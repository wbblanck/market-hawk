"""Paper bet log: decisions, closes, CLV and results (paper/shadow only, never a wager).

Two append-only DuckDB tables in the ``MarketStore`` file (``settings.database_path``):

* ``paper_bets``: one decision row per bet. ``bet_id`` is a deterministic hash of
  game/market/selection/line/book/decision time/model, so recording the same decision
  twice is a no-op. Holdout (2025) game_ids and decisions at or after the close cutoff
  (``kickoff - 15 min``) are refused.
* ``paper_bet_grades``: append-only rows per ``bet_id`` with ``graded_at_utc`` and
  ``kind`` (``close`` or ``result``). The latest row of each kind wins; nothing is updated
  in place.

Closes use the locked policy (``markets.closing.select_close``) twice per bet:

* **line close** (``line=None``): the main-line close of the selected side (spreads in the
  selected team's own line); ``line_clv = clv.line_clv(selection, bet_line, close_line)``.
* **price close** (``line=bet line``): the devigged close on the exact bet contract;
  ``price_clv = clv.probability_clv(bet_fair, close_fair)`` where ``bet_fair`` is the bet
  book's own devigged pair at decision time (NaN without an opposite price).

Each close records its own tier. Line CLV and price CLV are never summed. Moneylines
have no line close (tier ``missing``, flag ``NO_LINE_CLV_MONEYLINE``); their price close
uses the same selector with the moneyline treated as one fixed contract.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ..config import TIER_ALL_BOOK_MEDIAN, TIER_MISSING, ClosePolicy
from ..holdout import is_holdout
from .closing import canonical_book, select_close
from .clv import line_clv, probability_clv
from .interface import BET_LOG_COLUMNS, CLV_SUMMARY_COLUMNS, empty_frame
from .odds import american_to_decimal, devig_two_way

FLAG_NO_LINE_CLV_ML = "NO_LINE_CLV_MONEYLINE"
_ML_LINE = 0.0   # moneyline stand-in line for the price-close selector

DECISION_COLUMNS: tuple[str, ...] = (
    "bet_id", "game_id", "season", "week", "kickoff_utc", "decision_time_utc", "quote_captured_at_utc", "book",
    "market", "selection", "line", "odds_american", "opposite_odds_american", "model_id", "dist_id", "p_win",
    "p_push", "ev_per_unit", "kelly_fraction", "stake_units", "stake_limit_reason", "synthetic",
)
CLOSE_FIELDS: tuple[str, ...] = (
    "line_close", "line_close_tier", "line_clv", "price_close_fair_prob", "price_close_tier", "price_clv",
    "close_flags", "close_policy_fingerprint",
)
RESULT_FIELDS: tuple[str, ...] = ("result", "pnl_units")

DDL = (
    """CREATE TABLE IF NOT EXISTS paper_bets (
 bet_id VARCHAR PRIMARY KEY, game_id VARCHAR, season INTEGER, week INTEGER, kickoff_utc TIMESTAMP,
 decision_time_utc TIMESTAMP, quote_captured_at_utc TIMESTAMP, book VARCHAR, market VARCHAR, selection VARCHAR,
 line DOUBLE, odds_american INTEGER, opposite_odds_american INTEGER, model_id VARCHAR, dist_id VARCHAR,
 p_win DOUBLE, p_push DOUBLE, ev_per_unit DOUBLE, kelly_fraction DOUBLE, stake_units DOUBLE,
 stake_limit_reason VARCHAR, synthetic BOOLEAN, recorded_at_utc TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS paper_bet_grades (
 grade_id VARCHAR PRIMARY KEY, bet_id VARCHAR, kind VARCHAR, graded_at_utc TIMESTAMP,
 line_close DOUBLE, line_close_tier VARCHAR, line_clv DOUBLE, price_close_fair_prob DOUBLE,
 price_close_tier VARCHAR, price_clv DOUBLE, close_flags VARCHAR, close_policy_fingerprint VARCHAR,
 result VARCHAR, pnl_units DOUBLE, synthetic BOOLEAN)""",
)
_BET_TS = ("kickoff_utc", "decision_time_utc", "quote_captured_at_utc", "recorded_at_utc")


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _naive(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_localize(None)


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or (not isinstance(x, str) and pd.isna(x))


def make_bet_id(game_id, market, selection, line, book, decision_time, model_id) -> str:
    ln = "nan" if _nan(line) else f"{float(line):.3f}"
    blob = "|".join([str(game_id), str(market), str(selection).lower(), ln, canonical_book(book),
                     _utc(decision_time).isoformat(), str(model_id)])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _grade_id(bet_id: str, kind: str, graded_at: pd.Timestamp, payload: str) -> str:
    return hashlib.sha256(f"{bet_id}|{kind}|{graded_at.isoformat()}|{payload}".encode()).hexdigest()[:20]


# ------------------------------------------------------------------- settlement
def settle(market: str, selection: str, line, margin: float, total: float) -> str:
    """win / push / loss for one contract given home margin and total points."""
    sel = selection.lower()
    if market == "spread":
        x = margin + float(line) if sel == "home" else float(line) - margin
    elif market == "total":
        x = total - float(line) if sel == "over" else float(line) - total
    elif market == "moneyline":
        x = margin if sel == "home" else -margin
    else:
        raise ValueError(f"unknown market {market!r}")
    return "win" if x > 0 else "push" if x == 0 else "loss"


def pnl_units(result: str, stake: float, odds_american: int) -> float:
    if result == "win":
        return float(stake) * (american_to_decimal(int(odds_american)) - 1)
    return 0.0 if result == "push" else -float(stake)


# ------------------------------------------------------------------------ store
def _tables(c) -> set[str]:
    return {r[0] for r in c.execute("SELECT table_name FROM information_schema.tables").fetchall()}


def _assemble(bets: pd.DataFrame, grades: pd.DataFrame) -> pd.DataFrame:
    """Join decisions with the latest close row and latest result row (BET_LOG_COLUMNS)."""
    if bets.empty:
        return empty_frame(BET_LOG_COLUMNS)
    b = bets.copy()
    for col in _BET_TS:
        if col in b.columns:
            b[col] = pd.to_datetime(b[col]).dt.tz_localize("UTC")
    g = grades.sort_values(["graded_at_utc", "grade_id"]) if not grades.empty else grades
    closes = g[g["kind"] == "close"].drop_duplicates("bet_id", keep="last") if not g.empty else g
    results = g[g["kind"] == "result"].drop_duplicates("bet_id", keep="last") if not g.empty else g
    if not closes.empty:
        b = b.merge(closes[["bet_id", *CLOSE_FIELDS]], on="bet_id", how="left")
    if not results.empty:
        b = b.merge(results[["bet_id", *RESULT_FIELDS]], on="bet_id", how="left")
    for f in (*CLOSE_FIELDS, *RESULT_FIELDS):
        if f not in b.columns:
            b[f] = np.nan
    for f in ("line_close_tier", "price_close_tier", "close_flags", "close_policy_fingerprint", "result"):
        b[f] = b[f].fillna("").astype(str)
    b["status"] = np.where(b["result"] != "", "graded", np.where(b["close_policy_fingerprint"] != "", "closed", "open"))
    b["opposite_odds_american"] = b["opposite_odds_american"].astype("Int64")
    b = b.sort_values(["decision_time_utc", "bet_id"], ascending=[False, True], ignore_index=True)
    return b[list(BET_LOG_COLUMNS)]


def read_bet_log(path: str | Path) -> pd.DataFrame:
    """BET_LOG_COLUMNS from ``path`` opened read-only; empty when the file or tables are absent."""
    if not Path(path).exists():
        return empty_frame(BET_LOG_COLUMNS)
    try:
        with duckdb.connect(str(path), read_only=True) as c:
            t = _tables(c)
            if "paper_bets" not in t:
                return empty_frame(BET_LOG_COLUMNS)
            bets = c.execute("SELECT * FROM paper_bets").df()
            grades = c.execute("SELECT * FROM paper_bet_grades").df() if "paper_bet_grades" in t else pd.DataFrame()
    except duckdb.Error:
        return empty_frame(BET_LOG_COLUMNS)
    return _assemble(bets, grades)


class BetLogStore:
    """Append-only paper bet log in the DuckDB file at ``path``."""

    def __init__(self, path: str | Path = "data/nfl_market.duckdb", close_policy: ClosePolicy | None = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.close_policy = close_policy or ClosePolicy()
        with duckdb.connect(self.path) as c:
            for ddl in DDL:
                c.execute(ddl)

    # ---------------------------------------------------------------- decisions
    def _validate(self, x: pd.DataFrame) -> None:
        hold = is_holdout(x["game_id"].astype(str))
        if hold.any():
            raise ValueError(f"refusing holdout game_ids {sorted(x.loc[hold, 'game_id'].unique())[:5]}")
        cutoff = pd.to_datetime(x["kickoff_utc"], utc=True) - pd.Timedelta(minutes=self.close_policy.cutoff_minutes)
        late = ~(pd.to_datetime(x["decision_time_utc"], utc=True) < cutoff)
        if late.any():
            raise ValueError(f"refusing {int(late.sum())} bets decided at/after kickoff - "
                             f"{self.close_policy.cutoff_minutes} min (or with unknown kickoff)")
        bad = pd.to_datetime(x["quote_captured_at_utc"], utc=True) > pd.to_datetime(x["decision_time_utc"], utc=True)
        if bad.any():
            raise ValueError("a bet's quote was captured after its decision time")
        if (x["stake_units"] <= 0).any():
            raise ValueError("stake_units must be > 0 (filter zero stakes before recording)")

    def record(self, bets: pd.DataFrame, decision_time=None, *, recorded_at=None) -> int:
        """Append sized bets (``staking.size_stakes`` rows with stake > 0). Returns rows inserted.

        ``decision_time`` fills ``decision_time_utc`` when the frame has none. Re-recording the
        same decision inserts nothing (deterministic ``bet_id``).
        """
        if bets is None or bets.empty:
            return 0
        x = bets.copy()
        if "decision_time_utc" not in x.columns:
            if decision_time is None:
                raise ValueError("decision_time is required")
            x["decision_time_utc"] = _utc(decision_time)
        if "quote_captured_at_utc" not in x.columns:
            x["quote_captured_at_utc"] = x["captured_at_utc"]
        if "synthetic" not in x.columns:
            x["synthetic"] = False
        x["book"] = x["book"].map(canonical_book)
        self._validate(x)
        x["bet_id"] = [make_bet_id(*r) for r in zip(x["game_id"], x["market"], x["selection"], x["line"], x["book"],
                                                     x["decision_time_utc"], x["model_id"])]
        x["recorded_at_utc"] = _utc(recorded_at) if recorded_at is not None else pd.Timestamp.now(tz="UTC")
        x = x[list(DECISION_COLUMNS) + ["recorded_at_utc"]].drop_duplicates("bet_id")
        for col in _BET_TS:
            x[col] = _naive(x[col])
        x["opposite_odds_american"] = x["opposite_odds_american"].astype("Int64")
        with duckdb.connect(self.path) as c:
            c.register("incoming", x)
            before = c.execute("SELECT count(*) FROM paper_bets").fetchone()[0]
            c.execute("INSERT INTO paper_bets BY NAME SELECT * FROM incoming ON CONFLICT(bet_id) DO NOTHING")
            after = c.execute("SELECT count(*) FROM paper_bets").fetchone()[0]
        return int(after - before)

    # ------------------------------------------------------------------ grades
    def append_grades(self, rows: pd.DataFrame) -> int:
        """Append close/result rows (``bet_id``, ``kind``, ``graded_at_utc`` + fields). Append-only."""
        if rows is None or rows.empty:
            return 0
        x = rows.copy()
        for f in (*CLOSE_FIELDS, *RESULT_FIELDS):
            if f not in x.columns:
                x[f] = None
        if "synthetic" not in x.columns:
            x["synthetic"] = False
        x["graded_at_utc"] = pd.to_datetime(x["graded_at_utc"], utc=True)
        x["grade_id"] = [_grade_id(b, k, t, repr(tuple(r))) for b, k, t, r in
                         zip(x["bet_id"], x["kind"], x["graded_at_utc"],
                             x[[*CLOSE_FIELDS, *RESULT_FIELDS]].itertuples(index=False))]
        x["graded_at_utc"] = _naive(x["graded_at_utc"])
        cols = ["grade_id", "bet_id", "kind", "graded_at_utc", *CLOSE_FIELDS, *RESULT_FIELDS, "synthetic"]
        x = x[cols].drop_duplicates("grade_id")
        with duckdb.connect(self.path) as c:
            c.register("incoming", x)
            before = c.execute("SELECT count(*) FROM paper_bet_grades").fetchone()[0]
            c.execute("INSERT INTO paper_bet_grades BY NAME SELECT * FROM incoming ON CONFLICT(grade_id) DO NOTHING")
            after = c.execute("SELECT count(*) FROM paper_bet_grades").fetchone()[0]
        return int(after - before)

    def attach_closes(self, quotes: pd.DataFrame, policy: ClosePolicy | None = None, *, as_of=None,
                      graded_at=None) -> int:
        """Compute line and price closes for every bet whose close cutoff is <= ``as_of``.

        Appends one ``close`` row per bet stamped ``graded_at`` (default ``as_of``); re-running with
        the same quotes and ``graded_at`` inserts nothing. Returns the number of rows appended.
        """
        pol = policy or self.close_policy
        as_of = _utc(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
        graded_at = _utc(graded_at) if graded_at is not None else as_of
        bets = self.bets()
        if bets.empty:
            return 0
        cut = bets["kickoff_utc"] - pd.Timedelta(minutes=pol.cutoff_minutes)
        bets = bets[(cut <= as_of) & (bets["status"] != "void")]
        q = quotes.copy()
        if "event_id" not in q.columns:
            q = q.rename(columns={"game_id": "event_id"})
        q["market"] = q["market"].astype(str).str.lower()
        rows = [dict(bet_id=b["bet_id"], kind="close", graded_at_utc=graded_at, synthetic=bool(b["synthetic"]),
                     **close_fields(q, b, pol)) for b in bets.to_dict("records")]
        return self.append_grades(pd.DataFrame(rows))

    def grade(self, games: pd.DataFrame, *, graded_at=None) -> int:
        """Attach win/push/loss and ``pnl_units`` from final scores (``game_id, home_score, away_score``)."""
        graded_at = _utc(graded_at) if graded_at is not None else pd.Timestamp.now(tz="UTC")
        bets = self.bets()
        if bets.empty:
            return 0
        g = games[["game_id", "home_score", "away_score"]].dropna()
        b = bets.merge(g, on="game_id", how="inner")
        rows = []
        for r in b.to_dict("records"):
            m, t = r["home_score"] - r["away_score"], r["home_score"] + r["away_score"]
            res = settle(r["market"], r["selection"], r["line"], m, t)
            rows.append({"bet_id": r["bet_id"], "kind": "result", "graded_at_utc": graded_at, "result": res,
                             "pnl_units": pnl_units(res, r["stake_units"], r["odds_american"]),
                             "synthetic": bool(r["synthetic"])})
        return self.append_grades(pd.DataFrame(rows))

    def bets(self) -> pd.DataFrame:
        """All bets with their latest close and result (BET_LOG_COLUMNS), newest decision first."""
        return read_bet_log(self.path)


# ------------------------------------------------------------------- closes
def close_fields(quotes: pd.DataFrame, bet: dict, policy: ClosePolicy) -> dict:
    """Line close, price close and both CLVs for one bet (dict of CLOSE_FIELDS)."""
    market, sel = bet["market"], str(bet["selection"]).lower()
    flags: list[str] = []
    out = {"line_close": np.nan, "line_close_tier": TIER_MISSING, "line_clv": np.nan,
           "price_close_fair_prob": np.nan, "price_close_tier": TIER_MISSING, "price_clv": np.nan,
           "close_policy_fingerprint": policy.fingerprint()}
    if market == "moneyline":
        flags.append(FLAG_NO_LINE_CLV_ML)
        q = quotes[quotes["market"] == "moneyline"].copy()
        q["line"] = _ML_LINE
        price = select_close(q, bet["kickoff_utc"], policy, event_id=bet["game_id"], market="moneyline",
                             selection=sel, line=_ML_LINE)
    else:
        lc = select_close(quotes, bet["kickoff_utc"], policy, event_id=bet["game_id"], market=market,
                          selection=sel, line=None)
        flags += list(lc.flags)
        out["line_close_tier"] = lc.close_tier
        if not lc.is_missing and lc.close_line is not None:
            out["line_close"] = float(lc.close_line)
            out["line_clv"] = line_clv(sel, float(bet["line"]), float(lc.close_line), market=market)
        price = select_close(quotes, bet["kickoff_utc"], policy, event_id=bet["game_id"], market=market,
                             selection=sel, line=float(bet["line"]))
    flags += list(price.flags)
    if not price.is_missing and price.close_fair_probability is not None:
        out["price_close_tier"] = price.close_tier
        out["price_close_fair_prob"] = float(price.close_fair_probability)
        if not _nan(bet["opposite_odds_american"]):
            bet_fair = devig_two_way(int(bet["odds_american"]), int(bet["opposite_odds_american"]))[0]
            out["price_clv"] = probability_clv(bet_fair, out["price_close_fair_prob"])
    out["close_flags"] = ",".join(dict.fromkeys(flags))
    return out


# ------------------------------------------------------------------ summary
def clv_summary(bets: pd.DataFrame, by: Sequence[str] = ("close_tier", "market")) -> pd.DataFrame:
    """CLV aggregates per group (CLV_SUMMARY_COLUMNS).

    ``close_tier`` is always a grouping key (weak-fallback rows are never pooled with sharp
    ones): line-CLV statistics group bets by ``line_close_tier`` and price-CLV statistics by
    ``price_close_tier``, so one bet can count in two tier groups. Only bets with a close
    attached are included; a ``missing`` close is counted in ``n_bets`` but excluded from
    the means and shares.
    """
    if bets is None or bets.empty:
        return empty_frame(CLV_SUMMARY_COLUMNS)
    b = bets[bets["status"].isin(["closed", "graded"])]
    if b.empty:
        return empty_frame(CLV_SUMMARY_COLUMNS)
    extra = [k for k in by if k != "close_tier"]
    long = pd.concat([
        b.assign(close_tier=b["line_close_tier"], _kind="line", _v=b["line_clv"]),
        b.assign(close_tier=b["price_close_tier"], _kind="price", _v=b["price_clv"]),
    ], ignore_index=True)
    keys = ["close_tier", *extra]
    rows = []
    for gk, g in long.groupby(keys, sort=True, dropna=False):
        gk = gk if isinstance(gk, tuple) else (gk,)
        lv = g.loc[g["_kind"] == "line", "_v"].astype(float).dropna()
        pv = g.loc[g["_kind"] == "price", "_v"].astype(float).dropna()
        tier = gk[0]
        rows.append({
            "group": "|".join(f"{k}={v}" for k, v in zip(keys, gk)),
            "close_tier": tier,
            "market": dict(zip(keys, gk)).get("market", "all"),
            "n_bets": int(g["bet_id"].nunique()),
            "n_line_clv": int(lv.size),
            "mean_line_clv": float(lv.mean()) if lv.size else np.nan,
            "share_positive_line_clv": float((lv > 0).mean()) if lv.size else np.nan,
            "n_price_clv": int(pv.size),
            "mean_price_clv": float(pv.mean()) if pv.size else np.nan,
            "share_positive_price_clv": float((pv > 0).mean()) if pv.size else np.nan,
            "is_weak_fallback": tier == TIER_ALL_BOOK_MEDIAN,
        })
    return pd.DataFrame(rows, columns=list(CLV_SUMMARY_COLUMNS))


__all__ = [
    "DECISION_COLUMNS",
    "FLAG_NO_LINE_CLV_ML",
    "BetLogStore",
    "close_fields",
    "clv_summary",
    "make_bet_id",
    "pnl_units",
    "read_bet_log",
    "settle",
]
