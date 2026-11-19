"""Fair prices from game predictions and a fitted outcome distribution.

One prediction row (``models.interface.PREDICTION_COLUMNS``) plus its margin pmf (and,
optionally, a total pmf) gives ternary probabilities for every contract we price:

* spread: at each home line ``h`` both sides, home at ``h`` and away at ``-h``
  (each row carries the selected side's own line);
* total: over and under at each total ``L``;
* moneyline: home and away (``line`` NaN); a tie is a push (two-way moneyline).

``fair_american`` is the unrounded no-vig price of ``p_win_no_push`` (push returns the
stake). All probabilities come from ``distributions.markets`` on the same pmf, so the
spread ladder, the moneyline and the push probabilities are mutually coherent.

The default distributions (step-3 recommendation, ``docs/methodology/distributions.md``)
are family ``k1`` for margins and ``t`` for totals, fit on OOF rows strictly before the
pricing time (:func:`fit_default_distributions`). The 2025 holdout is refused everywhere.

Fair prices are stored in the ``fair_prices`` DuckDB table (same file as ``MarketStore``),
append-only; readers keep the latest row per contract by ``prediction_time_utc``.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ..distributions.evaluate import (
    FittedDistribution,
    build_pmfs,
    fit_distribution,
    with_outcomes,
)
from ..distributions.markets import (
    ThreeWay,
    away_spread_probs,
    conditional_cover,
    fair_american,
    home_spread_probs,
    moneyline_probs,
    total_probs,
)
from ..distributions.pmf import IntPMF
from ..holdout import exclude_holdout, is_holdout
from .interface import PRICE_COLUMNS, empty_frame

DEFAULT_MARGIN_FAMILY = "k1"
DEFAULT_TOTAL_FAMILY = "t"
#: Half-point steps either side of the model's implied line in the default ladder (+-3 points).
LADDER_STEPS = 6

FAIR_PRICES_DDL = """
CREATE TABLE IF NOT EXISTS fair_prices (
 price_key VARCHAR PRIMARY KEY, game_id VARCHAR, season INTEGER, week INTEGER, kickoff_utc TIMESTAMP,
 model_id VARCHAR, dist_id VARCHAR, prediction_time_utc TIMESTAMP, market VARCHAR, selection VARCHAR,
 line DOUBLE, p_win DOUBLE, p_push DOUBLE, p_loss DOUBLE, p_win_no_push DOUBLE, fair_american DOUBLE,
 synthetic BOOLEAN, written_at_utc TIMESTAMP
)"""
_TS_COLS = ("kickoff_utc", "prediction_time_utc", "written_at_utc")


# ------------------------------------------------------------------ helpers
def refuse_holdout(game_ids: Iterable[str], what: str = "pricing") -> None:
    """Raise if any id is a 2025 holdout game (by manifest id or by season)."""
    ids = pd.Series(list(game_ids), dtype=object).astype(str)
    if len(ids) and is_holdout(ids).any():
        bad = sorted(ids[is_holdout(ids)].unique())[:5]
        raise ValueError(f"{what}: refusing holdout game_ids {bad} (2025 is the locked holdout)")


def dist_id(fd: FittedDistribution) -> str:
    """``'k1@2015-2024'``: family (with ``+key``) at its training seasons."""
    fam = fd.family + ("+key" if fd.key_adjust else "")
    s = fd.train_seasons
    return f"{fam}@{min(s)}-{max(s)}" if s else fam


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _row(base: dict, market: str, selection: str, line: float, tw: ThreeWay) -> dict:
    p_np = conditional_cover(tw.win, tw.push) if tw.push < 1 else float("nan")
    fair = fair_american(p_np) if 0 < p_np < 1 else float("nan")
    return {**base, "market": market, "selection": selection, "line": float(line),
            "p_win": float(tw.win), "p_push": float(tw.push), "p_loss": float(tw.loss),
            "p_win_no_push": float(p_np), "fair_american": float(fair)}


def _uniq(lines: Iterable[float]) -> list[float]:
    out = sorted({round(float(x), 3) for x in lines if x is not None and math.isfinite(float(x))})
    return out


# ------------------------------------------------------------------ pricing
def price_game(pred_row, margin_pmf: IntPMF, total_pmf: IntPMF | None = None, *,
               spread_lines: Sequence[float] = (), total_lines: Sequence[float] = (),
               model_id: str, dist_id: str, total_model_id: str | None = None,
               total_dist_id: str | None = None) -> list[dict]:
    """Price one game. Returns one PRICE_COLUMNS dict per contract.

    ``spread_lines`` are **home** lines ``h``; each yields home at ``h`` and away at ``-h``.
    ``total_lines`` are ignored when ``total_pmf`` is None. ``total_model_id`` /
    ``total_dist_id`` label total rows when the total comes from another model (default:
    ``model_id`` / ``dist_id``).
    """
    r = dict(pred_row)
    refuse_holdout([r["game_id"]])
    ko = r.get("kickoff_utc")
    base = {"game_id": str(r["game_id"]), "season": int(r["season"]), "week": int(r["week"]),
            "kickoff_utc": _utc(ko) if ko is not None and not pd.isna(ko) else pd.NaT,
            "model_id": model_id, "dist_id": dist_id,
            "prediction_time_utc": _utc(r["prediction_time_utc"])}
    rows = []
    for h in _uniq(spread_lines):
        rows.append(_row(base, "spread", "home", h, home_spread_probs(margin_pmf, h)))
        rows.append(_row(base, "spread", "away", -h + 0.0, away_spread_probs(margin_pmf, -h)))
    ml = moneyline_probs(margin_pmf)
    rows.append(_row(base, "moneyline", "home", float("nan"), ml))
    rows.append(_row(base, "moneyline", "away", float("nan"), ThreeWay(ml.loss, ml.push, ml.win)))
    if total_pmf is not None:
        tbase = {**base, "model_id": total_model_id or model_id, "dist_id": total_dist_id or dist_id}
        for L in _uniq(total_lines):
            if L < 0:
                continue
            tp = total_probs(total_pmf, L)
            rows.append(_row(tbase, "total", "over", L, tp))
            rows.append(_row(tbase, "total", "under", L, ThreeWay(tp.loss, tp.push, tp.win)))
    return rows


def ladder_lines(preds: pd.DataFrame, steps: int = LADDER_STEPS) -> pd.DataFrame:
    """Default lines: half-point ladder of +-``steps`` around the model's implied lines.

    Implied home line ``-round(2 * pred_margin) / 2``; implied total ``round(2 * pred_total) / 2``.
    Returned as a ``game_id, market, selection, line`` frame (home and over rows only; the
    opposite sides are priced automatically).
    """
    rows = []
    off = np.arange(-steps, steps + 1) / 2
    for r in preds.to_dict("records"):
        h0 = -round(2 * float(r["pred_margin"])) / 2
        rows += [{"game_id": r["game_id"], "market": "spread", "selection": "home", "line": h0 + o} for o in off]
        t = r.get("pred_total")
        if t is not None and not pd.isna(t):
            t0 = round(2 * float(t)) / 2
            rows += [{"game_id": r["game_id"], "market": "total", "selection": "over", "line": t0 + o}
                     for o in off if t0 + o > 0]
    return pd.DataFrame(rows, columns=["game_id", "market", "selection", "line"])


def _lines_for(lines: pd.DataFrame, game_id: str) -> tuple[list[float], list[float]]:
    g = lines[lines["game_id"].astype(str) == str(game_id)]
    sp = g[g["market"] == "spread"]
    home = sp["line"].where(sp["selection"].str.lower() == "home", -sp["line"])  # away a -> home -a
    tot = g[g["market"] == "total"]["line"]
    return _uniq(home.dropna()), _uniq(tot.dropna())


def _pmf_rows(fd: FittedDistribution, frame: pd.DataFrame) -> list[IntPMF]:
    lo, mat = build_pmfs(fd, frame)
    return [IntPMF(lo, mat[i]) for i in range(mat.shape[0])]


def price_frame(preds: pd.DataFrame, margin_fd: FittedDistribution, total_fd: FittedDistribution | None = None,
                lines: pd.DataFrame | None = None, *, total_model_col: str | None = None) -> pd.DataFrame:
    """Price every row of ``preds`` (PREDICTION_COLUMNS, optionally ``kickoff_utc``).

    ``lines`` (``game_id, market, selection, line``, e.g. from stored quotes) selects the
    contracts; ``None`` uses :func:`ladder_lines`. Total rows need ``pred_total`` and
    ``total_fd``. ``total_model_col`` names a column holding the model id of ``pred_total``
    when it came from another model than ``pred_margin`` (see ``scripts/price_slate.py``).
    """
    if preds.empty:
        return empty_frame(PRICE_COLUMNS)
    refuse_holdout(preds["game_id"])
    preds = preds.reset_index(drop=True)
    if lines is None:
        lines = ladder_lines(preds)
    m_pmfs = _pmf_rows(margin_fd, preds)
    has_total = total_fd is not None and preds["pred_total"].notna().all()
    t_pmfs = _pmf_rows(total_fd, preds) if has_total else [None] * len(preds)
    m_id, t_id = dist_id(margin_fd), dist_id(total_fd) if total_fd is not None else None
    rows: list[dict] = []
    for i, r in enumerate(preds.to_dict("records")):
        sp, tl = _lines_for(lines, r["game_id"])
        rows += price_game(r, m_pmfs[i], t_pmfs[i], spread_lines=sp, total_lines=tl, model_id=r["model_id"],
                           dist_id=m_id, total_model_id=r.get(total_model_col) if total_model_col else None,
                           total_dist_id=t_id)
    out = pd.DataFrame(rows)
    return out[list(PRICE_COLUMNS)].sort_values(["game_id", "model_id", "market", "selection", "line"],
                                                 ignore_index=True)


# ------------------------------------------------------------ distributions
def default_training_rows(oof_with_outcomes: pd.DataFrame, as_of) -> pd.DataFrame:
    """OOF rows usable at ``as_of``: non-holdout, ``prediction_time_utc < as_of`` and (if present)
    ``label_available_at_utc < as_of`` (the ``evaluate.training_rows`` cutoff rule)."""
    cutoff = _utc(as_of)
    df = with_outcomes(exclude_holdout(oof_with_outcomes))
    keep = pd.to_datetime(df["prediction_time_utc"], utc=True) < cutoff
    if "label_available_at_utc" in df.columns:
        keep &= pd.to_datetime(df["label_available_at_utc"], utc=True) < cutoff
    return df[keep].dropna(subset=["margin", "total"])


def fit_default_distributions(oof_with_outcomes: pd.DataFrame, as_of, *, margin_model_id: str | None = None,
                              total_model_id: str | None = None, margin_family: str = DEFAULT_MARGIN_FAMILY,
                              total_family: str = DEFAULT_TOTAL_FAMILY
                              ) -> tuple[FittedDistribution, FittedDistribution | None]:
    """Fit the margin (``k1``) and total (``t``) distributions on OOF rows strictly before ``as_of``.

    ``*_model_id`` restricts the OOF rows to that model's predictions (the residual shape
    belongs to the model being priced). The total fit is None if no row has ``pred_total``.
    """
    tr = default_training_rows(oof_with_outcomes, as_of)
    mtr = tr if margin_model_id is None else tr[tr["model_id"] == margin_model_id]
    if mtr.empty:
        raise ValueError(f"no OOF rows before {as_of} for margin model {margin_model_id!r}")
    margin_fd = fit_distribution(mtr, margin_family, target="margin")
    ttr = tr if total_model_id is None else tr[tr["model_id"] == total_model_id]
    ttr = ttr.dropna(subset=["pred_total"])
    total_fd = fit_distribution(ttr, total_family, target="total") if not ttr.empty else None
    return margin_fd, total_fd


# ---------------------------------------------------------------- storage
def _line_key(x) -> str:
    return "nan" if x is None or pd.isna(x) else f"{float(x):.3f}"


def _naive_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_localize(None)


def append_fair_prices(path: str | Path, prices: pd.DataFrame, *, synthetic: bool = False,
                       written_at=None) -> int:
    """Append PRICE_COLUMNS rows to ``fair_prices`` (append-only; identical rows skipped).

    Returns the number of rows inserted. Refuses holdout game_ids.
    """
    if prices.empty:
        return 0
    x = prices[list(PRICE_COLUMNS)].copy()
    refuse_holdout(x["game_id"], "append_fair_prices")
    x["synthetic"] = bool(synthetic)
    x["written_at_utc"] = _utc(written_at) if written_at is not None else pd.Timestamp.now(tz="UTC")
    x["price_key"] = (x["game_id"].astype(str) + "|" + x["model_id"].astype(str) + "|" + x["dist_id"].astype(str)
                      + "|" + x["market"] + "|" + x["selection"] + "|" + x["line"].map(_line_key) + "|"
                      + pd.to_datetime(x["prediction_time_utc"], utc=True).map(lambda t: t.isoformat()))
    x = x.drop_duplicates("price_key")
    for c in _TS_COLS:
        x[c] = _naive_utc(x[c])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(path)) as c:
        c.execute(FAIR_PRICES_DDL)
        c.register("incoming", x)
        before = c.execute("SELECT count(*) FROM fair_prices").fetchone()[0]
        c.execute("INSERT INTO fair_prices BY NAME SELECT * FROM incoming ON CONFLICT(price_key) DO NOTHING")
        after = c.execute("SELECT count(*) FROM fair_prices").fetchone()[0]
    return int(after - before)


def latest_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Latest row per contract (game, model, market, selection, line) by prediction time."""
    if frame.empty:
        return frame
    x = frame.copy()
    x["_lk"] = x["line"].map(_line_key)
    sort = ["prediction_time_utc"] + (["written_at_utc"] if "written_at_utc" in x.columns else [])
    x = x.sort_values(sort).drop_duplicates(["game_id", "model_id", "market", "selection", "_lk"], keep="last")
    return x.drop(columns="_lk").reset_index(drop=True)


def read_fair_prices(path: str | Path, *, latest: bool = True) -> pd.DataFrame:
    """Read ``fair_prices`` read-only (PRICE_COLUMNS + ``synthetic``). Empty if absent."""
    cols = list(PRICE_COLUMNS) + ["synthetic"]
    if not Path(path).exists():
        return empty_frame(cols)
    try:
        with duckdb.connect(str(path), read_only=True) as c:
            names = {r[0] for r in c.execute("SELECT table_name FROM information_schema.tables").fetchall()}
            if "fair_prices" not in names:
                return empty_frame(cols)
            d = c.execute("SELECT * FROM fair_prices").df()
    except duckdb.Error:
        return empty_frame(cols)
    for col in _TS_COLS:
        d[col] = pd.to_datetime(d[col]).dt.tz_localize("UTC")
    if latest:
        d = latest_prices(d)
    return d[cols].reset_index(drop=True)


__all__ = [
    "DEFAULT_MARGIN_FAMILY", "DEFAULT_TOTAL_FAMILY", "append_fair_prices", "default_training_rows", "dist_id",
    "fit_default_distributions", "ladder_lines", "latest_prices", "price_frame", "price_game", "read_fair_prices",
    "refuse_holdout",
]
