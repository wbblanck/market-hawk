"""Edges: stored sportsbook quotes joined to model fair prices on the exact contract.

For each (game, book, market, selection, line) the latest quote with
``captured_at_utc <= as_of`` is compared with the latest fair price whose
``prediction_time_utc <= as_of`` (point in time on both sides).

Screens (rows dropped):

* ``is_live`` quotes (pre-game only);
* quotes captured at or after the close cutoff ``kickoff - 15 min``
  (``ClosePolicy.cutoff_minutes``, locked) and every quote of a game whose cutoff is
  at or before ``as_of`` (betting on it is closed);
* quotes with unknown kickoff or invalid American odds;
* quotes older than ``StakePolicy.max_quote_age_minutes`` at ``as_of`` (default 180,
  proposed, **not locked**). With ``include_stale=True`` they are kept and flagged.

Flags (comma-separated, "" when clean):

* ``NO_OPPOSITE``: the same book's snapshot has no paired opposite side, so
  ``market_fair_prob`` / ``edge_vs_market`` are NaN.
* ``STALE_QUOTE``: older than the max quote age (only with ``include_stale``).
* ``SUPERSEDED``: the book posted a later snapshot of this market that no longer
  contains this line (the book moved off it); the quote is probably not available.
* ``SYNTHETIC``: demo quote or demo price.

``opposite_odds_american`` comes from the same book and the same ``captured_at_utc``
at the paired line (spread ``h`` pairs with ``-h``, total and moneyline the same line).
``ev_per_unit = p_win * (decimal - 1) - p_loss``; a push returns the stake.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ClosePolicy, StakePolicy
from .closing import OPPOSITE, canonical_book
from .interface import EDGE_COLUMNS, empty_frame
from .odds import (
    american_to_decimal,
    devig_two_way,
    implied_probability,
    is_valid_american,
)

FLAG_NO_OPPOSITE = "NO_OPPOSITE"
FLAG_STALE_QUOTE = "STALE_QUOTE"
FLAG_SUPERSEDED = "SUPERSEDED"
FLAG_SYNTHETIC = "SYNTHETIC"
#: Flags that make an edge ineligible for a stake (``markets.staking``).
BLOCKING_FLAGS = frozenset({FLAG_STALE_QUOTE, FLAG_SUPERSEDED})

_ML_KEY = -9999.0  # moneyline line (NaN) as a join key


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _key(line: pd.Series) -> pd.Series:
    return line.astype(float).round(3).fillna(_ML_KEY)


def _paired_key(market: pd.Series, lk: pd.Series) -> pd.Series:
    return np.where((market == "spread") & (lk != _ML_KEY), -lk, lk) + 0.0


def prepare_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    """Normalise a store quote frame: ``game_id``, canonical ``book``, lower-case fields, UTC times."""
    q = quotes.copy()
    if "game_id" not in q.columns:
        q = q.rename(columns={"event_id": "game_id"})
    q["game_id"] = q["game_id"].astype(str)
    q["book"] = q["book"].map(canonical_book)
    q["market"] = q["market"].astype(str).str.lower()
    q["selection"] = q["selection"].astype(str).str.lower()
    q["line"] = pd.to_numeric(q["line"], errors="coerce").astype(float)
    q.loc[q["market"] == "moneyline", "line"] = np.nan
    for col in ("is_live", "synthetic"):
        q[col] = q[col].fillna(False).astype(bool) if col in q.columns else False
    q["captured_at_utc"] = pd.to_datetime(q["captured_at_utc"], utc=True)
    q["kickoff_utc"] = pd.to_datetime(q["kickoff_utc"], utc=True) if "kickoff_utc" in q.columns else pd.NaT
    q = q[q["odds_american"].map(is_valid_american)]
    q["odds_american"] = q["odds_american"].astype(int)
    q["_lk"] = _key(q["line"])
    return q


def _latest_prices(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    p = prices.copy()
    p["prediction_time_utc"] = pd.to_datetime(p["prediction_time_utc"], utc=True)
    p = p[p["prediction_time_utc"] <= as_of]
    p["_lk"] = _key(p["line"])
    p = p.sort_values("prediction_time_utc").drop_duplicates(["game_id", "model_id", "market", "selection", "_lk"],
                                                             keep="last")
    if "synthetic" not in p.columns:
        p["synthetic"] = False
    p["kickoff_utc"] = pd.to_datetime(p["kickoff_utc"], utc=True)
    keep = ["game_id", "season", "week", "kickoff_utc", "model_id", "dist_id", "market", "selection", "_lk",
            "p_win", "p_push", "p_loss", "p_win_no_push", "synthetic"]
    return p[keep].rename(columns={"season": "p_season", "week": "p_week", "kickoff_utc": "p_kickoff",
                                   "synthetic": "p_synthetic"})


def _superseded(latest: pd.DataFrame, pre: pd.DataFrame) -> pd.Series:
    """True where the book has a later snapshot of the same game/market than this quote."""
    last = pre.groupby(["game_id", "book", "market"])["captured_at_utc"].max().rename("_book_last")
    t = latest.join(last, on=["game_id", "book", "market"])
    return (t["_book_last"] > t["captured_at_utc"]).to_numpy()


def compute_edges(quotes: pd.DataFrame, prices: pd.DataFrame, as_of=None, *,
                  stake_policy: StakePolicy | None = None, close_policy: ClosePolicy | None = None,
                  include_stale: bool = False) -> pd.DataFrame:
    """Edges of the latest quote per book/contract at ``as_of`` (``EDGE_COLUMNS``), best EV first.

    ``quotes`` is a quote-store frame (``event_id`` or ``game_id``); ``prices`` has
    PRICE_COLUMNS (optionally ``synthetic``). ``as_of`` defaults to now (UTC).
    """
    sp = stake_policy or StakePolicy()
    cp = close_policy or ClosePolicy()
    if quotes is None or prices is None or quotes.empty or prices.empty:
        return empty_frame(EDGE_COLUMNS)
    as_of = _utc(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    q = prepare_quotes(quotes)
    q = q[~q["is_live"] & (q["captured_at_utc"] <= as_of)]
    p = _latest_prices(prices, as_of)
    if q.empty or p.empty:
        return empty_frame(EDGE_COLUMNS)

    # kickoff: the price row (nflverse schedule) wins over the quote's own field
    ko = p.groupby("game_id")["p_kickoff"].max()
    q["kickoff_utc"] = q["game_id"].map(ko).fillna(q["kickoff_utc"])
    q = q[q["kickoff_utc"].notna()]
    cutoff = q["kickoff_utc"] - pd.Timedelta(minutes=cp.cutoff_minutes)
    q = q[(q["captured_at_utc"] < cutoff) & (as_of < cutoff)]
    if q.empty:
        return empty_frame(EDGE_COLUMNS)

    snap = q.copy()  # every pre-as_of row, for opposite sides and supersession
    latest = (q.sort_values("captured_at_utc")
               .drop_duplicates(["game_id", "book", "market", "selection", "_lk"], keep="last").copy())
    latest["quote_age_minutes"] = (as_of - latest["captured_at_utc"]).dt.total_seconds() / 60
    latest["_stale"] = latest["quote_age_minutes"] > sp.max_quote_age_minutes
    if not include_stale:
        latest = latest[~latest["_stale"]]
    latest["_superseded"] = _superseded(latest, snap)

    # opposite side: same book, same capture time, paired line
    latest["_opp_sel"] = latest["selection"].map(OPPOSITE)
    latest["_opp_lk"] = _paired_key(latest["market"], latest["_lk"])
    opp = snap[["game_id", "book", "market", "captured_at_utc", "selection", "_lk", "odds_american"]].rename(
        columns={"selection": "_opp_sel", "_lk": "_opp_lk", "odds_american": "opposite_odds_american"})
    opp = opp.drop_duplicates(["game_id", "book", "market", "captured_at_utc", "_opp_sel", "_opp_lk"], keep="last")
    latest = latest.merge(opp, on=["game_id", "book", "market", "captured_at_utc", "_opp_sel", "_opp_lk"],
                          how="left")

    e = latest.merge(p, on=["game_id", "market", "selection", "_lk"], how="inner")
    if e.empty:
        return empty_frame(EDGE_COLUMNS)
    e["season"] = e["p_season"].fillna(e.get("season")).astype("Int64")
    e["week"] = e["p_week"].fillna(e.get("week")).astype("Int64")
    e["synthetic"] = e["synthetic"].astype(bool) | e["p_synthetic"].fillna(False).astype(bool)

    dec = e["odds_american"].map(american_to_decimal)
    e["breakeven_prob"] = e["odds_american"].map(implied_probability)
    has_opp = e["opposite_odds_american"].notna()
    e["market_fair_prob"] = [devig_two_way(int(o), int(oo))[0] if h else np.nan
                             for o, oo, h in zip(e["odds_american"], e["opposite_odds_american"], has_opp)]
    e["edge_prob"] = e["p_win_no_push"] - e["breakeven_prob"]
    e["edge_vs_market"] = e["p_win_no_push"] - e["market_fair_prob"]
    e["ev_per_unit"] = e["p_win"] * (dec - 1) - e["p_loss"]
    e["opposite_odds_american"] = e["opposite_odds_american"].astype("Int64")

    def flags(r) -> str:
        f = []
        if not r["_has_opp"]:
            f.append(FLAG_NO_OPPOSITE)
        if r["_stale"]:
            f.append(FLAG_STALE_QUOTE)
        if r["_superseded"]:
            f.append(FLAG_SUPERSEDED)
        if r["synthetic"]:
            f.append(FLAG_SYNTHETIC)
        return ",".join(f)

    e["_has_opp"] = has_opp
    e["flags"] = e.apply(flags, axis=1)
    e = e.sort_values(["ev_per_unit", "game_id", "market", "selection", "line", "book", "model_id"],
                      ascending=[False, True, True, True, True, True, True], ignore_index=True)
    return e[list(EDGE_COLUMNS)]


__all__ = ["BLOCKING_FLAGS", "FLAG_NO_OPPOSITE", "FLAG_STALE_QUOTE", "FLAG_SUPERSEDED", "FLAG_SYNTHETIC",
           "compute_edges", "prepare_quotes"]
