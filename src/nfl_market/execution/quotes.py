"""Quote-level execution helpers for paper decisions (no wager placement, ever).

* :func:`best_price_same_contract`: the best-paying quote among rows that are all the
  same exact contract (callers filter with ``markets.contracts.contract_view`` first).
* :func:`net_ev`: EV per unit of a binary bet at decimal odds, minus a fee.
* :func:`net_ev_ternary`: the same with a push probability (push returns the stake),
  matching ``markets.edges`` ``ev_per_unit``.
"""
from __future__ import annotations

import pandas as pd

from ..markets.odds import american_to_decimal


def best_price_same_contract(frame: pd.DataFrame) -> pd.Series | None:
    """Row with the highest decimal odds (ties: first in frame order); None for an empty frame.

    The returned row carries a ``decimal_odds`` column. All rows must be the same contract.
    """
    if frame.empty:
        return None
    keys = [c for c in ("event_id", "game_id", "market", "selection", "line") if c in frame.columns]
    for k in keys:
        vals = frame[k].astype(str).str.lower().unique() if k == "selection" else frame[k].dropna().unique()
        if len(vals) > 1:
            raise ValueError(f"best_price_same_contract: rows differ in {k!r}; filter to one contract first")
    x = frame.copy()
    x["decimal_odds"] = x["odds_american"].map(american_to_decimal)
    return x.sort_values("decimal_odds", ascending=False, kind="stable").iloc[0]


def net_ev(probability: float, decimal_odds: float, fee_rate: float = 0.0) -> float:
    """``p (d - 1) - (1 - p) - fee`` per unit staked, for a two-outcome bet."""
    if not 0 < probability < 1 or decimal_odds <= 1:
        raise ValueError("Invalid input")
    return probability * (decimal_odds - 1) - (1 - probability) - fee_rate


def net_ev_ternary(p_win: float, p_push: float, decimal_odds: float, fee_rate: float = 0.0) -> float:
    """``p_win (d - 1) - p_loss - fee`` per unit staked; a push returns the stake."""
    p_loss = 1 - p_win - p_push
    if min(p_win, p_push) < 0 or p_loss < -1e-12 or decimal_odds <= 1:
        raise ValueError("Invalid input")
    return p_win * (decimal_odds - 1) - max(p_loss, 0.0) - fee_rate


__all__ = ["best_price_same_contract", "net_ev", "net_ev_ternary"]
