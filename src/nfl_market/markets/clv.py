"""Closing line value (CLV).

Two separate metrics live here and must never be silently combined:

* **Line CLV** (points): how many points better the bet line was than the close.
* **Price CLV** (probability): how much the devigged fair probability of the
  selection moved in the bettor's favour between bet time and close, measured
  on the *same* contract (same event, market, selection and exact line).

Sign convention (positive is always favourable to the bettor):

* Spreads use the **selected team's own spread** (e.g. underdog ``+3.5``,
  favourite ``-2.5``), never an ambiguous home/away market line.
  ``line_clv = bet_line - close_line``.
* Over: ``line_clv = close_line - bet_line``.
* Under: ``line_clv = bet_line - close_line``.

Examples (locked by tests):

* Favourite -2.5, close -3   -> +0.5
* Underdog  +3.5, close +3   -> +0.5
* Over 44.5,  close 45.5     -> +1.0
* Under 46.5, close 45.5     -> +1.0
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .odds import devig_two_way

SPREAD_SELECTIONS = frozenset({"home", "away", "favorite", "underdog"})
TOTAL_SELECTIONS = frozenset({"over", "under"})
MARKET_SELECTIONS = {"spread": SPREAD_SELECTIONS, "total": TOTAL_SELECTIONS}


def _market_for(selection: str) -> str:
    side = selection.lower()
    if side in SPREAD_SELECTIONS:
        return "spread"
    if side in TOTAL_SELECTIONS:
        return "total"
    raise ValueError(f"Unsupported selection: {selection}")


def _finite(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def selection_spread(home_line: float, selection: str) -> float:
    """Convert a home-perspective spread into the selected team's own spread."""
    side = selection.lower()
    if side == "home":
        return float(home_line)
    if side == "away":
        return -float(home_line)
    raise ValueError("selection_spread needs 'home' or 'away'")


def line_clv(selection: str, bet_line: float, close_line: float, market: str | None = None) -> float:
    """Line CLV in points; positive is favourable.

    For spreads, ``bet_line`` and ``close_line`` must both be the selected
    team's own spread. Passing ``market`` checks the selection belongs to it.
    """
    side = selection.lower()
    inferred = _market_for(side)
    if market is not None and market.lower() != inferred:
        raise ValueError(f"Selection {selection!r} is not valid for market {market!r}")
    bet, close = _finite("bet_line", bet_line), _finite("close_line", close_line)
    if inferred == "total":
        if bet <= 0 or close <= 0:
            raise ValueError("Total lines must be positive")
        return close - bet if side == "over" else bet - close
    return bet - close


def probability_clv(bet_fair_probability: float, close_fair_probability: float) -> float:
    """Fair-probability CLV: close minus bet. Positive means the market moved toward the bet."""
    for name, p in (("bet_fair_probability", bet_fair_probability), ("close_fair_probability", close_fair_probability)):
        if not 0 < p < 1:
            raise ValueError(f"{name} must be strictly between 0 and 1")
    return close_fair_probability - bet_fair_probability


def price_clv(bet_odds: int, bet_opposite_odds: int, close_odds: int, close_opposite_odds: int) -> float:
    """Price CLV from devigged two-way prices on the same contract.

    Each pair is (selection, opposite side) American odds at the same line.
    Returns ``close_fair_p - bet_fair_p`` for the selection, in probability units.
    The caller is responsible for ensuring bet and close quotes are the same
    exact contract; if the line changed, use :func:`line_clv` separately.
    """
    bet_p, _ = devig_two_way(bet_odds, bet_opposite_odds)
    close_p, _ = devig_two_way(close_odds, close_opposite_odds)
    return probability_clv(bet_p, close_p)


@dataclass(frozen=True)
class CLVResult:
    """Line and price CLV reported side by side, never summed."""

    selection: str
    line_clv_points: float
    price_clv_probability: float | None  # None when bet/close lines differ (not comparable)


def clv_report(
    selection: str,
    bet_line: float,
    close_line: float,
    bet_odds: int | None = None,
    bet_opposite_odds: int | None = None,
    close_odds: int | None = None,
    close_opposite_odds: int | None = None,
    market: str | None = None,
) -> CLVResult:
    """Compute both CLV metrics. Price CLV is only reported for an identical line."""
    lc = line_clv(selection, bet_line, close_line, market=market)
    odds = (bet_odds, bet_opposite_odds, close_odds, close_opposite_odds)
    pc = None
    if all(o is not None for o in odds) and math.isclose(float(bet_line), float(close_line), abs_tol=1e-9):
        pc = price_clv(*odds)  # type: ignore[arg-type]
    return CLVResult(selection=selection.lower(), line_clv_points=lc, price_clv_probability=pc)
