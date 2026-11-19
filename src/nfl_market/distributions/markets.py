"""Spread, total and moneyline probabilities derived from one margin/total pmf.

Sign conventions follow ``nfl_market.markets.clv``: a spread is the selected
team's *own* line (home favourite ``-3``, away underdog ``+3``)::

    home spread h:  P(cover) = P(M + h > 0), P(push) = P(M + h = 0), P(no cover) = P(M + h < 0)
    away spread a:  P(cover) = P(a - M > 0), ...       (a = -h on the same market)
    total line L:   P(over)  = P(T > L),     P(push) = P(T = L),      P(under) = P(T < L)

Half-point lines have zero push probability by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .pmf import IntPMF


@dataclass(frozen=True)
class ThreeWay:
    """Ternary settlement probabilities; ``win + push + loss == 1``."""

    win: float
    push: float
    loss: float

    def __post_init__(self):
        s = self.win + self.push + self.loss
        if abs(s - 1) > 1e-9 or min(self.win, self.push, self.loss) < -1e-12:
            raise ValueError(f"incoherent probabilities {self}")

    @property
    def win_given_no_push(self) -> float:
        return conditional_cover(self.win, self.push)

    def as_dict(self) -> dict:
        return {"win": self.win, "push": self.push, "loss": self.loss}


def _three_way(pmf: IntPMF, threshold: float) -> ThreeWay:
    """P(X > thr), P(X = thr), P(X < thr) with the remainder assigned so they sum to 1 exactly."""
    gt = pmf.prob_gt(threshold)
    eq = pmf.prob_eq(threshold)
    lt = max(0.0, 1.0 - gt - eq)
    return ThreeWay(gt, eq, lt)


def _check_line(x: float) -> float:
    x = float(x)
    if not np.isfinite(x):
        raise ValueError("line must be finite")
    return x


def is_whole_line(line: float) -> bool:
    return abs(float(line) - round(float(line))) < 1e-9


# ------------------------------------------------------------- moneyline
def p_home_win(margin: IntPMF) -> float:
    return margin.prob_gt(0)


def p_away_win(margin: IntPMF) -> float:
    return margin.prob_lt(0)


def p_tie(margin: IntPMF) -> float:
    return margin.prob_eq(0)


def moneyline_probs(margin: IntPMF) -> ThreeWay:
    """(home win, tie, away win). Two-way moneylines usually void ties: use ``win_given_no_push``."""
    return _three_way(margin, 0.0)


# --------------------------------------------------------------- spreads
def home_spread_probs(margin: IntPMF, home_spread: float) -> ThreeWay:
    """Home side at its own line ``h``: cover iff ``M + h > 0`` i.e. ``M > -h``."""
    return _three_way(margin, -_check_line(home_spread))


def away_spread_probs(margin: IntPMF, away_spread: float) -> ThreeWay:
    """Away side at its own line ``a``: cover iff ``-M + a > 0`` i.e. ``M < a``."""
    a = _check_line(away_spread)
    lt, eq = margin.prob_lt(a), margin.prob_eq(a)
    return ThreeWay(lt, eq, max(0.0, 1.0 - lt - eq))


def spread_table(margin: IntPMF, home_spreads: Sequence[float]) -> list[dict]:
    """Home cover/push/no-cover for a ladder of home lines (used for coherence checks)."""
    return [{"home_spread": h, **home_spread_probs(margin, h).as_dict()} for h in home_spreads]


# ---------------------------------------------------------------- totals
def total_probs(total: IntPMF, line: float) -> ThreeWay:
    """(over, push, under) at total line ``L``."""
    line = _check_line(line)
    if line < 0:
        raise ValueError("total line must be non-negative")
    return _three_way(total, line)


# ----------------------------------------------------------- key numbers
def key_number_probs(margin: IntPMF, keys: Sequence[int] = (3, 7, 10)) -> dict[int, float]:
    """``P(|M| = k)`` for each key number."""
    return {int(k): margin.prob_eq(k) + (margin.prob_eq(-k) if k != 0 else 0.0) for k in keys}


# ------------------------------------------------------------------ odds
def conditional_cover(p_cover: float, p_push: float) -> float:
    """``P(cover | no push) = P(cover) / (1 - P(push))``."""
    if not 0 <= p_push < 1:
        raise ValueError("p_push must be in [0, 1)")
    return float(p_cover / (1 - p_push))


def fair_american(p: float) -> float:
    """Fair (no-vig) American odds for win probability ``p`` (unrounded).

    ``p >= 0.5`` gives a negative price ``-100 p / (1 - p)``; otherwise ``100 (1 - p) / p``.
    For ternary markets pass :func:`conditional_cover` (push returns stake).
    """
    p = float(p)
    if not 0 < p < 1:
        raise ValueError("probability must be strictly between 0 and 1")
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p


def market_summary(margin: IntPMF, total: IntPMF | None = None, *, home_spread: float | None = None,
                   total_line: float | None = None, keys: Sequence[int] = (3, 7, 10)) -> dict:
    """Flat dict of every derived probability for one game (all from the same pmf)."""
    ml = moneyline_probs(margin)
    out = {"p_home_win": ml.win, "p_tie": ml.push, "p_away_win": ml.loss,
           "p_home_win_no_tie": ml.win_given_no_push}
    out.update({f"p_abs_margin_{k}": v for k, v in key_number_probs(margin, keys).items()})
    if home_spread is not None:
        hs = home_spread_probs(margin, home_spread)
        out.update({"home_spread": home_spread, "p_home_cover": hs.win, "p_spread_push": hs.push,
                    "p_away_cover": hs.loss, "p_home_cover_no_push": hs.win_given_no_push if hs.push < 1 else np.nan})
    if total is not None and total_line is not None:
        tp = total_probs(total, total_line)
        out.update({"total_line": total_line, "p_over": tp.win, "p_total_push": tp.push, "p_under": tp.loss})
    return out
