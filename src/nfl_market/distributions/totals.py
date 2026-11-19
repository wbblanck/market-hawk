"""Total-points pmfs (T = home + away).

* Empirical residuals: :class:`~nfl_market.distributions.margin.ResidualPool`
  with ``target='total'`` re-centred on the predicted total (mass below 0 is
  moved to 0).
* Discretised alternative: Student-t (or normal, ``df=inf``) around the
  predicted total, fit in-fold on training residuals, discretised on 0..150.

A total of 1 is impossible in football (a team cannot score exactly 1 point
from scrimmage), so both families put zero mass on ``T = 1`` by default.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .margin import ResidualPool, TParams, discretize_cdf, fit_student_t
from .pmf import IntPMF

TOTAL_SUPPORT: tuple[int, int] = (0, 150)
IMPOSSIBLE_TOTALS: tuple[int, ...] = (1,)


def _drop_impossible(p: IntPMF, impossible=IMPOSSIBLE_TOTALS) -> IntPMF:
    if not impossible:
        return p
    probs = p.probs.copy()
    for k in impossible:
        if p.lo <= k <= p.hi:
            probs[k - p.lo] = 0.0
    if probs.sum() <= 0:
        return p
    return IntPMF(p.lo, probs)


def total_pmf_empirical(mean: float, pool: ResidualPool, *, condition_on_bucket: bool = False,
                        impossible=IMPOSSIBLE_TOTALS) -> IntPMF:
    """Empirical-residual total pmf. ``pool`` must be fit with ``target='total'``."""
    if pool.target != "total":
        raise ValueError("pool was not fit on totals")
    return _drop_impossible(pool.pmf(mean, condition_on_bucket=condition_on_bucket, min_value=0), impossible)


def fit_total_t(residuals, **kw) -> TParams:
    """In-fold t fit for total residuals (same estimator as margins)."""
    return fit_student_t(residuals, **kw)


def total_pmf_discretized(mean: float, params: TParams | None = None, *, sd: float | None = None,
                          support: tuple[int, int] = TOTAL_SUPPORT, impossible=IMPOSSIBLE_TOTALS) -> IntPMF:
    """Discretised t (or normal if only ``sd`` is given) total pmf on ``support``."""
    if params is None:
        if sd is None or not np.isfinite(sd) or sd <= 0:
            raise ValueError("need TParams or a positive sd")
        dist = stats.norm(loc=mean, scale=sd)
    else:
        scale = params.scale if sd is None or not np.isfinite(sd) else params.scale_for_sd(sd)
        dist = stats.t(df=params.df, loc=mean + params.loc, scale=scale)
    return _drop_impossible(discretize_cdf(dist.cdf, *support), impossible)
