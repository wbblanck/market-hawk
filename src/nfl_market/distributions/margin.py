"""Margin pmfs built around a model's predicted mean (predictive_modeling.md Section 4).

Families
--------
* **Empirical residuals** (:class:`ResidualPool`): realised ``outcome - predicted``
  residuals from *training-fold* out-of-fold rows are re-centred on a new
  predicted mean. Mass of each shifted residual is split linearly between the
  two neighbouring integers, which preserves the mean exactly. The pool can be
  restricted to rows in the same spread bucket.
* **Student-t** (:func:`fit_student_t`, :func:`t_pmf`): location = predicted
  mean, scale/df fit on training-fold residuals, discretised as
  ``P(M=k) = F(k+0.5) - F(k-0.5)`` with the tails folded into the end points.
* **Key-number adjustment** (:func:`fit_key_weights`, :func:`apply_key_weights`):
  multiplicative weights on the mass at ``|M| = k`` for key numbers (default
  0, 3, 4, 6, 7, 10, 14), fit so that the mean predicted ``P(|M|=k)`` over
  training-fold rows matches the observed training-fold frequency. After
  reweighting the pmf is renormalised (mass exactly 1) and exponentially tilted
  so the mean equals the pre-adjustment mean.

Ties: ``M = 0`` is a key number in its own right. Continuous families put about
3% on 0, NFL ties are roughly 0.2-0.3% of games, so the default key set
includes 0 and the adjustment pulls tie mass to the training-fold rate.

Nothing in this module looks at test rows: every fit function receives the
training rows explicitly, and :func:`fit_residual_pool` filters by a cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .pmf import IntPMF

#: Default integer support for margins (home - away).
MARGIN_SUPPORT: tuple[int, int] = (-80, 80)
#: Default key numbers for the margin adjustment (absolute margin values).
DEFAULT_MARGIN_KEYS: tuple[int, ...] = (0, 3, 4, 6, 7, 10, 14)
#: Preregistered spread buckets (calibration_check.md Section 2).
SPREAD_BUCKETS: tuple[str, ...] = ("0-2.5", "3", "3.5-6.5", "7", "7.5-9.5", "10+")


# ------------------------------------------------------------------ buckets
def spread_bucket(values) -> np.ndarray:
    """Bucket labels for the absolute value of a reference spread.

    Values are rounded to the nearest half point first, so a model's continuous
    predicted margin of 2.9 falls in the ``3`` bucket.
    """
    a = np.round(np.abs(np.asarray(values, dtype=float)) * 2) / 2
    out = np.full(a.shape, "10+", dtype=object)
    out[a <= 2.5] = "0-2.5"
    out[a == 3] = "3"
    out[(a >= 3.5) & (a <= 6.5)] = "3.5-6.5"
    out[a == 7] = "7"
    out[(a >= 7.5) & (a <= 9.5)] = "7.5-9.5"
    out[np.isnan(a)] = None
    return out


# ------------------------------------------------------------- matrix utils
def stack_pmfs(pmfs: Sequence[IntPMF]) -> tuple[int, np.ndarray]:
    """Stack pmfs onto one common support: returns ``(lo, P)`` with ``P[i, j] = P_i(lo + j)``."""
    lo, hi = min(p.lo for p in pmfs), max(p.hi for p in pmfs)
    mat = np.zeros((len(pmfs), hi - lo + 1))
    for i, p in enumerate(pmfs):
        mat[i, p.lo - lo: p.hi - lo + 1] = p.probs
    return lo, mat


def _tilt_to_mean(mat: np.ndarray, x: np.ndarray, target: np.ndarray, n_iter: int = 30) -> np.ndarray:
    """Exponentially tilt each row of ``mat`` so its mean equals ``target`` (vectorised Newton)."""
    xc = x - x.mean()
    tgt = target - x.mean()
    lam = np.zeros(mat.shape[0])
    for _ in range(n_iter):
        e = lam[:, None] * xc[None, :]
        e -= e.max(axis=1, keepdims=True)
        w = mat * np.exp(e)
        w /= w.sum(axis=1, keepdims=True)
        m = w @ xc
        v = w @ xc**2 - m**2
        step = (m - tgt) / np.maximum(v, 1e-9)
        lam -= np.clip(step, -0.5, 0.5)
        if np.max(np.abs(m - tgt)) < 1e-10:
            break
    e = lam[:, None] * xc[None, :]
    e -= e.max(axis=1, keepdims=True)
    w = mat * np.exp(e)
    return w / w.sum(axis=1, keepdims=True)


def _tilt_to_moments(mat: np.ndarray, x: np.ndarray, target_mean: np.ndarray, target_var: np.ndarray,
                     n_iter: int = 40) -> np.ndarray:
    """Tilt rows by ``exp(a x + b x^2)`` so mean and variance match targets (vectorised 2-d Newton)."""
    s = max(float(np.sqrt(np.max(target_var))), 1.0)
    xc = (x - x.mean()) / s
    t1 = (target_mean - x.mean()) / s
    t2 = target_var / s**2 + t1**2  # target E[xc^2]
    lam = np.zeros((mat.shape[0], 2))
    feats = np.stack([xc, xc**2])  # (2, S)
    for _ in range(n_iter):
        e = lam @ feats
        e -= e.max(axis=1, keepdims=True)
        w = mat * np.exp(e)
        w /= w.sum(axis=1, keepdims=True)
        m1, m2 = w @ xc, w @ xc**2
        m3, m4 = w @ xc**3, w @ xc**4
        g = np.stack([m1 - t1, m2 - t2], axis=1)
        if np.max(np.abs(g)) < 1e-11:
            break
        h11, h12, h22 = m2 - m1**2, m3 - m1 * m2, m4 - m2**2
        det = np.maximum(h11 * h22 - h12**2, 1e-12)
        step = np.stack([(h22 * g[:, 0] - h12 * g[:, 1]) / det, (h11 * g[:, 1] - h12 * g[:, 0]) / det], axis=1)
        lam -= np.clip(step, -0.5, 0.5)
    e = lam @ feats
    e -= e.max(axis=1, keepdims=True)
    w = mat * np.exp(e)
    return w / w.sum(axis=1, keepdims=True)


def _row_moments(mat: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = mat @ x
    return m, mat @ x**2 - m**2


# ----------------------------------------------------------- residual pool
@dataclass
class ResidualPool:
    """Training-fold residuals ``outcome - predicted`` for one target.

    ``ref`` holds the predicted values of the pool rows (used for spread-bucket
    conditioning). Build it with :func:`fit_residual_pool`.
    """

    residuals: np.ndarray
    ref: np.ndarray | None = None
    target: str = "margin"
    min_bucket_n: int = 200
    resolution: float = 0.01
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        r = np.asarray(self.residuals, dtype=float)
        if r.size == 0 or not np.all(np.isfinite(r)):
            raise ValueError("residual pool must be non-empty and finite")
        self.residuals = r
        if self.ref is not None:
            self.ref = np.asarray(self.ref, dtype=float)
            if self.ref.shape != r.shape:
                raise ValueError("ref must align with residuals")
            self._buckets = spread_bucket(self.ref)
        else:
            self._buckets = None

    def __len__(self) -> int:
        return self.residuals.size

    def subset(self, bucket: str | None) -> np.ndarray:
        """Residuals for a bucket, falling back to the full pool if it has < ``min_bucket_n`` rows."""
        if bucket is None or self._buckets is None:
            return self.residuals
        r = self.residuals[self._buckets == bucket]
        return r if r.size >= self.min_bucket_n else self.residuals

    def _frac_pmf(self, frac: float, bucket: str | None) -> IntPMF:
        key = (round(frac, 6), bucket)
        if key not in self._cache:
            v = frac + self.subset(bucket)
            lo_i = np.floor(v)
            w_hi = v - lo_i
            lo = int(lo_i.min())
            idx = (lo_i - lo).astype(np.int64)
            n = int(idx.max()) + 2
            p = np.bincount(idx, weights=1 - w_hi, minlength=n) + np.bincount(idx + 1, weights=w_hi, minlength=n)
            self._cache[key] = IntPMF(lo, p)
        return self._cache[key]

    def sf_continuous(self, t, mean) -> np.ndarray:
        """``P(mean + r > t)`` over the raw (continuous) residuals: the nfl_ats ``shifted`` survival function.

        :meth:`pmf` instead splits each shifted residual between neighbouring
        integers. For a half-point ``t`` that equals the average of this function
        over ``[t - 0.5, t + 0.5]``, and it equals this function exactly when
        ``mean + r`` is an integer.
        """
        r = np.sort(self.residuals)
        x = np.atleast_1d(np.asarray(t, float) - np.asarray(mean, float))
        return 1 - np.searchsorted(r, x, side="right") / r.size

    def pmf(self, mean: float, *, condition_on_bucket: bool = False, min_value: int | None = None) -> IntPMF:
        """pmf of ``mean + residual`` on the integers (linear split, mean preserved up to ``resolution``)."""
        if not np.isfinite(mean):
            raise ValueError("mean must be finite")
        m = np.round(mean / self.resolution) * self.resolution
        k = int(np.floor(m))
        bucket = spread_bucket([mean])[0] if condition_on_bucket else None
        out = self._frac_pmf(m - k, bucket).shift(k)
        if min_value is not None and out.lo < min_value:
            p = out.on_support(min_value, max(out.hi, min_value))
            p[0] += out.probs[: min_value - out.lo].sum()
            out = IntPMF(min_value, p)
        return out


def _outcome(frame: pd.DataFrame, target: str) -> pd.Series:
    if target in frame.columns:
        return frame[target].astype(float)
    if {"home_score", "away_score"} <= set(frame.columns):
        h, a = frame["home_score"].astype(float), frame["away_score"].astype(float)
        return h - a if target == "margin" else h + a
    raise ValueError(f"frame needs a '{target}' column or home_score/away_score")


def fit_residual_pool(
    oof_predictions_with_outcomes: pd.DataFrame,
    before_time,
    *,
    target: str = "margin",
    pred_col: str | None = None,
    time_col: str = "prediction_time_utc",
    label_time_col: str = "label_available_at_utc",
    min_bucket_n: int = 200,
) -> ResidualPool:
    """Residual pool from OOF rows with ``prediction_time < before_time`` only.

    If ``label_time_col`` is present, rows whose label became available at or
    after ``before_time`` are dropped too. Rows with missing outcomes are dropped.
    """
    if target not in ("margin", "total"):
        raise ValueError("target must be 'margin' or 'total'")
    df = oof_predictions_with_outcomes
    pred_col = pred_col or f"pred_{target}"
    cutoff = pd.Timestamp(before_time)
    cutoff = cutoff.tz_convert("UTC") if cutoff.tzinfo else cutoff.tz_localize("UTC")
    keep = pd.to_datetime(df[time_col], utc=True) < cutoff
    if label_time_col in df.columns:
        keep &= pd.to_datetime(df[label_time_col], utc=True) < cutoff
    sub = df.loc[keep]
    y = _outcome(sub, target)
    pred = sub[pred_col].astype(float)
    ok = y.notna() & pred.notna()
    if not ok.any():
        raise ValueError("no training rows before the cutoff")
    return ResidualPool(residuals=(y - pred)[ok].to_numpy(), ref=pred[ok].to_numpy(),
                        target=target, min_bucket_n=min_bucket_n)


# --------------------------------------------------------------- Student-t
@dataclass(frozen=True)
class TParams:
    scale: float
    df: float
    loc: float = 0.0  # residual bias (0 unless fit_loc)

    def scale_for_sd(self, sd: float) -> float:
        """t scale that gives standard deviation ``sd`` at this df (df > 2)."""
        if self.df <= 2:
            raise ValueError("SD undefined for df <= 2")
        return float(sd * np.sqrt((self.df - 2) / self.df))


def fit_student_t(residuals, *, fit_loc: bool = False, df_bounds: tuple[float, float] = (2.5, 200.0)) -> TParams:
    """MLE of t scale/df (and optionally location) on training-fold residuals."""
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 10:
        raise ValueError("need at least 10 residuals to fit a t")
    s0 = float(np.std(r)) or 1.0

    def nll(theta):
        log_s, log_df = theta[0], theta[1]
        loc = theta[2] if fit_loc else 0.0
        return -stats.t.logpdf(r, df=np.exp(log_df), loc=loc, scale=np.exp(log_s)).sum()

    x0 = [np.log(s0), np.log(10.0)] + ([float(np.mean(r))] if fit_loc else [])
    bounds = [(np.log(s0) - 3, np.log(s0) + 3), tuple(np.log(df_bounds))] + ([(None, None)] if fit_loc else [])
    res = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    loc = float(res.x[2]) if fit_loc else 0.0
    return TParams(scale=float(np.exp(res.x[0])), df=float(np.exp(res.x[1])), loc=loc)


def discretize_cdf(cdf: Callable[[np.ndarray], np.ndarray], lo: int, hi: int) -> IntPMF:
    """``P(k) = F(k+0.5) - F(k-0.5)`` on ``lo..hi``; tail mass folded into the end points."""
    edges = np.arange(lo, hi + 2) - 0.5
    c = cdf(edges)
    p = np.diff(c)
    p[0] += c[0]
    p[-1] += 1 - c[-1]
    return IntPMF(lo, np.clip(p, 0, None))


def t_pmf(mean: float, params: TParams, *, sd: float | None = None,
          support: tuple[int, int] = MARGIN_SUPPORT) -> IntPMF:
    """Discretised Student-t margin pmf. ``sd`` (e.g. the model's own predictive SD) overrides the fitted scale."""
    scale = params.scale if sd is None or not np.isfinite(sd) else params.scale_for_sd(sd)
    dist = stats.t(df=params.df, loc=mean + params.loc, scale=scale)
    return discretize_cdf(dist.cdf, *support)


# ---------------------------------------------------------- key numbers
@dataclass(frozen=True)
class KeyNumberWeights:
    """Multiplicative weights on mass at key numbers, fit on training-fold rows.

    ``weights`` maps a bucket label (or ``"all"``) to an array aligned with ``keys``.
    With ``fold_abs`` the weight for ``k`` applies to both ``+k`` and ``-k``.
    """

    keys: tuple[int, ...]
    weights: Mapping[str, np.ndarray]
    fold_abs: bool = True
    n_train: Mapping[str, int] = field(default_factory=dict)

    def for_bucket(self, bucket: str | None) -> np.ndarray:
        if bucket is not None and bucket in self.weights:
            return self.weights[bucket]
        return self.weights["all"]

    def as_frame(self) -> pd.DataFrame:
        rows = [{"bucket": b, "key": k, "weight": float(w[i]), "n_train": self.n_train.get(b)}
                for b, w in self.weights.items() for i, k in enumerate(self.keys)]
        return pd.DataFrame(rows)


def _key_masks(lo: int, width: int, keys: Sequence[int], fold_abs: bool) -> np.ndarray:
    x = np.arange(lo, lo + width)
    xv = np.abs(x) if fold_abs else x
    return np.stack([xv == k for k in keys])  # (K, S)


def _reweight(mat: np.ndarray, masks: np.ndarray, w_rows: np.ndarray) -> np.ndarray:
    """Multiply mass at key positions by per-row weights ``w_rows`` (n, K) and renormalise."""
    factor = np.ones_like(mat)
    for j in range(masks.shape[0]):
        factor[:, masks[j]] = w_rows[:, [j]]
    out = mat * factor
    return out / out.sum(axis=1, keepdims=True)


def fit_key_weights(
    base_pmfs: Sequence[IntPMF] | tuple[int, np.ndarray],
    outcomes,
    *,
    keys: Sequence[int] = DEFAULT_MARGIN_KEYS,
    ref=None,
    by_bucket: bool = False,
    min_bucket_n: int = 300,
    fold_abs: bool = True,
    smoothing: float = 0.5,
    n_iter: int = 4,
    bounds: tuple[float, float] = (0.02, 5.0),
) -> KeyNumberWeights:
    """Fit key weights on **training-fold** rows: base pmfs and realised outcomes.

    Iteratively sets ``w_k <- w_k * (obs_k + s) / (pred_k + s)`` where ``obs_k``
    is the count of outcomes at the key and ``pred_k`` the summed predicted mass
    after the current reweighting.
    """
    lo, mat = base_pmfs if isinstance(base_pmfs, tuple) else stack_pmfs(base_pmfs)
    y = np.asarray(outcomes, dtype=float)
    if y.shape[0] != mat.shape[0]:
        raise ValueError("outcomes must align with base pmfs")
    keys = tuple(int(k) for k in keys)
    masks = _key_masks(lo, mat.shape[1], keys, fold_abs)
    yv = np.abs(y) if fold_abs else y
    groups: dict[str, np.ndarray] = {"all": np.ones(len(y), bool)}
    if by_bucket:
        if ref is None:
            raise ValueError("by_bucket needs ref values")
        b = spread_bucket(ref)
        for name in SPREAD_BUCKETS:
            sel = b == name
            if sel.sum() >= min_bucket_n:
                groups[name] = sel
    weights, n_train = {}, {}
    for name, sel in groups.items():
        m, yy = mat[sel], yv[sel]
        obs = np.array([(yy == k).sum() for k in keys], dtype=float)
        w = np.ones(len(keys))
        for _ in range(n_iter):
            adj = _reweight(m, masks, np.broadcast_to(w, (m.shape[0], len(keys))))
            pred = (adj @ masks.T.astype(float)).sum(axis=0)
            w = np.clip(w * (obs + smoothing) / (pred + smoothing), *bounds)
        weights[name], n_train[name] = w, int(sel.sum())
    return KeyNumberWeights(keys=keys, weights=weights, fold_abs=fold_abs, n_train=n_train)


def apply_key_weights_matrix(lo: int, mat: np.ndarray, kw: KeyNumberWeights, *, ref=None,
                             preserve_mean: bool = True, preserve_var: bool = True,
                             strength: float = 1.0) -> np.ndarray:
    """Vectorised :func:`apply_key_weights` on a stacked pmf matrix."""
    masks = _key_masks(lo, mat.shape[1], kw.keys, kw.fold_abs)
    if ref is not None and len(kw.weights) > 1:
        buckets = spread_bucket(ref)
        w_rows = np.stack([kw.for_bucket(b) for b in buckets])
    else:
        w_rows = np.broadcast_to(kw.weights["all"], (mat.shape[0], len(kw.keys)))
    w_rows = np.asarray(w_rows) ** strength
    out = _reweight(mat, masks, w_rows)
    x = np.arange(lo, lo + mat.shape[1], dtype=float)
    if preserve_var:
        out = _tilt_to_moments(out, x, *_row_moments(mat, x))
    elif preserve_mean:
        out = _tilt_to_mean(out, x, mat @ x)
    return out


def apply_key_weights(pmf: IntPMF, kw: KeyNumberWeights, *, bucket_ref: float | None = None,
                      preserve_mean: bool = True, preserve_var: bool = True, strength: float = 1.0) -> IntPMF:
    """Reweight key-number mass; total mass exactly 1.

    The result is exponentially tilted so its mean (and with ``preserve_var``
    its variance) equals the pre-adjustment pmf's.
    """
    ref = None if bucket_ref is None else [bucket_ref]
    out = apply_key_weights_matrix(pmf.lo, pmf.probs[None, :], kw, ref=ref, preserve_mean=preserve_mean,
                                   preserve_var=preserve_var, strength=strength)
    return IntPMF(pmf.lo, out[0])


def adjust_to_key_targets(pmf: IntPMF, targets: Mapping[int, float], *, fold_abs: bool = True,
                          preserve_mean: bool = True, n_iter: int = 20) -> IntPMF:
    """Move ``P(|M|=k)`` to given target probabilities (e.g. training-fold frequencies) for one pmf."""
    keys = tuple(int(k) for k in targets)
    tgt = np.array([targets[k] for k in keys], dtype=float)
    if np.any(tgt < 0) or tgt.sum() >= 1:
        raise ValueError("targets must be non-negative and sum to < 1")
    masks = _key_masks(pmf.lo, pmf.probs.size, keys, fold_abs)
    x = pmf.support.astype(float)
    cur = pmf.probs[None, :].copy()
    for _ in range(n_iter):
        have = (cur @ masks.T.astype(float))[0]
        if np.any((have <= 0) & (tgt > 0)):
            raise ValueError("cannot move mass to a key with zero base mass")
        w = np.where(have > 0, tgt / np.where(have > 0, have, 1), 1.0)
        cur = _reweight(cur, masks, w[None, :])
        if preserve_mean:
            cur = _tilt_to_mean(cur, x, np.array([pmf.mean()]))
        if np.max(np.abs((cur @ masks.T.astype(float))[0] - tgt)) < 1e-9:
            break
    return IntPMF(pmf.lo, cur[0])


# -------------------------------------------------------------- front door
def margin_pmf(mean: float, *, family: str = "t", pool: ResidualPool | None = None,
               t_params: TParams | None = None, sd: float | None = None,
               key_weights: KeyNumberWeights | None = None, condition_on_bucket: bool = False,
               support: tuple[int, int] = MARGIN_SUPPORT) -> IntPMF:
    """Margin pmf for one game from its predicted mean (and optional SD).

    ``family='empirical'`` needs ``pool``; ``family='t'`` needs ``t_params``.
    ``key_weights`` (fit in-fold) are applied afterwards if given.
    """
    if family == "empirical":
        if pool is None:
            raise ValueError("empirical family needs a residual pool")
        out = pool.pmf(mean, condition_on_bucket=condition_on_bucket)
    elif family == "t":
        if t_params is None:
            raise ValueError("t family needs fitted TParams")
        out = t_pmf(mean, t_params, sd=sd, support=support)
    else:
        raise ValueError(f"unknown margin family {family!r}")
    if key_weights is not None:
        out = apply_key_weights(out, key_weights, bucket_ref=mean)
    return out


# ---------------------------------------------------- vectorised builders
def t_pmf_matrix(means, params: TParams, *, sds=None, support: tuple[int, int] = MARGIN_SUPPORT,
                 fold_tails: bool = True) -> tuple[int, np.ndarray]:
    """Discretised t pmfs for many means at once (same result as :func:`t_pmf` row by row).

    ``params.df = inf`` gives the discretised normal. ``fold_tails=False`` drops the
    mass beyond the support and renormalises instead of adding it to the end
    points (this is what nfl_ats ``keynum`` does).
    """
    mu = np.asarray(means, dtype=float)
    scale = np.full(mu.shape, params.scale)
    if sds is not None:
        sds = np.asarray(sds, dtype=float)
        ok = np.isfinite(sds)
        if ok.any():
            scale[ok] = sds[ok] * (np.sqrt((params.df - 2) / params.df) if np.isfinite(params.df) else 1.0)
    lo, hi = support
    edges = np.arange(lo, hi + 2) - 0.5
    z = (edges[None, :] - (mu[:, None] + params.loc)) / scale[:, None]
    c = stats.norm.cdf(z) if not np.isfinite(params.df) else stats.t.cdf(z, df=params.df)
    p = np.diff(c, axis=1)
    if fold_tails:
        p[:, 0] += c[:, 0]
        p[:, -1] += 1 - c[:, -1]
    p = np.clip(p, 0, None)
    return lo, p / p.sum(axis=1, keepdims=True)


def normal_pmf(mean: float, sd: float, *, support: tuple[int, int] = MARGIN_SUPPORT) -> IntPMF:
    """Discretised normal ``Phi((k+0.5-mu)/sd) - Phi((k-0.5-mu)/sd)``."""
    lo, mat = t_pmf_matrix([mean], TParams(scale=sd, df=np.inf), support=support)
    return IntPMF(lo, mat[0])


# ------------------------------------------- ported from nfl_ats (owner repo)
# Provenance: wbblanck/nfl_ats src/engine/models/margin_dist.py (MarginDist kinds
# "kernel", "kernel_sym", "keynum"; choose_h; TotalDist "shifted_total") and
# docs/phase4_preregistration.md, docs/keynumber_preregistration.md. Their
# conditioning variable is the market home spread c (negative = home favoured)
# with predicted margin -c; here it is the model's predicted home margin
# mu = -c, so "c_i - c" becomes "mu_i - mu" (same distance). Margin = home - away
# in both repos. Their survival-function API is re-expressed as integer pmfs.

#: Bandwidth grid of nfl_ats ``choose_h``.
KERNEL_H_GRID: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
#: nfl_ats keynum: weights for |k| = 0..20, w = 1 beyond.
KEYNUM_KEYS: tuple[int, ...] = tuple(range(21))


@dataclass
class KernelLocalEmpirical:
    """Kernel-local empirical margin distribution (nfl_ats estimator K / KS).

    ``P(M = k | mu) = sum_i w_i 1[m_i = k] / sum_i w_i`` with
    ``w_i = exp(-0.5 ((mu_i - mu) / h)^2)`` over training-fold games ``i``
    (predicted margin ``mu_i``, realised integer margin ``m_i``). Realised
    margins are used unshifted, so key-number mass (3, 7, 10, 14) and the low
    tie rate stay where they occur in real outcomes. ``mirrored=True`` adds each
    training game again as ``(-mu_i, -m_i)`` (nfl_ats ``kernel_sym``: assumes
    favourites and underdogs are symmetric around the line, doubles the sample).

    Note: the pmf mean is the kernel-weighted local mean of outcomes, not
    exactly ``mu``. For ``mu`` far outside the training range the nearest
    training games dominate (weights are normalised in log space).
    """

    pred: np.ndarray
    margin: np.ndarray
    h: float = 1.0
    mirrored: bool = False

    def __post_init__(self):
        p, m = np.asarray(self.pred, float), np.asarray(self.margin, float)
        ok = np.isfinite(p) & np.isfinite(m)
        p, m = p[ok], np.round(m[ok]).astype(np.int64)
        if p.size == 0:
            raise ValueError("kernel estimator needs training games")
        if self.mirrored:
            p, m = np.concatenate([p, -p]), np.concatenate([m, -m])
        if self.h <= 0:
            raise ValueError("bandwidth must be positive")
        self.pred, self.margin = p, m
        self._lo = int(m.min())
        self._onehot_idx = m - self._lo
        self._width = int(m.max()) - self._lo + 1
        self._onehot = np.zeros((m.size, self._width))
        self._onehot[np.arange(m.size), self._onehot_idx] = 1.0

    def pmf_matrix(self, means, chunk: int = 2000) -> tuple[int, np.ndarray]:
        mu = np.asarray(means, dtype=float)
        out = np.zeros((mu.size, self._width))
        for s in range(0, mu.size, chunk):
            e = -0.5 * ((self.pred[None, :] - mu[s:s + chunk, None]) / self.h) ** 2
            w = np.exp(e - e.max(axis=1, keepdims=True))
            blk = w @ self._onehot
            out[s:s + chunk] = blk / blk.sum(axis=1, keepdims=True)
        return self._lo, out

    def pmf(self, mean: float) -> IntPMF:
        lo, mat = self.pmf_matrix([mean])
        return IntPMF(lo, mat[0])


def choose_kernel_h(pred, margin, season, *, mirrored: bool = False,
                    grid: Sequence[float] = KERNEL_H_GRID, n_val_seasons: int = 3) -> float:
    """nfl_ats ``choose_h``: bandwidth by log loss on the last ``n_val_seasons`` training seasons.

    Fit on earlier training seasons; target ``1[m > t]`` with ``t = mu ± 0.5``
    alternating (the tail around the game's own line). Training data only.
    Falls back to ``1.0`` when fewer than ``n_val_seasons + 1`` seasons exist.
    """
    pred, margin, season = (np.asarray(x) for x in (pred, margin, season))
    seasons = np.unique(season)
    if seasons.size <= n_val_seasons:
        return 1.0
    val = np.isin(season, seasons[-n_val_seasons:])
    t = pred[val] + np.where(np.arange(val.sum()) % 2 == 0, 0.5, -0.5)
    y = (margin[val] > t).astype(float)
    best, best_ll = grid[0], np.inf
    for h in grid:
        est = KernelLocalEmpirical(pred[~val], margin[~val], h=h, mirrored=mirrored)
        lo, mat = est.pmf_matrix(pred[val])
        x = np.arange(lo, lo + mat.shape[1])
        p = np.clip((mat * (x[None, :] > t[:, None])).sum(axis=1), 1e-4, 1 - 1e-4)
        ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        if ll < best_ll - 1e-12:
            best, best_ll = h, ll
    return float(best)


def fit_keynum(pred, margin, *, sigma: float | None = None, fold_tails: bool = True) -> tuple[TParams, KeyNumberWeights]:
    """nfl_ats ``keynum``: discretised normal (sd of training residuals) times weights ``w_|k|``.

    ``w_|k| = observed count of |m| = |k| / expected count under the base``, for
    |k| = 0..20, one pass, no smoothing; apply with :func:`apply_key_weights`
    (``preserve_mean=False, preserve_var=False``) to reproduce nfl_ats exactly.
    """
    pred, margin = np.asarray(pred, float), np.round(np.asarray(margin, float))
    sd = float(np.std(margin - pred)) if sigma is None else float(sigma)
    params = TParams(scale=sd, df=np.inf)
    lo, mat = t_pmf_matrix(pred, params, fold_tails=fold_tails)
    kw = fit_key_weights((lo, mat), margin, keys=KEYNUM_KEYS, smoothing=0.0, n_iter=1, bounds=(0.0, np.inf))
    return params, kw


@dataclass
class TotalScaledPool:
    """Shifted empirical margin pmf whose width scales with the game total (nfl_ats ``TotalDist``).

    ``s(T) = sigma * (alpha + beta T) / mean|r|`` with ``|r|`` regressed on ``T``
    by least squares on training rows; standardised residuals ``z = r / s(T)``
    are re-scaled to the game's ``s(T)``. In nfl_ats ``T`` is the market total;
    here it is the model's ``pred_total`` (never the nflverse total_line).
    """

    alpha: float
    beta: float
    sigma: float
    mean_abs: float
    z: np.ndarray

    @classmethod
    def fit(cls, pred_margin, margin, total_ref) -> "TotalScaledPool":
        p, m, T = (np.asarray(x, float) for x in (pred_margin, margin, total_ref))
        ok = np.isfinite(p) & np.isfinite(m) & np.isfinite(T)
        p, m, T = p[ok], m[ok], T[ok]
        r = m - p
        a = np.abs(r)
        alpha, beta = np.linalg.lstsq(np.column_stack([np.ones_like(T), T]), a, rcond=None)[0]
        obj = cls(alpha=float(alpha), beta=float(beta), sigma=float(np.std(r)), mean_abs=float(a.mean()), z=np.zeros(0))
        obj.z = r / obj.scale(T)
        return obj

    def scale(self, T):
        s = self.sigma * (self.alpha + self.beta * np.asarray(T, float)) / self.mean_abs
        return np.maximum(s, 0.25 * self.sigma)  # guard against a negative fitted line far out of range

    def pmf(self, mean: float, total_ref: float) -> IntPMF:
        v = mean + self.z * float(self.scale(total_ref))
        lo_i = np.floor(v)
        w_hi = v - lo_i
        lo = int(lo_i.min())
        idx = (lo_i - lo).astype(np.int64)
        n = int(idx.max()) + 2
        p = np.bincount(idx, weights=1 - w_hi, minlength=n) + np.bincount(idx + 1, weights=w_hi, minlength=n)
        return IntPMF(lo, p)
