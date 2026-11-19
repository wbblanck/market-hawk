"""Discrete integer pmfs for margin, total and joint (home_pts, away_pts).

Conventions follow ``nfl_market.models.interface``: margin ``M = home - away``,
total ``T = home + away``. Supports are contiguous integer ranges stored as
``lo`` plus a probability vector; gaps are zeros.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

#: Absolute tolerance on the raw mass before normalisation.
MASS_TOL = 1e-6


def _clean_probs(probs, *, normalise: bool, tol: float = MASS_TOL) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("pmf probabilities must be a non-empty 1-d array")
    if not np.all(np.isfinite(p)):
        raise ValueError("pmf probabilities must be finite")
    if np.any(p < -1e-12):
        raise ValueError("pmf probabilities must be non-negative")
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s <= 0:
        raise ValueError("pmf has zero total mass")
    if not normalise and abs(s - 1.0) > tol:
        raise ValueError(f"pmf mass {s:.8f} differs from 1 by more than {tol}")
    return p / s


@dataclass(frozen=True, eq=False)
class IntPMF:
    """pmf over a contiguous integer support ``lo, lo+1, ..., lo+len(probs)-1``.

    Use :meth:`from_arrays` for arbitrary (possibly unsorted, gappy) supports.
    The probabilities are always normalised to sum to exactly 1 (floating point).
    """

    lo: int
    probs: np.ndarray

    def __init__(self, lo: int, probs, *, normalise: bool = True):
        if int(lo) != lo:
            raise ValueError("lo must be an integer")
        p = _clean_probs(probs, normalise=normalise)
        p.setflags(write=False)
        object.__setattr__(self, "lo", int(lo))
        object.__setattr__(self, "probs", p)

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_arrays(cls, support: Sequence[int], probs: Sequence[float], *, normalise: bool = True) -> "IntPMF":
        s = np.asarray(support, dtype=float)
        if s.shape != np.shape(probs):
            raise ValueError("support and probs must have the same shape")
        if not np.all(np.isfinite(s)) or np.any(s != np.round(s)):
            raise ValueError("support must be integers")
        s = s.astype(np.int64)
        lo, hi = int(s.min()), int(s.max())
        p = np.zeros(hi - lo + 1)
        np.add.at(p, s - lo, np.asarray(probs, dtype=float))
        return cls(lo, p, normalise=normalise)

    @classmethod
    def point(cls, k: int) -> "IntPMF":
        return cls(int(k), [1.0])

    @classmethod
    def from_samples(cls, values: Iterable[int], weights=None) -> "IntPMF":
        v = np.asarray(list(values), dtype=float)
        w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=float)
        return cls.from_arrays(v, w)

    # ------------------------------------------------------------- properties
    @property
    def hi(self) -> int:
        return self.lo + self.probs.size - 1

    @property
    def support(self) -> np.ndarray:
        return np.arange(self.lo, self.hi + 1)

    def mean(self) -> float:
        return float(np.dot(self.support, self.probs))

    def var(self) -> float:
        m = self.mean()
        return float(np.dot((self.support - m) ** 2, self.probs))

    def sd(self) -> float:
        return float(np.sqrt(self.var()))

    # ---------------------------------------------------------- probabilities
    def pmf(self, k) -> np.ndarray | float:
        """P(X = k); 0 for non-integer or out-of-support k."""
        k_arr = np.asarray(k, dtype=float)
        out = np.zeros(k_arr.shape)
        is_int = np.isfinite(k_arr) & (k_arr == np.round(k_arr))
        idx = np.where(is_int, k_arr, self.lo - 1).astype(np.int64) - self.lo
        ok = is_int & (idx >= 0) & (idx < self.probs.size)
        out[ok] = self.probs[idx[ok]]
        return float(out) if out.ndim == 0 else out

    def cdf(self, x) -> np.ndarray | float:
        """P(X <= x) for real x."""
        x_arr = np.asarray(x, dtype=float)
        c = np.concatenate([[0.0], np.cumsum(self.probs)])
        c[-1] = 1.0
        idx = np.floor(x_arr).astype(np.int64) - self.lo + 1
        out = c[np.clip(idx, 0, self.probs.size)]
        return float(out) if out.ndim == 0 else out

    def prob_gt(self, x: float) -> float:
        """P(X > x)."""
        return float(self.probs[self.support > x + 1e-9].sum())

    def prob_lt(self, x: float) -> float:
        """P(X < x)."""
        return float(self.probs[self.support < x - 1e-9].sum())

    def prob_eq(self, x: float) -> float:
        return float(self.probs[np.abs(self.support - x) <= 1e-9].sum())

    def quantile(self, q: float) -> int:
        """Smallest k with P(X <= k) >= q."""
        if not 0 <= q <= 1:
            raise ValueError("q must be in [0, 1]")
        c = np.cumsum(self.probs)
        idx = int(np.searchsorted(c, q - 1e-12, side="left"))
        return self.lo + min(idx, self.probs.size - 1)

    def central_interval(self, level: float) -> tuple[int, int]:
        """Central interval [Q((1-l)/2), Q((1+l)/2)]; covers at least ``level``."""
        a = (1 - level) / 2
        return self.quantile(a), self.quantile(1 - a)

    # ------------------------------------------------------------ transforms
    def abs(self) -> "IntPMF":
        return IntPMF.from_arrays(np.abs(self.support), self.probs)

    def shift(self, k: int) -> "IntPMF":
        return IntPMF(self.lo + int(k), self.probs)

    def negate(self) -> "IntPMF":
        return IntPMF(-self.hi, self.probs[::-1])

    def trim(self, eps: float = 0.0) -> "IntPMF":
        """Drop leading/trailing zero (or <= eps) mass and renormalise."""
        nz = np.nonzero(self.probs > eps)[0]
        return IntPMF(self.lo + int(nz[0]), self.probs[nz[0]: nz[-1] + 1])

    def on_support(self, lo: int, hi: int) -> np.ndarray:
        """Probabilities aligned to ``lo..hi`` (mass outside is dropped, not moved)."""
        out = np.zeros(hi - lo + 1)
        a, b = max(lo, self.lo), min(hi, self.hi)
        if a <= b:
            out[a - lo: b - lo + 1] = self.probs[a - self.lo: b - self.lo + 1]
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"IntPMF(lo={self.lo}, hi={self.hi}, mean={self.mean():.3f}, sd={self.sd():.3f})"


def mixture(pmfs: Sequence[IntPMF], weights=None) -> IntPMF:
    """Finite mixture of pmfs."""
    if not pmfs:
        raise ValueError("mixture needs at least one pmf")
    w = np.ones(len(pmfs)) if weights is None else np.asarray(weights, dtype=float)
    lo, hi = min(p.lo for p in pmfs), max(p.hi for p in pmfs)
    acc = np.zeros(hi - lo + 1)
    for wi, p in zip(w, pmfs):
        acc[p.lo - lo: p.hi - lo + 1] += wi * p.probs
    return IntPMF(lo, acc)


@dataclass(frozen=True, eq=False)
class JointPMF:
    """Joint pmf of (home_pts, away_pts) on ``0..H`` x ``0..A`` (matrix ``P[h, a]``).

    Margin and total pmfs derived from the same matrix are coherent by
    construction.
    """

    probs: np.ndarray

    def __init__(self, probs, *, normalise: bool = True):
        p = np.asarray(probs, dtype=float)
        if p.ndim != 2 or p.size == 0:
            raise ValueError("joint pmf must be a non-empty 2-d array")
        flat = _clean_probs(p.ravel(), normalise=normalise)
        p = flat.reshape(p.shape)
        p.setflags(write=False)
        object.__setattr__(self, "probs", p)

    @classmethod
    def independent(cls, home: IntPMF, away: IntPMF) -> "JointPMF":
        if home.lo < 0 or away.lo < 0:
            raise ValueError("team points must be non-negative")
        h = home.on_support(0, home.hi)
        a = away.on_support(0, away.hi)
        return cls(np.outer(h, a))

    def _grids(self):
        h = np.arange(self.probs.shape[0])[:, None]
        a = np.arange(self.probs.shape[1])[None, :]
        return h, a

    def home(self) -> IntPMF:
        return IntPMF(0, self.probs.sum(axis=1))

    def away(self) -> IntPMF:
        return IntPMF(0, self.probs.sum(axis=0))

    def margin(self) -> IntPMF:
        h, a = self._grids()
        d = (h - a).ravel()
        lo = int(d.min())
        return IntPMF(lo, np.bincount(d - lo, weights=self.probs.ravel()))

    def total(self) -> IntPMF:
        h, a = self._grids()
        t = (h + a).ravel()
        return IntPMF(0, np.bincount(t, weights=self.probs.ravel()))

    def cov(self) -> float:
        h, a = self._grids()
        mh, ma = self.home().mean(), self.away().mean()
        return float(((h - mh) * (a - ma) * self.probs).sum())

    def corr(self) -> float:
        return self.cov() / np.sqrt(self.home().var() * self.away().var())

    def pmf(self, home_pts: int, away_pts: int) -> float:
        if 0 <= home_pts < self.probs.shape[0] and 0 <= away_pts < self.probs.shape[1]:
            return float(self.probs[home_pts, away_pts])
        return 0.0
