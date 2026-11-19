"""Distribution and probability calibration metrics (calibration_check.md).

Every function takes pmfs either as a sequence of :class:`IntPMF` or as a
stacked ``(lo, matrix)`` pair (see :func:`~nfl_market.distributions.margin.stack_pmfs`)
plus aligned integer outcomes. Nothing here fits anything: these are pure
scoring functions to be run on held-out walk-forward predictions only.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .margin import SPREAD_BUCKETS, spread_bucket, stack_pmfs
from .pmf import IntPMF

PMFs = Sequence[IntPMF] | tuple[int, np.ndarray]
PROB_CLIP = 1e-6
MIN_BIN_N = 30


def as_matrix(pmfs: PMFs) -> tuple[int, np.ndarray]:
    if isinstance(pmfs, tuple) and len(pmfs) == 2 and isinstance(pmfs[1], np.ndarray):
        return int(pmfs[0]), pmfs[1]
    return stack_pmfs(list(pmfs))


def _outcomes(y, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.shape != (n,):
        raise ValueError(f"expected {n} outcomes, got shape {y.shape}")
    if not np.all(np.isfinite(y)) or np.any(y != np.round(y)):
        raise ValueError("outcomes must be finite integers")
    return y.astype(np.int64)


def pmf_at(pmfs: PMFs, y) -> np.ndarray:
    """``P_i(Y = y_i)`` (0 outside support)."""
    lo, mat = as_matrix(pmfs)
    y = _outcomes(y, mat.shape[0])
    idx = y - lo
    ok = (idx >= 0) & (idx < mat.shape[1])
    out = np.zeros(len(y))
    out[ok] = mat[np.nonzero(ok)[0], idx[ok]]
    return out


def cdf_at(pmfs: PMFs, x) -> np.ndarray:
    """``P_i(Y <= x_i)`` for real ``x_i``."""
    lo, mat = as_matrix(pmfs)
    x = np.asarray(x, dtype=float)
    cum = np.concatenate([np.zeros((mat.shape[0], 1)), np.cumsum(mat, axis=1)], axis=1)
    cum[:, -1] = 1.0
    idx = np.clip(np.floor(x).astype(np.int64) - lo + 1, 0, mat.shape[1])
    return cum[np.arange(mat.shape[0]), idx]


# ----------------------------------------------------------------- scores
def log_scores(pmfs: PMFs, outcomes, floor: float = 1e-12) -> np.ndarray:
    """Per-row negative log pmf of the realised outcome (floored to avoid inf)."""
    return -np.log(np.maximum(pmf_at(pmfs, outcomes), floor))


def log_score(pmfs: PMFs, outcomes, floor: float = 1e-12) -> float:
    """Mean negative log pmf of realised integer outcomes (lower is better)."""
    return float(np.mean(log_scores(pmfs, outcomes, floor)))


def randomized_pit(pmfs: PMFs, outcomes, seed: int = 0) -> np.ndarray:
    """``u = F(y-1) + V (F(y) - F(y-1))``, ``V ~ U(0,1)`` from a seeded generator."""
    y = np.asarray(outcomes, dtype=float)
    f_hi = cdf_at(pmfs, y)
    f_lo = cdf_at(pmfs, y - 1)
    v = np.random.default_rng(seed).uniform(size=len(y))
    return f_lo + v * (f_hi - f_lo)


def pit_histogram(u, n_bins: int = 10) -> pd.DataFrame:
    u = np.asarray(u, dtype=float)
    counts, edges = np.histogram(u, bins=n_bins, range=(0, 1))
    return pd.DataFrame({"bin_lo": edges[:-1], "bin_hi": edges[1:], "count": counts,
                         "frac": counts / max(len(u), 1), "expected_frac": 1 / n_bins})


def pit_summary(u) -> dict:
    """Mean (0.5), variance (1/12 = 0.0833; larger = too narrow), KS distance to U(0,1)."""
    u = np.asarray(u, dtype=float)
    ks = stats.kstest(u, "uniform")
    return {"n": int(len(u)), "pit_mean": float(u.mean()), "pit_var": float(u.var()),
            "pit_ks": float(ks.statistic), "pit_ks_p": float(ks.pvalue)}


def interval_coverage(pmfs: PMFs, outcomes, levels: Sequence[float] = (0.5, 0.8, 0.95)) -> pd.DataFrame:
    """Empirical coverage of central intervals ``[Q((1-l)/2), Q((1+l)/2)]``.

    Discrete intervals hold at least the nominal mass; ``predicted_mass`` is the
    mean model mass inside each interval, the exact comparator for ``coverage``.
    """
    lo, mat = as_matrix(pmfs)
    y = _outcomes(outcomes, mat.shape[0])
    cum = np.cumsum(mat, axis=1)
    rows = []
    for lev in levels:
        a = (1 - lev) / 2
        qlo = lo + np.argmax(cum >= a - 1e-12, axis=1)
        qhi = lo + np.argmax(cum >= 1 - a - 1e-12, axis=1)
        inside = (y >= qlo) & (y <= qhi)
        mass = cdf_at((lo, mat), qhi) - cdf_at((lo, mat), qlo - 1)
        rows.append({"level": lev, "n": int(len(y)), "coverage": float(inside.mean()),
                     "predicted_mass": float(mass.mean()), "mean_width": float(np.mean(qhi - qlo + 1))})
    return pd.DataFrame(rows)


def key_number_table(pmfs: PMFs, outcomes, ref_spread=None, keys: Sequence[int] = (3, 7, 10)) -> pd.DataFrame:
    """Mean predicted vs observed ``P(|M| = k)``, overall and by bucket of ``|ref_spread|``.

    ``ref_spread`` is a home spread (home's own line, negative = home favoured).
    Besides the six preregistered buckets, a ``home_fav_4-6`` slice (home line in
    [-6, -4]) matches the nfl_ats key-number report, and signed columns
    (``M = +k`` home wins by k, ``M = -k``) are given for every slice. The
    ``all`` row is directly comparable with nfl_ats "margins of exactly 3 / 7"
    (14.2% / 8.7% observed there).
    """
    lo, mat = as_matrix(pmfs)
    y = _outcomes(outcomes, mat.shape[0])
    x = np.arange(lo, lo + mat.shape[1])
    groups = [("all", np.ones(len(y), bool))]
    if ref_spread is not None:
        ref = np.asarray(ref_spread, dtype=float)
        b = spread_bucket(ref)
        groups += [(name, b == name) for name in SPREAD_BUCKETS]
        groups.append(("home_fav_4-6", (ref >= -6) & (ref <= -4)))
    rows = []
    for name, sel in groups:
        n = int(sel.sum())
        for k in keys:
            row = {"bucket": name, "key": k, "n": n}
            if n:
                pp = mat[sel][:, x == k].sum(axis=1).mean() if (x == k).any() else 0.0
                pn = mat[sel][:, x == -k].sum(axis=1).mean() if k and (x == -k).any() else 0.0
                op, on = float(np.mean(y[sel] == k)), float(np.mean(y[sel] == -k)) if k else 0.0
                row.update(predicted=float(pp + pn), observed=op + on, diff=op + on - float(pp + pn),
                           pred_pos=float(pp), obs_pos=op, pred_neg=float(pn), obs_neg=on)
            else:
                row.update(predicted=np.nan, observed=np.nan, diff=np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------ binary probabilities
def binary_metrics(p, y, *, n_bins: int = 10, strategy: str = "uniform") -> tuple[dict, pd.DataFrame]:
    """Brier, log loss (clipped, clip count reported), ECE and a reliability table with counts."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.shape != y.shape:
        raise ValueError("p and y must align")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("y must be 0/1")
    pc = np.clip(p, PROB_CLIP, 1 - PROB_CLIP)
    clip_count = int(np.sum(pc != p))
    if strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows, ece = [], 0.0
    for i in range(len(edges) - 1):
        sel = b == i
        n = int(sel.sum())
        mp = float(p[sel].mean()) if n else np.nan
        my = float(y[sel].mean()) if n else np.nan
        if n:
            ece += n / len(p) * abs(my - mp)
        rows.append({"bin_lo": edges[i], "bin_hi": edges[i + 1], "n": n, "mean_pred": mp,
                     "observed": my, "flag_small": n < MIN_BIN_N})
    summary = {"n": int(len(p)), "brier": float(np.mean((p - y) ** 2)),
               "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
               "ece": float(ece), "clip_count": clip_count}
    return summary, pd.DataFrame(rows)


def _threshold_probs(pmfs: PMFs, thr) -> tuple[np.ndarray, np.ndarray]:
    """``P(X > thr_i)`` and ``P(X = thr_i)`` row-wise."""
    thr = np.asarray(thr, dtype=float)
    gt = 1 - cdf_at(pmfs, thr)
    whole = np.abs(thr - np.round(thr)) < 1e-9
    eq = np.zeros(len(thr))
    if whole.any():
        lo, mat = as_matrix(pmfs)
        eq[whole] = pmf_at((lo, mat[whole]), np.round(thr[whole]))
    return gt, eq


def spread_outcome_probs(pmfs: PMFs, home_spreads) -> pd.DataFrame:
    """Row-wise home P(cover), P(push), P(no cover) at each game's home line."""
    gt, eq = _threshold_probs(pmfs, -np.asarray(home_spreads, dtype=float))
    return pd.DataFrame({"p_cover": gt, "p_push": eq, "p_no_cover": np.clip(1 - gt - eq, 0, 1)})


def push_calibration(pmfs: PMFs, outcomes, lines, *, kind: str = "spread") -> pd.DataFrame:
    """Mean predicted vs observed push rate grouped by whole-number line (half lines skipped).

    ``kind='spread'``: ``lines`` are home spreads, push iff ``M + h = 0``.
    ``kind='total'``: push iff ``T = L``.
    """
    lines = np.asarray(lines, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    thr = -lines if kind == "spread" else lines
    if kind not in ("spread", "total"):
        raise ValueError("kind must be 'spread' or 'total'")
    _, eq = _threshold_probs(pmfs, thr)
    whole = np.abs(lines - np.round(lines)) < 1e-9
    df = pd.DataFrame({"line": np.round(lines), "p_push": eq, "push": (y == thr).astype(float)})[whole]
    out = df.groupby("line").agg(n=("push", "size"), predicted=("p_push", "mean"), observed=("push", "mean"))
    return out.reset_index()


def cover_calibration(pmfs: PMFs, outcomes, home_spreads, *, n_bins: int = 10,
                      strategy: str = "quantile") -> tuple[dict, pd.DataFrame]:
    """Binary home-cover calibration conditional on no push, pushes excluded and counted."""
    y = np.asarray(outcomes, dtype=float)
    h = np.asarray(home_spreads, dtype=float)
    probs = spread_outcome_probs(pmfs, h)
    push = (y + h) == 0
    p = probs["p_cover"].to_numpy() / np.maximum(1 - probs["p_push"].to_numpy(), 1e-12)
    summary, rel = binary_metrics(p[~push], ((y + h) > 0)[~push].astype(float), n_bins=n_bins, strategy=strategy)
    summary["n_push_excluded"] = int(push.sum())
    return summary, rel


def home_win_calibration(pmfs: PMFs, outcomes, *, n_bins: int = 10, strategy: str = "uniform") -> tuple[dict, pd.DataFrame]:
    """P(home win | no tie) vs outcome on non-tie games (ties excluded and counted)."""
    y = np.asarray(outcomes, dtype=float)
    gt, eq = _threshold_probs(pmfs, np.zeros(len(y)))
    tie = y == 0
    p = gt / np.maximum(1 - eq, 1e-12)
    summary, rel = binary_metrics(p[~tie], (y > 0)[~tie].astype(float), n_bins=n_bins, strategy=strategy)
    summary["n_ties_excluded"] = int(tie.sum())
    return summary, rel


def distribution_summary(pmfs: PMFs, outcomes, *, seed: int = 0) -> dict:
    """Log score, PIT summary and 50/80/95 coverage in one flat dict."""
    lo, mat = as_matrix(pmfs)
    out = {"log_score": log_score((lo, mat), outcomes)}
    out.update(pit_summary(randomized_pit((lo, mat), outcomes, seed=seed)))
    cov = interval_coverage((lo, mat), outcomes)
    for _, r in cov.iterrows():
        out[f"cov{int(round(r.level * 100))}"] = r.coverage
        out[f"mass{int(round(r.level * 100))}"] = r.predicted_mass
    return out


def by_slice(pmfs: PMFs, outcomes, slices, fn=distribution_summary, **kw) -> pd.DataFrame:
    """Apply a summary function per slice value (e.g. season) and overall."""
    lo, mat = as_matrix(pmfs)
    y = np.asarray(outcomes)
    s = np.asarray(slices)
    rows = []
    for val in list(pd.unique(s)) + ["all"]:
        sel = np.ones(len(y), bool) if val == "all" else s == val
        rows.append({"slice": val, **fn((lo, mat[sel]), y[sel], **kw)})
    return pd.DataFrame(rows)
