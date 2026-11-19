"""Score models ported from the owner's nfl-betting-model repo (models that passed their tests).

Provenance: wbblanck/nfl-betting-model (read-only clone), ``src/nflbet/ats/batch5.py``
(K1 key-number margins, S1 shared-touchdown joint scores), ``src/nflbet/ats/tdfg.py``
(D2 touchdown/field-goal team points), and helpers from ``src/nflbet/ats/elo.py``
(``_golden``, ``MAXPTS``, ``margin_dist``). Frozen constants come from
``data/backtests/batch3_constants.json`` (D2, fit on 2016-2020) and
``data/backtests/batch5_constants.json`` (S1 ``c0``, K1 ``sigma``/``w``, fit on 2016-2020).
Their findings: K1 beat NB on exact-margin log probability by +0.143 and S1 by +0.035
(docs/batch5_report.md), and D2 beat NB on cover log loss (docs/batch3_report.md).
All three stayed worse than a coin flip against the closing line.

Inputs. In the original, D2 and S1 take an Elo gap ``d`` (rating points including
home field) and a league scoring level ``L`` (points per team-game). K1 takes an
expected margin ``mu``. Our prediction frame has ``pred_margin`` and ``pred_total``,
so :func:`tdfg_inputs` solves for ``(d, L)`` such that the model's expected margin
and total equal the prediction (the means are analytic, see the function).

Numerics follow the originals operation by operation, using ``math.lgamma`` tables
and golden-section search, so the parity goldens match to float precision. The one
intentional difference: returned pmfs are normalised (:class:`IntPMF`/:class:`JointPMF`),
while the originals keep the mass lost to truncation at 100 points per team
(5e-8 to 1.5e-5 of mass on the parity inputs, largest at high scoring levels).

Caution: the frozen constants were fit on 2016-2020 outcomes. Walk-forward scores
for 2016-2020 test seasons that use frozen constants are in-sample. K1 can instead
be refit in-fold with :func:`fit_k1`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from math import lgamma

import numpy as np
from scipy import optimize

from .pmf import IntPMF, JointPMF

MAXPTS = 100              # nflbet elo.MAXPTS: per-team points support 0..100
NT, NF, NS = 20, 15, 5    # tdfg truncation of touchdown, field-goal and safety counts
K1_M = np.arange(-60, 61)  # batch5.M
K1_KMAX = 40               # batch5.KMAX

#: batch3_constants.json (D2), fit 2016-2020 on 1,280 games.
D2_CONSTANTS: dict = {
    "c_T": 0.11287023406417737, "b_T": 0.09834042916839975, "theta_T": 93701390.74949771,
    "c_F": 0.07010492853147711, "b_F": 0.03180931604841374, "s": 0.03359375,
    "p": [0.10554561717352415, 0.8540548598688134, 0.04039952295766249],
}
#: batch5_constants.json s1.c0 (shared touchdowns per point of scoring level).
S1_C0: float = 0.027936337335281135
#: batch5_constants.json k1 (sigma after weighting, and w_|m| for |m| = 0..40).
K1_SIGMA: float = 13.165446423071288
K1_W: tuple[float, ...] = (
    0.16795173176581835, 0.7997274289434099, 0.7632249469944462, 2.6040513023124787, 0.8005486854359072,
    0.8630328884817671, 1.2840445314177058, 1.8936964648874224, 0.9331658802623221, 0.3754807933766474,
    1.1085710819026475, 0.621180627227967, 0.41819627266683745, 0.5081047473745856, 1.6963211912248686,
    0.5584734347424597, 0.9168387939107153, 1.1075453233115198, 0.8971971000675331, 0.47416677978021377,
    1.1177818962288937, 1.2768767076080143, 0.45756906763166805, 0.6129491093458401, 1.2605749769119137,
    0.7750782902378651, 1.0242957391529064, 0.9993680121511263, 1.6193959347615916, 0.9854142086268814,
    0.6323099065210861, 1.761309286713741, 1.1982758588026887, 1.003095901861309, 0.7087874369532072,
    2.2367376169031616, 0.3324778830388708, 2.383738370377349, 2.385219235354084, 0.575592622320045,
    1.395750291382274,
)


def golden(f, lo: float, hi: float, tol: float = 1e-5) -> float:
    """nflbet ``elo._golden`` (golden-section minimisation), copied verbatim."""
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2


# ======================================================================= K1
def k1_matrix(mu, sigma: float = K1_SIGMA, w=K1_W) -> np.ndarray:
    """nflbet ``batch5.k1_dist``: ``(G, 121)``, ``P(M=m)`` proportional to ``phi((m-mu)/sigma) * w_|m|`` on -60..60.

    ``w`` has ``K1_KMAX + 1`` entries and the weight is 1 beyond. Note that K1 uses
    the normal *density* at integers, not CDF differences.
    """
    ww = np.ones(len(K1_M))
    a = np.abs(K1_M)
    ww[a <= K1_KMAX] = np.asarray(w, float)[a[a <= K1_KMAX]]
    z = (K1_M[None, :] - np.asarray(mu, float)[:, None]) / sigma
    p = np.exp(-0.5 * z * z) * ww[None, :]
    return p / p.sum(1, keepdims=True)


def k1_pmf(mu: float, sigma: float = K1_SIGMA, w=K1_W) -> IntPMF:
    return IntPMF(int(K1_M[0]), k1_matrix([mu], sigma, w)[0])


def _k1_lp(pm: np.ndarray, m) -> np.ndarray:
    idx = np.clip(np.asarray(m, int) + 60, 0, len(K1_M) - 1)
    return np.log(np.maximum(pm[np.arange(len(idx)), idx], 1e-300))


@dataclass(frozen=True)
class K1Params:
    sigma0: float
    sigma: float
    w: tuple[float, ...]


K1_FROZEN = K1Params(sigma0=13.166218013125565, sigma=K1_SIGMA, w=K1_W)


def fit_k1(mu, m) -> K1Params:
    """nflbet ``batch5.fit_k1`` on training-fold rows only.

    The sigma of the plain normal comes from golden search on [5, 25]. Then
    ``w_k = share(|m| = k) / mean base P(|M| = k)``, with 0.05 for a margin never
    seen. Sigma is then refit with the weights.
    """
    mu = np.asarray(mu, float)
    ones = np.ones(K1_KMAX + 1)
    sig = golden(lambda s: -_k1_lp(k1_matrix(mu, s, ones), m).sum(), 5.0, 25.0)
    base = k1_matrix(mu, sig, ones)
    a = np.abs(np.asarray(m, int))
    w = np.array([np.mean(a == k) / max(base[:, np.abs(K1_M) == k].sum(1).mean(), 1e-9) for k in range(K1_KMAX + 1)])
    w = np.where(w > 0, w, 0.05)
    sig2 = golden(lambda s: -_k1_lp(k1_matrix(mu, s, w), m).sum(), 5.0, 25.0)
    return K1Params(sigma0=float(sig), sigma=float(sig2), w=tuple(float(x) for x in w))


# ======================================================================= D2
def _pois_counts(mu, n: int) -> np.ndarray:
    k = np.arange(n + 1)
    mu = np.asarray(mu, float)[:, None]
    return np.exp(k * np.log(mu) - mu - np.array([lgamma(i + 1) for i in k]))


def _pois_safe(mu, n: int) -> np.ndarray:
    """nflbet ``batch5.pois``: Poisson pmf with ``log(max(mu, 1e-300))`` so ``mu = 0`` is allowed."""
    k = np.arange(n + 1)
    mu = np.asarray(mu, float)[:, None]
    return np.exp(k * np.log(np.maximum(mu, 1e-300)) - mu - np.array([lgamma(i + 1) for i in k]))


def nb_counts(mu, theta: float, n: int) -> np.ndarray:
    """nflbet ``tdfg.nb_counts``: ``(G, n+1)`` NB probabilities; ``theta = inf`` gives Poisson."""
    if not np.isfinite(theta):
        return _pois_counts(mu, n)
    k = np.arange(n + 1)
    mu = np.asarray(mu, float)[:, None]
    lc = np.array([lgamma(i + theta) - lgamma(theta) - lgamma(i + 1) for i in k])
    return np.exp(lc + theta * np.log(theta / (theta + mu)) + k * np.log(mu / (theta + mu)))


def td_points_table(p) -> np.ndarray:
    """nflbet ``tdfg.td_points_table``: row n is the distribution of ``6n + conversions``."""
    conv = np.array([1.0])
    rows = np.zeros((NT + 1, MAXPTS + 1))
    for n in range(NT + 1):
        if n:
            conv = np.convolve(conv, p)
        seg = conv[: max(0, MAXPTS + 1 - 6 * n)]
        rows[n, 6 * n: 6 * n + len(seg)] = seg
    return rows


def _fg_safety(lf, s):
    pf = np.zeros((len(np.atleast_1d(lf)), MAXPTS + 1))
    pf[:, 0: 3 * NF + 1: 3] = nb_counts(lf, math.inf, NF)
    ps = np.zeros(MAXPTS + 1)
    ps[0: 2 * NS + 1: 2] = nb_counts([s], math.inf, NS)[0]
    return pf, ps


def points_pmf(lt, theta_t: float, lf, s: float, p) -> np.ndarray:
    """nflbet ``tdfg.points_pmf``: ``(G, MAXPTS+1)`` distribution of one team's points."""
    pt = nb_counts(lt, theta_t, NT) @ td_points_table(p)
    pf, ps = _fg_safety(lf, s)
    out = np.zeros_like(pt)
    for i in range(len(pt)):
        out[i] = np.convolve(np.convolve(pt[i], pf[i])[: MAXPTS + 1], ps)[: MAXPTS + 1]
    return out


def rates(d, level, c: float, b: float):
    """``c L exp(+-b d / 100)`` for (home, away)."""
    x = b * np.asarray(d, float) / 100.0
    base = c * np.asarray(level, float)
    return base * np.exp(x), base * np.exp(-x)


def d2_team_pmfs(d, level, k: dict = D2_CONSTANTS) -> tuple[np.ndarray, np.ndarray]:
    """nflbet ``tdfg.game_pmfs``: home and away points pmfs (independent teams)."""
    th, ta = rates(d, level, k["c_T"], k["b_T"])
    fh, fa = rates(d, level, k["c_F"], k["b_F"])
    p = np.asarray(k["p"], float)
    return points_pmf(th, k["theta_T"], fh, k["s"], p), points_pmf(ta, k["theta_T"], fa, k["s"], p)


def d2_joint(d: float, level: float, k: dict = D2_CONSTANTS) -> JointPMF:
    ph, pa = d2_team_pmfs([d], [level], k)
    return JointPMF(np.outer(ph[0], pa[0]))


# ======================================================================= S1
def lam0(c0: float, level, lt_h, lt_a):
    """Shared touchdown rate ``min(c0 L, 0.95 min(lambda_h, lambda_a))``."""
    return np.minimum(c0 * np.asarray(level, float), 0.95 * np.minimum(lt_h, lt_a))


def s1_joint_matrices(d, level, k: dict = D2_CONSTANTS, c0: float = S1_C0, kmax: int = 10) -> list[np.ndarray]:
    """S1 joint ``(home_pts, away_pts)`` matrices ``(MAXPTS+1)^2``, one per game.

    Home touchdowns are ``X_h + X_0`` and away touchdowns ``X_a + X_0``, with
    ``X_0 ~ Poisson(lambda_0)`` shared. Field goals, safeties and conversions follow
    D2. Given ``X_0 = s`` the teams are independent, so the joint is
    ``sum_s P(s) outer(ph_s, pa_s)``. Its margin equals nflbet ``batch5.s1_margin``
    (same truncation at ``kmax`` shared touchdowns, same skip of ``P(s) < 1e-12``).
    """
    th, ta = rates(d, level, k["c_T"], k["b_T"])
    fh, fa = rates(d, level, k["c_F"], k["b_F"])
    l0 = lam0(c0, level, th, ta)
    tab = td_points_table(np.asarray(k["p"], float))
    p0 = _pois_safe(l0, kmax)
    xh, xa = _pois_safe(th - l0, NT), _pois_safe(ta - l0, NT)
    pf_h, ps = _fg_safety(fh, k["s"])
    pf_a, _ = _fg_safety(fa, k["s"])
    out = []
    for i in range(len(th)):
        fsh = np.convolve(pf_h[i], ps)[: MAXPTS + 1]
        fsa = np.convolve(pf_a[i], ps)[: MAXPTS + 1]
        J = np.zeros((MAXPTS + 1, MAXPTS + 1))
        for s in range(kmax + 1):
            w = p0[i, s]
            if w < 1e-12:
                continue
            rows = tab[s: s + NT + 1]
            if len(rows) < NT + 1:
                rows = np.vstack([rows, np.zeros((NT + 1 - len(rows), MAXPTS + 1))])
            ph = np.convolve(xh[i] @ rows, fsh)[: MAXPTS + 1]
            pa = np.convolve(xa[i] @ rows, fsa)[: MAXPTS + 1]
            J += w * np.outer(ph, pa)
        out.append(J)
    return out


def s1_joint(d: float, level: float, k: dict = D2_CONSTANTS, c0: float = S1_C0) -> JointPMF:
    return JointPMF(s1_joint_matrices([d], [level], k, c0)[0])


def raw_margin(joint_matrix: np.ndarray) -> np.ndarray:
    """Unnormalised margin vector on ``-MAXPTS..MAXPTS`` (nflbet ``elo.margin_dist`` layout)."""
    n = joint_matrix.shape[0]
    h, a = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return np.bincount((h - a).ravel() + MAXPTS, weights=joint_matrix.ravel(), minlength=2 * MAXPTS + 1)


# ============================================================ input adapter
def _mean_coefs(k: dict):
    p = np.asarray(k["p"], float)
    td_pts = 6.0 + p[1] + 2.0 * p[2]
    return k["c_T"] * td_pts, k["b_T"], 3.0 * k["c_F"], k["b_F"], 2.0 * k["s"]


def tdfg_means(d, level, k: dict = D2_CONSTANTS):
    """Analytic expected (home, away) points under D2/S1 (S1's shared touchdowns do not change the means)."""
    aT, bT, aF, bF, saf = _mean_coefs(k)
    d, L = np.asarray(d, float), np.asarray(level, float)
    xT, xF = bT * d / 100.0, bF * d / 100.0
    return L * (aT * np.exp(xT) + aF * np.exp(xF)) + saf, L * (aT * np.exp(-xT) + aF * np.exp(-xF)) + saf


def tdfg_inputs(pred_margin, pred_total, k: dict = D2_CONSTANTS, d_bound: float = 2000.0):
    """Solve ``(d, L)`` so that D2/S1 expected margin and total equal the prediction.

    ``margin = L A(d)`` and ``total - 4s = L B(d)``, with ``A/B`` increasing in d,
    so d comes from a 1-d root find and then ``L = (total - 4s) / B(d)``. Ratios
    outside the attainable range are clamped at ``±d_bound``.
    """
    aT, bT, aF, bF, saf = _mean_coefs(k)

    def AB(d):
        xT, xF = bT * d / 100.0, bF * d / 100.0
        return (aT * (np.exp(xT) - np.exp(-xT)) + aF * (np.exp(xF) - np.exp(-xF)),
                aT * (np.exp(xT) + np.exp(-xT)) + aF * (np.exp(xF) + np.exp(-xF)))

    pm, pt = np.atleast_1d(np.asarray(pred_margin, float)), np.atleast_1d(np.asarray(pred_total, float))
    ds, Ls = np.zeros(pm.size), np.zeros(pm.size)
    for i, (m, t) in enumerate(zip(pm, pt)):
        net = t - 2 * saf
        if net <= 0:
            raise ValueError("predicted total too small for the TD/FG model")
        ratio = m / net
        f = lambda d: AB(d)[0] / AB(d)[1] - ratio  # noqa: E731
        if f(-d_bound) > 0:
            d = -d_bound
        elif f(d_bound) < 0:
            d = d_bound
        else:
            d = optimize.brentq(f, -d_bound, d_bound, xtol=1e-10)
        ds[i], Ls[i] = d, net / AB(d)[1]
    return ds, Ls
