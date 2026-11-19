"""Negative Binomial team points with a shared game-level gamma frailty.

Model (per game, means from the game model: ``mu_h = (pred_total + pred_margin)/2``,
``mu_a = (pred_total - pred_margin)/2``)::

    G ~ Gamma(shape=1/theta, scale=theta)          E[G] = 1, Var[G] = theta
    H | G ~ NB(mean = mu_h * G, size = r)          independent given G
    A | G ~ NB(mean = mu_a * G, size = r)

``theta`` is the dependence term: ``Cov(H, A) = mu_h * mu_a * theta`` and
``theta = 0`` is exact independence (P5 hypothesis). The frailty can only induce
*positive* dependence; the diagnostics show whether that matches the data.
The joint is integrated over ``n_nodes`` equal-probability gamma quantiles
(renormalised to mean 1). Parameters are fit by maximum likelihood on
training-fold rows only.

Known limitation (predictive_modeling.md Section 4): NB points ignore the
3/6/7 lumpiness of scoring, so margin key numbers derived from it are too flat.
:func:`independence_diagnostics` reports key-number frequencies so that is visible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .pmf import JointPMF

MAX_POINTS = 100


@dataclass(frozen=True)
class NBParams:
    r: float          # NB size (dispersion); Poisson as r -> inf
    theta: float      # frailty variance; 0 = independence

    def independent(self) -> "NBParams":
        return NBParams(r=self.r, theta=0.0)


def frailty_nodes(theta: float, n_nodes: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Equal-weight gamma quantile nodes with mean exactly 1."""
    if theta < 0:
        raise ValueError("theta must be >= 0")
    if theta <= 1e-9:
        return np.ones(1), np.ones(1)
    q = (np.arange(n_nodes) + 0.5) / n_nodes
    g = stats.gamma(a=1.0 / theta, scale=theta).ppf(q)
    return g / g.mean(), np.full(n_nodes, 1.0 / n_nodes)


def _nb_pmf_grid(mu: np.ndarray, r: float, k: np.ndarray) -> np.ndarray:
    """NB pmf, rows = means ``mu``, cols = counts ``k``."""
    mu = np.maximum(np.asarray(mu, float), 1e-9)
    p = r / (r + mu)
    return stats.nbinom.pmf(k[None, :], r, p[:, None])


def nb_joint_pmf(mu_home: float, mu_away: float, params: NBParams, *,
                 max_points: int = MAX_POINTS, n_nodes: int = 32) -> JointPMF:
    """Joint pmf of (home_pts, away_pts) on ``0..max_points`` squared (tail mass renormalised)."""
    if mu_home <= 0 or mu_away <= 0:
        raise ValueError("team point means must be positive")
    g, w = frailty_nodes(params.theta, n_nodes)
    k = np.arange(max_points + 1)
    ph = _nb_pmf_grid(mu_home * g, params.r, k)  # (J, K)
    pa = _nb_pmf_grid(mu_away * g, params.r, k)
    return JointPMF((ph * w[:, None]).T @ pa)


def team_means(pred_margin, pred_total) -> tuple[np.ndarray, np.ndarray]:
    """(mu_home, mu_away) from predicted margin and total, floored at 1 point."""
    m, t = np.asarray(pred_margin, float), np.asarray(pred_total, float)
    return np.maximum((t + m) / 2, 1.0), np.maximum((t - m) / 2, 1.0)


def nb_loglik(params: NBParams, mu_home, mu_away, home_pts, away_pts, n_nodes: int = 32) -> np.ndarray:
    """Per-game log-likelihood of observed (home, away) points."""
    g, w = frailty_nodes(params.theta, n_nodes)
    mh = np.asarray(mu_home, float)[:, None] * g[None, :]
    ma = np.asarray(mu_away, float)[:, None] * g[None, :]
    h = np.asarray(home_pts, float)[:, None]
    a = np.asarray(away_pts, float)[:, None]
    r = params.r
    lh = stats.nbinom.logpmf(h, r, r / (r + mh))
    la = stats.nbinom.logpmf(a, r, r / (r + ma))
    return special.logsumexp(lh + la + np.log(w)[None, :], axis=1)


def fit_nb_team_points(mu_home, mu_away, home_pts, away_pts, *, fit_dependence: bool = True,
                       theta_max: float = 1.0, n_nodes: int = 24) -> NBParams:
    """MLE of NB size ``r`` and frailty ``theta`` on training-fold rows."""
    h, a = np.asarray(home_pts, float), np.asarray(away_pts, float)
    ok = np.isfinite(h) & np.isfinite(a)
    mh, ma = np.asarray(mu_home, float)[ok], np.asarray(mu_away, float)[ok]
    h, a = h[ok], a[ok]
    if h.size < 20:
        raise ValueError("need at least 20 games to fit NB team points")

    def nll(x):
        theta = x[1] if fit_dependence else 0.0
        return -nb_loglik(NBParams(r=float(np.exp(x[0])), theta=float(theta)), mh, ma, h, a, n_nodes).sum()

    if fit_dependence:
        res = optimize.minimize(nll, [np.log(10.0), 0.02], method="L-BFGS-B",
                                bounds=[(np.log(0.5), np.log(1e4)), (0.0, theta_max)])
        return NBParams(r=float(np.exp(res.x[0])), theta=float(res.x[1]))
    res = optimize.minimize(nll, [np.log(10.0)], method="L-BFGS-B", bounds=[(np.log(0.5), np.log(1e4))])
    return NBParams(r=float(np.exp(res.x[0])), theta=0.0)


def _implied_moments(mu_home, mu_away, params: NBParams, keys, max_points, n_nodes) -> dict:
    eh, ea, vh, va, cov, em, vm, et, vt = ([] for _ in range(9))
    key_mass = np.zeros(len(keys))
    for mh, ma in zip(mu_home, mu_away):
        j = nb_joint_pmf(mh, ma, params, max_points=max_points, n_nodes=n_nodes)
        hp, ap, mp, tp = j.home(), j.away(), j.margin(), j.total()
        eh.append(hp.mean()); ea.append(ap.mean()); vh.append(hp.var()); va.append(ap.var())
        cov.append(j.cov()); em.append(mp.mean()); vm.append(mp.var()); et.append(tp.mean()); vt.append(tp.var())
        absm = mp.abs()
        key_mass += np.array([absm.pmf(k) for k in keys])
    eh, ea = np.array(eh), np.array(ea)
    var_h = np.mean(vh) + eh.var()
    var_a = np.mean(va) + ea.var()
    cov_ha = np.mean(cov) + np.mean((eh - eh.mean()) * (ea - ea.mean()))
    out = {
        "corr(home,away)": cov_ha / np.sqrt(var_h * var_a),
        "var(margin)": np.mean(vm) + np.var(em),
        "var(total)": np.mean(vt) + np.var(et),
        "mean(margin)": float(np.mean(em)),
        "mean(total)": float(np.mean(et)),
    }
    for k, m in zip(keys, key_mass / len(mu_home)):
        out[f"P(|M|={k})"] = float(m)
    return out


def independence_diagnostics(mu_home, mu_away, home_pts, away_pts, params: NBParams, *,
                             keys: Sequence[int] = (0, 3, 7, 10), max_points: int = MAX_POINTS,
                             n_nodes: int = 24, max_games: int | None = 2000, seed: int = 0) -> pd.DataFrame:
    """P5 diagnostics: observed vs implied (dependent fit and theta=0) moments and key numbers.

    Implied moments are mixtures over games (within-game variance plus the
    variance of game means), so they compare with the observed cross-game moments.
    ``max_games`` subsamples games (seeded) to bound run time.
    """
    mh, ma = np.asarray(mu_home, float), np.asarray(mu_away, float)
    h, a = np.asarray(home_pts, float), np.asarray(away_pts, float)
    idx = np.arange(len(h))
    if max_games is not None and len(h) > max_games:
        idx = np.sort(np.random.default_rng(seed).choice(len(h), max_games, replace=False))
    m, t = h - a, h + a
    observed = {
        "corr(home,away)": float(np.corrcoef(h, a)[0, 1]),
        "var(margin)": float(np.var(m)),
        "var(total)": float(np.var(t)),
        "mean(margin)": float(np.mean(m)),
        "mean(total)": float(np.mean(t)),
    }
    for k in keys:
        observed[f"P(|M|={k})"] = float(np.mean(np.abs(m) == k))
    dep = _implied_moments(mh[idx], ma[idx], params, keys, max_points, n_nodes)
    ind = _implied_moments(mh[idx], ma[idx], params.independent(), keys, max_points, n_nodes)
    pooled = np.concatenate([mh, ma])
    obs_pts = np.concatenate([h, a])
    rows = [{"statistic": s, "observed": observed[s], "implied_dependent": dep[s],
             "implied_independent": ind[s]} for s in observed]
    rows.append({"statistic": "dispersion (resid var / mean)",
                 "observed": float(np.mean((obs_pts - pooled) ** 2) / np.mean(pooled)),
                 "implied_dependent": float(1 + np.mean(pooled**2) / np.mean(pooled)
                                            * ((1 + params.theta) / params.r + params.theta)),
                 "implied_independent": float(1 + np.mean(pooled**2) / np.mean(pooled) / params.r)})
    out = pd.DataFrame(rows)
    out.attrs.update({"r": params.r, "theta": params.theta, "n_games": int(len(h)), "n_games_implied": int(len(idx))})
    return out
