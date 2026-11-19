import numpy as np
import pytest
from scipy import stats

from nfl_market.distributions.team_points import (NBParams, fit_nb_team_points, independence_diagnostics,
                                                  nb_joint_pmf)


def test_zero_dependence_is_independence():
    j = nb_joint_pmf(24, 20, NBParams(r=8, theta=0.0))
    h, a = j.home(), j.away()
    np.testing.assert_allclose(j.probs, np.outer(h.probs, a.probs), atol=1e-14)
    assert j.corr() == pytest.approx(0, abs=1e-12)
    assert h.mean() == pytest.approx(24, abs=1e-3)
    assert h.var() == pytest.approx(24 + 24**2 / 8, rel=1e-3)


def test_dependence_term_raises_correlation_and_total_variance():
    ind = nb_joint_pmf(24, 20, NBParams(r=8, theta=0.0))
    dep = nb_joint_pmf(24, 20, NBParams(r=8, theta=0.05))
    assert dep.corr() > ind.corr() + 0.05
    assert dep.total().var() > ind.total().var()
    assert dep.margin().mean() == pytest.approx(4, abs=1e-2)
    assert dep.margin().probs.sum() == pytest.approx(1)
    # implied covariance ~ mu_h mu_a theta (quadrature slightly shrinks Var G)
    assert dep.cov() == pytest.approx(24 * 20 * 0.05, rel=0.1)


def _simulate(theta, n, seed):
    rng = np.random.default_rng(seed)
    mh, ma, r = rng.uniform(18, 30, n), rng.uniform(16, 26, n), 10.0
    g = rng.gamma(1 / theta, theta, n) if theta > 0 else np.ones(n)
    h = stats.nbinom.rvs(r, r / (r + mh * g), random_state=rng)
    a = stats.nbinom.rvs(r, r / (r + ma * g), random_state=rng)
    return mh, ma, h, a


def test_fit_recovers_dependence_and_diagnostics():
    mh, ma, h, a = _simulate(0.04, 3000, 1)
    p = fit_nb_team_points(mh, ma, h, a)
    assert 0.015 < p.theta < 0.07
    assert 6 < p.r < 16
    d = independence_diagnostics(mh, ma, h, a, p, max_games=150).set_index("statistic")
    row = d.loc["corr(home,away)"]
    assert row["implied_independent"] < row["implied_dependent"]
    assert abs(row["implied_dependent"] - row["observed"]) < abs(row["implied_independent"] - row["observed"])
    assert {"var(margin)", "var(total)", "P(|M|=3)", "P(|M|=7)"} <= set(d.index)


def test_fit_without_dependence_gives_theta_zero():
    mh, ma, h, a = _simulate(0.0, 800, 2)
    p = fit_nb_team_points(mh, ma, h, a, fit_dependence=False)
    assert p.theta == 0.0
    p2 = fit_nb_team_points(mh, ma, h, a)
    assert p2.theta < 0.02
