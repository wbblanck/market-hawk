import numpy as np
import pytest

from nfl_market.distributions.margin import (KernelLocalEmpirical, ResidualPool, TParams, fit_residual_pool,
                                             normal_pmf, spread_bucket, t_pmf, t_pmf_matrix)
from nfl_market.distributions.pmf import IntPMF, JointPMF, mixture
from nfl_market.distributions.totals import total_pmf_discretized


def test_intpmf_normalises_and_validates():
    p = IntPMF(-1, [1, 2, 1])
    assert p.probs.sum() == pytest.approx(1)
    assert p.mean() == pytest.approx(0)
    assert p.var() == pytest.approx(0.5)
    with pytest.raises(ValueError):
        IntPMF(0, [0.5, -0.1])
    with pytest.raises(ValueError):
        IntPMF(0, [np.nan, 1])
    with pytest.raises(ValueError):
        IntPMF(0, [0, 0])
    with pytest.raises(ValueError):
        IntPMF(0, [0.2, 0.2], normalise=False)


def test_from_arrays_gaps_cdf_quantile():
    p = IntPMF.from_arrays([3, -3, 7], [0.5, 0.25, 0.25])
    assert p.lo == -3 and p.hi == 7
    assert p.pmf(0) == 0 and p.pmf(3) == 0.5 and p.pmf(2.5) == 0
    assert p.cdf(2.9) == pytest.approx(0.25)
    assert p.cdf(3) == pytest.approx(0.75)
    assert p.quantile(0.5) == 3
    assert p.prob_gt(3) == pytest.approx(0.25) and p.prob_lt(3) == pytest.approx(0.25)
    assert p.abs().pmf(3) == pytest.approx(0.75)
    assert p.negate().pmf(-7) == pytest.approx(0.25)
    lo, hi = p.central_interval(0.5)
    assert lo <= 3 <= hi


def test_mixture_and_shift():
    m = mixture([IntPMF.point(0), IntPMF.point(10)], [3, 1])
    assert m.mean() == pytest.approx(2.5)
    assert m.shift(2).mean() == pytest.approx(4.5)


def test_joint_margin_total_consistent():
    h = IntPMF(0, [0.2, 0.3, 0.5])
    a = IntPMF(0, [0.6, 0.4])
    j = JointPMF.independent(h, a)
    assert j.margin().mean() == pytest.approx(h.mean() - a.mean())
    assert j.total().mean() == pytest.approx(h.mean() + a.mean())
    assert j.margin().var() == pytest.approx(h.var() + a.var())
    assert j.corr() == pytest.approx(0, abs=1e-12)
    # P(M=0) by hand: (0,0)+(1,1)
    assert j.margin().pmf(0) == pytest.approx(0.2 * 0.6 + 0.3 * 0.4)
    assert j.total().pmf(1) == pytest.approx(0.2 * 0.4 + 0.3 * 0.6)


def test_t_pmf_discretisation_and_mean():
    p = t_pmf(2.5, TParams(scale=12, df=6))
    assert p.probs.sum() == pytest.approx(1)
    assert p.mean() == pytest.approx(2.5, abs=5e-3)  # tails folded at +-80
    lo, mat = t_pmf_matrix([2.5], TParams(scale=12, df=6))
    np.testing.assert_allclose(mat[0], p.probs, atol=1e-12)
    n = normal_pmf(0, 13)
    assert n.sd() == pytest.approx(np.sqrt(13**2 + 1 / 12), rel=1e-3)


def test_residual_pool_preserves_mean_and_buckets():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 13, 3000)
    pool = ResidualPool(r - r.mean(), ref=rng.uniform(-14, 14, 3000))
    p = pool.pmf(3.37)
    assert p.mean() == pytest.approx(3.37, abs=0.01)
    assert pool.pmf(-6.5, condition_on_bucket=True).probs.sum() == pytest.approx(1)
    assert list(spread_bucket([0, 2.9, 3.4, 5, 7.1, 8, 13])) == ["0-2.5", "3", "3.5-6.5", "3.5-6.5", "7", "7.5-9.5", "10+"]


def test_fit_residual_pool_uses_only_rows_before_cutoff():
    import pandas as pd
    t = pd.date_range("2020-01-01", periods=6, freq="D", tz="UTC")
    df = pd.DataFrame({"prediction_time_utc": t, "pred_margin": [0.0] * 6,
                       "home_score": [10, 20, 30, 40, 50, 60], "away_score": [0] * 6})
    pool = fit_residual_pool(df, t[3])
    assert sorted(pool.residuals) == [10, 20, 30]


def test_kernel_local_keeps_integer_outcomes_and_mirrors():
    pred = np.array([-3.0, -3.0, 3.0, 3.0, 10.0])
    marg = np.array([-3, -7, 3, 7, 14])
    k = KernelLocalEmpirical(pred, marg, h=0.5)
    p = k.pmf(3.0)
    assert p.pmf(3) == pytest.approx(0.5) and p.pmf(7) == pytest.approx(0.5)
    ks = KernelLocalEmpirical(pred, marg, h=0.5, mirrored=True)
    q = ks.pmf(0.0)
    assert q.mean() == pytest.approx(0, abs=1e-12)  # symmetric pool


def test_total_pmf_has_no_mass_on_one():
    p = total_pmf_discretized(44, TParams(scale=13, df=10))
    assert p.pmf(1) == 0 and p.lo == 0
    assert p.probs.sum() == pytest.approx(1)
