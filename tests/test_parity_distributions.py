"""Parity of our distribution ports with the owner's original code.

Goldens come from tests/fixtures/parity/distributions_goldens.json, produced by
scripts/parity/make_goldens_distributions.py from the original repos (commit
hashes are stored in the file). The originals are NOT imported here. Intentional
differences are asserted exactly (see docs/methodology/distributions.md,
"Parity with original code").
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from nfl_market.distributions import scoring_models as SM
from nfl_market.distributions.margin import (KernelLocalEmpirical, ResidualPool, TParams, apply_key_weights_matrix,
                                             choose_kernel_h, fit_keynum, t_pmf_matrix)

G = json.loads((Path(__file__).parent / "fixtures" / "parity" / "distributions_goldens.json").read_text())
C = np.asarray(G["training"]["close"], float)       # nfl_ats market home spread (negative = home favoured)
M = np.asarray(G["training"]["margin"], float)
SEASON = np.asarray(G["training"]["season"])
MU = -C                                               # our conditioning variable: predicted home margin
QC = np.asarray(G["query_c"], float)
T_HALF, T_WHOLE = np.asarray(G["t_half"]), np.asarray(G["t_whole"])
EPS = 1e-4                                            # nfl_ats clips sf to [EPS, 1 - EPS]; we do not
TIGHT = 1e-12


def clip(p):
    return np.clip(p, EPS, 1 - EPS)


def sf_from_pmf(lo, mat, t):
    x = np.arange(lo, lo + mat.shape[1])
    return (mat[:, None, :] * (x[None, None, :] > np.asarray(t)[None, :, None])).sum(axis=2)


def golden(kind, key):
    return np.asarray(G["margin_dist"][kind][key], float)


def test_goldens_record_source_commits():
    for repo in ("nfl_ats", "nfl-betting-model"):
        assert len(G["sources"][repo]["commit"]) == 40


# ------------------------------------------------------------ nfl_ats MarginDist
def test_normal_matches_at_half_point_lines():
    sd = np.std(M - MU)
    assert sd == pytest.approx(G["margin_dist"]["normal"]["sigma"], abs=TIGHT)
    lo, mat = t_pmf_matrix(-QC, TParams(scale=sd, df=np.inf))
    np.testing.assert_allclose(clip(sf_from_pmf(lo, mat, T_HALF)), golden("normal", "sf_half"), atol=TIGHT, rtol=0)


def test_normal_whole_lines_expected_difference():
    """Expected difference: ours is discrete, so P(M > t) at whole t equals their continuous sf at t + 0.5."""
    sd = np.std(M - MU)
    lo, mat = t_pmf_matrix(-QC, TParams(scale=sd, df=np.inf))
    ours = clip(sf_from_pmf(lo, mat, T_WHOLE))
    theirs = golden("normal", "sf_whole")
    half = golden("normal", "sf_half")
    idx = np.searchsorted(T_HALF, T_WHOLE + 0.5)
    ok = idx < T_HALF.size
    np.testing.assert_allclose(ours[:, ok], half[:, idx[ok]], atol=TIGHT, rtol=0)
    gap = np.abs(ours - theirs).max()
    assert 0.010 < gap < 0.016   # about one half-point of normal density, phi(0)/sigma/2


def test_shifted_survival_function_is_exact():
    pool = ResidualPool(M - MU)
    ours = np.array([pool.sf_continuous(T_HALF, -c) for c in QC])
    np.testing.assert_allclose(clip(ours), golden("shifted", "sf_half"), atol=TIGHT, rtol=0)
    ours_w = np.array([pool.sf_continuous(T_WHOLE, -c) for c in QC])
    np.testing.assert_allclose(clip(ours_w), golden("shifted", "sf_whole"), atol=TIGHT, rtol=0)


def test_shifted_integer_pmf_is_unit_window_average():
    """Expected difference: our integer pmf splits each shifted residual linearly between neighbouring integers,
    so at half point t it gives the average of their sf over [t - 0.5, t + 0.5]."""
    pool = ResidualPool(M - MU, resolution=1e-9)
    c = G["margin_dist"]["shifted"]["sf_fine_c"]
    fine_t = np.asarray(G["t_fine"])
    fine_sf = golden("shifted", "sf_fine")
    p = pool.pmf(-c)
    gaps = []
    for t in np.arange(-11.5, 12.0, 1.0):
        sel = (fine_t >= t - 0.5 - 1e-9) & (fine_t <= t + 0.5 + 1e-9)
        window = getattr(np, "trapezoid", getattr(np, "trapz", None))(fine_sf[sel], fine_t[sel])   # integral over a unit window = average
        ours = p.prob_gt(t)
        assert ours == pytest.approx(window, abs=3e-3)        # 0.01 grid on a step function
        exact = np.clip(pool.sf_continuous([t], -c)[0], EPS, 1 - EPS)
        gaps.append(abs(ours - exact))
    assert max(gaps) > 0.005   # the difference is real, not rounding


@pytest.mark.parametrize("kind,mirrored", [("kernel", False), ("kernel_sym", True)])
def test_kernel_estimators_exact(kind, mirrored):
    est = KernelLocalEmpirical(MU, M, h=G["margin_dist"][kind]["h"], mirrored=mirrored)
    lo, mat = est.pmf_matrix(-QC)
    for key, t in (("sf_half", T_HALF), ("sf_whole", T_WHOLE)):
        np.testing.assert_allclose(clip(sf_from_pmf(lo, mat, t)), golden(kind, key), atol=1e-14, rtol=0)


@pytest.mark.parametrize("kind,mirrored", [("kernel", False), ("kernel_sym", True)])
def test_choose_kernel_h_exact(kind, mirrored):
    assert choose_kernel_h(MU, M, SEASON, mirrored=mirrored) == G["choose_h"][kind]


def test_keynum_exact_without_tail_folding():
    params, kw = fit_keynum(MU, M, fold_tails=False)
    assert params.scale == pytest.approx(G["margin_dist"]["keynum"]["sigma"], abs=TIGHT)
    lo, base = t_pmf_matrix(-QC, params, fold_tails=False)
    # With their exact weights our pmf is identical: the pmf pipeline is exact.
    from nfl_market.distributions.margin import KeyNumberWeights
    kw_theirs = KeyNumberWeights(keys=kw.keys, weights={"all": golden("keynum", "w")})
    exact = apply_key_weights_matrix(lo, base, kw_theirs, preserve_mean=False, preserve_var=False)
    np.testing.assert_allclose(exact, golden("keynum", "pmf"), atol=TIGHT, rtol=0)
    np.testing.assert_allclose(clip(sf_from_pmf(lo, exact, T_HALF)), golden("keynum", "sf_half"), atol=TIGHT, rtol=0)
    # Expected difference: nfl_ats computes the expected key counts from base rows that do not sum to 1 (tail beyond
    # +-80 dropped, about 3e-8); we renormalise each row first. Weights differ by that relative amount.
    rel_w = np.abs(kw.weights["all"] / golden("keynum", "w") - 1).max()
    assert 1e-10 < rel_w < 1e-7
    mat = apply_key_weights_matrix(lo, base, kw, preserve_mean=False, preserve_var=False)
    assert np.abs(clip(sf_from_pmf(lo, mat, T_HALF)) - golden("keynum", "sf_half")).max() < 2e-8


def test_keynum_default_tail_folding_expected_difference():
    """Expected difference: by default our discretised normal folds tail mass beyond +-80 into the end points;
    nfl_ats drops it. The gap is bounded by that tail mass."""
    params, kw = fit_keynum(MU, M)
    lo, base = t_pmf_matrix(-QC, params)
    mat = apply_key_weights_matrix(lo, base, kw, preserve_mean=False, preserve_var=False)
    gap = np.abs(mat - golden("keynum", "pmf")).max()
    tail = (stats.norm.cdf((-80.5 + QC) / params.scale) + stats.norm.sf((80.5 + QC) / params.scale)).max()
    assert 1e-9 < gap <= 2 * tail + 1e-9 and tail < 1e-6


# ------------------------------------------------------------ nfl-betting-model
def test_k1_frozen_constants_and_pmf_exact():
    K = G["k1"]
    assert SM.K1_SIGMA == K["constants"]["sigma"] and list(SM.K1_W) == K["constants"]["w"]
    np.testing.assert_allclose(SM.k1_matrix(K["mu"]), K["frozen_pmf"], atol=TIGHT, rtol=0)


def test_fit_k1_exact():
    f = SM.fit_k1(MU, M)
    K = G["k1"]["fit"]
    assert f.sigma0 == pytest.approx(K["sigma0"], abs=TIGHT)
    assert f.sigma == pytest.approx(K["sigma"], abs=TIGHT)
    np.testing.assert_allclose(f.w, K["w"], atol=TIGHT, rtol=0)


def test_d2_team_pmfs_exact():
    D = G["d2"]
    for k in ("c_T", "b_T", "theta_T", "c_F", "b_F", "s", "p"):
        assert SM.D2_CONSTANTS[k] == D["constants"][k]
    ph, pa = SM.d2_team_pmfs(D["d"], D["level"])
    np.testing.assert_allclose(ph, D["home_pmf"], atol=TIGHT, rtol=0)
    np.testing.assert_allclose(pa, D["away_pmf"], atol=TIGHT, rtol=0)
    raw = np.array([SM.raw_margin(np.outer(a, b)) for a, b in zip(ph, pa)])
    np.testing.assert_allclose(raw, D["margin"], atol=TIGHT, rtol=0)


def _normalisation_expected_difference(ours_norm, theirs):
    """Expected difference: our JointPMF renormalises; the original keeps the truncation loss."""
    theirs = np.asarray(theirs)
    mass = theirs.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(ours_norm, theirs / mass, atol=TIGHT, rtol=0)
    return float((1 - mass).max())


def test_d2_joint_normalisation_expected_difference():
    D = G["d2"]
    ours = np.array([SM.d2_joint(d, l).margin().on_support(-100, 100) for d, l in zip(D["d"], D["level"])])
    lost = _normalisation_expected_difference(ours, D["margin"])
    assert 0 < lost < 2e-5


def test_s1_margin_exact_and_normalised():
    S = G["s1"]
    assert SM.S1_C0 == S["c0"]
    raw = np.array([SM.raw_margin(J) for J in SM.s1_joint_matrices(S["d"], S["level"], c0=S["c0"])])
    np.testing.assert_allclose(raw, S["margin"], atol=TIGHT, rtol=0)
    ours = np.array([SM.s1_joint(d, l).margin().on_support(-100, 100) for d, l in zip(S["d"], S["level"])])
    lost = _normalisation_expected_difference(ours, S["margin"])
    assert 0 < lost < 2e-5


def test_s1_with_zero_shared_rate_is_d2_with_poisson_touchdowns():
    """S1 uses Poisson touchdowns; D2's frozen theta_T = 9.4e7 is NB at the Poisson limit (property of the original,
    worth 8e-7 relative). With theta_T = inf, S1 at c0 = 0 reproduces D2 exactly."""
    D = G["d2"]
    J0 = SM.s1_joint_matrices(D["d"], D["level"], c0=0.0)
    k_pois = {**SM.D2_CONSTANTS, "theta_T": np.inf}
    ph, pa = SM.d2_team_pmfs(D["d"], D["level"], k_pois)
    np.testing.assert_allclose([SM.raw_margin(J) for J in J0], [SM.raw_margin(np.outer(a, b)) for a, b in zip(ph, pa)],
                               atol=TIGHT, rtol=0)
    assert np.abs(np.array([SM.raw_margin(J) for J in J0]) - np.asarray(D["margin"])).max() < 1e-7


def test_tdfg_input_adapter_reproduces_prediction():
    d, L = SM.tdfg_inputs([-6.5, 0.0, 3.0, 10.0], [41.0, 44.0, 47.5, 51.0])
    mh, ma = SM.tdfg_means(d, L)
    np.testing.assert_allclose(mh - ma, [-6.5, 0.0, 3.0, 10.0], atol=1e-8)
    np.testing.assert_allclose(mh + ma, [41.0, 44.0, 47.5, 51.0], atol=1e-8)
    j = SM.d2_joint(d[2], L[2])
    assert j.margin().mean() == pytest.approx(3.0, abs=1e-3)
    assert j.total().mean() == pytest.approx(47.5, abs=1e-3)
