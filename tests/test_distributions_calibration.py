import numpy as np
import pandas as pd
import pytest

from nfl_market.distributions import calibration as cal
from nfl_market.distributions.evaluate import evaluate_distributions, fit_distribution, training_rows
from nfl_market.distributions.margin import (TParams, adjust_to_key_targets, apply_key_weights, fit_key_weights,
                                             fit_keynum, t_pmf, t_pmf_matrix)
from nfl_market.models.interface import PREDICTION_COLUMNS


def _sample(lo, mat, rng):
    c = np.cumsum(mat, axis=1)
    u = rng.uniform(size=(mat.shape[0], 1))
    return lo + (c < u).sum(axis=1)


@pytest.fixture(scope="module")
def sim():
    rng = np.random.default_rng(42)
    mu = rng.normal(2, 5, 4000)
    lo, mat = t_pmf_matrix(mu, TParams(scale=13, df=7))
    y = _sample(lo, mat, rng)
    return mu, lo, mat, y


def test_pit_uniform_under_true_model(sim):
    mu, lo, mat, y = sim
    u = cal.randomized_pit((lo, mat), y, seed=1)
    s = cal.pit_summary(u)
    assert s["pit_ks_p"] > 0.01
    assert s["pit_var"] == pytest.approx(1 / 12, abs=0.004)
    # too-narrow model -> PIT variance too large (U shape)
    lo2, narrow = t_pmf_matrix(mu, TParams(scale=7, df=7))
    assert cal.pit_summary(cal.randomized_pit((lo2, narrow), y, seed=1))["pit_var"] > 0.095
    # deterministic given the seed
    np.testing.assert_array_equal(u, cal.randomized_pit((lo, mat), y, seed=1))


def test_log_score_prefers_true_distribution(sim):
    mu, lo, mat, y = sim
    lo_w, wrong = t_pmf_matrix(mu + 3, TParams(scale=9, df=30))
    assert cal.log_score((lo, mat), y) < cal.log_score((lo_w, wrong), y)
    # list-of-IntPMF input gives the same score
    from nfl_market.distributions.pmf import IntPMF
    pm = [IntPMF(lo, r) for r in mat[:50]]
    assert cal.log_score(pm, y[:50]) == pytest.approx(cal.log_score((lo, mat[:50]), y[:50]))


def test_interval_coverage_near_nominal(sim):
    mu, lo, mat, y = sim
    cov = cal.interval_coverage((lo, mat), y)
    for _, r in cov.iterrows():
        assert r.predicted_mass >= r.level - 1e-9
        assert abs(r.coverage - r.predicted_mass) < 0.03


def test_binary_metrics_by_hand():
    p = np.array([0.2, 0.8, 0.6, 0.4])
    y = np.array([0, 1, 0, 1])
    s, rel = cal.binary_metrics(p, y, n_bins=2)
    assert s["brier"] == pytest.approx((0.04 + 0.04 + 0.36 + 0.36) / 4)
    assert rel["n"].sum() == 4
    assert s["ece"] == pytest.approx(0.5 * abs(0.5 - 0.3) + 0.5 * abs(0.5 - 0.7))
    assert s["clip_count"] == 0


def test_push_and_cover_calibration(sim):
    mu, lo, mat, y = sim
    lines = -np.round(mu * 2) / 2
    push = cal.push_calibration((lo, mat), y, lines)
    assert set(push["line"]) <= set(np.round(lines))
    assert np.all(np.abs(push["line"] - np.round(push["line"])) == 0)
    s, rel = cal.cover_calibration((lo, mat), y, lines)
    whole = np.abs(lines - np.round(lines)) < 1e-9
    assert s["n_push_excluded"] == int(((y + lines) == 0).sum())
    assert s["n"] + s["n_push_excluded"] == len(y)
    assert whole.any()
    probs = cal.spread_outcome_probs((lo, mat), lines)
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-12)
    kn = cal.key_number_table((lo, mat), y, lines, keys=(3, 7))
    assert {"all", "3", "7", "10+", "home_fav_4-6"} <= set(kn["bucket"])


def test_key_adjustment_raises_p3_keeps_mass_and_mean():
    p = t_pmf(-2.0, TParams(scale=13, df=7))
    before = p.abs().pmf(3)
    q = adjust_to_key_targets(p, {3: 0.15, 0: 0.003})
    assert before < 0.07
    assert q.abs().pmf(3) == pytest.approx(0.15, abs=1e-6)
    assert q.pmf(0) == pytest.approx(0.003, abs=1e-6)
    assert q.probs.sum() == pytest.approx(1, abs=1e-12)
    assert q.mean() == pytest.approx(p.mean(), abs=1e-6)


def test_fitted_key_weights_move_toward_training_frequency():
    rng = np.random.default_rng(3)
    mu = rng.normal(2, 5, 3000)
    lo, mat = t_pmf_matrix(mu, TParams(scale=13, df=7))
    y = _sample(lo, mat, rng)
    snap = rng.uniform(size=y.size)
    y = np.where(snap < 0.10, np.sign(y + 0.1) * 3, np.where(snap < 0.15, np.sign(y + 0.1) * 7, y))
    target3 = np.mean(np.abs(y) == 3)
    kw = fit_key_weights((lo, mat), y, keys=(0, 3, 7))
    assert kw.weights["all"][1] > 1.5
    p = t_pmf(2.0, TParams(scale=13, df=7))
    q = apply_key_weights(p, kw)
    assert p.abs().pmf(3) < q.abs().pmf(3) <= target3 + 0.03
    assert q.probs.sum() == pytest.approx(1, abs=1e-12)
    assert q.mean() == pytest.approx(p.mean(), abs=1e-3)
    assert q.var() == pytest.approx(p.var(), rel=1e-3)
    # nfl_ats-compatible keynum: matches observed |M|=3 rate in-sample
    params, kwn = fit_keynum(mu, y)
    from nfl_market.distributions.margin import apply_key_weights_matrix
    adj = apply_key_weights_matrix(*t_pmf_matrix(mu, params), kwn, preserve_mean=False, preserve_var=False)
    lo_n = t_pmf_matrix(mu[:1], params)[0]
    x = np.arange(lo_n, lo_n + adj.shape[1])
    assert adj[:, np.abs(x) == 3].sum(axis=1).mean() == pytest.approx(target3, rel=0.05)


# ------------------------------------------------------------ evaluate
def synthetic_oof(seasons=(2016, 2017, 2018, 2019, 2020, 2021, 2025), n=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        mu = rng.normal(2, 5, n)
        tot = rng.normal(44, 4, n)
        lo, mat = t_pmf_matrix(mu, TParams(scale=13, df=7))
        m = _sample(lo, mat, rng)
        snap = rng.uniform(size=n)
        m = np.where(snap < 0.08, np.sign(m + 0.1) * 3, np.where(snap < 0.12, np.sign(m + 0.1) * 7, m)).astype(int)
        away = np.maximum(0, np.round(rng.normal(21, 7, n))).astype(int)
        home = away + m
        neg = home < 0
        away[neg] -= home[neg]
        home[neg] = 0
        t = pd.Timestamp(f"{s}-09-10", tz="UTC") + pd.to_timedelta(np.sort(rng.uniform(0, 110, n)), unit="D")
        rows.append(pd.DataFrame({
            "game_id": [f"{s}_{i:03d}_H_A" for i in range(n)], "season": s, "week": 1 + (np.arange(n) // 16),
            "model_id": "synth", "fold_id": str(s), "prediction_time_utc": t,
            "train_end_utc": t - pd.Timedelta(days=4), "pred_margin": mu, "pred_total": tot,
            "pred_margin_sd": np.nan, "pred_total_sd": np.nan, "pred_home_win_prob": np.nan,
            "home_score": home, "away_score": away}))
    return pd.concat(rows, ignore_index=True)


def test_synthetic_frame_satisfies_contract():
    df = synthetic_oof()
    assert all(c in df.columns for c in PREDICTION_COLUMNS)


@pytest.mark.parametrize("family", ["t+key", "empirical", "kernel_sym", "keynum", "k1"])
def test_evaluate_uses_no_rows_from_scored_season(family):
    df = synthetic_oof()
    base = evaluate_distributions(df, family, min_history_seasons=2)
    params = base["params"].set_index("season")
    assert 2025 not in params.index and 2025 not in set(base["summary"]["season"])
    assert list(params.index) == [2018, 2019, 2020, 2021]
    # perturb the 2019 outcomes: 2019 params (and earlier) unchanged, later ones may change
    pert = df.copy()
    sel = pert["season"] == 2019
    pert.loc[sel, "home_score"] = pert.loc[sel, "home_score"] + 17
    out = evaluate_distributions(pert, family, min_history_seasons=2)["params"].set_index("season")
    pd.testing.assert_frame_equal(params.loc[[2018, 2019]], out.loc[[2018, 2019]])
    num = params.select_dtypes("number").columns
    assert not np.allclose(params.loc[2020, num].to_numpy(float), out.loc[2020, num].to_numpy(float))
    # the 2019 scores themselves do change
    s0 = base["summary"].set_index("season").loc[2019, "log_score"]
    s1 = evaluate_distributions(pert, family, min_history_seasons=2)["summary"].set_index("season").loc[2019, "log_score"]
    assert s0 != s1


def test_training_rows_strictly_precede_test_season():
    df = synthetic_oof()
    tr = training_rows(df, 2020)
    assert tr["season"].max() == 2019
    assert tr["prediction_time_utc"].max() < df.loc[df.season == 2020, "prediction_time_utc"].min()


def test_evaluate_tables_and_totals():
    df = synthetic_oof(seasons=(2016, 2017, 2018, 2019))
    r = evaluate_distributions(df, "t", min_history_seasons=2)
    for k in ("summary", "params", "coverage", "pit_hist", "key_numbers", "binary", "reliability", "push"):
        assert not r[k].empty, k
    assert set(r["summary"]["season"]) == {2018, 2019, "all"}
    rt = evaluate_distributions(df, "empirical", target="total", min_history_seasons=2)
    assert rt["key_numbers"].empty and not rt["summary"].empty
    fd = fit_distribution(df[df.season < 2018], "t+key")
    assert fd.key_weights is not None and fd.t_params.df > 2


def test_cli_writes_markdown(tmp_path):
    from nfl_market.distributions.evaluate import main
    df = synthetic_oof(seasons=(2016, 2017, 2018, 2019), n=120)
    src = tmp_path / "p.csv"
    df.to_csv(src, index=False)
    out = tmp_path / "r.md"
    assert main(["--predictions", str(src), "--out", str(out), "--families", "t,kernel",
                 "--targets", "margin", "--min-history-seasons", "2"]) == 0
    text = out.read_text()
    assert "Key numbers" in text and "margin / kernel" in text


@pytest.mark.parametrize("family", ["k1_frozen", "s1", "d2"])
def test_frozen_score_models_run_in_evaluate(family):
    df = synthetic_oof(seasons=(2016, 2017, 2018, 2019), n=60)
    r = evaluate_distributions(df, family, min_history_seasons=2)
    assert set(r["summary"]["season"]) == {2018, 2019, "all"}
    assert (r["params"]["frozen_constants_fit"] == "2016-2020 (nfl-betting-model)").all()
    assert r["summary"]["log_score"].notna().all()
