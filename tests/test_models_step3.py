"""Step-3 game models on a real nflverse subset (seasons 2015-2018; 2025 rows dropped by the loader)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_market.data.games import load_games
from nfl_market.holdout import is_holdout
from nfl_market.models.bayesian import (KALMAN_PRIOR_VAR, BayesianStrengthModel, KalmanMarginModel, KalmanParams,
                                        kalman_update)
from nfl_market.models.constant import ConstantModel
from nfl_market.models.elo import EloModel, EloParams, EloState
from nfl_market.models.features import FEATURE_COLUMNS, team_features
from nfl_market.models.folds import prediction_time
from nfl_market.models.interface import PREDICTION_COLUMNS, GameModel, validate_predictions
from nfl_market.models.ridge_model import RidgeModel
from nfl_market.models.runner import metrics_table, paired_bootstrap, run_walkforward
from nfl_market.models.xgb_model import XGBModel

FIX = Path(__file__).parent / "fixtures" / "nflverse" / "games_sample.csv"


@pytest.fixture(scope="module")
def games():
    return load_games(FIX)


def _split(games, season=2018, week=5):
    test = games[(games.season == season) & (games.week == week)]
    cut = prediction_time(test).min()
    train = games[games.label_available_at_utc < cut]
    blank = test.assign(home_score=np.nan, away_score=np.nan, label_available_at_utc=pd.NaT)
    return train, blank


def _models():
    return [ConstantModel(), EloModel(), KalmanMarginModel(), BayesianStrengthModel(), RidgeModel(), XGBModel()]


def test_each_model_fits_and_predicts(games):
    train, test = _split(games)
    for m in _models():
        assert isinstance(m, GameModel)
        p = validate_predictions(m.fit(train).predict(test))
        assert list(p.columns) == list(PREDICTION_COLUMNS)
        assert len(p) == len(test) and set(p.game_id) == set(test.game_id)
        assert not is_holdout(p["game_id"]).any()
        assert (p["train_end_utc"] < p["prediction_time_utc"]).all()
        assert p["pred_margin"].abs().max() < 30
        pw = p["pred_home_win_prob"].dropna()
        assert ((pw > 0) & (pw < 1)).all()
        if m.model_id != "kalman_gs_v1":
            assert p["pred_total"].between(25, 65).all()


def test_elo_hand_case():
    # source known answer: 1500 v 1500, H = 0, home wins by 7, K = 20 -> +20.79
    st = EloState(EloParams(k=20, hfa=0))
    d, rh, ra = st.update("A", "B", 2020, False, 27, 20)
    assert d == 0 and rh == pytest.approx(1500 + 20.79, abs=0.005) and ra == pytest.approx(1500 - 20.79, abs=0.005)
    # tie: multiplier 1, 1500 v 1500 with H=35 -> home loses K*(0.5 - p)
    st = EloState(EloParams())
    _, rh, _ = st.update("A", "B", 2020, False, 17, 17)
    p = 1 / (1 + 10 ** (-35 / 400))
    assert rh == pytest.approx(1500 + 25 * (0.5 - p))
    # between-season regression halfway to 1505
    assert st.rating("A", 2021) == pytest.approx(rh + 0.5 * (1505 - rh))


def test_kalman_hand_case():
    mh, vh, ma, va, pred, s = kalman_update(0.0, 49.0, 0.0, 49.0, 10, 0.0, 13.0)
    assert mh == pytest.approx(10 * 49 / (98 + 169)) and mh == pytest.approx(1.835, abs=1e-3) and ma == -mh
    assert KALMAN_PRIOR_VAR == 49.0 and KalmanParams().sigma_m == pytest.approx(12.0424, abs=1e-4)


def test_bayesian_sd_positive_and_shrinks(games):
    test = games[(games.season == 2018) & (games.week == 1)]
    sds = []
    for last in (2015, 2016, 2017):
        # a fresh model per training size; early-2015 training only vs several seasons
        tr = games[games.season <= last]
        tr = tr[tr.week <= 4] if last == 2015 else tr
        p = BayesianStrengthModel().fit(tr).predict(test)
        assert (p["pred_margin_sd"] > 0).all() and (p["pred_total_sd"] > 0).all()
        sds.append(p["pred_margin_sd"].mean())
    assert sds[0] > sds[1] >= sds[2] - 1e-9
    # within a season the posterior tightens as weeks accrue
    m = BayesianStrengthModel()
    s4 = m.fit(games[(games.season == 2015) & (games.week <= 2)]).predict(test)["pred_margin_sd"].mean()
    s8 = m.fit(games[(games.season == 2015) & (games.week <= 8)]).predict(test)["pred_margin_sd"].mean()
    assert s8 < s4


def test_future_perturbation_features_unchanged(games):
    """History includes future games; perturbing them must not change any row with T before their labels."""
    rows = games[games.season == 2018]
    cut = pd.Timestamp("2018-11-01", tz="UTC")
    pert = games.copy()
    fut = pert["kickoff_utc"] > cut
    rng = np.random.default_rng(0)
    pert.loc[fut, "home_score"] = rng.integers(0, 60, fut.sum()).astype(float)
    pert.loc[fut, "away_score"] = rng.integers(0, 60, fut.sum()).astype(float)
    a = team_features(games, rows)
    b = team_features(pert, rows)
    early = a["prediction_time_utc"] <= cut
    pd.testing.assert_frame_equal(a.loc[early], b.loc[early])
    assert not a.loc[~early, list(FEATURE_COLUMNS)].equals(b.loc[~early, list(FEATURE_COLUMNS)])
    assert (a["features_available_at_utc"].dropna() < a.loc[a["features_available_at_utc"].notna(),
                                                             "prediction_time_utc"]).all()


def test_missing_stays_nan(games):
    f = team_features(games[games.season < 2018], games[(games.season == 2018) & (games.week == 1)])
    assert f["home_pf_std"].isna().all() and f["home_gp_season"].eq(0).all()


def test_walkforward_perturbation_and_metrics(games):
    sub = games[games.season <= 2018]
    models = lambda: [ConstantModel(), EloModel(), KalmanMarginModel(), BayesianStrengthModel()]  # noqa: E731
    p = run_walkforward(sub, models(), [2018])
    assert set(p.fold_id) == {"2018"} and p.groupby("model_id").size().eq((sub.season == 2018).sum()).all()
    # perturb every 2018 score from week 10 on: predictions for weeks <= 9 must be identical
    pert = sub.copy()
    late = (pert.season == 2018) & (pert.week >= 10)
    pert.loc[late, "home_score"] = 0.0
    pert.loc[late, "away_score"] = 50.0
    q = run_walkforward(pert, models(), [2018])
    k = ["game_id", "model_id"]
    a = p[p.week <= 9].sort_values(k).reset_index(drop=True)
    b = q[q.week <= 9].sort_values(k).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert not p[p.week >= 11].sort_values(k)["pred_margin"].reset_index(drop=True).equals(
        q[q.week >= 11].sort_values(k)["pred_margin"].reset_index(drop=True))
    m = metrics_table(p, sub, by=("model_id",))
    assert (m["n_games"] == (sub.season == 2018).sum()).all() and (m["mae_margin"] > 5).all()
    bs = paired_bootstrap(p, sub, "kalman_gs_v1", "elo_v1", B=200)
    assert bs["lo"] <= bs["diff"] <= bs["hi"]


def test_runner_rejects_holdout():
    hold = load_games(FIX, include_holdout=True)
    with pytest.raises(AssertionError, match="holdout"):
        run_walkforward(hold, [ConstantModel()], [2018])
