"""Step-4 stacking: simplex weights, walk-forward cutoffs, holdout refusal and the StackModel contract."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_market.data.games import load_games
from nfl_market.distributions import scoring_models as SM
from nfl_market.holdout import holdout_ids
from nfl_market.markets.interface import STACK_MODEL_ID
from nfl_market.models.folds import prediction_time
from nfl_market.models.interface import PREDICTION_COLUMNS, GameModel, validate_predictions
from nfl_market.models.stacking import (DEFAULT_WEIGHTS_FILE, MARGIN_COMPONENTS, TOTAL_COMPONENTS, StackModel,
                                        home_win_no_tie, refuse_holdout, season_weights, simplex_weights,
                                        walkforward_stack, wide_oof)

FIX = Path(__file__).parent / "fixtures" / "nflverse" / "games_sample.csv"


# ------------------------------------------------------------- hand cases
def test_simplex_squared_hand_cases():
    # a = (1, 0), b = (0, 1), y = (0.7, 0.3): w = (0.7, 0.3) fits exactly.
    w, b = simplex_weights([[1, 0], [0, 1]], [0.7, 0.3])
    assert np.allclose(w, [0.7, 0.3]) and b == 0.0
    # a = (0, 0), b = (1, 1), y = (0.25, 0.75): prediction = w_b, SSE min at w_b = mean(y) = 0.5.
    w, _ = simplex_weights([[0, 1], [0, 1]], [0.25, 0.75])
    assert np.allclose(w, [0.5, 0.5])
    # a = (1, 1), b = (3, 3), y = (0, 0): prediction = 1 + 2 w_b >= 1, best on the simplex is w = (1, 0)
    # (unconstrained LS with sum 1 would give w_b = -0.5).
    w, _ = simplex_weights([[1, 3], [1, 3]], [0, 0])
    assert np.allclose(w, [1, 0])
    # intercept on: a = (0, 2), b = (2, 0), y = (6, 6): w = (0.5, 0.5) gives a constant 1, intercept 5.
    w, b = simplex_weights([[0, 2], [2, 0]], [6, 6], intercept=True)
    assert np.allclose(w, [0.5, 0.5]) and b == pytest.approx(5.0)


def test_simplex_mae_hand_case():
    # a = 0, b = 10 on three games, y = (2, 2, 9): prediction 10 w_b; sum |10 w_b - y| is minimised at the
    # median of y (2), so w_b = 0.2 and w = (0.8, 0.2). Squared error would give the mean: w_b = 13/30.
    X = [[0, 10], [0, 10], [0, 10]]
    w, _ = simplex_weights(X, [2, 2, 9], objective="mae")
    assert np.allclose(w, [0.8, 0.2], atol=1e-8)
    w2, _ = simplex_weights(X, [2, 2, 9])
    assert np.allclose(w2, [1 - 13 / 30, 13 / 30])


def test_simplex_recovers_known_mix():
    rng = np.random.default_rng(0)
    n = 3000
    a, b, c = rng.normal(0, 6, n), rng.normal(0, 6, n), rng.normal(0, 6, n)
    y = 0.7 * a + 0.3 * b + rng.normal(0, 1, n)
    for obj in ("squared", "mae"):
        w, _ = simplex_weights(np.c_[a, b, c], y, objective=obj)
        assert (w >= 0).all() and w.sum() == pytest.approx(1.0)
        assert np.allclose(w, [0.7, 0.3, 0.0], atol=0.02), (obj, w)


def test_home_win_no_tie_hand():
    # P(M=-1)=0.2, P(0)=0.2, P(1)=0.6 -> P(win | no tie) = 0.6 / 0.8 = 0.75
    assert home_win_no_tie(-1, np.array([[0.2, 0.2, 0.6]]))[0] == pytest.approx(0.75)


# ------------------------------------------------------ synthetic OOF data
def _synthetic(seasons=range(2010, 2017), per_season=48, seed=1):
    """Synthetic OOF frame: margin = 0.6 elo + 0.4 ridge + noise; total = 0.5 ridge + 0.5 xgb + noise."""
    rng = np.random.default_rng(seed)
    games, oof = [], []
    for s in seasons:
        start = pd.Timestamp(f"{s}-09-07 17:00", tz="UTC")
        for i in range(per_season):
            ko = start + pd.Timedelta(days=7 * (i // 3))
            gid = f"{s}_{i // 3 + 1:02d}_SYN{i}_X"
            p = {m: rng.normal(1.5, 5) for m in MARGIN_COMPONENTS}
            t = {m: rng.normal(45, 4) for m in TOTAL_COMPONENTS}
            m_obs = round(0.6 * p["elo_v1"] + 0.4 * p["ridge_v2"] + rng.normal(0, 3))
            tot = round(0.5 * t["ridge_v2"] + 0.5 * t["xgb_v2"] + rng.normal(0, 3))
            hs = (tot + m_obs) / 2
            games.append({"game_id": gid, "season": s, "week": i // 3 + 1, "kickoff_utc": ko, "home_score": hs,
                          "away_score": tot - hs, "label_available_at_utc": ko + pd.Timedelta(days=3),
                          "game_type": "REG"})
            for mid in MARGIN_COMPONENTS:
                oof.append({"game_id": gid, "season": s, "week": i // 3 + 1, "model_id": mid, "fold_id": str(s),
                            "prediction_time_utc": ko - pd.Timedelta(hours=24),
                            "train_end_utc": ko - pd.Timedelta(days=5), "pred_margin": p[mid],
                            "pred_total": t.get(mid, np.nan), "pred_margin_sd": np.nan, "pred_total_sd": np.nan,
                            "pred_home_win_prob": 0.5 + p[mid] / 40})
    return pd.DataFrame(oof), pd.DataFrame(games)


def test_walkforward_stack_contract_and_weights():
    oof, games = _synthetic()
    run = walkforward_stack(oof, games)
    p = validate_predictions(run.predictions)
    assert list(p.columns) == list(PREDICTION_COLUMNS)
    assert set(p["model_id"]) == {STACK_MODEL_ID}
    assert sorted(p["season"].unique()) == [2013, 2014, 2015, 2016]     # 2010-2012 are history only
    assert (p["train_end_utc"] < p["prediction_time_utc"]).all()
    assert p["pred_home_win_prob"].between(0, 1).all() and p["pred_total"].notna().all()
    last = {(w.season, w.target): w for w in run.weights}
    wm, wt = last[(2016, "margin")].as_dict(), last[(2016, "total")].as_dict()
    assert wm["elo_v1"] == pytest.approx(0.6, abs=0.08) and wm["ridge_v2"] == pytest.approx(0.4, abs=0.08)
    assert wt["ridge_v2"] + wt["xgb_v2"] == pytest.approx(1.0, abs=0.1)
    # stacked margin is exactly the season's weights applied to the component predictions
    wide = wide_oof(oof, games, MARGIN_COMPONENTS, "margin").set_index("game_id")
    s16 = p[p.season == 2016].set_index("game_id")
    manual = wide.loc[s16.index, list(MARGIN_COMPONENTS)].to_numpy() @ np.array([wm[c] for c in MARGIN_COMPONENTS])
    assert np.allclose(s16["pred_margin"], manual)
    eq = walkforward_stack(oof, games, equal=True, model_id="stack_eq_v2")
    assert all(np.allclose(w.weights, 1 / len(w.weights)) for w in eq.weights)


def test_weights_unchanged_when_future_perturbed():
    oof, games = _synthetic()
    S = 2014
    wide = wide_oof(oof, games, MARGIN_COMPONENTS, "margin")
    before = season_weights(wide, S, MARGIN_COMPONENTS, "margin")
    oof2, games2 = oof.copy(), games.copy()
    fut = oof2["season"] >= S
    oof2.loc[fut, "pred_margin"] = oof2.loc[fut, "pred_margin"] * -3 + 7
    gf = games2["season"] >= S
    games2.loc[gf, "home_score"] = games2.loc[gf, "home_score"] + 30
    after = season_weights(wide_oof(oof2, games2, MARGIN_COMPONENTS, "margin"), S, MARGIN_COMPONENTS, "margin")
    assert before.weights == after.weights and before.n_train == after.n_train
    r1, r2 = walkforward_stack(oof, games), walkforward_stack(oof2, games2)
    w1 = [w.weights for w in r1.weights if w.season == S]
    w2 = [w.weights for w in r2.weights if w.season == S]
    assert w1 == w2
    assert r1.k1[S].k1_params == r2.k1[S].k1_params
    # and a label that arrives after S's first T is excluded: move one 2013 label past the cutoff
    games3 = games.copy()
    cutoff = oof.loc[oof.season == S, "prediction_time_utc"].min()
    games3.loc[games3.season == 2013, "label_available_at_utc"] = cutoff + pd.Timedelta(hours=1)
    late = season_weights(wide_oof(oof, games3, MARGIN_COMPONENTS, "margin"), S, MARGIN_COMPONENTS, "margin")
    assert late.n_train == before.n_train - 48 and late.train_seasons == (2010, 2011, 2012)


def test_too_few_training_seasons_not_scored():
    oof, games = _synthetic(seasons=range(2010, 2013))
    assert walkforward_stack(oof, games).predictions.empty


def test_refuses_holdout_ids():
    hid = sorted(holdout_ids())[0]
    with pytest.raises(ValueError, match="holdout"):
        refuse_holdout([hid], "test")
    oof, games = _synthetic()
    bad = oof.copy()
    bad.loc[bad.index[:4], "game_id"] = hid
    with pytest.raises(ValueError, match="holdout"):
        walkforward_stack(bad, games)


# ------------------------------------------------------- StackModel (fixture)
@pytest.fixture(scope="module")
def split():
    games = load_games(FIX)
    test = games[(games.season == 2018) & (games.week == 5)]
    cut = prediction_time(test).min()
    train = games[games.label_available_at_utc < cut]
    return train, test.assign(home_score=np.nan, away_score=np.nan, label_available_at_utc=pd.NaT)


@pytest.fixture(scope="module")
def fitted(split):
    train, test = split
    wm = {"elo_v1": 0.1, "kalman_gs_v1": 0.4, "ridge_v1": 0.2, "xgb_v1": 0.3}
    wt = {"elo_v1": 0.0, "ridge_v1": 0.5, "xgb_v1": 0.5}
    m = StackModel(wm, wt).fit(train)
    return m, m.predict(test), wm, wt


def test_stack_model_contract(split, fitted):
    train, test = split
    m, p, wm, wt = fitted
    assert isinstance(m, GameModel) and m.model_id == STACK_MODEL_ID
    p = validate_predictions(p.assign(fold_id="2018"))
    assert list(p.columns) == list(PREDICTION_COLUMNS) and len(p) == len(test)
    comp = {c: mm.predict(test).set_index("game_id").loc[p.game_id] for c, mm in m.components.items()}
    assert np.allclose(p["pred_margin"], sum(w * comp[c]["pred_margin"].to_numpy() for c, w in wm.items()))
    assert np.allclose(p["pred_total"], sum(w * comp[c]["pred_total"].to_numpy() for c, w in wt.items()))
    # default mapping: margin-weighted average of components' own P(home win)
    pw = sum(w * comp[c]["pred_home_win_prob"].to_numpy() for c, w in wm.items())
    assert np.allclose(p["pred_home_win_prob"], pw)
    assert p["pred_margin_sd"].isna().all() and p["pred_total_sd"].isna().all()
    assert (p["train_end_utc"] < p["prediction_time_utc"]).all()


def test_stack_model_k1_mapping_and_file(split, fitted, tmp_path):
    train, test = split
    m, p, wm, wt = fitted
    k1 = SM.K1_FROZEN
    m2 = StackModel(wm, wt, components=list(m.components.values()), k1=k1)
    p2 = m2.predict(test)
    assert np.allclose(p2["pred_margin"], p["pred_margin"])
    pk = p2["pred_home_win_prob"].to_numpy()
    assert ((pk > 0) & (pk < 1)).all()
    assert np.all(np.diff(pk[np.argsort(p2["pred_margin"].to_numpy())]) >= -1e-12)   # monotone in margin
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"model_id": STACK_MODEL_ID, "margin": {"weights": wm, "intercept": 0.0},
                                "total": {"weights": wt, "intercept": 0.0},
                                "win_prob": {"mapping": "k1", "k1_sigma0": k1.sigma0, "k1_sigma": k1.sigma,
                                             "k1_w": list(k1.w)}}))
    m3 = StackModel.from_weights_file(path, components=list(m.components.values()))
    assert np.allclose(m3.predict(test)["pred_home_win_prob"], pk)
    with pytest.raises(ValueError, match="sum to 1"):
        StackModel({"elo_v1": 0.5, "ridge_v1": 0.6}, components=list(m.components.values()))
    with pytest.raises(ValueError, match="components missing"):
        StackModel({"bayes_od_v1": 1.0}, components=list(m.components.values()))


def test_stack_model_refuses_holdout(fitted, split):
    m = fitted[0]
    _, test = split
    bad = test.head(1).assign(game_id=sorted(holdout_ids())[0])
    with pytest.raises(ValueError, match="holdout"):
        m.predict(bad)
    with pytest.raises(ValueError, match="holdout"):
        m.fit(bad)


def test_committed_forward_weights_file():
    if not DEFAULT_WEIGHTS_FILE.exists():
        pytest.skip("forward weights not generated yet (python scripts/run_stacking.py)")
    d = json.loads(DEFAULT_WEIGHTS_FILE.read_text())
    assert d["model_id"] == STACK_MODEL_ID and d["train_seasons"] == "2015-2024"
    for t, comps in (("margin", MARGIN_COMPONENTS), ("total", TOTAL_COMPONENTS)):
        w = d[t]["weights"]
        assert set(w) == set(comps) and sum(w.values()) == pytest.approx(1.0) and min(w.values()) >= 0
    assert len(d["win_prob"]["k1_w"]) == SM.K1_KMAX + 1
