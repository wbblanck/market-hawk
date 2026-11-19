"""Parity of our Elo / Kalman / XGBoost ports with the ORIGINAL code (goldens, no import of the originals).

Goldens: tests/fixtures/parity/models_goldens.json, written by scripts/parity/make_goldens_models.py from
wbblanck/nfl-betting-model (commit recorded in the file). Two timing modes are checked:

* source timing: our sequential passes (``elo_history``, ``kalman_history``) process every earlier game, as the
  source does; these must match exactly (floating-point tolerance).
* our label rule (walk-forward models): a game enters only once its label (next Wednesday 12:00 ET) is before T.
  On the 2018 fixture season no game is affected, so predictions match exactly; on the hand schedule a Wednesday
  game right after a Monday game is predicted without week 1, and the difference equals week 1's rating updates.

Intended differences asserted rather than loosened: NB moments analytic vs the source's 0-100 truncated pmf;
XGB ``dome`` = roof "dome" only (source: "dome" or "closed").
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_market.data.games import load_games
from nfl_market.models.bayesian import KalmanMarginModel, KalmanParams, kalman_history
from nfl_market.models.elo import EloModel, EloParams, elo_history, scoring_level, score_means, win_prob
from nfl_market.models.features import team_features
from nfl_market.models.folds import labelled, prediction_time
from nfl_market.models.runner import run_walkforward
from nfl_market.models.xgb_model import XGB_PARAMS, XGBModel

HERE = Path(__file__).parent
GOLD = json.loads((HERE / "fixtures" / "parity" / "models_goldens.json").read_text())
FIX = HERE / "fixtures" / "nflverse" / "games_sample.csv"
EXACT = 1e-9
#: max abs diff per component, filled by the tests (for the parity table in docs/results/step3_walkforward.md)
REPORT: dict[str, float] = {}


def _arr(x):
    return np.array([np.nan if v is None else v for v in x], float)


def _gold(section: str) -> pd.DataFrame:
    g = GOLD[section]
    cols = [c for c, v in g.items() if isinstance(v, list) and len(v) == len(g["game_id"]) and c != "game_id"]
    return pd.DataFrame({c: _arr(g[c]) for c in cols}, index=pd.Index(g["game_id"], name="game_id"))


def _maxdiff(name, a, b):
    d = float(np.nanmax(np.abs(np.asarray(a, float) - np.asarray(b, float))))
    REPORT[name] = max(REPORT.get(name, 0.0), d)
    return d


@pytest.fixture(scope="module")
def fixture_games():
    return load_games(FIX)


@pytest.fixture(scope="module")
def hand_games(tmp_path_factory):
    h = GOLD["inputs"]["hand"]
    df = pd.DataFrame(h["rows"], columns=h["columns"])
    df = df.assign(home_rest=7, away_rest=7, div_game=0, roof="outdoors", surface="grass")
    p = tmp_path_factory.mktemp("hand") / "hand.csv"
    df.to_csv(p, index=False)
    return load_games(p)


def test_constants_match_source():
    e, k = GOLD["source"]["constants"]["elo"], GOLD["source"]["constants"]["kalman"]
    ep, kp = EloParams(), KalmanParams()
    assert (ep.k, ep.hfa, ep.regress, ep.b, ep.theta) == (e["k"], e["h"], e["c"], e["b"], e["theta"])
    assert (kp.h, kp.sigma_m, kp.tau_w, kp.rho, kp.tau_s) == (k["h"], k["sigma_m"], k["tau_w"], k["rho"], k["tau_s"])
    x = GOLD["xgb"]["params"]
    assert (XGB_PARAMS["max_depth"], XGB_PARAMS["learning_rate"], XGB_PARAMS["subsample"],
            XGB_PARAMS["colsample_bytree"], XGB_PARAMS["min_child_weight"], XGB_PARAMS["reg_lambda"],
            XGB_PARAMS["random_state"], XGB_PARAMS["objective"], XGB_PARAMS["n_estimators"]) == (
        x["max_depth"], x["eta"], x["subsample"], x["colsample_bytree"], x["min_child_weight"], x["lambda"],
        x["seed"], x["objective"], GOLD["xgb"]["n_trees"])


def _pre_ratings(eh: pd.DataFrame, params=EloParams()) -> pd.DataFrame:
    """Pre-game ratings reconstructed from our sequential pass (post ratings + between-season regression)."""
    last: dict[str, tuple[float, int]] = {}
    rh, ra = [], []
    for ht, at, s, ph, pa in zip(eh.home_franchise, eh.away_franchise, eh.season, eh.home_post, eh.away_post):
        for team, out in ((ht, rh), (at, ra)):
            if team in last:
                r, s0 = last[team]
                out.append(params.mean + (r - params.mean) * (1 - params.regress) ** (s - s0))
            else:
                out.append(params.init)
        last[ht], last[at] = (ph, s), (pa, s)
    return pd.DataFrame({"r_home": rh, "r_away": ra}, index=eh.game_id)


@pytest.mark.parametrize("section", ["fixture", "hand"])
def test_elo_source_timing_exact(section, fixture_games, hand_games):
    games = fixture_games if section == "fixture" else hand_games
    gold = _gold(section)
    eh = elo_history(games).set_index("game_id", drop=False)
    pre = _pre_ratings(eh)
    ids = gold.index
    assert _maxdiff(f"elo d ({section})", eh.loc[ids, "pre_diff"], gold["d"]) < EXACT
    assert _maxdiff(f"elo r_home/r_away ({section})", pre.loc[ids, "r_home"], gold["r_home"]) < EXACT
    assert _maxdiff(f"elo r_home/r_away ({section})", pre.loc[ids, "r_away"], gold["r_away"]) < EXACT
    assert _maxdiff(f"elo win prob ({section})", win_prob(eh.loc[ids, "pre_diff"]), gold["p"]) < EXACT
    final = EloModel().fit(games).state.ratings
    fg = GOLD[section]["final_elo"]
    assert set(final) == set(fg)
    assert _maxdiff(f"elo final ratings ({section})", [final[k] for k in fg], list(fg.values())) < EXACT
    # scoring level and NB score means
    rows = games.set_index("game_id").loc[ids].reset_index()
    lvl = scoring_level(games, rows, source_timing=True)
    assert _maxdiff(f"elo scoring level ({section})", lvl, gold["level"]) < EXACT
    # our label rule: identical except a game whose T precedes the previous week's labels
    lab = scoring_level(games, rows)
    differs = set(np.asarray(ids)[np.abs(lab - gold["level"].to_numpy()) > EXACT])
    assert differs == (set() if section == "fixture" else {"2020_02_A_D"})
    if differs:
        REPORT["label-rule scoring level shift, hand Wednesday game (expected)"] = float(
            np.max(np.abs(lab - gold["level"].to_numpy())))
    mh, ma = score_means(gold["d"], lvl, EloParams().b)
    assert _maxdiff(f"elo NB means mu_h/mu_a ({section})", mh, gold["mh"]) < EXACT
    assert _maxdiff(f"elo NB means mu_h/mu_a ({section})", ma, gold["ma"]) < EXACT


def _truncated_nb_moments(mh, ma, th, maxpts=100):
    """Margin/total moments and P(win, half ties) of independent NB scores truncated at 0..maxpts (unnormalised)."""
    from scipy.stats import nbinom

    k = np.arange(maxpts + 1)
    out = np.empty((len(mh), 5))
    for i, (a, b) in enumerate(zip(mh, ma)):
        ph, pa = nbinom.pmf(k, th, th / (th + a)), nbinom.pmf(k, th, th / (th + b))
        pm = np.convolve(ph, pa[::-1])
        m = np.arange(-maxpts, maxpts + 1)
        pt = np.convolve(ph, pa)
        t = np.arange(len(pt))
        mm, tm = (pm * m).sum(), (pt * t).sum()
        out[i] = (mm, np.sqrt((pm * m * m).sum() - mm ** 2), tm, np.sqrt((pt * t * t).sum() - tm ** 2),
                  (pm * (m > 0)).sum() + 0.5 * (pm * (m == 0)).sum())
    return out


def test_nb_moments_expected_truncation_diff():
    """Ours: analytic NB moments. Source: pmf truncated at 100 points per team, unnormalised.

    The truncated pmf reproduces the source exactly, so the analytic-vs-source gap is the truncation alone.
    """
    g = _gold("fixture")
    th = EloParams().theta
    tr = _truncated_nb_moments(g.mh.to_numpy(), g.ma.to_numpy(), th)
    for j, c in enumerate(("nb_margin_mean", "nb_margin_sd", "nb_total_mean", "nb_total_sd", "nb_win")):
        assert _maxdiff(f"NB pmf moments, truncated like source ({c})", tr[:, j], g[c]) < 1e-9
    vh, va = g.mh + g.mh ** 2 / th, g.ma + g.ma ** 2 / th
    dm = _maxdiff("NB margin mean (analytic vs 0-100 pmf)", g.mh - g.ma, g.nb_margin_mean)
    dsd = _maxdiff("NB margin/total SD (analytic vs 0-100 pmf)", np.sqrt(vh + va), g.nb_margin_sd)
    dt = _maxdiff("NB total mean (analytic vs 0-100 pmf)", g.mh + g.ma, g.nb_total_mean)
    _maxdiff("NB margin/total SD (analytic vs 0-100 pmf)", np.sqrt(vh + va), g.nb_total_sd)
    assert dm < 0.2 and dt < 0.2 and REPORT["NB margin/total SD (analytic vs 0-100 pmf)"] < 0.5   # truncated tail


@pytest.mark.parametrize("section", ["fixture", "hand"])
def test_kalman_source_timing_exact(section, fixture_games, hand_games):
    games = fixture_games if section == "fixture" else hand_games
    gold = _gold(section)
    kh, _ = kalman_history(games)
    kh = kh.set_index("game_id")
    assert _maxdiff(f"kalman mean ({section})", kh.loc[gold.index, "pred"], gold["kalman"]) < EXACT
    assert _maxdiff(f"kalman variance ({section})", kh.loc[gold.index, "var"], gold["kalman_var"]) < EXACT


def test_walkforward_label_rule_matches_source_on_2018(fixture_games):
    """Our deployed models (weekly refit under the label rule) on 2018: no purged game, so exact parity."""
    p = run_walkforward(fixture_games, [EloModel(), KalmanMarginModel()], [2018]).set_index("game_id")
    gold = _gold("fixture")
    ids = [g for g in gold.index if g.startswith("2018")]
    e, k = p[p.model_id == "elo_v1"], p[p.model_id == "kalman_gs_v1"]
    assert _maxdiff("elo_v1 walk-forward win prob (2018)", e.loc[ids, "pred_home_win_prob"], gold.loc[ids, "p"]) < EXACT
    assert _maxdiff("elo_v1 walk-forward margin (2018)", e.loc[ids, "pred_margin"],
                    gold.loc[ids, "mh"] - gold.loc[ids, "ma"]) < EXACT
    assert _maxdiff("kalman_gs_v1 walk-forward mean (2018)", k.loc[ids, "pred_margin"], gold.loc[ids, "kalman"]) < EXACT
    assert _maxdiff("kalman_gs_v1 walk-forward variance (2018)", k.loc[ids, "pred_margin_sd"] ** 2,
                    gold.loc[ids, "kalman_var"]) < 1e-7


def test_label_rule_difference_on_hand_schedule(hand_games):
    """Week 2 opens with a Wednesday game whose T (Tuesday) precedes week 1's labels (Wednesday noon)."""
    gold = _gold("hand")
    T = prediction_time(hand_games)
    ours = {}
    for (s, w), wk in hand_games.groupby(["season", "week"]):
        train = hand_games[labelled(hand_games) & (hand_games.label_available_at_utc < T.loc[wk.index].min())]
        blank = wk.assign(home_score=np.nan, away_score=np.nan, label_available_at_utc=pd.NaT)
        e = EloModel().fit(train).predict(blank)          # week 2: empty training set (all of week 1 purged)
        k = KalmanMarginModel().fit(train).predict(blank)
        for i, gid in enumerate(wk.game_id):
            ours[gid] = (e.pred_home_win_prob.iat[i], k.pred_margin.iat[i],
                         k.pred_margin_sd.iat[i] ** 2)
    kp, ep = KalmanParams(), EloParams()
    week2 = [g for g in gold.index if g.startswith("2020_02")]
    same = [g for g in gold.index if not g.startswith("2020_02") and not g.startswith("2020_01")]
    # weeks after the purge: exact parity
    for g in same:
        assert win_prob(gold.loc[g, "d"]) == pytest.approx(ours[g][0], abs=EXACT)
        assert ours[g][1] == pytest.approx(gold.loc[g, "kalman"], abs=EXACT)
        assert ours[g][2] == pytest.approx(gold.loc[g, "kalman_var"], abs=1e-7)
    # week 2: ours has no week-1 information (all teams at their priors); source used week 1
    hg = hand_games.set_index("game_id")
    for g in week2:
        hfa = 0.0 if hg.loc[g, "neutral_site"] else ep.hfa
        # expected Elo difference = week-1 rating changes of the two teams (source r - 1500 start)
        d_ours = hfa
        expected = (gold.loc[g, "r_home"] - ep.init) - (gold.loc[g, "r_away"] - ep.init)
        assert gold.loc[g, "d"] - d_ours == pytest.approx(expected, abs=EXACT)
        assert ours[g][0] == pytest.approx(float(win_prob(d_ours)), abs=EXACT)
        assert abs(expected) > 1.0
        REPORT["label-rule Elo d shift, hand week 2 (expected)"] = max(
            REPORT.get("label-rule Elo d shift, hand week 2 (expected)", 0.0), abs(expected))
        assert ours[g][1] == pytest.approx(0.0 if hg.loc[g, "neutral_site"] else kp.h, abs=EXACT)
        assert ours[g][2] == pytest.approx(2 * 49.0 + kp.sigma_m ** 2, abs=1e-7)
        kshift = abs(gold.loc[g, "kalman"] - ours[g][1])
        REPORT["label-rule Kalman mean shift, hand week 2 (expected)"] = max(
            REPORT.get("label-rule Kalman mean shift, hand week 2 (expected)", 0.0), kshift)
    assert REPORT["label-rule Kalman mean shift, hand week 2 (expected)"] > 0.5


def test_features_match_source_and_dome_difference(fixture_games):
    xg = GOLD["xgb"]
    feats = xg["features"]
    keep = [c for c in feats if c not in xg["all_nan_features"]]
    X = pd.DataFrame(np.array([_arr(r) for r in xg["X"]]), columns=keep, index=xg["game_id"])
    ids = [g for g in X.index if g.startswith("2018")]
    rows = fixture_games.set_index("game_id").loc[ids].reset_index()
    ours = team_features(fixture_games, rows).set_axis(ids)
    nd = xg["round"]
    for c in ("elo_d", "elo_fair", "kalman", "kalman_var", "level", "home_rest", "away_rest", "neutral",
              "div_game", "week"):
        assert _maxdiff(f"XGB feature {c} (2018, rounded {nd}dp)", np.round(ours[c], nd), X.loc[ids, c]) < 1e-9
    # intended difference: our dome is roof == "dome"; the source counts "closed" retractable roofs too
    diff = ours["is_dome"].to_numpy() != X.loc[ids, "dome"].to_numpy()
    closed = rows["roof"].eq("closed").to_numpy()
    assert (diff == closed).all() and diff.sum() > 0
    REPORT["XGB feature dome: rows differing (= roof 'closed', 2018)"] = float(diff.sum())


def test_xgb_reproduces_source_predictions():
    xg = GOLD["xgb"]
    feats = xg["features"]
    keep = [c for c in feats if c not in xg["all_nan_features"]]
    M = np.array([_arr(r) for r in xg["X"]])
    X = np.full((len(M), len(feats)), np.nan)
    for j, c in enumerate(keep):
        X[:, feats.index(c)] = M[:, j]
    y = np.array(xg["y"], float)
    tr, te = np.array(xg["is_train"]), np.array(xg["is_test"])
    pred = XGBModel()._make("margin", None).fit(X[tr], y[tr]).predict(X[te])
    # Reference: the source's own training call (xgb.train with its PARAMS and tree count, as in
    # nflbet/ats/xgb.py), run in the installed xgboost. Booster output changes across xgboost releases
    # (CI resolves 3.2.0 on Python 3.11; goldens were made on 3.4.1), so the port is checked against the
    # source setup in the same xgboost, and against the stored goldens when the version matches.
    import xgboost as xgb

    ref = xgb.train(dict(xg["params"]), xgb.DMatrix(X[tr], label=y[tr]), num_boost_round=xg["n_trees"])
    ref_pred = ref.predict(xgb.DMatrix(X[te]))
    assert _maxdiff("xgb_v1 estimator vs source training call (same matrix, same xgboost)", pred, ref_pred) < 1e-5
    if xgb.__version__ == xg["xgboost_version"]:
        assert _maxdiff("source training call vs goldens (same xgboost version)", ref_pred, _arr(xg["pred"])) < 1e-5
    else:
        REPORT[f"XGB goldens made on xgboost {xg['xgboost_version']}, installed {xgb.__version__}: "
               "compared to the in-environment source call only"] = 0.0

