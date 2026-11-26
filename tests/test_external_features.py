"""nflverse EPA and QB-status features (features/external.py) and the v2 feature models, on synthetic panels."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_market.data.games import load_games
from nfl_market.data.nflverse_ext import reduce_pbp
from nfl_market.features import external as X
from nfl_market.features.external import (
    EXTERNAL_COLUMNS,
    QB_REPLACEMENT,
    ExternalFeatures,
)
from nfl_market.models.folds import prediction_time
from nfl_market.models.interface import validate_predictions
from nfl_market.models.ridge_model import ridge_v2
from nfl_market.models.xgb_model import xgb_v2

FIX = Path(__file__).parent / "fixtures" / "nflverse" / "games_sample.csv"


@pytest.fixture(scope="module")
def games():
    return load_games(FIX)


def make_panels(games: pd.DataFrame, seed: int = 0) -> dict:
    """Synthetic panels for every completed fixture game: one QB per team ("<team>_QB1")."""
    rng = np.random.default_rng(seed)
    g = games[games.home_score.notna()]
    rows, qbs = [], []
    for gid, s, ht, at in zip(g.game_id, g.season, g.home_team, g.away_team):
        for team in (ht, at):
            po, pd_ = 60.0, 62.0
            rows.append({"game_id": gid, "team": team, "season": int(s),
                         "plays_off": po, "epa_off": rng.normal(0, 6), "succ_off": 28.0,
                         "pass_plays_off": 35.0, "pass_epa_off": rng.normal(0, 5), "rush_plays_off": 25.0,
                         "rush_epa_off": rng.normal(0, 3),
                         "plays_def": pd_, "epa_def": rng.normal(0, 6), "succ_def": 29.0,
                         "pass_plays_def": 36.0, "pass_epa_def": rng.normal(0, 5), "rush_plays_def": 26.0,
                         "rush_epa_def": rng.normal(0, 3)})
            qbs.append({"game_id": gid, "team": team, "player_id": f"{team}_QB1", "dropbacks": 38.0,
                        "qb_epa": rng.normal(2, 6), "season": int(s)})
    inj = pd.DataFrame({"season": pd.Series([], dtype=int), "week": pd.Series([], dtype=int), "team": [],
                        "gsis_id": [], "report_status": [], "date_modified": pd.Series([], dtype="datetime64[ns, UTC]")})
    return {"team": pd.DataFrame(rows), "qb": pd.DataFrame(qbs), "injuries": inj}


def _week(games, season=2018, week=6):
    test = games[(games.season == season) & (games.week == week)]
    cut = prediction_time(test).min()
    return games[games.label_available_at_utc < cut], test


def test_columns_and_point_in_time(games):
    panels = make_panels(games)
    hist, test = _week(games)
    f = ExternalFeatures(hist, panels).features(test)
    assert list(f.columns) == list(EXTERNAL_COLUMNS)
    assert f[[c for c in EXTERNAL_COLUMNS if "epa" in c]].notna().all().all()
    # last-8 by hand for one team: its last 8 history games, play-weighted
    row = test.iloc[0]
    team = row.home_franchise
    T = prediction_time(test.iloc[[0]]).iloc[0]
    tg = hist[((hist.home_franchise == team) | (hist.away_franchise == team)) & (hist.label_available_at_utc < T)]
    last = tg.sort_values(["label_available_at_utc", "kickoff_utc", "game_id"]).tail(8)
    p = panels["team"].merge(last[["game_id"]], on="game_id")
    p = p[p.team.isin(last.home_team.tolist() + last.away_team.tolist())]
    p = p[p.apply(lambda r: r.team == (last.set_index("game_id").loc[r.game_id, "home_team"]
                                       if last.set_index("game_id").loc[r.game_id, "home_franchise"] == team
                                       else last.set_index("game_id").loc[r.game_id, "away_team"]), axis=1)]
    assert f.iloc[0]["home_off_epa_l8"] == pytest.approx(p.epa_off.sum() / p.plays_off.sum())


def test_future_games_cannot_change_features(games):
    panels = make_panels(games)
    hist, test = _week(games)
    base = ExternalFeatures(hist, panels).features(test)
    # perturb every panel row of games labelled at or after the week's first T, including the predicted games
    cut = prediction_time(test).min()
    late = set(games.loc[~(games.label_available_at_utc < cut), "game_id"])
    p2 = {k: v.copy() for k, v in panels.items()}
    m = p2["team"].game_id.isin(late)
    p2["team"].loc[m, ["epa_off", "epa_def", "pass_epa_off", "pass_epa_def"]] += 100.0
    p2["qb"].loc[p2["qb"].game_id.isin(late), "qb_epa"] += 100.0
    # and give the full game table (with future labels) as history: rows after T still cannot contribute
    again = ExternalFeatures(games, p2).features(test)
    pd.testing.assert_frame_equal(base, again)


def test_qb_out_uses_report_time(games):
    panels = make_panels(games)
    hist, test = _week(games)
    row = test.iloc[[0]]
    team, T = row.home_team.iloc[0], prediction_time(row).iloc[0]
    pid = f"{team}_QB1"

    def with_report(ts):
        p = dict(panels)
        p["injuries"] = pd.DataFrame({"season": [2018], "week": [6], "team": [team], "gsis_id": [pid],
                                      "report_status": ["Out"], "date_modified": pd.to_datetime([ts], utc=True)})
        return ExternalFeatures(hist, p).features(row).iloc[0]

    before = with_report(T - pd.Timedelta(hours=30))
    assert before.home_qb_out == 1.0 and before.home_qb_epa == QB_REPLACEMENT
    after = with_report(T + pd.Timedelta(hours=1))
    assert after.home_qb_out == 0.0 and after.home_qb_epa != QB_REPLACEMENT
    missing_ts = with_report(None)            # no timestamp: treated as available (documented)
    assert missing_ts.home_qb_out == 1.0
    assert after.away_qb_out == 0.0           # injury season known, starter not listed
    # season without injury data -> NaN
    none = ExternalFeatures(hist, panels).features(row).iloc[0]
    assert np.isnan(none.home_qb_out)


def test_qb_epa_shrinkage(games):
    panels = make_panels(games)
    hist, test = _week(games)
    row = test.iloc[[0]]
    T = prediction_time(row).iloc[0]
    q = panels["qb"][panels["qb"].player_id == f"{row.home_team.iloc[0]}_QB1"]
    lab = hist.set_index("game_id").label_available_at_utc
    q = q[q.game_id.map(lab).notna()]
    q = q.assign(label=q.game_id.map(lab)).sort_values(["label", "game_id"])
    q = q[q.label < T].tail(16)
    f = ExternalFeatures(hist, panels).features(row).iloc[0]
    assert f.home_qb_epa == pytest.approx(q.qb_epa.sum() / (q.dropbacks.sum() + 200.0))


def test_holdout_history_refused(games):
    panels = make_panels(games)
    bad = games.head(3).copy()
    bad["game_id"] = ["2025_01_KC_BAL", "2025_01_GB_PHI", "2025_01_PIT_ATL"]
    bad["season"] = 2025
    with pytest.raises(ValueError):
        ExternalFeatures(bad, panels)


def test_reduce_pbp_ignores_line_columns():
    pbp = pd.DataFrame({
        "game_id": ["g"] * 4, "season": [2020] * 4, "posteam": ["A", "A", "B", None],
        "defteam": ["B", "B", "A", None], "pass": [1, 0, 1, 0], "rush": [0, 1, 0, 0],
        "epa": [0.5, -0.2, 1.0, 3.0], "success": [1, 0, 1, 1], "qb_dropback": [1, 0, 1, 0],
        "qb_epa": [0.5, np.nan, 1.0, np.nan], "id": ["qa", "ra", "qb", None],
        "spread_line": [3.0] * 4, "vegas_wp": [0.6] * 4})
    team, qb = reduce_pbp(pbp)
    assert "spread_line" not in team.columns and "vegas_wp" not in team.columns
    a = team.set_index("team").loc["A"]
    assert a.plays_off == 2 and a.epa_off == pytest.approx(0.3) and a.plays_def == 1
    assert set(qb.player_id) == {"qa", "qb"}


@pytest.mark.parametrize("factory", [ridge_v2, xgb_v2])
def test_v2_models_fit_predict(games, factory):
    X.set_panels(make_panels(games))
    try:
        hist, test = _week(games)
        blank = test.assign(home_score=np.nan, away_score=np.nan, label_available_at_utc=pd.NaT)
        m = factory().fit(hist)
        p = m.predict(blank)
        validate_predictions(p)
        assert p.pred_margin.notna().all() and p.pred_total.notna().all()
        assert set(EXTERNAL_COLUMNS) <= set(m.features)
        # predictions ignore the predicted games' own scores
        p2 = factory().fit(hist).predict(test)
        np.testing.assert_allclose(p.pred_margin, p2.pred_margin)
    finally:
        X.set_panels(None)
