from pathlib import Path

import pandas as pd
import pytest

from nfl_market.data.games import load_games
from nfl_market.models.folds import assert_fold, fold_table, prediction_time, season_folds, week_origins

FIX = Path(__file__).parent / "fixtures" / "nflverse" / "games_sample.csv"


@pytest.fixture(scope="module")
def games():
    return load_games(FIX)


def test_min_train_seasons(games):
    with pytest.raises(ValueError):
        season_folds(games, [2017])       # only 2 prior seasons
    folds = season_folds(games, [2018])
    assert folds[0].fold_id == "2018" and folds[0].train_end_utc < folds[0].test_start_utc


def test_week_origins_v2_v3(games):
    f = season_folds(games, [2018])[0]
    origins = week_origins(games, f)
    assert [o.week for o in origins] == sorted(games.loc[games.season == 2018, "week"].unique())
    by = games.set_index("game_id")
    T = prediction_time(games).set_axis(games["game_id"])
    seen = set()
    for o in origins:
        assert o.train_end_utc < T.loc[list(o.test_game_ids)].min()
        assert not (o.train_game_ids & o.test_game_ids)
        assert by.loc[list(o.train_game_ids), "label_available_at_utc"].max() < o.cutoff_utc
        seen |= o.test_game_ids
    assert seen == f.test_game_ids
    # rolling origin: week-2 training includes week-1 games of the test season
    w1 = {g for g in f.test_game_ids if by.loc[g, "week"] == 1}
    assert w1 <= origins[1].train_game_ids
    tab = fold_table(games, [2018])
    assert len(tab) == len(origins) and (tab["n_train_games"].diff().dropna() >= 0).all()


def test_assertions_fire(games):
    tr = games[games.season == 2017]
    te = games[games.season == 2017].head(5)
    with pytest.raises(AssertionError, match="V2|V3"):
        assert_fold(tr, te)
    hold = load_games(FIX, include_holdout=True)
    with pytest.raises(AssertionError, match="holdout"):
        assert_fold(games[games.season == 2015], hold[hold.season == 2025])


def test_label_rule_purges_wednesday_case():
    # synthetic: a Wednesday game's label (next Wed noon) is after the next week's Thursday T -> purged
    g = pd.DataFrame({
        "game_id": ["a", "b"], "season": [2024, 2024], "week": [17, 18], "game_type": ["REG", "REG"],
        "kickoff_utc": pd.to_datetime(["2024-12-25 18:00", "2025-01-02 01:15"], utc=True),
        "home_team": ["PIT", "KC"], "away_team": ["KC", "PIT"], "neutral_site": [False, False],
        "home_score": [10.0, 20.0], "away_score": [29.0, 17.0],
        "label_available_at_utc": pd.to_datetime(["2025-01-01 17:00", "2025-01-08 17:00"], utc=True),
    })
    train, test = g.iloc[[0]], g.iloc[[1]]
    with pytest.raises(AssertionError, match="V2"):   # Wed-night kickoff: T = Tue 20:15 ET, before Wed-noon label
        assert_fold(train, test)
    test2 = test.assign(kickoff_utc=pd.to_datetime(["2025-01-05 18:00"], utc=True))   # Sunday game: T Sat
    assert_fold(train, test2)
