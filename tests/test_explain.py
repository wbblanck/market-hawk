"""Driver explanation of ridge models: contributions must add up to the model's own prediction."""
import numpy as np
import pandas as pd
import pytest

from nfl_market.models.explain import GROUPS, _group_of, top_drivers


def test_group_of_maps_home_away_prefixes():
    assert _group_of("home_off_epa_l8") == GROUPS[2][0]
    assert _group_of("away_qb_out") == GROUPS[3][0]
    assert _group_of("elo_diff") == GROUPS[0][0]
    assert _group_of("league_total_l256") == GROUPS[5][0]
    assert _group_of("something_new") == "Other"


def test_top_drivers_sorted_by_size_and_skips_small_and_bookkeeping():
    row = pd.Series({"A": 0.01, "B": -2.0, "C": 1.0, "intercept": 40.0, "prediction": 39.0})
    assert top_drivers(row, 3) == [("B", -2.0), ("C", 1.0)]


def test_contributions_add_up_to_prediction():
    from pathlib import Path

    from nfl_market.data.games import load_games
    from nfl_market.models.explain import contributions
    from nfl_market.models.folds import prediction_time
    from nfl_market.models.ridge_model import RidgeModel

    games = load_games(Path(__file__).parent / "fixtures" / "nflverse" / "games_sample.csv")
    test = games[(games.season == 2018) & (games.week == 5)]
    train = games[games.label_available_at_utc < prediction_time(test).min()]
    test = test.assign(home_score=np.nan, away_score=np.nan, label_available_at_utc=pd.NaT).reset_index(drop=True)
    m = RidgeModel().fit(train.reset_index(drop=True))
    p = m.predict(test)
    C = contributions(m, test)
    assert C["margin"]["prediction"].to_numpy() == pytest.approx(p["pred_margin"].to_numpy(), abs=1e-8)
    assert C["total"]["prediction"].to_numpy() == pytest.approx(p["pred_total"].to_numpy(), abs=1e-8)
