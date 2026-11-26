"""Driver explanation of ridge models: contributions must add up to the model's own prediction."""
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
    from nfl_market.data.games import load_games
    from nfl_market.models.explain import contributions
    from nfl_market.models.ridge_model import RidgeModel
    from tests.test_models_step3 import FIX, _split

    games = load_games(FIX)
    train, test = _split(games)
    test = test.reset_index(drop=True)
    m = RidgeModel().fit(train.reset_index(drop=True))
    p = m.predict(test)
    C = contributions(m, test)
    assert C["margin"]["prediction"].to_numpy() == pytest.approx(p["pred_margin"].to_numpy(), abs=1e-8)
    assert C["total"]["prediction"].to_numpy() == pytest.approx(p["pred_total"].to_numpy(), abs=1e-8)
