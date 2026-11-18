import pytest

from nfl_market.markets.clv import clv_report, line_clv, price_clv, probability_clv, selection_spread
from nfl_market.markets.odds import devig_two_way


# The four sign-convention examples from the spec. Positive is favourable.
@pytest.mark.parametrize("selection,bet,close,expected", [
    ("favorite", -2.5, -3.0, 0.5),   # favourite -2.5, closes -3
    ("underdog", 3.5, 3.0, 0.5),     # underdog +3.5, closes +3
    ("over", 44.5, 45.5, 1.0),       # over 44.5, closes 45.5
    ("under", 46.5, 45.5, 1.0),      # under 46.5, closes 45.5
])
def test_spec_examples(selection, bet, close, expected):
    assert line_clv(selection, bet, close) == pytest.approx(expected)


@pytest.mark.parametrize("selection,bet,close,expected", [
    ("favorite", -3.0, -2.5, -0.5),
    ("underdog", 3.0, 3.5, -0.5),
    ("over", 45.5, 44.5, -1.0),
    ("under", 45.5, 46.5, -1.0),
])
def test_adverse_moves_are_negative(selection, bet, close, expected):
    assert line_clv(selection, bet, close) == pytest.approx(expected)


def test_home_and_away_use_selected_team_spread():
    # Market moves home from -2.5 to -3.5: home bettor gained, away bettor lost.
    home_bet, home_close = selection_spread(-2.5, "home"), selection_spread(-3.5, "home")
    away_bet, away_close = selection_spread(-2.5, "away"), selection_spread(-3.5, "away")
    assert (away_bet, away_close) == (2.5, 3.5)
    assert line_clv("home", home_bet, home_close) == pytest.approx(1.0)
    assert line_clv("away", away_bet, away_close) == pytest.approx(-1.0)


def test_market_selection_mismatch_rejected():
    with pytest.raises(ValueError):
        line_clv("over", -2.5, -3, market="spread")
    with pytest.raises(ValueError):
        line_clv("moneyline", 1, 1)
    with pytest.raises(ValueError):
        line_clv("over", -1, 44.5)


def test_price_clv_uses_devigged_probabilities():
    # Bet at -110/-110 (fair 50%); closes -130/+110 on the same line.
    fair_close, _ = devig_two_way(-130, 110)
    assert price_clv(-110, -110, -130, 110) == pytest.approx(fair_close - 0.5)
    assert price_clv(-110, -110, -130, 110) > 0
    # Vig level alone does not create price CLV.
    assert price_clv(-105, -105, -115, -115) == pytest.approx(0.0)


def test_probability_clv_bounds():
    assert probability_clv(0.5, 0.55) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        probability_clv(0, 0.5)


def test_report_keeps_line_and_price_separate():
    same_line = clv_report("home", -3, -3, -110, -110, -125, 105)
    assert same_line.line_clv_points == 0
    assert same_line.price_clv_probability > 0
    moved = clv_report("home", -2.5, -3, -110, -110, -110, -110)
    assert moved.line_clv_points == pytest.approx(0.5)
    assert moved.price_clv_probability is None  # different contract: not comparable
