import pandas as pd
import pytest

from nfl_market.markets.clv import line_clv
from nfl_market.markets.contracts import Contract, contract_view, same_contract, selection_view
from nfl_market.markets.odds import american_to_decimal, devig_two_way, implied_probability, is_valid_american


def test_odds():
    assert american_to_decimal(-110) == pytest.approx(1.9090909)
    assert implied_probability(-110) == pytest.approx(110 / 210)
    assert sum(devig_two_way(-110, -110)) == pytest.approx(1)


@pytest.mark.parametrize("bad", [0, 50, -99, 110.5, float("nan"), "abc", None])
def test_malformed_odds_rejected(bad):
    assert not is_valid_american(bad)
    with pytest.raises(ValueError):
        implied_probability(bad)


def test_clv():
    assert line_clv("over", 44.5, 45) == pytest.approx(.5)
    assert line_clv("under", 45.5, 44.5) == pytest.approx(1)
    assert line_clv("home", -2.5, -3) == pytest.approx(.5)


def test_contract_line_matters():
    assert not same_contract(Contract("g", "spread", "home", -2.5), Contract("g", "spread", "home", -3))


def test_views_never_mix_sides_or_lines():
    t = pd.Timestamp("2026-09-01T12:00Z")
    q = pd.DataFrame([
        dict(event_id="g", market="spread", selection="home", line=-2.5, book="a", captured_at_utc=t, odds_american=-110),
        dict(event_id="g", market="spread", selection="away", line=2.5, book="a", captured_at_utc=t, odds_american=-110),
        dict(event_id="g", market="spread", selection="home", line=-3.0, book="b", captured_at_utc=t, odds_american=-105),
        dict(event_id="g", market="total", selection="over", line=44.5, book="a", captured_at_utc=t, odds_american=-110),
    ])
    home = selection_view(q, "g", "spread", "home")
    assert set(home.selection) == {"home"} and len(home) == 2
    exact = contract_view(q, "g", "spread", "home", -2.5)
    assert len(exact) == 1 and exact.contract_label.iloc[0] == "spread home -2.5"
