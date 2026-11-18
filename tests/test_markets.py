import pytest
from nfl_market.markets.odds import american_to_decimal,implied_probability,devig_two_way
from nfl_market.markets.clv import line_clv
from nfl_market.markets.contracts import Contract,same_contract
def test_odds():
    assert american_to_decimal(-110)==pytest.approx(1.9090909)
    assert implied_probability(-110)==pytest.approx(110/210)
    assert sum(devig_two_way(-110,-110))==pytest.approx(1)
def test_clv():
    assert line_clv("over",44.5,45)==pytest.approx(.5)
    assert line_clv("under",45.5,44.5)==pytest.approx(1)
    assert line_clv("home",-2.5,-3)==pytest.approx(.5)
def test_contract_line_matters():
    assert not same_contract(Contract("g","spread","home",-2.5),Contract("g","spread","home",-3))
