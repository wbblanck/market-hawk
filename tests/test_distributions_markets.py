import numpy as np
import pytest

from nfl_market.distributions.margin import TParams, t_pmf
from nfl_market.distributions.markets import (away_spread_probs, conditional_cover, fair_american,
                                              home_spread_probs, key_number_probs, market_summary,
                                              moneyline_probs, p_home_win, p_tie, total_probs)
from nfl_market.distributions.pmf import IntPMF

# Hand-built margin pmf: M in {-3, 0, 3, 4, 7} with known masses
M = IntPMF.from_arrays([-3, 0, 3, 4, 7], [0.2, 0.05, 0.3, 0.15, 0.3])


def test_moneyline_and_ties():
    assert p_home_win(M) == pytest.approx(0.75)
    assert p_tie(M) == pytest.approx(0.05)
    ml = moneyline_probs(M)
    assert ml.loss == pytest.approx(0.2)
    assert ml.win_given_no_push == pytest.approx(0.75 / 0.95)


def test_home_spread_whole_line_by_hand():
    s = home_spread_probs(M, -3)          # home -3: cover iff M > 3
    assert s.win == pytest.approx(0.45)   # 4, 7
    assert s.push == pytest.approx(0.3)   # M == 3
    assert s.loss == pytest.approx(0.25)
    assert s.win_given_no_push == pytest.approx(0.45 / 0.7)


def test_home_spread_half_line_has_no_push():
    s = home_spread_probs(M, -3.5)
    assert s.push == 0
    assert s.win == pytest.approx(0.45)
    s2 = home_spread_probs(M, -2.5)
    assert s2.win == pytest.approx(0.75) and s2.push == 0


def test_home_underdog_line():
    s = home_spread_probs(M, +3)          # cover iff M > -3
    assert s.win == pytest.approx(0.8) and s.push == pytest.approx(0.2) and s.loss == pytest.approx(0, abs=1e-12)


def test_away_side_mirrors_home_side():
    for h in (-7, -3.5, -3, 0, 2.5, 3):
        hs, as_ = home_spread_probs(M, h), away_spread_probs(M, -h)
        assert as_.win == pytest.approx(hs.loss)
        assert as_.push == pytest.approx(hs.push)
        assert as_.loss == pytest.approx(hs.win)


def test_totals_over_push_under():
    T = IntPMF.from_arrays([37, 41, 44, 47], [0.25, 0.25, 0.25, 0.25])
    t = total_probs(T, 44)
    assert (t.win, t.push, t.loss) == pytest.approx((0.25, 0.25, 0.5))
    t2 = total_probs(T, 44.5)
    assert (t2.win, t2.push, t2.loss) == pytest.approx((0.25, 0, 0.75))


def test_coherence_and_monotonicity_on_a_realistic_pmf():
    p = t_pmf(3.0, TParams(scale=13, df=7))
    lines = np.arange(-20, 20.5, 0.5)
    covers = []
    for h in lines:
        s = home_spread_probs(p, h)
        assert s.win + s.push + s.loss == pytest.approx(1, abs=1e-12)
        if h != round(h):
            assert s.push == 0
        covers.append(s.win)
    # P(cover) non-increasing as h becomes more negative == non-decreasing in h
    assert np.all(np.diff(covers) >= -1e-15)


def test_key_numbers_and_fair_odds():
    k = key_number_probs(M, (0, 3, 7))
    assert k == pytest.approx({0: 0.05, 3: 0.5, 7: 0.3})
    assert fair_american(0.5) == pytest.approx(-100)
    assert fair_american(0.75) == pytest.approx(-300)
    assert fair_american(0.25) == pytest.approx(300)
    assert conditional_cover(0.45, 0.1) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        fair_american(1.0)


def test_market_summary_is_consistent():
    T = IntPMF.from_arrays([40, 45], [0.5, 0.5])
    s = market_summary(M, T, home_spread=-3, total_line=42.5)
    assert s["p_home_cover"] + s["p_spread_push"] + s["p_away_cover"] == pytest.approx(1)
    assert s["p_over"] == pytest.approx(0.5)
