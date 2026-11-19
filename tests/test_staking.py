"""Kelly for ternary bets and the cap order of size_stakes (hand-checked)."""
import numpy as np
import pandas as pd
import pytest

from nfl_market.config import StakePolicy
from nfl_market.markets.staking import kelly_fraction, size_stakes


def test_kelly_hand_checked():
    # b = 100/110 = 0.90909; (0.55 * 0.90909 - 0.45) / 0.90909 = 0.0550
    assert kelly_fraction(0.55, 0.0, -110) == pytest.approx(0.055, abs=1e-9)
    # push mass .1: (0.5 * b - 0.4) / (b * 0.9) = 0.054545 / 0.818182 = 1/15
    assert kelly_fraction(0.5, 0.1, -110) == pytest.approx(1 / 15, abs=1e-9)
    # +150, p = .4: b = 1.5, (0.6 - 0.6)/1.5 = 0 (break-even); clipped negative -> 0
    assert kelly_fraction(0.4, 0.0, 150) == pytest.approx(0.0, abs=1e-12)
    assert kelly_fraction(0.3, 0.0, 150) == 0.0
    with pytest.raises(ValueError):
        kelly_fraction(0.8, 0.3, -110)


def edge(game, market, sel, line, p_win, odds=-110, season=2026, week=12, book="pinnacle", model="elo_v1", flags=""):
    dec = 1 + (odds / 100 if odds > 0 else 100 / abs(odds))
    return {"game_id": game, "season": season, "week": week, "market": market, "selection": sel, "line": line,
            "book": book, "odds_american": odds, "model_id": model, "p_win": p_win, "p_push": 0.0,
            "p_loss": 1 - p_win, "ev_per_unit": p_win * (dec - 1) - (1 - p_win), "flags": flags}


def test_kelly_then_max_bet_and_min_edge():
    e = pd.DataFrame([edge("G1", "spread", "home", -3.5, 0.55),     # ev .05; .25 * .055 * 100 = 1.375 -> max_bet 1
                      edge("G2", "spread", "home", -3.5, 0.535),    # ev .0214; f=.0235; .25*.0235*100=.5875 kelly
                      edge("G3", "spread", "home", -3.5, 0.53)])    # ev .0118 < .02 -> min_edge
    s = size_stakes(e, StakePolicy()).set_index("game_id")
    assert s.loc["G1", "stake_units"] == 1.0 and s.loc["G1", "stake_limit_reason"] == "max_bet"
    assert s.loc["G2", "stake_units"] == pytest.approx(0.25 * kelly_fraction(0.535, 0, -110) * 100, abs=1e-6)
    assert s.loc["G2", "stake_limit_reason"] == "kelly"
    assert s.loc["G3", "stake_units"] == 0 and s.loc["G3", "stake_limit_reason"] == "min_edge"


def test_one_margin_bet_per_game_and_game_cap():
    e = pd.DataFrame([edge("G1", "spread", "home", -3.5, 0.56),
                      edge("G1", "spread", "away", 3.5, 0.55),       # other side: never both
                      edge("G1", "moneyline", "home", np.nan, 0.57, odds=-120),  # correlated with home spread
                      edge("G1", "total", "over", 44.5, 0.56)])
    s = size_stakes(e, StakePolicy())
    assert len(s) == 2 and set(s.market) <= {"spread", "moneyline", "total"}
    assert (s.market == "total").sum() == 1
    # existing 1.5 units on G1 (margin group) blocks margin bets and leaves 0.5 for the total
    existing = pd.DataFrame([{"game_id": "G1", "season": 2026, "week": 12, "market": "spread",
                              "stake_units": 1.5, "status": "open"}])
    s2 = size_stakes(e, StakePolicy(), existing).set_index("market")
    tot = s2.loc["total"]
    assert tot.stake_units == pytest.approx(0.5) and tot.stake_limit_reason == "max_game"
    margin = s2.drop(index="total").iloc[0]
    assert margin.stake_units == 0 and margin.stake_limit_reason == "existing_bet"


def test_week_cap_in_ev_order():
    e = pd.DataFrame([edge(f"G{i}", "spread", "home", -3.5, 0.60 - i * 0.001) for i in range(10)])
    s = size_stakes(e, StakePolicy()).sort_values("ev_per_unit", ascending=False)
    assert s.stake_units.tolist() == [1.0] * 8 + [0.0, 0.0]
    assert s.stake_limit_reason.tolist()[-2:] == ["max_week", "max_week"]


def test_blocking_flags_and_multi_model():
    e = pd.DataFrame([edge("G1", "spread", "home", -3.5, 0.6, flags="STALE_QUOTE"),
                      edge("G2", "spread", "home", -3.5, 0.6, flags="NO_OPPOSITE")])
    s = size_stakes(e, StakePolicy())
    assert s.game_id.tolist() == ["G2"]
    two = pd.concat([e, e.assign(model_id="ridge_v1")])
    with pytest.raises(ValueError, match="several models"):
        size_stakes(two)
    assert len(size_stakes(two, model_id="ridge_v1")) == 1
