"""Edges: exact-contract join, point-in-time quotes/prices, cutoff and staleness screens, hand-checked EV."""
import math

import numpy as np
import pandas as pd
import pytest

from nfl_market.markets.edges import compute_edges
from nfl_market.markets.interface import EDGE_COLUMNS

KO = pd.Timestamp("2026-11-22 18:00", tz="UTC")
GID = "2026_12_BUF_KC"


def q(book, market, sel, line, odds, mins_before, **kw):
    return {"event_id": GID, "season": 2026, "week": 12, "kickoff_utc": KO,
            "captured_at_utc": KO - pd.Timedelta(minutes=mins_before), "book": book, "market": market,
            "selection": sel, "line": line, "odds_american": odds, "is_live": kw.get("is_live", False),
            "synthetic": kw.get("synthetic", False)}


def price(market, sel, line, p_win, p_push=0.0, t=None, model="elo_v1"):
    t = KO - pd.Timedelta(hours=6) if t is None else t
    pwnp = p_win / (1 - p_push)
    return {"game_id": GID, "season": 2026, "week": 12, "kickoff_utc": KO, "model_id": model, "dist_id": "k1@x",
            "prediction_time_utc": t, "market": market, "selection": sel, "line": line, "p_win": p_win,
            "p_push": p_push, "p_loss": 1 - p_win - p_push, "p_win_no_push": pwnp, "fair_american": np.nan}


PRICES = pd.DataFrame([price("spread", "home", -3.5, 0.55), price("spread", "away", 3.5, 0.45),
                       price("moneyline", "home", np.nan, 0.60, 0.01), price("moneyline", "away", np.nan, 0.39, 0.01)])
AS_OF = KO - pd.Timedelta(minutes=60)


def test_hand_checked_ev_and_devig():
    quotes = pd.DataFrame([q("Pinnacle", "spread", "home", -3.5, -110, 90), q("Pinnacle", "spread", "away", 3.5, -110, 90)])
    e = compute_edges(quotes, PRICES, AS_OF)
    assert list(e.columns) == list(EDGE_COLUMNS)
    h = e[e.selection == "home"].iloc[0]
    # -110: breakeven 110/210 = 0.52381; devig of -110/-110 = 0.5; EV at .55: .55 * 0.90909 - .45 = 0.0500
    assert h.breakeven_prob == pytest.approx(0.523810, abs=1e-6)
    assert h.market_fair_prob == pytest.approx(0.5)
    assert h.ev_per_unit == pytest.approx(0.05, abs=1e-6)
    assert h.edge_prob == pytest.approx(0.55 - 0.523810, abs=1e-6)
    assert h.opposite_odds_american == -110 and h.book == "pinnacle"
    assert h.quote_age_minutes == pytest.approx(30)
    assert h["flags"] == ""
    assert e.iloc[0].selection == "home"  # sorted by EV desc


def test_cutoff_live_stale_and_pit_prices():
    quotes = pd.DataFrame([
        q("Pinnacle", "spread", "home", -3.5, -110, 15),          # at kickoff - 15: excluded
        q("Circa", "spread", "home", -3.5, -105, 16, is_live=True),  # live: excluded
        q("BookMaker", "spread", "home", -3.5, -110, 300),        # 300 - 10 = 290 min old at as_of: stale
    ])
    as_of = KO - pd.Timedelta(minutes=10)
    # as_of is past the cutoff: nothing is actionable
    assert compute_edges(quotes, PRICES, as_of).empty
    as_of = KO - pd.Timedelta(minutes=20)
    assert compute_edges(quotes, PRICES, as_of).empty           # stale one excluded by default
    e = compute_edges(quotes, PRICES, as_of, include_stale=True)
    assert len(e) == 1 and "STALE_QUOTE" in e.iloc[0]["flags"] and "NO_OPPOSITE" in e.iloc[0]["flags"]
    assert math.isnan(e.iloc[0].market_fair_prob)
    # price made after as_of is not visible
    late = PRICES.assign(prediction_time_utc=KO - pd.Timedelta(minutes=5))
    assert compute_edges(pd.DataFrame([q("Pinnacle", "spread", "home", -3.5, -110, 90)]), late, AS_OF).empty


def test_latest_quote_wins_and_superseded_flag():
    quotes = pd.DataFrame([
        q("Book A", "spread", "home", -3.5, -120, 150), q("Book A", "spread", "away", 3.5, 100, 150),
        q("Book A", "spread", "home", -3.5, -110, 100), q("Book A", "spread", "away", 3.5, -110, 100),
        q("Book A", "spread", "home", -3.0, -120, 70), q("Book A", "spread", "away", 3.0, 100, 70),  # moved to -3
    ])
    e = compute_edges(quotes, PRICES, AS_OF)
    h = e[(e.selection == "home") & (e.line == -3.5)].iloc[0]
    assert h.odds_american == -110                               # latest at -3.5
    assert "SUPERSEDED" in h["flags"]                               # book later moved off -3.5
    assert not (e.line == -3.0).any()                            # no price at -3: no edge row


def test_moneyline_nan_line_join_and_synthetic():
    quotes = pd.DataFrame([q("Pinnacle", "moneyline", "home", np.nan, -150, 60, synthetic=True),
                           q("Pinnacle", "moneyline", "away", np.nan, 130, 60, synthetic=True)])
    e = compute_edges(quotes, PRICES, AS_OF)
    assert len(e) == 2 and e.line.isna().all() and e.synthetic.all()
    h = e[e.selection == "home"].iloc[0]
    # -150 -> decimal 1.6667; EV = .60 * .6667 - .39 = 0.0100; push mass .01 returns stake
    assert h.ev_per_unit == pytest.approx(0.60 * (2 / 3) - 0.39, abs=1e-9)
    assert h.opposite_odds_american == 130 and "SYNTHETIC" in h["flags"]


def test_empty_inputs():
    assert list(compute_edges(pd.DataFrame(), PRICES).columns) == list(EDGE_COLUMNS)
