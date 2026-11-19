"""Paper bet log: append-only idempotence, refusals, tiered closes with CLV signs, grading, summary."""
import math

import numpy as np
import pandas as pd
import pytest

from nfl_market.config import ClosePolicy
from nfl_market.markets.bet_log import BetLogStore, clv_summary, settle
from nfl_market.markets.interface import BET_LOG_COLUMNS, CLV_SUMMARY_COLUMNS

KO = pd.Timestamp("2026-11-22 18:00", tz="UTC")
DECIDE = KO - pd.Timedelta(hours=4)


def bet(game, market, sel, line, book, odds=-110, opp=-110, stake=1.0):
    return {"game_id": game, "season": 2026, "week": 12, "kickoff_utc": KO, "captured_at_utc": DECIDE - pd.Timedelta(minutes=5),
            "book": book, "market": market, "selection": sel, "line": line, "odds_american": odds,
            "opposite_odds_american": opp, "model_id": "elo_v1", "dist_id": "k1@2015-2024", "p_win": 0.55,
            "p_push": 0.0, "ev_per_unit": 0.05, "kelly_fraction": 0.055, "stake_units": stake,
            "stake_limit_reason": "max_bet", "synthetic": False}


def quote(game, book, market, sel, line, odds, mins_before):
    return {"event_id": game, "season": 2026, "week": 12, "kickoff_utc": KO,
            "captured_at_utc": KO - pd.Timedelta(minutes=mins_before), "book": book, "market": market,
            "selection": sel, "line": line, "odds_american": odds, "is_live": False, "synthetic": False}


BETS = pd.DataFrame([
    bet("2026_12_G1_H1", "spread", "home", -2.5, "Pinnacle"),        # favourite -2.5
    bet("2026_12_G2_H2", "spread", "away", 3.5, "Circa Sports"),     # underdog +3.5
    bet("2026_12_G3_H3", "total", "over", 44.5, "Book A"),           # over 44.5
])

CLOSE_QUOTES = pd.DataFrame([
    # G1: Pinnacle close (30 min before KO): main line -3 (balanced) plus -2.5 at -125/+105
    quote("2026_12_G1_H1", "Pinnacle", "spread", "home", -3.0, -110, 30),
    quote("2026_12_G1_H1", "Pinnacle", "spread", "away", 3.0, -110, 30),
    quote("2026_12_G1_H1", "Pinnacle", "spread", "home", -2.5, -125, 30),
    quote("2026_12_G1_H1", "Pinnacle", "spread", "away", 2.5, 105, 30),
    # G2: no Pinnacle; Circa and BookMaker both close away +3 -> sharp_median
    quote("2026_12_G2_H2", "Circa", "spread", "away", 3.0, -110, 30),
    quote("2026_12_G2_H2", "Circa", "spread", "home", -3.0, -110, 30),
    quote("2026_12_G2_H2", "BookMaker.eu", "spread", "away", 3.0, -110, 40),
    quote("2026_12_G2_H2", "BookMaker.eu", "spread", "home", -3.0, -110, 40),
    # G2 at the bet line +3.5 only 245 min before KO: stale at the cutoff -> price close missing
    quote("2026_12_G2_H2", "Circa", "spread", "away", 3.5, -110, 245),
    quote("2026_12_G2_H2", "Circa", "spread", "home", -3.5, -110, 245),
    # G3: only two retail books -> all_book_median weak fallback, FEW_BOOKS
    quote("2026_12_G3_H3", "Book A", "total", "over", 45.5, -110, 30),
    quote("2026_12_G3_H3", "Book A", "total", "under", 45.5, -110, 30),
    quote("2026_12_G3_H3", "Book B", "total", "over", 45.5, -110, 20),
    quote("2026_12_G3_H3", "Book B", "total", "under", 45.5, -110, 20),
])


@pytest.fixture
def store(tmp_path):
    s = BetLogStore(tmp_path / "bets.duckdb")
    assert s.record(BETS, decision_time=DECIDE) == 3
    return s


def test_record_is_append_only_and_idempotent(store):
    assert store.record(BETS, decision_time=DECIDE) == 0         # same decisions -> same bet_id
    b = store.bets()
    assert list(b.columns) == list(BET_LOG_COLUMNS) and len(b) == 3
    assert (b.status == "open").all() and b.bet_id.is_unique
    assert set(b.book) == {"pinnacle", "circa", "booka"}
    assert store.record(BETS, decision_time=DECIDE + pd.Timedelta(minutes=1)) == 3  # a new decision time is new


def test_refusals(tmp_path):
    s = BetLogStore(tmp_path / "r.duckdb")
    with pytest.raises(ValueError, match="holdout"):
        s.record(BETS.assign(game_id="2025_10_DAL_PHI"), decision_time=DECIDE)
    with pytest.raises(ValueError, match="kickoff - 15"):
        s.record(BETS, decision_time=KO - pd.Timedelta(minutes=15))   # exactly at the cutoff
    assert s.bets().empty


def test_tiered_closes_and_clv_signs(store):
    store.attach_closes(CLOSE_QUOTES, ClosePolicy(), as_of=KO)
    b = store.bets().set_index("game_id")
    g1, g2, g3 = b.loc["2026_12_G1_H1"], b.loc["2026_12_G2_H2"], b.loc["2026_12_G3_H3"]
    # README: favourite -2.5, close -3 -> +0.5 (Pinnacle close)
    assert (g1.line_close, g1.line_close_tier, g1.line_clv) == (-3.0, "pinnacle", 0.5)
    # price close on the exact -2.5: devig(-125, +105) = .555556 / (.555556 + .487805) = .532468; bet fair .5
    assert g1.price_close_tier == "pinnacle"
    assert g1.price_close_fair_prob == pytest.approx(0.532468, abs=1e-6)
    assert g1.price_clv == pytest.approx(0.032468, abs=1e-6)
    # README: underdog +3.5, close +3 -> +0.5 (median of Circa and BookMaker)
    assert (g2.line_close, g2.line_close_tier, g2.line_clv) == (3.0, "sharp_median", 0.5)
    assert g2.price_close_tier == "missing" and math.isnan(g2.price_clv)   # bet line stale at cutoff
    # README: over 44.5, close 45.5 -> +1.0 (weak fallback)
    assert (g3.line_close, g3.line_close_tier, g3.line_clv) == (45.5, "all_book_median", 1.0)
    assert "MARKET_MEDIAN_FALLBACK" in g3.close_flags and "FEW_BOOKS" in g3.close_flags
    assert (b.status == "closed").all() and (b.close_policy_fingerprint == ClosePolicy().fingerprint()).all()


def test_close_not_attached_before_cutoff(store):
    assert store.attach_closes(CLOSE_QUOTES, as_of=KO - pd.Timedelta(minutes=16)) == 0
    assert (store.bets().status == "open").all()


def test_grade_and_latest_wins(store):
    store.attach_closes(CLOSE_QUOTES, as_of=KO)
    games = pd.DataFrame({"game_id": ["2026_12_G1_H1", "2026_12_G2_H2", "2026_12_G3_H3"],
                          "home_score": [24, 20, 24], "away_score": [20, 17, 20]})
    store.grade(games, graded_at=KO + pd.Timedelta(hours=4))
    b = store.bets().set_index("game_id")
    # G1 home -2.5, margin +4 -> win, 1 unit at -110 pays 100/110 = 0.90909
    assert b.loc["2026_12_G1_H1", "result"] == "win"
    assert b.loc["2026_12_G1_H1", "pnl_units"] == pytest.approx(0.909091, abs=1e-6)
    # G2 away +3.5, margin +3 -> 3.5 - 3 > 0 win; G3 over 44.5, total 44 -> loss
    assert b.loc["2026_12_G2_H2", "result"] == "win"
    assert (b.loc["2026_12_G3_H3", "result"], b.loc["2026_12_G3_H3", "pnl_units"]) == ("loss", -1.0)
    assert (b.status == "graded").all()
    # a stat correction: a later result row wins, the old one stays in the table
    store.grade(games.assign(home_score=[22, 20, 24]), graded_at=KO + pd.Timedelta(days=2))
    assert store.bets().set_index("game_id").loc["2026_12_G1_H1", "result"] == "loss"   # 22-20 = 2 < 2.5
    import duckdb
    with duckdb.connect(store.path, read_only=True) as c:
        assert c.execute("SELECT count(*) FROM paper_bet_grades WHERE kind='result'").fetchone()[0] == 6


def test_settle_pushes():
    assert settle("spread", "home", -3, 3, 40) == "push"
    assert settle("spread", "away", 3, 3, 40) == "push"
    assert settle("total", "under", 44, 3, 44) == "push"
    assert settle("moneyline", "home", np.nan, 0, 40) == "push"
    assert settle("moneyline", "away", np.nan, -1, 40) == "win"


def test_clv_summary_tiers(store):
    store.attach_closes(CLOSE_QUOTES, as_of=KO)
    s = clv_summary(store.bets())
    assert list(s.columns) == list(CLV_SUMMARY_COLUMNS)
    s = s.set_index("group")
    assert s.loc["close_tier=pinnacle|market=spread", "mean_line_clv"] == 0.5
    assert s.loc["close_tier=pinnacle|market=spread", "n_price_clv"] == 1
    assert s.loc["close_tier=sharp_median|market=spread", "n_price_clv"] == 0
    miss = s.loc["close_tier=missing|market=spread"]
    assert miss.n_bets == 1 and miss.n_line_clv == 0 and math.isnan(miss.mean_price_clv)
    weak = s.loc["close_tier=all_book_median|market=total"]
    assert bool(weak.is_weak_fallback) and weak.mean_line_clv == 1.0
    assert not s.drop(index="close_tier=all_book_median|market=total").is_weak_fallback.any()
    assert clv_summary(store.bets().assign(status="open")).empty
