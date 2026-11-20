"""AnalyticsService: validation, caps, contract columns and pass-through behaviour.

The fake reader returns small hand-built frames. Hand-checked numbers used below:

* -110 decimal = 1 + 100/110 = 1.90909; breakeven = 110/210 = 0.52381;
  ev at p_win 0.55, p_loss 0.45 = 0.55 * 0.90909 - 0.45 = 0.05000.
* +150 decimal = 2.5; breakeven = 100/250 = 0.4; ev at p_win 0.45, p_loss 0.55 = 0.45 * 1.5 - 0.55 = 0.125.
* -105 at p_win 0.48, p_loss 0.52: ev = 0.48 * (100/105) - 0.52 = 0.45714 - 0.52 = -0.06286.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from nfl_market.markets.interface import (
    BET_LOG_COLUMNS,
    CLV_SUMMARY_COLUMNS,
    EDGE_COLUMNS,
    MODEL_HEALTH_COLUMNS,
    PRICE_COLUMNS,
    MarketReader,
)
from nfl_market.services.analytics_service import (
    MAX_LIMIT,
    AnalyticsService,
    EmptyMarketReader,
    ev_plus_eligible,
    tier_label,
    tier_shares,
)
from nfl_market.services.market_service import MarketService

NOW = pd.Timestamp("2026-11-22T17:00", tz="UTC")
KO = pd.Timestamp("2026-11-23T18:00", tz="UTC")


def _edge(**kw) -> dict:
    base = {c: np.nan for c in EDGE_COLUMNS}
    base.update(game_id="2026_12_BUF_KC", season=2026, week=12, kickoff_utc=KO,
                captured_at_utc=NOW - pd.Timedelta(minutes=10), quote_age_minutes=10.0, model_id="elo_v1",
                dist_id="k1@2015-2024", p_push=0.0, flags="", synthetic=False)
    base.update(kw)
    return base


def edges_frame() -> pd.DataFrame:
    rows = [
        _edge(book="pinnacle", market="spread", selection="home", line=-3.5, odds_american=-110,
              opposite_odds_american=-110, p_win=0.55, p_loss=0.45, p_win_no_push=0.55,
              breakeven_prob=110 / 210, market_fair_prob=0.5, edge_prob=0.55 - 110 / 210,
              edge_vs_market=0.05, ev_per_unit=0.05),
        _edge(book="circa", market="moneyline", selection="away", odds_american=150, p_win=0.45, p_loss=0.55,
              p_win_no_push=0.45, breakeven_prob=0.4, edge_prob=0.05, ev_per_unit=0.125, flags="STALE_QUOTE"),
        _edge(game_id="DEMO_BUF_NYJ", book="Book A", market="total", selection="over", line=44.5,
              odds_american=-105, p_win=0.48, p_loss=0.52, p_win_no_push=0.48, breakeven_prob=105 / 205,
              edge_prob=0.48 - 105 / 205, ev_per_unit=0.48 * 100 / 105 - 0.52, synthetic=True),
    ]
    return pd.DataFrame(rows)[list(EDGE_COLUMNS)].sort_values("ev_per_unit", ascending=False).reset_index(drop=True)


def _bet(i: int, **kw) -> dict:
    base = {c: np.nan for c in BET_LOG_COLUMNS}
    base.update(bet_id=f"b{i}", game_id="2026_12_BUF_KC", season=2026, week=12, kickoff_utc=KO,
                decision_time_utc=NOW - pd.Timedelta(hours=i), quote_captured_at_utc=NOW - pd.Timedelta(hours=i),
                book="circa", market="spread", selection="home", line=-3.0, odds_american=-110,
                opposite_odds_american=-110, model_id="elo_v1", dist_id="k1@2015-2024", p_win=0.55, p_push=0.08,
                ev_per_unit=0.03, kelly_fraction=0.02, stake_units=0.5, stake_limit_reason="", status="closed",
                line_close_tier="", price_close_tier="", close_flags="", close_policy_fingerprint="abc123",
                result="", synthetic=False)
    base.update(kw)
    return base


def bet_frame() -> pd.DataFrame:
    rows = [
        # bet -3, close -3.5 -> line CLV = -3 - (-3.5) = +0.5 (README sign convention)
        _bet(1, line_close=-3.5, line_close_tier="pinnacle", line_clv=0.5, price_close_tier="pinnacle",
             price_clv=np.nan, close_flags="LINE_MOVED"),
        _bet(2, line_close=-3.0, line_close_tier="sharp_median", line_clv=0.0, price_close_fair_prob=0.53,
             price_close_tier="sharp_median", price_clv=0.03),
        _bet(3, line_close=-2.5, line_close_tier="all_book_median", line_clv=-0.5, price_close_tier="all_book_median",
             close_flags="FEW_BOOKS"),
        _bet(4, status="open"),  # close not attached yet
        _bet(5, season=2024, game_id="2024_10_BUF_KC", line_close_tier="missing", price_close_tier="missing",
             status="graded", result="win", pnl_units=0.4545),
    ]
    return pd.DataFrame(rows)[list(BET_LOG_COLUMNS)]


def clv_frame() -> pd.DataFrame:
    rows = [
        dict(group="close_tier=pinnacle|market=spread", close_tier="pinnacle", market="spread", n_bets=1, n_line_clv=1,
             mean_line_clv=0.5, share_positive_line_clv=1.0, n_price_clv=0, mean_price_clv=np.nan,
             share_positive_price_clv=np.nan, is_weak_fallback=False),
        dict(group="close_tier=all_book_median|market=spread", close_tier="all_book_median", market="spread",
             n_bets=1, n_line_clv=1, mean_line_clv=-0.5, share_positive_line_clv=0.0, n_price_clv=0,
             mean_price_clv=np.nan, share_positive_price_clv=np.nan, is_weak_fallback=True),
    ]
    return pd.DataFrame(rows)[list(CLV_SUMMARY_COLUMNS)]


def health_frame() -> pd.DataFrame:
    rows = []
    for model, w in (("stack_v1", np.nan), ("elo_v1", 0.6), ("ridge_v1", 0.4)):
        for scope, fold in (("pooled", ""), ("clean", ""), ("fold", "2024")):
            rows.append(dict(model_id=model, scope=scope, fold_id=fold, n_games=272, mae_margin=10.1, mae_total=10.4,
                             logloss_win=0.66, brier_win=0.23, ece_win=0.03, cover_logloss=np.nan,
                             stack_weight_margin=w, stack_weight_total=w, experiment_id="20261119-step4-stacking-01",
                             generated_at_utc="2026-11-19T12:00:00Z"))
    return pd.DataFrame(rows)[list(MODEL_HEALTH_COLUMNS)]


def prices_frame() -> pd.DataFrame:
    rows = [dict(game_id="2026_12_BUF_KC", season=2026, week=12, kickoff_utc=KO, model_id="elo_v1",
                 dist_id="k1@2015-2024", prediction_time_utc=NOW, market="spread", selection=sel, line=line,
                 p_win=pw, p_push=0.08, p_loss=0.92 - pw, p_win_no_push=pw / 0.92, fair_american=np.nan)
            for sel, line, pw in (("home", -3.0, 0.46), ("away", 3.0, 0.46))]
    return pd.DataFrame(rows)[list(PRICE_COLUMNS)]


class FakeReader:
    """Hand-built contract frames; records the arguments it was called with."""

    def __init__(self, n_extra_edges: int = 0):
        self.calls: list[tuple] = []
        self._edges = edges_frame()
        if n_extra_edges:
            extra = pd.concat([self._edges.iloc[[2]]] * n_extra_edges, ignore_index=True)
            self._edges = pd.concat([self._edges, extra], ignore_index=True)

    def prices(self, game_ids=None):
        self.calls.append(("prices", game_ids))
        f = prices_frame()
        return f if game_ids is None else f[f.game_id.isin(game_ids)]

    def edges(self, as_of=None, *, min_ev=None):
        self.calls.append(("edges", as_of, min_ev))
        f = self._edges
        return f if min_ev is None else f[f.ev_per_unit >= min_ev]

    def bet_log(self, season=None):
        self.calls.append(("bet_log", season))
        f = bet_frame()
        return f if season is None else f[f.season == season]

    def clv_summary(self, by=("close_tier", "market")):
        self.calls.append(("clv_summary", tuple(by)))
        return clv_frame()

    def model_health(self):
        self.calls.append(("model_health",))
        return health_frame()


@pytest.fixture
def market_service(tmp_path):
    return MarketService(str(tmp_path / "db.duckdb"))


@pytest.fixture
def svc(market_service):
    return AnalyticsService(FakeReader(), market_service)


def test_fake_reader_satisfies_protocol():
    assert isinstance(FakeReader(), MarketReader)
    assert isinstance(EmptyMarketReader(), MarketReader)


def test_edges_keep_contract_columns_and_hand_checked_values(svc):
    e = svc.edges()
    assert list(e.columns) == list(EDGE_COLUMNS)
    assert list(e.book) == ["circa", "pinnacle", "Book A"]  # best EV first: 0.125, 0.05, -0.0629
    assert e.ev_per_unit.iloc[0] == pytest.approx(0.125)
    assert e.ev_per_unit.iloc[1] == pytest.approx(0.05, abs=1e-9)
    assert e.breakeven_prob.iloc[1] == pytest.approx(0.52381, abs=1e-5)
    assert e.ev_per_unit.iloc[2] == pytest.approx(-0.06286, abs=1e-5)


def test_edges_min_ev_limit_and_as_of_parsing(svc):
    e = svc.edges(as_of="2026-11-22T12:00:00-05:00", min_ev=0.06, limit=1)
    assert list(e.book) == ["circa"]
    _, as_of, min_ev = svc.reader.calls[-1]
    assert as_of == NOW and str(as_of.tz) == "UTC" and min_ev == 0.06
    assert svc.edges(as_of="2026-11-22T17:00").shape[0] == 3  # naive = UTC
    assert svc.reader.calls[-1][1] == NOW


def test_edges_result_cap():
    s = AnalyticsService(FakeReader(n_extra_edges=MAX_LIMIT + 10), MarketService(":memory:"))
    assert len(s.edges(limit=MAX_LIMIT)) == MAX_LIMIT


@pytest.mark.parametrize("kw, msg", [
    ({"limit": 0}, "limit"), ({"limit": MAX_LIMIT + 1}, "limit"), ({"limit": True}, "limit"), ({"limit": "5"}, "limit"),
    ({"min_ev": float("nan")}, "min_ev"), ({"min_ev": float("inf")}, "min_ev"), ({"min_ev": "0.1"}, "min_ev"),
    ({"as_of": "not a date"}, "as_of"), ({"as_of": 12345}, "as_of"),
])
def test_edges_validation(svc, kw, msg):
    with pytest.raises(ValueError, match=msg):
        svc.edges(**kw)


def test_bet_log_season_filter_limit_and_validation(svc):
    b = svc.bet_log()
    assert list(b.columns) == list(BET_LOG_COLUMNS) and len(b) == 5
    assert list(svc.bet_log(season=2024).bet_id) == ["b5"]
    assert len(svc.bet_log(limit=2)) == 2
    # missing closes stay missing (NaN), never 0
    assert math.isnan(b.loc[b.bet_id == "b4", "line_clv"].iloc[0])
    for bad in (2025, "2024", 1800, 2024.0, True):
        with pytest.raises(ValueError, match="season"):
            svc.bet_log(season=bad)
    with pytest.raises(ValueError, match="holdout"):
        svc.bet_log(season=2025)


def test_clv_summary_by_validation(svc):
    assert list(svc.clv_summary().columns) == list(CLV_SUMMARY_COLUMNS)
    svc.clv_summary(by=["market", "close_tier"])
    assert svc.reader.calls[-1] == ("clv_summary", ("close_tier", "market"))  # normalised order
    svc.clv_summary(by=None)
    assert svc.reader.calls[-1] == ("clv_summary", ("close_tier", "market"))
    for bad, msg in ((["market"], "close_tier"), (["close_tier", "book"], "unknown"),
                     (["close_tier", "close_tier"], "duplicate"), ("close_tier", "list")):
        with pytest.raises(ValueError, match=msg):
            svc.clv_summary(by=bad)


def test_prices_and_game_id_validation(svc):
    p = svc.prices(["2026_12_BUF_KC"])
    assert list(p.columns) == list(PRICE_COLUMNS) and len(p) == 2
    assert svc.prices(["2026_01_NOPE_X"]).empty
    for bad in ([], "2026_12_BUF_KC", [""], [1], ["x" * 65], ["g"] * 101):
        with pytest.raises(ValueError, match="game_id"):
            svc.prices(bad)


def test_model_health_passthrough(svc):
    h = svc.model_health()
    assert list(h.columns) == list(MODEL_HEALTH_COLUMNS) and len(h) == 9


def test_reader_missing_contract_column_raises(market_service):
    class Bad(FakeReader):
        def model_health(self):
            return health_frame().drop(columns=["ece_win"])

    with pytest.raises(ValueError, match="ece_win"):
        AnalyticsService(Bad(), market_service).model_health()


def test_ev_plus_badge_only_for_clean_real_positive_rows():
    e = edges_frame()
    # circa: ev 0.125 but flagged STALE_QUOTE -> no badge; pinnacle: clean, real, ev 0.05 -> badge;
    # Book A: synthetic and negative -> no badge.
    assert list(ev_plus_eligible(e)) == [False, True, False]
    e.loc[1, "synthetic"] = True
    assert not ev_plus_eligible(e).any()
    assert ev_plus_eligible(e.iloc[0:0]).empty


def test_tier_helpers():
    shares = tier_shares(bet_frame()["line_close_tier"])
    # 5 bets: pinnacle, sharp_median, all_book_median, pending (""), missing -> 0.2 each
    assert shares == {"all_book_median": 0.2, "missing": 0.2, "pending": 0.2, "pinnacle": 0.2, "sharp_median": 0.2}
    assert tier_label("all_book_median") == "all_book_median (weak fallback)"
    assert tier_label("pinnacle") == "pinnacle" and tier_label("") == "pending"


def test_overview(svc):
    o = svc.overview(now=datetime(2026, 11, 22, 17, tzinfo=timezone.utc))
    assert o["generated_at_utc"] == "2026-11-22T17:00:00Z"
    assert o["health_status"] == "no_data" and o["quote_rows"] == 0
    assert o["edge_rows"] == 3 and o["bet_count"] == 5 and o["price_rows"] == 2 and o["priced_games"] == 1
    assert o["bets_by_status"] == {"closed": 3, "graded": 1, "open": 1}
    assert o["bets_by_line_close_tier"]["pending"] == 0.2
    assert o["model_health_rows"] == 9 and o["model_health_generated_at_utc"] == "2026-11-19T12:00:00Z"
    assert o["synthetic_present"] is True  # the Book A edge row is synthetic


def test_empty_reader_behaviour(market_service):
    s = AnalyticsService(EmptyMarketReader(), market_service)
    for f, cols in ((s.edges(), EDGE_COLUMNS), (s.bet_log(), BET_LOG_COLUMNS), (s.prices(), PRICE_COLUMNS),
                    (s.clv_summary(), CLV_SUMMARY_COLUMNS), (s.model_health(), MODEL_HEALTH_COLUMNS)):
        assert f.empty and list(f.columns) == list(cols)
    assert not ev_plus_eligible(s.edges()).any()
    o = s.overview()
    assert o["bet_count"] == 0 and o["bets_by_line_close_tier"] == {} and o["synthetic_present"] is False
    assert o["model_health_generated_at_utc"] is None


def test_default_reader_against_real_empty_duckdb(tmp_path):
    """Default reader (StoreMarketReader when present) on an empty tmp DuckDB returns empty contract frames."""
    ms = MarketService(str(tmp_path / "empty.duckdb"))
    s = AnalyticsService(market_service=ms, model_health_path=str(tmp_path / "no_such_health.json"))
    assert isinstance(s.reader, MarketReader)
    for f, cols in ((s.edges(), EDGE_COLUMNS), (s.bet_log(), BET_LOG_COLUMNS), (s.prices(), PRICE_COLUMNS),
                    (s.clv_summary(), CLV_SUMMARY_COLUMNS), (s.model_health(), MODEL_HEALTH_COLUMNS)):
        assert f.empty and list(f.columns) == list(cols)
    assert s.overview()["health_status"] == "no_data"
