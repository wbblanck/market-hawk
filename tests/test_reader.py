"""StoreMarketReader: contract frames on an empty/missing DB and a round trip through the writers."""
import json

import pandas as pd

from nfl_market.data.store import MarketStore
from nfl_market.distributions.pmf import IntPMF
from nfl_market.markets.bet_log import BetLogStore
from nfl_market.markets.interface import (
    BET_LOG_COLUMNS,
    CLV_SUMMARY_COLUMNS,
    EDGE_COLUMNS,
    MODEL_HEALTH_COLUMNS,
    PRICE_COLUMNS,
    MarketReader,
)
from nfl_market.markets.pricing import append_fair_prices, price_game
from nfl_market.markets.reader import StoreMarketReader
from nfl_market.markets.staking import size_stakes

KO = pd.Timestamp("2026-11-22 18:00", tz="UTC")
GID = "2026_12_BUF_KC"


def _check_empty(r):
    for fn, cols in ((r.prices, PRICE_COLUMNS), (r.edges, EDGE_COLUMNS), (r.bet_log, BET_LOG_COLUMNS),
                     (r.clv_summary, CLV_SUMMARY_COLUMNS), (r.model_health, MODEL_HEALTH_COLUMNS)):
        f = fn()
        assert list(f.columns) == list(cols) and f.empty


def test_missing_and_empty_db(tmp_path):
    r = StoreMarketReader(str(tmp_path / "absent.duckdb"), tmp_path / "absent.json")
    assert isinstance(r, MarketReader)
    _check_empty(r)
    assert not (tmp_path / "absent.duckdb").exists()             # reading never creates the file
    db = tmp_path / "empty.duckdb"
    MarketStore(str(db))                                          # quote table only
    _check_empty(StoreMarketReader(str(db), tmp_path / "absent.json"))


def test_round_trip(tmp_path):
    db = str(tmp_path / "m.duckdb")
    T = KO - pd.Timedelta(hours=6)
    pmf = IntPMF.from_arrays([-7, -3, 0, 7], [0.10, 0.10, 0.05, 0.75])
    rows = price_game({"game_id": GID, "season": 2026, "week": 12, "kickoff_utc": KO, "prediction_time_utc": T},
                      pmf, spread_lines=[-3.5], model_id="elo_v1", dist_id="k1@2015-2024")
    append_fair_prices(db, pd.DataFrame(rows))
    quotes = pd.DataFrame([
        {"snapshot_id": f"s{i}", "event_id": GID, "season": 2026, "week": 12, "kickoff_utc": KO,
         "captured_at_utc": KO - pd.Timedelta(minutes=90), "book": "Pinnacle", "market": "spread",
         "selection": sel, "line": line, "odds_american": -110, "is_live": False, "source": "test", "synthetic": False}
        for i, (sel, line) in enumerate((("home", -3.5), ("away", 3.5)))])
    MarketStore(db).append_quotes(quotes)
    r = StoreMarketReader(db, tmp_path / "absent.json")
    assert len(r.prices()) == len(rows) and r.prices(["nope"]).empty
    as_of = KO - pd.Timedelta(minutes=60)
    e = r.edges(as_of)
    assert list(e.columns) == list(EDGE_COLUMNS)
    # home -3.5 covers iff M > 3.5: P = .75 ; EV = .75 * .90909 - .25 = 0.4318
    assert len(e) == 2 and e.iloc[0].ev_per_unit == __import__("pytest").approx(0.75 / 1.1 - 0.25)
    assert r.edges(as_of, min_ev=0.5).empty
    stakes = size_stakes(e)
    BetLogStore(db).record(stakes[stakes.stake_units > 0], decision_time=as_of)
    b = r.bet_log()
    assert list(b.columns) == list(BET_LOG_COLUMNS) and len(b) == 1
    assert r.bet_log(season=2024).empty
    assert list(r.clv_summary().columns) == list(CLV_SUMMARY_COLUMNS)


def test_model_health_json(tmp_path):
    p = tmp_path / "h.json"
    p.write_text(json.dumps([{"model_id": "stack_v1", "scope": "pooled", "fold_id": "", "n_games": 10}]))
    h = StoreMarketReader(str(tmp_path / "x.duckdb"), p).model_health()
    assert list(h.columns) == list(MODEL_HEALTH_COLUMNS) and h.iloc[0].model_id == "stack_v1"
    p.write_text("not json")
    assert StoreMarketReader(str(tmp_path / "x.duckdb"), p).model_health().empty
