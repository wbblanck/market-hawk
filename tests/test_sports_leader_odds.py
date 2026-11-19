"""Sports Leader get_game_odds -> sportsbook ProviderQuote, against the upstream NFL golden fixture."""
import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nfl_market.data.validation import validate_quotes
from nfl_market.integrations.base import VENUE_SPORTSBOOK, ProviderQuote
from nfl_market.integrations.sports_leader.odds import game_odds_to_quotes, to_american

FIX = Path(__file__).parent / "fixtures" / "sports_leader"


def load(name):
    return json.loads((FIX / f"{name}.json").read_text())


def by_key(quotes):
    return {(q.market, q.selection): q for q in quotes}


def test_golden_nfl_game_odds_convert_to_six_sides():
    quotes = game_odds_to_quotes(load("get_game_odds"))
    assert all(isinstance(q, ProviderQuote) for q in quotes)
    q = by_key(quotes)
    assert set(q) == {("spread", "home"), ("spread", "away"), ("total", "over"), ("total", "under"),
                      ("moneyline", "home"), ("moneyline", "away")}
    assert (q["spread", "home"].line, q["spread", "home"].odds_american) == (-6.5, -115)
    assert (q["spread", "away"].line, q["spread", "away"].odds_american) == (6.5, -105)
    assert (q["total", "over"].line, q["total", "over"].odds_american) == (42.5, -115)
    assert (q["total", "under"].line, q["total", "under"].odds_american) == (42.5, -105)
    assert (q["moneyline", "home"].line, q["moneyline", "home"].odds_american) == (None, -285)
    assert (q["moneyline", "away"].line, q["moneyline", "away"].odds_american) == (None, 230)
    for x in quotes:
        assert x.provider == "sports_leader" and x.venue == "Draft Kings" and x.venue_kind == VENUE_SPORTSBOOK
        assert x.provider_event_id == "401872981" and x.provider_market_id == "100"
        assert x.home_team == "JAX" and x.away_team == "PHI"
        assert x.kickoff_utc == datetime(2026, 10, 11, 13, 30, tzinfo=UTC)
        assert x.source_timestamp_utc is None and "no_source_timestamp" in x.quality_flags
        assert x.event_id is None and "unmapped_event" in x.quality_flags
        assert x.raw["provider"] == "Draft Kings" and x.raw["open"]["homeSpread"] == "+2.5"  # raw kept whole
        json.dumps(x.raw)


def test_open_lines_are_never_used():
    quotes = game_odds_to_quotes(load("get_game_odds"))
    lines = {q.line for q in quotes}
    assert not lines & {44.5, 2.5, -2.5}


def test_spread_pair_is_exact_opposite_and_total_shared():
    q = by_key(game_odds_to_quotes(load("get_game_odds")))
    assert q["spread", "home"].line == -q["spread", "away"].line
    assert q["total", "over"].line == q["total", "under"].line
    assert not any("spread_pair_not_opposite" in x.quality_flags for x in q.values())


def test_spread_and_total_rows_pass_repo_validation():
    quotes = [x for x in game_odds_to_quotes(load("get_game_odds")) if x.market in ("spread", "total")]
    cap = datetime(2026, 10, 10, 12, tzinfo=UTC)
    frame = pd.DataFrame([{"snapshot_id": f"s{i}", "event_id": "e", "captured_at_utc": cap,
                           "kickoff_utc": x.kickoff_utc, "book": x.venue, "market": x.market,
                           "selection": x.selection, "line": x.line, "odds_american": x.odds_american,
                           "source": x.provider} for i, x in enumerate(quotes)])
    assert validate_quotes(frame) == []


def test_season_week_give_nflverse_style_event_id():
    q = game_odds_to_quotes(load("get_game_odds"), season=2026, week=6)[0]
    assert q.event_id == "2026_06_PHI_JAX" and (q.season, q.week) == (2026, 6)
    assert "unmapped_event" not in q.quality_flags


def test_missing_price_skips_side_and_flags_partner():
    resp = load("get_game_odds")
    book = resp["odds"][0]
    del book["away"]["spreadOdds"]
    book["underOdds"] = None
    q = by_key(game_odds_to_quotes(resp))
    assert ("spread", "away") not in q and ("total", "under") not in q
    assert "partner_missing_price" in q["spread", "home"].quality_flags
    assert "partner_missing_price" in q["total", "over"].quality_flags
    assert q["spread", "home"].odds_american == -115  # nothing invented
    assert "partner_missing_price" not in q["moneyline", "home"].quality_flags


def test_missing_current_block_falls_back_to_normalized_fields_with_flags():
    resp = load("get_game_odds")
    del resp["odds"][0]["current"]
    q = by_key(game_odds_to_quotes(resp))
    assert q["spread", "home"].line == -6.5 and q["spread", "away"].line == 6.5
    assert "away_line_derived_from_home" in q["spread", "away"].quality_flags
    assert "line_from_normalized_field" in q["total", "over"].quality_flags
    assert q["total", "over"].line == 42.5


def test_non_opposite_pair_is_flagged_not_fixed():
    resp = load("get_game_odds")
    resp["odds"][0]["current"]["awaySpread"] = "+7"
    q = by_key(game_odds_to_quotes(resp))
    assert q["spread", "away"].line == 7.0
    assert "spread_pair_not_opposite" in q["spread", "home"].quality_flags


def test_invalid_odds_are_dropped_and_flagged():
    resp = load("get_game_odds")
    resp["odds"][0]["home"]["moneyLine"] = 50  # not valid American odds
    resp["odds"][0]["homeMoneyLine"] = None
    resp["odds"][0]["current"]["homeMoneyLine"] = "n/a"
    q = by_key(game_odds_to_quotes(resp))
    assert ("moneyline", "home") not in q
    assert "partner_missing_price" in q["moneyline", "away"].quality_flags
    assert to_american("EVEN") == (100, None)
    assert to_american(None) == (None, None)
    assert to_american(0) == (None, "invalid_odds")


def test_no_books_gives_no_quotes_and_input_not_mutated():
    resp = load("get_game_odds")
    snapshot = copy.deepcopy(resp)
    game_odds_to_quotes(resp)
    assert resp == snapshot
    resp["odds"] = []
    assert game_odds_to_quotes(resp) == []
