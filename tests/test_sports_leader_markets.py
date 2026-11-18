"""Sports Leader get_market_odds -> prediction-market ProviderQuote.

Fixtures were produced by running the upstream server's own get_market_odds tool against its
recorded NFL Kalshi / Gamma / Polymarket US fixtures (test/fixtures/{nfl,markets})."""
import json
from datetime import UTC, datetime
from pathlib import Path

from nfl_market.integrations.base import VENUE_PREDICTION_MARKET
from nfl_market.integrations.sports_leader.mapping import (
    match_contract_to_game,
    normalize_team,
)
from nfl_market.integrations.sports_leader.prediction_markets import (
    market_odds_to_quotes,
)

FIX = Path(__file__).parent / "fixtures" / "sports_leader"


def load(name):
    return json.loads((FIX / f"{name}.json").read_text())


def keyed(quotes):
    return {(q.venue, q.selection): q for q in quotes}


def test_all_three_venues_side_by_side_never_averaged():
    quotes, issues = market_odds_to_quotes(load("get_market_odds_401872981"))
    assert issues == []
    q = keyed(quotes)
    assert set(q) == {(v, s) for v in ("kalshi", "polymarket", "polymarket_us") for s in ("home", "away")}
    for x in quotes:
        assert x.venue_kind == VENUE_PREDICTION_MARKET and x.market == "winner" and x.line is None
        assert x.odds_american is None and x.provider_event_id == "401872981"
        assert (x.home_team, x.away_team) == ("JAX", "PHI")
        json.dumps(x.raw)

    k_home, k_away = q["kalshi", "home"], q["kalshi", "away"]
    assert k_home.provider_market_id == "KXNFLGAME-26OCT11PHIJAC-JAC"
    assert (k_home.bid, k_home.ask, k_home.price) == (0.57, 0.59, 0.58)
    assert k_away.provider_market_id == "KXNFLGAME-26OCT11PHIJAC-PHI"
    assert (k_away.bid, k_away.ask, k_away.price) == (0.41, 0.43, 0.42)
    assert "price_is_last_trade" in k_home.quality_flags and k_home.size is None

    p_home = q["polymarket", "home"]
    assert p_home.provider_market_id == "nfl-phi-jax-2026-10-11"
    assert (p_home.price, p_home.bid, p_home.ask) == (0.58, None, None)
    assert p_home.raw["outcome"]["name"] == "Jaguars"

    us_home, us_away = q["polymarket_us", "home"], q["polymarket_us", "away"]
    assert us_home.provider_market_id == "aec-nfl-phi-jax-2026-10-11"
    assert (us_home.bid, us_home.ask, us_home.price) == (0.57, 0.59, 0.58)
    assert (us_away.bid, us_away.ask, us_away.price) == (0.41, 0.43, 0.42)
    assert "price_is_mid" in us_home.quality_flags


def test_fetched_at_is_stored_but_flagged_as_fetch_time():
    quotes, _ = market_odds_to_quotes(load("get_market_odds_401872981"))
    for x in quotes:
        assert x.source_timestamp_utc == datetime(2026, 11, 18, 23, 49, 34, 792000, tzinfo=UTC)
        assert "source_ts_is_fetch_time" in x.quality_flags


def test_missing_fetched_at_flags_no_source_timestamp():
    resp = load("get_market_odds_401872981")
    del resp["fetchedAt"]
    quotes, _ = market_odds_to_quotes(resp)
    assert quotes and all(x.source_timestamp_utc is None and "no_source_timestamp" in x.quality_flags for x in quotes)


def test_no_market_found_yields_no_quotes_and_no_issue():
    quotes, issues = market_odds_to_quotes(load("get_market_odds_401872990_postponed"))
    assert {x.venue for x in quotes} == {"kalshi"}
    assert issues == []
    assert all("market_date_mismatch" in x.quality_flags for x in quotes)  # postponed: ticker keeps old date


def test_venue_error_is_an_issue_and_other_venues_survive():
    quotes, issues = market_odds_to_quotes(load("get_market_odds_kalshi_error"))
    assert {x.venue for x in quotes} == {"polymarket", "polymarket_us"}
    assert len(issues) == 1 and issues[0]["venue"] == "kalshi" and "500" in issues[0]["reason"]


def test_event_id_mismatch_is_rejected():
    quotes, issues = market_odds_to_quotes(load("get_market_odds_401872981"), requested_event_id="999")
    assert quotes == [] and {i["reason"] for i in issues} == {"market_event_id_mismatch"}


def test_clob_book_prices_use_book_bid_ask_and_mid():
    resp = load("get_market_odds_401872981")
    poly = resp["venues"][1]["events"][0]["markets"][0]
    poly["outcomes"][1] = {"name": "Jaguars", "tokenId": "222", "gammaPrice": 0.58, "bid": 0.56, "ask": 0.6,
                           "mid": 0.58, "priceSource": "clob_book"}
    q = keyed(market_odds_to_quotes(resp)[0])
    assert (q["polymarket", "home"].bid, q["polymarket", "home"].ask, q["polymarket", "home"].price) == (0.56, 0.6, 0.58)
    assert "price_is_mid" in q["polymarket", "home"].quality_flags


def test_contract_matching_uses_server_matched_by():
    ok = match_contract_to_game({"status": "ok", "matchedBy": "teams+scheduledDate"},
                                requested_event_id="1", response_event_id="1", game_matched_by="eventId")
    assert ok.matched and ok.flags == ()
    swapped = match_contract_to_game(
        {"status": "ok", "matchedBy": "teams+eventDate (ticker lists the teams in the other order)+constructed slug"},
        requested_event_id="1", response_event_id="1")
    assert swapped.matched and {"market_teams_swapped", "market_slug_constructed"} <= set(swapped.flags)
    none = match_contract_to_game({"status": "no market found", "matchedBy": None},
                                  requested_event_id="1", response_event_id="1")
    assert not none.matched
    by_team = match_contract_to_game({"status": "ok", "matchedBy": "teams+eventDate"},
                                     requested_event_id="1", response_event_id="1", game_matched_by="team")
    assert "espn_game_not_matched_by_event_id" in by_team.flags


def test_team_normalization_to_nflverse_codes():
    assert normalize_team("LAR") == "LA" and normalize_team("wsh") == "WAS" and normalize_team("JAX") == "JAX"
    assert normalize_team(None) is None
