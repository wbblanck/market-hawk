from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from nfl_market.config import ClosePolicy
from nfl_market.markets.closing import (
    FLAG_FEW_BOOKS,
    FLAG_KICKOFF_CHANGED,
    FLAG_MARKET_MEDIAN_FALLBACK,
    RULE_BOOK_PRIMARY,
    RULE_MOST_BALANCED,
    canonical_book,
    close_table,
    select_close,
    select_main_line,
)

KICKOFF = datetime(2024, 9, 8, 17, 0, tzinfo=UTC)
CUTOFF = KICKOFF - timedelta(minutes=15)
POLICY = ClosePolicy()
EVENT = "2024_01_PIT_ATL"


def at(minutes_before_kickoff: float) -> datetime:
    return KICKOFF - timedelta(minutes=minutes_before_kickoff)


def spread_pair(book, home_line, home_odds, away_odds, t, **extra):
    """Both sides of one spread line; away's own line is -home_line."""
    base = {"event_id": EVENT, "book": book, "market": "spread", "captured_at_utc": t, "is_live": False,
                "synthetic": False, "kickoff_utc": KICKOFF}
    base.update(extra)
    return [dict(base, selection="home", line=home_line, odds_american=home_odds),
            dict(base, selection="away", line=-home_line, odds_american=away_odds)]


def frame(*pairs):
    return pd.DataFrame([row for pair in pairs for row in pair])


def close(q, **kw):
    kw.setdefault("event_id", EVENT)
    kw.setdefault("market", "spread")
    kw.setdefault("selection", "home")
    return select_close(q, KICKOFF, POLICY, **kw)


# --- book names -------------------------------------------------------------------------

@pytest.mark.parametrize("raw,canon", [
    ("Pinnacle", "pinnacle"), ("PINNACLE", "pinnacle"),
    ("Circa", "circa"), ("Circa Sports", "circa"), ("circasports", "circa"), ("circa_sports", "circa"),
    ("BookMaker", "bookmaker"), ("bookmaker.eu", "bookmaker"), ("Bookmaker.EU", "bookmaker"),
    ("DraftKings", "draftkings"),
])
def test_canonical_book(raw, canon):
    assert canonical_book(raw) == canon


# --- tiers --------------------------------------------------------------------------------

def test_tier1_pinnacle_wins_over_everything():
    q = frame(spread_pair("Pinnacle", -3, -110, -110, at(20)),
              spread_pair("Circa Sports", -3.5, -105, -115, at(20)),
              spread_pair("bookmaker.eu", -3.5, -110, -110, at(20)),
              spread_pair("DraftKings", -2.5, -120, 100, at(20)))
    r = close(q)
    assert r.close_tier == "pinnacle"
    assert r.close_line == -3
    assert r.close_odds_american == -110 and r.close_opposite_odds_american == -110
    assert r.books_used == ("pinnacle",)
    assert r.quote_age_minutes == pytest.approx(5)
    assert r.flags == () and r.missing_reason is None and r.fallback_reason is None


def test_tier2_sharp_median_needs_both_circa_and_bookmaker():
    q = frame(spread_pair("CIRCA", -3, -110, -110, at(20)),
              spread_pair("BookMaker", -4, -110, -110, at(30)),
              spread_pair("FanDuel", -10, -110, -110, at(20)))
    r = close(q)
    assert r.close_tier == "sharp_median"
    assert r.close_line == -3.5
    assert r.books_used == ("bookmaker", "circa")
    assert r.close_odds_american is None  # no single quote behind a median
    assert r.quote_age_minutes == pytest.approx(15)
    assert FLAG_MARKET_MEDIAN_FALLBACK not in r.flags
    assert "pinnacle: 0 of 1" in r.fallback_reason


def test_tier2_with_one_sharp_book_falls_to_tier3_with_flags():
    q = frame(spread_pair("Circa", -3, -110, -110, at(20)),
              spread_pair("DraftKings", -3.5, -110, -110, at(20)))
    r = close(q)
    assert r.close_tier == "all_book_median"
    assert r.close_line == -3.25
    assert r.books_used == ("circa", "draftkings")
    assert FLAG_MARKET_MEDIAN_FALLBACK in r.flags
    assert FLAG_FEW_BOOKS in r.flags  # 2 books < locked minimum of 5
    assert "sharp_median: 1 of 2 required books (bookmaker=no_quote_before_cutoff)" in r.fallback_reason


def test_tier3_enough_books_no_few_books_flag():
    q = frame(*(spread_pair(b, -3, -110, -110, at(20)) for b in ("DraftKings", "FanDuel", "BetMGM", "Caesars", "BetRivers")))
    r = close(q)
    assert r.close_tier == "all_book_median"
    assert r.flags == (FLAG_MARKET_MEDIAN_FALLBACK,)
    assert r.n_books == 5


def test_missing_close_stays_missing():
    r = close(frame(spread_pair("Pinnacle", -3, -110, -110, at(20))), event_id="other_game")
    assert r.is_missing and r.close_tier == "missing"
    assert r.close_line is None and r.close_fair_probability is None and r.close_odds_american is None
    assert r.missing_reason == "no_quote_before_cutoff"
    assert r.books_used == () and r.n_books == 0


def test_stale_quote_is_missing_not_carried_forward():
    # Pinnacle's last pre-cutoff quote is 3 hours old; a fresh post-cutoff quote must not rescue it.
    q = frame(spread_pair("Pinnacle", -3, -110, -110, at(15 + 180)),
              spread_pair("Pinnacle", -3.5, -110, -110, at(5)))
    r = close(q)
    assert r.is_missing
    assert r.missing_reason == "stale"
    assert "pinnacle=stale" in r.fallback_reason


def test_stale_pinnacle_falls_to_fresh_sharp_books():
    q = frame(spread_pair("Pinnacle", -3, -110, -110, at(15 + 121)),
              spread_pair("Circa", -2.5, -110, -110, at(16)),
              spread_pair("BookMaker", -2.5, -110, -110, at(16)))
    r = close(q)
    assert r.close_tier == "sharp_median"
    assert "pinnacle=stale" in r.fallback_reason


def test_quote_exactly_at_cutoff_and_at_max_age_qualifies():
    r = close(frame(spread_pair("Pinnacle", -3, -110, -110, CUTOFF)))
    assert r.close_tier == "pinnacle" and r.quote_age_minutes == 0
    r = close(frame(spread_pair("Pinnacle", -3, -110, -110, CUTOFF - timedelta(minutes=120))))
    assert r.close_tier == "pinnacle" and r.quote_age_minutes == pytest.approx(120)


def test_post_cutoff_quotes_ignored():
    q = frame(spread_pair("Pinnacle", -3, -110, -110, at(30)),
              spread_pair("Pinnacle", -6, -110, -110, at(14)),   # after cutoff
              spread_pair("Pinnacle", -7, -110, -110, at(1)))
    r = close(q)
    assert r.close_line == -3
    assert r.quote_age_minutes == pytest.approx(15)


@pytest.mark.parametrize("flag", ["is_live", "synthetic"])
def test_live_and_synthetic_quotes_excluded(flag):
    q = frame(spread_pair("Pinnacle", -3, -110, -110, at(40)),
              spread_pair("Pinnacle", -6, -110, -110, at(20), **{flag: True}))
    r = close(q)
    assert r.close_line == -3
    only_bad = frame(spread_pair("Pinnacle", -6, -110, -110, at(20), **{flag: True}))
    assert close(only_bad).is_missing


def test_uses_captured_at_not_source_timestamp():
    # source timestamp claims pre-cutoff, but we captured it after the cutoff: ignored.
    q = frame(spread_pair("Pinnacle", -3, -110, -110, at(40)),
              spread_pair("Pinnacle", -6, -110, -110, at(5), source_timestamp_utc=at(60)))
    assert close(q).close_line == -3


def test_away_selection_uses_own_line():
    q = frame(spread_pair("Pinnacle", -3, -110, -110, at(20)))
    assert close(q, selection="away").close_line == 3


def test_kickoff_change_is_flagged():
    q = frame(spread_pair("Pinnacle", -3, -110, -110, at(20)))
    q["kickoff_utc"] = KICKOFF - timedelta(hours=3)
    assert FLAG_KICKOFF_CHANGED in close(q).flags


# --- main line ------------------------------------------------------------------------------

def test_main_line_book_flag_branch():
    q = frame(spread_pair("Pinnacle", -3, -110, -110, at(20), is_main=False),
              spread_pair("Pinnacle", -3.5, +105, -125, at(20), is_main=True),
              spread_pair("Pinnacle", -2.5, -130, +110, at(20), is_main=False))
    ml = select_main_line(q, "home")
    assert ml.line == -3.5 and ml.rule == RULE_BOOK_PRIMARY
    assert ml.odds_american == 105 and ml.opposite_odds_american == -125


def test_main_line_most_balanced_branch():
    q = frame(spread_pair("Pinnacle", -3, -112, -108, at(20)),
              spread_pair("Pinnacle", -3.5, +105, -125, at(20)),
              spread_pair("Pinnacle", -2.5, -130, +110, at(20)))
    ml = select_main_line(q, "home")
    assert ml.line == -3 and ml.rule == RULE_MOST_BALANCED
    assert ml.fair_probability == pytest.approx(0.5, abs=0.01)


def test_main_line_ambiguous_flags_fall_back_to_balanced():
    q = frame(spread_pair("Pinnacle", -3, -110, -110, at(20), is_main=True),
              spread_pair("Pinnacle", -3.5, +105, -125, at(20), is_main=True))
    assert select_main_line(q, "home").rule == RULE_MOST_BALANCED


def test_main_line_rule_recorded_on_close_row():
    flagged = frame(spread_pair("Pinnacle", -3, -110, -110, at(20), is_main=False),
                    spread_pair("Pinnacle", -3.5, +105, -125, at(20), is_main=True))
    r = close(flagged)
    assert r.close_line == -3.5 and r.main_line_rule == RULE_BOOK_PRIMARY
    balanced = frame(spread_pair("Pinnacle", -3, -110, -110, at(20)),
                     spread_pair("Pinnacle", -3.5, +105, -125, at(20)))
    r = close(balanced)
    assert r.close_line == -3 and r.main_line_rule == RULE_MOST_BALANCED
    mixed = frame(spread_pair("Circa", -3, -110, -110, at(20), is_main=True),
                  spread_pair("BookMaker", -3, -110, -110, at(20)))
    assert close(mixed).main_line_rule == "mixed"


def test_main_line_uses_last_snapshot_only():
    # An alternate line pulled before the close snapshot must not be considered.
    q = frame(spread_pair("Pinnacle", -3, -140, +120, at(40)),
              spread_pair("Pinnacle", -2.5, -110, -110, at(40)),
              spread_pair("Pinnacle", -3, -110, -110, at(20)))
    r = close(q)
    assert r.close_line == -3 and r.quote_age_minutes == pytest.approx(5)


# --- exact-contract price close --------------------------------------------------------------

def test_price_close_requires_exact_contract():
    q = frame(spread_pair("Pinnacle", -3, -120, +100, at(20)),
              spread_pair("Pinnacle", -2.5, -135, +115, at(20)))
    r = close(q, line=-2.5)
    assert r.close_kind == "price" and r.close_tier == "pinnacle"
    assert r.close_line == -2.5 and r.close_odds_american == -135
    assert r.main_line_rule is None
    assert 0.5 < r.close_fair_probability < 0.6
    assert close(q, line=-3.5).is_missing  # Pinnacle never offered -3.5


def test_price_close_needs_opposite_side():
    q = frame(spread_pair("Pinnacle", -3, -110, -110, at(20)))
    q = q[q.selection == "home"]
    r = close(q, line=-3)
    assert r.is_missing and r.missing_reason == "no_opposite_side"


def test_price_close_sharp_median_of_devigged_probabilities():
    q = frame(spread_pair("Circa", -3, -120, +100, at(20)),
              spread_pair("BookMaker", -3, -110, -110, at(20)))
    r = close(q, line=-3)
    assert r.close_tier == "sharp_median"
    assert r.close_fair_probability == pytest.approx((0.5 + 120 / 220 / (120 / 220 + 0.5)) / 2)


def test_totals_pair_over_under_at_same_line():
    base = {"event_id": EVENT, "book": "Pinnacle", "market": "total", "captured_at_utc": at(20), "is_live": False,
                "synthetic": False}
    q = pd.DataFrame([dict(base, selection="over", line=44.5, odds_american=-105),
                      dict(base, selection="under", line=44.5, odds_american=-115),
                      dict(base, selection="over", line=45.5, odds_american=110),
                      dict(base, selection="under", line=45.5, odds_american=-130)])
    r = close(q, market="total", selection="under")
    assert r.close_line == 44.5 and r.main_line_rule == RULE_MOST_BALANCED
    r = close(q, market="total", selection="over", line=45.5)
    assert r.close_odds_american == 110 and r.close_opposite_odds_american == -130


def test_close_table_rows_record_tier_and_fingerprint():
    q = frame(spread_pair("Pinnacle", -3, -110, -110, at(20)))
    req = pd.DataFrame([{"event_id": EVENT, "market": "spread", "selection": "home", "kickoff_utc": KICKOFF, "line": None},
                        {"event_id": EVENT, "market": "spread", "selection": "home", "kickoff_utc": KICKOFF, "line": -7.0}])
    out = close_table(q, req, POLICY)
    assert list(out.close_tier) == ["pinnacle", "missing"]
    assert list(out.close_kind) == ["line", "price"]
    assert (out.policy_fingerprint == POLICY.fingerprint()).all()


def test_custom_cutoff_policy():
    q = frame(spread_pair("Pinnacle", -3, -110, -110, at(10)))
    assert close(q).is_missing
    r = select_close(q, KICKOFF, ClosePolicy(cutoff_minutes=5), event_id=EVENT, market="spread", selection="home")
    assert r.close_tier == "pinnacle"


# --- sharp feed provider (feeds the close; no network) ---------------------------------------

ODDS_API_EVENT = {
    "id": "abc123", "sport_key": "americanfootball_nfl", "commence_time": "2024-09-08T17:00:00Z",
    "home_team": "Atlanta Falcons", "away_team": "Pittsburgh Steelers",
    "bookmakers": [
        {"key": "pinnacle", "title": "Pinnacle", "last_update": "2024-09-08T16:40:00Z", "markets": [
            {"key": "spreads", "last_update": "2024-09-08T16:40:00Z", "outcomes": [
                {"name": "Atlanta Falcons", "price": -105, "point": 3.0},
                {"name": "Pittsburgh Steelers", "price": -115, "point": -3.0}]},
            {"key": "totals", "last_update": "2024-09-08T16:40:00Z", "outcomes": [
                {"name": "Over", "price": -110, "point": 42.5},
                {"name": "Under", "price": -110, "point": 42.5}]},
            {"key": "h2h", "outcomes": [{"name": "Atlanta Falcons", "price": 130}]}]},
        {"key": "draftkings", "markets": [{"key": "spreads", "outcomes": [
            {"name": "Atlanta Falcons", "price": -110, "point": 3.5}]}]},
    ],
}


def test_odds_api_parser_maps_sharp_books_only():
    from nfl_market.integrations.sharp import parse_odds_api_events

    qs = parse_odds_api_events([ODDS_API_EVENT])
    assert {q.venue for q in qs} == {"pinnacle"}
    got = {(q.market, q.selection, q.line, q.odds_american) for q in qs}
    assert got == {("spread", "home", 3.0, -105), ("spread", "away", -3.0, -115),
                   ("total", "over", 42.5, -110), ("total", "under", 42.5, -110)}
    assert all("public_site_feed_may_delay" in q.quality_flags for q in qs)
    assert all(q.source_timestamp_utc == datetime(2024, 9, 8, 16, 40, tzinfo=UTC) for q in qs)
    assert all(q.kickoff_utc == KICKOFF for q in qs)


def test_odds_api_provider_needs_key_and_never_calls_without_it(monkeypatch):
    from nfl_market.integrations.base import ProviderUnavailable
    from nfl_market.integrations.sharp import OddsApiProvider

    monkeypatch.delenv("NFLMI_ODDS_API_KEY", raising=False)
    p = OddsApiProvider(get=lambda url, params: pytest.fail("no call without a key"))
    assert not p.available()
    with pytest.raises(ProviderUnavailable):
        p.fetch_quotes()
    seen = {}
    p = OddsApiProvider("k", get=lambda url, params: seen.update(params) or [ODDS_API_EVENT])
    assert len(p.fetch_quotes(date="2024-09-08")) == 4 and seen["bookmakers"] == "pinnacle"
    assert p.fetch_quotes(date="2024-09-09") == []


def test_parlay_provider_reads_existing_key_env_and_all_books(monkeypatch):
    from nfl_market.integrations.sharp import parlay_api_provider

    monkeypatch.delenv("NFLMI_PARLAY_API_KEY", raising=False)
    monkeypatch.delenv("PARLAY_API_KEY", raising=False)
    assert not parlay_api_provider().available()
    monkeypatch.setenv("PARLAY_API_KEY", "x")
    seen = {}
    p = parlay_api_provider(books=(), get=lambda url, params: seen.update(params, url=url) or [ODDS_API_EVENT])
    qs = p.fetch_quotes()
    assert seen["url"].startswith("https://parlay-api.com/v1/") and "bookmakers" not in seen
    assert {q.venue for q in qs} == {"pinnacle", "draftkings"}
    assert all(q.provider == "parlay_api" for q in qs)
    assert not any("public_site_feed_may_delay" in q.quality_flags for q in qs)
