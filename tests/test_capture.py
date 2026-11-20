"""Odds capture: feed parsing, event mapping, capture timing, request budget, export/import round trip."""
from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime

import pandas as pd
import pytest

from nfl_market.collectors.capture import (DEFAULT_BUDGETS, ProviderBudget, budget_check, close_slots, export_runs,
                                           import_export_file, last_remaining, month_spent)
from nfl_market.collectors.snapshot_collector import SnapshotCollector
from nfl_market.config import ClosePolicy
from nfl_market.data.store import MarketStore
from nfl_market.integrations.event_mapping import map_quotes_to_games, team_code
from nfl_market.integrations.sharp import (decimal_to_american, oddslink_provider, parlay_api_provider,
                                           parse_odds_api_events)
from nfl_market.markets.closing import select_close

KO = "2026-11-22T18:00:00Z"
GAMES = pd.DataFrame({
    "game_id": ["2026_12_PIT_ATL", "2026_12_NYJ_KC", "2026_12_LA_SF"],
    "season": [2026, 2026, 2026], "week": [12, 12, 12],
    "kickoff_utc": pd.to_datetime([KO, "2026-11-22T21:05:00Z", "2026-11-22T21:25:00Z"], utc=True),
    "home_team": ["ATL", "KC", "SF"], "away_team": ["PIT", "NYJ", "LA"],
})


def event(books, *, home="Atlanta Falcons", away="Pittsburgh Steelers", commence=KO, decimal=False):
    def px(a):  # American -> the feed's format
        return (1 + a / 100 if a > 0 else 1 + 100 / -a) if decimal else a
    return {"id": "ev1", "commence_time": commence, "home_team": home, "away_team": away, "bookmakers": [
        {"key": b, "last_update": "2026-11-22T17:30:00Z", "markets": [
            {"key": "spreads", "outcomes": [{"name": home, "price": px(-110), "point": -3.0},
                                            {"name": away, "price": px(-110), "point": 3.0}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": px(-105), "point": 44.5},
                                           {"name": "Under", "price": px(-115), "point": 44.5}]},
            {"key": "h2h", "outcomes": [{"name": home, "price": px(-160)}, {"name": away, "price": px(140)}]},
            {"key": "spreads", "period": "h1", "outcomes": [{"name": home, "price": px(-110), "point": -1.5}]},
        ]} for b in books]}


def test_decimal_to_american_hand_values():
    assert [decimal_to_american(x) for x in (2.0, 1.909, 2.5, 1.5, 1.625, 2.4)] == [100, -110, 150, -200, -160, 140]
    assert decimal_to_american(1.0) is None and decimal_to_american("x") is None


def test_parser_keeps_h2h_only_when_asked_and_skips_period_markets():
    default = parse_odds_api_events([event(["pinnacle"])])
    assert {q.market for q in default} == {"spread", "total"}
    assert all(q.line != -1.5 for q in default)  # first-half ladder dropped
    full = parse_odds_api_events([event(["pinnacle"])], markets=("spreads", "totals", "h2h"))
    ml = sorted((q.selection, q.line, q.odds_american) for q in full if q.market == "moneyline")
    assert ml == [("away", None, 140), ("home", None, -160)]
    assert not any("missing_point" in q.quality_flags for q in full if q.market == "moneyline")


def test_team_codes_and_event_mapping():
    assert team_code("Los Angeles Rams") == "LA" and team_code("Washington Commanders") == "WAS"
    assert team_code("kc") == "KC" and team_code("Nowhere Nomads") is None
    qs = map_quotes_to_games(parse_odds_api_events([event(["pinnacle"])]), GAMES)
    assert {q.event_id for q in qs} == {"2026_12_PIT_ATL"} and {(q.season, q.week) for q in qs} == {(2026, 12)}
    flexed = map_quotes_to_games(parse_odds_api_events([event(["pinnacle"], commence="2026-11-23T01:20:00Z")]), GAMES)
    assert all("kickoff_differs_from_schedule" in q.quality_flags for q in flexed)
    unknown = map_quotes_to_games(parse_odds_api_events([event(["pinnacle"], home="Denver Broncos")]), GAMES)
    assert all(q.event_id is None and "unmapped_event" in q.quality_flags for q in unknown)


def test_oddslink_sends_key_in_header_not_query(monkeypatch):
    monkeypatch.setenv("ODDSLINK_API_KEY", "secret-key")
    seen = {}
    p = oddslink_provider(get=lambda url, params: seen.update(params, url=url) or [event(["pinnacle", "circa"], decimal=True)])
    qs = p.fetch_quotes()
    assert seen["url"] == "https://oddslink.app/v1/sports/nfl/odds" and "apiKey" not in seen
    assert seen["bookmakers"] == "pinnacle,circa" and seen["oddsFormat"] == "decimal"
    assert {q.venue for q in qs} == {"pinnacle", "circa"}
    assert all("converted_from_decimal" in q.quality_flags for q in qs)
    assert {q.odds_american for q in qs if q.market == "spread"} == {-110}


def test_close_slots_merge_and_cutoff():
    # 16:50 UTC: the 1 PM ET group (18:00) captures at 17:35; 4:05 and 4:25 merge into one capture at 20:40.
    s = close_slots(GAMES, "2026-11-22T16:50:00Z", window_minutes=300)
    assert [str(x.capture_at_utc) for x in s] == ["2026-11-22 17:35:00+00:00", "2026-11-22 20:40:00+00:00"]
    assert s[1].game_ids == ("2026_12_NYJ_KC", "2026_12_LA_SF")
    # Started late (17:40): still before the 17:45 cutoff, so capture now; at 17:46 the slot is dropped.
    assert close_slots(GAMES, "2026-11-22T17:40:00Z", window_minutes=60)[0].capture_at_utc == pd.Timestamp("2026-11-22T17:40:00Z")
    assert close_slots(GAMES, "2026-11-22T17:46:00Z", window_minutes=60) == []
    with pytest.raises(ValueError):
        close_slots(GAMES, "2026-11-22T16:50:00Z", lead_minutes=15)


def test_budget_cap_floor_and_month_rollover():
    b = ProviderBudget("parlay_api", monthly_cap=10, cost_per_call=3, floor_early=400, floor_close=250)
    now = datetime(2026, 11, 22, 17, 0, tzinfo=UTC)
    use = [{"at": "2026-11-20T15:00:00Z", "provider": "parlay_api", "status": "called", "last": "3", "remaining": "500"},
           {"at": "2026-11-21T15:00:00Z", "provider": "parlay_api", "status": "called", "last": "", "cost_estimate": 3,
            "remaining": "402"},
           {"at": "2026-10-30T15:00:00Z", "provider": "parlay_api", "status": "called", "last": "50", "remaining": "10"},
           {"at": "2026-11-21T16:00:00Z", "provider": "parlay_api", "status": "skipped", "last": "", "remaining": ""}]
    assert month_spent(use, "parlay_api", now) == 6 and last_remaining(use, "parlay_api", now) == 402
    assert budget_check(b, use, now, "close")[0]                 # 6 + 3 <= 10, 402 - 3 >= 250
    assert not budget_check(b, use, now, "early")[0]             # 402 - 3 < 400 (early floor)
    assert not budget_check(b, use + [use[0]], now, "close")[0]  # 9 + 3 > 10 (cap)
    assert budget_check(b, use, datetime(2026, 12, 1, tzinfo=UTC), "early")[0]  # new month: nothing spent yet
    assert not budget_check(DEFAULT_BUDGETS["oddslink"], [], now, "early")[0]   # OddsLink only near the close


def test_capture_export_import_round_trip_gives_a_pinnacle_close(tmp_path, monkeypatch):
    monkeypatch.setenv("PARLAY_API_KEY", "k")
    books = ["pinnacle", "draftkings", "fanduel"]
    prov = parlay_api_provider(books=(), regions="us", markets=("spreads", "totals", "h2h"), games=GAMES,
                               get=lambda url, params: [event(books)])
    db = tmp_path / "run.duckdb"
    clock = lambda: datetime(2026, 11, 22, 17, 35, tzinfo=UTC)  # noqa: E731 - kickoff - 25 min
    report = SnapshotCollector(str(db), [prov], clock=clock).collect()
    assert report.runs[0].status == "ok"
    out = tmp_path / "out" / "x.jsonl.gz"
    counts = export_runs(db, [r.run_id for r in report.runs], out)
    assert counts == {"collection_runs": 1, "market_snapshots": 18, "quote_snapshots": 12}  # 3 books x (6, of which 4 routable)
    if shutil.which("openssl"):  # the workflow's encryption settings round-trip
        enc, dec = tmp_path / "x.enc", tmp_path / "y.jsonl.gz"
        env = {"CAPTURE_PASSPHRASE": "pw", "PATH": "/usr/bin:/bin"}
        base = ["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-iter", "200000", "-pass", "env:CAPTURE_PASSPHRASE"]
        subprocess.run([*base, "-salt", "-in", str(out), "-out", str(enc)], check=True, env=env)
        subprocess.run([*base, "-d", "-in", str(enc), "-out", str(dec)], check=True, env=env)
        assert dec.read_bytes() == out.read_bytes()
    local = tmp_path / "local.duckdb"
    assert import_export_file(out, local) == counts
    assert import_export_file(out, local) == {"collection_runs": 0, "market_snapshots": 0, "quote_snapshots": 0}
    q = MarketStore(str(local)).quotes()
    r = select_close(q, KO, ClosePolicy(), event_id="2026_12_PIT_ATL", market="spread", selection="away")
    assert (r.close_tier, r.close_line, r.close_odds_american, r.quote_age_minutes) == ("pinnacle", 3.0, -110, 10.0)


def test_the_odds_api_is_a_close_only_pinnacle_source(monkeypatch):
    b = DEFAULT_BUDGETS["the_odds_api"]
    now = datetime(2026, 11, 22, 17, 0, tzinfo=UTC)
    assert budget_check(b, [], now, "close")[0] and not budget_check(b, [], now, "early")[0]
    use = [{"at": "2026-11-21T15:00:00Z", "provider": "the_odds_api", "status": "called", "last": "2", "remaining": "151"}]
    assert not budget_check(b, use, now, "close")[0]  # 151 - 2 < 150 floor
    from nfl_market.integrations.sharp import OddsApiProvider

    monkeypatch.setenv("NFLMI_ODDS_API_KEY", "k")
    seen = {}
    p = OddsApiProvider(books=("pinnacle",), markets=("spreads", "totals"), games=GAMES,
                        get=lambda url, params: seen.update(params, url=url) or [event(["pinnacle", "draftkings"])])
    qs = p.fetch_quotes()
    assert seen["url"] == "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    assert seen["bookmakers"] == "pinnacle" and seen["markets"] == "spreads,totals" and "regions" not in seen
    assert {q.venue for q in qs} == {"pinnacle"} and {q.event_id for q in qs} == {"2026_12_PIT_ATL"}
    assert all("public_site_feed_may_delay" in q.quality_flags for q in qs)
