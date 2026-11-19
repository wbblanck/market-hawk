"""SportsLeaderProvider end to end with a fake transport, plus optional-dependency behaviour.
No network, no Node: every tool call is answered from recorded fixtures."""
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nfl_market.integrations.base import MarketDataProvider, ProviderUnavailable
from nfl_market.integrations.sports_leader import SportsLeaderProvider, nfl_data
from nfl_market.integrations.sports_leader.client import (
    ENV_COMMAND,
    SportsLeaderClient,
    SportsLeaderToolError,
    result_to_dict,
)

FIX = Path(__file__).parent / "fixtures" / "sports_leader"


def load(name):
    return json.loads((FIX / f"{name}.json").read_text())


class FakeTransport:
    """Answers get_scoreboard / get_game_odds / get_market_odds from fixtures; unknown games raise."""

    def __init__(self):
        self.calls = []

    def __call__(self, name, args):
        self.calls.append((name, dict(args)))
        if name == "get_scoreboard":
            return load("get_scoreboard")
        if name == "get_game_odds" and args["eventId"] == "401872981":
            return load("get_game_odds")
        if name == "get_market_odds":
            if args["eventId"] == "401872981":
                return load("get_market_odds_401872981")
            if args["eventId"] == "401872990":
                return load("get_market_odds_401872990_postponed")
        raise SportsLeaderToolError(f"Upstream error 404 for {name} {args.get('eventId')}")


def test_provider_satisfies_protocol():
    p = SportsLeaderProvider(FakeTransport())
    assert isinstance(p, MarketDataProvider) and p.name == "sports_leader" and p.available()


def test_fetch_by_date_end_to_end_with_isolated_failures():
    fake = FakeTransport()
    p = SportsLeaderProvider(fake)
    quotes = p.fetch_quotes(date="2026-10-11")
    first = fake.calls[0]
    assert first == ("get_scoreboard", {"sport": "football", "league": "nfl", "date": "20261011"})
    books = [q for q in quotes if q.venue_kind == "sportsbook"]
    markets = [q for q in quotes if q.venue_kind == "prediction_market"]
    assert len(books) == 6 and {q.provider_event_id for q in books} == {"401872981"}
    assert {(q.provider_event_id, q.venue) for q in markets} == {
        ("401872981", "kalshi"), ("401872981", "polymarket"), ("401872981", "polymarket_us"), ("401872990", "kalshi")}
    assert len(markets) == 8
    # 13 games on the board; failures for the 12 without odds and 11 without markets are recorded, not raised.
    stages = [i["stage"] for i in p.last_fetch_issues]
    assert stages.count("sportsbook_odds") == 12 and stages.count("prediction_markets") == 11
    # identity comes from the scoreboard
    q = next(q for q in quotes if q.provider_event_id == "401872990")
    assert q.home_team and q.away_team and q.kickoff_utc is not None and "unmapped_event" in q.quality_flags


def test_fetch_by_event_ids_skips_scoreboard_and_maps_with_season_week():
    fake = FakeTransport()
    quotes = SportsLeaderProvider(fake, season=2026, week=6).fetch_quotes(event_ids=["401872981"])
    assert [c[0] for c in fake.calls] == ["get_game_odds", "get_market_odds"]
    assert len(quotes) == 12 and {q.event_id for q in quotes} == {"2026_06_PHI_JAX"}


def test_final_games_are_skipped_and_live_games_flagged():
    board = load("get_scoreboard")
    board["events"][0]["status"]["state"] = "in"
    board["events"][1]["status"]["state"] = "post"
    fake = FakeTransport()

    def call(name, args):
        return board if name == "get_scoreboard" else fake(name, args)

    p = SportsLeaderProvider(call)
    quotes = p.fetch_quotes(date="2026-10-11")
    assert {q.provider_event_id for q in quotes} == {"401872981"}
    assert all("game_in_progress" in q.quality_flags for q in quotes)
    assert {"provider_event_id": "401872990", "stage": "game", "reason": "game_final"} in p.last_fetch_issues


def test_scoreboard_pagination_follows_page_note():
    board = load("get_scoreboard")
    pages = {0: dict(board, events=board["events"][:1], page={"offset": 0, "returned": 1, "total": 2, "note": "n"}),
             1: dict(board, events=board["events"][1:2], page={"offset": 1, "returned": 1, "total": 2, "note": "n"})}
    seen = []

    def call(name, args):
        if name == "get_scoreboard":
            seen.append(args.get("offset", 0))
            return pages[args.get("offset", 0)]
        return FakeTransport()(name, args)

    quotes = SportsLeaderProvider(call).fetch_quotes(date="2026-10-11")
    assert seen == [0, 1] and {q.provider_event_id for q in quotes} == {"401872981", "401872990"}


def test_scoreboard_failure_propagates():
    def call(name, args):
        raise SportsLeaderToolError("down")

    with pytest.raises(SportsLeaderToolError):
        SportsLeaderProvider(call).fetch_quotes(date="2026-10-11")


def test_unconfigured_provider_is_unavailable_and_raises(monkeypatch):
    monkeypatch.delenv(ENV_COMMAND, raising=False)
    p = SportsLeaderProvider()
    assert p.available() is False
    with pytest.raises(ProviderUnavailable):
        p.fetch_quotes(date="2026-10-11")


def test_import_and_unavailable_without_mcp_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "mcp", None)  # makes `import mcp` raise ImportError
    for mod in [m for m in sys.modules if m.startswith("nfl_market.integrations.sports_leader")]:
        monkeypatch.delitem(sys.modules, mod)
    pkg = importlib.import_module("nfl_market.integrations.sports_leader")
    monkeypatch.setenv(ENV_COMMAND, "node /nonexistent/dist/index.js")
    p = pkg.SportsLeaderProvider()
    assert p.available() is False
    with pytest.raises(ProviderUnavailable):
        p.fetch_quotes(date="2026-10-11")


def test_available_with_command_and_sdk(monkeypatch):
    pytest.importorskip("mcp")
    monkeypatch.setenv(ENV_COMMAND, "node /path/dist/index.js")
    assert SportsLeaderProvider().available() is True  # no process started, no network
    assert SportsLeaderClient().command == ["node", "/path/dist/index.js"]


def test_result_to_dict_handles_sdk_shapes_and_errors():
    text = SimpleNamespace(type="text", text='{"a": 1}')
    assert result_to_dict(SimpleNamespace(is_error=False, structured_content={"b": 2}, content=[text])) == {"b": 2}
    assert result_to_dict(SimpleNamespace(isError=False, structuredContent=None, content=[text])) == {"a": 1}
    err = SimpleNamespace(is_error=True, structured_content=None, content=[SimpleNamespace(text="Upstream error 500")])
    with pytest.raises(SportsLeaderToolError, match="500"):
        result_to_dict(err)


def test_client_drops_none_args_and_rejects_truncated_results():
    seen = []

    def call(name, args):
        seen.append(args)
        return {"truncated": True, "message": "too big"}

    with pytest.raises(SportsLeaderToolError, match="truncated"):
        SportsLeaderClient(call).call("get_scoreboard", sport="football", date=None)
    assert seen == [{"sport": "football"}]


def test_nfl_data_wrappers_return_frames_marked_as_statistics():
    seen = []

    def call(name, args):
        seen.append((name, args))
        return {"dataset": "games", "league": "nfl", "seasons": [2025], "updated_at": "2026-01-01T00:00:00Z",
                "rowCount": 1, "columns": ["game_id", "game_date", "home_team_abbr"],
                "rows": [{"game_id": 401772959, "game_date": "2025-09-07", "home_team_abbr": "JAX"}]}

    client = SportsLeaderClient(call)
    frame = nfl_data.games(client, season=2025, date_from="2025-09-01", team="JAX")
    assert list(frame.columns) == ["game_id", "game_date", "home_team_abbr"] and len(frame) == 1
    assert frame.attrs["kind"] == "statistics_not_quotes" and frame.attrs["updated_at"] == "2026-01-01T00:00:00Z"
    assert seen[0] == ("get_games", {"league": "nfl", "season": 2025, "team": "JAX", "from": "2025-09-01"})
    nfl_data.player_game_logs(client, "rushing", player="Jacobs")
    assert seen[1][0] == "get_player_game_logs" and seen[1][1]["statGroup"] == "rushing"
    with pytest.raises(ValueError):
        nfl_data.player_game_logs(client, "batting")
    for fn, tool in [(nfl_data.team_box_scores, "get_team_box_scores"), (nfl_data.rosters, "get_rosters"),
                     (nfl_data.injury_reports, "get_injury_reports"), (nfl_data.depth_charts, "get_depth_charts")]:
        fn(client, team="JAX")
        assert seen[-1][0] == tool
