import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from nfl_market.mcp_server.server import TRANSPORTS, frame_records, run, to_jsonable


def test_timestamps_serialize_as_iso_utc():
    f = pd.DataFrame({"t": [pd.Timestamp("2026-09-01T08:00", tz="America/New_York"), pd.NaT],
                      "n": [np.int64(3), None], "x": [1.5, float("nan")]})
    recs = frame_records(f)
    assert recs[0]["t"] == "2026-09-01T12:00:00Z" and recs[1]["t"] is None
    assert recs[0]["n"] == 3 and recs[1]["x"] is None
    json.dumps(recs)  # must be JSON-serialisable
    assert to_jsonable(datetime(2026, 9, 1, 12, tzinfo=timezone.utc)) == "2026-09-01T12:00:00Z"


def test_transports_declared_and_validated():
    assert TRANSPORTS == ("stdio", "sse", "streamable-http")
    with pytest.raises(ValueError, match="Unsupported"):
        run(transport="websocket")


def test_build_server_registers_tools(tmp_path):
    pytest.importorskip("mcp")
    from nfl_market.mcp_server.server import build_server
    from nfl_market.services.market_service import MarketService
    m = build_server(MarketService(str(tmp_path / "db.duckdb")))
    assert m is not None


# ---------------------------------------------------------------- step-4 analytics tools
ANALYTICS_TOOLS = ("get_edges", "get_bet_log", "get_clv_summary", "get_model_health", "get_fair_prices", "get_overview")
STARTER_TOOLS = ("get_nfl_slate", "get_market_snapshot", "get_resource_registry", "get_system_health")


@pytest.fixture
def analytics_server(tmp_path):
    pytest.importorskip("mcp")
    from test_analytics_service import FakeReader

    from nfl_market.mcp_server.server import build_server
    from nfl_market.services.analytics_service import AnalyticsService
    from nfl_market.services.market_service import MarketService
    ms = MarketService(str(tmp_path / "db.duckdb"))
    a = AnalyticsService(FakeReader(), ms)
    return build_server(ms, a), a


def _tool(m, name):
    return m._tool_manager.get_tool(name).fn


def test_analytics_tools_registered_and_no_write_tools(analytics_server):
    m, _ = analytics_server
    names = {t.name for t in m._tool_manager.list_tools()}
    assert set(ANALYTICS_TOOLS) | set(STARTER_TOOLS) == names
    assert not any(w in n for n in names for w in ("place", "write", "record", "delete", "update", "set_"))


@pytest.mark.parametrize("tool, kwargs, method, method_kwargs", [
    ("get_edges", {}, "edges", {}),
    ("get_edges", {"as_of": "2026-11-22T17:00:00Z", "min_ev": 0.06, "limit": 1}, "edges",
     {"as_of": "2026-11-22T17:00:00Z", "min_ev": 0.06, "limit": 1}),
    ("get_bet_log", {}, "bet_log", {}),
    ("get_bet_log", {"season": 2024, "limit": 3}, "bet_log", {"season": 2024, "limit": 3}),
    ("get_clv_summary", {}, "clv_summary", {"by": None}),
    ("get_clv_summary", {"by": ["close_tier"]}, "clv_summary", {"by": ["close_tier"]}),
    ("get_model_health", {}, "model_health", {}),
    ("get_fair_prices", {}, "prices", {}),
    ("get_fair_prices", {"game_id": "2026_12_BUF_KC"}, "prices", {"game_ids": ["2026_12_BUF_KC"]}),
])
def test_tool_parity_with_analytics_service(analytics_server, tool, kwargs, method, method_kwargs):
    m, a = analytics_server
    got = _tool(m, tool)(**kwargs)
    assert got == frame_records(getattr(a, method)(**method_kwargs))
    json.dumps(got)


def test_overview_tool_parity(analytics_server, monkeypatch):
    m, a = analytics_server
    fixed = a.overview(now=datetime(2026, 11, 22, 17, tzinfo=timezone.utc))
    monkeypatch.setattr(a, "overview", lambda: fixed)
    assert _tool(m, "get_overview")() == to_jsonable(fixed)
    json.dumps(_tool(m, "get_overview")())


def test_tools_through_sdk_call_path(analytics_server):
    """The SDK's own call path (argument validation + execution) returns the service's records."""
    import asyncio

    m, a = analytics_server
    if not hasattr(m, "call_tool"):
        pytest.skip("SDK without MCPServer.call_tool")
    res = asyncio.run(m.call_tool("get_edges", {"min_ev": 0.0, "limit": 2}))
    if isinstance(res, tuple):  # SDK 1.x: (content, structured)
        payload = res[1]
    else:  # SDK 2.x: CallToolResult
        payload = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
    assert isinstance(payload, dict)
    assert payload["result"] == frame_records(a.edges(min_ev=0.0, limit=2))


@pytest.mark.parametrize("tool, kwargs, msg", [
    ("get_edges", {"limit": 0}, "limit"),
    ("get_edges", {"limit": 501}, "limit"),
    ("get_edges", {"min_ev": float("nan")}, "min_ev"),
    ("get_edges", {"as_of": "yesterday-ish"}, "as_of"),
    ("get_bet_log", {"season": 2025}, "holdout"),
    ("get_bet_log", {"limit": 10_000}, "limit"),
    ("get_clv_summary", {"by": ["market"]}, "close_tier"),
    ("get_clv_summary", {"by": ["close_tier", "book"]}, "unknown"),
    ("get_fair_prices", {"game_id": ""}, "game_id"),
])
def test_tool_input_validation(analytics_server, tool, kwargs, msg):
    m, _ = analytics_server
    with pytest.raises(ValueError, match=msg):
        _tool(m, tool)(**kwargs)


def test_analytics_resources(analytics_server):
    import asyncio

    m, a = analytics_server
    uris = {str(r.uri) for r in asyncio.run(m.list_resources())}
    assert {"nfl://analytics/clv", "nfl://models/health", "nfl://markets/edges"} <= uris
    for uri, expected in (("nfl://analytics/clv", a.clv_summary()), ("nfl://models/health", a.model_health()),
                          ("nfl://markets/edges", a.edges())):
        contents = list(asyncio.run(m.read_resource(uri)))
        assert json.loads(contents[0].content) == frame_records(expected)


def test_default_server_on_empty_db(tmp_path):
    pytest.importorskip("mcp")
    from nfl_market.mcp_server.server import build_server
    from nfl_market.services.market_service import MarketService
    m = build_server(MarketService(str(tmp_path / "db.duckdb")))
    assert _tool(m, "get_bet_log")() == [] and _tool(m, "get_clv_summary")() == []
    assert _tool(m, "get_edges")() == [] and _tool(m, "get_fair_prices")() == []
    assert _tool(m, "get_overview")()["health_status"] == "no_data"
