"""Read-only MCP server over the local quote store.

Transports: ``stdio`` (default, local), ``sse`` and ``streamable-http``. Network
transports bind to ``NFLMI_MCP_HOST``/``NFLMI_MCP_PORT`` (127.0.0.1:8000 by
default) and have no authentication: keep them on localhost unless secured.
Works with the MCP Python SDK 1.x (``FastMCP``) and 2.x (``MCPServer``).
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from ..config import settings
from ..services.market_service import MarketService

TRANSPORTS = ("stdio", "sse", "streamable-http")


def to_jsonable(value: Any) -> Any:
    """Convert pandas/numpy/datetime values into JSON-safe primitives. Timestamps become ISO-8601 UTC."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        ts = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return ts.isoformat().replace("+00:00", "Z")
    if isinstance(value, datetime):
        ts = pd.Timestamp(value)
        return to_jsonable(ts)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def frame_records(frame: pd.DataFrame) -> list[dict]:
    return [to_jsonable(r) for r in frame.astype(object).to_dict("records")]


def _server_class():
    try:
        from mcp.server.mcpserver import MCPServer  # SDK 2.x

        return MCPServer, 2
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP  # SDK 1.x

        return FastMCP, 1
    except ImportError as e:
        raise RuntimeError("Install with pip install -e '.[mcp]'") from e


def build_server(service: MarketService | None = None, host: str | None = None, port: int | None = None):
    cls, major = _server_class()
    host = host or settings.mcp_host
    port = port or settings.mcp_port
    m = cls("Market Hawk") if major == 2 else cls("Market Hawk", host=host, port=port)
    m._market_hawk_net = {"host": host, "port": port, "sdk_major": major}
    s = service or MarketService()

    @m.tool()
    def get_nfl_slate() -> list[dict]:
        "Return events represented in the local quote store."
        return to_jsonable(s.slate())

    @m.tool()
    def get_market_snapshot(event_id: str | None = None) -> list[dict]:
        "Return stored quote snapshots, optionally for one event. Timestamps are ISO-8601 UTC."
        return frame_records(s.quotes(event_id))

    @m.tool()
    def get_resource_registry() -> list[dict]:
        "Return provider resources and verification status."
        return to_jsonable(s.providers())

    @m.tool()
    def get_system_health() -> dict:
        "Return health derived from the data actually stored."
        return to_jsonable(s.health())

    @m.resource("nfl://slate/current", mime_type="application/json")
    def current_slate() -> str:
        return json.dumps(to_jsonable(s.slate()))

    @m.resource("nfl://data/providers", mime_type="application/json")
    def providers() -> str:
        return json.dumps(to_jsonable(s.providers()))

    return m


def run(transport: str | None = None, host: str | None = None, port: int | None = None) -> None:
    transport = transport or settings.mcp_transport
    if transport not in TRANSPORTS:
        raise ValueError(f"Unsupported MCP transport {transport!r}; choose one of {TRANSPORTS}")
    m = build_server(host=host, port=port)
    net = m._market_hawk_net
    if transport == "stdio" or net["sdk_major"] == 1:
        m.run(transport=transport)
    else:
        m.run(transport=transport, host=net["host"], port=net["port"])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Market Hawk read-only MCP server")
    p.add_argument("--transport", choices=TRANSPORTS, default=None, help="default: NFLMI_MCP_TRANSPORT or stdio")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    a = p.parse_args(argv)
    run(a.transport, a.host, a.port)


if __name__ == "__main__":
    main()
