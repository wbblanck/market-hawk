MCP entry point: `python -m nfl_market.mcp_server.server` (or the `nflmi-mcp` script).

Transports: `stdio` (default), `sse`, `streamable-http`. Choose with `--transport` or `NFLMI_MCP_TRANSPORT`; network transports bind to `NFLMI_MCP_HOST`/`NFLMI_MCP_PORT` (127.0.0.1:8000). Network transports have no authentication: add authentication and a security review before any remote deployment.

Read-only tools: `get_nfl_slate`, `get_market_snapshot`, `get_resource_registry`, `get_system_health`, and the step-4 analytics tools `get_edges`, `get_bet_log`, `get_clv_summary`, `get_model_health`, `get_fair_prices`, `get_overview`. Resources: `nfl://slate/current`, `nfl://data/providers`, `nfl://analytics/clv`, `nfl://models/health`, `nfl://markets/edges`. The analytics tools return exactly what the dashboard shows (both call `AnalyticsService`); see `docs/MCP_SPEC.md`. No write tools, no wager placement.

All tool and resource output is JSON-safe; timestamps are ISO-8601 UTC strings ending in `Z`, and missing values are `null`.
