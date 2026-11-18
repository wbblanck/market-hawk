MCP entry point: `python -m nfl_market.mcp_server.server` (or the `nflmi-mcp` script).

Transports: `stdio` (default), `sse`, `streamable-http`. Choose with `--transport` or `NFLMI_MCP_TRANSPORT`; network transports bind to `NFLMI_MCP_HOST`/`NFLMI_MCP_PORT` (127.0.0.1:8000). Network transports have no authentication: add authentication and a security review before any remote deployment.

All tool and resource output is JSON-safe; timestamps are ISO-8601 UTC strings ending in `Z`, and missing values are `null`.
