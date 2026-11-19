# MCP specification

Server: `src/nfl_market/mcp_server/server.py` (`python -m nfl_market.mcp_server.server`, SDK 1.x `FastMCP`
or 2.x `MCPServer`). Read-only: there are no write tools and no wager placement.

**Service parity.** The analytics tools call `AnalyticsService` (`src/nfl_market/services/analytics_service.py`),
the same object the dashboard uses, and return its frames as JSON records (`frame_records`). For the same
inputs a tool returns exactly the service result (`tests/test_mcp.py` asserts this per tool).

## Tools (exist)
| Tool | Service call | Notes |
|---|---|---|
| `get_nfl_slate()` | `MarketService.slate()` | events in the quote store |
| `get_market_snapshot(event_id=None)` | `MarketService.quotes()` | stored quote snapshots |
| `get_resource_registry()` | `MarketService.providers()` | provider registry |
| `get_system_health()` | `MarketService.health()` | health from stored data |
| `get_edges(as_of=None, min_ev=None, limit=50)` | `AnalyticsService.edges` | `EDGE_COLUMNS`, best EV first; `as_of` ISO-8601 (naive = UTC); `limit` 1-500 |
| `get_bet_log(season=None, limit=200)` | `AnalyticsService.bet_log` | `BET_LOG_COLUMNS`, paper bets only; season 2025 (holdout) is refused |
| `get_clv_summary(by=None)` | `AnalyticsService.clv_summary` | `CLV_SUMMARY_COLUMNS`; `by` subset of `["close_tier","market"]` and must include `close_tier` |
| `get_model_health()` | `AnalyticsService.model_health` | `MODEL_HEALTH_COLUMNS` |
| `get_fair_prices(game_id=None)` | `AnalyticsService.prices` | `PRICE_COLUMNS` |
| `get_overview()` | `AnalyticsService.overview` | counts, latest capture, health status, bet share by close tier, `synthetic_present` |

## Resources (exist)
`nfl://slate/current`, `nfl://data/providers`, `nfl://analytics/clv` (default CLV summary),
`nfl://models/health`, `nfl://markets/edges` (default `get_edges()`).

## Rules
* Invalid input raises `ValueError` with a readable message (returned as a tool error).
* Result caps: `limit` at most 500; prices at most 2000 rows; CLV summary at most 500; model health at most 2000.
* JSON-safe output: timestamps ISO-8601 UTC ending in `Z`, missing values `null` (a missing close is `null`, never 0).
* Line CLV and price CLV are separate fields; every CLV value carries its close tier; rows with
  `is_weak_fallback=true` are the all-book median fallback.
* Rows with `synthetic=true` are demo data.

## Planned (not built)
Earlier planned `get_fair_probability` and `get_model_performance` now exist as `get_fair_prices` and
`get_model_health`.

Tools: get_spread_market, get_total_market, get_best_available_price, get_pinnacle_benchmark, get_market_movement,
get_market_dispersion, predict_clv, get_clv_distribution, get_game_projection, calculate_expected_value,
get_execution_analysis, rank_opportunities, get_prediction_history, grade_predictions, get_data_quality,
run_backtest, compare_models, get_existing_ats_prediction.

Resources: nfl://markets/current; nfl://markets/{event_id}/spread; nfl://markets/{event_id}/total;
nfl://models/registry; nfl://models/ats/latest; nfl://models/clv/latest; nfl://predictions/latest;
nfl://data/quality; nfl://system/health.

Security: read-only by default, no wager placement, validate inputs, cap result sizes, avoid secrets. Still
planned: audit logging of calls and authentication before any remote exposure.
