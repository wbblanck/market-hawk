# Dashboard specification

Run: `streamlit run apps/dashboard/app.py` (reads `NFLMI_DATABASE_PATH`, default `data/nfl_market.duckdb`).
The dashboard is read-only. It calls only `AnalyticsService` (`src/nfl_market/services/analytics_service.py`)
and `MarketService`; it computes no prices, edges or CLV itself, only formatting and display filters.
The MCP server exposes the same service methods (`docs/MCP_SPEC.md`).

## What exists (step 4)
| Tab | Source | Shows |
|---|---|---|
| Command Center | `MarketService.quotes()` | stored quotes for one event, grouped by market |
| Edges | `AnalyticsService.edges(min_ev, limit)` | book, contract, odds, p_win/p_push, breakeven, market fair prob, edge, EV per unit, flags, quote capture time and age, model id; filters by market, book, min EV, flagged rows |
| Market Explorer | `MarketService.quotes()` | line movement per selection; price history per exact contract |
| CLV | `AnalyticsService.bet_log(season, limit)`, `clv_summary()` | paper bet log with line CLV and price CLV in separate columns, each with its own close tier; CLV summary split into Pinnacle/sharp median, the all-book median **weak fallback**, and missing |
| Model Health | `AnalyticsService.model_health()` | pooled, clean and per-fold tables; latest stack weights; calibration (ECE, Brier, log loss) |
| Data Health | `MarketService.health()`, `AnalyticsService.overview()` | quote-store status and warnings, row counts, bets by close tier, synthetic flag |
| Data & Resources | `MarketService.providers()` | provider registry |
| MCP | static | how to run the MCP server and its tools |

Display rules implemented:
* A red **SYNTHETIC DEMO DATA** banner appears whenever any displayed row has `synthetic=True`.
* The **EV+** badge (`analytics_service.ev_plus_eligible`) appears only when `ev_per_unit > 0`, `p_win` is
  present, `flags` is empty and the quote is not synthetic. Execution costs and fills are not modelled, so
  the badge is not a recommendation.
* Line CLV (points) and price CLV (probability) are never summed. A missing close shows as `missing`; a
  bet whose close is not attached yet shows as `pending`. Neither is shown as 0.
* `all_book_median` is always labelled "weak fallback" and summarised in its own table.
* Edge and bet tables show a timestamp, model id and flags column. Model-health and CLV-summary rows
  have no flags column in the contract (`src/nfl_market/markets/interface.py`); the Model Health tab shows
  `generated_at_utc` and `experiment_id` instead.

## Planned (not built)
1. Opportunity Lab: predicted CLV distribution, P(positive CLV), net EV after costs, execution risk.
2. Model Intelligence: ATS/game/CLV model versions, baselines and disagreements.
3. Execution Lab: bet-now-vs-wait, fees, slippage, latency, fill and partial-fill scenarios.
4. CLV Analytics by season, book, timing and line bucket (today: by close tier and market).
5. Game Lab: expected margin/total and calibrated distributions, isolated from CLV labels.
6. Data Health: provider history, refresh latency, missingness, schema errors.
7. Research Lab: preregistration, experiments, ablations and negative results.
8. MCP & Integrations: server health, permissions and audit logs.

All values need source timestamp, model version and quality flag. Do not show an EV+ badge without
validated probability, executable quote and costs.
