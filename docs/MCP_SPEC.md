# MCP specification
Starter read-only tools: get_nfl_slate, get_market_snapshot, get_resource_registry, get_system_health.
Planned tools: get_spread_market, get_total_market, get_best_available_price, get_pinnacle_benchmark, get_market_movement, get_market_dispersion, predict_clv, get_clv_distribution, get_game_projection, get_fair_probability, calculate_expected_value, get_execution_analysis, rank_opportunities, get_model_performance, get_prediction_history, grade_predictions, get_data_quality, run_backtest, compare_models, get_existing_ats_prediction.

Planned resources: nfl://slate/current; nfl://markets/current; nfl://markets/{event_id}/spread; nfl://markets/{event_id}/total; nfl://models/registry; nfl://models/ats/latest; nfl://models/clv/latest; nfl://predictions/latest; nfl://analytics/clv; nfl://data/providers; nfl://data/quality; nfl://system/health.

Security: read-only by default, no wager placement, validate inputs, cap result sizes, avoid secrets, audit calls, require auth before remote exposure. UI and MCP share the same service methods.
