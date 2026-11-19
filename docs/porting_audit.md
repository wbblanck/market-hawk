# Porting audit: NFL_ATS and nfl-betting-model

Every Python module under `src/` in Will's two earlier repos, with what Market Hawk does with it. Source commits: `wbblanck/nfl-betting-model` @ `4bd6a96`, `wbblanck/NFL_ATS` @ `c51e678` (read-only, 2026-11-19). Generated from the full file list, so a module cannot be silently left out.

Statuses: **Ported** (logic reproduced, parity-tested against the original), **Adapted** (idea or settings reused, implementation differs on purpose), **Reviewed, already covered** (Market Hawk already has an equivalent), **Deferred (step 4)** (useful, not in step 3 scope), **Skipped** (not useful here, reason given).

| Status | Modules |
|---|---:|
| Adapted | 5 |
| Deferred (step 4) | 36 |
| Ported | 5 |
| Reviewed, already covered | 25 |
| Skipped | 99 |

Parity tests for every Ported row: `tests/test_parity_*.py`, goldens in `tests/fixtures/parity/` produced by `scripts/parity/` from the original code.

## wbblanck/nfl-betting-model

| Module | Status | Market Hawk | Reason |
|---|---|---|---|
| `nflbet/ats/batch5.py` | Ported | `src/nfl_market/distributions/scoring_models.py` | K1 key-number margin and S1 shared-touchdown joint score both PASSED vs NB on exact-margin log score (batch5_report). Early-season ratings/sim totals parts failed: not ported. |
| `nflbet/ats/elo.py` | Ported | `src/nfl_market/models/elo.py` | Elo + Poisson/NB baseline (rung 1c). Frozen constants from data/backtests/elo_constants.json; polars to pandas. |
| `nflbet/ats/kalman.py` | Ported | `src/nfl_market/models/bayesian.py` | Glickman-Stern Kalman margin rating = our Bayesian dynamic team-strength rung, frozen batch1 constants. Its own test (batch1 M8) did not beat Elo; kept as a benchmark. |
| `nflbet/ats/live.py` | Skipped | - | Live tracker / venue pricing for another project's picks; Market Hawk execution is step 4 and has its own rules. |
| `nflbet/ats/rating.py` | Deferred (step 4) | - | ATS power rating whose fair line predicts line movement and earned CLV (ats_clv C1/C2 PASS, C3 ATS FAIL). Belongs with step-4 CLV models, not the game-model rung. |
| `nflbet/ats/tdfg.py` | Ported | `src/nfl_market/distributions/scoring_models.py` | TD/FG score model (D2) priced covers better than NB (batch3 T2 PASS). Expected-starter QB part not ported (post-game QB ids are not pre-game knowledge). |
| `nflbet/ats/xgb.py` | Adapted | `src/nfl_market/models/xgb_model.py` | Fixed hyperparameters reused. Feature set adapted to our point-in-time loader; rolling EPA needs play-by-play (see open items). batch4: no better than Elo. |
| `nflbet/ats/your_venues.py` | Skipped | - | Live tracker / venue pricing for another project's picks; Market Hawk execution is step 4 and has its own rules. |
| `nflbet/config.py` | Skipped | - | Repo-specific paths. |
| `nflbet/dashboard.py` | Skipped | - | Tracker dashboard for the props project; Market Hawk has apps/dashboard. |
| `nflbet/engine/accounting.py` | Skipped | - | Player-prop engine (props only). Out of scope: Market Hawk game models predict game outcomes. |
| `nflbet/engine/baselines.py` | Skipped | - | Player-prop engine (props only). Out of scope: Market Hawk game models predict game outcomes. |
| `nflbet/engine/challenger/counts.py` | Skipped | - | Player-prop engine (props only). Out of scope: Market Hawk game models predict game outcomes. |
| `nflbet/engine/challenger/design.py` | Skipped | - | Player-prop engine (props only). Out of scope: Market Hawk game models predict game outcomes. |
| `nflbet/engine/challenger/joint.py` | Skipped | - | Player-prop engine (props only). Out of scope: Market Hawk game models predict game outcomes. |
| `nflbet/engine/challenger/model.py` | Skipped | - | Player-prop engine (props only). Out of scope: Market Hawk game models predict game outcomes. |
| `nflbet/engine/challenger/yardage.py` | Skipped | - | Player-prop engine (props only). Out of scope: Market Hawk game models predict game outcomes. |
| `nflbet/engine/contracts.py` | Deferred (step 4) | - | Exchange contract matching and executable pricing (Novig). Step 4 markets/execution. |
| `nflbet/engine/data/nflverse.py` | Adapted | `src/nfl_market/data/games.py` | Idea reused (season pulls with recorded file hash); Market Hawk loader is its own, pandas. |
| `nflbet/engine/data/player_week.py` | Skipped | - | Player-prop engine (props only). Out of scope: Market Hawk game models predict game outcomes. |
| `nflbet/engine/features/build.py` | Skipped | - | Player-prop engine (props only). Out of scope: Market Hawk game models predict game outcomes. |
| `nflbet/engine/features/facts.py` | Skipped | - | Player-prop engine (props only). Out of scope: Market Hawk game models predict game outcomes. |
| `nflbet/engine/forecast.py` | Skipped | - | Player-prop engine (props only). Out of scope: Market Hawk game models predict game outcomes. |
| `nflbet/engine/ledger.py` | Deferred (step 4) | - | Append-only hash-chained prediction record. Good pattern for step 4 immutable prediction history. |
| `nflbet/engine/novig_capture.py` | Deferred (step 4) | - | Exchange contract matching and executable pricing (Novig). Step 4 markets/execution. |
| `nflbet/engine/paper.py` | Deferred (step 4) | - | Exchange contract matching and executable pricing (Novig). Step 4 markets/execution. |
| `nflbet/engine/pricing.py` | Deferred (step 4) | - | Exchange contract matching and executable pricing (Novig). Step 4 markets/execution. |
| `nflbet/engine/quotes.py` | Deferred (step 4) | - | Exchange contract matching and executable pricing (Novig). Step 4 markets/execution. |
| `nflbet/engine/registry.py` | Skipped | - | Player-prop engine (props only). Out of scope: Market Hawk game models predict game outcomes. |
| `nflbet/engine/replication/efficiency.py` | Skipped | - | Player-prop engine (props only). Out of scope: Market Hawk game models predict game outcomes. |
| `nflbet/engine/replication/environment.py` | Skipped | - | Player-prop engine (props only). Out of scope: Market Hawk game models predict game outcomes. |
| `nflbet/engine/replication/events.py` | Skipped | - | Player-prop engine (props only). Out of scope: Market Hawk game models predict game outcomes. |
| `nflbet/engine/replication/model.py` | Skipped | - | Player-prop engine (props only). Out of scope: Market Hawk game models predict game outcomes. |
| `nflbet/engine/replication/opportunity.py` | Skipped | - | Player-prop engine (props only). Out of scope: Market Hawk game models predict game outcomes. |
| `nflbet/engine/scoring.py` | Reviewed, already covered | `src/nfl_market/distributions/calibration.py` | Log score / PIT for integer distributions; same definitions implemented in Market Hawk. |
| `nflbet/execution/clock.py` | Skipped | - | Run clock/record files for that repo's trackers. |
| `nflbet/execution/evalues.py` | Deferred (step 4) | - | Anytime-valid e-process for live tracking. Useful for step 4 live monitoring. |
| `nflbet/execution/records.py` | Skipped | - | Run clock/record files for that repo's trackers. |
| `nflbet/execution/wind.py` | Skipped | - | Wind-unders rule failed its 2000-2014 test. |
| `nflbet/gamesim/coverage.py` | Deferred (step 4) | - | Play-by-play simulator. Confirmed 2022-2025 margin MAE 9.983 vs Elo 9.964 (not distinguishable). Possible later challenger; no incremental value shown. |
| `nflbet/gamesim/engine.py` | Deferred (step 4) | - | Play-by-play simulator. Confirmed 2022-2025 margin MAE 9.983 vs Elo 9.964 (not distinguishable). Possible later challenger; no incremental value shown. |
| `nflbet/gamesim/league.py` | Deferred (step 4) | - | Play-by-play simulator. Confirmed 2022-2025 margin MAE 9.983 vs Elo 9.964 (not distinguishable). Possible later challenger; no incremental value shown. |
| `nflbet/gamesim/mix_live.py` | Skipped | - | Player/prop layers of the simulator (props only). |
| `nflbet/gamesim/personnel.py` | Deferred (step 4) | - | Play-by-play simulator. Confirmed 2022-2025 margin MAE 9.983 vs Elo 9.964 (not distinguishable). Possible later challenger; no incremental value shown. |
| `nflbet/gamesim/players.py` | Skipped | - | Player/prop layers of the simulator (props only). |
| `nflbet/gamesim/props.py` | Skipped | - | Player/prop layers of the simulator (props only). |
| `nflbet/gamesim/run.py` | Deferred (step 4) | - | Play-by-play simulator. Confirmed 2022-2025 margin MAE 9.983 vs Elo 9.964 (not distinguishable). Possible later challenger; no incremental value shown. |
| `nflbet/gamesim/teams.py` | Deferred (step 4) | - | Play-by-play simulator. Confirmed 2022-2025 margin MAE 9.983 vs Elo 9.964 (not distinguishable). Possible later challenger; no incremental value shown. |
| `nflbet/props/bestprice.py` | Skipped | - | Player-prop research (props only). |
| `nflbet/props/clv.py` | Reviewed, already covered | `src/nfl_market/markets/clv.py` | Props CLV record; Market Hawk has its own game-line CLV. |
| `nflbet/props/confirm.py` | Skipped | - | Player-prop research (props only). |
| `nflbet/props/core.py` | Skipped | - | Player-prop research (props only). |
| `nflbet/props/explain.py` | Skipped | - | Player-prop research (props only). |
| `nflbet/props/injury_redist.py` | Skipped | - | Player-prop research (props only). |
| `nflbet/props/ladder.py` | Skipped | - | Player-prop research (props only). |
| `nflbet/props/live.py` | Skipped | - | Player-prop research (props only). |
| `nflbet/props/market_ev.py` | Skipped | - | Player-prop research (props only). |
| `nflbet/props/player_proj.py` | Skipped | - | Player-prop research (props only). |
| `nflbet/props/pressure.py` | Skipped | - | Player-prop research (props only). |
| `nflbet/props/projection.py` | Skipped | - | Player-prop research (props only). |
| `nflbet/props/recap.py` | Skipped | - | Player-prop research (props only). |
| `nflbet/props/screener.py` | Skipped | - | Player-prop research (props only). |
| `nflbet/props/selective.py` | Skipped | - | Player-prop research (props only). |
| `nflbet/props/signals.py` | Skipped | - | Player-prop research (props only). |
| `nflbet/props/sim.py` | Skipped | - | Player-prop research (props only). |
| `nflbet/props/your_venues.py` | Skipped | - | Player-prop research (props only). |
| `nflbet/quant/fees.py` | Deferred (step 4) | - | Exchange taker fees: needed for step 4 Kalshi/Polymarket execution. |
| `nflbet/quant/odds.py` | Reviewed, already covered | `src/nfl_market/markets/odds.py` | Odds/devig arithmetic already in Market Hawk. |
| `nflbet/sources/asb.py` | Skipped | - | AusSportsBetting data is personal-use only; Market Hawk is public. |
| `nflbet/sources/depth.py` | Deferred (step 4) | - | Pre-game injury/depth status: needed before any QB feature is admitted. |
| `nflbet/sources/espn.py` | Skipped | - | Prop feeds / generic HTTP client (props only or duplicated by httpx). |
| `nflbet/sources/gc_props.py` | Skipped | - | Prop feeds / generic HTTP client (props only or duplicated by httpx). |
| `nflbet/sources/http.py` | Skipped | - | Prop feeds / generic HTTP client (props only or duplicated by httpx). |
| `nflbet/sources/ids.py` | Reviewed, already covered | `src/nfl_market/integrations/sports_leader/mapping.py` | Team keys; Market Hawk has its own mapping plus franchise map in models. |
| `nflbet/sources/injuries.py` | Deferred (step 4) | - | Pre-game injury/depth status: needed before any QB feature is admitted. |
| `nflbet/sources/openmarkets.py` | Deferred (step 4) | - | Prediction-market prices; step 4 (Sports Leader already covers Kalshi/Polymarket). |
| `nflbet/sources/parlay.py` | Adapted | `src/nfl_market/integrations/sharp/odds_api.py` | ParlayAPI reader; Market Hawk adapter reads the same key for game lines. |
| `nflbet/sources/sbro.py` | Deferred (step 4) | - | SBRO 2007-2021 open/close, book unknown, no timestamps: at most a labelled benchmark (closing_line.md L6). |
| `nflbet/sources/teamstats.py` | Deferred (step 4) | - | Team pass-rush/coverage stats; candidate features later. |
| `nflbet/sources/venues.py` | Deferred (step 4) | - | Prediction-market prices; step 4 (Sports Leader already covers Kalshi/Polymarket). |
| `nflbet/sources/weather.py` | Deferred (step 4) | - | ERA5 wind is reanalysis (not a forecast at T): REVIEW REQUIRED before use. |

## wbblanck/NFL_ATS

| Module | Status | Market Hawk | Reason |
|---|---|---|---|
| `engine/__main__.py` | Skipped | - | Pool pick pipeline, CLI, config, dashboard and alerts. |
| `engine/backtest/adaptive_eval.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/alt_markets.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/calibration.py` | Reviewed, already covered | `src/nfl_market/distributions/calibration.py` | Platt/isotonic. Finding reused: no calibrator improved raw probabilities (phase5). |
| `engine/backtest/cbs_lines.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/common.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/early2026.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/edge2.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/efficiency_residual.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/gap2022.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/holdout2025.py` | Skipped | - | 2025 holdout for the pool engine. Note: 2025 was already scored there. |
| `engine/backtest/keynumber.py` | Adapted | `src/nfl_market/distributions/calibration.py` | Key-number table layout matched so results are comparable. Its pick test FAILED (keynumber_report). |
| `engine/backtest/mining.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/pace_penalty.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/patterns.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/phase2.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/phase3.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/phase4.py` | Reviewed, already covered | `src/nfl_market/models/folds.py, distributions/evaluate.py` | Walk-forward design; Market Hawk has its own fold builder with V2/V3/holdout asserts. |
| `engine/backtest/phase5.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/phase6.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/players_eval.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/players_v3_eval.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/projection_eval.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/projection_v41_eval.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/projection_v4_eval.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/projection_v51_eval.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/projection_v5_eval.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/projection_v6_eval.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/search.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/sharp_line.py` | Reviewed, already covered | `src/nfl_market/markets/closing.py` | Sharp-book estimate check; precedent for the tiered close. |
| `engine/backtest/straightup_eval.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/tiebreak.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/total.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/turnover_luck.py` | Skipped | - | Pre-registered tests of pool-line pick strategies; results kept as prior evidence, code not needed. |
| `engine/backtest/walk_forward.py` | Reviewed, already covered | `src/nfl_market/models/folds.py, distributions/evaluate.py` | Walk-forward design; Market Hawk has its own fold builder with V2/V3/holdout asserts. |
| `engine/config.py` | Skipped | - | Pool pick pipeline, CLI, config, dashboard and alerts. |
| `engine/data/cbs_lines.py` | Skipped | - | CBS pool line capture (pool-specific). |
| `engine/data/league_lines.py` | Skipped | - | CBS pool line capture (pool-specific). |
| `engine/data/market.py` | Deferred (step 4) | - | Pinnacle backup sources (OddsLink, OddsPapi). Listed in docs/data_sources/sharp_feeds.md; adapters next. |
| `engine/data/nflverse.py` | Reviewed, already covered | `src/nfl_market/data/games.py` | nflreadpy ingestion; replaced by Market Hawk loader. |
| `engine/data/oddslink.py` | Deferred (step 4) | - | Pinnacle backup sources (OddsLink, OddsPapi). Listed in docs/data_sources/sharp_feeds.md; adapters next. |
| `engine/data/oddspapi.py` | Deferred (step 4) | - | Pinnacle backup sources (OddsLink, OddsPapi). Listed in docs/data_sources/sharp_feeds.md; adapters next. |
| `engine/data/sbro.py` | Reviewed, already covered | - | Same SBRO parser as nfl-betting-model. |
| `engine/data/store.py` | Reviewed, already covered | `src/nfl_market/data/store.py` | Append-only archive; Market Hawk has DuckDB store. |
| `engine/features/availability.py` | Deferred (step 4) | - | Player-availability features; need pre-game injury data. |
| `engine/features/efficiency.py` | Deferred (step 4) | - | Opponent-adjusted EPA ratings from play-by-play. Strongest feature candidate for step 4; v6 failed vs v1/v3. |
| `engine/features/efficiency_v6.py` | Deferred (step 4) | - | Opponent-adjusted EPA ratings from play-by-play. Strongest feature candidate for step 4; v6 failed vs v1/v3. |
| `engine/features/epa_components.py` | Deferred (step 4) | - | Player-availability features; need pre-game injury data. |
| `engine/features/players_v3.py` | Deferred (step 4) | - | Player-availability features; need pre-game injury data. |
| `engine/features/qb.py` | Deferred (step 4) | - | QB change: needs pre-game starter knowledge. |
| `engine/features/rest.py` | Adapted | `src/nfl_market/data/games.py` | Schedule rest days. |
| `engine/gamesim/batch5.py` | Reviewed, already covered | (see nflbet/gamesim, nflbet/ats) | Copy of the nfl-betting-model module; the original was used as the port source. |
| `engine/gamesim/coverage.py` | Reviewed, already covered | (see nflbet/gamesim, nflbet/ats) | Copy of the nfl-betting-model module; the original was used as the port source. |
| `engine/gamesim/elo.py` | Reviewed, already covered | (see nflbet/gamesim, nflbet/ats) | Copy of the nfl-betting-model module; the original was used as the port source. |
| `engine/gamesim/engine.py` | Reviewed, already covered | (see nflbet/gamesim, nflbet/ats) | Copy of the nfl-betting-model module; the original was used as the port source. |
| `engine/gamesim/kalman.py` | Reviewed, already covered | (see nflbet/gamesim, nflbet/ats) | Copy of the nfl-betting-model module; the original was used as the port source. |
| `engine/gamesim/league.py` | Reviewed, already covered | (see nflbet/gamesim, nflbet/ats) | Copy of the nfl-betting-model module; the original was used as the port source. |
| `engine/gamesim/personnel.py` | Reviewed, already covered | (see nflbet/gamesim, nflbet/ats) | Copy of the nfl-betting-model module; the original was used as the port source. |
| `engine/gamesim/run.py` | Reviewed, already covered | (see nflbet/gamesim, nflbet/ats) | Copy of the nfl-betting-model module; the original was used as the port source. |
| `engine/gamesim/tdfg.py` | Reviewed, already covered | (see nflbet/gamesim, nflbet/ats) | Copy of the nfl-betting-model module; the original was used as the port source. |
| `engine/gamesim/teams.py` | Reviewed, already covered | (see nflbet/gamesim, nflbet/ats) | Copy of the nfl-betting-model module; the original was used as the port source. |
| `engine/gamesim/xgb.py` | Reviewed, already covered | (see nflbet/gamesim, nflbet/ats) | Copy of the nfl-betting-model module; the original was used as the port source. |
| `engine/models/adaptive.py` | Skipped | - | Pool pick strategies (drift, adaptive, players_v3_live). |
| `engine/models/drift.py` | Skipped | - | Pool pick strategies (drift, adaptive, players_v3_live). |
| `engine/models/margin_dist.py` | Ported | `src/nfl_market/distributions/margin.py` | normal, shifted, kernel, kernel_sym, keynum estimators and bandwidth chooser. |
| `engine/models/null.py` | Reviewed, already covered | `src/nfl_market/models/constant.py` | Null baselines. |
| `engine/models/players_v3_live.py` | Skipped | - | Pool pick strategies (drift, adaptive, players_v3_live). |
| `engine/models/projection.py` | Deferred (step 4) | - | Score-projection models (Poisson v1 tracked; v4.1 failed validation; v5/v5.1 drive sims). Joint-score challengers for later. |
| `engine/models/projection_v4.py` | Deferred (step 4) | - | Score-projection models (Poisson v1 tracked; v4.1 failed validation; v5/v5.1 drive sims). Joint-score challengers for later. |
| `engine/models/projection_v41.py` | Deferred (step 4) | - | Score-projection models (Poisson v1 tracked; v4.1 failed validation; v5/v5.1 drive sims). Joint-score challengers for later. |
| `engine/models/projection_v5.py` | Deferred (step 4) | - | Score-projection models (Poisson v1 tracked; v4.1 failed validation; v5/v5.1 drive sims). Joint-score challengers for later. |
| `engine/models/projection_v51.py` | Deferred (step 4) | - | Score-projection models (Poisson v1 tracked; v4.1 failed validation; v5/v5.1 drive sims). Joint-score challengers for later. |
| `engine/picks/alerts.py` | Skipped | - | Pool pick pipeline, CLI, config, dashboard and alerts. |
| `engine/picks/curve.py` | Skipped | - | Pool pick pipeline, CLI, config, dashboard and alerts. |
| `engine/picks/dashboard.py` | Skipped | - | Pool pick pipeline, CLI, config, dashboard and alerts. |
| `engine/picks/explain.py` | Skipped | - | Pool pick pipeline, CLI, config, dashboard and alerts. |
| `engine/picks/final.py` | Skipped | - | Pool pick pipeline, CLI, config, dashboard and alerts. |
| `engine/picks/game_detail.py` | Skipped | - | Pool pick pipeline, CLI, config, dashboard and alerts. |
| `engine/picks/grading.py` | Reviewed, already covered | `src/nfl_market/markets/clv.py` | Spread sign conventions match (home spread negative = home favoured, margin = home - away). |
| `engine/picks/hybrid.py` | Skipped | - | Pool pick pipeline, CLI, config, dashboard and alerts. |
| `engine/picks/hypothetical.py` | Skipped | - | Pool pick pipeline, CLI, config, dashboard and alerts. |
| `engine/picks/live.py` | Skipped | - | Pool pick pipeline, CLI, config, dashboard and alerts. |
| `engine/picks/loss_review.py` | Skipped | - | Pool pick pipeline, CLI, config, dashboard and alerts. |
| `engine/picks/preview.py` | Skipped | - | Pool pick pipeline, CLI, config, dashboard and alerts. |
| `engine/picks/provisional.py` | Skipped | - | Pool pick pipeline, CLI, config, dashboard and alerts. |
| `engine/picks/season.py` | Skipped | - | Pool pick pipeline, CLI, config, dashboard and alerts. |
| `engine/picks/slate_replay.py` | Skipped | - | Pool pick pipeline, CLI, config, dashboard and alerts. |
| `engine/picks/tiebreak_total.py` | Skipped | - | Pool pick pipeline, CLI, config, dashboard and alerts. |
| `engine/teams.py` | Reviewed, already covered | - | Team code map. |
