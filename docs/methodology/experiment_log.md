# Experiment log

One immutable record per executed experiment, including failures. The log is **append-only**: entries are never edited or deleted after results exist; corrections, deviations and invalidations are appended as new dated lines under the entry.

Adapted from sports-analytic-skills `skills/experiment-log/SKILL.md` (create-before-execution, schema, decision rules, invalidation handling) and `skills/experiment-log/references/log_schema.md`.

## Rules

| # | Rule |
|---|---|
| E1 | Create the entry **before** fitting: hypothesis, charter, primary metric, baselines and success rule are frozen first. |
| E2 | Log every run, including failed, abandoned and losing ones. Winner-only logging is a violation. |
| E3 | A changed target, T, population, metric, fold table or hypothesis means a **new experiment id**. |
| E4 | Post-hoc metrics are labelled `post-hoc` and cannot drive the decision. |
| E5 | When leakage or a validation violation is found later, append an `invalidated` line to every affected entry; the repaired run gets a new id linking back. |
| E6 | Comparisons are valid only when target, T, data snapshot, rows, folds and metrics match. |

Storage location and a helper to create entries are **to be enforced in step 3** (proposed: one Markdown or YAML file per entry under a versioned `experiments/` directory, plus an index). This document defines the format only; entries never go in this file.

## Required fields

```text
experiment_id:            YYYYMMDD-<slug>-<nn>
created_at_utc / operator
hypothesis and expected direction (falsifiable)
target / grain / eligible population / prediction timestamp rule T
data: source tables, snapshot id or query, data window, checksum; holdout manifest id (excluded)
features: registry version and list of admitted features
validation charter ref: fold table id, embargo, inner tuning scheme
primary metric (+ direction), secondary diagnostics, success rule
baselines (ladder rungs, predictive_modeling.md)
model family / config / hyperparameter search space / seeds
code commit / environment lock / exact command
status: planned | running | completed | failed | invalidated
per-fold results table (below) and pooled result with game-clustered CI
calibration verdict (calibration_check.md)
leakage audit verdict (leakage_audit.md)
holdout access: none | yes (log line ref)
deviations from plan
decision: keep | discard | follow-up | invalid; reason; next action
artifacts: prediction files (OOF), fitted models, checksums
```

Per-fold table:

```text
fold | train_period | test_period | games | rows | baseline_primary | candidate_primary | diff | secondary | calibration | warnings
```

## Decision rules

| Decision | When |
|---|---|
| `keep` | predeclared success rule met on the locked evaluation, leakage audit `CLEAN` |
| `discard` | fails the baseline or success rule, or adds complexity without gain |
| `follow-up` | plausible signal with one named uncertainty and one concrete next test |
| `invalid` | leakage, execution failure, charter violation or incomparable rows |

## Template entry

```text
experiment_id:     20261118-elo-margin-baseline-01
created_at_utc:    2026-11-18T00:00:00Z
operator:          <name>
hypothesis:        As-of Elo with MOV and season regression lowers margin MAE vs the constant baseline on 2016-2024 weekly walk-forward. Direction: lower MAE.
target/grain/T:    home margin; one row per game; T = kickoff_utc - 24h
population:        regular season + postseason, cancelled games excluded
data:              schedule/results snapshot <id> sha256 <...>; window 2012-2024; holdout manifest <id> excluded
features:          registry v<..>: elo_diff, is_neutral
charter:           fold table <id>: season outer folds, weekly refit, 3 min train seasons, embargo = results + stat-correction allowance
primary metric:    MAE (lower better); secondary RMSE, bias by season, spread bucket
success rule:      MAE lower than rung 0 on >= 6/9 test seasons and paired game-bootstrap 95% CI of diff < 0
baselines:         rung 0 constant mean home margin; rung 1a market at T (report only)
config:            K=20, HFA=<fit in-fold>, MOV multiplier, regression 1/3 to mean; seed 0
code/env/command:  <commit> / <lockfile hash> / <command>
status:            planned
--- appended after run ---
results:           <per-fold table>; pooled MAE <..> [CI]
calibration:       n/a (point model) | <verdict>
leakage audit:     <verdict>, <auditor>, <date>
holdout access:    none
deviations:        none
decision:          <keep|discard|follow-up|invalid> because <...>; next: <...>
artifacts:         <OOF prediction path> sha256 <...>
```

## Entries

Until the per-entry `experiments/` directory exists, step-3 entries are appended here (append-only).

### 20261119-step3-game-models-01

```text
experiment_id:     20261119-step3-game-models-01
created_at_utc:    2026-11-19T00:00:00Z (charter written with the code, before the first full 2015-2024 run)
operator:          Claude (workstream 2), for the step-3 lead
hypothesis:        On season walk-forward with weekly refits, each rung-2/3 model (Kalman margin rating, offense/defense
                   Bayesian Kalman, Ridge, XGBoost) has lower margin MAE than the ported Elo baseline (rung 1c), and all
                   beat the constant baseline (rung 0). Direction: lower MAE / log loss.
target/grain/T:    home margin (primary), total, P(home win); one row per game; T = kickoff_utc - 24h
population:        all REG + postseason games of test seasons 2015-2024 (2,743 games); no cancelled games in source
label rule:        label_available_at_utc = next Wednesday 12:00 US/Eastern after kickoff; ties count in MAE, dropped
                   from log loss/Brier
data:              nflverse games.csv cache data/raw/nflverse_games.csv (sha256 in docs/results/step3_walkforward.md),
                   seasons 1999-2024; holdout manifest nfl_games_2025_v1 excluded by the loader; 2026 dropped
features:          registry v1 in src/nfl_market/models/features.py (status proposed): as-of Elo, Elo gap/fair margin,
                   ported Kalman fair margin/variance, last-8 and season-to-date points for/against, previous-season
                   points, games played, rest (REVIEW: final schedule), schedule flags, week, league scoring level and
                   last-256 total/margin. Excluded: nflverse lines/odds (unknown capture time), qb ids (post-game),
                   weather (observed).
charter:           fold table season_wf_weekly_v1 (src/nfl_market/models/folds.py): outer test seasons 2015-2024,
                   >= 3 training seasons (history from 1999), weekly rolling origin, refit before every week on games
                   with label < week's first T; embargo none beyond the label rule; V2/V3/holdout asserted per origin
inner tuning:      Ridge alpha in {1,10,100,1e3,1e4} per target on inner split (seasons < V -> V, V = last training
                   season with its Super Bowl), once per outer fold; same split gives Ridge/XGB residual SDs.
                   Elo, Kalman, XGB settings fixed a priori = frozen constants ported from wbblanck/nfl-betting-model
                   (elo_constants.json, batch1_constants.json, xgb.py PARAMS, 100 trees per batch4_model.json).
                   bayes_od_v1 constants fixed a priori (docstring).
primary metric:    margin MAE (lower better); secondary RMSE, bias, total MAE, home-win log loss and Brier, by fold,
                   pooled, clean seasons, REG/postseason/2020 slices
success rule:      candidate beats Elo if the paired game-clustered bootstrap 95% CI (B=2000) of the MAE difference is
                   below 0 and it is lower on a majority of the 10 folds
baselines:         rung 0 constant_v1; rung 1c elo_v1; rung 1b nflverse line reported separately as a labelled
                   benchmark ("nflverse line, capture time unknown"), not a model
models/seeds:      constant_v1, elo_v1, kalman_gs_v1, bayes_od_v1, ridge_v1, xgb_v1 (seed 0, nthread 1)
code/env/command:  uncommitted step-3 working tree on branch claude/project-thread-4716zq; python 3.13, pandas 3.0.5,
                   xgboost 3.4.1; `python scripts/run_walkforward.py`
status:            completed
--- appended after run (2026-11-19) ---
results:           docs/results/step3_walkforward.md (per-fold, pooled, clean-season and slice tables, bootstrap CIs).
                   Pooled margin MAE: constant 10.963, elo 10.176, kalman_gs 10.152, bayes_od 10.161, ridge 10.144,
                   xgb 10.190; nflverse-line benchmark 9.821. Margin MAE minus Elo: kalman -0.025 [-0.089, +0.036],
                   bayes_od -0.015 [-0.096, +0.065], ridge -0.033 [-0.095, +0.029], xgb +0.013 [-0.059, +0.089].
                   Total MAE minus Elo: ridge -0.190 [-0.285, -0.096] (9/10 folds), xgb -0.209 [-0.312, -0.104] (10/10).
calibration:       not assessed here (distributions workstream consumes the OOF file)
leakage audit:     REVIEW REQUIRED (author checks only: fold assertions, as-of feature construction with
                   assert_point_in_time, future-perturbation tests in tests/test_models_step3.py; reviewer pending)
holdout access:    none (no read_holdout call; loader excludes 2025)
deviations:        V5 caveat declared up front: ported constants were chosen by the source repo on 2016-2020 (Elo,
                   Kalman) / 2016-2021 (XGB tree count), so those folds are not nested for the ported models;
                   clean-season slice (2015, 2022-2024) reported alongside. QB and EPA features unavailable.
decision:          follow-up. Margin: no candidate beats Elo (all CIs cover 0), keep Elo as the margin baseline (P6).
                   Total: Ridge and XGBoost beat Elo's NB-implied total on pooled and clean seasons. Next: pre-game QB
                   and EPA features; in-fold refit of ported constants to remove the V5 caveat; leakage review.
artifacts:         data/processed/step3_oof_predictions.parquet (gitignored), 16,458 rows, sha256 cb2d2c5de84b64d653ea09401fd816696aa9208ccbaa7db245d0364ee88246db
```

### 20261119-step4-stacking-01

```text
experiment_id:     20261119-step4-stacking-01
created_at_utc:    2026-11-19T17:45:00Z (design and primary objective fixed in code before any test-season score)
operator:          Claude (step-4 stacking agent), for the step-4 lead
hypothesis:        A simplex-weighted stack (stack_v1) of elo_v1, kalman_gs_v1, ridge_v1, xgb_v1 (margin) and elo_v1,
                   ridge_v1, xgb_v1 (total) has lower margin/total MAE and home-win log loss than elo_v1 and than the
                   best single component on the same games. Direction: lower is better. Expectation: small or no gain.
target/grain/T:    home margin, total, P(home win); one row per game; T = kickoff_utc - 24h (inherited from step 3)
population:        test seasons 2018-2024 (1,942 games; 2015-2017 are weight history only: >= 3 training seasons)
data:              data/processed/step3_oof_predictions.parquet (experiment 20261119-step3-game-models-01; 16,458 rows)
                   + nflverse cache for outcomes/label times; holdout nfl_games_2025_v1 refused; 2026 not read
charter:           season-level walk-forward: weights for S fit on OOF rows of seasons < S with prediction time and
                   label time < S's first T (distributions.evaluate.training_rows); no weekly refit inside S
method:            non-negative weights summing to 1, no intercept; primary objective squared error (exact active-set
                   QP); variants reported separately: MAE objective (stack_mae_v1), bayes_od_v1 as fifth candidate
                   (stack_b5_v1); baseline stack equal weights (stack_eq_v1). Win prob: K1 margin distribution refit per
                   S on prior rows' stack margin, P(M>0)/(1-P(M=0))
primary metric:    margin MAE; secondary total MAE, home-win log loss/Brier, ECE (10 bins), K1 PIT/coverage, cover
                   calibration at the nflverse line (capture time unknown; context only)
success rule:      "beats" only if paired game-clustered bootstrap 95% CI (B=2000) excludes 0 below AND majority of folds
code/env/command:  uncommitted step-4 working tree on branch claude/project-thread-5axwmr; `python scripts/run_stacking.py`
status:            completed
--- appended after run (2026-11-19) ---
results:           docs/results/step4_stacking.md, docs/results/step4_model_health.json. Pooled 2018-2024 margin MAE:
                   stack_v1 10.211, stack_eq_v1 10.185, elo 10.221, kalman_gs 10.192, ridge 10.198, xgb 10.265.
                   stack_v1 minus elo -0.009 [-0.074, +0.056] (3/7 folds); minus best single (kalman_gs) +0.020
                   [-0.020, +0.057] (4/7). Total MAE stack_v1 10.743: minus elo -0.250 [-0.354, -0.136] (7/7, beats);
                   minus best single (xgb) -0.015 [-0.055, +0.024] (4/7). Home-win log loss 0.6371 vs elo 0.6369,
                   +0.000 [-0.004, +0.005]. Equal weights margin minus elo -0.035 [-0.086, +0.018] (3/7).
calibration:       P(home win) ECE 0.030 (elo 0.016); K1 PIT mean 0.492, coverage 50/80/95 = 0.535/0.808/0.951;
                   cover log loss at the nflverse line 0.713 (all components 0.711-0.716, worse than 0.693).
                   Verdict usable-with-caveats (2019 PIT mean 0.450; inherits step-3 distribution caveats).
leakage audit:     REVIEW REQUIRED (author checks: future-perturbation and late-label tests in tests/test_stacking.py;
                   validate_predictions train_end_utc < prediction_time_utc on every row)
holdout access:    none
deviations:        none from the charter; V5 caveat inherited for 2018-2021 folds; clean seasons = 2022-2024 (2015 is
                   not scored)
decision:          follow-up. No stack beats elo_v1 or the best single model on margin or win probability; the total
                   stack beats Elo's NB total (as ridge/xgb already did in step 3) but not xgb_v1. stack_v1 stays
                   experimental; it is a convenience blend for pricing, not a promoted improvement. Forward weights
                   (2015-2024): margin elo 0.081, kalman_gs 0.432, ridge 0.201, xgb 0.286; total ridge 0.540, xgb 0.460
                   (config/models/stack_v1_weights.json).
artifacts:         data/processed/step4_stack_predictions.parquet (gitignored), 1,942 rows, sha256
                   0164302238b75b37e8e25e3d942d1007e1e395187c56be8aebd216361b5ecb4a
```

### 20261125-v2-epa-qb-01

```text
experiment_id:     20261125-v2-epa-qb-01
created_at_utc:    2026-11-25T21:45:00Z
operator:          Claude (requested by Will)
hypothesis:        nflverse EPA team form and pre-game QB status lower margin/total MAE and win log loss of the
                   feature models (ridge, xgb) on the step-3 fold table. Direction: lower.
target/grain/T:    home margin, total, P(home win); one row per game; T = kickoff_utc - 24h
data:              nflverse games.csv; nflverse play-by-play 1999-2024 and injuries 2009-2024 reduced to panels
                   (data/raw/nflverse_ext/manifest.json sha256 per file); 2025 never downloaded
features:          registry v2 (feature_rules.md): *_epa_l8, *_epa_ewm, *_pass_epa_ewm, qb_out, qb_epa (both sides)
charter:           season_wf_weekly_v1, test seasons 2015-2024; constants fixed before the first scored run
success rule:      paired game-clustered bootstrap 95% CI (B=2000) of v2 minus v1 below 0
code/env/command:  scripts/experiments/recommendations_backtest.py (exploratory run), then scripts/run_walkforward.py and
                   scripts/run_stacking.py on the production classes (ridge_v2, xgb_v2, stack_v2)
status:            completed
--- appended after run (2026-11-26) ---
results:           docs/results/experiments/recommendations_backtest.md; docs/results/step3_walkforward.md. Pooled
                   2015-2024: ridge_v2 margin MAE 10.110 (ridge_v1 10.144), total 10.678 (10.792), log loss 0.631
                   (0.634); xgb_v2 margin 10.142 (xgb_v1 10.190), total 10.661 (10.773). Exploratory run CIs: ridge
                   EPA+QB total -0.114 [-0.166, -0.058], log loss -0.003 [-0.006, -0.001], margin -0.035 [-0.081,
                   +0.011]; xgb EPA+QB margin -0.047 [-0.080, -0.013], total -0.110 [-0.175, -0.048]. nflverse line
                   9.821 / 10.444. stack_v2 (docs/results/step4_stacking.md) margin 10.166 vs best single ridge_v2
                   10.156; total 10.635.
beyond the line:   (pred - line) does not predict (result - line) for any model, spreads or totals; ATS/OU hit
                   rates 47.6%-52.1%.
leakage audit:     REVIEW REQUIRED (author checks in tests/test_external_features.py; injury rows without
                   date_modified treated as available)
holdout access:    none
decision:          keep as production feature models (Will, 2026-11-25): ridge_v2 is the default pricing model and
                   stack_v2 replaces stack_v1. Not an edge over the close. LightGBM and fitted Kalman discarded.
```

```
experiment_id:     20261126-ebm-ngboost-01
created_at_utc:    2026-11-26T04:10:00Z
operator:          Claude (requested by Will)
hypothesis:        (1) an EBM on the xgb_v2 features is as accurate as ridge_v2/xgb_v2 and shows whether XGBoost's
                   learned relationships are sensible; (2) NGBoost per-game sigma improves margin/total pmf log score
                   over production (k1 margin, t total on ridge_v2 means). Direction: lower MAE, better log score.
target/grain/T:    home margin, total; one row per game; T = kickoff_utc - 24h
data:              nflverse games.csv; nflverse_ext panels (as 20261125-v2-epa-qb-01); step3 OOF predictions
charter:           test seasons 2015-2024, season-level refits (xgb season-refit control); distributions scored
                   2018-2024; settings fixed in the script docstring before the first scored run
success rule:      (season, week)-clustered bootstrap 95% CI (B=2000) of the difference excludes 0 in the good direction
code/env/command:  python scripts/experiments/ebm_ngboost_backtest.py (interpret 0.7.8, ngboost 0.6.1)
status:            completed
results:           docs/results/experiments/ebm_ngboost_backtest.md. EBM MAE margin 10.133 / total 10.666 vs ridge_v2
                   10.110 / 10.678 (CIs include 0). NGBoost sigma-only log score -0.171 margin, -0.066 total vs
                   production; NGBoost full -0.127 / -0.013 (all CIs below 0); corr(sigma, |resid|) about 0.
                   Diagnostic: signs agree across EBM/XGB/Ridge; xgb_v2 home-QB-out effect -1.0 pts vs EBM -3.0,
                   Ridge -4.0.
holdout access:    none
decision:          not adopted (EBM kept as a diagnostic only); production distributions unchanged. xgb QB-out
                   under-reaction recorded as a follow-up candidate.
```
