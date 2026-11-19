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
