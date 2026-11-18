# Validation design

Applies to every model from step 3 on: game models (margin, total, team points, win/cover probability) and CLV models. A result that does not follow this design is exploratory and may not be cited as a predictive claim.

Adapted from sports-analytic-skills `skills/validation-design/SKILL.md` (charter, fold diagnostics, nested tuning, anti-patterns) and `skills/predictive-modeling/references/walk_forward.md`.

## 1. Rules

| # | Rule |
|---|---|
| V1 | Primary predictive claims use chronological expanding-window walk-forward only. **No random train/test splits and no random K-fold** on games, snapshots or team-game rows. Random splits may appear only in a debugging notebook and never in a results table. |
| V2 | For every fold: `max(train label_available_at_utc) < min(test prediction_time_utc)` after the declared embargo. Checked in code, not by eye. |
| V3 | Rows from one `game_id` (both sides, all books, all snapshots, all markets) are never split across train and test of the same fold. |
| V4 | Every model in a comparison uses the **same fold table** (same fold ids, same train/test `game_id` sets, same eligible rows). A model that cannot score a row makes the row ineligible for all models, or the gap is reported as a coverage difference, never silently dropped. |
| V5 | Hyperparameters, early stopping, feature selection, calibration maps and stacking weights are chosen inside the outer training window (nested), never on the outer test fold. |
| V6 | The final holdout is locked before step 3 modeling begins and is scored once per frozen candidate (Section 5). |
| V7 | Uncertainty is clustered by game (Section 6). Row counts are not sample sizes. |
| V8 | Every fold is reported, including bad ones. Dropping a fold after seeing it invalidates the experiment. |

## 2. The validation charter (write before fitting)

Each experiment links to a charter stored with its experiment-log entry (`experiment_log.md`). Required fields:

```text
target / row grain / eligible population
prediction timestamp rule T (e.g. kickoff_utc - 24h; or the snapshot's captured_at_utc)
label definition and label_available_at_utc rule
fold table id (Section 3) and min training history
embargo / purge rule (Section 4)
primary metric (+ direction) and secondary diagnostics
baselines (predictive_modeling.md ladder)
inner tuning scheme
final holdout manifest id (Section 5)
success rule (metric gap, fold consistency, calibration limits)
```

If the charter changes after results are seen, the old result is relabelled exploratory and a new experiment id is opened.

## 3. NFL fold layout

Ordering key is the prediction timestamp, never file order or `game_id`.

**Season outer folds (default for game models).**

```text
fold 1: train seasons [s0 .. S-1]           test season S
fold 2: train seasons [s0 .. S]             test season S+1
...
```

- Minimum training history: 3 complete seasons before the first test season (preregister a different number in the charter if justified).
- Inside a test season, the default is **weekly rolling origin**: refit (or update ratings) before week `w` using all games whose labels were available before week `w`'s first prediction time; score week `w`; advance. A season-frozen variant (fit once before week 1) is allowed but must be declared.
- Week is the NFL week from the schedule (`season`, `week` in `quote_snapshots`; see `src/nfl_market/data/store.py`). Postseason weeks are separate folds and reported as their own slice. Thursday, Saturday, international and Monday games belong to their schedule week.
- Report slices that change the data-generating process: 2020 (limited crowds), 17-game seasons (2021+), rule-change seasons, postseason.

**CLV-model folds.** CLV rows are our own snapshots, so history begins at the first capture date. Fold by NFL week of `kickoff_utc` (all snapshots of a game go to that game's week). Minimum training history is stated in weeks; until enough weeks exist, CLV results are labelled preliminary.

**Existing code.** `src/nfl_market/models/walkforward.py::expanding_windows(frame, time_col, min_train_rows, test_rows)` sorts by `time_col` and yields expanding windows sized by row count. It does **not** respect week or game boundaries, has no embargo, and knows nothing about the holdout. It may be used only for smoke tests. A season/week fold builder that writes a fold table (fold_id, train game_ids, test game_ids, train_end_utc, test_start_utc, embargo) and asserts V2/V3 is **to be enforced in step 3**.

## 4. Purge and embargo

Labels resolve after the prediction time. A training row is usable only if its label was known before the test row was predicted.

```text
label_available_at_utc:
  game outcome labels   = game final time + stat-correction allowance (declare it; default: next Wednesday 12:00 ET)
  CLV labels            = close snapshot time for that contract (closing_line.md)
purge:   drop train rows with label_available_at_utc >= min(test prediction_time_utc)
embargo: additionally drop train rows whose label_available_at_utc falls within E of the test start
```

- Game models with weekly refits: last week's games are final by the next week's prediction time, so purge rarely bites; it matters for Thursday-to-Sunday overlap and for features built from stat-corrected data.
- CLV models: a snapshot at Tuesday has a label that resolves Sunday. Snapshots from the same week overlap heavily, so purge **every** training snapshot whose close falls after the test fold's first prediction time. Week-level folds plus purge satisfy this.
- Set E from the actual maximum availability delay (stat corrections, late result ingestion), not from a round number. Record E in the charter.

## 5. Locked final holdout

- **What.** A contiguous most-recent block that no tuning ever touches. Proposed default for game models: the most recent complete season available when step 3 freezes the dataset (currently the 2025 season including postseason). The in-progress 2026 season is forward/shadow data, not a tuning set. For CLV models: the most recent contiguous block of captured weeks, declared when the CLV dataset is frozen.
- **How it is locked (to be enforced in step 3).** A holdout manifest file (holdout id, `game_id` list, season/week range, SHA-256 of the sorted id list, created_at_utc, creator) is committed before any step-3 model fit. Data loaders exclude manifest ids by default; reading them requires an explicit holdout flag plus an experiment id.
- **How it is logged.** Every holdout read appends a line to an append-only holdout access log: timestamp, experiment id, model card id, code commit, purpose. A model is scored on the holdout once, after its features, hyperparameters and calibration are frozen.
- **Burn rule.** If any decision (feature, hyperparameter, model choice, threshold) is changed after looking at holdout results, the holdout is burned: record it in the log, label the result exploratory, and the next unseen season becomes the new holdout. Holdout results are never used for tuning, including "just checking" runs.

## 6. Uncertainty clustered by game

Multiple rows per game (two team perspectives, both spread sides, many books, many snapshots) are not independent.

- Report N as games (and weeks), not rows.
- Confidence intervals for a metric, or for a metric difference between two models, use a **game-cluster bootstrap**: resample `game_id` with replacement (all rows of a game together), recompute the metric, repeat B ≥ 2000. For effects that move a whole week (e.g. market-wide moves), also report a week-block bootstrap.
- Regression-style standard errors use cluster-robust SEs by `game_id`.
- Model comparisons use the **paired** difference per game on identical rows, then the cluster bootstrap on that difference.
- Game-level claims use one row per game (e.g. home perspective). A doubled team-game panel must not be reported as game accuracy.

## 7. Fold diagnostics (run for every fold table)

```python
for f in folds:
    tr, te = df.loc[f.train_idx], df.loc[f.test_idx]
    assert tr.label_available_at_utc.max() < te.prediction_time_utc.min() - f.embargo   # V2
    assert set(tr.game_id).isdisjoint(te.game_id)                                     # V3
    assert set(te.game_id).isdisjoint(holdout_ids)                                    # V6 (dev folds)
```

Record per fold: train/test period, games, rows, target base rate or mean, exclusions with reasons, embargo, null rates of required columns.

## 8. Anti-patterns that invalidate a result

Random K-fold; scaler/imputer fit on all data; early stopping on the outer test fold; repeated holdout peeks; dropping an ugly fold; pooled mean only; different rows per model; doubled team rows counted as games; fold boundaries cutting through a game. (From sports-analytic-skills `skills/validation-design/SKILL.md`, anti-pattern table.)
