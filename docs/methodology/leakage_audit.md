# Leakage audit

No result is accepted, logged as `keep`, or put on a model card until this audit returns `CLEAN`. Stance: assume leakage until the pipeline proves otherwise; a strong metric is never evidence of cleanliness.

Adapted from sports-analytic-skills `skills/leakage-audit/SKILL.md` (verdict scale, audit sequence, too-good triggers), `skills/leakage-audit/references/audit_checklist.md`, and `skills/feature-rules/SKILL.md` (as-of joins, future-perturbation test). Repo-specific timestamp rules come from `src/nfl_market/integrations/base.py`.

## Verdicts

| Verdict | Meaning | Effect |
|---|---|---|
| `CLEAN` | every item below passed with evidence | result may be cited |
| `REVIEW REQUIRED` | no proven leak, evidence incomplete | result stays exploratory |
| `NOT CLEAN` | any feature, join, transform, split or tuning path uses information after T | predictive claims blocked; rebuild, new experiment id, rerun |

Deleting a suspicious column after scoring does not rescue a result. The rebuilt pipeline gets a new experiment id that links to the invalidated one.

## Checklist

### A. Contract
- [ ] Target, row grain and prediction timestamp T written down. Ambiguous T means the audit fails.
- [ ] Label is post-T by construction. Cancelled, postponed and tied games have a written rule.

### B. Features knowable at T
- [ ] Every model input is in the feature registry (`feature_rules.md`) with its `available_at` rule.
- [ ] Every feature row has `available_at_utc <= T`. Test: `src/nfl_market/features/pit.py::assert_point_in_time(df, prediction_time, available_time)` raises on any row where available time > prediction time (exists; covered by `tests/test_pit.py`).
- [ ] Rolling and expanding stats shift before aggregating (the current game never contributes to its own feature).

### C. Market snapshots
- [ ] As-of joins to quotes use `captured_at_utc`, backward direction, `captured_at_utc <= T`. **Never** `source_timestamp_utc` (the provider's own clock, may be missing or wrong) and never kickoff date alone.
- [ ] No backfilled quotes: every quote used was captured by our collector at `captured_at_utc`. `base.py` states providers never assign capture time and `source_timestamp_utc` is never filled with capture time.
- [ ] `synthetic = True` rows (demo/test data, `quote_snapshots.synthetic`) excluded from primary claims.
- [ ] Provider statistics (aggregated "historical odds" endpoints, nflverse `spread_line`/`total_line` in schedules, any line without our capture time) are **not historical quotes**. They may not enter `quote_snapshots`, may not be given a fabricated `captured_at_utc`, and may not be used as a pre-close feature. If used as a benchmark they are labelled with their provider and their unknown capture time.
- [ ] Closing lines, results and injury news captured after T are excluded. A pre-close prediction never uses the close.

### D. Preprocessing inside folds
- [ ] Scalers, imputers, encoders, target encoders, feature selection, PCA, residual pools and calibration maps are fit on the training fold only, then applied to the test fold. All fitted transforms live inside the estimator object that the fold loop fits (e.g. the `make_pipeline(StandardScaler(), ...)` objects in `src/nfl_market/models/baselines.py`).
- [ ] No `fit`/`fit_transform` call on a frame that contains test-fold or holdout rows.
- [ ] Ratings (Elo/Glicko, hierarchical strength) are updated only with games whose labels were available before T.

### E. Stacking
- [ ] Meta-model inputs are historical out-of-fold predictions only: each base prediction for game g was produced by a model whose training data ended before g's T (`predictive_modeling.md` Section 5).
- [ ] No K-fold OOF predictions (those are trained on future games).

### F. Splits
- [ ] Fold table passes `validation_design.md` Section 7 assertions (time order, game disjointness, holdout exclusion).
- [ ] Hyperparameters and early stopping chosen on inner folds only. Holdout access log shows at most one scored read per frozen candidate.

### G. Too-good triggers (stop and re-audit)
- [ ] A game model beats the closing line by a wide margin on MAE or log loss.
- [ ] ATS hit rate or positive-CLV rate far above what the market baseline allows over a season-sized sample.
- [ ] One feature produces a dramatic jump; train and test metrics nearly equal; near-perfect correlation of a feature with the label.

## Concrete tests (to be enforced in step 3 unless marked existing)

| Test | What it proves | Status |
|---|---|---|
| `assert_point_in_time` over every feature column | no row uses a value available after T | **exists** in `pit.py`; wiring to every feature is step 3 |
| Future-perturbation test: change all labels, snapshots and stats after cutoff C, rebuild features, assert every row with T <= C is byte-identical | no future information reaches past rows | step 3 (lesson recorded in `docs/EXISTING_MODELS.md`) |
| As-of join test: after `merge_asof(..., left_on=T, right_on="captured_at_utc", direction="backward")` assert `captured_at_utc <= T` | quotes are as-of capture | step 3 |
| Static check: feature and model code never joins on `source_timestamp_utc` | provider clock is not used as availability | step 3 |
| Fold-local fit test: the fitted scaler mean equals the training-fold mean, not the full-data mean | preprocessing is fold-local | step 3 |
| Fold disjointness and time-order assertions | splits are forward-only and grouped by game | step 3 |
| OOF provenance test: for each meta row, `base_train_end_utc < T` and base model never saw that `game_id` | stacking uses history only | step 3 |
| Target alias scan: exact or near-perfect correlation of any feature with the label | no disguised label | step 3 |

## Known NFL leakage traps

| Trap | Why it leaks | Rule |
|---|---|---|
| Final-score derived stats (same-game EPA, success rate, yards, turnovers, box score) | computed from the game being predicted | label side only; features use prior games, shifted |
| Season-aggregate stats (full-season EPA/play, final standings, season DVOA-style tables) | week-2 row sees weeks 3-18 | expanding as-of values only |
| Post-game injury reports, IR moves, snap counts, "who actually started" from play-by-play | published after kickoff | use the last pre-T injury report / inactives with its publication time; current-game starter only from a pre-T source |
| Closing line as a feature for a pre-close prediction | the close is after T | features use the line as-of T (`captured_at_utc <= T`) |
| nflverse `spread_line` / `total_line` used at an early T | provider line with unknown capture time, close-like | not a feature; at most a labelled benchmark |
| nflverse play-by-play `vegas_wp`, `vegas_home_wp`, `vegas_wpa`, `spread_line`, `total_line` | the same unknown-capture-time line, embedded in every play | never read: `data/nflverse_ext.py` loads only the columns in `PBP_COLUMNS` (tested) |
| Schedule revisions (flexed games, postponements, 2020 reschedules, cancelled games, kickoff time changes) | rest days and T computed from the final schedule, not the one known at T | compute T and rest from the schedule as known at T; cancelled games excluded and flagged |
| Stat corrections | historical tables show corrected values that were not known on the day | use as-of vintages or embargo (validation_design.md Section 4) |
| Weather observations | actual conditions are post-T | forecast vintage as of T only |
| Opponent features joined from a table that includes the matchup | contains the game's own result | opponent's own pre-game row |
| Full-data target encoding of team/venue/referee | test labels enter the encoding | fold-local fit |

## Audit record (attach to the experiment-log entry)

```text
experiment_id / auditor / date
target / grain / T / publication policy
data snapshot id and feature registry version
checklist A-G: PASS | REVIEW | FAIL per item, with evidence (test name, commit, file)
contaminated fields and affected results
required repairs and retests
verdict: CLEAN | REVIEW REQUIRED | NOT CLEAN
```

## Known nonconformance in current code

`src/nfl_market/features/pit.py::add_market_features` computes `book_count_contract` and `market_dispersion` with `groupby(...).transform` over **all** rows of a contract, i.e. across all capture times, including snapshots after any given row. It is not as-of unless the caller first filters to `captured_at_utc <= T`. It also fills `market_dispersion` NaN with 0 (see `feature_rules.md` R4). Both must be fixed or wrapped before these features enter a model (step 3).
