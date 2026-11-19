# Model card and promotion

A model card is the contract for a promoted model. It describes evidence that already exists; it is never written to make an untested model look mature.

Adapted from sports-analytic-skills `skills/model-card/SKILL.md` (required evidence, status and freeze rules, kill conditions) and `skills/model-card/references/card_template.md`, `references/kill_conditions.md`.

## Promotion rule

**No promotion without documented validation results.** A model is promoted (used by the service layer, dashboard, MCP or any advisory output) only when all of these are linked from its card:

1. an experiment-log entry with decision `keep` (`experiment_log.md`);
2. walk-forward results on the locked fold table, per fold and pooled with game-clustered intervals (`validation_design.md`);
3. the baselines it beat on identical folds, with the paired difference (`predictive_modeling.md`);
4. a calibration verdict of `well-calibrated` or `usable-with-caveats` with the caveats stated (`calibration_check.md`);
5. a leakage audit verdict `CLEAN`, signed by someone other than the author (`leakage_audit.md`);
6. one final-holdout score, with its holdout access-log line.

Missing any item: status stays `experimental`. Unknown fields are written `unknown` or `not tested`, never inferred. Enforcement in code (service layer refusing unpromoted model ids) is **to be enforced in step 3**. Advisory release still requires the independent review gate in `docs/STAGES_AND_GATES.md` (stage 10).

## Status and versioning

| Event | Card action |
|---|---|
| First candidate meeting its charter | create `v1`, status `promoted` |
| Feature set, target or T changes | new version, fresh validation |
| Data window or source snapshot changes | new version, fresh validation |
| Hyperparameter change | new version linked to its experiment |
| Editorial fix only | dated correction note in place |
| Kill condition fires or leakage found | status `retired` or `invalidated`; evidence kept |

## Template

```text
# Model card: <model_name> v<N>
status: experimental | promoted | retired | invalidated
owner / reviewers / created / next review date
linked experiments: <experiment_ids>

## Intended use
decision supported; prohibited uses (e.g. no automated wagering; CBS pool picks are not market recommendations)

## Target and output
target; grain (one row per game / per snapshot); prediction timestamp T; output semantics
(e.g. pmf over integer home margin; derived P(cover), P(push), P(over))

## Data window
sources and snapshot ids; seasons/weeks; exclusions (cancelled games, synthetic rows); holdout manifest id

## Features
registry version; feature list with available_at rules; fold-fitted transforms

## Validation
fold table id; outer/inner scheme; embargo; min train history; number of test folds

## Baselines beaten
| baseline (rung) | primary metric | candidate | paired diff [95% game-bootstrap CI] | folds won / total |

## Results
primary metric per fold and pooled; slices: season, spread bucket, postseason; holdout score (single read, log ref)

## Calibration
verdict; reliability table with bin counts; by season and spread bucket; PIT / coverage; key-number (3/7/10) and push checks

## Known failure modes
where it is weak (e.g. early-season weeks, QB changes, extreme spreads, weather games, 2020); dependence assumptions (home/away score correlation); data gaps

## Leakage audit sign-off
verdict CLEAN; auditor; date; audit record ref

## Monitoring and kill conditions
observable trigger, minimum sample, owner, cadence, action. Examples: loses to the locked baseline for two consecutive review windows; calibration in a spread bucket outside stated band for >= N games; required feature missingness above X%; leakage audit reopened; schedule/rule regime outside evaluated scope

## Reproduction
code commit; environment lock; commands; artifact paths and checksums
```

## Hard constraints

- Training metrics are never presented as expected performance.
- Baseline, T, grain and primary metric are never omitted. Metric direction and N (games) are stated with every number.
- No claim beyond evaluated seasons, markets and populations.

## Note on sports_ds

The reference repo's optional `sports_ds` toolkit has NFL pipelines (`src/sports_ds/pipelines/nfl_elo_baseline.py`, `nfl_margin_model.py`, `nfl_win_model.py`) that could serve as an external benchmark row on a card later (`predictive_modeling.md` Section 6). It is not a dependency and its results do not substitute for our own validation.
