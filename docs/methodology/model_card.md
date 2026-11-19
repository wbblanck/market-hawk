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

## Model card: stack_v1 v1

Status: **experimental** (promotion items 1, 5 and 6 missing: decision `follow-up`, leakage audit unsigned, no
holdout score). Evidence: `docs/results/step4_stacking.md`, experiment `20261119-step4-stacking-01`. Regenerate with
`python scripts/run_stacking.py`.

- **Intended use:** a blended pre-game home margin, total and P(home win) for fair pricing (`markets.pricing`) and
  the dashboard. Research and paper tracking only. It does not place wagers.
- **Target and output:** `PREDICTION_COLUMNS`, model id `stack_v1`, T = kickoff - 24h. `pred_margin` = simplex blend of
  elo_v1, kalman_gs_v1, ridge_v1, xgb_v1. `pred_total` = blend of elo_v1, ridge_v1, xgb_v1 (Kalman has no total).
  `pred_home_win_prob` = P(home win | no tie) from a K1 margin pmf at the stacked margin. SDs are NaN.
- **Data window:** component OOF predictions for 2015-2024 (step 3). The walk-forward evaluation scores 2018-2024. The
  forward weights and K1 are fit on all 2015-2024 OOF rows. 2025 is the holdout: refused in every function and in
  `StackModel.fit`/`predict`.
- **Validation:** weights fit per test season on earlier seasons' rows before its first T, with at least 3 seasons.
  Squared-error objective, no intercept. Equal weights and an MAE objective are reported baselines/variants.
- **Baselines beaten** (pooled 2018-2024, 1,942 games, paired game bootstrap B = 2000):

| baseline | metric | stack_v1 minus baseline [95% CI] | folds won | verdict |
|---|---|---|---|---|
| elo_v1 | margin MAE | -0.009 [-0.074, +0.056] | 3/7 | no difference shown |
| kalman_gs_v1 (best single) | margin MAE | +0.020 [-0.020, +0.057] | 4/7 | no difference shown |
| elo_v1 | total MAE | -0.250 [-0.354, -0.136] | 7/7 | beats |
| xgb_v1 (best single) | total MAE | -0.015 [-0.055, +0.024] | 4/7 | no difference shown |
| elo_v1 | home-win log loss | +0.000 [-0.004, +0.005] | 3/7 | no difference shown |

- **Calibration:** usable-with-caveats. P(home win) ECE 0.030 over 10 bins (elo_v1 0.016). The K1 PIT mean is 0.492
  and 50/80/95 coverage is 0.535/0.808/0.951. The 2019 PIT mean is 0.450. The cover log loss at the nflverse line is
  0.713. That line's capture time is unknown, so this figure is context only.
- **Forward weights** (`config/models/stack_v1_weights.json`): margin elo 0.081, kalman_gs 0.432, ridge 0.201,
  xgb 0.286. Total ridge 0.540, xgb 0.460, elo 0.000.
- **Known failure modes:** the components are highly correlated, so the weights move between near-equivalent models
  from season to season. The weights are refit once per season, so drift within a season is not tracked. The stack
  inherits every component's gaps: no QB or EPA features, and the V5 caveat on 2016-2021.
- **Leakage audit:** not signed. Author checks are in `tests/test_stacking.py`: future perturbation, late labels and
  holdout refusal.
- **Monitoring / kill:** retire if it loses to elo_v1 on margin MAE in two consecutive review windows, each at least
  one season. Retire if its forward CLV (captured closes only) is worse than elo_v1's over at least 200 paper bets.
