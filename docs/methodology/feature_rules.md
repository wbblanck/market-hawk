# Feature rules

Every model input, including raw columns, is a registered feature. Unregistered inputs make a leakage audit fail.

Adapted from sports-analytic-skills `skills/feature-rules/SKILL.md` (legality test, feature card, shift-then-aggregate, as-of joins) and `skills/feature-rules/references/feature_card_template.md`; registry shape compared with that repo's `src/sports_ds/features/registry.py` (`FeatureSpec`: name, formula, legality, shift_rule).

## Rules

| # | Rule |
|---|---|
| R1 | **Registered.** A feature exists only once it has a registry entry (fields below). Registry location and loader are **to be enforced in step 3**; until then the entry lives in the experiment-log record. |
| R2 | **Knowable at T.** Legality test: could someone following our data policy compute this exact value from information captured at or before T? Unknown availability is `REVIEW REQUIRED`, never a pass. A shift does not repair a post-T source. |
| R3 | **Point-in-time construction.** History is built with explicit sort + shift + lookback, or a backward as-of join on the availability timestamp. For market data the availability timestamp is `captured_at_utc` (stamped by our collector, `src/nfl_market/integrations/base.py`), never `source_timestamp_utc`. |
| R4 | **Missing is missing.** A value that does not exist stays NaN/None. No zero-fill, no backfill from later records, no forward-fill past the declared max staleness. Imputation, if any, is a fold-fitted transform with a missingness indicator. Consistent with `docs/METHODOLOGY.md` ("A missing close is missing, not zero") and `base.py` ("Missing values stay None, never 0"). |
| R5 | **Fold-fitted parts are declared.** Scaling, encoding, target encoding, selection and imputation are listed on the entry and fit inside each training fold. |
| R6 | **Stable IDs.** Joins use `event_id`/`game_id`/team ids with asserted cardinality, never display names. |
| R7 | **Admission before use.** Only `admitted` features may appear in a candidate that is compared against baselines. |

## Registry entry (required fields)

```text
name / version
description (one line) and formula in plain language
source: table or provider, raw column(s), provider name
grain: game | team-game | snapshot (event, book, market, selection, line, captured_at_utc)
available_at rule: how available_at_utc is computed (e.g. captured_at_utc; report publication time; game final + stat-correction allowance)
availability lag: stable delay between event and availability (e.g. injury report Wed/Thu/Fri publication times; stats next day)
as-of rule: shift / lookback / min periods / as-of join key and direction; max staleness
missing policy: NaN meaning; staleness expiry; early-season / first-game behavior (prior? exclusion?)
fold-fitted components: none | list
owner
status: proposed | admitted | retired; date; reviewer
tests: point-in-time test name, perturbation test name
used by: experiment ids / model cards
```

## Point-in-time construction patterns

- **Shift, then aggregate.** `groupby(team).shift(1)` before `expanding()` or `rolling()`. Sort by (team, kickoff_utc, game_id) explicitly; input order is not chronology.
- **Ratings.** Elo/Glicko/hierarchical strength for game g use only games with `label_available_at_utc < T_g`. Preseason priors (carryover, regression to mean) are written on the entry.
- **Market features.** Filter `captured_at_utc <= T` first, then take the last quote per contract/book (with max quote age), then aggregate across books. Movement features (open-to-T change, reversals, quote age, dispersion) are computed from the filtered set only.
- **Opponent features.** Build each team's pre-game row first, then join the opponent's own pre-game row on `game_id`, one-to-one.
- **Schedule features (rest, travel, short week).** From the schedule as known at T, not the final revised schedule.
- **Injury / QB status.** From the last report published before T, with its publication timestamp; not from post-game participation.

## Admission procedure

1. Write the registry entry (status `proposed`).
2. Build the feature with the as-of rule; run `assert_point_in_time` (`src/nfl_market/features/pit.py`) on its `available_at_utc` column.
3. Run the future-perturbation test (`leakage_audit.md`); unchanged past rows required.
4. Hand-check boundary rows: first game of a season, first week after a bye, a flexed or rescheduled game, a game with a single book quoted.
5. Report null rate by season/week and the meaning of each null.
6. Reviewer (not the author) signs the entry; status becomes `admitted`.

Admission establishes legality, not value. A feature earns a place in a promoted model only by improving out-of-fold results on identical folds (`predictive_modeling.md`).

## Known nonconformance

`src/nfl_market/features/pit.py::add_market_features` sets `market_dispersion` = per-contract std with `.fillna(0)`. The std across a single book is undefined, so 0 asserts "no disagreement" that was never observed (breaks R4). It also computes across all capture times (breaks R3; see `leakage_audit.md`). To be fixed before admission in step 3.
