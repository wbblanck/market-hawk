# Methodology standards

Development rules for step 3 onward: NFL game models (ratings, regularised regression, boosting, hierarchical strength), margin/total/team-points distributions, chronological stacking, and CLV models on our own timestamped snapshots. These rules are enforced in review; a result that breaks one is exploratory and cannot be cited, logged as `keep`, or promoted. They refine `docs/METHODOLOGY.md`, `docs/DATA_CONTRACTS.md` and `docs/STAGES_AND_GATES.md`; where code exists it is named, otherwise the doc says "to be enforced in step 3".

## Documents

| Doc | Purpose |
|---|---|
| [validation_design.md](validation_design.md) | Season/week expanding walk-forward, purge/embargo, locked holdout, game-clustered uncertainty, identical folds. |
| [leakage_audit.md](leakage_audit.md) | Checklist, concrete tests and NFL leakage traps; verdict required before any result is accepted. |
| [feature_rules.md](feature_rules.md) | Feature registration fields, point-in-time construction, missing-is-missing, admission procedure. |
| [predictive_modeling.md](predictive_modeling.md) | Baseline ladder, beat-the-baseline rule, margin distribution as central output, chronological OOF stacking. |
| [calibration_check.md](calibration_check.md) | Brier, log loss, ECE/reliability with bins, slices by season and spread bucket, PIT/coverage, key numbers, pushes. |
| [closing_line.md](closing_line.md) | Close snapshot definition, Pinnacle vs labelled fallback, line CLV vs price CLV sign conventions (verified against `clv.py`). |
| [experiment_log.md](experiment_log.md) | Required fields, decision rules and template for append-only experiment records. |
| [model_card.md](model_card.md) | Promotion rule and model-card template. |

## Non-negotiable rules

| # | Rule | Enforced by |
|---|---|---|
| 1 | No random train/test splits for primary NFL predictive claims. | validation_design.md (V1) |
| 2 | All features must be knowable at the prediction timestamp. | feature_rules.md (R2, R3); leakage_audit.md (B, C) |
| 3 | Preprocessing is fitted within each training fold. | leakage_audit.md (D); feature_rules.md (R5) |
| 4 | Stacking uses historical out-of-sample predictions only. | predictive_modeling.md (Section 5); leakage_audit.md (E) |
| 5 | Every model has a baseline comparison. | predictive_modeling.md (P1, P2) |
| 6 | Model promotion requires documented validation results. | model_card.md (promotion rule) |
| 7 | Final holdout data cannot be repeatedly used for tuning. | validation_design.md (V6, Section 5) |
| 8 | Market snapshots are timestamped at capture and never backfilled; provider statistics are not historical quotes. | closing_line.md (L2, L6); leakage_audit.md (C); feature_rules.md (R3) |

## Attribution

Parts of these standards are adapted from WalrusQuant/sports-analytic-skills (MIT). Each doc names the source files it adapts. That repo's optional `sports_ds` toolkit is noted as a possible external benchmark only (`predictive_modeling.md` Section 6); it is not a dependency.
