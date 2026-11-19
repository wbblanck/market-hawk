# Calibration check

Run on held-out walk-forward predictions only (`validation_design.md`). Calibration of in-sample or holdout-tuned predictions is not evidence.

Adapted from sports-analytic-skills `skills/calibration-check/SKILL.md` (verdict scale, workflow, recalibration rules) and `skills/calibration-check/references/binning.md` (bin rules); NFL push, key-number and distribution checks are specific to this repo.

## Rules

| # | Rule |
|---|---|
| C1 | Report Brier, log loss and binned reliability (with counts) for every probability output. Discrimination (AUC, accuracy) never replaces calibration. |
| C2 | Report calibration overall, **by season**, and **by spread bucket**. |
| C3 | Distributions (margin, total, team points) get PIT and interval-coverage checks plus key-number checks. |
| C4 | Pushes are handled explicitly (Section 4), never silently counted as losses or dropped. |
| C5 | Recalibration (Platt, isotonic, scale/df refit) is fit inside training folds only and applied forward. Fitting a calibrator on the test fold or holdout and calling it validated is forbidden. |
| C6 | One row per game per market side for game-level calibration; do not double count both sides of a spread. |

## 1. Probability metrics

```text
Brier     = mean( (p_i - y_i)^2 )
Log loss  = -mean( y_i*log(p_i) + (1-y_i)*log(1-p_i) ),  p clipped to [1e-6, 1-1e-6]; report the clip count
ECE       = sum_b (n_b / N) * | mean(y in b) - mean(p in b) |
```

Bins: preregister. Default 10 equal-width bins on [0, 1]; use quantile bins when predictions crowd near 0.5 (typical for spreads). Report `n_b` for every bin; bins with `n_b < 30` games are flagged and not interpreted. Inspect tails (< 0.2, > 0.8) separately. Reliability diagram: mean predicted vs observed per bin with a game-cluster bootstrap band.

## 2. Slices

- **Season**: each test season separately; a model calibrated overall but drifting by season is `usable-with-caveats` at best.
- **Spread bucket** on the absolute home spread at T (preregistered): `0-2.5`, `3`, `3.5-6.5`, `7`, `7.5-9.5`, `10+`. Key numbers 3 and 7 get their own bucket.
- Also: home favourite vs home underdog; totals bucket (`<41`, `41-47`, `>47`, adjust to data quantiles); time-to-kickoff bucket for CLV models.

## 3. Distribution checks

- **Randomized PIT** for integer outcomes: `u_i = F_i(y_i - 1) + V_i * (F_i(y_i) - F_i(y_i - 1))`, `V_i ~ U(0,1)`. Histogram of `u` should be flat; U-shape = too narrow, hump = too wide, slope = biased. Report by season.
- **Interval coverage**: empirical coverage of central 50/80/95% intervals vs nominal, with counts.
- **Log score** of realised outcomes (primary distribution metric, `predictive_modeling.md`).
- **Key numbers 3 / 7 / 10** (margin, absolute value) and common totals: compare mean predicted `P(|M| = k)` with observed frequency in the test folds, overall and by spread bucket. A model whose implied P(|M|=3) is materially below the observed rate misprices pushes and half-point value around -3/+3.
- **Coherence**: P(cover)+P(push)+P(no cover) = 1 for every game; derived spread, total and moneyline probabilities come from the same distribution.

## 4. Push handling

- Half-point lines: push probability is 0 by construction; the distribution must still give integer support so that whole-number lines are priced from the same object.
- Whole-number lines: the outcome is ternary (cover / push / no cover). Report:
  - `P(push)` calibration grouped by line value (especially -3, -7, -10 and their positives): mean predicted vs observed push rate with counts;
  - binary cover calibration **conditional on no push**: `p_cover|no push = P(cover) / (1 - P(push))`, evaluated only on non-push games, with the number of pushes excluded stated.
- Totals: the same rule on whole-number totals.
- Settlement follows the book rule for the contract: a push returns stake and is neither a win nor a loss in hit-rate statistics.

## 5. Verdict

| Verdict | Meaning |
|---|---|
| `well-calibrated` | acceptable reliability overall, by season and by spread bucket, populated bins, PIT roughly flat |
| `usable-with-caveats` | disclosed miscalibration in a slice or tail; fold-local recalibration planned |
| `poorly-calibrated` | probabilities not usable as probabilities |
| `invalid-eval` | leakage or split problems block judgement |

No single ECE threshold makes a model well-calibrated. Calibration code (metrics, PIT, key-number tables) is **to be enforced in step 3**; nothing in `src/nfl_market/` computes these yet.
