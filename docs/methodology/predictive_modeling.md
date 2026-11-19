# Predictive modeling

Rules for fitting and comparing game models (margin, total, team points, win/cover probabilities) and CLV models. Validation follows `validation_design.md`; features follow `feature_rules.md`.

Adapted from sports-analytic-skills `skills/predictive-modeling/SKILL.md` and `references/model_ladder.md` (ladder, promotion rule), `skills/baseline-models/SKILL.md` (baselines are often the ship candidate), `skills/simulation-sports/SKILL.md` (state dependence assumptions), `skills/ratings-strength-models/SKILL.md` (Elo design choices).

## 1. Rules

| # | Rule |
|---|---|
| P1 | **Baseline first.** No complex model is fit until the baselines for that target exist on the same fold table. |
| P2 | **Every model beats a baseline on identical folds** (same rows, same fold ids) on the preregistered primary metric, or it is not progress. |
| P3 | **No ensemble entry without incremental OOF value** (Section 5). |
| P4 | **Margin distribution is the central output** of a game model (Section 4). Spread, total and moneyline probabilities are derived from one coherent distribution, not fitted by separate unrelated models. |
| P5 | **Home and away scores are not assumed independent.** Independence is a hypothesis to test, not a modeling default. |
| P6 | Keep the simplest model that wins. If boosting loses to Ridge/logistic, keep Ridge/logistic. |

## 2. Baseline ladder

Climb a rung only when the rung below is beaten on the primary metric on the majority of outer folds and the game-clustered bootstrap CI of the paired difference excludes zero (or the result is labelled inconclusive).

**Game models (margin, total, win/cover):**

| Rung | Model | Notes |
|---:|---|---|
| 0 | Constant: training-fold mean home margin / mean total / home win rate | sanity floor |
| 1a | Market at T (persistence): the spread/total quoted at T, as-of `captured_at_utc` | the honest pre-close baseline |
| 1b | Market close (benchmark): the closing line (`closing_line.md`) | a game model is not expected to beat it; any claim of beating it triggers a leakage audit |
| 1c | Elo (home-field, margin-of-victory multiplier, between-season regression), ratings as-of T | the non-market domain baseline |
| 2 | Ridge / Elastic Net / logistic on admitted features; Glicko; Bayesian hierarchical team and QB strength | classical |
| 3 | XGBoost / LightGBM / CatBoost | only after rung 2 exists on the same folds |
| 4 | Chronological stacking of rungs 1c-3 | Section 5 |

**CLV models** (`docs/METHODOLOGY.md`, `docs/STAGES_AND_GATES.md` stage 5): prior/persistence ("no move"), then logistic for P(positive CLV) and Ridge for CLV magnitude, then boosting and quantile regression. Existing code: `src/nfl_market/models/baselines.py` provides `classifiers()` = {`prior` DummyClassifier, `logistic` StandardScaler+LogisticRegression} and `regressors()` = {`mean` DummyRegressor, `ridge` StandardScaler+Ridge(alpha=10)}. Market-close, persistence and Elo baselines for game models do not exist yet (**to be enforced in step 3**).

## 3. Metrics (lock before fitting)

| Output | Primary | Secondary |
|---|---|---|
| Margin / total point prediction | MAE | RMSE, bias by season and spread bucket |
| Margin / total / team-points distribution | log score of the realised integer outcome (mean negative log pmf) | randomized PIT, interval coverage, key-number mass (`calibration_check.md`) |
| Win / cover / over probability | log loss | Brier, ECE, reliability by season and spread bucket |
| CLV sign | log loss | Brier, calibration |
| CLV magnitude | MAE | RMSE, quantile coverage |

Economic results (ROI, CLV captured) are reported separately from predictive metrics and never substitute for them (`docs/METHODOLOGY.md`).

## 4. Margin distribution as the central output

For each game the model emits a pmf over integer home margin `M = home_pts - away_pts`, preferably via a joint pmf over `(home_pts, away_pts)` so that margin and total come from the same object.

```text
home spread h (home's own line, e.g. -3):  P(cover) = P(M + h > 0)   P(push) = P(M + h = 0)
total line L:                               P(over)  = P(T > L)       P(push) = P(T = L)
moneyline:                                  P(home win) = P(M > 0), P(tie) = P(M = 0)
coherence: P(cover)+P(push)+P(no cover) = 1; P(cover) non-increasing as h gets more negative
```

Distribution families and their required checks:

| Family | Allowed use | Required check |
|---|---|---|
| Empirical residuals around a predicted mean | margin, total | residual pool built from training folds only; optionally conditioned on spread bucket |
| Student-t (location from model, scale/df fit in-fold) | margin | discretize: `P(M=k) = F(k+0.5) - F(k-0.5)`; continuous families flatten key numbers, so key-number mass must be checked and corrected if off |
| Negative Binomial | team points | overdispersion vs Poisson; NFL scoring is lumpy (3, 6, 7, 8), so NB on points alone misses key structure; check margin key numbers derived from it. (`docs/METHODOLOGY.md` says NB is not a default CLV model; this use is for team points only.) |
| Discrete scoring simulation (drives / TD / FG / safety / 2pt) | joint scores | report seeds and draw counts; no draw caps that truncate tails (`docs/EXISTING_MODELS.md`) |

**Independence (P5).** If team points are modeled per team, the joint must carry dependence (shared game pace/total effect, copula, or joint simulation). Test on OOF predictions: compare implied vs observed `corr(home_pts, away_pts)`, implied vs observed `Var(M)` and `Var(T)`, and key-number frequencies. Failing these means the per-team product is not acceptable as the margin distribution.

## 5. Chronological out-of-fold stacking

```text
for each outer fold k (chronological):
  base models trained on data with labels available before fold k -> predictions for fold k  (these are the OOF rows)
  meta-model for fold k trained only on OOF rows from folds < k
  meta-model scores fold k once
```

- OOF predictions are stored with `game_id, model_id, fold_id, base_train_end_utc, prediction_time_utc`; the meta-model never sees an in-sample base prediction (**to be enforced in step 3**).
- No K-fold OOF predictions; they are trained on future games.
- The meta-model starts only after a declared minimum number of OOF folds exist; earlier folds report base models only.
- **Incremental value rule (P3).** A base model enters the stack only if adding it improves the meta-model's primary metric on later folds versus the stack without it, on identical rows, majority of folds, paired game-clustered bootstrap. Otherwise it is dropped.
- Simple averages with fixed weights are allowed as a baseline ensemble; weights chosen on test folds are not.

## 6. Optional external benchmark (no dependency)

sports-analytic-skills ships an optional `sports_ds` toolkit (`src/sports_ds/`). It includes NFL pipelines worth benchmarking against later: `pipelines/nfl_elo_baseline.py` (as-of Elo, k=20, home_adv=65, seasons 2018-2024, 2 min train seasons), `pipelines/nfl_margin_model.py` (team-margin Ridge and HistGradientBoosting via `pipelines/team_margin.py`), `pipelines/nfl_win_model.py`, `ratings/elo.py`, `validation/splits.py` (season walk-forward masks), and `models/ensemble.py` (fixed-weight probability averaging, not OOF stacking). It loads data via nflreadpy on a doubled team-game panel and pulls polars, statsmodels and matplotlib. Use it at most as an external sanity check on identical games; **no mandatory dependency**.
