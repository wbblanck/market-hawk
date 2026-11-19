# Outcome distributions (step 3)

Code: `src/nfl_market/distributions/`. Spec: `predictive_modeling.md` Section 4 (P4, P5) and `calibration_check.md`. Conventions from `src/nfl_market/models/interface.py`: margin `M = home - away`, total `T = home + away`. Spreads are the selected team's own line (`markets/clv.py`): home `h = -3` means home is favoured by 3.

## 1. Objects

| Module | What it holds |
|---|---|
| `pmf.py` | `IntPMF` (integer support, normalised, validated; cdf, quantiles, central intervals) and `JointPMF` over `(home_pts, away_pts)` whose `margin()` and `total()` come from the same matrix |
| `margin.py` | margin families, residual pool, key-number weights, spread buckets |
| `totals.py` | total pmfs (empirical residuals; discretised t/normal); `T = 1` gets zero mass |
| `team_points.py` | Negative Binomial team points with a shared gamma frailty, plus P5 independence diagnostics |
| `markets.py` | moneyline, spread, total and key-number probabilities, fair American odds |
| `calibration.py` | log score, randomized PIT, interval coverage, key-number table, Brier/log loss/ECE with bin counts, push calibration, cover given no push, season slices |
| `evaluate.py` | season walk-forward fit-and-score, plus the CLI |

## 2. Families

| Family (`evaluate` name) | Construction | Fit in-fold |
|---|---|---|
| `empirical` | shifted empirical residuals: `mu + r_i`, where `r_i = outcome - prediction` on training rows. Each shifted value is split linearly between its two neighbouring integers, so the mean is preserved. Can be conditioned on spread bucket (falls back to the full pool below 200 rows). | residual pool |
| `t` | Student-t with location = predicted mean, discretised as `P(M=k) = F(k+0.5) - F(k-0.5)`; tail mass goes to the ±80 end points | scale and df (MLE, df in [2.5, 200]); the model's own `pred_margin_sd` can replace the scale |
| `normal` | discretised normal, sd = training residual sd | sd |
| `kernel` | **ported from nfl_ats** (estimator K). `P(M=k given mu) = sum_i w_i 1[m_i = k] / sum_i w_i`, with `w_i = exp(-0.5((mu_i - mu)/h)^2)`. Realised margins are used **unshifted**, so the mass on 3, 7, 10, 14 and the tie rate stay where they occur in real games. | bandwidth `h` from {0.5, 1, 1.5, 2, 3} by cover log loss on the last 3 training seasons (nfl_ats `choose_h`) |
| `kernel_sym` | `kernel` with each training game added again mirrored, `(-mu_i, -m_i)` (nfl_ats KS) | same |
| `keynum` | **ported from nfl_ats**. Discretised normal times `w_|k|` for `|k| = 0..20`, where `w = observed count / expected count` (one pass), then renormalised. No moment correction, which matches nfl_ats exactly. | sd and weights |
| `shifted_total` | **ported from nfl_ats** `TotalDist`. Shifted empirical whose width scales as `s(T) = sigma (alpha + beta T) / mean abs r`, where `abs r` is regressed on T. Here T is the model's `pred_total`, never the market total. | alpha, beta, sigma, standardised residuals |
| `nb` | NB team points: `H given G ~ NB(mu_h G, r)` and `A given G ~ NB(mu_a G, r)`, with `G ~ Gamma(1/theta, theta)`. `theta = 0` is exact independence, and `Cov(H, A) = mu_h mu_a theta`. `mu_h = (pred_total + pred_margin)/2`. | `r` and `theta` by MLE |
| `k1` | **ported from nfl-betting-model** `batch5.k1_dist`/`fit_k1`. On -60..60, `P(M=m)` is proportional to `phi((m - mu)/sigma) w_abs(m)`, using the normal *density* at integers; w covers abs(m) 0..40, and is 0.05 for a margin never seen | sigma (golden search), w, then sigma again |
| `k1_frozen` | K1 with the frozen `batch5_constants.json` (sigma 13.165, w_3 2.60, w_7 1.89) | none: fit 2016-2020 in that repo |
| `d2` | **ported from nfl-betting-model** `tdfg.game_pmfs`. Per team: touchdowns NB(c_T L e^(±b_T d/100), theta_T), field goals Poisson(c_F L e^(±b_F d/100)), safeties Poisson(s); each TD is worth 6 plus a 0/1/2-point conversion with probabilities p. The two teams are independent, giving a `JointPMF` | none: frozen `batch3_constants.json` (2016-2020) |
| `s1` | **ported from nfl-betting-model** `batch5.s1_margin`. D2 with shared touchdowns: home TD = X_h + X_0, away TD = X_a + X_0, X_0 ~ Poisson(lambda_0), lambda_0 = min(c0 L, 0.95 min(lambda_h, lambda_a)). Returned as the full joint | none: frozen c0 = 0.0279 |
| `<family>+key` | the base family followed by the key-number adjustment (Section 3) | key weights |

Inputs for D2/S1: the original takes an Elo gap `d` and a league scoring level `L`. `scoring_models.tdfg_inputs` solves for `(d, L)` so that the model's analytic expected margin and total equal our `pred_margin`/`pred_total`. K1's `mu` is our `pred_margin` (Elo's fair margin in the original). **The frozen constants were fit on 2016-2020 outcomes, so `k1_frozen`, `s1` and `d2` scores for test seasons 2016-2020 are in-sample.** Only `k1` is refit in-fold.

nfl_ats sign note: there, `c` is the market home spread, which is negative when home is favoured, and the predicted margin is `-c`. Here the conditioning variable is our predicted margin `mu = -c`, so the kernel distances are identical.

## 3. Key numbers, ties and pushes

- **Adjustment (`+key`).** Weights `w_k` are multiplied onto the mass at `M = ±k` for k in {0, 3, 4, 6, 7, 10, 14}. They are fit iteratively on training rows: `w <- w (obs_k + 0.5)/(pred_k + 0.5)`, clipped to [0.02, 5], optionally per spread bucket. The adjusted pmf is renormalised, so total mass is exactly 1. It is then exponentially tilted, `exp(a x + b x^2)`, so its mean and variance equal the base pmf's. Without the variance term, moving mass onto 3 and 7 narrows the distribution (central 80% coverage fell from 0.81 to 0.77 on a smoke run). `keynum` skips the tilt for nfl_ats comparability.
- **Ties.** 0 is a key number in its own right. Continuous families and the shifted empirical put about 2.7–2.9% on `M = 0`, while the observed rate is about 0.2%. The kernel and `+key` families get it right. `markets.moneyline_probs` returns (home win, tie, away win), and `win_given_no_push` gives the two-way moneyline with ties voided.
- **Pushes.** Every family has integer support. For home line h: cover is `M + h > 0`, push is `M + h = 0`, no cover is `M + h < 0`. The away side at `-h` is evaluated on the same pmf. Totals use over `T > L`, push `T = L`, under `T < L`. Half-point lines give zero push by construction. Whole lines are ternary, and `conditional_cover = P(cover) / (1 - P(push))`. Fair odds for a ternary market use the conditional probability.
- **Key-number table** (`calibration.key_number_table`). It compares mean predicted with observed `P(abs M = k)`. Rows: overall; by bucket of the absolute reference home spread (0-2.5, 3, 3.5-6.5, 7, 7.5-9.5, 10+; the value is rounded to the half point first); and `home_fav_4-6`. Signed columns split `M = +k` from `M = -k`. The `all` row is the nfl_ats "margins of exactly 3 / 7" statistic (14.2% / 8.7% observed there), and `home_fav_4-6` is their keynum diagnostic slice.

## 4. Fit-in-fold rules

1. `evaluate_distributions` handles each model and each test season S on its own. Parameters are fit only on rows with `season < S` and `prediction_time_utc` (and `label_available_at_utc`, if present) before the first prediction of S. S is then scored once. A test checks this by perturbing S's outcomes: S's fitted parameters stay the same.
2. Seasons are scored only after `min_history_seasons` earlier OOF seasons exist (default 3).
3. 2025 rows are always dropped (`holdout.exclude_holdout`). Seasons after `max_season` (default 2024) are dropped, so 2026 is neither fit nor scored by default.
4. Key weights for `empirical+key` are fit on the same training rows that make up the pool, which is mildly optimistic. This is not nested, and it is disclosed here.
5. Reference lines for buckets, push and cover use `ref_col` when it is given (a home spread at T). Otherwise they use the model's own implied line, `-round(2 pred_margin)/2`. nflverse `spread_line`/`total_line` are never read.
6. The input must satisfy `PREDICTION_COLUMNS` and include `home_score`/`away_score` (or `margin`/`total`).

## 5. Parity with original code

`scripts/parity/make_goldens_distributions.py` imports the original code from the read-only clones and writes `tests/fixtures/parity/distributions_goldens.json`. The run uses a small, seeded, synthetic training set (900 games, 6 seasons) and fixed query inputs, and stores the inputs, the outputs and the source commits (nfl_ats `c51e678`, nfl-betting-model `4bd6a96`). `tests/test_parity_distributions.py` checks our ports against that file without importing the originals.

| Component (original) | Our port | Max abs diff | Status |
|---|---|---|---|
| `MarginDist("normal")` sf at half-point t | `t_pmf_matrix(df=inf)` | 1e-16 | exact |
| `MarginDist("shifted")` sf | `ResidualPool.sf_continuous` | 0 | exact |
| `MarginDist("kernel")`, `("kernel_sym")` sf, half and whole t | `KernelLocalEmpirical` | 3e-16 | exact |
| `choose_h` (kernel, kernel_sym) | `choose_kernel_h` | identical h (1.5, 2.0) | exact |
| `MarginDist("keynum")` pmf, given their weights | `apply_key_weights_matrix(preserve_*=False)` | 2.5e-16 | exact |
| `batch5.k1_dist` (frozen), `fit_k1` (sigma0, sigma, w) | `k1_matrix`, `fit_k1` | 0 | exact |
| `tdfg.game_pmfs` home/away, `elo.margin_dist` | `d2_team_pmfs`, `raw_margin` | 3e-17 | exact |
| `batch5.s1_margin` | `s1_joint_matrices` + `raw_margin` | 3e-17 | exact |

Intentional or explained differences (each one asserted at its documented size, with no loosened tolerances):

1. **normal at whole-number t: 0.0146.** Ours is a discrete pmf, so `P(M > t)` at whole t equals their continuous sf at `t + 0.5`; the test asserts that identity to 1e-16. A discrete pmf is needed for pushes.
2. **shifted, integer pmf: up to 0.014.** `ResidualPool.pmf` splits each shifted residual linearly between its two neighbouring integers, so at half-point t it equals the average of their sf over `[t - 0.5, t + 0.5]` (asserted to 3e-3 on a 0.01 grid). The exact sf is available as `sf_continuous`. The integer pmf keeps the mean exact and supports whole-number lines.
3. **keynum weights: 7e-8 relative, so 8e-10 on the pmf and under 2e-8 on the sf.** nfl_ats computes expected key counts from base rows that do not sum to 1, because the tail beyond ±80 is dropped; we renormalise rows first. With their weights, our pmf is exact (above).
4. **keynum with our default `fold_tails=True`: 1.3e-7.** By default our discretised normal folds tail mass beyond ±80 into the end points, while nfl_ats drops it. The difference is bounded by twice the tail mass, which is asserted. `fit_keynum(fold_tails=False)` reproduces their convention.
5. **D2 and S1 normalised pmfs: up to 4.6e-7; truncation loss 5e-8 to 1.5e-5.** `JointPMF` renormalises, whereas the originals keep the mass lost by truncating at 100 points per team and at the count limits. Asserted: ours equals theirs divided by their total mass, to 1e-12.
6. **S1 with c0 = 0 against D2: 8e-7 relative.** This is a property of the original: S1 uses Poisson touchdowns, while D2's frozen theta_T = 9.4e7 is NB at the Poisson limit. With theta_T = inf, S1 at c0 = 0 equals D2 exactly (asserted).
7. **Not ported:** EPS clipping of sf to [1e-4, 1 - 1e-4] in nfl_ats (the tests clip ours before comparing); `TotalDist("normal_total")`; batch5 R1/R2 rating models; S1's `fit_c0` and D2's `fit_counts`, which need per-game touchdown and field-goal counts from play-by-play that our prediction frame does not carry.

## 6. Known limitations

- Key adjustment applies to the margin pmf only. For `nb+key`, the adjusted margin is no longer the exact margin of the joint pmf.
- The gamma frailty can only produce positive `corr(home, away)`. The P5 diagnostics show whether the data need more than that.
- NB points ignore the 3/7 lumpiness of scoring. On a crude smoke predictor it gave P(abs M = 3) of 5% against 15% observed, and it was too wide.

## 7. Results on real OOF predictions (pending)

_Placeholder for the lead: run `python -m nfl_market.distributions.evaluate --predictions data/processed/step3_oof_predictions.parquet --out docs/results/step3_distributions.md` and summarise here: log score by family, PIT/coverage, key-number table vs nfl_ats, push calibration, P5 diagnostics, verdict (`calibration_check.md` Section 5)._
