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
| `<family>+key` | the base family followed by the key-number adjustment (Section 3) | key weights |

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

## 5. Known limitations

- Key adjustment applies to the margin pmf only. For `nb+key`, the adjusted margin is no longer the exact margin of the joint pmf.
- The gamma frailty can only produce positive `corr(home, away)`. The P5 diagnostics show whether the data need more than that.
- NB points ignore the 3/7 lumpiness of scoring. On a crude smoke predictor it gave P(abs M = 3) of 5% against 15% observed, and it was too wide.

## 6. Results on real OOF predictions (pending)

_Placeholder for the lead: run `python -m nfl_market.distributions.evaluate --predictions data/processed/step3_oof_predictions.parquet --out docs/results/step3_distributions.md` and summarise here: log score by family, PIT/coverage, key-number table vs nfl_ats, push calibration, P5 diagnostics, verdict (`calibration_check.md` Section 5)._
