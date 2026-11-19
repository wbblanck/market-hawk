# Step 4 stacking results (stack_v1)

Generated 2026-11-19T18:06:59Z by `python scripts/run_stacking.py` (code commit `de22c97` plus uncommitted step-4 working tree; runtime 13s). Experiment `20261119-step4-stacking-01` (`docs/methodology/experiment_log.md`).

**Status: experimental.** Inputs are the step-3 OOF predictions (`data/processed/step3_oof_predictions.parquet`, test seasons 2015-2024). No 2025 (holdout `nfl_games_2025_v1`) or 2026 game is read, fit or scored.

## Design (fixed before scoring)

- **Components.** Margin stack over `elo_v1`, `kalman_gs_v1`, `ridge_v1`, `xgb_v1`. Total stack over the models that predict totals: `elo_v1`, `ridge_v1`, `xgb_v1` (`kalman_gs_v1` is margin-only). `constant_v1` and `bayes_od_v1` are reference rows; `bayes_od_v1` as a fifth candidate is a separately reported variant.
- **Walk-forward weights.** For test season S, weights are fit on OOF rows of seasons < S whose prediction time and label-available time are before S's first prediction time (`distributions.evaluate.training_rows`). One weight vector per season (no weekly refit inside S). At least 3 training seasons, so the first scored season is 2018 (trained on 2015-2017 OOF rows); scored seasons 2018-2024, 1942 games. Every single-model row below is on the same games.
- **Objective.** Primary: squared error on the simplex (weights >= 0, sum 1), no intercept, chosen before any test-season score was computed. Reported variant: MAE on the simplex (`stack_mae_v1`). Baseline stack: equal weights (`stack_eq_v1`).
- **Win probability.** Stacked margin -> K1 margin distribution (`fit_distribution(..., "k1")`) refit for each test season on the same training rows, with their stack margin computed from season S's weights (those weights are in-sample on the K1 training rows; nothing from S or later is used). `pred_home_win_prob = P(M > 0) / (1 - P(M = 0))`. `stack_eq_v1` uses the same procedure.
- **Clean seasons.** Step 3's clean seasons are 2015, 2022, 2023, 2024 (V5 caveat: ported constants chosen on 2016-2021). 2015 cannot be scored here (no 3 prior OOF seasons), so clean = 2022-2024.
- **Verdict rule.** "beats" only if the paired game-clustered bootstrap 95% CI (B = 2000) of the loss difference is below 0 AND the candidate wins a majority of folds. Negative difference = candidate better. "Best single" is the component with the lowest pooled value of that metric on these games (chosen on the test games, which favours the baseline).

## Headline

- stack_v1 margin MAE 10.211 vs elo_v1 10.221: -0.009 [-0.074, +0.056], folds won 3/7: **no difference shown**.
- stack_v1 margin vs best single (kalman_gs_v1, 10.192): +0.020 [-0.020, +0.057], folds won 4/7: **no difference shown**.
- stack_v1 total MAE 10.743 vs elo_v1 10.993: -0.250 [-0.354, -0.136], folds won 7/7: **beats**; vs best single (xgb_v1, 10.758): -0.015 [-0.055, +0.024], folds won 4/7: **no difference shown**.
- stack_v1 home-win log loss 0.6371 vs elo_v1 0.6369: +0.000 [-0.004, +0.005], folds won 3/7: **no difference shown**.
- Equal-weight stack margin MAE 10.185: vs elo_v1 -0.035 [-0.086, +0.018], folds won 3/7: **no difference shown**.
- Comparisons meeting the "beats" rule (pooled): stack_v1 vs elo_v1 on abs_total; stack_eq_v1 vs elo_v1 on abs_total. Gains over the best single model are small; read the CIs, not the point estimates.

## Pooled, scored seasons 2018-2024 (same games)

| model_id | n_games | mae_margin | rmse_margin | bias_margin | mae_total | n_win | logloss_win | brier_win |
|---|---|---|---|---|---|---|---|---|
| stack_v1 | 1942 | 10.211 | 13.173 | 0.271 | 10.743 | 1935 | 0.637 | 0.223 |
| stack_eq_v1 | 1942 | 10.185 | 13.151 | 0.229 | 10.776 | 1935 | 0.635 | 0.222 |
| elo_v1 | 1942 | 10.221 | 13.228 | -0.197 | 10.993 | 1935 | 0.637 | 0.223 |
| kalman_gs_v1 | 1942 | 10.192 | 13.152 | -0.078 |  | 1935 | 0.637 | 0.223 |
| ridge_v1 | 1942 | 10.198 | 13.186 | 0.539 | 10.763 | 1935 | 0.636 | 0.223 |
| xgb_v1 | 1942 | 10.265 | 13.195 | 0.651 | 10.758 | 1935 | 0.638 | 0.224 |
| bayes_od_v1 | 1942 | 10.205 | 13.166 | 0.009 | 10.833 | 1935 | 0.637 | 0.223 |
| constant_v1 | 1942 | 11.029 | 14.293 | 0.735 | 11.018 | 1935 | 0.690 | 0.249 |

### Clean seasons only (2022-2024)

| model_id | n_games | mae_margin | rmse_margin | bias_margin | mae_total | n_win | logloss_win | brier_win |
|---|---|---|---|---|---|---|---|---|
| stack_v1 | 854 | 9.879 | 12.840 | -0.496 | 10.379 | 852 | 0.636 | 0.223 |
| stack_eq_v1 | 854 | 9.859 | 12.831 | -0.415 | 10.410 | 852 | 0.634 | 0.222 |
| elo_v1 | 854 | 9.881 | 12.932 | -0.995 | 10.618 | 852 | 0.635 | 0.222 |
| kalman_gs_v1 | 854 | 9.918 | 12.876 | -0.818 |  | 852 | 0.637 | 0.223 |
| ridge_v1 | 854 | 9.872 | 12.850 | 0.013 | 10.393 | 852 | 0.636 | 0.223 |
| xgb_v1 | 854 | 9.898 | 12.822 | 0.140 | 10.394 | 852 | 0.636 | 0.223 |
| bayes_od_v1 | 854 | 9.946 | 12.893 | -0.300 | 10.519 | 852 | 0.639 | 0.224 |
| constant_v1 | 854 | 10.601 | 13.780 | -0.094 | 10.600 | 852 | 0.686 | 0.246 |

## Paired bootstrap and folds won

| scope | candidate | baseline | metric | n_games | diff | lo | hi | folds_won | verdict |
|---|---|---|---|---|---|---|---|---|---|
| pooled 2018-2024 | stack_v1 | elo_v1 | abs_margin | 1942 | -0.009 | -0.074 | 0.056 | 3/7 | no difference shown |
| pooled 2018-2024 | stack_v1 | kalman_gs_v1 | abs_margin | 1942 | 0.020 | -0.020 | 0.057 | 4/7 | no difference shown |
| pooled 2018-2024 | stack_v1 | elo_v1 | abs_total | 1942 | -0.250 | -0.354 | -0.136 | 7/7 | beats |
| pooled 2018-2024 | stack_v1 | xgb_v1 | abs_total | 1942 | -0.015 | -0.055 | 0.024 | 4/7 | no difference shown |
| pooled 2018-2024 | stack_v1 | elo_v1 | logloss_win | 1935 | 0.000 | -0.004 | 0.005 | 3/7 | no difference shown |
| pooled 2018-2024 | stack_v1 | ridge_v1 | logloss_win | 1935 | 0.001 | -0.002 | 0.003 | 4/7 | no difference shown |
| pooled 2018-2024 | stack_eq_v1 | elo_v1 | abs_margin | 1942 | -0.035 | -0.086 | 0.018 | 3/7 | no difference shown |
| pooled 2018-2024 | stack_eq_v1 | kalman_gs_v1 | abs_margin | 1942 | -0.007 | -0.052 | 0.038 | 3/7 | no difference shown |
| pooled 2018-2024 | stack_eq_v1 | elo_v1 | abs_total | 1942 | -0.217 | -0.289 | -0.138 | 7/7 | beats |
| pooled 2018-2024 | stack_eq_v1 | xgb_v1 | abs_total | 1942 | 0.018 | -0.041 | 0.074 | 4/7 | no difference shown |
| pooled 2018-2024 | stack_eq_v1 | elo_v1 | logloss_win | 1935 | -0.002 | -0.005 | 0.002 | 3/7 | no difference shown |
| pooled 2018-2024 | stack_eq_v1 | ridge_v1 | logloss_win | 1935 | -0.001 | -0.003 | 0.001 | 4/7 | no difference shown |
| clean 2022-2024 | stack_v1 | elo_v1 | abs_margin | 854 | -0.002 | -0.096 | 0.094 | 1/3 | no difference shown |
| clean 2022-2024 | stack_v1 | kalman_gs_v1 | abs_margin | 854 | -0.039 | -0.084 | 0.001 | 3/3 | no difference shown |
| clean 2022-2024 | stack_v1 | elo_v1 | abs_total | 854 | -0.239 | -0.393 | -0.089 | 3/3 | beats |
| clean 2022-2024 | stack_v1 | xgb_v1 | abs_total | 854 | -0.016 | -0.074 | 0.040 | 2/3 | no difference shown |
| clean 2022-2024 | stack_v1 | elo_v1 | logloss_win | 852 | 0.001 | -0.006 | 0.007 | 1/3 | no difference shown |
| clean 2022-2024 | stack_v1 | ridge_v1 | logloss_win | 852 | -0.000 | -0.004 | 0.003 | 2/3 | no difference shown |
| clean 2022-2024 | stack_eq_v1 | elo_v1 | abs_margin | 854 | -0.022 | -0.098 | 0.055 | 1/3 | no difference shown |
| clean 2022-2024 | stack_eq_v1 | kalman_gs_v1 | abs_margin | 854 | -0.059 | -0.129 | 0.004 | 3/3 | no difference shown |
| clean 2022-2024 | stack_eq_v1 | elo_v1 | abs_total | 854 | -0.207 | -0.317 | -0.100 | 3/3 | beats |
| clean 2022-2024 | stack_eq_v1 | xgb_v1 | abs_total | 854 | 0.016 | -0.060 | 0.094 | 2/3 | no difference shown |
| clean 2022-2024 | stack_eq_v1 | elo_v1 | logloss_win | 852 | -0.001 | -0.007 | 0.005 | 1/3 | no difference shown |
| clean 2022-2024 | stack_eq_v1 | ridge_v1 | logloss_win | 852 | -0.002 | -0.005 | 0.001 | 2/3 | no difference shown |

## Per fold

Margin MAE:

| fold_id | stack_v1 | stack_eq_v1 | elo_v1 | kalman_gs_v1 | ridge_v1 | xgb_v1 | bayes_od_v1 | constant_v1 |
|---|---|---|---|---|---|---|---|---|
| 2018 | 10.115 | 10.086 | 10.257 | 10.046 | 10.044 | 10.121 | 10.072 | 10.612 |
| 2019 | 10.508 | 10.441 | 10.406 | 10.375 | 10.548 | 10.614 | 10.412 | 11.618 |
| 2020 | 10.119 | 10.072 | 10.013 | 10.052 | 10.038 | 10.284 | 10.095 | 11.130 |
| 2021 | 11.107 | 11.123 | 11.225 | 11.109 | 11.142 | 11.152 | 11.015 | 12.051 |
| 2022 | 9.009 | 8.983 | 8.978 | 9.062 | 9.003 | 9.034 | 9.146 | 9.575 |
| 2023 | 10.495 | 10.492 | 10.564 | 10.528 | 10.455 | 10.557 | 10.466 | 11.081 |
| 2024 | 10.131 | 10.099 | 10.098 | 10.162 | 10.155 | 10.100 | 10.223 | 11.145 |

Total MAE:

| fold_id | stack_v1 | stack_eq_v1 | elo_v1 | ridge_v1 | xgb_v1 | bayes_od_v1 | constant_v1 |
|---|---|---|---|---|---|---|---|
| 2018 | 11.093 | 11.175 | 11.424 | 11.166 | 11.084 | 11.187 | 11.482 |
| 2019 | 11.003 | 11.002 | 11.215 | 10.982 | 11.041 | 11.007 | 11.164 |
| 2020 | 10.933 | 10.977 | 11.214 | 10.884 | 11.006 | 10.921 | 11.641 |
| 2021 | 11.082 | 11.095 | 11.295 | 11.176 | 11.044 | 11.196 | 11.109 |
| 2022 | 10.673 | 10.743 | 10.998 | 10.693 | 10.661 | 10.813 | 10.935 |
| 2023 | 10.470 | 10.485 | 10.716 | 10.499 | 10.492 | 10.617 | 10.708 |
| 2024 | 9.994 | 10.003 | 10.140 | 9.987 | 10.030 | 10.128 | 10.158 |

Home-win log loss:

| fold_id | stack_v1 | stack_eq_v1 | elo_v1 | kalman_gs_v1 | ridge_v1 | xgb_v1 | bayes_od_v1 | constant_v1 |
|---|---|---|---|---|---|---|---|---|
| 2018 | 0.6269 | 0.6252 | 0.6382 | 0.6244 | 0.6211 | 0.6268 | 0.6230 | 0.6756 |
| 2019 | 0.6374 | 0.6357 | 0.6349 | 0.6352 | 0.6416 | 0.6389 | 0.6368 | 0.6974 |
| 2020 | 0.6312 | 0.6267 | 0.6187 | 0.6294 | 0.6228 | 0.6396 | 0.6316 | 0.7032 |
| 2021 | 0.6564 | 0.6549 | 0.6607 | 0.6576 | 0.6591 | 0.6518 | 0.6497 | 0.6976 |
| 2022 | 0.6335 | 0.6322 | 0.6321 | 0.6362 | 0.6337 | 0.6327 | 0.6401 | 0.6841 |
| 2023 | 0.6527 | 0.6531 | 0.6631 | 0.6535 | 0.6506 | 0.6590 | 0.6498 | 0.6848 |
| 2024 | 0.6204 | 0.6175 | 0.6098 | 0.6217 | 0.6238 | 0.6166 | 0.6280 | 0.6894 |

## Weight path (walk-forward, one vector per test season)

Margin (stack_v1 = squared error, stack_mae_v1 = MAE variant):

| model_id | season | target | n_train | train_seasons | intercept | w_elo_v1 | w_kalman_gs_v1 | w_ridge_v1 | w_xgb_v1 |
|---|---|---|---|---|---|---|---|---|---|
| stack_v1 | 2018 | margin | 801 | 2015-2017 | 0.000 | 0.257 | 0.000 | 0.125 | 0.618 |
| stack_v1 | 2019 | margin | 1068 | 2015-2018 | 0.000 | 0.000 | 0.230 | 0.280 | 0.490 |
| stack_v1 | 2020 | margin | 1335 | 2015-2019 | 0.000 | 0.008 | 0.523 | 0.124 | 0.345 |
| stack_v1 | 2021 | margin | 1604 | 2015-2020 | 0.000 | 0.171 | 0.515 | 0.273 | 0.042 |
| stack_v1 | 2022 | margin | 1889 | 2015-2021 | 0.000 | 0.121 | 0.538 | 0.183 | 0.158 |
| stack_v1 | 2023 | margin | 2173 | 2015-2022 | 0.000 | 0.139 | 0.507 | 0.160 | 0.194 |
| stack_v1 | 2024 | margin | 2458 | 2015-2023 | 0.000 | 0.010 | 0.527 | 0.272 | 0.191 |
| stack_mae_v1 | 2018 | margin | 801 | 2015-2017 | 0.000 | 0.191 | 0.000 | 0.294 | 0.515 |
| stack_mae_v1 | 2019 | margin | 1068 | 2015-2018 | 0.000 | 0.000 | 0.199 | 0.619 | 0.183 |
| stack_mae_v1 | 2020 | margin | 1335 | 2015-2019 | 0.000 | 0.074 | 0.567 | 0.359 | 0.000 |
| stack_mae_v1 | 2021 | margin | 1604 | 2015-2020 | 0.000 | 0.090 | 0.554 | 0.356 | 0.000 |
| stack_mae_v1 | 2022 | margin | 1889 | 2015-2021 | 0.000 | 0.074 | 0.567 | 0.359 | 0.000 |
| stack_mae_v1 | 2023 | margin | 2173 | 2015-2022 | 0.000 | 0.106 | 0.479 | 0.415 | 0.000 |
| stack_mae_v1 | 2024 | margin | 2458 | 2015-2023 | 0.000 | 0.086 | 0.448 | 0.466 | 0.000 |

Total:

| model_id | season | target | n_train | train_seasons | intercept | w_elo_v1 | w_ridge_v1 | w_xgb_v1 |
|---|---|---|---|---|---|---|---|---|
| stack_v1 | 2018 | total | 801 | 2015-2017 | 0.000 | 0.062 | 0.248 | 0.690 |
| stack_v1 | 2019 | total | 1068 | 2015-2018 | 0.000 | 0.000 | 0.294 | 0.706 |
| stack_v1 | 2020 | total | 1335 | 2015-2019 | 0.000 | 0.010 | 0.458 | 0.532 |
| stack_v1 | 2021 | total | 1604 | 2015-2020 | 0.000 | 0.079 | 0.555 | 0.367 |
| stack_v1 | 2022 | total | 1889 | 2015-2021 | 0.000 | 0.084 | 0.469 | 0.446 |
| stack_v1 | 2023 | total | 2173 | 2015-2022 | 0.000 | 0.020 | 0.494 | 0.486 |
| stack_v1 | 2024 | total | 2458 | 2015-2023 | 0.000 | 0.000 | 0.571 | 0.429 |
| stack_mae_v1 | 2018 | total | 801 | 2015-2017 | 0.000 | 0.215 | 0.236 | 0.549 |
| stack_mae_v1 | 2019 | total | 1068 | 2015-2018 | 0.000 | 0.052 | 0.208 | 0.740 |
| stack_mae_v1 | 2020 | total | 1335 | 2015-2019 | 0.000 | 0.010 | 0.318 | 0.672 |
| stack_mae_v1 | 2021 | total | 1604 | 2015-2020 | 0.000 | 0.006 | 0.473 | 0.521 |
| stack_mae_v1 | 2022 | total | 1889 | 2015-2021 | 0.000 | 0.000 | 0.385 | 0.615 |
| stack_mae_v1 | 2023 | total | 2173 | 2015-2022 | 0.000 | 0.000 | 0.370 | 0.630 |
| stack_mae_v1 | 2024 | total | 2458 | 2015-2023 | 0.000 | 0.000 | 0.401 | 0.599 |

Forward weights (fit on all OOF seasons 2015-2024; `config/models/stack_v1_weights.json`, used by `StackModel.from_weights_file` and the dashboard's `stack_weight_*` fields):

| target | w_elo_v1 | w_kalman_gs_v1 | w_ridge_v1 | w_xgb_v1 | n_train |
|---|---|---|---|---|---|
| margin | 0.081 | 0.432 | 0.201 | 0.286 | 2743 |
| total | 0.000 |  | 0.540 | 0.460 | 2743 |

Forward K1 for the win-probability mapping: sigma 13.013, w_3 2.682, w_7 1.741.


K1 parameters of the stack per test season:

| season | n_train | train_seasons | k1_sigma | k1_w3 | k1_w7 |
|---|---|---|---|---|---|
| 2018 | 801 | 2015-2017 | 12.737 | 2.688 | 1.848 |
| 2019 | 1068 | 2015-2018 | 12.883 | 2.792 | 1.861 |
| 2020 | 1335 | 2015-2019 | 12.979 | 2.625 | 1.991 |
| 2021 | 1604 | 2015-2020 | 12.950 | 2.605 | 1.902 |
| 2022 | 1889 | 2015-2021 | 13.096 | 2.668 | 1.835 |
| 2023 | 2173 | 2015-2022 | 12.937 | 2.690 | 1.791 |
| 2024 | 2458 | 2015-2023 | 13.028 | 2.701 | 1.777 |

## Variants (not primary; reported separately)

- `stack_mae_v1`: MAE objective (reported variant)
- `stack_b5_v1`: squared error, bayes_od_v1 added as a fifth candidate (margin and total)

| model_id | n_games | mae_margin | rmse_margin | bias_margin | mae_total | n_win | logloss_win | brier_win |
|---|---|---|---|---|---|---|---|---|
| stack_mae_v1 | 1942 | 10.191 | 13.163 | 0.276 | 10.746 | 1935 | 0.636 | 0.223 |
| stack_b5_v1 | 1942 | 10.210 | 13.174 | 0.233 | 10.728 | 1935 | 0.637 | 0.223 |


| scope | candidate | baseline | metric | n_games | diff | lo | hi | folds_won | verdict |
|---|---|---|---|---|---|---|---|---|---|
| pooled 2018-2024 | stack_mae_v1 | elo_v1 | abs_margin | 1942 | -0.029 | -0.090 | 0.034 | 3/7 | no difference shown |
| pooled 2018-2024 | stack_mae_v1 | elo_v1 | abs_total | 1942 | -0.247 | -0.352 | -0.132 | 7/7 | beats |
| pooled 2018-2024 | stack_mae_v1 | elo_v1 | logloss_win | 1935 | -0.001 | -0.005 | 0.004 | 3/7 | no difference shown |
| pooled 2018-2024 | stack_b5_v1 | elo_v1 | abs_margin | 1942 | -0.010 | -0.074 | 0.053 | 3/7 | no difference shown |
| pooled 2018-2024 | stack_b5_v1 | elo_v1 | abs_total | 1942 | -0.265 | -0.384 | -0.143 | 7/7 | beats |
| pooled 2018-2024 | stack_b5_v1 | elo_v1 | logloss_win | 1935 | -0.000 | -0.004 | 0.004 | 3/7 | no difference shown |

`stack_b5_v1` weight path:

| model_id | season | target | n_train | train_seasons | intercept | w_elo_v1 | w_kalman_gs_v1 | w_ridge_v1 | w_xgb_v1 | w_bayes_od_v1 |
|---|---|---|---|---|---|---|---|---|---|---|
| stack_b5_v1 | 2018 | margin | 801 | 2015-2017 | 0.000 | 0.256 | 0.000 | 0.123 | 0.614 | 0.006 |
| stack_b5_v1 | 2018 | total | 801 | 2015-2017 | 0.000 | 0.124 |  | 0.047 | 0.533 | 0.295 |
| stack_b5_v1 | 2019 | margin | 1068 | 2015-2018 | 0.000 | 0.000 | 0.230 | 0.280 | 0.490 | 0.000 |
| stack_b5_v1 | 2019 | total | 1068 | 2015-2018 | 0.000 | 0.087 |  | 0.046 | 0.473 | 0.394 |
| stack_b5_v1 | 2020 | margin | 1335 | 2015-2019 | 0.000 | 0.024 | 0.451 | 0.109 | 0.330 | 0.086 |
| stack_b5_v1 | 2020 | total | 1335 | 2015-2019 | 0.000 | 0.119 |  | 0.103 | 0.302 | 0.476 |
| stack_b5_v1 | 2021 | margin | 1604 | 2015-2020 | 0.000 | 0.243 | 0.210 | 0.205 | 0.000 | 0.341 |
| stack_b5_v1 | 2021 | total | 1604 | 2015-2020 | 0.000 | 0.171 |  | 0.161 | 0.158 | 0.510 |
| stack_b5_v1 | 2022 | margin | 1889 | 2015-2021 | 0.000 | 0.196 | 0.213 | 0.121 | 0.101 | 0.369 |
| stack_b5_v1 | 2022 | total | 1889 | 2015-2021 | 0.000 | 0.166 |  | 0.087 | 0.262 | 0.485 |
| stack_b5_v1 | 2023 | margin | 2173 | 2015-2022 | 0.000 | 0.188 | 0.289 | 0.119 | 0.161 | 0.244 |
| stack_b5_v1 | 2023 | total | 2173 | 2015-2022 | 0.000 | 0.117 |  | 0.142 | 0.302 | 0.439 |
| stack_b5_v1 | 2024 | margin | 2458 | 2015-2023 | 0.000 | 0.078 | 0.232 | 0.210 | 0.147 | 0.333 |
| stack_b5_v1 | 2024 | total | 2458 | 2015-2023 | 0.000 | 0.075 |  | 0.259 | 0.274 | 0.392 |

## Win probability: stack K1 mapping vs the components' own probabilities

Non-tie games; ECE with 10 equal-width bins. The last row averages the components' own P(home win) with stack_v1's margin weights (no distribution step).

| model | n_win | logloss | brier | ece_10bin |
|---|---|---|---|---|
| stack_v1 | 1935 | 0.6371 | 0.2230 | 0.0304 |
| stack_eq_v1 | 1935 | 0.6352 | 0.2222 | 0.0271 |
| elo_v1 | 1935 | 0.6369 | 0.2228 | 0.0161 |
| kalman_gs_v1 | 1935 | 0.6371 | 0.2229 | 0.0283 |
| ridge_v1 | 1935 | 0.6363 | 0.2227 | 0.0346 |
| xgb_v1 | 1935 | 0.6380 | 0.2235 | 0.0241 |
| stack_v1 margin weights on component P(win) | 1935 | 0.6372 | 0.2231 | 0.0296 |

## Calibration of stack_v1

P(home win) ECE (10 bins) 0.0304. Home win | no tie from the K1 pmfs: log loss 0.6371, Brier 0.2230, 7 ties excluded.


Reliability (P(home win), equal-width bins; `flag_small` = fewer than 30 games):

| bin_lo | bin_hi | n | mean_pred | observed | flag_small |
|---|---|---|---|---|---|
| 0.000 | 0.100 | 0 |  |  | True |
| 0.100 | 0.200 | 15 | 0.162 | 0.333 | True |
| 0.200 | 0.300 | 91 | 0.263 | 0.297 | False |
| 0.300 | 0.400 | 195 | 0.356 | 0.323 | False |
| 0.400 | 0.500 | 377 | 0.452 | 0.403 | False |
| 0.500 | 0.600 | 483 | 0.553 | 0.524 | False |
| 0.600 | 0.700 | 437 | 0.646 | 0.670 | False |
| 0.700 | 0.800 | 250 | 0.741 | 0.752 | False |
| 0.800 | 0.900 | 81 | 0.836 | 0.840 | False |
| 0.900 | 1.000 | 6 | 0.910 | 1.000 | True |

K1 margin distribution: log score, randomized PIT and central-interval coverage by season (well calibrated: pit_mean 0.5, pit_var 1/12 = 0.083, covN close to massN):

| season | log_score | n | pit_mean | pit_var | pit_ks | pit_ks_p | cov50 | mass50 | cov80 | mass80 | cov95 | mass95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018 | 3.881 | 267 | 0.497 | 0.082 | 0.032 | 0.934 | 0.558 | 0.531 | 0.813 | 0.814 | 0.933 | 0.955 |
| 2019 | 3.915 | 267 | 0.450 | 0.085 | 0.090 | 0.026 | 0.498 | 0.531 | 0.798 | 0.816 | 0.955 | 0.955 |
| 2020 | 3.866 | 269 | 0.474 | 0.082 | 0.052 | 0.439 | 0.506 | 0.532 | 0.807 | 0.814 | 0.959 | 0.955 |
| 2021 | 3.971 | 285 | 0.502 | 0.091 | 0.041 | 0.719 | 0.502 | 0.530 | 0.789 | 0.814 | 0.930 | 0.955 |
| 2022 | 3.762 | 284 | 0.503 | 0.072 | 0.054 | 0.373 | 0.577 | 0.530 | 0.852 | 0.813 | 0.975 | 0.955 |
| 2023 | 3.866 | 285 | 0.518 | 0.084 | 0.046 | 0.554 | 0.533 | 0.532 | 0.786 | 0.813 | 0.944 | 0.955 |
| 2024 | 3.877 | 285 | 0.506 | 0.082 | 0.035 | 0.854 | 0.565 | 0.532 | 0.814 | 0.814 | 0.958 | 0.955 |
| all | 3.877 | 1942 | 0.492 | 0.083 | 0.022 | 0.283 | 0.535 | 0.531 | 0.808 | 0.814 | 0.951 | 0.955 |

Interval coverage, pooled:

| level | n | coverage | predicted_mass | mean_width |
|---|---|---|---|---|
| 0.500 | 1942 | 0.535 | 0.531 | 17.472 |
| 0.800 | 1942 | 0.808 | 0.814 | 34.044 |
| 0.950 | 1942 | 0.951 | 0.955 | 53.496 |

## Cover calibration at the nflverse spread line (nflverse line, capture time unknown; context only, never a feature)

P(home cover | no push) from each model's own walk-forward K1 distribution at the nflverse line, pushes excluded, 10 equal-width bins. The line is close-like, so the observed cover rate is near 50% in every bin; a log loss above ln 2 = 0.693 means worse than a constant 50% forecast.

| model | n_no_push | n_push_excluded | cover_logloss | brier | ece_10bin |
|---|---|---|---|---|---|
| stack_v1 | 1891 | 51 | 0.7131 | 0.2590 | 0.0793 |
| stack_eq_v1 | 1891 | 51 | 0.7119 | 0.2584 | 0.0754 |
| elo_v1 | 1891 | 51 | 0.7147 | 0.2595 | 0.0801 |
| kalman_gs_v1 | 1891 | 51 | 0.7107 | 0.2579 | 0.0814 |
| ridge_v1 | 1891 | 51 | 0.7135 | 0.2591 | 0.0735 |
| xgb_v1 | 1891 | 51 | 0.7162 | 0.2604 | 0.0886 |

stack_v1 reliability at the nflverse line:

| bin_lo | bin_hi | n | mean_pred | observed | flag_small |
|---|---|---|---|---|---|
| 0.000 | 0.100 | 0 |  |  | True |
| 0.100 | 0.200 | 6 | 0.175 | 0.333 | True |
| 0.200 | 0.300 | 31 | 0.260 | 0.452 | False |
| 0.300 | 0.400 | 254 | 0.362 | 0.492 | False |
| 0.400 | 0.500 | 663 | 0.453 | 0.487 | False |
| 0.500 | 0.600 | 643 | 0.545 | 0.465 | False |
| 0.600 | 0.700 | 246 | 0.637 | 0.537 | False |
| 0.700 | 0.800 | 40 | 0.737 | 0.500 | False |
| 0.800 | 0.900 | 6 | 0.845 | 0.833 | True |
| 0.900 | 1.000 | 2 | 0.930 | 0.000 | True |

## Outputs

- `data/processed/step4_stack_predictions.parquet` (gitignored): 1942 stack_v1 rows, sha256 `0164302238b75b37e8e25e3d942d1007e1e395187c56be8aebd216361b5ecb4a`; passes `validate_predictions` (train_end_utc < prediction_time_utc).
- `config/models/stack_v1_weights.json`: forward weights + K1 (fit 2015-2024 OOF rows).
- `docs/results/step4_model_health.json`: MODEL_HEALTH_COLUMNS records, scopes pooled (2018-2024), clean (2022-2024) and one per fold, for stack_v1, stack_eq_v1 and each component. `cover_logloss` is the nflverse-line cover log loss above; `stack_weight_*` are the forward weights (null where not a component).

## Caveats

- Inputs inherit step 3's open items: leakage audit not signed; V5 caveat for 2018-2021 folds (ported constants chosen on 2016-2021 in the source repo).
- Season-level weights: within-season drift is not tracked. K1 for the stack is fit with in-sample weights on its training rows (slightly optimistic spread, 4 weights on >= 800 games).
- The components are highly correlated, so weights move between near-equivalent models from season to season (see the weight path); this is expected and costs little in MAE.
- Real CLV for the stack only exists for games captured going forward; nothing here uses sportsbook odds.
