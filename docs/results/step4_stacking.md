# Step 4 stacking results (stack_v2)

Generated 2026-11-26T00:00:15Z by `python scripts/run_stacking.py` (code commit `a894dee` plus uncommitted step-4 working tree; runtime 10s). Experiment `20261125-v2-epa-qb-01` (`docs/methodology/experiment_log.md`).

**Status: experimental.** Inputs are the step-3 OOF predictions (`data/processed/step3_oof_predictions.parquet`, test seasons 2015-2024). No 2025 (holdout `nfl_games_2025_v1`) or 2026 game is read, fit or scored.

## Design (fixed before scoring)

- **Components.** Margin stack over `elo_v1`, `kalman_gs_v1`, `ridge_v2`, `xgb_v2`. Total stack over the models that predict totals: `elo_v1`, `ridge_v2`, `xgb_v2` (`kalman_gs_v1` is margin-only). `constant_v1` and `bayes_od_v1` are reference rows; `bayes_od_v1` as a fifth candidate is a separately reported variant.
- **Walk-forward weights.** For test season S, weights are fit on OOF rows of seasons < S whose prediction time and label-available time are before S's first prediction time (`distributions.evaluate.training_rows`). One weight vector per season (no weekly refit inside S). At least 3 training seasons, so the first scored season is 2018 (trained on 2015-2017 OOF rows); scored seasons 2018-2024, 1942 games. Every single-model row below is on the same games.
- **Objective.** Primary: squared error on the simplex (weights >= 0, sum 1), no intercept, chosen before any test-season score was computed. Reported variant: MAE on the simplex (`stack_mae_v2`). Baseline stack: equal weights (`stack_eq_v2`).
- **Win probability.** Stacked margin -> K1 margin distribution (`fit_distribution(..., "k1")`) refit for each test season on the same training rows, with their stack margin computed from season S's weights (those weights are in-sample on the K1 training rows; nothing from S or later is used). `pred_home_win_prob = P(M > 0) / (1 - P(M = 0))`. `stack_eq_v2` uses the same procedure.
- **Clean seasons.** Step 3's clean seasons are 2015, 2022, 2023, 2024 (V5 caveat: ported constants chosen on 2016-2021). 2015 cannot be scored here (no 3 prior OOF seasons), so clean = 2022-2024.
- **Verdict rule.** "beats" only if the paired game-clustered bootstrap 95% CI (B = 2000) of the loss difference is below 0 AND the candidate wins a majority of folds. Negative difference = candidate better. "Best single" is the component with the lowest pooled value of that metric on these games (chosen on the test games, which favours the baseline).

## Headline

- stack_v2 margin MAE 10.166 vs elo_v1 10.221: -0.055 [-0.126, +0.016], folds won 5/7: **no difference shown**.
- stack_v2 margin vs best single (ridge_v2, 10.156): +0.010 [-0.029, +0.048], folds won 4/7: **no difference shown**.
- stack_v2 total MAE 10.635 vs elo_v1 10.993: -0.358 [-0.480, -0.235], folds won 7/7: **beats**; vs best single (xgb_v2, 10.640): -0.004 [-0.038, +0.028], folds won 5/7: **no difference shown**.
- stack_v2 home-win log loss 0.6338 vs elo_v1 0.6369: -0.003 [-0.008, +0.001], folds won 4/7: **no difference shown**.
- Equal-weight stack margin MAE 10.153: vs elo_v1 -0.067 [-0.120, -0.014], folds won 5/7: **beats**.
- Comparisons meeting the "beats" rule (pooled): stack_v2 vs elo_v1 on abs_total; stack_eq_v2 vs elo_v1 on abs_margin; stack_eq_v2 vs elo_v1 on abs_total. Gains over the best single model are small; read the CIs, not the point estimates.

## Pooled, scored seasons 2018-2024 (same games)

| model_id | n_games | mae_margin | rmse_margin | bias_margin | mae_total | n_win | logloss_win | brier_win |
|---|---|---|---|---|---|---|---|---|
| stack_v2 | 1942 | 10.166 | 13.116 | 0.354 | 10.635 | 1935 | 0.634 | 0.222 |
| stack_eq_v2 | 1942 | 10.153 | 13.115 | 0.209 | 10.680 | 1935 | 0.633 | 0.221 |
| elo_v1 | 1942 | 10.221 | 13.228 | -0.197 | 10.993 | 1935 | 0.637 | 0.223 |
| kalman_gs_v1 | 1942 | 10.192 | 13.152 | -0.078 |  | 1935 | 0.637 | 0.223 |
| ridge_v2 | 1942 | 10.156 | 13.115 | 0.457 | 10.652 | 1935 | 0.633 | 0.221 |
| xgb_v2 | 1942 | 10.197 | 13.146 | 0.655 | 10.640 | 1935 | 0.635 | 0.222 |
| bayes_od_v1 | 1942 | 10.205 | 13.166 | 0.009 | 10.833 | 1935 | 0.637 | 0.223 |
| constant_v1 | 1942 | 11.029 | 14.293 | 0.735 | 11.018 | 1935 | 0.690 | 0.249 |

### Clean seasons only (2022-2024)

| model_id | n_games | mae_margin | rmse_margin | bias_margin | mae_total | n_win | logloss_win | brier_win |
|---|---|---|---|---|---|---|---|---|
| stack_v2 | 854 | 9.828 | 12.768 | -0.349 | 10.312 | 852 | 0.632 | 0.221 |
| stack_eq_v2 | 854 | 9.820 | 12.790 | -0.483 | 10.347 | 852 | 0.632 | 0.221 |
| elo_v1 | 854 | 9.881 | 12.932 | -0.995 | 10.618 | 852 | 0.635 | 0.222 |
| kalman_gs_v1 | 854 | 9.918 | 12.876 | -0.818 |  | 852 | 0.637 | 0.223 |
| ridge_v2 | 854 | 9.808 | 12.745 | -0.103 | 10.301 | 852 | 0.631 | 0.220 |
| xgb_v2 | 854 | 9.825 | 12.775 | -0.016 | 10.352 | 852 | 0.634 | 0.222 |
| bayes_od_v1 | 854 | 9.946 | 12.893 | -0.300 | 10.519 | 852 | 0.639 | 0.224 |
| constant_v1 | 854 | 10.601 | 13.780 | -0.094 | 10.600 | 852 | 0.686 | 0.246 |

## Paired bootstrap and folds won

| scope | candidate | baseline | metric | n_games | diff | lo | hi | folds_won | verdict |
|---|---|---|---|---|---|---|---|---|---|
| pooled 2018-2024 | stack_v2 | elo_v1 | abs_margin | 1942 | -0.055 | -0.126 | 0.016 | 5/7 | no difference shown |
| pooled 2018-2024 | stack_v2 | ridge_v2 | abs_margin | 1942 | 0.010 | -0.029 | 0.048 | 4/7 | no difference shown |
| pooled 2018-2024 | stack_v2 | elo_v1 | abs_total | 1942 | -0.358 | -0.480 | -0.235 | 7/7 | beats |
| pooled 2018-2024 | stack_v2 | xgb_v2 | abs_total | 1942 | -0.004 | -0.038 | 0.028 | 5/7 | no difference shown |
| pooled 2018-2024 | stack_v2 | elo_v1 | logloss_win | 1935 | -0.003 | -0.008 | 0.001 | 4/7 | no difference shown |
| pooled 2018-2024 | stack_v2 | ridge_v2 | logloss_win | 1935 | 0.001 | -0.001 | 0.003 | 2/7 | no difference shown |
| pooled 2018-2024 | stack_eq_v2 | elo_v1 | abs_margin | 1942 | -0.067 | -0.120 | -0.014 | 5/7 | beats |
| pooled 2018-2024 | stack_eq_v2 | ridge_v2 | abs_margin | 1942 | -0.002 | -0.049 | 0.044 | 4/7 | no difference shown |
| pooled 2018-2024 | stack_eq_v2 | elo_v1 | abs_total | 1942 | -0.313 | -0.399 | -0.224 | 7/7 | beats |
| pooled 2018-2024 | stack_eq_v2 | xgb_v2 | abs_total | 1942 | 0.040 | -0.015 | 0.094 | 2/7 | no difference shown |
| pooled 2018-2024 | stack_eq_v2 | elo_v1 | logloss_win | 1935 | -0.004 | -0.007 | 0.000 | 5/7 | no difference shown |
| pooled 2018-2024 | stack_eq_v2 | ridge_v2 | logloss_win | 1935 | 0.000 | -0.002 | 0.003 | 3/7 | no difference shown |
| clean 2022-2024 | stack_v2 | elo_v1 | abs_margin | 854 | -0.053 | -0.154 | 0.049 | 3/3 | no difference shown |
| clean 2022-2024 | stack_v2 | ridge_v2 | abs_margin | 854 | 0.019 | -0.040 | 0.079 | 1/3 | no difference shown |
| clean 2022-2024 | stack_v2 | elo_v1 | abs_total | 854 | -0.306 | -0.483 | -0.127 | 3/3 | beats |
| clean 2022-2024 | stack_v2 | xgb_v2 | abs_total | 854 | -0.040 | -0.084 | -0.000 | 3/3 | beats |
| clean 2022-2024 | stack_v2 | elo_v1 | logloss_win | 852 | -0.003 | -0.010 | 0.005 | 2/3 | no difference shown |
| clean 2022-2024 | stack_v2 | ridge_v2 | logloss_win | 852 | 0.001 | -0.002 | 0.005 | 0/3 | no difference shown |
| clean 2022-2024 | stack_eq_v2 | elo_v1 | abs_margin | 854 | -0.061 | -0.137 | 0.016 | 3/3 | no difference shown |
| clean 2022-2024 | stack_eq_v2 | ridge_v2 | abs_margin | 854 | 0.012 | -0.061 | 0.081 | 1/3 | no difference shown |
| clean 2022-2024 | stack_eq_v2 | elo_v1 | abs_total | 854 | -0.270 | -0.393 | -0.147 | 3/3 | beats |
| clean 2022-2024 | stack_eq_v2 | xgb_v2 | abs_total | 854 | -0.005 | -0.077 | 0.067 | 2/3 | no difference shown |
| clean 2022-2024 | stack_eq_v2 | elo_v1 | logloss_win | 852 | -0.003 | -0.009 | 0.003 | 2/3 | no difference shown |
| clean 2022-2024 | stack_eq_v2 | ridge_v2 | logloss_win | 852 | 0.001 | -0.003 | 0.005 | 1/3 | no difference shown |

## Per fold

Margin MAE:

| fold_id | stack_v2 | stack_eq_v2 | elo_v1 | kalman_gs_v1 | ridge_v2 | xgb_v2 | bayes_od_v1 | constant_v1 |
|---|---|---|---|---|---|---|---|---|
| 2018 | 10.066 | 10.069 | 10.257 | 10.046 | 9.990 | 10.091 | 10.072 | 10.612 |
| 2019 | 10.506 | 10.424 | 10.406 | 10.375 | 10.535 | 10.590 | 10.412 | 11.618 |
| 2020 | 10.033 | 10.026 | 10.013 | 10.052 | 10.038 | 10.166 | 10.095 | 11.130 |
| 2021 | 11.078 | 11.096 | 11.225 | 11.109 | 11.107 | 11.068 | 11.015 | 12.051 |
| 2022 | 8.942 | 8.932 | 8.978 | 9.062 | 8.901 | 8.936 | 9.146 | 9.575 |
| 2023 | 10.471 | 10.479 | 10.564 | 10.528 | 10.424 | 10.526 | 10.466 | 11.081 |
| 2024 | 10.067 | 10.047 | 10.098 | 10.162 | 10.098 | 10.011 | 10.223 | 11.145 |

Total MAE:

| fold_id | stack_v2 | stack_eq_v2 | elo_v1 | ridge_v2 | xgb_v2 | bayes_od_v1 | constant_v1 |
|---|---|---|---|---|---|---|---|
| 2018 | 11.026 | 11.067 | 11.424 | 11.041 | 10.914 | 11.187 | 11.482 |
| 2019 | 10.898 | 10.924 | 11.215 | 10.940 | 10.905 | 11.007 | 11.164 |
| 2020 | 10.655 | 10.756 | 11.214 | 10.657 | 10.674 | 10.921 | 11.641 |
| 2021 | 10.973 | 11.013 | 11.295 | 11.065 | 10.963 | 11.196 | 11.109 |
| 2022 | 10.631 | 10.706 | 10.998 | 10.614 | 10.662 | 10.813 | 10.935 |
| 2023 | 10.447 | 10.462 | 10.716 | 10.417 | 10.500 | 10.617 | 10.708 |
| 2024 | 9.859 | 9.875 | 10.140 | 9.874 | 9.895 | 10.128 | 10.158 |

Home-win log loss:

| fold_id | stack_v2 | stack_eq_v2 | elo_v1 | kalman_gs_v1 | ridge_v2 | xgb_v2 | bayes_od_v1 | constant_v1 |
|---|---|---|---|---|---|---|---|---|
| 2018 | 0.6236 | 0.6242 | 0.6382 | 0.6244 | 0.6202 | 0.6251 | 0.6230 | 0.6756 |
| 2019 | 0.6363 | 0.6343 | 0.6349 | 0.6352 | 0.6388 | 0.6371 | 0.6368 | 0.6974 |
| 2020 | 0.6261 | 0.6247 | 0.6187 | 0.6294 | 0.6237 | 0.6339 | 0.6316 | 0.7032 |
| 2021 | 0.6530 | 0.6528 | 0.6607 | 0.6576 | 0.6541 | 0.6494 | 0.6497 | 0.6976 |
| 2022 | 0.6294 | 0.6295 | 0.6321 | 0.6362 | 0.6266 | 0.6294 | 0.6401 | 0.6841 |
| 2023 | 0.6511 | 0.6526 | 0.6631 | 0.6535 | 0.6500 | 0.6571 | 0.6498 | 0.6848 |
| 2024 | 0.6161 | 0.6146 | 0.6098 | 0.6217 | 0.6159 | 0.6150 | 0.6280 | 0.6894 |

## Weight path (walk-forward, one vector per test season)

Margin (stack_v2 = squared error, stack_mae_v2 = MAE variant):

| model_id | season | target | n_train | train_seasons | intercept | w_elo_v1 | w_kalman_gs_v1 | w_ridge_v2 | w_xgb_v2 |
|---|---|---|---|---|---|---|---|---|---|
| stack_v2 | 2018 | margin | 801 | 2015-2017 | 0.000 | 0.213 | 0.000 | 0.341 | 0.446 |
| stack_v2 | 2019 | margin | 1068 | 2015-2018 | 0.000 | 0.000 | 0.114 | 0.457 | 0.429 |
| stack_v2 | 2020 | margin | 1335 | 2015-2019 | 0.000 | 0.000 | 0.352 | 0.421 | 0.227 |
| stack_v2 | 2021 | margin | 1604 | 2015-2020 | 0.000 | 0.109 | 0.320 | 0.418 | 0.153 |
| stack_v2 | 2022 | margin | 1889 | 2015-2021 | 0.000 | 0.029 | 0.377 | 0.359 | 0.235 |
| stack_v2 | 2023 | margin | 2173 | 2015-2022 | 0.000 | 0.034 | 0.341 | 0.352 | 0.273 |
| stack_v2 | 2024 | margin | 2458 | 2015-2023 | 0.000 | 0.000 | 0.330 | 0.409 | 0.261 |
| stack_mae_v2 | 2018 | margin | 801 | 2015-2017 | 0.000 | 0.091 | 0.000 | 0.474 | 0.435 |
| stack_mae_v2 | 2019 | margin | 1068 | 2015-2018 | 0.000 | 0.000 | 0.000 | 0.649 | 0.351 |
| stack_mae_v2 | 2020 | margin | 1335 | 2015-2019 | 0.000 | 0.024 | 0.366 | 0.439 | 0.170 |
| stack_mae_v2 | 2021 | margin | 1604 | 2015-2020 | 0.000 | 0.159 | 0.371 | 0.470 | 0.000 |
| stack_mae_v2 | 2022 | margin | 1889 | 2015-2021 | 0.000 | 0.032 | 0.440 | 0.460 | 0.068 |
| stack_mae_v2 | 2023 | margin | 2173 | 2015-2022 | 0.000 | 0.090 | 0.269 | 0.536 | 0.105 |
| stack_mae_v2 | 2024 | margin | 2458 | 2015-2023 | 0.000 | 0.074 | 0.285 | 0.627 | 0.014 |

Total:

| model_id | season | target | n_train | train_seasons | intercept | w_elo_v1 | w_ridge_v2 | w_xgb_v2 |
|---|---|---|---|---|---|---|---|---|
| stack_v2 | 2018 | total | 801 | 2015-2017 | 0.000 | 0.178 | 0.471 | 0.350 |
| stack_v2 | 2019 | total | 1068 | 2015-2018 | 0.000 | 0.004 | 0.357 | 0.639 |
| stack_v2 | 2020 | total | 1335 | 2015-2019 | 0.000 | 0.022 | 0.405 | 0.573 |
| stack_v2 | 2021 | total | 1604 | 2015-2020 | 0.000 | 0.025 | 0.430 | 0.545 |
| stack_v2 | 2022 | total | 1889 | 2015-2021 | 0.000 | 0.031 | 0.394 | 0.576 |
| stack_v2 | 2023 | total | 2173 | 2015-2022 | 0.000 | 0.001 | 0.431 | 0.568 |
| stack_v2 | 2024 | total | 2458 | 2015-2023 | 0.000 | 0.000 | 0.540 | 0.460 |
| stack_mae_v2 | 2018 | total | 801 | 2015-2017 | 0.000 | 0.017 | 0.383 | 0.600 |
| stack_mae_v2 | 2019 | total | 1068 | 2015-2018 | 0.000 | 0.000 | 0.109 | 0.891 |
| stack_mae_v2 | 2020 | total | 1335 | 2015-2019 | 0.000 | 0.000 | 0.135 | 0.865 |
| stack_mae_v2 | 2021 | total | 1604 | 2015-2020 | 0.000 | 0.000 | 0.208 | 0.792 |
| stack_mae_v2 | 2022 | total | 1889 | 2015-2021 | 0.000 | 0.000 | 0.202 | 0.798 |
| stack_mae_v2 | 2023 | total | 2173 | 2015-2022 | 0.000 | 0.000 | 0.295 | 0.705 |
| stack_mae_v2 | 2024 | total | 2458 | 2015-2023 | 0.000 | 0.000 | 0.395 | 0.605 |

Forward weights (fit on all OOF seasons 2015-2024; `config/models/stack_v2_weights.json`, used by `StackModel.from_weights_file` and the dashboard's `stack_weight_*` fields):

| target | w_elo_v1 | w_kalman_gs_v1 | w_ridge_v2 | w_xgb_v2 | n_train |
|---|---|---|---|---|---|
| margin | 0.000 | 0.221 | 0.488 | 0.291 | 2743 |
| total | 0.000 |  | 0.517 | 0.483 | 2743 |

Forward K1 for the win-probability mapping: sigma 12.999, w_3 2.690, w_7 1.745.


K1 parameters of the stack per test season:

| season | n_train | train_seasons | k1_sigma | k1_w3 | k1_w7 |
|---|---|---|---|---|---|
| 2018 | 801 | 2015-2017 | 12.729 | 2.689 | 1.849 |
| 2019 | 1068 | 2015-2018 | 12.867 | 2.798 | 1.864 |
| 2020 | 1335 | 2015-2019 | 12.972 | 2.631 | 1.995 |
| 2021 | 1604 | 2015-2020 | 12.961 | 2.617 | 1.909 |
| 2022 | 1889 | 2015-2021 | 13.094 | 2.676 | 1.839 |
| 2023 | 2173 | 2015-2022 | 12.935 | 2.699 | 1.795 |
| 2024 | 2458 | 2015-2023 | 13.018 | 2.707 | 1.780 |

## Variants (not primary; reported separately)

- `stack_mae_v2`: MAE objective (reported variant)
- `stack_b5_v2`: squared error, bayes_od_v1 added as a fifth candidate (margin and total)

| model_id | n_games | mae_margin | rmse_margin | bias_margin | mae_total | n_win | logloss_win | brier_win |
|---|---|---|---|---|---|---|---|---|
| stack_mae_v2 | 1942 | 10.167 | 13.116 | 0.356 | 10.631 | 1935 | 0.634 | 0.222 |
| stack_b5_v2 | 1942 | 10.167 | 13.120 | 0.347 | 10.639 | 1935 | 0.634 | 0.222 |


| scope | candidate | baseline | metric | n_games | diff | lo | hi | folds_won | verdict |
|---|---|---|---|---|---|---|---|---|---|
| pooled 2018-2024 | stack_mae_v2 | elo_v1 | abs_margin | 1942 | -0.053 | -0.125 | 0.018 | 5/7 | no difference shown |
| pooled 2018-2024 | stack_mae_v2 | elo_v1 | abs_total | 1942 | -0.362 | -0.488 | -0.230 | 7/7 | beats |
| pooled 2018-2024 | stack_mae_v2 | elo_v1 | logloss_win | 1935 | -0.003 | -0.008 | 0.001 | 4/7 | no difference shown |
| pooled 2018-2024 | stack_b5_v2 | elo_v1 | abs_margin | 1942 | -0.053 | -0.123 | 0.016 | 5/7 | no difference shown |
| pooled 2018-2024 | stack_b5_v2 | elo_v1 | abs_total | 1942 | -0.354 | -0.481 | -0.222 | 7/7 | beats |
| pooled 2018-2024 | stack_b5_v2 | elo_v1 | logloss_win | 1935 | -0.003 | -0.007 | 0.002 | 4/7 | no difference shown |

`stack_b5_v2` weight path:

| model_id | season | target | n_train | train_seasons | intercept | w_elo_v1 | w_kalman_gs_v1 | w_ridge_v2 | w_xgb_v2 | w_bayes_od_v1 |
|---|---|---|---|---|---|---|---|---|---|---|
| stack_b5_v2 | 2018 | margin | 801 | 2015-2017 | 0.000 | 0.213 | 0.000 | 0.341 | 0.446 | 0.000 |
| stack_b5_v2 | 2018 | total | 801 | 2015-2017 | 0.000 | 0.160 |  | 0.296 | 0.299 | 0.244 |
| stack_b5_v2 | 2019 | margin | 1068 | 2015-2018 | 0.000 | 0.000 | 0.114 | 0.457 | 0.429 | 0.000 |
| stack_b5_v2 | 2019 | total | 1068 | 2015-2018 | 0.000 | 0.008 |  | 0.176 | 0.566 | 0.250 |
| stack_b5_v2 | 2020 | margin | 1335 | 2015-2019 | 0.000 | 0.000 | 0.352 | 0.421 | 0.227 | 0.000 |
| stack_b5_v2 | 2020 | total | 1335 | 2015-2019 | 0.000 | 0.026 |  | 0.179 | 0.469 | 0.326 |
| stack_b5_v2 | 2021 | margin | 1604 | 2015-2020 | 0.000 | 0.144 | 0.131 | 0.403 | 0.115 | 0.207 |
| stack_b5_v2 | 2021 | total | 1604 | 2015-2020 | 0.000 | 0.021 |  | 0.203 | 0.480 | 0.296 |
| stack_b5_v2 | 2022 | margin | 1889 | 2015-2021 | 0.000 | 0.069 | 0.150 | 0.338 | 0.197 | 0.247 |
| stack_b5_v2 | 2022 | total | 1889 | 2015-2021 | 0.000 | 0.023 |  | 0.205 | 0.516 | 0.256 |
| stack_b5_v2 | 2023 | margin | 2173 | 2015-2022 | 0.000 | 0.054 | 0.227 | 0.342 | 0.256 | 0.122 |
| stack_b5_v2 | 2023 | total | 2173 | 2015-2022 | 0.000 | 0.002 |  | 0.252 | 0.510 | 0.236 |
| stack_b5_v2 | 2024 | margin | 2458 | 2015-2023 | 0.000 | 0.000 | 0.142 | 0.396 | 0.234 | 0.228 |
| stack_b5_v2 | 2024 | total | 2458 | 2015-2023 | 0.000 | 0.000 |  | 0.371 | 0.409 | 0.220 |

## Win probability: stack K1 mapping vs the components' own probabilities

Non-tie games; ECE with 10 equal-width bins. The last row averages the components' own P(home win) with stack_v2's margin weights (no distribution step).

| model | n_win | logloss | brier | ece_10bin |
|---|---|---|---|---|
| stack_v2 | 1935 | 0.6338 | 0.2216 | 0.0258 |
| stack_eq_v2 | 1935 | 0.6334 | 0.2214 | 0.0306 |
| elo_v1 | 1935 | 0.6369 | 0.2228 | 0.0161 |
| kalman_gs_v1 | 1935 | 0.6371 | 0.2229 | 0.0283 |
| ridge_v2 | 1935 | 0.6329 | 0.2212 | 0.0306 |
| xgb_v2 | 1935 | 0.6354 | 0.2224 | 0.0388 |
| stack_v2 margin weights on component P(win) | 1935 | 0.6342 | 0.2217 | 0.0193 |

## Calibration of stack_v2

P(home win) ECE (10 bins) 0.0258. Home win | no tie from the K1 pmfs: log loss 0.6338, Brier 0.2216, 7 ties excluded.


Reliability (P(home win), equal-width bins; `flag_small` = fewer than 30 games):

| bin_lo | bin_hi | n | mean_pred | observed | flag_small |
|---|---|---|---|---|---|
| 0.000 | 0.100 | 0 |  |  | True |
| 0.100 | 0.200 | 17 | 0.171 | 0.294 | True |
| 0.200 | 0.300 | 88 | 0.261 | 0.284 | False |
| 0.300 | 0.400 | 204 | 0.356 | 0.279 | False |
| 0.400 | 0.500 | 364 | 0.453 | 0.426 | False |
| 0.500 | 0.600 | 467 | 0.552 | 0.531 | False |
| 0.600 | 0.700 | 453 | 0.646 | 0.662 | False |
| 0.700 | 0.800 | 253 | 0.745 | 0.743 | False |
| 0.800 | 0.900 | 83 | 0.840 | 0.867 | False |
| 0.900 | 1.000 | 6 | 0.912 | 0.833 | True |

K1 margin distribution: log score, randomized PIT and central-interval coverage by season (well calibrated: pit_mean 0.5, pit_var 1/12 = 0.083, covN close to massN):

| season | log_score | n | pit_mean | pit_var | pit_ks | pit_ks_p | cov50 | mass50 | cov80 | mass80 | cov95 | mass95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018 | 3.874 | 267 | 0.497 | 0.081 | 0.035 | 0.895 | 0.577 | 0.531 | 0.820 | 0.814 | 0.936 | 0.955 |
| 2019 | 3.914 | 267 | 0.448 | 0.085 | 0.092 | 0.020 | 0.509 | 0.532 | 0.790 | 0.815 | 0.955 | 0.955 |
| 2020 | 3.861 | 269 | 0.474 | 0.082 | 0.060 | 0.280 | 0.506 | 0.533 | 0.814 | 0.813 | 0.955 | 0.955 |
| 2021 | 3.970 | 285 | 0.502 | 0.090 | 0.044 | 0.629 | 0.498 | 0.530 | 0.793 | 0.814 | 0.930 | 0.955 |
| 2022 | 3.759 | 284 | 0.501 | 0.072 | 0.054 | 0.357 | 0.577 | 0.531 | 0.852 | 0.814 | 0.975 | 0.954 |
| 2023 | 3.863 | 285 | 0.514 | 0.085 | 0.044 | 0.626 | 0.537 | 0.532 | 0.793 | 0.814 | 0.940 | 0.955 |
| 2024 | 3.867 | 285 | 0.503 | 0.081 | 0.029 | 0.962 | 0.561 | 0.530 | 0.811 | 0.813 | 0.965 | 0.955 |
| all | 3.872 | 1942 | 0.491 | 0.083 | 0.026 | 0.153 | 0.538 | 0.531 | 0.811 | 0.814 | 0.951 | 0.955 |

Interval coverage, pooled:

| level | n | coverage | predicted_mass | mean_width |
|---|---|---|---|---|
| 0.500 | 1942 | 0.538 | 0.531 | 17.444 |
| 0.800 | 1942 | 0.811 | 0.814 | 33.947 |
| 0.950 | 1942 | 0.951 | 0.955 | 53.347 |

## Cover calibration at the nflverse spread line (nflverse line, capture time unknown; context only, never a feature)

P(home cover | no push) from each model's own walk-forward K1 distribution at the nflverse line, pushes excluded, 10 equal-width bins. The line is close-like, so the observed cover rate is near 50% in every bin; a log loss above ln 2 = 0.693 means worse than a constant 50% forecast.

| model | n_no_push | n_push_excluded | cover_logloss | brier | ece_10bin |
|---|---|---|---|---|---|
| stack_v2 | 1891 | 51 | 0.7102 | 0.2577 | 0.0732 |
| stack_eq_v2 | 1891 | 51 | 0.7098 | 0.2574 | 0.0750 |
| elo_v1 | 1891 | 51 | 0.7147 | 0.2595 | 0.0801 |
| kalman_gs_v1 | 1891 | 51 | 0.7107 | 0.2579 | 0.0814 |
| ridge_v2 | 1891 | 51 | 0.7097 | 0.2574 | 0.0739 |
| xgb_v2 | 1891 | 51 | 0.7128 | 0.2589 | 0.0808 |

stack_v2 reliability at the nflverse line:

| bin_lo | bin_hi | n | mean_pred | observed | flag_small |
|---|---|---|---|---|---|
| 0.000 | 0.100 | 0 |  |  | True |
| 0.100 | 0.200 | 5 | 0.179 | 0.400 | True |
| 0.200 | 0.300 | 26 | 0.257 | 0.500 | True |
| 0.300 | 0.400 | 235 | 0.364 | 0.468 | False |
| 0.400 | 0.500 | 669 | 0.455 | 0.489 | False |
| 0.500 | 0.600 | 666 | 0.544 | 0.473 | False |
| 0.600 | 0.700 | 245 | 0.635 | 0.522 | False |
| 0.700 | 0.800 | 38 | 0.734 | 0.579 | False |
| 0.800 | 0.900 | 6 | 0.852 | 0.500 | True |
| 0.900 | 1.000 | 1 | 0.957 | 0.000 | True |

## Outputs

- `data/processed/step4_stack_predictions.parquet` (gitignored): 1942 stack_v2 rows, sha256 `dab2c6e6f1218c4aa462d6594a4dd560154655abf5d537d349dbbc2e4568ce92`; passes `validate_predictions` (train_end_utc < prediction_time_utc).
- `config/models/stack_v2_weights.json`: forward weights + K1 (fit 2015-2024 OOF rows).
- `docs/results/step4_model_health.json`: MODEL_HEALTH_COLUMNS records, scopes pooled (2018-2024), clean (2022-2024) and one per fold, for stack_v2, stack_eq_v2 and each component. `cover_logloss` is the nflverse-line cover log loss above; `stack_weight_*` are the forward weights (null where not a component).

## Caveats

- Inputs inherit step 3's open items: leakage audit not signed; V5 caveat for 2018-2021 folds (ported constants chosen on 2016-2021 in the source repo).
- Season-level weights: within-season drift is not tracked. K1 for the stack is fit with in-sample weights on its training rows (slightly optimistic spread, 4 weights on >= 800 games).
- The components are highly correlated, so weights move between near-equivalent models from season to season (see the weight path); this is expected and costs little in MAE.
- Real CLV for the stack only exists for games captured going forward; nothing here uses sportsbook odds.
