# Step 3 walk-forward results

Generated 2026-11-19 04:45 UTC by `python scripts/run_walkforward.py`
(code commit `ee6ca2d` plus uncommitted step-3 working tree). Experiment `20261119-step3-game-models-01`
(`docs/methodology/experiment_log.md`). Runtime 179s (fit+predict: constant_v1 1s, elo_v1 9s, kalman_gs_v1 10s, bayes_od_v1 8s, ridge_v1 65s, xgb_v1 79s).

**Status: exploratory until the leakage audit is signed (`leakage_audit.md`) and the feature registry entries
are reviewed.** No 2025 (holdout `nfl_games_2025_v1`) or 2026 game was scored or used.

## Setup

- Data: nflverse `games.csv` cache `data/raw/nflverse_games.csv` sha256 `be2cb49af61ed79c...`,
  seasons 1999-2024 (2025 excluded by `exclude_holdout`; 2026 dropped).
- Fold table `season_wf_weekly_v1`: outer test seasons 2015-2024 (>= 3 training seasons; history from 1999);
  weekly rolling origin inside each test season (postseason weeks are their own origins). Before week w every
  model is refit on all games with `label_available_at_utc` < week w's first T. V2/V3/holdout asserted per origin.
- T = kickoff - 24h. Label available = next Wednesday 12:00 US/Eastern after kickoff. Ties count in MAE and are
  dropped from log loss / Brier.
- Models: `constant_v1` (rung 0), `elo_v1` (rung 1c, ported Elo + NB score model, frozen constants),
  `kalman_gs_v1` (ported Glickman-Stern Kalman margin rating, frozen constants; margin only), `bayes_od_v1`
  (our offense/defense Kalman extension, a-priori constants; margin + total), `ridge_v1` (ours; alpha by inner
  season split), `xgb_v1` (ported a-priori XGBoost settings, 100 trees; QB and EPA features not available).
  Win probabilities: Elo logistic; Kalman/Bayes Normal; Ridge/XGB Normal with inner-split residual SD.
- **V5 caveat:** ported constants were chosen on 2016-2020 (Elo, Kalman) and 2016-2021 (XGB tree count) in the
  source repo. For those models the 2016-2021 folds are not nested. "Clean" columns use seasons
  2015, 2022, 2023, 2024 only.
- Bootstrap: paired, game-clustered (one row per game per model), B = 2000, 95% percentile CI; negative = better
  than Elo.

## Pooled, test seasons 2015-2024

| model_id | n_games | mae_margin | rmse_margin | bias_margin | mae_total | rmse_total | n_win | logloss_win | brier_win |
|---|---|---|---|---|---|---|---|---|---|
| bayes_od_v1 | 2743 | 10.161 | 13.102 | -0.049 | 10.844 | 13.696 | 2734 | 0.636 | 0.223 |
| constant_v1 | 2743 | 10.963 | 14.177 | 0.586 | 10.982 | 14.014 | 2734 | 0.689 | 0.248 |
| elo_v1 | 2743 | 10.176 | 13.145 | -0.465 | 10.982 | 13.881 | 2734 | 0.635 | 0.222 |
| kalman_gs_v1 | 2743 | 10.152 | 13.091 | -0.345 |  |  | 2734 | 0.636 | 0.222 |
| ridge_v1 | 2743 | 10.144 | 13.109 | 0.419 | 10.792 | 13.650 | 2734 | 0.634 | 0.222 |
| xgb_v1 | 2743 | 10.190 | 13.103 | 0.523 | 10.773 | 13.660 | 2734 | 0.635 | 0.222 |
| BENCHMARK: nflverse line, capture time unknown | 2743 | 9.821 | 12.769 | 0.012 | 10.444 | 13.224 | 2733 | 0.613 | 0.212 |

Benchmark row: nflverse line, capture time unknown; 2743/2743 games have spread_line; win probability from de-vigged nflverse moneylines where present
(n_win). It is close-like, reported for context only, never a feature, and not part of the prediction file.

### Clean seasons only (2015, 2022, 2023, 2024)

| model_id | n_games | mae_margin | rmse_margin | bias_margin | mae_total | rmse_total | n_win | logloss_win | brier_win |
|---|---|---|---|---|---|---|---|---|---|
| bayes_od_v1 | 1121 | 10.033 | 12.940 | -0.134 | 10.577 | 13.430 | 1119 | 0.640 | 0.225 |
| constant_v1 | 1121 | 10.676 | 13.884 | 0.174 | 10.624 | 13.616 | 1119 | 0.687 | 0.247 |
| elo_v1 | 1121 | 9.946 | 12.940 | -0.853 | 10.677 | 13.583 | 1119 | 0.637 | 0.223 |
| kalman_gs_v1 | 1121 | 10.007 | 12.923 | -0.704 |  |  | 1119 | 0.639 | 0.224 |
| ridge_v1 | 1121 | 9.949 | 12.905 | 0.134 | 10.520 | 13.337 | 1119 | 0.638 | 0.223 |
| xgb_v1 | 1121 | 9.998 | 12.896 | 0.323 | 10.484 | 13.330 | 1119 | 0.639 | 0.224 |

### Slices

Regular season:

| model_id | n_games | mae_margin | rmse_margin | bias_margin | mae_total | rmse_total | n_win | logloss_win | brier_win |
|---|---|---|---|---|---|---|---|---|---|
| bayes_od_v1 | 2623 | 10.172 | 13.124 | 0.054 | 10.832 | 13.669 | 2614 | 0.636 | 0.223 |
| constant_v1 | 2623 | 10.985 | 14.225 | 0.765 | 10.961 | 13.964 | 2614 | 0.690 | 0.248 |
| elo_v1 | 2623 | 10.181 | 13.164 | -0.352 | 10.969 | 13.838 | 2614 | 0.635 | 0.222 |
| kalman_gs_v1 | 2623 | 10.158 | 13.106 | -0.245 |  |  | 2614 | 0.636 | 0.222 |
| ridge_v1 | 2623 | 10.153 | 13.130 | 0.492 | 10.773 | 13.613 | 2614 | 0.635 | 0.222 |
| xgb_v1 | 2623 | 10.205 | 13.130 | 0.610 | 10.753 | 13.613 | 2614 | 0.636 | 0.223 |

Postseason:

| model_id | n_games | mae_margin | rmse_margin | bias_margin | mae_total | rmse_total | n_win | logloss_win | brier_win |
|---|---|---|---|---|---|---|---|---|---|
| bayes_od_v1 | 120 | 9.919 | 12.618 | -2.303 | 11.114 | 14.256 | 120 | 0.633 | 0.221 |
| constant_v1 | 120 | 10.493 | 13.088 | -3.331 | 11.449 | 15.048 | 120 | 0.662 | 0.234 |
| elo_v1 | 120 | 10.070 | 12.735 | -2.930 | 11.277 | 14.800 | 120 | 0.631 | 0.221 |
| kalman_gs_v1 | 120 | 10.008 | 12.762 | -2.532 |  |  | 120 | 0.637 | 0.224 |
| ridge_v1 | 120 | 9.937 | 12.633 | -1.185 | 11.200 | 14.442 | 120 | 0.629 | 0.219 |
| xgb_v1 | 120 | 9.856 | 12.486 | -1.389 | 11.205 | 14.638 | 120 | 0.631 | 0.220 |

2020 (limited crowds):

| model_id | n_games | mae_margin | rmse_margin | bias_margin | mae_total | rmse_total | n_win | logloss_win | brier_win |
|---|---|---|---|---|---|---|---|---|---|
| bayes_od_v1 | 269 | 10.095 | 12.929 | 0.506 | 10.921 | 13.817 | 268 | 0.632 | 0.220 |
| constant_v1 | 269 | 11.130 | 14.265 | 2.251 | 11.641 | 14.845 | 268 | 0.703 | 0.255 |
| elo_v1 | 269 | 10.013 | 12.842 | 1.381 | 11.214 | 13.937 | 268 | 0.619 | 0.213 |
| kalman_gs_v1 | 269 | 10.052 | 12.933 | 1.388 |  |  | 268 | 0.629 | 0.219 |
| ridge_v1 | 269 | 10.038 | 12.914 | 1.365 | 10.884 | 13.840 | 268 | 0.623 | 0.216 |
| xgb_v1 | 269 | 10.284 | 13.160 | 1.483 | 11.006 | 14.029 | 268 | 0.640 | 0.224 |

## Per fold

Origins and training sizes:

| fold_id | origins | min_train | max_train |
|---|---|---|---|
| 2015 | 21 | 4248 | 4514 |
| 2016 | 21 | 4515 | 4781 |
| 2017 | 21 | 4782 | 5048 |
| 2018 | 21 | 5049 | 5315 |
| 2019 | 21 | 5316 | 5582 |
| 2020 | 21 | 5583 | 5851 |
| 2021 | 22 | 5852 | 6136 |
| 2022 | 22 | 6137 | 6420 |
| 2023 | 22 | 6421 | 6705 |
| 2024 | 22 | 6706 | 6990 |

Margin MAE:

| fold_id | n_games | bayes_od_v1 | constant_v1 | elo_v1 | kalman_gs_v1 | ridge_v1 | xgb_v1 |
|---|---|---|---|---|---|---|---|
| 2015 | 267 | 10.312 | 10.913 | 10.153 | 10.292 | 10.195 | 10.317 |
| 2016 | 267 | 9.395 | 10.103 | 9.409 | 9.381 | 9.331 | 9.371 |
| 2017 | 267 | 10.453 | 11.401 | 10.645 | 10.488 | 10.510 | 10.336 |
| 2018 | 267 | 10.072 | 10.612 | 10.257 | 10.046 | 10.044 | 10.121 |
| 2019 | 267 | 10.412 | 11.618 | 10.406 | 10.375 | 10.548 | 10.614 |
| 2020 | 269 | 10.095 | 11.130 | 10.013 | 10.052 | 10.038 | 10.284 |
| 2021 | 285 | 11.015 | 12.051 | 11.225 | 11.109 | 11.142 | 11.152 |
| 2022 | 284 | 9.146 | 9.575 | 8.978 | 9.062 | 9.003 | 9.034 |
| 2023 | 285 | 10.466 | 11.081 | 10.564 | 10.528 | 10.455 | 10.557 |
| 2024 | 285 | 10.223 | 11.145 | 10.098 | 10.162 | 10.155 | 10.100 |

Total MAE:

| fold_id | bayes_od_v1 | constant_v1 | elo_v1 | kalman_gs_v1 | ridge_v1 | xgb_v1 |
|---|---|---|---|---|---|---|
| 2015 | 10.763 | 10.699 | 10.866 |  | 10.927 | 10.773 |
| 2016 | 10.103 | 10.634 | 10.520 |  | 10.334 | 10.232 |
| 2017 | 11.748 | 11.357 | 11.484 |  | 11.323 | 11.422 |
| 2018 | 11.187 | 11.482 | 11.424 |  | 11.166 | 11.084 |
| 2019 | 11.007 | 11.164 | 11.215 |  | 10.982 | 11.041 |
| 2020 | 10.921 | 11.641 | 11.214 |  | 10.884 | 11.006 |
| 2021 | 11.196 | 11.109 | 11.295 |  | 11.176 | 11.044 |
| 2022 | 10.813 | 10.935 | 10.998 |  | 10.693 | 10.661 |
| 2023 | 10.617 | 10.708 | 10.716 |  | 10.499 | 10.492 |
| 2024 | 10.128 | 10.158 | 10.140 |  | 9.987 | 10.030 |

Home-win log loss:

| fold_id | bayes_od_v1 | constant_v1 | elo_v1 | kalman_gs_v1 | ridge_v1 | xgb_v1 |
|---|---|---|---|---|---|---|
| 2015 | 0.644 | 0.691 | 0.644 | 0.644 | 0.643 | 0.649 |
| 2016 | 0.635 | 0.679 | 0.626 | 0.635 | 0.627 | 0.628 |
| 2017 | 0.620 | 0.684 | 0.623 | 0.620 | 0.620 | 0.610 |
| 2018 | 0.623 | 0.676 | 0.638 | 0.624 | 0.621 | 0.627 |
| 2019 | 0.637 | 0.697 | 0.635 | 0.635 | 0.642 | 0.639 |
| 2020 | 0.632 | 0.703 | 0.619 | 0.629 | 0.623 | 0.640 |
| 2021 | 0.650 | 0.698 | 0.661 | 0.658 | 0.659 | 0.652 |
| 2022 | 0.640 | 0.684 | 0.632 | 0.636 | 0.634 | 0.633 |
| 2023 | 0.650 | 0.685 | 0.663 | 0.654 | 0.651 | 0.659 |
| 2024 | 0.628 | 0.689 | 0.610 | 0.622 | 0.624 | 0.617 |

## Paired difference vs Elo (candidate minus `elo_v1`)

| candidate | metric | n_games | diff | lo | hi | folds_better | clean_diff | clean_lo | clean_hi | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| constant_v1 | abs_margin | 2743 | 0.787 | 0.606 | 0.968 | 0/10 | 0.730 | 0.459 | 0.998 | worse than Elo (CI excludes 0) |
| constant_v1 | abs_total | 2743 | 0.000 | -0.100 | 0.098 | 6/10 | -0.053 | -0.144 | 0.038 | inconclusive |
| constant_v1 | logloss_win | 2734 | 0.053 | 0.040 | 0.067 | 0/10 | 0.050 | 0.030 | 0.069 | worse than Elo (CI excludes 0) |
| kalman_gs_v1 | abs_margin | 2743 | -0.025 | -0.089 | 0.036 | 6/10 | 0.061 | -0.033 | 0.156 | inconclusive |
| kalman_gs_v1 | logloss_win | 2734 | 0.001 | -0.004 | 0.005 | 4/10 | 0.002 | -0.005 | 0.008 | inconclusive |
| bayes_od_v1 | abs_margin | 2743 | -0.015 | -0.096 | 0.065 | 5/10 | 0.087 | -0.034 | 0.201 | inconclusive |
| bayes_od_v1 | abs_total | 2743 | -0.138 | -0.280 | 0.013 | 9/10 | -0.100 | -0.332 | 0.124 | inconclusive |
| bayes_od_v1 | logloss_win | 2734 | 0.001 | -0.005 | 0.006 | 5/10 | 0.003 | -0.005 | 0.011 | inconclusive |
| ridge_v1 | abs_margin | 2743 | -0.033 | -0.095 | 0.029 | 5/10 | 0.003 | -0.095 | 0.101 | inconclusive |
| ridge_v1 | abs_total | 2743 | -0.190 | -0.285 | -0.096 | 9/10 | -0.157 | -0.315 | -0.015 | better than Elo (CI excludes 0) |
| ridge_v1 | logloss_win | 2734 | -0.001 | -0.005 | 0.003 | 5/10 | 0.001 | -0.006 | 0.007 | inconclusive |
| xgb_v1 | abs_margin | 2743 | 0.013 | -0.059 | 0.089 | 5/10 | 0.052 | -0.066 | 0.163 | inconclusive |
| xgb_v1 | abs_total | 2743 | -0.209 | -0.312 | -0.104 | 10/10 | -0.192 | -0.348 | -0.048 | better than Elo (CI excludes 0) |
| xgb_v1 | logloss_win | 2734 | 0.000 | -0.004 | 0.005 | 4/10 | 0.002 | -0.005 | 0.009 | inconclusive |

## Which models beat Elo?

Beats Elo (pooled CI below 0 and majority of folds): `ridge_v1` on abs_total (-0.190 [-0.285, -0.096], 9/10 folds; clean seasons -0.157 [-0.315, -0.015]); `xgb_v1` on abs_total (-0.209 [-0.312, -0.104], 10/10 folds; clean seasons -0.192 [-0.348, -0.048]). No model beats Elo on margin MAE or home-win log loss: every margin difference is within about 0.1 point and its CI covers 0, so for the margin the ported Elo remains the model to beat (P6: keep the simplest).

- `kalman_gs_v1` margin MAE minus Elo: -0.025 [-0.089, +0.036], lower on 6/10 folds; clean seasons +0.061 [-0.033, +0.156].
- `bayes_od_v1` margin MAE minus Elo: -0.015 [-0.096, +0.065], lower on 5/10 folds; clean seasons +0.087 [-0.034, +0.201].
- `ridge_v1` margin MAE minus Elo: -0.033 [-0.095, +0.029], lower on 5/10 folds; clean seasons +0.003 [-0.095, +0.101].
- `xgb_v1` margin MAE minus Elo: +0.013 [-0.059, +0.089], lower on 5/10 folds; clean seasons +0.052 [-0.066, +0.163].
- `constant_v1` margin MAE minus Elo: +0.787 [+0.606, +0.968], lower on 0/10 folds; clean seasons +0.730 [+0.459, +0.998].

Elo's total is its NB score model's mu_home + mu_away (league level times a cosh of the Elo gap), so the
total comparisons test team-specific scoring information that Elo does not carry.

Reading: a model "beats Elo" only if the pooled CI is below 0 and it wins a majority of folds
(`predictive_modeling.md` Section 2). See the table above for the verdict per metric. Every model is far from
the nflverse line benchmark (which has unknown capture time and is close-like), as expected for models without
market, QB or injury inputs.

## Reproduction context vs the source repo (descriptive)

Original figures: wbblanck/nfl-betting-model `data/backtests/elo_confirm.json` (Elo, close; `docs/elo_report.md`)
and `data/backtests/batch4.json` (XGBoost; `docs/batch4_report.md`). Their headline: plain Elo margin MAE 9.964 on
1023 games (2022-2025 REG weeks 2-18 with closing spread and both prices) vs closing spread 9.537. We cannot score
2025, so we compare 2022-2024 on the same sample definition. Kalman: the source reports Kalman MAE only for
2005-2015 (`batch1.json`, 10.907 on 2,640 games), so 2022-2024 is "not reported".

| season | n_ours | n_orig | elo_ours | elo_orig | kalman_ours | kalman_orig | xgb_ours | xgb_orig | close_ours_nflverse | close_orig |
|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | 255 | 255 | 8.986 | 8.986 | 9.070 | not reported | 9.015 | 9.056 | 8.718 | 8.718 |
| 2023 | 256 | 256 | 10.463 | 10.463 | 10.419 | not reported | 10.427 | 10.657 | 9.818 | 9.818 |
| 2024 | 256 | 256 | 10.076 | 10.071 | 10.121 | not reported | 10.107 | 9.998 | 9.648 | 9.648 |
| 2022-2024 pooled | 767 | 767 | 9.843 | 9.841 | 9.871 | not reported | 9.851 | 9.905 |  | 9.396 |

Our 2022 and 2023 Elo MAE and the nflverse close match the source to three decimals on identical game
counts; 2024 differs by 0.005, consistent with the label-timing rule (the week-16 Monday game's label arrives
Wednesday 2024-12-25 12:00 ET, after the Christmas-Day games' T, so here it enters the ratings one week later). XGBoost differs by design (rows below).

What differs from the source runs:

| item | source | here |
|---|---|---|
| seasons scored | 2022-2025 (confirm), REG weeks 2-18 | 2015-2024 all weeks incl. postseason; table above restricted to 2022-2024 REG weeks 2-18 with closing spread + prices |
| Elo / Kalman constants | fit 2016-2020, frozen | same frozen constants (ported) |
| Elo / Kalman update timing | week-batched; a week's forecasts use all earlier weeks | same, plus the label rule: a game is used only once its label (next Wed 12:00 ET) is before T, which purges e.g. Wednesday/Christmas games for one week |
| Elo scoring level | prev REG season + current-season REG weeks before w | same formula, as-of label availability |
| XGBoost features | Elo gap/fair, QB deltas (expected starter), Kalman fair/var, 8-game EPA/play off/def, level, rest, neutral, div, dome (dome or closed), week | QB deltas dropped; EPA replaced by 8-game points for/against; dome = roof `dome` only |
| XGBoost training | fit once on 2016-2021 REG weeks 2-18, frozen | refit every week on all labelled games since 2000 (expanding window), total model added |
| close benchmark | nflverse spread_line | same column, labelled capture time unknown |

## Open issues

- QB effects skipped: nflverse `home_qb_id`/`away_qb_id` are post-game starters. Needs pre-game depth chart /
  injury report data with publication times.
- EPA features (source XGBoost) need nflverse play-by-play; not ingested in this step.
- Rest days come from the final schedule (REVIEW REQUIRED for flexed/rescheduled games); weather not used.
- 1999 kickoff times are imputed (13:00 ET); 1999 is training history only.
- Home/away score independence (P5) in `bayes_od_v1` and the Elo NB score model is untested here; the
  distributions workstream tests it on these OOF predictions.
- Ported constants are not nested for 2016-2021 folds (see V5 caveat); an in-fold refit of the Elo grid and
  Kalman ML would remove it.
