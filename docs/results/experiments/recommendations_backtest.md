# Recommendations backtest (exploratory)

Generated 2026-11-25 22:14 UTC by `python scripts/experiments/recommendations_backtest.py`. Runtime 1753s. Test seasons 2015-2024 on the step-3 fold table (weekly rolling origin, T = kickoff - 24h). 2025 (holdout) and 2026 were not read.

Pre-registered settings are in the script docstring. Exploratory: not preregistered in the experiment log, no leakage audit sign-off.

## Pooled, all test seasons

| model_id | n_games | mae_margin | rmse_margin | bias_margin | mae_total | rmse_total | n_win | logloss_win | brier_win |
|---|---|---|---|---|---|---|---|---|---|
| bayes_od_fit | 2743 | 10.151 | 13.106 | -0.045 | 10.813 | 13.661 | 2734 | 0.636 | 0.223 |
| bayes_od_v1 | 2743 | 10.161 | 13.102 | -0.049 | 10.844 | 13.696 | 2734 | 0.636 | 0.223 |
| elo_v1 | 2743 | 10.176 | 13.145 | -0.465 | 10.982 | 13.881 | 2734 | 0.635 | 0.222 |
| kalman_gs_fit | 2743 | 10.196 | 13.118 | 0.699 |  |  | 2734 | 0.637 | 0.223 |
| kalman_gs_v1 | 2743 | 10.152 | 13.091 | -0.345 |  |  | 2734 | 0.636 | 0.222 |
| lgbm_epa_qb | 2743 | 10.150 | 13.079 | 0.313 | 10.736 | 13.621 | 2734 | 0.632 | 0.221 |
| lgbm_v1 | 2743 | 10.226 | 13.155 | 0.373 | 10.850 | 13.713 | 2734 | 0.636 | 0.223 |
| ridge_epa | 2743 | 10.115 | 13.074 | 0.329 | 10.736 | 13.613 | 2734 | 0.634 | 0.222 |
| ridge_epa_qb | 2743 | 10.109 | 13.047 | 0.306 | 10.678 | 13.562 | 2734 | 0.631 | 0.221 |
| ridge_qb | 2743 | 10.117 | 13.066 | 0.361 | 10.700 | 13.578 | 2734 | 0.632 | 0.221 |
| ridge_v1 | 2743 | 10.144 | 13.109 | 0.419 | 10.792 | 13.650 | 2734 | 0.634 | 0.222 |
| xgb_epa_qb | 2743 | 10.143 | 13.070 | 0.528 | 10.663 | 13.567 | 2734 | 0.633 | 0.222 |
| xgb_v1 | 2743 | 10.190 | 13.103 | 0.523 | 10.773 | 13.660 | 2734 | 0.635 | 0.222 |

## Clean seasons only (2015, 2022-2024)

| model_id | n_games | mae_margin | rmse_margin | bias_margin | mae_total | rmse_total | n_win | logloss_win | brier_win |
|---|---|---|---|---|---|---|---|---|---|
| bayes_od_fit | 1121 | 10.020 | 12.943 | -0.115 | 10.551 | 13.404 | 1119 | 0.641 | 0.225 |
| bayes_od_v1 | 1121 | 10.033 | 12.940 | -0.134 | 10.577 | 13.430 | 1119 | 0.640 | 0.225 |
| elo_v1 | 1121 | 9.946 | 12.940 | -0.853 | 10.677 | 13.583 | 1119 | 0.637 | 0.223 |
| kalman_gs_fit | 1121 | 10.047 | 12.928 | 0.255 |  |  | 1119 | 0.640 | 0.225 |
| kalman_gs_v1 | 1121 | 10.007 | 12.923 | -0.704 |  |  | 1119 | 0.639 | 0.224 |
| lgbm_epa_qb | 1121 | 9.940 | 12.835 | 0.196 | 10.487 | 13.392 | 1119 | 0.635 | 0.222 |
| lgbm_v1 | 1121 | 10.037 | 12.928 | 0.409 | 10.536 | 13.366 | 1119 | 0.639 | 0.224 |
| ridge_epa | 1121 | 9.907 | 12.860 | 0.091 | 10.465 | 13.313 | 1119 | 0.636 | 0.223 |
| ridge_epa_qb | 1121 | 9.895 | 12.803 | 0.048 | 10.386 | 13.246 | 1119 | 0.634 | 0.222 |
| ridge_qb | 1121 | 9.906 | 12.838 | 0.055 | 10.439 | 13.272 | 1119 | 0.635 | 0.222 |
| ridge_v1 | 1121 | 9.949 | 12.905 | 0.134 | 10.520 | 13.337 | 1119 | 0.638 | 0.223 |
| xgb_epa_qb | 1121 | 9.943 | 12.868 | 0.192 | 10.426 | 13.331 | 1119 | 0.638 | 0.223 |
| xgb_v1 | 1121 | 9.998 | 12.896 | 0.323 | 10.484 | 13.330 | 1119 | 0.639 | 0.224 |

## Paired bootstrap (candidate minus baseline; negative = candidate better)

| question | candidate | baseline | metric | n_games | diff | lo | hi |
|---|---|---|---|---|---|---|---|
| EPA | ridge_epa | ridge_v1 | abs_margin | 2743 | -0.0283 | -0.0650 | 0.0077 |
| EPA | ridge_epa | ridge_v1 | abs_total | 2743 | -0.0555 | -0.0956 | -0.0160 |
| EPA | ridge_epa | ridge_v1 | logloss_win | 2734 | -0.0007 | -0.0027 | 0.0015 |
| QB | ridge_qb | ridge_v1 | abs_margin | 2743 | -0.0269 | -0.0599 | 0.0029 |
| QB | ridge_qb | ridge_v1 | abs_total | 2743 | -0.0922 | -0.1467 | -0.0370 |
| QB | ridge_qb | ridge_v1 | logloss_win | 2734 | -0.0027 | -0.0045 | -0.0008 |
| EPA+QB | ridge_epa_qb | ridge_v1 | abs_margin | 2743 | -0.0349 | -0.0812 | 0.0114 |
| EPA+QB | ridge_epa_qb | ridge_v1 | abs_total | 2743 | -0.1137 | -0.1656 | -0.0583 |
| EPA+QB | ridge_epa_qb | ridge_v1 | logloss_win | 2734 | -0.0031 | -0.0057 | -0.0005 |
| EPA+QB | xgb_epa_qb | xgb_v1 | abs_margin | 2743 | -0.0465 | -0.0796 | -0.0130 |
| EPA+QB | xgb_epa_qb | xgb_v1 | abs_total | 2743 | -0.1103 | -0.1746 | -0.0479 |
| EPA+QB | xgb_epa_qb | xgb_v1 | logloss_win | 2734 | -0.0019 | -0.0040 | 0.0001 |
| LightGBM | lgbm_v1 | xgb_v1 | abs_margin | 2743 | 0.0362 | -0.0111 | 0.0803 |
| LightGBM | lgbm_v1 | xgb_v1 | abs_total | 2743 | 0.0771 | 0.0256 | 0.1246 |
| LightGBM | lgbm_v1 | xgb_v1 | logloss_win | 2734 | 0.0009 | -0.0018 | 0.0036 |
| EPA+QB | lgbm_epa_qb | lgbm_v1 | abs_margin | 2743 | -0.0759 | -0.1341 | -0.0163 |
| EPA+QB | lgbm_epa_qb | lgbm_v1 | abs_total | 2743 | -0.1145 | -0.1950 | -0.0356 |
| EPA+QB | lgbm_epa_qb | lgbm_v1 | logloss_win | 2734 | -0.0044 | -0.0079 | -0.0008 |
| fitted | kalman_gs_fit | kalman_gs_v1 | abs_margin | 2743 | 0.0448 | 0.0044 | 0.0845 |
| fitted | kalman_gs_fit | kalman_gs_v1 | logloss_win | 2734 | 0.0011 | -0.0011 | 0.0035 |
| fitted | bayes_od_fit | bayes_od_v1 | abs_margin | 2743 | -0.0094 | -0.0305 | 0.0131 |
| fitted | bayes_od_fit | bayes_od_v1 | abs_total | 2743 | -0.0314 | -0.0513 | -0.0126 |
| fitted | bayes_od_fit | bayes_od_v1 | logloss_win | 2734 | 0.0005 | -0.0003 | 0.0013 |
| vs Elo | ridge_epa_qb | elo_v1 | abs_margin | 2743 | -0.0677 | -0.1419 | 0.0046 |
| vs Elo | ridge_epa_qb | elo_v1 | abs_total | 2743 | -0.3040 | -0.4159 | -0.1929 |
| vs Elo | ridge_epa_qb | elo_v1 | logloss_win | 2734 | -0.0039 | -0.0086 | 0.0007 |
| vs Elo | xgb_epa_qb | elo_v1 | abs_margin | 2743 | -0.0330 | -0.1062 | 0.0407 |
| vs Elo | xgb_epa_qb | elo_v1 | abs_total | 2743 | -0.3196 | -0.4244 | -0.2049 |
| vs Elo | xgb_epa_qb | elo_v1 | logloss_win | 2734 | -0.0018 | -0.0067 | 0.0030 |

## Beyond the line

Regression of (result - line) on (prediction - line), nflverse spread_line (close-like, capture time unknown). A slope near 0 means the model adds nothing the line does not already have; a significantly positive slope means it does. ATS hit rate counts games where the model differs from the line by at least 1.5 points, pushes excluded.

| model_id | n | slope | slope_se | t | mean_abs_diff_from_line | ats_n_(|diff|>=1.5) | ats_hit | ats_p_vs_50% |
|---|---|---|---|---|---|---|---|---|
| bayes_od_fit | 2743 | 0.058 | 0.077 | 0.751 | 2.409 | 1612 | 0.512 | 0.331 |
| bayes_od_v1 | 2743 | 0.075 | 0.076 | 0.988 | 2.445 | 1629 | 0.511 | 0.400 |
| elo_v1 | 2743 | 0.060 | 0.074 | 0.802 | 2.553 | 1682 | 0.513 | 0.294 |
| kalman_gs_fit | 2743 | 0.040 | 0.079 | 0.509 | 2.381 | 1588 | 0.494 | 0.670 |
| kalman_gs_v1 | 2743 | 0.060 | 0.079 | 0.754 | 2.344 | 1599 | 0.508 | 0.516 |
| lgbm_epa_qb | 2743 | 0.131 | 0.075 | 1.748 | 2.517 | 1631 | 0.508 | 0.552 |
| lgbm_v1 | 2743 | 0.087 | 0.070 | 1.235 | 2.687 | 1731 | 0.497 | 0.810 |
| ridge_epa | 2743 | 0.100 | 0.078 | 1.272 | 2.365 | 1574 | 0.517 | 0.199 |
| ridge_epa_qb | 2743 | 0.106 | 0.081 | 1.306 | 2.282 | 1531 | 0.521 | 0.113 |
| ridge_qb | 2743 | 0.058 | 0.084 | 0.694 | 2.235 | 1528 | 0.504 | 0.778 |
| ridge_v1 | 2743 | 0.055 | 0.078 | 0.701 | 2.388 | 1596 | 0.508 | 0.565 |
| xgb_epa_qb | 2743 | 0.101 | 0.081 | 1.253 | 2.346 | 1524 | 0.496 | 0.778 |
| xgb_v1 | 2743 | 0.076 | 0.077 | 0.982 | 2.412 | 1584 | 0.497 | 0.860 |

## Feature coverage on test games

| feature | non_missing |
|---|---|
| home_off_epa_l8 | 1.000 |
| home_def_epa_l8 | 1.000 |
| home_off_epa_ewm | 1.000 |
| home_def_epa_ewm | 1.000 |
| home_off_pass_epa_ewm | 1.000 |
| home_def_pass_epa_ewm | 1.000 |
| home_qb_out | 1.000 |
| home_qb_epa | 1.000 |
| away_off_epa_l8 | 1.000 |
| away_def_epa_l8 | 1.000 |
| away_off_epa_ewm | 1.000 |
| away_def_epa_ewm | 1.000 |
| away_off_pass_epa_ewm | 1.000 |
| away_def_pass_epa_ewm | 1.000 |
| away_qb_out | 1.000 |
| away_qb_epa | 1.000 |

Share of team-games with the expected starting QB listed Out/Doubtful before T: 0.018.

Data checks: {"pbp_team_games_matched": 13976, "pbp_team_games_total": 13976, "injury_out_rows": 17985, "injury_out_rows_no_timestamp": 1003}

## Fitted parameters per fold

```
{
 "kalman": {
  "2014": {
   "h": 2.6758442411400325,
   "sigma_m": 12.720564299521762,
   "tau_w": 0.6321056013367001,
   "rho": 0.6088191936147954,
   "tau_s": 3.705872092303729
  },
  "2015": {
   "h": 2.6069614463392012,
   "sigma_m": 12.685044814712732,
   "tau_w": 0.6637692946225595,
   "rho": 0.6108943837197738,
   "tau_s": 3.622995040306253
  },
  "2016": {
   "h": 2.6244332932461387,
   "sigma_m": 12.604568753514727,
   "tau_w": 0.6703679866478338,
   "rho": 0.6043801274470723,
   "tau_s": 3.5661779463193852
  },
  "2017": {
   "h": 2.610998324657329,
   "sigma_m": 12.6066495588654,
   "tau_w": 0.6624088117863967,
   "rho": 0.5969119537863677,
   "tau_s": 3.609920891820613
  },
  "2018": {
   "h": 2.589650753364024,
   "sigma_m": 12.59744509973702,
   "tau_w": 0.6413551303030535,
   "rho": 0.5985833300207708,
   "tau_s": 3.6193785252169905
  },
  "2019": {
   "h": 2.4613864050548733,
   "sigma_m": 12.583712726041531,
   "tau_w": 0.6680079230294037,
   "rho": 0.6062556597117295,
   "tau_s": 3.5656764390246973
  },
  "2020": {
   "h": 2.3557987336882222,
   "sigma_m": 12.553980002458161,
   "tau_w": 0.6970210926361132,
   "rho": 0.6000972543267389,
   "tau_s": 3.5307570032112787
  },
  "2021": {
   "h": 2.3240555841998978,
   "sigma_m": 12.591058685865114,
   "tau_w": 0.6807537696698793,
   "rho": 0.6129286582020258,
   "tau_s": 3.4923434930438586
  },
  "2022": {
   "h": 2.317695889514604,
   "sigma_m": 12.503535167681472,
   "tau_w": 0.7261628417759258,
   "rho": 0.5939078512766458,
   "tau_s": 3.44068936771104
  },
  "2023": {
   "h": 2.335543164034527,
   "sigma_m": 12.525668988916273,
   "tau_w": 0.7057160218704822,
   "rho": 0.603357397500423,
   "tau_s": 3.399946464628643
  }
 },
 "bayes_od": {
  "2014": [
   {
    "sigma": 9.143756041743831,
    "prior_team_sd": 3.182397520857358,
    "prior_mu": 21.0,
    "prior_mu_sd": 3.0,
    "prior_h": 2.5,
    "prior_h_sd": 1.5,
    "q_game_sd": 0.386462014087971,
    "q_global_sd": 0.05,
    "rho": 0.6487980136432341,
    "q_season_sd": 2.2085667986419146
   },
   0.018373773151964576
  ],
  "2015": [
   {
    "sigma": 9.11251346801587,
    "prior_team_sd": 3.060641331421598,
    "prior_mu": 21.0,
    "prior_mu_sd": 3.0,
    "prior_h": 2.5,
    "prior_h_sd": 1.5,
    "q_game_sd": 0.4206637986006126,
    "q_global_sd": 0.05,
    "rho": 0.6346639960689556,
    "q_season_sd": 2.1686521566736974
   },
   0.012015476479917405
  ],
  "2016": [
   {
    "sigma": 9.068679378416961,
    "prior_team_sd": 3.143301677363473,
    "prior_mu": 21.0,
    "prior_mu_sd": 3.0,
    "prior_h": 2.5,
    "prior_h_sd": 1.5,
    "q_game_sd": 0.42687626789770633,
    "q_global_sd": 0.05,
    "rho": 0.6332906988932543,
    "q_season_sd": 2.140775958761292
   },
   0.022522016987903487
  ],
  "2017": [
   {
    "sigma": 9.134217913062455,
    "prior_team_sd": 3.147863460019912,
    "prior_mu": 21.0,
    "prior_mu_sd": 3.0,
    "prior_h": 2.5,
    "prior_h_sd": 1.5,
    "q_game_sd": 0.4211306679676987,
    "q_global_sd": 0.05,
    "rho": 0.6242311942887327,
    "q_season_sd": 2.1580325381700987
   },
   0.0063989684365136764
  ],
  "2018": [
   {
    "sigma": 9.117620513479425,
    "prior_team_sd": 3.5202079742415373,
    "prior_mu": 21.0,
    "prior_mu_sd": 3.0,
    "prior_h": 2.5,
    "prior_h_sd": 1.5,
    "q_game_sd": 0.4129440856174325,
    "q_global_sd": 0.05,
    "rho": 0.616654042136982,
    "q_season_sd": 2.201585309247151
   },
   0.019433125301791105
  ],
  "2019": [
   {
    "sigma": 9.067066067976537,
    "prior_team_sd": 3.0861740676053784,
    "prior_mu": 21.0,
    "prior_mu_sd": 3.0,
    "prior_h": 2.5,
    "prior_h_sd": 1.5,
    "q_game_sd": 0.43855935647566313,
    "q_global_sd": 0.05,
    "rho": 0.6116607366812216,
    "q_season_sd": 2.1980037096972427
   },
   0.006431556421754047
  ],
  "2020": [
   {
    "sigma": 9.076988667244173,
    "prior_team_sd": 3.099746772792868,
    "prior_mu": 21.0,
    "prior_mu_sd": 3.0,
    "prior_h": 2.5,
    "prior_h_sd": 1.5,
    "q_game_sd": 0.42158305922476774,
    "q_global_sd": 0.05,
    "rho": 0.6073044469985202,
    "q_season_sd": 2.203803711699501
   },
   0.0044398333212630585
  ],
  "2021": [
   {
    "sigma": 9.086127801179355,
    "prior_team_sd": 3.128731433319041,
    "prior_mu": 21.0,
    "prior_mu_sd": 3.0,
    "prior_h": 2.5,
    "prior_h_sd": 1.5,
    "q_game_sd": 0.43190467236864977,
    "q_global_sd": 0.05,
    "rho": 0.6181968525836157,
    "q_season_sd": 2.1424288119760013
   },
   0.030717379562985415
  ],
  "2022": [
   {
    "sigma": 9.071997048273717,
    "prior_team_sd": 3.08886812049294,
    "prior_mu": 21.0,
    "prior_mu_sd": 3.0,
    "prior_h": 2.5,
    "prior_h_sd": 1.5,
    "q_game_sd": 0.4307743496629662,
    "q_global_sd": 0.05,
    "rho": 0.6047502716975461,
    "q_season_sd": 2.1696208449185046
   },
   0.03229780659058305
  ],
  "2023": [
   {
    "sigma": 9.060658661591107,
    "prior_team_sd": 3.17750979331004,
    "prior_mu": 21.0,
    "prior_mu_sd": 3.0,
    "prior_h": 2.5,
    "prior_h_sd": 1.5,
    "q_game_sd": 0.46271117056930605,
    "q_global_sd": 0.05,
    "rho": 0.6093280876899605,
    "q_season_sd": 2.0648140868036773
   },
   0.033932337542456135
  ]
 }
}
```

Fit+predict seconds: {"elo_v1": 7.8, "kalman_gs_v1": 8.1, "bayes_od_v1": 6.9, "ridge_v1": 54.8, "xgb_v1": 68.0, "ridge_epa": 63.3, "ridge_qb": 61.1, "ridge_epa_qb": 64.1, "xgb_epa_qb": 94.1, "lgbm_v1": 88.7, "lgbm_epa_qb": 135.3, "kalman_gs_fit": 80.2, "bayes_od_fit": 1008.9}

## Beyond the line: totals (added by hand from the same prediction file)

Same regression for totals against the nflverse total_line. Over/under hit rate where the model differs from the line by at least 1.5 points.

| model_id | slope | se | t | ou_hit | n |
|---|---|---|---|---|---|
| bayes_od_fit | -0.301 | 0.091 | -3.31 | 0.476 | 1532 |
| bayes_od_v1 | -0.282 | 0.086 | -3.27 | 0.478 | 1602 |
| elo_v1 | -0.017 | 0.063 | -0.26 | 0.494 | 1991 |
| lgbm_epa_qb | -0.083 | 0.084 | -1.00 | 0.496 | 1660 |
| lgbm_v1 | -0.060 | 0.073 | -0.82 | 0.498 | 1802 |
| ridge_epa | -0.142 | 0.090 | -1.58 | 0.482 | 1580 |
| ridge_epa_qb | -0.119 | 0.096 | -1.24 | 0.492 | 1503 |
| ridge_qb | -0.088 | 0.092 | -0.96 | 0.504 | 1586 |
| ridge_v1 | -0.081 | 0.083 | -0.98 | 0.487 | 1663 |
| xgb_epa_qb | -0.046 | 0.089 | -0.51 | 0.510 | 1617 |
| xgb_v1 | -0.040 | 0.079 | -0.51 | 0.497 | 1731 |

## Conclusions

1. EPA + QB features make every feature model more accurate (totals MAE about -0.11 for Ridge, XGB and LightGBM, all
   significant; margin MAE -0.03 to -0.08, significant for XGB and LightGBM). Best margin model ridge_epa_qb 10.109
   vs the nflverse line 9.821.
2. None of it is information the line lacks: no model's (prediction - line) significantly predicts (result - line)
   for spreads (largest t = 1.75) or totals (all slopes <= 0); ATS and over/under hit rates 47.6%-52.1%, none
   different from 50% and all below the 52.4% needed at -110.
3. Fitted settings: kalman_gs_fit is worse than the frozen constants (margin MAE +0.045, significant; ML on
   1999-onward data puts home field at 2.3-2.7 points, so it overpredicts home teams in recent seasons).
   bayes_od_fit improves totals slightly (-0.031, significant) and estimates a home/away error correlation of
   about 0.03, which supports the independence assumption (P5).
4. LightGBM with the pre-registered settings is worse than XGBoost on totals (+0.077, significant) and no better
   on margin.
