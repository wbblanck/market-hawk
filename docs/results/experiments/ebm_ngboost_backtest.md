# EBM and NGBoost backtest (exploratory, 2026-11-26)

Script: `scripts/experiments/ebm_ngboost_backtest.py` (settings pre-registered in its docstring, not tuned).
Walk-forward test seasons 2015-2024 (distributions scored 2018-2024, the same 1,942 games for every candidate).
2025 never read. Every game's features are computed at its week's cutoff from games labelled before it.
EBM and NGBoost are refit once per season. An xgb_v2 refit on the same schedule is included as a control.
The nflverse line is a benchmark only and is never a feature.

## Answers

1. **EBM as a model: no gain.** EBM ties ridge_v2 and xgb_v2. Margin MAE is 10.133 vs 10.110, a difference of
   +0.023 [-0.021, +0.070]. Total MAE is 10.666 vs 10.678, a difference of -0.012 [-0.074, +0.051]. Neither
   difference is significant. It is not worth adding as a production model.
2. **EBM as a diagnostic: XGBoost's relationships are sensible, with one exception.** On every important
   feature, the EBM, XGBoost and Ridge agree in sign, and the sizes are similar. The exception is a starting QB
   ruled out. In the scenario the features encode (flag = 1, QB EPA set to replacement level), the predicted
   home margin changes by **-3.0 points under the EBM, -4.0 under Ridge, and only -1.0 under xgb_v2**. The
   xgb_v2 split rules (min_child_weight 20, 100 shallow trees) almost never split on a flag that is set in
   about 1.9% of games, so XGBoost gives a ruled-out QB about a third of the weight the other two models give
   it. The other models agree with each other, and market moves for a ruled-out starter are usually several
   points, so this looks like under-reaction rather than insight. The sample is about 50 such games per side
   in 2015-2024, which is too small to test directly.
3. **NGBoost per-game spread: rejected.** Every NGBoost variant scores worse than the production distribution
   layer, and the gaps are significant.
   - The sigma-only variant (NGBoost spread on ridge_v2 means) costs -0.171 [-0.200, -0.143] log score on
     margins and -0.066 [-0.091, -0.043] on totals. Its 80% intervals cover only 73-74% of outcomes, so it is
     overconfident.
   - The predicted spread does not track the actual error: the correlation between per-game sigma and
     |residual| is -0.01 on margins and 0.03 on totals. The features carry no detectable signal about which
     games are more uncertain, so NGBoost fits noise.
   - The full variant (NGBoost mean and spread) has a worse mean (margin MAE 10.204 vs 10.155 on the scored
     games) and a worse log score (margins -0.127, totals -0.013).
   - The production margin family (k1, key-number aware) remains best. The closest alternative is a t
     distribution with key-number weights and a constant spread, at -0.019. Key-number mass matters more than
     per-game width, as expected.
   - No candidate predicts cover or over outcomes against the line: hit rates are 49.4-50.9% on the games
     where the model's probability is more than 2.4 points from 50%.

## 1. Point accuracy (2015-2024, 2,743 games)

| target | model | MAE |
|---|---|---|
| margin | ridge_v2 (weekly refit) | 10.110 |
| margin | xgb_v2 (weekly refit) | 10.142 |
| margin | xgb_v2 settings (season refit) | 10.142 |
| margin | EBM (season refit) | 10.133 |
| margin | NGBoost mean (season refit) | 10.161 |
| margin | nflverse line (benchmark) | 9.821 |
| total | ridge_v2 (weekly refit) | 10.678 |
| total | xgb_v2 (weekly refit) | 10.661 |
| total | xgb_v2 settings (season refit) | 10.673 |
| total | EBM (season refit) | 10.666 |
| total | NGBoost mean (season refit) | 10.695 |
| total | nflverse line (benchmark) | 10.444 |

MAE minus ridge_v2's, with a 95% bootstrap CI clustered by (season, week), B = 2000:

| target | candidate | diff | 95% CI |
|---|---|---|---|
| margin | EBM | +0.023 | [-0.021, +0.070] |
| margin | xgb_v2 settings, season refit | +0.032 | [-0.019, +0.085] |
| margin | NGBoost mean | +0.050 | [-0.004, +0.107] |
| total | EBM | -0.012 | [-0.074, +0.051] |
| total | xgb_v2 settings, season refit | -0.005 | [-0.064, +0.056] |
| total | NGBoost mean | +0.017 | [-0.053, +0.086] |

The season-refit xgb control matches weekly xgb_v2 (10.142 both on margin), so refitting once per season does not
handicap EBM or NGBoost.

## 2. Distributions (2018-2024, 1,942 games)

"-log pmf" is the mean negative log probability of the observed integer outcome (lower is better). cov80 is the coverage of
80% central intervals (target 0.80). line_brier is the Brier score of P(cover or over | no push) against the
nflverse line. hit is the hit rate where |P - 0.5| > 0.024.

| target | candidate | -log pmf | PIT KS | cov80 | line Brier | sd of P | bets | hit | pred push | obs push |
|---|---|---|---|---|---|---|---|---|---|---|
| margin | **production: ridge_v2 + k1** | **3.872** | 0.031 | 0.811 | **0.2574** | 0.098 | 1490 | 0.501 | 0.023 | 0.026 |
| margin | ridge_v2 + t, constant spread | 3.991 | 0.038 | 0.795 | 0.2575 | 0.095 | 1495 | 0.496 | 0.016 | 0.026 |
| margin | ridge_v2 + t+key, constant spread | 3.891 | 0.031 | 0.786 | 0.2580 | 0.099 | 1475 | 0.495 | 0.023 | 0.026 |
| margin | ridge_v2 + t, NGBoost sigma | 4.043 | 0.049 | 0.732 | 0.2602 | 0.112 | 1556 | 0.498 | 0.019 | 0.026 |
| margin | ridge_v2 + t+key, NGBoost sigma | 3.951 | 0.052 | 0.728 | 0.2604 | 0.112 | 1525 | 0.494 | 0.025 | 0.026 |
| margin | NGBoost full (mean + sigma), t | 3.999 | 0.041 | 0.795 | 0.2596 | 0.102 | 1519 | 0.502 | 0.016 | 0.026 |
| total | **production: ridge_v2 + t** | **4.022** | 0.024 | 0.823 | **0.2564** | 0.082 | 1470 | 0.499 | 0.015 | 0.011 |
| total | ridge_v2 + t, NGBoost sigma | 4.089 | 0.054 | 0.738 | 0.2598 | 0.103 | 1536 | 0.493 | 0.019 | 0.011 |
| total | NGBoost full (mean + sigma), t | 4.035 | 0.037 | 0.783 | 0.2586 | 0.101 | 1567 | 0.509 | 0.017 | 0.011 |

Log-score gain over production (positive = better), clustered bootstrap 95% CI:

| target | candidate | gain | 95% CI |
|---|---|---|---|
| margin | ridge_v2 + t, constant spread | -0.119 | [-0.144, -0.096] |
| margin | ridge_v2 + t+key, constant spread | -0.019 | [-0.032, -0.006] |
| margin | ridge_v2 + t, NGBoost sigma | -0.171 | [-0.200, -0.143] |
| margin | ridge_v2 + t+key, NGBoost sigma | -0.079 | [-0.100, -0.056] |
| margin | NGBoost full, t | -0.127 | [-0.152, -0.101] |
| total | ridge_v2 + t, NGBoost sigma | -0.066 | [-0.091, -0.043] |
| total | NGBoost full, t | -0.013 | [-0.022, -0.005] |

Spread of the predicted sigma, 10th / 50th / 90th percentile:
- Sigma-only: margin 9.2 / 11.3 / 12.7; total 9.0 / 11.7 / 13.3. These are too narrow; the residual SD is about
  13.
- Full: margin 12.0 / 13.0 / 13.7; total 11.6 / 13.0 / 13.6.

The sigma-only model is trained on ridge_v2's out-of-fold residuals from earlier test seasons only, which is about
270 to 2,200 rows. It overfits those rows and shrinks its spread. The full model trains on every earlier game and
stays near a constant, but its small variations do not line up with actual errors.

## 3. EBM diagnostic (all labelled games 2000-2024 with EPA features, margin and total)

Effect = change in the mean prediction (points) when the feature moves from its 10th to its 90th percentile
(binary: 0 to 1), other features held at observed values (partial dependence on 1,500 sampled games).

| target | feature | EBM | xgb_v2 settings | Ridge |
|---|---|---|---|---|
| margin | kalman | +7.5 | +9.1 | +6.6 |
| margin | elo_fair | +1.0 | +1.2 | +2.4 |
| margin | elo_d | +0.9 | +1.9 | +2.4 |
| margin | home_qb_epa | +1.7 | +1.5 | +2.4 |
| margin | home_off_epa_ewm | +1.5 | +0.9 | +3.0 |
| margin | away_off_epa_ewm | -1.4 | -0.4 | -1.8 |
| margin | away_def_epa_ewm (EPA allowed) | +1.3 | +0.0 | +3.3 |
| margin | away_off_epa_l8 | -1.2 | -0.7 | -1.0 |
| margin | home_qb_out flag alone | -2.1 | 0.0 | -1.9 |
| margin | **home QB out scenario (flag 1, qb_epa -0.15)** | **-3.0** | **-1.0** | **-4.0** |
| total | is_dome | +2.5 | +1.6 | +2.7 |
| total | home_qb_epa | +2.3 | +2.4 | +2.9 |
| total | away_qb_epa | +2.6 | +1.7 | +2.5 |
| total | div_game | -1.1 | -0.2 | -1.1 |
| total | away_off_pass_epa_ewm | +1.3 | +1.3 | +1.4 |
| total | home_off_epa_ewm | +1.7 | +1.5 | +2.1 |
| total | home_pf_l8 | +1.7 | +1.1 | +2.5 |
| total | **home QB out scenario** | **-2.9** | **-1.7** | **-3.5** |

All signs agree. The EBM's most important margin terms are the Kalman rating, the Elo terms, home QB EPA and EPA
form. For totals they are dome, both QBs' EPA and division game.

XGBoost's effects are generally smaller than the other two (100 trees at eta 0.03 with min_child_weight 20 shrink
everything). They are close to zero for away defensive EPA, division game and the QB-out flag.

## Conclusions

- Do not add EBM or NGBoost to production. EBM stays a one-off diagnostic, rerun when features change.
- Keep k1 for margins and t for totals.
- Follow-up candidate, not adopted: xgb_v2 under-reacts to a ruled-out starting QB. The cheapest fix is to rely
  on ridge_v2 (the default pricing model) or the stack for QB-out games, or to pre-register an xgb_v3 with a
  smaller min_child_weight and test it walk-forward. Too few QB-out games exist to choose by backtest alone.

## Follow-up: which QB-out effect is right? (2026-11-26)

Script: `scripts/experiments/qb_out_market_effect.py` (research only, no production change).

No line from before the injury news exists in our data. The nflverse line is close-like (one line per game,
capture time unknown), and our own captures start 2026-11-21. So instead of a lookahead-to-close move, this
regresses the nflverse line itself on the team-strength features at T (the xgb_v2 set without the QB columns)
plus the two QB-out flags. The data are 2009-2024: 4,345 games, 159 flagged starters (82 home, 77 away).

The line already prices team strength, so the flag coefficient is the market's adjustment for a newly ruled-out
expected starter. Its residual SD is 2.7 points, against 13.3 for final margins, so it is about 5 times more
precise. CIs are from a bootstrap clustered by (season, week), B = 4,000.

| dependent | points against the team whose QB is out | 95% CI |
|---|---|---|
| nflverse spread line (market) | **3.05** | [2.49, 3.65] |
| final margin (results) | 3.53 | [1.41, 5.67] |
| margin minus line (anything the market misses) | 0.48 | [-1.48, 2.47] |
| nflverse total line (change in total) | -1.69 | [-2.19, -1.25] |
| final total | -3.51 | [-5.51, -1.59] |

Line effect by era: 2009-2016 home -2.9 / away +2.0; 2017-2024 home -3.5 / away +3.8.

**Reading.** The market prices a ruled-out starter at about 3 points, and results agree, at 3.5. xgb_v2's 1.0
point lies below both CIs, while the EBM (3.0) and Ridge (4.0) lie inside them. On totals, the market lowers the
total by 1.7 and xgb_v2 by 1.7, while results fall by 3.5, which is closer to the EBM and Ridge. The
results-versus-line gap (0.5, CI including 0) shows no edge from QB-out games: the market already prices them.

**Caveat.** The flag catches only starters listed Out or Doubtful who started the team's previous game. QBs hurt
mid-game the week before, where the backup already had the most dropbacks, are not flagged.
