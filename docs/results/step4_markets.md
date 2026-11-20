# Step 4 market benchmark (auto-generated; NOT CLV)

Regenerate: `python scripts/run_market_backtest.py --model elo_v1 --bootstrap 2000 --seed 0`
(reads `data/processed/step3_oof_predictions.parquet` and `data/raw/nflverse_games.csv`).

**What this is.** Model `elo_v1` OOF predictions (2015-2024 walk-forward, step 3) priced with a
margin `k1` and total `t` distribution fit walk-forward per season on earlier seasons only, then compared
with the nflverse provider line and price for each game: "nflverse line, capture time unknown". Because the capture time
is unknown (close-like), this is a **benchmark against a provider line, not closing line value**. No CLV
is computed or implied here. Real CLV only exists for games we capture going forward
(`markets.bet_log`, `docs/methodology/closing_line.md`). The 2025 holdout and 2026 are never read.

**Bets.** Per game at most one spread side and one total side (the higher-EV side) when
`ev_per_unit >= 0.02`; units from `markets.staking.size_stakes` with the default StakePolicy
(proposed, not locked: Kelly x 0.25, bankroll 100 units fixed, max
1/bet, 2/game, 8/week). Win rates exclude
pushes. ROI = P&L / units staked, 95% CI from a game-clustered bootstrap (2000 draws, seed 0).
Seasons 2015-2017 have fewer than 3 earlier OOF seasons to fit the distributions and are not scored.

Columns: `n_qual_*` / `win_rate_*` = every best-side bet with `ev_per_unit >= min_ev`, flat stakes, before
caps (`flat_roi` = P&L per unit on them); `n_staked`, `units_staked`, `pnl_units`, `roi*` = after StakePolicy sizing and caps; `mean_edge_*` = mean of `p_win_no_push - breakeven` of the model's preferred side over all
games (what the model *claims*); `cover_logloss_home` / `over_logloss` = log loss of P(home cover | no push)
and P(over | no push) at the nflverse line (a coin flip scores ln 2 = 0.693; lower is better).

## By season

| season | n_games | mean_edge_ats | n_qual_ats | win_rate_ats | mean_edge_ou | n_qual_ou | win_rate_ou | flat_roi | n_staked | cover_logloss_home | over_logloss | units_staked | pnl_units | roi | roi_lo | roi_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018 | 267 | 0.074 | 214 | 0.507 | 0.099 | 229 | 0.493 | -0.023 | 155 | 0.713 | 0.733 | 154.371 | 10.302 | 0.067 | -0.081 | 0.221 |
| 2019 | 267 | 0.071 | 227 | 0.518 | 0.084 | 228 | 0.524 | 0.017 | 154 | 0.710 | 0.717 | 153.592 | 9.542 | 0.062 | -0.103 | 0.225 |
| 2020 | 269 | 0.067 | 224 | 0.522 | 0.088 | 238 | 0.481 | -0.025 | 158 | 0.711 | 0.737 | 157.682 | 0.330 | 0.002 | -0.156 | 0.157 |
| 2021 | 285 | 0.077 | 243 | 0.490 | 0.086 | 244 | 0.473 | -0.065 | 162 | 0.727 | 0.731 | 162.000 | -8.381 | -0.052 | -0.197 | 0.105 |
| 2022 | 284 | 0.071 | 235 | 0.509 | 0.084 | 247 | 0.490 | -0.028 | 163 | 0.696 | 0.725 | 163.000 | 9.773 | 0.060 | -0.081 | 0.208 |
| 2023 | 285 | 0.058 | 206 | 0.444 | 0.077 | 227 | 0.504 | -0.086 | 162 | 0.730 | 0.727 | 162.000 | -4.570 | -0.028 | -0.175 | 0.110 |
| 2024 | 285 | 0.064 | 227 | 0.504 | 0.073 | 226 | 0.531 | -0.008 | 164 | 0.715 | 0.714 | 163.629 | -1.796 | -0.011 | -0.152 | 0.134 |
| pooled | 1942 | 0.069 | 1576 | 0.500 | 0.084 | 1639 | 0.499 | -0.031 | 1118 | 0.715 | 0.726 | 1116.274 | 15.199 | 0.014 | -0.045 | 0.071 |

## Reading

Pooled over 1942 games: 1576 qualifying spread bets won 0.500 and
1639 qualifying total bets won 0.499 (flat, pushes excluded; break-even at
-110 is 0.524). Flat 1-unit ROI on all of them: -0.031. Under the StakePolicy caps 1118 bets got a stake: ROI 0.014 on
1116.3 units (95% CI -0.045 to 0.071). The week cap binds
most weeks (8 units), so the staked set is the highest-EV subset of each week.
Cover log loss 0.7147 and over log loss 0.7261 vs 0.6931 for 50%.
The model's claimed edges come from disagreeing with the line; step 3 found no model beats the line on
accuracy, so a claimed edge here is mostly model error, and the interval says whether the realised ROI can be
told apart from the vig. This does not support staking real money on `elo_v1`. The capped ROI being above the flat ROI
is a property of one bet subset, inside a CI that includes zero; it is not evidence of skill. Caveat: `elo_v1`
uses ported constants tuned on 2016-2020 (step 3), so 2018-2020 rows are partly in-sample for the game model;
2022-2024 are the clean seasons.

## Distribution fits per season

| season | margin_dist | total_dist | k1_sigma | t_scale | t_df |
|---|---|---|---|---|---|
| 2018 | k1@2015-2017 | t@2015-2017 | 12.7819 | 13.4565 | 37.8547 |
| 2019 | k1@2015-2018 | t@2015-2018 | 12.9692 | 13.3478 | 19.6002 |
| 2020 | k1@2015-2019 | t@2015-2019 | 13.0511 | 13.3799 | 21.0963 |
| 2021 | k1@2015-2020 | t@2015-2020 | 13.0306 | 13.5007 | 26.4397 |
| 2022 | k1@2015-2021 | t@2015-2021 | 13.1763 | 13.6594 | 39.5093 |
| 2023 | k1@2015-2022 | t@2015-2022 | 13.0175 | 13.6567 | 42.4710 |
| 2024 | k1@2015-2023 | t@2015-2023 | 13.1192 | 13.5567 | 35.5109 |
