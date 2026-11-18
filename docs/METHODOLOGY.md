# Methodology and hypotheses
H1: point-in-time market-state features improve out-of-sample CLV prediction over persistence and simple market-movement baselines.
H2: gains survive fees, slippage, quote age, liquidity and missed fills.
H3: performance is stable across seasons, books, spreads/totals and time-to-kickoff buckets.

Start with prior/persistence baselines, logistic regression for P(positive CLV), and Ridge for CLV magnitude. Only then test LightGBM/XGBoost and quantile regression. Hierarchical effects are later-stage for sparse book/season data. Negative binomial is not a default CLV model.

Use chronological expanding-window walk-forward, purge overlapping label windows, cluster uncertainty by game, and lock the final holdout. Report Brier/log loss, calibration, MAE/RMSE, quantile coverage and economic outcomes separately. Compare to no-model and market baselines.

A missing close is missing, not zero. A quote is not a fill. Until measured order/fill data exists, execution outputs are hypothetical.
