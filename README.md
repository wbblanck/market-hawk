# Market Hawk

NFL market intelligence. Standalone research platform for NFL pre-game spreads and full-game totals. Includes an append-only quote store, fair pricing, edges, paper staking and bet log, a model stack, an odds-capture workflow, market/CLV primitives, point-in-time controls, baseline model utilities, Streamlit dashboard, provider registry, MCP server starter, tests, and staged implementation gates.

**This is a reviewable scaffold, not a validated profitable model.** Provider names in the registry do not establish API access, historical coverage, permission, or executable fills.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,mcp]"
cp .env.example .env
pytest
python scripts/load_demo_data.py
streamlit run apps/dashboard/app.py
```
Demo records are synthetic and clearly marked. Do not use them for betting decisions.

## CLV sign convention
Positive CLV is always favourable to the bettor, and line CLV and price CLV are reported separately, never summed.

| Bet | Close | Line CLV |
|---|---|---|
| Favourite -2.5 | -3 | +0.5 |
| Underdog +3.5 | +3 | +0.5 |
| Over 44.5 | 45.5 | +1.0 |
| Under 46.5 | 45.5 | +1.0 |

Spreads always use the selected team's own spread (`line_clv = bet - close`). Price CLV compares devigged fair probabilities on the same exact line and is left empty when the line moved.

## Market snapshots (data layer)
We build our own market history: each collection run stamps one capture time at receipt (never backfilled from source timestamps) and appends every quote, with its original payload and quality flags, to the `market_snapshots` table. Sportsbook spread and total quotes that pass validation are also routed into `quote_snapshots`.

```bash
pip install -e ".[sports-leader]"
export NFLMI_SPORTS_LEADER_CMD="node /path/to/sports-leader-mcp/dist/index.js"
python scripts/collect_snapshots.py --date 2026-10-11 --season 2026 --week 6
```

[Sports Leader MCP](https://github.com/WalrusQuant/sports-leader-mcp) is optional: without Node, the server, or the `mcp` SDK the provider reports unavailable and everything else runs. It supplies current sportsbook odds (no source timestamps), Kalshi/Polymarket/Polymarket US winner prices, and NFL statistics. Its statistics are not historical sportsbook quotes. Behind a proxy, Node 22 needs `NODE_USE_ENV_PROXY=1`.

Development standards for validation, leakage, features, calibration, closing lines, experiment logs and model cards are in [`docs/methodology/`](docs/methodology/README.md).

## Step 4: prices, edges, paper bets, stacking
All step-4 frames follow one contract, [`src/nfl_market/markets/interface.py`](src/nfl_market/markets/interface.py).

```bash
python scripts/price_slate.py --as-of 2026-11-22T12:00Z      # fair spread/total/moneyline prices -> fair_prices
python scripts/run_stacking.py                               # stack_v1 walk-forward -> docs/results/step4_stacking.md
python scripts/run_market_backtest.py --model elo_v1         # benchmark vs nflverse lines (NOT CLV) -> docs/results/step4_markets.md
```

- `markets.pricing` turns a game prediction plus its outcome distribution (K1 margin, t total) into ternary win/push/loss probabilities and fair odds; `markets.edges` joins them to stored quotes (breakeven, devigged market probability, EV per unit); `markets.staking` sizes paper stakes with fractional Kelly and caps ([`docs/methodology/staking.md`](docs/methodology/staking.md); defaults are proposals); `markets.bet_log` keeps an append-only paper bet log graded against the locked close, storing line CLV and price CLV separately with the tier each came from.
- `models.stacking` blends Elo, Kalman, Ridge and XGBoost with season walk-forward simplex weights. It does not beat the best single model ([results](docs/results/step4_stacking.md)).
- The 2025 holdout is refused by every writer. Forward 2026 prices therefore come from models trained without 2025.
- Paper bets only: there is no wager placement code.

## Odds capture (GitHub Actions)
`.github/workflows/capture-odds.yml` captures ParlayAPI (all books) and OddsLink (Pinnacle, Circa) on a request budget shared with NFL_ATS, and commits encrypted captures to the `market-data` branch. Setup (secrets `PARLAY_API_KEY`, `ODDS_LINK`, `CAPTURE_PASSPHRASE`, plus an external timer for close captures) is in [`docs/data_sources/odds_capture.md`](docs/data_sources/odds_capture.md).

## MCP server
`python -m nfl_market.mcp_server.server` runs over stdio. Read-only tools and resources are listed in [`docs/MCP_SPEC.md`](docs/MCP_SPEC.md) (edges, bet log, CLV summary, model health, fair prices, overview, plus the starter tools); they return the same results as the dashboard. Use `--transport sse` or `--transport streamable-http` (with `--host`/`--port`, default 127.0.0.1:8000) for network transports, which have no authentication. Works with MCP SDK 1.x and 2.x.

## Scope
- NFL pre-game spreads and full-game totals only.
- No props, live betting, parlays, futures, or automated wager placement.
- Keep line CLV, price CLV, fair probability, net EV, and execution risk separate.
- Primary validation is chronological walk-forward with a locked holdout.
- Missing Pinnacle benchmarks remain missing; consensus fallbacks must be labelled.
- A displayed quote is not a fill.

See `docs/` for architecture, data contracts, model methodology, known existing-model lessons, resources, dashboard/MCP specifications, and stage gates.
