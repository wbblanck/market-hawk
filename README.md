# NFL Market Intelligence

Standalone research platform for NFL pre-game spreads and full-game totals. Includes an append-only quote store, market/CLV primitives, point-in-time controls, baseline model utilities, Streamlit dashboard, provider registry, MCP server starter, tests, and staged implementation gates.

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

## Scope
- NFL pre-game spreads and full-game totals only.
- No props, live betting, parlays, futures, or automated wager placement.
- Keep line CLV, price CLV, fair probability, net EV, and execution risk separate.
- Primary validation is chronological walk-forward with a locked holdout.
- Missing Pinnacle benchmarks remain missing; consensus fallbacks must be labelled.
- A displayed quote is not a fill.

See `docs/` for architecture, data contracts, model methodology, known existing-model lessons, resources, dashboard/MCP specifications, and stage gates.
