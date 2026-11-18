# Stages and gates
0. Specification: freeze event/contract IDs, CLV conventions, close definition and preregistration. Gate: approved data dictionary.
1. Data foundation: append-only snapshots, validation, provenance, backfill manifest. Gate: reproducible dataset and mapping QA.
2. Existing model integration: versioned ATS export, preserve CBS-vs-market distinction. Gate: deterministic replay.
3. Market reconstruction: contract matching, same-line best price, devig, benchmark/fallback flags. Gate: hand-checked tests.
4. Features: movement, dispersion, reversals, quote age, time-to-kickoff, as-of contextual features. Gate: leakage tests.
5. CLV models: baselines then logistic/Ridge then boosting/quantiles. Gate: out-of-sample incremental value and calibration.
6. Execution: fees, slippage, latency, fill and missed-fill. Gate: measured data or explicitly hypothetical.
7. Dashboard: full pages and service parity. Gate: reconciled calculations and provenance.
8. MCP: read-only tools/resources, auth and audit. Gate: security review and service parity.
9. Shadow production: live quote capture, no wagering. Gate: data freshness, uptime, immutable predictions.
10. Advisory release: only validated signals with uncertainty. Gate: independent review. No automated wagering in scope.
