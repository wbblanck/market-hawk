# Data contracts
Quote row: snapshot_id, event_id, season, week, kickoff_utc, captured_at_utc, book, market, selection, line, odds_american, is_live, source, source_quote_id, displayed_liquidity, synthetic.

All timestamps are UTC with original source timestamp retained. Preserve provenance and raw payload. A contract is event + market + selection + exact line. A -2.5 and -3 are not identical contracts. Adjacent-line comparisons must be separately labelled.

Pinnacle close benchmark must have a preregistered cutoff and exact-contract match. Missing Pinnacle remains missing. A consensus fallback is explicitly labelled and never silently substituted. Preserve quote age and fallback reason.

CLV sign convention must be locked by tests. Line CLV and price/probability CLV are stored separately.
