# Closing line and CLV

Defines which snapshot is "the close" and how CLV is computed. Builds on `docs/DATA_CONTRACTS.md`, `docs/ARCHITECTURE.md` and `docs/EXISTING_MODELS.md`; sign conventions are those implemented in `src/nfl_market/markets/clv.py`.

## 1. Rules

| # | Rule |
|---|---|
| L1 | The close is defined by a **preregistered cutoff** relative to `kickoff_utc`, fixed before any CLV result is computed and recorded in config and in each experiment's charter. Changing it opens a new experiment id. |
| L2 | The close is selected by **`captured_at_utc`** (our capture time). `source_timestamp_utc` is retained for provenance only. |
| L3 | **Pinnacle** is the benchmark book. A **consensus fallback** is allowed only when explicitly labelled; it is never silently substituted. |
| L4 | **Missing close stays missing.** No zero, no carry-back from a later quote, no substitution from another market or line. The row is excluded from CLV metrics and counted. |
| L5 | **Line CLV and price CLV are separate quantities**, stored and reported side by side, never summed. |
| L6 | Market snapshots are timestamped at capture and never backfilled. Provider statistics (aggregated historical lines, nflverse `spread_line`/`total_line`) are not historical quotes and never act as our close. |

## 2. Close snapshot definition

```text
cutoff_utc     = kickoff_utc - CLOSE_CUTOFF            (preregistered; see open decision below)
close quote    = last quote with captured_at_utc <= cutoff_utc
                 and captured_at_utc >= cutoff_utc - MAX_CLOSE_AGE  (else: missing, reason "stale")
                 and is_live = false and synthetic = false
                 for book = Pinnacle, same event_id, market, selection
kickoff_utc    = the kickoff known at capture; if the game is rescheduled, the close is re-derived from the new kickoff and the change is flagged
```

- **Price close** requires an **exact-contract match**: same event, market, selection and exact line as the bet (`docs/DATA_CONTRACTS.md`: -2.5 and -3 are different contracts; contract key in `src/nfl_market/markets/contracts.py`).
- **Line close** is the benchmark's closing line for the same event, market and selection. When the book shows several lines (alternates), the main line is the one the book flags as primary, else the line with the two sides' prices closest to even; the rule used is stored per row.
- **Fallback**: if Pinnacle has no qualifying close, a consensus close = median line (and median devigged price on the exact contract) across books with a qualifying close quote. Stored with `close_source = consensus_median`, flags `MARKET_MEDIAN_FALLBACK` and, below the preregistered minimum book count, `FEW_BOOKS` (flag names from `docs/EXISTING_MODELS.md`), plus quote age and fallback reason (`docs/DATA_CONTRACTS.md`). Results are reported Pinnacle-only and Pinnacle+fallback separately.

**Open decision (must be set before step-3 CLV results).** `CLOSE_CUTOFF`, `MAX_CLOSE_AGE` and the minimum book count are not defined anywhere in the repo yet (`src/nfl_market/config.py` has no such setting). The earlier workflow used Pinnacle about 20 minutes before kickoff (`docs/EXISTING_MODELS.md`); this repo must preregister its own value. Enforcement code is **to be built in step 3**.

## 3. Line CLV (points)

Positive is always favourable to the bettor. Spreads use the **selected side's own line**, never an ambiguous home-perspective line (`selection_spread(home_line, selection)` converts home/away).

```text
spread (home | away | favorite | underdog):  line_clv = bet_line - close_line
over:                                        line_clv = close_line - bet_line
under:                                       line_clv = bet_line - close_line
```

| Bet | Close | Line CLV |
|---|---|---|
| Favourite -2.5 | -3 | -2.5 - (-3) = **+0.5** |
| Underdog +3.5 | +3 | 3.5 - 3 = **+0.5** |
| Over 44.5 | 45.5 | 45.5 - 44.5 = **+1.0** |
| Under 46.5 | 45.5 | 46.5 - 45.5 = **+1.0** |

**Verified** against `src/nfl_market/markets/clv.py::line_clv` (`return close - bet if side == "over" else bet - close` for totals; `bet - close` for spreads) and `tests/test_clv.py::test_spec_examples`, which parametrizes exactly these four cases; adverse moves are negative in `test_adverse_moves_are_negative`; home/away conversion is tested in `test_home_and_away_use_selected_team_spread`. `pytest tests/test_clv.py` passed (run during this write-up).

Line CLV in points is not a probability: half a point through 3 or 7 is worth far more than half a point through 41.5 on a total. Report line CLV with a key-number-crossing flag, and convert to probability with the game model's margin distribution (`predictive_modeling.md` Section 4) when a value is needed.

## 4. Price CLV (probability)

```text
price_clv = close_fair_p - bet_fair_p      (same exact contract only)
fair_p    = multiplicative two-way devig of (selection, opposite side) American odds
```

Implemented as `clv.py::price_clv(bet_odds, bet_opposite_odds, close_odds, close_opposite_odds)` using `src/nfl_market/markets/odds.py::devig_two_way` (proportional devig) and `probability_clv`, which rejects probabilities outside (0, 1). `clv.py::clv_report` returns `price_clv_probability = None` when bet and close lines differ, and `CLVResult` holds the two metrics side by side without summing them (tested in `test_report_keeps_line_and_price_separate`). Vig level alone does not create price CLV (`test_price_clv_uses_devigged_probabilities`).

## 5. Reporting

For every CLV table: close source mix (Pinnacle / fallback / missing counts), cutoff used, quote age distribution, line CLV and price CLV in separate columns, N in games, game-clustered intervals (`validation_design.md` Section 6). A quote is not a fill (`docs/METHODOLOGY.md`): CLV on quotes is a market-timing measure, not realised execution.
