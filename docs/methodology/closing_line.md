# Closing line and CLV

Defines which snapshot is "the close" and how CLV is computed. Builds on `docs/DATA_CONTRACTS.md`, `docs/ARCHITECTURE.md` and `docs/EXISTING_MODELS.md`; sign conventions are those implemented in `src/nfl_market/markets/clv.py`.

## 1. Rules

| # | Rule |
|---|---|
| L1 | The close is defined by a **preregistered cutoff** relative to `kickoff_utc`, fixed before any CLV result is computed and recorded in config and in each experiment's charter. Changing it opens a new experiment id. |
| L2 | The close is selected by **`captured_at_utc`** (our capture time). `source_timestamp_utc` is retained for provenance only. |
| L3 | **Pinnacle** is the benchmark book. Fallback tiers (sharp median, then all-book median) are allowed only when explicitly labelled per row; they are never silently substituted. |
| L4 | **Missing close stays missing.** No zero, no carry-back from a later quote, no substitution from another market or line. The row is excluded from CLV metrics and counted. |
| L5 | **Line CLV and price CLV are separate quantities**, stored and reported side by side, never summed. |
| L6 | Market snapshots are timestamped at capture and never backfilled. Provider statistics (aggregated historical lines, nflverse `spread_line`/`total_line`) are not historical quotes and never act as our close. |

## 2. Close snapshot definition

```text
cutoff_utc     = kickoff_utc - CLOSE_CUTOFF            (CLOSE_CUTOFF = 15 min, locked)
close quote    = last quote with captured_at_utc <= cutoff_utc
                 and captured_at_utc >= cutoff_utc - MAX_CLOSE_AGE  (else: not qualifying, reason "stale")
                 and is_live = false and synthetic = false
                 same event_id, market, selection; per book, then tiered (table below)
kickoff_utc    = the kickoff known at capture; if the game is rescheduled, the close is re-derived from the new kickoff and the change is flagged (KICKOFF_CHANGED)
```

- A book's **close snapshot** is all of its rows for the event and market captured at the latest `captured_at_utc <= cutoff_utc` at which it quoted the selection. Quotes after the cutoff are ignored, even when the pre-cutoff quote is stale (no carry-back).
- **Price close** requires an **exact-contract match**: same event, market, selection and exact line as the bet (`docs/DATA_CONTRACTS.md`: -2.5 and -3 are different contracts; contract key in `src/nfl_market/markets/contracts.py`). The fair probability is the two-way devig against the opposite side at the paired line in the same snapshot; a book without the opposite side does not qualify (`no_opposite_side`).
- **Line close** is each book's closing **main line** for the same event, market and selection. Main-line rule (locked, `book_primary_else_most_balanced`): the line the book flags as main/primary (`is_main` / `is_primary` column); else the line whose two sides' devigged prices are closest to 50/50 (ties: lower overround, then lower line). Open point for Will: a feed's main flag can be wrong (OddsLink flagged a junk Circa line in `nfl_ats`, see `docs/data_sources/sharp_feeds.md`); no guard is applied yet. The rule applied is stored per row (`main_line_rule` = `book_primary`, `most_balanced`, or `mixed` when median books differ).
- Line close and price close each walk the tiers independently and each row records its own `close_tier`.

### Close tiers (locked by Will, 2026-11-19)

| Tier | `close_tier` | Source | Requirement | Flags | Reporting |
|---|---|---|---|---|---|
| 1 | `pinnacle` | Pinnacle's own close quote | Pinnacle has a qualifying close | none | primary results |
| 2 | `sharp_median` | median of Circa and BookMaker | at least 2 of those books qualify (so both) | none (tier recorded) | primary results, tier mix shown |
| 3 | `all_book_median` | median across all books with a qualifying close | at least 1 book | `MARKET_MEDIAN_FALLBACK`; `FEW_BOOKS` below the fallback minimum book count | **reported separately** as a weak fallback |
| - | `missing` | none | no tier qualifies | `missing_reason` (e.g. `stale`, `no_quote_before_cutoff`) | excluded from CLV metrics, counted |

A tier-2 attempt with only one sharp book falls to tier 3 (the lone sharp book is then one of the tier-3 books). Every non-tier-1 row stores `fallback_reason` naming why each higher tier was skipped (e.g. `pinnacle: 0 of 1 required books (pinnacle=stale)`). Median tiers store no single odds; for a price close they store the median devigged probability. Flag names follow `docs/EXISTING_MODELS.md`. Book names are matched case-insensitively (`Circa Sports`, `circasports` -> `circa`; `BookMaker.eu` -> `bookmaker`).

### Parameters

| Parameter | Value | Status | Setting (`NFLMI_` env prefix) |
|---|---|---|---|
| `CLOSE_CUTOFF` | 15 min before kickoff | **locked** (Will) | `close_cutoff_minutes` |
| Tier order and books | Pinnacle; Circa + BookMaker; all books | **locked** (Will) | `config.default_tiers` |
| Tier-2 minimum books | 2 | **locked** (Will) | `close_sharp_min_books` |
| Main-line rule | `book_primary_else_most_balanced` | **locked** (Will) | `close_main_line_rule` |
| `MAX_CLOSE_AGE` | 120 min | locked (Will, 2026-11-19) | `close_max_age_minutes` |
| Tier-3 minimum book count (`FEW_BOOKS` below) | 5 (same as `nfl_ats`) | locked (Will, 2026-11-19) | `close_fallback_min_books` |

The policy lives in `src/nfl_market/config.py` (`ClosePolicy`, `Settings.close_policy()`); every close row carries `policy_fingerprint` so a charter can pin the exact policy. Changing any value after CLV results exist opens a new experiment id (rule L1). Enforcement: `src/nfl_market/markets/closing.py` (`select_main_line`, `select_close`, `close_table`), tested in `tests/test_closing.py` and `tests/test_config_close.py`. Where the Pinnacle, Circa and BookMaker quotes can come from: `docs/data_sources/sharp_feeds.md`.

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

For every CLV table: close tier mix (`pinnacle` / `sharp_median` / `all_book_median` / `missing` counts; `all_book_median` rows also reported separately), policy fingerprint, cutoff used, quote age distribution, line CLV and price CLV in separate columns, N in games, game-clustered intervals (`validation_design.md` Section 6). A quote is not a fill (`docs/METHODOLOGY.md`): CLV on quotes is a market-timing measure, not realised execution.
