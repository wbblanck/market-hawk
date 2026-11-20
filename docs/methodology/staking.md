# Paper staking and the bet log (step 4)

Paper/shadow only. Nothing in Market Hawk places, routes or simulates a wager with a sportsbook.
Code: `src/nfl_market/markets/{edges,staking,bet_log}.py`; policy: `nfl_market.config.StakePolicy`.

## Status of every value

All staking values below are **proposed, not locked by Will**. They change stake sizes only, never
which quote is the close or any CLV number. The close policy (`ClosePolicy`) is locked and separate.

| Setting (`NFLMI_STAKE_*`) | Default | Meaning |
|---|---|---|
| `bankroll_units` | 100 | fixed paper bankroll; no compounding |
| `kelly_multiplier` | 0.25 | fraction of full Kelly |
| `min_ev` | 0.02 | minimum EV per unit staked |
| `max_bet_units` | 1.0 | per bet |
| `max_game_units` | 2.0 | all bets on one game, existing bets included |
| `max_week_units` | 8.0 | all bets in one season-week, existing bets included |
| `max_quote_age_minutes` | 180 | edge screen: older quotes are dropped (or flagged `STALE_QUOTE`) |

## Edges

For each (game, book, market, selection, line) the latest quote captured at or before `as_of` is
joined on the exact contract to the latest fair price made at or before `as_of`. Live quotes,
quotes captured at or after kickoff - 15 min (the locked close cutoff), games whose cutoff has
passed and quotes older than `max_quote_age_minutes` are excluded. `opposite_odds_american` is the
same book's paired side in the same snapshot (spread `h` with `-h`; totals and moneylines the same
line). `market_fair_prob` is that pair devigged multiplicatively. `ev_per_unit = p_win (d - 1) - p_loss`
(a push returns the stake). Flags: `NO_OPPOSITE`, `STALE_QUOTE`, `SUPERSEDED` (the book later posted
the market without this line), `SYNTHETIC`.

## Kelly with pushes

At decimal odds `d`, `b = d - 1`, staking fraction `f`:
`G(f) = p_win log(1 + f b) + p_push log 1 + p_loss log(1 - f)`. The push term is zero. Setting
`G'(f) = 0` gives `f* = (p_win b - p_loss) / (b (p_win + p_loss))`, the two-outcome Kelly at
`q = p_win / (1 - p_push)`. With no push mass this is `(p b - (1 - p)) / b`
(e.g. `p = 0.55` at -110: `(0.55 x 0.90909 - 0.45) / 0.90909 = 0.055`). With push mass it is
slightly larger than `(p_win b - p_loss) / b`, because the push mass puts nothing at risk.
Negative values are clipped to zero. Note for the integrator: the step-4 brief wrote
`(p_win b - p_loss) / b`; that formula is exact only when `p_push = 0` and we use the derived one.

## Sizing order

1. drop edges flagged `STALE_QUOTE` or `SUPERSEDED`; edges from several models must be narrowed to one;
2. `min_edge`: `ev_per_unit < min_ev` gets no stake;
3. **correlated-bet rule**: per game at most one *margin* bet (spread or moneyline, either side) and
   one *total* bet, the highest-EV candidate of each. Home spread and home moneyline are one position on
   the margin and are never two full stakes; home moneyline with away spread is a middle and is also
   excluded. A game/group that already holds a recorded bet gets nothing (`existing_bet`);
4. `kelly`: `kelly_multiplier x f* x bankroll_units`;
5. `max_bet`; 6. `max_game` (in EV order); 7. `max_week` (in EV order).

`stake_limit_reason` records the step that bound last.

## Bet log

`paper_bets` (decisions) and `paper_bet_grades` (close and result rows) live in the quote-store
DuckDB file and are append-only. `bet_id` hashes game, market, selection, line, book, decision time
and model, so re-recording a decision is a no-op. Writers refuse 2025 holdout ids and decisions at
or after kickoff - 15 min. Each bet gets two closes from `markets.closing.select_close`:

* line close (main line of the selected side) -> `line_clv` in points;
* price close (the exact bet contract) -> `price_clv = close_fair - bet_fair` where `bet_fair` is the
  bet book's own devigged pair at decision time.

Each close keeps its own tier (`pinnacle`, `sharp_median`, `all_book_median`, `missing`); the two CLVs
are never summed. `clv_summary` always groups by tier, marks `all_book_median` rows `is_weak_fallback`,
and counts missing closes in `n_bets` while excluding them from means. Moneylines have no line close
(`NO_LINE_CLV_MONEYLINE`). Synthetic quotes never qualify as a close, so demo closes are written as
synthetic rows by `scripts/load_demo_data.py`.

No historical CLV exists: Will declined paid historical sharp odds, so CLV starts with the games we
capture. The nflverse benchmark in `docs/results/step4_markets.md` is not CLV.
