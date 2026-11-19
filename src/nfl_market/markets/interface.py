"""Step-4 market interface: the hand-off between pricing/execution, stacking and the read layer.

Three workstreams meet here and nowhere else:

* **markets/execution** (``nfl_market.markets.pricing``, ``edges``, ``staking``, ``bet_log``,
  ``reader``) turns game predictions plus a fitted outcome distribution into fair prices,
  joins them to stored sportsbook quotes (edges), sizes paper stakes and keeps a bet log
  graded against the locked close policy (``nfl_market.config.ClosePolicy``).
* **stacking** (``nfl_market.models.stacking``) writes stacked predictions that satisfy
  ``nfl_market.models.interface.PREDICTION_COLUMNS`` (model id ``STACK_MODEL_ID``) and a
  model-health table with ``MODEL_HEALTH_COLUMNS``.
* **dashboard + MCP** read everything through a :class:`MarketReader`; they never compute
  prices, edges or CLV themselves.

Conventions (inherited, locked)
-------------------------------
* ``game_id`` is the nflverse id (``2026_12_BUF_KC``). Stored quotes use the same value in
  ``event_id`` once mapped (``integrations.sports_leader.mapping``). Frames here call it
  ``game_id``; a quote's ``event_id`` is renamed on the way in.
* ``selection`` is lower case: spread ``home``/``away``, total ``over``/``under``,
  moneyline ``home``/``away``. ``line`` is the selected side's own line (home -3 and away +3
  are two rows); moneyline ``line`` is NaN.
* Probabilities are ternary: ``p_win + p_push + p_loss == 1``. A push returns the stake. A
  moneyline tie is a push (two-way moneyline). ``p_win_no_push = p_win / (1 - p_push)``.
* Line CLV (points) and price CLV (probability) are separate columns, never summed. Every
  CLV value carries the close tier it came from; ``all_book_median`` is the weak fallback
  and is reported separately from ``pinnacle``/``sharp_median``. Missing closes stay NaN.
* 2025 (holdout manifest ``nfl_games_2025_v1``) never enters pricing, edges, stakes or the
  bet log. Writers must refuse holdout ``game_id``s (``nfl_market.holdout.is_holdout``).
* Every row carries a timestamp, a model id and, where relevant, quality flags
  (``docs/DASHBOARD_SPEC.md``). Timestamps are tz-aware UTC.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import pandas as pd

STACK_MODEL_ID = "stack_v1"

PRICED_MARKETS: tuple[str, ...] = ("spread", "total", "moneyline")
SELECTIONS = {"spread": ("home", "away"), "total": ("over", "under"), "moneyline": ("home", "away")}

# ---------------------------------------------------------------------- fair prices
#: One row per game x model x market x selection x line. Written by ``markets.pricing``.
PRICE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "kickoff_utc",
    "model_id",               # game model that produced pred_margin / pred_total (e.g. elo_v1, stack_v1)
    "dist_id",                # distribution used, e.g. "k1@2015-2024" (family @ training seasons)
    "prediction_time_utc",    # T of the underlying prediction
    "market",                 # spread | total | moneyline
    "selection",
    "line",                   # selected side's own line; NaN for moneyline
    "p_win",
    "p_push",
    "p_loss",
    "p_win_no_push",
    "fair_american",          # unrounded fair odds for p_win_no_push (distributions.markets.fair_american)
)

# ---------------------------------------------------------------------------- edges
#: One row per stored quote x model. Written by ``markets.edges``.
EDGE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "kickoff_utc",
    "captured_at_utc",        # the quote's capture time (never the source timestamp)
    "quote_age_minutes",      # as_of - captured_at_utc
    "book",                   # canonical book (markets.closing.canonical_book)
    "market",
    "selection",
    "line",
    "odds_american",
    "opposite_odds_american", # same book, same snapshot, paired line; NaN if absent
    "model_id",
    "dist_id",
    "p_win",
    "p_push",
    "p_loss",
    "p_win_no_push",
    "breakeven_prob",         # implied probability of odds_american (stake back on push)
    "market_fair_prob",       # this book's devigged probability on this exact contract; NaN if no opposite
    "edge_prob",              # p_win_no_push - breakeven_prob
    "edge_vs_market",         # p_win_no_push - market_fair_prob (NaN if no opposite)
    "ev_per_unit",            # p_win * (decimal - 1) - p_loss, per 1 unit staked
    "flags",                  # comma-separated quality flags, "" when clean
    "synthetic",              # True for demo data
)

# -------------------------------------------------------------------------- bet log
#: Paper bets only (no wager placement, ever). One row per recorded bet. Written by ``markets.bet_log``.
BET_LOG_COLUMNS: tuple[str, ...] = (
    "bet_id",                 # deterministic hash of game/market/selection/line/book/decision time/model
    "game_id",
    "season",
    "week",
    "kickoff_utc",
    "decision_time_utc",      # when the bet was recorded; must be < kickoff - close cutoff
    "quote_captured_at_utc",
    "book",
    "market",
    "selection",
    "line",
    "odds_american",
    "opposite_odds_american",
    "model_id",
    "dist_id",
    "p_win",
    "p_push",
    "ev_per_unit",
    "kelly_fraction",         # full-Kelly fraction before the multiplier and caps
    "stake_units",            # after multiplier and every limit
    "stake_limit_reason",     # which limit bound ("" if none): kelly | max_bet | max_game | max_week | min_edge | existing_bet
    "status",                 # open | closed (close attached) | graded (result attached) | void
    # --- close and CLV (filled once the close is known; NaN / "" until then) ---
    "line_close",             # main-line close for the selection (selected side's own line)
    "line_close_tier",        # pinnacle | sharp_median | all_book_median | missing
    "line_clv",               # points, positive favourable (markets.clv.line_clv)
    "price_close_fair_prob",  # devigged close on the exact bet contract
    "price_close_tier",
    "price_clv",              # probability, close_fair - bet_fair; NaN when no exact-contract close
    "close_flags",            # union of the two closes' flags
    "close_policy_fingerprint",
    # --- result ---
    "result",                 # win | push | loss | "" (ungraded)
    "pnl_units",
    "synthetic",
)

#: Aggregated CLV, one row per group. Written by ``markets.bet_log.clv_summary``.
CLV_SUMMARY_COLUMNS: tuple[str, ...] = (
    "group",                  # e.g. "close_tier=pinnacle|market=spread"
    "close_tier",
    "market",
    "n_bets",
    "n_line_clv",
    "mean_line_clv",
    "share_positive_line_clv",
    "n_price_clv",
    "mean_price_clv",
    "share_positive_price_clv",
    "is_weak_fallback",       # True for all_book_median rows (reported separately)
)

# ---------------------------------------------------------------------- model health
#: One row per model x scope. Written by ``models.stacking`` (docs/results/step4_model_health.json).
MODEL_HEALTH_COLUMNS: tuple[str, ...] = (
    "model_id",
    "scope",                  # "pooled" | "fold" | "clean" (seasons with no ported-constant tuning)
    "fold_id",                # test season as string; "" for pooled / clean
    "n_games",
    "mae_margin",
    "mae_total",
    "logloss_win",
    "brier_win",
    "ece_win",                # expected calibration error of P(home win), 10 equal-width bins
    "cover_logloss",          # log loss of P(home cover | no push) at the benchmark line; NaN if not computed
    "stack_weight_margin",    # latest walk-forward weight in the stack; NaN for non-components
    "stack_weight_total",
    "experiment_id",
    "generated_at_utc",
)


def empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    """An empty frame with the given contract columns (for no-data reads)."""
    return pd.DataFrame({c: pd.Series(dtype=object) for c in columns})


def check_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> pd.DataFrame:
    """Raise if ``frame`` lacks a contract column; return it ordered by the contract."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} frame missing columns: {missing}")
    return frame[list(columns)]


@runtime_checkable
class MarketReader(Protocol):
    """Read-only access used by the dashboard and the MCP server (identical results in both).

    Every method returns a frame with exactly the contract columns (possibly empty), never
    raises for "no data", and never writes.
    """

    def prices(self, game_ids: Sequence[str] | None = None) -> pd.DataFrame:
        """Latest fair prices (``PRICE_COLUMNS``)."""
        ...

    def edges(self, as_of: pd.Timestamp | None = None, *, min_ev: float | None = None) -> pd.DataFrame:
        """Edges of the latest quote per book/contract at ``as_of`` (``EDGE_COLUMNS``), best EV first."""
        ...

    def bet_log(self, season: int | None = None) -> pd.DataFrame:
        """Paper bet log (``BET_LOG_COLUMNS``), newest decision first."""
        ...

    def clv_summary(self, by: Sequence[str] = ("close_tier", "market")) -> pd.DataFrame:
        """CLV aggregates (``CLV_SUMMARY_COLUMNS``); weak-fallback tier rows flagged."""
        ...

    def model_health(self) -> pd.DataFrame:
        """Model health rows (``MODEL_HEALTH_COLUMNS``)."""
        ...


__all__ = [
    "BET_LOG_COLUMNS", "CLV_SUMMARY_COLUMNS", "EDGE_COLUMNS", "MODEL_HEALTH_COLUMNS", "MarketReader",
    "PRICED_MARKETS", "PRICE_COLUMNS", "SELECTIONS", "STACK_MODEL_ID", "check_columns", "empty_frame",
]
