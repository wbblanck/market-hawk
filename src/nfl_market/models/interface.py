"""Shared contracts for step-3 game models and the distributions layer.

This module is the hand-off point between the game models (``nfl_market.models``)
and the outcome distributions (``nfl_market.distributions``). Both sides import
from here; neither imports the other's internals.

Conventions
-----------
* ``margin`` is always home points minus away points (home perspective).
* ``total`` is home points plus away points.
* Every prediction row carries the time it was made and the latest label time
  used to train the model that made it, so point-in-time order can be asserted
  (``validation_design.md`` V2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

# --------------------------------------------------------------------------- games
#: Columns every game frame passed to a model must have (one row per game).
GAME_COLUMNS: tuple[str, ...] = (
    "game_id",            # nflverse id, e.g. 2024_01_BAL_KC
    "season",
    "week",
    "game_type",          # REG, WC, DIV, CON, SB
    "kickoff_utc",        # tz-aware UTC timestamp
    "home_team",
    "away_team",
    "neutral_site",       # bool
    "home_score",         # float, NaN until final
    "away_score",
    "label_available_at_utc",  # when the final score may be used for training
)

# ---------------------------------------------------------------------- predictions
#: Columns of a prediction frame (one row per game per model per fold).
PREDICTION_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "model_id",               # e.g. "elo_v1", "ridge_v1"
    "fold_id",                # e.g. "2022" for the 2022 test season
    "prediction_time_utc",    # T: when this prediction was made
    "train_end_utc",          # max label_available_at_utc of rows used to fit
    "pred_margin",            # expected home margin (float)
    "pred_total",             # expected total points (float or NaN if model has none)
    "pred_margin_sd",         # model's own predictive SD of margin (NaN if none)
    "pred_total_sd",          # model's own predictive SD of total (NaN if none)
    "pred_home_win_prob",     # model's own P(home win) (NaN if none)
)

#: Outcome columns joined onto predictions for evaluation (never model inputs).
OUTCOME_COLUMNS: tuple[str, ...] = ("home_score", "away_score", "margin", "total")


@dataclass(frozen=True)
class FoldSpec:
    """One outer fold of the season walk-forward."""

    fold_id: str
    test_season: int
    train_game_ids: frozenset[str]
    test_game_ids: frozenset[str]
    train_end_utc: pd.Timestamp
    test_start_utc: pd.Timestamp


@runtime_checkable
class GameModel(Protocol):
    """A model that predicts home margin (and optionally total) for games.

    ``fit`` receives only games whose labels were available before the
    earliest prediction time of the games passed to ``predict``. ``predict``
    returns a frame with ``PREDICTION_COLUMNS`` (``model_id``/``fold_id``/
    ``train_end_utc`` may be filled in by the walk-forward runner).
    Implementations must not read scores of the games they predict.
    """

    model_id: str

    def fit(self, games: pd.DataFrame) -> "GameModel": ...

    def predict(self, games: pd.DataFrame) -> pd.DataFrame: ...


def validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Check a prediction frame against the contract. Returns it unchanged."""
    missing = [c for c in PREDICTION_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"prediction frame missing columns: {missing}")
    if frame["pred_margin"].isna().any():
        raise ValueError("pred_margin must not be missing")
    dup = frame.duplicated(["game_id", "model_id", "fold_id"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicate (game_id, model_id, fold_id) rows")
    late = frame["train_end_utc"] >= frame["prediction_time_utc"]
    if late.any():
        raise ValueError(f"point-in-time leakage: {int(late.sum())} rows trained on labels at/after T")
    return frame
