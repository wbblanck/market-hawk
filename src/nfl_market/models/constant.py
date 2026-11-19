"""Rung-0 constant baseline: training-window mean home margin, mean total, home win rate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_market.models.folds import labelled, prediction_frame, train_end


class ConstantModel:
    """GameModel: predicts training means (optionally over the last ``window_seasons`` seasons).

    Home win rate counts ties as 0.5. SDs are the training sample SDs.
    """

    def __init__(self, window_seasons: int | None = None, model_id: str = "constant_v1"):
        self.window_seasons = window_seasons
        self.model_id = model_id

    def fit(self, games: pd.DataFrame) -> "ConstantModel":
        g = games.loc[labelled(games)]
        self.train_end_ = train_end(g)
        if self.window_seasons is not None and len(g):
            g = g[g["season"] > g["season"].max() - self.window_seasons]
        m = (g["home_score"] - g["away_score"]).to_numpy(float)
        t = (g["home_score"] + g["away_score"]).to_numpy(float)
        if len(m) == 0:
            raise ValueError("ConstantModel.fit needs at least one labelled game")
        self.margin_, self.total_ = float(m.mean()), float(t.mean())
        self.margin_sd_ = float(m.std(ddof=1)) if len(m) > 1 else np.nan
        self.total_sd_ = float(t.std(ddof=1)) if len(t) > 1 else np.nan
        self.home_win_ = float(np.mean(np.where(m > 0, 1.0, np.where(m < 0, 0.0, 0.5))))
        return self

    def predict(self, games: pd.DataFrame) -> pd.DataFrame:
        return prediction_frame(games, self.model_id, self.train_end_,
                                pred_margin=self.margin_, pred_total=self.total_,
                                pred_margin_sd=self.margin_sd_, pred_total_sd=self.total_sd_,
                                pred_home_win_prob=self.home_win_)
