"""Ridge (rung 2) and shared machinery for feature-based margin/total models.

Training rows: labelled games in the training window from the second available
season on (the first season only seeds history). Features: ``features.team_features``
(point-in-time, built from the training games only; scores of predicted games
are never read).

Fold-local parts: median imputer with missingness indicators, StandardScaler,
Ridge alpha. Alpha is chosen per target on an inner season split inside the
training window: fit on seasons < V, score MAE on season V, where V is the most
recent training season whose Super Bowl is in the training window. The same
inner split gives an out-of-sample residual SD, used as ``pred_*_sd`` and for
``pred_home_win_prob = Phi(pred_margin / pred_margin_sd)``. The choice is cached
per V (i.e. once per outer fold), then the model is refit weekly on all
training rows with that alpha.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nfl_market.models.features import FEATURE_COLUMNS, team_features
from nfl_market.models.folds import labelled, prediction_frame, train_end

TARGETS = ("margin", "total")


def _targets(g: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"margin": (g["home_score"] - g["away_score"]).to_numpy(float),
            "total": (g["home_score"] + g["away_score"]).to_numpy(float)}


class FeatureModel:
    """Base: weekly refit on point-in-time features, inner-season SD/hyperparameter selection.

    ``extra`` names columns from ``features.external`` (nflverse EPA and QB status) appended to
    ``features``; they are built from the same training history, so the as-of rule is unchanged.
    """

    model_id = "feature_base"
    features: tuple[str, ...] = FEATURE_COLUMNS
    extra: tuple[str, ...] = ()

    def __init__(self):
        self._inner_cache: dict = {}
        self._feat_cache: pd.DataFrame | None = None
        self._feat_hist: dict[str, tuple[float, float]] = {}
        self._feat_max_label = None

    def _train_features(self, hist: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
        """Features for training rows, reusing rows computed at an earlier refit when provably unchanged.

        A cached row r has T_r < label_r <= max old label. If the new history
        contains the old one with identical scores and every added game has a
        label after the old maximum, no added game can enter r's as-of window,
        so r's features are unchanged. Otherwise everything is recomputed.
        """
        ok = self._feat_cache is not None
        if ok:
            ids = hist["game_id"].tolist()
            old = np.fromiter((i in self._feat_hist for i in ids), bool, len(ids))
            ok = old.sum() == len(self._feat_hist) and all(
                self._feat_hist[i] == (a, b) for i, a, b in
                zip(hist.loc[old, "game_id"].tolist(), hist.loc[old, "home_score"].tolist(),
                    hist.loc[old, "away_score"].tolist()))
            added = hist.loc[~old, "label_available_at_utc"]
            ok = ok and (added.empty or added.min() > self._feat_max_label)
        cache = self._feat_cache if ok else None
        need = rows if cache is None else rows[~rows["game_id"].isin(cache.index)]
        fresh = self._all_features(hist, need)
        fresh.index = need["game_id"].to_numpy()
        cache = fresh if cache is None else pd.concat([cache, fresh])
        self._feat_cache = cache
        self._feat_hist = dict(zip(hist["game_id"].tolist(), zip(hist["home_score"].tolist(), hist["away_score"].tolist())))
        self._feat_max_label = hist["label_available_at_utc"].max()
        out = cache.loc[rows["game_id"].to_numpy()]
        out.index = rows.index
        return out

    def _all_features(self, hist: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
        base = team_features(hist, rows)
        if not self.extra:
            return base
        if getattr(self, "_ext_hist_ref", None) is not hist:
            from nfl_market.features.external import ExternalFeatures

            self._ext, self._ext_hist_ref = ExternalFeatures(hist), hist
        return base.join(self._ext.features(rows)[list(self.extra)])

    # subclasses implement
    def _make(self, target: str, hp):
        raise NotImplementedError

    def _grid(self, target: str) -> list:
        return [None]

    def _inner(self, X: pd.DataFrame, y: dict, seasons: np.ndarray, val_season: int):
        """Pick hyperparameters per target on (seasons < V) -> V; return (hp, residual sd)."""
        key = (val_season, len(X.columns))
        if key in self._inner_cache:
            return self._inner_cache[key]
        tr, va = seasons < val_season, seasons == val_season
        res = {}
        for t in TARGETS:
            best = None
            for hp in self._grid(t):
                if tr.sum() < 50 or va.sum() < 20:
                    best = (np.inf, hp, np.nan)
                    break
                m = self._make(t, hp).fit(X[tr], y[t][tr])
                r = y[t][va] - m.predict(X[va])
                mae = float(np.mean(np.abs(r)))
                if best is None or mae < best[0]:
                    best = (mae, hp, float(np.sqrt(np.mean(r ** 2))))
            res[t] = (best[1], best[2])
        self._inner_cache[key] = res
        return res

    def fit(self, games: pd.DataFrame) -> "FeatureModel":
        hist = games.loc[labelled(games)].sort_values(["kickoff_utc", "game_id"])
        self.history_ = hist
        self.train_end_ = train_end(hist)
        rows = hist[hist["season"] > hist["season"].min()]
        X = self._train_features(hist, rows)[list(self.features)]
        y = _targets(rows)
        seasons = rows["season"].to_numpy(int)
        complete = hist.loc[hist["game_type"] == "SB", "season"]
        val_season = int(complete.max()) if len(complete) else int(seasons.max())
        self.inner_ = self._inner(X, y, seasons, val_season)
        self.models_ = {t: self._make(t, self.inner_[t][0]).fit(X, y[t]) for t in TARGETS}
        return self

    def predict(self, games: pd.DataFrame) -> pd.DataFrame:
        X = self._all_features(self.history_, games)[list(self.features)]
        pm = self.models_["margin"].predict(X)
        pt = self.models_["total"].predict(X)
        sm, stt = self.inner_["margin"][1], self.inner_["total"][1]
        pw = norm.cdf(pm / sm) if np.isfinite(sm) and sm > 0 else np.nan
        return prediction_frame(games, self.model_id, self.train_end_, pred_margin=pm, pred_total=pt,
                                pred_margin_sd=sm, pred_total_sd=stt, pred_home_win_prob=pw)


class RidgeModel(FeatureModel):
    """GameModel: Ridge on point-in-time team features (alpha by inner season split)."""

    ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)

    def __init__(self, model_id: str = "ridge_v1", extra: tuple[str, ...] = ()):
        super().__init__()
        self.model_id = model_id
        self.extra = tuple(extra)
        self.features = tuple(FEATURE_COLUMNS) + self.extra

    def _grid(self, target):
        return list(self.ALPHAS)

    def _make(self, target, alpha):
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                             StandardScaler(), Ridge(alpha=alpha))


def ridge_v2() -> RidgeModel:
    """``ridge_v2``: ridge_v1 plus nflverse EPA and pre-game QB status (``features.external``)."""
    from nfl_market.features.external import EXTERNAL_COLUMNS

    return RidgeModel("ridge_v2", EXTERNAL_COLUMNS)
