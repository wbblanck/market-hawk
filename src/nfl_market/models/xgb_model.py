"""XGBoost margin and total (rung 3) on point-in-time features.

Hyperparameters fixed a priori from wbblanck/nfl-betting-model ``src/nflbet/ats/xgb.py``
PARAMS (pre-registered in its ``docs/batch4_preregistration.md``): squared error,
max_depth 3, eta 0.03, subsample 0.8, colsample_bytree 0.8, min_child_weight 20,
lambda 1, seed 0, and 100 trees (the count that repo chose by leave-one-season-out
MAE on 2016-2021 and froze in ``data/backtests/batch4_model.json``; V5 caveat: test
folds 2016-2021 are not nested for the tree count). Features: ``features.XGB_FEATURES``
(their list minus QB; points-based stand-ins for EPA). Missing values stay NaN (XGBoost's
native missing-value split). No early stopping. Refit every week. The inner
season split (seasons < V -> V, V = last complete training season) is used only
to estimate the out-of-sample residual SD for ``pred_*_sd`` and
``pred_home_win_prob``.
"""
from __future__ import annotations

from nfl_market.models.features import XGB_FEATURES
from nfl_market.models.ridge_model import FeatureModel

XGB_PARAMS = dict(n_estimators=100, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=20, reg_lambda=1.0, tree_method="hist", random_state=0, n_jobs=1,
                  objective="reg:squarederror")


class XGBModel(FeatureModel):
    features = XGB_FEATURES

    def __init__(self, params: dict | None = None, model_id: str = "xgb_v1", extra: tuple[str, ...] = ()):
        super().__init__()
        self.params = dict(XGB_PARAMS, **(params or {}))
        self.model_id = model_id
        self.extra = tuple(extra)
        self.features = tuple(XGB_FEATURES) + self.extra

    def _make(self, target, hp):
        from xgboost import XGBRegressor

        return XGBRegressor(**self.params)


def xgb_v2() -> XGBModel:
    """``xgb_v2``: xgb_v1 (same a-priori settings) plus nflverse EPA and pre-game QB status."""
    from nfl_market.features.external import EXTERNAL_COLUMNS

    return XGBModel(model_id="xgb_v2", extra=EXTERNAL_COLUMNS)
