"""Plain-language drivers of a ridge model's prediction (``ridge_v1`` / ``ridge_v2``).

The model is a standard-scaled ridge, so each prediction is exactly::

    prediction = intercept + sum_j coef_j * z_j      (z_j = scaled feature value)

``intercept`` is the average game the model was trained on; ``coef_j * z_j`` is how many points feature
``j`` moves this game away from that average. Features are grouped into a few named drivers. The
contributions of the median-imputer's missing-value indicators are added to ``other``. Reading the
result is explanation, not causation: ridge splits credit between correlated features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Team strength (Elo and Kalman ratings)", ("elo", "elo_diff", "elo_d", "elo_fair", "kalman", "kalman_var")),
    ("Recent scoring (last 8 games, season, last year)", ("pf_l8", "pa_l8", "pf_std", "pa_std", "pf_prev", "pa_prev")),
    ("Play-by-play efficiency (EPA per play)", ("off_epa_l8", "def_epa_l8", "off_epa_ewm", "def_epa_ewm",
                                               "off_pass_epa_ewm", "def_pass_epa_ewm")),
    ("Quarterback (listed status and recent EPA)", ("qb_out", "qb_epa")),
    ("Rest, home field and schedule", ("rest", "rest_diff", "neutral", "div_game", "postseason", "is_dome",
                                       "is_retractable", "week", "gp_season")),
    ("League scoring environment", ("league_total_l256", "league_margin_l256", "level")),
)


def _group_of(feature: str) -> str:
    base = feature
    for pre in ("home_", "away_"):
        if feature.startswith(pre):
            base = feature[len(pre):]
    for name, members in GROUPS:
        if base in members:
            return name
    return "Other"


def contributions(model, games: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-game driver table for each target ('margin', 'total') of a fitted :class:`FeatureModel`.

    Returns ``{target: frame}`` indexed like ``games``: one column per driver plus ``intercept`` and
    ``prediction`` (which equals ``model.predict`` up to float error), in points.
    """
    feats = list(model.features)
    X = model._all_features(model.history_, games)[feats]
    out = {}
    for target, pipe in model.models_.items():
        imputer, scaler, ridge = pipe.steps[0][1], pipe.steps[1][1], pipe.steps[2][1]
        Z = scaler.transform(imputer.transform(X))
        contrib = Z * ridge.coef_
        names = feats + [f"missing:{feats[i]}" for i in imputer.indicator_.features_]
        frame = pd.DataFrame(contrib, index=games.index, columns=names)
        grouped = pd.DataFrame(index=games.index)
        for col in frame.columns:
            grouped[_group_of(col.replace("missing:", ""))] = grouped.get(_group_of(col.replace("missing:", "")), 0.0) + frame[col]
        grouped["intercept"] = float(ridge.intercept_)
        grouped["prediction"] = grouped.sum(axis=1)
        out[target] = grouped
    return out


def raw_facts(model, games: pd.DataFrame) -> pd.DataFrame:
    """The unscaled inputs worth quoting (EPA, QB flag, Elo, rest) for each game."""
    cols = ["home_elo", "away_elo", "home_off_epa_l8", "away_off_epa_l8", "home_def_epa_l8", "away_def_epa_l8",
            "home_qb_out", "away_qb_out", "home_qb_epa", "away_qb_epa", "home_rest", "away_rest",
            "home_pf_l8", "away_pf_l8", "home_pa_l8", "away_pa_l8"]
    X = model._all_features(model.history_, games)
    return X[[c for c in cols if c in X.columns]]


def top_drivers(row: pd.Series, k: int = 3) -> list[tuple[str, float]]:
    """The ``k`` largest absolute driver contributions (excluding intercept and prediction)."""
    d = row.drop(labels=[c for c in ("intercept", "prediction") if c in row.index])
    d = d[np.abs(d) >= 0.05].sort_values(key=np.abs, ascending=False)
    return [(str(n), float(v)) for n, v in d.head(k).items()]
