"""Season walk-forward runner with weekly rolling origins, plus evaluation metrics.

For each outer fold (test season) and each weekly origin: every model is fit on
the origin's training games (labels strictly before the week's first T), then
predicts the week's games from a copy whose scores and label times are blanked,
so a model cannot read the outcomes it predicts. All models share one fold table
(V4). Output passes ``interface.validate_predictions``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from nfl_market.holdout import is_holdout
from nfl_market.models.folds import season_folds, week_origins
from nfl_market.models.interface import PREDICTION_COLUMNS, validate_predictions

BOOTSTRAP_B = 2000


def _blank_outcomes(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    g["home_score"] = np.nan
    g["away_score"] = np.nan
    g["label_available_at_utc"] = pd.NaT
    return g


def run_walkforward(games: pd.DataFrame, models, test_seasons, min_train_seasons: int = 3,
                    verbose: bool = False) -> pd.DataFrame:
    """Run every model over the same season/week fold table. Returns a prediction frame."""
    if is_holdout(games["game_id"]).any():
        raise AssertionError("holdout games passed to run_walkforward")
    games = games.reset_index(drop=True)
    by_id = games.set_index("game_id", drop=False)
    timing = {m.model_id: 0.0 for m in models}
    out = []
    for fold in season_folds(games, test_seasons, min_train_seasons):
        for o in week_origins(games, fold):
            train = by_id.loc[sorted(o.train_game_ids)].reset_index(drop=True)
            test = _blank_outcomes(by_id.loc[sorted(o.test_game_ids)].reset_index(drop=True))
            for m in models:
                t0 = time.perf_counter()
                p = m.fit(train).predict(test)
                timing[m.model_id] += time.perf_counter() - t0
                if set(p["game_id"]) != set(o.test_game_ids) or len(p) != len(o.test_game_ids):
                    raise AssertionError(f"{m.model_id} did not return one row per test game")
                p = p.copy()
                p["model_id"] = m.model_id
                p["fold_id"] = o.fold_id
                p["train_end_utc"] = o.train_end_utc
                out.append(p[list(PREDICTION_COLUMNS)])
        if verbose:
            print(f"fold {fold.fold_id} done; cumulative seconds {', '.join(f'{k}={v:.1f}' for k, v in timing.items())}",
                  flush=True)
    preds = pd.concat(out, ignore_index=True)
    preds["train_end_utc"] = pd.to_datetime(preds["train_end_utc"], utc=True)
    preds.attrs["fit_predict_seconds"] = timing
    return validate_predictions(preds)


# ------------------------------------------------------------------------- metrics
def attach_outcomes(preds: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    g = games[["game_id", "home_score", "away_score", "game_type"]].copy()
    g["margin"] = g["home_score"] - g["away_score"]
    g["total"] = g["home_score"] + g["away_score"]
    return preds.merge(g, on="game_id", how="left", validate="many_to_one")


def _logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def game_metrics(df: pd.DataFrame) -> dict:
    """Metrics on a frame with predictions + outcomes (one row per game). Ties excluded from win metrics."""
    em = df["pred_margin"] - df["margin"]
    et = df["pred_total"] - df["total"]
    nt = df["margin"] != 0
    w = df.loc[nt & df["pred_home_win_prob"].notna()]
    y = (w["margin"] > 0).astype(float).to_numpy()
    p = w["pred_home_win_prob"].to_numpy(float)
    has_total = et.notna().any()
    return {
        "n_games": int(len(df)),
        "mae_margin": float(em.abs().mean()), "rmse_margin": float(np.sqrt((em ** 2).mean())),
        "bias_margin": float(em.mean()),
        "mae_total": float(et.abs().mean()) if has_total else np.nan,
        "rmse_total": float(np.sqrt((et ** 2).mean())) if has_total else np.nan,
        "n_win": int(len(w)),
        "logloss_win": float(_logloss(p, y).mean()) if len(w) else np.nan,
        "brier_win": float(((p - y) ** 2).mean()) if len(w) else np.nan,
    }


def metrics_table(preds: pd.DataFrame, games: pd.DataFrame, by=("model_id", "fold_id")) -> pd.DataFrame:
    df = attach_outcomes(preds, games)
    rows = []
    for key, grp in df.groupby(list(by), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **game_metrics(grp)})
    return pd.DataFrame(rows)


def paired_bootstrap(preds: pd.DataFrame, games: pd.DataFrame, candidate: str, baseline: str,
                     metric: str = "abs_margin", B: int = BOOTSTRAP_B, seed: int = 0) -> dict:
    """Paired game-clustered bootstrap of mean(candidate loss - baseline loss).

    One row per game per model in the pooled OOF set, so resampling games is
    resampling clusters. ``metric``: abs_margin | abs_total | logloss_win.
    Negative difference = candidate better.
    """
    df = attach_outcomes(preds, games)
    if metric == "abs_margin":
        df["loss"] = (df["pred_margin"] - df["margin"]).abs()
    elif metric == "abs_total":
        df["loss"] = (df["pred_total"] - df["total"]).abs()
    elif metric == "logloss_win":
        df = df[df["margin"] != 0].copy()
        df["loss"] = _logloss(df["pred_home_win_prob"].to_numpy(float), (df["margin"] > 0).to_numpy(float))
    else:
        raise ValueError(metric)
    wide = df.pivot_table(index="game_id", columns="model_id", values="loss", aggfunc="first")
    if candidate not in wide or baseline not in wide:
        return {"candidate": candidate, "baseline": baseline, "metric": metric, "n_games": 0,
                "diff": np.nan, "lo": np.nan, "hi": np.nan}
    d = (wide[candidate] - wide[baseline]).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"candidate": candidate, "baseline": baseline, "metric": metric, "n_games": int(len(d)),
            "diff": float(d.mean()), "lo": float(lo), "hi": float(hi)}


def folds_won(fold_metrics: pd.DataFrame, candidate: str, baseline: str, col: str) -> tuple[int, int]:
    w = fold_metrics.pivot(index="fold_id", columns="model_id", values=col)
    if candidate not in w or baseline not in w:
        return 0, 0
    d = (w[candidate] - w[baseline]).dropna()
    return int((d < 0).sum()), int(len(d))
