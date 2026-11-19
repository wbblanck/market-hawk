"""Step-4 stacking: simplex-weighted blends of the step-3 game models (``stack_v1``).

Two separate stacks are fit, because Kalman has no total:

* **margin** over ``elo_v1``, ``kalman_gs_v1``, ``ridge_v1``, ``xgb_v1`` (``MARGIN_COMPONENTS``);
* **total** over ``elo_v1``, ``ridge_v1``, ``xgb_v1`` (``TOTAL_COMPONENTS``).

Walk-forward rule (season level, no weekly refit inside a season): the weights for
test season ``S`` are fit only on OOF rows of seasons ``< S`` whose
``prediction_time_utc`` and ``label_available_at_utc`` are strictly before the
first prediction time of ``S`` (:func:`nfl_market.distributions.evaluate.training_rows`).
At least ``MIN_TRAIN_SEASONS`` (3) such seasons are required; seasons with fewer
are not scored. Weights are non-negative and sum to 1 (simplex); the primary
objective is squared error (``objective="squared"``), solved exactly by enumerating
active sets of the KKT system. ``objective="mae"`` (a linear programme) is a reported
variant. An intercept is off by default.

Win probability: the stacked margin is mapped through a K1 margin distribution
(``distributions.evaluate.fit_distribution(..., "k1")``) fit on the same training
rows, where the training rows' stack margin uses season ``S``'s weights (so those
weights are in-sample on the K1 training rows, but nothing from ``S`` or later is
used). ``pred_home_win_prob`` is ``P(M > 0) / (1 - P(M = 0))`` (home win given no
tie), matching how win metrics drop ties.

:class:`StackModel` is the deployable :class:`~nfl_market.models.interface.GameModel`
holding component models and fixed weights (``config/models/stack_v1_weights.json``).
The 2025 holdout is refused everywhere (``nfl_market.holdout.is_holdout``).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from nfl_market.distributions import scoring_models as SM
from nfl_market.distributions.evaluate import FittedDistribution, build_pmfs, fit_distribution, training_rows
from nfl_market.holdout import REPO_ROOT, is_holdout
from nfl_market.markets.interface import STACK_MODEL_ID
from nfl_market.models.interface import PREDICTION_COLUMNS, validate_predictions

MARGIN_COMPONENTS: tuple[str, ...] = ("elo_v1", "kalman_gs_v1", "ridge_v1", "xgb_v1")
TOTAL_COMPONENTS: tuple[str, ...] = ("elo_v1", "ridge_v1", "xgb_v1")
EQUAL_MODEL_ID = "stack_eq_v1"
MIN_TRAIN_SEASONS = 3
OBJECTIVES = ("squared", "mae")
DEFAULT_WEIGHTS_FILE = REPO_ROOT / "config" / "models" / "stack_v1_weights.json"


# ------------------------------------------------------------------ guards
def refuse_holdout(game_ids, where: str) -> None:
    """Raise if any id is in the 2025 holdout manifest."""
    ids = pd.Series(list(game_ids), dtype=object)
    if len(ids) and is_holdout(ids).any():
        raise ValueError(f"{where}: refusing {int(is_holdout(ids).sum())} holdout game_id(s)")


# ------------------------------------------------------------ simplex fits
def _simplex_squared(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """min ||y - Xw||^2 s.t. w >= 0, sum w = 1, by enumerating active sets (exact for a convex QP)."""
    n, k = X.shape
    if k > 10:
        raise ValueError("active-set enumeration supports at most 10 components")
    best, best_sse = None, np.inf
    for r in range(1, k + 1):
        for subset in itertools.combinations(range(k), r):
            Xs = X[:, subset]
            A = np.zeros((r + 1, r + 1))
            A[:r, :r] = 2 * Xs.T @ Xs
            A[:r, r] = 1.0
            A[r, :r] = 1.0
            rhs = np.concatenate([2 * Xs.T @ y, [1.0]])
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
            ws = sol[:r]
            if (ws < -1e-10).any():
                continue
            w = np.zeros(k)
            w[list(subset)] = np.clip(ws, 0, None)
            w /= w.sum()
            sse = float(((y - X @ w) ** 2).sum())
            if sse < best_sse - 1e-9:
                best, best_sse = w, sse
    return best


def _simplex_mae(X: np.ndarray, y: np.ndarray, intercept: bool) -> tuple[np.ndarray, float]:
    """min sum |y - b - Xw| s.t. w >= 0, sum w = 1 as a linear programme (HiGHS)."""
    from scipy import sparse
    from scipy.optimize import linprog

    n, k = X.shape
    nb = 1 if intercept else 0
    c = np.concatenate([np.zeros(k + nb), np.ones(2 * n)])
    eye = sparse.identity(n, format="csr")
    blocks = [sparse.csr_matrix(X)] + ([sparse.csr_matrix(np.ones((n, 1)))] if intercept else []) + [eye, -eye]
    A_eq = sparse.vstack([sparse.hstack(blocks),
                          sparse.csr_matrix(np.concatenate([np.ones(k), np.zeros(nb + 2 * n)])[None, :])]).tocsr()
    b_eq = np.concatenate([y, [1.0]])
    bounds = [(0, None)] * k + [(None, None)] * nb + [(0, None)] * (2 * n)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"MAE simplex fit failed: {res.message}")
    w = np.clip(res.x[:k], 0, None)
    return w / w.sum(), float(res.x[k]) if intercept else 0.0


def simplex_weights(X, y, *, objective: str = "squared", intercept: bool = False) -> tuple[np.ndarray, float]:
    """Simplex weights (non-negative, summing to 1) and intercept (0 unless ``intercept``)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.ndim != 2 or len(X) != len(y) or len(y) == 0:
        raise ValueError("X must be (n, k) with n = len(y) > 0")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must be finite")
    if objective == "mae":
        return _simplex_mae(X, y, intercept)
    if objective != "squared":
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    if not intercept:
        return _simplex_squared(X, y), 0.0
    xm, ym = X.mean(axis=0), y.mean()
    w = _simplex_squared(X - xm, y - ym)
    return w, float(ym - xm @ w)


# ------------------------------------------------------------- OOF frames
def wide_oof(oof: pd.DataFrame, games: pd.DataFrame, components: Sequence[str], target: str) -> pd.DataFrame:
    """One row per game with each component's ``pred_<target>`` as a column, plus timing and outcome.

    Only games where every component has a finite prediction are kept. Columns:
    ``game_id, season, week, prediction_time_utc, train_end_utc`` (max over components),
    ``label_available_at_utc``, ``home_score``, ``away_score``, ``margin``, ``total`` and one per component.
    """
    if target not in ("margin", "total"):
        raise ValueError("target must be 'margin' or 'total'")
    refuse_holdout(oof["game_id"], "wide_oof")
    sub = oof[oof["model_id"].isin(components)]
    col = f"pred_{target}"
    wide = sub.pivot_table(index="game_id", columns="model_id", values=col, aggfunc="first")
    missing = [c for c in components if c not in wide.columns]
    if missing:
        raise ValueError(f"OOF frame has no {col} for components {missing}")
    wide = wide[list(components)].dropna()
    meta = sub.groupby("game_id").agg(season=("season", "first"), week=("week", "first"),
                                      prediction_time_utc=("prediction_time_utc", "min"),
                                      t_max=("prediction_time_utc", "max"),
                                      train_end_utc=("train_end_utc", "max"))
    if (meta["prediction_time_utc"] != meta["t_max"]).any():
        raise ValueError("components disagree on prediction_time_utc for some games")
    g = games[["game_id", "label_available_at_utc", "home_score", "away_score"]].set_index("game_id")
    out = meta.drop(columns="t_max").join(g, how="left").join(wide, how="inner").reset_index()
    out["margin"] = out["home_score"] - out["away_score"]
    out["total"] = out["home_score"] + out["away_score"]
    return out.sort_values(["prediction_time_utc", "game_id"]).reset_index(drop=True)


@dataclass(frozen=True)
class SeasonWeights:
    """Simplex weights fit for one test season (or the forward fit)."""

    season: int | None
    target: str
    components: tuple[str, ...]
    weights: tuple[float, ...]
    intercept: float
    n_train: int
    train_seasons: tuple[int, ...]
    train_label_end_utc: pd.Timestamp

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.components, self.weights))

    def apply(self, frame: pd.DataFrame) -> np.ndarray:
        return self.intercept + frame[list(self.components)].to_numpy(float) @ np.asarray(self.weights)


def fit_weights(train: pd.DataFrame, components: Sequence[str], target: str, *, season: int | None = None,
                objective: str = "squared", intercept: bool = False, equal: bool = False) -> SeasonWeights:
    """Fit simplex weights on a wide training frame (``equal=True``: 1/k each, no fitting)."""
    train = train.dropna(subset=[target])
    refuse_holdout(train["game_id"], "fit_weights")
    k = len(components)
    if equal:
        w, b = np.full(k, 1.0 / k), 0.0
    else:
        w, b = simplex_weights(train[list(components)].to_numpy(float), train[target].to_numpy(float),
                               objective=objective, intercept=intercept)
    return SeasonWeights(season=season, target=target, components=tuple(components),
                         weights=tuple(float(x) for x in w), intercept=float(b), n_train=int(len(train)),
                         train_seasons=tuple(sorted(int(s) for s in train["season"].unique())),
                         train_label_end_utc=pd.to_datetime(train["label_available_at_utc"], utc=True).max())


def season_weights(wide: pd.DataFrame, test_season: int, components: Sequence[str], target: str, *,
                   objective: str = "squared", intercept: bool = False, equal: bool = False,
                   min_train_seasons: int = MIN_TRAIN_SEASONS) -> SeasonWeights | None:
    """Weights for ``test_season`` from earlier seasons' rows before its first T; None if too little history."""
    tr = training_rows(wide, test_season)
    if tr["season"].nunique() < min_train_seasons:
        return None
    return fit_weights(tr, components, target, season=test_season, objective=objective, intercept=intercept,
                       equal=equal)


# -------------------------------------------------------- win probability
def home_win_no_tie(lo: int, mat: np.ndarray) -> np.ndarray:
    """Row-wise ``P(M > 0) / (1 - P(M = 0))`` from stacked integer margin pmfs."""
    x = np.arange(lo, lo + mat.shape[1])
    gt = mat[:, x > 0].sum(axis=1)
    eq = mat[:, x == 0].sum(axis=1)
    return gt / np.maximum(1 - eq, 1e-12)


def fit_k1(frame: pd.DataFrame, pred_margin: np.ndarray) -> FittedDistribution:
    """K1 margin distribution (refit, ``distributions.evaluate``) on rows with outcomes and a given margin."""
    tr = pd.DataFrame({"season": frame["season"].to_numpy(), "prediction_time_utc": frame["prediction_time_utc"].to_numpy(),
                       "pred_margin": np.asarray(pred_margin, float),
                       "home_score": frame["home_score"].to_numpy(float), "away_score": frame["away_score"].to_numpy(float)})
    return fit_distribution(tr, "k1", target="margin")


def k1_pmfs(fd: FittedDistribution, pred_margin) -> tuple[int, np.ndarray]:
    return build_pmfs(fd, pd.DataFrame({"pred_margin": np.asarray(pred_margin, float)}))


def k1_win_prob(params: SM.K1Params) -> Callable[[np.ndarray], np.ndarray]:
    """Mapping stacked margin -> P(home win | no tie) under fixed K1 parameters."""
    def f(margin) -> np.ndarray:
        return home_win_no_tie(int(SM.K1_M[0]), SM.k1_matrix(np.asarray(margin, float), params.sigma, params.w))
    return f


def k1_walkforward(rows: pd.DataFrame, pred_col: str = "pred_margin",
                   min_train_seasons: int = MIN_TRAIN_SEASONS) -> dict[int, FittedDistribution]:
    """K1 fit per test season on a model's own earlier OOF rows (``training_rows`` cutoff)."""
    out = {}
    for s in sorted(int(x) for x in rows["season"].unique()):
        tr = training_rows(rows, s).dropna(subset=[pred_col, "margin"])
        if tr["season"].nunique() >= min_train_seasons:
            out[s] = fit_k1(tr, tr[pred_col].to_numpy(float))
    return out


# ------------------------------------------------------------ walk-forward
@dataclass
class StackRun:
    """Output of :func:`walkforward_stack`."""

    model_id: str
    predictions: pd.DataFrame
    weights: list[SeasonWeights] = field(default_factory=list)
    k1: dict[int, FittedDistribution] = field(default_factory=dict)

    def weights_table(self) -> pd.DataFrame:
        rows = []
        for sw in self.weights:
            rows.append({"model_id": self.model_id, "season": sw.season, "target": sw.target, "n_train": sw.n_train,
                         "train_seasons": f"{min(sw.train_seasons)}-{max(sw.train_seasons)}",
                         "intercept": sw.intercept, **{f"w_{c}": w for c, w in sw.as_dict().items()}})
        return pd.DataFrame(rows)


def walkforward_stack(oof: pd.DataFrame, games: pd.DataFrame, *, model_id: str = STACK_MODEL_ID,
                      margin_components: Sequence[str] = MARGIN_COMPONENTS,
                      total_components: Sequence[str] = TOTAL_COMPONENTS, objective: str = "squared",
                      intercept: bool = False, equal: bool = False,
                      min_train_seasons: int = MIN_TRAIN_SEASONS) -> StackRun:
    """Season walk-forward stack predictions (``PREDICTION_COLUMNS``) for every season with enough history."""
    refuse_holdout(oof["game_id"], "walkforward_stack")
    wm = wide_oof(oof, games, margin_components, "margin")
    wt = wide_oof(oof, games, total_components, "total").set_index("game_id")
    run = StackRun(model_id=model_id, predictions=pd.DataFrame(columns=list(PREDICTION_COLUMNS)))
    out = []
    for s in sorted(int(x) for x in wm["season"].unique()):
        kw = dict(objective=objective, intercept=intercept, equal=equal, min_train_seasons=min_train_seasons)
        swm = season_weights(wm, s, margin_components, "margin", **kw)
        if swm is None:
            continue
        swt = season_weights(wt.reset_index(), s, total_components, "total", **kw)
        tr = training_rows(wm, s).dropna(subset=["margin"])
        fd = fit_k1(tr, swm.apply(tr))
        run.weights += [w for w in (swm, swt) if w is not None]
        run.k1[s] = fd
        te = wm[wm["season"] == s].reset_index(drop=True)
        margin = swm.apply(te)
        lo, mat = k1_pmfs(fd, margin)
        total = np.full(len(te), np.nan)
        label_end = swm.train_label_end_utc
        if swt is not None:
            tt = wt.reindex(te["game_id"])
            ok = tt[list(total_components)].notna().all(axis=1).to_numpy()
            total[ok] = swt.apply(tt[ok])
            label_end = max(label_end, swt.train_label_end_utc)
        train_end = pd.to_datetime(te["train_end_utc"], utc=True).where(lambda t: t > label_end, label_end)
        out.append(pd.DataFrame({
            "game_id": te["game_id"], "season": te["season"], "week": te["week"], "model_id": model_id,
            "fold_id": str(s), "prediction_time_utc": te["prediction_time_utc"], "train_end_utc": train_end,
            "pred_margin": margin, "pred_total": total, "pred_margin_sd": np.nan, "pred_total_sd": np.nan,
            "pred_home_win_prob": home_win_no_tie(lo, mat)}))
    if out:
        preds = pd.concat(out, ignore_index=True)[list(PREDICTION_COLUMNS)]
        preds["train_end_utc"] = pd.to_datetime(preds["train_end_utc"], utc=True)
        run.predictions = validate_predictions(preds)
    return run


def forward_weights(oof: pd.DataFrame, games: pd.DataFrame, *, objective: str = "squared",
                    intercept: bool = False, margin_components: Sequence[str] = MARGIN_COMPONENTS,
                    total_components: Sequence[str] = TOTAL_COMPONENTS) -> dict:
    """Weights and K1 fit on ALL supplied OOF seasons (never 2025), as the JSON payload of the weights file."""
    wm = wide_oof(oof, games, margin_components, "margin").dropna(subset=["margin"])
    wt = wide_oof(oof, games, total_components, "total").dropna(subset=["total"])
    swm = fit_weights(wm, margin_components, "margin", objective=objective, intercept=intercept)
    swt = fit_weights(wt, total_components, "total", objective=objective, intercept=intercept)
    fd = fit_k1(wm, swm.apply(wm))
    p = fd.k1_params
    seasons = swm.train_seasons
    return {
        "model_id": STACK_MODEL_ID, "objective": objective, "intercept_fit": bool(intercept),
        "train_seasons": f"{min(seasons)}-{max(seasons)}",
        "train_label_end_utc": swm.train_label_end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "margin": {"weights": swm.as_dict(), "intercept": swm.intercept, "n_train": swm.n_train},
        "total": {"weights": swt.as_dict(), "intercept": swt.intercept, "n_train": swt.n_train},
        "win_prob": {"mapping": "k1", "note": "P(home win | no tie) from K1 margin pmf at the stacked margin",
                     "k1_sigma0": p.sigma0, "k1_sigma": p.sigma, "k1_w": list(p.w)},
    }


# --------------------------------------------------------------- GameModel
def make_component(model_id: str):
    """Construct a step-3 GameModel by id (same constructors as scripts/run_walkforward.py)."""
    if model_id == "elo_v1":
        from nfl_market.models.elo import EloModel
        return EloModel()
    if model_id == "kalman_gs_v1":
        from nfl_market.models.bayesian import KalmanMarginModel
        return KalmanMarginModel()
    if model_id == "bayes_od_v1":
        from nfl_market.models.bayesian import BayesianStrengthModel
        return BayesianStrengthModel()
    if model_id == "ridge_v1":
        from nfl_market.models.ridge_model import RidgeModel
        return RidgeModel()
    if model_id == "xgb_v1":
        from nfl_market.models.xgb_model import XGBModel
        return XGBModel()
    raise ValueError(f"unknown component model id {model_id!r}")


class StackModel:
    """GameModel: fixed simplex weights over component GameModels.

    ``pred_margin = margin_intercept + sum_i w_i * pred_margin_i`` and likewise for
    totals (NaN if a total component returns none). ``pred_home_win_prob`` comes from
    ``win_prob(pred_margin)`` if supplied, else from the K1 parameters (``k1``) if
    supplied (``from_weights_file`` loads them), else the margin-weighted average of
    the components' own ``pred_home_win_prob`` (renormalised over components that
    have one). ``pred_margin_sd``/``pred_total_sd`` are NaN. ``train_end_utc`` is the
    latest component ``train_end_utc``. Holdout games are refused in fit and predict.
    """

    def __init__(self, margin_weights: Mapping[str, float], total_weights: Mapping[str, float] | None = None, *,
                 components: Sequence | None = None, margin_intercept: float = 0.0, total_intercept: float = 0.0,
                 k1: SM.K1Params | None = None, win_prob: Callable[[np.ndarray], np.ndarray] | None = None,
                 model_id: str = STACK_MODEL_ID):
        self.margin_weights = {k: float(v) for k, v in margin_weights.items()}
        self.total_weights = {k: float(v) for k, v in (total_weights or {}).items()}
        for name, w in (("margin", self.margin_weights), ("total", self.total_weights)):
            v = np.array(list(w.values()), float)
            if len(v) and ((v < 0).any() or abs(v.sum() - 1) > 1e-6):
                raise ValueError(f"{name} weights must be non-negative and sum to 1, got {w}")
        if not self.margin_weights:
            raise ValueError("margin_weights must not be empty")
        needed = list(dict.fromkeys([*self.margin_weights, *self.total_weights]))
        if components is None:
            comps = {c: make_component(c) for c in needed}
        else:
            comps = {m.model_id: m for m in components}
            missing = [c for c in needed if c not in comps]
            if missing:
                raise ValueError(f"components missing for weighted ids {missing}")
        self.components = {c: comps[c] for c in needed}
        self.margin_intercept = float(margin_intercept)
        self.total_intercept = float(total_intercept)
        self.k1 = k1
        self.win_prob = win_prob if win_prob is not None else (k1_win_prob(k1) if k1 is not None else None)
        self.model_id = model_id

    @classmethod
    def from_weights_file(cls, path: str | Path = DEFAULT_WEIGHTS_FILE, components: Sequence | None = None,
                          **kw) -> "StackModel":
        d = json.loads(Path(path).read_text())
        wp = d.get("win_prob") or {}
        k1 = (SM.K1Params(sigma0=float(wp["k1_sigma0"]), sigma=float(wp["k1_sigma"]), w=tuple(map(float, wp["k1_w"])))
              if wp.get("mapping") == "k1" else None)
        return cls(d["margin"]["weights"], d.get("total", {}).get("weights"), components=components,
                   margin_intercept=d["margin"].get("intercept", 0.0),
                   total_intercept=d.get("total", {}).get("intercept", 0.0), k1=k1,
                   model_id=d.get("model_id", STACK_MODEL_ID), **kw)

    def fit(self, games: pd.DataFrame) -> "StackModel":
        refuse_holdout(games["game_id"], "StackModel.fit")
        for m in self.components.values():
            m.fit(games)
        return self

    def predict(self, games: pd.DataFrame) -> pd.DataFrame:
        refuse_holdout(games["game_id"], "StackModel.predict")
        preds = {c: m.predict(games).set_index("game_id").reindex(games["game_id"]) for c, m in self.components.items()}
        first = preds[next(iter(self.margin_weights))]
        margin = self.margin_intercept + sum(w * preds[c]["pred_margin"].to_numpy(float)
                                             for c, w in self.margin_weights.items())
        total = (self.total_intercept + sum(w * preds[c]["pred_total"].to_numpy(float)
                                            for c, w in self.total_weights.items())
                 if self.total_weights else np.full(len(games), np.nan))
        if self.win_prob is not None:
            pw = np.asarray(self.win_prob(margin), float)
        else:
            num = np.zeros(len(games)); den = np.zeros(len(games))
            for c, w in self.margin_weights.items():
                p = preds[c]["pred_home_win_prob"].to_numpy(float)
                ok = np.isfinite(p)
                num[ok] += w * p[ok]; den[ok] += w
            pw = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        train_end = pd.concat([pd.to_datetime(p["train_end_utc"], utc=True) for p in preds.values()], axis=1).max(axis=1)
        out = pd.DataFrame({
            "game_id": games["game_id"].to_numpy(), "season": games["season"].to_numpy(),
            "week": games["week"].to_numpy(), "model_id": self.model_id, "fold_id": None,
            "prediction_time_utc": first["prediction_time_utc"].to_numpy(), "train_end_utc": train_end.to_numpy(),
            "pred_margin": margin, "pred_total": total, "pred_margin_sd": np.nan, "pred_total_sd": np.nan,
            "pred_home_win_prob": pw})
        return out[list(PREDICTION_COLUMNS)]


__all__ = [
    "DEFAULT_WEIGHTS_FILE", "EQUAL_MODEL_ID", "MARGIN_COMPONENTS", "MIN_TRAIN_SEASONS", "SeasonWeights", "StackModel",
    "StackRun", "TOTAL_COMPONENTS", "fit_k1", "fit_weights", "forward_weights", "home_win_no_tie", "k1_walkforward",
    "k1_win_prob", "make_component", "refuse_holdout", "season_weights", "simplex_weights", "walkforward_stack",
    "wide_oof",
]
