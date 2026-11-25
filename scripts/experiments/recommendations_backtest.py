"""Backtest of the four Bayes Ratings recommendations (exploratory, 2026-11-25).

Questions, each answered on the step-3 fold table (season walk-forward, test seasons 2015-2024, weekly
rolling origin, T = kickoff - 24h, 2025 never read):

1. nflverse play-by-play EPA features: do Ridge/XGB improve when EPA/play team form is added?
2. Pre-game QB status from nflverse injury reports: does an "expected starter out" flag plus the expected
   starter's recent EPA/dropback help?
3. Fitting the Bayesian models' settings by marginal likelihood (per fold, earlier seasons only) instead of
   fixing them: kalman_gs_fit, bayes_od_fit (with a home/away error correlation).
4. LightGBM challenger (settings fixed below before any run).

Every candidate is compared with its own baseline (paired game-clustered bootstrap) and with the nflverse line
(close-like, capture time unknown; evaluation only, never a feature). "Beyond the line" = does
(prediction - line) predict (result - line)?

Pre-registered constants (chosen before the first scored run, not tuned):
* EPA: plays = pass or rush with non-missing EPA; last-8 = play-weighted over the team's last 8 available games
  (crossing seasons); EWMA over games with alpha = 0.10, continued across seasons without shrinkage.
* QB: a team's starter in a game = player with most QB dropbacks. Expected starter at T = starter of the team's
  latest game with label < T. qb_out = 1 when that player is listed Out or Doubtful on the team's final injury
  report for the game's (season, week) and the report's date_modified < T (rows with no date_modified are
  treated as available: flagged in the report). Seasons before 2009 have no injury data: qb_out = NaN.
  qb_epa = expected starter's EPA/dropback over his last 16 games played, shrunk toward 0 with 200 pseudo
  dropbacks; when qb_out = 1 it is set to -0.15 (replacement level, a priori).
* LightGBM: 300 trees, learning_rate 0.03, num_leaves 7, min_child_samples 40, subsample 0.8 (every tree),
  colsample_bytree 0.8, reg_lambda 1, seed 0.
* Fitted Kalman/Bayes: Nelder-Mead on the one-step-ahead predictive log likelihood of all labelled games in
  seasons <= the last complete training season, from the frozen/a-priori values; refit once per outer fold.

    python scripts/experiments/prep_nflverse_ext.py          # once
    python scripts/experiments/recommendations_backtest.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import binomtest, norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nfl_market.data.games import franchise, load_games, load_nflverse_lines  # noqa: E402
from nfl_market.holdout import is_holdout  # noqa: E402
from nfl_market.models.bayesian import (BayesianStrengthModel, BayesParams, KalmanMarginModel,  # noqa: E402
                                        KalmanParams, kalman_history)
from nfl_market.models.elo import EloModel  # noqa: E402
from nfl_market.models.features import FEATURE_COLUMNS, XGB_FEATURES, team_features  # noqa: E402
from nfl_market.models.folds import labelled, prediction_frame, prediction_time  # noqa: E402
from nfl_market.models.folds import to_ns  # noqa: E402
from nfl_market.models.ridge_model import FeatureModel, RidgeModel  # noqa: E402
from nfl_market.models.runner import attach_outcomes, metrics_table, paired_bootstrap, run_walkforward  # noqa: E402
from nfl_market.models.xgb_model import XGBModel  # noqa: E402

EXT = ROOT / "data" / "raw" / "nflverse_ext"
OUT_MD = ROOT / "docs" / "results" / "experiments" / "recommendations_backtest.md"
OUT_PRED = ROOT / "data" / "processed" / "recommendations_oof_predictions.parquet"
CLEAN_SEASONS = (2015, 2022, 2023, 2024)

EWM_ALPHA = 0.10
LAST_N = 8
QB_GAMES = 16
QB_PSEUDO = 200.0
QB_REPLACEMENT = -0.15
EPA_SIDE = ("off_epa_l8", "def_epa_l8", "off_epa_ewm", "def_epa_ewm", "off_pass_epa_ewm", "def_pass_epa_ewm")
QB_SIDE = ("qb_out", "qb_epa")
EPA_COLS = tuple(f"{s}_{c}" for s in ("home", "away") for c in EPA_SIDE)
QB_COLS = tuple(f"{s}_{c}" for s in ("home", "away") for c in QB_SIDE)
LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=40, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=0, n_jobs=1, verbose=-1)


# --------------------------------------------------------------------------- external point-in-time features
class ExternalFeatures:
    """EPA and QB features for any game, using only games whose label time is before the row's T."""

    def __init__(self, games: pd.DataFrame):
        g = games[["game_id", "season", "week", "kickoff_utc", "label_available_at_utc", "home_team", "away_team",
                   "home_franchise", "away_franchise"]].copy()
        g = g[g.label_available_at_utc.notna()]
        self.stats = {}
        team = pd.read_parquet(EXT / "team_game_epa.parquet")
        if is_holdout(team.game_id).any() or team.season.max() >= 2025:
            raise AssertionError("holdout season in pbp panel")
        side = pd.concat([
            g.assign(team=g.home_team, fr=g.home_franchise)[["game_id", "team", "fr", "label_available_at_utc", "kickoff_utc"]],
            g.assign(team=g.away_team, fr=g.away_franchise)[["game_id", "team", "fr", "label_available_at_utc", "kickoff_utc"]],
        ])
        team["team_fr"] = franchise(team.team).astype(str)
        side["fr"] = side.fr.astype(str)
        panel = side.merge(team.drop(columns=["team"]), left_on=["game_id", "fr"], right_on=["game_id", "team_fr"],
                           how="inner")
        self.stats["pbp_team_games_matched"] = len(panel)
        self.stats["pbp_team_games_total"] = len(team)
        panel = panel.sort_values(["fr", "label_available_at_utc", "kickoff_utc", "game_id"])
        self.teams = {}
        for fr, p in panel.groupby("fr"):
            arr = {"label": to_ns(p.label_available_at_utc), "game_id": p.game_id.to_numpy()}
            for k in ("off", "def"):
                plays = p[f"plays_{k}"].fillna(0).to_numpy(float)
                epa = p[f"epa_{k}"].fillna(0).to_numpy(float)
                arr[f"c_plays_{k}"] = np.concatenate([[0.0], np.cumsum(plays)])
                arr[f"c_epa_{k}"] = np.concatenate([[0.0], np.cumsum(epa)])
                for name, num, den in ((f"{k}_epa", epa, plays),
                                       (f"{k}_pass_epa", p[f"pass_epa_{k}"].fillna(0).to_numpy(float),
                                        p[f"pass_plays_{k}"].fillna(0).to_numpy(float))):
                    per = np.where(den > 0, num / np.maximum(den, 1), np.nan)
                    e = np.full(len(per) + 1, np.nan)   # e[i] = EWMA through game i-1 (prefix of length i)
                    cur = np.nan
                    for i, v in enumerate(per):
                        if np.isfinite(v):
                            cur = v if not np.isfinite(cur) else (1 - EWM_ALPHA) * cur + EWM_ALPHA * v
                        e[i + 1] = cur
                    arr[f"ewm_{name}"] = e
            self.teams[fr] = arr
        self._qb_init(g)
        self._cache: dict[str, dict] = {}

    def _qb_init(self, g: pd.DataFrame) -> None:
        qb = pd.read_parquet(EXT / "qb_game.parquet")
        qb["fr"] = franchise(qb.team).astype(str)
        lab = g.set_index("game_id")["label_available_at_utc"]
        qb["label"] = qb.game_id.map(lab)
        qb = qb[qb.label.notna()]
        starters = qb.sort_values("dropbacks", ascending=False).drop_duplicates(["game_id", "fr"])
        starters = starters.sort_values(["fr", "label", "game_id"])
        self.starters = {fr: (to_ns(s.label), s.player_id.to_numpy()) for fr, s in starters.groupby("fr")}
        qb = qb.sort_values(["player_id", "label", "game_id"])
        self.players = {}
        for pid, p in qb.groupby("player_id"):
            self.players[pid] = (to_ns(p.label), np.concatenate([[0.0], np.cumsum(p.dropbacks.to_numpy(float))]),
                                 np.concatenate([[0.0], np.cumsum(p.qb_epa.to_numpy(float))]))
        inj = pd.read_parquet(EXT / "injuries.parquet")
        inj = inj[inj.report_status.isin(["Out", "Doubtful"])].copy()
        inj["fr"] = franchise(inj.team).astype(str)
        self.stats["injury_out_rows"] = len(inj)
        self.stats["injury_out_rows_no_timestamp"] = int(inj.date_modified.isna().sum())
        self.inj = {(int(s), int(w), fr, pid): dm for s, w, fr, pid, dm in
                    zip(inj.season, inj.week, inj.fr, inj.gsis_id, inj.date_modified)}
        self.inj_seasons = set(int(s) for s in inj.season.unique())

    def _side(self, fr: str, T: int, season: int, week: int) -> dict:
        out = {c: np.nan for c in EPA_SIDE + QB_SIDE}
        a = self.teams.get(fr)
        if a is not None:
            k = int(np.searchsorted(a["label"], T, side="left"))
            if k > 0:
                j = max(k - LAST_N, 0)
                for s in ("off", "def"):
                    pl = a[f"c_plays_{s}"][k] - a[f"c_plays_{s}"][j]
                    out[f"{s}_epa_l8"] = (a[f"c_epa_{s}"][k] - a[f"c_epa_{s}"][j]) / pl if pl > 0 else np.nan
                    out[f"{s}_epa_ewm"] = a[f"ewm_{s}_epa"][k]
                    out[f"{s}_pass_epa_ewm"] = a[f"ewm_{s}_pass_epa"][k]
        st = self.starters.get(fr)
        if st is not None:
            k = int(np.searchsorted(st[0], T, side="left"))
            if k > 0:
                pid = st[1][k - 1]
                pl = self.players.get(pid)
                n = e = 0.0
                if pl is not None:
                    m = int(np.searchsorted(pl[0], T, side="left"))
                    j = max(m - QB_GAMES, 0)
                    n, e = pl[1][m] - pl[1][j], pl[2][m] - pl[2][j]
                qb_epa = e / (n + QB_PSEUDO)
                if season in self.inj_seasons:
                    dm = self.inj.get((season, week, fr, pid), "absent")
                    listed = not (isinstance(dm, str))
                    available = listed and (pd.isna(dm) or dm.value < T)
                    out["qb_out"] = 1.0 if available else 0.0
                    if available:
                        qb_epa = QB_REPLACEMENT
                out["qb_epa"] = qb_epa
        return out

    def features(self, rows: pd.DataFrame) -> pd.DataFrame:
        T = to_ns(prediction_time(rows))
        recs = []
        for gid, t, s, w, hf, af in zip(rows.game_id.tolist(), T.tolist(), rows.season.tolist(), rows.week.tolist(),
                                       rows.home_franchise.astype(str).tolist(), rows.away_franchise.astype(str).tolist()):
            key = f"{gid}|{t}"
            if key not in self._cache:
                h, a = self._side(hf, t, int(s), int(w)), self._side(af, t, int(s), int(w))
                self._cache[key] = {**{f"home_{c}": v for c, v in h.items()}, **{f"away_{c}": v for c, v in a.items()}}
            recs.append(self._cache[key])
        return pd.DataFrame(recs, index=rows.index)


EXTF: ExternalFeatures | None = None


class AugMixin:
    """Adds external EPA/QB columns to a FeatureModel's point-in-time features."""

    extra: tuple[str, ...] = ()

    def _train_features(self, hist, rows):
        base = super()._train_features(hist, rows)
        return base.join(EXTF.features(rows)[list(self.extra)]) if self.extra else base

    def predict(self, games):
        X = team_features(self.history_, games)
        if self.extra:
            X = X.join(EXTF.features(games)[list(self.extra)])
        X = X[list(self.features)]
        pm, pt = self.models_["margin"].predict(X), self.models_["total"].predict(X)
        sm, stt = self.inner_["margin"][1], self.inner_["total"][1]
        pw = norm.cdf(pm / sm) if np.isfinite(sm) and sm > 0 else np.nan
        return prediction_frame(games, self.model_id, self.train_end_, pred_margin=pm, pred_total=pt,
                                pred_margin_sd=sm, pred_total_sd=stt, pred_home_win_prob=pw)


class AugRidge(AugMixin, RidgeModel):
    def __init__(self, model_id, extra):
        super().__init__(model_id=model_id)
        self.extra = tuple(extra)
        self.features = tuple(FEATURE_COLUMNS) + self.extra


class AugXGB(AugMixin, XGBModel):
    def __init__(self, model_id, extra):
        super().__init__(model_id=model_id)
        self.extra = tuple(extra)
        self.features = tuple(XGB_FEATURES) + self.extra


class LGBMModel(AugMixin, FeatureModel):
    def __init__(self, model_id="lgbm_v1", extra=()):
        super().__init__()
        self.model_id = model_id
        self.extra = tuple(extra)
        self.features = tuple(XGB_FEATURES) + self.extra

    def _make(self, target, hp):
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**LGBM_PARAMS)


# --------------------------------------------------------------------------- fitted Bayesian models
def _complete_season(hist: pd.DataFrame) -> int:
    sb = hist.loc[hist.game_type == "SB", "season"]
    return int(sb.max()) if len(sb) else int(hist.season.max()) - 1


class FittedKalman(KalmanMarginModel):
    """kalman_gs with (h, sigma_m, tau_w, rho, tau_s) fit by ML on complete training seasons, once per fold."""

    def __init__(self, model_id="kalman_gs_fit"):
        super().__init__(model_id=model_id)
        self._fits: dict[int, KalmanParams] = {}

    @staticmethod
    def _nll(x, g):
        h, sm, tw, rho, ts = x[0], np.exp(x[1]), np.exp(x[2]), 1 / (1 + np.exp(-x[3])), np.exp(x[4])
        kh, _ = kalman_history(g, KalmanParams(h, sm, tw, rho, ts))
        m = (g.sort_values(["season", "week", "kickoff_utc", "game_id"]).eval("home_score - away_score")).to_numpy()
        return -norm.logpdf(m, kh["pred"].to_numpy(), np.sqrt(kh["var"].to_numpy())).sum()

    def fit(self, games):
        super().fit(games)
        cs = _complete_season(self.history_)
        if cs not in self._fits:
            g = self.history_[self.history_.season <= cs]
            p0 = KalmanParams()
            x0 = np.array([p0.h, np.log(p0.sigma_m), np.log(p0.tau_w), np.log(p0.rho / (1 - p0.rho)), np.log(p0.tau_s)])
            r = minimize(self._nll, x0, args=(g,), method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-3, "fatol": 0.05})
            x = r.x
            self._fits[cs] = KalmanParams(float(x[0]), float(np.exp(x[1])), float(np.exp(x[2])),
                                          float(1 / (1 + np.exp(-x[3]))), float(np.exp(x[4])))
            print(f"[{self.model_id}] seasons<= {cs}: {self._fits[cs]} nll {r.fun:.1f}", flush=True)
        self.params = self._fits[cs]
        return self


class CorrBayes(BayesianStrengthModel):
    """bayes_od with correlated home/away observation errors (corr r)."""

    def __init__(self, params=BayesParams(), r=0.0, model_id="bayes_od_fit"):
        self.r = r
        self.ll = 0.0
        super().__init__(params, model_id)

    def _R(self):
        s2 = self.params.sigma ** 2
        return s2 * np.array([[1.0, self.r], [self.r, 1.0]])

    def _update(self, home, away, season, neutral, hs, as_):
        p = self.params
        if self.season is None:
            self.season = int(season)
        elif season > self.season:
            self.x, self.P = self._roll_state(self.x, self.P, self.season, int(season))
            self.season = int(season)
        idx, H = self._design(home, away, neutral)
        P = self.P
        for j in idx[:4]:
            P[j, j] += p.q_game_sd ** 2
        P[self.i_mu, self.i_mu] += p.q_global_sd ** 2
        P[self.i_h, self.i_h] += p.q_global_sd ** 2
        PHt = P[:, idx] @ H.T
        S = H @ PHt[idx] + self._R()
        Si = np.linalg.inv(S)
        resid = np.array([hs, as_]) - H @ self.x[idx]
        self.ll += -0.5 * (resid @ Si @ resid + np.log(np.linalg.det(S)) + 2 * np.log(2 * np.pi))
        K = PHt @ Si
        self.x = self.x + K @ resid
        self.P = P - K @ PHt.T
        self.P = 0.5 * (self.P + self.P.T)

    def predict_components(self, games):
        h, a = (games["home_franchise"].astype(str), games["away_franchise"].astype(str))
        rows = []
        for ht, at, s, n in zip(h.tolist(), a.tolist(), games["season"].tolist(), games["neutral_site"].tolist()):
            x, P = self._roll_state(self.x, self.P, self.season, int(s))
            idx, H = self._design(ht, at, bool(n))
            mean = H @ x[idx]
            cov = H @ P[np.ix_(idx, idx)] @ H.T + self._R()
            dm, dt = np.array([1.0, -1.0]), np.array([1.0, 1.0])
            rows.append((dm @ mean, dt @ mean, np.sqrt(dm @ cov @ dm), np.sqrt(dt @ cov @ dt)))
        return pd.DataFrame(rows, columns=["margin", "total", "margin_sd", "total_sd"], index=games.index)


class FittedBayes:
    """bayes_od with (sigma, prior_team_sd, q_game_sd, rho, q_season_sd, r) fit by ML once per fold."""

    def __init__(self, model_id="bayes_od_fit"):
        self.model_id = model_id
        self._fits: dict[int, tuple[BayesParams, float]] = {}
        self._models: dict[int, CorrBayes] = {}

    @staticmethod
    def _unpack(x):
        p = replace(BayesParams(), sigma=float(np.exp(x[0])), prior_team_sd=float(np.exp(x[1])),
                    q_game_sd=float(np.exp(x[2])), rho=float(1 / (1 + np.exp(-x[3]))), q_season_sd=float(np.exp(x[4])))
        return p, float(np.tanh(x[5]))

    def _nll(self, x, g):
        p, r = self._unpack(x)
        m = CorrBayes(p, r)
        m.fit(g)
        return -m.ll

    def fit(self, games):
        hist = games.loc[labelled(games)]
        cs = _complete_season(hist)
        if cs not in self._fits:
            g = hist[hist.season <= cs]
            p0 = BayesParams()
            x0 = np.array([np.log(p0.sigma), np.log(p0.prior_team_sd), np.log(p0.q_game_sd),
                           np.log(p0.rho / (1 - p0.rho)), np.log(p0.q_season_sd), 0.0])
            r = minimize(self._nll, x0, args=(g,), method="Nelder-Mead",
                         options={"maxiter": 300, "xatol": 1e-3, "fatol": 0.05})
            self._fits[cs] = self._unpack(r.x)
            print(f"[{self.model_id}] seasons<= {cs}: {self._fits[cs]} nll {r.fun:.1f}", flush=True)
            self._models[cs] = CorrBayes(*self._fits[cs], model_id=self.model_id)
        self.cur = self._models[cs]
        self.cur.fit(games)
        return self

    def predict(self, games):
        return self.cur.predict(games)


# --------------------------------------------------------------------------- evaluation
def beyond_line(preds: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Does (pred - line) predict (result - line)? nflverse spread_line = expected home margin."""
    lines = load_nflverse_lines()[["game_id", "spread_line"]]
    df = attach_outcomes(preds, games).merge(lines, on="game_id")
    df = df[df.spread_line.notna()]
    rows = []
    for m, d in df.groupby("model_id"):
        x = (d.pred_margin - d.spread_line).to_numpy()
        y = (d.margin - d.spread_line).to_numpy()
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ beta
        XtXi = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(XtXi @ (X.T * e ** 2) @ X @ XtXi))
        pick = np.abs(x) >= 1.5
        nopush = y != 0
        hit = np.sign(x) == np.sign(y)
        sel = pick & nopush
        k, n = int(hit[sel].sum()), int(sel.sum())
        rows.append({"model_id": m, "n": len(d), "slope": beta[1], "slope_se": se[1], "t": beta[1] / se[1],
                     "mean_abs_diff_from_line": float(np.abs(x).mean()),
                     "ats_n_(|diff|>=1.5)": n, "ats_hit": k / n if n else np.nan,
                     "ats_p_vs_50%": binomtest(k, n, 0.5).pvalue if n else np.nan})
    return pd.DataFrame(rows)


def fmt(v, nd=3):
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def md(df: pd.DataFrame, nd=3) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(fmt(v, nd) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


PAIRS = [  # (candidate, baseline, question)
    ("ridge_epa", "ridge_v1", "EPA"), ("ridge_qb", "ridge_v1", "QB"), ("ridge_epa_qb", "ridge_v1", "EPA+QB"),
    ("xgb_epa_qb", "xgb_v1", "EPA+QB"), ("lgbm_v1", "xgb_v1", "LightGBM"), ("lgbm_epa_qb", "lgbm_v1", "EPA+QB"),
    ("kalman_gs_fit", "kalman_gs_v1", "fitted"), ("bayes_od_fit", "bayes_od_v1", "fitted"),
    ("ridge_epa_qb", "elo_v1", "vs Elo"), ("xgb_epa_qb", "elo_v1", "vs Elo"),
]


def main():
    global EXTF
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2015-2024")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    a, _, b = args.seasons.partition("-")
    seasons = list(range(int(a), int(b or a) + 1))
    if max(seasons) >= 2025:
        raise SystemExit("test seasons must be <= 2024 (2025 is the locked holdout)")
    t0 = time.time()
    games = load_games()
    games = games[games.season <= max(seasons)].reset_index(drop=True)
    EXTF = ExternalFeatures(games)
    models = [EloModel(), KalmanMarginModel(), BayesianStrengthModel(), RidgeModel(), XGBModel(),
              AugRidge("ridge_epa", EPA_COLS), AugRidge("ridge_qb", QB_COLS), AugRidge("ridge_epa_qb", EPA_COLS + QB_COLS),
              AugXGB("xgb_epa_qb", EPA_COLS + QB_COLS), LGBMModel(), LGBMModel("lgbm_epa_qb", EPA_COLS + QB_COLS),
              FittedKalman(), FittedBayes()]
    if args.only:
        keep = set(args.only.split(","))
        models = [m for m in models if m.model_id in keep]
    preds = run_walkforward(games, models, seasons, verbose=True)
    timing = preds.attrs.get("fit_predict_seconds", {})
    preds.attrs = {}
    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(OUT_PRED, index=False)

    pooled = metrics_table(preds, games, by=("model_id",))
    clean = metrics_table(preds[preds.season.isin(CLEAN_SEASONS)], games, by=("model_id",))
    have = set(preds.model_id)
    boots = []
    for cand, base, q in PAIRS:
        if cand in have and base in have:
            for metric in ("abs_margin", "abs_total", "logloss_win"):
                r = paired_bootstrap(preds, games, cand, base, metric, B=args.bootstrap)
                if r["n_games"]:
                    boots.append({"question": q, "candidate": cand, "baseline": base, **r})
    boots = pd.DataFrame(boots)
    bl = beyond_line(preds, games)

    # feature coverage on test games
    test = games[games.season.isin(seasons)]
    xf = EXTF.features(test)
    cov = {c: float(xf[c].notna().mean()) for c in xf.columns}
    qb_rate = float(np.nanmean(np.r_[xf.home_qb_out.to_numpy(), xf.away_qb_out.to_numpy()]))

    fits_k = {k: v.__dict__ if hasattr(v, "__dict__") else str(v) for m in models if isinstance(m, FittedKalman)
              for k, v in m._fits.items()}
    fits_b = {k: (vars(p), r) for m in models if isinstance(m, FittedBayes) for k, (p, r) in m._fits.items()}

    lines = [
        "# Recommendations backtest (exploratory)", "",
        f"Generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC by `python scripts/experiments/recommendations_backtest.py`."
        f" Runtime {time.time() - t0:.0f}s. Test seasons {seasons[0]}-{seasons[-1]} on the step-3 fold table"
        " (weekly rolling origin, T = kickoff - 24h). 2025 (holdout) and 2026 were not read.", "",
        "Pre-registered settings are in the script docstring. Exploratory: not preregistered in the experiment log,"
        " no leakage audit sign-off.", "",
        "## Pooled, all test seasons", "", md(pooled), "",
        "## Clean seasons only (2015, 2022-2024)", "", md(clean), "",
        "## Paired bootstrap (candidate minus baseline; negative = candidate better)", "", md(boots, 4), "",
        "## Beyond the line", "",
        "Regression of (result - line) on (prediction - line), nflverse spread_line (close-like, capture time"
        " unknown). A slope near 0 means the model adds nothing the line does not already have; a significantly"
        " positive slope means it does. ATS hit rate counts games where the model differs from the line by"
        " at least 1.5 points, pushes excluded.", "", md(bl), "",
        "## Feature coverage on test games", "",
        md(pd.DataFrame([{"feature": k, "non_missing": v} for k, v in cov.items()])), "",
        f"Share of team-games with the expected starting QB listed Out/Doubtful before T: {qb_rate:.3f}.", "",
        f"Data checks: {json.dumps(EXTF.stats)}", "",
        "## Fitted parameters per fold", "", "```", json.dumps({"kalman": {str(k): v for k, v in fits_k.items()},
                                                               "bayes_od": {str(k): v for k, v in fits_b.items()}},
                                                              indent=1, default=str), "```", "",
        f"Fit+predict seconds: {json.dumps({k: round(v, 1) for k, v in timing.items()})}", "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD)


if __name__ == "__main__":
    main()
