"""Bayesian dynamic team-strength models (Gaussian state space, exact Kalman updates).

1. ``KalmanMarginModel`` (default, ``kalman_gs_v1``): faithful port of
   wbblanck/nfl-betting-model ``src/nflbet/ats/kalman.py`` (polars -> pandas; read-only
   clone, no commit pinned), Glickman-Stern margin rating with a diagonal
   covariance, pre-registered in that repo's ``docs/batch1_preregistration.md``.
   Frozen constants from ``data/backtests/batch1_constants.json`` (fit by ML on
   2016-2020): h = 1.48964, sigma_m = 12.04238, tau_w = 0.83475, rho = 0.56240,
   tau_s = 3.18171; starting strength 0, variance 49. Margin = theta_home -
   theta_away + h (0 at neutral); forecast variance = v_home + v_away + sigma_m^2.
   Between weeks every variance grows by tau_w^2; between seasons means shrink by
   rho and variances become rho^2 v + tau_s^2. Provenance caveat (V5): constants
   chosen on 2016-2020, so test folds 2016-2020 are not nested for this model
   (exploratory); 2015 and 2021-2024 are clean. Margin only (``pred_total`` NaN).

2. ``BayesianStrengthModel`` (extension, ``bayes_od_v1``): our offense/defense
   extension with full covariance, giving totals as well (below).

Offense/defense extension
-------------------------
Dynamic Bayesian team strength: Gaussian state-space (Kalman filter) model.

State (per franchise): offense ``o`` and defense ``d`` (points above/below
league average; defense positive = allows fewer points), plus a league scoring
level ``mu`` and home advantage ``h``. Observation per final game (kickoff order):

    home_pts = mu + h*(1-neutral)/2 + o_home - d_away + e_h
    away_pts = mu - h*(1-neutral)/2 + o_away - d_home + e_a,   e ~ N(0, sigma^2) iid

Posterior is exact (conjugate Gaussian). Dynamics:
* each game adds process variance ``q_game`` to the two teams' o and d (in-season drift),
  and ``q_global`` to mu and h;
* at a season boundary team means shrink toward 0 by ``rho`` and the team block
  of the covariance becomes ``rho^2 P + q_season`` (season-to-season variance inflation).

All hyperparameters are fixed a priori (generic NFL scale, not tuned on our data):
sigma = 9.5, prior sd(o,d) = 3, prior mu ~ N(21, 3^2), h ~ N(2.5, 1.5^2),
sd(q_game) = 0.6, sd(q_global) = 0.05, rho = 0.7, sd(q_season) = 2.0.
Home/away score errors are assumed independent (P5: to be tested on OOF output).

Predictive: margin = a'x, var = a'Pa + 2 sigma^2 (same for total), and
``pred_home_win_prob = Phi(mean/sd)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from nfl_market.models.elo import _franchises
from nfl_market.models.folds import labelled, prediction_frame, prediction_time, to_ns, train_end


@dataclass(frozen=True)
class BayesParams:
    sigma: float = 9.5
    prior_team_sd: float = 3.0
    prior_mu: float = 21.0
    prior_mu_sd: float = 3.0
    prior_h: float = 2.5
    prior_h_sd: float = 1.5
    q_game_sd: float = 0.6
    q_global_sd: float = 0.05
    rho: float = 0.7
    q_season_sd: float = 2.0


KALMAN_PRIOR_VAR = 49.0


@dataclass(frozen=True)
class KalmanParams:
    """Frozen constants: wbblanck/nfl-betting-model data/backtests/batch1_constants.json ["kalman"]."""

    h: float = 1.4896432908005999
    sigma_m: float = 12.04238350275417
    tau_w: float = 0.8347519207357257
    rho: float = 0.562402169800117
    tau_s: float = 3.181714872336233


def kalman_update(mh, vh, ma, va, margin, h, sigma_m):
    """One game (source ``update``): returns new (mh, vh, ma, va), the forecast and its variance."""
    pred = mh - ma + h
    s = vh + va + sigma_m ** 2
    r = margin - pred
    return mh + vh / s * r, vh - vh * vh / s, ma - va / s * r, va - va * va / s, pred, s


def kalman_history(history: pd.DataFrame, p: KalmanParams = KalmanParams()):
    """Run the filter over labelled games in (season, week, kickoff) order.

    Returns (per-game frame with pre-game forecast and post-game team states,
    {season: sorted weeks seen}). Each team plays at most once per week, so the
    per-game sequential update equals the source's week-batched update.
    """
    g = history.loc[labelled(history)].sort_values(["season", "week", "kickoff_utc", "game_id"])
    h, a = _franchises(g)
    M: dict[str, float] = {}
    V: dict[str, float] = {}
    cur = (None, None)
    out = np.empty((len(g), 6))
    for i, (ht, at, s, w, n, hs, as_) in enumerate(zip(h.tolist(), a.tolist(), g["season"].tolist(), g["week"].tolist(),
                                                       g["neutral_site"].tolist(), g["home_score"].tolist(),
                                                       g["away_score"].tolist())):
        if (s, w) != cur:
            if s != cur[0]:
                for t in M:
                    M[t] *= p.rho
                    V[t] = p.rho * p.rho * V[t] + p.tau_s ** 2
            else:
                for t in V:
                    V[t] += p.tau_w ** 2
            cur = (s, w)
        mh, vh = M.get(ht, 0.0), V.get(ht, KALMAN_PRIOR_VAR)
        ma, va = M.get(at, 0.0), V.get(at, KALMAN_PRIOR_VAR)
        nmh, nvh, nma, nva, pred, var = kalman_update(mh, vh, ma, va, float(hs - as_), 0.0 if n else p.h, p.sigma_m)
        M[ht], V[ht], M[at], V[at] = nmh, nvh, nma, nva
        out[i] = (pred, var, nmh, nvh, nma, nva)
    frame = pd.DataFrame(out, columns=["pred", "var", "home_m", "home_v", "away_m", "away_v"])
    for c in ("game_id", "season", "week", "kickoff_utc", "label_available_at_utc"):
        frame[c] = g[c].array
    frame["home_franchise"], frame["away_franchise"] = h.to_numpy(), a.to_numpy()
    weeks = {int(s): np.array(sorted(set(x)), int) for s, x in g.groupby("season")["week"]}
    return frame, weeks


def kalman_asof(history: pd.DataFrame, rows: pd.DataFrame, p: KalmanParams = KalmanParams()):
    """Team strength (mean, variance) for both sides of each row as of its T.

    Uses each team's state after its last game with label < T, then applies the
    source's between-week variance growth and between-season shrinkage up to the
    row's (season, week). Returns arrays (m_home, v_home, m_away, v_away, avail_ns).
    """
    kh, weeks = kalman_history(history, p)
    panel = pd.concat([
        pd.DataFrame({"team": kh["home_franchise"], "m": kh["home_m"], "v": kh["home_v"], "season": kh["season"],
                      "week": kh["week"], "ko": kh["kickoff_utc"], "label": kh["label_available_at_utc"]}),
        pd.DataFrame({"team": kh["away_franchise"], "m": kh["away_m"], "v": kh["away_v"], "season": kh["season"],
                      "week": kh["week"], "ko": kh["kickoff_utc"], "label": kh["label_available_at_utc"]}),
    ]).sort_values(["team", "season", "week", "ko"])
    by_team = {t: (to_ns(g["label"]), g["m"].to_numpy(), g["v"].to_numpy(), g["season"].to_numpy(int),
                   g["week"].to_numpy(int)) for t, g in panel.groupby("team")}
    T = to_ns(prediction_time(rows))
    hf, af = _franchises(rows)
    hf, af = hf.tolist(), af.tolist()
    seasons, wks = rows["season"].to_numpy(int), rows["week"].to_numpy(int)
    tw2 = p.tau_w ** 2
    res = np.empty((len(rows), 4))
    avail = np.full(len(rows), np.iinfo("int64").min, dtype="int64")

    def n_weeks(season, lo, hi, cur_week):
        """League weeks in ``season`` with lo < week <= hi (the predicted week counts as started)."""
        ws = weeks.get(season, np.empty(0, int))
        ws = np.union1d(ws, [cur_week]) if cur_week is not None else ws
        return int(np.sum((ws > lo) & (ws <= hi)))

    for i in range(len(rows)):
        s, w = seasons[i], wks[i]
        for j, team in ((0, hf[i]), (2, af[i])):
            m, v = 0.0, KALMAN_PRIOR_VAR
            if team in by_team:
                lab, mm, vv, ss, ww = by_team[team]
                n = int(np.searchsorted(lab, T[i], side="left"))
                if n > 0:
                    k = n - 1
                    m, v, sg, wg = mm[k], vv[k], ss[k], ww[k]
                    avail[i] = max(avail[i], lab[k])
                    if sg == s:
                        v += tw2 * n_weeks(s, wg, w, w)
                    else:
                        v += tw2 * n_weeks(sg, wg, 10 ** 6, None)
                        for _ in range(s - sg):
                            m *= p.rho
                            v = p.rho * p.rho * v + p.tau_s ** 2
                        ws = np.union1d(weeks.get(s, np.empty(0, int)), [w])
                        v += tw2 * n_weeks(s, ws.min(), w, w)
            res[i, j], res[i, j + 1] = m, v
    return res[:, 0], res[:, 1], res[:, 2], res[:, 3], avail


class KalmanMarginModel:
    """GameModel: ported Glickman-Stern Kalman margin rating with frozen batch-1 constants."""

    def __init__(self, params: KalmanParams = KalmanParams(), model_id: str = "kalman_gs_v1"):
        self.params = params
        self.model_id = model_id

    def fit(self, games: pd.DataFrame) -> "KalmanMarginModel":
        self.history_ = games.loc[labelled(games)]
        self.train_end_ = train_end(self.history_)
        return self

    def predict(self, games: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        mh, vh, ma, va, _ = kalman_asof(self.history_, games, p)
        hh = np.where(games["neutral_site"].to_numpy(bool), 0.0, p.h)
        mean = mh - ma + hh
        sd = np.sqrt(vh + va + p.sigma_m ** 2)
        return prediction_frame(games, self.model_id, self.train_end_, pred_margin=mean, pred_margin_sd=sd,
                                pred_home_win_prob=norm.cdf(mean / sd))


class BayesianStrengthModel:
    """GameModel: Kalman-filtered offense/defense ratings with posterior uncertainty."""

    def __init__(self, params: BayesParams = BayesParams(), model_id: str = "bayes_od_v1"):
        self.params = params
        self.model_id = model_id
        self._reset()

    def _reset(self):
        self.teams: dict[str, int] = {}
        self.n_max = 40
        p = self.params
        n = 2 * self.n_max + 2
        self.x = np.zeros(n)
        self.P = np.eye(n) * p.prior_team_sd ** 2
        self.i_mu, self.i_h = n - 2, n - 1
        self.x[self.i_mu], self.x[self.i_h] = p.prior_mu, p.prior_h
        self.P[self.i_mu, self.i_mu], self.P[self.i_h, self.i_h] = p.prior_mu_sd ** 2, p.prior_h_sd ** 2
        self.season: int | None = None
        self._seen: dict[str, tuple[float, float]] = {}
        self._last_key = None

    def _idx(self, team: str) -> tuple[int, int]:
        if team not in self.teams:
            if len(self.teams) >= self.n_max:
                raise ValueError("too many teams")
            self.teams[team] = len(self.teams)
        k = self.teams[team]
        return k, self.n_max + k   # offense index, defense index

    def _team_block(self) -> np.ndarray:
        return np.arange(2 * self.n_max)

    def _roll_state(self, x, P, season_from: int | None, season_to: int):
        if season_from is None or season_to <= season_from:
            return x, P
        p = self.params
        x, P = x.copy(), P.copy()
        tb = self._team_block()
        for _ in range(season_to - season_from):
            x[tb] *= p.rho
            P[tb, :] *= p.rho
            P[:, tb] *= p.rho
            P[tb, tb] += p.q_season_sd ** 2
        return x, P

    def _design(self, home: str, away: str, neutral: bool) -> tuple[np.ndarray, np.ndarray]:
        oh, dh = self._idx(home)
        oa, da = self._idx(away)
        hh = 0.0 if neutral else 0.5
        idx = np.array([oh, da, oa, dh, self.i_mu, self.i_h])
        H = np.array([[1.0, -1.0, 0.0, 0.0, 1.0, hh],     # home points
                      [0.0, 0.0, 1.0, -1.0, 1.0, -hh]])   # away points
        return idx, H

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
        PHt = P[:, idx] @ H.T                       # n x 2
        S = H @ PHt[idx] + np.eye(2) * p.sigma ** 2
        K = PHt @ np.linalg.inv(S)
        resid = np.array([hs, as_]) - H @ self.x[idx]
        self.x = self.x + K @ resid
        self.P = P - K @ PHt.T
        self.P = 0.5 * (self.P + self.P.T)

    def _can_extend(self, g: pd.DataFrame) -> bool:
        if not self._seen:
            return False
        ids = g["game_id"].tolist()
        mask = np.fromiter((i in self._seen for i in ids), bool, len(ids))
        if mask.sum() != len(self._seen):
            return False
        old = g.loc[mask]
        if not all(self._seen[i] == (a, b) for i, a, b in zip(old["game_id"].tolist(), old["home_score"].tolist(), old["away_score"].tolist())):
            return False
        new = g.loc[~mask]
        return new.empty or min(zip(new["kickoff_utc"], new["game_id"])) > self._last_key

    def fit(self, games: pd.DataFrame) -> "BayesianStrengthModel":
        g = games.loc[labelled(games)].sort_values(["kickoff_utc", "game_id"])
        self.train_end_ = train_end(g)
        if self._can_extend(g):
            new = g.loc[~g["game_id"].isin(self._seen.keys())]
        else:
            self._reset()
            new = g
        h, a = _franchises(new)
        for gid, ht, at, s, n, hs, as_, ko in zip(new["game_id"].tolist(), h.tolist(), a.tolist(), new["season"].tolist(),
                                                 new["neutral_site"].tolist(), new["home_score"].tolist(),
                                                 new["away_score"].tolist(), new["kickoff_utc"].tolist()):
            self._update(ht, at, int(s), bool(n), float(hs), float(as_))
            self._seen[gid] = (hs, as_)
            self._last_key = (ko, gid)
        return self

    def predict_components(self, games: pd.DataFrame) -> pd.DataFrame:
        h, a = _franchises(games)
        sig2 = self.params.sigma ** 2
        rows = []
        for ht, at, s, n in zip(h.tolist(), a.tolist(), games["season"].tolist(), games["neutral_site"].tolist()):
            x, P = self._roll_state(self.x, self.P, self.season, int(s))
            idx, H = self._design(ht, at, bool(n))
            mean = H @ x[idx]
            cov = H @ P[np.ix_(idx, idx)] @ H.T + np.eye(2) * sig2
            dm, dt = np.array([1.0, -1.0]), np.array([1.0, 1.0])
            rows.append((dm @ mean, dt @ mean, np.sqrt(dm @ cov @ dm), np.sqrt(dt @ cov @ dt)))
        return pd.DataFrame(rows, columns=["margin", "total", "margin_sd", "total_sd"], index=games.index)

    def predict(self, games: pd.DataFrame) -> pd.DataFrame:
        c = self.predict_components(games)
        return prediction_frame(games, self.model_id, self.train_end_,
                                pred_margin=c["margin"], pred_total=c["total"],
                                pred_margin_sd=c["margin_sd"], pred_total_sd=c["total_sd"],
                                pred_home_win_prob=norm.cdf(c["margin"] / c["margin_sd"]))
