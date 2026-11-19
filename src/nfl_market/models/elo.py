"""As-of NFL Elo + negative-binomial score model (rung 1c baseline).

Ported from wbblanck/nfl-betting-model ``src/nflbet/ats/elo.py`` (polars -> pandas;
read-only clone, no commit pinned) with its frozen constants from
``data/backtests/elo_constants.json`` (pre-registered in that repo's
``docs/elo_preregistration.md``):

* K = 25, home field H = 35 Elo points (0 at neutral sites), between-season
  regression c = 1/2 toward 1505, new franchises start at 1500.
* MOV multiplier ``ln(|margin|+1) * 2.2 / (0.001*d_winner + 2.2)``, 1 for a tie.
* Score means ``log mu_home = a + b*d/100``, ``log mu_away = a - b*d/100``,
  ``a = log(level)``, b = 0.0852689; negative binomial size theta = 5.99587.
* ``level`` = league points per team-game known before the game:
  ``(128*prev_season_REG_mean + sum current-season REG points) / (128 + n team-games)``.

Provenance caveat (V5): those constants were chosen on 2016-2020 (grid by win
log loss; b, theta by ML), so test folds 2016-2020 here are NOT nested for this
model and are exploratory; folds 2015 and 2021-2024 are clean (2021 was that
repo's descriptive-only explore season).

Outputs: ``pred_home_win_prob`` = logistic(d), ``pred_margin`` = mu_h - mu_a
(the "fair spread"), ``pred_total`` = mu_h + mu_a, SDs from the NB variances
(mu + mu^2/theta per team, scores independent given the means).
Ratings are keyed by franchise (STL->LA, SD->LAC, OAK->LV).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nfl_market.data.games import franchise
from nfl_market.models.folds import labelled, prediction_frame, prediction_time, to_ns, train_end

PRIOR_N = 128.0
DEFAULT_LEVEL = 21.0


@dataclass(frozen=True)
class EloParams:
    k: float = 25.0
    hfa: float = 35.0
    mean: float = 1505.0
    regress: float = 0.5
    init: float = 1500.0
    b: float = 0.0852689212242553
    theta: float = 5.995866671817541


def win_prob(diff):
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(diff, dtype=float) / 400.0))


def mov_multiplier(mov: float, winner_diff: float) -> float:
    if mov == 0:
        return 1.0
    return math.log(abs(mov) + 1.0) * 2.2 / (0.001 * winner_diff + 2.2)


def _franchises(games: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    h = games["home_franchise"] if "home_franchise" in games else franchise(games["home_team"])
    a = games["away_franchise"] if "away_franchise" in games else franchise(games["away_team"])
    return h.astype(str), a.astype(str)


@dataclass
class EloState:
    params: EloParams
    ratings: dict = field(default_factory=dict)
    season: int | None = None

    def rating(self, team: str, season: int) -> float:
        """Pre-game rating for ``team`` in ``season`` (regressed if the state is from an earlier season)."""
        if team not in self.ratings:
            return self.params.init
        r = self.ratings[team]
        if self.season is not None:
            for _ in range(max(0, int(season) - self.season)):
                r = r + self.params.regress * (self.params.mean - r)
        return r

    def _roll_to(self, season: int) -> None:
        if self.season is None:
            self.season = int(season)
            return
        while self.season < season:
            p = self.params
            self.ratings = {t: r + p.regress * (p.mean - r) for t, r in self.ratings.items()}
            self.season += 1

    def update(self, home: str, away: str, season: int, neutral: bool, hs: float, as_: float) -> tuple[float, float, float]:
        """Process one final game. Returns (pre-game diff incl. HFA, post home, post away)."""
        self._roll_to(int(season))
        p = self.params
        rh, ra = self.ratings.get(home, p.init), self.ratings.get(away, p.init)
        d = rh - ra + (0.0 if neutral else p.hfa)
        exp_h = 1.0 / (1.0 + 10.0 ** (-d / 400.0))
        mov = hs - as_
        res = 1.0 if mov > 0 else (0.0 if mov < 0 else 0.5)
        shift = p.k * mov_multiplier(mov, d if mov > 0 else -d) * (res - exp_h)
        self.ratings[home] = rh + shift
        self.ratings[away] = ra - shift
        return d, rh + shift, ra - shift


def _sorted_labelled(games: pd.DataFrame) -> pd.DataFrame:
    return games.loc[labelled(games)].sort_values(["kickoff_utc", "game_id"])


def elo_history(history: pd.DataFrame, params: EloParams = EloParams()) -> pd.DataFrame:
    """Sequential Elo over labelled games (kickoff order); one row per game with pre diff and post ratings.

    Each team plays at most once per week, so per-game sequential updating equals
    the source's week-batched update.
    """
    g = _sorted_labelled(history)
    h, a = _franchises(g)
    st = EloState(params)
    rows = np.empty((len(g), 3))
    for i, (ht, at, s, n, hs, as_) in enumerate(zip(h.tolist(), a.tolist(), g["season"].tolist(),
                                                    g["neutral_site"].tolist(), g["home_score"].tolist(),
                                                    g["away_score"].tolist())):
        rows[i] = st.update(ht, at, int(s), bool(n), float(hs), float(as_))
    return pd.DataFrame({
        "game_id": g["game_id"].to_numpy(), "season": g["season"].to_numpy(),
        "kickoff_utc": g["kickoff_utc"].array, "label_available_at_utc": g["label_available_at_utc"].array,
        "home_franchise": h.to_numpy(), "away_franchise": a.to_numpy(),
        "pre_diff": rows[:, 0], "home_post": rows[:, 1], "away_post": rows[:, 2],
        "margin": (g["home_score"] - g["away_score"]).to_numpy(),
    })


def scoring_level(history: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """League points per team-game known at each row's T (source ``scoring_level``, made strictly as-of).

    Previous regular season's mean (21.0 if none) blended with current-season
    regular-season games whose label is available before T, weight 128 team-games.
    """
    hist = history.loc[labelled(history) & (history["game_type"] == "REG")]
    T = to_ns(prediction_time(rows))
    seasons = rows["season"].to_numpy(int)
    pts = (hist["home_score"] + hist["away_score"]).to_numpy(float)
    hs = hist["season"].to_numpy(int)
    lab = to_ns(hist["label_available_at_utc"])
    out = np.empty(len(rows))
    for s in np.unique(seasons):
        prev = hs == s - 1
        pm = float(pts[prev].mean() / 2.0) if prev.any() else DEFAULT_LEVEL
        cur = hs == s
        cl, cp = lab[cur], pts[cur]
        o = np.argsort(cl, kind="stable")
        cl, cc = cl[o], np.concatenate([[0.0], np.cumsum(cp[o])])
        q = np.flatnonzero(seasons == s)
        n = np.searchsorted(cl, T[q], side="left")
        out[q] = (PRIOR_N * pm + cc[n]) / (PRIOR_N + 2.0 * n)
    return out


def score_means(d, level, b: float):
    a = np.log(np.asarray(level, float))
    x = b * np.asarray(d, float) / 100.0
    return np.exp(a + x), np.exp(a - x)


class EloModel:
    """GameModel: as-of Elo + NB score model. Refits incrementally when the new training set extends the old one."""

    def __init__(self, params: EloParams = EloParams(), model_id: str = "elo_v1"):
        self.params = params
        self.model_id = model_id
        self._reset()

    def _reset(self):
        self.state = EloState(self.params)
        self._seen: dict[str, tuple[float, float]] = {}
        self._last_key = None

    def _can_extend(self, g: pd.DataFrame) -> bool:
        if not self._seen:
            return False
        ids = g["game_id"].tolist()
        seen_mask = np.fromiter((i in self._seen for i in ids), bool, len(ids))
        if seen_mask.sum() != len(self._seen):
            return False
        old = g.loc[seen_mask]
        if not all(self._seen[i] == (hs, as_) for i, hs, as_ in zip(old["game_id"].tolist(), old["home_score"].tolist(), old["away_score"].tolist())):
            return False
        new = g.loc[~seen_mask]
        return new.empty or min(zip(new["kickoff_utc"], new["game_id"])) > self._last_key

    def fit(self, games: pd.DataFrame) -> "EloModel":
        g = _sorted_labelled(games)
        self.train_end_ = train_end(g)
        self.history_ = g
        if self._can_extend(g):
            new = g.loc[~g["game_id"].isin(self._seen.keys())]
        else:
            self._reset()
            new = g
        h, a = _franchises(new)
        for gid, ht, at, s, n, hs, as_, ko in zip(new["game_id"].tolist(), h.tolist(), a.tolist(), new["season"].tolist(),
                                                 new["neutral_site"].tolist(), new["home_score"].tolist(),
                                                 new["away_score"].tolist(), new["kickoff_utc"].tolist()):
            self.state.update(ht, at, int(s), bool(n), float(hs), float(as_))
            self._seen[gid] = (hs, as_)
            self._last_key = (ko, gid)
        return self

    def pregame_diff(self, games: pd.DataFrame) -> np.ndarray:
        h, a = _franchises(games)
        return np.array([
            self.state.rating(ht, s) - self.state.rating(at, s) + (0.0 if n else self.params.hfa)
            for ht, at, s, n in zip(h.tolist(), a.tolist(), games["season"].tolist(), games["neutral_site"].tolist())
        ], dtype=float)

    def predict(self, games: pd.DataFrame) -> pd.DataFrame:
        d = self.pregame_diff(games)
        lvl = scoring_level(self.history_, games)
        mh, ma = score_means(d, lvl, self.params.b)
        th = self.params.theta
        vh, va = mh + mh ** 2 / th, ma + ma ** 2 / th
        sd = np.sqrt(vh + va)
        return prediction_frame(games, self.model_id, self.train_end_,
                                pred_margin=mh - ma, pred_total=mh + ma,
                                pred_margin_sd=sd, pred_total_sd=sd, pred_home_win_prob=win_prob(d))
