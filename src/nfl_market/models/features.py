"""Point-in-time team features for step-3 game models (feature_rules.md R2-R4).

Every value for a game with prediction time T is built only from ``history``
games whose ``label_available_at_utc < T`` (strict as-of). For each team the
labelled games are sorted by (kickoff, game_id); because the label rule is a
monotone function of kickoff, "games with label < T" is a prefix of that list,
so features are prefix aggregates (shift-then-aggregate generalised to the
availability timestamp). The game being predicted can never contribute (its
label time is after its own T), nor can any later game.

Feature registry v1 (status: proposed; author-checked, needs reviewer sign-off):

=====================  ============================================================  =========
name                   formula (team side; home_/away_ prefixed)                       missing
=====================  ============================================================  =========
elo                    as-of Elo (elo.py, ported frozen constants), regressed per gap  never (1500 start)
pf_l8 / pa_l8          mean points for/against over the last <=8 available games        NaN if no games
pf_std / pa_std        season-to-date mean points for/against                          NaN before 1st game
pf_prev / pa_prev      previous season mean points for/against (available games)       NaN if none
gp_season              season games available                                          0
rest                   nflverse home_rest/away_rest (final schedule; REVIEW for flexes) NaN
elo_diff               home_elo - away_elo                                             -
neutral, div_game, postseason, is_dome, is_retractable   schedule flags                 -
league_total_l256      mean total of the last 256 available league games              NaN if none
league_margin_l256     mean home margin of the last 256 available league games        NaN if none
elo_d / elo_fair       Elo gap incl. HFA / Elo+score-model fair margin (elo.py)        never
level                  league pts per team-game as of T (elo.scoring_level)            21 prior
kalman / kalman_var    ported Kalman fair margin and forecast variance as of T         never (prior)
week                   schedule week                                                   -
=====================  ============================================================  =========

``features_available_at_utc`` = latest label time of any game that contributed;
``assert_point_in_time`` is run on it against ``prediction_time_utc``.
Not used: QB identity (nflverse qb ids are post-game starters; needs pre-game
depth chart / injury data), weather (observed, post-T), nflverse lines (unknown capture time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_market.features.pit import assert_point_in_time
from nfl_market.models.bayesian import KalmanParams, kalman_asof
from nfl_market.models.elo import EloParams, _franchises, elo_history, score_means, scoring_level
from nfl_market.models.folds import labelled, prediction_time
from nfl_market.models.folds import to_ns as _ns

TEAM_STATS = ("elo", "pf_l8", "pa_l8", "pf_std", "pa_std", "pf_prev", "pa_prev", "gp_season", "rest")
GAME_FLAGS = ("neutral", "div_game", "postseason", "is_dome", "is_retractable")
LEAGUE = ("league_total_l256", "league_margin_l256", "level")
RATINGS = ("elo_d", "elo_fair", "kalman", "kalman_var")
FEATURE_COLUMNS: tuple[str, ...] = (
    *(f"home_{c}" for c in TEAM_STATS), *(f"away_{c}" for c in TEAM_STATS),
    "elo_diff", "rest_diff", *RATINGS, *GAME_FLAGS, "week", *LEAGUE,
)
#: XGBoost feature list: wbblanck/nfl-betting-model src/nflbet/ats/xgb.py FEATURES, minus the QB
#: deltas (need pre-game expected-starter data) and with last-8 points for/against standing in for
#: the last-8 EPA/play off/def (needs nflverse play-by-play; open item). dome = is_dome here.
XGB_FEATURES: tuple[str, ...] = (
    "elo_d", "elo_fair", "kalman", "kalman_var", "home_pf_l8", "home_pa_l8", "away_pf_l8", "away_pa_l8",
    "level", "home_rest", "away_rest", "neutral", "div_game", "is_dome", "week",
)
LAST_N = 8
LEAGUE_N = 256




def _team_panel(history: pd.DataFrame, params: EloParams) -> pd.DataFrame:
    eh = elo_history(history, params)
    g = history.set_index("game_id").loc[eh["game_id"]]
    home = pd.DataFrame({"team": eh["home_franchise"], "kickoff": eh["kickoff_utc"],
                         "game_id": eh["game_id"], "season": eh["season"],
                         "label": eh["label_available_at_utc"],
                         "pf": g["home_score"].to_numpy(), "pa": g["away_score"].to_numpy(),
                         "elo_post": eh["home_post"]})
    away = pd.DataFrame({"team": eh["away_franchise"], "kickoff": eh["kickoff_utc"],
                         "game_id": eh["game_id"], "season": eh["season"],
                         "label": eh["label_available_at_utc"],
                         "pf": g["away_score"].to_numpy(), "pa": g["home_score"].to_numpy(),
                         "elo_post": eh["away_post"]})
    return pd.concat([home, away]).sort_values(["team", "kickoff", "game_id"]).reset_index(drop=True)


def _side_stats(panel: pd.DataFrame, teams: np.ndarray, T: np.ndarray, seasons: np.ndarray,
                params: EloParams) -> tuple[dict, np.ndarray]:
    n_q = len(teams)
    out = {c: np.full(n_q, np.nan) for c in TEAM_STATS if c != "rest"}
    out["elo"][:] = params.init
    out["gp_season"][:] = 0.0
    avail = np.full(n_q, np.iinfo("int64").min, dtype="int64")
    for team, grp in panel.groupby("team", sort=False):
        q = np.flatnonzero(teams == team)
        if not len(q):
            continue
        lab = _ns(grp["label"])
        if np.any(np.diff(lab) < 0):
            raise AssertionError(f"label times not monotone in kickoff for {team}")
        sea = grp["season"].to_numpy()
        cpf = np.concatenate([[0.0], np.cumsum(grp["pf"].to_numpy(float))])
        cpa = np.concatenate([[0.0], np.cumsum(grp["pa"].to_numpy(float))])
        post = grp["elo_post"].to_numpy(float)
        n = np.searchsorted(lab, T[q], side="left")            # games with label < T
        s = seasons[q]
        s0 = np.searchsorted(sea, s, side="left")
        ps = np.searchsorted(sea, s - 1, side="left")
        with np.errstate(invalid="ignore", divide="ignore"):
            lo = np.maximum(n - LAST_N, 0)
            cnt = n - lo
            out["pf_l8"][q] = np.where(cnt > 0, (cpf[n] - cpf[lo]) / cnt, np.nan)
            out["pa_l8"][q] = np.where(cnt > 0, (cpa[n] - cpa[lo]) / cnt, np.nan)
            gp = np.clip(n - s0, 0, None)
            out["gp_season"][q] = gp
            out["pf_std"][q] = np.where(gp > 0, (cpf[np.maximum(n, s0)] - cpf[s0]) / gp, np.nan)
            out["pa_std"][q] = np.where(gp > 0, (cpa[np.maximum(n, s0)] - cpa[s0]) / gp, np.nan)
            pe = np.minimum(s0, n)
            pc = pe - ps
            pe_safe, ps_safe = np.maximum(pe, ps), ps
            out["pf_prev"][q] = np.where(pc > 0, (cpf[pe_safe] - cpf[ps_safe]) / pc, np.nan)
            out["pa_prev"][q] = np.where(pc > 0, (cpa[pe_safe] - cpa[ps_safe]) / pc, np.nan)
        has = n > 0
        last = np.maximum(n - 1, 0)
        gap = np.where(has, s - sea[last], 0)
        r = post[last]
        r = params.mean + (r - params.mean) * (1.0 - params.regress) ** np.clip(gap, 0, None)
        out["elo"][q] = np.where(has, r, params.init)
        avail[q] = np.where(has, lab[last], avail[q])
    return out, avail


def team_features(history: pd.DataFrame, rows: pd.DataFrame, params: EloParams = EloParams(),
                  kparams: KalmanParams = KalmanParams()) -> pd.DataFrame:
    """Features for ``rows`` (any games) built only from labelled ``history`` games available before each row's T.

    Scores of ``rows`` are never read. Returns FEATURE_COLUMNS plus
    ``prediction_time_utc`` and ``features_available_at_utc``, indexed like ``rows``.
    """
    hist = history.loc[labelled(history)]
    T_ser = prediction_time(rows)
    T = _ns(T_ser)
    seasons = rows["season"].to_numpy(int)
    hf, af = _franchises(rows)
    feats: dict[str, np.ndarray] = {}
    avail = np.full(len(rows), np.iinfo("int64").min, dtype="int64")
    if len(hist):
        panel = _team_panel(hist, params)
    else:
        panel = pd.DataFrame(columns=["team", "kickoff", "game_id", "season", "label", "pf", "pa", "elo_post"])
    for side, teams in (("home", hf.to_numpy()), ("away", af.to_numpy())):
        st, av = _side_stats(panel, teams, T, seasons, params)
        for c, v in st.items():
            feats[f"{side}_{c}"] = v
        avail = np.maximum(avail, av)
    for side in ("home", "away"):
        col = f"{side}_rest"
        feats[col] = rows[col].to_numpy(float) if col in rows else np.full(len(rows), np.nan)
    # league scoring environment
    lg = hist.sort_values(["label_available_at_utc", "kickoff_utc", "game_id"])
    llab = _ns(lg["label_available_at_utc"]) if len(lg) else np.empty(0, dtype="int64")
    ctot = np.concatenate([[0.0], np.cumsum((lg["home_score"] + lg["away_score"]).to_numpy(float))])
    cmar = np.concatenate([[0.0], np.cumsum((lg["home_score"] - lg["away_score"]).to_numpy(float))])
    n = np.searchsorted(llab, T, side="left")
    lo = np.maximum(n - LEAGUE_N, 0)
    cnt = n - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        feats["league_total_l256"] = np.where(cnt > 0, (ctot[n] - ctot[lo]) / cnt, np.nan)
        feats["league_margin_l256"] = np.where(cnt > 0, (cmar[n] - cmar[lo]) / cnt, np.nan)
    avail = np.maximum(avail, np.where(n > 0, llab[np.maximum(n - 1, 0)] if len(llab) else 0, avail))
    feats["elo_diff"] = feats["home_elo"] - feats["away_elo"]
    neutral = rows["neutral_site"].to_numpy(bool)
    feats["elo_d"] = feats["elo_diff"] + np.where(neutral, 0.0, params.hfa)
    feats["level"] = scoring_level(hist, rows)
    mh, ma = score_means(feats["elo_d"], feats["level"], params.b)
    feats["elo_fair"] = mh - ma
    km_h, kv_h, km_a, kv_a, kav = kalman_asof(hist, rows, kparams)
    feats["kalman"] = km_h - km_a + np.where(neutral, 0.0, kparams.h)
    feats["kalman_var"] = kv_h + kv_a + kparams.sigma_m ** 2
    avail = np.maximum(avail, kav)
    feats["week"] = rows["week"].to_numpy(float)
    feats["rest_diff"] = feats["home_rest"] - feats["away_rest"]
    roof = rows["roof"].astype("string").str.lower() if "roof" in rows else pd.Series(pd.NA, index=rows.index, dtype="string")
    feats["neutral"] = rows["neutral_site"].to_numpy(float)
    feats["div_game"] = rows["div_game"].to_numpy(float) if "div_game" in rows else np.full(len(rows), np.nan)
    feats["postseason"] = (rows["game_type"].astype(str) != "REG").to_numpy(float)
    feats["is_dome"] = roof.eq("dome").fillna(False).to_numpy(float)
    feats["is_retractable"] = roof.isin(["open", "closed"]).fillna(False).to_numpy(float)
    out = pd.DataFrame({c: feats[c] for c in FEATURE_COLUMNS}, index=rows.index)
    out["prediction_time_utc"] = T_ser.to_numpy()
    out["features_available_at_utc"] = pd.to_datetime(avail, utc=True)  # int64 min == NaT (no history)
    chk = out.dropna(subset=["features_available_at_utc"])
    assert_point_in_time(chk, "prediction_time_utc", "features_available_at_utc")
    if (chk["features_available_at_utc"] >= chk["prediction_time_utc"]).any():
        raise ValueError("Point-in-time leakage: feature label time not strictly before T")
    return out
