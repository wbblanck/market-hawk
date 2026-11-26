"""Point-in-time EPA and QB-status features from nflverse play-by-play and injury reports.

Feature registry v2 additions (status: proposed; see ``docs/methodology/feature_rules.md``). Each is computed
per side (``home_`` / ``away_``) for a game with prediction time T, using only games in the training history
whose ``label_available_at_utc < T``:

=================  ==========================================================================================
off_epa_l8         offensive EPA per play over the team's last <= 8 available games (play-weighted, crosses
                   seasons); NaN if none
def_epa_l8         EPA per play allowed over the same games
off_epa_ewm        EWMA (alpha 0.10 per game, continued across seasons) of per-game offensive EPA/play
def_epa_ewm        same for EPA/play allowed
off_pass_epa_ewm   same for offensive EPA per pass play
def_pass_epa_ewm   same for EPA per pass play allowed
qb_out             1 if the expected starter (most dropbacks in the team's latest available game) is listed
                   Out or Doubtful on the team's injury report for the game's (season, week) with
                   date_modified < T (rows without date_modified count as available); 0 if not listed;
                   NaN for seasons without injury data (before 2009)
qb_epa             expected starter's EPA per dropback over his last 16 games played, shrunk toward 0 with 200
                   pseudo-dropbacks; -0.15 (replacement level) when qb_out = 1
=================  ==========================================================================================

All constants were fixed before the first scored run (``scripts/experiments/recommendations_backtest.py``,
2026-11-25) and are not tuned. ``features_available_at`` for these columns is bounded by the same label rule
as ``models.features``; injury rows are bounded by ``date_modified < T``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_market.data.games import franchise
from nfl_market.holdout import is_holdout
from nfl_market.models.folds import labelled, prediction_time, to_ns

EWM_ALPHA = 0.10
LAST_N = 8
QB_GAMES = 16
QB_PSEUDO = 200.0
QB_REPLACEMENT = -0.15
QB_OUT_STATUSES = ("Out", "Doubtful")
EPA_SIDE: tuple[str, ...] = ("off_epa_l8", "def_epa_l8", "off_epa_ewm", "def_epa_ewm", "off_pass_epa_ewm",
                             "def_pass_epa_ewm")
QB_SIDE: tuple[str, ...] = ("qb_out", "qb_epa")
EPA_COLUMNS: tuple[str, ...] = tuple(f"{s}_{c}" for s in ("home", "away") for c in EPA_SIDE)
QB_COLUMNS: tuple[str, ...] = tuple(f"{s}_{c}" for s in ("home", "away") for c in QB_SIDE)
EXTERNAL_COLUMNS: tuple[str, ...] = EPA_COLUMNS + QB_COLUMNS

_PANELS: dict | None = None


def default_panels() -> dict:
    """Panels from ``data/raw/nflverse_ext`` (loaded once per process)."""
    global _PANELS
    if _PANELS is None:
        from nfl_market.data.nflverse_ext import load_panels

        _PANELS = load_panels()
    return _PANELS


def set_panels(panels: dict | None) -> None:
    """Override the process-wide panels (tests); ``None`` reloads from disk on next use."""
    global _PANELS
    _PANELS = panels


class ExternalFeatures:
    """Feature builder bound to one training history (only its labelled games can contribute)."""

    def __init__(self, history: pd.DataFrame, panels: dict | None = None):
        panels = panels if panels is not None else default_panels()
        hist = history.loc[labelled(history)]
        if len(hist) and is_holdout(hist["game_id"]).any():
            raise ValueError("ExternalFeatures: holdout games in history")
        side = pd.concat([
            pd.DataFrame({"game_id": hist["game_id"].to_numpy(), "fr": _fr(hist, "home"),
                          "label": hist["label_available_at_utc"].to_numpy(), "ko": hist["kickoff_utc"].to_numpy()}),
            pd.DataFrame({"game_id": hist["game_id"].to_numpy(), "fr": _fr(hist, "away"),
                          "label": hist["label_available_at_utc"].to_numpy(), "ko": hist["kickoff_utc"].to_numpy()}),
        ])
        team = panels["team"].copy()
        team["fr"] = franchise(team["team"]).astype(str)
        panel = side.merge(team.drop(columns=["team", "season"], errors="ignore"), on=["game_id", "fr"], how="inner")
        panel = panel.sort_values(["fr", "label", "ko", "game_id"]).reset_index(drop=True)
        for k in ("off", "def"):
            for c in ("plays", "epa", "pass_plays", "pass_epa"):
                panel[f"{c}_{k}"] = panel[f"{c}_{k}"].fillna(0.0).astype(float)
            panel[f"per_{k}"] = np.where(panel[f"plays_{k}"] > 0, panel[f"epa_{k}"] / panel[f"plays_{k}"].clip(lower=1), np.nan)
            panel[f"pper_{k}"] = np.where(panel[f"pass_plays_{k}"] > 0,
                                          panel[f"pass_epa_{k}"] / panel[f"pass_plays_{k}"].clip(lower=1), np.nan)
        self.teams: dict[str, dict] = {}
        for fr, p in panel.groupby("fr", sort=False):
            a = {"label": to_ns(p["label"])}
            for k in ("off", "def"):
                a[f"c_plays_{k}"] = np.concatenate([[0.0], np.cumsum(p[f"plays_{k}"].to_numpy())])
                a[f"c_epa_{k}"] = np.concatenate([[0.0], np.cumsum(p[f"epa_{k}"].to_numpy())])
                for name, col in ((f"{k}_epa", f"per_{k}"), (f"{k}_pass_epa", f"pper_{k}")):
                    e = p[col].ewm(alpha=EWM_ALPHA, adjust=False, ignore_na=True).mean().to_numpy()
                    a[f"ewm_{name}"] = np.concatenate([[np.nan], e])   # a[...][k] = EWMA through game k-1
            self.teams[fr] = a
        self._qb_init(hist, panels)

    def _qb_init(self, hist: pd.DataFrame, panels: dict) -> None:
        qb = panels["qb"].copy()
        qb["fr"] = franchise(qb["team"]).astype(str)
        lab = hist.set_index("game_id")["label_available_at_utc"]
        qb["label"] = qb["game_id"].map(lab)
        qb = qb[qb["label"].notna()]
        st = qb.sort_values(["dropbacks", "player_id"], ascending=[False, True]).drop_duplicates(["game_id", "fr"])
        st = st.sort_values(["fr", "label", "game_id"])
        self.starters = {fr: (to_ns(s["label"]), s["player_id"].to_numpy()) for fr, s in st.groupby("fr", sort=False)}
        qb = qb.sort_values(["player_id", "label", "game_id"])
        self.players = {pid: (to_ns(p["label"]), np.concatenate([[0.0], np.cumsum(p["dropbacks"].to_numpy(float))]),
                              np.concatenate([[0.0], np.cumsum(p["qb_epa"].to_numpy(float))]))
                        for pid, p in qb.groupby("player_id", sort=False)}
        inj = panels["injuries"]
        inj = inj[inj["report_status"].isin(QB_OUT_STATUSES)]
        frs = franchise(inj["team"]).astype(str)
        dm = pd.to_datetime(inj["date_modified"], utc=True, errors="coerce")
        dm_ns = np.where(dm.isna(), np.iinfo("int64").min, to_ns(dm.fillna(pd.Timestamp(0, tz="UTC"))))
        self.inj = {}
        for s, w, fr, pid, t in zip(inj["season"].astype(int), inj["week"].astype(int), frs, inj["gsis_id"], dm_ns):
            key = (int(s), int(w), fr, pid)
            self.inj[key] = min(self.inj.get(key, np.iinfo("int64").max), int(t))
        self.inj_seasons = set(panels["injuries"]["season"].astype(int).unique().tolist())

    def features(self, rows: pd.DataFrame) -> pd.DataFrame:
        """EXTERNAL_COLUMNS for ``rows`` (indexed like ``rows``); scores of ``rows`` are never read."""
        T = to_ns(prediction_time(rows))
        seasons, weeks = rows["season"].to_numpy(int), rows["week"].to_numpy(int)
        out = {}
        for side in ("home", "away"):
            frs = _fr(rows, side)
            cols = {c: np.full(len(rows), np.nan) for c in EPA_SIDE + QB_SIDE}
            for fr in np.unique(frs):
                idx = np.flatnonzero(frs == fr)
                self._epa(fr, T[idx], idx, cols)
                self._qb(fr, T[idx], seasons[idx], weeks[idx], idx, cols)
            out.update({f"{side}_{c}": v for c, v in cols.items()})
        return pd.DataFrame(out, index=rows.index)[list(EXTERNAL_COLUMNS)]

    def _epa(self, fr, T, idx, cols) -> None:
        a = self.teams.get(fr)
        if a is None:
            return
        k = np.searchsorted(a["label"], T, side="left")
        j = np.maximum(k - LAST_N, 0)
        for s in ("off", "def"):
            pl = a[f"c_plays_{s}"][k] - a[f"c_plays_{s}"][j]
            ep = a[f"c_epa_{s}"][k] - a[f"c_epa_{s}"][j]
            cols[f"{s}_epa_l8"][idx] = np.where(pl > 0, ep / np.maximum(pl, 1), np.nan)
            cols[f"{s}_epa_ewm"][idx] = a[f"ewm_{s}_epa"][k]
            cols[f"{s}_pass_epa_ewm"][idx] = a[f"ewm_{s}_pass_epa"][k]

    def _qb(self, fr, T, seasons, weeks, idx, cols) -> None:
        st = self.starters.get(fr)
        if st is None:
            return
        k = np.searchsorted(st[0], T, side="left")
        for r in range(len(idx)):
            if k[r] == 0:
                continue
            pid = st[1][k[r] - 1]
            n = e = 0.0
            pl = self.players.get(pid)
            if pl is not None:
                m = int(np.searchsorted(pl[0], T[r], side="left"))
                j = max(m - QB_GAMES, 0)
                n, e = pl[1][m] - pl[1][j], pl[2][m] - pl[2][j]
            val = e / (n + QB_PSEUDO)
            if int(seasons[r]) in self.inj_seasons:
                t = self.inj.get((int(seasons[r]), int(weeks[r]), fr, pid))
                out = t is not None and t < T[r]
                cols["qb_out"][idx[r]] = 1.0 if out else 0.0
                if out:
                    val = QB_REPLACEMENT
            cols["qb_epa"][idx[r]] = val


def _fr(games: pd.DataFrame, side: str) -> np.ndarray:
    col = f"{side}_franchise"
    s = games[col] if col in games.columns else franchise(games[f"{side}_team"])
    return s.astype(str).to_numpy()
