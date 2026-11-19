"""Season walk-forward fold table with weekly rolling origins (validation_design.md Section 3).

Outer folds: test season S, outer training = all labelled games of seasons < S
(at least ``min_train_seasons`` complete seasons). Inside S, origins are the
schedule weeks in order: before week w, a model is refit on every labelled game
(any season <= S, holdout excluded) whose ``label_available_at_utc`` is strictly
before week w's first prediction time, then scores week w.

Prediction time rule T: ``prediction_time_utc = kickoff_utc - 24h`` (charter default).
Embargo: none beyond the label-availability rule (labels become usable the next
Wednesday 12:00 ET, which already covers stat corrections).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nfl_market.holdout import is_holdout
from nfl_market.models.interface import FoldSpec

PREDICTION_LEAD = pd.Timedelta(hours=24)
FOLD_TABLE_ID = "season_wf_weekly_v1"


def prediction_time(games: pd.DataFrame) -> pd.Series:
    """T per game: the frame's own ``prediction_time_utc`` if present, else kickoff - 24h."""
    if "prediction_time_utc" in games.columns:
        return games["prediction_time_utc"]
    return games["kickoff_utc"] - PREDICTION_LEAD


def labelled(games: pd.DataFrame) -> pd.Series:
    return games["label_available_at_utc"].notna() & games["home_score"].notna() & games["away_score"].notna()


@dataclass(frozen=True)
class WeekOrigin:
    """One weekly rolling origin inside a test season."""

    fold_id: str
    season: int
    week: int
    cutoff_utc: pd.Timestamp          # min prediction time of the week's games
    train_game_ids: frozenset[str]
    test_game_ids: frozenset[str]
    train_end_utc: pd.Timestamp       # max label time among train games


def _assert_no_holdout(ids, what: str) -> None:
    s = pd.Series(sorted(ids), dtype=str)
    if len(s) and is_holdout(s).any():
        raise AssertionError(f"holdout game ids present in {what}")


def assert_fold(train: pd.DataFrame, test: pd.DataFrame, what: str = "fold") -> None:
    """V2 (train labels strictly before every test T), V3 (disjoint games), no holdout ids."""
    if len(train) and len(test):
        tmax = train["label_available_at_utc"].max()
        tmin = prediction_time(test).min()
        if not tmax < tmin:
            raise AssertionError(f"V2 violated in {what}: train label {tmax} >= test T {tmin}")
    if set(train["game_id"]) & set(test["game_id"]):
        raise AssertionError(f"V3 violated in {what}: game in both train and test")
    _assert_no_holdout(train["game_id"], f"{what} train")
    _assert_no_holdout(test["game_id"], f"{what} test")


def season_folds(games: pd.DataFrame, test_seasons, min_train_seasons: int = 3) -> list[FoldSpec]:
    """Outer season folds. Raises if a test season has fewer than ``min_train_seasons`` prior seasons."""
    seasons = sorted(games["season"].unique())
    lab = labelled(games)
    folds = []
    for s in sorted(int(x) for x in test_seasons):
        prior = [x for x in seasons if x < s]
        if len(prior) < min_train_seasons:
            raise ValueError(f"test season {s} has {len(prior)} training seasons < {min_train_seasons}")
        test = games[games["season"] == s]
        if test.empty:
            raise ValueError(f"no games for test season {s}")
        train = games[(games["season"] < s) & lab]
        assert_fold(train, test, f"fold {s}")
        folds.append(FoldSpec(
            fold_id=str(s), test_season=s,
            train_game_ids=frozenset(train["game_id"]), test_game_ids=frozenset(test["game_id"]),
            train_end_utc=train["label_available_at_utc"].max(),
            test_start_utc=prediction_time(test).min(),
        ))
    return folds


def week_origins(games: pd.DataFrame, fold: FoldSpec) -> list[WeekOrigin]:
    """Weekly rolling origins for one outer fold, each asserted (V2, V3, holdout)."""
    s = fold.test_season
    lab = labelled(games)
    T = prediction_time(games)
    test_season = games[games["season"] == s]
    out = []
    for w in sorted(test_season["week"].unique()):
        wk = test_season[test_season["week"] == w]
        cutoff = T.loc[wk.index].min()
        train = games[lab & (games["season"] <= s) & (games["label_available_at_utc"] < cutoff)]
        assert_fold(train, wk, f"fold {s} week {w}")
        out.append(WeekOrigin(
            fold_id=fold.fold_id, season=s, week=int(w), cutoff_utc=cutoff,
            train_game_ids=frozenset(train["game_id"]), test_game_ids=frozenset(wk["game_id"]),
            train_end_utc=train["label_available_at_utc"].max(),
        ))
    return out


def fold_table(games: pd.DataFrame, test_seasons, min_train_seasons: int = 3) -> pd.DataFrame:
    """Flat fold table (one row per weekly origin) for logging/diagnostics."""
    rows = []
    for f in season_folds(games, test_seasons, min_train_seasons):
        for o in week_origins(games, f):
            rows.append({
                "fold_table_id": FOLD_TABLE_ID, "fold_id": o.fold_id, "season": o.season, "week": o.week,
                "cutoff_utc": o.cutoff_utc, "train_end_utc": o.train_end_utc,
                "n_train_games": len(o.train_game_ids), "n_test_games": len(o.test_game_ids),
            })
    return pd.DataFrame(rows)


def prediction_frame(games: pd.DataFrame, model_id: str, train_end_utc, **cols) -> pd.DataFrame:
    """Build a PREDICTION_COLUMNS frame for ``games``; missing prediction columns are NaN."""
    import numpy as np

    from nfl_market.models.interface import PREDICTION_COLUMNS

    out = pd.DataFrame({
        "game_id": games["game_id"].to_numpy(), "season": games["season"].to_numpy(),
        "week": games["week"].to_numpy(), "model_id": model_id, "fold_id": None,
        "prediction_time_utc": prediction_time(games).to_numpy(),
        "train_end_utc": train_end_utc if train_end_utc is not None else pd.NaT,
    })
    for c in ("pred_margin", "pred_total", "pred_margin_sd", "pred_total_sd", "pred_home_win_prob"):
        v = cols.pop(c, np.nan)
        out[c] = np.broadcast_to(np.asarray(v, dtype=float), (len(out),)).copy() if len(out) else np.empty(0)
    if cols:
        raise TypeError(f"unknown prediction columns {sorted(cols)}")
    out["train_end_utc"] = pd.to_datetime(out["train_end_utc"], utc=True)
    return out[list(PREDICTION_COLUMNS)]


def train_end(games: pd.DataFrame):
    lab = games.loc[labelled(games), "label_available_at_utc"]
    return lab.max() if len(lab) else pd.NaT


def to_ns(s: pd.Series):
    """tz-aware timestamps -> int64 ns since epoch (NaT -> int64 min)."""
    return pd.to_datetime(s, utc=True).dt.as_unit("ns").astype("int64").to_numpy()
