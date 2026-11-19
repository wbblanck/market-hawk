"""nflverse schedule/results loader for step-3 game models.

Source: nflverse ``games.csv`` (https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv),
cached at ``data/raw/nflverse_games.csv`` (gitignored).

Time handling
-------------
* ``gameday`` + ``gametime`` are US/Eastern local wall-clock; converted to UTC.
  The 1999 season has no ``gametime`` in the source: those rows get 13:00 ET and
  ``kickoff_time_imputed = True`` (training history only; never a test season).
* ``label_available_at_utc`` = the next Wednesday 12:00 US/Eastern strictly after
  the kickoff's Eastern calendar date (validation_design.md Section 4 default
  stat-correction allowance). Games without a final score have NaT.

Columns
-------
Returns ``interface.GAME_COLUMNS`` plus schedule extras that are knowable before
the game: ``home_rest``, ``away_rest`` (days, nflverse-computed from the final
schedule; see note), ``div_game`` (bool), ``roof``, ``surface``, and the franchise
keys ``home_franchise`` / ``away_franchise`` (relocations OAK->LV, SD->LAC,
STL->LA mapped to the current code so ratings carry over).

* Rest days come from the published final schedule. Flexed/rescheduled games
  can change dates after T; feature status REVIEW REQUIRED for those rare rows.
* ``roof`` for retractable stadiums (``open``/``closed``) is decided on game day:
  only the fixed fact "dome" vs "retractable" vs "outdoors" may be used as a feature.
* ``temp`` / ``wind`` are returned only with ``include_weather=True``. They are
  observed (post-T) conditions of unknown capture time: REVIEW REQUIRED, never
  admitted features.
* nflverse ``spread_line`` / ``total_line`` / moneylines / odds are provider lines
  with unknown capture time. They are never returned by :func:`load_games`;
  :func:`load_nflverse_lines` returns them for a separately labelled benchmark.
* ``home_qb_id`` / ``away_qb_id`` are post-game starters and are not returned.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nfl_market.holdout import REPO_ROOT, exclude_holdout
from nfl_market.models.interface import GAME_COLUMNS

NFLVERSE_GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
DEFAULT_CACHE = REPO_ROOT / "data" / "raw" / "nflverse_games.csv"
EASTERN = "America/New_York"
DEFAULT_GAMETIME = "13:00"

#: Relocated franchises -> current code (franchise continuity for ratings).
FRANCHISE_MAP: dict[str, str] = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

#: Pre-game schedule extras returned besides GAME_COLUMNS.
SCHEDULE_EXTRAS: tuple[str, ...] = (
    "home_rest", "away_rest", "div_game", "roof", "surface",
    "home_franchise", "away_franchise", "kickoff_time_imputed",
)
WEATHER_COLUMNS: tuple[str, ...] = ("temp", "wind")
#: Provider line columns: benchmark only, capture time unknown, never features.
LINE_COLUMNS: tuple[str, ...] = (
    "spread_line", "total_line", "home_moneyline", "away_moneyline",
    "home_spread_odds", "away_spread_odds", "over_odds", "under_odds",
)
LINE_LABEL = "nflverse line, capture time unknown"


def download_games(cache: str | Path = DEFAULT_CACHE, refresh: bool = False, timeout: float = 60.0) -> Path:
    """Download nflverse games.csv to ``cache`` unless it already exists."""
    cache = Path(cache)
    if cache.exists() and not refresh:
        return cache
    import httpx

    cache.parent.mkdir(parents=True, exist_ok=True)
    r = httpx.get(NFLVERSE_GAMES_URL, follow_redirects=True, timeout=timeout)
    r.raise_for_status()
    tmp = cache.with_suffix(".tmp")
    tmp.write_bytes(r.content)
    tmp.replace(cache)
    return cache


def franchise(team: pd.Series) -> pd.Series:
    return team.replace(FRANCHISE_MAP)


def kickoff_utc(gameday: pd.Series, gametime: pd.Series) -> tuple[pd.Series, pd.Series]:
    """US/Eastern local gameday+gametime -> tz-aware UTC. Returns (kickoff, imputed flag)."""
    imputed = gametime.isna()
    local = pd.to_datetime(gameday.astype(str) + " " + gametime.fillna(DEFAULT_GAMETIME).astype(str),
                           format="%Y-%m-%d %H:%M")
    ko = local.dt.tz_localize(EASTERN, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    return ko, imputed


def label_available_at(kickoff: pd.Series) -> pd.Series:
    """Next Wednesday 12:00 US/Eastern strictly after the kickoff's Eastern date."""
    local = kickoff.dt.tz_convert(EASTERN)
    day = local.dt.normalize().dt.tz_localize(None)
    ahead = (2 - day.dt.weekday) % 7
    ahead = ahead.where(ahead != 0, 7)
    wed = day + pd.to_timedelta(ahead, unit="D") + pd.Timedelta(hours=12)
    return wed.dt.tz_localize(EASTERN).dt.tz_convert("UTC")


def _read_raw(path: str | Path | None) -> pd.DataFrame:
    p = Path(path) if path is not None else DEFAULT_CACHE
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run nfl_market.data.games.download_games() (source {NFLVERSE_GAMES_URL})")
    return pd.read_csv(p, low_memory=False)


def load_games(path: str | Path | None = None, include_holdout: bool = False,
               include_weather: bool = False) -> pd.DataFrame:
    """One row per game with GAME_COLUMNS + SCHEDULE_EXTRAS, sorted by kickoff.

    The locked holdout (2025) is excluded unless ``include_holdout=True``; that
    flag exists for manifest checks only. Scoring holdout games goes through
    ``nfl_market.holdout.read_holdout``.
    """
    raw = _read_raw(path)
    if not include_holdout:
        raw = exclude_holdout(raw)
    ko, imputed = kickoff_utc(raw["gameday"], raw["gametime"])
    out = pd.DataFrame({
        "game_id": raw["game_id"].astype(str),
        "season": raw["season"].astype(int),
        "week": raw["week"].astype(int),
        "game_type": raw["game_type"].astype(str),
        "kickoff_utc": ko,
        "home_team": raw["home_team"].astype(str),
        "away_team": raw["away_team"].astype(str),
        "neutral_site": raw["location"].astype(str).str.lower().eq("neutral"),
        "home_score": pd.to_numeric(raw["home_score"], errors="coerce").astype(float),
        "away_score": pd.to_numeric(raw["away_score"], errors="coerce").astype(float),
    })
    final = out["home_score"].notna() & out["away_score"].notna()
    out["label_available_at_utc"] = label_available_at(ko).where(final)
    out["home_rest"] = pd.to_numeric(raw["home_rest"], errors="coerce").astype(float)
    out["away_rest"] = pd.to_numeric(raw["away_rest"], errors="coerce").astype(float)
    out["div_game"] = pd.to_numeric(raw["div_game"], errors="coerce").fillna(0).astype(bool)
    out["roof"] = raw["roof"].astype("string")
    out["surface"] = raw["surface"].astype("string")
    out["home_franchise"] = franchise(out["home_team"])
    out["away_franchise"] = franchise(out["away_team"])
    out["kickoff_time_imputed"] = imputed.to_numpy()
    if include_weather:
        for c in WEATHER_COLUMNS:
            out[c] = pd.to_numeric(raw[c], errors="coerce").astype(float)
    if out["game_id"].duplicated().any():
        raise ValueError("duplicate game_id in nflverse games")
    if out["kickoff_utc"].isna().any():
        raise ValueError("unparseable kickoff time")
    cols = list(GAME_COLUMNS) + list(SCHEDULE_EXTRAS) + (list(WEATHER_COLUMNS) if include_weather else [])
    return out[cols].sort_values(["kickoff_utc", "game_id"]).reset_index(drop=True)


def load_nflverse_lines(path: str | Path | None = None, include_holdout: bool = False) -> pd.DataFrame:
    """nflverse provider lines (home-perspective ``spread_line``, ``total_line``, moneylines).

    BENCHMARK ONLY: capture time unknown, close-like. Never a model feature
    (leakage_audit.md C). Every row carries ``line_label``.
    """
    raw = _read_raw(path)
    if not include_holdout:
        raw = exclude_holdout(raw)
    out = raw[["game_id", *LINE_COLUMNS]].copy()
    for c in LINE_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    out["line_label"] = LINE_LABEL
    return out.reset_index(drop=True)


def moneyline_home_prob(home_ml: pd.Series, away_ml: pd.Series) -> pd.Series:
    """De-vigged (multiplicative) P(home win) from American moneylines; NaN if either missing."""
    def implied(o):
        o = o.astype(float)
        return pd.Series(np.where(o > 0, 100.0 / (o + 100.0), -o / (-o + 100.0)), index=o.index).where(o.notna())
    ph, pa = implied(home_ml), implied(away_ml)
    return ph / (ph + pa)
