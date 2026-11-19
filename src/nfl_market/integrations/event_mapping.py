"""Map odds-feed events (full team names + commence time) onto nflverse ``game_id``s.

odds-api-v4 style feeds (ParlayAPI, The Odds API, OddsLink) name teams in full
("Kansas City Chiefs") and give a commence time, but no season or week. Quotes only
reach ``quote_snapshots`` once ``event_id`` is set, and every step-4 frame keys on the
nflverse ``game_id`` (``markets.interface``), so the capture job maps each event to the
schedule game with the same home and away team whose kickoff is within
``max_kickoff_diff_hours`` of the feed's commence time.

Unmatched events keep ``event_id=None`` and get the flag ``unmapped_event`` (they are still
stored raw). A matched event whose feed kickoff differs from the schedule by more than
``KICKOFF_FLAG_MINUTES`` is flagged ``kickoff_differs_from_schedule``; the feed's kickoff is
kept, since it is usually the more current one (flexed games).
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pandas as pd

from .base import ProviderQuote

#: odds-api-v4 full team names -> nflverse team codes (current franchises).
TEAM_NAME_TO_CODE: dict[str, str] = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}
_ALIASES = {"washington football team": "WAS", "washington": "WAS", "la rams": "LA", "la chargers": "LAC",
            "ny giants": "NYG", "ny jets": "NYJ", "sf 49ers": "SF"}
_BY_LOWER = {k.lower(): v for k, v in TEAM_NAME_TO_CODE.items()} | _ALIASES
_CODES = set(TEAM_NAME_TO_CODE.values())

FLAG_UNMAPPED = "unmapped_event"
FLAG_KICKOFF_DIFFERS = "kickoff_differs_from_schedule"
KICKOFF_FLAG_MINUTES = 60


def team_code(name: str | None) -> str | None:
    """Full or short team name (or an nflverse code) -> nflverse code; None if unknown."""
    if not name:
        return None
    text = str(name).strip()
    if text.upper() in _CODES:
        return text.upper()
    return _BY_LOWER.get(text.lower())


def _utc(value) -> pd.Timestamp:
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def match_game(home: str | None, away: str | None, kickoff: datetime | None, games: pd.DataFrame,
               max_kickoff_diff_hours: float = 36.0) -> pd.Series | None:
    """The schedule row for this matchup nearest ``kickoff`` (within the limit), else None."""
    h, a = team_code(home), team_code(away)
    if h is None or a is None or kickoff is None or games.empty:
        return None
    cand = games[(games["home_team"] == h) & (games["away_team"] == a)]
    if cand.empty:
        return None
    diff = (pd.to_datetime(cand["kickoff_utc"], utc=True) - _utc(kickoff)).abs()
    best = diff.idxmin()
    if diff.loc[best] > pd.Timedelta(hours=max_kickoff_diff_hours):
        return None
    return cand.loc[best]


def map_quotes_to_games(quotes: list[ProviderQuote], games: pd.DataFrame,
                        max_kickoff_diff_hours: float = 36.0) -> list[ProviderQuote]:
    """Set ``event_id``/``season``/``week`` from the schedule on quotes that lack an event id."""
    out: list[ProviderQuote] = []
    cache: dict[tuple, pd.Series | None] = {}
    for q in quotes:
        if q.event_id is not None:
            out.append(q)
            continue
        key = (q.home_team, q.away_team, q.kickoff_utc)
        if key not in cache:
            cache[key] = match_game(q.home_team, q.away_team, q.kickoff_utc, games, max_kickoff_diff_hours)
        g = cache[key]
        if g is None:
            out.append(replace(q, quality_flags=tuple(dict.fromkeys((*q.quality_flags, FLAG_UNMAPPED)))))
            continue
        flags = list(q.quality_flags)
        if abs((_utc(g["kickoff_utc"]) - _utc(q.kickoff_utc)).total_seconds()) > KICKOFF_FLAG_MINUTES * 60:
            flags.append(FLAG_KICKOFF_DIFFERS)
        out.append(replace(q, event_id=str(g["game_id"]), season=int(g["season"]), week=int(g["week"]),
                           quality_flags=tuple(dict.fromkeys(flags))))
    return out


__all__ = ["FLAG_KICKOFF_DIFFERS", "FLAG_UNMAPPED", "TEAM_NAME_TO_CODE", "map_quotes_to_games", "match_game",
           "team_code"]
