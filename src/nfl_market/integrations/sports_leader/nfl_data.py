"""NFL statistics from the Sports Leader historical Parquet export, as pandas DataFrames.

IMPORTANT: these are game results, box scores, player logs, rosters, injury and depth-chart
rows. They are statistics and context, NOT historical sportsbook or exchange quotes. The
server does not expose historical odds; this repo's market history comes only from its own
captured snapshots.

Each wrapper calls one export tool (sports-leader-mcp ``src/data/tools.ts``) with
``league="nfl"`` and returns the ``rows`` of the result. Result metadata (``dataset``,
``seasons``, ``updated_at``, ``rowCount``, ``columns``) is attached as ``DataFrame.attrs``.
Columns are whatever the export publishes; ``game_id`` is the ESPN event id. The server
needs network access to GitHub Releases for these tools and caps ``limit`` at 250 rows.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .client import SportsLeaderClient
from .schemas import ExportResult

LEAGUE = "nfl"
NFL_STAT_GROUPS = ("passing", "rushing", "receiving")  # get_player_game_logs statGroup for nfl


def _frame(result: dict[str, Any]) -> pd.DataFrame:
    parsed = ExportResult.model_validate(result)
    frame = pd.DataFrame(parsed.rows, columns=parsed.columns or None)
    frame.attrs.update({
        "source": "sports_leader",
        "kind": "statistics_not_quotes",
        "dataset": parsed.dataset,
        "seasons": parsed.seasons,
        "updated_at": parsed.updated_at,
        "row_count": parsed.rowCount,
    })
    return frame


def _query(client: SportsLeaderClient, tool: str, *, season: int | None = None, seasons: list[int] | None = None,
           date_from: str | None = None, date_to: str | None = None, team: str | None = None,
           opponent: str | None = None, player: str | None = None, columns: list[str] | None = None,
           sort: str | None = None, order: str | None = None, limit: int | None = None,
           **extra: Any) -> pd.DataFrame:
    args = dict(league=LEAGUE, season=season, seasons=seasons, team=team, opponent=opponent, player=player,
                columns=columns, sort=sort, order=order, limit=limit, **extra)
    args["from"] = date_from  # tool argument names are `from` / `to`
    args["to"] = date_to
    return _frame(client.call(tool, **args))


def games(client: SportsLeaderClient, **kw: Any) -> pd.DataFrame:
    """Schedule and results, one row per game (``get_games``). Filters: season(s), date_from/date_to
    (YYYY-MM-DD), team, opponent, columns, sort, order, limit."""
    return _query(client, "get_games", **kw)


def team_box_scores(client: SportsLeaderClient, **kw: Any) -> pd.DataFrame:
    """ESPN team box line, one row per team per game (``get_team_box_scores``)."""
    return _query(client, "get_team_box_scores", **kw)


def team_game_stats(client: SportsLeaderClient, **kw: Any) -> pd.DataFrame:
    """Full team counting stats for and against, one row per team per game (``get_team_game_stats``)."""
    return _query(client, "get_team_game_stats", **kw)


def player_game_logs(client: SportsLeaderClient, stat_group: str, **kw: Any) -> pd.DataFrame:
    """Player box lines (``get_player_game_logs``); NFL needs stat_group passing, rushing or receiving."""
    if stat_group not in NFL_STAT_GROUPS:
        raise ValueError(f"stat_group must be one of {NFL_STAT_GROUPS}, got {stat_group!r}")
    return _query(client, "get_player_game_logs", statGroup=stat_group, **kw)


def rosters(client: SportsLeaderClient, **kw: Any) -> pd.DataFrame:
    """Roster rows with valid_from / valid_to (``get_rosters``)."""
    return _query(client, "get_rosters", **kw)


def injury_reports(client: SportsLeaderClient, **kw: Any) -> pd.DataFrame:
    """Historical injury rows with valid_from / valid_to (``get_injury_reports``; not the live report)."""
    return _query(client, "get_injury_reports", **kw)


def depth_charts(client: SportsLeaderClient, **kw: Any) -> pd.DataFrame:
    """NFL depth-chart slots with the span each was current (``get_depth_charts``)."""
    return _query(client, "get_depth_charts", **kw)
