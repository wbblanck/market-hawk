"""``SportsLeaderProvider``: the optional Sports Leader MCP server as a ``MarketDataProvider``.

Flow for ``fetch_quotes(date=...)``: ``get_scoreboard`` (sport football, league nfl, date
YYYYMMDD) lists ESPN event ids; then per game ``get_game_odds`` (sportsbooks) and
``get_market_odds`` (Kalshi / Polymarket / Polymarket US). With ``event_ids`` the scoreboard is
skipped. A failure for one game or venue is recorded in ``last_fetch_issues`` and the fetch
carries on; it never fails the whole call. Games whose scoreboard state is ``post`` are skipped.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from ..base import ProviderQuote, ProviderUnavailable
from .client import ENV_COMMAND, SportsLeaderClient
from .mapping import EventMapping, map_event
from .odds import game_odds_to_quotes
from .prediction_markets import market_odds_to_quotes
from .schemas import Scoreboard, ScoreboardEvent

log = logging.getLogger(__name__)

PROVIDER_NAME = "sports_leader"
SPORT, LEAGUE = "football", "nfl"
MAX_SCOREBOARD_PAGES = 10


class SportsLeaderProvider:
    """Optional provider backed by the Sports Leader MCP server over stdio.

    ``call_tool`` injects a transport (``(name, args) -> dict``); without it the server is
    launched from ``command`` or ``NFLMI_SPORTS_LEADER_CMD``. ``season`` / ``week``, when both
    are given, let quotes carry the repo ``event_id`` (``{season}_{week:02d}_{away}_{home}``);
    ``week`` alone also lets ``fetch_quotes()`` with no date ask the scoreboard for that week.
    """

    name = PROVIDER_NAME

    def __init__(self, call_tool: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None, *,
                 command: str | list[str] | None = None, season: int | None = None, week: int | None = None,
                 include_sportsbooks: bool = True, include_prediction_markets: bool = True,
                 timeout_s: float | None = None):
        self._call_tool = call_tool
        self._command = command
        self._timeout_s = timeout_s
        self.season = season
        self.week = week
        self.include_sportsbooks = include_sportsbooks
        self.include_prediction_markets = include_prediction_markets
        self.last_fetch_issues: list[dict[str, Any]] = []

    def _client(self) -> SportsLeaderClient:
        return SportsLeaderClient(self._call_tool, command=self._command, timeout_s=self._timeout_s)

    def available(self) -> bool:
        """SDK importable and a launch command configured (or a transport injected). No network call."""
        return self._client().configured()

    def fetch_quotes(self, *, date: str | None = None, event_ids: list[str] | None = None) -> list[ProviderQuote]:
        client = self._client()
        if not client.configured():
            raise ProviderUnavailable(
                f"Sports Leader MCP is not available: install the Python 'mcp' SDK and set {ENV_COMMAND} "
                "(e.g. 'node /path/to/sports-leader-mcp/dist/index.js')")
        self.last_fetch_issues = []
        quotes: list[ProviderQuote] = []
        with client:
            if event_ids:
                games: list[tuple[str, ScoreboardEvent | None]] = [(str(e), None) for e in event_ids]
            else:
                games = [(e.id, e) for e in self._scoreboard(client, date)]
            for event_id, event in games:
                quotes += self._fetch_game(client, event_id, event)
        return quotes

    # -- internals ---------------------------------------------------------------------------

    def _issue(self, event_id: str | None, stage: str, reason: str, **extra: Any) -> None:
        item = {"provider_event_id": event_id, "stage": stage, "reason": reason, **extra}
        log.warning("sports_leader skip: %s", item)
        self.last_fetch_issues.append(item)

    def _scoreboard(self, client: SportsLeaderClient, date: str | None) -> list[ScoreboardEvent]:
        args: dict[str, Any] = {"sport": SPORT, "league": LEAGUE}
        if date:
            args["date"] = date.replace("-", "")
        elif self.week is not None:
            args["week"] = self.week
        events: dict[str, ScoreboardEvent] = {}
        offset = 0
        for _ in range(MAX_SCOREBOARD_PAGES):
            try:
                board = Scoreboard.model_validate(client.call("get_scoreboard", **args, offset=offset or None))
            except Exception as e:
                if not events:
                    raise
                self._issue(None, "scoreboard", repr(e), offset=offset)
                break
            for ev in board.events:
                events.setdefault(ev.id, ev)
            page = board.page
            if page is None or page.returned <= 0 or page.offset + page.returned >= page.total:
                break
            offset = page.offset + page.returned
        return list(events.values())

    def _identity(self, event_id: str, event: ScoreboardEvent | None) -> EventMapping | None:
        if event is None:
            return None  # built from each response instead
        return map_event(event_id, home=event.side("home").abbrev, away=event.side("away").abbrev,
                         start_utc=event.date, season=self.season, week=self.week)

    def _fetch_game(self, client: SportsLeaderClient, event_id: str, event: ScoreboardEvent | None) -> list[ProviderQuote]:
        state = (event.status.state if event else None) or ""
        if state == "post":
            self._issue(event_id, "game", "game_final")
            return []
        ident = self._identity(event_id, event)
        live_flag = ("game_in_progress",) if state == "in" else ()
        out: list[ProviderQuote] = []
        if self.include_sportsbooks:
            try:
                resp = client.call("get_game_odds", sport=SPORT, league=LEAGUE, eventId=event_id)
                out += game_odds_to_quotes(resp, provider_event_id=event_id, season=self.season, week=self.week,
                                           identity=ident)
            except Exception as e:  # noqa: BLE001 - one game must not sink the fetch
                self._issue(event_id, "sportsbook_odds", repr(e))
        if self.include_prediction_markets:
            try:
                resp = client.call("get_market_odds", sport=SPORT, league=LEAGUE, eventId=event_id)
                pm, issues = market_odds_to_quotes(resp, requested_event_id=event_id, season=self.season,
                                                   week=self.week, identity=ident)
                out += pm
                for item in issues:
                    self._issue(item.pop("provider_event_id"), item.pop("stage"), item.pop("reason"), **item)
            except Exception as e:  # noqa: BLE001
                self._issue(event_id, "prediction_markets", repr(e))
        if live_flag:
            out = [_with_flags(q, live_flag) for q in out]
        return out


def _with_flags(q: ProviderQuote, flags: tuple[str, ...]) -> ProviderQuote:
    return replace(q, quality_flags=tuple(dict.fromkeys((*q.quality_flags, *flags))))
