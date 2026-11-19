"""Pydantic models for only the Sports Leader response fields this package consumes.

Field names are the server's compact output (sports-leader-mcp ``src/output-schemas.ts``,
``src/transforms.ts`` ``transformGameOdds``, ``src/markets/service.ts``). Every field is
optional because the server prunes undefined values; unknown fields are ignored here and the
original dicts are kept in ``ProviderQuote.raw``.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

Num = float | int | str | None  # ESPN sometimes sends numbers as strings


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# --- get_scoreboard -----------------------------------------------------------------------

class TeamRef(_Model):
    id: str | None = None
    abbrev: str | None = None
    name: str | None = None
    displayName: str | None = None
    location: str | None = None


class Status(_Model):
    state: str | None = None  # pre | in | post
    detail: str | None = None


class Competitor(_Model):
    homeAway: str | None = None
    team: TeamRef = TeamRef()


class ScoreboardEvent(_Model):
    id: str
    date: str | None = None  # start time, ISO-8601 UTC
    name: str | None = None
    status: Status = Status()
    competitors: list[Competitor] = []

    def side(self, home_away: str) -> TeamRef:
        for c in self.competitors:
            if (c.homeAway or "").lower() == home_away:
                return c.team
        return TeamRef()


class Page(_Model):
    offset: int = 0
    returned: int = 0
    total: int = 0


class Scoreboard(_Model):
    events: list[ScoreboardEvent] = []
    page: Page | None = None


# --- get_game_odds ------------------------------------------------------------------------

class OddsSide(_Model):
    moneyLine: Num = None
    spreadOdds: Num = None


class CurrentLines(_Model):
    total: Num = None
    homeSpread: Num = None
    awaySpread: Num = None
    homeMoneyLine: Num = None
    awayMoneyLine: Num = None


class BookOdds(_Model):
    provider: str | None = None
    providerId: str | None = None
    overUnder: Num = None
    overOdds: Num = None
    underOdds: Num = None
    home: OddsSide = OddsSide()
    away: OddsSide = OddsSide()
    current: CurrentLines | None = None
    homeMoneyLine: Num = None
    awayMoneyLine: Num = None
    homeSpread: Num = None  # normalized signed home spread
    total: Num = None


class GameOdds(_Model):
    eventId: str | None = None
    startUtc: str | None = None
    homeTeam: TeamRef = TeamRef()
    awayTeam: TeamRef = TeamRef()
    odds: list[dict[str, Any]] = []  # parsed per book with BookOdds; dict kept for raw


# --- get_market_odds ----------------------------------------------------------------------

class MarketGame(_Model):
    name: str | None = None
    startUtc: str | None = None
    easternDate: str | None = None
    away: TeamRef = TeamRef()
    home: TeamRef = TeamRef()


class VenueResult(_Model):
    venue: str
    status: str  # ok | no market found | error
    matchedBy: str | None = None
    error: str | None = None
    slug: str | None = None
    priceSource: str | None = None
    events: list[dict[str, Any]] = []


class MarketOdds(_Model):
    eventId: str | None = None
    fetchedAt: str | None = None
    matchedBy: str | None = None
    averaged: bool | None = None
    game: MarketGame | None = None
    venues: list[VenueResult] = []


class KalshiSide(_Model):
    code: str | None = None
    name: str | None = None
    nickname: str | None = None


class KalshiBook(_Model):
    error: str | None = None
    yesBidSize: Num = None


class KalshiMarket(_Model):
    ticker: str
    outcome: str | None = None
    title: str | None = None
    yesBid: Num = None
    yesAsk: Num = None
    last: Num = None
    priceSource: str | None = None
    book: KalshiBook | None = None


class KalshiEvent(_Model):
    eventTicker: str | None = None
    scheduledDate: str | None = None
    away: KalshiSide | None = None
    home: KalshiSide | None = None
    markets: list[dict[str, Any]] = []


class GammaSide(_Model):
    code: str | None = None
    name: str | None = None


class GammaOutcome(_Model):
    name: str | None = None
    tokenId: str | None = None
    price: Num = None  # gamma_outcome_prices
    bid: Num = None  # clob_book
    ask: Num = None
    mid: Num = None
    priceSource: str | None = None


class GammaMarket(_Model):
    slug: str | None = None
    sportsMarketType: str | None = None
    outcomes: list[dict[str, Any]] = []


class GammaEvent(_Model):
    slug: str | None = None
    startTime: str | None = None
    away: GammaSide | None = None
    home: GammaSide | None = None
    matchedBy: str | None = None
    markets: list[dict[str, Any]] = []


class UsSide(_Model):
    side: str | None = None  # home | away
    code: str | None = None
    name: str | None = None
    bid: Num = None
    ask: Num = None
    mid: Num = None
    priceSource: str | None = None


class UsEvent(_Model):
    slug: str | None = None
    usId: str | None = None
    matchedBy: str | None = None
    long: dict[str, Any] | None = None
    short: dict[str, Any] | None = None


# --- historical export tools (get_games, get_rosters, ...) ---------------------------------

class ExportResult(_Model):
    dataset: str | None = None
    league: str | None = None
    seasons: list[int] = []
    updated_at: str | None = None
    rowCount: int | None = None
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
