"""Live provider for odds-api-v4-compatible feeds: The Odds API and ParlayAPI.

ParlayAPI (parlay-api.com/v1) uses the same schema; Will already holds a key for it (used by
his nfl_ats capture-market workflow). Use :func:`parlay_api_provider` for it; its key is read
from ``NFLMI_PARLAY_API_KEY`` or, if unset, his existing ``PARLAY_API_KEY`` env name. ParlayAPI's
terms are unverified (nfl_ats docs/data_audit.md 0.5), and its feed has often lacked Pinnacle
since 2026-10-08.

Why this source (docs/data_sources/sharp_feeds.md): the free Starter tier (500 credits/month,
checked 2026-11-19) serves live odds; Pinnacle is in the ``eu`` region (bookmaker key
``pinnacle``; the provider notes these odds come from Pinnacle's public website "which may
incur a delay"). Its terms allow storing data and using it in analytical tools and to train
models; they forbid reselling or redistributing the data as a product. Circa and BookMaker
are not carried. Historical snapshots need a paid plan and are not implemented here.

The API key is read from the ``NFLMI_ODDS_API_KEY`` environment variable and is never
hard-coded or logged. Without a key (or without httpx) ``available()`` is False.

Response shape (v4 ``/sports/{sport}/odds``): a list of events with ``id``, ``commence_time``,
``home_team``, ``away_team`` and ``bookmakers[].markets[].outcomes[]`` (``name``, ``price``,
``point``). Like every provider, this one never assigns capture time: the snapshot collector
stamps ``captured_at_utc``. ``last_update`` is kept as ``source_timestamp_utc`` (provenance only).
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from ..base import VENUE_SPORTSBOOK, ProviderQuote, ProviderUnavailable

PROVIDER = "the_odds_api"
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
API_KEY_ENV = "NFLMI_ODDS_API_KEY"
PARLAY_PROVIDER = "parlay_api"
PARLAY_BASE_URL = "https://parlay-api.com/v1"
PARLAY_KEY_ENVS = ("NFLMI_PARLAY_API_KEY", "PARLAY_API_KEY")
SHARP_BOOKS = ("pinnacle",)          # the only one of Pinnacle/Circa/BookMaker it lists (2026-11-19)
MARKET_MAP = {"spreads": "spread", "totals": "total"}
FLAG_PUBLIC_SITE_DELAY = "public_site_feed_may_delay"
_EASTERN = ZoneInfo("America/New_York")


def _ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _american(price: Any) -> int | None:
    try:
        f = float(price)
    except (TypeError, ValueError):
        return None
    return int(f) if f == int(f) and (f <= -100 or f >= 100) else None


def parse_odds_api_events(events: list[dict[str, Any]], books: tuple[str, ...] = SHARP_BOOKS,
                          provider: str = PROVIDER) -> list[ProviderQuote]:
    """Turn a v4 odds response into ProviderQuotes for spread/total of the given books.

    Spread selections are home/away with each side's own point; total selections over/under.
    An empty ``books`` keeps every book. Outcomes that cannot be mapped (unknown team name, missing point, non-American price)
    are kept with ``None`` fields and a quality flag rather than guessed.
    """
    out: list[ProviderQuote] = []
    wanted = {b.lower() for b in books}
    for ev in events:
        home, away = ev.get("home_team"), ev.get("away_team")
        kickoff = _ts(ev.get("commence_time"))
        for bm in ev.get("bookmakers", []):
            key = str(bm.get("key", "")).lower()
            if wanted and key not in wanted:
                continue
            for mk in bm.get("markets", []):
                market = MARKET_MAP.get(mk.get("key"))
                if market is None:
                    continue
                for oc in mk.get("outcomes", []):
                    name = str(oc.get("name", ""))
                    flags: list[str] = []
                    if market == "spread":
                        selection = "home" if name == home else "away" if name == away else name.lower()
                        if selection not in ("home", "away"):
                            flags.append("unmapped_team_name")
                    else:
                        selection = name.lower()
                    point = oc.get("point")
                    odds = _american(oc.get("price"))
                    if point is None:
                        flags.append("missing_point")
                    if odds is None:
                        flags.append("invalid_american_odds")
                    src_ts = _ts(mk.get("last_update") or bm.get("last_update"))
                    if src_ts is None:
                        flags.append("no_source_timestamp")
                    if key == "pinnacle" and provider == PROVIDER:
                        flags.append(FLAG_PUBLIC_SITE_DELAY)
                    out.append(ProviderQuote(
                        provider=provider, venue=key, venue_kind=VENUE_SPORTSBOOK,
                        provider_event_id=str(ev.get("id")), market=market, selection=selection,
                        line=None if point is None else float(point), odds_american=odds,
                        kickoff_utc=kickoff, home_team=home, away_team=away, source_timestamp_utc=src_ts,
                        raw={"event_id": ev.get("id"), "bookmaker": key, "market": mk.get("key"), "outcome": oc},
                        quality_flags=tuple(flags)))
    return out


class OddsApiProvider:
    """MarketDataProvider for an odds-api-v4-compatible feed (sharp books only by default).

    ``api_key_env`` may name several env variables; the first one set is used. The key is
    never stored anywhere but this object and is never logged.
    """

    def __init__(self, api_key: str | None = None, *, books: tuple[str, ...] = SHARP_BOOKS,
                 base_url: str = BASE_URL, api_key_env: str | tuple[str, ...] = API_KEY_ENV,
                 name: str = PROVIDER, regions: str = "us,eu", timeout: float = 20.0, get=None):
        envs = (api_key_env,) if isinstance(api_key_env, str) else tuple(api_key_env)
        self._key_envs = envs
        self._api_key = api_key if api_key is not None else next(
            (os.environ[e] for e in envs if os.environ.get(e)), None)
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.books = books
        self.regions = regions
        self.timeout = timeout
        self._get = get  # injectable HTTP GET for tests: (url, params) -> list[dict]

    def available(self) -> bool:
        if not self._api_key:
            return False
        if self._get is not None:
            return True
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False
        return True

    def _request(self, url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if self._get is not None:
            return self._get(url, params)
        import httpx

        try:
            resp = httpx.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except httpx.HTTPError as e:  # never let the key (a query parameter) reach logs
            raise ProviderUnavailable(f"{self.name} request failed: {type(e).__name__}") from None
        return resp.json()

    def fetch_quotes(self, *, date: str | None = None, event_ids: list[str] | None = None) -> list[ProviderQuote]:
        if not self.available():
            raise ProviderUnavailable(f"{self.name} needs one of {', '.join(self._key_envs)} (and httpx)")
        params: dict[str, Any] = {"apiKey": self._api_key, "markets": "spreads,totals",
                                  "oddsFormat": "american", "dateFormat": "iso"}
        if self.books:
            params["bookmakers"] = ",".join(self.books)
        else:
            params["regions"] = self.regions
        if event_ids:
            params["eventIds"] = ",".join(event_ids)
        events = self._request(f"{self.base_url}/sports/{SPORT}/odds", params)
        quotes = parse_odds_api_events(events, self.books, provider=self.name)
        if date:
            quotes = [q for q in quotes if q.kickoff_utc is not None
                      and q.kickoff_utc.astimezone(_EASTERN).date().isoformat() == date]
        return quotes


def parlay_api_provider(books: tuple[str, ...] = SHARP_BOOKS, **kw) -> OddsApiProvider:
    """ParlayAPI with Will's existing key (env NFLMI_PARLAY_API_KEY, else PARLAY_API_KEY).
    Pass ``books=()`` for every book it carries (tier-3 all-book median)."""
    return OddsApiProvider(books=books, base_url=PARLAY_BASE_URL, api_key_env=PARLAY_KEY_ENVS,
                           name=PARLAY_PROVIDER, **kw)
