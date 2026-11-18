"""Convert ``get_market_odds`` responses (one ESPN game matched to Kalshi, Polymarket and
Polymarket US) into prediction-market ``ProviderQuote`` records.

``get_market_odds`` is the right tool for one NFL game: it takes the ESPN ``eventId``, matches
each venue to that game, keeps home/away aligned with ESPN, and lists venues side by side
(``averaged: false``). It compares the winner/moneyline market only, so every record here is
``market="winner"`` with ``line=None``. Venues are never averaged; each contract side is its
own record.

Prices are 0-1 dollars exactly as the server reports them:

- Kalshi (``priceSource: "dollars"``): bid = ``yesBid``, ask = ``yesAsk``, price = ``last``
  (flag ``price_is_last_trade``); size only when a ``book`` is attached (``yesBidSize``).
- Polymarket global: ``gamma_outcome_prices`` -> price = outcome ``price``; with
  ``clob_book`` -> bid/ask from the book and price = its ``mid`` (flag ``price_is_mid``).
- Polymarket US (``bbo``): bid/ask of the ``long`` / ``short`` side, price = the server's
  ``mid`` (flag ``price_is_mid``).

``fetchedAt`` is documented upstream as "when the response was built", i.e. the server's own
fetch time, not a venue quote time. It is stored as ``source_timestamp_utc`` with the flag
``source_ts_is_fetch_time``; without it the flag is ``no_source_timestamp``.
"""
from __future__ import annotations

from typing import Any

from ..base import VENUE_PREDICTION_MARKET, ProviderQuote
from .mapping import EventMapping, map_event, match_contract_to_game, parse_utc
from .odds import to_float
from .schemas import (
    GammaEvent,
    GammaMarket,
    GammaOutcome,
    KalshiEvent,
    KalshiMarket,
    MarketOdds,
    TeamRef,
    UsSide,
)

PROVIDER = "sports_leader"


def _price(value: Any) -> float | None:
    """A 0-1 dollar price, or None (out-of-range values are not clipped, they are dropped)."""
    num = to_float(value)
    return num if num is not None and 0.0 <= num <= 1.0 else None


def _side_for_name(name: str | None, home: list[str | None], away: list[str | None],
                   home_display: str | None, away_display: str | None) -> str | None:
    """Resolve a contract's team label to home/away using names the server reported for the game."""
    if not name:
        return None
    needle = name.strip().lower()

    def hit(cands: list[str | None], display: str | None) -> bool:
        if any(c and c.strip().lower() == needle for c in cands):
            return True
        return bool(display) and display.strip().lower().endswith(" " + needle)

    h, a = hit(home, home_display), hit(away, away_display)
    return "home" if h and not a else "away" if a and not h else None


def market_odds_to_quotes(response: dict[str, Any], *, requested_event_id: str | None = None,
                          season: int | None = None, week: int | None = None,
                          identity: EventMapping | None = None) -> tuple[list[ProviderQuote], list[dict[str, Any]]]:
    """Convert one ``get_market_odds`` response. Returns ``(quotes, issues)``; an issue is a
    venue or contract that was skipped, with the reason (``no market found`` is not an issue)."""
    mo = MarketOdds.model_validate(response)
    event_id = str(requested_event_id or mo.eventId or "")
    game = mo.game
    home_ref = game.home if game else TeamRef()
    away_ref = game.away if game else TeamRef()
    ident = identity or map_event(event_id, home=home_ref.abbrev, away=away_ref.abbrev,
                                  start_utc=game.startUtc if game else None, season=season, week=week)
    ts = parse_utc(mo.fetchedAt)
    ts_flags = ("source_ts_is_fetch_time",) if ts is not None else ("no_source_timestamp",)
    quotes: list[ProviderQuote] = []
    issues: list[dict[str, Any]] = []

    def issue(venue: str, reason: str, **extra: Any) -> None:
        issues.append({"provider_event_id": event_id, "stage": "prediction_markets", "venue": venue,
                       "reason": reason, **extra})

    def make(venue: str, selection: str, market_id: str | None, raw: dict[str, Any], flags: tuple[str, ...], *,
             price: float | None, bid: float | None = None, ask: float | None = None,
             size: float | None = None) -> ProviderQuote:
        extra: list[str] = []
        if price is None and bid is None and ask is None:
            extra.append("missing_price")
        return ProviderQuote(
            provider=PROVIDER, venue=venue, venue_kind=VENUE_PREDICTION_MARKET, provider_event_id=ident.provider_event_id,
            market="winner", selection=selection, line=None, price=price, bid=bid, ask=ask, size=size,
            provider_market_id=market_id, event_id=ident.event_id, season=ident.season, week=ident.week,
            kickoff_utc=ident.kickoff_utc, home_team=ident.home_team, away_team=ident.away_team,
            source_timestamp_utc=ts, raw=raw,
            quality_flags=tuple(dict.fromkeys((*ts_flags, *ident.flags, *flags, *extra))),
        )

    for venue_model, venue_raw in zip(mo.venues, response.get("venues") or []):
        venue = venue_model.venue
        if venue_model.status == "no market found":
            continue
        match = match_contract_to_game(venue_raw, requested_event_id=event_id, response_event_id=mo.eventId,
                                       game_matched_by=mo.matchedBy)
        if not match.matched:
            issue(venue, venue_model.error or ",".join(match.flags), status=venue_model.status)
            continue
        meta = {"venue": venue, "status": venue_model.status, "matchedBy": venue_model.matchedBy,
                "fetchedAt": mo.fetchedAt, "eventId": mo.eventId}
        if venue == "kalshi":
            quotes += _kalshi(venue_model.events, meta, match.flags, make, issue)
        elif venue == "polymarket":
            quotes += _polymarket(venue_model.events, meta, match.flags, home_ref, away_ref, make, issue)
        elif venue == "polymarket_us":
            quotes += _polymarket_us(venue_model.events, meta, match.flags, make, issue)
        else:
            issue(venue, "unknown_venue")
    return quotes, issues


def _kalshi(events, meta, flags, make, issue) -> list[ProviderQuote]:
    out = []
    for ev_raw in events:
        ev = KalshiEvent.model_validate(ev_raw)
        codes = {"home": (ev.home.code if ev.home else None), "away": (ev.away.code if ev.away else None)}
        for m_raw in ev.markets:
            m = KalshiMarket.model_validate(m_raw)
            side = next((s for s, c in codes.items() if c and m.outcome and c.upper() == m.outcome.upper()), None)
            if side is None:
                issue("kalshi", "outcome_side_unresolved", ticker=m.ticker)
                continue
            size, extra = None, ["price_is_last_trade"]
            if m.book is not None and m.book.error is None:
                size = to_float(m.book.yesBidSize)
                if size is not None:
                    extra.append("size_is_best_bid_size")
            raw = {**meta, "eventTicker": ev.eventTicker, "scheduledDate": ev.scheduledDate,
                   "away": ev_raw.get("away"), "home": ev_raw.get("home"), "market": m_raw}
            out.append(make("kalshi", side, m.ticker, raw, (*flags, *extra), price=_price(m.last),
                            bid=_price(m.yesBid), ask=_price(m.yesAsk), size=size))
    return out


def _polymarket(events, meta, flags, home_ref: TeamRef, away_ref: TeamRef, make, issue) -> list[ProviderQuote]:
    out = []
    for ev_raw in events:
        ev = GammaEvent.model_validate(ev_raw)
        home_names = [home_ref.name, home_ref.displayName, home_ref.abbrev, ev.home.name if ev.home else None,
                      ev.home.code if ev.home else None]
        away_names = [away_ref.name, away_ref.displayName, away_ref.abbrev, ev.away.name if ev.away else None,
                      ev.away.code if ev.away else None]
        for m_raw in ev.markets:
            m = GammaMarket.model_validate(m_raw)
            if (m.sportsMarketType or "moneyline") != "moneyline":
                continue
            for o_raw in m.outcomes:
                o = GammaOutcome.model_validate(o_raw)
                side = _side_for_name(o.name, home_names, away_names,
                                      home_ref.displayName or (ev.home.name if ev.home else None),
                                      away_ref.displayName or (ev.away.name if ev.away else None))
                if side is None:
                    issue("polymarket", "outcome_side_unresolved", slug=m.slug, outcome=o.name)
                    continue
                raw = {**meta, "eventSlug": ev.slug, "eventMatchedBy": ev.matchedBy, "startTime": ev.startTime,
                       "market": {k: v for k, v in m_raw.items() if k != "outcomes"}, "outcome": o_raw}
                if o.priceSource == "clob_book":
                    out.append(make("polymarket", side, m.slug or ev.slug, raw, (*flags, "price_is_mid"),
                                    price=_price(o.mid), bid=_price(o.bid), ask=_price(o.ask)))
                else:
                    out.append(make("polymarket", side, m.slug or ev.slug, raw, flags, price=_price(o.price)))
    return out


def _polymarket_us(events, meta, flags, make, issue) -> list[ProviderQuote]:
    out = []
    for ev_raw in events:
        for leg in ("long", "short"):
            leg_raw = ev_raw.get(leg)
            if not isinstance(leg_raw, dict):
                continue
            s = UsSide.model_validate(leg_raw)
            side = (s.side or "").lower()
            if side not in ("home", "away"):
                issue("polymarket_us", "outcome_side_unresolved", slug=ev_raw.get("slug"), leg=leg)
                continue
            raw = {**meta, "slug": ev_raw.get("slug"), "usId": ev_raw.get("usId"),
                   "eventMatchedBy": ev_raw.get("matchedBy"), "leg": leg, "quote": leg_raw}
            out.append(make("polymarket_us", side, ev_raw.get("slug"), raw, (*flags, "price_is_mid"),
                            price=_price(s.mid), bid=_price(s.bid), ask=_price(s.ask)))
    return out
