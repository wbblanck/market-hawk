"""Convert ``get_game_odds`` responses into sportsbook ``ProviderQuote`` records.

Per book (``odds[]`` entry, venue = its ``provider`` name such as "Draft Kings") this emits
up to six records: spread home/away, total over/under, moneyline home/away. Lines come from
the book's ``current`` block, falling back to the normalized numeric fields
(``homeSpread``, ``total`` / ``overUnder``); ``open`` lines are never used. Prices are
``home.spreadOdds`` / ``away.spreadOdds``, ``overOdds`` / ``underOdds`` and
``home.moneyLine`` / ``away.moneyLine`` (falling back to ``homeMoneyLine`` /
``awayMoneyLine`` and then the ``current`` strings). A side with no usable price is skipped;
its partner carries a ``partner_missing_price`` flag. Values are never invented.

ESPN odds carry no quote timestamp, so every record has ``source_timestamp_utc=None`` and the
flag ``no_source_timestamp``.
"""
from __future__ import annotations

import math
from typing import Any

from ...markets.odds import is_valid_american
from ..base import VENUE_SPORTSBOOK, ProviderQuote
from .mapping import EventMapping, map_event
from .schemas import BookOdds, GameOdds

PROVIDER = "sports_leader"


def to_float(value: Any) -> float | None:
    """Parse a number or a signed string ("+6.5", "-285"). Empty, non-numeric and NaN -> None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
    else:
        text = str(value).strip().replace("−", "-")
        if not text:
            return None
        try:
            out = float(text)
        except ValueError:
            return None
    return None if math.isnan(out) or math.isinf(out) else out


def to_american(value: Any) -> tuple[int | None, str | None]:
    """Return (odds, flag). Missing -> (None, None); malformed -> (None, 'invalid_odds').
    ESPN's display string "EVEN" is +100."""
    if isinstance(value, str) and value.strip().upper() == "EVEN":
        return 100, None
    num = to_float(value)
    if num is None:
        return None, ("invalid_odds" if value not in (None, "") else None)
    if num != int(num) or not is_valid_american(int(num)):
        return None, "invalid_odds"
    return int(num), None


def _first_odds(*candidates: Any) -> tuple[int | None, tuple[str, ...]]:
    flags: list[str] = []
    for c in candidates:
        odds, flag = to_american(c)
        if odds is not None:
            return odds, tuple(flags)
        if flag:
            flags.append(flag)
    return None, tuple(dict.fromkeys(flags))


def _book_quotes(book_raw: dict[str, Any], ident: EventMapping) -> list[ProviderQuote]:
    book = BookOdds.model_validate(book_raw)
    venue = (book.provider or "").strip() or (f"provider_{book.providerId}" if book.providerId else "unknown_book")
    cur = book.current
    base_flags = ["no_source_timestamp", *ident.flags]
    if not (book.provider or "").strip():
        base_flags.append("missing_book_name")

    def make(market: str, selection: str, line: float | None, odds: int | None, flags: list[str]) -> ProviderQuote:
        return ProviderQuote(
            provider=PROVIDER, venue=venue, venue_kind=VENUE_SPORTSBOOK, provider_event_id=ident.provider_event_id,
            market=market, selection=selection, line=line, odds_american=odds,
            provider_market_id=book.providerId, event_id=ident.event_id, season=ident.season, week=ident.week,
            kickoff_utc=ident.kickoff_utc, home_team=ident.home_team, away_team=ident.away_team,
            source_timestamp_utc=None, raw=book_raw, quality_flags=tuple(dict.fromkeys(base_flags + flags)),
        )

    def pair(market: str, sides: list[tuple[str, float | None, tuple[int | None, tuple[str, ...]]]],
             pair_flags: list[str]) -> list[ProviderQuote]:
        out = []
        priced = [s for s in sides if s[2][0] is not None]
        for selection, line, (odds, odds_flags) in priced:
            flags = [*pair_flags, *odds_flags]
            if len(priced) < len(sides):
                flags.append("partner_missing_price")
            if line is None and market != "moneyline":
                flags.append("missing_line")
            out.append(make(market, selection, line, odds, flags))
        return out

    quotes: list[ProviderQuote] = []

    # Spread: each side keeps its own line.
    home_line = to_float(cur.homeSpread if cur else None)
    away_line = to_float(cur.awaySpread if cur else None)
    spread_flags: list[str] = []
    if home_line is None and (fallback := to_float(book.homeSpread)) is not None:
        home_line = fallback
        spread_flags.append("line_from_normalized_field")
    if away_line is None and home_line is not None:
        away_line = -home_line if home_line != 0 else 0.0
        spread_flags.append("away_line_derived_from_home")
    if home_line is not None and away_line is not None and home_line != -away_line:
        spread_flags.append("spread_pair_not_opposite")  # reported as-is, never "fixed"
    quotes += pair("spread", [
        ("home", home_line, _first_odds(book.home.spreadOdds)),
        ("away", away_line, _first_odds(book.away.spreadOdds)),
    ], spread_flags)

    # Total: over and under share one total.
    total = to_float(cur.total if cur else None)
    total_flags: list[str] = []
    if total is None:
        total = to_float(book.total)
        if total is None:
            total = to_float(book.overUnder)
        if total is not None:
            total_flags.append("line_from_normalized_field")
    quotes += pair("total", [
        ("over", total, _first_odds(book.overOdds)),
        ("under", total, _first_odds(book.underOdds)),
    ], total_flags)

    # Moneyline.
    quotes += pair("moneyline", [
        ("home", None, _first_odds(book.home.moneyLine, book.homeMoneyLine, cur.homeMoneyLine if cur else None)),
        ("away", None, _first_odds(book.away.moneyLine, book.awayMoneyLine, cur.awayMoneyLine if cur else None)),
    ], [])
    return quotes


def game_odds_to_quotes(response: dict[str, Any], *, provider_event_id: str | None = None,
                        season: int | None = None, week: int | None = None,
                        identity: EventMapping | None = None) -> list[ProviderQuote]:
    """Convert one ``get_game_odds`` response. ``identity`` overrides the mapping built from the response."""
    game = GameOdds.model_validate(response)
    event_id = str(provider_event_id or game.eventId or "")
    if not event_id:
        raise ValueError("get_game_odds response has no eventId")
    ident = identity or map_event(event_id, home=game.homeTeam.abbrev, away=game.awayTeam.abbrev,
                                  start_utc=game.startUtc, season=season, week=week)
    quotes: list[ProviderQuote] = []
    for book_raw in game.odds:
        if isinstance(book_raw, dict):
            quotes += _book_quotes(book_raw, ident)
    return quotes
