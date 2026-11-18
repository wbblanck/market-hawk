"""Provider interface shared by every market-data integration and the snapshot collector.

A provider turns whatever its upstream source returns into a list of ``ProviderQuote``
records. It never assigns capture time and never writes to the database: the snapshot
collector stamps ``captured_at_utc`` at the moment it receives the records and owns
persistence. Optional providers must import cleanly without their extra dependencies
installed and report ``available() == False`` instead of raising.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

# Venue kinds. A sportsbook quotes American odds on a line; an exchange or prediction
# market quotes a contract price in dollars (0-1) with optional bid/ask and size.
VENUE_SPORTSBOOK = "sportsbook"
VENUE_PREDICTION_MARKET = "prediction_market"

# Markets the collector understands. Only spread and total currently flow into the
# validated ``quote_snapshots`` table; everything is kept in the raw snapshot table.
MARKETS = {"spread", "total", "moneyline", "winner"}


class ProviderUnavailable(RuntimeError):
    """Raised when an optional provider is called but its dependency or server is missing."""


@dataclass(frozen=True)
class ProviderQuote:
    """One quoted side of one market at one venue, as the provider reported it.

    Conventions:
    - ``selection`` is lower case: home/away for spread and moneyline/winner, over/under for total.
    - ``line`` is the selected side's own line (home -3.5 and away +3.5 are two records);
      ``None`` for moneyline/winner contracts.
    - ``odds_american`` is set for sportsbook quotes; ``price`` / ``bid`` / ``ask`` (0-1 dollars)
      for prediction-market contracts. Missing values stay ``None``, never 0.
    - ``source_timestamp_utc`` is the provider's own timestamp, timezone-aware UTC, or ``None``
      when the provider does not supply one. It is never filled with the capture time.
    - ``raw`` is the provider's original payload for this record (JSON-serialisable).
    - ``quality_flags`` holds short machine-readable codes, e.g. ``"no_source_timestamp"``.
    """

    provider: str
    venue: str
    venue_kind: str
    provider_event_id: str
    market: str
    selection: str
    line: float | None = None
    odds_american: int | None = None
    price: float | None = None
    bid: float | None = None
    ask: float | None = None
    size: float | None = None
    liquidity: float | None = None
    provider_market_id: str | None = None
    event_id: str | None = None
    season: int | None = None
    week: int | None = None
    kickoff_utc: datetime | None = None
    home_team: str | None = None
    away_team: str | None = None
    source_timestamp_utc: datetime | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    quality_flags: tuple[str, ...] = ()


@runtime_checkable
class MarketDataProvider(Protocol):
    """What the snapshot collector needs from a provider."""

    name: str

    def available(self) -> bool:
        """True when the provider's optional dependency and upstream are usable."""
        ...

    def fetch_quotes(self, *, date: str | None = None, event_ids: list[str] | None = None) -> list[ProviderQuote]:
        """Return current quotes for NFL games on ``date`` (YYYY-MM-DD, US-Eastern game day)
        or for the given provider event ids. Raises ``ProviderUnavailable`` if not available."""
        ...
