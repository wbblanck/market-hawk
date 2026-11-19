"""Read-only analytics service: the one place the dashboard and the MCP server read step-4 data.

Everything here is a validated, size-capped pass-through to a
:class:`~nfl_market.markets.interface.MarketReader` (prices, edges, the paper bet log, CLV
aggregates, model health) plus :class:`~nfl_market.services.market_service.MarketService`
for quote-store health. It computes no prices, edges or CLV of its own: the dashboard and
the MCP tools call the same methods with the same inputs, so they show identical results
(``docs/MCP_SPEC.md``, service parity).

Input rules (violations raise ``ValueError`` with a readable message):

* ``limit`` is an int in ``1..MAX_LIMIT``;
* ``season`` is an int in ``MIN_SEASON..MAX_SEASON`` and never the locked 2025 holdout;
* ``min_ev`` is a finite number;
* ``as_of`` is a timestamp or ISO-8601 string (naive means UTC);
* ``by`` is a subset of ``CLV_GROUP_KEYS`` that always includes ``close_tier`` (tiers are
  never pooled: the ``all_book_median`` weak fallback stays separate);
* ``game_ids`` is at most ``MAX_GAME_IDS`` non-empty strings.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from numbers import Real
from typing import Any, Sequence

import pandas as pd

from ..config import TIER_ALL_BOOK_MEDIAN
from ..holdout import holdout_seasons
from ..markets.interface import (
    BET_LOG_COLUMNS,
    CLV_SUMMARY_COLUMNS,
    EDGE_COLUMNS,
    MODEL_HEALTH_COLUMNS,
    PRICE_COLUMNS,
    MarketReader,
    check_columns,
    empty_frame,
)
from .market_service import MarketService

MAX_LIMIT = 500
MAX_PRICE_ROWS = 2000
MAX_SUMMARY_ROWS = 500
MAX_HEALTH_ROWS = 2000
MAX_GAME_IDS = 100
MAX_GAME_ID_LENGTH = 64
MIN_SEASON, MAX_SEASON = 1999, 2100
CLV_GROUP_KEYS: tuple[str, ...] = ("close_tier", "market")
#: Label shown for a bet whose close is not attached yet (``line_close_tier == ""``).
PENDING_TIER = "pending"
WEAK_FALLBACK_LABEL = "weak fallback"


class EmptyMarketReader:
    """A reader with no data: every method returns an empty contract frame."""

    def prices(self, game_ids: Sequence[str] | None = None) -> pd.DataFrame:
        return empty_frame(PRICE_COLUMNS)

    def edges(self, as_of: pd.Timestamp | None = None, *, min_ev: float | None = None) -> pd.DataFrame:
        return empty_frame(EDGE_COLUMNS)

    def bet_log(self, season: int | None = None) -> pd.DataFrame:
        return empty_frame(BET_LOG_COLUMNS)

    def clv_summary(self, by: Sequence[str] = CLV_GROUP_KEYS) -> pd.DataFrame:
        return empty_frame(CLV_SUMMARY_COLUMNS)

    def model_health(self) -> pd.DataFrame:
        return empty_frame(MODEL_HEALTH_COLUMNS)


def default_reader(database_path: str | None = None, model_health_path: str | None = None) -> MarketReader:
    """``StoreMarketReader`` when importable, else :class:`EmptyMarketReader`."""
    try:
        from ..markets.reader import StoreMarketReader
    except ImportError:
        return EmptyMarketReader()
    return StoreMarketReader(database_path, model_health_path)


# ------------------------------------------------------------------ validation
def validate_limit(limit: Any, maximum: int = MAX_LIMIT) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError(f"limit must be an integer between 1 and {maximum}, got {limit!r}")
    if not 1 <= limit <= maximum:
        raise ValueError(f"limit must be between 1 and {maximum}, got {limit}")
    return limit


def validate_season(season: Any) -> int | None:
    if season is None:
        return None
    if isinstance(season, bool) or not isinstance(season, int):
        raise ValueError(f"season must be an integer year, got {season!r}")
    if not MIN_SEASON <= season <= MAX_SEASON:
        raise ValueError(f"season must be between {MIN_SEASON} and {MAX_SEASON}, got {season}")
    if season in holdout_seasons():
        raise ValueError(f"season {season} is the locked holdout; it is never in the bet log")
    return season


def validate_min_ev(min_ev: Any) -> float | None:
    if min_ev is None:
        return None
    if isinstance(min_ev, bool) or not isinstance(min_ev, Real) or not math.isfinite(float(min_ev)):
        raise ValueError(f"min_ev must be a finite number, got {min_ev!r}")
    return float(min_ev)


def validate_as_of(as_of: Any) -> pd.Timestamp | None:
    """Parse ``as_of`` into a tz-aware UTC timestamp; naive values are taken as UTC."""
    if as_of is None:
        return None
    if not isinstance(as_of, (str, datetime, pd.Timestamp)):
        raise ValueError(f"as_of must be an ISO-8601 timestamp string, got {as_of!r}")
    try:
        ts = pd.Timestamp(as_of)
    except (ValueError, TypeError) as e:
        raise ValueError(f"as_of is not a valid ISO-8601 timestamp: {as_of!r}") from e
    if ts is pd.NaT:
        raise ValueError(f"as_of is not a valid ISO-8601 timestamp: {as_of!r}")
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def validate_by(by: Any) -> tuple[str, ...]:
    if by is None:
        return CLV_GROUP_KEYS
    if isinstance(by, str) or not isinstance(by, Sequence):
        raise ValueError(f"by must be a list of group keys from {list(CLV_GROUP_KEYS)}, got {by!r}")
    keys = list(by)
    unknown = [k for k in keys if k not in CLV_GROUP_KEYS]
    if unknown:
        raise ValueError(f"unknown CLV group keys {unknown}; allowed: {list(CLV_GROUP_KEYS)}")
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate CLV group keys in {keys}")
    if "close_tier" not in keys:
        raise ValueError("by must include 'close_tier': close tiers are always reported separately")
    return tuple(k for k in CLV_GROUP_KEYS if k in keys)


def validate_game_ids(game_ids: Any) -> list[str] | None:
    if game_ids is None:
        return None
    if isinstance(game_ids, str) or not isinstance(game_ids, Sequence):
        raise ValueError("game_ids must be a list of game id strings")
    ids = list(game_ids)
    if not ids:
        raise ValueError("game_ids must not be empty (pass None for all games)")
    if len(ids) > MAX_GAME_IDS:
        raise ValueError(f"at most {MAX_GAME_IDS} game_ids per request, got {len(ids)}")
    for g in ids:
        if not isinstance(g, str) or not g.strip() or len(g) > MAX_GAME_ID_LENGTH:
            raise ValueError(f"invalid game_id {g!r}")
    return [g.strip() for g in ids]


# --------------------------------------------------------------------- helpers
def ev_plus_eligible(edges: pd.DataFrame) -> pd.Series:
    """True where an edge row may carry an 'EV+' badge (docs/DASHBOARD_SPEC.md).

    Requires positive finite ``ev_per_unit``, a finite ``p_win``, no quality flags and a
    non-synthetic quote. Everything else gets no badge.
    """
    if edges.empty:
        return pd.Series(False, index=edges.index, dtype=bool)
    ev = pd.to_numeric(edges["ev_per_unit"], errors="coerce")
    p = pd.to_numeric(edges["p_win"], errors="coerce")
    flags = edges["flags"].fillna("").astype(str).str.strip()
    synthetic = edges["synthetic"].astype(object).where(edges["synthetic"].notna(), True).astype(bool)
    return ((ev > 0) & p.notna() & (flags == "") & ~synthetic).astype(bool)


def any_synthetic(*frames: pd.DataFrame) -> bool:
    """True if any row of any frame is marked synthetic (missing marks are not counted)."""
    for f in frames:
        if "synthetic" in f.columns and not f.empty and bool(f["synthetic"].fillna(False).astype(bool).any()):
            return True
    return False


def tier_shares(tiers: pd.Series) -> dict[str, float]:
    """Share of bets per close tier; an empty tier (close not attached yet) counts as ``pending``."""
    if tiers.empty:
        return {}
    labels = tiers.fillna("").astype(str).replace("", PENDING_TIER)
    counts = labels.value_counts()
    return {str(k): float(v / len(labels)) for k, v in counts.sort_index().items()}


def tier_label(tier: str) -> str:
    """Display label for a close tier; the all-book median is always called the weak fallback."""
    if tier == TIER_ALL_BOOK_MEDIAN:
        return f"{tier} ({WEAK_FALLBACK_LABEL})"
    return tier if tier else PENDING_TIER


def _iso(ts: Any) -> str | None:
    if ts is None or ts is pd.NaT or (isinstance(ts, float) and math.isnan(ts)):
        return None
    t = pd.Timestamp(ts)
    if t is pd.NaT:
        return None
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------- service
class AnalyticsService:
    """Validated, capped, read-only access to prices, edges, the paper bet log, CLV and model health."""

    def __init__(self, reader: MarketReader | None = None, market_service: MarketService | None = None,
                 *, model_health_path: str | None = None):
        self.market_service = market_service or MarketService()
        self.reader = reader if reader is not None else default_reader(
            self.market_service.database_path, model_health_path)

    def prices(self, game_ids: Sequence[str] | None = None) -> pd.DataFrame:
        ids = validate_game_ids(game_ids)
        f = check_columns(self.reader.prices(ids), PRICE_COLUMNS, "prices")
        return f.head(MAX_PRICE_ROWS).reset_index(drop=True)

    def edges(self, as_of: Any = None, min_ev: Any = None, limit: int = 50) -> pd.DataFrame:
        ts = validate_as_of(as_of)
        ev = validate_min_ev(min_ev)
        n = validate_limit(limit)
        f = check_columns(self.reader.edges(ts, min_ev=ev), EDGE_COLUMNS, "edges")
        if ev is not None and not f.empty:
            f = f[pd.to_numeric(f["ev_per_unit"], errors="coerce") >= ev]
        return f.head(n).reset_index(drop=True)

    def bet_log(self, season: Any = None, limit: int = 200) -> pd.DataFrame:
        s = validate_season(season)
        n = validate_limit(limit)
        f = check_columns(self.reader.bet_log(s), BET_LOG_COLUMNS, "bet_log")
        return f.head(n).reset_index(drop=True)

    def clv_summary(self, by: Sequence[str] | None = CLV_GROUP_KEYS) -> pd.DataFrame:
        keys = validate_by(by)
        f = check_columns(self.reader.clv_summary(keys), CLV_SUMMARY_COLUMNS, "clv_summary")
        return f.head(MAX_SUMMARY_ROWS).reset_index(drop=True)

    def model_health(self) -> pd.DataFrame:
        f = check_columns(self.reader.model_health(), MODEL_HEALTH_COLUMNS, "model_health")
        return f.head(MAX_HEALTH_ROWS).reset_index(drop=True)

    def overview(self, now: datetime | None = None) -> dict:
        """Small summary: row counts, latest capture, quote-store health, bets by close tier, synthetic flag."""
        now = now or datetime.now(timezone.utc)
        health = self.market_service.health(now=now)
        edges = check_columns(self.reader.edges(pd.Timestamp(now)), EDGE_COLUMNS, "edges")
        bets = check_columns(self.reader.bet_log(None), BET_LOG_COLUMNS, "bet_log")
        prices = check_columns(self.reader.prices(None), PRICE_COLUMNS, "prices")
        mh = check_columns(self.reader.model_health(), MODEL_HEALTH_COLUMNS, "model_health")
        status = bets["status"].fillna("").astype(str).value_counts() if not bets.empty else pd.Series(dtype=int)
        return {
            "generated_at_utc": _iso(now),
            "health_status": health.get("status"),
            "health_warnings": list(health.get("warnings", [])),
            "latest_capture_utc": health.get("latest_capture_utc"),
            "quote_rows": int(health.get("quote_rows", 0) or 0),
            "price_rows": int(len(prices)),
            "priced_games": int(prices["game_id"].nunique()) if not prices.empty else 0,
            "edge_rows": int(len(edges)),
            "bet_count": int(len(bets)),
            "bets_by_status": {str(k): int(v) for k, v in status.sort_index().items()},
            "bets_by_line_close_tier": tier_shares(bets["line_close_tier"]),
            "bets_by_price_close_tier": tier_shares(bets["price_close_tier"]),
            "model_health_rows": int(len(mh)),
            "model_health_generated_at_utc": _iso(pd.to_datetime(mh["generated_at_utc"], utc=True).max())
            if not mh.empty else None,
            "synthetic_present": any_synthetic(self._quotes_or_empty(), edges, bets),
        }

    def _quotes_or_empty(self) -> pd.DataFrame:
        try:
            return self.market_service.quotes()
        except Exception:  # noqa: BLE001 - an unreadable store is reported by health(), not here
            return pd.DataFrame()


__all__ = [
    "AnalyticsService", "CLV_GROUP_KEYS", "EmptyMarketReader", "MAX_LIMIT", "PENDING_TIER", "WEAK_FALLBACK_LABEL",
    "any_synthetic", "default_reader", "ev_plus_eligible", "tier_label", "tier_shares", "validate_as_of",
    "validate_by", "validate_game_ids", "validate_limit", "validate_min_ev", "validate_season",
]
