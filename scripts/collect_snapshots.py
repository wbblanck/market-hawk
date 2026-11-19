"""Collect one timestamped market snapshot from the configured providers.

Every record is stored raw in ``market_snapshots``; clean pre-kickoff sportsbook spread/total
quotes are also routed into ``quote_snapshots``. Example:

    python scripts/collect_snapshots.py --date 2026-11-22 --providers sports_leader
"""
from __future__ import annotations

import argparse
import sys
from datetime import date as _date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_market.config import settings


def _sports_leader(season: int | None = None, week: int | None = None):
    try:
        from nfl_market.integrations.sports_leader import SportsLeaderProvider
    except ImportError as exc:
        raise LookupError(
            f"provider 'sports_leader' is not installed in this checkout "
            f"(nfl_market.integrations.sports_leader.SportsLeaderProvider could not be imported: {exc})"
        ) from exc
    return SportsLeaderProvider(season=season, week=week)


def _schedule():
    """nflverse schedule for mapping odds-feed events to game ids (None if not downloaded)."""
    try:
        from nfl_market.data.games import load_games

        return load_games()
    except FileNotFoundError:
        return None


def _parlay_api(season: int | None = None, week: int | None = None):
    from nfl_market.integrations.sharp import parlay_api_provider

    return parlay_api_provider(books=(), regions="us", markets=("spreads", "totals", "h2h"), games=_schedule())


def _oddslink(season: int | None = None, week: int | None = None):
    from nfl_market.integrations.sharp import oddslink_provider

    return oddslink_provider(markets=("spreads", "totals"), games=_schedule())


PROVIDER_FACTORIES = {"sports_leader": _sports_leader, "parlay_api": _parlay_api, "oddslink": _oddslink}


def resolve_providers(names: list[str], **options) -> list:
    providers = []
    for name in names:
        factory = PROVIDER_FACTORIES.get(name)
        if factory is None:
            raise LookupError(f"unknown provider {name!r}; known: {sorted(PROVIDER_FACTORIES)}")
        providers.append(factory(**options))
    return providers


def _valid_date(value: str) -> str:
    try:
        _date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from exc
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Capture a timestamped NFL market snapshot into DuckDB.")
    p.add_argument("--date", type=_valid_date, help="Game day YYYY-MM-DD (US-Eastern), passed to providers")
    p.add_argument("--event-id", action="append", dest="event_ids", default=None,
                   help="Provider event id to collect (repeatable)")
    p.add_argument("--db", default=settings.database_path, help=f"DuckDB path (default: {settings.database_path})")
    p.add_argument("--providers", nargs="+", default=["sports_leader"],
                   help="Provider names, space or comma separated (default: sports_leader)")
    p.add_argument("--season", type=int, default=None,
                   help="NFL season; with --week, lets providers map games to canonical event ids")
    p.add_argument("--week", type=int, default=None, help="NFL week (see --season)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    names = [n.strip() for item in args.providers for n in item.split(",") if n.strip()]
    try:
        providers = resolve_providers(names, season=args.season, week=args.week)
    except LookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    from nfl_market.collectors.snapshot_collector import SnapshotCollector

    report = SnapshotCollector(args.db, providers).collect(date=args.date, event_ids=args.event_ids)
    print(f"Snapshot collection -> {args.db}")
    print(report.summary())
    return 0 if all(r.status in ("ok", "partial") for r in report.runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
