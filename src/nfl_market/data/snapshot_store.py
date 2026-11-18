"""Append-only raw market-snapshot store (our own market history, with full provenance).

Every ``ProviderQuote`` the collector receives is kept here exactly as reported, stamped
with the collector's capture time. Only clean sportsbook spread/total quotes are also
routed into the validated ``quote_snapshots`` table (see ``collectors.snapshot_collector``).

The tables live in the same DuckDB file as ``MarketStore``. There is deliberately no
UPDATE or DELETE API: re-inserting an existing ``snapshot_id`` / ``run_id`` is a counted
no-op. Timestamps are stored as naive UTC and read back timezone-aware UTC.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from ..integrations.base import ProviderQuote

SNAPSHOT_DDL = """
CREATE TABLE IF NOT EXISTS market_snapshots (
 snapshot_id VARCHAR PRIMARY KEY, run_id VARCHAR NOT NULL, captured_at_utc TIMESTAMP NOT NULL,
 source_timestamp_utc TIMESTAMP, provider VARCHAR NOT NULL, venue VARCHAR NOT NULL, venue_kind VARCHAR NOT NULL,
 provider_event_id VARCHAR NOT NULL, provider_market_id VARCHAR, market VARCHAR NOT NULL, selection VARCHAR NOT NULL,
 line DOUBLE, odds_american INTEGER, price DOUBLE, bid DOUBLE, ask DOUBLE, size DOUBLE, liquidity DOUBLE,
 event_id VARCHAR, season INTEGER, week INTEGER, kickoff_utc TIMESTAMP, home_team VARCHAR, away_team VARCHAR,
 raw_json VARCHAR NOT NULL, quality_flags VARCHAR NOT NULL, routed_to_quotes BOOLEAN NOT NULL
)"""

RUNS_DDL = """
CREATE TABLE IF NOT EXISTS collection_runs (
 run_id VARCHAR PRIMARY KEY, started_at_utc TIMESTAMP NOT NULL, finished_at_utc TIMESTAMP, provider VARCHAR NOT NULL,
 status VARCHAR NOT NULL, n_received INTEGER NOT NULL, n_stored INTEGER NOT NULL, n_routed INTEGER NOT NULL,
 n_rejected INTEGER NOT NULL, errors VARCHAR NOT NULL
)"""

SNAPSHOT_COLUMNS = [
    "snapshot_id", "run_id", "captured_at_utc", "source_timestamp_utc", "provider", "venue", "venue_kind",
    "provider_event_id", "provider_market_id", "market", "selection", "line", "odds_american", "price", "bid", "ask",
    "size", "liquidity", "event_id", "season", "week", "kickoff_utc", "home_team", "away_team", "raw_json",
    "quality_flags", "routed_to_quotes",
]
SNAPSHOT_TIMESTAMPS = ["captured_at_utc", "source_timestamp_utc", "kickoff_utc"]
RUN_COLUMNS = ["run_id", "started_at_utc", "finished_at_utc", "provider", "status", "n_received", "n_stored",
               "n_routed", "n_rejected", "errors"]
RUN_TIMESTAMPS = ["started_at_utc", "finished_at_utc"]


@dataclass(frozen=True)
class SnapshotAppendResult:
    received: int
    inserted: int
    duplicate_in_batch: int
    already_stored: int


def to_utc(value: datetime | None) -> datetime | None:
    """Timezone-aware UTC datetime, or None. Naive values are assumed to already be UTC."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _db_ts(value: datetime | None) -> datetime | None:
    """Naive-UTC value for DuckDB TIMESTAMP columns."""
    v = to_utc(value)
    return None if v is None else v.replace(tzinfo=None)


def _line_key(line: float | None) -> str | None:
    # Normalise so -3 and -3.0 hash identically.
    return None if line is None else repr(float(line))


def snapshot_id_for(q: ProviderQuote, captured_at_utc: datetime) -> str:
    """Deterministic id over provider, venue, provider_event_id, provider_market_id, market, selection,
    line and capture time. The same record captured at the same instant always gets the same id."""
    key = [q.provider, q.venue, q.provider_event_id, q.provider_market_id, q.market, q.selection,
           _line_key(q.line), to_utc(captured_at_utc).isoformat()]
    return "ms_" + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:32]


def _json_default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def snapshot_row(q: ProviderQuote, *, run_id: str, captured_at_utc: datetime, routed_to_quotes: bool,
                 extra_flags: tuple[str, ...] = ()) -> dict[str, Any]:
    """Database row for one provider record. Missing optional values stay None (NULL), never 0."""
    flags = list(dict.fromkeys([*q.quality_flags, *extra_flags]))
    return {
        "snapshot_id": snapshot_id_for(q, captured_at_utc),
        "run_id": run_id,
        "captured_at_utc": _db_ts(captured_at_utc),
        "source_timestamp_utc": _db_ts(q.source_timestamp_utc),
        "provider": q.provider, "venue": q.venue, "venue_kind": q.venue_kind,
        "provider_event_id": q.provider_event_id, "provider_market_id": q.provider_market_id,
        "market": q.market, "selection": q.selection,
        "line": None if q.line is None else float(q.line),
        "odds_american": None if q.odds_american is None else int(q.odds_american),
        "price": q.price, "bid": q.bid, "ask": q.ask, "size": q.size, "liquidity": q.liquidity,
        "event_id": q.event_id, "season": q.season, "week": q.week,
        "kickoff_utc": _db_ts(q.kickoff_utc), "home_team": q.home_team, "away_team": q.away_team,
        "raw_json": json.dumps(q.raw, default=_json_default, sort_keys=True),
        "quality_flags": json.dumps(flags),
        "routed_to_quotes": bool(routed_to_quotes),
    }


def _localize(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        d[col] = pd.to_datetime(d[col]).dt.tz_localize("UTC")
    return d


class SnapshotStore:
    """Append-only access to ``market_snapshots`` and ``collection_runs``."""

    def __init__(self, path: str = "data/nfl_market.duckdb"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with duckdb.connect(self.path) as c:
            c.execute(SNAPSHOT_DDL)
            c.execute(RUNS_DDL)

    def append_snapshots(self, rows: list[dict[str, Any]]) -> SnapshotAppendResult:
        """Insert rows; an existing snapshot_id is skipped and counted, never overwritten."""
        if not rows:
            return SnapshotAppendResult(0, 0, 0, 0)
        unique: dict[str, dict[str, Any]] = {}
        for r in rows:
            unique.setdefault(r["snapshot_id"], r)
        dup_batch = len(rows) - len(unique)
        params = [[r[c] for c in SNAPSHOT_COLUMNS] for r in unique.values()]
        placeholders = ",".join("?" * len(SNAPSHOT_COLUMNS))
        sql = (f"INSERT INTO market_snapshots ({','.join(SNAPSHOT_COLUMNS)}) VALUES ({placeholders}) "
               "ON CONFLICT (snapshot_id) DO NOTHING")
        with duckdb.connect(self.path) as c:
            c.execute("BEGIN TRANSACTION")
            before = c.execute("SELECT count(*) FROM market_snapshots").fetchone()[0]
            c.executemany(sql, params)
            after = c.execute("SELECT count(*) FROM market_snapshots").fetchone()[0]
            c.execute("COMMIT")
        inserted = after - before
        return SnapshotAppendResult(len(rows), inserted, dup_batch, len(unique) - inserted)

    def record_run(self, run: dict[str, Any]) -> bool:
        """Insert one finished run record. Returns False if that run_id already exists."""
        row = dict(run)
        row["started_at_utc"] = _db_ts(row["started_at_utc"])
        row["finished_at_utc"] = _db_ts(row.get("finished_at_utc"))
        row["errors"] = json.dumps(list(row.get("errors") or []))
        placeholders = ",".join("?" * len(RUN_COLUMNS))
        with duckdb.connect(self.path) as c:
            before = c.execute("SELECT count(*) FROM collection_runs").fetchone()[0]
            c.execute(f"INSERT INTO collection_runs ({','.join(RUN_COLUMNS)}) VALUES ({placeholders}) "
                      "ON CONFLICT (run_id) DO NOTHING", [row[k] for k in RUN_COLUMNS])
            after = c.execute("SELECT count(*) FROM collection_runs").fetchone()[0]
        return after > before

    def snapshots(self, run_id: str | None = None, provider_event_id: str | None = None) -> pd.DataFrame:
        """Stored snapshots with tz-aware UTC timestamps and quality_flags decoded to lists."""
        where, args = [], []
        if run_id:
            where.append("run_id = ?")
            args.append(run_id)
        if provider_event_id:
            where.append("provider_event_id = ?")
            args.append(provider_event_id)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        with duckdb.connect(self.path, read_only=True) as c:
            d = c.execute(f"SELECT * FROM market_snapshots{clause} ORDER BY captured_at_utc, snapshot_id", args).df()
        d = _localize(d, SNAPSHOT_TIMESTAMPS)
        d["quality_flags"] = d["quality_flags"].map(json.loads)
        return d

    def runs(self) -> pd.DataFrame:
        with duckdb.connect(self.path, read_only=True) as c:
            d = c.execute("SELECT * FROM collection_runs ORDER BY started_at_utc, run_id").df()
        d = _localize(d, RUN_TIMESTAMPS)
        d["errors"] = d["errors"].map(json.loads)
        return d

    def row_count(self) -> int:
        with duckdb.connect(self.path, read_only=True) as c:
            return c.execute("SELECT count(*) FROM market_snapshots").fetchone()[0]

    @staticmethod
    def raw_payload(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
        return json.loads(row["raw_json"])
