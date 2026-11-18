from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

DDL = '''
CREATE TABLE IF NOT EXISTS quote_snapshots (
 snapshot_id VARCHAR PRIMARY KEY,event_id VARCHAR,season INTEGER,week INTEGER,
 kickoff_utc TIMESTAMP,captured_at_utc TIMESTAMP,book VARCHAR,market VARCHAR,
 selection VARCHAR,line DOUBLE,odds_american INTEGER,is_live BOOLEAN,source VARCHAR,
 source_quote_id VARCHAR,displayed_liquidity DOUBLE,synthetic BOOLEAN
)'''

# Column order and pandas dtype used when normalising an incoming frame.
COLUMNS: dict[str, str] = {
    "snapshot_id": "string",
    "event_id": "string",
    "season": "Int64",
    "week": "Int64",
    "kickoff_utc": "timestamp",
    "captured_at_utc": "timestamp",
    "book": "string",
    "market": "string",
    "selection": "string",
    "line": "float64",
    "odds_american": "Int64",
    "is_live": "boolean",
    "source": "string",
    "source_quote_id": "string",
    "displayed_liquidity": "float64",
    "synthetic": "boolean",
}
REQUIRED = {"snapshot_id", "event_id", "captured_at_utc", "book", "market", "selection", "line", "odds_american", "source"}
OPTIONAL = set(COLUMNS) - REQUIRED
TIMESTAMP_COLUMNS = [c for c, t in COLUMNS.items() if t == "timestamp"]


@dataclass(frozen=True)
class AppendResult:
    received: int
    inserted: int
    duplicate_in_batch: int
    already_stored: int


def _to_utc_naive(series: pd.Series) -> pd.Series:
    """Store timestamps as naive UTC. Naive inputs are assumed to already be UTC."""
    s = pd.to_datetime(series, utc=True, errors="raise")
    return s.dt.tz_convert("UTC").dt.tz_localize(None)


def normalize_quotes(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with every store column, in order, with optional fields filled as null."""
    if missing := REQUIRED - set(frame.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if unknown := set(frame.columns) - set(COLUMNS):
        raise ValueError(f"Unknown columns: {sorted(unknown)}")
    x = frame.copy()
    for col in OPTIONAL - set(x.columns):
        x[col] = None
    for col, kind in COLUMNS.items():
        if kind == "timestamp":
            x[col] = _to_utc_naive(x[col])
        else:
            x[col] = x[col].astype(object).where(x[col].notna(), None).astype(kind)
    x["selection"] = x["selection"].str.lower()
    x["market"] = x["market"].str.lower()
    return x[list(COLUMNS)]


class MarketStore:
    def __init__(self, path="data/nfl_market.duckdb"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with duckdb.connect(path) as c:
            c.execute(DDL)

    def append_quotes_report(self, frame: pd.DataFrame) -> AppendResult:
        """Append quotes, skipping duplicates by snapshot_id. Reports what actually happened."""
        if frame.empty:
            return AppendResult(0, 0, 0, 0)
        x = normalize_quotes(frame)
        received = len(x)
        x = x.drop_duplicates("snapshot_id", keep="first")
        dup_batch = received - len(x)
        with duckdb.connect(self.path) as c:
            c.register("incoming", x)
            c.execute("BEGIN TRANSACTION")
            before = c.execute("SELECT count(*) FROM quote_snapshots").fetchone()[0]
            c.execute("INSERT INTO quote_snapshots BY NAME SELECT * FROM incoming ON CONFLICT(snapshot_id) DO NOTHING")
            after = c.execute("SELECT count(*) FROM quote_snapshots").fetchone()[0]
            c.execute("COMMIT")
        inserted = after - before
        return AppendResult(received, inserted, dup_batch, len(x) - inserted)

    def append_quotes(self, frame: pd.DataFrame) -> int:
        """Append quotes and return the number of rows actually inserted."""
        return self.append_quotes_report(frame).inserted

    def quotes(self, event_id=None) -> pd.DataFrame:
        with duckdb.connect(self.path, read_only=True) as c:
            if event_id:
                d = c.execute("SELECT * FROM quote_snapshots WHERE event_id=? ORDER BY captured_at_utc", [event_id]).df()
            else:
                d = c.execute("SELECT * FROM quote_snapshots ORDER BY captured_at_utc").df()
        for col in TIMESTAMP_COLUMNS:
            d[col] = pd.to_datetime(d[col]).dt.tz_localize("UTC")
        return d

    def row_count(self) -> int:
        with duckdb.connect(self.path, read_only=True) as c:
            return c.execute("SELECT count(*) FROM quote_snapshots").fetchone()[0]
