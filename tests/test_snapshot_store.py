from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest

from nfl_market.data.snapshot_store import SnapshotStore, snapshot_id_for, snapshot_row
from nfl_market.integrations.base import (
    VENUE_PREDICTION_MARKET,
    VENUE_SPORTSBOOK,
    ProviderQuote,
)

CAP = datetime(2026, 11, 22, 15, tzinfo=UTC)


def pq(**kw):
    base = {"provider": "fake", "venue": "BookA", "venue_kind": VENUE_SPORTSBOOK, "provider_event_id": "pe1",
                "market": "spread", "selection": "home", "line": -3.0, "odds_american": -110}
    base.update(kw)
    return ProviderQuote(**base)


def test_snapshot_id_is_deterministic_and_line_normalised():
    a = snapshot_id_for(pq(line=-3), CAP)
    assert a == snapshot_id_for(pq(line=-3.0), CAP)
    assert a != snapshot_id_for(pq(line=-3.5), CAP)
    assert a != snapshot_id_for(pq(), CAP + timedelta(seconds=1))
    assert a != snapshot_id_for(pq(provider_market_id="m2"), CAP)
    # same instant expressed in another zone -> same id
    assert a == snapshot_id_for(pq(), CAP.astimezone(timezone(timedelta(hours=-5))))


def test_append_only_roundtrip_nulls_and_tz(tmp_path):
    s = SnapshotStore(str(tmp_path / "db.duckdb"))
    q = pq(venue="Kal", venue_kind=VENUE_PREDICTION_MARKET, market="winner", line=None, odds_american=None,
           price=0.61, raw={"px": "0.61", "nested": {"a": [1, 2]}}, quality_flags=("no_source_timestamp",))
    row = snapshot_row(q, run_id="r1", captured_at_utc=CAP, routed_to_quotes=False)
    r = s.append_snapshots([row, row])
    assert (r.received, r.inserted, r.duplicate_in_batch, r.already_stored) == (2, 1, 1, 0)
    r2 = s.append_snapshots([row])
    assert (r2.inserted, r2.already_stored) == (0, 1)
    d = s.snapshots()
    assert len(d) == 1
    got = d.iloc[0]
    assert got.captured_at_utc == pd.Timestamp("2026-11-22T15:00Z")
    assert str(d.captured_at_utc.dt.tz) == "UTC"
    assert pd.isna(got.source_timestamp_utc) and pd.isna(got.kickoff_utc)
    for col in ["line", "odds_american", "bid", "ask", "size", "liquidity", "season", "week"]:
        assert pd.isna(got[col]), col
    assert got.price == pytest.approx(0.61)
    assert SnapshotStore.raw_payload(got) == {"px": "0.61", "nested": {"a": [1, 2]}}
    assert got.quality_flags == ["no_source_timestamp"]
    assert not hasattr(s, "delete") and not hasattr(s, "update")


def test_record_run_is_idempotent(tmp_path):
    s = SnapshotStore(str(tmp_path / "db.duckdb"))
    run = {"run_id": "r1", "started_at_utc": CAP, "finished_at_utc": CAP, "provider": "fake", "status": "ok",
               "n_received": 0, "n_stored": 0, "n_routed": 0, "n_rejected": 0, "errors": ["x"]}
    assert s.record_run(run) is True
    assert s.record_run(run) is False
    runs = s.runs()
    assert len(runs) == 1 and runs.errors.iloc[0] == ["x"]
    assert runs.started_at_utc.iloc[0] == pd.Timestamp("2026-11-22T15:00Z")
