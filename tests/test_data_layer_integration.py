"""Provider -> collector -> store, end to end, using recorded Sports Leader fixtures (no network)."""
from datetime import UTC, datetime

from test_sports_leader_adapter import FakeTransport

from nfl_market.collectors.snapshot_collector import SnapshotCollector
from nfl_market.data.snapshot_store import SnapshotStore
from nfl_market.data.store import MarketStore
from nfl_market.integrations.sports_leader import SportsLeaderProvider

# Fixture game PHI @ JAX kicks off 2026-10-11 13:30 UTC; capture an hour before.
CAPTURE = datetime(2026, 10, 11, 12, 30, tzinfo=UTC)


def test_sports_leader_snapshot_routes_spreads_and_totals(tmp_path):
    db = tmp_path / "m.duckdb"
    provider = SportsLeaderProvider(FakeTransport(), season=2026, week=6)
    report = SnapshotCollector(db, [provider], clock=lambda: CAPTURE).collect(date="2026-10-11")
    run = report.by_provider("sports_leader")
    assert run.status in ("ok", "partial")

    raw = SnapshotStore(str(db)).snapshots()
    assert len(raw) > 0
    assert (raw.captured_at_utc == CAPTURE).all()  # capture time from the clock, never the source
    assert raw.source_timestamp_utc[raw.venue_kind == "sportsbook"].isna().all()

    quotes = MarketStore(str(db)).quotes()
    assert set(quotes.market) == {"spread", "total"}
    assert set(quotes.event_id) == {"2026_06_PHI_JAX"}
    spread = dict(zip(quotes[quotes.market == "spread"].selection, quotes[quotes.market == "spread"].line))
    assert spread == {"home": -6.5, "away": 6.5}
    assert set(quotes[quotes.market == "total"].line) == {42.5}
    # Moneyline and prediction-market contracts are kept raw only.
    assert not raw[raw.market.isin(["moneyline", "winner"])].routed_to_quotes.any()

    # Re-running the same capture is idempotent.
    SnapshotCollector(db, [SportsLeaderProvider(FakeTransport(), season=2026, week=6)],
                      clock=lambda: CAPTURE).collect(date="2026-10-11")
    assert len(SnapshotStore(str(db)).snapshots()) == len(raw)
    assert len(MarketStore(str(db)).quotes()) == len(quotes)
