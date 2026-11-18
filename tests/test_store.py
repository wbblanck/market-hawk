from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from nfl_market.data.store import MarketStore


def quote(sid="s", **kw):
    now = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
    base = dict(snapshot_id=sid, event_id="e", season=2026, week=1, kickoff_utc=now + timedelta(days=1), captured_at_utc=now,
                book="b", market="spread", selection="home", line=-2.5, odds_american=-110, is_live=False, source="test",
                source_quote_id=None, displayed_liquidity=None, synthetic=False)
    base.update(kw)
    return base


def test_append_and_read(tmp_path):
    s = MarketStore(str(tmp_path / "db.duckdb"))
    assert s.append_quotes(pd.DataFrame([quote()])) == 1
    q = s.quotes("e")
    assert len(q) == 1
    assert q.captured_at_utc.iloc[0] == pd.Timestamp("2026-09-01T12:00Z")


def test_optional_fields_may_be_missing(tmp_path):
    s = MarketStore(str(tmp_path / "db.duckdb"))
    required_only = {k: v for k, v in quote().items() if k in
                     {"snapshot_id", "event_id", "captured_at_utc", "book", "market", "selection", "line", "odds_american", "source"}}
    assert s.append_quotes(pd.DataFrame([required_only])) == 1
    q = s.quotes()
    assert pd.isna(q.kickoff_utc.iloc[0]) and pd.isna(q.season.iloc[0])


def test_inserted_count_excludes_duplicates(tmp_path):
    s = MarketStore(str(tmp_path / "db.duckdb"))
    assert s.append_quotes(pd.DataFrame([quote("a"), quote("b")])) == 2
    r = s.append_quotes_report(pd.DataFrame([quote("a"), quote("c"), quote("c")]))
    assert (r.received, r.inserted, r.duplicate_in_batch, r.already_stored) == (3, 1, 1, 1)
    assert s.row_count() == 3
    assert s.append_quotes(pd.DataFrame([quote("a")])) == 0


def test_non_utc_timestamps_converted(tmp_path):
    s = MarketStore(str(tmp_path / "db.duckdb"))
    eastern = pd.Timestamp("2026-09-01T08:00", tz="America/New_York")
    s.append_quotes(pd.DataFrame([quote(captured_at_utc=eastern)]))
    assert s.quotes().captured_at_utc.iloc[0] == pd.Timestamp("2026-09-01T12:00Z")


def test_bad_columns_rejected(tmp_path):
    s = MarketStore(str(tmp_path / "db.duckdb"))
    with pytest.raises(ValueError, match="Missing"):
        s.append_quotes(pd.DataFrame([{"snapshot_id": "x"}]))
    with pytest.raises(ValueError, match="Unknown"):
        s.append_quotes(pd.DataFrame([quote(extra=1)]))
