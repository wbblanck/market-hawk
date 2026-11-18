import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from nfl_market.collectors import SnapshotCollector
from nfl_market.data.snapshot_store import SnapshotStore
from nfl_market.data.store import MarketStore
from nfl_market.integrations.base import (
    VENUE_PREDICTION_MARKET,
    VENUE_SPORTSBOOK,
    ProviderQuote,
    ProviderUnavailable,
)

ROOT = Path(__file__).resolve().parents[1]
CAP = datetime(2026, 11, 22, 15, tzinfo=UTC)
KO = datetime(2026, 11, 22, 18, tzinfo=UTC)
SOURCE_TS = datetime(2026, 11, 22, 9, tzinfo=UTC)  # deliberately different from capture time


class FakeProvider:
    def __init__(self, quotes, name="fake", available=True, error=None):
        self.name, self._quotes, self._available, self._error = name, quotes, available, error
        self.calls = []

    def available(self):
        return self._available

    def fetch_quotes(self, *, date=None, event_ids=None):
        self.calls.append((date, event_ids))
        if self._error:
            raise self._error
        return list(self._quotes)


def clock_at(t=CAP):
    return lambda: t


def q(market="spread", selection="home", line=-3.5, odds=-110, book="BookA", event="E1", **kw):
    base = {"provider": "fake", "venue": book, "venue_kind": VENUE_SPORTSBOOK, "provider_event_id": f"p-{event}",
                "market": market, "selection": selection, "line": line, "odds_american": odds, "event_id": event,
                "season": 2026, "week": 12, "kickoff_utc": KO, "home_team": "BUF", "away_team": "NYJ",
                "provider_market_id": f"{book}-{event}-{market}-{selection}", "source_timestamp_utc": SOURCE_TS,
                "raw": {"book": book, "sel": selection, "line": line}}
    base.update(kw)
    return ProviderQuote(**base)


def full_book(book="BookA", event="E1", spread=-3.5, total=44.5):
    return [q("spread", "home", spread, book=book, event=event), q("spread", "away", -spread, book=book, event=event),
            q("total", "over", total, book=book, event=event), q("total", "under", total, book=book, event=event)]


def collect(tmp_path, providers, clock=None, **kw):
    db = str(tmp_path / "db.duckdb")
    report = SnapshotCollector(db, providers, clock=clock or clock_at()).collect(**kw)
    return db, report


def test_capture_time_from_clock_not_source(tmp_path):
    db, rep = collect(tmp_path, [FakeProvider(full_book())])
    snaps = SnapshotStore(db).snapshots()
    assert (snaps.captured_at_utc == pd.Timestamp(CAP)).all()
    assert (snaps.source_timestamp_utc == pd.Timestamp(SOURCE_TS)).all()
    assert rep.runs[0].captured_at_utc == CAP
    quotes = MarketStore(db).quotes()
    assert (quotes.captured_at_utc == pd.Timestamp(CAP)).all()


def test_one_capture_stamp_per_batch_taken_after_fetch(tmp_path):
    ticks = iter(CAP + timedelta(seconds=i) for i in range(100))
    db, rep = collect(tmp_path, [FakeProvider(full_book())], clock=lambda: next(ticks))
    run = rep.runs[0]
    snaps = SnapshotStore(db).snapshots()
    assert snaps.captured_at_utc.nunique() == 1
    # started_at (tick 0) < call start (tick 1) < capture (tick 2)
    assert snaps.captured_at_utc.iloc[0] == pd.Timestamp(CAP + timedelta(seconds=2))
    assert run.started_at_utc < run.captured_at_utc < run.finished_at_utc


def test_source_timestamp_none_stays_none_and_raw_roundtrips(tmp_path):
    raw = {"id": 7, "odds": {"am": "-110"}, "tags": ["a", "b"]}
    db, _ = collect(tmp_path, [FakeProvider([q(source_timestamp_utc=None, raw=raw)])])
    row = SnapshotStore(db).snapshots().iloc[0]
    assert pd.isna(row.source_timestamp_utc)
    assert SnapshotStore.raw_payload(row) == raw


def test_duplicate_rerun_is_idempotent(tmp_path):
    db = str(tmp_path / "db.duckdb")
    c = SnapshotCollector(db, [FakeProvider(full_book())], clock=clock_at())
    first = c.collect().runs[0]
    second = c.collect().runs[0]
    assert (first.n_stored, first.n_routed_inserted) == (4, 4)
    assert (second.n_stored, second.n_already_stored, second.n_routed_inserted) == (0, 4, 0)
    assert SnapshotStore(db).row_count() == 4 and MarketStore(db).row_count() == 4
    assert len(SnapshotStore(db).runs()) == 2


def test_valid_spread_and_total_pairs_are_routed(tmp_path):
    db, rep = collect(tmp_path, [FakeProvider(full_book("BookA") + full_book("BookB", spread=-3.0, total=45.0))])
    run = rep.runs[0]
    assert (run.status, run.n_candidates, run.n_routed, run.n_rejected) == ("ok", 8, 8, 0)
    quotes = MarketStore(db).quotes("E1")
    assert len(quotes) == 8
    r = quotes[(quotes.book == "BookA") & (quotes.selection == "home")].iloc[0]
    assert r.source == "fake" and r.source_quote_id == "BookA-E1-spread-home"
    assert not r.synthetic and not r.is_live
    assert SnapshotStore(db).snapshots().routed_to_quotes.all()


def test_mismatched_spread_pair_not_routed_but_stored(tmp_path):
    bad = [q("spread", "home", -3.5, book="BadBook"), q("spread", "away", 3.0, book="BadBook")]
    db, rep = collect(tmp_path, [FakeProvider(full_book("BookA") + bad)])
    run = rep.runs[0]
    assert run.status == "partial" and run.n_rejected == 2 and run.n_routed == 4
    assert any("BadBook" in e and "spread pairs not opposite" in e for e in run.errors)
    snaps = SnapshotStore(db).snapshots()
    assert len(snaps) == 6
    badrows = snaps[snaps.venue == "BadBook"]
    assert not badrows.routed_to_quotes.any()
    assert all("not_routed_validation_failed" in f for f in badrows.quality_flags)
    assert set(MarketStore(db).quotes().book) == {"BookA"}
    stored_run = SnapshotStore(db).runs().iloc[0]
    assert stored_run.status == "partial" and stored_run.n_rejected == 2
    assert any("spread pairs not opposite" in e for e in stored_run.errors)


def test_post_kickoff_quote_not_routed(tmp_path):
    db, rep = collect(tmp_path, [FakeProvider(full_book())], clock=clock_at(KO))
    run = rep.runs[0]
    assert run.n_post_kickoff == 4 and run.n_routed == 0
    snaps = SnapshotStore(db).snapshots()
    assert len(snaps) == 4 and not snaps.routed_to_quotes.any()
    assert all("captured_at_or_after_kickoff" in f for f in snaps.quality_flags)
    assert MarketStore(db).row_count() == 0


def test_prediction_market_and_moneyline_stored_not_routed(tmp_path):
    recs = [
        q("moneyline", "home", None, -150),
        q("winner", "home", None, None, book="Kalshi", venue_kind=VENUE_PREDICTION_MARKET, price=0.6, bid=0.59,
          ask=0.61, size=1200.0),
        q("spread", "home", -3.5, None, book="Kalshi", venue_kind=VENUE_PREDICTION_MARKET, price=0.5),
    ]
    db, rep = collect(tmp_path, [FakeProvider(recs)])
    assert rep.runs[0].n_candidates == 0 and rep.runs[0].status == "ok"
    snaps = SnapshotStore(db).snapshots()
    assert len(snaps) == 3 and not snaps.routed_to_quotes.any()
    assert MarketStore(db).row_count() == 0


def test_nulls_preserved(tmp_path):
    rec = q(odds=None, season=None, week=None, liquidity=None, provider_market_id=None, home_team=None)
    db, _ = collect(tmp_path, [FakeProvider([rec])])
    row = SnapshotStore(db).snapshots().iloc[0]
    for col in ["odds_american", "season", "week", "liquidity", "price", "bid", "ask", "size",
                "provider_market_id", "home_team"]:
        assert pd.isna(row[col]), col
    assert not row.routed_to_quotes and "not_routed_missing_fields" in row.quality_flags
    assert MarketStore(db).row_count() == 0


def test_unavailable_provider_skipped_and_recorded(tmp_path):
    off = FakeProvider(full_book(), name="off", available=False)
    db, rep = collect(tmp_path, [off, FakeProvider(full_book(), name="on")])
    assert off.calls == []
    assert rep.by_provider("off").status == "unavailable"
    assert rep.by_provider("on").n_routed == 4
    runs = SnapshotStore(db).runs().set_index("provider")
    assert runs.loc["off", "status"] == "unavailable" and runs.loc["off", "n_received"] == 0


def test_raising_provider_does_not_stop_others(tmp_path):
    boom = FakeProvider([], name="boom", error=RuntimeError("upstream 500"))
    gone = FakeProvider([], name="gone", error=ProviderUnavailable("server down"))
    good = FakeProvider(full_book(), name="good")
    db, rep = collect(tmp_path, [boom, gone, good], date="2026-11-22", event_ids=["p-E1"])
    assert rep.by_provider("boom").status == "failed"
    assert "upstream 500" in rep.by_provider("boom").errors[0]
    assert rep.by_provider("gone").status == "unavailable"
    assert rep.by_provider("good").status == "ok" and rep.by_provider("good").n_stored == 4
    assert good.calls == [("2026-11-22", ["p-E1"])]
    runs = SnapshotStore(db).runs().set_index("provider")
    assert runs.loc["boom", "status"] == "failed" and "upstream 500" in runs.loc["boom", "errors"][0]
    assert MarketStore(db).row_count() == 4


def test_script_help_runs():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "collect_snapshots.py"), "--help"],
                         capture_output=True, text=True, timeout=60, check=False)
    assert out.returncode == 0, out.stderr
    assert "--event-id" in out.stdout and "--providers" in out.stdout


def test_script_reports_unknown_provider(tmp_path):
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "collect_snapshots.py"), "--db",
                          str(tmp_path / "db.duckdb"), "--providers", "nope"], capture_output=True, text=True, timeout=60, check=False)
    assert out.returncode == 2 and "unknown provider" in out.stderr
