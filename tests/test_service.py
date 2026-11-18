from datetime import datetime, timedelta, timezone

import pandas as pd

from nfl_market.services.market_service import MarketService
from test_store import quote

NOW = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)


def pair(prefix, synthetic=False, **kw):
    return [quote(f"{prefix}h", synthetic=synthetic, **kw), quote(f"{prefix}a", selection="away", line=2.5, synthetic=synthetic, **kw)]


def test_empty_store_is_not_healthy(tmp_path):
    h = MarketService(str(tmp_path / "db.duckdb")).health(now=NOW)
    assert h["status"] == "no_data" and h["quote_rows"] == 0


def test_synthetic_only_is_degraded(tmp_path):
    s = MarketService(str(tmp_path / "db.duckdb"))
    s.store.append_quotes(pd.DataFrame(pair("x", synthetic=True)))
    h = s.health(now=NOW)
    assert h["status"] == "degraded" and h["synthetic_only"]


def test_real_upcoming_data_is_ok(tmp_path):
    s = MarketService(str(tmp_path / "db.duckdb"))
    s.store.append_quotes(pd.DataFrame(pair("x")))
    h = s.health(now=NOW)
    assert h["status"] == "ok", h["warnings"]
    assert h["upcoming_events"] == 1 and h["latest_capture_utc"].startswith("2026-09-01T12:00")


def test_no_upcoming_events_is_degraded(tmp_path):
    s = MarketService(str(tmp_path / "db.duckdb"))
    s.store.append_quotes(pd.DataFrame(pair("x")))
    assert s.health(now=NOW + timedelta(days=5))["status"] == "degraded"


def test_unreadable_database_is_error(tmp_path):
    s = MarketService(str(tmp_path / "db.duckdb"))
    (tmp_path / "db.duckdb").write_bytes(b"not a database")
    assert s.health(now=NOW)["status"] == "error"
