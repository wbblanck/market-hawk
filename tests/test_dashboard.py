"""Dashboard smoke tests with Streamlit's AppTest (in-process, no browser).

One run on an empty tmp DuckDB, one with the hand-built FakeReader frames (synthetic row,
flagged row, weak-fallback CLV row, model health) injected via the service's default reader.
"""
from __future__ import annotations

from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "apps" / "dashboard" / "app.py"
TABS = ["Command Center", "Edges", "Market Explorer", "CLV", "Model Health", "Data Health", "Data & Resources", "MCP"]


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from nfl_market.config import settings
    db = str(tmp_path / "dash.duckdb")
    monkeypatch.setenv("NFLMI_DATABASE_PATH", db)
    monkeypatch.setattr(settings, "database_path", db)
    return db


def _run(timeout: float = 60):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == TABS
    return at


def test_dashboard_renders_on_empty_db(app_env, monkeypatch):
    from nfl_market.services import analytics_service as svc_mod
    monkeypatch.setattr(svc_mod, "default_reader", lambda *a, **k: svc_mod.EmptyMarketReader())
    at = _run()
    infos = " ".join(i.value for i in at.info)
    assert "No edges" in infos and "No paper bets" in infos and "No model health" in infos
    assert not any("SYNTHETIC DEMO DATA" in e.value for e in at.error)


def test_dashboard_renders_fake_reader_with_banners(app_env, monkeypatch):
    from test_analytics_service import FakeReader

    from nfl_market.services import analytics_service as svc_mod
    monkeypatch.setattr(svc_mod, "default_reader", lambda *a, **k: FakeReader())
    at = _run()
    assert any("SYNTHETIC DEMO DATA" in e.value for e in at.error)
    md = " ".join(m.value for m in at.markdown)
    assert "weak fallback" in md and "Stack weights" in md and "Calibration" in md
    frames = [d.value for d in at.dataframe]
    edges = next(f for f in frames if "ev_plus" in f.columns)
    # only the clean, real pinnacle row (ev 0.05) gets the badge; the flagged circa row (ev 0.125) does not
    assert list(edges.ev_plus) == ["", "EV+", ""]
    for col in ("captured_at_utc", "model_id", "flags"):
        assert col in edges.columns
    bets = next(f for f in frames if "line_clv" in f.columns and "bet_id" in f.columns)
    by_id = bets.set_index("bet_id")
    assert by_id.loc["b4", "line_clv"] == "pending" and by_id.loc["b5", "line_clv"] == "missing"
    assert by_id.loc["b3", "line_close_tier"] == "all_book_median (weak fallback)"
    assert "line_clv" in bets.columns and "price_clv" in bets.columns and "clv" not in bets.columns


def test_dashboard_renders_with_default_reader(app_env):
    """No injection: the real default reader (StoreMarketReader) over the empty tmp DuckDB."""
    _run()
