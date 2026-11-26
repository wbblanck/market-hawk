"""NFL_ATS spread-capture importer: price conversion, routing, idempotence, holdout refusal."""
import importlib.util
from pathlib import Path

import duckdb
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("imp", Path(__file__).parents[1] / "scripts" / "import_nfl_ats_lines.py")
imp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imp)

COLS = ("capture_id,captured_at,game_id,season,week,home_team,away_team,kickoff_utc,lock_utc,minutes_to_lock,book,"
        "home_spread,home_price_decimal,book_last_update,market_last_update,provider_event_id,status,reason")


def _write(tmp_path, rows):
    d = tmp_path / "data" / "lines"
    d.mkdir(parents=True)
    (d / "market_captures.csv").write_text(COLS + "\n" + "\n".join(rows) + "\n")
    return tmp_path


def test_decimal_to_american():
    assert imp.decimal_to_american(1.909) == -110
    assert imp.decimal_to_american(2.5) == 150
    with pytest.raises(ValueError):
        imp.decimal_to_american(1.0)


def test_import_routes_home_side_and_is_idempotent(tmp_path):
    base = "2026-11-22T17:55:00+00:00,2026_12_KC_BUF,2026,12,BUF,KC,2026-11-22T18:20:00+00:00,2026-11-22T18:05:00+00:00,10,"
    root = _write(tmp_path, [
        "c1," + base + "pinnacle,-2.5,1.952,2026-11-22T17:54:00Z,,e1,ok,",
        "c1," + base + "fanduel,-3.0,1.909,,,e1,ok,",
        "c1," + base + "bovada,,1.9,,,e1,fetch_error,timeout",
        "c2,2025-11-22T17:55:00+00:00,2025_12_KC_BUF,2025,12,BUF,KC,2025-11-22T18:20:00+00:00,x,10,pinnacle,-2.5,1.95,,,e2,ok,",
    ])
    db = str(tmp_path / "t.duckdb")
    imp.main(["--nfl-ats", str(root), "--db", db])
    imp.main(["--nfl-ats", str(root), "--db", db])
    con = duckdb.connect(db, read_only=True)
    q = con.execute("select book, line, odds_american, selection, captured_at_utc, source from quote_snapshots order by book").df()
    assert q.book.tolist() == ["fanduel", "pinnacle"] and q.selection.unique().tolist() == ["home"]
    assert q.odds_american.tolist() == [-110, -105]
    assert pd.Timestamp(q.captured_at_utc.iloc[0]) == pd.Timestamp("2026-11-22 17:55:00")
    assert set(q.source) == {"nfl_ats_import"}
    assert con.execute("select count(*) from market_snapshots").fetchone()[0] == 2   # 2025 row refused
