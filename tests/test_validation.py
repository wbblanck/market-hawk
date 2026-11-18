from datetime import datetime, timedelta, timezone

import pandas as pd

from nfl_market.data.validation import validate_quotes

KO = datetime(2026, 9, 10, 0, 20, tzinfo=timezone.utc)


def row(**kw):
    base = dict(snapshot_id="s1", event_id="e", captured_at_utc=KO - timedelta(hours=1), kickoff_utc=KO, book="b",
                market="spread", selection="home", line=-2.5, odds_american=-110, source="t", is_live=False)
    base.update(kw)
    return base


def pair(**kw):
    return [row(**kw), row(snapshot_id="s2", selection="away", line=2.5, **{k: v for k, v in kw.items() if k not in {"selection", "line"}})]


def has(errors, text):
    return any(text in e for e in errors)


def test_clean_pair_passes():
    assert validate_quotes(pd.DataFrame(pair())) == []


def test_missing_columns():
    assert has(validate_quotes(pd.DataFrame([{"snapshot_id": "x"}])), "missing columns")


def test_duplicate_snapshot_id_and_duplicate_content():
    assert has(validate_quotes(pd.DataFrame([row(), row()])), "duplicate snapshot_id")
    assert has(validate_quotes(pd.DataFrame([row(), row(snapshot_id="s9")])), "duplicate quote records")


def test_invalid_selection_for_market():
    assert has(validate_quotes(pd.DataFrame([row(selection="over")])), "invalid selection")
    assert has(validate_quotes(pd.DataFrame([row(market="moneyline")])), "unsupported market")


def test_malformed_prices():
    assert has(validate_quotes(pd.DataFrame([row(odds_american=0)])), "cannot be zero")
    assert has(validate_quotes(pd.DataFrame([row(odds_american=50)])), "malformed American odds")
    assert has(validate_quotes(pd.DataFrame([row(odds_american=-110.5)])), "malformed American odds")


def test_post_kickoff_quote():
    assert has(validate_quotes(pd.DataFrame([row(captured_at_utc=KO + timedelta(minutes=1))])), "post-kickoff")
    assert has(validate_quotes(pd.DataFrame([row(captured_at_utc=KO)])), "post-kickoff")


def test_timestamp_problems():
    naive = KO.replace(tzinfo=None)
    assert has(validate_quotes(pd.DataFrame([row(kickoff_utc=naive, captured_at_utc=naive - timedelta(hours=1))])), "timezone-aware")
    assert has(validate_quotes(pd.DataFrame([row(captured_at_utc="not a time")])), "unparseable")
    assert has(validate_quotes(pd.DataFrame([row(captured_at_utc=None)])), "missing timestamps")


def test_paired_quote_consistency():
    bad_spread = [row(), row(snapshot_id="s2", selection="away", line=3.0)]
    assert has(validate_quotes(pd.DataFrame(bad_spread)), "not opposite")
    bad_total = [row(market="total", selection="over", line=44.5),
                 row(snapshot_id="s2", market="total", selection="under", line=45.0)]
    assert has(validate_quotes(pd.DataFrame(bad_total)), "totals differ")


def test_nonpositive_total():
    assert has(validate_quotes(pd.DataFrame([row(market="total", selection="over", line=0)])), "positive")
