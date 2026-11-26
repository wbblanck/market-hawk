"""Injury-report snapshots: dedupe, first-seen times, filling date_modified, holdout refusal."""
import pandas as pd
import pytest

from nfl_market.data import injury_snapshots as S


def _rep(rows):
    return pd.DataFrame(rows, columns=["season", "week", "team", "gsis_id", "report_status", "full_name"])


def test_snapshot_dedupes_and_first_seen(tmp_path):
    t1, t2, t3 = (pd.Timestamp(x, tz="UTC") for x in ("2026-11-24 15:00", "2026-11-25 15:00", "2026-11-26 15:00"))
    a = _rep([(2026, 13, "KC", "qb1", "Questionable", "A")])
    b = _rep([(2026, 13, "KC", "qb1", "Out", "A"), (2026, 13, "BUF", "wr1", "Out", "B")])
    e1 = S.snapshot(2026, tmp_path, now=t1, frame=a)
    assert e1["rows"] == 1
    assert S.snapshot(2026, tmp_path, now=t2, frame=a.iloc[::-1])["skipped"] == "unchanged"
    S.snapshot(2026, tmp_path, now=t3, frame=b)
    m = S.read_manifest(tmp_path)
    assert len(m) == 2 and (tmp_path / m.file.iloc[1]).exists()
    seen = S.first_seen(tmp_path).set_index(["team", "report_status"]).first_seen_utc
    assert seen[("KC", "Questionable")] == t1
    assert seen[("KC", "Out")] == t3          # the Out listing first appears in the third capture


def test_fill_report_times_keeps_feed_times(tmp_path):
    t = pd.Timestamp("2026-11-26 15:00", tz="UTC")
    S.snapshot(2026, tmp_path, now=t, frame=_rep([(2026, 13, "KC", "qb1", "Out", "A")]))
    inj = _rep([(2026, 13, "KC", "qb1", "Out", "A"), (2026, 13, "KC", "rb1", "Out", "C"),
                (2026, 13, "KC", "te1", "Doubtful", "D")])
    feed = pd.Timestamp("2026-11-20", tz="UTC")
    inj["date_modified"] = [pd.NaT, pd.NaT, feed]
    out = S.fill_report_times(inj, S.first_seen(tmp_path))
    assert out.date_modified.iloc[0] == t and out.report_time_source.iloc[0] == "snapshot_first_seen"
    assert pd.isna(out.date_modified.iloc[1]) and out.report_time_source.iloc[1] == "missing"
    assert out.date_modified.iloc[2] == feed and out.report_time_source.iloc[2] == "nflverse_date_modified"


def test_empty_store_and_holdout(tmp_path):
    inj = _rep([(2026, 13, "KC", "qb1", "Out", "A")]).assign(date_modified=pd.NaT)
    out = S.fill_report_times(inj, S.first_seen(tmp_path))
    assert out.report_time_source.tolist() == ["missing"]
    with pytest.raises(ValueError):
        S.snapshot(2025, tmp_path, frame=_rep([(2025, 1, "KC", "qb1", "Out", "A")]))
    with pytest.raises(ValueError):
        S.snapshot(2026, tmp_path, frame=_rep([(2024, 1, "KC", "qb1", "Out", "A")]))
