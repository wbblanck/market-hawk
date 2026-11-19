import numpy as np
import pandas as pd
import pytest

from nfl_market.features.pit import add_market_features, assert_point_in_time


def test_future_feature_rejected():
    x = pd.DataFrame({"p": pd.to_datetime(["2026-09-01"], utc=True), "a": pd.to_datetime(["2026-09-02"], utc=True)})
    with pytest.raises(ValueError, match="leakage"):
        assert_point_in_time(x, "p", "a")


def _q(book, t, odds, sid):
    return {"snapshot_id": sid, "event_id": "E1", "market": "h2h", "selection": "home", "line": np.nan,
            "book": book, "captured_at_utc": pd.Timestamp(t, tz="UTC"), "odds_american": odds}


def _quotes():
    return pd.DataFrame([
        _q("A", "2026-09-01 10:00", -110, "s1"),
        _q("B", "2026-09-01 11:00", -120, "s2"),
        _q("A", "2026-09-01 12:00", -130, "s3"),
    ])


def test_market_features_as_of_and_no_zero_fill():
    f = add_market_features(_quotes()).set_index("snapshot_id")
    # single book at 10:00: count 1, dispersion missing (not 0)
    assert f.loc["s1", "book_count_contract"] == 1
    assert np.isnan(f.loc["s1", "market_dispersion"])
    # 11:00: A(-110) and B(-120)
    pa, pb = 110 / 210, 120 / 220
    assert f.loc["s2", "book_count_contract"] == 2
    assert f.loc["s2", "market_dispersion"] == pytest.approx(pd.Series([pa, pb]).std())
    # 12:00: latest A(-130) replaces A(-110)
    assert f.loc["s3", "market_dispersion"] == pytest.approx(pd.Series([130 / 230, pb]).std())


def test_future_quote_does_not_change_earlier_rows():
    base = add_market_features(_quotes()).set_index("snapshot_id")
    more = pd.concat([_quotes(), pd.DataFrame([_q("C", "2026-09-01 13:00", +150, "s4"),
                                               _q("B", "2026-09-01 14:00", -200, "s5")])])
    pert = add_market_features(more).set_index("snapshot_id")
    cols = ["book_count_contract", "market_dispersion", "implied_prob"]
    pd.testing.assert_frame_equal(base[cols], pert.loc[base.index, cols], check_dtype=False)
    assert pert.loc["s4", "book_count_contract"] == 3


def test_max_quote_age_drops_stale_books():
    f = add_market_features(_quotes(), max_quote_age=pd.Timedelta(minutes=90)).set_index("snapshot_id")
    # at 12:00 only A(12:00) and B(11:00) are within 90 minutes; A(10:00) superseded anyway
    assert f.loc["s3", "book_count_contract"] == 2
    f2 = add_market_features(_quotes(), max_quote_age=pd.Timedelta(minutes=30)).set_index("snapshot_id")
    assert f2.loc["s3", "book_count_contract"] == 1
    assert np.isnan(f2.loc["s3", "market_dispersion"])
