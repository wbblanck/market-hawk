import importlib.util
from datetime import UTC
from pathlib import Path

import pandas as pd

from nfl_market.data.validation import validate_quotes

spec = importlib.util.spec_from_file_location("load_demo_data", Path(__file__).resolve().parents[1] / "scripts" / "load_demo_data.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_demo_pairs_are_consistent():
    f = pd.DataFrame(demo.demo_rows())
    assert validate_quotes(f) == []
    key = ["event_id", "book", "captured_at_utc"]
    sp = f[f.market == "spread"].pivot_table(index=key, columns="selection", values="line")
    assert ((sp.home + sp.away).abs() < 1e-9).all()
    tot = f[f.market == "total"].pivot_table(index=key, columns="selection", values="line")
    assert ((tot.over - tot.under).abs() < 1e-9).all()
    assert f.synthetic.all()


def test_demo_prices_bets_are_synthetic_and_not_holdout(tmp_path):
    from datetime import datetime

    from nfl_market.holdout import is_holdout
    from nfl_market.markets.interface import PRICE_COLUMNS

    now = datetime(2026, 11, 19, 18, tzinfo=UTC)
    prices = demo.demo_prices(now)
    assert list(prices.columns) == list(PRICE_COLUMNS) and (prices.model_id == demo.DEMO_MODEL_ID).all()
    bets, grades = demo.demo_past_bets(now)
    assert bets.synthetic.all() and grades.synthetic.all()
    ids = pd.concat([prices.game_id, bets.game_id, pd.Series([r["event_id"] for r in demo.demo_rows(now)])])
    assert not is_holdout(ids.astype(str)).any() and ids.str.startswith("DEMO_").all()
    # each close tier appears once and every demo quote is captured before now and before kickoff - 15 min
    assert set(grades.dropna(subset=["line_close_tier"]).line_close_tier) == {"pinnacle", "sharp_median",
                                                                              "all_book_median"}
    f = pd.DataFrame(demo.demo_rows(now))
    assert (f.captured_at_utc <= now).all()
    assert (f.captured_at_utc < f.kickoff_utc - pd.Timedelta(minutes=15)).all()
