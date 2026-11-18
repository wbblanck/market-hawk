import importlib.util
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
