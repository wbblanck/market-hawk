"""Point-in-time feature helpers (feature_rules.md R3/R4, leakage_audit.md B/C)."""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTRACT_KEYS = ["event_id", "market", "selection", "line"]


def assert_point_in_time(df: pd.DataFrame, prediction_time: str, available_time: str):
    bad = df[available_time] > df[prediction_time]
    if bad.any():
        raise ValueError(f"Point-in-time leakage in {int(bad.sum())} rows")


def _implied(o) -> float:
    if pd.isna(o):
        return np.nan
    o = float(o)
    return 100.0 / (o + 100.0) if o > 0 else abs(o) / (abs(o) + 100.0)


def add_market_features(df: pd.DataFrame, max_quote_age: pd.Timedelta | None = None) -> pd.DataFrame:
    """Add as-of market features to quote snapshot rows.

    For each row (contract = event_id, market, selection, line; time t = its
    ``captured_at_utc``) only quotes of the same contract with
    ``captured_at_utc <= t`` are used (and, with ``max_quote_age``, captured at
    or after ``t - max_quote_age``). The latest such quote per book is taken, then:

    * ``book_count_contract``: number of books with a quote as of t;
    * ``market_dispersion``: sample SD (ddof=1) of those books' implied
      probabilities; NaN when fewer than 2 books (never zero-filled, R4);
    * ``implied_prob``: the row's own implied probability from ``odds_american``.

    ``source_timestamp_utc`` is never used. Rows are returned sorted by
    ``captured_at_utc`` (stable) with a fresh RangeIndex, all input columns kept.
    """
    if df.empty:
        return df.copy()
    x = df.sort_values("captured_at_utc", kind="stable").reset_index(drop=True)
    x["implied_prob"] = x["odds_american"].map(_implied).astype(float)
    count = np.full(len(x), np.nan)
    disp = np.full(len(x), np.nan)
    pos = pd.Series(np.arange(len(x)), index=x.index)
    keys = [k for k in CONTRACT_KEYS if k in x.columns]
    for _, grp in x.groupby(keys, dropna=False, sort=False):
        t = grp["captured_at_utc"]
        for when in t.unique():
            asof = grp[t <= when]
            if max_quote_age is not None:
                asof = asof[asof["captured_at_utc"] >= when - max_quote_age]
            latest = asof.groupby("book", dropna=False).tail(1)   # sorted by capture time: last = latest
            rows = pos[grp.index[t == when]].to_numpy()
            count[rows] = latest["book"].nunique(dropna=False)
            p = latest["implied_prob"].dropna()
            disp[rows] = p.std(ddof=1) if len(p) >= 2 else np.nan
    x["book_count_contract"] = count.astype(int) if not np.isnan(count).any() else count
    x["market_dispersion"] = disp
    return x
