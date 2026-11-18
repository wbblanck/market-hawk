"""Quote-snapshot validation. Returns human-readable error strings; empty list means valid."""
from __future__ import annotations

import pandas as pd

from ..markets.odds import is_valid_american

REQUIRED = ["snapshot_id", "event_id", "captured_at_utc", "kickoff_utc", "book", "market", "selection", "line", "odds_american", "source"]
VALID_SELECTIONS = {"spread": {"home", "away"}, "total": {"over", "under"}}
# Same book, same event, same market, same side, same moment, same line: one record only.
CONTENT_KEY = ["event_id", "book", "market", "selection", "line", "captured_at_utc"]


def _parse_utc(series: pd.Series, name: str, errors: list[str]) -> pd.Series | None:
    if series.isna().any():
        errors.append(f"{name}: {int(series.isna().sum())} missing timestamps")
    present = series.dropna()
    naive = [v for v in present if isinstance(v, pd.Timestamp) and v.tzinfo is None]
    if pd.api.types.is_datetime64_any_dtype(series) and getattr(series.dt, "tz", None) is None and len(present):
        errors.append(f"{name}: timestamps must be timezone-aware UTC")
    elif naive:
        errors.append(f"{name}: {len(naive)} timestamps are not timezone-aware")
    try:
        return pd.to_datetime(series, utc=True, errors="raise")
    except (ValueError, TypeError):
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
        errors.append(f"{name}: {int((parsed.isna() & series.notna()).sum())} unparseable timestamps")
        return parsed


def validate_quotes(df: pd.DataFrame, allow_live: bool = False) -> list[str]:
    if missing := [x for x in REQUIRED if x not in df]:
        return [f"missing columns: {missing}"]
    errors: list[str] = []
    if df.empty:
        return errors

    for col in ["snapshot_id", "event_id", "book", "market", "selection", "source"]:
        if (n := int(df[col].isna().sum())):
            errors.append(f"{col}: {n} missing values")

    # Duplicates
    if df.snapshot_id.duplicated().any():
        errors.append("duplicate snapshot_id")

    # Markets and selections
    market = df.market.astype(str).str.lower()
    selection = df.selection.astype(str).str.lower()
    if not set(market.dropna()).issubset(VALID_SELECTIONS):
        errors.append("unsupported market")
    bad_sel = [
        (m, s) for m, s in zip(market, selection) if m in VALID_SELECTIONS and s not in VALID_SELECTIONS[m]
    ]
    if bad_sel:
        errors.append(f"invalid selection for market: {sorted(set(bad_sel))}")

    # Prices
    if (df.odds_american == 0).any():
        errors.append("American odds cannot be zero")
    malformed = [o for o in df.odds_american if o != 0 and not is_valid_american(o)]
    if malformed:
        errors.append(f"malformed American odds: {len(malformed)} rows (e.g. {malformed[0]!r})")

    # Lines
    line = pd.to_numeric(df.line, errors="coerce")
    if line.isna().any():
        errors.append(f"line: {int(line.isna().sum())} missing or non-numeric")
    if ((market == "total") & (line <= 0)).any():
        errors.append("total lines must be positive")

    # Timestamps
    cap = _parse_utc(df.captured_at_utc, "captured_at_utc", errors)
    ko = _parse_utc(df.kickoff_utc, "kickoff_utc", errors)
    if cap is not None and ko is not None:
        post = (cap >= ko) & cap.notna() & ko.notna()
        if not allow_live:
            post = post & ~df.get("is_live", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        if post.any():
            errors.append(f"post-kickoff quotes: {int(post.sum())} rows captured at or after kickoff")
        if "is_live" in df and not allow_live and df.is_live.fillna(False).astype(bool).any():
            errors.append("live quotes are out of scope (pre-game only)")

    # Duplicate content under different snapshot ids
    keyed = df.assign(market=market, selection=selection, line=line, captured_at_utc=cap)
    dup_content = keyed.duplicated(CONTENT_KEY, keep=False) & ~keyed.snapshot_id.duplicated(keep=False)
    if dup_content.any():
        errors.append(f"duplicate quote records: {int(dup_content.sum())} rows share event/book/market/selection/line/time")

    errors.extend(_paired_quote_errors(keyed))
    return errors


def _paired_quote_errors(df: pd.DataFrame) -> list[str]:
    """Home/away spreads must be opposites; over/under must share a total (same book and moment)."""
    errors = []
    group = ["event_id", "book", "captured_at_utc"]
    # Only compare when each side has a single line at that moment (skip alternate-line ladders).
    def single(v):
        u = v.dropna().unique()
        return u[0] if len(u) == 1 else float("nan")
    spreads = df[df.market == "spread"].pivot_table(index=group, columns="selection", values="line", aggfunc=single)
    if {"home", "away"}.issubset(spreads.columns):
        both = spreads.dropna(subset=["home", "away"])
        if (n := int(((both.home + both.away).abs() > 1e-9).sum())):
            errors.append(f"spread pairs not opposite: {n} book snapshots where home line != -away line")
    totals = df[df.market == "total"].pivot_table(index=group, columns="selection", values="line", aggfunc=single)
    if {"over", "under"}.issubset(totals.columns):
        both = totals.dropna(subset=["over", "under"])
        if (n := int(((both.over - both.under).abs() > 1e-9).sum())):
            errors.append(f"total pairs mismatched: {n} book snapshots where over and under totals differ")
    return errors
