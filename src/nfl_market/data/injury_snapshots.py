"""Timestamped snapshots of the nflverse injury-report file.

Why: from 2025 on, nflverse ``injuries_<season>.parquet`` has no ``date_modified`` and is overwritten in place,
so a past season cannot tell when a player was first listed Out. Without that time, the QB-out feature cannot be
backtested point in time for those seasons. Depth charts need no snapshot: nflverse ``depth_charts_<season>``
already keeps every update with its own ``dt`` timestamp.

Store layout (plaintext; nflverse data is CC-BY 4.0, attribution in ``docs/data_sources/nflverse.md``)::

    <store>/manifest.jsonl                       one line per stored snapshot
    <store>/<season>/injuries_<season>_<UTC>.parquet

A snapshot is written only when the file's content differs from the season's latest stored snapshot.
``captured_at_utc`` is our receipt time, never a feed timestamp. The workflow ``capture-injuries.yml`` runs this
at a few set times a week (Wed-Sat evening, Sun morning, Mon and Thu afternoon) and commits the store to ``injuries/`` on the ``market-data`` branch.

``first_seen`` gives, for each report row (season, week, team, player, report_status), the earliest capture that
contained it. ``fill_report_times`` uses it as ``date_modified`` where nflverse gives none, so a row counts as
known only from the first snapshot that showed it (conservative: the report was public no later than that).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import pandas as pd

from nfl_market.holdout import REPO_ROOT, holdout_seasons

INJURIES_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet"
DEFAULT_STORE = REPO_ROOT / "data" / "raw" / "injury_snapshots"
ROW_KEY = ["season", "week", "team", "gsis_id", "report_status"]
MANIFEST = "manifest.jsonl"


def store_dir(path: str | Path | None = None) -> Path:
    """Explicit path, else ``$NFLMI_INJURY_SNAPSHOTS``, else ``data/raw/injury_snapshots``."""
    return Path(path or os.environ.get("NFLMI_INJURY_SNAPSHOTS") or DEFAULT_STORE)


def _check_season(season: int) -> None:
    if int(season) in holdout_seasons():
        raise ValueError(f"season {season} is the locked holdout; injury snapshots refuse it")


def content_hash(df: pd.DataFrame) -> str:
    """Order-independent hash of the file's rows (row order in the feed is not meaningful)."""
    cols = sorted(df.columns)
    canon = df[cols].astype(str).sort_values(cols).reset_index(drop=True)
    return hashlib.sha256(canon.to_csv(index=False).encode()).hexdigest()


def read_manifest(store: str | Path | None = None) -> pd.DataFrame:
    p = store_dir(store) / MANIFEST
    if not p.exists():
        return pd.DataFrame(columns=["captured_at_utc", "season", "sha256", "rows", "file"])
    m = pd.DataFrame([json.loads(x) for x in p.read_text().splitlines() if x.strip()])
    m["captured_at_utc"] = pd.to_datetime(m["captured_at_utc"], utc=True)
    return m


def fetch(season: int, timeout: float = 120.0) -> pd.DataFrame:
    import httpx

    r = httpx.get(INJURIES_URL.format(season=season), follow_redirects=True, timeout=timeout)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def snapshot(season: int, store: str | Path | None = None, *, now: pd.Timestamp | None = None,
             frame: pd.DataFrame | None = None) -> dict:
    """Fetch (or take ``frame``) and store it if its content changed. Returns the manifest entry or a skip note."""
    _check_season(season)
    out = store_dir(store)
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    df = fetch(season) if frame is None else frame
    if len(df) and set(df["season"].astype(int).unique()) - {int(season)}:
        raise ValueError(f"injury file for {season} holds other seasons")
    h = content_hash(df)
    m = read_manifest(out)
    prev = m[m["season"].astype(int) == int(season)] if len(m) else m
    if len(prev) and prev.sort_values("captured_at_utc")["sha256"].iloc[-1] == h:
        return {"season": int(season), "skipped": "unchanged", "sha256": h}
    rel = f"{int(season)}/injuries_{int(season)}_{now.strftime('%Y%m%dT%H%M%SZ')}.parquet"
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / rel, index=False)
    entry = {"captured_at_utc": now.isoformat(), "season": int(season), "sha256": h, "rows": int(len(df)), "file": rel}
    with open(out / MANIFEST, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def first_seen(store: str | Path | None = None, seasons=None) -> pd.DataFrame:
    """Earliest capture time of each report row ``ROW_KEY`` across stored snapshots."""
    out = store_dir(store)
    m = read_manifest(out)
    if seasons is not None and len(m):
        m = m[m["season"].astype(int).isin([int(s) for s in seasons])]
    parts = []
    for t, rel in zip(m["captured_at_utc"], m["file"]):
        d = pd.read_parquet(out / rel)
        d = d[d["report_status"].notna()][ROW_KEY].drop_duplicates()
        parts.append(d.assign(first_seen_utc=t))
    if not parts:
        return pd.DataFrame(columns=ROW_KEY + ["first_seen_utc"])
    allrows = pd.concat(parts, ignore_index=True)
    allrows["season"] = allrows["season"].astype(int)
    allrows["week"] = allrows["week"].astype(int)
    return allrows.groupby(ROW_KEY, as_index=False, dropna=False)["first_seen_utc"].min()


def fill_report_times(inj: pd.DataFrame, seen: pd.DataFrame) -> pd.DataFrame:
    """Set ``date_modified`` from ``first_seen_utc`` where it is missing; flag the source in ``report_time_source``."""
    out = inj.copy()
    out["date_modified"] = pd.to_datetime(out.get("date_modified"), utc=True, errors="coerce")
    src = pd.Series("nflverse_date_modified", index=out.index).where(out["date_modified"].notna(), "missing")
    if len(seen):
        k = out[ROW_KEY].copy()
        k["season"], k["week"] = k["season"].astype(int), k["week"].astype(int)
        t = k.merge(seen, on=ROW_KEY, how="left")["first_seen_utc"].to_numpy()
        t = pd.to_datetime(pd.Series(t, index=out.index), utc=True)
        fill = out["date_modified"].isna() & t.notna()
        out.loc[fill, "date_modified"] = t[fill]
        src[fill] = "snapshot_first_seen"
    out["report_time_source"] = src
    return out


def current_season(now: pd.Timestamp | None = None) -> int:
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    return now.year - (1 if now.month < 8 else 0)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Snapshot the nflverse injury-report file if it changed")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--store", default=None)
    a = ap.parse_args()
    print(json.dumps(snapshot(a.season or current_season(), a.store), default=str))
