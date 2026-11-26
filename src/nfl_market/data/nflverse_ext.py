"""nflverse play-by-play and injury reports, reduced to small panels for EPA and QB-status features.

Sources (CC-BY 4.0, nflverse; attribution in ``docs/data_sources/nflverse.md``):

* play-by-play ``play_by_play_<season>.parquet`` (1999 onward)
* injuries ``injuries_<season>.parquet`` (2009 onward)

Holdout seasons (2025) are never downloaded. Raw play-by-play is reduced on download and deleted;
only the panels below are kept, under ``data/raw/nflverse_ext/`` (gitignored), with a sha256 manifest.

Panels
------
``team_game_epa.parquet``  one row per (game_id, team): offensive and defensive plays, EPA sums, success
                           counts, pass/rush splits. Plays = pass or rush with non-missing EPA.
``qb_game.parquet``        one row per (game_id, team, player_id): QB dropbacks and QB EPA.
``injuries.parquet``       nflverse injury rows (final weekly report), ``date_modified`` as UTC.

Leakage notes
-------------
* Play-by-play also carries ``vegas_wp``, ``vegas_home_wp``, ``spread_line`` and ``total_line`` (the nflverse
  line, capture time unknown). Those columns are never read here.
* Panels describe completed games; features use them only through ``label_available_at_utc < T``
  (``nfl_market.features.external``).
* Injury ``date_modified`` is missing in 2009 and from 2025 on (the file is overwritten in place). For the
  live season the file is fetched at pricing time, so its contents are known at that time. From 2026-11 the
  ``capture-injuries`` workflow snapshots the file every 3 hours (``nfl_market.data.injury_snapshots``); rows it
  saw get their first capture time as ``date_modified``. Other rows without a timestamp are treated as available
  (flagged in the feature card); ``report_time_source`` says which applies.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from nfl_market.holdout import REPO_ROOT, holdout_seasons

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
INJURIES_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet"
EXT_DIR = REPO_ROOT / "data" / "raw" / "nflverse_ext"
FIRST_PBP_SEASON = 1999
FIRST_INJURY_SEASON = 2009
PBP_COLUMNS = ["game_id", "season", "posteam", "defteam", "pass", "rush", "epa", "success", "qb_dropback",
               "qb_epa", "id"]
TEAM_PANEL, QB_PANEL, INJURY_PANEL = "team_game_epa.parquet", "qb_game.parquet", "injuries.parquet"


def _seasons(first: int, last: int) -> list[int]:
    banned = holdout_seasons()
    return [s for s in range(first, last + 1) if s not in banned]


def _fetch(url: str, path: Path, timeout: float = 300.0) -> str:
    import httpx

    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reduce_pbp(pbp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce one season of play-by-play to the team-game EPA panel and the QB-game panel."""
    p = pbp[pbp["posteam"].notna() & pbp["epa"].notna() & ((pbp["pass"] == 1) | (pbp["rush"] == 1))].copy()
    p["is_pass"] = (p["pass"] == 1).astype(float)
    p["is_rush"] = 1.0 - p["is_pass"]
    p["pass_epa"] = p["epa"] * p["is_pass"]
    p["rush_epa"] = p["epa"] * p["is_rush"]
    agg = dict(plays=("epa", "size"), epa=("epa", "sum"), succ=("success", "sum"), pass_plays=("is_pass", "sum"),
               pass_epa=("pass_epa", "sum"), rush_plays=("is_rush", "sum"), rush_epa=("rush_epa", "sum"))
    off = p.groupby(["game_id", "posteam"]).agg(**agg).reset_index().rename(columns={"posteam": "team"})
    dfn = p.groupby(["game_id", "defteam"]).agg(**agg).reset_index().rename(columns={"defteam": "team"})
    team = off.merge(dfn, on=["game_id", "team"], suffixes=("_off", "_def"), how="outer")
    team["season"] = p["season"].astype(int).to_numpy()[0] if len(p) else pd.NA
    db = p[(p["qb_dropback"] == 1) & p["id"].notna()]
    qb = (db.groupby(["game_id", "posteam", "id"]).agg(dropbacks=("qb_epa", "size"), qb_epa=("qb_epa", "sum"))
          .reset_index().rename(columns={"posteam": "team", "id": "player_id"}))
    qb["season"] = team["season"].iloc[0] if len(team) else pd.NA
    return team, qb


def download(last_season: int, out_dir: str | Path = EXT_DIR, refresh_seasons: tuple[int, ...] = (),
             snapshot_store: str | Path | None = None) -> Path:
    """Build or update the panels for seasons ``FIRST_*..last_season`` (holdout seasons skipped).

    Seasons already in the panels are kept unless listed in ``refresh_seasons`` (use this for the live season).
    Injury rows without ``date_modified`` get the first capture time from the injury snapshots
    (``nfl_market.data.injury_snapshots``; store from ``snapshot_store`` or ``$NFLMI_INJURY_SNAPSHOTS``) when the
    store exists; ``report_time_source`` records which time each row carries.
    """
    import httpx

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    old_team = pd.read_parquet(out / TEAM_PANEL) if (out / TEAM_PANEL).exists() else pd.DataFrame(columns=["season"])
    old_qb = pd.read_parquet(out / QB_PANEL) if (out / QB_PANEL).exists() else pd.DataFrame(columns=["season"])
    old_inj = pd.read_parquet(out / INJURY_PANEL) if (out / INJURY_PANEL).exists() else pd.DataFrame(columns=["season"])
    refresh = set(refresh_seasons)
    teams, qbs, injs = [old_team[~old_team.season.isin(refresh)]], [old_qb[~old_qb.season.isin(refresh)]], \
        [old_inj[~old_inj.season.isin(refresh)]]
    have_pbp, have_inj = set(teams[0].season.astype(int)), set(injs[0].season.astype(int))
    for s in _seasons(FIRST_PBP_SEASON, last_season):
        if s in have_pbp:
            continue
        raw = out / f"pbp_{s}.parquet"
        try:
            manifest[f"pbp_{s}"] = _fetch(PBP_URL.format(season=s), raw)
        except httpx.HTTPError as exc:  # the live season's file may not exist yet in the preseason
            print(f"pbp {s}: {exc}")
            continue
        t, q = reduce_pbp(pd.read_parquet(raw, columns=PBP_COLUMNS))
        raw.unlink()
        teams.append(t)
        qbs.append(q)
    for s in _seasons(FIRST_INJURY_SEASON, last_season):
        if s in have_inj:
            continue
        raw = out / f"injuries_{s}.parquet"
        try:
            manifest[f"injuries_{s}"] = _fetch(INJURIES_URL.format(season=s), raw)
        except httpx.HTTPError as exc:
            print(f"injuries {s}: {exc}")
            continue
        inj = pd.read_parquet(raw)
        raw.unlink()
        if "date_modified" not in inj.columns:
            inj["date_modified"] = pd.NaT
        inj["date_modified"] = pd.to_datetime(inj["date_modified"], utc=True, errors="coerce")
        injs.append(inj)
    banned = holdout_seasons()
    for name, frames in ((TEAM_PANEL, teams), (QB_PANEL, qbs), (INJURY_PANEL, injs)):
        df = pd.concat([f for f in frames if len(f)], ignore_index=True)
        if name == INJURY_PANEL:
            from nfl_market.data.injury_snapshots import fill_report_times, first_seen, store_dir

            store = store_dir(snapshot_store)
            seen = first_seen(store) if (store / "manifest.jsonl").exists() else None
            df = fill_report_times(df, seen if seen is not None else pd.DataFrame())
        if df["season"].astype(int).isin(banned).any():
            raise AssertionError(f"holdout season in {name}")
        df.to_parquet(out / name, index=False)
    man_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return out


def load_panels(out_dir: str | Path = EXT_DIR) -> dict[str, pd.DataFrame]:
    """Load the three panels; raises FileNotFoundError with the fix if they are missing."""
    out = Path(out_dir)
    missing = [n for n in (TEAM_PANEL, QB_PANEL, INJURY_PANEL) if not (out / n).exists()]
    if missing:
        raise FileNotFoundError(f"{missing} missing in {out}; run `python -m nfl_market.data.nflverse_ext`")
    panels = {"team": pd.read_parquet(out / TEAM_PANEL), "qb": pd.read_parquet(out / QB_PANEL),
              "injuries": pd.read_parquet(out / INJURY_PANEL)}
    banned = holdout_seasons()
    for k, df in panels.items():
        if df["season"].astype(int).isin(banned).any():
            raise AssertionError(f"holdout season in nflverse {k} panel")
    return panels


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download nflverse play-by-play and injury panels")
    ap.add_argument("--last-season", type=int, default=pd.Timestamp.now(tz="UTC").year
                    - (1 if pd.Timestamp.now(tz="UTC").month < 8 else 0))
    ap.add_argument("--refresh", type=int, nargs="*", default=[], help="seasons to re-download (live season)")
    a = ap.parse_args()
    print(download(a.last_season, refresh_seasons=tuple(a.refresh)))
