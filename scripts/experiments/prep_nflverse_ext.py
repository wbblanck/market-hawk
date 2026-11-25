"""Download nflverse play-by-play (1999-2024) and injuries (2009-2024) and reduce them to small panels.

Exploratory (recommendations backtest, 2026-11-25). Never fetches 2025 (locked holdout) or 2026.
Raw pbp files are deleted after reduction; outputs go to data/raw/nflverse_ext/ (gitignored).

    python scripts/experiments/prep_nflverse_ext.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "nflverse_ext"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{s}.parquet"
INJ_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{s}.parquet"
COLS = ["game_id", "season", "posteam", "defteam", "play_type", "pass", "rush", "epa", "success",
        "qb_dropback", "qb_epa", "id"]


def fetch(url: str, path: Path) -> str:
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reduce_pbp(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.read_parquet(path, columns=COLS)
    p = p[p.posteam.notna() & p.epa.notna() & ((p["pass"] == 1) | (p["rush"] == 1))].copy()
    p["is_pass"] = (p["pass"] == 1).astype(float)
    p["is_rush"] = 1.0 - p["is_pass"]
    p["pass_epa"] = p.epa * p.is_pass
    p["rush_epa"] = p.epa * p.is_rush
    agg = dict(plays=("epa", "size"), epa=("epa", "sum"), succ=("success", "sum"), pass_plays=("is_pass", "sum"),
               pass_epa=("pass_epa", "sum"), rush_plays=("is_rush", "sum"), rush_epa=("rush_epa", "sum"))
    off = p.groupby(["game_id", "posteam"]).agg(**agg).reset_index().rename(columns={"posteam": "team"})
    dfn = p.groupby(["game_id", "defteam"]).agg(**agg).reset_index().rename(columns={"defteam": "team"})
    team = off.merge(dfn, on=["game_id", "team"], suffixes=("_off", "_def"), how="outer")
    team["season"] = int(p.season.iloc[0])
    db = p[(p.qb_dropback == 1) & p["id"].notna()]
    qb = db.groupby(["game_id", "posteam", "id"]).agg(dropbacks=("qb_epa", "size"), qb_epa=("qb_epa", "sum")).reset_index()
    qb = qb.rename(columns={"posteam": "team", "id": "player_id"})
    qb["season"] = int(p.season.iloc[0])
    return team, qb


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, teams, qbs, injs = {}, [], [], []
    for s in range(1999, 2025):
        raw = OUT / f"pbp_{s}.parquet"
        manifest[f"pbp_{s}"] = fetch(PBP_URL.format(s=s), raw)
        t, q = reduce_pbp(raw)
        teams.append(t)
        qbs.append(q)
        raw.unlink()
        print("pbp", s, len(t), flush=True)
    for s in range(2009, 2025):
        raw = OUT / f"inj_{s}.parquet"
        manifest[f"injuries_{s}"] = fetch(INJ_URL.format(s=s), raw)
        injs.append(pd.read_parquet(raw))
        raw.unlink()
        print("inj", s, flush=True)
    pd.concat(teams, ignore_index=True).to_parquet(OUT / "team_game_epa.parquet", index=False)
    pd.concat(qbs, ignore_index=True).to_parquet(OUT / "qb_game.parquet", index=False)
    inj = pd.concat(injs, ignore_index=True)
    inj["date_modified"] = pd.to_datetime(inj["date_modified"], utc=True, errors="coerce")
    inj.to_parquet(OUT / "injuries.parquet", index=False)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
