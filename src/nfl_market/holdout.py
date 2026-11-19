"""Locked final holdout (validation_design.md Section 5).

The 2025 season (regular season and postseason) is the locked holdout. Its
game ids live in a committed manifest with a SHA-256 of the sorted id list.
Loaders call :func:`exclude_holdout` by default. Reading holdout rows requires
:func:`read_holdout` with an experiment id and purpose, which appends a line to
the append-only access log.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "config" / "holdout" / "nfl_games_2025_v1.json"
DEFAULT_ACCESS_LOG = REPO_ROOT / "config" / "holdout" / "access_log.jsonl"


def ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


@lru_cache(maxsize=8)
def _load(path: str) -> dict:
    m = json.loads(Path(path).read_text())
    if ids_sha256(m["game_ids"]) != m["sha256"]:
        raise ValueError(f"holdout manifest {path} hash mismatch: id list was edited")
    return m


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict:
    return _load(str(path))


def holdout_ids(path: str | Path = DEFAULT_MANIFEST) -> frozenset[str]:
    return frozenset(load_manifest(path)["game_ids"])


def holdout_seasons(path: str | Path = DEFAULT_MANIFEST) -> frozenset[int]:
    return frozenset(int(s) for s in load_manifest(path)["seasons"])


def is_holdout(game_ids: pd.Series, path: str | Path = DEFAULT_MANIFEST) -> pd.Series:
    """True for games in the holdout: listed ids, or any game in a holdout season."""
    ids = holdout_ids(path)
    mask = game_ids.isin(ids)
    seasons = holdout_seasons(path)
    season_of_id = pd.to_numeric(game_ids.astype(str).str[:4], errors="coerce")
    return mask | season_of_id.isin(seasons)


def exclude_holdout(games: pd.DataFrame, path: str | Path = DEFAULT_MANIFEST) -> pd.DataFrame:
    """Drop holdout games (by id and by season). Default for every loader."""
    mask = is_holdout(games["game_id"], path)
    if "season" in games.columns:
        mask = mask | games["season"].isin(holdout_seasons(path))
    return games.loc[~mask].copy()


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def read_holdout(games: pd.DataFrame, *, experiment_id: str, purpose: str,
                 model_card_id: str | None = None, path: str | Path = DEFAULT_MANIFEST,
                 access_log: str | Path = DEFAULT_ACCESS_LOG) -> pd.DataFrame:
    """Return only holdout games and log the access. Score a frozen model once."""
    if not experiment_id or not purpose:
        raise ValueError("holdout reads need an experiment_id and a purpose")
    entry = {
        "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_id": load_manifest(path)["holdout_id"],
        "experiment_id": experiment_id,
        "model_card_id": model_card_id,
        "code_commit": _commit(),
        "purpose": purpose,
    }
    log = Path(access_log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    mask = is_holdout(games["game_id"], path)
    return games.loc[mask].copy()
