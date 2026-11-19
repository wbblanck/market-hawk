import json
import shutil

import pandas as pd
import pytest

from nfl_market import holdout
from nfl_market.holdout import (
    DEFAULT_ACCESS_LOG,
    DEFAULT_MANIFEST,
    exclude_holdout,
    holdout_ids,
    ids_sha256,
    load_manifest,
    read_holdout,
)


@pytest.fixture
def manifest_copy(tmp_path):
    dst = tmp_path / "manifest.json"
    shutil.copy(DEFAULT_MANIFEST, dst)
    return dst


@pytest.fixture
def games():
    return pd.DataFrame({
        "game_id": ["2024_01_BAL_KC", "2025_01_DAL_PHI", "2025_22_KC_PHI_X", "2026_01_DAL_PHI", "custom_id"],
        "season": [2024, 2025, 2025, 2026, 2025],
    })


@pytest.fixture(autouse=True)
def real_log_untouched():
    before = DEFAULT_ACCESS_LOG.read_bytes() if DEFAULT_ACCESS_LOG.exists() else None
    yield
    after = DEFAULT_ACCESS_LOG.read_bytes() if DEFAULT_ACCESS_LOG.exists() else None
    assert before == after, "tests must never write to the real holdout access log"


def test_committed_manifest_hash_matches():
    m = load_manifest()
    assert m["holdout_id"] == "nfl_games_2025_v1"
    assert m["seasons"] == [2025]
    assert ids_sha256(m["game_ids"]) == m["sha256"]
    assert len(m["game_ids"]) == m["n_games"]
    assert all(g.startswith("2025_") for g in m["game_ids"])


def test_copy_passes_hash_check(manifest_copy):
    assert holdout_ids(manifest_copy) == holdout_ids()


def test_tampered_manifest_fails(manifest_copy, tmp_path):
    m = json.loads(manifest_copy.read_text())
    m["game_ids"] = m["game_ids"][1:]  # drop one game without updating the hash
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_manifest(tampered)


def test_tampered_by_adding_id_fails(manifest_copy, tmp_path):
    m = json.loads(manifest_copy.read_text())
    m["game_ids"].append("2024_01_BAL_KC")
    tampered = tmp_path / "added.json"
    tampered.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        holdout_ids(tampered)


def test_exclude_holdout_by_id_and_season(games, manifest_copy):
    kept = exclude_holdout(games, manifest_copy)
    assert list(kept.game_id) == ["2024_01_BAL_KC", "2026_01_DAL_PHI"]


def test_exclude_by_id_without_season_column(games, manifest_copy):
    kept = exclude_holdout(games.drop(columns="season"), manifest_copy)
    # listed id and 2025-prefixed id dropped; a non-standard id without a season column survives
    assert list(kept.game_id) == ["2024_01_BAL_KC", "2026_01_DAL_PHI", "custom_id"]


def test_exclude_by_season_only(manifest_copy):
    g = pd.DataFrame({"game_id": ["abc", "def"], "season": [2025, 2024]})
    assert list(exclude_holdout(g, manifest_copy).game_id) == ["def"]


def test_exclude_by_listed_id(manifest_copy):
    gid = load_manifest(manifest_copy)["game_ids"][0]
    g = pd.DataFrame({"game_id": [gid, "2023_01_X_Y"]})
    assert list(exclude_holdout(g, manifest_copy).game_id) == ["2023_01_X_Y"]


def test_read_holdout_logs_to_tmp_log(games, manifest_copy, tmp_path, monkeypatch):
    monkeypatch.setattr(holdout, "_commit", lambda: "deadbeef")
    log = tmp_path / "logs" / "access.jsonl"
    out = read_holdout(games.drop(columns="season"), experiment_id="exp-001", purpose="final score",
                       model_card_id="mc-1", path=manifest_copy, access_log=log)
    assert set(out.game_id) == {"2025_01_DAL_PHI", "2025_22_KC_PHI_X"}
    read_holdout(games, experiment_id="exp-002", purpose="second", path=manifest_copy, access_log=log)
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert [e["experiment_id"] for e in lines] == ["exp-001", "exp-002"]  # append-only
    assert lines[0]["holdout_id"] == "nfl_games_2025_v1"
    assert lines[0]["model_card_id"] == "mc-1" and lines[0]["code_commit"] == "deadbeef"
    assert lines[0]["purpose"] == "final score" and lines[0]["accessed_at_utc"]


@pytest.mark.parametrize("experiment_id,purpose", [("", "p"), (None, "p"), ("exp", "")])
def test_read_holdout_refuses_without_id_or_purpose(games, manifest_copy, tmp_path, experiment_id, purpose):
    log = tmp_path / "access.jsonl"
    with pytest.raises(ValueError):
        read_holdout(games, experiment_id=experiment_id, purpose=purpose, path=manifest_copy, access_log=log)
    assert not log.exists()
