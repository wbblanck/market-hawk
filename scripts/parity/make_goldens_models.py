"""Generate parity goldens for the step-3 model ports from the ORIGINAL code.

Imports the owner's read-only clone (never edits it):
  * nfl-betting-model: nflbet.ats.elo (run, scoring_level, means, nb_pmf, margin_dist, probs),
                       nflbet.ats.kalman (run), nflbet.ats.xgb (PARAMS, FEATURES, model)

Inputs are deterministic: the committed fixture tests/fixtures/nflverse/games_sample.csv
(seasons 2015-2018; 2025 rows dropped) and a hand-built 4-team schedule (HAND below)
that contains a Monday game followed by a Wednesday game, the case where our
label-timing rule differs from the source. Frozen constants are read from the
clone's data/backtests/{elo_constants,batch1_constants}.json.

Writes tests/fixtures/parity/models_goldens.json. tests/test_parity_models.py
checks our ports against it without importing the originals.

    pip install polars   # the originals import it
    python scripts/parity/make_goldens_models.py [--nfl-betting-model /home/user/wbblanck/nfl-betting-model]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures" / "nflverse" / "games_sample.csv"
OUT = REPO / "tests" / "fixtures" / "parity" / "models_goldens.json"
OUTPUT_SEASONS = (2017, 2018)       # per-game outputs stored for these seasons (ratings run from 2015)
XGB_TRAIN, XGB_TEST = (2016, 2017), (2018,)
ROUND = 6                           # XGB inputs are rounded before the source sees them

#: hand schedule: (game_id, season, week, game_type, gameday, gametime ET, away, home, away_score, home_score, location)
HAND = [
    ("2020_01_A_B", 2020, 1, "REG", "2020-09-13", "13:00", "A", "B", 17, 24, "Home"),
    ("2020_01_C_D", 2020, 1, "REG", "2020-09-14", "20:15", "C", "D", 30, 10, "Home"),   # Monday
    ("2020_02_A_D", 2020, 2, "REG", "2020-09-16", "20:00", "A", "D", 21, 20, "Home"),   # Wednesday, next week
    ("2020_02_C_B", 2020, 2, "REG", "2020-09-20", "13:00", "C", "B", 14, 14, "Neutral"),
    ("2020_03_B_A", 2020, 3, "REG", "2020-09-27", "13:00", "B", "A", 3, 31, "Home"),
    ("2020_03_D_C", 2020, 3, "REG", "2020-09-27", "16:25", "D", "C", 27, 24, "Home"),
    ("2021_01_A_C", 2021, 1, "REG", "2021-09-12", "13:00", "A", "C", 20, 13, "Home"),
    ("2021_01_D_B", 2021, 1, "REG", "2021-09-12", "13:00", "D", "B", 10, 13, "Home"),
]
HAND_COLS = ["game_id", "season", "week", "game_type", "gameday", "gametime", "away_team", "home_team",
             "away_score", "home_score", "location"]


def commit(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def r(x, nd=15):
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(f"{v:.{nd}g}") for v in x]


def schedule(df: pd.DataFrame):
    import polars as pl

    d = df.copy()
    d["result"] = d["home_score"] - d["away_score"]
    return pl.from_pandas(d)


def run_source(sched, ek: dict, kp: dict, E, K):
    import polars as pl

    games = E.prep(sched)
    er = E.run(games, ek["k"], ek["h"], ek["c"]).join(E.scoring_level(games), on="game_id")
    mh, ma = E.means(er["d"], er["level"], ek["b"])
    er = er.with_columns(pl.Series("mh", mh), pl.Series("ma", ma))
    kf, _, (M, V) = K.run(games, kp)
    out = er.join(kf, on="game_id").sort("season", "week", "game_id")
    ph, pa = E.nb_pmf(out["mh"].to_numpy(), ek["theta"]), E.nb_pmf(out["ma"].to_numpy(), ek["theta"])
    marg, pm = E.margin_dist(ph, pa)
    win, _ = E.probs(ph, pa, np.zeros(len(out)))
    mmean = (pm * marg).sum(1)
    msd = np.sqrt((pm * marg ** 2).sum(1) - mmean ** 2)
    pt = np.array([np.convolve(ph[i], pa[i]) for i in range(len(out))])
    tt = np.arange(pt.shape[1])
    tmean = (pt * tt).sum(1)
    tsd = np.sqrt((pt * tt ** 2).sum(1) - tmean ** 2)
    final = E.final_ratings(games, ek["k"], ek["h"], ek["c"])
    return out, {"nb_margin_mean": mmean, "nb_margin_sd": msd, "nb_win": win, "nb_total_mean": tmean,
                 "nb_total_sd": tsd}, final, (M, V), games


def per_game(out, nb, seasons=None) -> dict:
    keep = np.ones(len(out), bool) if seasons is None else out["season"].is_in(list(seasons)).to_numpy()
    res = {"game_id": [g for g, k in zip(out["game_id"].to_list(), keep) if k]}
    for c in ("r_home", "r_away", "d", "p", "level", "mh", "ma", "kalman", "kalman_var"):
        res[c] = r(out[c].to_numpy()[keep])
    for c, v in nb.items():
        res[c] = r(np.asarray(v)[keep])
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfl-betting-model", default="/home/user/wbblanck/nfl-betting-model")
    args = ap.parse_args()
    nbm = Path(args.nfl_betting_model)
    sys.path[:0] = [str(nbm / "src")]
    import polars as pl
    from nflbet.ats import elo as E, kalman as K, xgb as X

    ek = json.loads((nbm / "data" / "backtests" / "elo_constants.json").read_text())
    kp = json.loads((nbm / "data" / "backtests" / "batch1_constants.json").read_text())["kalman"]
    kp = {k: kp[k] for k in K.PARAMS}

    fx = pd.read_csv(FIX)
    fx = fx[fx["season"] < 2025]          # holdout rows never reach the source either
    out, nb, final, (M, V), games = run_source(schedule(fx), ek, kp, E, K)

    hand = pd.DataFrame(HAND, columns=HAND_COLS)
    hout, hnb, hfinal, (hM, hV), _ = run_source(schedule(hand), ek, kp, E, K)

    # XGB: source features on the fixture (QB and EPA columns are NaN: no pbp/injury inputs here)
    s = schedule(fx).filter(pl.col("game_type") == "REG").with_columns(
        pl.col("home_team").replace(E.FRANCHISE).alias("ht"), pl.col("away_team").replace(E.FRANCHISE).alias("at"),
        pl.col("roof").is_in(["dome", "closed"]).cast(int).alias("dome"), pl.col("div_game").cast(int))
    f = out.select("game_id", pl.col("d").alias("elo_d"), "level", (pl.col("mh") - pl.col("ma")).alias("elo_fair"),
                   "kalman", "kalman_var")
    s = s.join(games.select("game_id", pl.col("neutral").cast(int)), on="game_id").join(f, on="game_id")
    for c in ("qb_h", "qb_a", "off8_h", "def8_h", "off8_a", "def8_a"):
        s = s.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    s = s.filter(pl.col("week").is_between(2, 18)).sort("game_id")
    Xall = np.round(s.select(X.FEATURES).to_numpy().astype(float), ROUND)
    y = s["result"].to_numpy().astype(float)
    se = s["season"].to_numpy()
    tr, te = np.isin(se, XGB_TRAIN), np.isin(se, XGB_TEST)
    n_trees = 100   # frozen count, data/backtests/batch4_model.json (num_trees 100)
    m = X.model(n_trees).fit(Xall[tr], y[tr])
    pred = m.predict(Xall[te])
    nan_cols = [c for j, c in enumerate(X.FEATURES) if np.isnan(Xall[:, j]).all()]
    keep = [j for j, c in enumerate(X.FEATURES) if c not in nan_cols]

    gold = {
        "source": {"repo": "wbblanck/nfl-betting-model", "commit": commit(nbm),
                   "modules": ["src/nflbet/ats/elo.py", "src/nflbet/ats/kalman.py", "src/nflbet/ats/xgb.py"],
                   "constants": {"elo": {k: ek[k] for k in ("k", "h", "c", "b", "theta")}, "kalman": kp}},
        "inputs": {"fixture": str(FIX.relative_to(REPO)), "fixture_seasons": [2015, 2018],
                   "hand": {"columns": HAND_COLS, "rows": [list(x) for x in HAND]}},
        "fixture": {"output_seasons": list(OUTPUT_SEASONS), **per_game(out, nb, OUTPUT_SEASONS),
                    "final_elo": {k: float(v) for k, v in sorted(final.items())},
                    "final_kalman": {k: [float(M[k]), float(V[k])] for k in sorted(M)}},
        "hand": {**per_game(hout, hnb), "final_elo": {k: float(v) for k, v in sorted(hfinal.items())}},
        "xgb": {"params": X.PARAMS, "n_trees": n_trees, "features": X.FEATURES, "all_nan_features": nan_cols,
                "round": ROUND, "train_seasons": list(XGB_TRAIN), "test_seasons": list(XGB_TEST),
                "game_id": s["game_id"].to_list(), "is_train": tr.tolist(), "is_test": te.tolist(),
                "X": [r(row, 10) for row in Xall[:, keep]], "y": y.tolist(), "pred": r(pred.astype(float), 9),
                "dome_source": s["dome"].to_list()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gold, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB), commit {gold['source']['commit'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
