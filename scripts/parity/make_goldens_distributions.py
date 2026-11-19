"""Generate parity goldens for the distribution ports from the ORIGINAL code.

Imports the owner's read-only clones (never edits them):
  * nfl_ats:            engine.models.margin_dist (MarginDist kinds, choose_h)
  * nfl-betting-model:  nflbet.ats.batch5 (K1, S1), nflbet.ats.tdfg (D2), nflbet.ats.elo (margin_dist)

Writes tests/fixtures/parity/distributions_goldens.json with the inputs, the
outputs and each source repo's commit hash. tests/test_parity_distributions.py
checks our ports against it without importing the originals.

    pip install polars   # the originals import it
    python scripts/parity/make_goldens_distributions.py \
        [--nfl-ats /home/user/wbblanck/nfl_ats] [--nfl-betting-model /home/user/wbblanck/nfl-betting-model]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "tests" / "fixtures" / "parity" / "distributions_goldens.json"


def commit(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def training_set(seed: int = 20240901, seasons=range(2015, 2021), n: int = 150) -> dict:
    """Small deterministic training set: market home spread c (nfl_ats sign: negative = home favoured),
    integer home margin with a key-number bump, season."""
    rng = np.random.default_rng(seed)
    c, m, s = [], [], []
    for season in seasons:
        cc = np.round(rng.normal(-2.0, 5.5, n) * 2) / 2
        mm = np.round(-cc + rng.standard_t(7, n) * 12.5)
        snap = rng.uniform(size=n)
        mm = np.where(snap < 0.08, np.sign(mm + 0.1) * 3, np.where(snap < 0.12, np.sign(mm + 0.1) * 7, mm))
        c += cc.tolist(); m += mm.astype(int).tolist(); s += [season] * n
    return {"close": c, "margin": m, "season": s}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfl-ats", default="/home/user/wbblanck/nfl_ats")
    ap.add_argument("--nfl-betting-model", default="/home/user/wbblanck/nfl-betting-model")
    args = ap.parse_args()
    ats, nbm = Path(args.nfl_ats), Path(args.nfl_betting_model)
    sys.path[:0] = [str(ats / "src"), str(nbm / "src")]
    from engine.models import margin_dist as MD
    from nflbet.ats import batch5 as B5, elo as E, tdfg as TD

    tr = training_set()
    c, m, season = (np.asarray(tr[k], float) for k in ("close", "margin", "season"))
    query_c = [-10.0, -7.0, -3.0, -2.5, 0.0, 3.5, 6.0]
    t_half = np.arange(-20.5, 21.0, 1.0)
    t_whole = np.arange(-20.0, 21.0, 1.0)
    t_fine = np.round(np.arange(-12.5, 12.5 + 1e-9, 0.01), 2)

    margin_dist = {}
    for kind, h in [("normal", 1.0), ("shifted", 1.0), ("kernel", 1.0), ("kernel_sym", 1.5), ("keynum", 1.0)]:
        md = MD.MarginDist(kind=kind, h=h).fit(c, m)
        rec = {"h": h, "sigma": md.sigma, "sf_half": [], "sf_whole": []}
        for cq in query_c:
            rec["sf_half"].append(md.sf(t_half, np.full(t_half.size, cq)).tolist())
            rec["sf_whole"].append(md.sf(t_whole, np.full(t_whole.size, cq)).tolist())
        if kind == "keynum":
            rec["w"] = md.w.tolist()
            rec["pmf"] = md.pmf(np.asarray(query_c)).tolist()
        if kind == "shifted":
            rec["sf_fine_c"] = -2.5
            rec["sf_fine"] = md.sf(t_fine, np.full(t_fine.size, -2.5)).tolist()
        margin_dist[kind] = rec
    choose_h = {kind: MD.choose_h(c, m, season, kind=kind) for kind in ("kernel", "kernel_sym")}

    # nfl-betting-model: K1, D2, S1
    mu_q = [-10.0, -3.0, 0.0, 2.5, 7.0, 13.5]
    k5 = json.loads((nbm / "data" / "backtests" / "batch5_constants.json").read_text())
    k3 = json.loads((nbm / "data" / "backtests" / "batch3_constants.json").read_text())
    k1_frozen = B5.k1_dist(np.asarray(mu_q), k5["k1"]["sigma"], k5["k1"]["w"])
    k1_fit = B5.fit_k1(-c, m)
    d_q = [-150.0, -50.0, 0.0, 40.0, 120.0]
    lv_q = [20.0, 22.0, 23.0, 24.0, 26.0]
    ph, pa = TD.game_pmfs(np.asarray(d_q), np.asarray(lv_q), k3)
    _, d2_margin = E.margin_dist(ph, pa)
    s1_margin = B5.s1_margin(np.asarray(d_q), np.asarray(lv_q), k3, k5["s1"]["c0"])

    out = {
        "sources": {"nfl_ats": {"path": str(ats), "commit": commit(ats),
                                "modules": ["src/engine/models/margin_dist.py"]},
                    "nfl-betting-model": {"path": str(nbm), "commit": commit(nbm),
                                          "modules": ["src/nflbet/ats/batch5.py", "src/nflbet/ats/tdfg.py",
                                                      "src/nflbet/ats/elo.py"]}},
        "generator": "scripts/parity/make_goldens_distributions.py",
        "training": tr,
        "query_c": query_c, "t_half": t_half.tolist(), "t_whole": t_whole.tolist(), "t_fine": t_fine.tolist(),
        "margin_dist": margin_dist,
        "choose_h": choose_h,
        "k1": {"mu": mu_q, "constants": k5["k1"], "frozen_pmf": k1_frozen.tolist(), "fit": k1_fit},
        "d2": {"d": d_q, "level": lv_q, "constants": k3, "home_pmf": ph.tolist(), "away_pmf": pa.tolist(),
               "margin": d2_margin.tolist()},
        "s1": {"d": d_q, "level": lv_q, "c0": k5["s1"]["c0"], "margin": s1_margin.tolist()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
