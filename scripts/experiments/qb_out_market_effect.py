"""How many points does the market move for a ruled-out starting QB? (research only, 2026-11-26)

No line from before the injury news exists in our data (nflverse carries one close-like line per game; our own
captures start 2026-11-21), so this uses the close-like nflverse line as the dependent variable instead:

    spread_line ~ team-strength features at T + home_qb_out + away_qb_out

on 2009-2024 games (injury data from 2009; 2025 never read). Because the line already prices team strength, the
flag coefficients estimate the market's adjustment for a newly ruled-out expected starter, with far less noise
than final margins (residual SD of the line around the features is ~3 points, of results ~13). The same regression
with the final margin as the dependent variable is shown for comparison. CIs: bootstrap clustered by (season, week),
B = 4000. Features: the xgb_v2 set without the QB columns (strength, form, rest, venue), computed at T.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nfl_market.data.games import load_games, load_nflverse_lines  # noqa: E402
from nfl_market.models.features import XGB_FEATURES  # noqa: E402
from nfl_market.features.external import EPA_COLUMNS  # noqa: E402

f = pd.read_parquet(ROOT / "data" / "processed" / "ebm_ngb_features.parquet")  # from ebm_ngboost_backtest.py
g = load_games()[["game_id", "home_score", "away_score"]]
d = (f.merge(g, on="game_id").merge(load_nflverse_lines()[["game_id", "spread_line", "total_line"]], on="game_id"))
d = d[(d.season >= 2009) & (d.season <= 2024) & d.home_qb_out.notna() & d.spread_line.notna() & d.home_score.notna()]
d["margin"], d["total"] = d.home_score - d.away_score, d.home_score + d.away_score
d["margin_minus_line"] = d.margin - d.spread_line
ctrl = [c for c in list(XGB_FEATURES) + list(EPA_COLUMNS)]
X = d[ctrl].fillna(d[ctrl].median())
X = (X - X.mean()) / X.std().replace(0, 1)
X = np.column_stack([np.ones(len(d)), X.to_numpy(), d.home_qb_out, d.away_qb_out])
blk = pd.factorize(d.season.astype(str) + "_" + d.week_no.astype(str))[0]
rng = np.random.default_rng(0)
nb = blk.max() + 1
idx_by = [np.flatnonzero(blk == b) for b in range(nb)]
rows = []
for dep in ("spread_line", "margin", "margin_minus_line", "total_line", "total"):
    y = d[dep].to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid_sd = float(np.std(y - X @ beta))
    bs = []
    for _ in range(4000):
        ii = np.concatenate([idx_by[b] for b in rng.integers(0, nb, nb)])
        bs.append(np.linalg.lstsq(X[ii], y[ii], rcond=None)[0][-2:])
    bs = np.array(bs)
    for j, side in ((0, "home_qb_out"), (1, "away_qb_out")):
        rows.append({"dependent": dep, "flag": side, "coef": beta[-2 + j],
                     "ci_lo": np.quantile(bs[:, j], .025), "ci_hi": np.quantile(bs[:, j], .975), "resid_sd": resid_sd})
    # pooled: home-perspective effect (home out = -x, away out = +x)
    pooled = (beta[-1] - beta[-2]) / 2
    pb = (bs[:, 1] - bs[:, 0]) / 2
    rows.append({"dependent": dep, "flag": "pooled (points against the team whose QB is out)",
                 "coef": pooled if dep in ("spread_line", "margin", "margin_minus_line") else (beta[-1] + beta[-2]) / 2,
                 "ci_lo": np.quantile(pb if dep in ("spread_line", "margin", "margin_minus_line") else (bs[:, 1] + bs[:, 0]) / 2, .025),
                 "ci_hi": np.quantile(pb if dep in ("spread_line", "margin", "margin_minus_line") else (bs[:, 1] + bs[:, 0]) / 2, .975),
                 "resid_sd": resid_sd})
out = pd.DataFrame(rows)
print(f"games {len(d)}, home flags {int(d.home_qb_out.sum())}, away flags {int(d.away_qb_out.sum())}")
print(out.round(2).to_string(index=False))
# per-season-era stability for the line
for lo, hi in ((2009, 2016), (2017, 2024)):
    m = ((d.season >= lo) & (d.season <= hi)).to_numpy()
    b = np.linalg.lstsq(X[m], d.spread_line.to_numpy(float)[m], rcond=None)[0]
    print(lo, hi, "line: home", round(b[-2], 2), "away", round(b[-1], 2), "n flags", int(d.home_qb_out[m].sum() + d.away_qb_out[m].sum()))

# ---------------------------------------------------------------- open -> close (SBRO, 2009-2021)
# SBRO open/close spreads (copied from wbblanck/NFL_ATS data/processed/sbro_lines.csv into the gitignored
# data/raw/external/; never committed here). The open is posted before most of the week's injury news, so the
# open-to-close move in games where the expected starter is ruled out estimates the market's QB-out
# adjustment directly. When the news was already out at the open the move understates it (lower bound).
sb_path = ROOT / "data" / "raw" / "external" / "sbro_lines.csv"
if sb_path.exists():
    sb = pd.read_csv(sb_path)
    sb = sb[(sb.open_status == "ok") & (sb.close_status == "ok")]
    e = sb.merge(f[["game_id", "home_qb_out", "away_qb_out", "week_no", "season"]].rename(columns={"season": "s2"}),
                 on="game_id")
    e = e[e.home_qb_out.notna()]
    e["move_home"] = -(e.close_home_spread - e.open_home_spread)   # home-perspective points, + = toward home
    Xo = np.column_stack([np.ones(len(e)), e.home_qb_out, e.away_qb_out])
    yo = e.move_home.to_numpy(float)
    bo = np.linalg.lstsq(Xo, yo, rcond=None)[0]
    blk2 = pd.factorize(e.season.astype(str) + "_" + e.week_no.astype(str))[0]
    idx2 = [np.flatnonzero(blk2 == b) for b in range(blk2.max() + 1)]
    bs2 = []
    for _ in range(4000):
        ii = np.concatenate([idx2[b] for b in rng.integers(0, len(idx2), len(idx2))])
        bs2.append(np.linalg.lstsq(Xo[ii], yo[ii], rcond=None)[0][1:])
    bs2 = np.array(bs2)
    pooled = (bo[2] - bo[1]) / 2
    pb2 = (bs2[:, 1] - bs2[:, 0]) / 2
    print(f"\nSBRO open->close, {len(e)} games 2009-{int(e.season.max())}, flags home {int(e.home_qb_out.sum())} "
          f"away {int(e.away_qb_out.sum())}; move SD {yo.std():.2f}")
    print(f"home_qb_out {bo[1]:.2f} [{np.quantile(bs2[:, 0], .025):.2f}, {np.quantile(bs2[:, 0], .975):.2f}]  "
          f"away_qb_out {bo[2]:.2f} [{np.quantile(bs2[:, 1], .025):.2f}, {np.quantile(bs2[:, 1], .975):.2f}]")
    print(f"pooled move against the team whose QB is out: {pooled:.2f} "
          f"[{np.quantile(pb2, .025):.2f}, {np.quantile(pb2, .975):.2f}]")
    fl = e[(e.home_qb_out == 1) | (e.away_qb_out == 1)]
    mv = np.where(fl.home_qb_out == 1, -fl.move_home, fl.move_home)
    print("flagged-game moves against the out team: share >= 3 pts", round(float(np.mean(mv >= 3)), 2),
          "share <= 0", round(float(np.mean(mv <= 0)), 2), "median", float(np.median(mv)))
