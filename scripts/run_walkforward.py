"""Step-3 season walk-forward (test seasons 2015-2024, weekly rolling origin).

Writes data/processed/step3_oof_predictions.parquet (gitignored) and
docs/results/step3_walkforward.md. Never reads the 2025 holdout (the loader
drops it) and leaves the in-progress 2026 season out.

    python scripts/run_walkforward.py [--seasons 2015-2024] [--bootstrap 2000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_market.data.games import DEFAULT_CACHE, LINE_LABEL, load_games, load_nflverse_lines, moneyline_home_prob  # noqa: E402
from nfl_market.holdout import load_manifest  # noqa: E402
from nfl_market.models.bayesian import BayesianStrengthModel, KalmanMarginModel  # noqa: E402
from nfl_market.models.constant import ConstantModel  # noqa: E402
from nfl_market.models.elo import EloModel  # noqa: E402
from nfl_market.models.folds import FOLD_TABLE_ID, fold_table  # noqa: E402
from nfl_market.models.ridge_model import RidgeModel  # noqa: E402
from nfl_market.models.runner import (attach_outcomes, folds_won, game_metrics, metrics_table,  # noqa: E402
                                      paired_bootstrap, run_walkforward)
from nfl_market.models.xgb_model import XGBModel  # noqa: E402

OUT_PRED = ROOT / "data" / "processed" / "step3_oof_predictions.parquet"
OUT_MD = ROOT / "docs" / "results" / "step3_walkforward.md"
BASELINE = "elo_v1"
#: seasons where no ported constant was chosen (Elo/Kalman fit 2016-2020, XGB tree count 2016-2021)
CLEAN_SEASONS = (2015, 2022, 2023, 2024)
#: original repo figures, regular season weeks 2-18 with nflverse closing spread and both spread prices.
ORIG = {  # season -> (n, elo, xgb, close); wbblanck/nfl-betting-model data/backtests/elo_confirm.json + batch4.json
    2022: (255, 8.98597096879588, 9.056054023580224, 8.717647058823529),
    2023: (256, 10.462842674529039, 10.657189692894462, 9.818359375),
    2024: (256, 10.07075420715708, 9.998395847695065, 9.6484375),
}
ORIG_POOLED_2022_2025 = {"n": 1023, "elo": 9.964047803799682, "xgb": 10.00329451889659, "close": 9.536656891495602}


def parse_seasons(s: str) -> list[int]:
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(x, nd=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def md_table(df: pd.DataFrame, nd=3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(v, nd) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def benchmark_frame(games: pd.DataFrame, seasons) -> pd.DataFrame:
    """nflverse provider lines as a pseudo-prediction (capture time unknown; never a feature)."""
    lines = load_nflverse_lines()
    g = games[games.season.isin(seasons)][["game_id", "season", "week"]].merge(lines, on="game_id", how="left")
    return pd.DataFrame({"game_id": g.game_id, "season": g.season, "week": g.week, "model_id": "nflverse_line",
                         "fold_id": g.season.astype(str), "pred_margin": g.spread_line, "pred_total": g.total_line,
                         "pred_home_win_prob": moneyline_home_prob(g.home_moneyline, g.away_moneyline),
                         "_orig_sample": g.spread_line.notna() & g.home_spread_odds.notna() & g.away_spread_odds.notna()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2015-2024")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    seasons = parse_seasons(args.seasons)
    if max(seasons) >= 2025:
        raise SystemExit("2025 is the locked holdout and 2026 is forward-only: test seasons must be <= 2024")

    t0 = time.time()
    games = load_games()
    games = games[games.season <= max(seasons)].reset_index(drop=True)   # 2026 never enters
    models = [ConstantModel(), EloModel(), KalmanMarginModel(), BayesianStrengthModel(), RidgeModel(), XGBModel()]
    preds = run_walkforward(games, models, seasons, verbose=True)
    runtime = time.time() - t0
    timing = preds.attrs.get("fit_predict_seconds", {})
    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    preds.attrs = {}   # keep run timings out of the file so its checksum is reproducible
    preds.to_parquet(OUT_PRED, index=False)

    ft = fold_table(games, seasons)
    fold_m = metrics_table(preds, games, by=("model_id", "fold_id"))
    pooled = metrics_table(preds, games, by=("model_id",))
    clean = metrics_table(preds[preds.season.isin(CLEAN_SEASONS)], games, by=("model_id",))
    out = attach_outcomes(preds, games)
    reg = metrics_table(preds[out.game_type.eq("REG").to_numpy()], games, by=("model_id",))
    post = metrics_table(preds[out.game_type.ne("REG").to_numpy()], games, by=("model_id",))
    s2020 = metrics_table(preds[preds.season == 2020], games, by=("model_id",))

    # benchmark row
    bench = benchmark_frame(games, seasons)
    bo = bench.merge(games[["game_id", "home_score", "away_score"]], on="game_id")
    bo["margin"], bo["total"] = bo.home_score - bo.away_score, bo.home_score + bo.away_score
    have = bo.pred_margin.notna()
    bench_m = game_metrics(bo[have])
    bench_cov = f"{int(have.sum())}/{len(bo)} games have spread_line"

    # paired bootstrap vs Elo
    boots = []
    for m in [x.model_id for x in models if x.model_id != BASELINE]:
        for metric in ("abs_margin", "abs_total", "logloss_win"):
            r = paired_bootstrap(preds, games, m, BASELINE, metric, B=args.bootstrap)
            if r["n_games"]:
                col = {"abs_margin": "mae_margin", "abs_total": "mae_total", "logloss_win": "logloss_win"}[metric]
                w, n = folds_won(fold_m, m, BASELINE, col)
                rc = paired_bootstrap(preds[preds.season.isin(CLEAN_SEASONS)], games, m, BASELINE, metric,
                                      B=args.bootstrap)
                boots.append({**r, "folds_better": f"{w}/{n}", "clean_diff": rc["diff"], "clean_lo": rc["lo"],
                              "clean_hi": rc["hi"]})
    boots = pd.DataFrame(boots)

    # reproduction context: original repo sample analog (REG, weeks 2-18, closing spread + both prices)
    orig_ids = set(bench.loc[bench._orig_sample, "game_id"])
    rep_rows = []
    for s in (2022, 2023, 2024):
        n0, e0, x0, c0 = ORIG[s]
        sel = out[(out.season == s) & out.game_type.eq("REG") & out.week.between(2, 18) & out.game_id.isin(orig_ids)]
        bsel = bo[bo.game_id.isin(sel.game_id.unique())]
        ours = {m: (sel[sel.model_id == m].pred_margin - sel[sel.model_id == m].margin).abs().mean()
                for m in ("elo_v1", "kalman_gs_v1", "xgb_v1")}
        rep_rows.append({"season": s, "n_ours": int(sel.game_id.nunique()), "n_orig": n0,
                         "elo_ours": ours["elo_v1"], "elo_orig": e0,
                         "kalman_ours": ours["kalman_gs_v1"], "kalman_orig": "not reported",
                         "xgb_ours": ours["xgb_v1"], "xgb_orig": x0,
                         "close_ours_nflverse": (bsel.pred_margin - bsel.margin).abs().mean(), "close_orig": c0})
    rep = pd.DataFrame(rep_rows)
    sel = out[out.season.isin([2022, 2023, 2024]) & out.game_type.eq("REG") & out.week.between(2, 18)
              & out.game_id.isin(orig_ids)]
    rep_pool = {m: (sel[sel.model_id == m].pred_margin - sel[sel.model_id == m].margin).abs().mean()
                for m in ("elo_v1", "kalman_gs_v1", "xgb_v1")}
    rep_pool_n = int(sel.game_id.nunique())
    orig_3 = {k: sum(ORIG[s][0] * ORIG[s][i] for s in ORIG) / sum(ORIG[s][0] for s in ORIG)
              for k, i in (("elo", 1), ("xgb", 2), ("close", 3))}

    write_md(seasons, games, ft, fold_m, pooled, clean, reg, post, s2020, bench_m, bench_cov, boots, rep, rep_pool,
             rep_pool_n, orig_3, runtime, timing, args.bootstrap)
    summary = {"runtime_s": round(runtime, 1), "n_pred_rows": len(preds),
               "pooled": pooled[["model_id", "n_games", "mae_margin", "mae_total", "logloss_win"]].round(4).to_dict("records"),
               "pred_sha256": sha256(OUT_PRED)}
    print(json.dumps(summary, indent=1, default=str))


def write_md(seasons, games, ft, fold_m, pooled, clean, reg, post, s2020, bench_m, bench_cov, boots, rep, rep_pool,
             rep_pool_n, orig_3, runtime, timing, B):
    cols = ["model_id", "n_games", "mae_margin", "rmse_margin", "bias_margin", "mae_total", "rmse_total",
            "n_win", "logloss_win", "brier_win"]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip()
    except Exception:
        commit = "unknown"
    bench_row = pd.DataFrame([{"model_id": f"BENCHMARK: {LINE_LABEL}", **bench_m}])[cols]
    fm = fold_m.pivot(index="fold_id", columns="model_id", values="mae_margin").reset_index()
    ft_tot = fold_m.pivot(index="fold_id", columns="model_id", values="mae_total").reset_index()
    fl = fold_m.pivot(index="fold_id", columns="model_id", values="logloss_win").reset_index()
    ng = fold_m[fold_m.model_id == BASELINE][["fold_id", "n_games"]]
    fm = ng.merge(fm, on="fold_id")
    per_week = ft.groupby("fold_id").agg(origins=("week", "size"), min_train=("n_train_games", "min"),
                                         max_train=("n_train_games", "max")).reset_index()

    def verdict(row):
        if row.hi < 0:
            return "better than Elo (CI excludes 0)"
        if row.lo > 0:
            return "worse than Elo (CI excludes 0)"
        return "inconclusive"
    b = boots.copy()
    b["verdict"] = b.apply(verdict, axis=1)
    b = b[["candidate", "metric", "n_games", "diff", "lo", "hi", "folds_better", "clean_diff", "clean_lo", "clean_hi",
           "verdict"]]

    def pick(metric, cand):
        r = boots[(boots.metric == metric) & (boots.candidate == cand)]
        return None if r.empty else r.iloc[0]

    notes = []
    for c in ("kalman_gs_v1", "bayes_od_v1", "ridge_v1", "xgb_v1", "constant_v1"):
        r = pick("abs_margin", c)
        if r is not None:
            notes.append(f"- `{c}` margin MAE minus Elo: {r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}], "
                         f"lower on {r['folds_better']} folds; clean seasons {r['clean_diff']:+.3f} "
                         f"[{r['clean_lo']:+.3f}, {r['clean_hi']:+.3f}].")
    winners = []
    for _, r in boots.iterrows():
        w, n = (int(x) for x in r["folds_better"].split("/"))
        if r["hi"] < 0 and w > n / 2:
            winners.append(f"`{r['candidate']}` on {r['metric']} ({r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}], "
                           f"{r['folds_better']} folds; clean seasons {r['clean_diff']:+.3f} "
                           f"[{r['clean_lo']:+.3f}, {r['clean_hi']:+.3f}])")
    win_text = ("Beats Elo (pooled CI below 0 and majority of folds): " + "; ".join(winners) + "."
                if winners else "No model beats Elo on any metric by this rule.")
    margin_wins = [x for x in winners if "abs_margin" in x or "logloss_win" in x]
    if not margin_wins:
        win_text += (" No model beats Elo on margin MAE or home-win log loss: every margin difference is within "
                     "about 0.1 point and its CI covers 0, so for the margin the ported Elo remains the model to beat "
                     "(P6: keep the simplest).")
    timing_s = ", ".join(f"{k} {v:.0f}s" for k, v in timing.items())
    rp = pd.DataFrame([{"season": "2022-2024 pooled", "n_ours": rep_pool_n, "n_orig": sum(ORIG[s][0] for s in ORIG),
                        "elo_ours": rep_pool["elo_v1"], "elo_orig": orig_3["elo"],
                        "kalman_ours": rep_pool["kalman_gs_v1"], "kalman_orig": "not reported",
                        "xgb_ours": rep_pool["xgb_v1"], "xgb_orig": orig_3["xgb"],
                        "close_ours_nflverse": np.nan, "close_orig": orig_3["close"]}])
    rep_all = pd.concat([rep.astype(object), rp.astype(object)], ignore_index=True)

    text = f"""# Step 3 walk-forward results

Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `python scripts/run_walkforward.py`
(code commit `{commit}` plus uncommitted step-3 working tree). Experiment `20261119-step3-game-models-01`
(`docs/methodology/experiment_log.md`). Runtime {runtime:.0f}s (fit+predict: {timing_s}).

**Status: exploratory until the leakage audit is signed (`leakage_audit.md`) and the feature registry entries
are reviewed.** No 2025 (holdout `{load_manifest()['holdout_id']}`) or 2026 game was scored or used.

## Setup

- Data: nflverse `games.csv` cache `{DEFAULT_CACHE.relative_to(ROOT)}` sha256 `{sha256(DEFAULT_CACHE)[:16]}...`,
  seasons {int(games.season.min())}-{max(seasons)} (2025 excluded by `exclude_holdout`; 2026 dropped).
- Fold table `{FOLD_TABLE_ID}`: outer test seasons {seasons[0]}-{seasons[-1]} (>= 3 training seasons; history from 1999);
  weekly rolling origin inside each test season (postseason weeks are their own origins). Before week w every
  model is refit on all games with `label_available_at_utc` < week w's first T. V2/V3/holdout asserted per origin.
- T = kickoff - 24h. Label available = next Wednesday 12:00 US/Eastern after kickoff. Ties count in MAE and are
  dropped from log loss / Brier.
- Models: `constant_v1` (rung 0), `elo_v1` (rung 1c, ported Elo + NB score model, frozen constants),
  `kalman_gs_v1` (ported Glickman-Stern Kalman margin rating, frozen constants; margin only), `bayes_od_v1`
  (our offense/defense Kalman extension, a-priori constants; margin + total), `ridge_v1` (ours; alpha by inner
  season split), `xgb_v1` (ported a-priori XGBoost settings, 100 trees; QB and EPA features not available).
  Win probabilities: Elo logistic; Kalman/Bayes Normal; Ridge/XGB Normal with inner-split residual SD.
- **V5 caveat:** ported constants were chosen on 2016-2020 (Elo, Kalman) and 2016-2021 (XGB tree count) in the
  source repo. For those models the 2016-2021 folds are not nested. "Clean" columns use seasons
  {', '.join(map(str, CLEAN_SEASONS))} only.
- Bootstrap: paired, game-clustered (one row per game per model), B = {B}, 95% percentile CI; negative = better
  than Elo.

## Pooled, test seasons {seasons[0]}-{seasons[-1]}

{md_table(pooled[cols])}
{md_table(bench_row).splitlines()[-1]}

Benchmark row: {LINE_LABEL}; {bench_cov}; win probability from de-vigged nflverse moneylines where present
(n_win). It is close-like, reported for context only, never a feature, and not part of the prediction file.

### Clean seasons only ({', '.join(map(str, CLEAN_SEASONS))})

{md_table(clean[cols])}

### Slices

Regular season:

{md_table(reg[cols])}

Postseason:

{md_table(post[cols])}

2020 (limited crowds):

{md_table(s2020[cols]) if len(s2020) else "(2020 not in the test seasons)"}

## Per fold

Origins and training sizes:

{md_table(per_week, 0)}

Margin MAE:

{md_table(fm)}

Total MAE:

{md_table(ft_tot)}

Home-win log loss:

{md_table(fl)}

## Paired difference vs Elo (candidate minus `elo_v1`)

{md_table(b)}

## Which models beat Elo?

{win_text}

{chr(10).join(notes)}

Elo's total is its NB score model's mu_home + mu_away (league level times a cosh of the Elo gap), so the
total comparisons test team-specific scoring information that Elo does not carry.

Reading: a model "beats Elo" only if the pooled CI is below 0 and it wins a majority of folds
(`predictive_modeling.md` Section 2). See the table above for the verdict per metric. Every model is far from
the nflverse line benchmark (which has unknown capture time and is close-like), as expected for models without
market, QB or injury inputs.

## Reproduction context vs the source repo (descriptive)

Original figures: wbblanck/nfl-betting-model `data/backtests/elo_confirm.json` (Elo, close; `docs/elo_report.md`)
and `data/backtests/batch4.json` (XGBoost; `docs/batch4_report.md`). Their headline: plain Elo margin MAE 9.964 on
1023 games (2022-2025 REG weeks 2-18 with closing spread and both prices) vs closing spread 9.537. We cannot score
2025, so we compare 2022-2024 on the same sample definition. Kalman: the source reports Kalman MAE only for
2005-2015 (`batch1.json`, 10.907 on 2,640 games), so 2022-2024 is "not reported".

{md_table(rep_all)}

Our 2022 and 2023 Elo MAE and the nflverse close match the source to three decimals on identical game
counts; 2024 differs by 0.005, consistent with the label-timing rule (the week-16 Monday game's label arrives
Wednesday 2024-12-25 12:00 ET, after the Christmas-Day games' T, so here it enters the ratings one week later). XGBoost differs by design (rows below).

What differs from the source runs:

| item | source | here |
|---|---|---|
| seasons scored | 2022-2025 (confirm), REG weeks 2-18 | 2015-2024 all weeks incl. postseason; table above restricted to 2022-2024 REG weeks 2-18 with closing spread + prices |
| Elo / Kalman constants | fit 2016-2020, frozen | same frozen constants (ported) |
| Elo / Kalman update timing | week-batched; a week's forecasts use all earlier weeks | same, plus the label rule: a game is used only once its label (next Wed 12:00 ET) is before T, which purges e.g. Wednesday/Christmas games for one week |
| Elo scoring level | prev REG season + current-season REG weeks before w | same formula, as-of label availability |
| XGBoost features | Elo gap/fair, QB deltas (expected starter), Kalman fair/var, 8-game EPA/play off/def, level, rest, neutral, div, dome (dome or closed), week | QB deltas dropped; EPA replaced by 8-game points for/against; dome = roof `dome` only |
| XGBoost training | fit once on 2016-2021 REG weeks 2-18, frozen | refit every week on all labelled games since 2000 (expanding window), total model added |
| close benchmark | nflverse spread_line | same column, labelled capture time unknown |

## Open issues

- QB effects skipped: nflverse `home_qb_id`/`away_qb_id` are post-game starters. Needs pre-game depth chart /
  injury report data with publication times.
- EPA features (source XGBoost) need nflverse play-by-play; not ingested in this step.
- Rest days come from the final schedule (REVIEW REQUIRED for flexed/rescheduled games); weather not used.
- 1999 kickoff times are imputed (13:00 ET); 1999 is training history only.
- Home/away score independence (P5) in `bayes_od_v1` and the Elo NB score model is untested here; the
  distributions workstream tests it on these OOF predictions.
- Ported constants are not nested for 2016-2021 folds (see V5 caveat); an in-fold refit of the Elo grid and
  Kalman ML would remove it.
"""
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(text)


if __name__ == "__main__":
    print(f"python {platform.python_version()}", flush=True)
    main()
