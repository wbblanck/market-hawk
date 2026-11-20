"""Step-4 stacking evaluation (stack_v1) on the step-3 OOF predictions.

Reads data/processed/step3_oof_predictions.parquet (run scripts/run_walkforward.py first) and the
nflverse cache, then writes:

* data/processed/step4_stack_predictions.parquet (gitignored; stack_v1 rows, PREDICTION_COLUMNS)
* config/models/stack_v1_weights.json (forward weights + K1 fit on all OOF seasons 2015-2024)
* docs/results/step4_stacking.md
* docs/results/step4_model_health.json (MODEL_HEALTH_COLUMNS records, read by the dashboard)

Never reads 2025 (holdout) or 2026.

    python scripts/run_stacking.py [--bootstrap 2000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_market.data.games import LINE_LABEL, load_games, load_nflverse_lines  # noqa: E402
from nfl_market.distributions import calibration as cal  # noqa: E402
from nfl_market.markets.interface import MODEL_HEALTH_COLUMNS, STACK_MODEL_ID  # noqa: E402
from nfl_market.models.interface import validate_predictions  # noqa: E402
from nfl_market.models.runner import folds_won, metrics_table, paired_bootstrap  # noqa: E402
from nfl_market.models.stacking import (EQUAL_MODEL_ID, MARGIN_COMPONENTS, TOTAL_COMPONENTS,  # noqa: E402
                                        forward_weights, home_win_no_tie, k1_pmfs, k1_walkforward,
                                        refuse_holdout, walkforward_stack)

OOF = ROOT / "data" / "processed" / "step3_oof_predictions.parquet"
OUT_PRED = ROOT / "data" / "processed" / "step4_stack_predictions.parquet"
OUT_WEIGHTS = ROOT / "config" / "models" / "stack_v1_weights.json"
OUT_MD = ROOT / "docs" / "results" / "step4_stacking.md"
OUT_HEALTH = ROOT / "docs" / "results" / "step4_model_health.json"
EXPERIMENT_ID = "20261119-step4-stacking-01"
BASELINE = "elo_v1"
#: step-3 clean seasons are 2015, 2022-2024; 2015 has no 3 prior OOF seasons, so it is never scored here
CLEAN_SEASONS = (2022, 2023, 2024)
VARIANTS = {"stack_mae_v1": "MAE objective (reported variant)",
            "stack_b5_v1": "squared error, bayes_od_v1 added as a fifth candidate (margin and total)"}
METRICS = (("abs_margin", "mae_margin"), ("abs_total", "mae_total"), ("logloss_win", "logloss_win"))


def fmt(x, nd=3):
    if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)):
        return ""
    if isinstance(x, (float, np.floating)):
        return f"{x:.{nd}f}"
    return str(x)


def md_table(df: pd.DataFrame, nd=3) -> str:
    if df.empty:
        return "_(empty)_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v, nd) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def with_outcomes(frame: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    g = games[["game_id", "label_available_at_utc", "home_score", "away_score"]]
    df = frame.merge(g, on="game_id", how="left", validate="many_to_one")
    df["margin"] = df["home_score"] - df["away_score"]
    df["total"] = df["home_score"] + df["away_score"]
    return df


def ece_table(p: np.ndarray, margin: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Binary metrics of P(home win) on non-tie games, 10 equal-width bins."""
    ok = (margin != 0) & np.isfinite(p)
    return cal.binary_metrics(p[ok], (margin[ok] > 0).astype(float), n_bins=10, strategy="uniform")


def pmf_block(fds: dict, rows: pd.DataFrame) -> tuple[int, np.ndarray]:
    """K1 pmfs for each row using the fit of its season (rows must all be scored seasons)."""
    mats, lo = [], None
    for s, grp in rows.groupby("season", sort=False):
        lo, m = k1_pmfs(fds[int(s)], grp["pred_margin"].to_numpy(float))
        mats.append(pd.DataFrame(m, index=grp.index))
    return lo, pd.concat(mats).loc[rows.index].to_numpy()


def verdict(boot: dict, won: tuple[int, int]) -> str:
    w, n = won
    if boot["hi"] < 0 and w > n / 2:
        return "beats"
    if boot["lo"] > 0 and (n - w) > n / 2:
        return "worse"
    return "no difference shown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    t0 = time.time()
    if not OOF.exists():
        raise SystemExit(f"{OOF} missing: run scripts/run_walkforward.py first")
    oof = pd.read_parquet(OOF)
    refuse_holdout(oof["game_id"], "run_stacking")
    if oof["season"].max() > 2024:
        raise SystemExit("OOF file contains seasons after 2024; refusing (2025 holdout, 2026 forward-only)")
    games = load_games()
    games = games[games["season"] <= 2024].reset_index(drop=True)

    # ------------------------------------------------------------------ stacks
    runs = {
        STACK_MODEL_ID: walkforward_stack(oof, games),
        EQUAL_MODEL_ID: walkforward_stack(oof, games, model_id=EQUAL_MODEL_ID, equal=True),
        "stack_mae_v1": walkforward_stack(oof, games, model_id="stack_mae_v1", objective="mae"),
        "stack_b5_v1": walkforward_stack(oof, games, model_id="stack_b5_v1",
                                         margin_components=(*MARGIN_COMPONENTS, "bayes_od_v1"),
                                         total_components=(*TOTAL_COMPONENTS, "bayes_od_v1")),
    }
    stack = runs[STACK_MODEL_ID].predictions
    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    stack.to_parquet(OUT_PRED, index=False)
    pred_sha = hashlib.sha256(OUT_PRED.read_bytes()).hexdigest()

    fw = forward_weights(oof, games)
    fw = {"experiment_id": EXPERIMENT_ID, **fw,
          "generated_by": "python scripts/run_stacking.py", "source_oof": str(OOF.relative_to(ROOT))}
    OUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    OUT_WEIGHTS.write_text(json.dumps(fw, indent=2) + "\n")

    eval_ids = set(stack["game_id"])
    seasons = sorted(int(s) for s in stack["season"].unique())
    singles = oof[oof["game_id"].isin(eval_ids)]
    allp = validate_predictions(pd.concat([singles, *[r.predictions for r in runs.values()]], ignore_index=True))
    main_ids = [STACK_MODEL_ID, EQUAL_MODEL_ID, *MARGIN_COMPONENTS, "bayes_od_v1", "constant_v1"]
    main_p = allp[allp["model_id"].isin(main_ids)]

    fold_m = metrics_table(allp, games, by=("model_id", "fold_id"))
    pooled = metrics_table(allp, games, by=("model_id",))
    clean = metrics_table(allp[allp["season"].isin(CLEAN_SEASONS)], games, by=("model_id",))
    clean_fold = fold_m[fold_m["fold_id"].isin([str(s) for s in CLEAN_SEASONS])]

    pooled_main = pooled.set_index("model_id")
    best = {"abs_margin": pooled_main.loc[list(MARGIN_COMPONENTS), "mae_margin"].idxmin(),
            "abs_total": pooled_main.loc[list(TOTAL_COMPONENTS), "mae_total"].idxmin(),
            "logloss_win": pooled_main.loc[list(MARGIN_COMPONENTS), "logloss_win"].idxmin()}

    def comparisons(cands, frame, fm, tag):
        rows = []
        for cand in cands:
            for metric, col in METRICS:
                for base in dict.fromkeys([BASELINE, best[metric]]):
                    b = paired_bootstrap(frame, games, cand, base, metric=metric, B=args.bootstrap, seed=0)
                    won = folds_won(fm, cand, base, col)
                    rows.append({"scope": tag, "candidate": cand, "baseline": base, "metric": metric,
                                 "n_games": b["n_games"], "diff": b["diff"], "lo": b["lo"], "hi": b["hi"],
                                 "folds_won": f"{won[0]}/{won[1]}", "verdict": verdict(b, won)})
        return pd.DataFrame(rows)

    boot = comparisons([STACK_MODEL_ID, EQUAL_MODEL_ID], allp, fold_m, "pooled 2018-2024")
    boot_clean = comparisons([STACK_MODEL_ID, EQUAL_MODEL_ID], allp[allp["season"].isin(CLEAN_SEASONS)],
                             clean_fold, "clean 2022-2024")
    boot_var = comparisons(list(VARIANTS), allp, fold_m, "pooled 2018-2024")
    boot_var = boot_var[boot_var["baseline"] == BASELINE]

    # ------------------------------------------------------ win probabilities
    out = with_outcomes(allp, games)
    win_rows, rel_stack = [], None
    sw = {w.season: w for w in runs[STACK_MODEL_ID].weights if w.target == "margin"}
    wide_p = out.pivot_table(index="game_id", columns="model_id", values="pred_home_win_prob", aggfunc="first")
    st = out[out["model_id"] == STACK_MODEL_ID].set_index("game_id")
    pavg = np.array([sum(w * wide_p.loc[g, c] for c, w in sw[int(s)].as_dict().items())
                     for g, s in zip(st.index, st["season"])])
    probs = {m: out[out["model_id"] == m].set_index("game_id").loc[st.index, "pred_home_win_prob"].to_numpy(float)
             for m in [STACK_MODEL_ID, EQUAL_MODEL_ID, *MARGIN_COMPONENTS]}
    probs["stack_v1 margin weights on component P(win)"] = pavg
    marg = st["margin"].to_numpy(float)
    for name, p in probs.items():
        summ, rel = ece_table(p, marg)
        win_rows.append({"model": name, "n_win": summ["n"], "logloss": summ["log_loss"], "brier": summ["brier"],
                         "ece_10bin": summ["ece"]})
        if name == STACK_MODEL_ID:
            rel_stack = rel
    win_tab = pd.DataFrame(win_rows)

    # --------------------------------------- K1 pmfs: PIT, coverage, cover calibration
    lines = load_nflverse_lines()[["game_id", "spread_line"]]
    oofo = with_outcomes(oof, games)
    fds = {STACK_MODEL_ID: runs[STACK_MODEL_ID].k1, EQUAL_MODEL_ID: runs[EQUAL_MODEL_ID].k1}
    for c in MARGIN_COMPONENTS:
        fds[c] = k1_walkforward(oofo[oofo["model_id"] == c])
    pmfs, ev_rows = {}, {}
    for m in fds:
        rows = out[out["model_id"] == m].merge(lines, on="game_id", how="left").reset_index(drop=True)
        ev_rows[m] = rows
        pmfs[m] = pmf_block(fds[m], rows)
    lo, mat = pmfs[STACK_MODEL_ID]
    srows = ev_rows[STACK_MODEL_ID]
    y = srows["margin"].to_numpy(float)
    pit = cal.by_slice((lo, mat), y, srows["season"].to_numpy(), seed=0)
    cov = cal.interval_coverage((lo, mat), y)
    cover_rows, cover_ll, rel_cover = [], {}, None
    for m in fds:
        r = ev_rows[m]
        lo_m, mat_m = pmfs[m]
        has = r["spread_line"].notna().to_numpy()
        for scope, sel in [("pooled", has), ("clean", has & r["season"].isin(CLEAN_SEASONS).to_numpy()),
                           *[(str(s), has & (r["season"] == s).to_numpy()) for s in seasons]]:
            summ, rel = cal.cover_calibration((lo_m, mat_m[sel]), r["margin"].to_numpy(float)[sel],
                                              -r["spread_line"].to_numpy(float)[sel], n_bins=10, strategy="uniform")
            cover_ll[(m, scope)] = summ["log_loss"]
            if scope == "pooled":
                cover_rows.append({"model": m, "n_no_push": summ["n"], "n_push_excluded": summ["n_push_excluded"],
                                   "cover_logloss": summ["log_loss"], "brier": summ["brier"], "ece_10bin": summ["ece"]})
                if m == STACK_MODEL_ID:
                    rel_cover = rel
    cover_tab = pd.DataFrame(cover_rows)
    k1_hw = cal.home_win_calibration((lo, mat), y)[0]

    # ----------------------------------------------------------- model health
    gen = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    health_models = [STACK_MODEL_ID, EQUAL_MODEL_ID, *MARGIN_COMPONENTS]
    o_main = out[out["model_id"].isin(health_models)]
    records = []
    for m in health_models:
        om = o_main[o_main["model_id"] == m]
        for scope, fold_id, sel in [("pooled", "", np.ones(len(om), bool)),
                                    ("clean", "", om["season"].isin(CLEAN_SEASONS).to_numpy()),
                                    *[("fold", str(s), (om["season"] == s).to_numpy()) for s in seasons]]:
            g = om[sel]
            met = metrics_table(allp[allp["game_id"].isin(g["game_id"]) & (allp["model_id"] == m)], games,
                                by=("model_id",)).iloc[0]
            ece = ece_table(g["pred_home_win_prob"].to_numpy(float), g["margin"].to_numpy(float))[0]["ece"]
            wm = fw["margin"]["weights"].get(m, np.nan) if m in MARGIN_COMPONENTS else np.nan
            wt = fw["total"]["weights"].get(m, np.nan) if m in MARGIN_COMPONENTS else np.nan
            rec = {"model_id": m, "scope": scope, "fold_id": fold_id, "n_games": int(met["n_games"]),
                   "mae_margin": met["mae_margin"], "mae_total": met["mae_total"], "logloss_win": met["logloss_win"],
                   "brier_win": met["brier_win"], "ece_win": ece,
                   "cover_logloss": cover_ll.get((m, fold_id or scope), np.nan),
                   "stack_weight_margin": wm, "stack_weight_total": wt,
                   "experiment_id": EXPERIMENT_ID, "generated_at_utc": gen}
            records.append({k: (None if isinstance(v, float) and np.isnan(v) else
                                (float(v) if isinstance(v, (float, np.floating)) else v)) for k, v in rec.items()})
    assert all(list(r) == list(MODEL_HEALTH_COLUMNS) for r in records)
    OUT_HEALTH.write_text(json.dumps(records, indent=2) + "\n")

    # --------------------------------------------------------------- markdown
    def wide(col, models):
        w = fold_m[fold_m["model_id"].isin(models)].pivot(index="fold_id", columns="model_id", values=col)
        w = w[[m for m in models if m in w.columns]].reset_index()
        return w

    cols = ["model_id", "n_games", "mae_margin", "rmse_margin", "bias_margin", "mae_total", "n_win", "logloss_win",
            "brier_win"]
    order = {m: i for i, m in enumerate(main_ids + list(VARIANTS))}
    srt = lambda d: d.assign(_o=d["model_id"].map(order)).sort_values("_o").drop(columns="_o")  # noqa: E731
    wt_tab = pd.concat([runs[m].weights_table() for m in (STACK_MODEL_ID, "stack_mae_v1")], ignore_index=True)
    wt_m = wt_tab[wt_tab["target"] == "margin"].dropna(axis=1, how="all")
    wt_t = wt_tab[wt_tab["target"] == "total"].dropna(axis=1, how="all")
    b5 = runs["stack_b5_v1"].weights_table()
    fwd = pd.DataFrame([{"target": t, **{f"w_{c}": w for c, w in fw[t]["weights"].items()},
                         "n_train": fw[t]["n_train"]} for t in ("margin", "total")])
    k1p = pd.DataFrame([{"season": s, **fd.params_row()} for s, fd in runs[STACK_MODEL_ID].k1.items()])
    k1p = k1p[["season", "n_train", "train_seasons", "k1_sigma", "k1_w3", "k1_w7"]]

    def vline(df, cand, metric, base):
        r = df[(df.candidate == cand) & (df.metric == metric) & (df.baseline == base)].iloc[0]
        return f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}], folds won {r['folds_won']}: **{r['verdict']}**"

    pm = pooled_main
    any_beats = (boot["verdict"] == "beats").any()
    beats_list = boot[boot["verdict"] == "beats"]
    beats_txt = ("; ".join(f"{r.candidate} vs {r.baseline} on {r.metric}" for r in beats_list.itertuples())
                 if any_beats else "none")
    runtime = time.time() - t0
    md = [
        "# Step 4 stacking results (stack_v1)\n",
        f"Generated {gen} by `python scripts/run_stacking.py` (code commit `{commit()}` plus uncommitted step-4 "
        f"working tree; runtime {runtime:.0f}s). Experiment `{EXPERIMENT_ID}` (`docs/methodology/experiment_log.md`).\n",
        "**Status: experimental.** Inputs are the step-3 OOF predictions (`data/processed/step3_oof_predictions.parquet`, "
        "test seasons 2015-2024). No 2025 (holdout `nfl_games_2025_v1`) or 2026 game is read, fit or scored.\n",
        "## Design (fixed before scoring)\n",
        "- **Components.** Margin stack over `elo_v1`, `kalman_gs_v1`, `ridge_v1`, `xgb_v1`. Total stack over the models "
        "that predict totals: `elo_v1`, `ridge_v1`, `xgb_v1` (`kalman_gs_v1` is margin-only). `constant_v1` and "
        "`bayes_od_v1` are reference rows; `bayes_od_v1` as a fifth candidate is a separately reported variant.",
        "- **Walk-forward weights.** For test season S, weights are fit on OOF rows of seasons < S whose "
        "prediction time and label-available time are before S's first prediction time "
        "(`distributions.evaluate.training_rows`). One weight vector per season (no weekly refit inside S). "
        "At least 3 training seasons, so the first scored season is 2018 (trained on 2015-2017 OOF rows); "
        f"scored seasons {seasons[0]}-{seasons[-1]}, {len(eval_ids)} games. Every single-model row below is on the same games.",
        "- **Objective.** Primary: squared error on the simplex (weights >= 0, sum 1), no intercept, chosen before any "
        "test-season score was computed. Reported variant: MAE on the simplex (`stack_mae_v1`). Baseline stack: equal "
        "weights (`stack_eq_v1`).",
        "- **Win probability.** Stacked margin -> K1 margin distribution (`fit_distribution(..., \"k1\")`) refit for each "
        "test season on the same training rows, with their stack margin computed from season S's weights (those "
        "weights are in-sample on the K1 training rows; nothing from S or later is used). "
        "`pred_home_win_prob = P(M > 0) / (1 - P(M = 0))`. `stack_eq_v1` uses the same procedure.",
        "- **Clean seasons.** Step 3's clean seasons are 2015, 2022, 2023, 2024 (V5 caveat: ported constants chosen on "
        "2016-2021). 2015 cannot be scored here (no 3 prior OOF seasons), so clean = 2022-2024.",
        f"- **Verdict rule.** \"beats\" only if the paired game-clustered bootstrap 95% CI (B = {args.bootstrap}) of "
        "the loss difference is below 0 AND the candidate wins a majority of folds. Negative difference = candidate "
        "better. \"Best single\" is the component with the lowest pooled value of that metric on these games (chosen "
        "on the test games, which favours the baseline).\n",
        "## Headline\n",
        f"- stack_v1 margin MAE {pm.loc[STACK_MODEL_ID, 'mae_margin']:.3f} vs elo_v1 {pm.loc[BASELINE, 'mae_margin']:.3f}: "
        + vline(boot, STACK_MODEL_ID, "abs_margin", BASELINE) + ".",
        f"- stack_v1 margin vs best single ({best['abs_margin']}, {pm.loc[best['abs_margin'], 'mae_margin']:.3f}): "
        + vline(boot, STACK_MODEL_ID, "abs_margin", best["abs_margin"]) + ".",
        f"- stack_v1 total MAE {pm.loc[STACK_MODEL_ID, 'mae_total']:.3f} vs elo_v1 {pm.loc[BASELINE, 'mae_total']:.3f}: "
        + vline(boot, STACK_MODEL_ID, "abs_total", BASELINE) + f"; vs best single ({best['abs_total']}, "
        f"{pm.loc[best['abs_total'], 'mae_total']:.3f}): " + vline(boot, STACK_MODEL_ID, "abs_total", best["abs_total"]) + ".",
        f"- stack_v1 home-win log loss {pm.loc[STACK_MODEL_ID, 'logloss_win']:.4f} vs elo_v1 "
        f"{pm.loc[BASELINE, 'logloss_win']:.4f}: " + vline(boot, STACK_MODEL_ID, "logloss_win", BASELINE) + ".",
        f"- Equal-weight stack margin MAE {pm.loc[EQUAL_MODEL_ID, 'mae_margin']:.3f}: vs elo_v1 "
        + vline(boot, EQUAL_MODEL_ID, "abs_margin", BASELINE) + ".",
        f"- Comparisons meeting the \"beats\" rule (pooled): {beats_txt}. Gains over the best single model are "
        "small; read the CIs, not the point estimates.\n",
        "## Pooled, scored seasons 2018-2024 (same games)\n",
        md_table(srt(pooled[pooled.model_id.isin(main_ids)])[cols]),
        "\n### Clean seasons only (2022-2024)\n",
        md_table(srt(clean[clean.model_id.isin(main_ids)])[cols]),
        "\n## Paired bootstrap and folds won\n",
        md_table(pd.concat([boot, boot_clean], ignore_index=True)),
        "\n## Per fold\n",
        "Margin MAE:\n", md_table(wide("mae_margin", main_ids)),
        "\nTotal MAE:\n", md_table(wide("mae_total", [m for m in main_ids if m != "kalman_gs_v1"])),
        "\nHome-win log loss:\n", md_table(wide("logloss_win", main_ids), nd=4),
        "\n## Weight path (walk-forward, one vector per test season)\n",
        "Margin (stack_v1 = squared error, stack_mae_v1 = MAE variant):\n", md_table(wt_m),
        "\nTotal:\n", md_table(wt_t),
        "\nForward weights (fit on all OOF seasons 2015-2024; `config/models/stack_v1_weights.json`, used by "
        "`StackModel.from_weights_file` and the dashboard's `stack_weight_*` fields):\n", md_table(fwd),
        f"\nForward K1 for the win-probability mapping: sigma {fw['win_prob']['k1_sigma']:.3f}, w_3 "
        f"{fw['win_prob']['k1_w'][3]:.3f}, w_7 {fw['win_prob']['k1_w'][7]:.3f}.\n",
        "\nK1 parameters of the stack per test season:\n", md_table(k1p),
        "\n## Variants (not primary; reported separately)\n",
        "\n".join(f"- `{k}`: {v}" for k, v in VARIANTS.items()) + "\n",
        md_table(srt(pooled[pooled.model_id.isin(VARIANTS)])[cols]), "\n",
        md_table(boot_var),
        "\n`stack_b5_v1` weight path:\n", md_table(b5.dropna(axis=1, how="all")),
        "\n## Win probability: stack K1 mapping vs the components' own probabilities\n",
        "Non-tie games; ECE with 10 equal-width bins. The last row averages the components' own P(home win) with "
        "stack_v1's margin weights (no distribution step).\n",
        md_table(win_tab, nd=4),
        "\n## Calibration of stack_v1\n",
        f"P(home win) ECE (10 bins) {win_tab.loc[win_tab.model == STACK_MODEL_ID, 'ece_10bin'].iloc[0]:.4f}. "
        f"Home win | no tie from the K1 pmfs: log loss {k1_hw['log_loss']:.4f}, Brier {k1_hw['brier']:.4f}, "
        f"{k1_hw['n_ties_excluded']} ties excluded.\n",
        "\nReliability (P(home win), equal-width bins; `flag_small` = fewer than 30 games):\n",
        md_table(rel_stack, nd=3),
        "\nK1 margin distribution: log score, randomized PIT and central-interval coverage by season "
        "(well calibrated: pit_mean 0.5, pit_var 1/12 = 0.083, covN close to massN):\n",
        md_table(pit.rename(columns={"slice": "season"}), nd=3),
        "\nInterval coverage, pooled:\n", md_table(cov),
        f"\n## Cover calibration at the nflverse spread line ({LINE_LABEL}; context only, never a feature)\n",
        "P(home cover | no push) from each model's own walk-forward K1 distribution at the nflverse line, "
        "pushes excluded, 10 equal-width bins. The line is close-like, so the observed cover rate is near 50% in "
        "every bin; a log loss above ln 2 = 0.693 means worse than a constant 50% forecast.\n",
        md_table(cover_tab, nd=4),
        "\nstack_v1 reliability at the nflverse line:\n", md_table(rel_cover, nd=3),
        "\n## Outputs\n",
        f"- `data/processed/step4_stack_predictions.parquet` (gitignored): {len(stack)} stack_v1 rows, "
        f"sha256 `{pred_sha}`; passes `validate_predictions` (train_end_utc < prediction_time_utc).",
        "- `config/models/stack_v1_weights.json`: forward weights + K1 (fit 2015-2024 OOF rows).",
        "- `docs/results/step4_model_health.json`: MODEL_HEALTH_COLUMNS records, scopes pooled (2018-2024), clean "
        "(2022-2024) and one per fold, for stack_v1, stack_eq_v1 and each component. `cover_logloss` is the "
        "nflverse-line cover log loss above; `stack_weight_*` are the forward weights (null where not a component).",
        "\n## Caveats\n",
        "- Inputs inherit step 3's open items: leakage audit not signed; V5 caveat for 2018-2021 folds (ported "
        "constants chosen on 2016-2021 in the source repo).",
        "- Season-level weights: within-season drift is not tracked. K1 for the stack is fit with in-sample weights "
        "on its training rows (slightly optimistic spread, 4 weights on >= 800 games).",
        "- The components are highly correlated, so weights move between near-equivalent models from season to season "
        "(see the weight path); this is expected and costs little in MAE.",
        "- Real CLV for the stack only exists for games captured going forward; nothing here uses sportsbook odds.",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"wrote {OUT_MD}, {OUT_HEALTH}, {OUT_WEIGHTS}, {OUT_PRED} in {runtime:.0f}s")


if __name__ == "__main__":
    main()
