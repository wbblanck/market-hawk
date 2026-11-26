"""Backtest of an Explainable Boosting Machine (EBM) and NGBoost (exploratory, 2026-11-26).

Questions, on the step-3 walk-forward (test seasons 2015-2024, 2025 never read):

1. EBM as a model: is an EBM on the xgb_v2 feature set as accurate as ridge_v2 / xgb_v2?
2. EBM as a diagnostic: fit on all labelled games <= 2024, do XGBoost's learned relationships (partial
   dependence) agree in sign and size with the EBM's and Ridge's? Is the QB-out effect plausible?
3. NGBoost per-game uncertainty: does a game-specific spread (sigma) improve the margin and total
   distributions over the production layer (margin ``k1``, total ``t``, both on ridge_v2 means)?
   (a) "sigma-only": mean = ridge_v2 OOF prediction, NGBoost models the residual's spread from the features,
       trained on earlier seasons' OOF residuals only;
   (b) "full": NGBoost predicts its own mean and spread from the features.
   Both are discretized with the repo's ``t`` (and ``t+key``) families via ``use_model_sd``, so every candidate
   is scored by the same integer-pmf code (``nfl_market.distributions.evaluate``), seasons 2018-2024.

Feature rows: every game's features are computed at its own week's cutoff (min T of the week, T = kickoff - 24h)
from games labelled before that cutoff, exactly as in the weekly walk-forward. Season-level refits (one fit per
test season, on all games labelled before the season's first T) are used for EBM and NGBoost; an xgb_v2 refit
on the same schedule is included as a control for the refit schedule.

Pre-registered settings (fixed before the first scored run, not tuned):
* EBM: interpret ExplainableBoostingRegressor, xgb_v2 features (XGB_FEATURES + EXTERNAL_COLUMNS),
  interactions=10, outer_bags=8, inner_bags=0, random_state=0, other settings default (interpret 0.7.8).
* NGBoost: Normal distribution, 400 iterations, learning_rate 0.01, default base learner (depth-3 tree),
  minibatch_frac 1.0, natural gradient, random_state 0. Inputs: median imputation with missing indicators.
  sigma-only inputs: xgb_v2 features + ridge_v2 pred_margin, |pred_margin|, pred_total.
* Scoring: integer-pmf log score, PIT, 80% coverage, cover|no-push vs the nflverse line (benchmark only, never a
  feature), and a paired bootstrap of per-game log-score differences clustered by (season, week), 2000 draws.

    python scripts/experiments/ebm_ngboost_backtest.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nfl_market.data.games import load_games, load_nflverse_lines  # noqa: E402
from nfl_market.distributions import calibration as cal  # noqa: E402
from nfl_market.distributions.evaluate import build_pmfs, fit_distribution, training_rows, with_outcomes  # noqa: E402
from nfl_market.features.external import EXTERNAL_COLUMNS, ExternalFeatures  # noqa: E402
from nfl_market.holdout import is_holdout  # noqa: E402
from nfl_market.models.features import FEATURE_COLUMNS, XGB_FEATURES, team_features  # noqa: E402
from nfl_market.models.folds import labelled, prediction_time  # noqa: E402
from nfl_market.models.xgb_model import XGB_PARAMS  # noqa: E402

FEAT_CACHE = ROOT / "data" / "processed" / "ebm_ngb_features.parquet"
OOF = ROOT / "data" / "processed" / "step3_oof_predictions.parquet"
OUT_PRED = ROOT / "data" / "processed" / "ebm_ngb_predictions.parquet"
OUT_MD = ROOT / "docs" / "results" / "experiments" / "ebm_ngboost_backtest.md"
TEST_SEASONS = range(2015, 2025)
SCORE_FROM = 2018
EBM_FEATURES = tuple(XGB_FEATURES) + tuple(EXTERNAL_COLUMNS)
ALL_FEATURES = tuple(dict.fromkeys(tuple(FEATURE_COLUMNS) + EBM_FEATURES))
EBM_PARAMS = dict(interactions=10, outer_bags=8, inner_bags=0, random_state=0, n_jobs=4)
NGB_PARAMS = dict(n_estimators=400, learning_rate=0.01, minibatch_frac=1.0, natural_gradient=True,
                  random_state=0, verbose=False)


# ------------------------------------------------------------------ features
def build_features(games: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time features for every game of seasons first+1..2024, computed at its week's cutoff."""
    lab = labelled(games)
    T = prediction_time(games)
    first = int(games["season"].min())
    rows = games[(games["season"] > first) & (games["season"] <= 2024)]
    out = []
    for (s, w), wk in rows.groupby(["season", "week"], sort=True):
        cut = T.loc[wk.index].min()
        hist = games[lab & (games["label_available_at_utc"] < cut)]
        f = team_features(hist, wk)
        try:
            f = f.join(ExternalFeatures(hist).features(wk))
        except Exception as exc:  # seasons before play-by-play coverage
            print(f"external features {s} wk {w}: {exc}")
            for c in EXTERNAL_COLUMNS:
                f[c] = np.nan
        f.insert(0, "game_id", wk["game_id"].to_numpy())
        f["season"], f["week_no"], f["T"] = int(s), int(w), cut
        out.append(f)
    feat = pd.concat(out, ignore_index=True)
    assert not is_holdout(feat["game_id"]).any()
    return feat


def attach(feat: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    g = games[["game_id", "home_score", "away_score", "label_available_at_utc", "kickoff_utc"]]
    d = feat.merge(g, on="game_id", how="left")
    d["margin"] = d["home_score"] - d["away_score"]
    d["total"] = d["home_score"] + d["away_score"]
    return d


def season_split(d: pd.DataFrame, s: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    te = d[d["season"] == s]
    cut = te["T"].min()
    tr = d[(d["label_available_at_utc"] < cut) & d["margin"].notna()]
    assert tr["label_available_at_utc"].max() < cut
    return tr, te


# -------------------------------------------------------------------- models
def _impute(tr: pd.DataFrame, te: pd.DataFrame, cols) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.impute import SimpleImputer

    imp = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    return imp.fit_transform(tr[list(cols)]), imp.transform(te[list(cols)])


def fit_ebm(X, y):
    from interpret.glassbox import ExplainableBoostingRegressor

    return ExplainableBoostingRegressor(feature_names=list(EBM_FEATURES), **EBM_PARAMS).fit(X, y)


def fit_xgb(X, y):
    from xgboost import XGBRegressor

    return XGBRegressor(**XGB_PARAMS).fit(X, y)


def fit_ngb(X, y):
    from ngboost import NGBRegressor
    from ngboost.distns import Normal

    return NGBRegressor(Dist=Normal, **NGB_PARAMS).fit(X, y)


def ngb_predict(m, X) -> tuple[np.ndarray, np.ndarray]:
    p = m.pred_dist(X).params
    return np.asarray(p["loc"], float), np.asarray(p["scale"], float)


def season_models(d: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """EBM, xgb (season refit) and NGBoost full / sigma-only predictions for test seasons 2015-2024."""
    rv2 = oof[oof["model_id"] == "ridge_v2"].set_index("game_id")
    out = []
    for s in TEST_SEASONS:
        t0 = time.time()
        tr, te = season_split(d, s)
        rec = te[["game_id", "season", "week_no"]].copy()
        for tgt in ("margin", "total"):
            y = tr[tgt].to_numpy(float)
            Xtr, Xte = tr[list(EBM_FEATURES)], te[list(EBM_FEATURES)]
            rec[f"ebm_{tgt}"] = fit_ebm(Xtr, y).predict(Xte)
            rec[f"xgbS_{tgt}"] = fit_xgb(Xtr, y).predict(Xte)
            A, B = _impute(tr, te, EBM_FEATURES)
            mu, sd = ngb_predict(fit_ngb(A, y), B)
            rec[f"ngb_{tgt}"], rec[f"ngb_{tgt}_sd"] = mu, sd
            # sigma-only: residual of ridge_v2 OOF, trained on earlier test seasons' OOF rows
            otr = tr[tr["game_id"].isin(rv2.index) & (tr["season"] >= TEST_SEASONS[0])]
            if otr["season"].nunique() >= 1 and len(otr) > 200:
                cols = list(EBM_FEATURES) + ["rv2_margin", "rv2_abs", "rv2_total"]
                a, b = otr.copy(), te.copy()
                for z in (a, b):
                    z["rv2_margin"] = z["game_id"].map(rv2["pred_margin"])
                    z["rv2_total"] = z["game_id"].map(rv2["pred_total"])
                    z["rv2_abs"] = z["rv2_margin"].abs()
                resid = a[tgt].to_numpy(float) - a[f"rv2_{tgt}"].to_numpy(float)
                A2, B2 = _impute(a, b, cols)
                _, sd2 = ngb_predict(fit_ngb(A2, resid), B2)
                rec[f"ngbsig_{tgt}_sd"] = sd2
            else:
                rec[f"ngbsig_{tgt}_sd"] = np.nan
        out.append(rec)
        print(f"season {s}: {len(te)} games, {time.time() - t0:.0f}s", flush=True)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- scoring
def block_boot(diff: np.ndarray, blocks: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of per-game differences with a (season, week)-clustered bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    ub, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    cnts = np.bincount(inv)
    draws = rng.integers(0, len(ub), size=(n, len(ub)))
    means = sums[draws].sum(1) / cnts[draws].sum(1)
    return float(diff.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def oof_frame(base: pd.DataFrame, pred: pd.Series, sd: pd.Series | None, tgt: str) -> pd.DataFrame:
    f = base.copy()
    f[f"pred_{tgt}"] = f["game_id"].map(pred)
    if sd is not None:
        f[f"pred_{tgt}_sd"] = f["game_id"].map(sd)
    return f


def score_family(frame: pd.DataFrame, family: str, tgt: str, use_sd: bool, ref: pd.Series) -> dict:
    """Walk-forward fit (earlier seasons' OOF rows) and score 2018-2024; returns per-game arrays."""
    frame = with_outcomes(frame).dropna(subset=[f"pred_{tgt}", tgt]).sort_values("prediction_time_utc")
    ls, pit_y, mats, ys, refs, gids, blocks = [], [], [], [], [], [], []
    for s in range(SCORE_FROM, 2025):
        tr = training_rows(frame, s)
        te = frame[frame["season"] == s]
        if use_sd:
            te = te.dropna(subset=[f"pred_{tgt}_sd"])
        fd = fit_distribution(tr, family, target=tgt, use_model_sd=use_sd)
        lo, mat = build_pmfs(fd, te)
        mats.append((lo, mat))
        ys.append(te[tgt].to_numpy(float))
        refs.append(te["game_id"].map(ref).to_numpy(float))
        gids.append(te["game_id"].to_numpy())
        blocks.append((te["season"].astype(str) + "_" + te["week"].astype(str)).to_numpy())
    lo = min(m[0] for m in mats)
    hi = max(m[0] + m[1].shape[1] - 1 for m in mats)
    mat = np.vstack([np.pad(m, ((0, 0), (l - lo, hi - (l + m.shape[1] - 1)))) for l, m in mats])
    y, r = np.concatenate(ys), np.concatenate(refs)
    return {"pmfs": (lo, mat), "y": y, "ref": r, "game_id": np.concatenate(gids), "block": np.concatenate(blocks),
            "ls": cal.log_scores((lo, mat), y)}


def summarize(name: str, res: dict, tgt: str) -> dict:
    lo, mat = res["pmfs"]
    y, ref = res["y"], res["ref"]
    x = np.arange(lo, lo + mat.shape[1])
    mean = mat @ x
    pit = cal.pit_summary(cal.randomized_pit((lo, mat), y))
    cov = cal.interval_coverage((lo, mat), y)
    c80 = float(cov.loc[np.isclose(cov["level"], 0.8), "coverage"].iloc[0])
    ok = np.isfinite(ref)
    # nflverse spread_line is home-perspective (home covers when margin > spread_line); total: over when > line
    thr = ref[ok]
    m2 = mat[ok]
    above = (m2 * (x[None, :] > thr[:, None])).sum(1)
    push = (m2 * (x[None, :] == thr[:, None])).sum(1)
    p = above / np.clip(1 - push, 1e-9, None)
    yy = y[ok]
    win = (yy > thr).astype(float)
    np_ = yy != thr
    brier = float(np.mean((p[np_] - win[np_]) ** 2))
    pick = np_ & (np.abs(p - 0.5) > 0.024)
    hit = float(np.mean((p[pick] > 0.5) == (win[pick] == 1))) if pick.any() else np.nan
    return {"candidate": name, "n": len(y), "mae_mean": float(np.mean(np.abs(y - mean))),
            "log_score": float(res["ls"].mean()), "pit_ks": pit.get("pit_ks", np.nan), "cov80": c80,
            "line_brier": brier, "p_line_sd": float(np.std(p)), "n_bets_52.4": int(pick.sum()), "hit_rate": hit,
            "pred_push": float(push[np_ | ~np_].mean()), "obs_push": float(np.mean(~np_))}


# -------------------------------------------------------------- diagnostic
def pdp(predict, X: pd.DataFrame, col: str, grid: np.ndarray, n: int = 1500, seed: int = 0) -> np.ndarray:
    Xs = X.sample(min(n, len(X)), random_state=seed)
    vals = []
    for v in grid:
        Z = Xs.copy()
        Z[col] = v
        vals.append(float(np.mean(predict(Z))))
    return np.array(vals)


def diagnostic(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """EBM vs XGB vs Ridge partial dependence on all labelled games <= 2024 (margin and total)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr = d[d["margin"].notna() & (d["season"] <= 2024) & d["home_off_epa_l8"].notna()]
    X = tr[list(EBM_FEATURES)]
    imp_rows, rows = [], []
    for tgt in ("margin", "total"):
        y = tr[tgt].to_numpy(float)
        ebm = fit_ebm(X, y)
        xgb = fit_xgb(X, y)
        rid = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                            StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30))).fit(X, y)
        imp = pd.Series(ebm.term_importances(), index=ebm.term_names_).sort_values(ascending=False)
        for k, v in imp.head(12).items():
            imp_rows.append({"target": tgt, "term": k, "ebm_importance": float(v)})
        mains = [t for t in imp.index if t in EBM_FEATURES][:8]
        for c in dict.fromkeys(mains + ["home_qb_out", "away_qb_out"]):
            xs = X[c].dropna()
            if xs.nunique() <= 2:
                grid = np.array([0.0, 1.0])
                labels = ["0", "1"]
            else:
                grid = np.quantile(xs, [0.1, 0.5, 0.9])
                labels = ["p10", "p50", "p90"]
            e, g, r = (pdp(m.predict, X, c, grid) for m in (ebm, xgb, rid))
            e, g, r = e - e[len(e) // 2 if len(e) > 2 else 0], g - g[len(g) // 2 if len(g) > 2 else 0], \
                r - r[len(r) // 2 if len(r) > 2 else 0]
            row = {"target": tgt, "feature": c, "grid": " / ".join(f"{lab}={v:.3g}" for lab, v in zip(labels, grid))}
            span = lambda a: float(a[-1] - a[0])  # noqa: E731
            row.update(ebm_effect=span(e), xgb_effect=span(g), ridge_effect=span(r),
                       sign_agree=bool(np.sign(span(e)) == np.sign(span(g)) == np.sign(span(r))))
            rows.append(row)
        # the QB-out scenario as the features encode it: flag = 1 and qb_epa = replacement level (-0.15),
        # versus the same games with the flag = 0 and the starter's own qb_epa
        Xs = X[X["home_qb_out"] == 0].sample(1500, random_state=0)
        Z = Xs.copy()
        Z["home_qb_out"], Z["home_qb_epa"] = 1.0, -0.15
        for name, m in (("ebm", ebm), ("xgb", xgb), ("ridge", rid)):
            eff = float(np.mean(m.predict(Z)) - np.mean(m.predict(Xs)))
            rows.append({"target": tgt, "feature": f"home QB out scenario ({name})", "grid": "flag 1, qb_epa -0.15",
                         f"{name}_effect": eff})
    return pd.DataFrame(imp_rows), pd.DataFrame(rows)


# ---------------------------------------------------------------- report
def md(df: pd.DataFrame, nd: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else f"{v:.{nd}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--skip-models", action="store_true", help="reuse saved season-model predictions")
    a = ap.parse_args()
    t_start = time.time()
    games = load_games()
    games = games[games["season"] <= 2024]
    if FEAT_CACHE.exists() and not a.rebuild_features:
        feat = pd.read_parquet(FEAT_CACHE)
    else:
        feat = build_features(games)
        feat.to_parquet(FEAT_CACHE, index=False)
    d = attach(feat, games)
    oof = pd.read_parquet(OOF)
    assert not is_holdout(oof["game_id"]).any()
    if OUT_PRED.exists() and a.skip_models:
        sm = pd.read_parquet(OUT_PRED)
    else:
        sm = season_models(d, oof)
        sm.to_parquet(OUT_PRED, index=False)

    lines = load_nflverse_lines().set_index("game_id")
    spread_ref, total_ref = lines["spread_line"], lines["total_line"]
    out = d.set_index("game_id")
    report = {}

    # 1. point accuracy, 2015-2024 and 2018-2024
    acc_rows, boots = [], []
    for tgt in ("margin", "total"):
        y = out[tgt]
        cands = {"ridge_v2 (weekly)": oof[oof.model_id == "ridge_v2"].set_index("game_id")[f"pred_{tgt}"],
                 "xgb_v2 (weekly)": oof[oof.model_id == "xgb_v2"].set_index("game_id")[f"pred_{tgt}"],
                 "xgb_v2 settings (season refit)": sm.set_index("game_id")[f"xgbS_{tgt}"],
                 "EBM (season refit)": sm.set_index("game_id")[f"ebm_{tgt}"],
                 "NGBoost mean (season refit)": sm.set_index("game_id")[f"ngb_{tgt}"],
                 "nflverse line (benchmark)": (spread_ref if tgt == "margin" else total_ref)}
        common = sm["game_id"]
        for name, p in cands.items():
            e = (y.reindex(common) - p.reindex(common)).abs().to_numpy()
            acc_rows.append({"target": tgt, "model": name, "n": int(np.isfinite(e).sum()), "mae": float(np.nanmean(e))})
        base = (y.reindex(common) - cands["ridge_v2 (weekly)"].reindex(common)).abs().to_numpy()
        blk = (sm["season"].astype(str) + "_" + sm["week_no"].astype(str)).to_numpy()
        for name in ("EBM (season refit)", "xgb_v2 settings (season refit)", "NGBoost mean (season refit)"):
            e = (y.reindex(common) - cands[name].reindex(common)).abs().to_numpy()
            m, lo_, hi_ = block_boot(e - base, blk)
            boots.append({"target": tgt, "candidate - ridge_v2": name, "mae_diff": m, "ci_lo": lo_, "ci_hi": hi_})
    acc, boot = pd.DataFrame(acc_rows), pd.DataFrame(boots)

    # 2. distributions (2018-2024)
    rv2 = oof[oof.model_id == "ridge_v2"].merge(
        games[["game_id", "home_score", "away_score", "label_available_at_utc"]], on="game_id", how="left")
    smi = sm.set_index("game_id")
    dist_rows, dboot = [], []
    for tgt, ref, prod in (("margin", spread_ref, "k1"), ("total", total_ref, "t")):
        rv2_pred = rv2.set_index("game_id")[f"pred_{tgt}"]
        variants = {
            f"production: ridge_v2 + {prod}": (rv2, prod, False),
            "ridge_v2 + t (constant spread)": (rv2, "t", False),
            "ridge_v2 + t, NGBoost sigma": (oof_frame(rv2, rv2_pred, smi[f"ngbsig_{tgt}_sd"], tgt), "t", True),
            "NGBoost full (mean + sigma), t": (oof_frame(rv2, smi[f"ngb_{tgt}"], smi[f"ngb_{tgt}_sd"], tgt), "t", True),
        }
        if tgt == "margin":
            variants["ridge_v2 + t+key (constant spread)"] = (rv2, "t+key", False)
            variants["ridge_v2 + t+key, NGBoost sigma"] = (oof_frame(rv2, rv2_pred, smi["ngbsig_margin_sd"], tgt),
                                                           "t+key", True)
        res = {k: score_family(f, fam, tgt, use_sd, ref) for k, (f, fam, use_sd) in variants.items()}
        ids = set.intersection(*(set(r["game_id"]) for r in res.values()))
        for k, r in res.items():
            keep = np.isin(r["game_id"], list(ids))
            r2 = {**r, "pmfs": (r["pmfs"][0], r["pmfs"][1][keep]), "y": r["y"][keep], "ref": r["ref"][keep],
                  "game_id": r["game_id"][keep], "block": r["block"][keep], "ls": r["ls"][keep]}
            order = np.argsort(r2["game_id"])
            res[k] = {kk: (vv[order] if isinstance(vv, np.ndarray) else (vv[0], vv[1][order]) if kk == "pmfs" else vv)
                      for kk, vv in r2.items()}
            dist_rows.append({"target": tgt, **summarize(k, res[k], tgt)})
        base_k = f"production: ridge_v2 + {prod}"
        for k in res:
            if k == base_k:
                continue
            m, lo_, hi_ = block_boot(res[base_k]["ls"] - res[k]["ls"], res[k]["block"])
            dboot.append({"target": tgt, "candidate": k, "log_score_gain_vs_production": m, "ci_lo": lo_, "ci_hi": hi_})
        # how much does sigma vary?
        sd = smi[f"ngbsig_{tgt}_sd"].reindex(list(ids)).dropna()
        sdf = smi[f"ngb_{tgt}_sd"].reindex(list(ids)).dropna()
        report[f"sigma_{tgt}"] = {"sigma_only_p10_p50_p90": [round(float(q), 2) for q in sd.quantile([.1, .5, .9])],
                                  "full_p10_p50_p90": [round(float(q), 2) for q in sdf.quantile([.1, .5, .9])]}
        r_abs = np.abs(out.loc[sd.index, tgt] - rv2_pred.reindex(sd.index))
        report[f"sigma_{tgt}"]["corr_sigma_abs_resid"] = round(float(np.corrcoef(sd, r_abs)[0, 1]), 3)
    dist, dboot = pd.DataFrame(dist_rows), pd.DataFrame(dboot)

    # 3. diagnostic
    imp, shapes = diagnostic(d)

    report["runtime_s"] = round(time.time() - t_start, 1)
    print(json.dumps(report, indent=1))
    print(acc.to_string()); print(boot.to_string()); print(dist.to_string()); print(dboot.to_string())
    print(imp.to_string()); print(shapes.to_string())
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    tables = {"accuracy": acc, "accuracy_boot": boot, "distributions": dist, "distribution_boot": dboot,
              "ebm_importance": imp, "shapes": shapes}
    (ROOT / "data" / "processed" / "ebm_ngb_tables.json").write_text(
        json.dumps({k: v.to_dict(orient="records") for k, v in tables.items()} | {"report": report}, default=str))
    return tables, report


if __name__ == "__main__":
    main()
