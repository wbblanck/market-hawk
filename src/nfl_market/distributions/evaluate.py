"""Season walk-forward evaluation of outcome distributions on OOF game predictions.

For each test season ``S`` in the prediction frame (per ``model_id``), the
distribution parameters (residual pool, t scale/df, key-number weights, NB
size/frailty) are fit **only** on rows of earlier seasons whose
``prediction_time_utc`` (and ``label_available_at_utc`` if present) is before
the first prediction time of season ``S``. Season ``S`` is then scored once.

Input: a frame satisfying ``nfl_market.models.interface.PREDICTION_COLUMNS``
plus outcome columns ``home_score`` and ``away_score`` (or ``margin``/``total``).
Holdout rows (2025) are always dropped via :func:`nfl_market.holdout.exclude_holdout`;
seasons after ``max_season`` (default 2024, i.e. the 2026 forward season) are dropped too.

Reference lines: bucket/push/cover tables use ``ref_col`` (a home spread, home's
own line) if given, otherwise the model's own implied home line
``-round(pred_margin * 2) / 2``. Market lines are never read from nflverse.

CLI::

    python -m nfl_market.distributions.evaluate \
        --predictions data/processed/step3_oof_predictions.parquet \
        --out docs/results/step3_distributions.md
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..holdout import exclude_holdout
from ..models.interface import PREDICTION_COLUMNS
from . import calibration as cal
from .margin import (DEFAULT_MARGIN_KEYS, MARGIN_SUPPORT, KernelLocalEmpirical, KeyNumberWeights, ResidualPool,
                     TotalScaledPool, TParams, apply_key_weights_matrix, choose_kernel_h, fit_key_weights,
                     fit_keynum, fit_student_t, stack_pmfs, t_pmf_matrix)
from . import scoring_models as SM
from .pmf import JointPMF
from .team_points import NBParams, fit_nb_team_points, nb_joint_pmf, team_means
from .totals import TOTAL_SUPPORT, total_pmf_empirical

#: Distribution families. ``+key`` (margins only) adds the in-fold key-number adjustment.
FAMILIES = ("empirical", "t", "normal", "kernel", "kernel_sym", "keynum", "shifted_total", "nb",
            "k1", "k1_frozen", "s1", "d2")
_MARGIN_ONLY = {"kernel_sym", "keynum", "shifted_total", "k1", "k1_frozen"}
#: Families that need pred_total (team-level models).
_NEEDS_TOTAL = {"nb", "shifted_total", "s1", "d2"}
#: Families whose parameters are frozen constants fit on 2016-2020 in nfl-betting-model (in-sample there).
FROZEN_FAMILIES = {"k1_frozen", "s1", "d2"}


def parse_family(family: str) -> tuple[str, bool]:
    """``'t+key'`` -> ``('t', True)``."""
    base, _, suffix = family.partition("+")
    if base not in FAMILIES or suffix not in ("", "key") or (base == "keynum" and suffix):
        raise ValueError(f"family must be one of {FAMILIES} optionally with '+key', got {family!r}")
    return base, suffix == "key"


def with_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``margin``/``total`` from ``home_score``/``away_score`` if missing."""
    df = frame.copy()
    if "margin" not in df.columns or "total" not in df.columns:
        if not {"home_score", "away_score"} <= set(df.columns):
            raise ValueError("predictions need home_score/away_score (or margin/total) outcome columns")
        df["margin"] = df["home_score"] - df["away_score"]
        df["total"] = df["home_score"] + df["away_score"]
    return df


# --------------------------------------------------------------- fitting
@dataclass
class FittedDistribution:
    family: str
    target: str
    key_adjust: bool
    n_train: int
    train_seasons: tuple[int, ...]
    pool: ResidualPool | None = None
    t_params: TParams | None = None
    nb_params: NBParams | None = None
    kernel: KernelLocalEmpirical | None = None
    total_scaled: TotalScaledPool | None = None
    k1_params: SM.K1Params | None = None
    key_weights: KeyNumberWeights | None = None
    key_preserve_moments: bool = True
    condition_on_bucket: bool = False
    use_model_sd: bool = False

    def params_row(self) -> dict:
        row = {"family": self.family + ("+key" if self.key_adjust else ""), "target": self.target,
               "n_train": self.n_train,
               "train_seasons": f"{min(self.train_seasons)}-{max(self.train_seasons)}" if self.train_seasons else ""}
        if self.pool is not None:
            row.update(pool_n=len(self.pool), pool_mean=float(self.pool.residuals.mean()),
                       pool_sd=float(self.pool.residuals.std()))
        if self.t_params is not None:
            row.update(t_scale=self.t_params.scale, t_df=self.t_params.df, t_loc=self.t_params.loc)
        if self.kernel is not None:
            row.update(kernel_h=self.kernel.h, kernel_n=int(self.kernel.pred.size))
        if self.total_scaled is not None:
            row.update(ts_alpha=self.total_scaled.alpha, ts_beta=self.total_scaled.beta,
                       ts_sigma=self.total_scaled.sigma)
        if self.nb_params is not None:
            row.update(nb_r=self.nb_params.r, nb_theta=self.nb_params.theta)
        if self.k1_params is not None:
            row.update(k1_sigma=self.k1_params.sigma, k1_w3=self.k1_params.w[3], k1_w7=self.k1_params.w[7])
        if self.family in FROZEN_FAMILIES:
            row["frozen_constants_fit"] = "2016-2020 (nfl-betting-model)"
        if self.key_weights is not None:
            for k, w in zip(self.key_weights.keys, self.key_weights.weights["all"]):
                if k in (0, 3, 4, 6, 7, 10, 14):
                    row[f"w_key_{k}"] = float(w)
        return row


def _base_pmfs(fd: FittedDistribution, frame: pd.DataFrame) -> tuple[int, np.ndarray]:
    pred = frame[f"pred_{fd.target}"].to_numpy(float)
    sd_col = f"pred_{fd.target}_sd"
    sds = frame[sd_col].to_numpy(float) if (fd.use_model_sd and sd_col in frame) else None
    if fd.family == "empirical":
        if fd.target == "margin":
            return stack_pmfs([fd.pool.pmf(m, condition_on_bucket=fd.condition_on_bucket) for m in pred])
        return stack_pmfs([total_pmf_empirical(m, fd.pool, condition_on_bucket=fd.condition_on_bucket)
                           for m in pred])
    if fd.family in ("t", "normal", "keynum"):
        lo, mat = t_pmf_matrix(pred, fd.t_params, sds=sds,
                               support=MARGIN_SUPPORT if fd.target == "margin" else TOTAL_SUPPORT)
        if fd.target == "total":  # drop impossible totals (T = 1)
            mat[:, 1 - lo] = 0.0
            mat = mat / mat.sum(axis=1, keepdims=True)
        return lo, mat
    if fd.family in ("kernel", "kernel_sym"):
        return fd.kernel.pmf_matrix(pred)
    if fd.family in ("k1", "k1_frozen"):
        p = fd.k1_params
        return int(SM.K1_M[0]), SM.k1_matrix(pred, p.sigma, p.w)
    if fd.family in ("s1", "d2"):
        d, L = SM.tdfg_inputs(frame["pred_margin"].to_numpy(float), frame["pred_total"].to_numpy(float))
        if fd.family == "d2":
            joints = [SM.d2_joint(di, li) for di, li in zip(d, L)]
        else:
            joints = [JointPMF(J) for J in SM.s1_joint_matrices(d, L)]
        return stack_pmfs([j.margin() if fd.target == "margin" else j.total() for j in joints])
    if fd.family == "shifted_total":
        return stack_pmfs([fd.total_scaled.pmf(m, t) for m, t in zip(pred, frame["pred_total"].to_numpy(float))])
    mh, ma = team_means(frame["pred_margin"], frame["pred_total"])
    joints = [nb_joint_pmf(h, a, fd.nb_params) for h, a in zip(mh, ma)]
    return stack_pmfs([j.margin() if fd.target == "margin" else j.total() for j in joints])


def build_pmfs(fd: FittedDistribution, frame: pd.DataFrame) -> tuple[int, np.ndarray]:
    """Stacked pmfs ``(lo, matrix)`` for each row of ``frame`` under a fitted distribution."""
    lo, mat = _base_pmfs(fd, frame)
    if fd.key_weights is not None:
        mat = apply_key_weights_matrix(lo, mat, fd.key_weights, ref=frame[f"pred_{fd.target}"].to_numpy(float),
                                       preserve_mean=fd.key_preserve_moments,
                                       preserve_var=fd.key_preserve_moments)
    return lo, mat


def fit_distribution(train: pd.DataFrame, family: str = "t", *, target: str = "margin",
                     condition_on_bucket: bool = False, use_model_sd: bool = False,
                     keys: Sequence[int] = DEFAULT_MARGIN_KEYS, key_by_bucket: bool = False,
                     kernel_h: float | None = None, max_key_fit_rows: int = 4000) -> FittedDistribution:
    """Fit a distribution family on training-fold OOF rows (with outcomes) only.

    ``kernel_h=None`` chooses the kernel bandwidth with :func:`choose_kernel_h`
    on the training rows (nfl_ats rule).
    """
    base, key_adjust = parse_family(family)
    if target not in ("margin", "total"):
        raise ValueError("target must be 'margin' or 'total'")
    if target != "margin" and (key_adjust or base in _MARGIN_ONLY):
        raise ValueError(f"family {family!r} is implemented for margins only")
    train = with_outcomes(train)
    pred_col = f"pred_{target}"
    need = [pred_col, target] + (["pred_margin", "pred_total"] if base in _NEEDS_TOTAL else [])
    train = train.dropna(subset=list(dict.fromkeys(need)))
    if train.empty:
        raise ValueError("no usable training rows")
    pred = train[pred_col].to_numpy(float)
    y = train[target].to_numpy(float)
    resid = y - pred
    fd = FittedDistribution(family=base, target=target, key_adjust=key_adjust, n_train=len(train),
                            train_seasons=tuple(sorted(int(s) for s in train["season"].unique())),
                            condition_on_bucket=condition_on_bucket, use_model_sd=use_model_sd)
    if base == "empirical":
        fd.pool = ResidualPool(resid, ref=pred, target=target)
    elif base == "t":
        fd.t_params = fit_student_t(resid)
    elif base == "normal":
        fd.t_params = TParams(scale=float(np.std(resid)), df=np.inf)
    elif base in ("kernel", "kernel_sym"):
        mirrored = base == "kernel_sym"
        h = kernel_h if kernel_h is not None else choose_kernel_h(pred, y, train["season"].to_numpy(), mirrored=mirrored)
        fd.kernel = KernelLocalEmpirical(pred, y, h=h, mirrored=mirrored)
    elif base == "keynum":
        fd.t_params, fd.key_weights = fit_keynum(pred, y)
        fd.key_preserve_moments = False
    elif base == "k1":
        fd.k1_params = SM.fit_k1(pred, y)
    elif base == "k1_frozen":
        fd.k1_params = SM.K1_FROZEN
    elif base in ("s1", "d2"):
        pass  # frozen constants (SM.D2_CONSTANTS, SM.S1_C0)
    elif base == "shifted_total":
        fd.total_scaled = TotalScaledPool.fit(pred, y, train["pred_total"].to_numpy(float))
    else:
        mh, ma = team_means(train["pred_margin"], train["pred_total"])
        fd.nb_params = fit_nb_team_points(mh, ma, train["home_score"], train["away_score"])
    if key_adjust:
        kt = train.sort_values("prediction_time_utc").tail(max_key_fit_rows)
        lo, mat = _base_pmfs(fd, kt)
        fd.key_weights = fit_key_weights((lo, mat), kt[target].to_numpy(float), keys=keys,
                                         ref=kt[pred_col].to_numpy(float), by_bucket=key_by_bucket)
    return fd


def training_rows(frame: pd.DataFrame, test_season: int) -> pd.DataFrame:
    """Rows usable to fit parameters for ``test_season``: earlier seasons, strictly before its first prediction."""
    test = frame[frame["season"] == test_season]
    cutoff = pd.to_datetime(test["prediction_time_utc"], utc=True).min()
    tr = frame[frame["season"] < test_season]
    keep = pd.to_datetime(tr["prediction_time_utc"], utc=True) < cutoff
    if "label_available_at_utc" in tr.columns:
        keep &= pd.to_datetime(tr["label_available_at_utc"], utc=True) < cutoff
    return tr[keep]


# ------------------------------------------------------------ evaluation
def _home_ref(frame: pd.DataFrame, ref_col: str | None) -> np.ndarray:
    if ref_col is not None:
        return frame[ref_col].to_numpy(float)
    return -np.round(frame["pred_margin"].to_numpy(float) * 2) / 2


def evaluate_distributions(predictions_with_outcomes: pd.DataFrame, family: str = "t", *,
                           target: str = "margin", min_history_seasons: int = 3,
                           model_id: str | None = None, ref_col: str | None = None,
                           condition_on_bucket: bool = False, use_model_sd: bool = False,
                           key_by_bucket: bool = False, max_season: int | None = 2024,
                           keys_report: Sequence[int] = (0, 3, 7, 10), seed: int = 0) -> dict[str, pd.DataFrame]:
    """Walk-forward fit-and-score of one distribution family. Returns tidy tables.

    Keys: ``summary`` (log score, PIT, coverage per season and ``all``), ``params``
    (fitted parameters per season), ``coverage``, ``pit_hist``, ``key_numbers``
    (margin only), ``binary`` (home win | no tie and cover | no push, or over | no
    push for totals), ``reliability``, ``push``.
    """
    missing = [c for c in PREDICTION_COLUMNS if c not in predictions_with_outcomes.columns]
    if missing:
        raise ValueError(f"prediction frame missing columns: {missing}")
    df = with_outcomes(exclude_holdout(predictions_with_outcomes))
    if max_season is not None:
        df = df[df["season"] <= max_season]
    df = df.dropna(subset=["margin", "total"])
    if model_id is not None:
        df = df[df["model_id"] == model_id]
    fam_label = family
    out: dict[str, list] = {k: [] for k in ("summary", "params", "coverage", "pit_hist", "key_numbers",
                                             "binary", "reliability", "push")}
    for mid, mdf in df.groupby("model_id", sort=True):
        mdf = mdf.sort_values("prediction_time_utc")
        seasons = sorted(int(s) for s in mdf["season"].unique())
        mats, ys, refs, season_tags = [], [], [], []
        for s in seasons:
            tr = training_rows(mdf, s)
            if tr["season"].nunique() < min_history_seasons:
                continue
            te = mdf[mdf["season"] == s]
            if target == "total" or parse_family(family)[0] in _NEEDS_TOTAL:
                te = te.dropna(subset=["pred_total"])
            if te.empty:
                continue
            fd = fit_distribution(tr, family, target=target, condition_on_bucket=condition_on_bucket,
                                  use_model_sd=use_model_sd, key_by_bucket=key_by_bucket)
            out["params"].append({"model_id": mid, "season": s, **fd.params_row()})
            lo, mat = build_pmfs(fd, te)
            mats.append((lo, mat)); ys.append(te[target].to_numpy(float))
            refs.append(_home_ref(te, ref_col) if target == "margin"
                        else np.round(te["pred_total"].to_numpy(float) * 2) / 2)
            season_tags.append(np.full(len(te), s))
        if not mats:
            continue
        lo = min(m[0] for m in mats)
        hi = max(m[0] + m[1].shape[1] - 1 for m in mats)
        mat = np.vstack([np.pad(m, ((0, 0), (l - lo, hi - (l + m.shape[1] - 1)))) for l, m in mats])
        y, ref, stag = np.concatenate(ys), np.concatenate(refs), np.concatenate(season_tags)
        tag = {"model_id": mid, "family": fam_label, "target": target}
        summ = cal.by_slice((lo, mat), y, stag, seed=seed)
        x = np.arange(lo, lo + mat.shape[1])
        for i, row in summ.iterrows():
            sel = np.ones(len(y), bool) if row["slice"] == "all" else stag == row["slice"]
            extra = {"mean_pred": float((mat[sel] @ x).mean()), "mean_obs": float(y[sel].mean())}
            if target == "margin":
                extra.update(pred_tie=float(mat[sel][:, x == 0].sum(axis=1).mean()), obs_tie=float(np.mean(y[sel] == 0)))
            out["summary"].append({**tag, "season": row["slice"], **row.drop("slice").to_dict(), **extra})
        cov = cal.interval_coverage((lo, mat), y)
        out["coverage"].append(cov.assign(**tag))
        out["pit_hist"].append(cal.pit_histogram(cal.randomized_pit((lo, mat), y, seed=seed)).assign(**tag))
        if target == "margin":
            kn = cal.key_number_table((lo, mat), y, ref, keys=keys_report).assign(season="all")
            per_season = [cal.key_number_table((lo, mat[stag == s]), y[stag == s], None, keys=keys_report)
                          .assign(season=int(s)) for s in np.unique(stag)]
            out["key_numbers"].append(pd.concat([kn, *per_season]).assign(**tag))
            hw, hw_rel = cal.home_win_calibration((lo, mat), y)
            cv, cv_rel = cal.cover_calibration((lo, mat), y, ref)
            out["binary"] += [{**tag, "market": "home_win|no_tie", **hw}, {**tag, "market": "home_cover|no_push", **cv}]
            out["reliability"] += [hw_rel.assign(market="home_win|no_tie", **tag), cv_rel.assign(market="home_cover|no_push", **tag)]
            out["push"].append(cal.push_calibration((lo, mat), y, ref, kind="spread").assign(**tag))
        else:
            ov, ov_rel = cal.cover_calibration((lo, mat), y, -ref)
            out["binary"].append({**tag, "market": "over|no_push", **ov})
            out["reliability"].append(ov_rel.assign(market="over|no_push", **tag))
            out["push"].append(cal.push_calibration((lo, mat), y, ref, kind="total").assign(**tag))
        for s in np.unique(stag):
            sel = stag == s
            if target == "margin":
                hw, _ = cal.home_win_calibration((lo, mat[sel]), y[sel])
                cv, _ = cal.cover_calibration((lo, mat[sel]), y[sel], ref[sel])
                out["binary"] += [{**tag, "season": int(s), "market": "home_win|no_tie", **hw},
                                  {**tag, "season": int(s), "market": "home_cover|no_push", **cv}]
    tables = {}
    for k, v in out.items():
        if not v:
            tables[k] = pd.DataFrame()
        elif isinstance(v[0], pd.DataFrame):
            tables[k] = pd.concat(v, ignore_index=True)
        else:
            tables[k] = pd.DataFrame(v)
    if "season" in tables["binary"]:
        tables["binary"]["season"] = tables["binary"]["season"].fillna("all")
    return tables


# --------------------------------------------------------------------- CLI
def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    if df.empty:
        return "_(empty)_\n"
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def render_markdown(results: dict[str, dict[str, pd.DataFrame]], source: str) -> str:
    parts = ["# Step 3 outcome distributions (walk-forward, auto-generated)\n",
             f"Source predictions: `{source}`. Generated by `python -m nfl_market.distributions.evaluate`. "
             "Parameters for each test season are fit only on earlier seasons' OOF rows; the 2025 holdout "
             "and the 2026 forward season are excluded. Families k1_frozen, s1 and d2 use constants fit on 2016-2020 outcomes in nfl-betting-model, so their 2016-2020 scores are in-sample. Reference lines are the model's own implied lines "
             "unless a market line column was supplied.\n"]
    for name, sections in [("Summary (log score, PIT, coverage)", "summary"), ("Fitted parameters", "params"),
                           ("Key numbers", "key_numbers"), ("Binary markets", "binary"),
                           ("Push calibration (whole-number lines)", "push"), ("Interval coverage", "coverage")]:
        parts.append(f"\n## {name}\n")
        for label, tabs in results.items():
            t = tabs.get(sections, pd.DataFrame())
            if sections == "key_numbers" and not t.empty:
                t = t[t["season"] == "all"]
            parts.append(f"\n### {label}\n\n" + _md_table(t))
    return "".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--predictions", required=True, help="parquet/csv with PREDICTION_COLUMNS + home_score/away_score")
    ap.add_argument("--out", required=True, help="markdown output path")
    ap.add_argument("--families", default="empirical,empirical+key,t,t+key,kernel,kernel_sym,keynum,shifted_total,nb,k1,k1_frozen,s1,d2",
                    help="comma list of families (empirical, t, nb, each optionally +key)")
    ap.add_argument("--targets", default="margin,total")
    ap.add_argument("--min-history-seasons", type=int, default=3)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--ref-col", default=None, help="column with a home spread to use for buckets/push/cover")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    path = Path(args.predictions)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    for c in ("prediction_time_utc", "train_end_utc", "label_available_at_utc"):
        if c in frame.columns:
            frame[c] = pd.to_datetime(frame[c], utc=True)
    results = {}
    for target in args.targets.split(","):
        for fam in args.families.split(","):
            if target == "total" and ("+key" in fam or fam.split("+")[0] in _MARGIN_ONLY):
                continue
            if target == "total" and frame["pred_total"].isna().all():
                continue
            if fam.split("+")[0] in _NEEDS_TOTAL and frame["pred_total"].isna().all():
                continue
            results[f"{target} / {fam}"] = evaluate_distributions(
                frame, fam, target=target, min_history_seasons=args.min_history_seasons,
                model_id=args.model_id, ref_col=args.ref_col, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(results, str(path)))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
