"""BENCHMARK (not CLV): price nflverse spread/total lines with walk-forward elo_v1 distributions.

Usage::

    python scripts/run_market_backtest.py [--model elo_v1] [--bootstrap 2000] [--seed 0] [--out docs/results/step4_markets.md]

For each test season ``S`` in the step-3 OOF predictions (2015-2024; the 2025 holdout and
2026 are never read), the margin distribution (``k1``) and total distribution (``t``) are
fit on the model's OOF rows of seasons ``< S`` strictly before ``S``'s first prediction
(``distributions.evaluate.training_rows``; at least 3 earlier seasons, so 2018-2024 are
scored). Each game's nflverse ``spread_line`` / ``total_line`` is priced at the nflverse
prices (``home_spread_odds`` etc.).

**These nflverse lines have unknown capture time** (``data.games.LINE_LABEL``; close-like).
This is a benchmark of the model against a provider line, not closing line value: no CLV
is computed here and none of these numbers may be called CLV.

Bets: per game at most one spread side and one total side (the higher-EV side), placed if
``ev_per_unit >= StakePolicy.min_ev``. Units follow ``markets.staking.size_stakes`` with the
default StakePolicy (fixed 100-unit bankroll, no compounding; caps per game and season-week).
ROI = P&L / units staked; the 95% CI resamples games (both of a game's bets move together)
with replacement, ``--bootstrap`` draws, seed ``--seed``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from nfl_market.config import StakePolicy
from nfl_market.data.games import (
    LINE_LABEL,
    load_games,
    load_nflverse_lines,
)
from nfl_market.distributions.evaluate import (
    fit_distribution,
    training_rows,
    with_outcomes,
)
from nfl_market.holdout import exclude_holdout
from nfl_market.markets.bet_log import pnl_units, settle
from nfl_market.markets.odds import (
    american_to_decimal,
    implied_probability,
)
from nfl_market.markets.pricing import dist_id, price_frame
from nfl_market.markets.staking import size_stakes
from nfl_market.models.runner import attach_outcomes

OOF = ROOT / "data" / "processed" / "step3_oof_predictions.parquet"
OUT_MD = ROOT / "docs" / "results" / "step4_markets.md"
MIN_HISTORY_SEASONS = 3


def load_inputs(model_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = load_games()
    oof = pd.read_parquet(OOF)
    oof = oof[(oof["model_id"] == model_id) & (oof["season"] <= 2024)]
    oof = with_outcomes(attach_outcomes(exclude_holdout(oof), games))
    oof = oof.merge(games[["game_id", "kickoff_utc", "label_available_at_utc"]], on="game_id", how="left")
    lines = load_nflverse_lines()
    return oof.sort_values("prediction_time_utc").reset_index(drop=True), lines


def season_candidates(oof: pd.DataFrame, lines: pd.DataFrame, season: int) -> tuple[pd.DataFrame, dict] | None:
    """Priced nflverse contracts for one test season (both sides), with EV at nflverse prices."""
    tr = training_rows(oof, season)
    if tr["season"].nunique() < MIN_HISTORY_SEASONS:
        return None
    te = oof[oof["season"] == season].merge(lines, on="game_id", how="inner")
    te = te[te["spread_line"].notna() & te["home_spread_odds"].notna() & te["away_spread_odds"].notna()]
    mfd = fit_distribution(tr, "k1", target="margin")
    tfd = fit_distribution(tr.dropna(subset=["pred_total"]), "t", target="total")
    want = pd.concat([
        pd.DataFrame({"game_id": te.game_id, "market": "spread", "selection": "home", "line": -te.spread_line}),
        pd.DataFrame({"game_id": te.game_id, "market": "total", "selection": "over", "line": te.total_line}),
    ]).dropna()
    pr = price_frame(te[[c for c in te.columns if c not in lines.columns or c == "game_id"]], mfd, tfd, want)
    pr = pr[pr["market"] != "moneyline"]
    odds = te.set_index("game_id")
    pick = {("spread", "home"): "home_spread_odds", ("spread", "away"): "away_spread_odds",
            ("total", "over"): "over_odds", ("total", "under"): "under_odds"}
    pr["odds_american"] = [odds.at[g, pick[(m, s)]] for g, m, s in zip(pr.game_id, pr.market, pr.selection)]
    pr = pr[pr["odds_american"].notna()].copy()
    pr["odds_american"] = pr["odds_american"].astype(int)
    dec = pr["odds_american"].map(american_to_decimal)
    pr["breakeven_prob"] = pr["odds_american"].map(implied_probability)
    pr["edge_prob"] = pr["p_win_no_push"] - pr["breakeven_prob"]
    pr["ev_per_unit"] = pr["p_win"] * (dec - 1) - pr["p_loss"]
    out = te.set_index("game_id")
    pr["margin"] = pr.game_id.map(out["margin"])
    pr["total_pts"] = pr.game_id.map(out["total"])
    pr["result"] = [settle(m, s, ln, mg, t) for m, s, ln, mg, t in
                    zip(pr.market, pr.selection, pr.line, pr.margin, pr.total_pts)]
    pr["book"] = "nflverse"
    pr["flags"] = ""
    info = {"season": season, "margin_dist": dist_id(mfd), "total_dist": dist_id(tfd), "k1_sigma": mfd.k1_params.sigma,
            "t_scale": tfd.t_params.scale, "t_df": tfd.t_params.df}
    return pr, info


def cover_logloss(pr: pd.DataFrame, market: str, sel: str) -> tuple[float, int]:
    """Log loss of P(sel wins | no push) on non-push games; 0.5 baseline is ln 2 = 0.6931."""
    x = pr[(pr.market == market) & (pr.selection == sel) & (pr.result != "push")]
    p = np.clip(x["p_win_no_push"].to_numpy(float), 1e-6, 1 - 1e-6)
    y = (x["result"] == "win").to_numpy(float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), len(x)


def bootstrap_roi(bets: pd.DataFrame, B: int, seed: int) -> tuple[float, float]:
    if bets.empty or bets.stake_units.sum() == 0:
        return np.nan, np.nan
    g = bets.groupby("game_id")[["pnl", "stake_units"]].sum()
    pnl, stake = g["pnl"].to_numpy(), g["stake_units"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(B, len(g)))
    s = stake[idx].sum(axis=1)
    roi = np.where(s > 0, pnl[idx].sum(axis=1) / np.where(s > 0, s, 1), np.nan)
    lo, hi = np.nanpercentile(roi, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(pr: pd.DataFrame, bets: pd.DataFrame, label, B: int, seed: int, min_ev: float) -> dict:
    row = {"season": label, "n_games": int(pr.game_id.nunique())}
    best = pr.sort_values("ev_per_unit", ascending=False).drop_duplicates(["game_id", "market"])
    for m, tag in (("spread", "ats"), ("total", "ou")):
        b = best[best.market == m]
        row[f"mean_edge_{tag}"] = float(b.edge_prob.mean())
        q = b[b.ev_per_unit >= min_ev]                      # every qualifying bet, flat (before caps)
        w, lo = (q.result == "win").sum(), (q.result == "loss").sum()
        row[f"n_qual_{tag}"] = len(q)
        row[f"win_rate_{tag}"] = float(w / (w + lo)) if w + lo else np.nan
    qual = best[best.ev_per_unit >= min_ev]
    flat = [pnl_units(r, 1.0, o) for r, o in zip(qual.result, qual.odds_american)]
    row["flat_roi"] = float(np.mean(flat)) if flat else np.nan    # 1 unit on every qualifying bet
    row["n_staked"] = len(bets)
    row["cover_logloss_home"], _ = cover_logloss(pr, "spread", "home")
    row["over_logloss"], _ = cover_logloss(pr, "total", "over")
    row["units_staked"] = float(bets.stake_units.sum())
    row["pnl_units"] = float(bets.pnl.sum())
    row["roi"] = row["pnl_units"] / row["units_staked"] if row["units_staked"] else np.nan
    row["roi_lo"], row["roi_hi"] = bootstrap_roi(bets, B, seed)
    return row


def run(model_id: str, B: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof, lines = load_inputs(model_id)
    policy = StakePolicy()
    rows, infos, all_pr, all_bets = [], [], [], []
    for s in sorted(oof["season"].unique()):
        res = season_candidates(oof, lines, int(s))
        if res is None:
            continue
        pr, info = res
        pr["model_id"] = model_id
        sized = size_stakes(pr, policy)
        bets = sized[sized.stake_units > 0].copy()
        bets["pnl"] = [pnl_units(r, st, o) for r, st, o in zip(bets.result, bets.stake_units, bets.odds_american)]
        rows.append(summarize(pr, bets, int(s), B, seed, policy.min_ev))
        infos.append(info)
        all_pr.append(pr); all_bets.append(bets)
    pr, bets = pd.concat(all_pr, ignore_index=True), pd.concat(all_bets, ignore_index=True)
    rows.append(summarize(pr, bets, "pooled", B, seed, policy.min_ev))
    return pd.DataFrame(rows), pd.DataFrame(infos)


def fmt(v, nd=3) -> str:
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def md_table(df: pd.DataFrame, nd=3) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(fmt(v, nd) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def render(summary: pd.DataFrame, infos: pd.DataFrame, model_id: str, B: int, seed: int) -> str:
    pooled = summary[summary.season == "pooled"].iloc[0]
    pol = StakePolicy()
    return f"""# Step 4 market benchmark (auto-generated; NOT CLV)

Regenerate: `python scripts/run_market_backtest.py --model {model_id} --bootstrap {B} --seed {seed}`
(reads `data/processed/step3_oof_predictions.parquet` and `data/raw/nflverse_games.csv`).

**What this is.** Model `{model_id}` OOF predictions (2015-2024 walk-forward, step 3) priced with a
margin `k1` and total `t` distribution fit walk-forward per season on earlier seasons only, then compared
with the nflverse provider line and price for each game: "{LINE_LABEL}". Because the capture time
is unknown (close-like), this is a **benchmark against a provider line, not closing line value**. No CLV
is computed or implied here. Real CLV only exists for games we capture going forward
(`markets.bet_log`, `docs/methodology/closing_line.md`). The 2025 holdout and 2026 are never read.

**Bets.** Per game at most one spread side and one total side (the higher-EV side) when
`ev_per_unit >= {pol.min_ev}`; units from `markets.staking.size_stakes` with the default StakePolicy
(proposed, not locked: Kelly x {pol.kelly_multiplier}, bankroll {pol.bankroll_units:g} units fixed, max
{pol.max_bet_units:g}/bet, {pol.max_game_units:g}/game, {pol.max_week_units:g}/week). Win rates exclude
pushes. ROI = P&L / units staked, 95% CI from a game-clustered bootstrap ({B} draws, seed {seed}).
Seasons 2015-2017 have fewer than {MIN_HISTORY_SEASONS} earlier OOF seasons to fit the distributions and are not scored.

Columns: `n_qual_*` / `win_rate_*` = every best-side bet with `ev_per_unit >= min_ev`, flat stakes, before
caps (`flat_roi` = P&L per unit on them); `n_staked`, `units_staked`, `pnl_units`, `roi*` = after StakePolicy sizing and caps; `mean_edge_*` = mean of `p_win_no_push - breakeven` of the model's preferred side over all
games (what the model *claims*); `cover_logloss_home` / `over_logloss` = log loss of P(home cover | no push)
and P(over | no push) at the nflverse line (a coin flip scores ln 2 = 0.693; lower is better).

## By season

{md_table(summary)}

## Reading

Pooled over {int(pooled.n_games)} games: {int(pooled.n_qual_ats)} qualifying spread bets won {fmt(pooled.win_rate_ats)} and
{int(pooled.n_qual_ou)} qualifying total bets won {fmt(pooled.win_rate_ou)} (flat, pushes excluded; break-even at
-110 is 0.524). Flat 1-unit ROI on all of them: {fmt(pooled.flat_roi)}. Under the StakePolicy caps {int(pooled.n_staked)} bets got a stake: ROI {fmt(pooled.roi)} on
{fmt(pooled.units_staked, 1)} units (95% CI {fmt(pooled.roi_lo)} to {fmt(pooled.roi_hi)}). The week cap binds
most weeks (8 units), so the staked set is the highest-EV subset of each week.
Cover log loss {fmt(pooled.cover_logloss_home, 4)} and over log loss {fmt(pooled.over_logloss, 4)} vs 0.6931 for 50%.
The model's claimed edges come from disagreeing with the line; step 3 found no model beats the line on
accuracy, so a claimed edge here is mostly model error, and the interval says whether the realised ROI can be
told apart from the vig. This does not support staking real money on `{model_id}`. The capped ROI being above the flat ROI
is a property of one bet subset, inside a CI that includes zero; it is not evidence of skill. Caveat: `elo_v1`
uses ported constants tuned on 2016-2020 (step 3), so 2018-2020 rows are partly in-sample for the game model;
2022-2024 are the clean seasons.

## Distribution fits per season

{md_table(infos, 4)}
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="elo_v1")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_MD))
    args = ap.parse_args(argv)
    summary, infos = run(args.model, args.bootstrap, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(render(summary, infos, args.model, args.bootstrap, args.seed))
    print(summary.to_string())
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
