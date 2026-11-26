"""Price the upcoming NFL slate and write fair prices to the ``fair_prices`` DuckDB table.

Usage::

    python scripts/price_slate.py [--as-of 2026-11-19T18:00Z] [--days 8] [--model default|elo_v1|ridge_v1|ridge_v2|xgb_v2|stack_v2]
                                  [--oof data/processed/step3_oof_predictions.parquet] [--record-bets] [--db PATH]

Steps (all point in time at ``--as-of``, default now):

1. Fit the game model on every labelled, non-holdout game with ``label_available_at_utc < as_of``
   and predict the unplayed games kicking off in ``(as_of, as_of + days]``.
2. Fit the default distributions (margin ``k1``, total ``t``) on the OOF predictions of the
   same model(s) with prediction and label times before ``as_of``.
3. Price every line seen in stored quotes for those games, plus a +-3 point ladder around
   the model's own implied lines, and append to ``fair_prices``.
4. ``--record-bets``: compute edges on the stored quotes, size paper stakes
   (``settings.stake_policy()``) and append them to the paper bet log. Paper only.

``--model default`` takes ``pred_margin`` and ``pred_total`` from ``ridge_v2`` (ridge_v1 plus nflverse EPA
and pre-game QB status; it beat elo_v1 and ridge_v1 on margin, total and win log loss in the 2015-2024
walk-forward, docs/results/experiments/recommendations_backtest.md). Before fitting, the nflverse
play-by-play and injury panels are refreshed for the live season (``nfl_market.data.nflverse_ext``), so the
injury report used is the one published at pricing time. Each price row's ``model_id`` names the model that
produced it. ``stack_v2`` is used only if
``nfl_market.models.stacking.StackModel`` is importable; its distributions are fit on
``data/processed/step4_stack_predictions.parquet`` when that exists.

Caveat (locked holdout): the 2025 season is never loaded, so the forward model is trained
on 1999-2024 plus the completed 2026 games, with a one-season gap. Elo carries 2024
ratings (regressed at the season boundary) straight into 2026; ratings are staler than a
model that saw 2025 would be. Prices are research output, not betting advice.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from nfl_market.config import settings
from nfl_market.data.games import load_games
from nfl_market.data.store import MarketStore
from nfl_market.holdout import exclude_holdout
from nfl_market.markets.bet_log import BetLogStore
from nfl_market.markets.edges import compute_edges
from nfl_market.markets.pricing import (
    append_fair_prices,
    fit_default_distributions,
    ladder_lines,
    price_frame,
    read_fair_prices,
)
from nfl_market.markets.staking import size_stakes
from nfl_market.models.runner import attach_outcomes

STEP3_OOF = ROOT / "data" / "processed" / "step3_oof_predictions.parquet"
STACK_OOF = ROOT / "data" / "processed" / "step4_stack_predictions.parquet"
DEFAULT_MODEL = "ridge_v2"
#: models that read the nflverse play-by-play / injury panels
EXTERNAL_MODELS = {"ridge_v2", "xgb_v2", "stack_v2"}


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def make_model(model_id: str):
    if model_id == "elo_v1":
        from nfl_market.models.elo import EloModel
        return EloModel()
    if model_id == "ridge_v1":
        from nfl_market.models.ridge_model import RidgeModel
        return RidgeModel()
    if model_id == "ridge_v2":
        from nfl_market.models.ridge_model import ridge_v2
        return ridge_v2()
    if model_id == "xgb_v2":
        from nfl_market.models.xgb_model import xgb_v2
        return xgb_v2()
    if model_id == "stack_v2":
        try:
            from nfl_market.models.stacking import StackModel
        except ImportError as exc:  # agent 2's module may be absent
            raise SystemExit(f"stack_v2 unavailable: {exc}")
        return StackModel.from_weights_file()
    raise SystemExit(f"unknown model {model_id!r}")


def forward_predictions(games: pd.DataFrame, model_id: str, as_of: pd.Timestamp, upcoming: pd.DataFrame) -> pd.DataFrame:
    """Fit ``model_id`` on games labelled before ``as_of``; predict ``upcoming`` with outcomes blanked."""
    train = games[games["label_available_at_utc"].notna() & (games["label_available_at_utc"] < as_of)]
    test = upcoming.copy()
    test["home_score"] = float("nan")
    test["away_score"] = float("nan")
    test["label_available_at_utc"] = pd.NaT
    p = make_model(model_id).fit(train.reset_index(drop=True)).predict(test.reset_index(drop=True))
    p = p.copy()
    p["model_id"] = model_id
    p["fold_id"] = "forward"
    p["prediction_time_utc"] = as_of
    p["train_end_utc"] = pd.to_datetime(train["label_available_at_utc"], utc=True).max()
    return p


def build_predictions(games, model: str, as_of, upcoming) -> tuple[pd.DataFrame, str, str]:
    """One prediction row per game. Returns (preds, margin_model_id, total_model_id)."""
    if model == "default":
        model = DEFAULT_MODEL
    p = forward_predictions(games, model, as_of, upcoming)
    p["total_model_id"] = model
    return p, model, model


def oof_with_outcomes(path: Path, games: pd.DataFrame) -> pd.DataFrame:
    oof = pd.read_parquet(path)
    out = attach_outcomes(oof, games)
    return out.merge(games[["game_id", "label_available_at_utc"]], on="game_id", how="left")


def quote_lines(quotes: pd.DataFrame, game_ids) -> pd.DataFrame:
    if quotes.empty:
        return pd.DataFrame(columns=["game_id", "market", "selection", "line"])
    q = quotes[quotes["event_id"].isin(set(game_ids)) & quotes["market"].isin(["spread", "total"])]
    return q.rename(columns={"event_id": "game_id"})[["game_id", "market", "selection", "line"]].drop_duplicates()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--as-of", default=None, help="pricing time (UTC); default now")
    ap.add_argument("--days", type=float, default=8.0, help="price games kicking off within this many days")
    ap.add_argument("--model", default="default", choices=["default", "elo_v1", "ridge_v1", "ridge_v2", "xgb_v2", "stack_v2"])
    ap.add_argument("--no-refresh", action="store_true", help="use the cached nflverse panels as they are")
    ap.add_argument("--oof", default=None, help="OOF predictions for the distribution fit")
    ap.add_argument("--db", default=settings.database_path)
    ap.add_argument("--record-bets", action="store_true", help="size and record paper bets on current edges")
    args = ap.parse_args(argv)

    as_of = _utc(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC")
    games = exclude_holdout(load_games())          # 2025 never loaded (load_games drops it; belt and braces)
    upcoming = games[(games["kickoff_utc"] > as_of) & (games["kickoff_utc"] <= as_of + pd.Timedelta(days=args.days))
                     & games["home_score"].isna()]
    if upcoming.empty:
        print(f"no unplayed games between {as_of} and +{args.days} days")
        return 0
    if (DEFAULT_MODEL if args.model == "default" else args.model) in EXTERNAL_MODELS:
        from nfl_market.data.nflverse_ext import EXT_DIR, TEAM_PANEL, download
        live = tuple(sorted(set(upcoming["season"].astype(int))))
        if not (args.no_refresh and (EXT_DIR / TEAM_PANEL).exists()):
            download(max(live), refresh_seasons=() if args.no_refresh else live)
    preds, m_id, t_id = build_predictions(games, args.model, as_of, upcoming)
    preds = preds.merge(upcoming[["game_id", "kickoff_utc"]], on="game_id", how="left")

    oof_path = Path(args.oof) if args.oof else (STACK_OOF if args.model == "stack_v2" and STACK_OOF.exists()
                                                else STEP3_OOF)
    oof = oof_with_outcomes(oof_path, games)
    margin_fd, total_fd = fit_default_distributions(oof, as_of, margin_model_id=m_id, total_model_id=t_id)

    store_quotes = MarketStore(args.db).quotes()
    lines = pd.concat([quote_lines(store_quotes, preds["game_id"]), ladder_lines(preds)], ignore_index=True)
    prices = price_frame(preds, margin_fd, total_fd, lines, total_model_col="total_model_id")
    n = append_fair_prices(args.db, prices, written_at=pd.Timestamp.now(tz="UTC"))
    print(f"priced {preds['game_id'].nunique()} games ({len(prices)} contracts; margin {m_id} "
          f"{margin_fd.family}@{min(margin_fd.train_seasons)}-{max(margin_fd.train_seasons)}, total {t_id}) "
          f"-> {args.db} fair_prices: inserted {n}")

    if args.record_bets:
        edges = compute_edges(store_quotes, read_fair_prices(args.db, latest=False), as_of,
                              stake_policy=settings.stake_policy(), close_policy=settings.close_policy())
        log = BetLogStore(args.db, settings.close_policy())
        bets = []
        for mid in sorted(edges["model_id"].unique()) if not edges.empty else []:
            s = size_stakes(edges, settings.stake_policy(), log.bets(), model_id=mid)
            bets.append(s[s["stake_units"] > 0])
        bets = pd.concat(bets, ignore_index=True) if bets else pd.DataFrame()
        k = log.record(bets, decision_time=as_of) if not bets.empty else 0
        print(f"edges {len(edges)}; paper bets recorded {k} (paper only, no wagers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
