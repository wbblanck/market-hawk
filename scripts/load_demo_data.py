"""Load clearly-marked synthetic demo data (every row ``synthetic=True``; demo only, never results).

Writes, into ``settings.database_path``:

* quotes for two upcoming demo games (captured in the last three hours, kicking off in
  three and four hours). Each book snapshot is internally consistent: the away spread is
  always the exact opposite of the home spread, and Over/Under always share one total;
* synthetic fair prices (model ``demo_synthetic``, dist ``synthetic_demo``) for those games;
* paper bets sized on the demo edges (open), plus one finished demo game with three paper
  bets whose closes (one per close tier) and results are synthetic rows.

Demo game ids start with ``DEMO_`` (never a 2025 holdout id). Quote snapshot ids are
deterministic, so re-running adds no quotes; it appends a fresh set of demo prices and open
bets, and adds the finished demo game only once.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from nfl_market.config import (
    TIER_ALL_BOOK_MEDIAN,
    TIER_PINNACLE,
    TIER_SHARP_MEDIAN,
    settings,
)
from nfl_market.data.store import MarketStore
from nfl_market.data.validation import validate_quotes
from nfl_market.distributions.pmf import IntPMF
from nfl_market.markets.bet_log import BetLogStore, pnl_units, settle
from nfl_market.markets.edges import compute_edges
from nfl_market.markets.pricing import append_fair_prices, price_game
from nfl_market.markets.staking import size_stakes

BOOKS = ["Pinnacle", "Book A", "Book B", "Book C"]
EVENTS = [("DEMO_BUF_NYJ", -2.5, 44.5), ("DEMO_MIA_NE", -6.5, 41.5)]
MINUTES_BEFORE = [180, 120, 60, 30, 10]   # capture times, minutes before "now"
DEMO_MODEL_ID = "demo_synthetic"
DEMO_DIST_ID = "synthetic_demo"
#: synthetic model means (home margin, total) per demo event: deliberately off the market for a visible edge
DEMO_MEANS = {"DEMO_BUF_NYJ": (4.5, 42.0), "DEMO_MIA_NE": (6.0, 43.5)}
PAST_EVENT = "DEMO_KC_DEN"


def _kickoff(now: datetime, ei: int) -> datetime:
    return now + timedelta(hours=3 + ei)


def demo_rows(now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(UTC)
    rows = []
    for ei, (event, home_open, total_open) in enumerate(EVENTS):
        ko = _kickoff(now, ei)
        for si, mins in enumerate(MINUTES_BEFORE):
            ts = now - timedelta(minutes=mins)
            for bi, book in enumerate(BOOKS):
                home = home_open - (0.5 if (si + bi) % 4 == 0 else 0)
                total = total_open + (0.5 if (si + bi) % 3 == 0 else 0)
                juice = -105 if bi == 0 else -110  # sharper book shows a lower hold
                markets = [
                    ("spread", "home", home, juice),
                    ("spread", "away", -home, juice),
                    ("total", "over", total, juice),
                    ("total", "under", total, juice),
                ]
                for market, sel, line, odds in markets:
                    rows.append({
                        "snapshot_id": f"{event}-{si}-{book}-{market}-{sel}", "event_id": event, "season": 2026, "week": 1,
                        "kickoff_utc": ko, "captured_at_utc": ts, "book": book, "market": market, "selection": sel, "line": line,
                        "odds_american": odds, "is_live": False, "source": "synthetic_demo", "source_quote_id": None,
                        "displayed_liquidity": None, "synthetic": True,
                    })
    return rows


def _normal_pmf(mean: float, sd: float, lo: int, hi: int) -> IntPMF:
    x = np.arange(lo, hi + 1)
    p = np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return IntPMF(lo, p / p.sum())


def demo_prices(now: datetime | None = None) -> pd.DataFrame:
    """Synthetic fair prices at every demo quote line (normal pmfs; not a real model)."""
    now = now or datetime.now(UTC)
    quotes = pd.DataFrame(demo_rows(now))
    rows = []
    for ei, (event, _, _) in enumerate(EVENTS):
        q = quotes[quotes.event_id == event]
        sp = q[(q.market == "spread") & (q.selection == "home")].line.unique()
        tl = q[q.market == "total"].line.unique()
        m, t = DEMO_MEANS[event]
        pred = {"game_id": event, "season": 2026, "week": 1, "kickoff_utc": _kickoff(now, ei),
                "prediction_time_utc": now - timedelta(minutes=200)}
        rows += price_game(pred, _normal_pmf(m, 13.5, -60, 60), _normal_pmf(t, 13.0, 2, 120), spread_lines=sp,
                           total_lines=tl, model_id=DEMO_MODEL_ID, dist_id=DEMO_DIST_ID)
    return pd.DataFrame(rows)


def demo_past_bets(now: datetime | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One finished demo game with three bets and synthetic closes (one per tier) and results."""
    now = now or datetime.now(UTC)
    ko = now - timedelta(days=2)
    decided = ko - timedelta(hours=5)
    base = {"game_id": PAST_EVENT, "season": 2026, "week": 1, "kickoff_utc": ko, "captured_at_utc": decided - timedelta(minutes=3),
                "decision_time_utc": decided, "opposite_odds_american": -110, "model_id": DEMO_MODEL_ID, "dist_id": DEMO_DIST_ID,
                "p_win": 0.55, "p_push": 0.0, "ev_per_unit": 0.05, "kelly_fraction": 0.055, "stake_units": 1.0,
                "stake_limit_reason": "max_bet", "synthetic": True}
    bets = pd.DataFrame([
        {**base, "book": "Pinnacle", "market": "spread", "selection": "home", "line": -2.5, "odds_american": -110},
        {**base, "book": "Circa", "market": "spread", "selection": "away", "line": 3.5, "odds_american": -110},
        {**base, "book": "Book A", "market": "total", "selection": "over", "line": 44.5, "odds_american": -110},
    ])
    # synthetic closes mirroring the README CLV examples, one tier each
    closes = [(-3.0, TIER_PINNACLE, 0.5, 0.53, TIER_PINNACLE, 0.03, ""),
              (3.0, TIER_SHARP_MEDIAN, 0.5, 0.52, TIER_SHARP_MEDIAN, 0.02, ""),
              (45.5, TIER_ALL_BOOK_MEDIAN, 1.0, np.nan, "missing", np.nan, "MARKET_MEDIAN_FALLBACK,FEW_BOOKS")]
    home, away = 24, 20  # margin +4, total 44
    grades = []
    for b, (lc, lt, lclv, pf, pt, pclv, fl) in zip(bets.to_dict("records"), closes):
        grades.append({"bet_key": (b["market"], b["selection"], b["line"]), "kind": "close",
                           "graded_at_utc": ko, "line_close": lc, "line_close_tier": lt, "line_clv": lclv,
                           "price_close_fair_prob": pf, "price_close_tier": pt, "price_clv": pclv,
                           "close_flags": ",".join(filter(None, [fl, "SYNTHETIC"])),
                           "close_policy_fingerprint": settings.close_policy().fingerprint(), "synthetic": True})
        res = settle(b["market"], b["selection"], b["line"], home - away, home + away)
        grades.append({"bet_key": (b["market"], b["selection"], b["line"]), "kind": "result",
                           "graded_at_utc": ko + timedelta(hours=4), "result": res,
                           "pnl_units": pnl_units(res, b["stake_units"], b["odds_american"]), "synthetic": True})
    return bets, pd.DataFrame(grades)


def main() -> None:
    now = datetime.now(UTC)
    frame = pd.DataFrame(demo_rows(now))
    if errors := validate_quotes(frame):
        raise SystemExit(f"Demo data failed validation: {errors}")
    db = settings.database_path
    result = MarketStore(db).append_quotes_report(frame)
    print(f"Synthetic demo rows -> {db}: received={result.received} inserted={result.inserted} "
          f"already_stored={result.already_stored}")

    n_prices = append_fair_prices(db, demo_prices(now), synthetic=True)
    log = BetLogStore(db, settings.close_policy())
    edges = compute_edges(MarketStore(db).quotes(), demo_prices(now).assign(synthetic=True), now,
                          stake_policy=settings.stake_policy())
    edges = edges[edges.synthetic]
    stakes = size_stakes(edges, settings.stake_policy(), log.bets(), model_id=DEMO_MODEL_ID) if not edges.empty else edges
    n_open = log.record(stakes[stakes.stake_units > 0], decision_time=now) if not stakes.empty else 0

    bets, grades = demo_past_bets(now)
    n_past = 0 if (log.bets().game_id == PAST_EVENT).any() else log.record(bets)
    if n_past:
        ids = log.bets().query("game_id == @PAST_EVENT")
        key = {(r.market, r.selection, float(r.line)): r.bet_id for r in ids.itertuples()}
        grades["bet_id"] = grades.bet_key.map(lambda k: key[(k[0], k[1], float(k[2]))])
        log.append_grades(grades.drop(columns="bet_key"))
    print(f"Synthetic fair prices inserted={n_prices}; demo paper bets open={n_open} graded={n_past} (all synthetic)")


if __name__ == "__main__":
    main()
