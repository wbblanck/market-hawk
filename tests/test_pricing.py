"""Fair pricing: hand-checked probabilities on a known pmf, ladders, holdout refusal, fit cutoffs."""
import math

import numpy as np
import pandas as pd
import pytest

from nfl_market.distributions.pmf import IntPMF
from nfl_market.markets.interface import PRICE_COLUMNS
from nfl_market.markets.pricing import (
    append_fair_prices,
    default_training_rows,
    fit_default_distributions,
    ladder_lines,
    price_frame,
    price_game,
    read_fair_prices,
)

T = pd.Timestamp("2026-11-22 15:00", tz="UTC")
ROW = {"game_id": "2026_12_BUF_KC", "season": 2026, "week": 12, "prediction_time_utc": T,
       "kickoff_utc": T + pd.Timedelta(hours=3)}
# Margin mass: M=-7: .20, -3: .10, 0: .05, +3: .30, +7: .35 (home perspective)
MARGIN = IntPMF.from_arrays([-7, -3, 0, 3, 7], [0.20, 0.10, 0.05, 0.30, 0.35])
# Total mass: 40: .50, 44: .25, 48: .25
TOTAL = IntPMF.from_arrays([40, 44, 48], [0.50, 0.25, 0.25])


def _get(rows, market, sel, line=None):
    for r in rows:
        if r["market"] == market and r["selection"] == sel and (
                (line is None and math.isnan(r["line"])) or (line is not None and r["line"] == line)):
            return r
    raise KeyError((market, sel, line))


def test_hand_checked_spread_total_moneyline():
    rows = price_game(ROW, MARGIN, TOTAL, spread_lines=[-3, -3.5], total_lines=[44], model_id="m", dist_id="d")
    # home -3: cover iff M > 3 -> .35; push M = 3 -> .30; loss .35
    h3 = _get(rows, "spread", "home", -3.0)
    assert (h3["p_win"], h3["p_push"], h3["p_loss"]) == pytest.approx((0.35, 0.30, 0.35))
    assert h3["p_win_no_push"] == pytest.approx(0.5)            # .35 / .70
    assert h3["fair_american"] == pytest.approx(-100.0)
    # home -3.5: half point, no push: cover .35, loss .65
    h35 = _get(rows, "spread", "home", -3.5)
    assert (h35["p_win"], h35["p_push"], h35["p_loss"]) == pytest.approx((0.35, 0.0, 0.65))
    # away +3 (paired with home -3): cover iff M < 3 -> .20+.10+.05 = .35, push .30
    a3 = _get(rows, "spread", "away", 3.0)
    assert (a3["p_win"], a3["p_push"], a3["p_loss"]) == pytest.approx((0.35, 0.30, 0.35))
    a35 = _get(rows, "spread", "away", 3.5)
    assert a35["p_win"] == pytest.approx(0.65)                  # M < 3.5 includes M = 3
    # moneyline: home M > 0 = .65, tie .05 (push), away .30; fair home = -100 * .65/.30 = -216.67
    mh, ma = _get(rows, "moneyline", "home"), _get(rows, "moneyline", "away")
    assert (mh["p_win"], mh["p_push"], mh["p_loss"]) == pytest.approx((0.65, 0.05, 0.30))
    assert (ma["p_win"], ma["p_push"], ma["p_loss"]) == pytest.approx((0.30, 0.05, 0.65))
    assert mh["fair_american"] == pytest.approx(-216.6667, abs=1e-3)
    # total 44: over T > 44 = .25, push .25, under .50
    ov, un = _get(rows, "total", "over", 44.0), _get(rows, "total", "under", 44.0)
    assert (ov["p_win"], ov["p_push"], ov["p_loss"]) == pytest.approx((0.25, 0.25, 0.50))
    assert (un["p_win"], un["p_push"], un["p_loss"]) == pytest.approx((0.50, 0.25, 0.25))
    for r in rows:
        assert r["p_win"] + r["p_push"] + r["p_loss"] == pytest.approx(1.0)
        assert set(r) == set(PRICE_COLUMNS)


def test_holdout_refused():
    with pytest.raises(ValueError, match="holdout"):
        price_game({**ROW, "game_id": "2025_01_DAL_PHI", "season": 2025}, MARGIN, model_id="m", dist_id="d")


def test_ladder_and_quote_lines():
    preds = pd.DataFrame([{**ROW, "model_id": "m", "pred_margin": 2.8, "pred_total": 44.2}])
    lad = ladder_lines(preds, steps=2)
    # implied home line -round(5.6)/2 = -3.0; ladder -4..-2 in halves; total 44.0 +- 1
    assert sorted(lad[lad.market == "spread"].line) == [-4.0, -3.5, -3.0, -2.5, -2.0]
    assert sorted(lad[lad.market == "total"].line) == [43.0, 43.5, 44.0, 44.5, 45.0]


class _FakeFD:
    """Stand-in for build_pmfs input: we monkeypatch build_pmfs below."""


def test_price_frame_uses_quote_lines(monkeypatch):
    import nfl_market.markets.pricing as P
    monkeypatch.setattr(P, "build_pmfs", lambda fd, frame: (MARGIN.lo, np.tile(MARGIN.probs, (len(frame), 1)))
                        if fd == "m" else (TOTAL.lo, np.tile(TOTAL.probs, (len(frame), 1))))
    monkeypatch.setattr(P, "dist_id", lambda fd: f"{fd}@x")
    preds = pd.DataFrame([{**ROW, "model_id": "elo_v1", "pred_margin": 2.8, "pred_total": 44.2}])
    lines = pd.DataFrame([{"game_id": ROW["game_id"], "market": "spread", "selection": "away", "line": 3.0},
                          {"game_id": ROW["game_id"], "market": "total", "selection": "under", "line": 44.0}])
    out = price_frame(preds, "m", "t", lines)
    assert list(out.columns) == list(PRICE_COLUMNS)
    got = set(zip(out.market, out.selection, out.line.fillna(-1)))
    assert got == {("spread", "home", -3.0), ("spread", "away", 3.0), ("total", "over", 44.0),
                   ("total", "under", 44.0), ("moneyline", "home", -1), ("moneyline", "away", -1)}
    assert set(out[out.market == "total"].dist_id) == {"t@x"}


def _oof(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in (2018, 2019, 2020, 2021):
        for i in range(300):
            pm, pt = rng.normal(0, 5), rng.normal(45, 4)
            m, t = round(pm + rng.normal(0, 13)), max(3, round(pt + rng.normal(0, 13)))
            hs = (t + m) / 2
            rows.append({"game_id": f"{s}_01_G{i}", "season": s, "week": 1, "model_id": "elo_v1", "fold_id": str(s),
                         "prediction_time_utc": pd.Timestamp(f"{s}-09-10", tz="UTC") + pd.Timedelta(hours=i),
                         "train_end_utc": pd.Timestamp(f"{s}-09-01", tz="UTC"), "pred_margin": pm, "pred_total": pt,
                         "pred_margin_sd": np.nan, "pred_total_sd": np.nan, "pred_home_win_prob": np.nan,
                         "home_score": hs, "away_score": t - hs,
                         "label_available_at_utc": pd.Timestamp(f"{s}-09-20", tz="UTC") + pd.Timedelta(hours=i)})
    return pd.DataFrame(rows)


def test_fit_default_distributions_respects_as_of():
    oof = _oof()
    as_of = pd.Timestamp("2021-01-01", tz="UTC")
    tr = default_training_rows(oof, as_of)
    assert set(tr.season) == {2018, 2019, 2020}
    m, t = fit_default_distributions(oof, as_of, margin_model_id="elo_v1", total_model_id="elo_v1")
    assert m.family == "k1" and t.family == "t"
    assert m.train_seasons == (2018, 2019, 2020)
    from nfl_market.markets.pricing import dist_id
    assert dist_id(m) == "k1@2018-2020"
    with pytest.raises(ValueError):
        fit_default_distributions(oof, as_of, margin_model_id="nope")


def test_fair_prices_store_roundtrip(tmp_path):
    db = tmp_path / "x.duckdb"
    rows = pd.DataFrame(price_game(ROW, MARGIN, TOTAL, spread_lines=[-3], total_lines=[44], model_id="m", dist_id="d"))
    assert append_fair_prices(db, rows) == len(rows)
    assert append_fair_prices(db, rows) == 0                      # identical rows: append-only no-op
    later = rows.assign(prediction_time_utc=T + pd.Timedelta(hours=1), p_win=rows.p_win)
    assert append_fair_prices(db, later) == len(rows)
    got = read_fair_prices(db)
    assert len(got) == len(rows) and (got.prediction_time_utc == T + pd.Timedelta(hours=1)).all()
    with pytest.raises(ValueError, match="holdout"):
        append_fair_prices(db, rows.assign(game_id="2025_02_A_B"))
