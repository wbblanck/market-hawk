"""Paper stake sizing: fractional Kelly with caps (``docs/methodology/staking.md``).

Paper/shadow only: nothing here places, routes or simulates a wager with a book.
Every policy value lives in ``nfl_market.config.StakePolicy`` and is **proposed,
not locked by Will**.

Kelly for a ternary bet (win / push / loss) at decimal odds ``d``, ``b = d - 1``.
Staking a fraction ``f`` of the bankroll, the expected log growth is::

    G(f) = p_win * log(1 + f b) + p_push * log(1) + p_loss * log(1 - f)

A push returns the stake, so its term is ``log 1 = 0`` and drops out of ``G``.
Setting ``G'(f) = p_win b / (1 + f b) - p_loss / (1 - f) = 0`` gives::

    f* = (p_win b - p_loss) / (b (p_win + p_loss)) = (q b - (1 - q)) / b,   q = p_win / (1 - p_push)

i.e. the two-outcome Kelly at the no-push win probability. With ``p_push = 0`` this is
the textbook ``(p b - (1 - p)) / b``; with pushes the denominator ``p_win + p_loss < 1``
makes ``f*`` slightly larger than ``(p_win b - p_loss) / b`` (the bankroll is not at risk
on the push mass). Negative values are clipped to 0.

Sizing order (each later step can only lower a stake; the binding step is recorded in
``stake_limit_reason``):

1. eligibility: rows with a blocking flag (``STALE_QUOTE``, ``SUPERSEDED``) are dropped;
2. ``min_edge``: ``ev_per_unit < min_ev`` gets stake 0;
3. correlated-bet rule: per game at most one **margin** bet (spread or moneyline, either
   side: home spread and home moneyline are the same position on the margin, home
   moneyline and away spread would be a middle) and one **total** bet; the highest-EV
   candidate of each group wins (ties: book, line). A game/group already holding an
   ``existing_bets`` row gets nothing (``existing_bet``);
4. ``kelly``: ``kelly_multiplier * f* * bankroll_units``;
5. ``max_bet``: per-bet cap;
6. ``max_game``: all bets on one game together (existing ones included), allocated in EV order;
7. ``max_week``: all bets of one season-week together (existing ones included), in EV order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StakePolicy
from .edges import BLOCKING_FLAGS
from .odds import american_to_decimal

REASON_KELLY = "kelly"
REASON_MAX_BET = "max_bet"
REASON_MAX_GAME = "max_game"
REASON_MAX_WEEK = "max_week"
REASON_MIN_EDGE = "min_edge"
REASON_EXISTING = "existing_bet"

MARKET_GROUP = {"spread": "margin", "moneyline": "margin", "total": "total"}
_EPS = 1e-12


def kelly_fraction(p_win: float, p_push: float, odds_american: int) -> float:
    """Full-Kelly bankroll fraction for a win/push/loss bet; ``0`` when there is no edge."""
    if not (0 <= p_win <= 1 and 0 <= p_push < 1 and p_win + p_push <= 1 + 1e-12):
        raise ValueError("need 0 <= p_win, 0 <= p_push < 1, p_win + p_push <= 1")
    b = american_to_decimal(odds_american) - 1
    p_loss = max(0.0, 1.0 - p_win - p_push)
    f = (p_win * b - p_loss) / (b * (p_win + p_loss))
    return float(max(0.0, f))


def _blocked(flags: pd.Series) -> pd.Series:
    return flags.fillna("").map(lambda s: bool(set(filter(None, str(s).split(","))) & BLOCKING_FLAGS))


def _existing(existing_bets: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["game_id", "season", "week", "market", "stake_units"]
    if existing_bets is None or existing_bets.empty:
        return pd.DataFrame(columns=cols)
    x = existing_bets.copy()
    if "status" in x.columns:
        x = x[x["status"] != "void"]
    return x[cols]


def size_stakes(edges: pd.DataFrame, policy: StakePolicy | None = None,
                existing_bets: pd.DataFrame | None = None, *, model_id: str | None = None) -> pd.DataFrame:
    """Size paper stakes for an edge frame (``EDGE_COLUMNS``).

    Returns one row per surviving candidate (at most one per game x market group) with the
    edge columns plus ``kelly_fraction``, ``stake_units`` and ``stake_limit_reason``,
    sorted by EV. Rows with ``stake_units == 0`` are kept so the reason is visible.
    Edges from several models must be narrowed with ``model_id`` (choosing the best model
    per contract after the fact would be selection on the outcome of the comparison).
    """
    pol = policy or StakePolicy()
    if edges is None or edges.empty:
        return pd.DataFrame(columns=[*(edges.columns if edges is not None else []), "kelly_fraction",
                                     "stake_units", "stake_limit_reason"])
    e = edges.copy()
    if model_id is not None:
        e = e[e["model_id"] == model_id]
    if e["model_id"].nunique() > 1:
        raise ValueError(f"edges from several models {sorted(e['model_id'].unique())}: pass model_id")
    e = e[~_blocked(e["flags"])]
    e = e[e["ev_per_unit"].notna()].copy()
    e["_group"] = e["market"].map(MARKET_GROUP)

    # 3. one candidate per game x market group
    e = e.sort_values(["ev_per_unit", "book", "line"], ascending=[False, True, True])
    e = e.drop_duplicates(["game_id", "_group"], keep="first").copy()
    ex = _existing(existing_bets)
    ex["_group"] = ex["market"].map(MARKET_GROUP)
    held = set(zip(ex["game_id"], ex["_group"]))
    e["kelly_fraction"] = [kelly_fraction(float(w), float(pp), int(o))
                           for w, pp, o in zip(e["p_win"], e["p_push"], e["odds_american"])]

    stake, reason = [], []
    game_used = ex.groupby("game_id")["stake_units"].sum().to_dict()
    week_used = ex.groupby(["season", "week"])["stake_units"].sum().to_dict()
    for r in e.to_dict("records"):
        if r["ev_per_unit"] < pol.min_ev:                          # 2
            stake.append(0.0); reason.append(REASON_MIN_EDGE); continue
        if (r["game_id"], r["_group"]) in held:                    # 3
            stake.append(0.0); reason.append(REASON_EXISTING); continue
        s, why = pol.kelly_multiplier * r["kelly_fraction"] * pol.bankroll_units, REASON_KELLY   # 4
        if s > pol.max_bet_units + _EPS:                           # 5
            s, why = pol.max_bet_units, REASON_MAX_BET
        g_left = max(0.0, pol.max_game_units - game_used.get(r["game_id"], 0.0))
        if s > g_left + _EPS:                                      # 6
            s, why = g_left, REASON_MAX_GAME
        wk = (r["season"], r["week"])
        w_left = max(0.0, pol.max_week_units - week_used.get(wk, 0.0))
        if s > w_left + _EPS:                                      # 7
            s, why = w_left, REASON_MAX_WEEK
        s = float(np.round(s, 6))
        game_used[r["game_id"]] = game_used.get(r["game_id"], 0.0) + s
        week_used[wk] = week_used.get(wk, 0.0) + s
        stake.append(s); reason.append(why)
    e["stake_units"] = stake
    e["stake_limit_reason"] = reason
    return e.drop(columns="_group").reset_index(drop=True)


__all__ = ["MARKET_GROUP", "REASON_EXISTING", "REASON_KELLY", "REASON_MAX_BET", "REASON_MAX_GAME",
           "REASON_MAX_WEEK", "REASON_MIN_EDGE", "kelly_fraction", "size_stakes"]
