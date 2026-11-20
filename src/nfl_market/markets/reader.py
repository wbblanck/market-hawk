"""Read-only :class:`~nfl_market.markets.interface.MarketReader` over the DuckDB store.

The dashboard and the MCP server read prices, edges, the paper bet log, CLV summaries and
model health through :class:`StoreMarketReader` only, so both show identical numbers.

* ``prices``: the ``fair_prices`` table written by ``scripts/price_slate.py`` (append-only;
  latest row per contract by ``prediction_time_utc``).
* ``edges``: :func:`markets.edges.compute_edges` over the stored quotes and every stored
  price (point in time at ``as_of``).
* ``bet_log`` / ``clv_summary``: ``paper_bets`` + ``paper_bet_grades`` (``markets.bet_log``).
* ``model_health``: ``docs/results/step4_model_health.json`` (written by ``models.stacking``).

Every method opens the database read-only, returns exactly the contract columns (possibly
empty) and never raises for missing files, tables or rows.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import duckdb
import pandas as pd

from ..config import Settings, StakePolicy, settings
from ..holdout import REPO_ROOT
from .bet_log import clv_summary as _clv_summary
from .bet_log import read_bet_log
from .edges import compute_edges
from .interface import (
    BET_LOG_COLUMNS,
    CLV_SUMMARY_COLUMNS,
    EDGE_COLUMNS,
    MODEL_HEALTH_COLUMNS,
    PRICE_COLUMNS,
    empty_frame,
)
from .pricing import read_fair_prices

DEFAULT_MODEL_HEALTH_PATH = REPO_ROOT / "docs" / "results" / "step4_model_health.json"


def _read_quotes(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    try:
        with duckdb.connect(path, read_only=True) as c:
            names = {r[0] for r in c.execute("SELECT table_name FROM information_schema.tables").fetchall()}
            if "quote_snapshots" not in names:
                return pd.DataFrame()
            d = c.execute("SELECT * FROM quote_snapshots ORDER BY captured_at_utc").df()
    except duckdb.Error:
        return pd.DataFrame()
    for col in ("kickoff_utc", "captured_at_utc"):
        d[col] = pd.to_datetime(d[col]).dt.tz_localize("UTC")
    return d


class StoreMarketReader:
    """``MarketReader`` backed by the DuckDB file at ``database_path`` (default ``settings.database_path``)."""

    def __init__(self, database_path: str | None = None, model_health_path: str | Path | None = None,
                 *, config: Settings | None = None):
        cfg = config or settings
        self.database_path = str(database_path or cfg.database_path)
        self.model_health_path = Path(model_health_path) if model_health_path else DEFAULT_MODEL_HEALTH_PATH
        self.stake_policy: StakePolicy = cfg.stake_policy()
        self.close_policy = cfg.close_policy()

    def prices(self, game_ids: Sequence[str] | None = None) -> pd.DataFrame:
        try:
            p = read_fair_prices(self.database_path, latest=True)
        except Exception:  # noqa: BLE001 - a reader never raises for bad/absent data
            return empty_frame(PRICE_COLUMNS)
        if game_ids is not None:
            p = p[p["game_id"].isin(list(game_ids))]
        return p[list(PRICE_COLUMNS)].reset_index(drop=True)

    def edges(self, as_of: pd.Timestamp | None = None, *, min_ev: float | None = None) -> pd.DataFrame:
        try:
            quotes = _read_quotes(self.database_path)
            prices = read_fair_prices(self.database_path, latest=False)
            e = compute_edges(quotes, prices, as_of, stake_policy=self.stake_policy, close_policy=self.close_policy)
        except Exception:  # noqa: BLE001
            return empty_frame(EDGE_COLUMNS)
        if min_ev is not None and not e.empty:
            e = e[e["ev_per_unit"] >= min_ev]
        return e.reset_index(drop=True)

    def bet_log(self, season: int | None = None) -> pd.DataFrame:
        try:
            b = read_bet_log(self.database_path)
        except Exception:  # noqa: BLE001
            return empty_frame(BET_LOG_COLUMNS)
        if season is not None and not b.empty:
            b = b[b["season"] == season]
        return b.reset_index(drop=True)

    def clv_summary(self, by: Sequence[str] = ("close_tier", "market")) -> pd.DataFrame:
        try:
            return _clv_summary(self.bet_log(), by)
        except Exception:  # noqa: BLE001
            return empty_frame(CLV_SUMMARY_COLUMNS)

    def model_health(self) -> pd.DataFrame:
        try:
            records = json.loads(self.model_health_path.read_text())
            if isinstance(records, dict):  # tolerate {"rows": [...]}
                records = records.get("rows", records.get("records", []))
            d = pd.DataFrame(records)
        except (OSError, ValueError, TypeError, AttributeError):
            return empty_frame(MODEL_HEALTH_COLUMNS)
        if d.empty:
            return empty_frame(MODEL_HEALTH_COLUMNS)
        for c in MODEL_HEALTH_COLUMNS:
            if c not in d.columns:
                d[c] = pd.NA
        return d[list(MODEL_HEALTH_COLUMNS)].reset_index(drop=True)


__all__ = ["DEFAULT_MODEL_HEALTH_PATH", "StoreMarketReader"]
