"""Scheduled odds capture: when to call the feeds, how many requests we may spend, and export.

Used by ``scripts/capture_odds.py`` and ``.github/workflows/capture-odds.yml``
(docs/data_sources/odds_capture.md). Two modes:

* ``early``: one snapshot of every upcoming game (bet-time quotes). Started by GitHub's own
  cron, which can run hours late; that is fine for early quotes.
* ``close``: quotes that can serve as the locked close (last quote at or before
  ``kickoff - 15 min``, at most 120 min old). Started by an external timer about an hour
  before a kickoff slot (GitHub cron was measured 2.7-5.7 h late in nfl_ats), then the job
  waits until ``kickoff - lead_minutes`` for each kickoff group and captures.

Budget. The request quotas are shared with Will's nfl_ats repo, so every call is checked
against (a) a Market Hawk monthly cap counted from our own usage log and (b) a floor on the
provider-reported remaining requests, which is set above nfl_ats's own floor so its lock runs
keep their requests. A call that would break either is skipped and logged, never made.
"""
from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import duckdb
import pandas as pd

MODE_EARLY = "early"
MODE_CLOSE = "close"
MODES = (MODE_EARLY, MODE_CLOSE)


@dataclass(frozen=True)
class ProviderBudget:
    """Request budget for one provider. Defaults are proposals, not locked by Will."""

    provider: str
    monthly_cap: int            # Market Hawk's own requests (cost units) per UTC calendar month
    cost_per_call: int          # worst-case cost of one call, used before the provider reports it
    floor_early: int            # skip early captures when the provider reports fewer remaining
    floor_close: int            # skip close captures below this (kept above nfl_ats's floor)
    modes: tuple[str, ...] = MODES


#: ParlayAPI: 1,000 requests/month on Will's key (nfl_ats data/lines/api_usage.csv), nfl_ats used
#: 311 in Sept 2026 and 67 by Nov 18 and keeps optional captures above 150. One call here asks for
#: spreads, totals and h2h; nfl_ats measured cost 1 per single-market call, so 3 is the worst case.
#: OddsLink: 1,000 requests/month, 1 per call (nfl_ats probe 2026-10-18); used only near the close.
DEFAULT_BUDGETS: dict[str, ProviderBudget] = {
    "parlay_api": ProviderBudget("parlay_api", monthly_cap=300, cost_per_call=3, floor_early=400, floor_close=250),
    "oddslink": ProviderBudget("oddslink", monthly_cap=120, cost_per_call=1, floor_early=10**9, floor_close=200,
                               modes=(MODE_CLOSE,)),
    # The Odds API free Starter tier: 500 credits/month; one call = markets x regions = 2 (spreads, totals;
    # Pinnacle only, so one region). Close captures only: its role is a second Pinnacle source.
    "the_odds_api": ProviderBudget("the_odds_api", monthly_cap=100, cost_per_call=2, floor_early=10**9,
                                   floor_close=150, modes=(MODE_CLOSE,)),
}


def _utc(value) -> pd.Timestamp:
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# ------------------------------------------------------------------ schedule
def upcoming_games(games: pd.DataFrame, now, horizon_hours: float) -> pd.DataFrame:
    """Games with kickoff in ``(now, now + horizon]``."""
    now = _utc(now)
    ko = pd.to_datetime(games["kickoff_utc"], utc=True)
    return games[(ko > now) & (ko <= now + pd.Timedelta(hours=horizon_hours))].sort_values("kickoff_utc")


@dataclass(frozen=True)
class CaptureSlot:
    capture_at_utc: pd.Timestamp        # when to call the feeds
    kickoffs_utc: tuple[pd.Timestamp, ...]
    game_ids: tuple[str, ...]


def close_slots(games: pd.DataFrame, now, *, window_minutes: int = 75, lead_minutes: int = 25,
                merge_minutes: int = 30, cutoff_minutes: int = 15) -> list[CaptureSlot]:
    """Kickoff groups starting within ``window_minutes`` and the capture time for each.

    A group's capture is at ``first kickoff - lead_minutes``. Kickoffs within ``merge_minutes`` of a
    group's first kickoff share its capture (e.g. 4:05 and 4:25 PM ET: one call at 3:40 serves both,
    and the 4:25 close is then 30 min old at its cutoff, well inside the 120-min limit). A group
    whose capture time has passed is captured now if its close cutoff (kickoff - cutoff_minutes)
    has not; otherwise it is dropped (a late capture could not be a close).
    """
    if lead_minutes <= cutoff_minutes:
        raise ValueError("lead_minutes must exceed the close cutoff so the capture lands before it")
    now = _utc(now)
    up = upcoming_games(games, now, window_minutes / 60)
    slots: list[CaptureSlot] = []
    group: list[pd.Series] = []

    def flush():
        if not group:
            return
        first = _utc(group[0]["kickoff_utc"])
        at = first - pd.Timedelta(minutes=lead_minutes)
        if at < now:
            if now >= first - pd.Timedelta(minutes=cutoff_minutes):
                return
            at = now
        kos = tuple(sorted({_utc(g["kickoff_utc"]) for g in group}))
        slots.append(CaptureSlot(at, kos, tuple(str(g["game_id"]) for g in group)))

    for _, g in up.iterrows():
        if group and _utc(g["kickoff_utc"]) - _utc(group[0]["kickoff_utc"]) > pd.Timedelta(minutes=merge_minutes):
            flush()
            group = []
        group.append(g)
    flush()
    return slots


# -------------------------------------------------------------------- budget
def read_usage(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def append_usage(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _int(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def month_spent(usage: Iterable[dict[str, Any]], provider: str, now) -> int:
    """Cost units Market Hawk spent with ``provider`` this UTC month (measured cost when known)."""
    now = _utc(now)
    total = 0
    for r in usage:
        if r.get("provider") != provider or r.get("status") != "called":
            continue
        at = _utc(r["at"])
        if (at.year, at.month) == (now.year, now.month):
            cost = _int(r.get("last"))
            total += cost if cost is not None else int(r.get("cost_estimate", 0))
    return total


def last_remaining(usage: Iterable[dict[str, Any]], provider: str, now) -> int | None:
    """Provider-reported remaining requests from our latest call this month (None if unknown)."""
    now = _utc(now)
    best = None
    for r in usage:
        if r.get("provider") != provider or r.get("status") != "called":
            continue
        at = _utc(r["at"])
        rem = _int(r.get("remaining"))
        if rem is None or (at.year, at.month) != (now.year, now.month):
            continue
        if best is None or at >= best[0]:
            best = (at, rem)
    return None if best is None else best[1]


def budget_check(budget: ProviderBudget, usage: list[dict[str, Any]], now, mode: str) -> tuple[bool, str]:
    """(allowed, reason). Never allows a call that could break the monthly cap or the floor."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode not in budget.modes:
        return False, f"{budget.provider} is not used for {mode} captures"
    spent = month_spent(usage, budget.provider, now)
    if spent + budget.cost_per_call > budget.monthly_cap:
        return False, f"monthly cap: spent {spent} + {budget.cost_per_call} > {budget.monthly_cap}"
    remaining = last_remaining(usage, budget.provider, now)
    floor = budget.floor_early if mode == MODE_EARLY else budget.floor_close
    if remaining is not None and remaining - budget.cost_per_call < floor:
        return False, f"provider floor: {remaining} remaining - {budget.cost_per_call} < {floor}"
    return True, f"ok (spent {spent}/{budget.monthly_cap} this month, provider remaining {remaining})"


def usage_row(provider: str, mode: str, now, status: str, budget: ProviderBudget, reason: str = "",
              counters: dict[str, str] | None = None) -> dict[str, Any]:
    c = counters or {}
    return {"at": _utc(now).isoformat().replace("+00:00", "Z"), "provider": provider, "mode": mode,
            "status": status, "reason": reason, "cost_estimate": budget.cost_per_call,
            "remaining": c.get("remaining", ""), "used": c.get("used", ""), "last": c.get("last", "")}


# -------------------------------------------------------------------- export
def _jsonable(v: Any) -> Any:
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return _utc(v).isoformat().replace("+00:00", "Z")
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def export_runs(db_path: str | Path, run_ids: list[str], out_path: str | Path) -> dict[str, int]:
    """Write the given collection runs (runs, raw snapshots, routed quotes) as gzipped JSON lines.

    One line per row: ``{"table": <name>, "row": {...}}``. Returns row counts per table.
    """
    counts = {"collection_runs": 0, "market_snapshots": 0, "quote_snapshots": 0}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not run_ids:
        return counts
    ph = ",".join("?" * len(run_ids))
    queries = {
        "collection_runs": f"SELECT * FROM collection_runs WHERE run_id IN ({ph})",
        "market_snapshots": f"SELECT * FROM market_snapshots WHERE run_id IN ({ph})",
        "quote_snapshots": ("SELECT q.* FROM quote_snapshots q JOIN market_snapshots m USING (snapshot_id) "
                            f"WHERE m.run_id IN ({ph})"),
    }
    with duckdb.connect(str(db_path), read_only=True) as c, gzip.open(out, "wt") as f:
        for table, sql in queries.items():
            frame = c.execute(sql, run_ids).df()
            for rec in frame.astype(object).to_dict("records"):
                f.write(json.dumps({"table": table, "row": {k: _jsonable(v) for k, v in rec.items()}}) + "\n")
                counts[table] += 1
    return counts


def import_export_file(path: str | Path, db_path: str | Path) -> dict[str, int]:
    """Load one export file (plain ``.jsonl.gz``) into a local DuckDB. Idempotent (append-only, ids skip)."""
    from ..data.snapshot_store import RUN_COLUMNS, SNAPSHOT_COLUMNS, SnapshotStore
    from ..data.store import MarketStore

    rows: dict[str, list[dict]] = {"collection_runs": [], "market_snapshots": [], "quote_snapshots": []}
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                rows[rec["table"]].append(rec["row"])
    snaps = SnapshotStore(str(db_path))
    quotes = MarketStore(str(db_path))

    def naive(v):
        return None if v is None else _utc(v).tz_localize(None).to_pydatetime()

    snap_rows = []
    for r in rows["market_snapshots"]:
        r = {k: r.get(k) for k in SNAPSHOT_COLUMNS}
        for k in ("captured_at_utc", "source_timestamp_utc", "kickoff_utc"):
            r[k] = naive(r[k])
        snap_rows.append(r)
    res = snaps.append_snapshots(snap_rows)
    n_runs = 0
    for r in rows["collection_runs"]:
        r = {k: r.get(k) for k in RUN_COLUMNS}
        r["started_at_utc"] = _utc(r["started_at_utc"]).to_pydatetime()
        r["finished_at_utc"] = None if r["finished_at_utc"] is None else _utc(r["finished_at_utc"]).to_pydatetime()
        r["errors"] = json.loads(r["errors"]) if isinstance(r["errors"], str) else (r["errors"] or [])
        n_runs += bool(snaps.record_run(r))
    q_inserted = quotes.append_quotes(pd.DataFrame(rows["quote_snapshots"])) if rows["quote_snapshots"] else 0
    return {"collection_runs": n_runs, "market_snapshots": res.inserted, "quote_snapshots": q_inserted}


def utcnow() -> datetime:
    return datetime.now(UTC)


__all__ = ["DEFAULT_BUDGETS", "MODES", "MODE_CLOSE", "MODE_EARLY", "CaptureSlot", "ProviderBudget",
           "append_usage", "budget_check", "close_slots", "export_runs", "import_export_file", "last_remaining",
           "month_spent", "read_usage", "upcoming_games", "usage_row", "utcnow"]
