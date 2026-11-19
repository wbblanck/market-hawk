"""Budgeted odds capture for the GitHub Actions workflow (docs/data_sources/odds_capture.md).

    # bet-time snapshot of every upcoming game (GitHub cron; lateness is fine)
    python scripts/capture_odds.py --mode early
    # close capture: wait until kickoff - 25 min for each kickoff group starting within 75 min
    python scripts/capture_odds.py --mode close --wait --window-minutes 75
    # show what would run, make no request
    python scripts/capture_odds.py --mode close --dry-run

Keys come from env only (PARLAY_API_KEY / NFLMI_PARLAY_API_KEY, ODDSLINK_API_KEY /
NFLMI_ODDSLINK_API_KEY). Each capture writes one ``.jsonl.gz`` export under ``--out-dir``
and one usage line per provider to ``--usage``. The workflow encrypts the export before it
is committed anywhere; this script never prints keys or odds.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_market.collectors.capture import (DEFAULT_BUDGETS, MODE_CLOSE, MODE_EARLY, append_usage,  # noqa: E402
                                           budget_check, close_slots, export_runs, read_usage, upcoming_games,
                                           usage_row)
from nfl_market.collectors.snapshot_collector import SnapshotCollector  # noqa: E402
from nfl_market.data.games import load_games  # noqa: E402
from nfl_market.integrations.sharp import oddslink_provider, parlay_api_provider  # noqa: E402

EARLY_HORIZON_HOURS = 8 * 24


def make_provider(name: str, games: pd.DataFrame):
    if name == "parlay_api":
        return parlay_api_provider(books=(), regions="us", markets=("spreads", "totals", "h2h"), games=games)
    if name == "oddslink":
        return oddslink_provider(markets=("spreads", "totals"), games=games)
    raise SystemExit(f"unknown provider {name!r}")


def capture_once(mode: str, provider_names: list[str], games: pd.DataFrame, args, now: datetime) -> int:
    usage = read_usage(args.usage)
    log = (lambda row: None) if args.dry_run else (lambda row: append_usage(args.usage, row))
    chosen = []
    for name in provider_names:
        budget = DEFAULT_BUDGETS[name]
        provider = make_provider(name, games)
        if not provider.available():
            log(usage_row(name, mode, now, "skipped", budget, "no API key in env"))
            print(f"[{name}] skipped: no API key in env")
            continue
        ok, reason = budget_check(budget, usage, now, mode)
        print(f"[{name}] budget: {reason}")
        if not ok:
            log(usage_row(name, mode, now, "skipped", budget, reason))
            continue
        chosen.append((name, provider, budget))
    if not chosen or args.dry_run:
        print("dry run: no request made" if args.dry_run else "no provider called")
        return 0
    collector = SnapshotCollector(args.db, [p for _, p, _ in chosen])
    report = collector.collect()
    print(report.summary())
    for name, provider, budget in chosen:
        run = report.by_provider(provider.name)
        status = "called" if run.status != "unavailable" else "failed"
        append_usage(args.usage, usage_row(name, mode, now, status, budget, run.status, provider.last_usage))
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out_dir) / now.strftime("%Y/%m/%d") / f"{stamp}_{mode}.jsonl.gz"
    counts = export_runs(args.db, [r.run_id for r in report.runs], out)
    print(f"exported {counts} to {out}")
    failed = all(r.status in ("failed", "unavailable") for r in report.runs)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=(MODE_EARLY, MODE_CLOSE), required=True)
    p.add_argument("--wait", action="store_true", help="close mode: sleep until each group's capture time")
    p.add_argument("--window-minutes", type=int, default=75)
    p.add_argument("--lead-minutes", type=int, default=25)
    p.add_argument("--providers", default=None, help="comma separated; default parlay_api (+ oddslink for close)")
    p.add_argument("--db", default="data/capture_run.duckdb")
    p.add_argument("--out-dir", default="captures")
    p.add_argument("--usage", default="state/usage.jsonl")
    p.add_argument("--games", default=None, help="nflverse games.csv (default data/raw/nflverse_games.csv)")
    p.add_argument("--now", default=None, help="ISO time override (testing)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    games = load_games(a.games)  # the 2025 holdout is excluded; only upcoming games matter here
    now = (pd.Timestamp(a.now).tz_convert("UTC") if a.now else pd.Timestamp(datetime.now(UTC))).to_pydatetime()
    names = a.providers.split(",") if a.providers else (
        ["parlay_api"] if a.mode == MODE_EARLY else ["parlay_api", "oddslink"])

    if a.mode == MODE_EARLY:
        up = upcoming_games(games, now, EARLY_HORIZON_HOURS)
        if up.empty:
            print("no games in the next 8 days: nothing to capture")
            return 0
        print(f"{len(up)} games in the next 8 days; first kickoff {up.kickoff_utc.iloc[0]}")
        return capture_once(MODE_EARLY, names, games, a, now)

    slots = close_slots(games, now, window_minutes=a.window_minutes, lead_minutes=a.lead_minutes)
    if not slots:
        print(f"no kickoff within {a.window_minutes} min whose close cutoff is still ahead: nothing to capture")
        return 0
    rc = 0
    for s in slots:
        print(f"slot: capture at {s.capture_at_utc} for kickoffs {[str(k) for k in s.kickoffs_utc]} "
              f"({len(s.game_ids)} games)")
        if a.dry_run:
            continue
        if a.wait:
            delay = (s.capture_at_utc - pd.Timestamp(datetime.now(UTC))).total_seconds()
            if delay > 0:
                print(f"waiting {delay / 60:.1f} min", flush=True)
                time.sleep(delay)
        now = datetime.now(UTC) if not a.now else s.capture_at_utc.to_pydatetime()
        rc |= capture_once(MODE_CLOSE, names, games, a, now)
    if a.dry_run:
        capture_once(MODE_CLOSE, names, games, a, now)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
