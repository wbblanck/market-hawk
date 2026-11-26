"""Import NFL_ATS's 2026 spread captures into a local Market Hawk DuckDB (nothing is copied into this repo).

wbblanck/NFL_ATS captured every book's spread (ParlayAPI feed, 15 books incl. Pinnacle) from 2026 week 3 on,
including captures shortly before its lock (kickoff - 15 min, the same rule as our close). This replays those
captures through the normal snapshot collector, so they land in ``market_snapshots`` and ``quote_snapshots``
exactly like our own captures:

* ``captured_at_utc`` = NFL_ATS's recorded capture time (its receipt time); the feed's book timestamp is kept as
  ``source_timestamp_utc``.
* provider ``nfl_ats_import``; flags ``imported_from_nfl_ats`` and ``home_side_only`` (NFL_ATS stored the home
  side only: home spread and decimal price). Away-side quotes are not invented.
* Spreads only (NFL_ATS has no totals). CBS "open" lines are not imported as quotes: the page shows the opening
  number at a later capture time, so it was not a price available at that time.
* Holdout seasons are refused; only ``status == ok`` rows are used. Re-running is idempotent (snapshot ids are
  deterministic; already-stored rows are skipped).

    git clone --depth 1 https://github.com/wbblanck/nfl_ats ../nfl_ats
    python scripts/import_nfl_ats_lines.py --nfl-ats ../nfl_ats --db data/nfl_market.duckdb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_market.collectors.snapshot_collector import SnapshotCollector  # noqa: E402
from nfl_market.config import settings  # noqa: E402
from nfl_market.holdout import holdout_seasons  # noqa: E402
from nfl_market.integrations.base import VENUE_SPORTSBOOK, ProviderQuote  # noqa: E402

PROVIDER = "nfl_ats_import"


def decimal_to_american(d: float) -> int:
    if not d > 1.0:
        raise ValueError(f"bad decimal price {d}")
    return int(round((d - 1.0) * 100)) if d >= 2.0 else int(round(-100.0 / (d - 1.0)))


def load_captures(nfl_ats: Path) -> pd.DataFrame:
    m = pd.read_csv(nfl_ats / "data" / "lines" / "market_captures.csv")
    m = m[(m["status"] == "ok") & m["home_spread"].notna() & m["home_price_decimal"].notna()]
    m = m[~m["season"].astype(int).isin(holdout_seasons())]
    for c in ("captured_at", "kickoff_utc", "book_last_update"):
        m[c] = pd.to_datetime(m[c], utc=True, errors="coerce")
    return m


def quotes_for(batch: pd.DataFrame) -> list[ProviderQuote]:
    out = []
    for r in batch.itertuples(index=False):
        out.append(ProviderQuote(
            provider=PROVIDER, venue=str(r.book), venue_kind=VENUE_SPORTSBOOK,
            provider_event_id=str(r.provider_event_id), market="spread", selection="home",
            line=float(r.home_spread), odds_american=decimal_to_american(float(r.home_price_decimal)),
            provider_market_id=f"{r.capture_id}:{r.book}:{r.game_id}", event_id=str(r.game_id),
            season=int(r.season), week=int(r.week), kickoff_utc=r.kickoff_utc.to_pydatetime(),
            home_team=str(r.home_team), away_team=str(r.away_team),
            source_timestamp_utc=None if pd.isna(r.book_last_update) else r.book_last_update.to_pydatetime(),
            raw={"source": "wbblanck/NFL_ATS data/lines/market_captures.csv", **{k: (None if pd.isna(v) else str(v))
                 for k, v in r._asdict().items()}},
            quality_flags=("imported_from_nfl_ats", "home_side_only"),
        ))
    return out


class _Replay:
    name = PROVIDER

    def __init__(self, quotes):
        self._q = quotes

    def available(self) -> bool:
        return True

    def fetch_quotes(self, *, date=None, event_ids=None):
        return list(self._q)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--nfl-ats", required=True, help="path to a clone of wbblanck/NFL_ATS")
    p.add_argument("--db", default=settings.database_path)
    a = p.parse_args(argv)
    m = load_captures(Path(a.nfl_ats))
    stored = routed = 0
    for cid, batch in m.groupby("capture_id", sort=True):
        t = batch["captured_at"].iloc[0].to_pydatetime()
        rep = SnapshotCollector(a.db, [_Replay(quotes_for(batch))], clock=lambda t=t: t).collect().runs[0]
        stored += rep.n_stored
        routed += rep.n_routed_inserted
        if rep.errors:
            print(cid, rep.errors)
    print(f"captures {m['capture_id'].nunique()}, rows {len(m)}, newly stored {stored}, newly routed {routed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
