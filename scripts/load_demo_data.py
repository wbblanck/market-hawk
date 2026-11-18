"""Load clearly-marked synthetic demo quotes.

Each book snapshot is internally consistent: the away spread is always the exact
opposite of the home spread, and Over/Under always share one total.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datetime import datetime, timedelta, timezone  # noqa: E402

import pandas as pd  # noqa: E402

from nfl_market.config import settings  # noqa: E402
from nfl_market.data.store import MarketStore  # noqa: E402
from nfl_market.data.validation import validate_quotes  # noqa: E402

BOOKS = ["Pinnacle", "Book A", "Book B", "Book C"]
EVENTS = [("DEMO_BUF_NYJ", -2.5, 44.5), ("DEMO_MIA_NE", -6.5, 41.5)]
MINUTES_BEFORE = [180, 120, 60, 30, 10]


def demo_rows(now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    rows = []
    for ei, (event, home_open, total_open) in enumerate(EVENTS):
        ko = now + timedelta(days=3 + ei)
        for si, mins in enumerate(MINUTES_BEFORE):
            ts = ko - timedelta(minutes=mins)
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
                    rows.append(dict(
                        snapshot_id=f"{event}-{si}-{book}-{market}-{sel}", event_id=event, season=2026, week=1,
                        kickoff_utc=ko, captured_at_utc=ts, book=book, market=market, selection=sel, line=line,
                        odds_american=odds, is_live=False, source="synthetic_demo", source_quote_id=None,
                        displayed_liquidity=None, synthetic=True,
                    ))
    return rows


def main() -> None:
    frame = pd.DataFrame(demo_rows())
    if errors := validate_quotes(frame):
        raise SystemExit(f"Demo data failed validation: {errors}")
    result = MarketStore(settings.database_path).append_quotes_report(frame)
    print(f"Synthetic demo rows -> {settings.database_path}: received={result.received} inserted={result.inserted} "
          f"already_stored={result.already_stored}")


if __name__ == "__main__":
    main()
