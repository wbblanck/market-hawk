from __future__ import annotations

from datetime import datetime, timezone

from ..config import settings
from ..data.store import MarketStore
from ..data.validation import validate_quotes
from ..integrations.registry import provider_registry


class MarketService:
    def __init__(self, database_path: str | None = None):
        self.database_path = database_path or settings.database_path
        self.store = MarketStore(self.database_path)

    def quotes(self, event_id=None):
        return self.store.quotes(event_id)

    def slate(self):
        q = self.store.quotes()
        return [] if q.empty else q[["event_id", "season", "week", "kickoff_utc"]].drop_duplicates("event_id").to_dict("records")

    def providers(self):
        return provider_registry()

    def health(self, now: datetime | None = None) -> dict:
        """Report health from the data actually present, not from the service starting.

        status:
          * ``error``    - the database could not be read
          * ``no_data``  - the database is readable but holds no quotes
          * ``degraded`` - quotes exist but are synthetic-only, fail validation, or no event is upcoming
          * ``ok``       - real quotes exist, pass validation and cover at least one upcoming event
        """
        now = now or datetime.now(timezone.utc)
        base = {"database_path": self.database_path, "provider_count": len(provider_registry()), "checked_at_utc": now.isoformat()}
        try:
            q = self.store.quotes()
        except Exception as e:  # noqa: BLE001 - surface any read failure as unhealthy
            return {**base, "status": "error", "quote_rows": 0, "warnings": [f"database unreadable: {e}"]}

        if q.empty:
            return {**base, "status": "no_data", "quote_rows": 0, "event_count": 0, "book_count": 0,
                    "latest_capture_utc": None, "synthetic_only": False, "upcoming_events": 0,
                    "warnings": ["no quote snapshots stored"]}

        warnings: list[str] = []
        synthetic = q.synthetic.fillna(False).astype(bool)
        synthetic_only = bool(synthetic.all())
        if synthetic_only:
            warnings.append("only synthetic demo quotes are stored")
        issues = validate_quotes(q)
        warnings.extend(f"validation: {i}" for i in issues)
        upcoming = int(q.loc[q.kickoff_utc.notna() & (q.kickoff_utc > now), "event_id"].nunique())
        if upcoming == 0:
            warnings.append("no upcoming events in the quote store")

        status = "ok" if not (synthetic_only or issues or upcoming == 0) else "degraded"
        latest = q.captured_at_utc.max()
        return {
            **base,
            "status": status,
            "quote_rows": int(len(q)),
            "event_count": int(q.event_id.nunique()),
            "book_count": int(q.book.nunique()),
            "latest_capture_utc": latest.isoformat() if latest is not None else None,
            "synthetic_only": synthetic_only,
            "upcoming_events": upcoming,
            "warnings": warnings,
        }
