"""Timestamped market-snapshot collector.

For each provider: check ``available()``, call ``fetch_quotes``, stamp ONE ``captured_at_utc``
from the injected clock the moment the batch is received, keep every record in
``market_snapshots`` (raw, with provenance), and route clean pre-kickoff sportsbook
spread/total quotes into the validated ``quote_snapshots`` table.

Capture time is ours: it is never taken or backfilled from a provider's source timestamp.
Provider statistics are never treated as historical quotes; history is built only from
what this collector itself captured.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from ..data.snapshot_store import SnapshotStore, snapshot_id_for, snapshot_row, to_utc
from ..data.store import MarketStore
from ..data.validation import validate_quotes
from ..integrations.base import (
    VENUE_SPORTSBOOK,
    MarketDataProvider,
    ProviderQuote,
    ProviderUnavailable,
)

ROUTABLE_MARKETS = {"spread", "total"}

# Statuses recorded on collection_runs.
STATUS_OK = "ok"
STATUS_PARTIAL = "partial"  # stored, but some routing candidates were rejected or routing failed
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class ProviderRunReport:
    run_id: str
    provider: str
    started_at_utc: datetime
    finished_at_utc: datetime | None = None
    captured_at_utc: datetime | None = None
    status: str = STATUS_OK
    n_received: int = 0
    n_stored: int = 0  # newly inserted raw snapshots
    n_already_stored: int = 0  # raw snapshots that already existed (idempotent re-run) or repeated in batch
    n_candidates: int = 0  # sportsbook spread/total records eligible for routing
    n_post_kickoff: int = 0  # eligible except captured at/after kickoff: kept raw, never routed
    n_routed: int = 0  # candidates that passed validation and were sent to quote_snapshots
    n_routed_inserted: int = 0  # of those, rows newly inserted into quote_snapshots
    n_rejected: int = 0  # candidates rejected by validation
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        line = (f"[{self.provider}] run={self.run_id} status={self.status} captured_at={self.captured_at_utc} "
                f"received={self.n_received} stored={self.n_stored} already_stored={self.n_already_stored} "
                f"candidates={self.n_candidates} routed={self.n_routed} (new={self.n_routed_inserted}) "
                f"rejected={self.n_rejected} post_kickoff={self.n_post_kickoff}")
        return "\n".join([line, *(f"  error: {e}" for e in self.errors)])


@dataclass
class CollectionReport:
    runs: list[ProviderRunReport] = field(default_factory=list)

    def by_provider(self, name: str) -> ProviderRunReport:
        return next(r for r in self.runs if r.provider == name)

    def summary(self) -> str:
        return "\n".join(r.summary() for r in self.runs) if self.runs else "no providers configured"


def _quote_row(q: ProviderQuote, sid: str, captured: datetime) -> dict:
    return {
        "snapshot_id": sid, "event_id": q.event_id, "season": q.season, "week": q.week,
        "kickoff_utc": pd.Timestamp(to_utc(q.kickoff_utc)), "captured_at_utc": pd.Timestamp(captured),
        "book": q.venue, "market": q.market.lower(), "selection": q.selection.lower(), "line": float(q.line),
        "odds_american": int(q.odds_american), "is_live": False, "source": q.provider, "source_quote_id": q.provider_market_id,
        "displayed_liquidity": q.liquidity, "synthetic": False,
    }


class SnapshotCollector:
    def __init__(self, store_path: str, providers: Iterable[MarketDataProvider], clock: Callable[[], datetime] = utcnow):
        self.store_path = str(store_path)
        self.providers = list(providers)
        self.clock = clock
        self.snapshots = SnapshotStore(self.store_path)
        self.quotes = MarketStore(self.store_path)

    def _now(self) -> datetime:
        t = self.clock()
        if not isinstance(t, datetime) or t.tzinfo is None:
            raise ValueError("collector clock must return a timezone-aware datetime")
        return t.astimezone(UTC)

    def collect(self, date: str | None = None, event_ids: list[str] | None = None) -> CollectionReport:
        report = CollectionReport()
        for provider in self.providers:
            report.runs.append(self._collect_one(provider, date, event_ids))
        return report

    # -- one provider -------------------------------------------------------------------------
    def _collect_one(self, provider: MarketDataProvider, date, event_ids) -> ProviderRunReport:
        name = str(getattr(provider, "name", type(provider).__name__))
        run = ProviderRunReport(run_id=uuid.uuid4().hex, provider=name, started_at_utc=self._now())
        try:
            self._run(provider, run, date, event_ids)
        except Exception as exc:  # noqa: BLE001 - one provider failing must never stop the others
            run.status = STATUS_FAILED
            run.errors.append(f"{type(exc).__name__}: {exc}")
        run.finished_at_utc = self._now()
        try:
            self.snapshots.record_run({
                "run_id": run.run_id, "started_at_utc": run.started_at_utc, "finished_at_utc": run.finished_at_utc,
                "provider": run.provider, "status": run.status, "n_received": run.n_received, "n_stored": run.n_stored,
                "n_routed": run.n_routed, "n_rejected": run.n_rejected, "errors": run.errors,
            })
        except Exception as exc:  # noqa: BLE001 - recorded on the run
            run.errors.append(f"failed to record run: {type(exc).__name__}: {exc}")
        return run

    def _run(self, provider: MarketDataProvider, run: ProviderRunReport, date, event_ids) -> None:
        if not provider.available():
            run.status = STATUS_UNAVAILABLE
            run.errors.append("provider not available; skipped")
            return
        call_started = self._now()
        try:
            records = list(provider.fetch_quotes(date=date, event_ids=event_ids))
        except ProviderUnavailable as exc:
            run.status = STATUS_UNAVAILABLE
            run.errors.append(f"ProviderUnavailable: {exc}")
            return
        # ONE capture time for the whole batch, stamped at receipt from our clock.
        captured = self._now()
        if captured < call_started:
            raise RuntimeError(f"clock went backwards ({captured} < call start {call_started}); batch not stored")
        run.captured_at_utc = captured
        run.n_received = len(records)

        try:
            routed, extra_flags = self._plan_routing(records, captured, run)
        except Exception as exc:  # noqa: BLE001 - keep the raw batch even if routing cannot be planned
            routed, extra_flags = None, {}
            run.status = STATUS_PARTIAL
            run.errors.append(f"routing plan failed, nothing routed: {type(exc).__name__}: {exc}")
        route_ids = set() if routed is None else set(routed.snapshot_id)

        rows = []
        for q in records:
            sid = snapshot_id_for(q, captured)
            rows.append(snapshot_row(q, run_id=run.run_id, captured_at_utc=captured,
                                     routed_to_quotes=sid in route_ids, extra_flags=extra_flags.get(sid, ())))
        res = self.snapshots.append_snapshots(rows)  # raw history first, whatever routing decides
        run.n_stored = res.inserted
        run.n_already_stored = res.already_stored + res.duplicate_in_batch

        if routed is not None:
            try:
                qres = self.quotes.append_quotes_report(routed)
                run.n_routed = len(routed)
                run.n_routed_inserted = qres.inserted
            except Exception as exc:  # noqa: BLE001 - recorded on the run
                run.status = STATUS_PARTIAL
                run.errors.append(f"routing to quote_snapshots failed: {type(exc).__name__}: {exc}")
        if run.n_rejected and run.status == STATUS_OK:
            run.status = STATUS_PARTIAL

    def _plan_routing(self, records: list[ProviderQuote], captured: datetime, run: ProviderRunReport):
        """Decide which records go to quote_snapshots. Validation runs per (event_id, book) group so one
        bad book+event does not block the rest. Returns (quote_snapshots frame or None, extra flags by id)."""
        flags: dict[str, tuple[str, ...]] = {}
        candidates: dict[str, dict] = {}
        for q in records:
            if q.venue_kind != VENUE_SPORTSBOOK or (q.market or "").lower() not in ROUTABLE_MARKETS:
                continue
            sid = snapshot_id_for(q, captured)
            if q.event_id is None or q.kickoff_utc is None or q.line is None or q.odds_american is None:
                flags[sid] = ("not_routed_missing_fields",)
                continue
            if q.kickoff_utc.tzinfo is None:
                flags[sid] = ("not_routed_kickoff_not_tz_aware",)
                continue
            if captured >= to_utc(q.kickoff_utc):
                run.n_post_kickoff += 1
                flags[sid] = ("captured_at_or_after_kickoff",)
                continue
            candidates.setdefault(sid, _quote_row(q, sid, captured))
        run.n_candidates = len(candidates)
        if not candidates:
            return None, flags

        frame = pd.DataFrame(list(candidates.values()))
        accepted = []
        for (event_id, book), group in frame.groupby(["event_id", "book"], sort=True):
            if errs := validate_quotes(group.reset_index(drop=True)):
                run.n_rejected += len(group)
                run.errors.extend(f"validation [{event_id} / {book}]: {e}" for e in errs)
                for sid in group.snapshot_id:
                    flags[sid] = (*flags.get(sid, ()), "not_routed_validation_failed")
            else:
                accepted.append(group)
        if not accepted:
            return None, flags
        routed = pd.concat(accepted, ignore_index=True)
        # Belt and braces: the combined set must validate as a whole too.
        if errs := validate_quotes(routed):
            run.n_rejected += len(routed)
            run.errors.extend(f"validation [combined batch]: {e}" for e in errs)
            for sid in routed.snapshot_id:
                flags[sid] = (*flags.get(sid, ()), "not_routed_validation_failed")
            return None, flags
        return routed, flags
