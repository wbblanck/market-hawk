"""Close selection: which quote is "the close" (docs/methodology/closing_line.md).

Enforces the preregistered close policy (``nfl_market.config.ClosePolicy``):

* cutoff: the close is the last qualifying quote with ``captured_at_utc <= kickoff_utc - cutoff``
  (15 minutes, locked). Later quotes are ignored; a quote older than ``max_close_age_minutes``
  at the cutoff is stale and does not qualify (no carry-back).
* qualifying: ``is_live`` false and ``synthetic`` false. Only ``captured_at_utc`` is used;
  ``source_timestamp_utc`` is provenance only.
* tiers, first that qualifies wins, recorded per row: ``pinnacle`` -> ``sharp_median``
  (median of Circa and BookMaker, both required) -> ``all_book_median`` (weak fallback,
  flagged ``MARKET_MEDIAN_FALLBACK`` and ``FEW_BOOKS`` below the minimum book count)
  -> ``missing`` (no value, never zero).
* two kinds of close, never mixed:
  - **line close** (``line=None``): each book's main line for the selection, chosen by
    :func:`select_main_line` (book-flagged primary line, else the most balanced line).
  - **price close** (``line=x``): the exact contract (same event, market, selection, line),
    devigged against the opposite side at the paired line in the same snapshot.

A book's close snapshot is all of its rows for the event and market captured at the
latest ``captured_at_utc <= cutoff`` at which it quoted the selection (on the exact line,
for a price close). Quote frames follow the quote-row contract in ``docs/DATA_CONTRACTS.md``;
an optional boolean ``is_main`` (or ``is_primary``) column carries the book's main-line flag.
"""
from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field, replace
from datetime import timedelta

import pandas as pd

from ..config import TIER_MISSING, ClosePolicy, CloseTier
from .odds import devig_two_way, implied_probability

FLAG_MARKET_MEDIAN_FALLBACK = "MARKET_MEDIAN_FALLBACK"
FLAG_FEW_BOOKS = "FEW_BOOKS"
FLAG_KICKOFF_CHANGED = "KICKOFF_CHANGED"

RULE_BOOK_PRIMARY = "book_primary"
RULE_MOST_BALANCED = "most_balanced"
MIXED_RULES = "mixed"

REASON_NO_QUOTE = "no_quote_before_cutoff"
REASON_STALE = "stale"
REASON_NO_MAIN_LINE = "main_line_undetermined"
REASON_NO_OPPOSITE = "no_opposite_side"

MAIN_FLAG_COLUMNS = ("is_main", "is_primary")

OPPOSITE = {"home": "away", "away": "home", "over": "under", "under": "over",
            "favorite": "underdog", "underdog": "favorite"}

_BOOK_ALIASES = {
    "pinnacle": "pinnacle", "pinnaclesports": "pinnacle", "pinnaclecom": "pinnacle", "pinny": "pinnacle",
    "circa": "circa", "circasports": "circa", "circasportsbook": "circa",
    "bookmaker": "bookmaker", "bookmakereu": "bookmaker", "bookmakerag": "bookmaker",
}


def canonical_book(name) -> str:
    """Case-insensitive canonical book name: 'Circa Sports' / 'circasports' -> 'circa',
    'BookMaker.eu' -> 'bookmaker', 'Pinnacle' -> 'pinnacle'. Unknown books are returned
    lower case with punctuation and spaces removed."""
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return _BOOK_ALIASES.get(key, key)


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _paired_line(market: str, line: float) -> float:
    """The opposite side's own line: spread home -3 pairs with away +3; total over 44.5 with under 44.5."""
    return -float(line) if market == "spread" else float(line)


def _odds_at(rows: pd.DataFrame, selection: str, line: float) -> int | None:
    hit = rows[(rows["_sel"] == selection) & ((rows["line"] - float(line)).abs() < 1e-9)]
    if hit.empty:
        return None
    return int(hit["odds_american"].iloc[-1])


@dataclass(frozen=True)
class MainLine:
    line: float | None
    rule: str | None           # book_primary | most_balanced | None (undetermined)
    odds_american: int | None = None
    opposite_odds_american: int | None = None
    fair_probability: float | None = None


def select_main_line(quotes: pd.DataFrame, selection: str, market: str | None = None) -> MainLine:
    """Main line for one side of one book's snapshot of one event/market.

    Rule (locked): the line the book flags as main/primary (``is_main`` / ``is_primary``),
    else the line whose two sides' devigged prices are closest to 50/50. Ties break on lower
    overround, then on the lower line value, so the choice is deterministic.
    """
    sel = selection.lower()
    q = quotes.copy()
    q["_sel"] = q["selection"].astype(str).str.lower()
    q = q[q["line"].notna() & q["odds_american"].notna()]
    if market is None:
        markets = q["market"].unique() if "market" in q.columns else []
        if len(markets) != 1:
            raise ValueError("select_main_line needs quotes for exactly one market")
        market = str(markets[0])
    opp = OPPOSITE[sel]
    mine = q[q["_sel"] == sel]
    if mine.empty:
        return MainLine(None, None)

    def build(line: float, rule: str) -> MainLine:
        o = _odds_at(q, sel, line)
        oo = _odds_at(q, opp, _paired_line(market, line))
        fair = devig_two_way(o, oo)[0] if o is not None and oo is not None else None
        return MainLine(float(line), rule, o, oo, fair)

    flag_col = next((c for c in MAIN_FLAG_COLUMNS if c in mine.columns), None)
    if flag_col is not None:
        flagged = mine[mine[flag_col].fillna(False).astype(bool)]["line"].unique()
        if len(flagged) == 1:
            return build(flagged[0], RULE_BOOK_PRIMARY)

    candidates = []
    for line in sorted(mine["line"].unique()):
        o, oo = _odds_at(q, sel, line), _odds_at(q, opp, _paired_line(market, line))
        if o is None or oo is None:
            continue
        p = devig_two_way(o, oo)[0]
        overround = implied_probability(o) + implied_probability(oo) - 1
        candidates.append((abs(p - 0.5), overround, float(line)))
    if candidates:
        return build(min(candidates)[2], RULE_MOST_BALANCED)
    lines = mine["line"].unique()
    if len(lines) == 1:  # a single one-sided line is trivially the most balanced available
        return build(lines[0], RULE_MOST_BALANCED)
    return MainLine(None, None)


@dataclass(frozen=True)
class BookClose:
    book: str
    captured_at_utc: pd.Timestamp | None
    age_minutes: float | None
    line: float | None = None
    odds_american: int | None = None
    opposite_odds_american: int | None = None
    fair_probability: float | None = None
    main_line_rule: str | None = None
    reason: str | None = None   # None when the book has a qualifying close

    @property
    def qualifies(self) -> bool:
        return self.reason is None


@dataclass(frozen=True)
class CloseResult:
    event_id: str
    market: str
    selection: str
    close_kind: str                       # "line" or "price"
    requested_line: float | None          # exact contract line for a price close
    cutoff_utc: pd.Timestamp
    close_tier: str                       # pinnacle | sharp_median | all_book_median | missing
    close_line: float | None = None
    close_odds_american: int | None = None            # single-book tier only
    close_opposite_odds_american: int | None = None   # single-book tier only
    close_fair_probability: float | None = None       # price close: devigged (median across tier books)
    main_line_rule: str | None = None                 # line close: book_primary | most_balanced | mixed
    books_used: tuple[str, ...] = ()
    n_books: int = 0
    quote_age_minutes: float | None = None            # oldest quote used, measured at the cutoff
    flags: tuple[str, ...] = ()
    fallback_reason: str | None = None                # why higher tiers were skipped
    missing_reason: str | None = None
    policy_fingerprint: str = ""
    book_detail: tuple[BookClose, ...] = field(default=(), repr=False)

    @property
    def is_missing(self) -> bool:
        return self.close_tier == TIER_MISSING

    def as_row(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "book_detail"}
        d["books_used"] = ",".join(self.books_used)
        d["flags"] = ",".join(self.flags)
        return d


def _prepare(quotes: pd.DataFrame, event_id: str, market: str) -> pd.DataFrame:
    q = quotes[(quotes["event_id"].astype(str) == str(event_id)) & (quotes["market"] == market)].copy()
    for col in ("is_live", "synthetic"):
        q[col] = q[col].fillna(False).astype(bool) if col in q.columns else False
    q = q[~q["is_live"] & ~q["synthetic"]]
    q["captured_at_utc"] = pd.to_datetime(q["captured_at_utc"], utc=True)
    q["_sel"] = q["selection"].astype(str).str.lower()
    q["_book"] = q["book"].map(canonical_book)
    return q


def _book_close(rows: pd.DataFrame, book: str, market: str, sel: str, line: float | None,
                cutoff: pd.Timestamp, max_age: timedelta) -> BookClose:
    pre = rows[rows["captured_at_utc"] <= cutoff]
    target = pre[pre["_sel"] == sel]
    if line is not None:
        target = target[(target["line"] - float(line)).abs() < 1e-9]
    if target.empty:
        return BookClose(book, None, None, reason=REASON_NO_QUOTE)
    t = target["captured_at_utc"].max()
    age = (cutoff - t).total_seconds() / 60
    if cutoff - t > max_age:
        return BookClose(book, t, age, reason=REASON_STALE)
    snap = pre[pre["captured_at_utc"] == t]
    if line is None:
        ml = select_main_line(snap, sel, market)
        if ml.line is None:
            return BookClose(book, t, age, reason=REASON_NO_MAIN_LINE)
        return BookClose(book, t, age, ml.line, ml.odds_american, ml.opposite_odds_american,
                         ml.fair_probability, ml.rule)
    o = _odds_at(snap, sel, line)
    oo = _odds_at(snap, OPPOSITE[sel], _paired_line(market, line))
    if oo is None:
        return BookClose(book, t, age, float(line), o, reason=REASON_NO_OPPOSITE)
    return BookClose(book, t, age, float(line), o, oo, devig_two_way(o, oo)[0])


def _tier_books(tier: CloseTier, closes: dict[str, BookClose]) -> list[BookClose]:
    pool = closes.values() if not tier.books else (closes[b] for b in tier.books if b in closes)
    return sorted((c for c in pool if c.qualifies), key=lambda c: c.book)


def _tier_skip_reason(tier: CloseTier, closes: dict[str, BookClose], n: int) -> str:
    books = tier.books or tuple(sorted(closes))
    per = [f"{b}={closes[b].reason if b in closes else REASON_NO_QUOTE}" for b in books
           if b not in closes or not closes[b].qualifies]
    detail = f" ({', '.join(per)})" if per else ""
    return f"{tier.name}: {n} of {tier.min_books} required books{detail}"


def select_close(quotes: pd.DataFrame, kickoff_utc, policy: ClosePolicy, *, event_id: str,
                 market: str, selection: str, line: float | None = None) -> CloseResult:
    """The close for one event/market/selection under ``policy``.

    ``line=None`` gives the line close (main line); a value gives the exact-contract price close.
    """
    sel = selection.lower()
    if sel not in OPPOSITE:
        raise ValueError(f"unsupported selection {selection!r}")
    if line is not None and not math.isfinite(float(line)):
        raise ValueError("line must be finite")
    kickoff = _utc(kickoff_utc)
    cutoff = kickoff - timedelta(minutes=policy.cutoff_minutes)
    max_age = timedelta(minutes=policy.max_close_age_minutes)
    q = _prepare(quotes, event_id, market)

    flags: list[str] = []
    if "kickoff_utc" in q.columns and q["kickoff_utc"].notna().any():
        known = pd.to_datetime(q["kickoff_utc"].dropna(), utc=True)
        if (known != kickoff).any():
            flags.append(FLAG_KICKOFF_CHANGED)

    closes = {b: _book_close(rows, b, market, sel, line, cutoff, max_age) for b, rows in q.groupby("_book")}
    base = CloseResult(event_id=str(event_id), market=market, selection=sel,
                       close_kind="line" if line is None else "price",
                       requested_line=None if line is None else float(line), cutoff_utc=cutoff,
                       close_tier=TIER_MISSING, policy_fingerprint=policy.fingerprint(),
                       book_detail=tuple(closes[b] for b in sorted(closes)))

    skipped: list[str] = []
    for tier in policy.tiers:
        used = _tier_books(tier, closes)
        if len(used) < tier.min_books:
            skipped.append(_tier_skip_reason(tier, closes, len(used)))
            continue
        row_flags = list(flags)
        if tier.aggregate == "median" and not tier.books:
            row_flags.append(FLAG_MARKET_MEDIAN_FALLBACK)
            if len(used) < policy.fallback_min_books:
                row_flags.append(FLAG_FEW_BOOKS)
        rules = {c.main_line_rule for c in used}
        fair = [c.fair_probability for c in used if c.fair_probability is not None]
        single = tier.aggregate == "single"
        if line is None:
            close_line = used[0].line if single else float(statistics.median(c.line for c in used))
            rule = rules.pop() if len(rules) == 1 else MIXED_RULES
            close_fair = used[0].fair_probability if single else None  # lines differ across books
        else:
            close_line, rule = float(line), None
            close_fair = float(statistics.median(fair)) if fair else None
        return replace(
            base, close_tier=tier.name, close_line=close_line,
            close_odds_american=used[0].odds_american if single else None,
            close_opposite_odds_american=used[0].opposite_odds_american if single else None,
            close_fair_probability=close_fair, main_line_rule=rule,
            books_used=tuple(c.book for c in used), n_books=len(used),
            quote_age_minutes=max(c.age_minutes for c in used), flags=tuple(row_flags),
            fallback_reason="; ".join(skipped) or None)

    reasons = sorted({c.reason for c in closes.values() if c.reason}) or [REASON_NO_QUOTE]
    return replace(base, flags=tuple(flags), missing_reason=",".join(reasons),
                   fallback_reason="; ".join(skipped) or None)


def close_table(quotes: pd.DataFrame, requests: pd.DataFrame, policy: ClosePolicy) -> pd.DataFrame:
    """One close row per request. ``requests`` needs event_id, market, selection, kickoff_utc and
    optionally line (NaN or missing = line close)."""
    rows = []
    for r in requests.to_dict("records"):
        ln = r.get("line")
        ln = None if ln is None or (isinstance(ln, float) and math.isnan(ln)) else float(ln)
        res = select_close(quotes, r["kickoff_utc"], policy, event_id=r["event_id"], market=r["market"],
                           selection=r["selection"], line=ln)
        rows.append(res.as_row())
    return pd.DataFrame(rows)


__all__ = [
    "FLAG_FEW_BOOKS",
    "FLAG_KICKOFF_CHANGED",
    "FLAG_MARKET_MEDIAN_FALLBACK",
    "RULE_BOOK_PRIMARY",
    "RULE_MOST_BALANCED",
    "BookClose",
    "CloseResult",
    "MainLine",
    "canonical_book",
    "close_table",
    "select_close",
    "select_main_line",
]
