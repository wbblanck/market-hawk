"""Map Sports Leader (ESPN) game identity onto the repo's ``event_id`` and match
prediction-market contracts to a game.

The repo has no canonical event-id scheme of its own (demo data uses ad-hoc ids), so this
module uses the nflverse ``game_id`` format ``{season}_{week:02d}_{away}_{home}`` with
nflverse team codes, and only when season *and* week are known. The Sports Leader
scoreboard and odds tools do not report season or week, so by default a game keeps its
ESPN id as ``provider_event_id``, ``event_id`` stays ``None`` and the record is flagged
``unmapped_event``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

# ESPN abbreviations that differ from nflverse team codes. Everything else is identical.
ESPN_TO_NFLVERSE = {"LAR": "LA", "WSH": "WAS"}


def normalize_team(abbrev: str | None) -> str | None:
    """ESPN team abbreviation -> nflverse code (upper case). None stays None."""
    if not abbrev:
        return None
    code = str(abbrev).strip().upper()
    return ESPN_TO_NFLVERSE.get(code, code)


def parse_utc(value: str | None) -> datetime | None:
    """Parse the server's ISO-8601 UTC strings (``2026-10-11T13:30Z``) to aware UTC datetimes."""
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None  # never guess a timezone
    return dt.astimezone(UTC)


@dataclass(frozen=True)
class EventMapping:
    provider_event_id: str
    event_id: str | None
    season: int | None
    week: int | None
    home_team: str | None
    away_team: str | None
    kickoff_utc: datetime | None
    flags: tuple[str, ...] = ()


def map_event(provider_event_id: str, *, home: str | None, away: str | None, start_utc: str | datetime | None,
              season: int | None = None, week: int | None = None) -> EventMapping:
    """Build the repo identity for one ESPN game. ``event_id`` is set only with season, week and both teams."""
    home_team, away_team = normalize_team(home), normalize_team(away)
    kickoff = start_utc if isinstance(start_utc, datetime) else parse_utc(start_utc)
    flags: list[str] = []
    if kickoff is None:
        flags.append("no_kickoff_time")
    if home_team is None or away_team is None:
        flags.append("missing_team")
    event_id = None
    if season is not None and week is not None and home_team and away_team:
        event_id = f"{int(season)}_{int(week):02d}_{away_team}_{home_team}"
    else:
        flags.append("unmapped_event")
    return EventMapping(str(provider_event_id), event_id, season, week, home_team, away_team, kickoff, tuple(flags))


@dataclass(frozen=True)
class ContractMatch:
    matched: bool
    matched_by: str | None
    flags: tuple[str, ...] = field(default_factory=tuple)


def match_contract_to_game(venue_result: dict, *, requested_event_id: str, response_event_id: str | None,
                           game_matched_by: str | None = None) -> ContractMatch:
    """Decide whether one ``get_market_odds`` venue entry belongs to the requested ESPN game.

    Uses only what the server reports: the venue ``status``, the venue ``matchedBy`` text and
    the response's ``eventId`` / top-level ``matchedBy`` (how the ESPN game was picked).
    A match the server made on weaker evidence is kept but flagged.
    """
    status = str(venue_result.get("status") or "")
    matched_by = venue_result.get("matchedBy")
    if status != "ok":
        return ContractMatch(False, matched_by, (f"venue_status:{status or 'missing'}",))
    if response_event_id is not None and str(response_event_id) != str(requested_event_id):
        return ContractMatch(False, matched_by, ("market_event_id_mismatch",))
    flags: list[str] = []
    if game_matched_by not in (None, "eventId"):
        flags.append("espn_game_not_matched_by_event_id")
    text = (matched_by or "").lower()
    if not text:
        flags.append("no_match_basis")
    if "postponed" in text or "off the espn date" in text:
        flags.append("market_date_mismatch")
    elif text and "date" not in text:
        flags.append("market_matched_without_date")
    if "other order" in text:
        flags.append("market_teams_swapped")
    if "constructed slug" in text:
        flags.append("market_slug_constructed")
    return ContractMatch(True, matched_by, tuple(flags))
