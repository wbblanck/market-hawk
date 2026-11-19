"""Runtime settings (env prefix ``NFLMI_``) and the preregistered close policy.

Close policy values marked LOCKED were decided by Will (owner) and must not be changed
without opening a new experiment id (``docs/methodology/closing_line.md`` rule L1).
All close values are LOCKED: MAX_CLOSE_AGE and the tier-3 minimum were confirmed by Will on 2026-11-19.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .holdout import DEFAULT_MANIFEST

# Close tiers, in priority order (LOCKED by Will, 2026-11-19).
TIER_PINNACLE = "pinnacle"
TIER_SHARP_MEDIAN = "sharp_median"
TIER_ALL_BOOK_MEDIAN = "all_book_median"
TIER_MISSING = "missing"

# Main-line rule (LOCKED): the book-flagged main/primary line, else the line whose two
# sides' prices are closest to even. The rule actually applied is stored per row.
MAIN_LINE_RULE = "book_primary_else_most_balanced"
MAIN_LINE_RULES = frozenset({MAIN_LINE_RULE})

LOCKED_CLOSE_CUTOFF_MINUTES = 15          # LOCKED: close = last quote <= kickoff - 15 min
MAX_CLOSE_AGE_MINUTES = 120               # LOCKED (Will, 2026-11-19)
LOCKED_SHARP_MIN_BOOKS = 2                # LOCKED: tier 2 needs Circa and BookMaker both
FALLBACK_MIN_BOOKS = 5                    # LOCKED (Will, 2026-11-19; matches nfl_ats); FEW_BOOKS below it


@dataclass(frozen=True)
class CloseTier:
    """One close source tier.

    ``books`` are canonical book names (see ``markets.closing.canonical_book``); an empty
    tuple means "every book with a qualifying close". ``aggregate`` is ``single`` (exactly
    one book's quote) or ``median`` (median across the qualifying books in the tier).
    ``min_books`` is the hard requirement for the tier to be used.
    """

    name: str
    books: tuple[str, ...]
    min_books: int
    aggregate: str

    def __post_init__(self):
        if self.aggregate not in ("single", "median"):
            raise ValueError(f"tier {self.name}: aggregate must be 'single' or 'median'")
        if self.min_books < 1:
            raise ValueError(f"tier {self.name}: min_books must be >= 1")
        if self.aggregate == "single" and len(self.books) != 1:
            raise ValueError(f"tier {self.name}: a single-book tier needs exactly one book")
        if self.books and self.min_books > len(self.books):
            raise ValueError(f"tier {self.name}: min_books {self.min_books} exceeds its {len(self.books)} books")


def default_tiers(sharp_min_books: int = LOCKED_SHARP_MIN_BOOKS) -> tuple[CloseTier, ...]:
    return (
        CloseTier(TIER_PINNACLE, ("pinnacle",), 1, "single"),
        CloseTier(TIER_SHARP_MEDIAN, ("circa", "bookmaker"), sharp_min_books, "median"),
        # Tier 3 is used with any number of books; below fallback_min_books it is flagged FEW_BOOKS.
        CloseTier(TIER_ALL_BOOK_MEDIAN, (), 1, "median"),
    )


@dataclass(frozen=True)
class ClosePolicy:
    """Preregistered definition of "the close" (closing_line.md Section 2)."""

    cutoff_minutes: int = LOCKED_CLOSE_CUTOFF_MINUTES
    max_close_age_minutes: int = MAX_CLOSE_AGE_MINUTES
    fallback_min_books: int = FALLBACK_MIN_BOOKS
    main_line_rule: str = MAIN_LINE_RULE
    tiers: tuple[CloseTier, ...] = field(default_factory=default_tiers)

    def __post_init__(self):
        if self.cutoff_minutes < 0:
            raise ValueError("cutoff_minutes must be >= 0")
        if self.max_close_age_minutes <= 0:
            raise ValueError("max_close_age_minutes must be > 0")
        if self.fallback_min_books < 1:
            raise ValueError("fallback_min_books must be >= 1")
        if self.main_line_rule not in MAIN_LINE_RULES:
            raise ValueError(f"unknown main_line_rule {self.main_line_rule!r}; known: {sorted(MAIN_LINE_RULES)}")
        if not self.tiers:
            raise ValueError("at least one close tier is required")
        names = [t.name for t in self.tiers]
        if len(set(names)) != len(names) or TIER_MISSING in names:
            raise ValueError("tier names must be unique and must not be 'missing'")

    def tier(self, name: str) -> CloseTier:
        for t in self.tiers:
            if t.name == name:
                return t
        raise KeyError(name)

    def fingerprint(self) -> str:
        """Short stable hash of the policy, for experiment charters and result rows."""
        blob = json.dumps(asdict(self), sort_keys=True, default=list)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NFLMI_", env_file=".env", extra="ignore")
    database_path: str = "data/nfl_market.duckdb"
    mcp_transport: str = "stdio"
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8000

    # Close policy. Changing any of these after CLV results exist opens a new experiment id.
    close_cutoff_minutes: int = Field(LOCKED_CLOSE_CUTOFF_MINUTES, ge=0)          # LOCKED
    close_max_age_minutes: int = Field(MAX_CLOSE_AGE_MINUTES, gt=0)      # LOCKED
    close_sharp_min_books: int = Field(LOCKED_SHARP_MIN_BOOKS, ge=1)              # LOCKED
    close_fallback_min_books: int = Field(FALLBACK_MIN_BOOKS, ge=1)      # LOCKED
    close_main_line_rule: str = MAIN_LINE_RULE                                    # LOCKED

    # Locked final holdout manifest (validation_design.md Section 5).
    holdout_manifest_path: str = str(DEFAULT_MANIFEST)

    def close_policy(self) -> ClosePolicy:
        return ClosePolicy(
            cutoff_minutes=self.close_cutoff_minutes,
            max_close_age_minutes=self.close_max_age_minutes,
            fallback_min_books=self.close_fallback_min_books,
            main_line_rule=self.close_main_line_rule,
            tiers=default_tiers(self.close_sharp_min_books),
        )


settings = Settings()
