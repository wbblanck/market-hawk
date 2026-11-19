import pytest

from nfl_market.config import (
    MAIN_LINE_RULE,
    TIER_ALL_BOOK_MEDIAN,
    TIER_PINNACLE,
    TIER_SHARP_MEDIAN,
    ClosePolicy,
    CloseTier,
    Settings,
)
from nfl_market.holdout import DEFAULT_MANIFEST


def test_locked_defaults(monkeypatch):
    for var in ("NFLMI_CLOSE_CUTOFF_MINUTES", "NFLMI_CLOSE_MAX_AGE_MINUTES", "NFLMI_CLOSE_SHARP_MIN_BOOKS",
                "NFLMI_CLOSE_FALLBACK_MIN_BOOKS", "NFLMI_CLOSE_MAIN_LINE_RULE", "NFLMI_HOLDOUT_MANIFEST_PATH"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.database_path == "data/nfl_market.duckdb" and s.mcp_port == 8000  # existing fields kept
    p = s.close_policy()
    assert p.cutoff_minutes == 15
    assert p.max_close_age_minutes == 120
    assert p.fallback_min_books == 5
    assert p.main_line_rule == MAIN_LINE_RULE == "book_primary_else_most_balanced"
    assert [t.name for t in p.tiers] == [TIER_PINNACLE, TIER_SHARP_MEDIAN, TIER_ALL_BOOK_MEDIAN]
    assert p.tier(TIER_PINNACLE).books == ("pinnacle",)
    sharp = p.tier(TIER_SHARP_MEDIAN)
    assert sharp.books == ("circa", "bookmaker") and sharp.min_books == 2 and sharp.aggregate == "median"
    assert p.tier(TIER_ALL_BOOK_MEDIAN).books == ()
    assert s.holdout_manifest_path == str(DEFAULT_MANIFEST)
    assert p == ClosePolicy()


def test_env_prefix_overrides(monkeypatch):
    monkeypatch.setenv("NFLMI_CLOSE_MAX_AGE_MINUTES", "60")
    monkeypatch.setenv("NFLMI_CLOSE_FALLBACK_MIN_BOOKS", "4")
    p = Settings(_env_file=None).close_policy()
    assert p.max_close_age_minutes == 60 and p.fallback_min_books == 4
    assert p.fingerprint() != ClosePolicy().fingerprint()


def test_fingerprint_is_stable():
    assert ClosePolicy().fingerprint() == ClosePolicy().fingerprint()
    assert len(ClosePolicy().fingerprint()) == 12


@pytest.mark.parametrize("kwargs", [
    {"cutoff_minutes": -1}, {"max_close_age_minutes": 0}, {"fallback_min_books": 0},
    {"main_line_rule": "closest_to_minus_110"}, {"tiers": ()},
])
def test_policy_validation(kwargs):
    with pytest.raises(ValueError):
        ClosePolicy(**kwargs)


def test_tier_validation():
    with pytest.raises(ValueError):
        CloseTier("x", ("a", "b"), 1, "single")
    with pytest.raises(ValueError):
        CloseTier("x", ("a",), 1, "mean")
    with pytest.raises(ValueError):
        CloseTier("x", ("circa", "bookmaker"), 3, "median")
    with pytest.raises(ValueError):
        ClosePolicy(tiers=(CloseTier("missing", (), 1, "median"),))


def test_settings_reject_bad_env(monkeypatch):
    monkeypatch.setenv("NFLMI_CLOSE_SHARP_MIN_BOOKS", "3")  # more than the two sharp books
    with pytest.raises(ValueError):
        Settings(_env_file=None).close_policy()
    monkeypatch.setenv("NFLMI_CLOSE_SHARP_MIN_BOOKS", "2")
    monkeypatch.setenv("NFLMI_CLOSE_CUTOFF_MINUTES", "-5")
    with pytest.raises(ValueError):
        Settings(_env_file=None)
