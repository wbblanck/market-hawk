"""nflverse games loader. Fixture: real subset of nflverse games.csv (seasons 2015-2018 + 2025 week 1)."""
from pathlib import Path

import pandas as pd

from nfl_market.data.games import (LINE_LABEL, kickoff_utc, label_available_at, load_games, load_nflverse_lines,
                                   moneyline_home_prob)
from nfl_market.models.interface import GAME_COLUMNS

FIX = Path(__file__).parent / "fixtures" / "nflverse" / "games_sample.csv"


def test_columns_and_holdout_excluded():
    g = load_games(FIX)
    assert list(g.columns[: len(GAME_COLUMNS)]) == list(GAME_COLUMNS)
    assert 2025 not in set(g["season"])
    assert not g["game_id"].str.startswith("2025").any()
    assert len(load_games(FIX, include_holdout=True)) == len(g) + 16
    assert str(g["kickoff_utc"].dt.tz) == "UTC"
    for banned in ("spread_line", "total_line", "home_moneyline", "home_qb_id", "temp", "wind"):
        assert banned not in g.columns
    assert g["home_score"].isna().sum() == 0


def test_eastern_to_utc_and_dst():
    ko, imp = kickoff_utc(pd.Series(["2018-09-09", "2018-12-09", "1999-09-12"]), pd.Series(["13:00", "13:00", None]))
    assert ko.iloc[0] == pd.Timestamp("2018-09-09 17:00", tz="UTC")   # EDT
    assert ko.iloc[1] == pd.Timestamp("2018-12-09 18:00", tz="UTC")   # EST
    assert list(imp) == [False, False, True]


def test_label_available_next_wednesday_noon_et():
    ko = pd.Series(pd.to_datetime(["2018-09-09 17:00", "2018-09-11 00:15", "2024-12-25 18:00", "2018-09-07 00:20"],
                                  utc=True))
    lab = label_available_at(ko)
    # Sunday and Monday-night (Mon 20:15 ET) -> Wed 2018-09-12 12:00 EDT
    assert lab.iloc[0] == pd.Timestamp("2018-09-12 16:00", tz="UTC")
    assert lab.iloc[1] == pd.Timestamp("2018-09-12 16:00", tz="UTC")
    # Wednesday game -> the following Wednesday (EST)
    assert lab.iloc[2] == pd.Timestamp("2025-01-01 17:00", tz="UTC")
    # Thursday-night opener (Thu 20:20 ET) -> next Wed
    assert lab.iloc[3] == pd.Timestamp("2018-09-12 16:00", tz="UTC")
    g = load_games(FIX)
    assert (g["label_available_at_utc"] > g["kickoff_utc"]).all()


def test_franchise_continuity():
    g = load_games(FIX)
    assert set(g.loc[g["home_team"] == "OAK", "home_franchise"]) == {"LV"}
    assert set(g.loc[g["home_team"] == "SD", "home_franchise"]) == {"LAC"}
    assert set(g.loc[g["home_team"] == "STL", "home_franchise"]) == {"LA"}
    assert g["home_franchise"].nunique() == 32


def test_lines_are_labelled_benchmark_only():
    lines = load_nflverse_lines(FIX)
    assert (lines["line_label"] == LINE_LABEL).all()
    assert not lines["game_id"].str.startswith("2025").any()
    p = moneyline_home_prob(pd.Series([-200.0, None]), pd.Series([170.0, 120.0]))
    assert 0.6 < p.iloc[0] < 0.7 and pd.isna(p.iloc[1])
