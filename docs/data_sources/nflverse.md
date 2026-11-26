# nflverse data

Market Hawk reads three nflverse datasets, all from GitHub releases of
[nflverse-data](https://github.com/nflverse/nflverse-data):

| dataset | loader | seasons | used for |
|---|---|---|---|
| schedules `games.csv` | `nfl_market.data.games` | 1999 onward | games, scores, label times |
| play-by-play `play_by_play_<season>.parquet` | `nfl_market.data.nflverse_ext` | 1999 onward | EPA team form, QB dropbacks and EPA |
| injuries `injuries_<season>.parquet` | `nfl_market.data.nflverse_ext` | 2009 onward | pre-game QB status |

2025 (the locked holdout) is never downloaded. Play-by-play is reduced to small panels on download and the raw
file deleted; everything lives under `data/raw/` (gitignored). Build or refresh the panels with:

```bash
python -m nfl_market.data.nflverse_ext --last-season 2026 --refresh 2026
```

`scripts/price_slate.py` refreshes the live season's panels itself before fitting a v2 model (`--no-refresh` to skip).

## Licence and attribution

nflverse data is released under CC-BY 4.0 (FTN charting, which we do not use, is CC-BY-SA 4.0). Attribution:
"Data: nflverse (https://nflverse.nflverse.com), CC-BY 4.0." The underlying NFL data belongs to its owners.
No nflverse data is committed to this public repo; only aggregate results are. Pro-Football-Reference-derived
nflverse tables (snap counts, PFR advanced stats) are not used.

## Timing caveats

- Play-by-play updates nightly after games; stat corrections land by Thursday. Features only use games whose
  `label_available_at_utc` (Wednesday noon ET after kickoff) is before T.
- Injury `date_modified` is missing in 2009 and from 2025 on; nflverse overwrites the live season's file in
  place with no history. Forward pricing uses the report as fetched at pricing time, which is legitimate, but a
  later backtest of 2026 would see final reports. A daily snapshot of the live injury file (stamped with our own
  `captured_at_utc`) is the follow-up that fixes this.
