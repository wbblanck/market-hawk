# Odds capture workflow (GitHub Actions)

Will declined paid historical sharp odds (2026-11-19), so Market Hawk builds its own history: CLV exists only for
games we capture. `.github/workflows/capture-odds.yml` runs `scripts/capture_odds.py`, which calls the feeds,
stores every record through the normal snapshot collector (raw `market_snapshots` plus validated
`quote_snapshots`), maps each event to its nflverse `game_id`, and commits an **encrypted** export to the
`market-data` branch.

## What Will needs to set up

Repository secrets (Settings > Secrets and variables > Actions > New repository secret) in `wbblanck/market-hawk`.
The names match the ones already used in `NFL_ATS`, so the values can be copied as they are:

| Secret | Value | Used for |
|---|---|---|
| `PARLAY_API_KEY` | the same ParlayAPI key as in NFL_ATS | all books (Pinnacle plus about 14 US books): tiers 1 and 3, bet-time quotes |
| `ODDS_LINK` | the same OddsLink key as in NFL_ATS | second Pinnacle source and the only Circa source, close captures only |
| `THEODDSAPI` | The Odds API key (free Starter tier) | third source, Pinnacle only (eu region), close captures only. ParlayAPI's feed has lacked Pinnacle since Nov 10 |
| `CAPTURE_PASSPHRASE` | a new long random string (e.g. `openssl rand -base64 32`), kept somewhere safe | encrypts every capture before it is committed. Without it the job stops before making any request |

External timer for close captures (same setup as NFL_ATS `docs/scheduling.md`, which measured GitHub's own cron
2.7-5.7 hours late): on cron-job.org, time zone America/New_York, one job per kickoff slot sending

- `POST https://api.github.com/repos/wbblanck/market-hawk/actions/workflows/capture-odds.yml/dispatches`
- headers `Accept: application/vnd.github+json`, `Authorization: Bearer <token>`, `X-GitHub-Api-Version: 2022-11-28`
- body `{"ref":"main","inputs":{"mode":"close","wait":"true","window_minutes":"75","lead_minutes":"25"}}`
- token: a fine-grained personal access token limited to `wbblanck/market-hawk` with only **Actions: Read and
  write**. It lives only in cron-job.org.

| Fires (ET) | Covers | `window_minutes` |
|---|---|---|
| Thu 7:15 PM | 8:15 PM kickoff | 75 |
| Sun 8:30 AM | 9:30 AM international | 75 |
| Sun 12:00 PM | 1:00 PM games | 75 |
| Sun 3:00 PM | 4:05 and 4:25 PM games | 100 |
| Sun 7:20 PM | 8:20 PM kickoff | 75 |
| Mon 7:15 PM | 8:15 PM kickoff (an earlier doubleheader game needs a one-off job) | 75 |

Saturday, holiday and other one-off kickoffs need a one-off job ~60 minutes before kickoff. A job that fires late
still captures while the close cutoff (kickoff - 15 min) is ahead; after it, the slot is skipped, because a later
quote could not be a close.

Early (bet-time) captures need nothing extra: GitHub cron runs them Tue, Thu, Sat and Sun mornings ET. Being late
by hours does not matter for them.

## Timing and the locked close

The close is the last quote with `captured_at_utc <= kickoff - 15 min`, at most 120 minutes old at that cutoff
(locked). A close job captures each kickoff group at `kickoff - 25 min` (`lead_minutes`, must be over 15).
Kickoffs within 30 minutes of a group's first kickoff share its capture: the 4:05 and 4:25 PM games are captured
once at 3:40 PM, so the 4:25 close is 30 minutes old at its 4:10 cutoff. `captured_at_utc` is the collector's
receipt time, never the feed's timestamp.

## Request budget (shared with NFL_ATS)

| Provider | Quota (from NFL_ATS evidence) | NFL_ATS use | Market Hawk cap per month | Skip early captures below | Skip close captures below |
|---|---|---|---|---|---|
| ParlayAPI | 1,000 requests/month; cost 1 per single-market call (`NFL_ATS data/lines/api_usage.csv`); its optional captures stop below 150 | 311 in Sept 2026, 67 by Nov 18 | 300 cost units | 400 remaining | 250 remaining |
| OddsLink | 1,000 requests/month, 1 per call, 2/second (NFL_ATS probe 2026-10-18) | fallback only | 120 requests | not used | 200 remaining |
| The Odds API (Starter) | 500 credits/month; one call = 2 credits (spreads and totals, Pinnacle only) | none known | 100 credits | not used | 150 remaining |
| OddsPapi, PinnRadar | 250/month and 100/day, shared with kmodel and props | in use | **not used** by Market Hawk | | |
| EndZoneAPI, SportsGameOdds | see "Other secrets" below | | **not used** | | |

Each ParlayAPI call asks for spreads, totals and moneylines at once, budgeted as 3 units until the provider's
`x-requests-last` header reports the real cost (the usage log then uses the measured value). Expected spend:
4 early + about 6 close captures a week = about 10 ParlayAPI calls (at most 30 units) and 6 OddsLink calls a week,
roughly 130 and 26 a month. Both floors sit above NFL_ATS's own floor of 150, so its lock runs keep their
requests. The usage log `state/usage.jsonl` on `market-data` holds only counters, never odds. All caps and floors
are proposals, not locked: they live in `nfl_market.collectors.capture.DEFAULT_BUDGETS`.

## Other secrets in the repo (reviewed 2026-11-20)

- **`THEODDSAPI`**: wired in (above). Pinnacle comes from Pinnacle's public site and "may incur a delay", so its rows carry the flag `public_site_feed_may_delay`; size of the delay is unverified. Each source keeps its own `source` label, and a book's close is built from one source's snapshot, never a mix of sources.
- **`SPORTSGAMEODDS`**: not wired in. The free Amateur tier has 9 retail books (no Pinnacle, Circa or BookMaker), 10-minute updates and 2,500 objects a month. Pinnacle and Circa are in the Pro tier ($299/month), which Will declined for history; BookMaker is not in its published book list.
- **`ENDZONEAPI_API_KEY`**: not wired in. Its main data is a *consensus* closing line (one number per game, source books not named) and, from 2020, per-book open and close for up to 8 books incl. Pinnacle. For 2024-25 the per-book close is "the last game-day snapshot, a few hours before kickoff", which is not our close (15 min before kickoff, at most 120 min old), so it cannot be stored as one. Possibly useful later: 2026 hourly line movement (`get_line_movement`) would be at most 60 minutes old at our cutoff if it includes Pinnacle; unverified, and its terms are unread. Free tier 500 credits a month, 5 per call.
- **`ODDSPAPI_API_KEY`, `PINN_API`**: not used on purpose. Their quotas (250/month, 100/day) are shared with NFL_ATS and kmodel.

## Storage and terms

This repo is public, and vendor terms do not allow redistributing odds (The Odds API forbids it; ParlayAPI and
OddsLink terms are unverified). So the workflow encrypts each export with
`openssl enc -aes-256-cbc -pbkdf2 -iter 200000` before committing it to `market-data`, and never uploads plaintext
as an artifact. To use the data locally:

```bash
git fetch origin market-data && git worktree add ../market-data FETCH_HEAD
export CAPTURE_PASSPHRASE=...   # same value as the secret
python scripts/import_captures.py ../market-data/captures --db data/nfl_market.duckdb
```

Imports are idempotent (snapshot and run ids are skipped if present). Making the repository private would allow
plaintext storage instead; that is Will's call.

## Checks before relying on it

1. Run the workflow once by hand with `mode=early, dry_run=true`: it prints the plan and budget and makes no
   request.
2. Run `mode=early` for real and import the result: ParlayAPI should show about 15 books with event ids mapped
   (`unmapped_event` flags mean a team name the map does not know).
3. OddsLink totals and moneylines in the same call are unverified; the first close capture's stored payload
   shows whether they come back. Spreads are verified.
4. Moneyline quotes are stored raw in `market_snapshots` only; `quote_snapshots` validation still accepts only
   spreads and totals, so moneyline edges and CLV need that routing extended (open item).
