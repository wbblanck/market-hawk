"""Load an ESPN scoreboard JSON (DraftKings close prices) into a MarketStore as paper-pricing quotes.

    python scripts/load_espn_odds.py ESPN_JSON DB_PATH

The cloud sandbox blocks site.api.espn.com, so the JSON is fetched with Firecrawl
(`https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?limit=100`, no dates parameter: date
ranges return HTTP 400). Only unplayed games are loaded. ``captured_at_utc`` is the load time, never ESPN's.
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nfl_market.data.store import MarketStore

j = json.load(open(sys.argv[1]))
games = pd.read_csv(ROOT / "data" / "raw" / "nflverse_games.csv")
fix = {"LAR": "LA", "WSH": "WAS"}
now = pd.Timestamp.now(tz="UTC")
rows = []
def am(x):
    return int(str(x).replace("+", ""))
for e in j["events"]:
    if e["status"]["type"]["name"] != "STATUS_SCHEDULED":
        continue
    c = e["competitions"][0]
    t = {x["homeAway"]: fix.get(x["team"]["abbreviation"], x["team"]["abbreviation"]) for x in c["competitors"]}
    gm = games[(games.home_team == t["home"]) & (games.away_team == t["away"]) & (games.season == 2026)]
    if len(gm) != 1:
        print("no game", t); continue
    g = gm.iloc[0]
    ko = pd.Timestamp(e["date"]).tz_convert("UTC")
    o = c["odds"][0]
    def add(market, sel, line, odds):
        sid = hashlib.sha1(f"{g.game_id}|{market}|{sel}|{line}|{now}".encode()).hexdigest()[:20]
        rows.append(dict(snapshot_id=sid, event_id=g.game_id, season=2026, week=int(g.week), kickoff_utc=ko,
                         captured_at_utc=now, book="draftkings", market=market, selection=sel, line=line,
                         odds_american=am(odds), is_live=False, source="espn_scoreboard"))
    ps = o["pointSpread"]
    add("spread", "home", float(ps["home"]["close"]["line"]), ps["home"]["close"]["odds"])
    add("spread", "away", float(ps["away"]["close"]["line"]), ps["away"]["close"]["odds"])
    tt = o["total"]
    add("total", "over", float(tt["over"]["close"]["line"].replace("o", "")), tt["over"]["close"]["odds"])
    add("total", "under", float(tt["under"]["close"]["line"].replace("u", "")), tt["under"]["close"]["odds"])
    ml = o["moneyline"]
    for s in ("home", "away"):
        add("moneyline", s, float("nan"), ml[s]["close"]["odds"])
df = pd.DataFrame(rows)
print(len(df), df.event_id.nunique(), "games")
store = MarketStore(sys.argv[2])
print(store.append_quotes_report(df))
