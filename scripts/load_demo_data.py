import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from datetime import datetime,timedelta,timezone
import pandas as pd
from nfl_market.data.store import MarketStore
now=datetime.now(timezone.utc); rows=[]
for ei,event in enumerate(["DEMO_BUF_NYJ","DEMO_MIA_NE"]):
    ko=now+timedelta(days=3+ei)
    for si,mins in enumerate([180,120,60,30,10]):
        ts=ko-timedelta(minutes=mins)
        for bi,book in enumerate(["Pinnacle","Book A","Book B","Book C"]):
            for market,sel,line in [("spread","home",-2.5+(0.5 if (si+bi)%4==0 else 0)),("spread","away",2.5),("total","over",44.5+(0.5 if (si+bi)%3==0 else 0)),("total","under",44.5)]:
                rows.append(dict(snapshot_id=f"{event}-{si}-{book}-{market}-{sel}",event_id=event,season=2026,week=1,kickoff_utc=ko,captured_at_utc=ts,book=book,market=market,selection=sel,line=line,odds_american=-110,is_live=False,source="synthetic_demo",source_quote_id=None,displayed_liquidity=None,synthetic=True))
print("Inserted synthetic demo rows:",MarketStore().append_quotes(pd.DataFrame(rows)))
