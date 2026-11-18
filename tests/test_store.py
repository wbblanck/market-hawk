from datetime import datetime,timezone,timedelta
import pandas as pd
from nfl_market.data.store import MarketStore
def test_append_and_read(tmp_path):
    s=MarketStore(str(tmp_path/"db.duckdb")); now=datetime.now(timezone.utc)
    f=pd.DataFrame([dict(snapshot_id="s",event_id="e",season=2026,week=1,kickoff_utc=now+timedelta(days=1),captured_at_utc=now,book="b",market="spread",selection="home",line=-2.5,odds_american=-110,is_live=False,source="test",source_quote_id=None,displayed_liquidity=None,synthetic=False)])
    s.append_quotes(f); assert len(s.quotes("e"))==1
