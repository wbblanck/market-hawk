from pathlib import Path
import duckdb
import pandas as pd
DDL = '''
CREATE TABLE IF NOT EXISTS quote_snapshots (
 snapshot_id VARCHAR PRIMARY KEY,event_id VARCHAR,season INTEGER,week INTEGER,
 kickoff_utc TIMESTAMP,captured_at_utc TIMESTAMP,book VARCHAR,market VARCHAR,
 selection VARCHAR,line DOUBLE,odds_american INTEGER,is_live BOOLEAN,source VARCHAR,
 source_quote_id VARCHAR,displayed_liquidity DOUBLE,synthetic BOOLEAN
)'''
class MarketStore:
    def __init__(self,path="data/nfl_market.duckdb"):
        Path(path).parent.mkdir(parents=True,exist_ok=True); self.path=path
        with duckdb.connect(path) as c: c.execute(DDL)
    def append_quotes(self,frame: pd.DataFrame):
        if frame.empty:return 0
        required={"snapshot_id","event_id","captured_at_utc","book","market","selection","line","odds_american","source"}
        if missing:=required-set(frame.columns): raise ValueError(f"Missing columns: {sorted(missing)}")
        with duckdb.connect(self.path) as c:
            c.register("incoming",frame)
            c.execute("INSERT INTO quote_snapshots BY NAME SELECT * FROM incoming ON CONFLICT(snapshot_id) DO NOTHING")
        return len(frame)
    def quotes(self,event_id=None):
        with duckdb.connect(self.path,read_only=True) as c:
            sql="SELECT * FROM quote_snapshots ORDER BY captured_at_utc"
            if event_id: return c.execute("SELECT * FROM quote_snapshots WHERE event_id=? ORDER BY captured_at_utc",[event_id]).df()
            return c.execute(sql).df()
