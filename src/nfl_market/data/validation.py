import pandas as pd
REQUIRED=["snapshot_id","event_id","captured_at_utc","kickoff_utc","book","market","selection","line","odds_american","source"]
def validate_quotes(df: pd.DataFrame)->list[str]:
    if missing:=[x for x in REQUIRED if x not in df]: return [f"missing columns: {missing}"]
    errors=[]
    if df.snapshot_id.duplicated().any(): errors.append("duplicate snapshot_id")
    if (df.odds_american==0).any(): errors.append("American odds cannot be zero")
    if not set(df.market.dropna()).issubset({"spread","total"}): errors.append("unsupported market")
    return errors
