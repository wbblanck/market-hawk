import pandas as pd
def assert_point_in_time(df: pd.DataFrame,prediction_time: str,available_time: str):
    bad=df[available_time]>df[prediction_time]
    if bad.any(): raise ValueError(f"Point-in-time leakage in {int(bad.sum())} rows")
def add_market_features(df: pd.DataFrame)->pd.DataFrame:
    if df.empty:return df.copy()
    x=df.sort_values("captured_at_utc").copy()
    x["book_count_contract"]=x.groupby(["event_id","market","selection","line"]).book.transform("nunique")
    x["implied_prob"]=x.odds_american.map(lambda o:100/(o+100) if o>0 else abs(o)/(abs(o)+100))
    x["market_dispersion"]=x.groupby(["event_id","market","selection","line"]).implied_prob.transform("std").fillna(0)
    return x
