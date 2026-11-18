def best_price_same_contract(frame):
    if frame.empty:return None
    x=frame.copy()
    x["decimal_odds"]=x.odds_american.map(lambda o:1+(o/100 if o>0 else 100/abs(o)))
    return x.sort_values("decimal_odds",ascending=False).iloc[0]
def net_ev(probability,decimal_odds,fee_rate=0.0):
    if not 0<probability<1 or decimal_odds<=1: raise ValueError("Invalid input")
    return probability*(decimal_odds-1)-(1-probability)-fee_rate
