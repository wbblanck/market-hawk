def american_to_decimal(odds: int) -> float:
    if odds == 0: raise ValueError("American odds cannot be zero")
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))
def implied_probability(odds: int) -> float:
    if odds == 0: raise ValueError("American odds cannot be zero")
    return 100/(odds+100) if odds > 0 else abs(odds)/(abs(odds)+100)
def devig_two_way(a: int, b: int) -> tuple[float,float]:
    pa,pb=implied_probability(a),implied_probability(b)
    return pa/(pa+pb),pb/(pa+pb)
