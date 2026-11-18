def line_clv(selection: str, bet_line: float, close_line: float) -> float:
    # Positive is favorable. Spread lines are from the selected team's perspective.
    side=selection.lower()
    if side in {"home","away","favorite","underdog"}: return bet_line-close_line
    if side=="over": return close_line-bet_line
    if side=="under": return bet_line-close_line
    raise ValueError(f"Unsupported selection: {selection}")
def probability_clv(bet_fair_probability: float, close_fair_probability: float) -> float:
    return close_fair_probability-bet_fair_probability
