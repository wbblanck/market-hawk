from dataclasses import dataclass

import pandas as pd

CONTRACT_COLUMNS = ["event_id", "market", "selection", "line"]


@dataclass(frozen=True)
class Contract:
    event_id: str
    market: str
    selection: str
    line: float

    def key(self):
        return (self.event_id, self.market, self.selection.lower(), round(self.line, 3))


def same_contract(a: Contract, b: Contract) -> bool:
    return a.key() == b.key()


def contract_label(market: str, selection: str, line: float) -> str:
    """Human label for one exact contract, e.g. 'spread home -2.5' or 'total over 44.5'."""
    line = float(line)
    shown = f"{line:+g}" if market == "spread" else f"{line:g}"
    return f"{market} {selection.lower()} {shown}"


def with_contract_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``contract_key`` and ``contract_label`` columns (event + market + selection + exact line)."""
    x = frame.copy()
    if x.empty:
        x["contract_key"] = pd.Series(dtype=str)
        x["contract_label"] = pd.Series(dtype=str)
        return x
    x["contract_label"] = [contract_label(m, s, l) for m, s, l in zip(x.market, x.selection, x.line)]
    x["contract_key"] = x.event_id.astype(str) + "|" + x.contract_label
    return x


def selection_view(frame: pd.DataFrame, event_id: str, market: str, selection: str) -> pd.DataFrame:
    """Quotes for one side of one market of one event, so different sides are never mixed."""
    sel = selection.lower()
    x = frame[(frame.event_id == event_id) & (frame.market == market) & (frame.selection.str.lower() == sel)]
    return with_contract_keys(x).sort_values("captured_at_utc")


def contract_view(frame: pd.DataFrame, event_id: str, market: str, selection: str, line: float) -> pd.DataFrame:
    """Quotes for exactly one contract. A -2.5 and a -3 are different contracts."""
    x = selection_view(frame, event_id, market, selection)
    return x[(x.line - float(line)).abs() < 1e-9]
