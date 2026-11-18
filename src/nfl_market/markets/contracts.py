from dataclasses import dataclass
@dataclass(frozen=True)
class Contract:
    event_id: str
    market: str
    selection: str
    line: float
    def key(self): return (self.event_id,self.market,self.selection.lower(),round(self.line,3))
def same_contract(a: Contract,b: Contract) -> bool: return a.key()==b.key()
