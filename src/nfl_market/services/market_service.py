from ..config import settings
from ..data.store import MarketStore
from ..integrations.registry import provider_registry
class MarketService:
    def __init__(self): self.store=MarketStore(settings.database_path)
    def quotes(self,event_id=None): return self.store.quotes(event_id)
    def slate(self):
        q=self.store.quotes()
        return [] if q.empty else q[["event_id","season","week","kickoff_utc"]].drop_duplicates("event_id").to_dict("records")
    def providers(self): return provider_registry()
    def health(self): return {"status":"ok","quote_rows":len(self.store.quotes()),"provider_count":len(provider_registry())}
