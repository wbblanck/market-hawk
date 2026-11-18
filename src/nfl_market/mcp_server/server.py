from ..services.market_service import MarketService
def build_server():
    try: from mcp.server.fastmcp import FastMCP
    except ImportError as e: raise RuntimeError("Install with pip install -e '.[mcp]'") from e
    m=FastMCP("NFL Market Intelligence"); s=MarketService()
    @m.tool()
    def get_nfl_slate()->list[dict]:
        "Return events represented in the local quote store."
        return s.slate()
    @m.tool()
    def get_market_snapshot(event_id: str|None=None)->list[dict]:
        "Return stored quote snapshots, optionally for one event."
        d=s.quotes(event_id); return d.astype(object).where(d.notna(),None).to_dict("records")
    @m.tool()
    def get_resource_registry()->list[dict]:
        "Return provider resources and verification status."
        return s.providers()
    @m.tool()
    def get_system_health()->dict:
        "Return local database health."
        return s.health()
    @m.resource("nfl://slate/current")
    def current_slate()->str: return str(s.slate())
    @m.resource("nfl://data/providers")
    def providers()->str: return str(s.providers())
    return m
def main(): build_server().run()
if __name__=="__main__": main()
