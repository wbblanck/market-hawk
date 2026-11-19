"""Sharp-book market data (Pinnacle, Circa, BookMaker). See docs/data_sources/sharp_feeds.md.

Only sources that are free and whose terms allow our use, or that Will already holds a key
for, get code here: the live odds-api-v4 feeds (The Odds API free Starter tier, and ParlayAPI
via ``parlay_api_provider``). Keys come from env only.
Circa and BookMaker have no free, terms-compatible source; nothing here scrapes any site.
"""
from .odds_api import OddsApiProvider, parlay_api_provider, parse_odds_api_events

__all__ = ["OddsApiProvider", "parlay_api_provider", "parse_odds_api_events"]
