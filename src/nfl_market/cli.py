import argparse
from .services.market_service import MarketService
def main():
    p=argparse.ArgumentParser(); p.add_argument("command",choices=["health","providers","slate"]); a=p.parse_args(); s=MarketService()
    print({"health":s.health,"providers":s.providers,"slate":s.slate}[a.command]())
