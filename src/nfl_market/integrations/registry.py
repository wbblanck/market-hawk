PROVIDERS=[
 {"name":"nflverse","status":"available_dataset","url":"https://github.com/nflverse/nflverse-data","use":"Schedules, play-by-play, team context"},
 {"name":"nflfastR","status":"available_dataset","url":"https://github.com/nflverse/nflfastR","use":"Play-by-play analytics"},
 {"name":"nflreadpy","status":"available_library","url":"https://github.com/nflverse/nflreadpy","use":"Python nflverse access"},
 {"name":"nflseedR","status":"existing_project_dependency","url":"https://github.com/nflverse/nflseedR","use":"Existing project schedule support"},
 {"name":"ESPN","status":"research_source","url":"https://www.espn.com/nfl/","use":"Schedule/score context"},
 {"name":"Pinnacle","status":"planned_adapter","url":None,"use":"Sharp benchmark; credentials/history/terms unverified"},
 {"name":"Novig","status":"planned_adapter","url":"https://novig.com/","use":"Exchange quote/execution research"},
 {"name":"Kalshi","status":"planned_adapter","url":"https://kalshi.com/","use":"Event-contract comparison"},
 {"name":"Polymarket US","status":"planned_adapter","url":"https://polymarket.us/","use":"Event-contract comparison"},
 {"name":"Action Network","status":"research_source","url":"https://www.actionnetwork.com/","use":"Market context"},
 {"name":"OpenMarkets","status":"research_source","url":"https://openmarkets.io/","use":"Odds/data research"},
 {"name":"ParlayAPI","status":"research_source","url":"https://parlayapi.com/","use":"Odds API exploration"}
]
def provider_registry(): return PROVIDERS
