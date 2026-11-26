"""Build the private paper-picks page (HTML) from a priced scratch DuckDB, with a plain-language
"why" dropdown under every game.

    python scripts/build_picks_page.py --db /tmp/picks.duckdb --out picks.html [--json picks.json]

Run after ``load_espn_odds.py`` and ``price_slate.py --record-bets`` on the same ``--db``. For every game the
dropdown refits ``ridge_v2`` (as of now, same rule as ``price_slate``), splits its margin and total prediction
into drivers (``nfl_market.models.explain``; contributions add up to the prediction) and compares model with
market. Paper only; the page says so and says what the model cannot see.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb
import numpy as np
import pandas as pd

from nfl_market.data.games import load_games
from nfl_market.models.explain import contributions, raw_facts, top_drivers

ET = ZoneInfo("America/New_York")


def dec(a: float) -> float:
    return 1 + a / 100 if a > 0 else 1 + 100 / -a


def odds(o) -> str:
    o = int(o)
    return f"+{o}" if o > 0 else str(o)


def pts(x: float) -> str:
    return f"{abs(x):.1f}"


def fair_line(sub: pd.DataFrame) -> float | None:
    """Line where the 'no push' win probability crosses 50% (linear interpolation on the priced ladder)."""
    sub = sub.sort_values("line").drop_duplicates("line")
    x, y = sub["line"].to_numpy(), sub["p_win_no_push"].to_numpy()
    for i in range(len(x) - 1):
        if (y[i] - .5) * (y[i + 1] - .5) <= 0 and y[i] != y[i + 1]:
            return float(x[i] + (.5 - y[i]) / (y[i + 1] - y[i]) * (x[i + 1] - x[i]))
    return None


def spread_text(home: str, away: str, line: float) -> str:
    if line < 0:
        return f"{home} -{abs(round(line, 1)):g}"
    if line > 0:
        return f"{away} -{abs(round(line, 1)):g}"
    return "PK"


def pick_text(market: str, sel: str, line, g: dict) -> str:
    if market == "total":
        return f"{sel.capitalize()} {line:g}"
    team = g["home"] if sel == "home" else g["away"]
    return f"{team} ML" if market == "moneyline" else f"{team} {line:+g}"


def collect(db: str) -> list[dict]:
    con = duckdb.connect(db, read_only=True)
    fp = con.execute("select * from fair_prices").df()
    q = con.execute("select * from quote_snapshots").df()
    pb = con.execute("select * from paper_bets").df()
    sched = load_games().set_index("game_id")
    out = []
    for gid, gq in q.groupby("event_id"):
        f = fp[fp.game_id == gid]
        ms = gq[(gq.market == "spread") & (gq.selection == "home")].iloc[0]
        mt = gq[(gq.market == "total") & (gq.selection == "over")].iloc[0]
        row = sched.loc[gid]
        g = dict(
            id=gid, away=row.away_team, home=row.home_team, kick=pd.Timestamp(gq.kickoff_utc.iloc[0]).tz_localize("UTC").isoformat(),
            fair_home_spread=fair_line(f[(f.market == "spread") & (f.selection == "home")]),
            mkt_home_spread=float(ms.line), mkt_spread_odds=int(ms.odds_american),
            fair_total=fair_line(f[(f.market == "total") & (f.selection == "over")]),
            mkt_total=float(mt.line), p_home=float(f[(f.market == "moneyline") & (f.selection == "home")].p_win_no_push.iloc[0]),
        )
        cand = []
        for _, r in gq.iterrows():
            m = f[(f.market == r.market) & (f.selection == r.selection)]
            if r.market != "moneyline":
                m = m[np.isclose(m.line, r.line)]
            if m.empty:
                continue
            m = m.iloc[0]
            cand.append(dict(market=r.market, sel=r.selection, line=None if pd.isna(r.line) else float(r.line),
                             odds=int(r.odds_american), p=float(m.p_win),
                             ev=float(m.p_win * (dec(int(r.odds_american)) - 1) - m.p_loss)))
        g["cand"] = cand
        g["bets"] = [dict(market=b.market, sel=b.selection, line=None if pd.isna(b.line) else float(b.line),
                          odds=int(b.odds_american), p=float(b.p_win), ev=float(b.ev_per_unit), stake=float(b.stake_units))
                     for b in pb[pb.game_id == gid].itertuples()]
        out.append(g)
    return sorted(out, key=lambda g: g["kick"])


def explain_all(games_out: list[dict], as_of: pd.Timestamp) -> dict[str, dict]:
    """Refit ridge_v2 as of now and return drivers + raw facts per game id."""
    import price_slate as ps

    sched = load_games()
    ids = [g["id"] for g in games_out]
    test = sched[sched.game_id.isin(ids)].copy()
    test["home_score"] = np.nan
    test["away_score"] = np.nan
    test["label_available_at_utc"] = pd.NaT
    test = test.reset_index(drop=True)
    train = sched[sched["label_available_at_utc"].notna() & (sched["label_available_at_utc"] < as_of)]
    model = ps.make_model("ridge_v2").fit(train.reset_index(drop=True))
    C = contributions(model, test)
    R = raw_facts(model, test)
    res = {}
    for i, gid in enumerate(test.game_id):
        res[gid] = dict(
            margin=C["margin"].iloc[i].to_dict(), total=C["total"].iloc[i].to_dict(),
            facts={k: (None if pd.isna(v) else float(v)) for k, v in R.iloc[i].items()},
            sd_margin=float(model.inner_["margin"][1]), sd_total=float(model.inner_["total"][1]))
    return res


def _fact_line(group: str, g: dict, f: dict) -> str:
    h, a = g["home"], g["away"]
    v = lambda k: f.get(k)
    try:
        if group.startswith("Play-by-play"):
            return (f"Offence EPA per play over the last 8 games: {h} {v('home_off_epa_l8'):+.2f}, {a} {v('away_off_epa_l8'):+.2f}. "
                    f"Defence (lower is better): {h} {v('home_def_epa_l8'):+.2f}, {a} {v('away_def_epa_l8'):+.2f}.")
        if group.startswith("Quarterback"):
            out = [t for t, k in ((h, "home_qb_out"), (a, "away_qb_out")) if v(k) == 1.0]
            status = (f"{' and '.join(out)}'s expected starter is listed Out or Doubtful." if out
                      else "Neither expected starter is listed Out or Doubtful.")
            return f"{status} Recent EPA per dropback: {h} {v('home_qb_epa'):+.2f}, {a} {v('away_qb_epa'):+.2f}."
        if group.startswith("Team strength"):
            return f"Elo ratings: {h} {v('home_elo'):.0f}, {a} {v('away_elo'):.0f}."
        if group.startswith("Recent scoring"):
            return (f"Last 8 games, points scored / allowed: {h} {v('home_pf_l8'):.1f} / {v('home_pa_l8'):.1f}, "
                    f"{a} {v('away_pf_l8'):.1f} / {v('away_pa_l8'):.1f}.")
        if group.startswith("Rest"):
            return f"Days of rest: {h} {v('home_rest'):.0f}, {a} {v('away_rest'):.0f}. Home field is built into this group."
    except (TypeError, ValueError):
        return ""
    return ""


def why_html(g: dict, ex: dict) -> str:
    h, a = g["home"], g["away"]
    m, t, f = ex["margin"], ex["total"], ex["facts"]
    pm, pt = m["prediction"], t["prediction"]
    favored = h if pm >= 0 else a
    parts = []
    # Spread
    mk = spread_text(h, a, g["mkt_home_spread"])
    fs = spread_text(h, a, g["fair_home_spread"]) if g["fair_home_spread"] is not None else "n/a"
    gap_s = (g["mkt_home_spread"] - g["fair_home_spread"]) if g["fair_home_spread"] is not None else 0.0
    side = h if gap_s > 0 else a  # market home line higher than fair means the market likes home less
    parts.append(f"<h4>Who wins and by how much</h4><p>The model expects {favored} to win by about {pts(pm)} points on average. "
                 f"Its fair spread is {fs}; the market has {mk}. "
                 + (f"That is a {pts(gap_s)}-point gap, with the model liking {side} more than the market does." if abs(gap_s) >= 0.5
                    else "The two are within half a point.") + "</p>")
    items = []
    for name, val in top_drivers(pd.Series(m), 3):
        toward = h if val > 0 else a
        fl = _fact_line(name, g, f)
        items.append(f"<li><b>{html.escape(name)}</b>: {abs(val):.1f} points toward {toward}. {html.escape(fl)}</li>")
    base = m["intercept"]
    parts.append(f"<ul>{''.join(items)}</ul><p class=\"fine\">Starting point: the average game in the training data is won by the home team by {base:.1f} points. "
                 "The drivers shown are the three biggest; the rest are small. The fair spread differs a little from the average margin because final margins bunch up on numbers like 3 and 7.</p>")
    # Total
    gap_t = g["fair_total"] - g["mkt_total"] if g["fair_total"] is not None else 0.0
    parts.append(f"<h4>How many points</h4><p>The model expects {pt:.1f} total points; the market has {g['mkt_total']:g}. "
                 + (f"The model is {pts(gap_t)} points {'higher' if gap_t > 0 else 'lower'}." if abs(gap_t) >= 0.5 else "They are within half a point.") + "</p>")
    items = []
    for name, val in top_drivers(pd.Series(t), 3):
        items.append(f"<li><b>{html.escape(name)}</b>: {'adds' if val > 0 else 'takes off'} {abs(val):.1f} points. {html.escape(_fact_line(name, g, f))}</li>")
    parts.append(f"<ul>{''.join(items)}</ul><p class=\"fine\">Starting point: the average game in the training data has {t['intercept']:.1f} total points.</p>")
    # Bets
    if g["bets"]:
        bl = "".join(f"<li><b>{html.escape(pick_text(b['market'], b['sel'], b['line'], g))} {odds(b['odds'])}</b>: the model gives it {b['p']*100:.0f}% to win. "
                     f"At that price you need {100/dec(b['odds']):.0f}% to break even, so the paper edge is {b['ev']*100:+.0f}% per unit staked.</li>" for b in g["bets"])
        parts.append(f"<h4>Why a paper bet</h4><ul>{bl}</ul>")
    big = abs(gap_s) >= 3 or abs(gap_t) >= 3
    parts.append("<p class=\"fine\">" + ("A gap this big usually means the market knows something the model does not, such as weather, an injury that is not a quarterback, "
                 "or a late lineup change. " if big else "") +
                 "The model cannot see weather, injuries beyond the listed starting quarterback, or anything from the 2025 season (held out on purpose). "
                 "It has not beaten the closing line in any backtest.</p>")
    return "".join(parts)


CSS = """
:root{--bg:#f5f7f6;--surface:#ffffff;--fg:#17201c;--muted:#5d6b64;--line:#d9e0dc;--accent:#1f6f4f;--warn-bg:#fff4dc;--warn-fg:#6b4a00;--gap:#b4501f;--tonight:#e8f3ed;
--display:"Archivo",system-ui,sans-serif;--body:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#111613;--surface:#182019;--fg:#e4ebe7;--muted:#9aa8a1;--line:#2b3630;--accent:#6fcf9f;--warn-bg:#332811;--warn-fg:#f2cf80;--gap:#f09a6c;--tonight:#1b2b22;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#111613;--surface:#182019;--fg:#e4ebe7;--muted:#9aa8a1;--line:#2b3630;--accent:#6fcf9f;--warn-bg:#332811;--warn-fg:#f2cf80;--gap:#f09a6c;--tonight:#1b2b22;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--body);line-height:1.5;margin:0}
.wrap{max-width:1040px;margin:0 auto;padding-block:28px 48px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:20px}
h1{font-family:var(--display);font-weight:800;font-size:1.9rem;margin:0;letter-spacing:-.01em;text-wrap:balance}
.sub{color:var(--muted);margin:4px 0 0;max-width:68ch}
.warn{background:var(--warn-bg);color:var(--warn-fg);border-radius:8px;padding:14px 16px;max-width:80ch}
.warn b{display:block;margin-bottom:4px}.warn p{margin:6px 0 0}
.tbl{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;min-width:760px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}
thead th{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600}
tr.day th{font-family:var(--display);font-size:.95rem;background:var(--bg);color:var(--fg)}
td.match b{display:block}.t{color:var(--muted);font-size:.85rem}
td{font-family:var(--mono);font-size:.9rem}td.match,td.bets,td.why{font-family:var(--body)}
.o{color:var(--muted);font-size:.8rem}td.gap{color:var(--gap);font-weight:500}
tr.tonight td{background:var(--tonight)}
tr.game td{border-bottom:none}tr.whyrow td{padding-top:0}
.bet{display:flex;flex-wrap:wrap;gap:2px 8px;align-items:baseline}.bet+.bet{margin-top:6px}
.pick{font-weight:600}.odds{font-family:var(--mono)}.meta{color:var(--muted);font-size:.8rem;flex-basis:100%}
.none{color:var(--muted);font-size:.85rem}
details summary{cursor:pointer;color:var(--accent);font-weight:500;font-size:.88rem;padding:2px 0}
details .body{padding:6px 0 4px;max-width:78ch;font-size:.9rem}
details h4{font-family:var(--display);font-size:.9rem;margin:12px 0 2px}
details p{margin:4px 0}details ul{margin:4px 0;padding-left:20px}details li+li{margin-top:4px}
.fine{color:var(--muted);font-size:.82rem}
.notes{color:var(--muted);font-size:.88rem;max-width:80ch}.notes li+li{margin-top:4px}
"""


def build(games_out: list[dict], explain: dict[str, dict], stamp: str) -> str:
    rows, last = [], None
    for g in games_out:
        kick = pd.Timestamp(g["kick"])
        ko = kick.astimezone(ET)
        day, tm = ko.strftime("%a %b %-d"), ko.strftime("%-I:%M %p ET")
        if day != last:
            rows.append(f'<tr class="day"><th colspan="6">{day}</th></tr>')
            last = day
        gap_s = abs(g["fair_home_spread"] - g["mkt_home_spread"]) >= 3 if g["fair_home_spread"] is not None else False
        gap_t = abs(g["fair_total"] - g["mkt_total"]) >= 3 if g["fair_total"] is not None else False
        cell = "".join(
            f'<div class="bet"><span class="pick">{pick_text(b["market"], b["sel"], b["line"], g)}</span> <span class="odds">{odds(b["odds"])}</span>'
            f'<span class="meta">model {b["p"]*100:.1f}% · EV {b["ev"]*100:+.1f}% · 1u</span></div>' for b in g["bets"])
        started = pd.Timestamp.now(tz="UTC") >= kick - pd.Timedelta(minutes=15)
        if not cell and started:
            top = sorted(g["cand"], key=lambda z: -z["ev"])[:2]
            lean = "; ".join(f'{pick_text(c["market"], c["sel"], c["line"], g)} {odds(c["odds"])} (EV {c["ev"]*100:+.0f}%)' for c in top)
            cell = (f'<span class="none">Inside the 15-minute pre-kickoff window, so no paper bet can be recorded. Model leaned {lean}.</span>')
        elif not cell:
            cell = '<span class="none">No bet (edge under 2%)</span>'
        why = ""
        if g["id"] in explain:
            why = f'<details><summary>Why the model says this</summary><div class="body">{why_html(g, explain[g["id"]])}</div></details>'
        fs = spread_text(g["home"], g["away"], g["fair_home_spread"]) if g["fair_home_spread"] is not None else "n/a"
        rows.append(
            f'<tr class="game"><td class="match"><b>{g["away"]} @ {g["home"]}</b><span class="t">{tm}</span></td>'
            f'<td{" class=gap" if gap_s else ""}>{fs}</td>'
            f'<td>{spread_text(g["home"], g["away"], g["mkt_home_spread"])} <span class="o">{odds(g["mkt_spread_odds"])}</span></td>'
            f'<td{" class=gap" if gap_t else ""}>{g["fair_total"]:.1f}</td><td>{g["mkt_total"]:g}</td><td class="bets">{cell}</td></tr>'
            f'<tr class="whyrow"><td colspan="6" class="why">{why}</td></tr>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Market Hawk Week 12</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Archivo:wght@700;800&display=swap">
<style>{CSS}</style></head><body><div class="wrap">
<header><h1>Market Hawk · paper picks</h1>
<p class="sub">{html.escape(stamp)} Model: ridge_v2 (adds nflverse EPA form and listed-starter QB status). Model fair lines against DraftKings prices (via ESPN). Paper bets only, quarter-Kelly capped at 1 unit on a 100-unit bankroll. Open "Why the model says this" under any game for the reasoning.</p></header>
<div class="warn"><b>Treat these as a test of the plumbing, not as betting advice.</b>
<p>In the 2015 to 2024 backtest the model is a little better than the old one but still not better than the closing line. It does not see weather or anything from the 2025 season (held out on purpose), and the injury data is whatever nflverse had at pricing time.</p>
<p>The edges below are far too large to be real in NFL markets. They mostly show where the market knows something the model does not. Orange cells mark a gap of 3+ points from the market.</p></div>
<div class="tbl"><table>
<thead><tr><th>Game</th><th>Model spread</th><th>Market spread</th><th>Model total</th><th>Market total</th><th>Paper bet</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<ul class="notes">
<li>Spreads and totals both come from ridge_v2, trained on 1999 to 2024 plus completed 2026 games.</li>
<li>The staking rules allow one side bet per game (spread or moneyline, whichever has the higher EV) plus one total. Underdog moneylines win that contest here, which is why no spread bets appear.</li>
<li>Prices move. A quote older than 3 hours is too stale to stake by the model's own rules, and no bet can be recorded within 15 minutes of kickoff, so re-run before each kickoff.</li>
</ul></div></body></html>"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None, help="also write the per-game data (spreads, totals, bets, drivers)")
    a = ap.parse_args(argv)
    now = pd.Timestamp.now(tz="UTC")
    games_out = collect(a.db)
    explain = explain_all(games_out, now)
    stamp = f"Updated {now.astimezone(ET).strftime('%a %b %-d at %-I:%M %p ET')}."
    Path(a.out).write_text(build(games_out, explain, stamp))
    if a.json:
        Path(a.json).write_text(json.dumps([dict(g, explain=explain.get(g["id"])) for g in games_out], indent=1, default=str))
    print(f"wrote {a.out}: {len(games_out)} games, {sum(len(g['bets']) for g in games_out)} paper bets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
