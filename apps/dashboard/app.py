"""Market Hawk research dashboard (read-only).

Every number comes from :class:`AnalyticsService` / :class:`MarketService`; this file only
formats and filters what they return (docs/DASHBOARD_SPEC.md). No wager placement.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from nfl_market.markets.contracts import (  # noqa: E402
    contract_view,
    selection_view,
    with_contract_keys,
)
from nfl_market.services.analytics_service import (  # noqa: E402
    MAX_LIMIT,
    PENDING_TIER,
    AnalyticsService,
    any_synthetic,
    ev_plus_eligible,
    tier_label,
)
from nfl_market.services.market_service import MarketService  # noqa: E402

TAB_LABELS = ["Command Center", "Edges", "Market Explorer", "CLV", "Model Health", "Data Health",
              "Data & Resources", "MCP"]
SYNTHETIC_BANNER = ("SYNTHETIC DEMO DATA: some rows below are synthetic (synthetic=True). "
                    "They are for demonstrating the dashboard only and must not be used for betting decisions.")
EDGE_DISPLAY = ["captured_at_utc", "quote_age_minutes", "game_id", "kickoff_utc", "book", "market", "selection",
                "line", "odds_american", "opposite_odds_american", "model_id", "dist_id", "p_win", "p_push",
                "p_win_no_push", "breakeven_prob", "market_fair_prob", "edge_prob", "edge_vs_market", "ev_per_unit",
                "flags", "synthetic"]
BET_DISPLAY = ["decision_time_utc", "bet_id", "game_id", "season", "week", "book", "market", "selection", "line",
               "odds_american", "model_id", "dist_id", "stake_units", "stake_limit_reason", "status",
               "line_close", "line_close_tier", "line_clv", "price_close_fair_prob", "price_close_tier", "price_clv",
               "close_flags", "result", "pnl_units", "synthetic"]
HEALTH_DISPLAY = ["generated_at_utc", "model_id", "scope", "fold_id", "n_games", "mae_margin", "mae_total",
                  "logloss_win", "brier_win", "ece_win", "cover_logloss", "stack_weight_margin", "stack_weight_total",
                  "experiment_id"]


# ------------------------------------------------------------------ formatting only
def missing_as_text(series: pd.Series, text: str = "missing") -> pd.Series:
    """Show NaN as ``text`` (a missing close is missing, not 0)."""
    return series.astype(object).where(series.notna(), text)


def tier_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(tier_label)


def synthetic_banner(*frames: pd.DataFrame) -> None:
    if any_synthetic(*frames):
        st.error(SYNTHETIC_BANNER)


# ------------------------------------------------------------------------- tabs
def command_center(q: pd.DataFrame) -> None:
    st.subheader("Slate / command center")
    if q.empty:
        st.info("No quotes loaded. Run python scripts/load_demo_data.py for synthetic demo data.")
        return
    synthetic_banner(q)
    ev = st.selectbox("Event", sorted(q.event_id.unique()))
    e = with_contract_keys(q[q.event_id == ev]).sort_values(["market", "selection", "captured_at_utc"],
                                                            ascending=[True, True, False])
    for mk, grp in e.groupby("market"):
        st.markdown(f"**{mk.title()}**")
        st.dataframe(grp, width="stretch")


def edges_tab(a: AnalyticsService) -> None:
    st.subheader("Edges")
    st.caption("Model probability vs the stored quote, latest quote per book and contract. EV is per 1 unit staked; "
               "a push returns the stake. A displayed quote is not a fill. No wager placement.")
    c1, c2, c3 = st.columns(3)
    use_min = c1.checkbox("Filter by minimum EV", value=False)
    min_ev = c2.number_input("Minimum EV per unit", value=0.0, step=0.01, format="%.3f") if use_min else None
    limit = int(c3.number_input("Max rows", min_value=1, max_value=MAX_LIMIT, value=50, step=10))
    e = a.edges(min_ev=min_ev, limit=limit)
    synthetic_banner(e)
    if e.empty:
        st.info("No edges: needs stored quotes and fair prices (scripts/price_slate.py, or the synthetic demo loader).")
        return
    f1, f2, f3 = st.columns(3)
    markets = f1.multiselect("Market", sorted(e.market.dropna().unique()))
    books = f2.multiselect("Book", sorted(e.book.dropna().unique()))
    clean_only = f3.checkbox("Hide flagged rows", value=False)
    view = e.assign(ev_plus=ev_plus_eligible(e))
    if markets:
        view = view[view.market.isin(markets)]
    if books:
        view = view[view.book.isin(books)]
    if clean_only:
        view = view[view.flags.fillna("").astype(str).str.strip() == ""]
    view = view.assign(ev_plus=view.ev_plus.map({True: "EV+", False: ""}))
    st.dataframe(view[["ev_plus", *EDGE_DISPLAY]], width="stretch", hide_index=True)
    st.caption("EV+ is shown only for rows with positive EV, no quality flags and a non-synthetic quote. "
               "Execution costs and fills are not modelled.")


def clv_tab(a: AnalyticsService) -> None:
    st.subheader("CLV (paper bet log)")
    st.caption("Line CLV is in points, price CLV in probability; they are separate and never summed. Positive is "
               "favourable to the bettor. Each value shows the close tier it came from; a missing close is shown as "
               "missing, never 0. all_book_median is the weak fallback and is reported separately.")
    all_bets = a.bet_log(limit=MAX_LIMIT)
    seasons = sorted(int(s) for s in all_bets.season.dropna().unique())
    pick = st.selectbox("Season", ["All", *seasons])
    bets = all_bets if pick == "All" else a.bet_log(season=int(pick), limit=MAX_LIMIT)
    summary = a.clv_summary()
    synthetic_banner(bets)
    if bets.empty:
        st.info("No paper bets recorded yet. Real CLV exists only for games captured going forward.")
    else:
        b = bets.assign(line_close_tier=tier_text(bets.line_close_tier),
                        price_close_tier=tier_text(bets.price_close_tier),
                        line_clv=missing_as_text(bets.line_clv), price_clv=missing_as_text(bets.price_clv),
                        line_close=missing_as_text(bets.line_close),
                        price_close_fair_prob=missing_as_text(bets.price_close_fair_prob))
        b.loc[bets.line_close_tier.fillna("").astype(str) == "", ["line_clv", "line_close"]] = PENDING_TIER
        b.loc[bets.price_close_tier.fillna("").astype(str) == "", ["price_clv", "price_close_fair_prob"]] = PENDING_TIER
        st.dataframe(b[BET_DISPLAY], width="stretch", hide_index=True)
    st.markdown("**CLV summary by close tier**")
    if summary.empty:
        st.info("No CLV summary yet.")
        return
    weak = summary.is_weak_fallback.fillna(False).astype(bool)
    missing = summary.close_tier.astype(str) == "missing"
    strong = summary[~weak & ~missing]
    st.markdown("Pinnacle and sharp median closes")
    st.dataframe(strong, width="stretch", hide_index=True)
    st.markdown("All-book median: **weak fallback** (reported separately; FEW_BOOKS below 5 books)")
    st.dataframe(summary[weak].assign(close_tier=tier_text(summary[weak].close_tier)), width="stretch",
                 hide_index=True)
    if missing.any():
        st.markdown("Missing closes (no CLV computed)")
        st.dataframe(summary[missing], width="stretch", hide_index=True)


def model_health_tab(a: AnalyticsService) -> None:
    st.subheader("Model health")
    h = a.model_health()
    if h.empty:
        st.info("No model health file yet (docs/results/step4_model_health.json, written by the stacking script).")
        return
    st.caption(f"Generated {h.generated_at_utc.astype(str).max()} | experiment {', '.join(sorted(h.experiment_id.astype(str).unique()))}. "
               "Walk-forward out-of-fold metrics on 2015-2024 test seasons; the 2025 holdout is never scored here. "
               "Model-health rows carry no quality flags column.")
    for scope, title in (("pooled", "Pooled"), ("clean", "Clean seasons (no ported-constant tuning)"),
                         ("fold", "Per fold (test season)")):
        st.markdown(f"**{title}**")
        st.dataframe(h[h.scope == scope][HEALTH_DISPLAY], width="stretch", hide_index=True)
    w = h[(h.scope == "pooled") & (h.stack_weight_margin.notna() | h.stack_weight_total.notna())]
    st.markdown("**Stack weights (latest walk-forward)**")
    st.dataframe(w[["generated_at_utc", "model_id", "stack_weight_margin", "stack_weight_total", "experiment_id"]],
                 width="stretch", hide_index=True)
    pooled = h[h.scope == "pooled"]
    st.markdown("**Calibration (pooled)**")
    st.dataframe(pooled[["generated_at_utc", "model_id", "ece_win", "brier_win", "logloss_win", "cover_logloss"]],
                 width="stretch", hide_index=True)
    if not pooled.empty and pooled.ece_win.notna().any():
        st.plotly_chart(px.bar(pooled, x="model_id", y="ece_win", labels={"ece_win": "ECE of P(home win)"}),
                        width="stretch")


def market_explorer(q: pd.DataFrame) -> None:
    st.subheader("Market explorer")
    st.caption("Each chart shows one side of one market. Prices are only compared within one exact contract (same line).")
    if q.empty:
        st.info("Load quote snapshots to explore movement.")
        return
    synthetic_banner(q)
    ev = st.selectbox("Event", sorted(q.event_id.unique()), key="ev2")
    mk = st.selectbox("Market", sorted(q[q.event_id == ev].market.unique()))
    sel = st.selectbox("Selection", sorted(q[(q.event_id == ev) & (q.market == mk)].selection.str.lower().unique()))
    v = selection_view(q, ev, mk, sel)
    line_label = "Selected team's spread" if mk == "spread" else "Total"
    st.markdown(f"**Line movement: {mk} {sel}**")
    st.plotly_chart(px.line(v, x="captured_at_utc", y="line", color="book", markers=True, labels={"line": line_label}),
                    width="stretch")
    lines = sorted(v.line.unique())
    ln = st.selectbox("Exact line (contract) for price comparison", lines,
                      format_func=lambda x: f"{x:+g}" if mk == "spread" else f"{x:g}")
    c = contract_view(q, ev, mk, sel, ln)
    st.markdown(f"**Price history: {c.contract_label.iloc[0] if not c.empty else ''}**")
    st.plotly_chart(px.line(c, x="captured_at_utc", y="odds_american", color="book", markers=True), width="stretch")
    st.dataframe(v, width="stretch")


def data_health_tab(h: dict, overview: dict) -> None:
    st.subheader("Data health")
    st.markdown(f"Status: **{h['status']}** (checked {h.get('checked_at_utc')})")
    for w in h.get("warnings", []):
        st.warning(w)
    st.json(h)
    st.markdown("**Analytics overview**")
    st.json(overview)


def main() -> None:
    st.set_page_config(page_title="NFL Market Intelligence", layout="wide")
    s = MarketService()
    a = AnalyticsService(market_service=s)
    q = s.quotes()
    st.title("NFL Market Intelligence")
    st.caption("Pre-game spreads, totals & moneylines | Research terminal | Paper bets only | No automated wagering")
    h = s.health()
    o = a.overview()
    if h["status"] != "ok":
        st.warning(f"Data health: {h['status']}. " + "; ".join(h.get("warnings", [])))
    if o["synthetic_present"]:
        st.error(SYNTHETIC_BANNER)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Quote snapshots", f"{h['quote_rows']:,}")
    m2.metric("Events", q.event_id.nunique() if not q.empty else 0)
    m3.metric("Books", q.book.nunique() if not q.empty else 0)
    m4.metric("Paper bets", o["bet_count"])
    m5.metric("Provider registry", h["provider_count"])
    tabs = st.tabs(TAB_LABELS)
    with tabs[0]:
        command_center(q)
    with tabs[1]:
        edges_tab(a)
    with tabs[2]:
        market_explorer(q)
    with tabs[3]:
        clv_tab(a)
    with tabs[4]:
        model_health_tab(a)
    with tabs[5]:
        data_health_tab(h, o)
    with tabs[6]:
        st.subheader("Provider registry")
        st.dataframe(pd.DataFrame(s.providers()), width="stretch")
        st.caption("Registry entries do not prove current API access, historical depth, permissions or fills.")
    with tabs[7]:
        st.subheader("MCP")
        st.code("pip install -e '.[mcp]'\npython -m nfl_market.mcp_server.server  # stdio\n"
                "python -m nfl_market.mcp_server.server --transport streamable-http --port 8000")
        st.write("Read-only tools: get_nfl_slate, get_market_snapshot, get_resource_registry, get_system_health, "
                 "get_edges, get_bet_log, get_clv_summary, get_model_health, get_fair_prices, get_overview. "
                 "They return the same data as this dashboard (shared AnalyticsService).")


main()
