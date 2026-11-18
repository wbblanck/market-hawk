import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
import streamlit as st
import plotly.express as px
import pandas as pd
from nfl_market.markets.contracts import contract_view, selection_view, with_contract_keys
from nfl_market.services.market_service import MarketService
st.set_page_config(page_title="NFL Market Intelligence",layout="wide")
s=MarketService(); q=s.quotes()
st.title("NFL Market Intelligence")
st.caption("Pre-game spreads & full-game totals | Research terminal | No automated wagering")
h=s.health(); a,b,c,d=st.columns(4)
if h["status"]!="ok": st.warning(f"Data health: {h['status']}. "+"; ".join(h.get("warnings",[])))
a.metric("Quote snapshots",f"{h['quote_rows']:,}"); b.metric("Events",q.event_id.nunique() if not q.empty else 0)
c.metric("Books",q.book.nunique() if not q.empty else 0); d.metric("Provider registry",h["provider_count"])
tabs=st.tabs(["Command Center","Market Explorer","CLV & Models","Data & Resources","MCP"])
with tabs[0]:
    st.subheader("Slate / command center")
    if q.empty: st.info("No quotes loaded. Run python scripts/load_demo_data.py for synthetic demo data.")
    else:
        ev=st.selectbox("Event",sorted(q.event_id.unique()))
        e=with_contract_keys(q[q.event_id==ev]).sort_values(["market","selection","captured_at_utc"],ascending=[True,True,False])
        for mk,grp in e.groupby("market"):
            st.markdown(f"**{mk.title()}**")
            st.dataframe(grp,width="stretch")
with tabs[1]:
    st.subheader("Market explorer")
    st.caption("Each chart shows one side of one market. Prices are only compared within one exact contract (same line).")
    if not q.empty:
        ev=st.selectbox("Event",sorted(q.event_id.unique()),key="ev2")
        mk=st.selectbox("Market",sorted(q[q.event_id==ev].market.unique()))
        sel=st.selectbox("Selection",sorted(q[(q.event_id==ev)&(q.market==mk)].selection.str.lower().unique()))
        v=selection_view(q,ev,mk,sel)
        line_label="Selected team's spread" if mk=="spread" else "Total"
        st.markdown(f"**Line movement: {mk} {sel}**")
        st.plotly_chart(px.line(v,x="captured_at_utc",y="line",color="book",markers=True,labels={"line":line_label}),width="stretch")
        lines=sorted(v.line.unique())
        ln=st.selectbox("Exact line (contract) for price comparison",lines,format_func=lambda x:f"{x:+g}" if mk=="spread" else f"{x:g}")
        c=contract_view(q,ev,mk,sel,ln)
        st.markdown(f"**Price history: {c.contract_label.iloc[0] if not c.empty else ''}**")
        st.plotly_chart(px.line(c,x="captured_at_utc",y="odds_american",color="book",markers=True),width="stretch")
        st.dataframe(v,width="stretch")
    else: st.info("Load quote snapshots to explore movement.")
with tabs[2]:
    st.subheader("CLV and model research")
    st.warning("No model is promoted by default. Closing labels require a timestamped, matched contract.")
    st.markdown("Keep line CLV, price CLV, fair probability, net EV and execution risk distinct. Validate chronologically; report calibration and uncertainty.")
with tabs[3]:
    st.subheader("Provider registry")
    st.dataframe(pd.DataFrame(s.providers()),width="stretch")
    st.caption("Registry entries do not prove current API access, historical depth, permissions or fills.")
with tabs[4]:
    st.subheader("MCP")
    st.code("pip install -e '.[mcp]'\npython -m nfl_market.mcp_server.server  # stdio\npython -m nfl_market.mcp_server.server --transport streamable-http --port 8000")
    st.write("Read-only tools: get_nfl_slate, get_market_snapshot, get_resource_registry, get_system_health.")
