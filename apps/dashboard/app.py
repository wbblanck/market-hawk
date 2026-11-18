import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
import streamlit as st
import plotly.express as px
import pandas as pd
from nfl_market.services.market_service import MarketService
st.set_page_config(page_title="NFL Market Intelligence",layout="wide")
s=MarketService(); q=s.quotes()
st.title("NFL Market Intelligence")
st.caption("Pre-game spreads & full-game totals | Research terminal | No automated wagering")
h=s.health(); a,b,c,d=st.columns(4)
a.metric("Quote snapshots",f"{h['quote_rows']:,}"); b.metric("Events",q.event_id.nunique() if not q.empty else 0)
c.metric("Books",q.book.nunique() if not q.empty else 0); d.metric("Provider registry",h["provider_count"])
tabs=st.tabs(["Command Center","Market Explorer","CLV & Models","Data & Resources","MCP"])
with tabs[0]:
    st.subheader("Slate / command center")
    if q.empty: st.info("No quotes loaded. Run python scripts/load_demo_data.py for synthetic demo data.")
    else:
        ev=st.selectbox("Event",sorted(q.event_id.unique())); st.dataframe(q[q.event_id==ev].sort_values("captured_at_utc",ascending=False),use_container_width=True)
with tabs[1]:
    st.subheader("Market explorer")
    if not q.empty:
        ev=st.selectbox("Event",sorted(q.event_id.unique()),key="ev2")
        mk=st.selectbox("Market",sorted(q[q.event_id==ev].market.unique()))
        v=q[(q.event_id==ev)&(q.market==mk)]
        st.plotly_chart(px.line(v,x="captured_at_utc",y="line",color="book",line_group="selection",markers=True),use_container_width=True)
        st.dataframe(v,use_container_width=True)
    else: st.info("Load quote snapshots to explore movement.")
with tabs[2]:
    st.subheader("CLV and model research")
    st.warning("No model is promoted by default. Closing labels require a timestamped, matched contract.")
    st.markdown("Keep line CLV, price CLV, fair probability, net EV and execution risk distinct. Validate chronologically; report calibration and uncertainty.")
with tabs[3]:
    st.subheader("Provider registry")
    st.dataframe(pd.DataFrame(s.providers()),use_container_width=True)
    st.caption("Registry entries do not prove current API access, historical depth, permissions or fills.")
with tabs[4]:
    st.subheader("MCP")
    st.code("pip install -e '.[mcp]'\npython -m nfl_market.mcp_server.server")
    st.write("Read-only tools: get_nfl_slate, get_market_snapshot, get_resource_registry, get_system_health.")
