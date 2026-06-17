"""
dashboard/app.py
SmartSales ML — Interactive Streamlit Dashboard

Run: streamlit run dashboard/app.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#  Config 
API_URL = os.environ.get("API_URL", "http://localhost:8000")

SEGMENT_COLORS = {
    "Champions":       "#10B981",
    "Loyal Customers": "#3B82F6",
    "At Risk":         "#F59E0B",
    "Lost / Inactive": "#EF4444",
}
RISK_COLORS = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}

st.set_page_config(
    page_title = "SmartSales ML Dashboard",
    page_icon  = "🛒",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

#  CSS tweaks 
st.markdown("""
<style>
  .metric-card {
    background: #F8FAFC; border-radius: 10px;
    padding: 16px 20px; border-left: 4px solid #3B82F6;
  }
  .risk-high   { color: #EF4444; font-weight: 700; }
  .risk-medium { color: #F59E0B; font-weight: 700; }
  .risk-low    { color: #10B981; font-weight: 700; }
  .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


#  API helpers 

@st.cache_data(ttl=300)
def fetch_segments():
    r = requests.get(f"{API_URL}/segments/all", timeout=10)
    r.raise_for_status()
    return pd.DataFrame(r.json())

@st.cache_data(ttl=300)
def fetch_churn():
    r = requests.get(f"{API_URL}/churn/all", timeout=10)
    r.raise_for_status()
    return pd.DataFrame(r.json())

@st.cache_data(ttl=300)
def fetch_history():
    r = requests.get(f"{API_URL}/sales/history", timeout=10)
    r.raise_for_status()
    return pd.DataFrame(r.json())

def call_forecast(horizon):
    r = requests.post(f"{API_URL}/forecast", json={"horizon_days": horizon}, timeout=15)
    r.raise_for_status()
    return r.json()

def call_segment(recency, frequency, monetary):
    r = requests.post(f"{API_URL}/segment",
                      json={"recency": recency, "frequency": frequency, "monetary": monetary},
                      timeout=10)
    r.raise_for_status()
    return r.json()

def call_churn(payload):
    r = requests.post(f"{API_URL}/churn", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


#  Sidebar 

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=60)
    st.title("SmartSales ML")
    st.caption("E-commerce Intelligence Dashboard")
    st.divider()

    st.markdown("**API Status**")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.success("🟢  API Connected")
        else:
            st.error("🔴  API Error")
    except Exception:
        st.error("🔴  API Unreachable")

    st.divider()
    st.markdown("**Models Running**")
    st.markdown("🔵  K-Means Segmentation")
    st.markdown("🟠  XGBoost Forecasting")
    st.markdown("🟢  LightGBM Churn")
    st.divider()
    st.caption("SmartSales ML · Capstone Project")


#  Header 

st.title("🛒  SmartSales ML Dashboard")
st.caption("Real-time e-commerce intelligence powered by machine learning")
st.divider()


#  Tabs 

tab1, tab2, tab3 = st.tabs([
    "👥  Customer Segments",
    "📈  Sales Forecast",
    "⚠️  Churn Risk",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Customer Segmentation
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Customer Segmentation")
    st.caption("RFM-based K-Means clustering — who are your customers?")

    try:
        df_seg = fetch_segments()

        #  KPI row 
        col1, col2, col3, col4 = st.columns(4)
        total = len(df_seg)
        for col, seg, emoji in zip(
            [col1, col2, col3, col4],
            ["Champions", "Loyal Customers", "At Risk", "Lost / Inactive"],
            ["🏆", "💙", "⚠️", "😴"]
        ):
            n = (df_seg["segment"] == seg).sum()
            col.metric(f"{emoji} {seg}", f"{n:,}", f"{n/total:.0%} of customers")

        st.divider()

        #  Scatter plot 
        col_l, col_r = st.columns([2, 1])
        with col_l:
            fig = px.scatter(
                df_seg, x="Recency", y="Monetary",
                color="segment", size="Frequency",
                color_discrete_map=SEGMENT_COLORS,
                hover_data=["CustomerID", "Frequency", "AvgOrderValue"],
                title="Customer Segments — Recency vs Monetary Value",
                labels={"Recency": "Recency (days)", "Monetary": "Lifetime Value (£)"},
                height=420,
            )
            fig.update_layout(legend_title_text="Segment",
                              plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            # Segment distribution donut
            seg_counts = df_seg["segment"].value_counts().reset_index()
            seg_counts.columns = ["segment", "count"]
            fig2 = px.pie(seg_counts, names="segment", values="count",
                          color="segment", color_discrete_map=SEGMENT_COLORS,
                          hole=0.5, title="Segment Distribution")
            fig2.update_traces(textposition="inside", textinfo="percent+label")
            fig2.update_layout(showlegend=False, height=420)
            st.plotly_chart(fig2, use_container_width=True)

        #  Segment stats table 
        st.subheader("Segment Profiles")
        summary = df_seg.groupby("segment").agg(
            Customers  = ("CustomerID",    "count"),
            Avg_Recency= ("Recency",       "median"),
            Avg_Freq   = ("Frequency",     "median"),
            Avg_LTV    = ("Monetary",      "median"),
        ).round(1).reset_index()
        summary.columns = ["Segment", "Customers", "Median Recency (days)",
                           "Median Orders", "Median LTV (£)"]
        st.dataframe(summary, use_container_width=True, hide_index=True)

        #  Lookup tool 
        st.divider()
        st.subheader("🔍  Segment a New Customer")
        with st.form("segment_form"):
            c1, c2, c3 = st.columns(3)
            r = c1.number_input("Recency (days)",  min_value=0.0,  value=30.0)
            f = c2.number_input("Frequency (orders)", min_value=1.0, value=5.0)
            m = c3.number_input("Monetary (£)",    min_value=0.0,  value=200.0)
            submitted = st.form_submit_button("Predict Segment →")

        if submitted:
            result = call_segment(r, f, m)
            seg_col = SEGMENT_COLORS.get(result["segment_name"], "#6B7280")
            st.markdown(f"""
            <div style='background:{seg_col}22; border-left: 5px solid {seg_col};
                        border-radius:8px; padding: 16px; margin-top:12px'>
            <h3 style='color:{seg_col}; margin:0'>{result["segment_name"]}</h3>
            <p style='margin:6px 0 0 0; color:#374151'>{result["description"]}</p>
            <p style='margin:6px 0 0 0; color:#6B7280'>
              RFM Scores — R: {result["rfm_scores"]["R"]}/5 &nbsp;|&nbsp;
              F: {result["rfm_scores"]["F"]}/5 &nbsp;|&nbsp;
              M: {result["rfm_scores"]["M"]}/5
            </p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Could not load segmentation data: {e}")
        st.info("Make sure the API is running: `uvicorn api.main:app --reload`")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Sales Forecast
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Sales Forecast")
    st.caption("XGBoost time-series regression — predict future revenue")

    col_ctrl, _ = st.columns([1, 3])
    horizon = col_ctrl.slider("Forecast horizon (days)", 7, 90, 30, step=7)

    try:
        history = fetch_history()
        df_hist = pd.DataFrame(history)
        df_hist["date"] = pd.to_datetime(df_hist["date"])

        fc = call_forecast(horizon)
        df_fc = pd.DataFrame({"date": pd.to_datetime(fc["dates"]), "revenue": fc["predicted"]})

        #  KPIs 
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Forecast Revenue", f"£{fc['total_forecast']:,.0f}")
        k2.metric("Avg Daily Revenue",      f"£{fc['avg_daily']:,.0f}")
        k3.metric("Forecast Period",         f"{fc['horizon_days']} days")
        k4.metric("Forecast From",           fc["forecast_from"])

        st.divider()

        #  Combined chart 
        fig = go.Figure()

        # Historical (last 90 days)
        df_recent = df_hist.tail(90)
        fig.add_trace(go.Scatter(
            x=df_recent["date"], y=df_recent["revenue"],
            name="Historical", line=dict(color="#3B82F6", width=2),
            hovertemplate="%{x|%b %d}<br>Revenue: £%{y:,.0f}<extra></extra>"
        ))

        # 7-day rolling average on historical
        df_recent = df_recent.copy()
        df_recent["roll7"] = df_recent["revenue"].rolling(7).mean()
        fig.add_trace(go.Scatter(
            x=df_recent["date"], y=df_recent["roll7"],
            name="7-day avg", line=dict(color="#93C5FD", width=1, dash="dot"),
        ))

        # Forecast
        fig.add_trace(go.Scatter(
            x=df_fc["date"], y=df_fc["revenue"],
            name="Forecast", line=dict(color="#F59E0B", width=2.5, dash="dash"),
            hovertemplate="%{x|%b %d}<br>Forecast: £%{y:,.0f}<extra></extra>"
        ))

        # Shaded forecast area
        fig.add_vrect(
            x0=df_fc["date"].iloc[0], x1=df_fc["date"].iloc[-1],
            fillcolor="#FEF3C7", opacity=0.2, line_width=0,
            annotation_text="Forecast", annotation_position="top left",
        )

        fig.update_layout(
            title="Revenue: Historical + Forecast",
            xaxis_title="Date", yaxis_title="Daily Revenue (£)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="white", paper_bgcolor="white",
            height=450,
        )
        fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
        fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9",
                         tickprefix="£", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

        #  Forecast table (collapsed) 
        with st.expander("📋  View Full Forecast Table"):
            df_fc_display = df_fc.copy()
            df_fc_display["date"]    = df_fc_display["date"].dt.strftime("%a, %b %d %Y")
            df_fc_display["revenue"] = df_fc_display["revenue"].apply(lambda x: f"£{x:,.2f}")
            df_fc_display.columns    = ["Date", "Predicted Revenue"]
            st.dataframe(df_fc_display, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Could not load forecast: {e}")
        st.info("Make sure the API is running: `uvicorn api.main:app --reload`")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Churn Risk
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Churn Risk")
    st.caption("LightGBM binary classifier — who is about to leave?")

    try:
        df_churn = fetch_churn()
        df_churn["churn_probability"] = df_churn["churn_probability"].astype(float)

        #  KPIs 
        total   = len(df_churn)
        high    = (df_churn["risk_level"] == "High").sum()
        medium  = (df_churn["risk_level"] == "Medium").sum()
        low     = (df_churn["risk_level"] == "Low").sum()
        at_risk_revenue = df_churn[df_churn["risk_level"] == "High"]["Monetary"].sum()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🔴  High Risk Customers", f"{high:,}",   f"{high/total:.0%} of base")
        k2.metric("🟡  Medium Risk",          f"{medium:,}", f"{medium/total:.0%} of base")
        k3.metric("🟢  Low Risk",             f"{low:,}",    f"{low/total:.0%} of base")
        k4.metric("💸  At-Risk Revenue (LTV)", f"£{at_risk_revenue:,.0f}")

        st.divider()

        col_l, col_r = st.columns([3, 2])

        with col_l:
            #  Histogram of churn probability 
            fig = px.histogram(
                df_churn, x="churn_probability", nbins=40,
                color="risk_level",
                color_discrete_map={"High":"#EF4444","Medium":"#F59E0B","Low":"#10B981"},
                title="Distribution of Churn Probability",
                labels={"churn_probability": "Churn Probability", "count": "Customers"},
                height=380,
            )
            fig.update_layout(bargap=0.05, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            #  Risk level breakdown 
            risk_df = pd.DataFrame({
                "Risk Level": ["High", "Medium", "Low"],
                "Customers":  [high, medium, low],
            })
            fig2 = px.bar(risk_df, x="Risk Level", y="Customers",
                          color="Risk Level",
                          color_discrete_map=RISK_COLORS,
                          title="Customers by Risk Level", height=380,
                          text="Customers")
            fig2.update_traces(textposition="outside")
            fig2.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)

        #  Filterable customer table 
        st.subheader("🎯  Customer Risk Table")
        col_f1, col_f2 = st.columns([1, 2])
        risk_filter = col_f1.selectbox("Filter by Risk Level", ["All", "High", "Medium", "Low"])
        min_prob    = col_f2.slider("Min Churn Probability", 0.0, 1.0, 0.0, 0.05)

        display = df_churn.copy()
        if risk_filter != "All":
            display = display[display["risk_level"] == risk_filter]
        display = display[display["churn_probability"] >= min_prob]

        # Format columns
        display_fmt = display[["CustomerID","Recency","Frequency",
                                "Monetary","churn_probability","risk_level"]].copy()
        display_fmt["Monetary"]          = display_fmt["Monetary"].apply(lambda x: f"£{x:,.0f}")
        display_fmt["churn_probability"] = display_fmt["churn_probability"].apply(lambda x: f"{x:.1%}")
        display_fmt.columns = ["Customer ID", "Recency (days)", "Orders",
                                "Lifetime Value", "Churn Probability", "Risk Level"]

        st.dataframe(display_fmt, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(display_fmt):,} of {total:,} customers")

        #  Single customer lookup 
        st.divider()
        st.subheader("🔍  Check a Customer")
        with st.form("churn_form"):
            cc1, cc2, cc3, cc4 = st.columns(4)
            cr = cc1.number_input("Recency (days)",    min_value=0.0,  value=95.0)
            cf = cc2.number_input("Frequency (orders)",min_value=1.0,  value=3.0)
            cm = cc3.number_input("Monetary (£)",      min_value=0.0,  value=120.0)
            ca = cc4.number_input("Avg Order (£)",     min_value=0.0,  value=40.0)
            cc5, cc6 = st.columns(2)
            cs = cc5.number_input("Purchase Span (days)", min_value=0.0, value=180.0)
            cp = cc6.number_input("Purchase Rate (orders/month)", min_value=0.0, value=0.3)
            submitted_c = st.form_submit_button("Predict Churn Risk →")

        if submitted_c:
            result = call_churn({
                "recency": cr, "frequency": cf, "monetary": cm,
                "avg_order_value": ca, "purchase_span": cs,
                "total_items": cf * 4, "days_since_first": cs + cr,
                "purchase_rate": cp,
            })
            risk_c = RISK_COLORS.get(result["risk_level"], "#6B7280")
            st.markdown(f"""
            <div style='background:{risk_c}22; border-left: 5px solid {risk_c};
                        border-radius:8px; padding:16px; margin-top:12px'>
            <h3 style='color:{risk_c}; margin:0'>{result["risk_level"]} Risk
              &nbsp;·&nbsp; {result["churn_probability"]:.1%} probability</h3>
            <p style='margin:8px 0 0 0; color:#374151'>
              <strong>Recommended Action:</strong> {result["recommendation"]}
            </p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Could not load churn data: {e}")
        st.info("Make sure the API is running: `uvicorn api.main:app --reload`")
