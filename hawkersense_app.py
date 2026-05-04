"""
HawkerSense v2.0 — Streamlit Dashboard
Run with: streamlit run hawkersense_app.py

Install: pip install streamlit plotly
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import os
import datetime

from hawkersense_engine import (
    compute_decision, log_run, load_log,
    BASE_DEMAND, DAY_MULTIPLIER, WEATHER_MULTIPLIER,
    SALE_PRICE, COST_PRICE, EVENT_MULTIPLIER
)

# ── Page config ──────────────────────────────────────
st.set_page_config(
    page_title="HawkerSense · Decision Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: #0f1117; color: #e0e0e0; }
  .metric-card {
    background: #1a1d26;
    border: 1px solid #2e3140;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.06em; }
  .metric-value { font-size: 28px; font-weight: 600; color: #5DCAA5; margin: 4px 0; }
  .metric-sub { font-size: 12px; color: #666; }
  .decision-box {
    border-left: 4px solid #1D9E75;
    padding: 12px 16px;
    background: #141a18;
    border-radius: 0 10px 10px 0;
    font-size: 15px;
    line-height: 1.7;
    color: #d0d0d0;
    margin: 1rem 0;
  }
  .risk-low    { color: #5DCAA5; font-weight: 600; }
  .risk-medium { color: #EF9F27; font-weight: 600; }
  .risk-high   { color: #E24B4A; font-weight: 600; }
  h1 { color: #5DCAA5 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar inputs ─────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙ Inputs")
    product  = st.selectbox("Product", list(BASE_DEMAND.keys()), format_func=str.title)
    day      = st.selectbox("Day of week", list(DAY_MULTIPLIER.keys()), index=4, format_func=str.title)
    weather  = st.selectbox("Weather", ["hot", "normal", "rainy", "cold"], index=1, format_func=str.title)
    event    = st.selectbox("Nearby event", list(EVENT_MULTIPLIER.keys()), format_func=str.title)

    st.markdown("---")
    st.markdown("### Pricing")
    sale_price = st.number_input("Sale price (Rs)", value=SALE_PRICE[product], min_value=1, max_value=1000)
    cost_price = st.number_input("Cost price (Rs)", value=COST_PRICE[product], min_value=1, max_value=500)

    st.markdown("---")
    st.markdown("### 🧠 Adaptive Learning")
    waste_yesterday = st.number_input("Units wasted yesterday", value=0, min_value=0, max_value=500)
    if waste_yesterday > 0:
        st.info(f"System will reduce today's forecast by ~{round(waste_yesterday*0.6)} units based on yesterday's waste.")

    run = st.button("▶ Run Decision Engine", type="primary", use_container_width=True)

# ── Main Panel ─────────────────────────────────────────
st.markdown("# HawkerSense · Decision Engine")
st.caption("Probabilistic demand forecasting for informal street vendors · Pakistan")

if run:
    with st.spinner("Running 500 Monte Carlo simulations..."):
        d = compute_decision(
            product=product, day=day, weather=weather, event=event,
            sale_price=sale_price, cost_price=cost_price,
            waste_yesterday=waste_yesterday, n_simulations=500
        )
        log_run(d, waste_actual=waste_yesterday)

    # ── Decision recommendation ─────────────────────────────
    risk_class = {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}[d["risk_level"]]
    st.markdown(f"""
    <div class="decision-box">
    <b>Recommendation:</b> {d['recommendation']}
    <br><span class="{risk_class}">Risk level: {d['risk_level']}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric cards ────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Stock range</div>
            <div class="metric-value">{d['stock_recommendation']['low']}–{d['stock_recommendation']['high']}</div>
            <div class="metric-sub">median {d['forecast']['median']} units</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Unsold risk</div>
            <div class="metric-value">{d['unsold_risk_pct']}%</div>
            <div class="metric-sub">of stock wasted</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Est. profit</div>
            <div class="metric-value">Rs. {d['estimated_profit_rs']:,}</div>
            <div class="metric-sub">at Rs. {sale_price}/unit</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{d['confidence_pct']}%</div>
            <div class="metric-sub">forecast confidence</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Charts ──────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Demand forecast range")
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="number+gauge+delta",
            value=d['forecast']['median'],
            delta={'reference': d['forecast']['p10'], 'relative': False},
            gauge={
                'axis': {'range': [0, d['forecast']['p90'] * 1.3]},
                'bar': {'color': "#1D9E75"},
                'steps': [
                    {'range': [0, d['forecast']['p10']], 'color': "#1a1d26"},
                    {'range': [d['forecast']['p10'], d['forecast']['p90']], 'color': "#141a18"},
                ],
                'threshold': {
                    'line': {'color': "#5DCAA5", 'width': 3},
                    'thickness': 0.75,
                    'value': d['stock_recommendation']['mid']
                }
            },
            title={'text': "Median demand (units)"}
        ))
        fig.update_layout(
            height=240, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0', margin=dict(t=40, b=20, l=30, r=30)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Weekly demand pattern")
        days_order = list(DAY_MULTIPLIER.keys())
        week_demand = [
            round(BASE_DEMAND[product] * DAY_MULTIPLIER[d_] *
                  WEATHER_MULTIPLIER[product].get(weather, 1.0) *
                  EVENT_MULTIPLIER.get(event, 1.0))
            for d_ in days_order
        ]
        colors = ['#1D9E75' if d_ == day.lower() else '#2e3140' for d_ in days_order]
        fig2 = go.Figure(go.Bar(
            x=[d_.title() for d_ in days_order],
            y=week_demand,
            marker_color=colors,
            text=week_demand,
            textposition='outside',
            textfont=dict(color='#888', size=11)
        ))
        fig2.update_layout(
            height=240, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0', showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#2e3140')
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Learning Log ────────────────────────────────────────
    log = load_log()
    if log:
        st.markdown("#### Adaptive learning log")
        df = pd.DataFrame(log[-10:])
        df = df[["timestamp","product","day","weather","event",
                 "recommended_stock","unsold_risk_pct","waste_actual"]]
        df.columns = ["Time","Product","Day","Weather","Event","Rec. Stock","Risk %","Waste"]
        df["Time"] = pd.to_datetime(df["Time"]).dt.strftime("%d %b %H:%M")
        st.dataframe(df.iloc[::-1], use_container_width=True, hide_index=True)

else:
    st.info("Configure inputs in the sidebar and click **Run Decision Engine** to generate your forecast.")
    st.markdown("""
    **What HawkerSense does:**
    - Runs 500 Monte Carlo simulations per forecast for uncertainty-aware prediction
    - Accounts for day of week, weather, and local events — separately per product
    - Recommends a stock range (not just one number), with risk % and confidence score
    - Adapts from yesterday's waste using a simple learning feedback loop
    - Suggests dynamic late-hour pricing to clear remaining stock
    """)
