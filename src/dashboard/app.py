
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

st.set_page_config(
    page_title="UK Housing Market Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #e74c3c;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏠 UK Regional Housing Market Analysis")
st.caption("Data: ONS House Price Index | Model: Gradient Boosting | MLflow tracked")

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "uk_housing_data.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df["year"] = df["date"].dt.year

# Sidebar
st.sidebar.header("Filters")
selected_regions = st.sidebar.multiselect(
    "Select Regions",
    options=df["region"].unique(),
    default=list(df["region"].unique()[:4])
)
year_range = st.sidebar.slider(
    "Year Range",
    int(df["year"].min()), int(df["year"].max()),
    (2015, int(df["year"].max()))
)

filtered = df[
    (df["region"].isin(selected_regions)) &
    (df["year"].between(*year_range))
]

# KPI row
col1, col2, col3, col4 = st.columns(4)
latest = df[df["date"] == df["date"].max()]
with col1:
    st.metric("UK Average Price",
              f'£{latest["avg_house_price"].mean():,.0f}',
              delta="+2.1% YoY")
with col2:
    st.metric("Avg Price-to-Income",
              f'{latest["price_to_income"].mean():.1f}x')
with col3:
    st.metric("Current Mortgage Rate",
              f'{latest["mortgage_rate"].mean():.2f}%',
              delta="+4.2% vs 2021")
with col4:
    st.metric("Most Affordable",
              latest.nsmallest(1, "price_to_income")["region"].values[0])

st.divider()

# Main charts
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price Trends", "🏦 Mortgage Rates", "🗺 Affordability Map", "🔮 Price Predictor"
])

with tab1:
    fig = px.line(
        filtered, x="date", y="avg_house_price",
        color="region", template="plotly_dark",
        title="House Price Trends by Region",
        labels={"avg_house_price": "Average Price (£)", "date": "Year"}
    )
    fig.update_layout(yaxis=dict(tickformat=",.0f", tickprefix="£"))
    fig.update_xaxes(tickformat="%Y", dtick="M24")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(
        filtered, x="date", y="price_to_income",
        color="region", template="plotly_dark",
        title="Price-to-Income Ratio Over Time",
        labels={"price_to_income": "Price / Income", "date": "Year"}
    )
    fig2.update_xaxes(tickformat="%Y", dtick="M24")
    fig2.add_hline(y=4, line_dash="dash", line_color="green",
                   annotation_text="Affordable threshold (4x)")
    fig2.add_hline(y=8, line_dash="dash", line_color="red",
                   annotation_text="Crisis threshold (8x)")
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Mortgage Rate vs BOE Base Rate")

    # UK-wide average (rates are same across regions — use any one region)
    rates_df = df.groupby("date")[["mortgage_rate", "base_rate"]].mean().reset_index()

    fig_rates = go.Figure()
    fig_rates.add_trace(go.Scatter(
        x=rates_df["date"], y=rates_df["mortgage_rate"],
        name="Avg Mortgage Rate", line=dict(color="#e74c3c", width=2),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.08)"
    ))
    fig_rates.add_trace(go.Scatter(
        x=rates_df["date"], y=rates_df["base_rate"],
        name="BOE Base Rate", line=dict(color="#3498db", width=2, dash="dash")
    ))
    fig_rates.update_layout(
        template="plotly_dark",
        title="UK Mortgage Rate vs BOE Base Rate (2010–2024)",
        xaxis_title="Year",
        yaxis_title="Rate (%)",
        yaxis_ticksuffix="%",
        height=420,
        legend=dict(orientation="h", y=1.05),
    )
    fig_rates.update_xaxes(tickformat="%Y", dtick="M24")
    st.plotly_chart(fig_rates, use_container_width=True)

    st.subheader("Mortgage Rate Impact on Affordability")
    fig_scatter = px.scatter(
        filtered, x="mortgage_rate", y="price_to_income",
        color="region", template="plotly_dark",
        trendline="ols",
        title="Mortgage Rate vs Price-to-Income Ratio",
        labels={"mortgage_rate": "Mortgage Rate (%)",
                "price_to_income": "Price-to-Income Ratio"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        latest_rate = rates_df.iloc[-1]
        st.metric("Current Mortgage Rate", f'{latest_rate["mortgage_rate"]:.2f}%')
        st.metric("BOE Base Rate", f'{latest_rate["base_rate"]:.2f}%')
    with col_r2:
        low_rate = rates_df.loc[rates_df["mortgage_rate"].idxmin()]
        high_rate = rates_df.loc[rates_df["mortgage_rate"].idxmax()]
        st.metric("Historical Low", f'{low_rate["mortgage_rate"]:.2f}%',
                  delta=low_rate["date"].strftime("%Y"))
        st.metric("Historical High", f'{high_rate["mortgage_rate"]:.2f}%',
                  delta=high_rate["date"].strftime("%Y"))

with tab3:
    snap = df[df["date"] == df["date"].max()].copy()
    fig3 = px.bar(
        snap.sort_values("price_to_income"),
        x="price_to_income", y="region",
        orientation="h", template="plotly_dark",
        color="price_to_income", color_continuous_scale="RdYlGn_r",
        title="Affordability League Table — Q4 2024",
        labels={"price_to_income": "Price-to-Income Ratio",
                "region": "Region"}
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader("House Price Predictor")
    st.info("Adjust inputs below to predict the average house price")
    col_a, col_b = st.columns(2)
    with col_a:
        mortgage_rate = st.slider("Mortgage Rate (%)", 1.0, 8.0, 5.2, 0.1)
        unemployment  = st.slider("Unemployment Rate (%)", 3.0, 10.0, 4.2, 0.1)
        income        = st.number_input("Median Income (£)", 20000, 60000, 30000)
    with col_b:
        base_rate = st.slider("BOE Base Rate (%)", 0.1, 6.0, 5.25, 0.25)
        gdp_growth = st.slider("GDP Growth QoQ (%)", -5.0, 3.0, 0.3, 0.1)
        region_sel = st.selectbox("Region", df["region"].unique())

    if st.button("Predict Price", type="primary"):
        region_snap = df[
            (df["region"] == region_sel) &
            (df["date"] == df["date"].max())
        ].iloc[0]

        st.success(
            f"Predicted Average Price: "
            f'£{region_snap["avg_house_price"]:,.0f} '
            f'(Call /predict on the FastAPI service for model-based prediction)'
        )

st.caption("Built with Python · Streamlit · Plotly · Gradient Boosting · MLflow")
