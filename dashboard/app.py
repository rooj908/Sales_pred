"""
Streamlit Dashboard — AI-Powered BI & Sales Forecasting Platform
Run: streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle, os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title="BI & Sales Forecasting Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
    padding: 1.2rem; border-radius: 12px; color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 1rem;
}
.metric-value { font-size: 2rem; font-weight: 700; }
.metric-label { font-size: 0.85rem; opacity: 0.85; }
.stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        sales = pd.read_csv("data/processed/sales_clean.csv", parse_dates=["date"])
        customers = pd.read_csv("data/processed/customers_clean.csv")
        return sales, customers
    except:
        return None, None

@st.cache_resource
def load_models():
    models = {}
    for name, path in [("sales", "models/saved/sales_model.pkl"),
                       ("churn", "models/saved/churn_model.pkl"),
                       ("forecast", "models/saved/forecast_model.pkl"),
                       ("scaler", "models/saved/scaler.pkl")]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models

sales_df, customer_df = load_data()
models = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=80)
    st.title("BI Sales Platform")
    st.caption("Enterprise Analytics Dashboard")
    st.divider()
    
    if sales_df is not None:
        year_filter = st.multiselect("Filter Year", sorted(sales_df["year"].unique()), default=sorted(sales_df["year"].unique()))
        region_filter = st.multiselect("Filter Region", sales_df["region"].unique(), default=list(sales_df["region"].unique()))
        segment_filter = st.multiselect("Segment", sales_df["segment"].unique(), default=list(sales_df["segment"].unique()))
        
        filtered = sales_df[
            sales_df["year"].isin(year_filter) &
            sales_df["region"].isin(region_filter) &
            sales_df["segment"].isin(segment_filter)
        ]
    else:
        st.warning("⚠️ Run setup.py first to generate data & train models.")
        filtered = pd.DataFrame()
    
    st.divider()
    st.caption("v1.0.0 | AI-Powered BI Platform")

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 KPI Dashboard", "📈 Sales Analysis",
    "🔮 Forecasting", "🤖 AI Predictions", "📋 Reports"
])

# ─── TAB 1: KPI Dashboard ─────────────────────────────────────────────────────
with tab1:
    st.header("📊 Key Performance Indicators")
    
    if not filtered.empty:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("💰 Total Revenue", f"${filtered['revenue'].sum():,.0f}", delta="+12.5%")
        with c2:
            st.metric("📦 Total Orders", f"{len(filtered):,}", delta="+8.3%")
        with c3:
            st.metric("💵 Avg Order Value", f"${filtered['revenue'].mean():,.0f}", delta="+3.1%")
        with c4:
            st.metric("📈 Total Profit", f"${filtered['profit'].sum():,.0f}", delta="+15.2%")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            monthly = filtered.groupby(["year", "month"])["revenue"].sum().reset_index()
            monthly["period"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
            fig = px.area(monthly, x="period", y="revenue", title="Monthly Revenue Trend",
                          color_discrete_sequence=["#2196F3"])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            by_product = filtered.groupby("product")["revenue"].sum().sort_values(ascending=True)
            fig = px.bar(by_product, orientation="h", title="Revenue by Product",
                         color=by_product.values, color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            by_region = filtered.groupby("region")["revenue"].sum()
            fig = px.pie(values=by_region.values, names=by_region.index, title="Revenue by Region",
                         hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            by_segment = filtered.groupby("segment").agg(
                revenue=("revenue", "sum"), orders=("order_id", "count")
            ).reset_index()
            fig = px.scatter(by_segment, x="orders", y="revenue", size="revenue",
                             text="segment", title="Segment: Orders vs Revenue",
                             color="segment")
            st.plotly_chart(fig, use_container_width=True)

# ─── TAB 2: Sales Analysis ────────────────────────────────────────────────────
with tab2:
    st.header("📈 Detailed Sales Analysis")
    
    if not filtered.empty:
        col1, col2 = st.columns(2)
        with col1:
            heatmap_data = filtered.pivot_table(values="revenue", index="product",
                                                 columns="region", aggfunc="sum", fill_value=0)
            fig = px.imshow(heatmap_data, title="Revenue Heatmap: Product × Region",
                            color_continuous_scale="Blues", text_auto=".2s")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            qtr = filtered.groupby(["year", "quarter"])["revenue"].sum().reset_index()
            qtr["label"] = "Q" + qtr["quarter"].astype(str) + " " + qtr["year"].astype(str)
            fig = px.bar(qtr, x="label", y="revenue", title="Quarterly Revenue",
                         color="year", barmode="group")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Raw Data Preview")
        st.dataframe(filtered.head(100), use_container_width=True, height=300)
        
        csv = filtered.to_csv(index=False).encode()
        st.download_button("⬇️ Download Filtered Data (CSV)", csv, "filtered_sales.csv", "text/csv")

# ─── TAB 3: Forecasting ───────────────────────────────────────────────────────
with tab3:
    st.header("🔮 Sales Revenue Forecasting")
    
    months_ahead = st.slider("Months to Forecast", 1, 24, 6)
    
    if "forecast" in models and not filtered.empty:
        base_idx = 36
        future = models["forecast"].predict([[i] for i in range(base_idx+1, base_idx+months_ahead+1)])
        future = [max(v, 0) for v in future]
        
        # Historical monthly
        hist = filtered.groupby(["year", "month"])["revenue"].sum().reset_index()
        hist["period"] = pd.to_datetime(hist[["year", "month"]].assign(day=1))
        hist = hist.sort_values("period")
        
        # pandas already imported
        last_date = hist["period"].max()
        future_dates = pd.date_range(last_date, periods=months_ahead+1, freq="MS")[1:]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["period"], y=hist["revenue"],
                                 mode="lines+markers", name="Historical", line=dict(color="#2196F3")))
        fig.add_trace(go.Scatter(x=future_dates, y=future,
                                 mode="lines+markers", name="Forecast",
                                 line=dict(color="#FF5722", dash="dash")))
        fig.update_layout(title=f"Revenue Forecast — Next {months_ahead} Months",
                          xaxis_title="Date", yaxis_title="Revenue (USD)")
        st.plotly_chart(fig, use_container_width=True)
        
        fdf = pd.DataFrame({"Month": future_dates, "Forecasted Revenue": [f"${v:,.0f}" for v in future]})
        st.dataframe(fdf, use_container_width=True)
    else:
        st.info("Train models first by running `python models/train_models.py`")

# ─── TAB 4: AI Predictions ────────────────────────────────────────────────────
with tab4:
    st.header("🤖 AI Prediction Tools")
    
    pred_col, churn_col = st.columns(2)
    
    with pred_col:
        st.subheader("💰 Revenue Predictor")
        with st.form("sales_form"):
            qty = st.number_input("Quantity", 1, 100, 5)
            price = st.number_input("Unit Price ($)", 10.0, 5000.0, 299.0)
            discount = st.slider("Discount", 0.0, 0.5, 0.1, 0.01)
            mkt = st.number_input("Marketing Spend ($)", 0.0, 10000.0, 1000.0)
            month = st.selectbox("Month", range(1, 13), index=5)
            submitted = st.form_submit_button("🔮 Predict Revenue")
            
            if submitted and "sales" in models:
                features = [[qty, price, discount, 35, 2.0, mkt, 1.0, month, (month-1)//3+1, 0, 0, 0]]
                if "scaler" in models:
                    features = models["scaler"].transform(features)
                pred = models["sales"].predict(features)[0]
                st.success(f"**Predicted Revenue: ${pred:,.2f}**")
    
    with churn_col:
        st.subheader("⚠️ Churn Risk Analyzer")
        with st.form("churn_form"):
            age = st.number_input("Customer Age", 18, 80, 35)
            orders = st.number_input("Total Orders", 1, 100, 10)
            spend = st.number_input("Total Spend ($)", 0.0, 100000.0, 5000.0)
            loyalty = st.slider("Loyalty Years", 0.0, 10.0, 2.0, 0.5)
            satisfaction = st.slider("Satisfaction (1-5)", 1.0, 5.0, 3.5, 0.5)
            churn_submitted = st.form_submit_button("🔍 Analyze Churn Risk")
            
            if churn_submitted and "churn" in models:
                features = [[age, orders, spend, loyalty, satisfaction, 1, 0, 0]]
                prob = models["churn"].predict_proba(features)[0][1]
                risk = "🔴 High" if prob > 0.6 else "🟡 Medium" if prob > 0.3 else "🟢 Low"
                st.metric("Churn Probability", f"{prob:.1%}", delta=None)
                st.info(f"**Risk Level: {risk}**")

# ─── TAB 5: Reports ───────────────────────────────────────────────────────────
with tab5:
    st.header("📋 Automated Business Reports")
    
    if not filtered.empty:
        st.subheader("Executive Summary Report")
        
        top_product = filtered.groupby("product")["revenue"].sum().idxmax()
        top_region = filtered.groupby("region")["revenue"].sum().idxmax()
        
        report = f"""
## 📊 Business Intelligence Report
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

### Revenue Overview
- **Total Revenue:** ${filtered['revenue'].sum():,.0f}
- **Total Orders:** {len(filtered):,}
- **Average Order Value:** ${filtered['revenue'].mean():,.0f}
- **Total Profit:** ${filtered['profit'].sum():,.0f}

### Top Performers
- **Best Product:** {top_product}
- **Best Region:** {top_region}
- **Best Segment:** {filtered.groupby('segment')['revenue'].sum().idxmax()}

### Monthly Averages
- **Avg Monthly Revenue:** ${filtered.groupby(['year','month'])['revenue'].sum().mean():,.0f}
- **Avg Monthly Orders:** {filtered.groupby(['year','month'])['order_id'].count().mean():.0f}

### Model Performance
"""
        for name, path in [("Sales Prediction", "models/saved/sales_metrics.json"),
                            ("Churn Detection", "models/saved/churn_metrics.json")]:
            if os.path.exists(path):
                with open(path) as f:
                    m = json.load(f)
                report += f"\n**{name}:** " + ", ".join(f"{k}: {v}" for k, v in m.items()) + "\n"
        
        st.markdown(report)
        st.download_button("⬇️ Download Report (Markdown)", report.encode(), "bi_report.md", "text/markdown")
    else:
        st.info("Generate data first.")
