import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

DEFAULT_DATABASE_URL = "postgresql+psycopg2://admin:admin@localhost:5432/retail_dw"
BASE_DIR = os.path.dirname(__file__) or "."
FALLBACK_SQLITE = os.path.join(BASE_DIR, "db", "retail_dw_fallback.db")

@st.cache_resource
def get_engine():
    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    # Try PostgreSQL first
    if "postgresql" in database_url:
        try:
            engine = create_engine(database_url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine
        except Exception:
            pass
    # Fallback to SQLite
    return create_engine(f"sqlite:///{FALLBACK_SQLITE}")

# ═══════════════════════════════════════════════════════════════════════════
# Data queries
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def fetch_fact_sales(limit=1000):
    try:
        return pd.read_sql(f'SELECT * FROM "Fact_Sales" LIMIT {limit}', get_engine())
    except Exception:
        return pd.read_sql(f"SELECT * FROM Fact_Sales LIMIT {limit}", get_engine())

@st.cache_data
def fetch_monthly_revenue():
    try:
        query = """
        SELECT t."Month" AS month, t."Year" AS year, SUM(f."TotalPrice") AS revenue
        FROM "Fact_Sales" f
        JOIN "Dim_Time" t ON f."Date" = t."Date"
        GROUP BY t."Year", t."Month"
        ORDER BY t."Year", t."Month"
        """
        return pd.read_sql(query, get_engine())
    except Exception:
        query = """
        SELECT t.Month AS month, t.Year AS year, SUM(f.TotalPrice) AS revenue
        FROM Fact_Sales f
        JOIN Dim_Time t ON f.Date = t.Date
        GROUP BY t.Year, t.Month
        ORDER BY t.Year, t.Month
        """
        return pd.read_sql(query, get_engine())

@st.cache_data
def fetch_top_countries():
    try:
        query = """
        SELECT c."Country" AS country, SUM(f."TotalPrice") AS revenue
        FROM "Fact_Sales" f
        JOIN "Dim_Customer" c ON f."CustomerID" = c."CustomerID"
        GROUP BY c."Country"
        ORDER BY revenue DESC
        LIMIT 10
        """
        return pd.read_sql(query, get_engine())
    except Exception:
        query = """
        SELECT c.Country AS country, SUM(f.TotalPrice) AS revenue
        FROM Fact_Sales f
        JOIN Dim_Customer c ON f.CustomerID = c.CustomerID
        GROUP BY c.Country
        ORDER BY revenue DESC
        LIMIT 10
        """
        return pd.read_sql(query, get_engine())

@st.cache_data
def fetch_top_products():
    try:
        query = """
        SELECT p."Description" AS product, SUM(f."TotalPrice") AS revenue
        FROM "Fact_Sales" f
        JOIN "Dim_Product" p ON f."StockCode" = p."StockCode"
        WHERE p."Description" IS NOT NULL
        GROUP BY p."Description"
        ORDER BY revenue DESC
        LIMIT 10
        """
        return pd.read_sql(query, get_engine())
    except Exception:
        query = """
        SELECT p.Description AS product, SUM(f.TotalPrice) AS revenue
        FROM Fact_Sales f
        JOIN Dim_Product p ON f.StockCode = p.StockCode
        WHERE p.Description IS NOT NULL
        GROUP BY p.Description
        ORDER BY revenue DESC
        LIMIT 10
        """
        return pd.read_sql(query, get_engine())

@st.cache_data
def fetch_all_rfm():
    try:
        return pd.read_sql('SELECT * FROM "Customer_RFM"', get_engine())
    except Exception:
        return pd.read_sql("SELECT * FROM Customer_RFM", get_engine())

@st.cache_data
def fetch_cluster_data():
    try:
        return pd.read_sql('SELECT * FROM "Customer_Cluster"', get_engine())
    except Exception:
        return pd.read_sql("SELECT * FROM Customer_Cluster", get_engine())

@st.cache_data
def fetch_customer_items(customer_id: float):
    try:
        query = text("""
            SELECT DISTINCT p."Description" AS item
            FROM "Fact_Sales" f
            JOIN "Dim_Product" p ON f."StockCode" = p."StockCode"
            WHERE f."CustomerID" = :customer_id
              AND p."Description" IS NOT NULL
        """)
    except Exception:
        query = text("""
            SELECT DISTINCT p.Description AS item
            FROM Fact_Sales f
            JOIN Dim_Product p ON f.StockCode = p.StockCode
            WHERE f.CustomerID = :customer_id
              AND p.Description IS NOT NULL
        """)
    with get_engine().connect() as conn:
        rows = conn.execute(query, {"customer_id": customer_id}).fetchall()
    return {row.item for row in rows if row.item}

@st.cache_resource
def load_models():
    default_model_dir = os.path.join(BASE_DIR, "output", "models")
    
    model_dir = os.getenv("MODEL_OUTPUT_DIR", default_model_dir)

    expected_files = ["kmeans.pkl", "xgb_model.pkl", "rules.pkl", "scaler.pkl"]
    missing = [name for name in expected_files if not os.path.exists(os.path.join(model_dir, name))]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(f"Missing model files in '{model_dir}': {missing_list}")

    kmeans = joblib.load(os.path.join(model_dir, "kmeans.pkl"))
    xgb_model = joblib.load(os.path.join(model_dir, "xgb_model.pkl"))
    rules = joblib.load(os.path.join(model_dir, "rules.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    return kmeans, xgb_model, rules, scaler

def recommend_items(purchased_items, rules_df, top_n=5):
    recommendations = []
    for _, row in rules_df.iterrows():
        antecedents = set(row["antecedents"])
        consequents = set(row["consequents"])
        if antecedents and antecedents.issubset(purchased_items):
            for candidate in consequents:
                if candidate not in purchased_items and candidate not in recommendations:
                    recommendations.append(candidate)
                if len(recommendations) >= top_n:
                    return recommendations
    return recommendations

# ═══════════════════════════════════════════════════════════════════════════
# Plotly layout helper — Natural Dark UI
# ═══════════════════════════════════════════════════════════════════════════
def modern_dark_layout(fig, title="", x_title=None, y_title=None):
    """Apply a professional dark mode style to Plotly figures."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Inter, sans-serif", size=15, color="#F3F4F6"),
            x=0,
        ),
        font=dict(family="Inter, sans-serif", size=12, color="#9CA3AF"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=True,
        legend=dict(
            font=dict(size=11, family="Inter, sans-serif", color="#D1D5DB"),
            bgcolor="rgba(17, 24, 39, 0.8)",
            bordercolor="#374151",
            borderwidth=1,
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=12, color="#9CA3AF")) if x_title else None,
            showgrid=True,
            gridcolor="#374151",
            zeroline=False,
            linecolor="#4B5563",
            linewidth=1,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=12, color="#9CA3AF")) if y_title else None,
            showgrid=True,
            gridcolor="#374151",
            gridwidth=1,
            zeroline=False,
            linecolor="#4B5563",
            linewidth=1,
            tickfont=dict(size=11),
        ),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Theme — Professional Dark Mode
# ═══════════════════════════════════════════════════════════════════════════
def apply_theme():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

        :root {
            --bg: #111827;
            --card-bg: #1F2937;
            --fg: #F9FAFB;
            --muted: #9CA3AF;
            --border: #374151;
            --accent: #3B82F6;
        }

        .stApp {
            background-color: var(--bg) !important;
            color: var(--fg);
            font-family: 'Inter', sans-serif !important;
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--fg) !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
        }
        h1 {
            font-size: 1.6rem !important;
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background-color: transparent;
            border-bottom: 1px solid var(--border);
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: var(--muted);
            border: none;
            border-bottom: 2px solid transparent;
            border-radius: 0;
            padding: 0.5rem 0.2rem;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            font-size: 0.9rem;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: var(--fg);
        }
        .stTabs [aria-selected="true"] {
            color: var(--accent) !important;
            border-bottom-color: var(--accent) !important;
        }

        div[data-testid="stMetric"] {
            background: var(--card-bg);
            border: 1px solid var(--border);
            padding: 1rem;
            border-radius: 6px;
        }
        div[data-testid="stMetric"] label {
            color: var(--muted) !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.8rem !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: var(--fg) !important;
            font-size: 1.5rem !important;
            font-weight: 600;
        }

        .stDataFrame {
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
        }

        .stTextInput input, .stSelectbox > div > div {
            border: 1px solid var(--border) !important;
            border-radius: 4px !important;
            background: var(--card-bg) !important;
            color: var(--fg) !important;
        }
        .stTextInput input:focus, .stSelectbox > div > div:focus {
            border-color: var(--accent) !important;
            box-shadow: none !important;
        }

        hr { border-color: var(--border); }
        </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Dashboard Tab
# ═══════════════════════════════════════════════════════════════════════════
def dashboard_tab():
    st.header("Business Intelligence")

    monthly = fetch_monthly_revenue()
    countries = fetch_top_countries()
    products = fetch_top_products()
    rfm_df = fetch_all_rfm()
    cluster_df = fetch_cluster_data()
    recent_transactions = fetch_fact_sales(limit=100)

    if not rfm_df.empty:
        total_rev = rfm_df["Monetary"].sum() if "Monetary" in rfm_df.columns else 0
        total_cust = len(rfm_df)
        avg_monetary = rfm_df["Monetary"].mean() if "Monetary" in rfm_df.columns else 0
        avg_frequency = rfm_df["Frequency"].mean() if "Frequency" in rfm_df.columns else 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Revenue", f"£{total_rev:,.0f}")
        k2.metric("Total Customers", f"{total_cust:,}")
        k3.metric("Avg Spend", f"£{avg_monetary:,.0f}")
        k4.metric("Avg Frequency", f"{avg_frequency:.1f}")

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        monthly["label"] = monthly.apply(lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1)
        fig = px.area(monthly, x="label", y="revenue", color_discrete_sequence=["#3B82F6"])
        modern_dark_layout(fig, "Monthly Revenue", "Month", "Revenue (£)")
        fig.update_traces(fillcolor='rgba(59, 130, 246, 0.2)', line=dict(width=2))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(countries, values="revenue", names="country", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Plotly)
        modern_dark_layout(fig, "Revenue by Country")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=False) 
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.bar(products, x="revenue", y="product", orientation="h",
                     color_discrete_sequence=["#10B981"])
        modern_dark_layout(fig, "Top Products", "Revenue (£)", "Product")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("**Recent Transactions**")
        st.dataframe(recent_transactions, height=300, use_container_width=True)

    if not rfm_df.empty:
        st.markdown("---")
        col5, col6 = st.columns([1, 2])
        with col5:
            rfm_col = st.selectbox("Metric Distribution", ["Recency", "Frequency", "Monetary"])
            data_col = rfm_col if rfm_col in rfm_df.columns else rfm_col.lower()
            fig = px.histogram(rfm_df, x=data_col, nbins=40, color_discrete_sequence=["#8B5CF6"])
            modern_dark_layout(fig, f"{rfm_col} Distribution", rfm_col, "Count")
            st.plotly_chart(fig, use_container_width=True)
        
        with col6:
            if not cluster_df.empty and "Cluster" in cluster_df.columns:
                cluster_strs = cluster_df["Cluster"].astype(str)
                fig = px.scatter(cluster_df, x="Frequency", y="Monetary",
                                 color=cluster_strs,
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                modern_dark_layout(fig, "Customer Segments", "Frequency", "Monetary (£)")
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# AI Customer Intelligence Tab
# ═══════════════════════════════════════════════════════════════════════════
def ai_customer_tab():
    st.header("Customer Intelligence")

    rfm_df = fetch_all_rfm()
    cluster_df = fetch_cluster_data()

    if rfm_df.empty:
        st.info("No customer data available.")
        return

    id_col = "CustomerID" if "CustomerID" in rfm_df.columns else "customerid"
    if not cluster_df.empty:
        cluster_id_col = "CustomerID" if "CustomerID" in cluster_df.columns else "customerid"
        merged_df = pd.merge(rfm_df, cluster_df[[cluster_id_col, "Cluster"]], left_on=id_col, right_on=cluster_id_col, how='left')
    else:
        merged_df = rfm_df.copy()
        merged_df["Cluster"] = "Unknown"

    st.markdown("**Customer Directory**")
    st.dataframe(merged_df, use_container_width=True, height=350)

    st.markdown("---")

    col_input, _ = st.columns([1, 2])
    with col_input:
        customers_list = sorted(merged_df[id_col].dropna().unique().tolist())
        selected_id = st.selectbox("Select Customer", customers_list)

    if not selected_id:
        return

    customer_row = merged_df[np.isclose(merged_df[id_col], selected_id)]

    try:
        kmeans, xgb_model, rules, scaler = load_models()
    except FileNotFoundError:
        st.error("Models not found. Please ensure the pipeline has generated models in 'output/models/'.")
        return

    rec_col = "Recency" if "Recency" in customer_row.columns else "recency"
    freq_col = "Frequency" if "Frequency" in customer_row.columns else "frequency"
    mon_col = "Monetary" if "Monetary" in customer_row.columns else "monetary"

    recency = float(customer_row.iloc[0][rec_col])
    frequency = float(customer_row.iloc[0][freq_col])
    monetary = float(customer_row.iloc[0][mon_col])

    features = pd.DataFrame(
        [[recency, frequency, monetary]],
        columns=["Recency", "Frequency", "Monetary"],
    )

    scaled = scaler.transform(features)
    predicted_cluster = int(kmeans.predict(scaled)[0])
    clv_prediction = float(xgb_model.predict(features)[0])

    purchased_items = fetch_customer_items(selected_id)
    recommendations = recommend_items(purchased_items, rules, top_n=5)

    cluster_labels = {0: "At Risk", 1: "Loyal", 2: "Champions", 3: "New"}
    cluster_name = cluster_labels.get(predicted_cluster, f"Segment {predicted_cluster}")

    st.markdown(f"**Customer Profile: {int(selected_id)}**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Segment", f"{cluster_name}")
    m2.metric("Predicted LTV", f"£{clv_prediction:,.2f}")
    m3.metric("Items Purchased", len(purchased_items))
    m4.metric("Avg Frequency", f"{frequency:.0f}")

    # Fetch segment average
    segment_df = merged_df[merged_df["Cluster"] == (predicted_cluster if "Cluster" in merged_df.columns else "Unknown")]
    if not segment_df.empty:
        avg_rec = segment_df[rec_col].mean()
        avg_freq = segment_df[freq_col].mean()
        avg_mon = segment_df[mon_col].mean()
    else:
        avg_rec, avg_freq, avg_mon = recency, frequency, monetary

    col1, col2 = st.columns([1, 1])
    with col1:
        # Comparison with Segment
        data = {
            "Metric": ["Recency", "Recency", "Frequency", "Frequency", "Monetary", "Monetary"],
            "Target": ["Customer", "Segment Avg", "Customer", "Segment Avg", "Customer", "Segment Avg"],
            "Value": [recency, avg_rec, frequency, avg_freq, monetary, avg_mon]
        }
        comp_df = pd.DataFrame(data)
        
        fig = px.bar(comp_df, x="Metric", y="Value", color="Target", barmode="group",
             color_discrete_sequence=["#3B82F6", "#374151"])
        modern_dark_layout(fig, "Customer vs Segment Average", "Metric", "Value")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Positioning scatter plot
        focus_df = merged_df.copy()
        focus_df["Label"] = focus_df[id_col].apply(lambda x: "Current" if x == selected_id else "Others")
        focus_df = focus_df.sort_values("Label")
        
        fig2 = px.scatter(focus_df, x=freq_col, y=mon_col, color="Label",
                          color_discrete_map={"Current": "#10B981", "Others": "rgba(107, 114, 128, 0.3)"},
                          hover_data=[id_col, "Cluster"])
        modern_dark_layout(fig2, "Customer Position (Frequency vs Monetary)", "Frequency", "Monetary (£)")
        fig2.update_traces(marker=dict(size=[12 if x == "Current" else 4 for x in focus_df["Label"]]))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("**Cross-Selling / Association Insights**")
    col3, col4 = st.columns([1, 1])
    with col3:
        if recommendations:
            st.info("Targeted Items highly associated with this customer's behavior:")
            for item in recommendations:
                st.markdown(f"- {item}")
        else:
            st.write("No strong association rules found for this customer.")
            
    with col4:
        st.markdown("**Historically Purchased Items:**")
        if purchased_items:
            # Showing fewer items directly to look better
            buy_list = list(purchased_items)[:8]
            for item in buy_list:
                st.write(f"- {item}")
            if len(purchased_items) > 8:
                st.write(f"*(and {len(purchased_items) - 8} more...)*")
        else:
            st.write("No items found.")


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Log Viewer Tab
# ═══════════════════════════════════════════════════════════════════════════
def log_viewer_tab():
    st.header("System Logs")
    
    log_path = os.path.join("report", "generated", "pipeline.log")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            log_content = f.read()
        st.code(log_content, language="log")
    else:
        st.write("No pipeline logs found.")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="Retail Dashboard",
        layout="wide",
    )
    apply_theme()

    st.title("Retail Data Warehouse")

    tab1, tab2, tab3 = st.tabs(["Dashboard", "Customer Intelligence", "System Logs"])

    with tab1:
        dashboard_tab()
    with tab2:
        ai_customer_tab()
    with tab3:
        log_viewer_tab()

if __name__ == "__main__":
    main()
