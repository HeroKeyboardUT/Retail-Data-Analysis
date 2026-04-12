"""
Retail ETL + Data Warehouse + AI Pipeline
==========================================
End-to-end pipeline: Extract → Transform → Load → Train → Visualize → Report.
Supports PostgreSQL (via Docker) with automatic SQLite fallback.
"""

import argparse
import datetime as dt
import json
import logging
import os
import sys
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from mlxtend.frequent_patterns import association_rules, fpgrowth
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_KAGGLE_PATH = r"C:\Users\USER\Desktop\DW DSS\online_retail.csv"
DEFAULT_DATABASE_URL = "postgresql+psycopg2://admin:admin@localhost:5432/retail_dw"
FALLBACK_SQLITE_PATH = os.path.join(os.path.dirname(__file__) or ".", "retail_dw_fallback.db")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__) or ".", "output")
DEFAULT_REPORT_DIR = os.path.join("report", "generated")


# ═══════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════
def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("retail_dw_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(os.path.join(output_dir, "pipeline.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _get_engine(database_url: str, logger):
    """Try PostgreSQL first; fall back to SQLite if unreachable."""
    if "postgresql" in database_url:
        try:
            engine = create_engine(database_url)
            with engine.connect() as conn:
                conn.execute(engine.dialect.server_version_info.__class__.__mro__[0].__init__.__class__("SELECT 1").__class__("SELECT 1"))
            logger.info("Connected to PostgreSQL: %s", engine.url.render_as_string(hide_password=True))
            return engine
        except Exception:
            pass
        # Simpler approach: just try a raw connection test
        try:
            engine = create_engine(database_url)
            with engine.connect() as conn:
                from sqlalchemy import text
                conn.execute(text("SELECT 1"))
            logger.info("Connected to PostgreSQL: %s", engine.url.render_as_string(hide_password=True))
            return engine
        except Exception as exc:
            logger.warning("PostgreSQL unavailable (%s). Falling back to SQLite.", exc)

    sqlite_url = f"sqlite:///{FALLBACK_SQLITE_PATH}"
    engine = create_engine(sqlite_url)
    logger.info("Using SQLite fallback: %s", FALLBACK_SQLITE_PATH)
    return engine


def _save_plot(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 1. EXTRACT + TRANSFORM
# ═══════════════════════════════════════════════════════════════════════════
def extract_transform(file_path: str, logger):
    logger.info("=== PHASE 1: EXTRACT + TRANSFORM ===")
    logger.info("Reading CSV: %s", file_path)
    df_raw = pd.read_csv(file_path, encoding="ISO-8859-1")
    logger.info("Raw dataset: %d rows x %d columns", *df_raw.shape)
    logger.info("Columns: %s", list(df_raw.columns))
    logger.info("Data types:\n%s", df_raw.dtypes.to_string())

    # --- Missing value analysis ---
    missing = df_raw.isnull().sum()
    missing_pct = (missing / len(df_raw) * 100).round(2)
    missing_report = pd.DataFrame({"Missing": missing, "Pct": missing_pct})
    logger.info("Missing values:\n%s", missing_report[missing_report["Missing"] > 0].to_string())

    # --- Duplicates ---
    dup_count = df_raw.duplicated().sum()
    logger.info("Duplicate rows: %d", dup_count)
    df_raw = df_raw.drop_duplicates()
    logger.info("After deduplication: %d rows", len(df_raw))

    # --- Step-by-step cleaning ---
    df_step_1 = df_raw.dropna(subset=["CustomerID"])
    logger.info("After removing missing CustomerID: %d rows (removed %d)", len(df_step_1), len(df_raw) - len(df_step_1))

    df_step_2 = df_step_1[~df_step_1["InvoiceNo"].astype(str).str.startswith("C")]
    logger.info("After removing cancelled invoices: %d rows (removed %d)", len(df_step_2), len(df_step_1) - len(df_step_2))

    df = df_step_2[(df_step_2["Quantity"] > 0) & (df_step_2["UnitPrice"] > 0)].copy()
    logger.info("After removing invalid qty/price: %d rows (removed %d)", len(df), len(df_step_2) - len(df))

    # --- Transformation ---
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["CustomerID"] = df["CustomerID"].astype(int)
    logger.info("Created TotalPrice = Quantity × UnitPrice")
    logger.info("Converted InvoiceDate to datetime, CustomerID to int")

    # --- Descriptive statistics ---
    logger.info("Descriptive statistics of numerical columns:\n%s", df[["Quantity", "UnitPrice", "TotalPrice"]].describe().to_string())

    # --- RFM ---
    snapshot_date = df["InvoiceDate"].max() + dt.timedelta(days=1)
    split_date = df["InvoiceDate"].max() - pd.DateOffset(months=3)

    df_train_period = df[df["InvoiceDate"] < split_date]
    df_test_period = df[df["InvoiceDate"] >= split_date]
    logger.info("Train period: %s to %s (%d rows)", df_train_period["InvoiceDate"].min().date(), split_date.date(), len(df_train_period))
    logger.info("Test period: %s to %s (%d rows)", split_date.date(), df["InvoiceDate"].max().date(), len(df_test_period))

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum",
    }).rename(columns={"InvoiceDate": "Recency", "InvoiceNo": "Frequency", "TotalPrice": "Monetary"})
    logger.info("RFM table created: %d customers", len(rfm))
    logger.info("RFM statistics:\n%s", rfm.describe().to_string())

    rfm_features = df_train_period.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (split_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum",
    }).rename(columns={"InvoiceDate": "Recency", "InvoiceNo": "Frequency", "TotalPrice": "Monetary"})

    future_spend = df_test_period.groupby("CustomerID")["TotalPrice"].sum().rename("FutureSpend")
    clv_data = rfm_features.join(future_spend).fillna(0)
    logger.info("CLV dataset: %d customers (with FutureSpend)", len(clv_data))

    # --- Feature selection rationale ---
    logger.info("Feature selection: Using RFM (Recency, Frequency, Monetary) — standard customer analytics features")
    logger.info("  Recency  — Days since last purchase (behavioural signal)")
    logger.info("  Frequency — Number of distinct transactions (engagement)")
    logger.info("  Monetary — Total revenue contribution (value)")

    stats = {
        "raw_rows": int(len(df_raw)) + dup_count,
        "after_dedup": int(len(df_raw)),
        "duplicates_removed": int(dup_count),
        "rows_after_customer_filter": int(len(df_step_1)),
        "rows_after_cancellation_filter": int(len(df_step_2)),
        "clean_rows": int(len(df)),
        "removed_missing_customerid": int(len(df_raw) - len(df_step_1)),
        "removed_cancelled_invoices": int(len(df_step_1) - len(df_step_2)),
        "removed_invalid_quantity_or_price": int(len(df_step_2) - len(df)),
        "unique_customers": int(df["CustomerID"].nunique()),
        "unique_products": int(df["StockCode"].nunique()),
        "unique_invoices": int(df["InvoiceNo"].nunique()),
        "unique_countries": int(df["Country"].nunique()),
        "total_revenue": float(df["TotalPrice"].sum()),
        "date_min": df["InvoiceDate"].min().strftime("%Y-%m-%d"),
        "date_max": df["InvoiceDate"].max().strftime("%Y-%m-%d"),
        "training_rows": int(len(df_train_period)),
        "testing_rows": int(len(df_test_period)),
        "rfm_rows": int(len(rfm)),
        "clv_rows": int(len(clv_data)),
        "num_attributes": int(df_raw.shape[1]),
        "missing_customerid_pct": float(missing_pct.get("CustomerID", 0)),
        "missing_description_pct": float(missing_pct.get("Description", 0)),
    }

    logger.info("=== EXTRACT + TRANSFORM COMPLETE ===")
    return df, rfm, clv_data, stats


# ═══════════════════════════════════════════════════════════════════════════
# 2. LOAD — Star Schema
# ═══════════════════════════════════════════════════════════════════════════
def load_star_schema(df: pd.DataFrame, rfm: pd.DataFrame, engine, logger):
    logger.info("=== PHASE 2: LOAD STAR SCHEMA ===")

    dim_customer = df[["CustomerID", "Country"]].drop_duplicates()
    dim_customer.to_sql("Dim_Customer", engine, index=False, if_exists="replace")
    logger.info("Dim_Customer: %d rows", len(dim_customer))

    dim_product = df[["StockCode", "Description", "UnitPrice"]].drop_duplicates(subset=["StockCode"])
    dim_product.to_sql("Dim_Product", engine, index=False, if_exists="replace")
    logger.info("Dim_Product: %d rows", len(dim_product))

    dim_time = pd.DataFrame({"Date": df["InvoiceDate"].dt.date.unique()})
    dim_time["Date_Obj"] = pd.to_datetime(dim_time["Date"])
    dim_time["Year"] = dim_time["Date_Obj"].dt.year
    dim_time["Quarter"] = dim_time["Date_Obj"].dt.quarter
    dim_time["Month"] = dim_time["Date_Obj"].dt.month
    dim_time["DayOfWeek"] = dim_time["Date_Obj"].dt.dayofweek
    dim_time["Date"] = dim_time["Date"].astype(str)
    dim_time.drop(columns=["Date_Obj"], inplace=True)
    dim_time.to_sql("Dim_Time", engine, index=False, if_exists="replace")
    logger.info("Dim_Time: %d rows (added Quarter, DayOfWeek)", len(dim_time))

    fact_sales = df[["InvoiceNo", "StockCode", "CustomerID", "Quantity", "TotalPrice"]].copy()
    fact_sales["Date"] = df["InvoiceDate"].dt.strftime("%Y-%m-%d")
    fact_sales.to_sql("Fact_Sales", engine, index=False, if_exists="replace")
    logger.info("Fact_Sales: %d rows", len(fact_sales))

    rfm_to_store = rfm.reset_index().copy()
    rfm_to_store.to_sql("Customer_RFM", engine, index=False, if_exists="replace")
    logger.info("Customer_RFM: %d rows", len(rfm_to_store))

    logger.info("=== STAR SCHEMA LOAD COMPLETE ===")


# ═══════════════════════════════════════════════════════════════════════════
# 3. TRAIN MODELS
# ═══════════════════════════════════════════════════════════════════════════
def find_optimal_k(rfm_scaled, k_range=range(2, 11)):
    """Elbow method + silhouette for optimal k."""
    inertias = []
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(rfm_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(rfm_scaled, labels))
    return list(k_range), inertias, sil_scores


def train_kmeans_model(rfm: pd.DataFrame, logger):
    logger.info("--- Training KMeans Clustering ---")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    logger.info("StandardScaler applied (mean=0, std=1 normalization)")

    k_range, inertias, sil_scores = find_optimal_k(rfm_scaled)
    for k, inert, sil in zip(k_range, inertias, sil_scores):
        logger.info("  k=%d → inertia=%.2f, silhouette=%.4f", k, inert, sil)

    optimal_k = 4  # Chosen based on elbow + domain knowledge
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(rfm_scaled)
    logger.info("Chosen k=%d (elbow + domain knowledge)", optimal_k)

    rfm_with_cluster = rfm.copy()
    rfm_with_cluster["Cluster"] = clusters

    sil_score = silhouette_score(rfm_scaled, clusters)
    ch_score = calinski_harabasz_score(rfm_scaled, clusters)
    logger.info("KMeans Silhouette Score: %.4f", sil_score)
    logger.info("KMeans Calinski-Harabasz Score: %.2f", ch_score)

    # Cluster profiling
    cluster_profile = rfm_with_cluster.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    logger.info("Cluster profiles (mean RFM):\n%s", cluster_profile.to_string())

    return scaler, kmeans, rfm_with_cluster, sil_score, ch_score, k_range, inertias, sil_scores


def train_xgboost_model(clv_data: pd.DataFrame, logger):
    logger.info("--- Training XGBoost Regression (CLV) ---")
    X = clv_data[["Recency", "Frequency", "Monetary"]]
    y = clv_data["FutureSpend"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info("Train/Test split: %d / %d (80/20)", len(X_train), len(X_test))

    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
    )
    logger.info("Hyperparameters: n_estimators=100, learning_rate=0.1, max_depth=5")

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info("XGBoost RMSE: %.2f", rmse)
    logger.info("XGBoost MAE: %.2f", mae)
    logger.info("XGBoost R²: %.4f", r2)

    # Feature importance
    importance = dict(zip(X.columns, xgb_model.feature_importances_))
    logger.info("Feature importances: %s", {k: f"{v:.4f}" for k, v in importance.items()})

    return xgb_model, rmse, mae, r2, y_test, y_pred


def train_fpgrowth_rules(df: pd.DataFrame, logger):
    logger.info("--- Training FP-Growth Association Rules ---")
    basket = (
        df[df["Country"] == "United Kingdom"]
        .groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum()
        .unstack()
        .reset_index()
        .fillna(0)
        .set_index("InvoiceNo")
    )
    logger.info("Basket matrix: %d transactions x %d items", *basket.shape)

    def encode_units(x):
        return 1 if x >= 1 else 0

    basket_sets = basket.map(encode_units)  # Fixed: was applymap (deprecated)

    frequent_itemsets = fpgrowth(basket_sets, min_support=0.015, use_colnames=True)
    logger.info("Frequent itemsets found: %d (min_support=0.015)", len(frequent_itemsets))

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)
    rules.sort_values("lift", ascending=False, inplace=True)
    logger.info("Association rules generated: %d (metric=lift, min_threshold=1.5)", len(rules))

    if not rules.empty:
        logger.info("Top 5 rules by lift:\n%s", rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(5).to_string())

    return rules, frequent_itemsets


# ═══════════════════════════════════════════════════════════════════════════
# 4. SAVE ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════
def save_artifacts(output_dir: str, scaler, kmeans, xgb_model, rules, logger):
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(kmeans, os.path.join(models_dir, "kmeans.pkl"))
    joblib.dump(xgb_model, os.path.join(models_dir, "xgb_model.pkl"))
    joblib.dump(rules, os.path.join(models_dir, "rules.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    logger.info("Model artifacts saved to: %s", models_dir)

    # Also save to root for backward compatibility with streamlit_app.py
    root_dir = os.path.dirname(__file__) or "."
    joblib.dump(kmeans, os.path.join(root_dir, "kmeans.pkl"))
    joblib.dump(xgb_model, os.path.join(root_dir, "xgb_model.pkl"))
    joblib.dump(rules, os.path.join(root_dir, "rules.pkl"))
    joblib.dump(scaler, os.path.join(root_dir, "scaler.pkl"))


# ═══════════════════════════════════════════════════════════════════════════
# 5. GENERATE VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════
def generate_visualizations(
    df: pd.DataFrame,
    rfm: pd.DataFrame,
    rfm_with_cluster: pd.DataFrame,
    y_test, y_pred,
    rules: pd.DataFrame,
    xgb_model,
    k_range, inertias, sil_scores,
    report_dir: str,
    logger,
):
    figure_dir = os.path.join(report_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    logger.info("=== PHASE 4: GENERATING VISUALIZATIONS ===")

    stats = df.attrs if hasattr(df, "attrs") else {}

    # 1. Cleaning flow overview
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = ["Raw\nrows", "After\ndedup", "Remove\nmissing ID", "Remove\ncancelled", "Final\nclean"]
    values = [
        stats.get("raw_rows", len(df)),
        stats.get("after_dedup", len(df)),
        stats.get("rows_after_customer_filter", len(df)),
        stats.get("rows_after_cancellation_filter", len(df)),
        stats.get("clean_rows", len(df)),
    ]
    colors = ["#95a5a6", "#3498db", "#e67e22", "#f39c12", "#27ae60"]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Data Cleaning Pipeline — Row Counts at Each Step", fontweight="bold")
    ax.set_ylabel("Number of Rows")
    _save_plot(fig, os.path.join(figure_dir, "cleaning_flow.png"))
    logger.info("  ✓ cleaning_flow.png")

    # 2. Monthly revenue
    monthly = df.copy()
    monthly["Month"] = monthly["InvoiceDate"].dt.to_period("M").astype(str)
    monthly_revenue = monthly.groupby("Month", as_index=False)["TotalPrice"].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly_revenue["Month"], monthly_revenue["TotalPrice"], marker="o", color="#2c3e50", linewidth=2)
    ax.fill_between(range(len(monthly_revenue)), monthly_revenue["TotalPrice"], alpha=0.15, color="#2c3e50")
    ax.set_title("Monthly Revenue Trend", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (£)")
    ax.tick_params(axis="x", rotation=45)
    _save_plot(fig, os.path.join(figure_dir, "monthly_revenue.png"))
    logger.info("  ✓ monthly_revenue.png")

    # 3. Top 10 countries
    top_countries = df.groupby("Country", as_index=False)["TotalPrice"].sum().sort_values("TotalPrice", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_countries["Country"], top_countries["TotalPrice"], color="#8e44ad")
    ax.set_title("Top 10 Countries by Revenue", fontweight="bold")
    ax.set_xlabel("Revenue (£)")
    ax.invert_yaxis()
    _save_plot(fig, os.path.join(figure_dir, "top_countries.png"))
    logger.info("  ✓ top_countries.png")

    # 4. Top 10 products
    top_products = df.groupby("Description", as_index=False)["TotalPrice"].sum().sort_values("TotalPrice", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_products["Description"], top_products["TotalPrice"], color="#1abc9c")
    ax.set_title("Top 10 Products by Revenue", fontweight="bold")
    ax.set_xlabel("Revenue (£)")
    ax.invert_yaxis()
    _save_plot(fig, os.path.join(figure_dir, "top_products.png"))
    logger.info("  ✓ top_products.png")

    # 5. RFM distributions (histograms)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col, color, title in zip(
        axes,
        ["Recency", "Frequency", "Monetary"],
        ["#2980b9", "#16a085", "#c0392b"],
        ["Recency (days)", "Frequency (transactions)", "Monetary (£)"],
    ):
        sns.histplot(rfm[col], bins=40, ax=ax, color=color, kde=True)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(col)
    _save_plot(fig, os.path.join(figure_dir, "rfm_distributions.png"))
    logger.info("  ✓ rfm_distributions.png")

    # 6. Elbow method
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(k_range, inertias, marker="o", color="#e74c3c", linewidth=2)
    ax1.axvline(x=4, color="#27ae60", linestyle="--", label="Chosen k=4")
    ax1.set_title("Elbow Method — Inertia", fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.legend()

    ax2.plot(k_range, sil_scores, marker="s", color="#3498db", linewidth=2)
    ax2.axvline(x=4, color="#27ae60", linestyle="--", label="Chosen k=4")
    ax2.set_title("Silhouette Score vs k", fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.legend()
    _save_plot(fig, os.path.join(figure_dir, "elbow_silhouette.png"))
    logger.info("  ✓ elbow_silhouette.png")

    # 7. Cluster distribution
    cluster_counts = rfm_with_cluster["Cluster"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_cluster = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    ax.bar(cluster_counts.index.astype(str), cluster_counts.values, color=colors_cluster[:len(cluster_counts)])
    for i, (idx, val) in enumerate(zip(cluster_counts.index, cluster_counts.values)):
        ax.text(i, val + 10, str(val), ha="center", fontweight="bold")
    ax.set_title("Customer Cluster Distribution (KMeans k=4)", fontweight="bold")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    _save_plot(fig, os.path.join(figure_dir, "cluster_distribution.png"))
    logger.info("  ✓ cluster_distribution.png")

    # 8. Cluster profile heatmap
    cluster_profile = rfm_with_cluster.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(cluster_profile, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Cluster Profiles — Average RFM Values", fontweight="bold")
    ax.set_ylabel("Cluster")
    _save_plot(fig, os.path.join(figure_dir, "cluster_heatmap.png"))
    logger.info("  ✓ cluster_heatmap.png")

    # 9. 3D Scatter of clusters
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    for c in sorted(rfm_with_cluster["Cluster"].unique()):
        mask = rfm_with_cluster["Cluster"] == c
        ax.scatter(
            rfm_with_cluster.loc[mask, "Recency"],
            rfm_with_cluster.loc[mask, "Frequency"],
            rfm_with_cluster.loc[mask, "Monetary"],
            label=f"Cluster {c}", alpha=0.5, s=15,
        )
    ax.set_xlabel("Recency")
    ax.set_ylabel("Frequency")
    ax.set_zlabel("Monetary")
    ax.set_title("3D RFM Cluster Visualization", fontweight="bold")
    ax.legend()
    _save_plot(fig, os.path.join(figure_dir, "cluster_3d.png"))
    logger.info("  ✓ cluster_3d.png")

    # 10. XGBoost actual vs predicted
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.4, color="#2980b9", s=20, edgecolors="white", linewidth=0.3)
    min_val = float(min(np.min(y_test), np.min(y_pred)))
    max_val = float(max(np.max(y_test), np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#e74c3c", linewidth=2, label="Perfect prediction")
    ax.set_title("XGBoost — Actual vs Predicted Future Spend", fontweight="bold")
    ax.set_xlabel("Actual Future Spend (£)")
    ax.set_ylabel("Predicted Future Spend (£)")
    ax.legend()
    _save_plot(fig, os.path.join(figure_dir, "xgb_actual_vs_predicted.png"))
    logger.info("  ✓ xgb_actual_vs_predicted.png")

    # 11. XGBoost residuals
    residuals = np.array(y_test) - np.array(y_pred)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(y_pred, residuals, alpha=0.4, color="#8e44ad", s=15)
    ax1.axhline(y=0, color="#e74c3c", linestyle="--")
    ax1.set_title("Residual Plot", fontweight="bold")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Residual")

    sns.histplot(residuals, bins=50, ax=ax2, color="#2ecc71", kde=True)
    ax2.set_title("Residual Distribution", fontweight="bold")
    ax2.set_xlabel("Residual")
    _save_plot(fig, os.path.join(figure_dir, "xgb_residuals.png"))
    logger.info("  ✓ xgb_residuals.png")

    # 12. Feature importance
    importance = xgb_model.feature_importances_
    features = ["Recency", "Frequency", "Monetary"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(features, importance, color=["#e74c3c", "#3498db", "#2ecc71"])
    for bar, val in zip(bars, importance):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", ha="left", va="center", fontweight="bold")
    ax.set_title("XGBoost Feature Importance", fontweight="bold")
    ax.set_xlabel("Importance Score")
    _save_plot(fig, os.path.join(figure_dir, "feature_importance.png"))
    logger.info("  ✓ feature_importance.png")

    # 13. Association rules
    if not rules.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(rules["support"], rules["lift"], alpha=0.5, c=rules["confidence"], cmap="viridis", s=30)
        axes[0].set_title("Support vs Lift (color = Confidence)", fontweight="bold")
        axes[0].set_xlabel("Support")
        axes[0].set_ylabel("Lift")

        axes[1].scatter(rules["support"], rules["confidence"], alpha=0.5, c=rules["lift"], cmap="magma", s=30)
        axes[1].set_title("Support vs Confidence (color = Lift)", fontweight="bold")
        axes[1].set_xlabel("Support")
        axes[1].set_ylabel("Confidence")
        _save_plot(fig, os.path.join(figure_dir, "rules_support_lift.png"))
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No association rules generated", ha="center", va="center", fontsize=14)
        ax.axis("off")
        _save_plot(fig, os.path.join(figure_dir, "rules_support_lift.png"))
    logger.info("  ✓ rules_support_lift.png")

    # 14. Star schema diagram
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Star Schema Design", fontweight="bold", fontsize=16, pad=20)

    # Center: Fact table
    fact_rect = plt.Rectangle((4, 3), 4, 2, fill=True, facecolor="#e74c3c", edgecolor="white", linewidth=2, alpha=0.9)
    ax.add_patch(fact_rect)
    ax.text(6, 4.3, "Fact_Sales", ha="center", va="center", fontweight="bold", fontsize=12, color="white")
    ax.text(6, 3.7, "InvoiceNo | StockCode | CustomerID\nQuantity | TotalPrice | Date", ha="center", va="center", fontsize=8, color="white")

    # Dimension tables
    dims = [
        (0.5, 5.5, 3, 1.5, "Dim_Customer", "CustomerID | Country", "#3498db"),
        (8.5, 5.5, 3, 1.5, "Dim_Product", "StockCode | Description\nUnitPrice", "#2ecc71"),
        (0.5, 1, 3, 1.5, "Dim_Time", "Date | Year | Quarter\nMonth | DayOfWeek", "#f39c12"),
        (8.5, 1, 3, 1.5, "Customer_RFM", "CustomerID | Recency\nFrequency | Monetary", "#9b59b6"),
    ]
    for x, y, w, h, name, cols, color in dims:
        rect = plt.Rectangle((x, y), w, h, fill=True, facecolor=color, edgecolor="white", linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.65, name, ha="center", va="center", fontweight="bold", fontsize=10, color="white")
        ax.text(x + w / 2, y + h * 0.3, cols, ha="center", va="center", fontsize=7, color="white")

    # Connection lines
    connections = [
        (4, 4, 3.5, 6.25),    # Fact → Dim_Customer
        (8, 4, 8.5, 6.25),    # Fact → Dim_Product
        (4, 4, 3.5, 1.75),    # Fact → Dim_Time
        (8, 4, 8.5, 1.75),    # Fact → Customer_RFM
    ]
    for x1, y1, x2, y2 in connections:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color="#7f8c8d", lw=2))
    _save_plot(fig, os.path.join(figure_dir, "star_schema.png"))
    logger.info("  ✓ star_schema.png")

    # 15. Missing value heatmap (before cleaning)
    # We'll create a summary version since raw data is already cleaned
    fig, ax = plt.subplots(figsize=(10, 3))
    missing_data = {
        "CustomerID": stats.get("missing_customerid_pct", 0),
        "Description": stats.get("missing_description_pct", 0),
        "InvoiceNo": 0, "StockCode": 0, "Quantity": 0, "UnitPrice": 0, "InvoiceDate": 0, "Country": 0,
    }
    colors_miss = ["#e74c3c" if v > 0 else "#2ecc71" for v in missing_data.values()]
    bars = ax.barh(list(missing_data.keys()), list(missing_data.values()), color=colors_miss)
    ax.set_title("Missing Value Percentage by Column (Before Cleaning)", fontweight="bold")
    ax.set_xlabel("Missing %")
    for bar, val in zip(bars, missing_data.values()):
        if val > 0:
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontweight="bold")
    _save_plot(fig, os.path.join(figure_dir, "missing_values.png"))
    logger.info("  ✓ missing_values.png")

    # 16. Revenue by day of week
    df_dow = df.copy()
    df_dow["DayOfWeek"] = df_dow["InvoiceDate"].dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_revenue = df_dow.groupby("DayOfWeek")["TotalPrice"].sum().reindex(dow_order).dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(dow_revenue.index, dow_revenue.values, color="#34495e")
    ax.set_title("Revenue by Day of Week", fontweight="bold")
    ax.set_ylabel("Revenue (£)")
    ax.tick_params(axis="x", rotation=30)
    _save_plot(fig, os.path.join(figure_dir, "revenue_by_day.png"))
    logger.info("  ✓ revenue_by_day.png")

    # 17. Hourly distribution
    df_hour = df.copy()
    df_hour["Hour"] = df_hour["InvoiceDate"].dt.hour
    hourly = df_hour.groupby("Hour")["TotalPrice"].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(hourly.index, hourly.values, color="#e67e22")
    ax.set_title("Revenue by Hour of Day", fontweight="bold")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Revenue (£)")
    _save_plot(fig, os.path.join(figure_dir, "revenue_by_hour.png"))
    logger.info("  ✓ revenue_by_hour.png")

    logger.info("=== VISUALIZATION COMPLETE: %d figures generated ===", 17)
    return figure_dir


# ═══════════════════════════════════════════════════════════════════════════
# 6. REPORT ASSETS
# ═══════════════════════════════════════════════════════════════════════════
class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types that are not JSON serializable."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_report_assets(report_dir: str, summary: dict, logger):
    os.makedirs(report_dir, exist_ok=True)

    with open(os.path.join(report_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    logger.info("summary.json written")

    with open(os.path.join(report_dir, "metrics.tex"), "w", encoding="utf-8") as fh:
        fh.write(
            """\\begin{{table}}[H]
\\centering
\\caption{{Pipeline Data Profile}}
\\begin{{tabular}}{{|l|r|}}
\\hline
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\hline
Raw rows (incl. duplicates) & {raw_rows:,} \\\\
After deduplication & {after_dedup:,} \\\\
Duplicates removed & {duplicates_removed:,} \\\\
Rows after customer filter & {rows_after_customer_filter:,} \\\\
Rows after cancellation filter & {rows_after_cancellation_filter:,} \\\\
Final clean rows & {clean_rows:,} \\\\
Unique customers & {unique_customers:,} \\\\
Unique products & {unique_products:,} \\\\
Unique invoices & {unique_invoices:,} \\\\
Unique countries & {unique_countries:,} \\\\
Total revenue & {total_revenue:,.2f} \\\\
Date range & {date_min} to {date_max} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}

\\begin{{table}}[H]
\\centering
\\caption{{AI Model Evaluation Metrics}}
\\begin{{tabular}}{{|l|r|}}
\\hline
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\hline
KMeans Silhouette Score & {silhouette:.4f} \\\\
KMeans Calinski-Harabasz & {calinski_harabasz:.2f} \\\\
XGBoost RMSE & {rmse:.2f} \\\\
XGBoost MAE & {mae:.2f} \\\\
XGBoost R² & {r2:.4f} \\\\
FP-Growth rules generated & {rule_count} \\\\
FP-Growth frequent itemsets & {frequent_itemset_count} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
""".format(**summary))

    logger.info("metrics.tex written")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
def run_pipeline(file_path: str, database_url: str, output_dir: str, report_dir: str):
    logger = setup_logging(report_dir)
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  RETAIL DATA WAREHOUSE + AI PIPELINE                ║")
    logger.info("╚══════════════════════════════════════════════════════╝")
    logger.info("Input file: %s", file_path)
    logger.info("Output dir: %s", output_dir)
    logger.info("Report dir: %s", report_dir)

    engine = _get_engine(database_url, logger)

    # Phase 1: Extract + Transform
    df, rfm, clv_data, stats = extract_transform(file_path, logger)
    df.attrs.update(stats)

    # Phase 2: Load Star Schema
    load_star_schema(df, rfm, engine, logger)

    # Phase 3: Train Models
    logger.info("=== PHASE 3: MODEL TRAINING ===")
    scaler, kmeans, rfm_with_cluster, sil_score, ch_score, k_range, inertias, sil_scores = train_kmeans_model(rfm, logger)
    xgb_model, rmse, mae, r2, y_test, y_pred = train_xgboost_model(clv_data, logger)
    rules, frequent_itemsets = train_fpgrowth_rules(df, logger)

    # Store cluster labels in DW
    rfm_with_cluster.reset_index().to_sql("Customer_Cluster", engine, index=False, if_exists="replace")
    logger.info("Customer_Cluster table saved to database")
    logger.info("=== MODEL TRAINING COMPLETE ===")

    # Phase 4: Save artifacts
    save_artifacts(output_dir, scaler, kmeans, xgb_model, rules, logger)

    # Phase 5: Visualizations
    figure_dir = generate_visualizations(
        df, rfm, rfm_with_cluster, y_test, y_pred, rules,
        xgb_model, k_range, inertias, sil_scores,
        report_dir, logger,
    )

    # Phase 6: Report assets
    summary = {
        **stats,
        "silhouette": float(sil_score),
        "calinski_harabasz": float(ch_score),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "rule_count": int(len(rules)),
        "frequent_itemset_count": int(len(frequent_itemsets)),
        "cluster_counts": {str(k): int(v) for k, v in rfm_with_cluster["Cluster"].value_counts().sort_index().items()},
        "feature_importance": {feat: float(imp) for feat, imp in zip(["Recency", "Frequency", "Monetary"], xgb_model.feature_importances_)},
        "cluster_profiles": {
            col: {str(k): float(v) for k, v in vals.items()}
            for col, vals in rfm_with_cluster.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().to_dict().items()
        },
    }
    write_report_assets(report_dir, summary, logger)

    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  PIPELINE COMPLETED SUCCESSFULLY                    ║")
    logger.info("╚══════════════════════════════════════════════════════╝")


def parse_args():
    parser = argparse.ArgumentParser(description="Retail ETL + DW + AI Pipeline")
    parser.add_argument("--file-path", default=os.getenv("ONLINE_RETAIL_CSV", DEFAULT_KAGGLE_PATH))
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL))
    parser.add_argument("--output-dir", default=os.getenv("MODEL_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-dir", default=os.getenv("REPORT_OUTPUT_DIR", DEFAULT_REPORT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.file_path, args.database_url, args.output_dir, args.report_dir)
