import os

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from ..utilities import save_plot

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")


def generate_visualizations(
    df: pd.DataFrame,
    rfm: pd.DataFrame,
    rfm_with_cluster: pd.DataFrame,
    y_test,
    y_pred,
    rules: pd.DataFrame,
    xgb_model,
    k_range,
    inertias,
    sil_scores,
    report_dir: str,
    logger,
):
    figure_dir = os.path.join(report_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    logger.info("=== PHASE 4: GENERATING VISUALIZATIONS ===")

    stats = df.attrs if hasattr(df, "attrs") else {}

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
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500, f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Data Cleaning Pipeline — Row Counts at Each Step", fontweight="bold")
    ax.set_ylabel("Number of Rows")
    save_plot(fig, os.path.join(figure_dir, "cleaning_flow.png"))
    logger.info("  ✓ cleaning_flow.png")

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
    save_plot(fig, os.path.join(figure_dir, "monthly_revenue.png"))
    logger.info("  ✓ monthly_revenue.png")

    top_countries = df.groupby("Country", as_index=False)["TotalPrice"].sum().sort_values("TotalPrice", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_countries["Country"], top_countries["TotalPrice"], color="#8e44ad")
    ax.set_title("Top 10 Countries by Revenue", fontweight="bold")
    ax.set_xlabel("Revenue (£)")
    ax.invert_yaxis()
    save_plot(fig, os.path.join(figure_dir, "top_countries.png"))
    logger.info("  ✓ top_countries.png")

    top_products = df.groupby("Description", as_index=False)["TotalPrice"].sum().sort_values("TotalPrice", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_products["Description"], top_products["TotalPrice"], color="#1abc9c")
    ax.set_title("Top 10 Products by Revenue", fontweight="bold")
    ax.set_xlabel("Revenue (£)")
    ax.invert_yaxis()
    save_plot(fig, os.path.join(figure_dir, "top_products.png"))
    logger.info("  ✓ top_products.png")

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
    save_plot(fig, os.path.join(figure_dir, "rfm_distributions.png"))
    logger.info("  ✓ rfm_distributions.png")

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
    save_plot(fig, os.path.join(figure_dir, "elbow_silhouette.png"))
    logger.info("  ✓ elbow_silhouette.png")

    cluster_counts = rfm_with_cluster["Cluster"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_cluster = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    ax.bar(cluster_counts.index.astype(str), cluster_counts.values, color=colors_cluster[: len(cluster_counts)])
    for i, (_, val) in enumerate(zip(cluster_counts.index, cluster_counts.values)):
        ax.text(i, val + 10, str(val), ha="center", fontweight="bold")
    ax.set_title("Customer Cluster Distribution (KMeans k=4)", fontweight="bold")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    save_plot(fig, os.path.join(figure_dir, "cluster_distribution.png"))
    logger.info("  ✓ cluster_distribution.png")

    cluster_profile = rfm_with_cluster.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(cluster_profile, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Cluster Profiles — Average RFM Values", fontweight="bold")
    ax.set_ylabel("Cluster")
    save_plot(fig, os.path.join(figure_dir, "cluster_heatmap.png"))
    logger.info("  ✓ cluster_heatmap.png")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    for c in sorted(rfm_with_cluster["Cluster"].unique()):
        mask = rfm_with_cluster["Cluster"] == c
        ax.scatter(
            rfm_with_cluster.loc[mask, "Recency"],
            rfm_with_cluster.loc[mask, "Frequency"],
            rfm_with_cluster.loc[mask, "Monetary"],
            label=f"Cluster {c}",
            alpha=0.5,
            s=15,
        )
    ax.set_xlabel("Recency")
    ax.set_ylabel("Frequency")
    ax.set_zlabel("Monetary")
    ax.set_title("3D RFM Cluster Visualization", fontweight="bold")
    ax.legend()
    save_plot(fig, os.path.join(figure_dir, "cluster_3d.png"))
    logger.info("  ✓ cluster_3d.png")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.4, color="#2980b9", s=20, edgecolors="white", linewidth=0.3)
    min_val = float(min(np.min(y_test), np.min(y_pred)))
    max_val = float(max(np.max(y_test), np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#e74c3c", linewidth=2, label="Perfect prediction")
    ax.set_title("XGBoost — Actual vs Predicted Future Spend", fontweight="bold")
    ax.set_xlabel("Actual Future Spend (£)")
    ax.set_ylabel("Predicted Future Spend (£)")
    ax.legend()
    save_plot(fig, os.path.join(figure_dir, "xgb_actual_vs_predicted.png"))
    logger.info("  ✓ xgb_actual_vs_predicted.png")

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
    save_plot(fig, os.path.join(figure_dir, "xgb_residuals.png"))
    logger.info("  ✓ xgb_residuals.png")

    importance = xgb_model.feature_importances_
    features = ["Recency", "Frequency", "Monetary"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(features, importance, color=["#e74c3c", "#3498db", "#2ecc71"])
    for bar, val in zip(bars, importance):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", ha="left", va="center", fontweight="bold")
    ax.set_title("XGBoost Feature Importance", fontweight="bold")
    ax.set_xlabel("Importance Score")
    save_plot(fig, os.path.join(figure_dir, "feature_importance.png"))
    logger.info("  ✓ feature_importance.png")

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
        save_plot(fig, os.path.join(figure_dir, "rules_support_lift.png"))
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No association rules generated", ha="center", va="center", fontsize=14)
        ax.axis("off")
        save_plot(fig, os.path.join(figure_dir, "rules_support_lift.png"))
    logger.info("  ✓ rules_support_lift.png")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Star Schema Design", fontweight="bold", fontsize=16, pad=20)

    fact_rect = plt.Rectangle((4, 3), 4, 2, fill=True, facecolor="#e74c3c", edgecolor="white", linewidth=2, alpha=0.9)
    ax.add_patch(fact_rect)
    ax.text(6, 4.3, "Fact_Sales", ha="center", va="center", fontweight="bold", fontsize=12, color="white")
    ax.text(6, 3.7, "InvoiceNo | StockCode | CustomerID\nQuantity | TotalPrice | Date", ha="center", va="center", fontsize=8, color="white")

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

    connections = [
        (4, 4, 3.5, 6.25),
        (8, 4, 8.5, 6.25),
        (4, 4, 3.5, 1.75),
        (8, 4, 8.5, 1.75),
    ]
    for x1, y1, x2, y2 in connections:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#7f8c8d", lw=2))
    save_plot(fig, os.path.join(figure_dir, "star_schema.png"))
    logger.info("  ✓ star_schema.png")

    fig, ax = plt.subplots(figsize=(10, 3))
    missing_data = {
        "CustomerID": stats.get("missing_customerid_pct", 0),
        "Description": stats.get("missing_description_pct", 0),
        "InvoiceNo": 0,
        "StockCode": 0,
        "Quantity": 0,
        "UnitPrice": 0,
        "InvoiceDate": 0,
        "Country": 0,
    }
    colors_miss = ["#e74c3c" if v > 0 else "#2ecc71" for v in missing_data.values()]
    bars = ax.barh(list(missing_data.keys()), list(missing_data.values()), color=colors_miss)
    ax.set_title("Missing Value Percentage by Column (Before Cleaning)", fontweight="bold")
    ax.set_xlabel("Missing %")
    for bar, val in zip(bars, missing_data.values()):
        if val > 0:
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontweight="bold")
    save_plot(fig, os.path.join(figure_dir, "missing_values.png"))
    logger.info("  ✓ missing_values.png")

    df_dow = df.copy()
    df_dow["DayOfWeek"] = df_dow["InvoiceDate"].dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_revenue = df_dow.groupby("DayOfWeek")["TotalPrice"].sum().reindex(dow_order).dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(dow_revenue.index, dow_revenue.values, color="#34495e")
    ax.set_title("Revenue by Day of Week", fontweight="bold")
    ax.set_ylabel("Revenue (£)")
    ax.tick_params(axis="x", rotation=30)
    save_plot(fig, os.path.join(figure_dir, "revenue_by_day.png"))
    logger.info("  ✓ revenue_by_day.png")

    df_hour = df.copy()
    df_hour["Hour"] = df_hour["InvoiceDate"].dt.hour
    hourly = df_hour.groupby("Hour")["TotalPrice"].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(hourly.index, hourly.values, color="#e67e22")
    ax.set_title("Revenue by Hour of Day", fontweight="bold")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Revenue (£)")
    save_plot(fig, os.path.join(figure_dir, "revenue_by_hour.png"))
    logger.info("  ✓ revenue_by_hour.png")

    logger.info("=== VISUALIZATION COMPLETE: %d figures generated ===", 17)
    return figure_dir
