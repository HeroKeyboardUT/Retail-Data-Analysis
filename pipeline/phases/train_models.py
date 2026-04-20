import numpy as np
import pandas as pd
import xgboost as xgb
from mlxtend.frequent_patterns import association_rules, fpgrowth
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


def _safe_smape(y_true, y_pred, epsilon: float = 1e-8) -> float:
    """Symmetric MAPE in percentage, robust when targets contain zeros."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true_arr) + np.abs(y_pred_arr) + epsilon
    return float(np.mean((2.0 * np.abs(y_pred_arr - y_true_arr)) / denom) * 100.0)


def train_kmeans_model(rfm: pd.DataFrame, logger):
    logger.info("--- Training KMeans Clustering ---")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    logger.info("StandardScaler applied (mean=0, std=1 normalization)")

    k_range, inertias, sil_scores = find_optimal_k(rfm_scaled)
    for k, inert, sil in zip(k_range, inertias, sil_scores):
        logger.info("  k=%d → inertia=%.2f, silhouette=%.4f", k, inert, sil)

    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(rfm_scaled)
    logger.info("Chosen k=%d (elbow + domain knowledge)", optimal_k)

    rfm_with_cluster = rfm.copy()
    rfm_with_cluster["Cluster"] = clusters

    sil_score = silhouette_score(rfm_scaled, clusters)
    ch_score = calinski_harabasz_score(rfm_scaled, clusters)
    db_score = davies_bouldin_score(rfm_scaled, clusters)
    logger.info("KMeans Silhouette Score: %.4f", sil_score)
    logger.info("KMeans Calinski-Harabasz Score: %.2f", ch_score)
    logger.info("KMeans Davies-Bouldin Score: %.4f (lower is better)", db_score)

    cluster_profile = rfm_with_cluster.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    logger.info("Cluster profiles (mean RFM):\n%s", cluster_profile.to_string())

    return scaler, kmeans, rfm_with_cluster, sil_score, ch_score, db_score, k_range, inertias, sil_scores


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
    medae = median_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    explained_var = explained_variance_score(y_test, y_pred)
    smape = _safe_smape(y_test, y_pred)

    logger.info("XGBoost RMSE: %.2f", rmse)
    logger.info("XGBoost MAE: %.2f", mae)
    logger.info("XGBoost MedAE: %.2f", medae)
    logger.info("XGBoost R²: %.4f", r2)
    logger.info("XGBoost Explained Variance: %.4f", explained_var)
    logger.info("XGBoost sMAPE: %.2f%%", smape)

    importance = dict(zip(X.columns, xgb_model.feature_importances_))
    logger.info("Feature importances: %s", {k: f"{v:.4f}" for k, v in importance.items()})

    return xgb_model, rmse, mae, medae, r2, explained_var, smape, y_test, y_pred


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

    basket_sets = basket.map(encode_units)

    frequent_itemsets = fpgrowth(basket_sets, min_support=0.015, use_colnames=True)
    logger.info("Frequent itemsets found: %d (min_support=0.015)", len(frequent_itemsets))

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)
    rules.sort_values("lift", ascending=False, inplace=True)
    logger.info("Association rules generated: %d (metric=lift, min_threshold=1.5)", len(rules))

    if not rules.empty:
        rule_metrics = {
            "avg_support": float(rules["support"].mean()),
            "avg_confidence": float(rules["confidence"].mean()),
            "avg_lift": float(rules["lift"].mean()),
            "max_lift": float(rules["lift"].max()),
        }
    else:
        rule_metrics = {
            "avg_support": 0.0,
            "avg_confidence": 0.0,
            "avg_lift": 0.0,
            "max_lift": 0.0,
        }

    logger.info(
        "FP-Growth metrics: avg_support=%.4f, avg_confidence=%.4f, avg_lift=%.4f, max_lift=%.4f",
        rule_metrics["avg_support"],
        rule_metrics["avg_confidence"],
        rule_metrics["avg_lift"],
        rule_metrics["max_lift"],
    )

    if not rules.empty:
        logger.info(
            "Top 5 rules by lift:\n%s",
            rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(5).to_string(),
        )

    return rules, frequent_itemsets, rule_metrics
