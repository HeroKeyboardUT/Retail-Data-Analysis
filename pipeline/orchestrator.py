from .phases.extract_transform import extract_transform
from .phases.generate_visualizations import generate_visualizations
from .phases.load_star_schema import load_star_schema
from .phases.report_assets import write_report_assets
from .phases.save_artifacts import save_artifacts
from .phases.train_models import train_fpgrowth_rules, train_kmeans_model, train_xgboost_model
from .utilities import get_engine, setup_logging


def run_pipeline(file_path: str, database_url: str, output_dir: str, report_dir: str):
    logger = setup_logging(report_dir)
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  RETAIL DATA WAREHOUSE + AI PIPELINE                ║")
    logger.info("╚══════════════════════════════════════════════════════╝")
    logger.info("Input file: %s", file_path)
    logger.info("Output dir: %s", output_dir)
    logger.info("Report dir: %s", report_dir)

    engine = get_engine(database_url, logger)

    df, rfm, clv_data, stats = extract_transform(file_path, logger)
    df.attrs.update(stats)

    load_star_schema(df, rfm, engine, logger)

    logger.info("=== PHASE 3: MODEL TRAINING ===")
    scaler, kmeans, rfm_with_cluster, sil_score, ch_score, db_score, k_range, inertias, sil_scores = train_kmeans_model(rfm, logger)
    xgb_model, rmse, mae, medae, r2, explained_var, smape, y_test, y_pred = train_xgboost_model(clv_data, logger)
    rules, frequent_itemsets, rule_metrics = train_fpgrowth_rules(df, logger)

    rfm_with_cluster.reset_index().to_sql("Customer_Cluster", engine, index=False, if_exists="replace")
    logger.info("Customer_Cluster table saved to database")
    logger.info("=== MODEL TRAINING COMPLETE ===")

    save_artifacts(output_dir, scaler, kmeans, xgb_model, rules, logger)

    generate_visualizations(
        df,
        rfm,
        rfm_with_cluster,
        y_test,
        y_pred,
        rules,
        xgb_model,
        k_range,
        inertias,
        sil_scores,
        report_dir,
        logger,
    )

    summary = {
        **stats,
        "silhouette": float(sil_score),
        "calinski_harabasz": float(ch_score),
        "davies_bouldin": float(db_score),
        "rmse": float(rmse),
        "mae": float(mae),
        "medae": float(medae),
        "r2": float(r2),
        "explained_variance": float(explained_var),
        "smape": float(smape),
        "rule_count": int(len(rules)),
        "frequent_itemset_count": int(len(frequent_itemsets)),
        "avg_rule_support": float(rule_metrics["avg_support"]),
        "avg_rule_confidence": float(rule_metrics["avg_confidence"]),
        "avg_rule_lift": float(rule_metrics["avg_lift"]),
        "max_rule_lift": float(rule_metrics["max_lift"]),
        "cluster_counts": {str(k): int(v) for k, v in rfm_with_cluster["Cluster"].value_counts().sort_index().items()},
        "feature_importance": {
            feat: float(imp)
            for feat, imp in zip(["Recency", "Frequency", "Monetary"], xgb_model.feature_importances_)
        },
        "cluster_profiles": {
            col: {str(k): float(v) for k, v in vals.items()}
            for col, vals in rfm_with_cluster.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().to_dict().items()
        },
    }
    write_report_assets(report_dir, summary, logger)

    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  PIPELINE COMPLETED SUCCESSFULLY                    ║")
    logger.info("╚══════════════════════════════════════════════════════╝")
