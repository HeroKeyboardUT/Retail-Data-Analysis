import json
import os

import numpy as np


class NumpyEncoder(json.JSONEncoder):
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
        json.dump(summary, fh, indent=2, ensure_ascii=False, cls=NumpyEncoder)
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
KMeans Davies-Bouldin & {davies_bouldin:.4f} \\\\
XGBoost RMSE & {rmse:.2f} \\\\
XGBoost MAE & {mae:.2f} \\\\
XGBoost MedAE & {medae:.2f} \\\\
XGBoost R² & {r2:.4f} \\\\
XGBoost Explained Variance & {explained_variance:.4f} \\\\
XGBoost sMAPE (\\%) & {smape:.2f} \\\\
FP-Growth rules generated & {rule_count} \\\\
FP-Growth frequent itemsets & {frequent_itemset_count} \\\\
FP-Growth avg support & {avg_rule_support:.4f} \\\\
FP-Growth avg confidence & {avg_rule_confidence:.4f} \\\\
FP-Growth avg lift & {avg_rule_lift:.4f} \\\\
FP-Growth max lift & {max_rule_lift:.4f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
""".format(**summary)
        )

    logger.info("metrics.tex written")
