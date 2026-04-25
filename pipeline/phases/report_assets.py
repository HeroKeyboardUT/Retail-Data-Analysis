import json
import os

import numpy as np
import pandas as pd


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

    def latex_escape(text: str) -> str:
        return (
            str(text)
            .replace("\\", r"\textbackslash{}")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("_", r"\_")
            .replace("#", r"\#")
            .replace("$", r"\$")
            .replace("{", r"\{")
            .replace("}", r"\}")
        )

    optional_sections = []

    if summary.get("kmeans_probe_predictions"):
        kmeans_probe_df = pd.DataFrame(summary["kmeans_probe_predictions"])
        kmeans_rows = []
        for row in kmeans_probe_df.itertuples(index=False):
            profile_escaped = row.profile.replace('_', '\\_')
            kmeans_rows.append(f"{profile_escaped} & {row.Recency:.2f} & {row.Frequency:.2f} & {row.Monetary:.2f} & {int(row.predicted_cluster)}")
        optional_sections.append(
            r"""
\begin{{table}}[H]
\centering
\caption{{KMeans Probe Predictions on New Customer Profiles}}
\smallskip
\begin{{tabular}}{{lrrrr}}
\toprule
\textbf{{Profile}} & \textbf{{Recency}} & \textbf{{Frequency}} & \textbf{{Monetary}} & \textbf{{Cluster}} \\
\midrule
{rows} \\
\bottomrule
\end{{tabular}}
\end{{table}}
""".format(
                rows=" \\\\\n".join(kmeans_rows)
            )
        )

    if summary.get("xgb_holdout_examples"):
        xgb_holdout_df = pd.DataFrame(summary["xgb_holdout_examples"])
        xgb_rows = []
        for row in xgb_holdout_df.itertuples(index=False):
            xgb_rows.append(f"{row.Recency:.2f} & {row.Frequency:.2f} & {row.Monetary:.2f} & {row.actual_future_spend:.2f} & {row.predicted_future_spend:.2f} & {row.error:.2f}")
        optional_sections.append(
            r"""
\begin{{table}}[H]
\centering
\caption{{XGBoost Holdout Test Examples}}
\smallskip
\begin{{tabular}}{{rrrrrr}}
\toprule
\textbf{{Recency}} & \textbf{{Frequency}} & \textbf{{Monetary}} & \textbf{{Actual}} & \textbf{{Predicted}} & \textbf{{Error}} \\
\midrule
{rows} \\
\bottomrule
\end{{tabular}}
\end{{table}}
""".format(
                rows=" \\\\\n".join(xgb_rows)
            )
        )

    if summary.get("top_rules_examples"):
        top_rules_df = pd.DataFrame(summary["top_rules_examples"])
        top_rules_df = top_rules_df.head(5)
        rules_rows = []
        for row in top_rules_df.itertuples(index=False):
            antecedents_escaped = latex_escape("; ".join(row.antecedents))
            consequents_escaped = latex_escape("; ".join(row.consequents))
            rules_rows.append(
                f"{antecedents_escaped} & {consequents_escaped} & {row.support:.4f} & {row.confidence:.4f} & {row.lift:.4f}"
            )
        optional_sections.append(
            r"""
\begin{{table}}[H]
\centering
\caption{{Top Association Rules}}
\smallskip
\footnotesize
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{4.2cm}}>{{\raggedright\arraybackslash}}p{{4.2cm}}rrr}}
\toprule
\textbf{{Antecedents}} & \textbf{{Consequents}} & \textbf{{Support}} & \textbf{{Confidence}} & \textbf{{Lift}} \\
\midrule
{rows} \\
\bottomrule
\end{{tabular}}
\end{{table}}
""".format(
                rows=" \\\\\n".join(rules_rows)
            )
        )

    if summary.get("customer_recommendation_example"):
        recommendation = summary["customer_recommendation_example"]
        purchased_items = recommendation["purchased_items"]
        purchased_preview = ", ".join(purchased_items[:3])
        if len(purchased_items) > 3:
            purchased_preview += f", ... (+{len(purchased_items) - 3} more)"

        purchased_items_escaped = latex_escape(purchased_preview)
        matched_antecedents_escaped = latex_escape("; ".join(recommendation["matched_antecedents"]))
        recommendations_escaped = latex_escape("; ".join(recommendation["recommendations"]))
        optional_sections.append(
            r"""
\begin{{table}}[H]
\centering
\caption{{Customer Recommendation Example}}
\smallskip
\footnotesize
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{3.6cm}}>{{\raggedright\arraybackslash}}p{{8.6cm}}}}
\toprule
\textbf{{Field}} & \textbf{{Value}} \\
\midrule
CustomerID & {customer_id} \\
Purchased items (sample) & {purchased_items} \\
Matched antecedents & {matched_antecedents} \\
Recommended item(s) & {recommendations} \\
Support & {support:.4f} \\
Confidence & {confidence:.4f} \\
Lift & {lift:.4f} \\
\bottomrule
\end{{tabular}}
\end{{table}}
""".format(
                customer_id=recommendation["customer_id"],
                purchased_items=purchased_items_escaped,
                matched_antecedents=matched_antecedents_escaped,
                recommendations=recommendations_escaped,
                support=float(recommendation["support"]),
                confidence=float(recommendation["confidence"]),
                lift=float(recommendation["lift"]),
            )
        )

    with open(os.path.join(report_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    logger.info("summary.json written")

    with open(os.path.join(report_dir, "metrics.tex"), "w", encoding="utf-8") as fh:
        fh.write(
            """\\begin{{table}}[H]
\\centering
\\caption{{Pipeline Data Profile}}
\\smallskip
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
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
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{table}}[H]
\\centering
\\caption{{AI Model Evaluation Metrics}}
\\smallskip
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
KMeans Silhouette Score & {silhouette:.4f} \\\\
KMeans Calinski-Harabasz & {calinski_harabasz:.2f} \\\\
KMeans Davies-Bouldin & {davies_bouldin:.4f} \\\\
XGBoost RMSE & {rmse:.2f} \\\\
XGBoost MAE & {mae:.2f} \\\\
XGBoost MedAE & {medae:.2f} \\\\
XGBoost R$^2$ & {r2:.4f} \\\\
XGBoost Explained Variance & {explained_variance:.4f} \\\\
XGBoost sMAPE (\\%) & {smape:.2f} \\\\
FP-Growth rules generated & {rule_count} \\\\
FP-Growth frequent itemsets & {frequent_itemset_count} \\\\
FP-Growth avg support & {avg_rule_support:.4f} \\\\
FP-Growth avg confidence & {avg_rule_confidence:.4f} \\\\
FP-Growth avg lift & {avg_rule_lift:.4f} \\\\
FP-Growth max lift & {max_rule_lift:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
{optional_sections}
""".format(optional_sections="\n".join(optional_sections), **summary)
        )

    logger.info("metrics.tex written")
