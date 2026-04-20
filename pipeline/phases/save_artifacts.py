import os

import joblib


def save_artifacts(output_dir: str, scaler, kmeans, xgb_model, rules, logger):
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(kmeans, os.path.join(models_dir, "kmeans.pkl"))
    joblib.dump(xgb_model, os.path.join(models_dir, "xgb_model.pkl"))
    joblib.dump(rules, os.path.join(models_dir, "rules.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    logger.info("Model artifacts saved to: %s", models_dir)
