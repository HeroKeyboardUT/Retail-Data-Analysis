"""
Retail ETL + Data Warehouse + AI Pipeline
==========================================
Thin CLI entrypoint that delegates to modular pipeline phases.
"""

import argparse
import os
import warnings

from pipeline.config import (
    DEFAULT_DATABASE_URL,
    DEFAULT_KAGGLE_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_DIR,
)
from pipeline.orchestrator import run_pipeline

warnings.filterwarnings("ignore")


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
