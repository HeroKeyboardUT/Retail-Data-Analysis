import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__)) or "."

DEFAULT_KAGGLE_PATH = os.path.join(PROJECT_ROOT, "data", "online_retail.csv")
DEFAULT_DATABASE_URL = "postgresql+psycopg2://admin:admin@localhost:5432/retail_dw"
FALLBACK_SQLITE_PATH = os.path.join(PROJECT_ROOT, "db", "retail_dw_fallback.db")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DEFAULT_REPORT_DIR = os.path.join(PROJECT_ROOT, "report", "generated")
