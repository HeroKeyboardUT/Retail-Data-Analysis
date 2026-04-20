import logging
import os

import matplotlib
from sqlalchemy import create_engine, text

from .config import FALLBACK_SQLITE_PATH

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def get_engine(database_url: str, logger):
    """Try PostgreSQL first; fall back to SQLite if unreachable."""
    if "postgresql" in database_url:
        try:
            engine = create_engine(database_url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Connected to PostgreSQL: %s", engine.url.render_as_string(hide_password=True))
            return engine
        except Exception as exc:
            logger.warning("PostgreSQL unavailable (%s). Falling back to SQLite.", exc)

    sqlite_url = f"sqlite:///{FALLBACK_SQLITE_PATH}"
    engine = create_engine(sqlite_url)
    logger.info("Using SQLite fallback: %s", FALLBACK_SQLITE_PATH)
    return engine


def save_plot(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
