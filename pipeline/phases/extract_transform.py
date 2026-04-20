import datetime as dt

import pandas as pd


def extract_transform(file_path: str, logger):
    logger.info("=== PHASE 1: EXTRACT + TRANSFORM ===")
    logger.info("Reading CSV: %s", file_path)
    df_raw = pd.read_csv(file_path, encoding="ISO-8859-1")
    logger.info("Raw dataset: %d rows x %d columns", *df_raw.shape)
    logger.info("Columns: %s", list(df_raw.columns))
    logger.info("Data types:\n%s", df_raw.dtypes.to_string())

    missing = df_raw.isnull().sum()
    missing_pct = (missing / len(df_raw) * 100).round(2)
    missing_report = pd.DataFrame({"Missing": missing, "Pct": missing_pct})
    logger.info("Missing values:\n%s", missing_report[missing_report["Missing"] > 0].to_string())

    dup_count = df_raw.duplicated().sum()
    logger.info("Duplicate rows: %d", dup_count)
    df_raw = df_raw.drop_duplicates()
    logger.info("After deduplication: %d rows", len(df_raw))

    df_step_1 = df_raw.dropna(subset=["CustomerID"])
    logger.info("After removing missing CustomerID: %d rows (removed %d)", len(df_step_1), len(df_raw) - len(df_step_1))

    df_step_2 = df_step_1[~df_step_1["InvoiceNo"].astype(str).str.startswith("C")]
    logger.info("After removing cancelled invoices: %d rows (removed %d)", len(df_step_2), len(df_step_1) - len(df_step_2))

    df = df_step_2[(df_step_2["Quantity"] > 0) & (df_step_2["UnitPrice"] > 0)].copy()
    logger.info("After removing invalid qty/price: %d rows (removed %d)", len(df), len(df_step_2) - len(df))

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["CustomerID"] = df["CustomerID"].astype(int)
    logger.info("Created TotalPrice = Quantity × UnitPrice")
    logger.info("Converted InvoiceDate to datetime, CustomerID to int")

    logger.info("Descriptive statistics of numerical columns:\n%s", df[["Quantity", "UnitPrice", "TotalPrice"]].describe().to_string())

    snapshot_date = df["InvoiceDate"].max() + dt.timedelta(days=1)
    split_date = df["InvoiceDate"].max() - pd.DateOffset(months=3)

    df_train_period = df[df["InvoiceDate"] < split_date]
    df_test_period = df[df["InvoiceDate"] >= split_date]
    logger.info("Train period: %s to %s (%d rows)", df_train_period["InvoiceDate"].min().date(), split_date.date(), len(df_train_period))
    logger.info("Test period: %s to %s (%d rows)", split_date.date(), df["InvoiceDate"].max().date(), len(df_test_period))

    rfm = df.groupby("CustomerID").agg(
        {
            "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
            "InvoiceNo": "nunique",
            "TotalPrice": "sum",
        }
    ).rename(columns={"InvoiceDate": "Recency", "InvoiceNo": "Frequency", "TotalPrice": "Monetary"})
    logger.info("RFM table created: %d customers", len(rfm))
    logger.info("RFM statistics:\n%s", rfm.describe().to_string())

    rfm_features = df_train_period.groupby("CustomerID").agg(
        {
            "InvoiceDate": lambda x: (split_date - x.max()).days,
            "InvoiceNo": "nunique",
            "TotalPrice": "sum",
        }
    ).rename(columns={"InvoiceDate": "Recency", "InvoiceNo": "Frequency", "TotalPrice": "Monetary"})

    future_spend = df_test_period.groupby("CustomerID")["TotalPrice"].sum().rename("FutureSpend")
    clv_data = rfm_features.join(future_spend).fillna(0)
    logger.info("CLV dataset: %d customers (with FutureSpend)", len(clv_data))

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
