import pandas as pd


def load_star_schema(df: pd.DataFrame, rfm: pd.DataFrame, engine, logger):
    logger.info("=== PHASE 2: LOAD STAR SCHEMA ===")

    dim_customer = df[["CustomerID", "Country"]].drop_duplicates()
    dim_customer.to_sql("Dim_Customer", engine, index=False, if_exists="replace")
    logger.info("Dim_Customer: %d rows", len(dim_customer))

    dim_product = df[["StockCode", "Description", "UnitPrice"]].drop_duplicates(subset=["StockCode"])
    dim_product.to_sql("Dim_Product", engine, index=False, if_exists="replace")
    logger.info("Dim_Product: %d rows", len(dim_product))

    dim_time = pd.DataFrame({"Date": df["InvoiceDate"].dt.date.unique()})
    dim_time["Date_Obj"] = pd.to_datetime(dim_time["Date"])
    dim_time["Year"] = dim_time["Date_Obj"].dt.year
    dim_time["Quarter"] = dim_time["Date_Obj"].dt.quarter
    dim_time["Month"] = dim_time["Date_Obj"].dt.month
    dim_time["DayOfWeek"] = dim_time["Date_Obj"].dt.dayofweek
    dim_time["Date"] = dim_time["Date"].astype(str)
    dim_time.drop(columns=["Date_Obj"], inplace=True)
    dim_time.to_sql("Dim_Time", engine, index=False, if_exists="replace")
    logger.info("Dim_Time: %d rows (added Quarter, DayOfWeek)", len(dim_time))

    fact_sales = df[["InvoiceNo", "StockCode", "CustomerID", "Quantity", "TotalPrice"]].copy()
    fact_sales["Date"] = df["InvoiceDate"].dt.strftime("%Y-%m-%d")
    fact_sales.to_sql("Fact_Sales", engine, index=False, if_exists="replace")
    logger.info("Fact_Sales: %d rows", len(fact_sales))

    rfm_to_store = rfm.reset_index().copy()
    rfm_to_store.to_sql("Customer_RFM", engine, index=False, if_exists="replace")
    logger.info("Customer_RFM: %d rows", len(rfm_to_store))

    logger.info("=== STAR SCHEMA LOAD COMPLETE ===")
