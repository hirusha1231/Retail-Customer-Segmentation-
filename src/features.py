import numpy as np
import pandas as pd

def create_rfm_features(df: pd.DataFrame, reference_date: str = "2011-12-10") -> pd.DataFrame:
    """
    Computes customer-level RFM features using calendar-day recency and log1p transformations.
    Ensures input immutability, unique CustomerID index, and strict data quality assertions.
    """
    df_work = df.copy()
    df_work["InvoiceDate"] = pd.to_datetime(df_work["InvoiceDate"])
    cutoff = pd.to_datetime(reference_date).normalize()

    # Pre-condition assertion: Cutoff must strictly exceed the latest transaction
    assert cutoff > df_work["InvoiceDate"].max(), "Reference date must strictly exceed the latest transaction date."

    # Customer aggregation using calendar-day recency
    rfm = df_work.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (cutoff - x.max().normalize()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"
    }).rename(columns={
        "InvoiceDate": "Recency",
        "InvoiceNo": "Frequency",
        "TotalPrice": "Monetary"
    }).reset_index()

    # Log1p feature transformation to manage right-skewed distributions
    rfm["Recency_log"] = np.log1p(rfm["Recency"])
    rfm["Frequency_log"] = np.log1p(rfm["Frequency"])
    rfm["Monetary_log"] = np.log1p(rfm["Monetary"])

    # Strict Quality Assertions
    assert rfm["CustomerID"].is_unique, "CustomerID must be unique across all aggregated rows."
    assert (rfm["Recency"] >= 1).all(), "Recency must be >= 1 calendar day."
    assert (rfm["Frequency"] >= 1).all(), "Frequency must be >= 1 order."
    assert (rfm["Monetary"] > 0).all(), "Monetary spend must be strictly > 0 for positive sales cohort."
    assert np.isfinite(rfm[["Recency_log", "Frequency_log", "Monetary_log"]].values).all(), "Transformed features contain non-finite values."

    return rfm
