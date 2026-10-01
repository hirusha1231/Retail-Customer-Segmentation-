import hashlib
import pandas as pd

def compute_file_sha256(filepath: str) -> str:
    """
    Computes SHA-256 cryptographic hash of a raw dataset file for provenance and auditability.
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def audit_and_clean_data(filepath: str, return_audit: bool = False):
    """
    Cleans raw retail transaction records and tracks row/customer retention at each stage.
    Separates return transactions for downstream validation and produces positive sales cohort.
    """
    df_raw = pd.read_excel(filepath)
    audit_records = []
    
    # 0. Initial Ingestion Baseline
    audit_records.append({
        "Stage": "0. Ingestion (Raw Data)",
        "Rows_Retained": len(df_raw),
        "Rows_Dropped": 0,
        "Unique_Customers": df_raw["CustomerID"].nunique(dropna=True),
        "Description": "Raw unedited data loaded directly from source repository."
    })
    
    # 1. Missing CustomerID Filter
    has_customer = df_raw["CustomerID"].notna()
    df_cust = df_raw[has_customer].copy()
    audit_records.append({
        "Stage": "1. CustomerID Imputation Filter",
        "Rows_Retained": len(df_cust),
        "Rows_Dropped": len(df_raw) - len(df_cust),
        "Unique_Customers": df_cust["CustomerID"].nunique(),
        "Description": "Filtered records missing CustomerID where behavioral linkage is impossible."
    })
    
    # Cast CustomerID to standard integer format
    df_cust["CustomerID"] = df_cust["CustomerID"].astype(int)
    
    # 2. Exact Duplicates Removal
    df_dedup = df_cust.drop_duplicates()
    audit_records.append({
        "Stage": "2. Duplicate Deduplication",
        "Rows_Retained": len(df_dedup),
        "Rows_Dropped": len(df_cust) - len(df_dedup),
        "Unique_Customers": df_dedup["CustomerID"].nunique(),
        "Description": "Dropped exact duplicate transactions across all matching fields."
    })
    
    # 3. Non-Merchandise StockCode Removal
    non_merch_codes = {
        "POST", "D", "M", "PADS", "DOT", "CRUK", "BANK CHARGES", 
        "AMAZONFEE", "S", "B", "gift_0001_10", "gift_0001_20", 
        "gift_0001_30", "gift_0001_40", "gift_0001_50"
    }
    clean_stock = ~df_dedup["StockCode"].astype(str).str.strip().isin(non_merch_codes)
    df_merch = df_dedup[clean_stock].copy()
    audit_records.append({
        "Stage": "3. Non-Merchandise Removal",
        "Rows_Retained": len(df_merch),
        "Rows_Dropped": len(df_dedup) - len(df_merch),
        "Unique_Customers": df_merch["CustomerID"].nunique(),
        "Description": "Removed postage, discount adjustments, debt write-offs, and manual codes."
    })
    
    # 4. Returns Isolation (Preserved separately for downstream operational analysis)
    is_return = (
        df_merch["InvoiceNo"].astype(str).str.startswith("C") | 
        (df_merch["Quantity"] <= 0) | 
        (df_merch["UnitPrice"] < 0)
    )
    returns_df = df_merch[is_return].copy()
    
    # Positive Sales Cohort
    df_sales = df_merch[~is_return].copy()
    
    # Filter valid positive price and positive quantity records
    valid_sales = (df_sales["Quantity"] > 0) & (df_sales["UnitPrice"] > 0)
    df_clean = df_sales[valid_sales].copy()
    df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["UnitPrice"]
    
    audit_records.append({
        "Stage": "4. Valid Positive Sales Cohort",
        "Rows_Retained": len(df_clean),
        "Rows_Dropped": len(df_merch) - len(df_clean),
        "Unique_Customers": df_clean["CustomerID"].nunique(),
        "Description": "Filtered out returns, zero/negative quantities, and unpriced promotions."
    })
    
    audit_df = pd.DataFrame(audit_records)
    
    if return_audit:
        return df_clean, audit_df, returns_df
    return df_clean

def compute_file_sha256(filepath):
  sha256_hash = hashlib.sha256()
  with open(filepath, 'rb') as f:
    for byte_block in iter(lambda: f.read(4096), b''):
      sha256_hash.update(byte_block)
  return sha256_hash.hexdigest()