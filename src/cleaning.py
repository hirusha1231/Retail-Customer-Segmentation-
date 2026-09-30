import pandas as pd
import numpy as np

def audit_and_clean_data(raw_filepath: str) -> pd.DataFrame:
    """
    Loads raw Online Retail data, applies cleaning rules, and returns a positive sales view.
    """
    df = pd.read_excel(raw_filepath) if raw_filepath.endswith('.xlsx') else pd.read_csv(raw_filepath, encoding='ISO-8859-1')
    
    # Exclude records without CustomerID
    df_cleaned = df.dropna(subset=['CustomerID']).copy()
    df_cleaned['CustomerID'] = df_cleaned['CustomerID'].astype(int)
    
    # Separate cancelled transactions and keep positive quantity & price
    df_cleaned = df_cleaned[~df_cleaned['InvoiceNo'].astype(str).str.startswith('C')]
    df_cleaned = df_cleaned[(df_cleaned['Quantity'] > 0) & (df_cleaned['UnitPrice'] > 0)]
    
    # Compute total transaction value
    df_cleaned['TotalPrice'] = df_cleaned['Quantity'] * df_cleaned['UnitPrice']
    
    return df_cleaned
