import pandas as pd
import numpy as np

def create_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transaction data by CustomerID to build Recency, Frequency, and Monetary features.
    """
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # Log transformation for right-skewed features
    rfm['Recency_log'] = np.log1p(rfm['Recency'])
    rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
    rfm['Monetary_log'] = np.log1p(rfm['Monetary'])
    
    return rfm
