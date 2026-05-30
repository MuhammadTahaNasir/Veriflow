import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def engineer_audit_features(df):
    """
    Transforms the integrated dataset into machine learning features
    specifically designed for B2B financial auditing.
    """
    df = df.copy()
    
    # 1. Handle dates
    date_cols = ['PODate', 'ReceivingDate', 'InvoiceDate', 'PayDate']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    # Calculate days elapsed
    if 'PODate' in df.columns and 'InvoiceDate' in df.columns:
        df['days_to_invoice'] = (df['InvoiceDate'] - df['PODate']).dt.days.fillna(0)
    else:
        df['days_to_invoice'] = 0
        
    if 'InvoiceDate' in df.columns and 'PayDate' in df.columns:
        df['days_to_pay'] = (df['PayDate'] - df['InvoiceDate']).dt.days.fillna(0)
    else:
        df['days_to_pay'] = 0

    # 2. Financial Discrepancies
    # Price Variance (Overcharge Amount)
    df['price_variance'] = df['InvoicedDollars'] - df['OrderedDollars']
    # Quantity Discrepancy (Ordered vs Invoiced items)
    df['quantity_discrepancy'] = df['InvoicedQuantity'] - df['OrderedQuantity']
    # Freight Ratios
    df['freight_per_unit'] = df['InvoicedFreight'] / (df['InvoicedQuantity'] + 1e-5)
    df['freight_to_dollar_ratio'] = df['InvoicedFreight'] / (df['InvoicedDollars'] + 1e-5)
    
    # Fill any NaNs or infinities that may have resulted from division by zero
    df['freight_per_unit'] = df['freight_per_unit'].replace([np.inf, -np.inf], 0).fillna(0)
    df['freight_to_dollar_ratio'] = df['freight_to_dollar_ratio'].replace([np.inf, -np.inf], 0).fillna(0)

    # 3. Formulate the Target for Classification (AP Audit Risk Flag)
    # If the raw data has 'ApprovalStatus' or similar, we can map it.
    # Otherwise, or in addition, we define an invoice as high-risk if:
    # - Price variance is positive (overcharge)
    # - Billed quantity exceeds ordered quantity
    # - Freight is abnormally high (ratio > 10% of dollars AND freight > $10)
    # - Or if ApprovalStatus explicitly marks it as rejected/pending review
    
    anomaly_conditions = (
        (df['price_variance'] > 5.0) |  # Invoiced more than PO by at least $5
        (df['quantity_discrepancy'] > 0) | # Billed more units than ordered
        (df['freight_to_dollar_ratio'] > 0.008)  # High freight charges (outliers > 1.5x standard contract)
    )
    
    # Check if raw ApprovalStatus has explicit labels
    if 'ApprovalStatus' in df.columns:
        # Check if there are any negative indicators in the text
        explicit_reject = df['ApprovalStatus'].astype(str).str.lower().str.contains('reject|manual|hold|pending')
        df['AuditRequired'] = np.where(anomaly_conditions | explicit_reject, 1, 0)
    else:
        df['AuditRequired'] = np.where(anomaly_conditions, 1, 0)
        
    return df

def get_regression_pipeline(numeric_features, categorical_features):
    """
    Creates a scikit-learn ColumnTransformer pipeline for regression.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    return preprocessor

def get_classification_pipeline(numeric_features, categorical_features):
    """
    Creates a scikit-learn ColumnTransformer pipeline for classification.
    """
    # Identical structure, but keeping modular for specific scaling or encoding needs
    return get_regression_pipeline(numeric_features, categorical_features)

def prepare_ml_splits(df, task_type='regression', test_size=0.2, random_state=42):
    """
    Performs feature selection and train-test splits for the specified task.
    """
    df_engineered = engineer_audit_features(df)
    
    # Base features for both models
    # Predictors: InvoicedQuantity, InvoicedDollars, UniqueBrandsOrdered, days_to_invoice, VendorName
    X = df_engineered[['InvoicedQuantity', 'InvoicedDollars', 'UniqueBrandsOrdered', 'days_to_invoice', 'VendorName']].copy()
    
    # Convert VendorName to string to avoid categorization issues
    X['VendorName'] = X['VendorName'].astype(str)
    
    if task_type == 'regression':
        y = df_engineered['InvoicedFreight'].copy()
    elif task_type == 'classification':
        y = df_engineered['AuditRequired'].copy()
    else:
        raise ValueError(f"Invalid task_type: {task_type}")
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Test Scaffolding
    print("Testing preprocessing pipelines scaffolding: OK")
