import os
import sys
import pickle
import numpy as np
import pandas as pd

# Add the parent directory of this file to system path to allow local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# Import database and preprocessing modules
from database import load_integrated_audit_data
from preprocess import prepare_ml_splits, get_regression_pipeline, get_classification_pipeline, engineer_audit_features

def train_and_save_models():
    """
    Extracts data, splits it, trains regression & classification models, 
    evaluates them, and saves the final production-ready pipelines.
    """
    # 1. Fetch B2B integrated transaction records from the real SQLite database
    print("Attempting to connect to real SQLite database 'data/inventory.db'...")
    df = load_integrated_audit_data()
        
    # Ensure models/ and data/ directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Core features list
    numeric_features = ['InvoicedQuantity', 'InvoicedDollars', 'UniqueBrandsOrdered', 'days_to_invoice']
    categorical_features = ['VendorName']
    
    # ----------------------------------------------------
    # PHASE A: Train Price & Freight Regressor (Module 1)
    # ----------------------------------------------------
    print("\n--- Phase A: Training B2B Cost & Freight Regressor ---")
    X_train_r, X_test_r, y_train_r, y_test_r = prepare_ml_splits(df, task_type='regression')
    
    # Instantiate Pipeline (Preprocessor + Regressor)
    reg_preprocessor = get_regression_pipeline(numeric_features, categorical_features)
    reg_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    
    reg_pipeline = Pipeline(steps=[
        ('preprocessor', reg_preprocessor),
        ('model', reg_model)
    ])
    
    print("Fitting Random Forest Regressor on training splits...")
    reg_pipeline.fit(X_train_r, y_train_r)
    
    # Evaluate Regressor
    y_pred_r = reg_pipeline.predict(X_test_r)
    mae = mean_absolute_error(y_test_r, y_pred_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    r2 = r2_score(y_test_r, y_pred_r)
    
    print(f"Regression Evaluation Results:")
    print(f"  * Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"  * Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"  * R² Coefficient of Determination: {r2:.4f}")
    
    # Save serialized Regressor
    reg_path = "models/price_regressor.pkl"
    with open(reg_path, 'wb') as f:
        pickle.dump(reg_pipeline, f)
    print(f"Successfully saved trained B2B Regressor to '{reg_path}'")
    
    # ----------------------------------------------------
    # PHASE B: Train Audit Risk Classifier (Module 2)
    # ----------------------------------------------------
    print("\n--- Phase B: Training B2B Audit Risk Classifier ---")
    X_train_c, X_test_c, y_train_c, y_test_c = prepare_ml_splits(df, task_type='classification')
    
    # Instantiate Pipeline (Preprocessor + Classifier)
    clf_preprocessor = get_classification_pipeline(numeric_features, categorical_features)
    # Since credit/billing anomalies can be imbalanced, use class_weight
    clf_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    
    clf_pipeline = Pipeline(steps=[
        ('preprocessor', clf_preprocessor),
        ('model', clf_model)
    ])
    
    print("Fitting Random Forest Classifier on training splits...")
    clf_pipeline.fit(X_train_c, y_train_c)
    
    # Evaluate Classifier
    y_pred_c = clf_pipeline.predict(X_test_c)
    f1 = f1_score(y_test_c, y_pred_c)
    precision = precision_score(y_test_c, y_pred_c)
    recall = recall_score(y_test_c, y_pred_c)
    
    print(f"Classification Evaluation Results:")
    print(f"  * F1-Score: {f1:.4f}")
    print(f"  * Precision (Accuracy of Risk Alarm): {precision:.4f}")
    print(f"  * Recall (Percentage of Anomalies Detected): {recall:.4f}")
    print("\nFull Classification Report:")
    print(classification_report(y_test_c, y_pred_c))
    
    # Save serialized Classifier
    clf_path = "models/audit_classifier.pkl"
    with open(clf_path, 'wb') as f:
        pickle.dump(clf_pipeline, f)
    print(f"Successfully saved trained B2B Classifier to '{clf_path}'")
    
    # Save a copy of engineered training data for quick Streamlit display
    df_engineered = engineer_audit_features(df)
    df_engineered.to_csv("data/audited_transactions_cache.csv", index=False)
    print("Saved transaction records cache to 'data/audited_transactions_cache.csv'")

if __name__ == "__main__":
    train_and_save_models()
