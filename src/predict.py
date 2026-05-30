import os
import sys
import pickle
import pandas as pd

# Add parent directory of this file to system path to allow local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def predict_freight(quantity, dollars, brands, delay, vendor_name):
    """
    Inference function for predicting fair-value freight cost.
    """
    model_path = os.path.join("models", "price_regressor.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run training first.")
        
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
        
    input_df = pd.DataFrame([{
        'InvoicedQuantity': quantity,
        'InvoicedDollars': dollars,
        'UniqueBrandsOrdered': brands,
        'days_to_invoice': delay,
        'VendorName': vendor_name
    }])
    
    return float(pipeline.predict(input_df)[0])

def predict_risk(quantity, dollars, brands, delay, vendor_name):
    """
    Inference function for predicting AP audit risk.
    """
    model_path = os.path.join("models", "audit_classifier.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run training first.")
        
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
        
    input_df = pd.DataFrame([{
        'InvoicedQuantity': quantity,
        'InvoicedDollars': dollars,
        'UniqueBrandsOrdered': brands,
        'days_to_invoice': delay,
        'VendorName': vendor_name
    }])
    
    pred = int(pipeline.predict(input_df)[0])
    prob = float(pipeline.predict_proba(input_df)[0][1])
    
    return pred, prob

if __name__ == "__main__":
    # Test inference
    try:
        f = predict_freight(1200, 8500, 4, 5, "A&G Logistics")
        r, p = predict_risk(1200, 8500, 4, 5, "A&G Logistics")
        print("Inference Check Successful!")
        print(f"  * Expected Freight: ${f:.2f}")
        print(f"  * Risk Flag: {r} (Confidence: {p*100:.1f}%)")
    except Exception as e:
        print(f"Local inference check: {e}")
