import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier

# Load the saved model and preprocessing components
print("Loading the saved model and preprocessing components...")
model = joblib.load('best_xgboost_classifier_model.pkl')
imputer = joblib.load('simple_imputer.pkl')
scaler = joblib.load('minmax_scaler.pkl')
poly = joblib.load('polynomial_features.pkl')
rfe_selector = joblib.load('rfe_selector.pkl')
print("Model and preprocessing components loaded.")

# Function to predict risk for a new set of feature values
def predict_risk(heart_rate, systolic_bp, resp_rate, o2_sats, temperature):
    print("\nPredicting risk for new data...")
    new_data = pd.DataFrame({
        'HEART_RATE': [heart_rate],
        'SYSTOLIC_BP': [systolic_bp],
        'RESP_RATE': [resp_rate],
        'O2_SATS': [o2_sats],
        'TEMPERATURE': [temperature]
    })

    # Handle missing values using SimpleImputer
    new_data_imputed = imputer.transform(new_data)
    
    # Feature engineering using PolynomialFeatures
    new_data_poly = poly.transform(new_data_imputed)
    
    # Combine original and polynomial features
    new_data_combined = np.hstack((new_data_imputed, new_data_poly))
    
    # Normalize the data
    new_data_scaled = scaler.transform(new_data_combined)
    
    # Select features using RFECV
    new_data_selected = rfe_selector.transform(new_data_scaled)
    
    # Predict risk
    risk_score = model.predict_proba(new_data_selected)[:, 1][0]
    print(f"Predicted Risk Score: {risk_score:.4f}")
    return risk_score

# Example usage
if __name__ == "__main__":
    heart_rate = 100
    systolic_bp = 110
    resp_rate = 25
    o2_sats = 98
    temperature = 37.6

    risk_score = predict_risk(heart_rate, systolic_bp, resp_rate, o2_sats, temperature)
