import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the saved model and scaler
model_file = 'calibrated_xgboost_classifier_model.pkl'
scaler_file = 'scaler.pkl'

calibrated_xgb = joblib.load(model_file)
scaler = joblib.load(scaler_file)

# Imputer setup, ensure it's consistent with training
imputer = SimpleImputer(strategy='mean')
imputer.fit_transform([[1, 2, 3, 4, 5]])  # Dummy fit to use it for transformation

# Function to predict risk for new data
def predict_risk(heart_rate, systolic_bp, resp_rate, o2_sats, temperature):
    print("\nPredicting risk for new data...")

    # Create a DataFrame for new data
    new_data = pd.DataFrame({
        'HEART_RATE': [heart_rate],
        'SYSTOLIC_BP': [systolic_bp],
        'RESP_RATE': [resp_rate],
        'O2_SATS': [o2_sats],
        'TEMPERATURE': [temperature]
    })

    # Impute missing values (if any)
    new_data_imputed = imputer.transform(new_data)

    # Scale the data using the loaded scaler
    new_data_scaled = scaler.transform(new_data_imputed)

    # Predict the risk score
    risk_score = calibrated_xgb.predict_proba(new_data_scaled)[:, 1][0]
    print(f"Predicted Risk Score: {risk_score:.2f}")
    return risk_score

# Example usage
heart_rate = 149
systolic_bp = 123
resp_rate = 25
o2_sats = 110
temperature = 36.6

risk_score = predict_risk(heart_rate, systolic_bp, resp_rate, o2_sats, temperature)
print(f"Predicted Risk Score: {risk_score:.2f}")
