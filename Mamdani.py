import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_excel('ANFIS.xlsx')

# Display the first few rows to check the format
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Handle missing values using SimpleImputer
features = ['HEART_RATE', 'SYSTOLIC_BP', 'RESP_RATE', 'O2_SATS', 'TEMPERATURE']
target = '4_HOURS_FROM_ANNOTATED_EVENT'

# Ensure that only relevant features are used
df_imputed = df[features + [target]].copy()
imputer = SimpleImputer(strategy='mean')
df_imputed[features] = imputer.fit_transform(df_imputed[features])

# Separate features and labels
X = df_imputed[features].values
Y = df_imputed[target].values

# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)

# Predict on the test set using the trained Random Forest model
rf_predicted_probs = rf.predict_proba(X_test)[:, 1]

# Define a simple MLP classifier for the adaptive fuzzy-like decision making
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
mlp.fit(X_train, Y_train)

# Predict on the test set using the MLP classifier
mlp_predicted_probs = mlp.predict_proba(X_test)[:, 1]

# Combine Random Forest and MLP predictions
fuzzy_predictions = []
for i in range(len(X_test)):
    rf_score = rf_predicted_probs[i]
    mlp_score = mlp_predicted_probs[i]
    combined_prediction = (rf_score + mlp_score) / 2  # Simple average of RF and MLP output
    fuzzy_predictions.append(combined_prediction)

# Convert combined predictions to binary labels (0 or 1)
predicted_labels_binary = np.array(fuzzy_predictions) >= 0.5

# Calculate accuracy
accuracy = accuracy_score(Y_test, predicted_labels_binary)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Feature Importance Plot for Random Forest
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(len(features)), importances[indices], align="center")
plt.xticks(range(len(features)), np.array(features)[indices], rotation=90)
plt.xlim([-1, len(features)])
plt.tight_layout()
plt.show()

# ROC Curve for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(Y_test, rf_predicted_probs)
roc_auc_rf = roc_auc_score(Y_test, rf_predicted_probs)

plt.figure(figsize=(10, 6))
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# ROC Curve for MLP
fpr_mlp, tpr_mlp, _ = roc_curve(Y_test, mlp_predicted_probs)
roc_auc_mlp = roc_auc_score(Y_test, mlp_predicted_probs)

plt.figure(figsize=(10, 6))
plt.plot(fpr_mlp, tpr_mlp, color='green', lw=2, label=f'MLP (AUC = {roc_auc_mlp:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, predicted_labels_binary)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['Class 0', 'Class 1'])

plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Distribution of Predicted Risk Scores
plt.figure(figsize=(10, 6))
plt.hist(fuzzy_predictions, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Risk Score')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Risk Scores')
plt.grid(True)
plt.show()

# Function to predict risk for a new set of feature values
def predict_risk(heart_rate, systolic_bp, resp_rate, o2_sats, temperature):
    # Create a DataFrame for the new input with the same feature names as the original data
    new_data = pd.DataFrame({
        'HEART_RATE': [heart_rate],
        'SYSTOLIC_BP': [systolic_bp],
        'RESP_RATE': [resp_rate],
        'O2_SATS': [o2_sats],
        'TEMPERATURE': [temperature]
    })

    # Ensure the new data has the same columns as the training data
    new_data = new_data[features]
    
    # Impute missing values in the input (if any)
    new_data_imputed = imputer.transform(new_data)
    
    # Normalize the input features
    new_data_scaled = scaler.transform(new_data_imputed)
    
    # Predict using Random Forest
    rf_score = rf.predict_proba(new_data_scaled)[0, 1]
    
    # Predict using MLP
    mlp_score = mlp.predict_proba(new_data_scaled)[0, 1]
    
    # Combine the scores
    combined_prediction = (rf_score + mlp_score) / 2  # Simple average of RF and MLP output
    
    # Return the risk score
    return combined_prediction

# Example usage
# Replace these with actual values for heart_rate, systolic_bp, resp_rate, o2_sats, and temperature
heart_rate = 98
systolic_bp = 123
resp_rate = 19
o2_sats = 97
temperature = 36.6

risk_score = predict_risk(heart_rate, systolic_bp, resp_rate, o2_sats, temperature)
print(f"Predicted Risk Score: {risk_score:.2f}")
