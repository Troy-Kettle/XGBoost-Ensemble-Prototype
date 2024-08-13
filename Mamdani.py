import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib
from imblearn.over_sampling import SMOTE

# Load the dataset
print("Loading dataset...")
df = pd.read_excel('ANFIS.xlsx')
print("Dataset loaded. Shape:", df.shape)
print("First few rows:\n", df.head())

# Check for missing values
print("\nChecking for missing values...")
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values)

# Define features and target
features = ['HEART_RATE', 'SYSTOLIC_BP', 'RESP_RATE', 'O2_SATS', 'TEMPERATURE']
target = '4_HOURS_FROM_ANNOTATED_EVENT'

# Handle missing values using SimpleImputer
print("\nHandling missing values...")
imputer = SimpleImputer(strategy='mean')
df[features] = imputer.fit_transform(df[features])
print("Missing values after imputation:\n", df[features].isnull().sum())

# Separate features and target
X = df[features].values
y = df[target].values

# Normalize the data
print("\nNormalizing data...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Add polynomial features
print("\nAdding polynomial features...")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
print(f"Polynomial features added. New shape: {X_poly.shape}")

# Split the data into training and testing sets
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42, stratify=y)
print("Data split. Training set size:", X_train.shape, "Testing set size:", X_test.shape)

# Handle class imbalance using SMOTE
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Resampled training set size: {X_train_resampled.shape}")

# Define XGBoost model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Define hyperparameter grid for XGBoost
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Perform Grid Search for hyperparameter tuning
print("\nPerforming Grid Search for XGBClassifier...")
grid_xgb = GridSearchCV(xgb, xgb_params, cv=5, scoring='accuracy')
grid_xgb.fit(X_train_resampled, y_train_resampled)
print("Best XGBoost parameters:", grid_xgb.best_params_)

# Best model
best_xgb = grid_xgb.best_estimator_

# Train the best XGBoost model
print("\nTraining XGBoost Classifier...")
best_xgb.fit(X_train_resampled, y_train_resampled)
print("XGBoost Classifier trained.")

# Predict on the test set using the XGBoost model
print("\nMaking predictions...")
y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
y_pred_binary = (y_pred_proba >= 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy * 100:.2f}%")

# ROC Curve for XGBoost Classifier
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'XGBoost Classifier (AUC = {roc_auc:.2f})')
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
conf_matrix = confusion_matrix(y_test, y_pred_binary)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['Class 0', 'Class 1'])

plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))

# Distribution of Predicted Risk Scores
plt.figure(figsize=(10, 6))
plt.hist(y_pred_proba, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Risk Score')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Risk Scores')
plt.grid(True)
plt.show()

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

    new_data_imputed = imputer.transform(new_data)
    new_data_scaled = scaler.transform(new_data_imputed)
    new_data_poly = poly.transform(new_data_scaled)
    risk_score = best_xgb.predict_proba(new_data_poly)[0, 1]
    print(f"Predicted Risk Score: {risk_score:.2f}")
    return risk_score

# Example usage
heart_rate = 98
systolic_bp = 123
resp_rate = 19
o2_sats = 97
temperature = 36.6

risk_score = predict_risk(heart_rate, systolic_bp, resp_rate, o2_sats, temperature)
print(f"Predicted Risk Score: {risk_score:.2f}")

# Save the model
print("\nSaving the model...")
joblib.dump(best_xgb, 'xgboost_classifier_model.pkl')
print("Model saved as 'xgboost_classifier_model.pkl'")

# Load the model (for future use)
# loaded_model = joblib.load('xgboost_classifier_model.pkl')
# print("Model loaded successfully.")
