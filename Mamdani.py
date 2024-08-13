import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_curve
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV

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

# Split the data into training and testing sets
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print("Data split. Training set size:", X_train.shape, "Testing set size:", X_test.shape)

# Handle class imbalance using SMOTE
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Resampled training set size: {X_train_resampled.shape}")

# Define XGBoost model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Define hyperparameter search space
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [0.1, 1.0]
}

# Perform Randomized Search for hyperparameter tuning (faster, then refine with GridSearch)
print("\nPerforming Randomized Search for XGBClassifier...")
random_search = RandomizedSearchCV(xgb, xgb_params, n_iter=50, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1, random_state=42)
random_search.fit(X_train_resampled, y_train_resampled)
print("Best parameters from Randomized Search:", random_search.best_params_)

# Use the best model from Randomized Search for further GridSearchCV tuning
best_xgb = random_search.best_estimator_

# Optionally, calibrate the classifier for better probability estimates
print("\nCalibrating the classifier...")
calibrated_xgb = CalibratedClassifierCV(best_xgb, method='isotonic', cv=5)
calibrated_xgb.fit(X_train_resampled, y_train_resampled)
print("Classifier calibrated.")

# Predict on the test set
print("\nMaking predictions on the test set...")
y_pred_proba = calibrated_xgb.predict_proba(X_test)[:, 1]
y_pred_binary = (y_pred_proba >= 0.5).astype(int)

# Calculate accuracy and AUC
accuracy = accuracy_score(y_test, y_pred_binary)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"ROC AUC: {roc_auc:.2f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Calibrated XGBoost (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid()
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['Class 0', 'Class 1'])

plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

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

# Save the model
print("\nSaving the model...")
joblib.dump(calibrated_xgb, 'calibrated_xgboost_classifier_model.pkl')
print("Model saved as 'calibrated_xgboost_classifier_model.pkl'")
