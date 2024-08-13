import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_curve
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from bayes_opt import BayesianOptimization
import shap

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
imputer = SimpleImputer(strategy='median')
df[features] = imputer.fit_transform(df[features])
print("Missing values after imputation:\n", df[features].isnull().sum())

# Feature engineering
print("\nPerforming feature engineering...")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[features])
feature_names = poly.get_feature_names(features)
df_poly = pd.DataFrame(X_poly, columns=feature_names)

# Combine original and polynomial features
X = pd.concat([df[features], df_poly], axis=1)
y = df[target].values

# Normalize the data
print("\nNormalizing data...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using ADASYN
print("\nApplying ADASYN...")
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_scaled, y)
print(f"Resampled dataset size: {X_resampled.shape}")

# Feature selection using RFECV
print("\nPerforming feature selection...")
base_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
rfe_selector = RFECV(estimator=base_model, step=1, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1)
X_selected = rfe_selector.fit_transform(X_resampled, y_resampled)
print(f"Selected {X_selected.shape[1]} features")

# Define models for stacking
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
lgbm = lgb.LGBMClassifier(random_state=42)
catboost = CatBoostClassifier(verbose=0, random_state=42)

# Define stacking ensemble
stacking_model = StackingClassifier(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('catboost', catboost)],
    final_estimator=LogisticRegression(random_state=42),
    cv=5
)

# Bayesian optimization for XGBoost hyperparameters
def xgb_evaluate(max_depth, learning_rate, n_estimators, gamma, min_child_weight, subsample, colsample_bytree):
    params = {
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),
        'gamma': gamma,
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree
    }
    
    cv_result = cross_val_score(XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', random_state=42),
                                X_selected, y_resampled, cv=StratifiedKFold(5), scoring='roc_auc')
    return cv_result.mean()

# Define the search space
pbounds = {
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.3),
    'n_estimators': (100, 1000),
    'gamma': (0, 1),
    'min_child_weight': (1, 10),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1)
}

# Perform Bayesian optimization
print("\nPerforming Bayesian optimization for XGBoost...")
optimizer = BayesianOptimization(f=xgb_evaluate, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=5, n_iter=25)

# Get the best parameters and convert n_estimators and max_depth to int
best_params = optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
print("Best parameters:", best_params)

# Train the final model using the best parameters
print("\nTraining the final model...")
best_xgb = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
best_xgb.fit(X_selected, y_resampled)

# Cross-validation
print("\nPerforming cross-validation...")
cv_scores = cross_val_score(best_xgb, X_selected, y_resampled, cv=StratifiedKFold(5), scoring='roc_auc')
print(f"Cross-validation ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# SHAP values for model explainability
print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_selected)

# Plot SHAP summary
shap.summary_plot(shap_values, X_selected, plot_type="bar")

# Save preprocessing components
print("\nSaving preprocessing components...")
joblib.dump(imputer, 'simple_imputer.pkl')
joblib.dump(scaler, 'minmax_scaler.pkl')
joblib.dump(poly, 'polynomial_features.pkl')
joblib.dump(rfe_selector, 'rfe_selector.pkl')
print("Preprocessing components saved.")

# Save the model
print("\nSaving the model...")
joblib.dump(best_xgb, 'best_xgboost_classifier_model.pkl')
print("Model saved as 'best_xgboost_classifier_model.pkl'")
