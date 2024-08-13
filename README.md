# XGBoost Classifier with Advanced Feature Engineering and Hyperparameter Optimization
## Overview
This project involves building a classification model to predict outcomes based on various medical features. The model employs several techniques, including feature engineering, imputation, normalization, and advanced hyperparameter optimization using Bayesian Optimization. The final model is an XGBoost classifier trained with the best hyperparameters found through optimization.

## Project Structure
### Data Loading and Preprocessing
- Load the dataset from an Excel file.
- Handle missing values using SimpleImputer.
- Perform feature engineering with PolynomialFeatures.
- Normalize the features using MinMaxScaler.
- Address class imbalance using ADASYN.

### Feature Selection
- Select the most relevant features using RFECV with an XGBoost classifier.

### Model Building and Evaluation
- Train a stacking classifier combining XGBoost, LightGBM, and CatBoost.
- Optimize XGBoost hyperparameters using Bayesian Optimization.
- Evaluate the final model using cross-validation and ROC AUC score.
- Compute SHAP values for model explainability.

### Saving Components
- Save preprocessing components and the final model using joblib.

## Requirements
Ensure you have the following Python packages installed:
- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost
- lightgbm
- catboost
- imblearn
- bayesian-optimization
- shap
- joblib
- openpyxl (for reading Excel files)

You can install the required packages using pip:
```bash
pip install pandas numpy matplotlib scikit-learn xgboost lightgbm catboost imbalanced-learn bayesian-optimization shap joblib openpyxl
```

## Data
The dataset is expected to be in Excel format (ANFIS.xlsx) with the following columns:
- Features: HEART_RATE, SYSTOLIC_BP, RESP_RATE, O2_SATS, TEMPERATURE
- Target: 4_HOURS_FROM_ANNOTATED_EVENT

## Running the Script
To run the script, execute it in a Python environment:
```bash
python your_script_name.py
```
Replace `your_script_name.py` with the name of the script file.

## Output
### Preprocessing Components:
- simple_imputer.pkl: Imputer for handling missing values.
- minmax_scaler.pkl: Scaler for normalizing features.
- polynomial_features.pkl: Polynomial feature transformer.
- rfe_selector.pkl: RFECV feature selector.

### Model:
- best_xgboost_classifier_model.pkl: The trained XGBoost classifier with optimized hyperparameters.

### SHAP Summary Plot:
SHAP summary plot will be displayed, showing feature importance and the impact of each feature on model predictions.

## Bayesian Optimization
Hyperparameters optimized for XGBoost include:
- max_depth
- learning_rate
- n_estimators
- gamma
- min_child_weight
- subsample
- colsample_bytree

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- XGBoost, LightGBM, and CatBoost for their powerful implementations of gradient boosting.
- SHAP for model explainability.
- imblearn for addressing class imbalance.
- bayesian-optimization for efficient hyperparameter tuning
