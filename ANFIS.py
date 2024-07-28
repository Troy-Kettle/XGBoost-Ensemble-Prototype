import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_excel('ANFIS.xlsx')

# Assuming the last column is the label
X = df.iloc[:, :-1].values  # Features: all columns except the last one
Y = df.iloc[:, -1].values   # Labels: the last column

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Define fuzzy membership functions for inputs
num_features = X_train.shape[1]
input_vars = []
for i in range(num_features):
    var = ctrl.Antecedent(np.arange(0, 1.01, 0.01), f'input_{i}')
    var['low'] = fuzz.trimf(var.universe, [0, 0, 0.5])
    var['medium'] = fuzz.trimf(var.universe, [0, 0.5, 1])
    var['high'] = fuzz.trimf(var.universe, [0.5, 1, 1])
    input_vars.append(var)

# Define fuzzy membership function for output
output_var = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'output')
output_var['very_low'] = fuzz.trimf(output_var.universe, [0, 0, 0.2])
output_var['low'] = fuzz.trimf(output_var.universe, [0.1, 0.3, 0.5])
output_var['medium'] = fuzz.trimf(output_var.universe, [0.4, 0.5, 0.6])
output_var['high'] = fuzz.trimf(output_var.universe, [0.5, 0.7, 0.9])
output_var['very_high'] = fuzz.trimf(output_var.universe, [0.8, 1, 1])

# Define fuzzy rules
rules = []
for i in range(num_features):
    rules.append(ctrl.Rule(input_vars[i]['low'], output_var['very_low']))
    rules.append(ctrl.Rule(input_vars[i]['medium'], output_var['medium']))
    rules.append(ctrl.Rule(input_vars[i]['high'], output_var['very_high']))

# Create control system and simulation
control_system = ctrl.ControlSystem(rules)
simulation = ctrl.ControlSystemSimulation(control_system)

# Train a machine learning model for adaptive learning
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)

# Function to categorize risk
def categorize_risk(probability):
    if probability < 0.2:
        return 'Very Low'
    elif probability < 0.4:
        return 'Low'
    elif probability < 0.6:
        return 'Medium'
    elif probability < 0.8:
        return 'High'
    else:
        return 'Very High'

# Prompt user for input values
input_values = []
features = ['HEART_RATE', 'SYSTOLIC_BP', 'RESP_RATE', 'O2_SATS', 'TEMPERATURE']
for feature in features:
    value = float(input(f"Enter {feature}: "))
    input_values.append(value)

# Scale the input values
input_values_scaled = scaler.transform([input_values])

# Predict using the Random Forest model
predicted_prob = rf.predict_proba(input_values_scaled)[:, 1][0]

# Apply fuzzy logic rules to refine the prediction
for i in range(num_features):
    simulation.input[f'input_{i}'] = input_values_scaled[0, i]
simulation.compute()
fuzzy_output = simulation.output['output']

# Combine predictions (simple average)
combined_prediction = (predicted_prob + fuzzy_output) / 2

# Categorize the risk
risk_category = categorize_risk(combined_prediction)

# Show the results
print(f"Predicted probability of event: {combined_prediction * 100:.2f}%")
print(f"Risk Category: {risk_category}")
