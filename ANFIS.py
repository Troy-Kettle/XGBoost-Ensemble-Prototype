import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

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

# Apply fuzzy c-means clustering to define the classes
n_clusters = 5  # Number of clusters/classes
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_scaled.T, n_clusters, 2, error=0.005, maxiter=1000, init=None)

# Determine the cluster membership for each patient
cluster_membership = np.argmax(u, axis=0)

# Add the cluster membership to the dataframe for analysis
df['Cluster'] = cluster_membership

# Distribution of patients per class
class_distribution = df['Cluster'].value_counts().sort_index()
print("Distribution of patients per class:")
print(class_distribution)

# Visualize the class distribution
plt.bar(class_distribution.index, class_distribution.values)
plt.xlabel('Cluster')
plt.ylabel('Number of Patients')
plt.title('Distribution of Patients per Cluster')
plt.show()

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, cluster_membership, test_size=0.2, random_state=42)

# Train a machine learning model for adaptive learning
rf = RandomForestClassifier(n_estimators=1000, random_state=42)
rf.fit(X_train, Y_train)

# Function to categorize risk based on cluster
def categorize_risk(cluster):
    risk_categories = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    return risk_categories[cluster]

# Prompt user for input values
input_values = []
features = ['HEART_RATE', 'SYSTOLIC_BP', 'RESP_RATE', 'O2_SATS', 'TEMPERATURE']
for feature in features:
    value = float(input(f"Enter {feature}: "))
    input_values.append(value)

# Scale the input values
input_values_scaled = scaler.transform([input_values])

# Predict using the Random Forest model
predicted_cluster = rf.predict(input_values_scaled)[0]

# Categorize the risk based on the cluster
risk_category = categorize_risk(predicted_cluster)

# Show the results
print(f"Predicted Cluster: {predicted_cluster}")
print(f"Risk Category: {risk_category}")
