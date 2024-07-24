import pandas as pd
import numpy as np
import anfis
import membership.mf as mf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_excel('ANFIS.xlsx')

# Assuming the last column is the label
X = df.iloc[:, :-1].values  # Features: all columns except the last one
Y = df.iloc[:, -1].values   # Labels: the last column

# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Convert labels to a 2D array for ANFIS
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

#########


# Create membership functions for each input variable
num_features = X_train.shape[1]
mfs = []
for _ in range(num_features):
    mfs.append([mf.trimf([0, 0, 0.5]), mf.trimf([0, 0.5, 1]), mf.trimf([0.5, 1, 1])])

mfc = mf.MemFuncs(mfs)

########

# Initialize ANFIS model
anf = anfis.ANFIS(X_train, Y_train, mfc)

# Train the model
anf.trainHybridJangOffLine(epochs=20)


#######


# Initialize ANFIS model
anf = anfis.ANFIS(X_train, Y_train, mfc)

# Train the model
anf.trainHybridJangOffLine(epochs=20)



########

# Predict on the test set
predicted_labels = anfis.predict(X_test)

# Convert predictions to binary labels (0 or 1)
predicted_labels_binary = (predicted_labels >= 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(predicted_labels_binary == Y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
