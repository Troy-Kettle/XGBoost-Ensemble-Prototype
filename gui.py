import tkinter as tk
from tkinter import messagebox

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
    
    # Predict using the Voting Classifier
    combined_prediction = voting_clf.predict_proba(new_data_scaled)[0, 1]
    
    # Return the risk score
    return combined_prediction

# Function to handle the prediction when the button is clicked
def on_predict_button_click():
    try:
        # Get input values from the user
        heart_rate = float(entry_heart_rate.get())
        systolic_bp = float(entry_systolic_bp.get())
        resp_rate = float(entry_resp_rate.get())
        o2_sats = float(entry_o2_sats.get())
        temperature = float(entry_temperature.get())

        # Predict the risk score
        risk_score = predict_risk(heart_rate, systolic_bp, resp_rate, o2_sats, temperature)

        # Display the result in a message box
        messagebox.showinfo("Prediction", f"Predicted Risk Score: {risk_score:.2f}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")

# Create the main window
root = tk.Tk()
root.title("Risk Prediction")

# Create and place labels and entries for each input
label_heart_rate = tk.Label(root, text="Heart Rate:")
label_heart_rate.grid(row=0, column=0, padx=10, pady=5)
entry_heart_rate = tk.Entry(root)
entry_heart_rate.grid(row=0, column=1, padx=10, pady=5)

label_systolic_bp = tk.Label(root, text="Systolic BP:")
label_systolic_bp.grid(row=1, column=0, padx=10, pady=5)
entry_systolic_bp = tk.Entry(root)
entry_systolic_bp.grid(row=1, column=1, padx=10, pady=5)

label_resp_rate = tk.Label(root, text="Respiratory Rate:")
label_resp_rate.grid(row=2, column=0, padx=10, pady=5)
entry_resp_rate = tk.Entry(root)
entry_resp_rate.grid(row=2, column=1, padx=10, pady=5)

label_o2_sats = tk.Label(root, text="O2 Saturation:")
label_o2_sats.grid(row=3, column=0, padx=10, pady=5)
entry_o2_sats = tk.Entry(root)
entry_o2_sats.grid(row=3, column=1, padx=10, pady=5)

label_temperature = tk.Label(root, text="Temperature:")
label_temperature.grid(row=4, column=0, padx=10, pady=5)
entry_temperature = tk.Entry(root)
entry_temperature.grid(row=4, column=1, padx=10, pady=5)

# Create and place the predict button
predict_button = tk.Button(root, text="Predict Risk", command=on_predict_button_click)
predict_button.grid(row=5, columnspan=2, pady=20)

# Start the GUI event loop
root.mainloop()
