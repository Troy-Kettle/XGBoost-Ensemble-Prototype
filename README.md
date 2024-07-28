# ANFIS-Diagnosis-System

# Medical Event Risk Prediction

This project combines Machine Learning and Fuzzy Logic to predict the risk of a medical event occurring within 4 hours based on patient physiological measurements. It uses a Random Forest classifier to predict the probability of an event, which is then refined using fuzzy logic rules.

## Features

- **HEART_RATE**: The heart rate of the patient.
- **SYSTOLIC_BP**: The systolic blood pressure of the patient.
- **RESP_RATE**: The respiratory rate of the patient.
- **O2_SATS**: The oxygen saturation level of the patient.
- **TEMPERATURE**: The body temperature of the patient.

## Categories of Risk

The predicted risk of an event is categorized into five levels:

- **Very Low**
- **Low**
- **Medium**
- **High**
- **Very High**

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/medical-event-risk-prediction.git
   cd medical-event-risk-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure your dataset is named `ANFIS.xlsx` and placed in the project directory. The dataset should have the following columns:
   - `HEART_RATE`
   - `SYSTOLIC_BP`
   - `RESP_RATE`
   - `O2_SATS`
   - `TEMPERATURE`
   - `4_HOURS_FROM_ANNOTATED_EVENT` (the label column indicating whether an event occurs within 4 hours)

2. Run the prediction script:
   ```bash
   python predict_risk.py
   ```

3. Enter the required physiological measurements when prompted:
   ```
   Enter HEART_RATE: 
   Enter SYSTOLIC_BP: 
   Enter RESP_RATE: 
   Enter O2_SATS: 
   Enter TEMPERATURE: 
   ```

4. View the predicted probability and risk category:
   ```
   Predicted probability of event: xx.xx%
   Risk Category: [Very Low | Low | Medium | High | Very High]
   ```

## Example

```
$ python predict_risk.py
Enter HEART_RATE: 72
Enter SYSTOLIC_BP: 120
Enter RESP_RATE: 16
Enter O2_SATS: 98
Enter TEMPERATURE: 37.0
Predicted probability of event: 23.45%
Risk Category: Low
```

## Dependencies

- `pandas`
- `numpy`
- `scikit-fuzzy`
- `scikit-learn`
- `openpyxl`

You can install these dependencies using:
```bash
pip install pandas numpy scikit-fuzzy scikit-learn openpyxl
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
