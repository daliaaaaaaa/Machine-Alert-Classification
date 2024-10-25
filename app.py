from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('lstm_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/detect', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.json
    input_data = pd.DataFrame(data['input'])

    # Ensure 'Timestamp' column is in datetime format
    input_data['Timestamp'] = pd.to_datetime(input_data['Timestamp'])

    # Convert KPI_Value to numpy array and normalize
    kpi_values = np.array(input_data['KPI_Value']).reshape(-1, 1)
    normalized_input = scaler.transform(kpi_values)

    # Create time series sequences for the LSTM model
    time_step = 10  # Same time step used in training
    X = []
    for i in range(len(normalized_input) - time_step):
        X.append(normalized_input[i:(i + time_step), 0])

    # Convert to numpy array and reshape for LSTM input
    X = np.array(X).reshape(len(X), time_step, 1)

    # Make predictions using the loaded model
    predictions = model.predict(X)

    # Inverse transform the predictions to original scale
    predictions = scaler.inverse_transform(predictions)

    # Convert predictions to list for JSON response
    return jsonify(predictions.flatten().tolist())


@app.route('/')

if __name__ == '__main__':
    app.run(debug=True)
