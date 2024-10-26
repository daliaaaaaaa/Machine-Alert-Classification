from flask import Flask, request, jsonify
import pickle
import joblib
import pandas as pd
import numpy as np

# Initialiser l'application Flask
app = Flask(__name__)

# Charger le modèle et les transformateurs nécessaires
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

label_encoder = joblib.load('label_encoder.pkl')
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')  # Assuming scaler.pkl is your scaler file
num_features = 130  # Assurez-vous de correspondre au nombre exact de caractéristiques

# Route pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données d'entrée JSON
    data = request.get_json()
    kpi_value = data.get('KPI_Value')
    timestamp = data.get('Timestamp')
    kpi_name = data.get('KPI_Name')

    # Créer un DataFrame avec les données reçues
    real_time_data = pd.DataFrame({
        'Timestamp': [timestamp],
        'KPI_Value': [kpi_value],
        'KPI_Name': [kpi_name]
    })

    # Prétraitement des données
    real_time_data['Timestamp'] = pd.to_datetime(real_time_data['Timestamp'])
    real_time_data.set_index('Timestamp', inplace=True)
    real_time_data['KPI_Value'] = scaler.transform(real_time_data[['KPI_Value']])
    real_time_data['KPI_Name'] = label_encoder.transform(real_time_data['KPI_Name'])

    # Ajouter des décalages (lags)
    for lag in [1, 3, 5, 12]:
        real_time_data[f'lag_{lag}'] = real_time_data['KPI_Value'].shift(lag)
    
    # Ajouter des statistiques de rolling
    real_time_data['rolling_mean_3'] = real_time_data['KPI_Value'].rolling(window=3).mean()
    real_time_data['rolling_std_3'] = real_time_data['KPI_Value'].rolling(window=3).std()
    real_time_data['KPI_diff'] = real_time_data['KPI_Value'].diff().fillna(0)
    real_time_data.fillna(0, inplace=True)

    # Prédire le cluster KPI
    KPI_Cluster = kmeans.predict(real_time_data[['KPI_Value', 'KPI_diff', 'rolling_mean_3', 'rolling_std_3']].fillna(0))[0]

    # Préparation des données pour le modèle
    exog_real_time = pd.DataFrame(np.zeros((1, num_features)))
    for idx, col in enumerate(real_time_data.columns):
        if idx < num_features:
            exog_real_time.iloc[0, idx] = real_time_data[col].values[0]
    exog_real_time.index = [11982]

    # Prédiction
    y_pred = model.predict(steps=1, exog=exog_real_time)

    # Vérification de l'anomalie
    threshold = 1
    anomaly_status = 1 if abs(y_pred[0]) > threshold else 0

    # Retourner le résultat sous format JSON
    return jsonify({
        'Predicted_Value': y_pred[0],
        'Anomaly_Status': anomaly_status
    })

# Démarrer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
