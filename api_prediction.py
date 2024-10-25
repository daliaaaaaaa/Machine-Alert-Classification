from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

# Charger le modèle pré-entraîné
reg = xgb.XGBRegressor()
reg.load_model("predict_KPI_model.json")  # Remplacez par le chemin de votre modèle

# Définir les fonctionnalités utilisées par le modèle
FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'lag_1', 'lag_2', 'lag_3']

app = Flask(__name__)

def create_features(df):
    """Générer les caractéristiques temporelles pour les prévisions."""
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df

def create_lag_features(df, lags=3):
    """Créer des caractéristiques de décalage pour les séries temporelles."""
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['KPI_Value'].shift(lag)
    return df

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Récupérer les paramètres de la requête
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        frequency = request.args.get('frequency', 'W')  # par défaut chaque semaine

        # Générer la série temporelle future
        future = pd.date_range(start=start_date, end=end_date, freq=frequency)
        future_df = pd.DataFrame(index=future)
        future_df['isFuture'] = True

        # Charger les données historiques et ajouter les futures dates avec les caractéristiques nécessaires
        df = pd.read_csv("dataset/train_set_rec.csv", nrows=15000, parse_dates=['Timestamp'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.set_index('Timestamp')
        df['isFuture'] = False
        df_and_future = pd.concat([df, future_df])
        df_and_future = create_features(df_and_future)
        df_and_future = create_lag_features(df_and_future, lags=3)

        # Préparer les données pour la prédiction
        future_features = df_and_future[df_and_future['isFuture']][FEATURES]
        future_features = future_features.fillna(method='bfill').fillna(method='ffill')  # Traiter les NaN

        # Prédictions
        future_preds = reg.predict(future_features)

        # Préparer les résultats dans un format de tableau
        predictions = [{"date": str(date), "prediction": float(pred)} for date, pred in zip(future.index, future_preds)]
        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
