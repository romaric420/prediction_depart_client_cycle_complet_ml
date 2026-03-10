# src/predict.py
# Chargement du modèle et prédiction sur de nouvelles données
# Utilisé par FastAPI pour exposer les prédictions via API REST

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from src.preprocessing import preprocess
from src.features import add_features

# Chemins vers les fichiers sauvegardés
MODEL_PATH         = Path(__file__).parent.parent / "models" / "logistic_regression.joblib"
FEATURE_NAMES_PATH = Path(__file__).parent.parent / "models" / "feature_names.joblib"


def load_model():
    """Charge le modèle et les noms de features depuis le disque."""
    model         = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return model, feature_names


def prepare(data: dict) -> pd.DataFrame:
    """Transforme un dictionnaire de données brutes en DataFrame prêt pour le modèle.
    
    Applique dans l'ordre :
    1. preprocessing — nettoyage et encodage
    2. feature engineering — création des nouvelles features
    3. alignement des colonnes — même ordre qu'à l'entraînement
    """
    df = pd.DataFrame([data])
    df = preprocess(df)
    df = add_features(df)

    # Aligner les colonnes dans le même ordre qu'à l'entraînement
    # Les colonnes manquantes sont créées avec 0
    # Les colonnes en trop sont supprimées
    _, feature_names = load_model()
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    return df


def predict(data: dict) -> dict:
    """Prédit le churn pour un client donné.
    
    Args:
        data: dictionnaire avec les données brutes du client

    Returns:
        dictionnaire avec :
        - churn : 0 ou 1
        - probability : probabilité de churn entre 0 et 1
        - risk : Low / Medium / High
    """
    model, _ = load_model()
    df       = prepare(data)

    churn       = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    # Niveau de risque basé sur la probabilité
    if probability < 0.35:
        risk = "Low"
    elif probability < 0.60:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "churn"      : churn,
        "probability": round(probability, 4),
        "risk"       : risk
    }