# backend/api.py
# API REST pour exposer le modèle ChurnGuard
# Endpoint : POST /predict → reçoit les données d'un client
#            GET  /health  → vérifie que l'API est en vie

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import sys
from pathlib import Path

# Ajouter le dossier racine au path pour importer src/
sys.path.append(str(Path(__file__).parent.parent))

from src.predict import predict

app = FastAPI(
    title="ChurnGuard API",
    description="API de prédiction de churn bancaire",
    version="1.0.0"
)


class ClientData(BaseModel):
    """Schéma des données d'entrée pour la prédiction.
    
    Tous les champs correspondent aux colonnes brutes du dataset.
    La validation et le preprocessing sont gérés dans src/predict.py.
    """
    CreditScore     : float = Field(..., example=650)
    Geography       : str   = Field(..., example="France")
    Gender          : str   = Field(..., example="Male")
    Age             : float = Field(..., example=35)
    Tenure          : float = Field(..., example=5)
    Balance         : float = Field(..., example=75000)
    NumOfProducts   : float = Field(..., example=2)
    HasCrCard       : float = Field(..., example=1)
    IsActiveMember  : float = Field(..., example=1)
    EstimatedSalary : float = Field(..., example=50000)
    SatisfactionScore: float = Field(..., example=3)
    NumComplaints   : float = Field(..., example=0)


@app.get("/health")
def health():
    """Vérifie que l'API est en vie et le modèle chargeable."""
    return {"status": "ok", "model": "logistic_regression"}


@app.post("/predict")
def predict_churn(client: ClientData):
    """Prédit le churn pour un client donné.
    
    Returns:
        churn       : 0 = restera, 1 = va churner
        probability : probabilité de churn entre 0 et 1
        risk        : Low / Medium / High
    """
    try:
        data   = client.model_dump()
        result = predict(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))