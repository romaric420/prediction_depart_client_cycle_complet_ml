# ChurnGuard — Système de Prédiction de Churn Bancaire

Pipeline MLOps complet de la donnée brute au déploiement en production.
Classification binaire pour prédire le départ de clients bancaires,
avec exposition via API REST, dashboard interactif, containerisation Docker,
orchestration Kubernetes et automatisation CI/CD.

---

## Apercu du Projet

Le dataset contient 15 750 clients bancaires avec 18 variables.
Apres nettoyage, preprocessing et feature engineering, trois modeles
ont ete entraines et compares. Le modele retenu (Logistic Regression)
est expose via une API REST consommee par un dashboard Streamlit,
le tout containerise et deploye sur Kubernetes.

---

## Stack Technique

**Machine Learning**
Python 3.12, scikit-learn, pandas, numpy, matplotlib, seaborn, scipy

**Versioning et Tracking**
Git, DVC (versioning donnees), MLflow (tracking experiences)

**Deploiement**
FastAPI, Uvicorn, Streamlit, Docker, Kubernetes, GitHub Actions

**Gestion de l'environnement**
uv (gestionnaire de packages et environnements virtuels)

---

## Architecture du Projet

```
classification_churn/
├── notebooks/
│   ├── 01_eda.ipynb                   # Analyse exploratoire complete
│   ├── 02_hypothesis_testing.ipynb    # Tests statistiques (Mann-Whitney, Chi-deux, Kruskal-Wallis)
│   ├── 03_preprocessing.ipynb         # Nettoyage et preparation des donnees
│   ├── 04_feature_engineering.ipynb   # Creation de nouvelles features
│   ├── 05_model_training.ipynb        # Entrainement et comparaison des modeles
│   └── 06_model_evaluation.ipynb      # Evaluation approfondie et feature importance
├── src/
│   ├── preprocessing.py               # Pipeline de nettoyage reutilisable
│   ├── features.py                    # Pipeline de feature engineering reutilisable
│   ├── predict.py                     # Chargement modele et prediction
│   └── evaluate.py                    # Fonctions d'evaluation
├── backend/
│   └── api.py                         # API REST FastAPI
├── frontend/
│   └── app.py                         # Dashboard Streamlit
├── docker/
│   ├── Dockerfile.api                 # Image Docker API
│   ├── Dockerfile.streamlit           # Image Docker Streamlit
│   └── docker-compose.yml             # Orchestration locale
├── k8s/
│   ├── api-deployment.yaml            # Deploiement Kubernetes API (2 replicas)
│   ├── api-service.yaml               # Service Kubernetes API (ClusterIP)
│   ├── streamlit-deployment.yaml      # Deploiement Kubernetes Streamlit
│   └── streamlit-service.yaml         # Service Kubernetes Streamlit (LoadBalancer)
├── models/
│   ├── logistic_regression.joblib     # Modele entraine sauvegarde
│   └── feature_names.joblib           # Noms des features dans l'ordre d'entrainement
├── data/
│   ├── raw/                           # Donnees brutes trackees par DVC
│   └── processed/                     # Donnees preprocessees et feature engineerees
└── .github/workflows/
    └── ci.yml                         # Pipeline CI/CD GitHub Actions
```

---

## Pipeline MLOps

### Etape 1 — Analyse Exploratoire (EDA)

- Detection de 750 doublons (4.76%)
- Identification de 12 types d'encodages incoherents
- Detection de valeurs aberrantes sur CreditScore, Balance, EstimatedSalary, Age, Tenure
- Analyse des correlations et distribution de la variable cible (26.1% churn)

### Etape 2 — Tests d'Hypotheses

Tests statistiques pour valider l'influence de chaque variable sur le churn.

Shapiro-Wilk confirme la non-normalite de toutes les distributions continues,
ce qui oriente le choix vers des tests non parametriques.

- Mann-Whitney : Age (p=0.000), Balance (p=0.344), CreditScore (p=0.517)
- Chi-deux : Geography (p=0.799), IsActiveMember (p=0.446)
- Kruskal-Wallis : NumOfProducts (p=0.000)

Variables confirmees influentes : Age et NumOfProducts.

### Etape 3 — Preprocessing

- Suppression des 750 doublons et des 5 colonnes non pertinentes
- Correction de tous les encodages incoherents via mapping explicite
- Clipping CreditScore [300, 850], Winsorizing Balance et EstimatedSalary
- Imputation par mediane (variables continues) et mode (variables categorielle/binaires)
- One-Hot Encoding Geography, Label Encoding Gender
- Train/Test split 80/20 stratifie (random_state=42)
- Standardisation StandardScaler fittee sur train uniquement (prevention data leakage)

### Etape 4 — Feature Engineering

Six nouvelles features creees a partir de la logique metier bancaire :

- AgeGroup : tranches d'age (correlation 0.0577 avec Exited)
- IsMultiProduct : client avec 2+ produits (correlation 0.0533, confirme par Kruskal-Wallis)
- EngagementScore : score d'engagement combine IsActiveMember et NumOfProducts
- SeniorInactive : client senior et inactif (profil a risque)
- HasBalance : presence d'un solde
- BalanceRatio : rapport Balance / EstimatedSalary

### Etape 5 — Entrainement des Modeles

Quatre modeles entraines et traces dans MLflow :

| Modele               | F1 (test) | ROC-AUC (test) | Remarque               |
|----------------------|-----------|----------------|------------------------|
| Baseline (Dummy)     | 0.0000    | 0.5000         | Reference minimale     |
| Logistic Regression  | 0.3923    | 0.5936         | Meilleur F1 et Recall  |
| Decision Tree        | 0.3862    | 0.6001         | Meilleur ROC-AUC       |
| Random Forest v1     | 0.3294    | 0.5792         | Overfitting detecte    |
| Random Forest v2     | 0.3867    | 0.5893         | Overfitting corrige    |

Modele retenu : Logistic Regression — meilleur recall (0.41),
priorite a la detection des churners en contexte bancaire.

### Etape 6 — Evaluation

- Matrice de confusion : 1668 TN, 547 FP, 460 FN, 325 TP
- AUC = 0.5936, au-dessus de la baseline aleatoire
- Top features : IsMultiProduct (+0.60), AgeGroup (-0.63), IsActiveMember (-0.30)
- Les deux features creees en feature engineering sont les plus importantes du modele

---

## Installation et Lancement

### Pre-requis

- Python 3.12
- uv
- Docker Desktop
- kubectl

### Installation

```bash
git clone https://github.com/romaric420/prediction_depart_client_cycle_complet_ml.git
cd classification_churn
uv sync --frozen
```

### Lancement local

```bash
# Terminal 1 — API
uv run uvicorn backend.api:app --reload --port 8000

# Terminal 2 — Dashboard
uv run streamlit run frontend/app.py
```

API disponible sur : http://localhost:8000
Dashboard disponible sur : http://localhost:8501
Documentation API (Swagger) : http://localhost:8000/docs

### Lancement avec Docker

```bash
docker compose -f docker/docker-compose.yml up
```

### Deploiement Kubernetes

```bash
# Build des images
docker build -f docker/Dockerfile.api -t churnguard-api:latest .
docker build -f docker/Dockerfile.streamlit -t churnguard-streamlit:latest .

# Deploiement
kubectl apply -f k8s/

# Verification
kubectl get pods
kubectl get services
```

---

## API Reference

### GET /health

Verifie que l'API est operationnelle.

```json
{"status": "ok", "model": "logistic_regression"}
```

### POST /predict

Predit le churn pour un client.

Exemple de requete :

```json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Male",
  "Age": 35,
  "Tenure": 5,
  "Balance": 75000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 50000,
  "SatisfactionScore": 3,
  "NumComplaints": 0
}
```

Exemple de reponse :

```json
{
  "churn": 0,
  "probability": 0.2341,
  "risk": "Low"
}
```

Niveaux de risque : Low (< 35%), Medium (35-60%), High (> 60%)

---

## CI/CD

Le pipeline GitHub Actions se declenche a chaque push sur main :

1. Job test : validation des imports src/, installation des dependances
2. Job build : build et push des images Docker vers Docker Hub

---

## Format de Sauvegarde du Modele

Le modele est sauvegarde au format joblib, recommande officiellement
par scikit-learn pour sa performance sur les arrays numpy internes.

- models/logistic_regression.joblib : modele entraine
- models/feature_names.joblib : ordre des features a l'entrainement

---

## Auteur

Romaric TCHOFFO
Etudiant Master Data Science & IA — EPSI Paris
Developpeur Full Stack & IA