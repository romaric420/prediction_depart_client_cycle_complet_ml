# frontend/app.py
# Dashboard de prédiction de churn bancaire
# Interface utilisateur pour interagir avec le modèle ChurnGuard

import streamlit as st
import requests
import json

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="ChurnGuard",
    layout="wide"
)

st.title("ChurnGuard — Prédiction de Churn Bancaire")
st.markdown("Renseignez les informations du client pour prédire s'il va churner.")

# Formulaire de saisie
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Informations personnelles")
    geography  = st.selectbox("Pays",        ["France", "Germany", "Spain"])
    gender     = st.selectbox("Genre",       ["Male", "Female"])
    age        = st.slider("Âge",            18, 95, 35)
    tenure     = st.slider("Ancienneté (ans)", 0, 10, 5)

with col2:
    st.subheader("Informations financières")
    credit_score      = st.slider("Credit Score",       300, 850, 650)
    balance           = st.number_input("Solde (€)",    0, 209055, 75000)
    estimated_salary  = st.number_input("Salaire (€)",  0, 236792, 50000)

with col3:
    st.subheader("Informations produits")
    num_products       = st.selectbox("Nombre de produits", [1, 2, 3, 4])
    has_cr_card        = st.selectbox("Carte de crédit",    [1, 0], format_func=lambda x: "Oui" if x == 1 else "Non")
    is_active_member   = st.selectbox("Membre actif",       [1, 0], format_func=lambda x: "Oui" if x == 1 else "Non")
    satisfaction_score = st.slider("Score satisfaction",    1, 5, 3)
    num_complaints     = st.slider("Nombre de plaintes",    0, 5, 0)

st.divider()

if st.button("Prédire le churn", use_container_width=True):
    payload = {
        "CreditScore"      : credit_score,
        "Geography"        : geography,
        "Gender"           : gender,
        "Age"              : age,
        "Tenure"           : tenure,
        "Balance"          : balance,
        "NumOfProducts"    : num_products,
        "HasCrCard"        : has_cr_card,
        "IsActiveMember"   : is_active_member,
        "EstimatedSalary"  : estimated_salary,
        "SatisfactionScore": satisfaction_score,
        "NumComplaints"    : num_complaints
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        result   = response.json()

        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            churn_label = "🔴 Va churner" if result["churn"] == 1 else "🟢 Va rester"
            st.metric("Prédiction", churn_label)

        with col_res2:
            st.metric("Probabilité de churn", f"{result['probability']*100:.1f}%")

        with col_res3:
            risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            st.metric("Niveau de risque", f"{risk_colors[result['risk']]} {result['risk']}")

    except Exception as e:
        st.error(f"Erreur API : {e}")