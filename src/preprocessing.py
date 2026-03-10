# src/preprocessing.py
# Fonctions de nettoyage extraites du notebook 03
# Utilisées par FastAPI pour préparer les nouvelles données

import pandas as pd
import numpy as np


def clean_geography(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les valeurs incohérentes de la colonne Geography."""
    mapping = {
        "Germany": "Germany", "GERMANY": "Germany", "Allemagne": "Germany",
        "France": "France",   "france": "France",   "Frnace": "France",
        "Spain": "Spain",     "Espagne": "Spain"
    }
    df["Geography"] = df["Geography"].map(mapping)
    return df


def clean_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les valeurs incohérentes de la colonne Gender."""
    mapping = {
        "Male": "Male",     "MALE": "Male",   "M": "Male",
        "Homme": "Male",    "1": "Male",
        "Female": "Female", "Femme": "Female", "F": "Female",
        "0": "Female",      "Unknown": np.nan
    }
    df["Gender"] = df["Gender"].map(mapping)
    return df


def clean_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise HasCrCard et IsActiveMember vers 0/1."""
    mapping_card = {
        1: "1", 0: "0", "1": "1", "0": "0",
        "Yes": "1", "No": "0", "True": "1", "False": "0",
        "Oui": "1", "Non": "0", 2: np.nan
    }
    mapping_active = {
        1: "1", 0: "0", "1": "1", "0": "0",
        "active": "1", "inactive": "0", "Y": "1", "N": "0"
    }
    df["HasCrCard"]      = df["HasCrCard"].map(mapping_card).astype(float)
    df["IsActiveMember"] = df["IsActiveMember"].map(mapping_active).astype(float)
    return df


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les colonnes numériques et filtre les valeurs aberrantes."""
    cols = ["CreditScore", "Age", "Tenure", "Balance",
            "NumOfProducts", "EstimatedSalary",
            "SatisfactionScore", "NumComplaints"]

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clipping et filtering des valeurs aberrantes
    df["CreditScore"]       = df["CreditScore"].clip(lower=300, upper=850)
    df["Balance"]           = df["Balance"].clip(upper=209_055)
    df["EstimatedSalary"]   = df["EstimatedSalary"].clip(upper=236_792)
    df["Balance"]           = df["Balance"].where(df["Balance"] >= 0)
    df["EstimatedSalary"]   = df["EstimatedSalary"].where(df["EstimatedSalary"] >= 0)
    df["Tenure"]            = df["Tenure"].where(df["Tenure"].between(0, 10))
    df["Age"]               = df["Age"].where(df["Age"].between(18, 95))
    df["SatisfactionScore"] = df["SatisfactionScore"].where(
                              df["SatisfactionScore"].between(1, 5))
    df["NumOfProducts"]     = df["NumOfProducts"].where(
                              df["NumOfProducts"].between(1, 4))
    df["NumComplaints"]     = df["NumComplaints"].where(
                              df["NumComplaints"].between(0, 10))
    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Impute les valeurs manquantes — médiane pour numériques, mode pour catégorielles."""
    cols_mediane = ["CreditScore", "Age", "Tenure", "Balance",
                    "EstimatedSalary", "SatisfactionScore", "NumComplaints"]
    for col in cols_mediane:
        df[col] = df[col].fillna(df[col].median())

    df["NumOfProducts"]  = df["NumOfProducts"].fillna(df["NumOfProducts"].mode()[0])
    df["HasCrCard"]      = df["HasCrCard"].fillna(df["HasCrCard"].mode()[0])
    df["IsActiveMember"] = df["IsActiveMember"].fillna(df["IsActiveMember"].mode()[0])
    df["Geography"]      = df["Geography"].fillna(df["Geography"].mode()[0])
    df["Gender"]         = df["Gender"].fillna(df["Gender"].mode()[0])
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Geography en One-Hot et Gender en Label Encoding."""
    df = pd.get_dummies(df, columns=["Geography"], drop_first=True)
    df["Gender"] = (df["Gender"] == "Male").astype(int)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet de preprocessing — appelle toutes les fonctions dans l'ordre."""
    df = df.copy()
    df = clean_geography(df)
    df = clean_gender(df)
    df = clean_binary_columns(df)
    df = clean_numeric_columns(df)
    df = impute(df)
    df = encode(df)
    return df