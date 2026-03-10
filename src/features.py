# src/features.py
# Fonctions de feature engineering extraites du notebook 04
# Utilisées par FastAPI pour créer les nouvelles features

import pandas as pd
import numpy as np


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Crée AgeGroup — tranches d'âge encodées en 0/1/2/3.
    
    Logique métier : les churners ont en moyenne 2.3 ans de plus
    que les non-churners (confirmé Mann-Whitney p=0.000).
    """
    df["AgeGroup"] = pd.cut(df["Age"],
                            bins=[-np.inf, 0, 1, 2, np.inf],
                            labels=[0, 1, 2, 3]).astype(int)
    return df


def add_has_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Crée HasBalance — 1 si le client a un solde, 0 sinon.
    
    Logique métier : un client sans solde est moins engagé financièrement.
    """
    df["HasBalance"] = (df["Balance"] > 0).astype(int)
    return df


def add_balance_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Crée BalanceRatio — rapport Balance / EstimatedSalary clippé à 10.
    
    Logique métier : un solde élevé par rapport au salaire
    indique un client très impliqué dans la banque.
    """
    df["BalanceRatio"] = (df["Balance"] / (df["EstimatedSalary"] + 1)).clip(upper=10)
    return df


def add_is_multi_product(df: pd.DataFrame) -> pd.DataFrame:
    """Crée IsMultiProduct — 1 si NumOfProducts >= 2, 0 sinon.
    
    Logique métier : confirmé par Kruskal-Wallis (p=0.000),
    les clients avec 3-4 produits churent massivement.
    """
    df["IsMultiProduct"] = (df["NumOfProducts"] >= 2).astype(int)
    return df


def add_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    """Crée EngagementScore — score d'engagement global du client.
    
    Combine IsActiveMember et NumOfProducts pour mesurer
    l'implication globale du client dans la banque.
    """
    df["EngagementScore"] = df["IsActiveMember"] + (df["NumOfProducts"] / 4)
    return df


def add_senior_inactive(df: pd.DataFrame) -> pd.DataFrame:
    """Crée SeniorInactive — 1 si client senior ET inactif.
    
    Logique métier : profil à risque élevé identifié en EDA.
    Age standardisé → 0 correspond à la médiane (40 ans).
    """
    df["SeniorInactive"] = ((df["Age"] > 0) & (df["IsActiveMember"] == 0)).astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet de feature engineering — appelle toutes les fonctions."""
    df = df.copy()
    df = add_age_group(df)
    df = add_has_balance(df)
    df = add_balance_ratio(df)
    df = add_is_multi_product(df)
    df = add_engagement_score(df)
    df = add_senior_inactive(df)
    return df