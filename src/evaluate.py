# src/evaluate.py
# Fonctions d'évaluation extraites du notebook 06
# Utilisées pour comparer les modèles et générer les métriques

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report,
                             roc_curve)


def compute_metrics(model, X, y) -> dict:
    """Calcule toutes les métriques d'évaluation pour un modèle.
    
    Args:
        model : modèle sklearn entraîné
        X     : features
        y     : cible réelle

    Returns:
        dictionnaire avec accuracy, precision, recall, f1, roc_auc
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "accuracy"  : round(accuracy_score(y, y_pred), 4),
        "precision" : round(precision_score(y, y_pred, zero_division=0), 4),
        "recall"    : round(recall_score(y, y_pred, zero_division=0), 4),
        "f1"        : round(f1_score(y, y_pred, zero_division=0), 4),
        "roc_auc"   : round(roc_auc_score(y, y_prob), 4)
    }


def plot_confusion_matrix(model, X, y, title: str = "Matrice de Confusion"):
    """Affiche la matrice de confusion pour un modèle.
    
    Args:
        model : modèle sklearn entraîné
        X     : features
        y     : cible réelle
        title : titre du graphique
    """
    y_pred = model.predict(X)
    cm     = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Prédit Non-Churn", "Prédit Churn"],
                yticklabels=["Réel Non-Churn",   "Réel Churn"])
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Valeur Réelle")
    ax.set_xlabel("Valeur Prédite")
    plt.tight_layout()
    plt.show()

    print(classification_report(y, y_pred,
          target_names=["Non-Churn", "Churn"]))


def plot_roc_curve(model, X, y, title: str = "Courbe ROC"):
    """Affiche la courbe ROC pour un modèle.
    
    Args:
        model : modèle sklearn entraîné
        X     : features
        y     : cible réelle
        title : titre du graphique
    """
    y_prob     = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc        = roc_auc_score(y, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2E86AB", lw=2,
            label=f"Modèle (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--",
            label="Baseline aléatoire (AUC = 0.5)")
    ax.set_xlabel("Taux de Faux Positifs")
    ax.set_ylabel("Taux de Vrais Positifs (Recall)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.show()


def compare_models(results: list) -> pd.DataFrame:
    """Affiche un tableau comparatif de plusieurs modèles.
    
    Args:
        results : liste de tuples (nom, metrics_dict)

    Returns:
        DataFrame comparatif trié par F1 décroissant
    """
    rows = []
    for nom, metrics in results:
        row = {"modele": nom}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    print(df.to_string(index=False))
    return df