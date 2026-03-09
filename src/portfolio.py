"""
portfolio.py
------------
Gestion de portefeuille : rendements, pondérations,
optimisation Markowitz et métriques de performance/risque.

Auteur : Yvonne Nyame
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ─────────────────────────────────────────────
# 1. CHARGEMENT ET PREPARATION DES DONNEES
# ─────────────────────────────────────────────

def calculer_rendements(prix: pd.DataFrame) -> pd.DataFrame:
    """Calcule les rendements journaliers logarithmiques."""
    return np.log(prix / prix.shift(1)).dropna()


def statistiques_actifs(rendements: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne les statistiques clés par actif :
    rendement annualisé, volatilité annualisée, ratio de Sharpe, skewness, kurtosis.
    """
    stats = pd.DataFrame({
        "Rendement annualisé (%)": rendements.mean() * 252 * 100,
        "Volatilité annualisée (%)": rendements.std() * np.sqrt(252) * 100,
        "Sharpe (rf=0)": (rendements.mean() * 252) / (rendements.std() * np.sqrt(252)),
        "Skewness": rendements.skew(),
        "Kurtosis": rendements.kurtosis(),
    }).round(4)
    return stats


# ─────────────────────────────────────────────
# 2. METRIQUES PORTEFEUILLE
# ─────────────────────────────────────────────

def performance_portefeuille(
    rendements: pd.DataFrame, poids: np.ndarray
) -> dict:
    """
    Calcule le rendement et la volatilité d'un portefeuille donné.

    Paramètres
    ----------
    rendements : pd.DataFrame — rendements journaliers des actifs
    poids      : np.ndarray   — vecteur des pondérations (somme = 1)

    Retourne
    --------
    dict avec rendement annualisé, volatilité annualisée, Sharpe
    """
    poids = np.array(poids)
    rend = np.dot(rendements.mean() * 252, poids)
    vol = np.sqrt(np.dot(poids.T, np.dot(rendements.cov() * 252, poids)))
    sharpe = rend / vol if vol > 0 else 0

    return {
        "rendement": round(rend * 100, 4),
        "volatilite": round(vol * 100, 4),
        "sharpe": round(sharpe, 4),
    }


# ─────────────────────────────────────────────
# 3. OPTIMISATION MARKOWITZ
# ─────────────────────────────────────────────

def portefeuille_minimum_variance(rendements: pd.DataFrame) -> dict:
    """
    Trouve le portefeuille de variance minimale (frontière efficiente).
    Contraintes : poids >= 0, somme des poids = 1 (pas de vente à découvert).
    """
    n = rendements.shape[1]
    cov = rendements.cov() * 252

    def variance(poids):
        return np.dot(poids.T, np.dot(cov, poids))

    contraintes = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    bornes = [(0, 1)] * n
    poids_init = np.ones(n) / n

    result = minimize(
        variance, poids_init,
        method="SLSQP",
        bounds=bornes,
        constraints=contraintes,
        options={"ftol": 1e-12}
    )

    poids_opt = result.x
    perf = performance_portefeuille(rendements, poids_opt)

    return {
        "poids": dict(zip(rendements.columns, poids_opt.round(4))),
        "rendement_annualise (%)": perf["rendement"],
        "volatilite_annualisee (%)": perf["volatilite"],
        "sharpe": perf["sharpe"],
    }


def portefeuille_max_sharpe(rendements: pd.DataFrame, rf: float = 0.0) -> dict:
    """
    Trouve le portefeuille qui maximise le ratio de Sharpe.

    Paramètres
    ----------
    rendements : pd.DataFrame — rendements journaliers
    rf         : float        — taux sans risque annualisé (ex: 0.03 = 3%)
    """
    n = rendements.shape[1]
    cov = rendements.cov() * 252
    mu = rendements.mean() * 252

    def neg_sharpe(poids):
        rend = np.dot(mu, poids)
        vol = np.sqrt(np.dot(poids.T, np.dot(cov, poids)))
        return -(rend - rf) / vol if vol > 0 else 0

    contraintes = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    bornes = [(0, 1)] * n
    poids_init = np.ones(n) / n

    result = minimize(
        neg_sharpe, poids_init,
        method="SLSQP",
        bounds=bornes,
        constraints=contraintes,
        options={"ftol": 1e-12}
    )

    poids_opt = result.x
    perf = performance_portefeuille(rendements, poids_opt)

    return {
        "poids": dict(zip(rendements.columns, poids_opt.round(4))),
        "rendement_annualise (%)": perf["rendement"],
        "volatilite_annualisee (%)": perf["volatilite"],
        "sharpe": perf["sharpe"],
    }


# ─────────────────────────────────────────────
# 4. VAR PORTEFEUILLE
# ─────────────────────────────────────────────

def var_portefeuille_historique(
    rendements: pd.DataFrame,
    poids: np.ndarray,
    confidence: float = 0.95,
    valeur_portefeuille: float = 1_000_000,
) -> dict:
    """
    Calcule la VaR historique d'un portefeuille pondéré.

    Paramètres
    ----------
    rendements           : pd.DataFrame — rendements journaliers par actif
    poids                : np.ndarray   — pondérations
    confidence           : float        — niveau de confiance
    valeur_portefeuille  : float        — valeur en euros (pour VaR absolue)
    """
    rend_ptf = rendements.dot(poids)
    seuil = np.percentile(rend_ptf, (1 - confidence) * 100)
    var_pct = -seuil
    var_eur = var_pct * valeur_portefeuille

    pertes = rend_ptf[rend_ptf <= seuil]
    es_pct = -pertes.mean()
    es_eur = es_pct * valeur_portefeuille

    return {
        "VaR (%)": round(var_pct * 100, 4),
        "VaR (€)": round(var_eur, 2),
        "ES (%)": round(es_pct * 100, 4),
        "ES (€)": round(es_eur, 2),
        "confidence": confidence,
        "valeur_portefeuille (€)": valeur_portefeuille,
    }
