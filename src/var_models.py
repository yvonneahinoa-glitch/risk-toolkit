"""
var_models.py
-------------
Calcul de la Value-at-Risk (VaR) et de l'Expected Shortfall (ES)
par trois méthodes : Paramétrique, Historique, Monte Carlo.

Auteur : Yvonne Nyame
"""

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────
# 1. VaR PARAMETRIQUE (méthode variance-covariance)
# ─────────────────────────────────────────────

def var_parametrique(rendements: pd.Series, confidence: float = 0.95) -> dict:
    """
    Calcule la VaR paramétrique en supposant une distribution normale.

    Paramètres
    ----------
    rendements : pd.Series  — série des rendements journaliers
    confidence : float      — niveau de confiance (ex: 0.95, 0.99)

    Retourne
    --------
    dict avec VaR, ES, moyenne, écart-type, z-score
    """
    mu = rendements.mean()
    sigma = rendements.std()
    z = stats.norm.ppf(1 - confidence)

    var = -(mu + z * sigma)
    # Expected Shortfall = E[perte | perte > VaR]
    es = -(mu - sigma * stats.norm.pdf(z) / (1 - confidence))

    return {
        "methode": "Paramétrique",
        "confidence": confidence,
        "VaR": round(var, 6),
        "ES": round(es, 6),
        "mu": round(mu, 6),
        "sigma": round(sigma, 6),
        "z_score": round(z, 4),
    }


# ─────────────────────────────────────────────
# 2. VaR HISTORIQUE
# ─────────────────────────────────────────────

def var_historique(rendements: pd.Series, confidence: float = 0.95) -> dict:
    """
    Calcule la VaR historique à partir des rendements observés.
    Aucune hypothèse de distribution : utilise directement les quantiles empiriques.

    Paramètres
    ----------
    rendements : pd.Series  — série des rendements journaliers
    confidence : float      — niveau de confiance

    Retourne
    --------
    dict avec VaR, ES, nombre d'observations
    """
    seuil = np.percentile(rendements, (1 - confidence) * 100)
    var = -seuil

    # ES = moyenne des pertes dépassant la VaR
    pertes_extremes = rendements[rendements <= seuil]
    es = -pertes_extremes.mean() if len(pertes_extremes) > 0 else var

    return {
        "methode": "Historique",
        "confidence": confidence,
        "VaR": round(var, 6),
        "ES": round(es, 6),
        "nb_observations": len(rendements),
        "nb_exceptions": len(pertes_extremes),
    }


# ─────────────────────────────────────────────
# 3. VaR MONTE CARLO
# ─────────────────────────────────────────────

def var_monte_carlo(
    rendements: pd.Series,
    confidence: float = 0.95,
    n_simulations: int = 10_000,
    horizon: int = 1,
    seed: int = 42,
) -> dict:
    """
    Calcule la VaR par simulation Monte Carlo.
    Simule N trajectoires de rendements sous hypothèse normale.

    Paramètres
    ----------
    rendements    : pd.Series — série des rendements journaliers
    confidence    : float     — niveau de confiance
    n_simulations : int       — nombre de scénarios simulés
    horizon       : int       — horizon en jours (scaling par racine du temps)
    seed          : int       — graine pour reproductibilité

    Retourne
    --------
    dict avec VaR, ES, paramètres de simulation
    """
    np.random.seed(seed)
    mu = rendements.mean()
    sigma = rendements.std()

    # Simulation de rendements sur l'horizon
    simulations = np.random.normal(
        loc=mu * horizon,
        scale=sigma * np.sqrt(horizon),
        size=n_simulations
    )

    seuil = np.percentile(simulations, (1 - confidence) * 100)
    var = -seuil
    es = -simulations[simulations <= seuil].mean()

    return {
        "methode": "Monte Carlo",
        "confidence": confidence,
        "VaR": round(var, 6),
        "ES": round(es, 6),
        "n_simulations": n_simulations,
        "horizon_jours": horizon,
        "mu_annualisee": round(mu * 252, 4),
        "vol_annualisee": round(sigma * np.sqrt(252), 4),
    }


# ─────────────────────────────────────────────
# 4. COMPARAISON DES TROIS METHODES
# ─────────────────────────────────────────────

def comparer_methodes(
    rendements: pd.Series,
    confidence: float = 0.95,
    n_simulations: int = 10_000,
) -> pd.DataFrame:
    """
    Calcule et compare la VaR et l'ES selon les trois méthodes.

    Retourne un DataFrame récapitulatif.
    """
    resultats = [
        var_parametrique(rendements, confidence),
        var_historique(rendements, confidence),
        var_monte_carlo(rendements, confidence, n_simulations),
    ]

    df = pd.DataFrame([
        {"Méthode": r["methode"], "VaR (%)": r["VaR"] * 100, "ES (%)": r["ES"] * 100}
        for r in resultats
    ])
    df = df.set_index("Méthode").round(4)
    return df


# ─────────────────────────────────────────────
# 5. BACKTESTING (critère Bâle)
# ─────────────────────────────────────────────

def backtesting(
    rendements: pd.Series,
    var_series: pd.Series,
    confidence: float = 0.99,
) -> dict:
    """
    Backtest d'un modèle VaR selon le cadre Bâle II/III.
    Compare les pertes réalisées aux prévisions VaR sur 250 jours.

    Zones Bâle :
        0–4  exceptions → Zone verte  (modèle validé)
        5–9  exceptions → Zone orange (avertissement)
        10+  exceptions → Zone rouge  (modèle rejeté)

    Paramètres
    ----------
    rendements : pd.Series — rendements réalisés
    var_series : pd.Series — VaR prévue (valeurs positives = pertes)
    confidence : float     — niveau de confiance utilisé

    Retourne
    --------
    dict avec nombre d'exceptions, zone Bâle, taux d'exception
    """
    exceptions = rendements < -var_series
    nb_exceptions = exceptions.sum()
    taux = nb_exceptions / len(rendements)

    if nb_exceptions <= 4:
        zone = "🟢 Verte — modèle validé"
    elif nb_exceptions <= 9:
        zone = "🟡 Orange — avertissement"
    else:
        zone = "🔴 Rouge — modèle rejeté"

    return {
        "nb_observations": len(rendements),
        "nb_exceptions": int(nb_exceptions),
        "taux_exception": round(taux * 100, 2),
        "taux_theorique": round((1 - confidence) * 100, 2),
        "zone_bale": zone,
    }
