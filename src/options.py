"""
options.py
----------
Pricing d'options européennes par le modèle Black-Scholes.
Calcul des Greeks : Delta, Gamma, Vega, Theta, Rho.

Auteur : Yvonne Nyame
"""

import numpy as np
from scipy.stats import norm


# ─────────────────────────────────────────────
# 1. PRICING BLACK-SCHOLES
# ─────────────────────────────────────────────

def black_scholes(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> dict:
    """
    Prix d'une option européenne selon Black-Scholes.

    Paramètres
    ----------
    S           : float — prix spot du sous-jacent
    K           : float — prix d'exercice (strike)
    T           : float — maturité en années (ex: 0.5 = 6 mois)
    r           : float — taux sans risque annualisé (ex: 0.03)
    sigma       : float — volatilité implicite annualisée (ex: 0.20)
    option_type : str   — "call" ou "put"

    Retourne
    --------
    dict avec prix, d1, d2 et tous les Greeks
    """
    if T <= 0:
        raise ValueError("La maturité T doit être strictement positive.")
    if sigma <= 0:
        raise ValueError("La volatilité sigma doit être strictement positive.")

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        prix = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        ) / 365

    elif option_type == "put":
        prix = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        ) / 365

    else:
        raise ValueError("option_type doit être 'call' ou 'put'.")

    # Greeks communs call et put
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # pour 1% de vol

    return {
        "type": option_type,
        "prix": round(prix, 4),
        "d1": round(d1, 4),
        "d2": round(d2, 4),
        "Greeks": {
            "Delta": round(delta, 4),
            "Gamma": round(gamma, 6),
            "Vega":  round(vega, 4),
            "Theta": round(theta, 4),
            "Rho":   round(rho, 4),
        },
        "parametres": {
            "S": S, "K": K, "T": T, "r": r, "sigma": sigma
        }
    }


# ─────────────────────────────────────────────
# 2. PARITE CALL-PUT
# ─────────────────────────────────────────────

def parite_call_put(
    S: float, K: float, T: float, r: float, sigma: float
) -> dict:
    """
    Vérifie la parité call-put : C - P = S - K * e^(-rT)

    Utile pour détecter des opportunités d'arbitrage.
    """
    call = black_scholes(S, K, T, r, sigma, "call")["prix"]
    put  = black_scholes(S, K, T, r, sigma, "put")["prix"]

    membre_gauche = call - put
    membre_droit  = S - K * np.exp(-r * T)
    ecart = abs(membre_gauche - membre_droit)

    return {
        "C - P": round(membre_gauche, 6),
        "S - K*e^(-rT)": round(membre_droit, 6),
        "écart": round(ecart, 8),
        "parité respectée": ecart < 1e-6,
    }


# ─────────────────────────────────────────────
# 3. SURFACE DE VOLATILITE IMPLICITE SIMPLIFIEE
# ─────────────────────────────────────────────

def volatilite_implicite(
    prix_marche: float,
    S: float, K: float, T: float, r: float,
    option_type: str = "call",
    precision: float = 1e-6,
    max_iter: int = 200,
) -> float:
    """
    Calcule la volatilité implicite par méthode de Newton-Raphson.

    Paramètres
    ----------
    prix_marche : float — prix observé sur le marché
    S, K, T, r  : float — paramètres Black-Scholes
    option_type : str   — "call" ou "put"
    precision   : float — critère de convergence
    max_iter    : int   — nombre max d'itérations

    Retourne
    --------
    volatilité implicite (float) ou None si non convergé
    """
    sigma = 0.2  # estimation initiale

    for _ in range(max_iter):
        bs = black_scholes(S, K, T, r, sigma, option_type)
        prix_bs = bs["prix"]
        vega    = bs["Greeks"]["Vega"] * 100  # annulation du /100 appliqué dans bs

        diff = prix_bs - prix_marche
        if abs(diff) < precision:
            return round(sigma, 6)

        if abs(vega) < 1e-10:
            break

        sigma -= diff / vega
        sigma = max(sigma, 1e-6)  # sigma strictement positif

    return None  # non convergé
