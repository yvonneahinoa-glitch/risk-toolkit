"""
main.py
-------
Script principal : télécharge des données réelles et applique tous les modèles.
Lance ce fichier pour voir le projet en action.

Usage : python main.py

Auteur : Yvonne Nyame
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from var_models import comparer_methodes, backtesting, var_historique
from portfolio import (
    calculer_rendements, statistiques_actifs,
    portefeuille_minimum_variance, portefeuille_max_sharpe,
    var_portefeuille_historique
)
from options import black_scholes, parite_call_put, volatilite_implicite

SEPARATEUR = "=" * 60


def section(titre):
    print(f"\n{SEPARATEUR}")
    print(f"  {titre}")
    print(SEPARATEUR)


def demo_sans_telechargement():
    """
    Démonstration complète avec données simulées
    (utilise les mêmes modèles que les données réelles).
    """
    np.random.seed(42)

    # Simulation de rendements réalistes
    ACTIFS = ["BNP Paribas", "TotalEnergies", "LVMH", "Airbus", "Sanofi"]
    n_jours = 1000
    mu = np.array([0.0003, 0.0004, 0.0005, 0.0002, 0.0001])
    vol = np.array([0.018, 0.015, 0.016, 0.020, 0.013])
    dates = pd.date_range("2020-01-01", periods=n_jours, freq="B")

    rendements = pd.DataFrame(
        np.random.normal(mu, vol, size=(n_jours, len(ACTIFS))),
        index=dates,
        columns=ACTIFS
    )

    # ── 1. STATISTIQUES PAR ACTIF ──────────────────────────────────
    section("1. STATISTIQUES DES ACTIFS")
    stats = statistiques_actifs(rendements)
    print(stats.to_string())

    # ── 2. VAR PAR LES TROIS METHODES ─────────────────────────────
    section("2. COMPARAISON VAR — BNP Paribas (confiance 95%)")
    bnp = rendements["BNP Paribas"]
    comparaison = comparer_methodes(bnp, confidence=0.95)
    print(comparaison.to_string())
    print("\nInterprétation : avec une confiance de 95%, la perte journalière")
    print("ne devrait pas dépasser la VaR sur 19 jours sur 20.")

    # ── 3. BACKTESTING ─────────────────────────────────────────────
    section("3. BACKTESTING BALE (VaR 99%, 250 derniers jours)")
    bnp_250 = bnp.tail(250)
    res_var = var_historique(bnp_250, confidence=0.99)
    var_constante = pd.Series(res_var["VaR"], index=bnp_250.index)
    backtest = backtesting(bnp_250, var_constante, confidence=0.99)
    for k, v in backtest.items():
        print(f"  {k:25s} : {v}")

    # ── 4. OPTIMISATION PORTEFEUILLE ───────────────────────────────
    section("4. OPTIMISATION MARKOWITZ")

    print("\n→ Portefeuille Variance Minimale :")
    min_var = portefeuille_minimum_variance(rendements)
    for actif, poids in min_var["poids"].items():
        print(f"   {actif:15s} : {poids*100:.1f}%")
    print(f"   Rendement   : {min_var['rendement_annualise (%)']:.2f}%")
    print(f"   Volatilité  : {min_var['volatilite_annualisee (%)']:.2f}%")
    print(f"   Sharpe      : {min_var['sharpe']:.2f}")

    print("\n→ Portefeuille Max Sharpe :")
    max_sh = portefeuille_max_sharpe(rendements)
    for actif, poids in max_sh["poids"].items():
        print(f"   {actif:15s} : {poids*100:.1f}%")
    print(f"   Rendement   : {max_sh['rendement_annualise (%)']:.2f}%")
    print(f"   Volatilité  : {max_sh['volatilite_annualisee (%)']:.2f}%")
    print(f"   Sharpe      : {max_sh['sharpe']:.2f}")

    # ── 5. VAR PORTEFEUILLE ────────────────────────────────────────
    section("5. VAR PORTEFEUILLE (1 000 000 €, confiance 99%)")
    poids_opt = np.array(list(min_var["poids"].values()))
    var_ptf = var_portefeuille_historique(
        rendements, poids_opt, confidence=0.99, valeur_portefeuille=1_000_000
    )
    for k, v in var_ptf.items():
        print(f"  {k:30s} : {v}")

    # ── 6. PRICING OPTIONS BLACK-SCHOLES ──────────────────────────
    section("6. PRICING OPTIONS — BLACK-SCHOLES")
    print("\nExemple : Option Call sur BNP Paribas")
    print("  S=50€, K=52€, T=6 mois, r=3%, σ=20%\n")

    bs_call = black_scholes(S=50, K=52, T=0.5, r=0.03, sigma=0.20, option_type="call")
    print(f"  Prix du Call : {bs_call['prix']} €")
    print(f"  d1={bs_call['d1']}, d2={bs_call['d2']}")
    print("\n  Greeks :")
    for g, v in bs_call["Greeks"].items():
        print(f"    {g:6s} = {v}")

    print("\n→ Put équivalent (parité call-put) :")
    bs_put = black_scholes(S=50, K=52, T=0.5, r=0.03, sigma=0.20, option_type="put")
    print(f"  Prix du Put  : {bs_put['prix']} €")

    parite = parite_call_put(S=50, K=52, T=0.5, r=0.03, sigma=0.20)
    print(f"\n  Vérification parité call-put :")
    for k, v in parite.items():
        print(f"    {k:25s} : {v}")

    # ── 7. VOLATILITE IMPLICITE ────────────────────────────────────
    section("7. VOLATILITE IMPLICITE (Newton-Raphson)")
    prix_marche = 3.20
    vol_impl = volatilite_implicite(
        prix_marche=prix_marche, S=50, K=52, T=0.5, r=0.03, option_type="call"
    )
    print(f"\n  Prix observé sur le marché : {prix_marche} €")
    print(f"  Volatilité implicite       : {vol_impl*100:.2f}%")
    print(f"  (La vol implicite à σ=20% donne un prix de {bs_call['prix']} €)")

    print(f"\n{SEPARATEUR}")
    print("  ✅ Analyse terminée — tous les modèles ont tourné correctement.")
    print(SEPARATEUR)


if __name__ == "__main__":
    print("\n🔵 RISK TOOLKIT — Yvonne Nyame")
    print("   VaR · Monte Carlo · Markowitz · Black-Scholes")
    demo_sans_telechargement()
