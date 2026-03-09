# Risk Toolkit — Quantitative Finance

> Boîte à outils Python pour la mesure du risque de marché, l'optimisation de portefeuille et le pricing d'options.  
> Projet développé dans le cadre du Master Finance Banque & Risque (ESSCA).

**Auteur :** Yvonne Nyame  
**Stack :** Python 3.10+ · NumPy · Pandas · SciPy

---

## Structure du projet

```
risk-toolkit/
│
├── src/
│   ├── var_models.py     # VaR paramétrique, historique, Monte Carlo + backtesting Bâle
│   ├── portfolio.py      # Optimisation Markowitz, VaR portefeuille, métriques Sharpe
│   └── options.py        # Black-Scholes, Greeks, volatilité implicite (Newton-Raphson)
│
├── main.py               # Script principal — lance toute l'analyse
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yvonne-nyame/risk-toolkit.git
cd risk-toolkit
pip install -r requirements.txt
python main.py
```

---

## Modèles implémentés

### 1. Value-at-Risk (VaR) & Expected Shortfall

| Méthode | Description |
|---|---|
| **Paramétrique** | Hypothèse normale, variance-covariance |
| **Historique** | Quantile empirique, sans hypothèse de distribution |
| **Monte Carlo** | Simulation de N trajectoires sous hypothèse normale |

```python
from src.var_models import comparer_methodes

# Comparer les 3 méthodes sur une série de rendements
comparaison = comparer_methodes(rendements, confidence=0.95)
print(comparaison)
#               VaR (%)  ES (%)
# Paramétrique   2.9261  3.6776
# Historique     2.9494  3.6432
# Monte Carlo    2.9441  3.7004
```

**Limites de la VaR** (connues des régulateurs) :
- Ne renseigne pas sur l'amplitude des pertes au-delà du seuil
- Suppose souvent des distributions normales (fat tails ignorés)
- Non sous-additive : la VaR d'un portefeuille peut dépasser la somme des VaR individuelles
- → L'Expected Shortfall (ES) corrige ces limites et est au cœur de **FRTB / Bâle IV**

---

### 2. Backtesting Bâle II/III

```python
from src.var_models import backtesting

result = backtesting(rendements_realises, var_prevue, confidence=0.99)
# nb_exceptions   : 3
# zone_bale       : 🟢 Verte — modèle validé
```

Zones réglementaires sur 250 jours d'observation :

| Exceptions | Zone | Conséquence |
|---|---|---|
| 0 – 4 | 🟢 Verte | Modèle validé |
| 5 – 9 | 🟡 Orange | Avertissement, multiplicateur augmenté |
| 10+ | 🔴 Rouge | Modèle rejeté, passage en approche standard |

---

### 3. Optimisation de portefeuille (Markowitz)

```python
from src.portfolio import portefeuille_max_sharpe, portefeuille_minimum_variance

# Portefeuille qui maximise le ratio de Sharpe
max_sh = portefeuille_max_sharpe(rendements)
# → poids optimaux, rendement, volatilité, Sharpe annualisés

# Portefeuille de variance minimale (frontière efficiente)
min_var = portefeuille_minimum_variance(rendements)
```

---

### 4. Pricing Black-Scholes & Greeks

```python
from src.options import black_scholes, volatilite_implicite

# Prix et Greeks d'un call européen
bs = black_scholes(S=50, K=52, T=0.5, r=0.03, sigma=0.20, option_type="call")
# Prix  : 2.28 €
# Delta : 0.46   → le call se comporte comme 46% d'une position sur le sous-jacent
# Vega  : 0.14   → +0.14€ pour +1% de volatilité
# Theta : -0.009 → perte de valeur temps de 0.9 centimes/jour

# Volatilité implicite par Newton-Raphson
vol_impl = volatilite_implicite(prix_marche=3.20, S=50, K=52, T=0.5, r=0.03)
# → 26.52%
```

---

## Résultats (données simulées — actifs CAC 40)

```
Portefeuille Max Sharpe (5 actifs) :
  Rendement annualisé : 11.63%
  Volatilité          : 12.20%
  Ratio de Sharpe     : 0.95

VaR portefeuille (99%, 1 000 000 €) :
  VaR journalière : 16 928 €
  ES journalière  : 19 792 €
```

---

## Références

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*
- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance
- MacKinlay, A.C. (1997). *Event Studies in Economics and Finance*
- Bâle III/IV — Fundamental Review of the Trading Book (FRTB), BCBS 2019

---

## Prochaines extensions

- [ ] Simulation Monte Carlo multivariée (corrélations entre actifs)
- [ ] Modèle GARCH pour la volatilité conditionnelle
- [ ] Surface de volatilité implicite complète
- [ ] Stress testing sur scénarios historiques (COVID-19, GFC 2008)
- [ ] Dashboard interactif avec Streamlit
