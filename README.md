# Churn Analytica

## Analyse et Prédiction du Churn Télécom — Pipeline End-to-End

Ce projet constitue une analyse complète du churn client dans le secteur des télécommunications. Il couvre l'intégralité du cycle de vie d'un projet de data science : ingestion de données multi-sources, prétraitement rigoureux, analyse statistique descriptive, segmentation non-supervisée par clustering, extraction de règles métier interprétables, et modélisation supervisée déployée dans une application interactive.

---

## Contexte et Objectifs

Le churn désigne la résiliation volontaire d'un abonnement par un client. Dans le secteur télécom, retenir un client existant coûte significativement moins cher qu'en acquérir un nouveau. Ce projet vise deux objectifs distincts et complémentaires :

- **Prédire** la probabilité de départ pour chaque client à partir de ses attributs comportementaux et contractuels.
- **Prescrire** des actions de rétention personnalisées selon le segment auquel appartient chaque client.

---

## Dataset

Le jeu de données est le Telco Customer Churn dataset, réparti en cinq fichiers CSV correspondant à cinq domaines fonctionnels distincts.

| Fichier | Contenu | Clé de jointure |
|---|---|---|
| `Telecom_Demographie.csv` | Âge, genre, situation familiale | `CustomerID` |
| `Telecom_Localisation.csv` | Ville, code postal, coordonnées GPS | `CustomerID` |
| `Telecom_Services.csv` | Abonnements internet, téléphone, services optionnels | `CustomerID` |
| `Telecom_Statut.csv` | Churn Value, Churn Reason, CLTV, Tenure Months | `CustomerID` |
| `Telecom_Population.csv` | Population par code postal | `Zip Code` |

Après fusion : **7 043 clients**, **35 variables**, **0 doublon**. Taux de churn : 26.5 %.

---

## Structure du Pipeline

Le pipeline se décompose en dix étapes séquentielles, chacune alimentant la suivante.

```
1. Fusion des 5 sources CSV
         |
2. Exploration initiale (types, NaN, distribution cible)
         |
3. Pretraitement & Feature Engineering
         |
4. Feature Selection (MI, RF Importance, Pearson, Cramér)
         |
5. Analyse Descriptive (Boxplots, Skewness, Lorenz, Gini)
         |
6. Questions Metier (4 analyses ciblées)
         |
7. Clustering non-supervisé (K-Means + ACP)
         |
8. Arbres de décision & Règles IF-THEN
         |
9. Modélisation supervisée (LR, RF, Gradient Boosting)
         |
10. Déploiement — Application Streamlit
```

---

## Étapes Détaillées

### Étape 1 — Fusion des Sources

Les cinq fichiers sont fusionnés progressivement sur `CustomerID`. La table Population fait l'objet d'une agrégation préalable par `Zip Code` pour éviter une jointure de type Many-to-Many. Une vérification de qualité garantit l'unicité de chaque `CustomerID` dans la table finale.

```python
df_pop_clean = df_pop.groupby('Zip Code', as_index=False)['Population'].mean()
df = df_demo.merge(df_loc, on='CustomerID')
           .merge(df_svc, on='CustomerID')
           .merge(df_stat, on='CustomerID')
           .merge(df_pop_clean, on='Zip Code', how='left')
```

### Étape 2 — Exploration Initiale

Trois points d'attention identifiés :

1. **Déséquilibre de classes** : 26.5 % de churners contre 73.5 % de fidèles, ce qui impose l'usage de métriques adaptées (AUC-ROC, F1, Recall) plutôt que l'accuracy simple.
2. **Churn Reason vide pour les clients fidèles** : absence normale par construction, non traitée comme une valeur manquante.
3. **Total Charges au format texte** : des espaces vides existent pour les clients dont l'ancienneté est nulle. Correction par conversion forcée puis imputation `Monthly Charges × Tenure Months`.

### Étape 3 — Prétraitement et Feature Engineering

**Variables supprimées (9)** :

| Variable | Motif |
|---|---|
| `CustomerID` | Identifiant unique, aucune information prédictive |
| `Churn Label`, `Churn Score`, `Churn Reason` | Data leakage : disponibles uniquement après le churn |
| `CLTV Category` | Discrétisation redondante de `CLTV` |
| `Count`, `Country` | Constantes (variance nulle) |
| `Lat Long` | Redondant avec `Latitude` et `Longitude` séparées |

**Variables créées (3)** :

| Variable | Construction | Intérêt |
|---|---|---|
| `Tenure Group` | Découpage en 4 tranches : 0-1 an, 1-2 ans, 2-4 ans, 4-6 ans | Capture l'effet cycle de vie client |
| `Charge Group` | Découpage en 4 niveaux : <35$, 35-60$, 60-90$, >90$ | Segmentation par niveau de prix |
| `N Services` | Compte des 6 services optionnels souscrits (0 à 6) | Mesure quantitative de l'engagement |

### Étape 4 — Feature Selection

Deux méthodes complémentaires sont appliquées en parallèle. Une variable est retenue si elle satisfait au moins l'un des deux critères.

**Mutual Information** (seuil MI > 0.02) : mesure la dépendance statistique entre chaque variable et la cible, y compris les relations non-linéaires. Formellement :

```
I(X ; Y) = H(X) - H(X|Y) = sum_x sum_y p(x,y) * log[ p(x,y) / (p(x)*p(y)) ]
```

**Random Forest Importance** (seuil > 0.01) : contribution moyenne à la réduction d'impureté de Gini sur l'ensemble des arbres. Capture les interactions entre variables.

Pour les variables numériques, la **corrélation de Pearson** est calculée. Pour les variables catégorielles, le **V de Cramér** (basé sur le chi-deux normalisé) est utilisé.

Résultats principaux :

| Variable | MI Score | RF Importance | Pearson / Cramér |
|---|---|---|---|
| Contract | Élevé | Élevé | V = 0.410 (fort) |
| Tenure Months | Élevé | Élevé | r = -0.352 (modéré négatif) |
| Monthly Charges | Modéré | Modéré | r = +0.193 (faible positif) |
| Online Security | Modéré | Modéré | V = 0.340 (fort) |
| Dependents | Modéré | Faible | V significatif |

### Étape 5 — Analyse Descriptive

Les distributions des variables numériques sont analysées selon quatre angles :

- **Boxplots comparatifs** Churners vs Fidèles : les churners présentent une ancienneté médiane plus faible, des charges mensuelles plus élevées, et un CLTV plus faible.
- **Skewness et Kurtosis** : Total Charges affiche une asymétrie positive (skewness > 1), ce qui signifie que quelques clients très anciens tirent la moyenne vers le haut.
- **Coefficient de variation** : CLTV (CV ≈ 99 %) est beaucoup plus hétérogène que Monthly Charges (CV ≈ 27 %), justifiant la standardisation avant le clustering.
- **Courbes de Lorenz et indice de Gini** : un Gini élevé sur CLTV confirme qu'une minorité de clients génère la majorité de la valeur.

```python
Gini = 1 - 2 * integral_0_1 L(p) dp
```

### Étape 6 — Questions Métier

Quatre questions analytiques sont adressées directement :

**Q1 — Facteurs d'influence** : d'après la RF Importance, Contract, Tenure Months, Monthly Charges, CLTV et N Services sont les cinq facteurs les plus déterminants.

**Q2 — Impact du contrat** : taux de churn de l'ordre de 43 % pour le contrat Mois-à-Mois, contre 11 % pour le contrat annuel et moins de 3 % pour le contrat bi-annuel.

**Q3 — Services protecteurs** : Online Security et Tech Support réduisent le taux de churn de plusieurs points de pourcentage. Les clients sans ces services churne significativement plus.

**Q4 — Profil financier** : les churners sont typiquement des clients récents sur des forfaits premium, sans engagement contractuel long terme et avec un CLTV plus faible.

### Étape 7 — Clustering K-Means + ACP

**Préparation** : cinq variables numériques sélectionnées (`Tenure Months`, `Monthly Charges`, `Total Charges`, `CLTV`, `N Services`), standardisées avec `StandardScaler` (moyenne 0, écart-type 1). La standardisation est obligatoire car K-Means est sensible aux échelles via la distance euclidienne.

**Choix du K** : méthode du coude sur la courbe inertie (WCSS) pour K allant de 1 à 10. Le coude est identifié à K = 3.

```
WCSS = sum_k sum_{x in C_k} ||x - mu_k||^2
```

**Résultats K = 3** :

| Cluster | Nom métier | Ancienneté médiane | Taux de churn |
|---|---|---|---|
| 0 | Fidèles Premium | > 48 mois | < 15 % |
| 1 | Nouveaux à Risque | < 15 mois | > 35 % |
| 2 | Gros Consommateurs | Variable | Modéré |

**Visualisation ACP** : projection sur les deux premières composantes principales (PC1 + PC2). Le cercle des corrélations révèle que PC1 est principalement associé à l'ancienneté et à la valeur (Tenure, Total Charges, CLTV), tandis que PC2 capte davantage les charges mensuelles actuelles.

```
Sigma * v_k = lambda_k * v_k   (décomposition spectrale de la matrice de covariance)
loadings = vecteurs propres * sqrt(valeurs propres)
```

### Étape 8 — Arbres de Décision et Règles Métier

**Optimisation de la profondeur** : courbe AUC-ROC Train vs Validation Croisée 5-fold pour des profondeurs allant de 2 à 14. La profondeur optimale est identifiée à 5 (AUC-CV = 0.8388, gap Train-CV = -0.064).

Paramètres de régularisation : `min_samples_split=100`, `min_samples_leaf=50`, `class_weight='balanced'`.

**Extraction des règles IF-THEN** : parcours récursif de l'arbre (DFS). Exemple de règle principale extraite :

```
SI   Contrat = Month-to-month
AND  Dependents = No
AND  Online Security = No
AND  Ancienneté <= 10.5 mois
ALORS  CHURN  (confiance : 84.9 %)
```

Les noms de variables techniques sont traduits en langage métier pour une exploitation directe par les équipes non-techniques.

### Étape 9 — Modélisation Supervisée

**Trois modèles** évalués en validation croisée stratifiée 5-fold :

```python
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

| Modèle | Paramètres clés | Principe |
|---|---|---|
| Logistic Regression | `class_weight='balanced'`, `C=1.0` | Modèle linéaire, baseline interprétable |
| Random Forest | `n_estimators=200`, `max_depth=8`, `class_weight='balanced'` | Bagging de 200 arbres |
| Gradient Boosting | `n_estimators=200`, `max_depth=4`, `learning_rate=0.05` | Boosting séquentiel par descente de gradient |

**Coût asymétrique des erreurs** : un faux négatif (churner non détecté) coûte significativement plus qu'un faux positif (fidèle classé churner). Le seuil de décision est optimisé pour maximiser le Recall sur la classe churn.

```
Recall = TP / (TP + FN)
```

**Meilleur modèle** : Gradient Boosting (AUC-ROC le plus élevé des trois). Les matrices de confusion sont analysées pour quantifier les faux négatifs (churners manqués) de chaque modèle.

### Étape 10 — Déploiement Streamlit

Le modèle final est sérialisé et intégré dans une application Streamlit comprenant deux modules :

- **Simulateur de risque** : saisie des attributs d'un client, retour de la probabilité de churn en temps réel.
- **Moteur de recommandation** : selon le cluster K-Means du client, une offre de rétention personnalisée est proposée.

---

## Stack Technique

```
Langage          Python 3.x
Manipulation     pandas, numpy
Visualisation    matplotlib, seaborn
Machine Learning sklearn (LabelEncoder, StandardScaler, KMeans, PCA,
                          DecisionTreeClassifier, RandomForestClassifier,
                          GradientBoostingClassifier, LogisticRegression,
                          StratifiedKFold, cross_validate)
Statistiques     scipy.stats (chi2_contingency, skew, kurtosis)
Déploiement      Streamlit
Notebook         Jupyter
```

---

## Paramètres Globaux

Tous les paramètres déterministes sont centralisés en début de notebook pour garantir la reproductibilité complète des résultats.

```python
RANDOM_STATE = 42
TEST_SIZE    = 0.25
N_FOLDS      = 5
```

---

## Résultats Principaux

- **Variable la plus prédictive** : le type de contrat (Contract), avec un V de Cramér de 0.410 et la RF Importance la plus élevée parmi toutes les variables.
- **Segmentation** : trois profils clients distincts identifiés par K-Means, visuellement séparables dans la projection ACP.
- **Règle principale** : les clients en contrat mensuel, sans dépendants, sans sécurité en ligne et avec moins de 10.5 mois d'ancienneté churne avec une confiance de 84.9 %.
- **Modèle optimal** : Gradient Boosting, évalué sur cinq métriques complémentaires (Accuracy, AUC-ROC, F1, Recall, Precision) en validation croisée stratifiée 5-fold.

---

## Lancer le Projet

```bash
# Cloner le dépôt
git clone https://github.com/votre-utilisateur/churn-analytica.git
cd churn-analytica

# Installer les dépendances
pip install -r requirements.txt

# Lancer le notebook d'analyse
jupyter notebook Analyse_notebook.ipynb

# Lancer l'application Streamlit
streamlit run app.py
```

---

## Organisation du Dépôt

```
churn-analytica/
|-- data/
|   |-- Telecom_Demographie.csv
|   |-- Telecom_Localisation.csv
|   |-- Telecom_Services.csv
|   |-- Telecom_Statut.csv
|   |-- Telecom_Population.csv
|
|-- Analyse_notebook.ipynb 
|-- app.py                      
|-- requirements.txt
|-- README.md
```