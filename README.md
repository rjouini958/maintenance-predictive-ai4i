# Maintenance prédictive — AI4I 2020

Prioriser les inspections d'un atelier à partir de mesures capteurs, pour limiter les pannes manquées et les inspections inutiles.

**Résultat sur 2 000 observations test réservées : 59/68 pannes détectées, 24 fausses alertes, rappel 86.8%, précision 71.1%, AP 0.886.** Modèle : Random Forest, seuil 0.17. Le dataset officiel est synthétique : aucun résultat industriel réel ni préavis à 24 h n'est démontré.

![Matrice de confusion](results/figures/05_confusion.png)

## Trois notebooks exécutés

| Notebook | Contenu | Axes des consignes |
|---|---|---|
| [01_cadrage_eda.ipynb](notebooks/01_cadrage_eda.ipynb) | Cadrage, source, dictionnaire, qualité, exploration train, audit de fuite, variables physiques | 1, 2, 3 |
| [02_modelisation.ipynb](notebooks/02_modelisation.ipynb) | Baseline et trois familles, CV, réglages imbriqués, comparaison train/validation, choix du seuil par coût | 4, 5, 6 |
| [03_decision_roi.ipynb](notebooks/03_decision_roi.ipynb) | Test final, erreurs, importances, incertitude, ROI et risques | 5, 6, 7, 8, 9 |

Les tableaux et graphiques sont enregistrés dans les notebooks et visibles sur GitHub sans exécuter Python. **Télécharger le dossier complet** : les notebooks utilisent `src/`, `data/` et les résultats sauvegardés. Le notebook 03 reconstruit le pipeline à partir du choix JSON produit par 02 et le réajuste sur le train uniquement ; il peut être relancé seul avec les fichiers fournis, sans modèle binaire.

## Installation et exécution

Version testée : **Python 3.12.14**. Les versions exactes des dépendances sont dans requirements.txt.
Depuis la racine du projet :

```bash
python -m venv .venv
# Linux / macOS :
source .venv/bin/activate
# Windows PowerShell (à la place de la ligne précédente) :
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/execute_local.py
python scripts/verify_results.py
```

`execute_local.py` lance un **nouveau processus Python et une session IPython neuve pour chaque notebook**, exécute toutes ses cellules dans l'ordre et réenregistre les sorties. C'est la méthode utilisée ici, car l'environnement d'exécution interdit les sockets des noyaux Jupyter. Les trois notebooks ont passé ce contrôle. Le test de l'interface graphique « Restart & Run All » n'a pas été réalisé dans cet environnement ; aucune cellule ne dépend d'un état interactif préalable.

Alternative avec de vrais noyaux Jupyter, sur une machine autorisant les sockets :
```bash
python scripts/run_all.py
```
Pour ouvrir les notebooks interactivement, utiliser VS Code avec l'extension Jupyter ou installer `jupyterlab` puis lancer `jupyter lab`. Exécuter 01 → 02 → 03. Compter quelques minutes selon la machine (environ 228 secondes pour l'exécution vérifiée ici).

Aucun téléchargement de données n'est nécessaire à l'exécution. Le modèle joblib est conservé comme export facultatif ; il n’est plus nécessaire à l’exécution du notebook 03.

## Protocole et choix

- CSV officiel non modifié : 10 000 lignes, 14 colonnes, 339 pannes (3,39 %), aucune valeur manquante et aucun doublon de ligne complète.
- Source, licence et empreinte SHA-256 dans `data/`. Le programme vérifie l'empreinte.
- Split stratifié 80/20, graine 42. Exploration des relations et choix de modèles sur train seulement.
- Exclusion des modes TWF/HDF/PWF/OSF/RNF et identifiants UDI/Product ID. Pas de réétiquetage des 27 divergences entre cible et OU des modes.
- Trois variables physiques : écart de température, puissance en W et usure × couple. Imputation, encodage et standardisation dans le pipeline.
- Comparaison de sept configurations : baseline, logistique, arbre, forêt, puis trois versions réglées. CV externe 5 plis ; réglage interne 3 plis pour chaque pli externe. Moyennes et écarts-types exportés.
- L'AP sert au réglage ; rappel ≥ 80 % et précision ≥ 1/3 sont les contraintes métier. Le modèle et le seuil minimisent le coût sur prédictions hors-pli train.
- Le modèle final est une forêt non réglée : la recherche d'hyperparamètres a bien été effectuée sur les autres configurations, sans supposer qu'elle doit forcément gagner.
- Test ouvert après sélection ; les diagnostics postérieurs (courbe, permutations, erreurs) ne servent pas à réajuster le modèle.
- Le choix parmi les familles et seuils crée un optimisme sur le train hors-pli : seuls les scores test servent de bilan final. Relancer le code sur le même test n'apporte pas une nouvelle validation indépendante.

## Valeur métier sous hypothèses

Coûts : FN 3 000 €, FP 150 €, TP 500 €, TN 0 €. Volume 5 000 cycles/an ; coût projet année 1 de 55 000 €.

- Gain brut central : **358 692 €/an** ; gain net année 1 : **303 692 €** ; ROI : **552,2 %**.
- Scénario pessimiste : gain net de **49 267 €**, avec rappel et précision réduits de 20 % et panne à 1 500 €.
- Scénario défavorable (panne à 500 €) : gain net de **−63 974 €**.

Ces montants sont des projections pédagogiques avec efficacité de prévention supposée. Un scénario additionnel réduit cette efficacité à 70 %. La référence « sans alerte » ne mesure pas la maintenance calendaire réellement pratiquée. Un pilote de trois mois est nécessaire pour mesurer coûts, disponibilité précoce des capteurs, capacité d'inspection et efficacité réelle des interventions.

## Vérifications et livrables

- `results/execution_report.json` : rapport des trois exécutions isolées réussies.
- `results/verification.txt` : contrôles indépendants des partitions, prédictions hors-pli, exclusion des fuites, seuil, métriques, conversion physique et ROI.
- `results/` : scores par pli, comparatif, toutes les valeurs de seuil, prédictions test, bootstrap, importances, scénarios et modèle final.
- `results/figures/` : huit figures exportées.
- `docs/corrections_diaporama.md` : valeurs à reporter page par page. **Le PDF initial n'a pas été modifié et ses anciens chiffres ne correspondent pas à ce code.**
- `docs/DEPOT_GITHUB.md` : procédure de publication. Aucune publication GitHub n'a été effectuée dans cette livraison.

## Limites

Dataset synthétique à règles physiques explicites, peu de pannes test (68), absence de dates et d'identifiant machine, dépendances entre cycles, pas de mesure causale de l'effet d'inspection. Les intervalles bootstrap supposent des observations indépendantes et ne couvrent pas le changement de domaine. Une validation temporelle/par équipement sur des données réelles est nécessaire avant une utilisation opérationnelle.

## Source et licence des données

AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository. DOI : https://doi.org/10.24432/C5HS5C. Licence CC BY 4.0. Consulté le 23 octobre 2026. Voir [data/README.md](data/README.md).

## Déclaration d'utilisation de l'IA

Projet préparé avec l'assistance de **ChatGPT / Codex (OpenAI)** : lecture des consignes et du diaporama, rédaction du code et des explications, exécution des notebooks, contrôle des résultats et préparation du dépôt. Les données proviennent du fichier officiel UCI.

**Équipe :**

- Rayan Jouini
- Wessam El Alaoui
- Fatimata Kane
- Ahmed Habibou

## Révision des notebooks

Les trois fichiers ont été revus : guides de lecture, synthèses pour l'oral, tableau de validation plus lisible, reconstruction du pipeline final sans dépendre d'un modèle binaire, contrôle des hypothèses de ROI et vérification de reproduction des probabilités.

Des versions HTML sont incluses dans `lecture/` pour consulter les résultats sans Jupyter. Elles sont des copies de lecture ; les sources exécutables restent les fichiers `.ipynb` dans `notebooks/`.

Une procédure GitHub Actions est fournie pour exécuter les trois notebooks et les vérifications après dépôt. Elle n'a pas encore été lancée sur GitHub : les vérifications ont été réalisées localement.
