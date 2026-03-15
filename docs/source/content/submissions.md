# Soumettre un modèle

Cette page décrit la procédure complète pour formater une prévision, la valider localement et
soumettre les résultats à DC2.

---

## 1. Prérequis et installation

Cloner le dépôt et installer le package en mode "*editable*" :

```bash
git clone https://github.com/ppr-ocean-ia/dc2-forecasting-global-ocean-dynamics.git
cd dc2-forecasting-global-ocean-dynamics
pip install -e .
```

L'installation fournit la commande CLI `dc-submit` (également invocable via
`python -m dc.submit`).

---

## 2. Format de soumission requis

### 2.1 Grille DC2

Toute prévision doit être fournie sur la grille globale DC2 :

| Dimension | Valeurs | Nb. points |
|---|---|---|
| `lat` | −78 to +90 °, pas 0.25 ° | 672 |
| `lon` | −180 to +180 °, pas 0.25 ° | 1 440 |
| `depth` (niveaux) | 0.494 / 47.374 / 92.327 / 155.851 / 222.475 / 318.127 / 380.213 / 453.938 / 541.089 / 643.567 / 763.333 / 902.339 / 1 245.292 / 1 684.284 / 2 225.078 / 3 220.820 / 3 597.032 / 3 992.484 / 4 405.225 / 4 833.291 / 5 274.784 m | 21 |
| `lead_time` | 0, 1, 2, …, 9 (jours après l'initialisation) | 10 |

### 2.2 Variables requises

| Variable | Dimensions | Shape | Unité | Description |
|---|---|---|---|---|
| `zos` | `(time, lat, lon)` | (10, 672, 1 440) | m | Hauteur de surface de la mer |
| `thetao` | `(time, depth, lat, lon)` | (10, 21, 672, 1 440) | °C | Température potentielle |
| `so` | `(time, depth, lat, lon)` | (10, 21, 672, 1 440) | PSU | Salinité |
| `uo` | `(time, depth, lat, lon)` | (10, 21, 672, 1 440) | m s⁻¹ | Courant zonal |
| `vo` | `(time, depth, lat, lon)` | (10, 21, 672, 1 440) | m s⁻¹ | Courant méridional |

> La dimension `time` encode les **dates valides** (date d'initialisation + lead-time), pas les
> indices. Les métadonnées CF (`units`, `long_name`) sont obligatoires pour chaque coordonnée.

### 2.3 Noms de variables acceptés (alias)

Le pipeline de validation accepte les alias courants :

| Coordonnée | Noms acceptés |
|---|---|
| latitude | `lat`, `latitude` |
| longitude | `lon`, `longitude` |
| profondeur | `depth`, `lev` |
| temps | `time` |
| SSH | `zos`, `ssh`, `ssha` |
| Température | `thetao`, `temperature` |
| Salinité | `so`, `salinity` |
| Courant zonal | `uo`, `u` |
| Courant méridional | `vo`, `v` |

---

## 3. Formats de fichiers acceptés

La commande `dc-submit info` liste les formats supportés. Quatre layouts sont reconnus :

| Layout | Description | Exemple |
|---|---|---|
| **A** — dossier de Zarr par date | *Recommandé.* Un `.zarr` par date d'initialisation dans un dossier | `model/20240103.zarr`, `model/20240110.zarr`, … |
| **B** — `zarr` unique | Un seul store Zarr couvrant toute la période | `model/all_forecasts.zarr` |
| **C** — fichier NetCDF unique | Un seul fichier `.nc` ou `.nc4` | `model/forecasts.nc` |
| **D** — glob de NetCDF | Tout chemin accepté par `glob` | `/data/model/*.nc` |

Le layout A est recommandé pour les grandes soumissions car il permet le chargement paresseux
par Dask et une meilleure tolérance aux erreurs.

### Structure du layout A (dossier de Zarr par date)

```
my_model/
    2024-01-03.zarr
    2024-01-10.zarr
    2024-01-17.zarr
    ...
    2024-12-25.zarr
```

---

## 4. Générer une soumission d'exemple

Le script `scripts/create_sample_submission.py` crée un jeu de données compliant rempli de
bruit aléatoire, utile pour tester le pipeline avant d'avoir un vrai modèle :

```bash
python scripts/create_sample_submission.py \
    --output /tmp/sample_model \
    --variables zos thetao so uo vo \
    --seed 42
```

Ce script génère les 52 fichiers Zarr correspondant à la période d'évaluation 2024-01-01 →
2025-01-01 (un par semaine). Chaque store respecte la grille DC2 décrite ci-dessus.

---

## 5. Valider la soumission

Avant d'exécuter toute l'évaluation, vérifier localement que le format est correct :

```bash
dc-submit validate <data_path> --model-name <NOM_DU_MODELE> [options]
```

### Options de validation

| Option | Description |
|---|---|
| `--model-name NAME` | Identifiant du modèle *(obligatoire)* |
| `--quick` | Valide uniquement les premières dates (test rapide) |
| `--save-report PATH` | Enregistre le rapport de validation dans un fichier JSON |
| `--max-nan-fraction F` | Fraction maximale de NaN tolérée (défaut : `0.10`, soit 10 %) |
| `--variables V [V …]` | Restreindre la vérification à certaines variables |
| `--config {dc2,…}` | Profil de configuration (défaut : `dc2`) |

### Ce que la validation vérifie

1. **Présence des variables** : `zos`, `thetao`, `so`, `uo`, `vo` (ou sous-ensemble si
   `--variables` est précisé).
2. **Conformité de la grille** : lat, lon, depth et lead_time correspondent à la spécification
   DC2.
3. **Fraction de NaN** : aucune variable ne dépasse `max_nan_fraction` (10 % par défaut).
4. **Couverture temporelle** : les dates d'initialisation attendues sont présentes.
5. **Types et unités** : les tableaux sont en virgule flottante et les unités CF sont
   renseignées.

---

## 6. Lancer l'évaluation complète

```bash
dc-submit run <data_path> --model-name <NOM_DU_MODELE> [options]
```

### Options d'exécution

| Option | Description |
|---|---|
| `-d DIR`, `--data-directory DIR` | Répertoire de sortie pour les résultats et les catalogues |
| `--force` | Écrase des résultats existants sans confirmation |
| `--skip-validation` | Saute la validation initiale (déconseillé) |
| `--quick-validation` | Lance une validation rapide avant l'évaluation |
| `--description TEXT` | Description courte du modèle |
| `--team TEXT` | Nom de l'équipe |
| `--email TEXT` | Contact |
| `--url TEXT` | URL du modèle (article, code, …) |

### Étapes du pipeline

1. **Téléchargement des catalogues** : les catalogues d'observations (SARAL, Jason-3, SWOT,
   Argo, GLORYS12) sont téléchargés depuis le bucket S3 Wasabi de DC2.
2. **Interpolation** : les champs de la prévision sont interpolés spatialement et temporellement
   aux positions de chaque jeu de référence (`pyinterp`, bilinéaire, fenêtre ±12 h).
3. **Calcul des métriques** : RMSD, RMSD courants géostrophiques, RMSD MLD, déviation
   lagrangienne, score Class 4 (voir [métriques](metrics.md)).
4. **Sortie** : les résultats sont écrits dans `<data_directory>/results/results_<NOM>.json`.
5. **Leaderboard** : les pages HTML du leaderboard sont reconstruites dans
   `<data_directory>/leaderboard/`.

---

## 7. Inspecter la spécification

La sous-commande `dc-submit info` affiche la configuration complète (grille, variables, métriques,
formats acceptés) sans effectuer aucune évaluation :

```bash
dc-submit info --config dc2
```

---

## 8. Participer au leaderboard public

Pour apparaître sur le leaderboard officiel, contactez les organisateurs de DC2 en fournissant :

- le fichier `results_<NOM>.json` généré par `dc-submit run` ;
- une brève description du modèle et des données d'entraînement utilisées ;
- une référence (article, préimpression, dépôt GitHub).

> **Note** : le module `dctools.submission` (backend de soumission distant) est en cours de
> développement. La procédure actuelle passe par le lancement local de `dc-submit run` et
> l'envoi manuel des résultats aux organisateurs. Ouvrez une
> [issue GitHub](https://github.com/ppr-ocean-ia/dc2-forecasting-global-ocean-dynamics/issues)
> pour toute question sur la soumission.

Voir aussi [`dc2/submit.py`](https://github.com/ppr-ocean-ia/dc2-forecasting-global-ocean-dynamics/blob/main/dc2/submit.py)
pour le code complet de l'interface CLI.
