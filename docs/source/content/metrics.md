# Métriques d'évaluation

Toutes les métriques sont calculées par la bibliothèque `dctools`
([`dctools.metrics`](https://github.com/ocean-ai-data-challenges/dc-tools)), qui s'appuie sur
le backend [OceanBench](https://github.com/jejjohnson/oceanbench) de Mercator Océan. La classe
orchestratrice est `DC2Evaluation` (`dc2/evaluation/evaluation.py`), héritée de
`BaseDCEvaluation` dans `dctools`.

---

## Pipeline d'évaluation

Pour chaque date d'initialisation de la période 2024-01-01 → 2025-01-01 (une prévision tous
les 7 jours, soit 52 cycles) :

1. Le modèle soumis est chargé ; ses champs sont interpolés spatialement et temporellement
   aux positions exactes de chaque jeu de référence à l'aide de **`pyinterp`** (interpolation
   bilinéaire).
2. La **fenêtre de correspondance temporelle** est de ±12 heures autour de chaque observation.
3. Les métriques sont calculées par variables, par niveau de profondeur et par délai de
   prévision (lead-time 0 à 9 jours).
4. Les résultats sont agrégés par date d'initialisation, puis publiés sur le leaderboard.

### Correspondance variables DC2 ↔ noms internes OceanBench

| Variable DC2 | Nom CF standard | Identifiant OceanBench |
|---|---|---|
| `zos` | `sea_surface_height_above_geoid` | `SEA_SURFACE_HEIGHT_ABOVE_GEOID` |
| `thetao` | `sea_water_potential_temperature` | `SEA_WATER_POTENTIAL_TEMPERATURE` |
| `so` | `sea_water_salinity` | `SEA_WATER_SALINITY` |
| `uo` | `eastward_sea_water_velocity` | `EASTWARD_SEA_WATER_VELOCITY` |
| `vo` | `northward_sea_water_velocity` | `NORTHWARD_SEA_WATER_VELOCITY` |

---

## Métriques par jeu de données de référence

Le tableau suivant résume les métriques assignées dans `dc2/config/dc2_wasabi.yaml` :

| Jeu de référence | Métrique(s) appliquée(s) | Variables évaluées |
|---|---|---|
| SARAL/AltiKa | `rmsd` | `zos` (SSH anomaly) |
| Jason-3 | `rmsd` | `zos` (SSH anomaly) |
| SWOT (KaRIn + nadir) | `rmsd` | `zos` (SSH filtrée) |
| Argo (profils `thetao`/`so`) | `rmsd` + `class4` | `thetao`, `so` |
| Argo (vélocités `uo`/`vo`) | `rmsd` | `uo`, `vo` |
| GLORYS12 (vérité terrain) | `rmsd` + `lagrangian` + `rmsd_geostrophic_currents` + `rmsd_mld` | `zos`, `thetao`, `so`, `uo`, `vo` |

---

## 1. RMSD — Écart quadratique moyen

La métrique centrale de DC2. Pour chaque paire *(prévision, observation)*, le champ prédit est
interpolé aux positions de l'observation ; l'RMSD est ensuite :

$$
\text{RMSD} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} \left( \hat{x}_i - x_i \right)^2}
$$

où $\hat{x}_i$ est la valeur prévue à la position $i$ et $x_i$ la valeur observée.

Deux variantes coexistent dans `dctools.metrics.oceanbench_metrics` :

| Cas | Fonction utilisée |
|---|---|
| Référence disponible (obs en temps réel) | `func_with_ref: rmsd` (oceanbench) |
| Pas de référence (comparaison GLORYS12) | `func_no_ref: rmsd_of_variables_compared_to_glorys` |

### Cartes spatiales d'RMSD par bins

En plus du score global, le pipeline calcule des **cartes d'RMSD par cellule** à la résolution
configurable (défaut `bin_resolution = 4°`). Ces cartes sont publiées sur le leaderboard sous
forme de visualisations interactives, permettant un diagnostic régional de l'erreur.

---

## 2. RMSD des courants géostrophiques

Les courants de surface géostrophiques $(u_g, v_g)$ sont dérivés du champ de SSH $\eta$ par
les relations d'équilibre géostrophique :

$$
u_g = -\frac{g}{f} \frac{\partial \eta}{\partial y}, \qquad
v_g = \frac{g}{f} \frac{\partial \eta}{\partial x}
$$

avec $g = 9.81\,\text{m s}^{-2}$ et $f = 2\Omega\sin\phi$ le paramètre de Coriolis.

Cette métrique est appliquée au jeu de référence GLORYS12. La fonction de pré-traitement
`preprocess_ref: add_geostrophic_currents` est appelée avant le calcul de l'RMSD
(`func_with_ref: rmsd`). Lorsque GLORYS12 n'est pas disponible comme référence directe,
`func_no_ref: rmsd_of_geostrophic_currents_compared_to_glorys` est utilisée.

> **Avantage** : cette métrique évalue la qualité du gradient de SSH indépendamment de tout
> décalage absolu d'altitude, ce qui la rend sensible à la mésoéchelle (tourbillons, fronts).

---

## 3. RMSD de la profondeur de la couche de mélange (MLD)

La profondeur de la couche de mélange est diagnostiquée depuis les profils de température et de
salinité prévus, via un critère de densité potentielle :

$$
\sigma_\theta(z_{\text{MLD}}) - \sigma_\theta(10\,\text{m}) = \Delta\sigma_\theta
= 0.03\,\text{kg m}^{-3}
$$

La fonction `preprocess_ref: add_mixed_layer_depth` effectue ce calcul avant l'évaluation
(`func_with_ref: rmsd`). En mode sans référence : `func_no_ref: rmsd_of_mixed_layer_depth_compared_to_glorys`.

Cette métrique teste la capacité du modèle à reproduire la stratification verticale de l'océan,
critique pour les échanges air-mer, la production primaire et les prévisions de cyclones.

---

## 4. Déviation de trajectoires lagrangiennes

Des particules virtuelles sont advectées par le champ de vitesse prévu $(u, v)$ sur tout
l'horizon de prévision. La déviation lagrangienne est définie comme l'écart spatial moyen
(en km) entre les positions des particules issues du modèle évalué et celles issues de la
référence GLORYS12 :

$$
\delta_L = \frac{1}{N_p} \sum_{p=1}^{N_p} \left\| \mathbf{r}_p^{\text{pred}}(T)
          - \mathbf{r}_p^{\text{ref}}(T) \right\|_2
$$

**Paramètres d'implémentation** (depuis `oceanbench_metrics.py`) :

| Paramètre | Valeur |
|---|---|
| Domaine spatial (`ZoneCoordinates`) | lat −90 → +90°, lon −180 → +180° (global) |
| Dimension requise | `depth` (la méthode ne s'applique qu'aux champs 3D de vitesse) |
| Fonction (avec référence) | `deviation_of_lagrangian_trajectories` |
| Fonction (sans référence) | `deviation_of_lagrangian_trajectories_compared_to_glorys` |

> **Applications** : recherche et sauvetage en mer, suivi de polluants, connectivité
> écologique, dérive de glace.

---

## 5. Score Class 4 (Argo in-situ)

La métrique **Class 4** est un standard de la prévision océanique opérationnelle (GODAE /
Copernicus Marine). Elle compare directement les profils prévus aux profils Argo in-situ
via le `Class4Evaluator` d'OceanBench :

```python
ClassEvaluator.run(model_ds, obs_ds, variables=["thetao", "so"])
```

Les profils Argo sont chargés depuis le catalogue S3 Wasabi et mâtchés en espace-temps avec
les prévisions (fenêtre : ±12 h, distance horizontale minimisée par `pyinterp`).

Les résultats distinguent :
- **biais** (erreur systématique) ;
- **RMSD non biaisé** (ubRMSD) ;
- décomposition par profondeur (0 – 2000 m).

---

## Agrégation et leaderboard

Les scores sont calculés pour chaque date d'initialisation, puis agrégés (moyenne ± écart-type)
sur toute la période 2024-2025. Le leaderboard publie :

- un score global par métrique et par variable ;
- des cartes spatiales d'RMSD par bin (résolution 4°×4°) pour chaque délai de prévision ;
- une décomposition par profondeur pour `thetao`, `so`, `uo`, `vo`.

Voir le [leaderboard](leaderboard.md) pour les scores courants et les cartes interactives.
