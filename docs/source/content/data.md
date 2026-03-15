# Data

## Données d'entraînement

Les participants sont **libres de choisir leurs données d'entraînement**. Il n'existe pas de jeu
de données imposé : le challenge est volontairement ouvert aux approches physiques, statistiques
et d'apprentissage automatique. À titre indicatif, les sources suivantes sont couramment utilisées :

- La réanalyse **GLORYS12** (voir ci-dessous) jusqu'au 1er janvier 2024.
- Tout produit du [Copernicus Marine Service (CMEMS)](https://marine.copernicus.eu/).
- Les réanalyses atmosphériques ECMWF (ERA5, ERA-Interim) pour forcer un modèle physique.
- Des sous-ensembles des données satellitaires et Argo décrites ci-dessous, antérieurs à la
  période d'évaluation.

---

## Données d'évaluation

Les prévisions soumises sont évaluées contre les jeux de données indépendants décrits ci-dessous.
Toutes les données sont stockées au format Zarr (ou NetCDF) dans un bucket S3 privé Wasabi
(`ppr-ocean-climat`) et téléchargées automatiquement par le pipeline d'évaluation.

La période d'évaluation couvre **du 1er janvier 2024 au 1er janvier 2025**.
La correspondance temporelle entre prévision et observation utilise une tolérance de ±12 heures.

---

### GLORYS12 — Réanalyse physique globale de l'océan

> **Référence modèle** — jeu de données gridded utilisé pour évaluer toutes les variables 3D

| Caractéristique | Valeur |
|---|---|
| **Identifiant CMEMS** | [`GLOBAL_MULTIYEAR_PHY_001_030`](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) |
| **Fournisseur** | Copernicus Marine Service (CMEMS) / Mercator Ocean International |
| **Type** | Réanalyse numérique (modèle NEMO, Niveau 4) |
| **Résolution horizontale** | 1/12° (~8 km) régulière |
| **Niveaux verticaux** | 50 niveaux standard (surface → ~5 500 m) |
| **Couverture spatiale** | Global, 80 °S – 90 °N, 180 °W – 180 °E |
| **Couverture temporelle** | 1 janvier 1993 → quasi-présent |
| **Résolution temporelle** | Moyenne journalière et mensuelle |
| **Forçage atmosphérique** | ECMWF ERA-Interim (1993–2018) puis ERA5 (2018–présent) |
| **Assimilation de données** | Filtre de Kalman d'ordre réduit + correction 3D-VAR |
| **Observations assimilées** | Altimétrietrack-by-track (SLA), SST satellite, concentration glace de mer, profils T/S in-situ |

**Variables évaluées dans DC2 :**

| Nom standard CF | Alias pipeline | Unité | Description |
|---|---|---|---|
| `sea_surface_height_above_geoid` | `ssh` | m | Hauteur de surface de la mer |
| `sea_water_potential_temperature` | `temperature` | °C | Température potentielle |
| `sea_water_salinity` | `salinity` | PSU | Salinité |
| `eastward_sea_water_velocity` | `u_current` | m s⁻¹ | Courant zonal |
| `northward_sea_water_velocity` | `v_current` | m s⁻¹ | Courant méridional |

**Limites et qualité :**
GLORYS12 ne résout pas les processus sous-méso-échelle (< 8 km). Les zones côtières et les mers
peu profondes sont moins bien représentées. Proche des pôles, la glace de mer introduit des
incertitudes supplémentaires. La précision de la SSH est de l'ordre de ~3–5 cm RMS en mer ouverte.

---

### SARAL/AltiKa — Altimètre nadir Ka-bande

> **Référence satellite** — mesures track-by-track de la hauteur de surface

| Caractéristique | Valeur |
|---|---|
| **Mission** | SARAL (*Satellite with ARgos and ALtika*) |
| **Partenaires** | ISRO (Inde) / CNES (France) |
| **Instrument principal** | Altimètre radar Ka-bande (35 GHz, longueur d'onde 0,86 cm) |
| **Instruments auxiliaires** | Radiomètre double-fréquence (23,8 et 37 GHz), DORIS, LRA |
| **Lancement** | 25 février 2013 |
| **Orbite** | Quasi-polaire, altitude ~800 km, inclinaison 98,55° |
| **Cycle orbital (phase 1)** | 35 jours (même trace que ERS/Envisat, jusqu'à juillet 2016) |
| **Phase drifting (SARAL-DP)** | Depuis juillet 2016 : orbite dérivante, sans trace répétée fixe |
| **Couverture spatiale** | Globale (hors zones très côtières < ~5 km) |
| **Résolution along-track** | ~7 km (meilleure qu'en Ku-bande grâce à la Ka-bande) |
| **Séparation inter-traces** | ~900 km à l'équateur (phase répétitive) |

**Variable évaluée dans DC2 :**

| Variable | Description | Précision typique |
|---|---|---|
| `ssha` | Anomalie de hauteur de surface de la mer (Sea Surface Height Anomaly) | ~2–3 cm RMS |

**Contexte et qualité :**
SARAL/AltiKa est le **premier altimètre Ka-bande opérationnel** pour l'océanographie. La bande Ka
offre : (1) une empreinte au sol plus petite → meilleure résolution spatiale ; (2) moins de bruit
ionosphérique qu'en Ku-bande ; (3) meilleure sensibilité aux vagues de courte longueur d'onde.
Limitations : sensibilité plus importante à la pluie (atténuation du signal), et orbite dérivante
depuis 2016 (couverture spatiale moins homogène mais plus grande diversité des inter-profils).

---

### Jason-3 — Altimètre nadir Ku-bande

> **Référence satellite** — continuité de la mesure altimétriques de référence depuis 1992

| Caractéristique | Valeur |
|---|---|
| **Mission** | Jason-3 |
| **Partenaires** | CNES / NASA / EUMETSAT / NOAA (programme Copernicus) |
| **Instrument principal** | Altimètre radar Poseidon-3B (Ku-bande + C-bande) |
| **Instruments auxiliaires** | Radiomètre micro-ondes avancé (AMR), DORIS, GPSP, LRA |
| **Lancement** | Janvier 2016 |
| **Orbite** | Basse orbite terrestre, altitude 1 336 km, inclinaison 66° |
| **Cycle orbital** | 10 jours (même trace que TOPEX/Poseidon, Jason-1, Jason-2) |
| **Couverture spatiale** | Globale jusqu'à ±66° de latitude |
| **Séparation inter-traces** | ~315 km à l'équateur |
| **Résolution along-track** | ~7 km (20 Hz), ~300 m (haute résolution expérimentale) |

**Variable évaluée dans DC2 :**

| Variable | Description | Précision typique |
|---|---|---|
| `data_01__ku__ssha` | Anomalie SSH (canal Ku, groupe `data_01`) | ~2–3 cm RMS |

**Contexte et qualité :**
Jason-3 assure la **continuité de la série temporelle de référence en niveau de mer** qui débute
avec TOPEX/Poseidon en 1992. C'est l'altimètre dont la calibration est la mieux documentée et
dont les données sont les plus homogènes temporellement. Sa complémentarité avec SARAL/AltiKa
(cycle de 35 j vs 10 j, bandes Ka vs Ku) enrichit la couverture altimétriques de l'évaluation.
Limitation : inclinaison à 66° → pas de mesure au-delà de cette latitude.

---

### SWOT — Interféromètre radar à fauchée large

> **Référence satellite** — première mission fournissant des cartes 2D de SSH à méso- et
> sous-méso-échelle

| Caractéristique | Valeur |
|---|---|
| **Mission** | SWOT (*Surface Water and Ocean Topography*) |
| **Partenaires** | NASA / CNES, contribution CSA (Canada) et UKSA (Royaume-Uni) |
| **Instrument principal** | KaRIn (*Ka-band Radar Interferometer*) — fauchée large |
| **Instruments auxiliaires** | Altimètre nadir, radiomètre AMR-C, DORIS, GPSP, LRA |
| **Lancement** | 16 décembre 2022 (lanceur SpaceX Falcon 9) |
| **Orbite opérationnelle** | Altitude 891 km, inclinaison 77,6°, cycle 21 jours |
| **Phase CalVal initiale** | Orbite 1 jour (altitude 857 km), 6 mois après lancement |
| **Largeur de fauchée** | 120 km (deux faisceaux de 50 km séparés par un nadir de ~20 km) |
| **Résolution spatiale SSH** | ~1 km × 1 km (objectif : structures ≥ 15 km) |
| **Couverture spatiale** | ≥ 90 % de la surface terrestre |
| **Temps de revisite moyen** | ~11 jours (orbite 21 j avec fauchée 120 km) |
| **Couverture en latitude** | ±77,6° |

**Variable évaluée dans DC2 :**

| Variable | Description | Précision cible |
|---|---|---|
| `ssha_filtered` | Anomalie SSH filtrée (produit L2 ou L3 débruitée) | ~1–2 cm RMS (méso-échelle) |

**Contexte et qualité :**
SWOT est la **première mission altimétriques 2D** capable de cartographier la SSH en 2 dimensions
(vs un simple profil along-track pour les altimètres nadir). Elle résout les structures de
méso-échelle (50–500 km) et potentiellement sous-méso-échelle (15–50 km). Cela en fait un outil
particulièrement exigeant pour évaluer la finesse spatiale des prévisions.
Limitations : les produits de 2024 sont encore en phase de validation opérationnelle ; le bruit
de l'interféromètre est significatif en dessous de ~15 km ; la nadir gap (~20 km) crée une bande
aveugle au centre de chaque orbite.

---

### Flotteurs Argo — Profils in-situ T/S dans la colonne d'eau

> **Référence in-situ** — seul jeu de données évaluant les prévisions **sous la surface** (3D)

| Caractéristique | Valeur |
|---|---|
| **Programme** | Argo international (~30 pays participants) |
| **Type d'instrument** | Flotteur profilant autonome CTD (pression, température, conductivité) |
| **Parc actif** | ~3 800 flotteurs (2024) |
| **Profondeur de parcage** | ~1 000 m (entre deux profils) |
| **Profondeur de plongée** | ~2 000 m (standard), jusqu'à 6 000 m (Deep Argo) |
| **Période de cycle** | ~10 jours par profil |
| **Couverture spatiale** | Globale, hors zones peu profondes et sous la banquise |
| **Résolution verticale** | Variable, typiquement ~2–10 m en surface, ~25–50 m en profondeur |
| **Production** | ~13 000 profils/mois (> 400/jour), depuis 2000 |
| **Disponibilité temps réel** | Dans les 12 heures suivant la remontée en surface |
| **Disponibilité différée** | Données QC retardées (DMQC) disponibles sous 1–2 ans |

**Variables évaluées dans DC2 :**

| Variable ARGO | Nom standard CF | Unité | Description |
|---|---|---|---|
| `TEMP` | `sea_water_potential_temperature` | °C | Température de l'eau |
| `PSAL` | `sea_water_salinity` | PSU | Salinité pratique |
| `PRES` | `sea_water_pressure` | dbar | Pression (proxy de la profondeur) |

Seules `TEMP` et `PSAL` entrent dans le calcul des métriques ; `PRES` sert à la localisation
verticale.

**Niveaux de qualité utilisés :**
Le pipeline utilise les données temps réel avec contrôles automatiques. Les données délayées
(DMQC, meilleure qualité) sont utilisées quand disponibles. La correspondance avec les prévisions
est faite avec une tolérance temporelle de ±12 heures.

**Contexte et qualité :**
Argo est le **seul système d'observation global et systématique de la colonne d'eau**. Il permet
d'évaluer la qualité 3D des prévisions (température et salinité de la surface jusqu'à 2 000 m),
ce qui est impossible avec les données satellitaires seules.
Limitations : couverture spatiale non-uniforme (sous-échantillonnage aux hautes latitudes et en
mer Méditerranée) ; pas de mesure sous la banquise (flotteurs standards) ; densité insuffisante
pour résoudre les structures sous-méso-échelle.

> **Vitesses Argo** : les composantes horizontales de courant (`U`, `V`) estimées à partir de la
> dérive des flotteurs pendant leur phase de parcage (~1 000 m) sont disponibles comme jeu de
> données supplémentaire (`argo_velocities`) mais **ne sont pas actives dans l'évaluation par
> défaut**.

---

## GloNet — Modèle de référence (baseline)

GloNet (*Global Neural Ocean Forecasting System*) est le modèle de référence contre lequel toutes
les soumissions sont comparées. Développé par Mercator Ocean International dans le cadre du
PPR Océan & Climat, c'est un modèle de prévision océanique basé sur l'apprentissage profond.

| Caractéristique | Valeur |
|---|---|
| **Fournisseur** | Mercator Ocean International / PPR Océan & Climat |
| **Type** | Modèle neuronal de prévision de l'état de l'océan |
| **Résolution horizontale** | 1/4° (grille régulière globale) |
| **Variables de sortie** | `zos`, `thetao`, `so`, `uo`, `vo` (identiques aux variables prévues) |
| **Horizon de prévision** | 10 jours (lead times 0 à 9) |
| **Stockage (DC2)** | Format Zarr sur Wasabi S3 (`DC2/ZARR/Glonet`) |

Les scores GloNet constituent le **plancher de référence** du leaderboard : un modèle concurrent
représente un progrès s'il dépasse GloNet sur au moins une des métriques d'évaluation.

---

## Résumé des jeux de données

| Jeu de données | Type | Dimension évaluée | Variables | Période |
|---|---|---|---|---|
| GLORYS12 | Réanalyse grillée | 3D (colonne entière) | SSH, T, S, U, V | 1993 → présent |
| SARAL/AltiKa | Satellite nadir | 2D surface (along-track) | SSHA | 2013 → présent |
| Jason-3 | Satellite nadir | 2D surface (along-track) | SSHA | 2016 → présent |
| SWOT | Satellite fauchée large | 2D surface (grille 2D) | SSHA filtrée | 2022 → présent |
| Argo profils | In-situ flotteurs | 3D (profils verticaux) | T, S | 2000 → présent |
| Argo vitesses | In-situ flotteurs | 3D (parcage ~1 000 m) | U, V | 2000 → présent (inactif) |

- The **GLORYS12** global ocean reanalysis (1993–present) — the same product used as one of the
  evaluation references (see below).
- Any Copernicus Marine Service (CMEMS) product, ERA5/ECMWF atmospheric re-analyses, or
  satellite-derived gridded analyses.
- Subsets of the **Argo** and altimetric observation datasets described below up to the start of
  the evaluation period (before 1 January 2024).

There is no prescribed training set: the challenge is intentionally open to both physics-based,
statistical, and deep-learning approaches.

## Evaluation data

Forecasts are evaluated against the following independent datasets. All datasets are stored in
Zarr or NetCDF format on a private Wasabi S3 bucket (`ppr-ocean-climat`) and are fetched
automatically by the evaluation pipeline.

### GLORYS12 — Global Ocean Physics Reanalysis

| | |
|---|---|
| **Provider** | Copernicus Marine Service (CMEMS) / Mercator Ocean International |
| **Type** | Gridded model reanalysis (0.083 °, 50 levels) |
| **Coverage** | Global, 80 °S – 90 °N, 1993 – present, daily |
| **Evaluated variables** | `salinity`, `ssh`, `temperature`, `u_current`, `v_current` |
| **CMEMS product** | [`GLOBAL_MULTIYEAR_PHY_001_030`](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) |

GLORYS12 assimilates along-track altimeter SSH, satellite SST, sea-ice concentration, and
in-situ T/S profiles via a reduced-order Kalman filter. It provides a physically consistent
gridded reference for all five predicted variables across the full water column.

### SARAL/AltiKa — Ka-band nadir altimeter

| | |
|---|---|
| **Provider** | CNES / ISRO |
| **Type** | Nadir radar altimeter (Ka-band, 35 GHz) |
| **Coverage** | Global, launched February 2013; 35-day exact-repeat then drifting orbit |
| **Evaluated variable** | Sea surface height anomaly (`ssha`) |
| **Documentation** | [AVISO SARAL page](https://www.aviso.altimetry.fr/en/missions/current-missions/saral.html) |

The first Ka-band oceanographic altimeter. Ka-band offers finer along-track spatial resolution
and lower ionospheric noise than Ku-band instruments.

### Jason-3 — Ku-band nadir altimeter

| | |
|---|---|
| **Provider** | CNES / NASA / EUMETSAT / NOAA |
| **Type** | Nadir radar altimeter (Ku/C-band, Poseidon-3B) |
| **Coverage** | Global, operational since January 2016; 10-day exact-repeat orbit |
| **Evaluated variable** | Sea surface height anomaly (`data_01__ku__ssha`) |
| **Documentation** | [EUMETSAT Jason-3 page](https://www.eumetsat.int/jason-3) |

Jason-3 continues the long-term sea-level record dating back to TOPEX/Poseidon (1992).
Its 10-day repeat orbit provides complementary along-track SSH tracks to SARAL.

### SWOT — Wide-swath radar interferometer

| | |
|---|---|
| **Provider** | NASA / CNES |
| **Type** | Ka-band Radar Interferometer (KaRIn) — wide-swath SSH |
| **Coverage** | Global (≥ 90 % of Earth); 120 km swath; launched December 2022; 21-day repeat orbit |
| **Evaluated variable** | Filtered SSH anomaly (`ssha_filtered`) |
| **Documentation** | [SWOT Mission overview](https://swot.jpl.nasa.gov/mission/overview/) |

Unlike nadir altimeters, SWOT produces 2-D SSH maps at 1 km × 1 km resolution,
resolving mesoscale and sub-mesoscale ocean features (≥ 15 km).

### Argo — Global array of profiling floats

| | |
|---|---|
| **Provider** | International Argo programme (~30 nations) |
| **Type** | Autonomous profiling CTD floats |
| **Coverage** | Global, ~3 800 active floats; ~1 profile per float per 10 days |
| **Evaluated variables** | Temperature (`TEMP`), salinity (`PSAL`) profiles to 2 000 m |
| **Documentation** | [argo.ucsd.edu](https://argo.ucsd.edu/about/) |

Argo floats drift at ~1 000 m parking depth before profiling from 2 000 m to the surface.
They provide the only systematic in-situ sampling of the global subsurface ocean, making them
essential for evaluating the 3-D skill of forecasts below the surface.

> **Argo velocities** (horizontal current `U`, `V` from surface drift) are also available as an
> additional reference dataset but are not active in the default evaluation run.
