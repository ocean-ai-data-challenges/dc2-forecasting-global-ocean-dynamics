# Data

## Training data

Participants choose their training data and their inputs: there is no prescribed dataset, the
challenge is open to physics-based, statistical and machine-learning approaches. What is
**forbidden** is any data concomitant with the evaluated dates, in training, and any information
from after a forecast's initialization time, in its inputs: see the {doc}`rules`. There are
two tracks: 2023–2024, training data **up to 31 December 2022**, and 2024, training data up to
31 December 2023. Commonly used sources, up to those dates:

- The **GLORYS12** reanalysis (see below).
- Any [Copernicus Marine Service (CMEMS)](https://marine.copernicus.eu/) product.
- ECMWF atmospheric reanalyses (ERA5, ERA-Interim) for forcing a physical model.
- The archives of the satellite and Argo observation datasets described below.

---

## Evaluation data

Submitted forecasts are evaluated against the independent datasets described below.
Depending on the active profile, data are fetched either from a private Wasabi S3 bucket
(`ppr-ocean-climat`, profile `dc2_wasabi`) or from the public EDITO MinIO gateway
(`https://minio.dive.edito.eu`, bucket `project-ppr-ocean-ia`, profile `dc2_edito`). Files are
stored in Zarr (along-track L3, SWOT, drifters, GLORYS12, GloNet) or NetCDF (Argo velocities,
SSS); the Argo profiles are read from the Argo GDAC and the SST through the Copernicus Marine API
(a Copernicus Marine account is needed). The pipeline fetches them all.

The evaluation period is profile-dependent:

- `dc2_wasabi`: **1 January 2024 to 1 January 2025**
- `dc2_edito`: **1 January 2023 to 31 December 2024**

Temporal matching between forecasts and observations uses a tolerance of ±12 hours.

---

### GLORYS12 — Global Ocean Physics Reanalysis

> **Model reference** — gridded dataset used to evaluate all 3-D variables

| Feature | Value |
|---|---|
| **CMEMS identifier** | [`GLOBAL_MULTIYEAR_PHY_001_030`](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) |
| **Provider** | Copernicus Marine Service (CMEMS) / Mercator Ocean International |
| **Type** | Numerical reanalysis (NEMO model, Level 4) |
| **Horizontal resolution** | 1/12° (~8 km) regular grid; DC2's copy is on DC2's 0.25° grid |
| **Vertical levels** | 50 standard levels (surface → ~5 500 m); DC2's copy has 21 of them |
| **Storage (DC2)** | `DC2/ZARR/Glorys/<YYYYMMDD>.zarr`, daily, 2023–2024, the grid of the forecasts (672 × 1 440 × 21) |
| **Spatial coverage** | Global, 80 °S – 90 °N, 180 °W – 180 °E |
| **Temporal coverage** | 1 January 1993 → near-present |
| **Temporal resolution** | Daily and monthly means |
| **Atmospheric forcing** | ECMWF ERA-Interim (1993–2018) then ERA5 (2018–present) |
| **Data assimilation** | Reduced-order Kalman filter + 3D-VAR correction |
| **Assimilated observations** | Along-track altimetry (SLA), satellite SST, sea-ice concentration, in-situ T/S profiles |

**Variables evaluated in DC2:**

| CF standard name | Pipeline alias | Unit | Description |
|---|---|---|---|
| `sea_surface_height_above_geoid` | `ssh` | m | Sea surface height |
| `sea_water_potential_temperature` | `temperature` | °C | Potential temperature |
| `sea_water_salinity` | `salinity` | PSU | Salinity |
| `eastward_sea_water_velocity` | `u_current` | m s⁻¹ | Zonal current |
| `northward_sea_water_velocity` | `v_current` | m s⁻¹ | Meridional current |

**Limitations and quality:**
GLORYS12 does not resolve sub-mesoscale processes (< 8 km). Coastal areas and shallow seas
are less well represented. Near the poles, sea ice introduces additional uncertainties. SSH
accuracy is on the order of ~3–5 cm RMS in the open ocean.

---

### Nadir altimetry — DUACS L3 along-track (`nadir_l3`)

> **Satellite reference** — the sea surface height of all the nadir altimeters of 2023–2024,
> edited and corrected

| Feature | Value |
|---|---|
| **CMEMS product** | `SEALEVEL_GLO_PHY_L3_MY_008_062` (DUACS, delayed time, 1 Hz), version 202411, downloaded 2026-10-28 |
| **Missions** | SARAL/AltiKa, Jason-3 (interleaved orbit), Sentinel-3A, Sentinel-3B, Sentinel-6A, HY-2B, SWOT nadir (from July 2023) |
| **Variables** | `sla_filtered` (sea level anomaly, noise-filtered ~65 km), `mdt` (CNES-CLS mean dynamic topography) |
| **Evaluated as** | ADT = `sla_filtered` + `mdt` against the model's `zos`; `crmse` leads (the two reference surfaces differ by a constant) |
| **Storage** | `DC2/ZARR/L3/<mission>/<MISSION>_L3_<YYYYMMDD>.zarr`, built by `scripts/build_alongtrack_l3.py` |
| **Coverage 2023–2024** | days with data: SARAL 722, Jason-3 731, S3A 729, S3B 731, S6A 731, HY-2B 621 (ends 2024-12-20), SWOT nadir 525 (from 2023-07-21); 3.4 GB |
| **Consistency** | GloNet 2024-01-03, lead 0: bias −0.47 m and crmse 0.049–0.052 m for each of the seven missions |

The previous along-track references (`DC2/ZARR/Saral`, `Jason3`, `Sentinel`) are the missions' L2
files with no editing flag kept: a single day of SARAL holds thousands of land, ice and rain values
of tens of metres. They are no longer evaluated. The same L3 product is the along-track reference of
`oceanml3d-eval`, so both benchmarks score against the same points. The mission descriptions below
are those of the missions of `nadir_l3`.

### SARAL/AltiKa — Ka-band nadir altimeter

> **Satellite reference** — along-track sea surface height measurements

| Feature | Value |
|---|---|
| **Mission** | SARAL (*Satellite with ARgos and ALtika*) |
| **Partners** | ISRO (India) / CNES (France) |
| **Main instrument** | Ka-band radar altimeter (35 GHz, wavelength 0.86 cm) |
| **Auxiliary instruments** | Dual-frequency radiometer (23.8 and 37 GHz), DORIS, LRA |
| **Launch** | 25 February 2013 |
| **Orbit** | Near-polar, altitude ~800 km, inclination 98.55° |
| **Orbital cycle (phase 1)** | 35 days (same ground track as ERS/Envisat, until July 2016) |
| **Drifting phase (SARAL-DP)** | Since July 2016: drifting orbit, no fixed repeat track |
| **Spatial coverage** | Global (excluding very coastal areas < ~5 km) |
| **Along-track resolution** | ~7 km (finer than Ku-band thanks to Ka-band) |
| **Inter-track spacing** | ~900 km at the equator (repeat phase) |

**Evaluated in DC2:** the mission's points of `nadir_l3` (DUACS L3, `sla_filtered` + `mdt`),
see above.

**Context and quality:**
SARAL/AltiKa is the **first operational Ka-band altimeter** for oceanography. The Ka-band
offers: (1) a smaller ground footprint → better spatial resolution; (2) less ionospheric noise
than Ku-band; (3) better sensitivity to short-wavelength waves.
Limitations: higher sensitivity to rain (signal attenuation), and drifting orbit since 2016
(less uniform spatial coverage but greater inter-profile diversity).

---

### Jason-3 — Ku-band nadir altimeter

> **Satellite reference** — the reference altimetric record from 2016 to 2022, on an interleaved
> orbit since 2022

| Feature | Value |
|---|---|
| **Mission** | Jason-3 |
| **Partners** | CNES / NASA / EUMETSAT / NOAA (Copernicus program) |
| **Main instrument** | Poseidon-3B radar altimeter (Ku-band + C-band) |
| **Auxiliary instruments** | Advanced Microwave Radiometer (AMR), DORIS, GPSP, LRA |
| **Launch** | January 2016 |
| **Orbit** | Low Earth orbit, altitude 1 336 km, inclination 66° |
| **Orbital cycle** | 10 days; on the TOPEX/Poseidon–Jason ground track until April 2022, then on the interleaved track midway between (Sentinel-6A took over the reference track) |
| **Spatial coverage** | Global up to ±66° latitude |
| **Inter-track spacing** | ~315 km at the equator |
| **Along-track resolution** | ~7 km (20 Hz), ~300 m (experimental high resolution) |

**Evaluated in DC2:** the mission's points of `nadir_l3` (DUACS L3, `sla_filtered` + `mdt`),
see above.

**Context and quality:**
Jason-3 carried the **reference sea-level time series** dating back to TOPEX/Poseidon in 1992
until Sentinel-6A took it over (2022); over 2023–2024 its interleaved track doubles the coverage
of the reference one. It is the altimeter with the best-documented calibration and the most
temporally homogeneous data record. Its complementarity with SARAL/AltiKa (35-day vs 10-day
cycles, Ka vs Ku bands) enriches the altimetric coverage of the evaluation.
Limitation: inclination at 66° → no measurements beyond this latitude.

---

### Sentinel-3A/B, Sentinel-6A and HY-2B — Ku-band nadir altimeters

> **Satellite references** — four more nadir missions of `nadir_l3`, each scored with the others

| Mission | Partners | Instrument | Orbit |
|---|---|---|---|
| Sentinel-3A, Sentinel-3B | ESA / EUMETSAT (Copernicus) | SRAL, SAR-mode Ku/C-band | sun-synchronous, 27-day repeat, the two satellites 140° apart |
| Sentinel-6A "Michael Freilich" | EUMETSAT / ESA / NASA / NOAA / CNES (Copernicus) | Poseidon-4, SAR-mode Ku/C-band | the TOPEX/Poseidon–Jason reference track (10 days) since April 2022 |
| HY-2B | NSOAS (China) | Ku/C-band altimeter | sun-synchronous; its L3 ends on 2024-12-20 |

**Evaluated in DC2:** the missions' points of `nadir_l3` (DUACS L3, `sla_filtered` + `mdt`), see
above. Sentinel-6A carries the reference sea-level record since Jason-3 left that track.

---

### SWOT — Wide-swath radar interferometer

> **Satellite reference** — first mission providing 2-D SSH maps at mesoscale and
> sub-mesoscale resolution

| Feature | Value |
|---|---|
| **Mission** | SWOT (*Surface Water and Ocean Topography*) |
| **Partners** | NASA / CNES, with contributions from CSA (Canada) and UKSA (United Kingdom) |
| **Main instrument** | KaRIn (*Ka-band Radar Interferometer*) — wide swath |
| **Auxiliary instruments** | Nadir altimeter, AMR-C radiometer, DORIS, GPSP, LRA |
| **Launch** | 16 December 2022 (SpaceX Falcon 9) |
| **Operational orbit** | Altitude 891 km, inclination 77.6°, 21-day cycle |
| **Initial CalVal phase** | 1-day orbit (altitude 857 km), 6 months after launch |
| **Swath width** | 120 km (two 50 km beams separated by a ~20 km nadir gap) |
| **SSH spatial resolution** | ~1 km × 1 km (target: features ≥ 15 km) |
| **Spatial coverage** | ≥ 90 % of Earth's surface |
| **Mean revisit time** | ~11 days (21-day orbit with 120 km swath) |
| **Latitude coverage** | ±77.6° |

**Variable evaluated in DC2:**

| Variable | Description | Target accuracy |
|---|---|---|
| `ssha_filtered` | Filtered SSH anomaly (denoised L2 or L3 product) | ~1–2 cm RMS (mesoscale) |

**Context and quality:**
SWOT is the **first 2-D altimetric mission** capable of mapping SSH in two dimensions
(vs. a simple along-track profile for nadir altimeters). It resolves mesoscale structures
(50–500 km) and potentially sub-mesoscale structures (15–50 km), making it a particularly
demanding tool for assessing the spatial finesse of forecasts.
Limitations: 2024 products are still in the operational validation phase; interferometer noise
is significant below ~15 km; the nadir gap (~20 km) creates a blind strip at the centre of
each orbit.

---

### Argo floats — In-situ T/S profiles in the water column

> **In-situ reference** — the only observations of temperature and salinity **below the surface**

| Feature | Value |
|---|---|
| **Program** | International Argo (~30 participating countries) |
| **Instrument type** | Autonomous profiling CTD float (pressure, temperature, conductivity) |
| **Active fleet** | ~3 800 floats (2024) |
| **Parking depth** | ~1 000 m (between two profiles) |
| **Maximum profiling depth** | ~2 000 m (standard), up to 6 000 m (Deep Argo) |
| **Cycle period** | ~10 days per profile |
| **Spatial coverage** | Global, excluding shallow areas and under sea ice |
| **Vertical resolution** | Variable, typically ~2–10 m near the surface, ~25–50 m at depth |
| **Production** | ~13 000 profiles/month (> 400/day), since 2000 |
| **Real-time availability** | Within 12 hours of surfacing |
| **Delayed-mode availability** | QC-controlled data (DMQC) available within 1–2 years |

**Variables evaluated in DC2:**

| ARGO variable | CF standard name | Unit | Description |
|---|---|---|---|
| `TEMP` | `sea_water_temperature` | °C | In-situ temperature, converted to potential temperature (TEOS-10) before scoring |
| `PSAL` | `sea_water_salinity` | PSU | Practical salinity |
| `PRES` | `sea_water_pressure` | dbar | Pressure (depth proxy) |

Only `TEMP` and `PSAL` are used in metric computation; `PRES` is used for vertical positioning.

**Quality levels used:**
Argo's standard mode: the adjusted values of the delayed-mode and adjusted profiles (DMQC), the
real-time ones otherwise, with the flags 1 and 2 kept (details in {doc}`metrics`, section 3).
Matching with forecasts uses a temporal tolerance of ±12 hours.

**Context and quality:**
Argo is the **only global and systematic observation system for the water column**. It enables
3-D quality assessment of forecasts (temperature and salinity from the surface down to 2 000 m),
which is impossible with satellite data alone.
Limitations: non-uniform spatial coverage (undersampling at high latitudes and in the
Mediterranean Sea); no measurements under sea ice (standard floats); insufficient density
to resolve sub-mesoscale structures.

---

### Argo velocities — In-situ horizontal currents from float drift

> **In-situ reference** — active dataset (`argo_velocities`) evaluating the `u_current`/`v_current`
> model variables against float-drift-derived currents

Horizontal current components (`U`, `V`) are estimated from Argo float drift, produced by Copernicus
Marine Service as an in-situ velocity product (`INSITU_GLO_PHY_UV_DISCRETE_NRT_013_048`; the
delayed-mode `MY_013_044` stops in August 2024) and distributed here as a single NetCDF file
(~537 000 points, 2023–2024). Half of the points are the drift **at the surface** during the
float's transmission, pushed by the wind: `depth_range: [100, 3000]` keeps the **parking-depth**
drift (~1 000 m), and each point is scored against the model interpolated to its depth (reported
as `1000m`).

| Feature | Value |
|---|---|
| **Provider** | Copernicus Marine Service (in-situ TAC) |
| **Type** | Float-drift-derived horizontal velocity, single-depth (parking depth ~1 000 m) |
| **Coverage** | Global, 2023–2024 evaluation window |
| **Evaluated variables** | Eastward (`U`) and northward (`V`) velocity |

**Variables evaluated in DC2:**

| NetCDF variable | Pipeline alias | Description |
|---|---|---|
| `U` | `U` | Eastward current component |
| `V` | `V` | Northward current component |

Because every task loads the same full-size file, `argo_velocities` uses lower parallelism
(fewer concurrent workers, smaller batch size) than other datasets to bound memory usage.

### Drifters — daily 15 m surface currents (`drifters_15m`)

> **In-situ reference** — surface currents (`uo`, `vo` at 15 m)

| Feature | Value |
|---|---|
| **CMEMS dataset** | `cmems_obs-ins_glo_phy-cur_my_drifter_PT1H` (`INSITU_GLO_PHY_UV_DISCRETE_MY_013_044`, delayed mode), version 202511 (retired by CMEMS on 2026-11-24: the DC2 stores are its frozen copy), downloaded 2026-10-28 |
| **Selection** | drogued drifters (15 m; the undrogued ones, at 0.5 m, slip with the wind), `value_qc` 1 |
| **Processing** | hourly velocities averaged per drifter and day (at least 12 hours), stamped at 00:00 of the day: ~400 drifter-days a day, all 731 days of 2023–2024 (106 MB) |
| **Evaluated as** | the model interpolated to 15 m, reported as `15m` |
| **Storage** | `DC2/ZARR/Drifters15m/DRIFTERS15M_<YYYYMMDD>.zarr`, built by `scripts/build_drifters_15m.py` |

The "daily 15 m currents against drogued drifters" setting of `oceanml3d-eval`'s
`surface_currents_15m` benchmark.

---

## Quality of the references (2023–2024)

Checked against GloNet's forecast of 2024-01-03, lead time 0 (an error that includes the model's):

| Reference | Points | Bias | Centred error | 99th percentile of the deviation |
|---|---|---|---|---|
| SARAL L2 store (`ssha` + `mean_topography`, \|SLA\| < 2 m) | 17 133 | −0.31 m | 0.27 m | 1.4 m |
| SARAL DUACS L3 (`sla_filtered` + `mdt`) | 27 739 | −0.47 m | **0.046 m** | — |
| SWOT KaRIn L3 Expert v2.0.1 (`ssha_filtered` + `mdt`) | 6.2 M | −0.48 m | 0.053 m | 0.18 m |
| SST L3S (quality levels 4–5 only in the product) | 100 k | −0.17 °C | 0.50 °C | 1.7 °C |
| SSS SMOS L3, daily | 497 k | −0.02 | 0.45 | 1.5 |
| Drifters 15 m, daily | 401 | 0.00 / 0.01 m/s | 0.10 / 0.12 m/s (u / v) | — |

The SMOS daily salinity is noisy (its own error is ~0.3): the noise adds the same floor to every
model's score. Gaps that cannot be filled: SWOT has no data before March 2023 (January and February
2023 are missing), and the GloNet baseline exists for 2024 only.

---

## GloNet — Reference model (baseline)

GloNet (*Global Neural Ocean Forecasting System*) is the reference model against which all
submissions are compared. Developed by Mercator Ocean International within the PPR Océan &
Climat framework, it is a deep-learning-based ocean forecasting model.

| Feature | Value |
|---|---|
| **Provider** | Mercator Ocean International / PPR Océan & Climat |
| **Type** | Neural ocean state forecasting model |
| **Horizontal resolution** | 1/4° (regular global grid) |
| **Output variables** | `zos`, `thetao`, `so`, `uo`, `vo` (same as the predicted variables) |
| **Forecast horizon** | 10 days (lead times 0 to 9) |
| **Storage (DC2)** | Zarr, one store per forecast (`DC2/ZARR/Glonet/<YYYYMMDD>.zarr`, Wednesdays of 2024) on EDITO (and the legacy Wasabi bucket) |
| **Reference** | El Aouni et al. (2025), *JGR: Machine Learning and Computation* (see {doc}`references`) |

**Declaration** (rule 4 of the {doc}`rules`): 2024 track. GloNet's training had no data of
2024; its initial state at $t_0$ is the CMEMS real-time analysis
(`GLOBAL_ANALYSISFORECAST_PHY_001_024`). It is the reference of the leaderboard's 2024 view and
is not shown in the 2023–2024 one.

## GLO4ENS — Probabilistic baseline (`glo4ens`)

The global ensemble forecasts of Mercator Ocean (`source: MOI GLO4ENS`), a physical model
(NEMO, ORCA025, 75 levels), are DC2's probabilistic baseline, and the reference of the leaderboard's 2023–2024 view (the
2024 view compares it with GloNet over 2024).

| Feature | Value |
|---|---|
| **Provider** | Mercator Ocean International |
| **Type** | Ensemble of a numerical ocean model, 50 members |
| **Cycles** | weekly, on Thursdays; 3-D fields 2022-12-29 .. 2024-12-26 (104 cycles over 2023–2024) |
| **Forecast horizon** | 28 daily means (DC2 keeps `n_days_forecast`, 10) |
| **Source** | public bucket `MOISICEEF` at `https://s3.waw3-1.cloudferro.com` (anonymous read), native grid, int16 |
| **Put on DC2's grid** | `scripts/build_glo4ens.py`: `zos − ssh_ib` (the source's `zos` holds the inverse-barometer response, which the DUACS SLA and GLORYS do not); velocities from the C grid (U/V points, i/j axes) averaged to T points and rotated to east/north; linear in depth to the 21 levels; bilinear (xESMF, NaN-aware) to 0.25°; daily means stamped at 00:00 |
| **Storage (DC2)** | none: the forecasts stay in their bucket, only their scores are kept. `scripts/evaluate_glo4ens.py` puts one cycle at a time on DC2's grid in a temporary directory (~31 GB at 50 members and 10 days), scores it, then deletes it with the download caches |
| **Checks** | regridded positions within 5 km (p99) of the target grid; 0.5 % of GLORYS ocean points are land on its coastline. Cycle 2024-01-04, lead 0 (2 members): SSH crmse 0.045 m against the L3 along-track (0.12–0.14 m before removing `ssh_ib`) and 0.048 m against SWOT, bias −0.52 m (the reference-surface offset GloNet has too); parking-depth velocities rmse 0.087 / 0.076 m/s, ssr ~1; drifters 0.095 m/s, ssr 0.6–0.7; surface salinity 0.80 against GLORYS (GloNet 0.59), the difference in the Arctic, a model difference |

It is an operational forecast (initial state from the real-time analysis): it follows the data
rules, in the 2023–2024 track (shown in both views).

Scoring it (e.g. on Datarmor; the conversion needs xESMF, the evaluation dctools):

```bash
python scripts/evaluate_glo4ens.py --work /work/$USER/glo4ens --first 2024-01-01 --last 2024-12-31
```

(on Datarmor's Slurm cluster: `scripts/datarmor/glo4ens.sbatch`, see
`dc2/leaderboard_results/README.md`)

The work directory holds one converted cycle at a time, the reference catalogs and the scores
(`run/results/results_glo4ens.json`); an interrupted run resumes at the cycle it stopped at.
A full cycle reads ~150 GB from the bucket (10 lead days, 50 members): run it where the network
is fast. Its source holds a few unphysical values (sea water down to −7.6 °C on the Siberian
shelves, 9 120 points of a member's day under −2.5 °C), kept as they are.

---

### Sea Surface Temperature (SST_fields) — CMEMS L3S gridded SST

> **Satellite reference (gridded)** — evaluates the `sst` variable derived from GloNet's
> surface-level `thetao`

| Feature | Value |
|---|---|
| **Provider** | Copernicus Marine Service (CMEMS) |
| **CMEMS product** | `cmems_obs-sst_glo_phy_l3s_pir_P1D-m` |
| **Type** | Level-3 super-collated (L3S) gridded daily "foundation" SST |
| **Access** | Fetched directly via the `copernicusmarine` API (no local Zarr/NetCDF mirror) |
| **Evaluated variable** | `sea_surface_temperature` |

**Context:**
`sst` is not a native GloNet variable — it is derived by the pipeline's surface-alias transform
as the shallowest depth level of the 3-D `thetao` (temperature) field, so that GloNet's
full-depth predictions can be matched against this surface-only observation product.

---

### Sea Surface Salinity (SSS_fields) — SMOS L3 gridded SSS

> **Satellite reference (gridded)** — evaluates the `sss` variable derived from GloNet's
> surface-level `so`

| Feature | Value |
|---|---|
| **Provider** | SMOS L3G (CATDS/CPDC, product `MIR_CS3G09`) |
| **Type** | Level-3 daily gridded rain-corrected SSS, ~25 km horizontal resolution |
| **Access** | NetCDF files on the EDITO MinIO gateway (`DC2/SSS_fields`) |
| **Evaluated variable** | `Sea_Surface_Salinity_Rain_Corrected` |

**Context:**
Like SST_fields, `sss` is derived from GloNet's surface-level `so` (salinity) by the same
surface-alias transform, since SMOS SSS is a surface-only product with no depth dimension.
SMOS L3G files carry no time coordinate; observation dates are recovered from the file name.

---

## Dataset summary

| Dataset | Type | Evaluated dimension | Variables | Period |
|---|---|---|---|---|
| GLORYS12 | Gridded reanalysis | 3-D (21 levels, 0.25°) | SSH, T, S, U, V | 2023–2024 (product: 1993 → present) |
| Nadir altimetry L3 (7 missions) | Nadir satellites, edited | 2-D surface (along-track) | SLA + MDT | 2023–2024 |
| SWOT | Wide-swath satellite | 2-D surface (2-D grid) | Filtered SSHA + MDT | March 2023 – 2024 |
| Argo profiles | In-situ floats | 3-D (vertical profiles) | T, S | 2023–2024 (programme: 2000 → present) |
| Argo velocities | In-situ floats | Parking depth (~1 000 m) | U, V | 2023–2024 |
| Drifters 15 m | In-situ drifters (drogued) | 15 m, daily means | U, V | 2023–2024 |
| SST_fields | Gridded satellite (CMEMS L3S) | 2-D surface | SST | 2023–2024 |
| SSS_fields | Gridded satellite (SMOS L3G) | 2-D surface | SSS | 2023–2024 |

---

## Accessing the data with Python

The evaluation datasets of the default profile are on the public EDITO MinIO gateway
(`https://minio.dive.edito.eu`, bucket `project-ppr-ocean-ia`, anonymous read). The examples
below open a few of them for exploration and model development.

### GLORYS12 reanalysis

GLORYS12 is publicly available on the Copernicus Marine Service (a Copernicus Marine account is
needed); DC2's copy, on DC2's grid, is on EDITO (below).

```python
import copernicusmarine

ds = copernicusmarine.open_dataset(
    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
    variables=["zos", "thetao", "so", "uo", "vo"],
    minimum_longitude=-180, maximum_longitude=180,
    minimum_latitude=-78, maximum_latitude=90,
    start_datetime="2022-12-01",       # training: up to 2022-12-31 (rules)
    end_datetime="2022-12-10",
)
```

### DC2's stores on EDITO

```python
import s3fs
import xarray as xr

fs = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": "https://minio.dive.edito.eu"})
root = "project-ppr-ocean-ia/DC2/ZARR"

# GloNet's forecast of 2024-01-03: dims (time: 10, depth: 21, lat: 672, lon: 1440)
glonet = xr.open_zarr(fs.get_mapper(f"{root}/Glonet/20240103.zarr"))

# GLORYS12 on DC2's grid, 2024-01-03: dims (time: 1, depth: 21, lat: 672, lon: 1440)
glorys = xr.open_zarr(fs.get_mapper(f"{root}/Glorys/20240103.zarr"))

# One day of SARAL's L3 along-track points: sla_filtered, sla_unfiltered, mdt
saral = xr.open_zarr(fs.get_mapper(f"{root}/L3/alg/SARAL_L3_20240105.zarr"))

# The daily 15 m drifter velocities of a day
drifters = xr.open_zarr(fs.get_mapper(f"{root}/Drifters15m/DRIFTERS15M_20240103.zarr"))
```

### Argo profiles

The Argo profiles are read from the Argo GDAC, e.g. with
[argopy](https://argopy.readthedocs.io/):

```python
import argopy

ds = argopy.DataFetcher(mode="standard").region(
    [-60, -30, 20, 40, 0, 2000, "2022-12-01", "2022-12-31"]).to_xarray()
```

> **Tip:** The evaluation pipeline handles all data fetching, interpolation, and caching
> automatically. You only need to access the raw data if you want to explore it for
> model development or diagnostic purposes.
