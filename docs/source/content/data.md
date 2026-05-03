# Data

## Training data

Participants are **free to choose their training data**. There is no prescribed dataset: the
challenge is intentionally open to physics-based, statistical, and machine-learning approaches.
The following sources are commonly used:

- The **GLORYS12** reanalysis (see below) up to 1 January 2024.
- Any [Copernicus Marine Service (CMEMS)](https://marine.copernicus.eu/) product.
- ECMWF atmospheric reanalyses (ERA5, ERA-Interim) for forcing a physical model.
- Subsets of the satellite and Argo observation datasets described below, prior to the
  evaluation period.

---

## Evaluation data

Submitted forecasts are evaluated against the independent datasets described below.
All data are stored in Zarr (or NetCDF) format on a private Wasabi S3 bucket
(`ppr-ocean-climat`) and are fetched automatically by the evaluation pipeline.

The evaluation period covers **1 January 2024 to 1 January 2025**.
Temporal matching between forecasts and observations uses a tolerance of ±12 hours.

---

### GLORYS12 — Global Ocean Physics Reanalysis

> **Model reference** — gridded dataset used to evaluate all 3-D variables

| Feature | Value |
|---|---|
| **CMEMS identifier** | [`GLOBAL_MULTIYEAR_PHY_001_030`](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) |
| **Provider** | Copernicus Marine Service (CMEMS) / Mercator Ocean International |
| **Type** | Numerical reanalysis (NEMO model, Level 4) |
| **Horizontal resolution** | 1/12° (~8 km) regular grid |
| **Vertical levels** | 50 standard levels (surface → ~5 500 m) |
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

**Variable evaluated in DC2:**

| Variable | Description | Typical accuracy |
|---|---|---|
| `ssha` | Sea Surface Height Anomaly | ~2–3 cm RMS |

**Context and quality:**
SARAL/AltiKa is the **first operational Ka-band altimeter** for oceanography. The Ka-band
offers: (1) a smaller ground footprint → better spatial resolution; (2) less ionospheric noise
than Ku-band; (3) better sensitivity to short-wavelength waves.
Limitations: higher sensitivity to rain (signal attenuation), and drifting orbit since 2016
(less uniform spatial coverage but greater inter-profile diversity).

---

### Jason-3 — Ku-band nadir altimeter

> **Satellite reference** — continuity of the reference altimetric record since 1992

| Feature | Value |
|---|---|
| **Mission** | Jason-3 |
| **Partners** | CNES / NASA / EUMETSAT / NOAA (Copernicus program) |
| **Main instrument** | Poseidon-3B radar altimeter (Ku-band + C-band) |
| **Auxiliary instruments** | Advanced Microwave Radiometer (AMR), DORIS, GPSP, LRA |
| **Launch** | January 2016 |
| **Orbit** | Low Earth orbit, altitude 1 336 km, inclination 66° |
| **Orbital cycle** | 10 days (same ground track as TOPEX/Poseidon, Jason-1, Jason-2) |
| **Spatial coverage** | Global up to ±66° latitude |
| **Inter-track spacing** | ~315 km at the equator |
| **Along-track resolution** | ~7 km (20 Hz), ~300 m (experimental high resolution) |

**Variable evaluated in DC2:**

| Variable | Description | Typical accuracy |
|---|---|---|
| `data_01__ku__ssha` | SSH anomaly (Ku channel, `data_01` group) | ~2–3 cm RMS |

**Context and quality:**
Jason-3 ensures the **continuity of the reference sea-level time series** dating back to
TOPEX/Poseidon in 1992. It is the altimeter with the best-documented calibration and the most
temporally homogeneous data record. Its complementarity with SARAL/AltiKa (35-day vs 10-day
cycles, Ka vs Ku bands) enriches the altimetric coverage of the evaluation.
Limitation: inclination at 66° → no measurements beyond this latitude.

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

> **In-situ reference** — the only dataset evaluating forecasts **below the surface** (3-D)

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
| `TEMP` | `sea_water_potential_temperature` | °C | Water temperature |
| `PSAL` | `sea_water_salinity` | PSU | Practical salinity |
| `PRES` | `sea_water_pressure` | dbar | Pressure (depth proxy) |

Only `TEMP` and `PSAL` are used in metric computation; `PRES` is used for vertical positioning.

**Quality levels used:**
The pipeline uses real-time data with automatic quality controls. Delayed-mode data (DMQC,
higher quality) are used when available. Matching with forecasts uses a temporal tolerance
of ±12 hours.

**Context and quality:**
Argo is the **only global and systematic observation system for the water column**. It enables
3-D quality assessment of forecasts (temperature and salinity from the surface down to 2 000 m),
which is impossible with satellite data alone.
Limitations: non-uniform spatial coverage (undersampling at high latitudes and in the
Mediterranean Sea); no measurements under sea ice (standard floats); insufficient density
to resolve sub-mesoscale structures.

> **Argo velocities**: horizontal current components (`U`, `V`) estimated from float drift
> during their parking phase (~1 000 m) are available as an additional dataset
> (`argo_velocities`) but **are not active in the default evaluation run**.

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
| **Storage (DC2)** | Zarr format on Wasabi S3 (`DC2/ZARR/Glonet`) |

GloNet scores serve as the **baseline floor** on the leaderboard: a competing model represents
progress if it outperforms GloNet on at least one of the evaluation metrics.

---

## Dataset summary

| Dataset | Type | Evaluated dimension | Variables | Period |
|---|---|---|---|---|
| GLORYS12 | Gridded reanalysis | 3-D (full water column) | SSH, T, S, U, V | 1993 → present |
| SARAL/AltiKa | Nadir satellite | 2-D surface (along-track) | SSHA | 2013 → present |
| Jason-3 | Nadir satellite | 2-D surface (along-track) | SSHA | 2016 → present |
| SWOT | Wide-swath satellite | 2-D surface (2-D grid) | Filtered SSHA | 2022 → present |
| Argo profiles | In-situ floats | 3-D (vertical profiles) | T, S | 2000 → present |

---

## Accessing the data with Python

All evaluation datasets are stored as Zarr on a Wasabi S3 bucket and fetched automatically
by the evaluation pipeline. The examples below show how to open them interactively for
exploration and model development.

### GLORYS12 reanalysis

GLORYS12 is publicly available on the Copernicus Marine Service:

```python
import xarray as xr

# Via Copernicus Marine (requires copernicusmarine credentials)
import copernicusmarine
ds = copernicusmarine.open_dataset(
    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
    variables=["zos", "thetao", "so", "uo", "vo"],
    minimum_longitude=-180, maximum_longitude=180,
    minimum_latitude=-78, maximum_latitude=90,
    start_datetime="2024-01-01",
    end_datetime="2024-01-10",
)
print(ds)
# <xarray.Dataset>
# Dimensions:  (time: 10, depth: 50, latitude: 2041, longitude: 4320)
# Coordinates:
#   * time       (time)  datetime64[ns] …
#   * depth      (depth) float32         0.494 … 5275.0
#   * latitude   (latitude)  float32     -80.0 … 90.0
#   * longitude  (longitude) float32     -180.0 … 180.0
# Data variables:
#     zos        (time, latitude, longitude)        float32 …
#     thetao     (time, depth, latitude, longitude) float32 …
#     so         (time, depth, latitude, longitude) float32 …
#     uo         (time, depth, latitude, longitude) float32 …
#     vo         (time, depth, latitude, longitude) float32 …
```

### GloNet baseline (DC2 Wasabi S3)

```python
import s3fs
import xarray as xr

fs = s3fs.S3FileSystem(
    key="A10PTBP92GOXQW8OPE8G",
    secret="1C1Iscst7rfDnXp5eM8DtL7AkFiUAycINzctMEGQ",
    client_kwargs={"endpoint_url": "https://s3.eu-west-2.wasabisys.com"},
)

# Open a single forecast date
store = s3fs.S3Map(root="ppr-ocean-climat/DC2/ZARR/Glonet/2024-01-03.zarr", s3=fs)
ds = xr.open_zarr(store)
print(ds)
# <xarray.Dataset>
# Dimensions:  (time: 10, depth: 21, lat: 672, lon: 1440)
# Data variables:
#     zos     (time, lat, lon)              float32 …
#     thetao  (time, depth, lat, lon)       float32 …
#     so      (time, depth, lat, lon)       float32 …
#     uo      (time, depth, lat, lon)       float32 …
#     vo      (time, depth, lat, lon)       float32 …
```

### Satellite altimetry (SARAL example)

```python
store = s3fs.S3Map(
    root="ppr-ocean-climat/DC2/ZARR/Saral/2024/001.zarr", s3=fs
)
ds_saral = xr.open_zarr(store)
print(ds_saral)
# <xarray.Dataset>  — along-track observations
# Coordinates: time, lat, lon
# Data variables: ssha, mean_topography, …
```

### Argo profiles

```python
store = s3fs.S3Map(
    root="ppr-ocean-climat/DC2/ZARR/Argo/2024/argo_2024_01.zarr", s3=fs
)
ds_argo = xr.open_zarr(store)
print(ds_argo)
# <xarray.Dataset>
# Dimensions:  (N_PROF: …, N_LEVELS: …)
# Data variables: TEMP, PSAL, PRES, LATITUDE, LONGITUDE, TIME, …
```

> **Tip:** The evaluation pipeline handles all data fetching, interpolation, and caching
> automatically. You only need to access the raw data if you want to explore it for
> model development or diagnostic purposes.
| Argo velocities | In-situ floats | 3-D (parking ~1 000 m) | U, V | 2000 → present (inactive) |

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
| **Provider** | International Argo program (~30 nations) |
| **Type** | Autonomous profiling CTD floats |
| **Coverage** | Global, ~3 800 active floats; ~1 profile per float per 10 days |
| **Evaluated variables** | Temperature (`TEMP`), salinity (`PSAL`) profiles to 2 000 m |
| **Documentation** | [argo.ucsd.edu](https://argo.ucsd.edu/about/) |

Argo floats drift at ~1 000 m parking depth before profiling from 2 000 m to the surface.
They provide the only systematic in-situ sampling of the global subsurface ocean, making them
essential for evaluating the 3-D skill of forecasts below the surface.

> **Argo velocities** (horizontal current `U`, `V` from surface drift) are also available as an
> additional reference dataset but are not active in the default evaluation run.
