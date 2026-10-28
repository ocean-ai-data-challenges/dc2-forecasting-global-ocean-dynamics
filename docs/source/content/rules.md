# Data rules

A forecast is scored against what the ocean did after it was issued. Any information about
that future, in the inputs or in what the model learnt, turns the forecast into a reconstruction
and its score into an overestimate. These rules define which data a DC2 submission may use. They
follow the practice of forecast verification in numerical weather prediction (WMO verification
standards; WeatherBench 2, Rasp et al. 2024) and of the GODAE OceanView Class 4 intercomparison
of ocean forecasting systems (Ryan et al. 2015), where a forecast is verified against
observations it has not seen.

The key words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119.

## Definitions

- **Evaluation period** $P$: 1 January 2023 to 31 December 2024 (the `start_time` /
  `end_time` of the configuration).
- **Initialization time** $t_0$: the date of a forecast (one every `n_days_interval` days in $P$).
  Its **lead times** are $t_0, t_0 + 1\,\text{d}, \ldots, t_0 + (H - 1)\,\text{d}$, with
  $H$ = `n_days_forecast`. Its **verification window** is $[t_0, t_0 + H\,\text{d}]$.
- **Valid time** of a datum: the time it describes (measurement time of an observation, date of
  a field), not the time it was produced or published.
- **Concomitant data**: any datum whose valid time falls in $P$ (or in $[P_{start}, P_{end} + H]$,
  the last forecasts' verification windows) and that describes the ocean, the sea ice or the
  atmosphere above them. Such data are correlated with the predicted variables at the evaluated
  dates, whether they measure them directly or not.

## Rule 1 — Training uses no concomitant data

Training, validation, hyper-parameter tuning, model selection, bias correction and the
computation of any statistic the model uses (normalisation, climatology, error covariance) MUST
NOT use concomitant data, of any kind and from any source:

- observations (altimetry, SST, SSS, Argo, drifters, moorings, gliders, ...), including those that
  are not DC2 references;
- gridded observation products (L3/L4: DUACS, OSTIA, ARMOR3D, SMOS/SMAP L4, ...);
- reanalyses and analyses (GLORYS12, ORAS5, ERA5, ...) and model simulations forced by, or
  nudged to, any of the above over $P$.

Everything with a valid time **before 1 January 2023** MAY be used for training: GLORYS12 and
ERA5 up to 31 December 2022, the satellite and Argo archives, simulations such as eORCA36.

A model's parameters are therefore fixed before it sees any day of $P$. Learning online during
$P$ (updating the model forecast after forecast) is a different setting: it MAY be submitted to a
separate, declared track, where the update at $t_0$ uses only the inputs allowed by Rule 2.

## Rule 2 — A forecast uses what was known at its initialization time

The inputs of the forecast initialized at $t_0$ MUST be data that existed at $t_0$ in real time:

- observations with a valid time **at or before** $t_0$, at any processing level (near-real-time
  or delayed-time reprocessing of the same measurements);
- **real-time analyses** valid at or before $t_0$, produced by a system that had no data after
  $t_0$: the CMEMS analysis-and-forecast products (e.g. `GLOBAL_ANALYSISFORECAST_PHY_001_024`,
  the initial state of the GloNet baseline), near-real-time L4 products;
- **atmospheric forecasts issued at or before** $t_0$ (e.g. archived ECMWF IFS or AIFS forecasts),
  up to the forecast horizon.

They MUST NOT include:

- any datum with a valid time after $t_0$ (observations, analyses, atmospheric analyses of the
  verification window: ERA5 over $[t_0, t_0 + H]$ is the future weather, not a forecast of it);
- **reanalyses and delayed-time gridded products valid in $P$**, even at times before $t_0$:
  GLORYS12, DUACS DT/MY L4, ARMOR3D MY, ERA5 and the like are produced afterwards, with smoothers
  and observation windows that reach past $t_0$, and they assimilated the observations DC2
  verifies against.

## Rule 3 — Lead time 0 is not an independent score

The observations a system assimilates up to $t_0$ include those of the matching window of lead
time 0 (±12 h around $t_0$). Lead time 0 measures how well the initial state fits them; it is
reported but SHOULD NOT be used to rank forecasts. From lead time 1 on, the matching windows
start after $t_0$ and the verifying observations are independent of any allowed input.

## Rule 4 — Declaration

A submission MUST declare, for training and for inference: each dataset (product identifier,
version, variables), its time span, and how the initial state at $t_0$ is obtained. The organisers
MAY ask for the code, the weights or a re-run on dates of their choice, and MAY exclude a
submission that breaks these rules from the leaderboard. Baselines follow the same rules and
their declaration is published with them.

## Examples

| Use | Allowed? |
|---|---|
| Train on GLORYS12 1993–2022, initialize from the CMEMS analysis of $t_0$ | yes |
| Train on GLORYS12 up to 2024 | **no** (Rule 1) |
| Initialize from GLORYS12 at $t_0$ (in $P$) | **no** (Rule 2: a reanalysis) |
| Use the Argo profiles and along-track SLA of the 10 days before $t_0$ as inputs | yes |
| Use the DUACS DT L4 map of $t_0$ as input | **no** (Rule 2: delayed-time L4) |
| Use the DUACS NRT L4 map of $t_0$ as input | yes |
| Force the model with the ECMWF forecast issued at $t_0$ | yes |
| Force the model with ERA5 over the forecast's 10 days | **no** (Rule 2: the future) |
| Compute input normalisation statistics over 2010–2022 | yes |
| Pick the best checkpoint by its score on 2023 | **no** (Rule 1: model selection) |
