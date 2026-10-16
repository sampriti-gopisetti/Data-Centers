# Data Center Siting in the U.S.

CSCI 461 – AI for Sustainable Development (Prof. Bistra Dilkina), Fall 2026
Ishika Agarwal, Malina Freeman, Mansimran Kaur, Sampriti Gopisetti

**Question:** What geographic, socioeconomic, demographic, and environmental characteristics
distinguish U.S. census tracts that receive data centers from comparable tracts that don't —
and do these patterns risk widening inequality? (SDG 10, 11, 12)

## Pipeline

| Stage | Module | Output |
|---|---|---|
| 1. Ingest raw sources | `src/ingest` | `data/raw/`, `data/interim/` |
| 2. Data center registry (dedup, tract join, construction year via NLCD) | `src/registry` | `dc_registry.parquet` |
| 3. Tract feature store (Census, USDA, EJScreen, NOAA, infrastructure, land cover) | `src/features` | `tract_features.parquet` |
| 4. Case–control sampling (pre-construction features) | `src/sampling` | `analysis_set.parquet` |
| 5. Models: baselines, conditional logit, RF / GBM, spatial CV, SHAP | `src/models` | metrics, SHAP values |
| 6. Analysis: clustering, temporal trends, figures | `src/analysis` | `reports/` |

## Layout

```
config/       settings (years, sampling, CV)
data/         raw / interim / processed — gitignored
docs/         proposal, design notes
notebooks/    EDA only; reusable code goes in src/
reports/      figures and final report
src/          pipeline code, one package per stage
tests/
```

## Setup

```bash
make setup                  # creates .venv and installs requirements
cp .env.example .env        # add your Census API key and GEE project
earthengine authenticate
```

Run a stage: `make registry`, or everything: `make all`.

## Data sources

PNNL IM3 Data Center Atlas · Compute Atlas · Epoch AI · Annual NLCD (via Google Earth Engine) ·
Census ACS / Decennial · USDA RUCA · EJScreen (archived) · NOAA NCEI · Census TIGER · EIA
