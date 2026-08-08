# PROJECT: Urban Heat Island Intensity in Colombo, Sri Lanka (2000–2025)

## What this is
Undergraduate spatial-analytics practicum. Remote-sensing analysis of Land Surface
Temperature (LST) trends in Colombo using ONLY free, public Google Earth Engine data.
Three deliverables: (1) LST trend analysis, (2) future hotspot scenario projection,
(3) urban greening priority recommendations.

## Hard constraints
- **Free/public data only.** No paywalled sources, no proprietary sensors, no field data.
- **Execution environment is Google Colab.** Code must run in Colab with
  `earthengine-api` + `geemap` installed via pip. Never assume a local GEE credential
  file, a local GDAL build, or ArcGIS.
- **You (Claude Code) cannot run Earth Engine.** You have no GEE credentials and no
  network access to EE servers. Write code, write tests for the pure-Python parts,
  and clearly mark cells that must be executed by the user in Colab.
- Author code as importable `.py` modules under `src/`, and thin `.ipynb` notebooks
  that import from `src/`. Notebooks orchestrate; modules hold logic.
- Target Python 3.10+.

## Repo layout (create and maintain this)
colombo-uhi/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── config/
│   └── params.yaml          # ALL dataset IDs, dates, thresholds, CRS live here
├── src/colombo_uhi/
│   ├── __init__.py
│   ├── auth.py              # ee.Authenticate/Initialize helpers for Colab
│   ├── aoi.py               # study-area geometries (CMC, district, rural ref)
│   ├── landsat.py           # C2 L2 masking, scaling, harmonised collection
│   ├── modis.py             # MOD11A2/MYD11A2 loading + QC
│   ├── indices.py           # NDVI, NDBI, MNDWI, EVI, SAVI, albedo
│   ├── composites.py        # seasonal/annual compositing + valid-obs counts
│   ├── uhi_metrics.py       # SUHII, UTFVI, z-scores
│   ├── trends.py            # Mann-Kendall, Sen's slope, FDR
│   ├── spatial_stats.py     # Moran's I, Getis-Ord Gi*, GWR wrappers
│   ├── prediction.py        # RF regression + CA-Markov glue
│   ├── greening.py          # MCDA / AHP weighted overlay
│   ├── exports.py           # Export.image.toDrive / table.toDrive wrappers
│   └── viz.py               # geemap map builders, matplotlib figure helpers
├── notebooks/
│   ├── 00_setup_and_auth.ipynb
│   ├── 01_aoi_and_boundaries.ipynb
│   ├── 02_lst_pipeline.ipynb
│   ├── 03_uhi_metrics.ipynb
│   ├── 04_trend_analysis.ipynb
│   ├── 05_spatial_statistics.ipynb
│   ├── 06_prediction.ipynb
│   ├── 07_greening_priority.ipynb
│   └── 08_figures_for_report.ipynb
├── data/
│   ├── raw/                 # .gitignore'd
│   ├── interim/
│   └── outputs/             # exported CSVs, small GeoJSONs (committed)
├── figures/
└── tests/

## Study area (authoritative values — do not invent others)
- Colombo, Sri Lanka. Centre ≈ **6.93° N, 79.85° E**. Köppen **Af** (tropical rainforest/monsoon).
- **Colombo Municipal Council (CMC)**: ~**37 km²** (gazetted 37.31) — the urban core.
  CMC = union of the **55 GN divisions** the CMC's own GIS Unit lists as inside the
  municipality (`aoi.cmc.gn_division_names`). Do **not** define it as the Colombo +
  Thimbirigasyaya DS pair: with COD-AB polygons that measures **46.87 km²**, because
  COD-AB's Colombo DS encloses the Port's outer harbour. No DS union yields 37 km².
- **Colombo District**: 13 Divisional Secretariat divisions, **557 Grama Niladhari (GN) divisions**.
- Western Province = Colombo + Gampaha + Kalutara districts.
- Analysis CRS: **EPSG:32644 (UTM 44N)**. Analysis grid: **30 m**.
- Water bodies to mask: Indian Ocean, **Beira Lake**, **Bolgoda Lake**, **Kelani River**,
  **Diyawanna (Parliament) Lake**.
- Monsoons: **SW May–Sep**, **NE Dec–Feb**. Driest/clearest window ≈ **Jan–Mar** →
  use as the primary "dry season" comparison window.

## Datasets (exact IDs — use these, do not substitute)
| Purpose | GEE ID | Notes |
|---|---|---|
| Landsat 5 TM L2 | `LANDSAT/LT05/C02/T1_L2` | ST band `ST_B6` |
| Landsat 7 ETM+ L2 | `LANDSAT/LE07/C02/T1_L2` | ST band `ST_B6`; SLC-off after 2003-05-31 |
| Landsat 8 L2 | `LANDSAT/LC08/C02/T1_L2` | ST band `ST_B10` |
| Landsat 9 L2 | `LANDSAT/LC09/C02/T1_L2` | ST band `ST_B10` |
| MODIS Terra LST | `MODIS/061/MOD11A2` | 8-day, 1 km, `LST_Day_1km`/`LST_Night_1km` |
| MODIS Aqua LST | `MODIS/061/MYD11A2` | Aqua daytime ≈ peak heating |
| Land cover | `GOOGLE/DYNAMICWORLD/V1` | 10 m, 2015-06-27 → present |
| Land cover | `ESA/WorldCover/v100`, `ESA/WorldCover/v200` | 2020, 2021 |
| Built-up | `JRC/GHSL/P2023A/GHS_BUILT_S` | 100 m, 1975–2030 |
| Local Climate Zones | `RUB/RUBCLIM/LCZ/global_lcz_map/latest` | 100 m, Demuzere et al. 2022 |
| Elevation | `USGS/SRTMGL1_003` | 30 m |
| Population | `WorldPop/GP/100m/pop` | annual |
| Night lights | `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` | 2012+ |
| Surface water | `JRC/GSW1_4/GlobalSurfaceWater` | for water masking |
| Emissivity | `NASA/ASTER_GED/AG100_003` | if computing LST manually |
| Reanalysis air temp | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | `temperature_2m` validation |
| Station air temp | `NOAA/GSOD` | Katunayake WMO 43450 |
| Boundaries | `FAO/GAUL_SIMPLIFIED_500m/2015/level2` | Colombo District |

**GN-division polygons are NOT in GAUL/GADM.** They must be uploaded by the user as a
GEE asset. Write code that reads an asset path from `config/params.yaml` and fails with
a clear, actionable message if it is missing.

## Landsat Collection 2 Level-2 constants (memorise; never hardcode elsewhere)
- Surface Temperature → Kelvin: `DN * 0.00341802 + 149.0`; °C = K − 273.15
- Surface Reflectance: `DN * 0.0000275 - 0.2`
- `ST_EMIS`: `DN * 0.0001`
- `ST_QA` (uncertainty): `DN * 0.01` Kelvin
- Valid ST DN range 293–65535; fill = 0
- `QA_PIXEL` bits: 0 Fill, 1 Dilated Cloud, 2 Cirrus, 3 Cloud, 4 Cloud Shadow,
  5 Snow, 6 Clear, 7 Water, 8–9 Cloud Conf, 10–11 Shadow Conf, 14–15 Cirrus Conf
- Standard mask: bits 0–4 all zero, AND `QA_RADSAT == 0`
- C2 is inter-calibrated across TM/ETM+/OLI — **no manual harmonisation coefficients needed.**
  Still verify empirically on overlapping years.

## MODIS constants
- MOD/MYD11A2 LST scale factor **0.02** → Kelvin
- MOD11A2 is a plain average of MOD11A1 with **no built-in QA filtering** — you must
  filter `QC_Day`/`QC_Night` for "good quality" and "avg error ≤ 1 K"
- Terra overpass ≈ 10:30 / 22:30 local; Aqua ≈ 13:30 / 01:30 local

## Method definitions (use exactly these)
- **SUHII** = mean urban LST − mean rural LST. Compute under **≥2 rural definitions**
  (buffer method AND LCZ-based) and report sensitivity.
- **UTFVI** = (Ts − Tmean) / Tmean. Six classes: Excellent <0; Good 0–0.005;
  Normal 0.005–0.010; Bad 0.010–0.015; Worse 0.015–0.020; Worst >0.020.
- **Trend**: pixel-wise Mann-Kendall (`ee.Reducer.kendallsCorrelation`) + Sen's slope
  (`ee.Reducer.sensSlope`) on the ANNUAL composite series. Report °C/yr.
- **Significance**: derive MK Z and p, then apply **Benjamini-Hochberg FDR** correction
  in Python on the exported p-value raster. Map only FDR-significant trends.
- **Hot spots**: Getis-Ord Gi* and Local Moran's I via `esda`/`libpysal` on exported
  zonal statistics; plus Emerging Hot Spot Analysis (MK on Gi* space-time bins).
- **Driver attribution**: OLS → test residual Moran's I → spatial lag/error → **GWR/MGWR**
  via the `mgwr` package.

## Non-negotiable scientific caveats — enforce these in code comments and outputs
1. This measures **Land Surface Temperature, not air temperature**. Never label an
   output "air temperature" or "temperature felt by residents." SUHI can be roughly
   2× canopy-air UHI.
2. **Always emit a per-pixel valid-observation count** alongside any composite or trend
   product. Tropical cloud cover means only a minority of scenes are usable.
3. **Never present prediction as forecast.** It is conditional scenario projection.
   Every predictive output must ship with validation metrics (RMSE, R², Kappa) and
   explicit uncertainty language.
4. Landsat captures a single ~10:30 local overpass. Night-time UHI only via MODIS.
5. Report rural-reference and aggregation-unit (MAUP) sensitivity, not single numbers.

## Coding standards
- Type hints on all public functions. Google-style docstrings.
- Every function that touches EE takes and returns `ee.Image`/`ee.ImageCollection`
  explicitly typed; no hidden global state.
- **No magic numbers in `src/`.** Every constant comes from `config/params.yaml`.
- Server-side EE operations only — never `.getInfo()` inside a loop.
- Wrap exports in `src/colombo_uhi/exports.py` with consistent naming:
  `{product}_{aoi}_{startyear}_{endyear}_{res}m`.
- Pure-Python logic (FDR, UTFVI classification, MCDA weighting, AHP consistency ratio)
  must have `pytest` unit tests with synthetic arrays. EE-dependent code does not.
- Add a `# COLAB: RUN THIS CELL` marker comment on notebook cells requiring the user's
  authenticated session.

## Workflow rules for you (Claude Code)
- Work **one phase at a time**. Stop at the end of each phase and report what the user
  must run in Colab before you continue.
- After writing any module, run `pytest` on the pure-Python tests and fix failures.
- If a dataset ID, band name, or algorithm detail is uncertain, **say so explicitly and
  ask** rather than guessing. Wrong band names cost the user hours of Colab debugging.
- Keep a running `PROGRESS.md` with what is done, what is pending, and what the user
  needs to verify.
  - Before writing code against any GEE collection, open its page in the Browser pane on
  developers.google.com/earth-engine/datasets and verify the band names, scale factors,
  QA bit layout, and temporal coverage against what CLAUDE.md claims. If the catalog
  disagrees with CLAUDE.md, STOP and tell me — do not silently follow either one.