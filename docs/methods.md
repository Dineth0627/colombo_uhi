# Methods

> **This file is generated.** Every parameter below is read from
> `config/params.yaml` and from the committed products in `data/outputs/`
> by `colombo_uhi.reporting`. Do not edit it by hand - edit the config, then
> re-run `reporting.write_docs(params)`. `tests/test_reporting.py` fails if
> the committed copy disagrees with what the code produces, which is what
> stops a threshold being changed in one place and left stale in another.


## 1. Study area and analysis frame

The study area is Colombo, Sri Lanka, centred near 6.93° N, 79.85° E, in the Köppen **Af** tropical zone. Two nested units are used throughout. The **Colombo Municipal Council (CMC)** is the urban core, defined as the union of the 55 Grama Niladhari (GN) divisions the CMC's own GIS Unit lists as inside the municipality. The **Colombo District** is the wider analysis frame: 13 Divisional Secretariat divisions and 557 GN divisions.

The CMC's area is **scale-dependent and must always be quoted with its reduction scale**. The gazetted figure is 37.31 km². The COD-AB polygon measures about 47 km² because it encloses the Colombo Port outer harbour; once water is masked the land area is **about 40 km² at 30 m**, and lower again at coarser reduction scales. The residual over the gazetted figure is polygon generalisation plus the water-mask thresholds, and is reported as a sensitivity rather than tuned away.

All analysis is carried out in **EPSG:32644** (UTM 44N) on a **30 m** grid. GN names are **not unique within the district**, so every GN-name filter is scoped to its parent DS division or keyed on `adm4_pcode`.

GN-division polygons are in no public Earth Engine dataset. They are uploaded by the project owner as an Earth Engine asset whose path is configured in `aoi.assets`; the code fails with an actionable message if the asset is absent.

## 2. Data

Every source is free and public. The study period is **2000-2025**. The complete inventory - collection ID, bands, native resolution, scale factors, temporal coverage and the module that reads each one - is in `data/outputs/data_provenance.csv` and in figure 11; it is generated from `config/params.yaml`, so it cannot disagree with the code.

The primary comparison window is the **dry season, Jan-Mar**, the driest and least cloudy part of the year. Monsoon seasons are defined as SW monsoon (May-Sep); NE monsoon (Dec-Feb); Inter-monsoon, which partition the twelve months exactly once each.

**4 configured datasets are not referenced by any analysis step**: `worldcover_2020`, `viirs_nightlights`, `aster_ged`, `era5_land`. They are retained in the configuration because `CLAUDE.md` names them, and they appear in the provenance table marked as unreferenced. The consequence for `era5_land` in particular is stated in `docs/limitations.md`: no reanalysis air-temperature comparison was carried out.

## 3. Land surface temperature

### 3.1 Landsat Collection 2 Level-2

Surface temperature comes from the Collection 2 Level-2 science products of Landsat 5 TM, 7 ETM+, 8 OLI/TIRS and 9 OLI-2/TIRS-2. Digital numbers are converted with the published constants: `ST x 0.00341802 + 149.0` gives kelvin, from which 273.15 is subtracted for degrees Celsius; surface reflectance is `DN x 0.0000275 - 0.2`. Valid ST digital numbers run 293-65535, with 0 as fill.

Cloud masking uses `QA_PIXEL` bits 0, 1, 2, 3, 4 (fill, dilated cloud, cirrus, cloud, cloud shadow), all required to be zero, together with `QA_RADSAT == 0`. Reflectance bands are renamed to `blue`...`swir2` so that index code is sensor-agnostic across the TM/ETM+ and OLI band-numbering change.

### 3.2 MODIS

MOD11A2 (Terra) and MYD11A2 (Aqua) 8-day 1 km LST are scaled by **0.02** to kelvin. These products are a plain average of the daily product with **no built-in quality filtering**, so `QC_Day` and `QC_Night` are decoded explicitly, and the day and night policies differ because the daytime field is far more heavily contaminated: day requires mandatory QA <= 0 and LST-error class <= 0, night requires <= 1 and <= 2. Overpass times are terra day 10:30, terra night 22:30, aqua day 13:30, aqua night 01:30 local.

### 3.3 Compositing and the observation count

Annual and dry-season composites reduce Landsat with the **median** and MODIS with the **mean**, and every composite carries a per-pixel `obs_count` band. That band is not a diagnostic: under tropical cloud only a minority of scenes are usable, so no composite or trend product may be read without it. It is figure 3, and it is what the untested grey on the trend map means.

## 4. Cross-sensor continuity - a measured failure

Collection 2 is inter-calibrated across TM, ETM+ and OLI, so the documentation implies no manual harmonisation is needed. **This was tested over Colombo and it failed.** Dry-season CMC means at 100 m give:

| pair | mean offset (°C) | t | overlap years | window | verdict |
|---|---|---|---|---|---|
| landsat5 - landsat7 | +1.78 | +2.72 | 8 | 2004-2011 | **material** |
| landsat7 - landsat8 | -2.48 | -3.60 | 10 | 2014-2023 | **material** |
| landsat8 - landsat9 | -0.40 | -0.67 | 4 | 2022-2025 | **negligible** |
| landsat5 - landsat8 | - | - | 0 | - | **insufficient_overlap** |
| landsat5 - landsat9 | - | - | 0 | - | **insufficient_overlap** |
| landsat7 - landsat9 | - | - | 2 | - | **insufficient_overlap** |

The first two offsets are several times the entire 26-year trend signal, and the L7-to-L8 step alone predicts the observed decadal jump. **No multi-year trend is fitted across a Landsat changeover.** The trend products use `landsat_oli_dry` - a single-sensor-family series - accepting reduced statistical power in exchange for a defensible slope. Offsets are deliberately **not** estimated and subtracted: eight to ten noisy overlap years would inject a new error rather than remove one.

SUHII is unaffected, because it is a within-year urban-minus-rural difference in which a spatially common-mode step cancels.

## 5. Surface urban heat island intensity (SUHII)

SUHII is the mean urban LST minus the mean rural LST, computed under **2 independent rural definitions** and reported as a sensitivity rather than a single number:

1. **Buffer ring** - an annulus from 15 to 25 km beyond the CMC boundary, excluding water and built-up pixels.
2. **LCZ-based** - Local Climate Zone classes 1-10 as urban and 11-17 as rural, both clipped to the district.

Both rural masks are additionally capped at **100 m** elevation: the ring reaches inland relief, and at a tropical lapse rate that is up to several tenths of a degree of elevation-driven cooling that would otherwise be counted as urban heat island.

The series is computed for `landsat_dry`, `landsat_oli_dry`, `terra_day`, `terra_night`, `aqua_day`, `aqua_night`, `terra_day_relaxed`, each reduced at its own native scale, and Landsat stays on the median while MODIS stays on the mean so that no part of a Landsat-versus-MODIS difference is a reducer artefact.

## 6. Urban thermal field variance index (UTFVI)

UTFVI is `(Ts - Tmean) / Tmean`, classified on the published breaks 0.0, 0.005, 0.01, 0.015, 0.02 into the six classes Excellent, Good, Normal, Bad, Worse, Worst. The index is computed in **celsius**, which is load-bearing rather than documentation: UTFVI is a ratio, so on kelvin every break would mean roughly a tenth of what it means on Celsius.

`Tmean` is the **per year AOI mean** - each year's own spatial mean over the AOI. That is the standard formulation and keeps the published breaks meaningful, but it has a consequence that must travel with every UTFVI output: **the index measures within-year spatial structure only.** A city that warms uniformly by 2 °C shows no class change at all, so epoch-to-epoch class drift is a redistribution of heat and never evidence of warming.

Epochs are 2000s (2000-2009), 2010s (2010-2019), 2020s (2020-2025); the last is short by design because the study period ends in 2025.

## 7. Trend analysis

Trends are fitted pixel-wise on the **annual composite series** at 100 m, using the non-parametric Mann-Kendall test (`kendallsCorrelation`) and Sen's slope (`sensSlope`), reported in degrees Celsius per year. A pixel is fitted only where it has at least **10 valid years**, each resting on at least **3 valid observations**; pixels below either floor are neither significant nor non-significant, and are drawn as untested grey rather than as zero trend.

Significance is corrected for multiple testing **in Python on the exported p-value raster**, because the procedure needs every p-value at once and cannot be done server-side. The headline correction is **Benjamini-hochberg** at alpha = 0.05. Benjamini-Hochberg controls the false discovery rate under independence or positive regression dependency, and a 100 m LST raster is strongly spatially autocorrelated, so **Benjamini-Yekutieli is reported beside it** as the bound valid under arbitrary dependence. Both denominators are reported: the tested set and the total, because untested pixels belong to neither.

The autocorrelation-corrected Mann-Kendall (`hamed_rao`) is run alongside the plain test on the aggregate series, and the pair is reported: serial correlation inflates the variance of S, and the size of that inflation is itself a result.

Decadal windows are 2000-2010, 2011-2020, 2021-2025, weighted `equal_years` so every year counts once. **The windows are unequal by construction** (11 / 10 / 5 years), so any difference involving the last one rests on about half the sample and carries roughly √2 the standard error - which is why the decadal product emits a standard error and a z band rather than a difference alone.

## 8. Spatial statistics

Every spatial statistic is computed at **both** the GN (557 units) and DS (13 units) levels. What survives the coarsening is itself the reported result: this is the modifiable areal unit problem, and reporting one level only would hide it.

Spatial weights are **queen contiguity**, r-standardised, with islands attached by `attach_knn1`. Global Moran's I, Local Moran's I (LISA) and Getis-Ord Gi* are computed with **999 conditional permutations** at seed 42, and local p-values carry a Benjamini-hochberg correction at alpha = 0.05. Emerging hot spot analysis applies Mann-Kendall to the Gi* time series of each unit's space-time bins.

Driver attribution follows the escalation path required by the project specification: ordinary least squares, then a test of the residual Moran's I, then a spatial lag or error model, then geographically weighted regression and multiscale GWR. Variance inflation factors are reported at every step, because the candidate drivers over Colombo are strongly collinear.

## 9. Conditional scenario projection

**Nothing in this section is a forecast.** The framing is `conditional_scenario_projection_NOT_forecast` throughout, and every predictive product ships with validation metrics and explicit uncertainty language.

A random forest of **500 trees** (seed 42, minimum leaf population 5, bag fraction 0.5) is fitted to `LST_C` on the 2020s `landsat_oli_dry` composite at 100 m, from 20,000 sampled pixels, with predictors `NDVI`, `NDBI`, `built_fraction`, `elevation_m`, `dist_coast_km`, `pop_density`, `lcz_class` (`lcz_class` categorical).

Validation uses a **spatially blocked** split at 2000 m blocks, not a random one: adjacent LST pixels are near-duplicates, so a random split leaks the test set into the training set and reports an optimistic score. Held-out performance is **RMSE 1.13 °C, R² 0.894**.

The **land-cover component is a measured negative result.** A CA-Markov model calibrated on Dynamic World reproduces the *quantity* of land-cover change over Colombo but cannot *allocate* it: Kappa 0.810 against a persistence Kappa of 0.844 - that is -0.034 against a no-change map - with a figure of merit of 0.074. A validation guard therefore **refuses to export any projected land-cover product**, and no map of the 2030 or 2036 horizons is produced. The transition matrix, class areas and validation sensitivities are published as the evidence for the negative result.

The **greening counterfactual** rests on the validated random forest alone, with no land-cover projection underneath it, which is why it can be mapped while the future horizons cannot. It applies a **20% shift of each priority cell's surface character toward the observed canopy signature** and re-predicts. It is a counterfactual on observed predictors - 'if these zones were greened today' - and it assumes both that the planting happens and that the fitted relationship holds under a surface the model never observed. Extrapolation beyond the training envelope is measured and reported.

## 10. Greening priority (MCDA / AHP)

All 557 GN divisions are ranked on the following 5 observed criteria:

| criterion | direction | what it measures |
|---|---|---|
| `lst_hot` | benefit | Land surface temperature (2020s dry-season mean) |
| `utfvi_severe_share` | benefit | Share of zone in UTFVI Bad/Worse/Worst |
| `ndvi_deficit` | cost | Vegetation deficit (inverse NDVI) |
| `pop_density` | benefit | Population density (WorldPop 2020, people/km2) |
| `green_access_deficit` | cost | Residents beyond 300 m of a green space >= 0.5 ha (3-30-300) |

Weights are derived from a pairwise comparison matrix by power iteration on that matrix, never set by hand - `greening.criteria_weights` is pinned to null by a test so the weights cannot be reverse-engineered from a desired answer. The resulting weights are lst_hot 0.319, utfvi_severe_share 0.068, ndvi_deficit 0.184, pop_density 0.319, green_access_deficit 0.109, with lambda_max 5.0364, consistency index 0.0091 and **consistency ratio 0.0081** against Saaty's 0.1 threshold.

The ranking is cross-checked three ways: an independent **TOPSIS** ranking under the same weights, a **leave-one-out ablation** against a heat-only ranking, and a **DS-level re-run** as the MAUP sensitivity. The ablation, not the consistency ratio, is what says how much the multi-criteria method adds - and over Colombo the answer is that it adds very little, which is a finding about the city and is reported as one.

Compliance with the **3-30-300 rule** is assessed per division: a 30% tree-class share, and a green space of at least 0.5 ha within 300 m by euclidean distance, bounded by a variant at a 1.3 detour ratio. The '3 trees from every window' component is marked `not_remotely_sensable` (not remotely sensable) and does not enter the score. The 30% figure is a **Dynamic World tree-class share** of a 10 m modal classification, not crown cover from a canopy-height model, and must never be quoted as canopy cover.

The exported priority set is the **top 60** divisions, each carrying its score gap at the cut, its tie flag, its wetland status and its land-cover coverage flag.

## 11. Figures and colour

Report figures are written at **300 dpi** into `figures/report/`. Every palette in the configuration is verified for colour-vision deficiency under simulated deuteranopia, protanopia, tritanopia (Viénot, Brettel & Mollon 1999), using two different tests: categorical palettes must keep a minimum pairwise CIE76 difference of 12.0, and sequential or diverging ramps must keep a monotonic L* profile spanning at least 25.0. Judging a ramp by pairwise difference is the mistake that split avoids: adjacent stops of a continuous ramp are meant to be close. The measured result is in `data/outputs/palette_cvd_check.csv`. Two palettes failed and were changed; two are exempt for stated reasons and are given redundant encoding instead, because colour never carries a class alone.

## 12. Software

All processing runs in Google Colab against the Earth Engine Python API. Analysis logic lives in the importable package `colombo_uhi`; the notebooks orchestrate and display. Pure-Python logic - false-discovery correction, UTFVI classification, AHP weighting and its consistency ratio, TOPSIS, the colour-vision checks - is covered by a `pytest` suite that runs without Earth Engine credentials. Spatial statistics use `libpysal`, `esda`, `spreg` and `mgwr`; raster post-processing uses `rasterio`.
