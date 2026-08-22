# Urban Heat Island Intensity in Colombo, Sri Lanka (2000–2025)

Undergraduate spatial-analytics practicum: a remote-sensing analysis of **Land
Surface Temperature (LST)** trends in Colombo using **only free, public Google
Earth Engine data**, run entirely in **Google Colab**.

Three deliverables:

1. **LST trend analysis** — pixel-wise Mann-Kendall + Sen's slope (°C/yr) with
   Benjamini-Hochberg FDR-corrected significance. Fitted **within one sensor
   family** (Landsat 8+9, 2014–2025), because Collection 2's inter-calibration
   was tested over Colombo and failed: see *The one result that shapes
   everything else*, below.
2. **Future hotspot scenario projection** — RF regression + CA-Markov,
   presented strictly as *conditional scenario projection*, never a forecast.
   The random-forest LST model is **validated**: held-out RMSE 1.13 °C, R² 0.894
   on spatially blocked data, led by NDBI, built fraction and LCZ class. The
   CA-Markov land-cover component is a **measured negative result** — it
   reproduces the *quantity* of land-cover change over Colombo but cannot
   *allocate* it, and never beats a no-change map, across two class schemes and
   both calibration intervals Dynamic World's record supports. No projected
   land-cover product is exported, because a validation guard refuses it.
   Deliverable 2 therefore carries a stated, quantified limitation rather than
   an unvalidated map.
3. **Urban greening priority recommendations** — an MCDA/AHP weighted overlay
   ranking all 557 GN divisions on five observed criteria: surface heat, the
   share of the zone in severe UTFVI classes, vegetation deficit, population
   density, and green-space access deficit. Weights come from a pairwise
   comparison matrix with a consistency ratio of **0.0081** against Saaty's 0.10
   threshold; the ranking is cross-checked against an independent **TOPSIS**
   ranking, a **3-30-300** assessment, and the Colombo Wetland Complex. Phase 6
   supplies the quantitative half as a **validated greening counterfactual**:
   shifting 27.7 km² of priority-zone surface 20 % toward the observed canopy
   signature implies **−0.84 °C** mean LST inside those zones. It rests on the
   validated random forest alone — no land-cover projection is involved — and it
   assumes the planting happens.

   Two things about the ranking must travel with it. The **weights are
   judgements, not measurements** — the consistency ratio tests whether they are
   self-consistent, never whether they are right. And the **criteria are
   near-collinear over Colombo**, so a leave-one-out ablation against a heat-only
   ranking, not the weights, is what says how much the multi-criteria method
   actually adds. Over Colombo, the honest answer is: very little.

> ⚠️ **This project measures Land Surface Temperature, not air temperature.**
> Surface UHI can be roughly 2× the canopy-air UHI. No output may be labelled
> "air temperature" or "temperature felt by residents".

**Start here for the write-up:** [`docs/methods.md`](docs/methods.md) and
[`docs/limitations.md`](docs/limitations.md). Both are **generated** from
`config/params.yaml` and from the committed measurements, so they cannot drift
from the code. The report figures are in [`figures/report/`](figures/report/).

## The one result that shapes everything else

Landsat Collection 2 is documented as inter-calibrated across TM, ETM+ and OLI.
Tested empirically on dry-season CMC means at 100 m, it is not, over Colombo:

| pair | mean offset | t | overlap years |
|---|---|---|---|
| Landsat 5 − Landsat 7 | **+1.78 °C** | +2.72 | 8 |
| Landsat 7 − Landsat 8 | **−2.48 °C** | −3.60 | 10 |
| Landsat 8 − Landsat 9 | −0.40 °C | −0.67 | 4 |

The first two are 2.4× and 3.4× the entire 26-year trend signal, and the
Landsat 7→8 step alone predicts the observed decadal jump to within 0.3 °C.
Consequences, enforced in code:

- **No multi-year trend is fitted across a changeover.** The trend products use
  `landsat_oli_dry` (L8+L9, 2014–2025) — 12 years instead of 26, less power, but
  a defensible slope instead of an artefact.
- **Offsets are deliberately not estimated and subtracted.** Eight to ten noisy
  overlap years would inject a new error rather than remove one.
- **SUHII is unaffected** — it is a within-year urban-minus-rural difference, so
  a spatially common-mode step cancels.
- **Figure 1 of the report draws the pooled series anyway**, labelled as a
  sensor-step diagnostic, above a MODIS Terra row that spans all three decades
  on one sensor. The gap between the two rows *is* the result.

## How this repo works

- **`config/params.yaml` is the single source of truth** — every dataset ID,
  band name, scale factor, date range, threshold, palette and CRS lives there.
  No magic numbers anywhere in `src/`.
- **`src/colombo_uhi/`** holds all logic as importable, typed modules.
- **`notebooks/`** are thin orchestrators run *by you* in Colab. Cells marked
  `# COLAB: RUN THIS CELL` need your authenticated Google session.
- **`tests/`** cover the pure-Python logic; Earth-Engine calls are not
  unit-tested.
- **`docs/methods.md` and `docs/limitations.md` are generated.** Never edit them
  by hand — edit `params.yaml`, then re-run `reporting.write_docs(params)`. A
  test fails if the committed copy disagrees with what the code produces.

---

# Reproduction guide

## 0. Prerequisites

| | |
|---|---|
| **Google account** | for Colab and Google Drive |
| **Earth Engine project** | `research-uhi-484404`, registered and set in `config/params.yaml` (`gcp.ee_project_id`). Override per session with `os.environ["EE_PROJECT"]` or `init_ee(project_id=...)`. |
| **Drive folder** | `colombo_uhi_exports` (`exports.drive_folder`). Batch exports land here and you copy them into `data/interim/`. |
| **GN/DS boundary assets** | **You must upload these.** They are in no public Earth Engine dataset. |

### Uploading the boundary assets

Grama Niladhari polygons are not in GAUL, GADM or any other public GEE
collection. Download the Sri Lanka COD-AB admin levels 3 and 4, upload them as
Earth Engine table assets, and set the paths under `aoi.assets` in
`config/params.yaml`:

```yaml
aoi:
  assets:
    ds_divisions: "projects/<your-project>/assets/lka_admin3"
    gn_divisions: "projects/<your-project>/assets/lka_admin4"
```

Notebook 01 has the step-by-step upload instructions and a probe that reports
exactly which property names your upload carries. Every module that needs these
assets fails with an actionable message if they are absent — nothing guesses.

> **GN names are not unique within Colombo District.** Always scope a GN-name
> filter to its parent DS division, or key on `adm4_pcode`. Unscoped, you will
> pull in unrelated same-named divisions from Dehiwala, Moratuwa and Kolonnawa.

## 1. Push, then open notebook 00 in Colab

The notebooks clone this repository from GitHub, so **uncommitted or unpushed
work is invisible to them**.

```bash
git push
```

Then open:
<https://colab.research.google.com/github/Dineth0627/colombo_uhi/blob/main/notebooks/00_setup_and_auth.ipynb>

Run all cells top to bottom. `REPO_URL` is preconfigured. The notebook ends with
an Earth Engine smoke test; expect `one_plus_one == 2` and
`srtm_bands == ['elevation']`.

Each fresh Colab VM re-prompts for Google authentication once — that is normal.
Within a session `init_ee()` is idempotent and never re-prompts.

## 2. Run the notebooks in order

Each opens the same way (clone → pip → auth) and is safe to re-run. Every one
begins with a **staleness guard** that fails loudly if the checked-out `src/`
predates the notebook — the failure mode it exists to prevent is a live runtime
serving cached modules after a `git pull`, so new cells silently run old code.

| # | Notebook | Produces | Approx. Colab time | Exports to copy back into `data/interim/` |
|---|---|---|---|---|
| 00 | `00_setup_and_auth` | auth + smoke test | 3 min | — |
| 01 | `01_aoi_and_boundaries` | AOI geometries, water mask, `*_divisions_colombo.geojson` | 15 min | — |
| 02 | `02_lst_pipeline` | harmonised Landsat + MODIS LST, composites, obs counts | 25 min | — |
| 03 | `03_uhi_metrics` | `suhii_2000_2025.csv`, UTFVI class shares, z-scores | 40 min | — |
| 04 | `04_trend_analysis` | MK + Sen's slope, BH-FDR, decadal product, sensor offsets | 60 min + exports | `lst_trend_*`, `lst_decadal_*` |
| 05 | `05_spatial_statistics` | Gi*, LISA, EHSA, GWR/MGWR, MAUP, landscape metrics | 60 min | zonal covariate tables |
| 06 | `06_prediction` | RF + blocked CV, CA-Markov, greening counterfactual | 50 min | predictor rasters |
| 07 | `07_greening_priority` | AHP/TOPSIS ranking, 3-30-300, wetland cross | 45 min + exports | green/canopy, population, wetland rasters |
| 08 | `08_figures_for_report` | the eleven report figures, `docs/`, provenance | 30 min + 3 exports | UTFVI class, obs count, MODIS decadal |

**The export loop, every time it appears:**

1. A cell submits batch tasks and prints their descriptions.
2. Re-run the **poll cell** until every state reads `COMPLETED`. Do not block a
   cell waiting — a Colab runtime disconnected mid-wait loses the session.
3. Mount Drive and copy the results in:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

   ```bash
   cp /content/drive/MyDrive/colombo_uhi_exports/*.tif data/interim/
   ```

4. Continue with Part 2 of that notebook.

**Bring the results home.** Every notebook ends with a bundle cell that zips
`data/outputs/`, `figures/` and (for notebook 08) `docs/` for download. Unzip
into the repo root locally and commit. `data/interim/` is deliberately excluded:
GeoTIFFs are tens of MB and fully reproducible from the Drive exports, while the
CSVs and figures are not reproducible without another full run.

## 3. Notebook 08 specifically

Phase 8 adds only **three** Earth Engine tasks. Everything else is either
committed in `data/outputs/` or already exported by Phase 4:

- MODIS Terra decadal means — figure 1's single-sensor row
- total valid dry-season observations per pixel — figure 3
- UTFVI six-class images, one band per epoch — figure 4

Before drawing anything, a discovery cell locates every input and names every
gap at once. Figures 1 and 2 reuse the Phase 4 rasters, so copy those back too
if this is a fresh runtime.

The notebook ends with a ten-point sign-off checklist. The one that matters
most: **on figure 1, the Landsat row should step between decades and the MODIS
row should not.** If both step together, the cross-sensor finding is wrong, and
every trend product in this project rests on it.

## 4. Run the tests

```bash
python -m pytest tests/ -q
```

1592 pass, 3 skip. The suite needs only `pyyaml`, `pytest`, `pandas`,
`numpy`, `matplotlib` and `rasterio` — **no Earth Engine credentials**: tests
fake `ee` and pin `params.yaml` values against the verified catalog constants.

Three of them are worth knowing about, because they hold decisions in place
rather than checking behaviour:

- `test_every_non_exempt_palette_passes_the_colour_vision_check` — if a palette
  changes and stops being legible under simulated dichromacy, fix the palette,
  not the exemption.
- `test_committed_document_matches_what_the_code_generates_now` — if you edit a
  threshold in `params.yaml`, regenerate the docs:

  ```bash
  python -c "from colombo_uhi import load_params, reporting; reporting.write_docs(load_params())"
  ```

- `test_provenance_covers_every_configured_dataset` — a dataset added to
  `params.yaml` and left out of the provenance table fails here.

---

## Repository structure

```
├── CLAUDE.md                # project spec (authoritative)
├── README.md
├── PROGRESS.md              # phase status + what to verify next
├── requirements.txt         # Colab-compatible pins + conflict notes
├── config/
│   └── params.yaml          # SINGLE SOURCE OF TRUTH for all constants
├── src/colombo_uhi/
│   ├── __init__.py          # load_params(), repo_root()
│   ├── auth.py              # Colab-friendly EE auth/init (idempotent)
│   ├── aoi.py               # Phase 1 — boundaries, CMC, water mask, rural refs
│   ├── landsat.py           # Phase 2 — harmonised L5/7/8/9 C2 L2 LST collection
│   ├── modis.py             # Phase 2 — MOD11A2/MYD11A2 + real QC bit filtering
│   ├── indices.py           # Phase 2 — NDVI, NDBI, MNDWI, EVI, SAVI, albedo
│   ├── composites.py        # Phase 2 — annual/dry-season + valid-obs counts
│   ├── uhi_metrics.py       # Phase 3 — SUHII, UTFVI, z-scores, zonal, driver OLS
│   ├── trends.py            # Phase 4 — Mann-Kendall, Sen's slope, BH-FDR, modified MK
│   ├── landcover.py         # Phase 4 — WorldCover/LCZ/Dynamic World + stratified stats
│   ├── exports.py           # Phase 4 — Export.image/table.toDrive + task status
│   ├── spatial_stats.py     # Phase 5 — Moran/LISA/Gi*, EHSA, OLS→lag/error→GWR/MGWR,
│   │                        #           MAUP, landscape metrics
│   ├── prediction.py        # Phase 6 — RF regression, blocked splits, CA-Markov,
│   │                        #           scenarios, and the validation export guard
│   ├── greening.py          # Phase 7 — AHP (power iteration + CR), TOPSIS, criterion
│   │                        #           prep, 3-30-300, wetland cross, guarded writers
│   ├── viz.py               # Phases 2–8 — thumbnails, every figure, CVD verification
│   └── reporting.py         # Phase 8 — provenance table, generated methods/limitations
├── notebooks/               # 00–08, all written and Colab-verified
├── docs/
│   ├── methods.md           # GENERATED — methods draft for the report
│   ├── limitations.md       # GENERATED — every caveat, standing and discovered
│   └── molusce_handoff.md   # Phase 6 — CA-Markov handoff to MOLUSCE in QGIS
├── data/
│   ├── raw/                 # git-ignored
│   ├── interim/             # git-ignored — downloaded GeoTIFFs
│   └── outputs/             # exported CSVs / small GeoJSONs / TIFs (committed)
├── figures/                 # per-phase diagnostics, 150 dpi
│   └── report/              # the eleven report figures, 300 dpi
└── tests/                   # pytest — pure-Python logic only
```

## Phase roadmap

| Phase | Notebook | Content | Status |
|---|---|---|---|
| 0 | `00_setup_and_auth` | scaffold, config, EE auth | ✅ done + Colab-verified |
| 1 | `01_aoi_and_boundaries` | AOI geometries, GN asset, water mask | ✅ done + Colab-verified |
| 2 | `02_lst_pipeline` | Landsat/MODIS LST, composites + valid-obs counts | ✅ done + Colab-verified |
| 3 | `03_uhi_metrics` | SUHII (≥2 rural defs), UTFVI, z-scores | ✅ done + Colab-verified |
| 4 | `04_trend_analysis` | MK + Sen's slope, BH-FDR, decadal, modified MK | ✅ done + Colab-verified |
| 5 | `05_spatial_statistics` | Gi*, Moran's I, EHSA, GWR/MGWR, MAUP, landscape metrics | ✅ done + Colab-verified |
| 6 | `06_prediction` | RF + CA-Markov scenario projection, spatially blocked validation, greening counterfactual, MOLUSCE handoff | ✅ done + Colab-verified |
| 7 | `07_greening_priority` | MCDA/AHP weighted overlay (CR 0.0081), TOPSIS cross-check, 3-30-300 compliance, Ramsar wetland cross, ranked priority table | ✅ done + Colab-verified |
| 8 | `08_figures_for_report` | eleven 300 dpi report figures, colour-vision verification, generated methods and limitations, data provenance | 🟡 Colab run 3: ten of eleven drawn, **figure 1 outstanding** |

## The report figures

Written to `figures/report/` by notebook 08, at `report.dpi` (300).

| # | Figure | What to read it for |
|---|---|---|
| 1 | Decadal mean dry-season LST | Landsat row = geography; MODIS row = level. The gap between them is the sensor step. |
| 2 | Sen's slope, FDR-stippled | Rate and confidence in one place. Grey is *untested*, not *no trend*. |
| 3 | Valid observations per pixel | The denominator for everything else. Read figure 2 against it. |
| 4 | UTFVI six-class, per epoch | Within-epoch redistribution of heat — **never** evidence of warming. |
| 5 | Getis-Ord Gi* and EHSA | Cluster geography. No epoch-to-epoch magnitude. |
| 6 | SUHII by source and rural definition | Both rural definitions, day and night. The spread is the result. |
| 7 | LST vs NDVI and NDBI, by epoch | Slopes are comparable across rows; levels are not. |
| 8 | GWR local coefficients | Where the drivers act differently. A property of polygons. |
| 9 | Greening counterfactual | A conditional counterfactual on observed predictors, not a forecast. |
| 10 | Greening priority, top zones labelled | The deliverable. Weights are judgements. |
| 11 | Data provenance | Every source, band and scale factor, generated from the config. |

### Colour

Every palette in `config/params.yaml` is verified under simulated deuteranopia,
protanopia and tritanopia (Viénot, Brettel & Mollon 1999), using two tests
because one cannot judge both kinds of palette: **categorical** palettes on
minimum pairwise CIE76 difference, **ramps** on lightness monotonicity. The
measured result is `data/outputs/palette_cvd_check.csv`.

Two palettes failed and were changed — the UTFVI six-class scale, whose
lightness was not monotonic so the coolest and second-hottest classes looked
alike, and the 3-30-300 compliance scheme, whose two *partial*-compliance
categories collapsed onto each other under both red-green deficiencies. Two are
exempt with stated reasons and given redundant encoding instead: the Dynamic
World legend colours, which are fixed by the catalog, and the 17-category
emerging-hot-spot scheme, which cannot be separated by colour even in normal
vision. Colour never carries a class alone.

## Data policy

Free/public GEE datasets only — no paywalled sources, no proprietary sensors,
no field data. GN-division boundary polygons are **not** in any public GEE
dataset and must be uploaded by the project owner as an EE asset (path set in
`params.yaml` → `aoi.assets.gn_divisions`).

Four configured datasets are referenced by no analysis step and are marked as
such in the provenance table: `worldcover_2020`, `viirs_nightlights`,
`aster_ged` and `era5_land`. The consequential one is `era5_land` — the
specification names it for reanalysis air-temperature validation and **that
comparison was never carried out**, so nothing here independently corroborates
the satellite land surface temperatures against an air-temperature record. See
`docs/limitations.md`.
