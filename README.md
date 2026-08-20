# Urban Heat Island Intensity in Colombo, Sri Lanka (2000–2025)

Undergraduate spatial-analytics practicum: a remote-sensing analysis of **Land
Surface Temperature (LST)** trends in Colombo using **only free, public Google
Earth Engine data**, run entirely in **Google Colab**.

Three deliverables:

1. **LST trend analysis** — pixel-wise Mann-Kendall + Sen's slope (°C/yr) with
   Benjamini-Hochberg FDR-corrected significance, 2000–2025.
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
   (Phase 7) ranking all 557 GN divisions on five observed criteria: surface
   heat, the share of the zone in severe UTFVI classes, vegetation deficit,
   population density, and green-space access deficit. Weights come from a
   pairwise comparison matrix with a consistency ratio of **0.0081** against
   Saaty's 0.10 threshold; the ranking is cross-checked against an independent
   **TOPSIS** ranking, a **3-30-300** assessment, and the Colombo Wetland
   Complex. Phase 6 supplies the quantitative half as a **validated greening
   counterfactual**: shifting 27.7 km² of priority-zone surface 20 % toward the
   observed canopy signature implies **−0.84 °C** mean LST inside those zones.
   It rests on the validated random forest alone — no land-cover projection is
   involved — and it assumes the planting happens.

   Two things about Phase 7 must travel with its output. The **weights are
   judgements, not measurements** — the consistency ratio tests whether they are
   self-consistent, never whether they are right. And the **criteria are
   near-collinear over Colombo** (`rho(LST, green fraction) = −0.9147`), so a
   leave-one-out ablation against a heat-only ranking, not the weights, is what
   says how much the multi-criteria method actually adds.

> ⚠️ **This project measures Land Surface Temperature, not air temperature.**
> Surface UHI can be roughly 2× the canopy-air UHI. No output may be labelled
> "air temperature" or "temperature felt by residents".

## How this repo works

- **`config/params.yaml` is the single source of truth** — every dataset ID,
  band name, scale factor, date range, threshold and CRS lives there. No magic
  numbers anywhere in `src/`.
- **`src/colombo_uhi/`** holds all logic as importable, typed modules.
- **`notebooks/`** are thin orchestrators run *by you* in Colab. Cells marked
  `# COLAB: RUN THIS CELL` need your authenticated Google session.
- **`tests/`** cover the pure-Python logic; Earth-Engine calls are not
  unit-tested.

## Quick start (GitHub → Colab workflow)

1. **Push this project to GitHub**: <https://github.com/Dineth0627/colombo_uhi>
   (first-time commit/push commands are in `PROGRESS.md`).
2. **Earth Engine project**: `research-uhi-484404` — registered and set in
   `config/params.yaml` (`gcp.ee_project_id`). Per-session override if ever
   needed: `os.environ["EE_PROJECT"] = "..."` or `init_ee(project_id="...")`.
3. After the first push, open notebook 00 directly in Colab:
   <https://colab.research.google.com/github/Dineth0627/colombo_uhi/blob/main/notebooks/00_setup_and_auth.ipynb>
4. Run all cells top-to-bottom (`REPO_URL` is preconfigured).
5. The notebook ends with an Earth Engine smoke test. Expected output:
   `one_plus_one == 2` and `srtm_bands == ['elevation']`.

Each fresh Colab VM re-prompts for Google authentication once — that is
normal. Within a session `init_ee()` is idempotent and never re-prompts.

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
│   ├── viz.py               # Phase 2/3/4 — thumbnails + report figures
│   ├── uhi_metrics.py       # Phase 3 — SUHII, UTFVI, z-scores, zonal, driver OLS
│   ├── trends.py            # Phase 4 — Mann-Kendall, Sen's slope, BH-FDR, modified MK
│   ├── landcover.py         # Phase 4 — WorldCover/LCZ/Dynamic World + stratified stats
│   ├── exports.py           # Phase 4 — Export.image/table.toDrive + task status
│   ├── spatial_stats.py     # Phase 5 — Moran/LISA/Gi*, EHSA, OLS→lag/error→GWR/MGWR,
│   │                        #           MAUP, landscape metrics
│   ├── prediction.py        # Phase 6 — RF regression, blocked splits, CA-Markov,
│   │                        #           scenarios, and the validation export guard
│   └── greening.py          # Phase 7 — AHP (power iteration + CR), TOPSIS, criterion
│                            #           prep, 3-30-300, wetland cross, guarded writers
├── notebooks/               # 00–07 written + Colab-verified; 08 is a stub
├── docs/
│   └── molusce_handoff.md   # Phase 6 — CA-Markov handoff to MOLUSCE in QGIS
├── data/
│   ├── raw/                 # git-ignored
│   ├── interim/
│   └── outputs/             # exported CSVs / small GeoJSONs (committed)
├── figures/
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
| 8 | `08_figures_for_report` | final figures | ⬜ |

## Running tests locally

```bash
python -m pytest tests/ -q
```

The suite needs only `pyyaml` and `pytest` (no `earthengine-api`): tests fake
`ee` and pin `params.yaml` values against the verified catalog constants.

## Data policy

Free/public GEE datasets only — no paywalled sources, no proprietary sensors,
no field data. GN-division boundary polygons are **not** in any public GEE
dataset and must be uploaded by the project owner as an EE asset (path set in
`params.yaml` → `aoi.assets.gn_divisions`).
