# PROGRESS — Colombo UHI practicum

_Last updated: 2026-08-08 (Phase 1c session)_

## Status snapshot

| Phase | Content | Status |
|---|---|---|
| 0 | Scaffold, params.yaml, auth, notebook 00 | ✅ done + Colab-verified |
| 1 | AOI & boundaries | 🔄 Colab runs 1–2 done; schema now known, run 3 pending (see Phase 1c) |
| 2 | LST pipeline (Landsat + MODIS) | ⬜ |
| 3 | UHI metrics (SUHII, UTFVI) | ⬜ |
| 4 | Trend analysis (MK/Sen + FDR) | ⬜ |
| 5 | Spatial statistics (Gi*, Moran, EHSA, GWR) | ⬜ |
| 6 | Scenario projection (RF + CA-Markov) | ⬜ |
| 7 | Greening priority (MCDA/AHP) | ⬜ |
| 8 | Report figures | ⬜ |

## Phase 0 — done this session (2026-08-08)

- [x] Full repo skeleton per CLAUDE.md (dirs, `.gitignore`, `.gitkeep`s)
- [x] `requirements.txt` — Colab-compatible floors + conflict notes
- [x] `config/params.yaml` — every dataset id/band/scale/threshold/CRS, with
      provenance comments (`[CLAUDE.md]` / `[GEE catalog]` / `[DEFAULT]`)
- [x] `src/colombo_uhi/__init__.py` (`load_params`) + `auth.py`
      (idempotent `init_ee`, `resolve_project_id`, `ee_smoke_test`)
- [x] 12 docstring-only module stubs (no fake implementations)
- [x] `notebooks/00_setup_and_auth.ipynb` (functional) + 01–08 stubs
- [x] `tests/` — 30 tests, all passing locally (`python -m pytest tests/ -q`)
- [ ] **USER: run notebook 00 in Colab end-to-end** (see "What you must run")

## Phase 1c — Colab run 2 results + fixes (2026-08-08)

The diagnostic-first approach paid off: the new `describe_asset` cell printed
both assets' real schemas, so **nothing about the boundary data is guesswork any
more.**

### ✅ Asset schema — AUTHORITATIVE (verified in Colab, do not re-guess)

The uploads are **OCHA COD-AB v03 with LOWERCASE field names**:

| Need | Property | Example value |
|---|---|---|
| DS name (admin3) | `adm3_name` | `Colombo`, `Kolonnawa` |
| GN name (admin4) | `adm4_name` | `Sammanthranapura`, `Mattakkuliya` |
| Parent district | `adm2_name` | `Colombo` |
| Province | `adm1_name` | `Western` |
| P-codes | `adm2_pcode` / `adm3_pcode` | `LK11` / `LK1103` |
| Stated area | `area_sqkm` | Colombo DS = 24.54 km² |

`adm*_name1` is Sinhala and `adm*_name2` Tamil — never filter on those.
Colombo DS at 24.54 km² makes the Colombo + Thimbirigasyaya pair land near the
expected 37 km² CMC, independent support for that two-DS definition.

### Bugs found and fixed

4. **Candidate lists were all uppercase** (`ADM3_EN`, …), so none matched.
   The district column resolved to None → the code took its spatial fallback.
   Fixed: lowercase COD-AB names are now the **first** candidate in each list
   (pinned by a test), uppercase variants kept as fallbacks.
5. **The spatial fallback silently returned 0 features** — latent bug of mine.
   `ee.Geometry.contains()` yields an `ee.Boolean` that round-trips as JSON
   `true`, which `ee.Filter.eq(prop, 1)` does not match; last session I
   "hardened" that comparison from `True` to `1`, which was exactly backwards.
   Fixed by casting to `ee.Number` so the stored value is unambiguously 1/0,
   with a comment telling future readers not to simplify it back.

### Phase 1c decision (user-approved 2026-08-08)

7. **LCZ masks are scoped to Colombo District** (`uhi.suhii.lcz_based.scope`).
   Unscoped they spanned Western Province + 25 km, making "urban" 2464 km² of
   built-up LCZ across Gampaha and Kalutara — a regional statistic, not
   Colombo's. The LCZ method is now an intra-district built-vs-vegetated
   contrast while the buffer method stays CMC-vs-ring, so the two definitions
   stay genuinely independent. Caveat recorded in code: the LCZ rural reference
   sits closer to the core, so advection may damp its SUHII relative to the ring
   — that divergence is the sensitivity to report, not a defect.

### Figure review (run 2's PNGs, decoded from the notebook)

`viz.save_thumbnail` is **Colab-proven**. What the figures show:

- **Water mask** — ocean, **Bolgoda Lake**, and the Labugama/Kalatuwawa
  reservoirs in eastern Seethawaka are all clearly captured. Beira and Diyawanna
  are **not resolvable** at the district-wide scale (~50 m/px for a 0.65 km²
  lake), so a zoomed `aoi_water_mask_core.png` (10 km around the centre,
  ~22 m/px) was added to notebook 01 to verify them. Water is visibly excluded
  from the LCZ rural mask along the coast and lagoons.
- **Boundaries** — province/district/urban-extent nest correctly. The GHSL urban
  extent is scattered speckle across the whole province, confirming exactly why
  a ring based on it came out at 4049 km².
- **LCZ masks** — red/green covered the entire buffered province with the tiny
  district outline lost inside it: the visual proof of the scoping problem that
  decision 7 fixes.

## Phase 1b — Colab run 1 results + fixes (2026-08-08)

User uploaded `projects/research-uhi-484404/assets/lka_admin3` (DS) and
`.../lka_admin4` (GN) and ran notebook 01 end-to-end. **No crashes**; EE auth,
water mask, both rural references and the map all built.

| Check | Result | Verdict |
|---|---|---|
| Colombo District | 1 feature, **685.6 km²** (exp. 699) | ✅ |
| Western Province | 3 features (Colombo/Gampaha/Kalutara), **3761.7 km²** (exp. 3684) | ✅ |
| DS divisions | **339** (expected 13) | ❌ → fixed |
| GN divisions | **14043** (expected 557) | ❌ → fixed |
| CMC | **0.0 km²** (expected ~37) | ❌ → fixed |
| Urban extent (GHSL) | 510.7 km² | ✅ plausible |
| Rural buffer ring | **4049 km²** | ⚠️ → base changed |

### Bugs found and fixed

1. **No district filter.** 339 / 14 043 are Sri Lanka's *national* DS/GN counts
   — the assets are correct, but the loaders returned the raw country-wide
   collections. Fixed: `ds_divisions()` / `gn_divisions()` take
   `district_only=True` and filter by the asset's parent-district column, or —
   when the asset has none — by a centroid-within-GAUL-district spatial test.
2. **CMC returned 0 km² silently.** The `ADM3_EN` guess did not match, and an
   empty filter + `union()` yields an empty geometry indistinguishable from a
   real answer. Root cause: HDX serves **two** Sri Lanka boundary products with
   different schemas — OCHA COD-AB (`ADM3_EN`, `ADM2_EN`) and geoBoundaries
   (`shapeName`, no parent-district column) — and we cannot know which was
   downloaded. Fixed three ways:
   - `aoi.assets.*_candidates` lists in params; `_resolve_property()` picks the
     first present and otherwise **raises listing the asset's real property
     names** (one `getInfo` per asset per session, cached).
   - `cmc_boundary()` now **raises when zero DS divisions match**, printing the
     DS names actually present in Colombo District.
   - `aoi.describe_asset()` + a new first notebook cell print each asset's
     count, schema and sample values up front.
3. **Ring base.** 4049 km² because the ring was built around the province-wide
   GHSL urban extent (every built patch), not Colombo's core.

### Phase 1b decisions (user-approved 2026-08-08)

5. **`buffer_ring.base` → `"cmc"`** — textbook SUHII: a 15–25 km ring around the
   ~37 km² municipal core. `urban_extent()` stays as a map layer and an
   alternative base.
6. **Rural elevation cap = 100 m** (`uhi.suhii.rural_filters.max_elevation_m`,
   SRTM), applied to rural masks under **both** definitions, urban masks
   untouched. Rationale: the ring reaches inland relief (~50–150 m); at a
   ~6.5 °C/km lapse rate that is up to ~0.65 °C of elevation-driven cooling
   contaminating SUHII.

### Also added

- `src/colombo_uhi/viz.py` — `outline_image`, `elevation_backdrop`,
  `save_thumbnail`. Notebook 01 now writes **four PNGs to `figures/`**
  (`aoi_boundaries`, `aoi_water_mask`, `aoi_rural_lcz`,
  `aoi_rural_buffer_ring`). The `geemap.Map` widget renders nothing once a
  notebook is saved, so those PNGs are the reviewable evidence.
- `aoi.mask_area_km2()` + a notebook cell printing every mask's area — the guard
  against the elevation cap silently emptying the rural reference.
- Notebook 01 degrades gracefully: if the CMC still fails, the LCZ method, water
  mask and their figures still run.
- 82 tests passing (was 71).

## Phase 1 — done this session (2026-08-08)

- [x] `src/colombo_uhi/aoi.py` — district + Western Province (GAUL), DS/GN
      asset loaders with prominent-warning fallback, `cmc_boundary()` (dissolve
      of Colombo + Thimbirigasyaya DS divisions; actionable error while the DS
      asset is missing), GHSL-derived `urban_extent()`, combined water mask
      (MNDWI ∨ QA_PIXEL water frequency ∨ JRC occurrence, optional shoreline
      buffer), and BOTH rural references behind `rural_reference(method, params)`
      (`"buffer_ring"` / `"lcz_based"`).
- [x] `src/colombo_uhi/landsat.py` — Phase 2 front-load, deliberately minimal:
      `bits_to_mask`, `qa_clear_mask` (bits 0–4 + QA_RADSAT), `qa_water_flag`,
      `scale_sr`. The water mask needed them; Phase 2 extends this module.
- [x] `config/params.yaml` — `aoi.assets.ds_divisions`, `aoi.cmc.*`,
      `aoi.expected_areas_km2`, `aoi.urban_extent`, `aoi.water_mask`,
      `buffer_ring.base`, LCZ `rural_classes` → A–G (11–17).
- [x] `notebooks/01_aoi_and_boundaries.ipynb` — functional: boundaries, area
      sanity table, water mask (+60 m shoreline demo), both rural refs, geemap
      layer stack, HDX asset-upload instructions, visual checklist.
- [x] `tests/test_aoi.py` + `test_params.py` additions — 71 tests passing.
- [ ] **USER: run notebook 01 in Colab** (see "What you must run").
- [ ] **USER: source + upload DS/GN assets** (HDX admin3/admin4; instructions in
      notebook 01) — until then DS/GN fall back to the GAUL district and CMC
      raises its actionable error.

### Phase 1 decisions (user-approved 2026-08-08)

1. **GAUL depth**: Sri Lanka GAUL stops at ADM2 = district — the originally
   requested "fall back to DS divisions from GAUL" is impossible. Approved
   design: nullable asset ids (`aoi.assets.ds_divisions` / `gn_divisions`,
   source OCHA/HDX admin3/admin4); CMC = dissolve of Colombo + Thimbirigasyaya
   DS divisions; GAUL-district fallback + prominent warning meanwhile.
2. **LCZ rural default = A–G (classes 11–17)**, water mask then removes G and
   the shoreline; E (paved) kept knowingly (flagged in params comment).
   Configurable list — Phase 3 sensitivity runs swap it.
3. Western Province is built from the three verified ADM2 district names, not
   an ADM1 filter (exact GAUL `ADM1_NAME` string unverified).
4. `aoi.cmc.ds_name_property` defaults to `ADM3_EN` (OCHA/HDX admin3 field) —
   verify against the uploaded asset's attribute table.

## Dataset verification vs GEE catalog (Browser pane, 2026-08-08)

| Dataset | ID | Verdict |
|---|---|---|
| Landsat 5 TM C2 L2 | `LANDSAT/LT05/C02/T1_L2` | ✅ ST_B6, scales OK; 1984-03-16→2012-05-05 |
| Landsat 7 ETM+ C2 L2 | `LANDSAT/LE07/C02/T1_L2` | ✅ ST_B6; 1999-05-28→**2024-01-19 (ended)** |
| Landsat 8 C2 L2 | `LANDSAT/LC08/C02/T1_L2` | ✅ ST_B10; 2013-03-18→present |
| Landsat 9 C2 L2 | `LANDSAT/LC09/C02/T1_L2` | ✅ ST_B10; 2021-10-31→present |
| MODIS Terra LST | `MODIS/061/MOD11A2` | ✅ bands + ×0.02 + QC bits; 2000-02-18→present |
| MODIS Aqua LST | `MODIS/061/MYD11A2` | ✅ same; starts **2002-07-04** (no Aqua 2000–mid-2002) |
| Dynamic World | `GOOGLE/DYNAMICWORLD/V1` | ✅ 10 m, 2015-06-27→present, `label` 0–8 (built=6) |
| ESA WorldCover 2020 | `ESA/WorldCover/v100` | ✅ band `Map`, legend captured |
| ESA WorldCover 2021 | `ESA/WorldCover/v200` | ✅ band `Map` |
| GHSL built-up | `JRC/GHSL/P2023A/GHS_BUILT_S` | ✅ `built_surface`, 100 m, 1975–2030 |
| LCZ global map | `RUB/RUBCLIM/LCZ/global_lcz_map/latest` | ✅ `/latest` alias valid; `LCZ_Filter` recommended; snapshot 2018 |
| SRTM | `USGS/SRTMGL1_003` | ✅ `elevation`, 30 m |
| WorldPop | `WorldPop/GP/100m/pop` | ✅ `population`; **2000–2020 only** |
| VIIRS night lights | `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` | ⚠️ `avg_rad`; starts **2014-01**, not 2012 (see below) |
| Global Surface Water | `JRC/GSW1_4/GlobalSurfaceWater` | ✅ `occurrence`/`seasonality`/`max_extent`, 30 m |
| ASTER GED | `NASA/ASTER_GED/AG100_003` | ✅ `emissivity_band10–14` ×0.001, 100 m |
| ERA5-Land monthly | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | ✅ `temperature_2m` (K); also `skin_temperature` |
| FAO GAUL level 2 | `FAO/GAUL_SIMPLIFIED_500m/2015/level2` | ✅ `ADM0/1/2_NAME` props; district = ADM2 |
| NOAA GSOD | `NOAA/GSOD` | ❌ **NOT IN THE EE CATALOG** (page 404s; search: 0 hits) |

## ⚠️ Catalog discrepancies vs CLAUDE.md — user decisions needed

1. **`NOAA/GSOD` does not exist in Earth Engine** (CLAUDE.md lists it for
   Katunayake WMO 43450 station air temperature). Parked in `params.yaml`
   under `non_ee_sources`; no code will be written against it. Free options:
   (a) use **ERA5-Land `temperature_2m`** for air-temp validation (already in
   the stack), (b) download GSOD CSVs directly from NOAA NCEI and load with
   pandas, (c) BigQuery public dataset `bigquery-public-data.noaa_gsod`.
   **→ Decide before Phase 2 (validation design).**
2. **VIIRS**: CLAUDE.md says "2012+", but the stray-light-corrected
   `VCMSLCFG` series starts **2014-01**. 2012-04–2013-12 exists only in the
   uncorrected `VCMCFG` sibling. Default: use VCMSLCFG from 2014.
   **→ Only matters if 2012–13 night lights are needed (Phase 5/6).**
3. Minor, no action needed (recorded in `params.yaml` comments):
   - `QA_PIXEL` bit 2 is *Cirrus* only on L8/L9 (OLI); it is *Unused* on
     L5/L7 — always 0 there, so the shared bits-0–4 mask stays valid.
   - CLAUDE.md's QA bit list omits bits 12–13 (snow/ice confidence); full
     layout captured in params.
   - ST bands are **fully masked when `PROCESSING_LEVEL` = `L2SR`** → Phase 2
     must filter to `L2SP` (encoded in params).
   - WorldPop ends 2020; L5 ends 2012-05; L7 collection ended 2024-01.

## User inputs — resolved 2026-08-08

1. **EE Cloud project id**: `research-uhi-484404` → set in `params.yaml`
   (`gcp.ee_project_id`).
2. **GN-division boundaries**: user does **not** have a shapefile/GeoJSON.
   → Phase 1 will (a) build the district/DS-level AOIs from GAUL first, and
   (b) evaluate free GN sources before any GN-level analysis:
   OCHA/HDX "Sri Lanka — Subnational Administrative Boundaries" (admin-4 =
   GN level; licence + Colombo's 557-GN count must be verified) or Survey
   Dept / Census & Statistics layers. If no usable GN layer is found, the
   MAUP-sensitivity fallback is DS divisions (13 units) + a regular grid —
   a deviation from CLAUDE.md's GN plan that the user must sign off on.
3. **GitHub repo**: <https://github.com/Dineth0627/colombo_uhi> → baked into
   notebook 00 (`REPO_URL`) and README.
4. **Drive export folder**: `colombo_uhi_exports` confirmed (params
   `exports.drive_folder`).
5. Commit style: **no Claude co-author trailer** in commit messages.

Still open: GSOD replacement decision (discrepancy #1) — needed by Phase 2.

## What you must run to verify Phase 1 (run 3, after the 1c fixes)

1. Commit + push, then re-run `notebooks/01_aoi_and_boundaries.ipynb` in Colab.
2. Confirm: DS = **13**, GN = **557**, CMC ≈ **37 km²** and close to the
   `area_sqkm` sum printed next to it, ring compact around the CMC, LCZ urban a
   district-scale number (not 2464 km²) with a non-empty rural counterpart, no
   mask flagged near-empty.
3. If CMC still fails, the printed DS-name list shows the exact spellings —
   paste the right two into `aoi.cmc.ds_division_names`.
4. **Send back the PNGs from `figures/`** — especially the new
   `aoi_water_mask_core.png`, which is the only view that can resolve Beira Lake
   and Diyawanna Lake.
5. Locally: `python -m pytest tests/ -q` → 84 passed.

## What you must run to verify Phase 0

1. Push the repo to GitHub (from the repo root):

   ```
   git add -A
   git commit -m "Phase 0: scaffold, params, EE auth, setup notebook"
   git remote add origin https://github.com/Dineth0627/colombo_uhi.git
   git branch -M main
   git push -u origin main
   ```

2. Open `notebooks/00_setup_and_auth.ipynb` in Colab (README quick-start has
   the direct link) and run all cells. Success = final cell prints
   `Earth Engine session OK` with `one_plus_one == 2` and
   `srtm_bands == ['elevation']`.
3. Optionally run `python -m pytest tests/ -q` locally (should be 30 passed).

## Session log

- **2026-08-08 — Phase 0**: scaffolded repo; verified 18 datasets in the GEE
  catalog Browser pane (17 ✅, 1 ❌ GSOD); wrote params.yaml, auth module,
  tests (30 passing), notebook 00, stubs, README. No LST processing code
  written (per session scope). Nothing committed to git yet.
- **2026-08-08 — Phase 1c**: second Colab run. Diagnostic cell revealed the
  assets are COD-AB v03 with **lowercase** fields (`adm3_name`/`adm4_name`/
  `adm2_name`) — my candidates were uppercase, so DS/GN came back 0/0. Also
  fixed a latent `ee.Boolean` vs `1` filter bug that made the spatial fallback
  match nothing. Scoped both LCZ masks to Colombo District per user decision.
  Reviewed run 2's PNGs (water mask confirmed over ocean/Bolgoda/reservoirs;
  added a zoomed core figure for Beira + Diyawanna). 84 tests passing.
- **2026-08-08 — Phase 1b**: first Colab run of notebook 01. Boundaries/areas
  correct; found 3 bugs (no district filter on the uploaded assets → 339/14043;
  silent 0 km² CMC from an unmatched attribute name; ring based on the
  province-wide GHSL extent → 4049 km²). Fixed with candidate-based schema
  resolution + loud failures, district filtering (attribute or centroid
  fallback), `base: "cmc"`, and a 100 m SRTM cap on rural masks. Added `viz.py`
  + four persistent PNGs and `mask_area_km2` emptiness guard. 82 tests passing.
- **2026-08-08 — Phase 1**: study-area layer. Confirmed GAUL has no DS/GN/CMC
  for Sri Lanka → asset-slot design approved by user; LCZ rural set to A–G.
  Wrote `aoi.py` (boundaries, urban extent, water mask, dual rural refs),
  minimal `landsat.py` QA helpers, notebook 01, params additions, tests
  (71 passing). Pending user: Colab run of notebook 01 + HDX DS/GN upload.
- **2026-08-08 — Phase 0 config update**: user supplied EE project id
  (`research-uhi-484404` → params.yaml), GitHub URL (→ notebook 00 + README),
  confirmed Drive folder; user has NO GN boundary data (Phase 1 sourcing plan
  recorded above); commits must carry no co-author trailer. Notebooks
  regenerated; 30 tests still passing.
