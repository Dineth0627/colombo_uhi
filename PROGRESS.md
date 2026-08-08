# PROGRESS — Colombo UHI practicum

_Last updated: 2026-08-08 (Phase 2 rev 6 — notebook runs end to end; night QC + offset decomposition pending run 6)_

## Status snapshot

| Phase | Content | Status |
|---|---|---|
| 0 | Scaffold, params.yaml, auth, notebook 00 | ✅ done + Colab-verified |
| 1 | AOI & boundaries | ✅ **done + Colab-verified** (run 5, 5 iterations) |
| 2 | LST pipeline (Landsat + MODIS) | 🟡 **runs end to end (run 5); rev 6 fixes night QC + offset decomposition — 249 tests passing** |
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

## 🟡 PHASE 2 — LST pipeline; runs end to end, two findings pending run 6

`landsat.py`, `indices.py`, `composites.py`, `modis.py`, a `viz.py` addition and
`notebooks/02_lst_pipeline.ipynb` are complete. **249 tests pass locally** (was 90);
every module still imports without `earthengine-api`. The notebook **runs end to end**
as of Colab run 5. Two findings from that run are fixed in rev 6 and need one more run
to confirm: MODIS night-time LST (empty under strict QC) and the Landsat-vs-MODIS
offset (not yet attributable).

Sections below are in reverse chronological order: latest run first, then the
as-designed record from before any run.

### Catalog re-verification (CLAUDE.md requires it before writing code)

Re-checked on the GEE catalog 2026-08-08 for the Phase 2 specifics. **No discrepancies**
with CLAUDE.md or params for: L5/L7 `ST_B6`, L8/L9 `ST_B10`; `ST_QA`/`ST_EMIS`/`QA_PIXEL`/
`QA_RADSAT`/`PROCESSING_LEVEL` on **all four** collections (confirmed on LT05, not only
LC08); ST `0.00341802`/`149`; SR `2.75e-05`/`-0.2`; `ST_QA` `0.01` K; TM vs OLI band
numbering; MODIS QC bits 0-1 (mandatory QA) and 6-7 (LST error), LST scale `0.02`.

Three **new** facts, now in params:

1. **`Clear_sky_days`/`Clear_sky_nights` are 8-bit BITMASKS, not counts** — one bit per
   day of the 8-day window. A fully clear period reads as **255**, not 8. `modis
   .clear_sky_count()` does the popcount. This one would have produced a wrong number
   that looked like a number.
2. **MODIS `LST_*_1km` valid DN range starts at 7500** (fill 0) → `modis_lst
   .valid_dn_range`, gated before scaling. 7500 × 0.02 = 150 K, the same physical floor
   as the Landsat DN gate.
3. `QA_RADSAT` bit 11 is **terrain occlusion**, not saturation, so `QA_RADSAT == 0` is
   marginally stricter than CLAUDE.md's wording. Harmless on flat coastal Colombo;
   recorded in params so nobody "fixes" it.

Also confirmed: Aqua `MYD11A2` runs 2002-07-04 → current, QC layout identical to Terra.

### Phase 2 decisions (user-approved 2026-08-08)

1. **Valid-observation floor: flag, never mask.** Every composite emits `obs_count`;
   no pixel is dropped. `composites.min_valid_obs: null` so Phase 4 sets its own floor.
2. **ST_QA uncertainty filter off by default**, `ST_QA_K` band always emitted
   (`landsat_c2l2.st_qa_max_kelvin: null`) — pick a threshold from the real distribution
   in Colab, not from a guess.
3. **Air-temperature validation deferred to Phase 3.** Phase 2 validates Landsat LST
   against MODIS LST. `NOAA/GSOD` stays parked under `non_ee_sources` (Phase 0
   discrepancy #1 remains open, but is no longer blocking).
4. **Albedo — DECISION REVERSED after design review; needs your sign-off.** You approved
   *per-sensor* coefficient sets on my recommendation, for the stated purpose of avoiding
   a step change at the 2013 L7→L8 transition. That recommendation was wrong on the
   facts: Liang (2001) predates OLI by twelve years, so there is **no separate "Liang OLI"
   fit** — the universally-cited Landsat 8 form is the *same* coefficients applied to the
   wavelength-matched OLI bands, which our `blue…swir2` rename already performs.
   Substituting a genuinely different published fit (Silva 2016) at 2013 would **create**
   the discontinuity the decision existed to prevent. Shipped as **one set
   (`liang2001`) across all four sensors**, with Silva available as
   `indices.albedo.sets.silva2016_oli` for sensitivity runs — flip
   `indices.albedo.active_set` if you disagree.

### Design-review corrections applied before the first run

A review pass against the installed `earthengine-api` source caught several things that
would have cost Colab round trips:

- **Empty-year composites.** Reducing a year with no scenes yields a *band-less* image
  that dies at the next `select()`. Originally fixed with an `ee.Algorithms.If`
  placeholder; **superseded after run 1** by `composites._padding_collection()` (see
  the run 1 notes above) because `If` evaluates both branches. Either way the year
  axis stays complete for Phase 4 Mann-Kendall, with `obs_count == 0`.
- **`neq("L2SR")`, not `eq("L2SP")`.** `neq` is `eq().Not()`, so a renamed
  `PROCESSING_LEVEL` fails **open** instead of silently emptying the collection.
  Notebook 02 prints filtered vs unfiltered counts per sensor as the backstop.
- **`sharedInputs=True`** on every `Reducer.combine`, and `.select([band])` *before*
  `.reduce(...)` — otherwise a single-input reducer runs on all 8 bands (24 outputs).
- **`ee.Image()` cast around `copyProperties`**, which returns an `ee.Element` and would
  otherwise fail inside `.map()`.
- **SLC-off off-by-one.** `filterDate`'s end is exclusive and `slc_off_after` is the last
  SLC-*on* day, so the code advances one day.
- **`month_filter` moved into `landsat.py`** with `aoi._month_filter` delegating to it —
  `aoi` already imports `landsat`, so the shared helper had to move *down* the dependency
  arrow or the import would be circular.
- **`time.season_partition`** added: `monsoon_season()` must iterate a declared partition,
  because `time.seasons` also holds `dry_window` (Jan-Mar), which overlaps `ne_monsoon`
  and would map January to two seasons.
- **`viz.plot_annual_lst_comparison` avoids pyplot entirely** (object-oriented
  `Figure` + `FigureCanvasAgg`); forcing a backend would break Colab's inline rendering
  for every later cell.

### Colab run 1 (2026-08-08) — `User memory limit exceeded`, three real bugs

Run 1 got through Steps 1-2 (collection built, both scene inventories printed)
and died on `recent.bandNames().getInfo()` in the dry-season cell. Not a quota
problem — three genuine design faults in my code, all now fixed:

1. **`ee.Algorithms.If` evaluates BOTH branches.** I used it to substitute a
   correctly-shaped placeholder for years with no scenes, so every one of the 26
   composites carried the placeholder graph *and* the real reduce. Replaced by
   merging a single fully-masked image into each year's collection
   (`composites._padding_collection`): the reducer then always emits the full
   band set, and because masked pixels are skipped by median, percentile and
   count alike, `obs_count` stays honest at 0 for an empty year. Cheaper and
   simpler than the construct it replaces.
2. **The notebook built all 26 years to use one.** `dry_season_composites(...)`
   then `.filter(year == 2025)` forces Earth Engine to evaluate every year's
   graph just to read its `year` property. Now composites the mapped year alone
   via `start_year == end_year`.
3. **Everything ran over `analysis_region`** (Western Province + 25 km) when the
   district would do. `aoi.water_mask()` gained an optional `region` argument —
   it builds a Landsat reflectance composite internally, so its scope dominates
   the cost of the whole notebook. Default unchanged, so the verified Phase 1
   numbers still reproduce exactly.

Also: `bandNames().getInfo()` forces evaluation of the whole image graph purely
to learn names that `composites.composite_band_names()` already knows
client-side. The notebook now uses the helper, and fetches the four scene
properties in one metadata-only `ee.Dictionary` round trip.

New notebook knobs, both documented in the notebook header: `WORK_REGION`
(shrink first if memory fails) and `ZONAL_SCALE_M` (100 m for the 26-year
comparison series; a spatial mean over the ~47 km2 CMC is insensitive to it, and
the scale is stated alongside the number exactly as Phase 1 does for CMC area).

`modis.annual_lst()` gained `region` (bounds the work) alongside `geometry`
(bounds the output).

### Colab run 2 (2026-08-08) — memory again, this time in the comparison series

Run 2 got through Steps 1-5 (collection, inventories, dry-season map,
observation counts, MODIS) and died in `zonal_annual_means` on the Landsat
series. Cause: **one request cannot hold 26 annual composite graphs at once.**
The single-round-trip design was right for alignment and wrong for memory. Two
fixes, neither of which changes any number:

1. **`annual_composites(..., with_percentile=False)`** — the biggest saving. A
   percentile reducer must retain every observation per pixel in order to sort
   them; mean and count are streaming accumulators. The comparison plot uses
   only the mean, so `p90` was pure cost across 26 years. The percentile is
   still produced by default everywhere else (the dry-season map keeps it).
2. **Batched fetching** in `zonal_annual_means` (`composites.zonal_batch_years`,
   default 4, `batch_years=` to override, 1 is valid). Batches are cut on a
   client-side year list, so no extra round trip is needed to discover them, and
   the alignment guarantee that motivated the single-request design is preserved
   *within* each batch — a fully-masked year still comes back as a `NaN` row.

The notebook also splits the Landsat and MODIS series into separate cells, so a
failure in one does not cost the other, and its header now lists the four memory
levers in order of effect.

**Lesson for Phases 4-7**, where 26-year series get much heavier: assume any
whole-series `getInfo` will exceed the memory limit, and reach for
drop-unneeded-reducers → batch → narrow region → coarsen scale, in that order.
Only the last one changes the numbers.

### ✅ Colab run 3 (2026-08-08) — Steps 1-5 VERIFIED

Everything up to the comparison plot now works. Verified outputs:

| Check | Result | Verdict |
|---|---|---|
| Scenes over the district, 2000-2025 | **1674** | ✅ |
| `L2SP only` vs unfiltered | 214/214, 402/402, 794/794, 264/264 | ✅ filter is a **no-op** here |
| Scene inventory | TM ends 2011, OLI from 2013, OLI-2 from 2021, ETM+ ends 2024 | ✅ as predicted |
| Dry-window years with zero scenes | **none** (min 2 in 2000, typically 15-35) | ✅ better than feared |
| Dry-season 2025 composite | 30 scenes, bands `LST_C`/`LST_C_p90`/`obs_count` | ✅ |
| `obs_count` over CMC, dry 2025 | **min 4, median 10, max 14** | ✅ healthy — no zero-coverage pixels |
| Terra day 8-day granules | 1189 | ✅ |
| LST + obs-count PNGs | rendered | ✅ |

Two findings worth carrying forward:

1. **The `PROCESSING_LEVEL` filter removes nothing** over this AOI — all four
   sensors report identical counts filtered and unfiltered. It stays enabled
   (harmless, and it fails open), but it is not load-bearing here.
2. **Landsat 5 shows 0 scenes in 2012**, not a partial year. Correct: L5 stopped
   *acquiring* in November 2011 after the electronics failure; `2012-05-05` is
   the collection's end timestamp, not its last acquisition over Sri Lanka. So
   the L7-only gap is really **2012-01 to 2013-03**, wider than CLAUDE.md's
   "2012-05" implies. Relevant to Phase 4: that stretch of the series rests on
   SLC-off ETM+ alone.

### ✅ Colab run 5 (2026-08-08) — notebook 02 runs end to end

Step 6 completed. Everything mechanical in Phase 2 now works; what remains is
driven by the numbers, not the plumbing.

| Check | Result | Verdict |
|---|---|---|
| Water mask: static (JRC) vs combined, CMC mean 2025 | **−0.074 °C** | ✅ the cheap mask is a fair substitute |
| Landsat zonal series, 26 years | complete, 4206 valid pixels/yr | ✅ |
| MODIS Terra day | 31.7–35.1 °C | ✅ plausible |
| MODIS Aqua day | 35.5–39.0 °C | ✅ warmer than Terra day, as expected at 13:30 |
| **MODIS Terra + Aqua NIGHT** | **all 26 years empty** | ⛔ see below |
| Index means over CMC (Jan–Mar 2025) | NDVI 0.372, NDBI 0.025, MNDWI −0.415, EVI 0.196, SAVI 0.192, albedo 0.124 | ✅ all physically plausible |

### ⛔ Two defects run 5 exposed, and the rev 6 fixes

**1. MODIS night LST was completely empty** — 0 valid pixels, all 26 years, both
satellites, while day worked. The code path is identical apart from band names,
so the cause is the QC policy: tropical night retrievals rarely reach "good
quality AND avg error ≤ 1 K". Strict night QC does not produce a conservative
answer, it produces **no** answer — and MODIS is the only night-time source
(CLAUDE.md caveat 4), so night-time UHI would have been unobtainable.

**2. The pipeline stayed silent about it.** `zonal_annual_means_by_year` returned
26 rows of `None` with `valid_pixels == 0` and no warning. Emitting a count that
nobody reads does not satisfy caveat 2 — an all-empty product is a defect, not a
datum. `composites._warn_if_series_is_empty` now warns loudly for a fully empty
series and for a majority-empty one, while still returning the frame (downstream
needs the shape, and swallowing it would trade one silent failure for another).

### Phase 2 decision 7 (user-approved 2026-08-08) — night QC deviation

**`modis_lst.qc_filter` is now split by overpass.** Day stays CLAUDE.md-strict
(mandatory QA ≤ 0, LST error ≤ 0). **Night is relaxed to mandatory QA ≤ 1 and
LST error ≤ 2 (≤ 3 K)** — a deliberate, documented deviation from CLAUDE.md,
justified by the measured fact above and recorded in the params comment.

**This asymmetry must travel into Phase 3:** night LST is accepted at up to 3 K
stated uncertainty against ≤ 1 K for day, so night-time SUHII is weaker evidence
than daytime SUHII and must never be reported as an equal-confidence pair.

`modis.qc_class_histogram()` was added so the next run *proves* which bit field
was doing the killing rather than leaving it to inference. If the histogram does
not show night mass above the strict ceiling, the night policy is the wrong fix.

### Open question for run 6 — the Landsat/MODIS offset

Landsat reads ~39–41 °C against Terra day ~32–35 °C **at the same overpass
time**, and warmer than Aqua day (13:30, nearer peak heating), which is the wrong
way round. Meanwhile the MODIS CMC means rest on only **13–23** 1 km pixels
(Terra day), as few as **2** in 2023 for Aqua. Three causes are tangled:

1. too few pixels → tested by rerunning over Colombo District (~700 MODIS pixels);
2. resolution/mixing → tested by reducing Landsat at 1000 m over the same polygon;
3. genuine sensor/emissivity difference → whatever offset survives both.

Notebook 02 now runs all three and prints the decomposition. **No offset should
be quoted in the report until that table comes back.** The figure also gained a
second panel plotting valid-pixel counts on a log scale, because nothing in the
top panel revealed that one point rested on 2 pixels and another on 4200.

Separately, the ~40 °C annual means are not implausible but must be presented
carefully: they are clear-sky, ~10:30, 30 m **land surface** temperatures over a
dense urban core. The clear-sky sampling bias is uncorrected anywhere in this
pipeline and belongs in the report.

### Colab run 4 — the actual cause: filtering a computed collection

`BATCH_YEARS = 1` still failed, which ruled out sheer volume and exposed the
real bug. **Batching by `ee.Filter` saves nothing.** An annual series is built
with `ee.ImageCollection.fromImages(ee.List.map(...))`; calling
`.filter(year == 2000)` on it forces Earth Engine to materialise **all 26**
composite graphs just to evaluate the predicate on each one. Every "batch" was
therefore the full 26-year computation plus a filter.

This is exactly the trap I had already identified and fixed in Step 4 (build one
year rather than filter 26) and then failed to apply to the batching itself.

Fixed by splitting the API so the distinction is impossible to miss:

* **`composites.zonal_annual_means_by_year(source, ...)`** — takes the SCENE
  collection and calls `annual_composites(start_year=lo, end_year=hi)` once per
  batch, so only those years' graphs are ever constructed. It also owns the
  compositing knobs (`reducer`, `months`, `with_percentile`, `mask`) because it
  builds the composites, and `progress=True` prints per batch so a slow run
  visibly advances. This is what any long series should use.
* **`composites.zonal_annual_means(collection, ...)`** — unchanged single-request
  version for collections that are already small. `batch_years` was **removed**
  from it rather than left as a false promise, and its docstring now carries the
  warning. A test pins that it takes no batching argument.

**Rule for Phases 4-7:** never subset a computed `ee` collection with `.filter()`
to save work — it does the opposite. Rebuild the subset you want.

### Colab run 3 — Step 6 exceeded memory; rev 4 changes

Batching to 4 years was not enough. The cost I had not accounted for: **a mask
built by compositing gets embedded into every image it masks.**
`aoi.water_mask` internally composites ~100 Landsat scenes, so masking 26 annual
images instantiated that composite 26 times, on top of the annual reductions
themselves. Three further changes:

1. **`aoi.static_water_mask()`** — JRC Global Surface Water occurrence alone, a
   single static image. For permanent water (ocean, Port harbour, Beira, Kelani)
   it agrees closely with the combined mask; it misses seasonal and shallow
   water. Used for long series only; `water_mask` still backs the maps. **This is
   the one change that moves a number**, so notebook 02 now measures the
   difference on the 2025 CMC mean and prints it, rather than assuming it away.
2. **`harmonised_collection(include_sr=False, include_st_qa=False)`** — an
   LST-only collection. Scaling and renaming six reflectance bands on ~1670
   scenes is pure weight when only `LST_C` is wanted.
3. `BATCH_YEARS` lowered to 2, and MODIS masking moved from Step 5 to Step 6 so
   every plotted series is masked identically.

### Known limitations carried into Phase 3+

1. **`LST_C_p90` is near-max at low observation counts.** With a handful of clear scenes
   per pixel per year, the 90th percentile is close to the sample maximum. Use it as
   hot-tail behaviour, not a stable statistic; Phase 4 should fit trends on the
   central-tendency band under an `obs_count` floor.
2. **Calendar-year slicing breaks wrapping seasons.** `annual_composites(months=[12,1,2])`
   groups December with the January/February of the *same* calendar year, which is not
   one continuous NE monsoon. The Jan-Mar dry window is unaffected. Documented in the
   docstring; fix it properly if Phase 3 wants NE-monsoon composites.
3. **Reducer mismatch is a comparison trap.** Landsat composites default to `median`,
   MODIS to `mean`. Notebook 02 uses `mean` for **both** sides of the comparison plot so
   the offset is not partly a reducer artefact. Keep that discipline in Phase 3.
4. **Clear-sky sampling bias is uncorrected.** Every LST number here is a clear-sky
   value; the true annual mean is cooler. Not corrected anywhere in the pipeline.
5. **C2 inter-calibration is assumed, not yet verified.** `landsat_c2l2.harmonisation:
   none` per CLAUDE.md. Verify empirically on the overlap years (2000-2012 L5/L7,
   2021-2024 L8/L9) before quoting cross-sensor trends.

### What you must run in Colab (Phase 2)

1. Push, then open `notebooks/02_lst_pipeline.ipynb` in Colab and run top to bottom.
2. Report back with: the two scenes-per-year tables, the `L2SP only` vs unfiltered
   counts, the observation-count statistics, and the three PNGs
   (`figures/lst_dry_season_2025.png`, `figures/lst_obs_count_2025.png`,
   `figures/lst_landsat_vs_modis_cmc.png`).
3. Success looks like: explainable gaps only (L7-only 2012-05→2013-03, no Aqua before
   2002-07); a warm CMC core with water absent; non-zero `obs_count` across the CMC;
   Landsat and MODIS Terra-day tracking in shape with a reportable offset.

## ✅ PHASE 1 SIGNED OFF — verified reference values (Colab run 5, 2026-08-08)

These are the authoritative Phase 1 outputs. Quote them in the report; re-running
notebook 01 should reproduce them.

| Quantity | Verified | Note |
|---|---|---|
| Colombo District | **685.6 km²**, 1 feature | GAUL 500 m-simplified (gazetted 699) |
| Western Province | **3761.7 km²**, 3 features | Colombo + Gampaha + Kalutara |
| DS / GN divisions in district | **13** / **557** | both match CLAUDE.md exactly |
| CMC name audit | **55/55 names, 55 features** | `missing` and `extra` both empty |
| CMC administrative | **47.07 km²** | = the DS pair (46.87 stated); the 55 GN divisions tile it exactly |
| CMC land @ 30 m | **40.18 km²** | +7.7% vs gazetted 37.31 |
| CMC land @ 300 m | **37.70 km²** | +1.0% — same quantity, coarser grid |
| water inside CMC | **6.89 km²** | Port harbour + Beira + Kelani mouth |
| Rural buffer ring (geometry) | **1603.5 km²** | 15–25 km annulus around the CMC |
| rural mask — buffer_ring | **206.1 km²** | after water, built-up, 100 m cap |
| urban mask — buffer_ring | **37.7 km²** | water excluded from urban masks too |
| urban / rural mask — LCZ | **458.5** / **152.2 km²** | district-scoped |
| water mask (analysis region) | **3700.2 km²** | mostly ocean, as expected |

Both Phase 1e bugs are confirmed fixed in the strongest available way: the ring
(1603.5) and its rural mask (206.1) returned to run 3's values **to the decimal**,
which only happens if the stray duplicate-name polygons are gone. `extra` coming
back empty independently proves the 55-name CMC list is complete and correct.

Figures all reviewed: `aoi_water_mask_core.png` confirms **Beira**, **Diyawanna**
and the **Kelani**; `aoi_water_mask.png` confirms ocean, **Bolgoda** and the
Labugama/Kalatuwawa reservoirs; `aoi_boundaries.png` shows one compact CMC nested
correctly; `aoi_rural_buffer_ring.png` shows a single coastal core with a
landward-only ring stopping short of the high ground.

### ⚠️ Caveats that must travel into later phases

1. **CMC area is scale-dependent** (40.18 km² @30 m vs 37.70 @300 m). Always
   quote the reduction scale. The residual over the gazetted 37.31 km² is COD-AB
   polygon generalisation plus the `aoi.water_mask` thresholds — report it as
   sensitivity, never tune it away.
2. **GN names are NOT unique within Colombo District.** Every GN-level filter must
   be scoped to its parent DS division (or use `adm4_pcode`). Directly relevant to
   the GN-level zonal statistics and MAUP work in **Phases 5–7**.
3. The two rural definitions differ **by design** (CMC-vs-ring vs
   built-vs-vegetated inside the district), so their SUHII values will differ.
   Reporting both is the CLAUDE.md requirement, not a discrepancy to resolve.
4. Still open from Phase 0: **the GSOD replacement decision** for air-temperature
   validation (ERA5-Land / NCEI CSV / BigQuery) — needed by Phase 2.

## Phase 1e — Colab run 4 results + fixes (2026-08-08)

The audit cell paid for itself immediately: **50/55 names matched**, and it printed
COD-AB's own spellings for the five that did not.

### ✅ CMC AREA — RESOLVED (do not revisit)

Three facts, established arithmetically:

1. The `extra` list came back as *exactly* the five correct spellings and nothing
   else ⇒ the Colombo + Thimbirigasyaya DS divisions contain **precisely the 55**
   GN divisions on the CMC's list. **GN-union ≡ DS-pair — the same polygon.**
   Switching units was never going to change the area.
2. **46.9 km² is COD-AB's polygon, and it is not wrong** — it encloses the
   **Colombo Port outer harbour**, plainly visible in
   `figures/aoi_water_mask_core.png` (run 3). The excess over the gazetted
   37.31 km² is **9.56 km²**, which matches the harbour's size.
3. Therefore the **land-only** area is the figure to compare against 37.31, and
   it is also what LST statistics actually cover. New `aoi.cmc_land_area_km2()`
   measures it; run 5 confirms the number. This doubles as an independent
   validation of the water mask.

`aoi.expected_areas_km2` now carries **both**: `cmc: 37` (checked against land)
and `cmc_administrative: 47` (the raw polygon, legitimately larger).

### Bugs found and fixed

6. **Five spelling errors** in my transcription from the CMC map. Corrected in
   params and pinned by a test (reverting any would silently undersize the CMC):
   Ibanwala→**Ibbanwala**, Kettarama→**Khettarama**, Kirulapona→**Kirulapone**,
   Kotehena East/West→**Kotahena East/West**.
7. **GN names are NOT unique within Colombo District** — a genuine bug of mine.
   Matching names across all 557 district GN divisions also pulled in same-named
   divisions from Dehiwala/Moratuwa/Kolonnawa. The tell was arithmetic: 50 of 55
   units measured **47.50 km²**, *more than both parent DS divisions combined*
   (46.87) — children cannot exceed their parents. Downstream corroboration: the
   ring inflated 1603 → **2100 km²** and its rural mask 206 → **468 km²**, and
   `figures/aoi_rural_buffer_ring.png` shows stray red fragments inland, each
   generating its own annulus. Fixed: `_cmc_units()` scopes GN selection to the
   parent DS divisions (`aoi.cmc.parent_ds_scope`) before name matching, and
   `cmc_boundary()` now warns when the matched **feature count** exceeds the
   expected division count. **This trap applies to all GN-level work in Phases
   5–7** (zonal statistics, MAUP) — scope by parent DS or use `adm4_pcode`.
8. **`urban_mask()` did not exclude water** while `rural_mask()` did. With a
   coastal CMC enclosing ~10 km² of harbour, open water would have dragged the
   urban LST mean down and inflated SUHII — and CLAUDE.md requires water masked
   before *any* LST statistic. Water is now excluded from urban masks under both
   definitions. The elevation cap stays rural-only (urban cores are not
   elevation-matched by construction).

## Phase 1d — Colab run 3 results + CMC redefinition (2026-08-08)

Run 3 verified **everything except the CMC**:

| Check | Run 3 | Verdict |
|---|---|---|
| DS / GN in Colombo District | **13** / **557** | ✅ exact |
| Colombo District / Western Province | 685.6 / 3761.7 km² | ✅ |
| Rural buffer ring | 1603 km² (was 4049) | ✅ proper 15–25 km annulus |
| Rural ring mask after exclusions | 206 km² | ✅ usable sample |
| LCZ urban / rural (district-scoped) | 477 / 152 km² | ✅ sums to the district |
| **CMC (DS pair)** | **47.1 km²** vs ~37 | ⛔ wrong definition |

### ✅ CMC DEFINITION — AUTHORITATIVE (do not revisit)

**CMC = union of the 55 GN divisions listed by the Colombo Municipal Council's
own GIS Unit** ("GN DIVISIONS" map, CMC GIS Unit / ID Center, supplied by the
project owner 2026-08-08). Encoded in `aoi.cmc.gn_division_names`;
`aoi.cmc.definition: "gn_union"`.

**The DS-pair definition is disproven arithmetically:** Colombo DS = 24.54 km²,
Thimbirigasyaya DS = 22.33 km², sum **46.87 km²** — and no other combination of
COD-AB DS polygons gives 37.31 km². The computed geometry (47.1 km²) agreed with
the asset's own `area_sqkm`, so the code was never wrong; the *definition* was.

**Why the DS pair over-covers by ~10 km²:** `figures/aoi_water_mask_core.png`
from run 3 shows COD-AB's Colombo DS polygon enclosing the **Colombo Port outer
harbour and breakwaters** — a large semicircle of open sea counted as CMC. The
GN divisions do not include the port waters. `ds_union` is kept in params only as
a sensitivity variant, with that reason recorded.

**Guard against silent partial matches:** the risk with 55 hand-transcribed names
is spelling drift vs COD-AB (Wellawatta/Wellawatte, Kettarama/Khettarama). New
`aoi.cmc_name_audit()` reports `matched` / `missing` / `extra` in both directions
plus the asset-stated area; `cmc_boundary()` raises on zero matches and **warns
loudly on a partial match** (an undersized CMC would look plausible). Notebook 01
runs the audit before building the geometry.

### Operational trap fixed: stale modules in a live Colab runtime

Run 4 failed immediately with
`AttributeError: module 'colombo_uhi.aoi' has no attribute 'cmc_name_audit'`.
Not a code bug — re-running a notebook in a **live** runtime keeps the previous
run's modules in `sys.modules`, so `git pull` updates the files on disk while the
import silently returns the OLD code. New functions look absent and fixed bugs
look unfixed. This will recur every phase, so it is now handled structurally:

- Both notebooks **purge `colombo_uhi*` from `sys.modules`** immediately before
  importing it (the clone cell's `git pull` then actually takes effect).
- The clone cell prints `HEAD <sha> <subject> <date>` — quote this whenever a
  result looks impossible.
- Notebook 01 asserts the functions it needs exist and, if not, raises with the
  fix steps and the loaded module's `__file__` instead of a bare AttributeError.
- Notebook 00's troubleshooting table has the symptom and remedy.

Any Phase 2+ notebook must copy the purge block.

### Figure review (run 3 PNGs, decoded from the notebook)

- **`aoi_water_mask_core.png`** (new, ~22 m/px) — settles the water checklist:
  **Beira Lake**, **Diyawanna (Parliament) Lake** and a continuous **Kelani
  River** are all captured. Also the figure that revealed the port-in-CMC problem.
- **`aoi_rural_buffer_ring.png`** — textbook result: compact red CMC core, clean
  green annulus 15–25 km out on the **landward side only** (the seaward half is
  removed by the water mask), thinning out before the high ground (elevation cap
  working). Green is sparse because built-up LCZ is excluded and the ring crosses
  the conurbation — expected, and 206 km² is a workable sample.
- **`aoi_rural_lcz.png` / `aoi_boundaries.png`** — district-scoped LCZ masks now
  sit inside the district outline; boundaries nest correctly.

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

## Phase 1 → Phase 2 handover

**Nothing is required to close Phase 1** — run 5 verified it. Optionally re-run
notebook 01 once to capture the refreshed sign-off table and the paired
30 m/300 m land-area lines in a saved notebook for the report appendix.

One decision is outstanding before Phase 2 code is written:

- **GSOD replacement** for air-temperature validation. `NOAA/GSOD` is not in the
  Earth Engine catalog (Phase 0 discrepancy #1). Options: (a) ERA5-Land
  `temperature_2m`, already in the stack and in `datasets`; (b) NOAA NCEI GSOD
  CSVs for Katunayake (WMO 43450) loaded with pandas; (c) BigQuery public
  `bigquery-public-data.noaa_gsod`. Needed for Phase 2's validation design.

Phase 2 scope (`02_lst_pipeline.ipynb` plus `src/colombo_uhi/landsat.py`,
`modis.py`, `composites.py`): harmonised L5/L7/L8/L9 C2 L2 LST with `L2SP`
filtering and the standard QA mask — `landsat.py` already carries
`bits_to_mask`, `qa_clear_mask`, `qa_water_flag` and `scale_sr` from Phase 1;
MOD11A2/MYD11A2 with explicit `QC_Day`/`QC_Night` filtering; annual and
dry-season composites, each shipping a **per-pixel valid-observation count**
(CLAUDE.md caveat 2).

Locally: `python -m pytest tests/ -q` → **90 passed** at the end of Phase 1;
**222 passed** after Phase 2.

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

- **2026-08-08 — Phase 2 rev 6**: notebook 02 ran end to end (run 5). Two defects in the
  results: MODIS **night** LST was empty for all 26 years on both satellites under the
  strict QC, and the pipeline **said nothing** about a 100%-empty series. Split
  `qc_filter` by overpass (day stays CLAUDE.md-strict; night relaxed to QA ≤ 1 /
  error ≤ 3 K as a documented deviation, with the uncertainty asymmetry flagged for
  Phase 3), added `modis.qc_class_histogram` so the next run proves the cause, and made
  `build_zonal_frame` warn on empty/mostly-empty series. Also added the offset
  decomposition (district scope + Landsat at 1 km) and a valid-pixel panel on the
  comparison figure. 249 tests passing.
- **2026-08-08 — Phase 2 rev 5**: `BATCH_YEARS = 1` still failed, which ruled
  out volume and exposed the real bug: **batching by `ee.Filter` saves
  nothing**, because filtering a collection built from `ee.List.map()`
  materialises every element's graph to test the predicate. The same trap I
  had already fixed in Step 4 and failed to apply to the batching itself.
  Split the API: `zonal_annual_means_by_year` builds each batch's composites
  from the scene collection via `start_year`/`end_year`, so only those years
  exist; `zonal_annual_means` keeps the single-request path and **lost** its
  `batch_years` argument rather than keep a false promise. 233 tests passing.
- **2026-08-08 — Phase 2 rev 4**: third Colab run **verified Steps 1-5** (1674
  scenes, inventory exactly as predicted, dry-2025 `obs_count` min 4 / median 10
  over the CMC, both PNGs rendered). Step 6 still exceeded memory: the cost I had
  missed is that a mask built by *compositing* is instantiated once per image it
  masks, so `aoi.water_mask` was rebuilt 26 times. Added
  `aoi.static_water_mask` (JRC-only, one static image) for series work, an
  LST-only collection mode (`include_sr=False`), and dropped `BATCH_YEARS` to 2.
  The static mask is the only change that moves a number, so the notebook now
  measures it against the combined mask on 2025 and prints the difference. Also
  learned that Landsat 5 has **zero** 2012 scenes (it stopped acquiring in Nov
  2011), so the L7-only gap is 2012-01 to 2013-03, wider than CLAUDE.md implies.
  231 tests passing.
- **2026-08-08 — Phase 2 rev 3**: second Colab run cleared Steps 1-5 and died in
  `zonal_annual_means` — one request cannot hold 26 annual composite graphs.
  Added `with_percentile=False` (a percentile reducer retains every observation
  per pixel to sort them; mean/count stream) and batched the fetch
  (`composites.zonal_batch_years`, default 4). Neither changes a number. Landsat
  and MODIS series split into separate notebook cells. 229 tests passing.
- **2026-08-08 — Phase 2 rev 2**: first Colab run failed with `User memory limit
  exceeded` on the dry-season cell. Three real faults, not a quota issue:
  `ee.Algorithms.If` evaluates both branches (replaced with a merged fully-masked
  padding image); the notebook built 26 annual composites to display one (now
  composites the single year); and everything ran over the province-wide
  analysis region (now district-scoped, `aoi.water_mask` gained an optional
  `region`). Also stopped calling `bandNames().getInfo()`, which forces full
  graph evaluation to learn names already known client-side. 224 tests passing.
- **2026-08-08 — Phase 2**: LST pipeline written. Re-verified the Phase 2 catalog
  specifics (no discrepancies) and found three new facts, the important one being that
  MODIS `Clear_sky_days` is an 8-bit **bitmask, not a count** — reading it raw would have
  reported 255 clear days instead of 8. Wrote `landsat.harmonised_collection` (four
  sensors, one schema, thermal-only quality gates so bad LST does not delete good
  reflectance), `indices` (six indices, one albedo coefficient set), `composites`
  (annual + dry-season, `obs_count` on every product, single-round-trip tables), `modis`
  (real QC bit filtering, clear-sky popcount, Aqua launch clamp), and notebook 02. A
  design-review pass caught the empty-year band-less-image bug, the `eq`-vs-`neq`
  fail-closed filter, the missing `sharedInputs=True`, and an SLC-off off-by-one — all
  fixed before the first Colab run. **Reversed the albedo decision** (one coefficient set,
  not per-sensor) and flagged it for user sign-off. 222 tests passing. Nothing
  Colab-verified yet.
- **2026-08-08 — Phase 0**: scaffolded repo; verified 18 datasets in the GEE
  catalog Browser pane (17 ✅, 1 ❌ GSOD); wrote params.yaml, auth module,
  tests (30 passing), notebook 00, stubs, README. No LST processing code
  written (per session scope). Nothing committed to git yet.
- **2026-08-08 — Phase 1f**: fifth Colab run — **Phase 1 signed off**. Audit
  clean (55/55 names, 55 features, no leftovers); both 1e bugs confirmed fixed by
  the ring and rural mask returning to run-3 values to the decimal; all five
  figures reviewed. Resolved the last inconsistency: "CMC minus water" was being
  reported at two reduction scales (40.18 @30 m vs 37.70 @300 m), so
  `cmc_land_area_km2` now takes an explicit `scale_m` and the notebook prints
  both as a labelled sensitivity pair rather than one number chosen for looking
  closest to 37.31. 90 tests passing.
- **2026-08-08 — Phase 1e**: fourth Colab run. Audit caught 5 spelling errors and
  printed COD-AB's spellings. Proved GN-union ≡ DS-pair (the 55 GN divisions tile
  the two DS divisions exactly), so the 46.9-vs-37.31 gap is the **Port harbour**,
  not a unit-choice error — added `cmc_land_area_km2()` to measure it. Fixed a
  real bug: GN names are not unique within the district, so name matching pulled
  in stray divisions (50 units measured 47.50 km², more than both parents) —
  selection is now scoped to the parent DS divisions. Also closed a correctness
  gap: `urban_mask()` now excludes water like `rural_mask()` does. 90 tests.
- **2026-08-08 — Phase 1d**: third Colab run — DS/GN counts exact (13/557), ring
  and LCZ masks all correct. CMC came out 47.1 km² vs ~37; proved no COD-AB DS
  union can give 37.31, and the zoomed water figure showed why (Colombo DS
  encloses the Port harbour). User supplied the CMC GIS Unit's GN-division map →
  CMC redefined as the union of its **55 GN divisions**, with `cmc_name_audit()`
  guarding against partial name matches. Reviewed all five run-3 PNGs: Beira,
  Diyawanna, Kelani, Bolgoda and the eastern reservoirs all confirmed in the
  water mask. 87 tests passing.
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
