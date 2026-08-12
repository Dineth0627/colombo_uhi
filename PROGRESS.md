# PROGRESS — Colombo UHI practicum

_Last updated: 2026-08-12 (Phase 5 code written; **awaiting the first Colab run of notebook 05**)_

## Status snapshot

| Phase | Content | Status |
|---|---|---|
| 0 | Scaffold, params.yaml, auth, notebook 00 | ✅ done + Colab-verified |
| 1 | AOI & boundaries | ✅ **done + Colab-verified** (run 5, 5 iterations) |
| 2 | LST pipeline (Landsat + MODIS) | ✅ **done + Colab-verified** (run 6, 6 iterations) |
| 3 | UHI metrics (SUHII, UTFVI) | ✅ **done + Colab-verified** (runs 7–9) |
| 4 | Trend analysis (MK/Sen + FDR) | ✅ **done + Colab-verified (runs 14–18).** MODIS Terra night = trend evidence; class contrasts stable across 2 configs; Landsat = quantified negative result (detection limit 0.33 °C/yr) |
| 5 | Spatial statistics (Gi*, Moran, EHSA, GWR) | 🟡 **code written 2026-08-12, NOT YET RUN.** 736 tests pass locally |
| 6 | Scenario projection (RF + CA-Markov) | ⬜ |
| 7 | Greening priority (MCDA/AHP) | ⬜ |
| 8 | Report figures | ⬜ |

## PHASE 5 — implementation record (2026-08-12, NOT YET RUN)

**Nothing below has been executed against Earth Engine.** Claude Code has no EE
credentials. Every server-side function is unverified until notebook 05 runs.

**736 tests pass locally, 8 skip** (was 594 passing at the end of Phase 4). The
skips are the 5 `rasterio` raster round-trip tests from Phase 4 plus **3 new
cross-validation tests** that need `esda`, `libpysal` and `spreg` — absent from
the local dev environment, present in Colab. Those three are the whole safety
argument for the design decision below, so **run `pytest tests/ -q` in Colab and
confirm 736 passing before signing Phase 5 off.**

### New / changed files

| File | What |
|---|---|
| `src/colombo_uhi/spatial_stats.py` | stub → full module (~2 600 lines): weights, Moran/LISA/Gi*, EHSA, the regression ladder, MAUP, landscape metrics, zone geometry + covariate exports |
| `config/params.yaml` | `spatial_stats` 8 lines → ~170; 2 new caveats (`within_epoch_only`, `zonal_not_pixel`) |
| `src/colombo_uhi/viz.py` | **additive only**: `spatial_palette`, LISA/Gi*/EHSA maps, GWR coefficient small multiples, MAUP table, landscape change |
| `tests/test_spatial_stats.py` | **new**, 115 tests |
| `tests/test_params.py` | + 24 structural tests for the new keys |
| `notebooks/05_spatial_statistics.ipynb` | stub → 43 cells |
| `.gitignore` | `data/interim/*.geojson`, `*.csv` — staging copies of the raw exports |

**No `requirements.txt` change.** `libpysal`, `esda`, `spreg`, `mgwr`,
`geopandas`, `rasterio` and `scipy` were already declared in Phase 0.

### The central design decision: the statistics are implemented, not imported

Moran's I (global and local), Gi\*, the OLS diagnostics and the Lagrange
Multiplier tests are computed **analytically in numpy** rather than delegated to
`esda`/`spreg`. Four reasons, in order of weight:

1. **Testability.** The local dev environment has no PySAL, so a module built on
   `esda` would have had **zero** local test coverage of its numerical core —
   against a project rule that pure-Python logic must be pytest-covered.
2. **Reproducibility.** The permutation p-values are driven by
   `spatial_stats.random_seed`, so a "significant" cluster cannot vanish on a
   re-run.
3. **API risk.** `Queen.from_dataframe`'s `use_index`, `Moran_Local`'s `seed` and
   `island_weight`, `G_Local`'s `star` — all have moved between releases. The
   plan flagged these as the phase's main uncertainty; implementing the
   statistics removes the uncertainty instead of guessing at it.
4. **Islands.** See below — the library behaviour here is silently wrong for
   this study area, and it had to be intercepted anyway.

**This is only safe because it is checked.** `esda_cross_check` and
`spreg_cross_check` run the reference implementations on the same inputs and
report the differences; notebook 05 **Step 1** prints both tables, and
`test_our_statistics_match_esda` / `test_our_ols_and_lm_tests_match_spreg`
assert agreement to 1e-9 and 1e-6 via `importorskip`. **Read the Step 1 output
before trusting any map in the notebook.**

`spreg` and `mgwr` ARE used for what they uniquely provide: ML spatial
lag/error estimation, and GWR/MGWR bandwidth search.

**One convention difference, found while writing the cross-check and worth
knowing before Step 1 prints it.** Local Moran's I has two published
normalisations: Anselin (1995) divides by the population second moment, GeoDa
and `esda` by `(n-1)`. They differ by a single constant identical for every
zone, which cannot change a quadrant, a permutation p-value or a cluster map —
only the printed magnitude of `local_i`. `esda_cross_check` therefore compares
that row **after rescaling**, prints the ratio, and adds a separate row
asserting the **quadrant labels agree exactly**. Global Moran's I and Gi\* have
one convention each and are compared directly. Had this been left as a naive
equality test it would have failed in Colab at ~3 % and looked like a bug in the
statistic.

### Five things that would have produced wrong maps silently

1. **An island is not a zero.** A GN division with no queen neighbour gets a
   local statistic over an *empty* neighbourhood — and with the standard
   pseudo-p formula it receives `1/(permutations+1)`, the **smallest achievable
   p-value**. The island would render as the single most significant cluster on
   the map. Colombo's coast is ragged and COD-AB encloses the port outer
   harbour, so this is live here, not theoretical. `build_weights` counts
   islands, repairs them (`island_policy: attach_knn1`) and reports the count;
   the statistics emit NaN for any that remain. Two tests pin it.
2. **Gi\* is undefined for negative values** — it is a ratio of a neighbourhood
   sum to the global sum. On `LST_z` or an anomaly it returns finite, plausible
   numbers that mean nothing. Guarded, and the error names the fix.
3. **Moran's I and Gi\* need *different* weights.** Row-standardising Gi\* forces
   every neighbourhood sum to 1 and collapses the variance term that lets a
   large neighbourhood outweigh a small one. `build_weights` returns the binary
   matrix alongside the row-standardised one so the two can never be confused.
4. **557 local tests are 557 tests.** Uncorrected at α=0.05 they manufacture ~28
   clusters from noise. Every local statistic carries a BH-adjusted p, and both
   counts are reported — the Phase 4 pixel-vs-GN discipline, applied locally.
5. **A GWR at n=13 still returns numbers.** `require_estimable` refuses, and the
   refusal (with its reason) becomes a row in the MAUP table.

### The EHSA classifier: three ordering bugs found and fixed before it ran

Written, then tested against all 13 categories, which is how these surfaced:

* An **all-hot** series is trivially "one unbroken final run", so `consecutive`
  swallowed `persistent`, `intensifying` and `diminishing` entirely. Fixed by
  gating `consecutive` on `share < persistent_share`.
* `historical` was **arithmetically unreachable**: with `persistent_share` 0.90
  a 10-bin series can afford one quiet bin, but the rule demanded three.
  `historical_recent_bins` 3 → 1, and `test_params` now pins the reachability
  relation so the combination cannot be reintroduced.
* The hot/cold **side was chosen by majority**, so a zone that was a cold spot
  for a decade and is a hot spot now was filed as a *cold* spot. The final bin
  now decides, and `oscillating` is tested before `new` — otherwise the flip,
  the most informative thing about such a zone, is lost.

### Decisions taken with the user (2026-08-12)

| # | Decision | Reasoning |
|---|---|---|
| D1 | EHSA on **both** `landsat_oli_dry` (12 bins, 100 m, GN) and `terra_night` (26 bins, 1 km, GN+DS) | Neither suffices: one has intra-urban detail, the other temporal power. `test_params` pins that neither is a **pooled** Landsat series — a Mann-Kendall across a changeover would measure the sensor step (Phase 4, run 16) |
| D2 | Epoch LISA/Gi\* on the **pooled** `landsat_dry`, three epochs | These are *within-epoch* statistics on deviations from that epoch's own mean, so a spatially uniform sensor step cancels as it does in SUHII. **No epoch-to-epoch magnitude may be quoted** (`caveats.within_epoch_only`). Step 9 re-runs the 2020s on `landsat_oli_dry` and compares cluster geography — so the argument is *tested*, not just asserted |
| D3 | GN full ladder; **DS only what n=13 supports** | 13 units × 6 predictors leaves a GWR with effectively no degrees of freedom. `require_estimable` gates it and the refusal is a reported result |
| D4 | Landscape metrics in **pure Python** (`scipy.ndimage`), Dynamic World 2016 vs 2024 + WorldCover 2021 | Keeps them pytest-covered like the rest of the pure-Python core, and two dates make the deliverable fragmentation *change*. Cropland is excluded from "green" in both schemes |

### Two performance decisions that are not micro-optimisation

* **Conditional randomisation reuses one index matrix.** Drawing a fresh
  permutation per zone per replicate is O(permutations × n) *per zone*; drawing
  one `(permutations, n−1)` matrix and re-mapping it around each zone is that
  cost in total. At 557 zones × 26 bins that is the difference between an EHSA
  panel taking seconds and taking a quarter of an hour.
* **`unit_noise_detection_limit` is memoised on (n_bins, α).**
  `trends.minimum_detectable_slope` bisects over a Monte Carlo power
  simulation — thousands of MK evaluations per call. **Mann-Kendall is scale
  invariant**, so detecting slope `s` in noise `σ` is detecting `s/σ` in unit
  noise: the limit scales linearly and only the unit-noise value needs
  simulating, once per series length.

### Things Colab must settle (notebook 05) — record the answers here

| # | Unknown | If it fails |
|---|---|---|
| S1 | **Do the analytic statistics match `esda`/`spreg`?** (Step 1 probe) | STOP. Nothing below Step 1 is trustworthy until explained |
| S2 | Installed PySAL API signatures, and `esda`'s `.q` quadrant coding | documentation only — nothing depends on them, but a `.q` mismatch would invert a cluster legend |
| S3 | How many GN divisions are **islands** under queen contiguity? | expected small and repairable; if large, revisit `weights.scheme` |
| S4 | Does the GN GeoJSON fit under `geometry.max_geojson_mb` (8 MB)? | raise `geometry.simplify_m` and re-export rather than committing it |
| S5 | Is global Moran's I on 2020s GN LST positive and significant? | if not, the covariate table and the weights are in different zone orders — far likelier than Colombo being unusual |
| S6 | **Does the 2020s cluster geography survive swapping to the single-sensor series?** (Step 9) | if not, D2 is wrong: re-scope the epoch maps onto `landsat_oli_dry` and accept the shorter record |
| S7 | Which model does the LM rule select at GN, and is residual Moran's I significant? | a non-significant residual Moran's I would mean the spatial models were unnecessary — itself reportable |
| S8 | Do MGWR bandwidths differ per covariate? | identical bandwidths mean the multiscale search collapsed and the result is only GWR |
| S9 | Does `spreg.ML_Lag`/`ML_Error` converge at n≈557? | switch `regression.lag_estimator` to `"gm"` |
| S10 | What fraction of EHSA "no pattern" zones are `underpowered`, per series? | over the 12-bin Landsat panel a high fraction is **expected** and is the honest headline |

### Caveats that travel into Phase 6+

1. **Zonal ≠ pixel ≠ person.** A coefficient fitted across 557 polygons is a
   property of that aggregation. Reading it as an individual-level relationship
   is the ecological fallacy (`caveats.zonal_not_pixel`).
2. **Epoch cluster maps carry no magnitude.** Only Phase 4's Sen's slope
   measures change (`caveats.within_epoch_only`).
3. **Population is the 2020 WorldPop layer** and built fraction is a 5-year GHSL
   epoch, whatever epoch they sit beside. Both travel with their real year.
4. **Landscape metrics are scale dependent.** Computed at 10 m; the same city at
   30 m gives different patch counts, edge density and aggregation index. The
   grid size is returned with the metrics and must be quoted.
5. **Distance-to-coast excludes inland water** via a connected-component floor.
   Without it, every division around Beira Lake would read as coastal.

## PHASE 4 — implementation record (2026-08-09, NOT YET RUN)

**Nothing below has been executed against Earth Engine.** Claude Code has no EE
credentials. Every server-side function is unverified until notebook 04 runs.

**540 tests pass locally, 5 skip** (was 388 at the end of Phase 3). The 5 skips
are the `apply_fdr_to_raster` / `read_trend_raster` round-trip tests, which need
`rasterio` — declared in `requirements.txt` and present in Colab, but not
installed in the local dev environment. They are real tests, not placeholders:
they write a synthetic 9-band GeoTIFF with a nodata block and assert the
nodata→NaN conversion, the band-count guard, the BH/BY ordering, and that
"untested" and "tested but not significant" come out as **different** values.
**Run `pytest tests/test_trends.py -k raster` in Colab and confirm 5 passes**
before signing Phase 4 off — that is the only pure-Python path not yet executed.

### New / changed files

| File | What |
|---|---|
| `src/colombo_uhi/trends.py` | stub → full module: MK/Sen products, FDR, MMK, decadal, raster post-processing |
| `src/colombo_uhi/exports.py` | stub → full module: `export_name`, Drive/asset wrappers, task status |
| `src/colombo_uhi/landcover.py` | **new**: WorldCover / LCZ / Dynamic World class images + grouped-reducer stratified stats |
| `src/colombo_uhi/composites.py` | **additive only**: `annual_composites` now sets `series_basis` and `window_months` |
| `src/colombo_uhi/viz.py` | + `trend_vis_params`, `build_trend_map_figure`, `build_mk_comparison_figure`, `build_trend_by_class_figure` |
| `config/params.yaml` | `trends` expanded to 27 keys; new `landcover` section; `exports` + 4 keys; 2 new caveats |
| `tests/test_trends.py`, `tests/test_exports.py` | **new** |
| `notebooks/04_trend_analysis.ipynb` | stub → 41 cells |

### Two corrections to the GEE community tutorial — deliberate, do not "fix" back

1. **`.int()` truncation.** The tutorial's `sign(i,j)` is
   `j.neq(i).multiply(j.subtract(i).clamp(-1,1)).int()`. `.int()` truncates
   toward zero, so a **+0.3 °C** year-to-year difference yields sign **0**.
   `LST_C` is float °C and most annual differences are well under 1 °C, so this
   would zero out most of S. The tutorial states its own scope: *"discrete data
   (i.e. not floating point)"*. We use `diff.gt(0).subtract(diff.lt(0))`.
   Pinned by `test_the_tutorials_truncating_reference_disagrees_on_a_sub_degree_difference`.
2. **One-sided p.** The tutorial emits `1 − Φ(|Z|)` and compensates by
   thresholding at 0.025. Benjamini-Hochberg needs **two-sided** input, so we
   emit `mk_p_two_sided = 2(1 − Φ(|Z|)) = 1 − erf(|Z|/√2)`. Feeding BH the
   one-sided form would roughly **double** the area reported as significant.

### The 325-image self-join is avoided, not merely discouraged

A 26-year series makes 26·25/2 = **325** pairwise sign images, each carrying two
full annual-composite graphs — and Colab run 2 died on **26** in one request. So
`mk_method: "tau_derived"` derives the statistics from the mandated
`ee.Reducer.kendallsCorrelation` instead:

* `τ_b = S / √((n₀−n₁)(n₀−n₂))`, `n₀ = n(n−1)/2`. x is the calendar **year**, so
  ties in x are impossible; ties in float LST effectively so. Hence `n₁ = n₂ = 0`
  and **`S = τ · n(n−1)/2` exactly**.
* `Var(S) = n(n−1)(2n+5)/18` — the tutorial's `factors()` with the tie term zero.
* Under ties this is **conservative, not wrong**: τ_b understates `S/n₀` and the
  untied variance overstates Var(S), so |Z| comes out too small and p too large.

`mk_method: "pairwise"` raises `NotImplementedError` with that reasoning.
`test_mk_statistics_from_tau_match_pymannkendall` pins agreement with
`pymannkendall` on 25 random series to 1e-9 for s, var_s, z and p.

### The structural guard (the brief's hard requirement)

* **Layer 1** — `trends.trend_image()` takes a source **key**, not a collection,
  and builds the series itself. Pinned by
  `test_trend_image_takes_a_source_key_not_a_collection`, which asserts the first
  parameter is `source` and that no `series`/`stack` parameter exists, so a later
  refactor cannot quietly reopen the hole.
* **Layer 2** — `require_annual_series()` costs ONE `getInfo` (all property
  arrays in a single `ee.List`) and delegates to the pure
  `validate_series_metadata()`. It rejects: a missing/partial/wrong
  `series_basis`; the presence of a `month` property (**the signal that catches a
  raw scene stack** — scenes carry `month`, composites do not); duplicate or
  non-ascending years; and a gap against an explicit range.

`composites.annual_composites` had to gain the `series_basis` marker because
**nothing else distinguished a composite from a scene** — scenes carry `year` too.

### Things Colab must settle (notebook 04, Step 2) — record the answers here

| # | Unknown | If it fails |
|---|---|---|
| V1 | Output band names of `sensSlope` and `kendallsCorrelation(1)`/`(2)` | edit `trends.bands` / `trends.mk_num_inputs` |
| V2 | Is the reducer's own p-value one- or two-sided? | documentation only — `mk_p_two_sided` is derived from Z and is unaffected |
| V3 | **`sensSlope` input order** — a known slope of 2.0 must return 2.0, not 0.5 | swap `trends.sen_input_order`; a reversed order returns the RECIPROCAL, not an error |
| V4 | Does the full-district reduction fit interactively, or must it be exported? | export only; last resort `trends.annual_stack_asset` |
| V5 | Grouped-reducer band-index convention for `trend_by_class` | reorder bands / `groupField` |
| V6 | Does `Export.image.toDrive` preserve band descriptions? | reader already falls back to `trends.export_band_order` with a warning |
| V7 | `aggregate_array` on a property no image carries — `[]` or raise? | one-line `try/except` in `require_annual_series` |

`mk_num_inputs: 2` is chosen because `composites._composite_reducer` already
records the rule empirically: a **multi-input** reducer makes
`ImageCollection.reduce` emit **bare** output names. So `kendallsCorrelation(2)`
should give `tau`/`p-value`, matching `sensSlope`'s `slope`/`offset`.

### Colab run 11 (2026-08-09) — Step 2 SETTLED, Step 3 guard reworked

**V1 band names — measured, and exactly as assumed. No params change needed.**

| Reduce | Output bands |
|---|---|
| `ee.Reducer.sensSlope()` | `['slope', 'offset']` — **bare** |
| `ee.Reducer.kendallsCorrelation(2)` | `['tau', 'p-value']` — **bare** |
| `ee.Reducer.kendallsCorrelation(1)` | `['fit_y_tau', 'fit_y_p-value']` — **band-prefixed** |

This confirms the rule `composites._composite_reducer` records: a **multi-input**
reducer makes `ImageCollection.reduce` emit bare output names, a single-input one
prefixes with the band. It also vindicates `mk_num_inputs: 2` — with `1` the band
names would depend on the fit-stack band name.
`resolve_reduced_band_names` matches on the exact-suffix path, no positional
fallback, no warning.

**V3 SETTLED: `sensSlope` takes x THEN y.** The known-slope probe returned
`{'offset': 10, 'slope': 2}` for a series built as `y = 10 + 2x`. Both outputs
correct, so `trends.sen_input_order: ["x", "y"]` stands.

**V2 SETTLED: `ee` tau agrees, `ee` p-value is unusable.** `ee` tau = 0.9090909,
identical to scipy and pymannkendall. But the reducer returned **`p-value: None`**
on a clean 12-point series with τ = 0.909 — it simply does not populate that
output. **Consequence: the exported `mk_p_ee` band will be entirely masked.**
That is harmless and was anticipated — `mk_p_ee` is comparison-only and never
reaches the FDR correction; `mk_p_two_sided` is derived from Z. `verify_trend_bands`
already skips bands whose min/max come back `None`.

Also confirmed locally: our `two_sided_p` matches pymannkendall's normal
approximation to 1e-6 (5.215e-5), while `scipy.stats.kendalltau` returns an
**exact** p (1.47e-6) at n=12. Different method, expected difference, not a bug.

### Run 18 (2026-08-11) — PHASE 4 RESOLVED. The zero is real, and now bounded.

**The sampling hypothesis is REJECTED, and the answer turned out to be simpler
and more defensible.**

`corr(obs_count, LST) = +0.076` over the CMC — essentially zero. Observation
counts do rise (+0.31/yr, and Landsat 9 clearly doubles them from 2022) but LST
does not track them. **Landsat 9 is not manufacturing the decline.**

#### The actual answer: the series cannot resolve the signal

Mann-Kendall on the CMC mean of the clean 12-year series:

| | |
|---|---|
| Sen's slope | −0.157 °C/yr |
| tau / S / Z | −0.364 / −24 / −1.577 |
| **p** | **0.115 — NO significant trend** |
| interannual sd | **1.22 °C** |

**−0.157 °C/yr is not cooling.** It is an unconstrained estimate from a short,
noisy series. *Sen's slope always returns a number; it does not tell you whether
that number means anything.* Mann-Kendall does, and it says no.

#### The detection limit — this is the reportable result

New `trends.mk_detection_threshold` and `trends.minimum_detectable_slope`:

| series | n | sd | smallest detectable trend |
|---|---|---|---|
| `landsat_oli_dry`, CMC | 12 | 1.22 °C | **0.342 °C/yr** (3.76 °C over the record) — *measured in Colab, run 19* |
| hypothetically, 26 years | 26 | 1.22 °C | 0.102 °C/yr |
| MODIS-like (26 yr, lower noise) | 26 | 0.50 °C | 0.042 °C/yr |

A plausible urban warming signal is ~0.03 °C/yr. The 12-year Landsat series can
only resolve trends **an order of magnitude larger**. MK at n=12 needs
|S| > 30 of a maximum 66, i.e. |tau| > 0.448; we observed 24 / 0.364.

> **The correct statement is NOT "Colombo shows no warming".** It is: *a 12-year
> dry-season Landsat series over Colombo cannot resolve trends below ~0.34 °C/yr,
> an order of magnitude above the expected signal.* The zero FDR-significant
> pixels is the honest consequence of that, not a measurement of the climate.

(The limit assumes white noise; real annual LST is positively autocorrelated, so
the true limit is **worse**. Quote it as a lower bound.)

#### The three zeros, and why only one of them was informative

| series | n | sensors | result | why |
|---|---|---|---|---|
| pooled Landsat | 26 | L5+L7+L8+L9 | 0 significant | **wrong reason** — sensor steps cancelled |
| OLI Landsat | 12 | L8+L9 | 0 significant | **right reason** — below the detection limit |
| **MODIS Terra night** | **26** | **Terra only** | **17.6% BH / 0.33% BY** | length **and** consistency |

That is a complete and coherent story. Terra night is the only series with both
a long record and a single sensor, and it is the only one that detects anything.

#### Phase 4's defensible deliverables

1. **MODIS Terra night**: 17.6% (BH) / 0.33% (BY) of tested area significantly
   warming — 121.0 / 2.3 km². The trend evidence.
2. **Class contrasts**, stable across two independent configurations: built-up
   warms ~0.056 °C/yr faster than tree cover; LCZ 6 open low-rise ~0.108 °C/yr
   faster than LCZ A dense trees. Identical ranking in both.
3. **A quantified negative result** for Landsat, with its detection limit — a
   methodological finding relevant to anyone attempting Landsat LST trends in
   the humid tropics.

#### Also fixed

* `landsat_oli_dry.start_year` **2013 → 2014**: run 18 measured `obs_count = 0`
  for 2013, because Landsat 8 opens 2013-03 and the dry window is Jan–Mar. The
  usable series is 12 years, not 13.

**Tests: 594 passing, 5 skipped.**

### Run 17 (2026-08-11) — the single-sensor fix works, and exposes a SECOND artefact

Clean run on `landsat_oli_dry` (L8+L9, 2013–2025). Sensor offsets re-confirmed
identically. **The fix removed the step and revealed a different problem.**

#### The headline: absolute magnitudes are not reportable, contrasts are

| | pooled 26-yr | OLI 13-yr |
|---|---|---|
| median Sen's slope | **+0.028 °C/yr** | **−0.176 °C/yr** |
| slope p1 / p99 | −0.205 / +0.237 | −0.472 / +0.133 |
| FDR-significant | 0 | **0** |
| n_tested | 66,787 | 62,887 |

−0.176 °C/yr is **−2.3 °C over 13 years** — no more credible as cooling than the
pooled series was as "no trend". **The absolute level swings by 0.204 °C/yr
between two defensible configurations of the same data.**

**But the class CONTRASTS are stable across both:**

| contrast | pooled | OLI | Δ |
|---|---|---|---|
| Built-up − Tree cover | +0.0339 | +0.0559 | +0.022 |
| LCZ 6 open low-rise − LCZ A dense trees | +0.0929 | +0.1081 | +0.015 |
| LCZ 9 sparsely built − LCZ A dense trees | +0.0586 | +0.0768 | +0.018 |

**Identical ranking in both**, agreement to ~0.02 °C/yr while the level moves ten
times that. That is the signature of a **common-mode bias** — it shifts every
class equally, cancels in a difference, survives in a level. Two independent
configurations now confirm it empirically.

> **The Phase 4 deliverable is the CONTRAST, not the magnitude.** "Built-up warms
> ~0.06 °C/yr faster than tree cover, and open low-rise ~0.11 °C/yr faster than
> dense trees" is supported. Any absolute Landsat °C/yr figure is not.

Per-GN on the clean series: **0 of 557** FDR-significant (was 8). Fort is still
the top warmer at +0.474 °C/yr but with n=12, p=0.034, p_adj=0.298 — it no longer
survives correction. The earlier "Fort +0.365 °C/yr, p_adj=1.1e-4" is **withdrawn**.

MODIS Terra night is unchanged and remains the only robust *magnitude*:
**17.6% BH / 0.33% BY** of tested area significantly warming.

#### Leading hypothesis for the −0.176 °C/yr: growing observation counts

Landsat 9 launched late 2021, roughly doubling dry-season scene availability from
2022. An annual composite is a **median over whatever clear-sky days existed**:
with few scenes the median is pinned to a handful of clear (and in a tropical dry
season, HOT) days; with more scenes it regresses toward a cooler, more
representative value. **A growing constellation can manufacture apparent cooling
with no change in climate.**

**Notebook Step 6b now tests this directly** — per-year CMC mean LST against
per-year mean `obs_count`, with an explicit verdict from
`trends.obs_count_verdict`. If the correlation is strongly negative while
obs_count rises, the magnitude is unreportable at any window.

> **Bug in the first Step 6b, fixed:** it asked
> `composites.zonal_annual_means_by_year(band="obs_count")`. That `band` argument
> selects from the **scene** collection, and scenes carry only `LST_C` —
> `obs_count` is *produced by* compositing and does not exist as an input, so
> Earth Engine failed with *"Band pattern 'obs_count' did not match any bands"*.
> Replaced by `trends.obs_count_series`, which reads both means off the
> **composite** in one reduction, batched by year. The distinction is worth
> remembering: anything `annual_composites` *creates* (`obs_count`, the
> percentile band) can only be read after compositing.

#### Bugs found and fixed this run

* **`zonal_annual_series` ignored the source's `start_year`.** It computed
  2000–2025 for a 2013 source, spending seven round trips on years that cannot
  contain data and then warning that 54% of every division's series was
  "missing". Now routed through a new pure `trends.resolve_source_years`.
* **`trend_image` centred the x axis on `time.start_year`,** so `sen_offset` for
  the OLI product was the fitted LST in 2000 — thirteen years outside the data.
  Same fix.
* `trends.slope_vis` widened to **±0.50** (13.28% saturated at ±0.25).

**Tests: 580 passing, 5 skipped.**

### Run 16 (2026-08-11) — C2 INTER-CALIBRATION FAILS OVER COLOMBO

**The most consequential result of Phase 4, and it is a negative one.** The
cross-sensor check ran clean and refuted `landsat_c2l2.harmonisation: none` for
this AOI.

#### Measured offsets, CMC dry season (Jan–Mar), 100 m

| pair | mean offset | sd | overlap yrs | t | verdict |
|---|---|---|---|---|---|
| L5 − L7 | **+1.783 °C** | 1.851 | 8 (2004–2011) | +2.72 | **material** |
| L7 − L8 | **−2.478 °C** (L8 hotter) | 2.177 | 10 (2014–2023) | −3.60 | **material** |
| L8 − L9 | −0.397 °C | 1.187 | 4 (2022–2025) | −0.67 | negligible |

**Scale:** the whole 26-year trend signal is +0.73 °C (Sen's slope × 26). The
L7→L8 step is **3.4×** that; L5→L7 is **2.4×**.

**It reconciles quantitatively.** L8 flies 2013–2020 = 8 of the 10 years in the
2011–2020 window and none of 2000–2010, so the step alone predicts a decadal
difference of 0.8 × 2.478 = **+1.98 °C** against the **+1.70 °C** observed. The
pooled Landsat Mann-Kendall was measuring the changeover, not the climate — and a
step up followed by a step down is exactly how a warming series reports
"no trend".

**Caveat on the diagnostic itself:** `sd_offset` (1.85–2.18 °C) is comparable to
the offsets, and the overlaps are only 8–10 years. Sensors observe different
dates within the same dry season, so weather is in there. What makes it decisive
is the combination — significant, in the directions that explain the zigzag, and
several times the signal they contaminate. Any one alone would be weak.

#### What this invalidates, and what survives

| Quantity | Status |
|---|---|
| **SUHII** (urban − rural, same year) | **ROBUST** — a common-mode step cancels. **Phase 3 is unaffected.** |
| **Class contrasts** (built-up vs tree cover ≈ 2.8×) | **Largely robust** — same cancellation |
| **MODIS Terra night** (one sensor) | **ROBUST** — 17.6% BH / 0.33% BY stands |
| Absolute per-GN slopes (Fort +0.365 °C/yr) | **CONTAMINATED — withdrawn pending recompute** |
| Absolute by-class slopes (+0.053 °C/yr) | **CONTAMINATED** — quote the contrast instead |
| Landsat pooled pixel-wise MK ("0% significant") | **INVALID** — a step artefact |

> Fort's +0.365 °C/yr = +9.5 °C over 26 years, with τ = 0.73. That is not
> credible as warming. A division whose early years are L7-dominated and later
> years L8-dominated gets a ~2.5 °C step smeared across the changeover, which MK
> reads as a *strong monotonic trend*. **This is the more dangerous failure mode
> than the zero: it looks significant.**

#### The fix

**New source `landsat_oli_dry`**: Landsat 8 + 9 only, **2013–2025**, dry season,
100 m. L8−L9 is negligible so OLI/TIRS pools safely. `trends.pixel_sources` now
reads `["landsat_oli_dry", "terra_night", "aqua_night"]`.

* `uhi_metrics.source_collection` gained a `sensors` passthrough from the source
  mapping; `trends.annual_series` honours a per-source `start_year` so the OLI
  series does not fabricate empty pre-2013 years.
* `landsat_dry` is **retained** for SUHII, where the step cancels.
* The decadal product is retargeted to the pooled `landsat_dry` series and
  **relabelled a sensor-step diagnostic** — its windows only mean anything over
  the full record, and its job now is to show the discontinuity.
* Notebook Step 11 now prints class **contrasts** relative to the coolest
  well-sampled class, alongside the absolute slopes.

**Cost:** 13 years instead of 26. Lower MK power, and `min_years: 10` is now 10
of 13 — a strict floor. Read `n_tested` in Step 8; if it collapses, drop the
floor to 8 rather than re-pooling sensors.

**Pending:** rerun Parts 1 and 2 on `landsat_oli_dry` to recompute the per-GN and
by-class numbers on the clean series.

### Run 15 (2026-08-11) — PHASE 4 RAN END TO END. One finding blocks the headline.

**No errors in any cell.** All 6 exports submitted, all downloaded, every figure
and table produced — Part 1 and Part 2 both complete.

#### Results that stand

| Result | Value |
|---|---|
| **Fort (CBD), per-GN** | **+0.365 °C/yr**, τ = 0.729, S = 237, p_adj = 1.1e-4 |
| Kotikawatta East | +0.189 °C/yr, τ = 0.575, p_adj = 0.0115 |
| GN divisions FDR-significant | 8 of 557 (1.4%) |
| Built-up vs Tree cover (WorldCover) | **+0.0528 vs +0.0189 °C/yr** — ~2.8× |
| LCZ 6 open low-rise | **+0.0659 °C/yr** (highest LCZ) |
| LCZ A dense trees | **−0.0271 °C/yr** (cooling) |
| LCZ 9 sparsely built | +0.0315 °C/yr |
| MODIS Terra night, FDR-significant | **17.6% of tested (BH) / 0.33% (BY)** — 121.0 / 2.3 km² |
| Landsat dry, tested area | 66,787 px = 667.9 km² of 126,054 px |

SUHII trends (26 yr, Hamed-Rao): `landsat_dry|lcz_based` **increasing**
(p = 0.0048, +0.048 °C/yr); `terra_day|lcz_based` increasing (p = 2.1e-4);
`aqua_day|lcz_based` increasing (p = 2.2e-4); `terra_day_relaxed|buffer_ring`
**decreasing** (p = 0.0042 — consistent with the Phase 2 orbital-drift finding).
Both buffer-ring night series: no trend.

**`var_inflation` came out exactly 1.000 for all 12 SUHII series**, with
identical original and Hamed-Rao p-values. That means pymannkendall found **no
significant autocorrelation lag** on a 24–26 point series, so the correction
factor is 1. Report it as "the correction changed nothing", **not** as "the
correction was applied" — and note it is a low-power test at this length.

#### The finding that blocks the Landsat headline

**Landsat dry-season reported ZERO FDR-significant pixels** under both BH and
BY. That is arithmetically correct and scientifically uninformative, because:

| | |
|---|---|
| decadal path | **+1.70 °C** (2000s→2010s), then **−1.23 °C** (2010s→2020s) |
| net | +0.47 °C — which *matches* Sen's slope × 26 (+0.73 °C) |
| Sen's slope percentiles | p1 −0.205, p50 **+0.028**, p99 +0.237 °C/yr |

**The net agrees; the path does not.** Up-then-down is a zigzag, and Mann-Kendall
tests *monotonic* trend — concordant and discordant pairs cancel. A series with a
sensor step up followed by a step down reports "no trend" while warming
throughout, and that is the most likely reading of the zero.

The decade boundaries coincide with the Landsat changeovers (L5+L7 → L7-SLC-off
alone 2012-01→2013-03 → L8 → L8+L9), and **single-sensor MODIS Terra night is the
series that came out significant.** `landsat_c2l2.harmonisation: none` has always
been an *assumption*; this is the first evidence against it.

> **DO NOT report "0% of Colombo shows significant warming."** Until the
> cross-sensor check is run, the honest statement is that the Landsat series is
> not demonstrably monotonic, and the trend evidence rests on MODIS Terra night
> plus the per-GN and by-class results (which aggregate away much of the noise).

#### Added in response

* **`trends.sensor_annual_means` + `trends.build_sensor_offset_summary`** — the
  empirical inter-calibration check CLAUDE.md always asked for ("Still verify
  empirically on overlapping years"). Builds each sensor's dry-season annual mean
  separately and reports the pairwise offset over the years both flew, with its
  standard error and a `material`/`negligible` verdict against
  `trends.sensor_check.material_degc` (0.5 °C). Reduces SCENES in 4-year batches,
  not the trend graph, so it is affordable interactively.
  **Notebook Step 6 — run this first; its verdict conditions everything else.**
* `trends.slope_vis` widened to **±0.25** (was ±0.15): run 15 measured 5.66% of
  pixels saturating.
* The decadal cell now reports **`diff_z`** and `n_years_min`, not just the median
  difference — with 11/10/5-year windows the difference alone is not
  interpretable.

#### Still open

* One GN division (`LK1136035`) hit the degenerate Hamed-Rao branch
  (var_s = −15.9) and is correctly labelled `degenerate_variance`.
* `mk_p_ee` is an all-masked band, as predicted — the reducer does not populate
  it.

### Run 14 (2026-08-11) — through the wall: exports submitted

**The batch path works.** Run 14 reached Step 5 and submitted every export.

| Step | Result |
|---|---|
| 2 — reducer probes | ✅ all three settled (see below) |
| 3 — structural guard | ✅ scene rejected; constructor self-test passed |
| 4 — trend image | ❌ on `TREND.bandNames().getInfo()` — a call left **outside** the try/except |
| 5 — **exports** | ✅ **3 tasks submitted, READY** |
| 6 — FDR preview | ✅ degraded gracefully, as designed |
| 7 — MODIS | ❌ `NameError: _bands` — pure cascade from step 4 aborting |

`TREND` itself was assigned fine; only its *inspection* failed. That completes
the picture: **no interactive question about the trend graph is affordable, down
to and including `bandNames()`** — while a batch `Export` of the very same image
submits without complaint.

**Measured, and now pinned in `trends.reducer_outputs`:**

| Reduce | Output bands |
|---|---|
| `ee.Reducer.sensSlope()` | `['slope', 'offset']` — bare |
| `ee.Reducer.kendallsCorrelation(2)` | `['tau', 'p-value']` — bare |
| `ee.Reducer.kendallsCorrelation(1)` | `['fit_y_tau', 'fit_y_p-value']` — prefixed |

Sen's input order confirmed **x then y** (`slope=2, offset=10` for `y = 10+2x`).
`ee` tau = 0.9090909, matching scipy exactly. **`ee`'s own p-value came back
`None`** — expect `mk_p_ee` to be an entirely masked band; it is exported for
comparison only and never reaches the FDR correction.

### The restructure: build → export → download → analyse

Phase 4's notebook is now **two parts**, which is what the ceiling forces and is
also the better design — `viz.plot_trend_map` already worked from local arrays,
and a downloaded GeoTIFF makes a real map rather than a `getThumbURL` preview.

* **Part 1** builds the products and submits **six** batch tasks: three trend
  rasters, one decadal raster, two by-class tables. It asks Earth Engine nothing
  it does not have to, and every remaining interactive call goes through a
  `_try_ee` helper that degrades to a note.
* **Part 2** is pure numpy/pandas over `data/interim/`: FDR, the significant-area
  figures under BH **and** BY, the slope maps, the decadal differences, and the
  by-class tables. Only the per-GN reduction still touches Earth Engine, and it
  works on annual composites in 4-year batches (the Phase 3 pattern that
  succeeded), not on the trend graph.

**New code this run:**

* `trends.trend_by_class_collection` / `landcover.stratified_stats_collection` —
  the *unevaluated* twins of the grouped reduction, so it can run inside an
  `Export.table` task. The same reduction that fails interactively succeeds as a
  submitted task.
* `trends.decadal_product` + `trends.decadal_band_order` — the decadal statistics
  as one exportable image with a derived, duplicate-free band order.
  `decadal_difference` now suffixes `diff_se`/`diff_z` with the window pair so
  several differences can share one image.
* `viz.build_decadal_difference_figure` — the difference **beside its
  signal-to-noise panel**, because with 11/10/5-year windows the difference alone
  is not interpretable.
* Every import and constant moved into Step 0, so Part 2 runs as a standalone
  session and a skipped step cannot cascade into a `NameError`.

**Test count: 566 passing, 5 skipped** (the rasterio round-trip tests).

### Runs 10–13 (2026-08-09) — the interactive memory ceiling, and the retreat

**Outcome: `require_annual_series` cannot be run against the production series,
and Phase 4 no longer tries.** Four attempts, four diagnoses, three of them
wrong:

| Run | Hypothesis | Change | Result |
|---|---|---|---|
| 10 | cost is region-driven | validate a 2 km probe series | **failed** |
| 11 | cost is per-image count | sample 4 images via `toList` | **failed** |
| 12 | cost is `n_scenes` forcing value evaluation | `propertyNames()` + 2 targeted `get()` on 2 images | **failed** |
| 13 | *no interactive question about a 26-composite graph is affordable* | stop asking | — |

**The principle, which belongs to Phases 5–7 too:**

> **Interactive `getInfo` has a low memory ceiling; batch `Export` tasks do not.**
> A 26-composite graph is not something you can ask questions about
> interactively — not its properties, not even its property *names*. Push real
> computation into `Export` tasks and keep interactive calls to small,
> already-reduced results.

**What changed in run 13:**

* **`trend_image` now builds with ZERO `getInfo` calls.** `sen_slope_image` and
  `kendall_image` each used to call `reduced.bandNames().getInfo()` on a reduce
  over all 26 composites — defensive code against band-name ambiguity that run 11
  had already resolved by measurement. The names now come from
  `trends.reducer_outputs` (measured, pinned by a test) and are selected
  directly. `resolve_reduced_band_names` survives as the **probe** helper the
  notebook uses to discover names, which is where discovery belongs.
* **`trends.selftest_annual_series(source, params, region, years=3)`** validates
  the *constructor*: it builds three years over a 2 km box through the same
  `annual_series()` call the production path uses, and runs the guard on that. A
  constructor that emits `series_basis`, omits `month`, and yields one image per
  calendar year for 3 years over a small box does the same for 26 over a
  district — the difference is data volume, not structure. Combined with layer 1
  (`trend_image` takes a source KEY), that is what Phase 4's structural guarantee
  now rests on.
* `require_annual_series` stays on by default in `fit_stack`, for a collection of
  unknown provenance passed in directly. That is the case it was always for.
* **Steps 4 and 6 degrade instead of aborting.** `verify_trend_bands` and the
  in-session FDR preview are the last two interactive questions asked of the
  trend graph; both now catch an EE memory error, print what was lost, and let
  the notebook continue to the exports. Neither is load-bearing — Step 11
  produces the authoritative significant-area figure from the exported raster.

**Still unknown:** exactly where the interactive wall sits. Deliberately not
chased further — the constructor self-test removes the need to know.

**Decisions:** `n_scenes` left as-is (Phase-2 signed off, notebook 02 reads it,
and it was never proven to be the cause).

### Colab run 12 (2026-08-09) — the guard's real cost, as diagnosed at the time

Sampling four images **also** failed. The root cause, and the reason two earlier
fixes missed it:

> **An annual composite's property dictionary contains `n_scenes`, which is
> `yearly.size()` — a COMPUTED count over the filtered four-sensor collection.**
> Anything that forces the dictionary to be built evaluates that count as well.
> `aggregate_array` does exactly that, for every image at once. So *reading
> properties off a composite is not a metadata operation at all* — it drags a
> filtered multi-sensor collection count along with it, per image.

That is why neither a smaller region (run 10) nor fewer images (run 11) helped:
the cost is per-image and intrinsic to the property dict.

**The fix — read names, not values:**

* `propertyNames()` lists KEYS without evaluating any value. That is how the
  guard asks "is this a scene?" (scenes carry `month`) and "is this Phase 4
  code?" (composites carry `series_basis`) for free.
* Two targeted `get()` calls fetch only `series_basis` and `year`, both cheap
  client-side constants. `n_scenes` is never requested.
* Two images are inspected: the **first**, and the one at index
  `end_year - start_year`. Both endpoints landing on the right year is only
  possible if the collection holds exactly one image per calendar year across
  the range — so the endpoints stand in for the size.
* `collection.size()` became opt-in (`check_size=False`): sizing a computed
  collection can force it to be materialised, which re-triggers the whole chain.
* `check_scenes` was **removed** — it requested the single most expensive
  property in the dictionary.
* `validate_series_metadata` now accepts `size=None` ("not measured") and gains
  a `property_names` input.

**Design lesson for Phases 5–7:** if a computed property's value is itself an
`ee` computation, treat every property read on that object as expensive. Prefer
`propertyNames()` + targeted `get()` over `aggregate_array` / `toDictionary()`
whenever the object is a composite this project built.

**Step 3 — the guard blew the memory limit, and my first fix was wrong.**

Run 10's fix assumed the cost was region-driven and validated a 2 km probe
series. Run 11 showed that **also** fails: the driver is the **graph**, not the
pixels. Twenty-six composite reduces in one request exceed the limit whatever the
region — the run-2 wall, reached from a different direction.

The working fix is to **sample**, which is sound rather than a shortcut because
`annual_composites` emits exactly one image per calendar year *including empty
ones*:

* `require_annual_series` now reads `collection.size()` (the length of the
  underlying list — no image built) plus the properties of a
  `toList(probe, offset)` slice, `probe = trends.batch_years = 4`. One round trip.
* Any gap or extra image changes `size`, so the **size check covers the years
  that were not inspected**. A scene stack fails on size (1674 vs 26) *and* on
  the first image inspected.
* `full=True` inspects every year in batches for the paranoid case.
* The "missing years" check was **removed as dead code**: with size matching the
  range, inspected years distinct, and none straying outside it, a gap is
  arithmetically impossible.
* `validate_series_metadata` now takes `probed` alongside `size`, so length
  checks are against what was actually inspected.

### Colab run 10 (2026-08-09) — Step 2 probe, partial

* **V3 SETTLED: `ee.Reducer.sensSlope` takes x THEN y.** The known-slope-2.0
  probe passed, so `trends.sen_input_order: ["x", "y"]` is correct.
* **V2 not settled on the first attempt** — two bugs in the *probe cell* (the
  library was never involved), both now fixed:
  1. **`ee.List.reduce` with a multi-input reducer wants a list of PAIRS**
     `[[x0,y0],[x1,y1],...]`, not two parallel arrays. The cell passed the
     transposed shape, so Earth Engine saw **2 samples** and returned a trivial
     `tau = 1` with a NaN p-value. The probe now runs through an
     `ee.ImageCollection` of two-band images — the same shape
     `trends.kendall_image` uses — so it actually exercises the production path.
     **The pipeline was never affected**; only the probe was wrong.
  2. **Earth Engine returns NaN through JSON as the STRING `'NaN'`.** So
     `if _ee_p:` was truthy and the next arithmetic raised
     `TypeError: ufunc 'divide' not supported`. The probe now coerces with a
     `try/float()` helper and branches on `np.isfinite`.
* **A legitimate difference worth not re-investigating**: at n=12 `scipy.stats.
  kendalltau` returns an **exact** p (1.47e-6) while `pymannkendall` and our
  `two_sided_p` use the **normal approximation with continuity correction**
  (5.22e-5). Ours must match `pymannkendall`, not scipy's exact value —
  confirmed locally to 1e-6, with S = 60 and Var(S) = 212.667 matching exactly.
  The probe now prints all three and says which is the reference.
* **Layer 2 of the structural guard WORKS**: the scene collection was rejected
  with *"only 0 of 40 images carry a 'series_basis' property"*.
* **But `require_annual_series` itself blew the memory limit on the full-region
  series** — and that was a design error on my part, violating this project's
  own documented doctrine. Reading a collection's *properties* still forces
  Earth Engine to materialise every image, and an annual series built with
  `fromImages(List.map(...))` carries a full composite graph per year; 26 of
  those in one request over Colombo District is exactly the run-2 failure.

  > **⚠️ The fix described below was WRONG and is superseded by run 11.** It
  > assumed the cost was region-driven; it is graph-driven, so the probe-region
  > version failed too. Kept here because the reasoning is instructive.

  The (incorrect) reasoning was: the STRUCTURE of an annual series is
  region-independent — one image per year, identical properties, whatever
  geometry it was built over. That is *true*, but irrelevant to the cost. So:
  * the notebook validated a probe series built over a ~2 km box — **this still
    exceeded the memory limit**;
  * `require_annual_series` gained `check_scenes=False` (the `n_scenes` fetch
    forces the per-year scene filter to evaluate on top of the composite graph,
    and it only powers a warning), plus a `RuntimeError` on out-of-memory that
    states the probe-region fix instead of surfacing the raw EE error;
  * **`trend_image` now passes `validate=False`** to `fit_stack`. This is not a
    hole: layer 1 guarantees provenance because `trend_image` takes a source KEY
    and builds the series itself, so layer 2 would only re-establish what is
    already certain, at the cost of a memory blow-up. `fit_stack`'s default
    stays `validate=True` for the other door — a collection of unknown
    provenance passed in directly. Both defaults are pinned by signature tests.
* **Still to record from the next run**: the V1 band names printed by Step 2a,
  and the V2 verdict on whether `kendallsCorrelation`'s own p is one- or
  two-sided (it may simply be NaN, which is fine — `mk_p_ee` is
  comparison-only and never reaches the FDR correction).

### Findings from writing the code (not from Colab)

* **`pymannkendall`'s Sen slope uses the array INDEX as x**, so on a gapped
  series it silently reports °C per *observation*, not per year. Measured: a
  20-year linear series with a 14-year gap gives **0.38** index-based against a
  true **0.10** °C/yr. `mk_comparison` therefore takes the test statistics
  (`s`, `var_s`, `z`, `p`, `tau` — all order-only) from `pymannkendall` but the
  slope from our own year-based implementation. Worth knowing: a *narrow*
  contiguous gap leaves the median unchanged, so this bug **hides in a spot
  check** while corrupting the divisions that have scattered gaps.
* **Hamed & Rao returns a non-finite Var(S) on a near-deterministic series.** A
  perfect arithmetic progression detrends to exactly-zero residuals, the lag-0
  autocovariance is zero, and the correction divides by it → NaN z and p. Those
  rows are labelled `degenerate_variance` rather than left blank.
* **BH must refuse p-values outside [0, 1].** A GeoTIFF read back from Drive
  carries a nodata fill; an unmasked `-9999` sorts **first** and becomes the most
  significant pixel in the AOI.
* Our BH and BY match `statsmodels` `fdr_bh`/`fdr_by` exactly on random inputs.

### Decisions taken with the user

* Pixel-wise rasters for `landsat_dry` + `terra_night` + `aqua_night`. **MODIS
  daytime is excluded** — Phase 2 recorded Terra's post-2020 orbital drift
  contaminating end-of-series daytime trends, and night is what Landsat
  structurally cannot provide.
* FDR twice: coarse in-session preview **and** the authoritative exported raster.
* Harmonic regression and the BFAST export are **deferred** out of Phase 4.
* Land cover went into a new `landcover.py`, since Phases 5–7 need it too.
* `fit_scale_m: 100`, not 30 — adjacent 30 m LST pixels are near-duplicates, so
  fitting at 30 m multiplies the FDR test count ~11× without adding independent
  information and drags the BH threshold down for every genuine pixel.
* Decades are **11 / 10 / 5 years, unequal by construction** (the study period
  ends in 2025), kept separate from `uhi.utfvi.epochs` because the two answer
  different questions. `decadal_means` emits `sd_`/`n_years_` and
  `decadal_difference` emits `diff_se`/`diff_z` so the asymmetry lands on the map.

### Caveats that must travel into Phase 5+

1. **Trend ≠ attribution.** Stratifying by WorldCover 2021 / LCZ 2018–19 answers
   "where is the warming, by TODAY'S land cover", not "did land-cover change
   cause it". Phase 6 is where attribution belongs.
2. **BH assumes independence or PRDS**, which a 100 m LST raster violates. Report
   Benjamini-Yekutieli beside it, and report the pixel-wise and per-GN
   significant fractions as a pair — that spread is the MAUP sensitivity applied
   to significance.
3. **The significance map inherits WRS-2 side-lap striping** from `obs_count`,
   one strip of which crosses the CMC. Report it; do not tune it away.
4. **C2 inter-calibration is assumed, not verified** (`harmonisation: none`), and
   2012-01 → 2013-03 rests on SLC-off ETM+ alone.
5. **Decadal differences are not the warming number.** Sen's slope is. A decadal
   difference conflates trend with interannual variability and with changing
   observation counts; the maps are for spatial pattern only.

## PHASE 3 — Colab run 7 results (2026-08-09)

Ran end to end on the first attempt. **388 tests pass locally** (was 249 at the
end of Phase 2). Every acceptance check passed; the numbers below are the
verified reference values Phase 4 may quote without re-running.

### SUHII, 2000–2025, mean °C (buffer_ring / lcz_based)

| Source | buffer_ring | lcz_based | urban px (median) |
|---|---|---|---|
| `landsat_dry` | **6.68** | **3.47** | 4101 / 46785 |
| `aqua_day` (~13:30) | **4.72** | **1.82** | 7.5 / 386 |
| `terra_day_relaxed` | **3.60** | **1.79** | 34 / 435 |
| `terra_day` (strict) | **2.98** | **1.56** | 17 / 397 |
| `aqua_night` (~01:30) | **2.46** | **0.55** | 34 / 435 |
| `terra_night` (~22:30) | **2.29** | **0.60** | 34 / 435 |

Aqua covers 24 of 26 years (launched 2002-07-04); the 2000–2001 rows are
correctly empty rather than absent.

**The nocturnal prediction held.** Terra-night buffer_ring SUHII is **+2.29 °C**
(range +1.40 to +3.08), matching the ~+2 °C CMC-vs-District night difference
Phase 2 measured before Phase 3 existed. This is the strongest internal
consistency check the project has produced so far.

### The relaxed-QC sensitivity was worth running — Phase 2's obligation, discharged

Strict `terra_day` rests on a median of **17** CMC pixels and reads **0.62 °C
cooler** (2.98) than `terra_day_relaxed` on **34** pixels (3.60). The strict
daytime series is the spatially biased one, exactly as Phase 2 predicted from the
QC histogram. **Never quote strict daytime SUHII alone.**

### Other verified values

* **`ee.Reducer.stdDev()` is the POPULATION standard deviation** (numpy `ddof=0`).
  Probe `[10,12,14,16]`: `stdDev()` → 2.2360679775 = `np.std(ddof=0)`;
  `sampleStdDev()` → 2.5819888975 = `np.std(ddof=1)`. `uhi.zscore.ddof = 0` was
  already correct. **The last open question in Phase 3 is closed.**
* Hot pixels over Colombo District, dry season 2025, at 100 m: **225.2 km² at 1σ**,
  **31.0 km² at 2σ**.
* Zonal: **GN 557 / DS 13** features, both as expected. GN mean LST **30.1–42.7 °C**
  (median 121 px per division, min 18); DS **32.7–40.6 °C** (median 2817 px).
  Hottest GN divisions: New Bazaar 42.74, Lakshapathiya North 42.66,
  Kajugahawatta 42.04 °C.
* **MAUP is real and measurable**: the same surface has sd 2.74 / range 12.65 at
  GN against sd 2.62 / range **7.98** at DS. Coarser units average the extremes
  away — a DS hot-spot map is not a downsampled GN one.
* Drivers, 26 years × 5000 sampled pixels, R² **0.30–0.78**:
  NDBI **+10.34** °C/unit (positive in **26/26** years), built_fraction **+6.09**
  (**26/26**), MNDWI −3.42 (negative in 22/26), NDVI −3.12 (negative in 21/26).

### A finding to carry into Phase 5 — not a bug

NDVI's **partial** coefficient flips sign in 5 of 26 years and its sd (4.27)
exceeds its mean (3.12), while its **bivariate** correlation is a clean −0.51.
That gap is NDVI/NDBI collinearity in a dense city, and it is the concrete
motivation for CLAUDE.md's escalation path: emitting both the multivariate
coefficient and the simple correlation is what made it visible.

## PHASE 3b — figure legibility fixes (2026-08-09)

Run 7's figures were hard to read, and diagnosing them turned up a defect in an
acceptance check.

### The SUHII figure was rebuilt as small multiples

The original drew **12 lines on one axes** (6 sources × 2 rural definitions) with
**one colour per source shared by both definitions**. A legend swatch is far too
short to show a dash pattern, so the legend read as six duplicated pairs — the
reported problem. It is now a 2×3 grid, one panel per source, carrying the same
two lines everywhere: `buffer_ring` solid blue circles, `lcz_based` dashed orange
squares. Colour, dash and marker all vary together, so the figure survives
greyscale and colour-vision deficiencies. **The legend is now two entries
regardless of source count**, and a parametrised test pins that. The gap between
the lines is shaded, since that gap is the sensitivity the figure exists to show.
Panels share a y-axis deliberately: free axes would draw `aqua_night`'s ~0.5 °C
gap the same size as `landsat_dry`'s ~3 °C one. Pixel counts moved from a
separate panel into the panel titles, keeping caveat 2 satisfied.

### The UTFVI legend was drawn on top of the band it named

`loc="upper left"` put the legend inside the axes, so the red "Worst" swatch
landed on the red "Worst" fill and was invisible. It now sits below the axes with
an opaque frame, plus thin white separators between bands so Good/Normal/Bad
(2–3 % each) stop blurring together.

### The scatter's fitted line was unlabelled and over-extended

Added a two-entry legend, and the line is now **drawn** across the 1st–99th
percentile of x while still being **fitted** on every row — previously it ran out
to NDVI ≈ −0.9 where a handful of pixels live, overstating its support.

### ⚠️ The Step 1 mask check raised a FALSE ALARM — fixed

Run 7 printed `buffer_ring/urban 40.4 vs 37.7 km² (+7.3%) <-- CHECK THIS`. **The
masks were fine.** The check measured at **100 m** and compared against Phase 1's
**300 m** figure. The same quantity is recorded in this file as **40.18 km² at
30 m** and **37.70 km² at 300 m**, so 40.4 at 100 m is exactly what the
scale-dependence predicts.

The check now measures every mask at **30 m, 100 m and 300 m**, asserts pass/fail
**only on scale-matched pairs** (buffer_ring/urban against 40.18 @30 m and 37.70
@300 m, 2 % tolerance), and prints the other three masks as *informational* —
because this file never recorded which scale their Phase 1 values used, and
inventing a tolerance against an unknown-scale reference is how a false alarm
becomes a habit. A new cell also isolates the **water-mask substitution** effect
separately from the scale effect, so the two confounds are reported as two
numbers.

**Rule for later phases: never compare an area against a reference without
matching the reduction scale first.**

## PHASE 3b — Colab run 8 (2026-08-09)

Ran clean, no errors. **The mask diagnosis was confirmed completely, and the
figure fixes did not reach Colab.**

### The false alarm is fully explained ✅

| mask | 30 m | 100 m | 300 m | Phase 1 |
|---|---|---|---|---|
| buffer_ring/urban | 40.69 | 40.45 | **38.17** | 37.7 |
| buffer_ring/rural | 216.26 | 214.95 | **206.18** | 206.1 |
| lcz_based/urban | 462.28 | 464.17 | **458.96** | 458.5 |
| lcz_based/rural | 155.19 | 155.53 | **152.42** | 152.2 |

Both scale-matched checks passed: buffer_ring/urban **+1.3 % at 30 m** and
**+1.2 % at 300 m**. The masks were never wrong.

**Two new facts fall straight out of that table.**

1. **Phase 1 measured every mask at 300 m.** The 300 m column reproduces all four
   Phase 1 values to within **0.15 %** (206.18 vs 206.1; 458.96 vs 458.5; 152.42
   vs 152.2). The "unknown scale" caveat is therefore closed, and the notebook now
   asserts pass/fail on **all four** masks at 300 m instead of only two.
2. **The residual on buffer_ring/urban is entirely the water-mask substitution.**
   Isolating it: the combined mask gives **37.72 km²** at 300 m against Phase 1's
   **37.70** — a match to 0.02 km². The static mask gives 38.17, so the
   substitution is **+0.44 km² (+1.2 %)**, and the scale effect over 30→300 m is a
   separate 2.5 km². Two confounds, two numbers, as intended.

### ⚠️ The figure redesign never reached Colab — and the guard let it through

Run 8's SUHII figure is still the 12-line overlay and the UTFVI legend is still
inside the axes. Cause: Colab cloned **HEAD 62f6e94** (the Phase 3a commit), whose
`viz.py` has `plot_suhii_sensitivity` but none of the Phase 3b work. The notebook
being run was current; `src/` was one revision behind. **Nothing errored.**

**The stale-module guard should have caught this and did not.** Its `viz` entry
listed only `plot_suhii_sensitivity`, `plot_utfvi_class_shares` and
`plot_lst_vs_index` — names present in *both* revisions — so the guard was
vacuous for exactly the change it needed to detect.

Fixed, and the fix is verified by replaying the guard against `62f6e94`'s actual
`viz.py`: it now raises `missing {'viz': ['build_suhii_figure',
'build_utfvi_shares_figure', 'build_lst_vs_index_figure']}`, and passes on the
current tree.

> **RULE for every later phase: the stale-module guard must name at least one
> function introduced by the MOST RECENT revision of each module.** A guard listing
> only pre-existing names passes while running old code, which is worse than no
> guard — it manufactures confidence. Notebook 03's guard now says this in a
> comment, and its error message names "committed but not pushed" as the most
> likely cause, plus the hand-uploaded-notebook case where the `.ipynb` and `src/`
> can sit at different revisions.

## PHASE 3 SIGNED OFF — Colab run 9 (2026-08-09, HEAD 7d5896d)

Ran clean against the pushed Phase 3b commit. **Every check passes and all three
figures render correctly on real data.** Phase 3 is complete.

### All five scale-matched mask checks PASS

| mask | scale | measured | Phase 1 | Δ |
|---|---|---|---|---|
| buffer_ring/urban | 30 m | 40.69 | 40.18 | +1.3 % |
| buffer_ring/urban | 300 m | 38.17 | 37.70 | +1.2 % |
| buffer_ring/rural | 300 m | 206.18 | 206.10 | **+0.0 %** |
| lcz_based/urban | 300 m | 458.96 | 458.50 | **+0.1 %** |
| lcz_based/rural | 300 m | 152.42 | 152.20 | **+0.1 %** |

Three of the five reproduce Phase 1 to a tenth of a percent — the Phase 3 masks
are demonstrably the Phase 1 masks. The residual on `buffer_ring/urban` is the
water-mask substitution (**+0.44 km²**; the combined mask returns 37.72 against
Phase 1's 37.70), not a geometry change.

### Figures verified on real data

The SUHII figure is now small multiples with a **two-entry legend**, panel gaps
matching the summary table exactly: `landsat_dry` **3.21 °C**, `aqua_day` 2.89,
`aqua_night` 1.91, `terra_day_relaxed` 1.81, `terra_night` 1.69, `terra_day` 1.42.
The UTFVI legend sits below the axes with all six classes visible including
"Worst". The scatter carries a labelled OLS fit clipped to the 1st–99th
percentile.

`ee.Reducer.stdDev()` re-confirmed as `ddof=0`; Terra-night buffer_ring SUHII
re-confirmed at **+2.29 °C**.

### Phase 3 deliverables (all in `data/outputs/`)

`suhii_2000_2025.csv` · `utfvi_class_shares_2000_2025.csv` ·
`lst_by_gn_2020s.csv` (557 rows) · `lst_by_ds_2020s.csv` (13 rows) ·
`lst_driver_ols_by_year.csv` · `lst_driver_correlations_by_year.csv`, plus the
figures in `figures/`.

**Phase 4 (Mann-Kendall + Sen's slope + Benjamini-Hochberg FDR) is unblocked.**

## PHASE 3 — implementation record (code written 2026-08-09)

`uhi_metrics.py` (new, replacing the stub), three `viz.py` figures, an additive
`aoi.py` change, one `composites.py` reuse refactor, `config/params.yaml` additions,
`tests/test_uhi_metrics.py` (new) and `notebooks/03_uhi_metrics.ipynb` (rebuilt from
the 1-cell stub). **366 tests pass locally** (was 249).

> **Nothing here has been executed against Earth Engine.** Claude Code has no EE
> credentials. Every server-side function is unverified until notebook 03 runs.

### User decisions taken this session (2026-08-09)

1. **UTFVI reference = per-year AOI mean** (`uhi.utfvi.reference`), the standard
   formulation, so the published class breaks keep their meaning. **Consequence that
   must travel with every UTFVI output:** the index measures *within*-year spatial
   structure, so a uniformly warming city shows **no class change at all**. Epoch-to-
   epoch drift is redistribution of heat, never evidence of warming.
2. **Six SUHII sources**: Landsat dry-season, MODIS Terra day/night, Aqua day/night,
   plus `terra_day_relaxed` — the relaxed-QC daytime run Phase 2 sign-off required.
3. **Driver OLS on sampled pixels**, 5000/year at 100 m (not GN-aggregated).

### Two blockers found and fixed

1. **`aoi.urban_mask`/`rural_mask` hard-coded `water_exclusion_mask(params)`** with no
   region and no override. That function composites Landsat internally and the composite
   is re-instantiated *for every image it masks* — the Colab run 3 failure, which across
   26 years × 6 sources would have been unrunnable. All three rural-reference functions
   now take an optional `water` image, **defaulting to `None` = existing behaviour**, so
   Phase 1's notebooks and call sites are untouched. Phase 3 passes
   `aoi.static_water_mask`. Cost of that substitution is the already-measured **−0.074 °C**
   on the CMC 2025 mean (notebook 02) — do not re-derive it.
2. **No mask-based zonal reduction existed.** Every `composites.py` zonal path takes an
   `ee.Geometry`. `uhi_metrics.masked_pair_image` stacks the LST band masked to every
   method's urban and rural mask into one image, so a **single `reduceRegion` returns all
   four means and all four counts** for both rural definitions at once.

### Open question — RESOLVED in run 7 ✅

**`ee.Reducer.stdDev()`: population (ddof 0) or sample (ddof 1)?** The Earth Engine API
docs never state it, and a separate `ee.Reducer.sampleStdDev()` exists. Rather than
guess, `uhi.zscore.ddof` was made an explicit params value and notebook 03 Step 6
measured it. **Answer: population, `ddof=0`** (see the run 7 section above for the
probe and the four numbers). The configured 0 was already right, so `zscore_array` and
`lst_zscore` compute the same statistic **exactly**, not to O(1/n). The cell stays in
the notebook as a regression guard in case Earth Engine ever changes the reducer.

### Bug found and fixed while testing

`pandas.Series.std(ddof=0)` of a **constant** column is not reliably exactly `0.0` — it
came back as `2.78e-17` in one frame construction and `0.0` in another. Both the
`fit_drivers` constant-predictor guard and the `plot_lst_vs_index` fit guard used
`std() == 0` / `std() > 0`, so a constant predictor slipped through into a singular
design matrix (statsmodels returns a NaN row that reads like a coefficient; `np.polyfit`
emitted `RankWarning` and drew a meaningless line). All three sites now count distinct
values (`nunique() <= 1`), which is exact. Pinned by parametrised regression tests.

### Catalog verification for the Phase 3 additions

Re-checked `JRC/GHSL/P2023A/GHS_BUILT_S` on the GEE catalog 2026-08-09: `built_surface`
is **m² of built surface per 100 m cell**, 5-year epochs 1975–2030. So the built **fraction**
is `built_surface / 10000` → `datasets.ghsl_built.cell_area_m2: 10000` and
`epoch_interval_years: 5`. `uhi_metrics.built_up_fraction()` snaps the requested year
**down** to its epoch and sets an `epoch` property: a 2023 built fraction is really the
2020 layer, and must be reported as such. No discrepancy with CLAUDE.md or params.

### Round-trip budget (what to expect in Colab)

| Product | Round trips |
|---|---|
| SUHII table, 6 sources × 2 definitions × 26 yr, batched 4 | ~42 |
| UTFVI class series, 26 yr batched 4 | ~7 |
| UTFVI epoch maps + thumbnails | ~6 |
| z-scores, hot-pixel areas, thumbnails | ~5 |
| Zonal by division (GN + DS) | 2 |
| Drivers, 1 sample per year | 26 |

### New caveats that must travel into Phase 4+

1. **UTFVI's per-year reference hides uniform warming.** The epoch maps show
   redistribution only. Phase 4's Mann-Kendall/Sen products are what measure trend, and
   must never be described as "confirming" the UTFVI maps.
2. **MODIS SUHII is a coarse-unit statistic.** At the 1 km reduction scale the 100 m LCZ
   masks and the rasterised CMC polygon are resampled, so a 1 km pixel with *any* urban
   fraction counts as urban. The CMC holds ~40 MODIS pixels (Phase 2, measured), so the
   MODIS urban mean is edge-contaminated. Always read it against `urban_pixels`.
3. **Day and night are not an equal-confidence pair** — 1 K vs 3 K accepted uncertainty
   (Phase 2 caveat 2, unchanged and still binding).
4. **Driver p-values are anti-conservative.** Sampled pixels are spatially autocorrelated;
   OLS standard errors are too small. This is a screening device until Phase 5 runs
   residual Moran's I → spatial lag/error → GWR/MGWR.
5. **Built-up fraction is a 5-year epoch value**, not an annual measurement.
6. **The two rural definitions differ by construction** (Phase 1 caveat 3, unchanged):
   their spread is the required sensitivity, not a discrepancy to reconcile. Run 7
   measured that spread at **1.4–3.2 °C** depending on source — large enough that a
   single unqualified SUHII number for Colombo would be indefensible.
7. **Never compare an area against a reference without matching the reduction scale**
   (new, from the run 7 false alarm). Areas on a ragged coastal mask are
   scale-dependent by several percent; record the scale with every area you write
   down. **All Phase 1 and Phase 3 mask areas are at 300 m** (established in run 8).
8. **Strict MODIS daytime QC is spatially biased, measurably** (Phase 2 predicted it,
   run 7 quantified it): strict `terra_day` reads 0.62 °C cooler on half the pixels.
   Quote it only alongside `terra_day_relaxed`.

### What run 7 confirmed (this section is now history, not a to-do)

`notebooks/03_uhi_metrics.ipynb` ran top to bottom on the first attempt. The two
cells designed to fail loudly both behaved:

* **Step 3** checked Terra-night buffer_ring SUHII against the ~+2 °C
  CMC-vs-District nocturnal difference Phase 2 measured. Result **+2.29 °C** —
  **PASS**, and the single strongest internal consistency check in the project.
* **Step 1** flagged `buffer_ring/urban` — and was itself wrong. It compared a 100 m
  measurement against a 300 m reference. See "PHASE 3b" above; the check is now
  scale-matched and the masks were never in question.

Also as expected and **not** a bug: Aqua rows for 2000–mid-2002 came back empty with
`urban_pixels = 0` (Aqua launched 2002-07-04), tripping the caveat-2 "mostly empty"
warning — which is that warning working correctly.

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

## PHASE 2 — LST pipeline (complete)

`landsat.py`, `indices.py`, `composites.py`, `modis.py`, a `viz.py` addition and
`notebooks/02_lst_pipeline.ipynb`. **249 tests pass locally** (was 90); every module
still imports without `earthengine-api`. Six Colab iterations; the verified outputs
are in the sign-off table below.

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

## ✅ PHASE 2 SIGNED OFF — verified reference values (Colab run 6, 2026-08-08)

These are the authoritative Phase 2 outputs. Quote them in the report; re-running
notebook 02 should reproduce them.

| Quantity | Verified | Note |
|---|---|---|
| Landsat scenes over the district, 2000–2025 | **1674** | L5 214 / L7 402 / L8 794 / L9 264 |
| `PROCESSING_LEVEL` filter effect | **none** | identical counts filtered/unfiltered, all four sensors |
| Dry-window years with zero scenes | **none** | min 2 (2000), typically 15–36 |
| Dry-season 2025 `obs_count` over CMC | **min 4, median 10, max 14** | healthy; no zero-coverage pixels |
| Static vs combined water mask, CMC 2025 mean | **−0.074 °C** | the cheap mask is a fair substitute for series work |
| Landsat annual mean, CMC (100 m) | **37.3–42.2 °C** | 4206 valid pixels every year |
| MODIS Terra day, CMC | **31.7–35.1 °C** | only 13–23 of 40 possible 1 km pixels |
| MODIS Aqua day, CMC | **35.5–39.0 °C** | warmer than Terra day, correct for 13:30 |
| MODIS Terra night, CMC | **22.7–25.2 °C** | 40/40 pixels — full coverage |
| MODIS Aqua night, CMC | **22.5–24.1 °C** | 40/40 pixels |
| MODIS Terra night, District | **20.8–23.0 °C** | 643/643 pixels |
| Index means, CMC Jan–Mar 2025 | NDVI 0.372, NDBI 0.025, MNDWI −0.415, EVI 0.196, SAVI 0.192, albedo 0.124 | all physically plausible |

**A nocturnal UHI signal is already visible**: CMC Terra night runs ~2 °C above
Colombo District Terra night in every year. That is Phase 3's result arriving
early, and it is the cleanest signal in the whole dataset — night has full pixel
coverage where daytime MODIS does not.

### Landsat-vs-MODIS offset, decomposed (the run 6 deliverable)

| Comparison | Mean offset | Attributes to |
|---|---|---|
| CMC, Landsat 100 m vs Terra day | **+5.92 °C** | (baseline) |
| CMC, Landsat 1000 m vs Terra day | **+5.13 °C** | resolution ≈ **0.79 °C** |
| District, Landsat 100 m vs Terra day | **+3.20 °C** | scope/sample ≈ **2.72 °C** |

So the ~6 °C CMC gap is mostly **small-sample and scope**, only ~0.8 °C is
resolution, and roughly **+3.2 °C** survives as a genuine sensor/sampling
difference. That residual — not the 5.9 — is the number the report may quote,
and it must be quoted with its cause stated: Landsat is an instantaneous
clear-sky snapshot, MOD11A2 is an 8-day average of clear-sky daily retrievals,
and the two carry different emissivity treatments. At district scale the two
series track each other closely in shape, which is the meaningful agreement.

### ⚠️ Caveats that must travel into later phases

1. **The strict daytime MODIS QC is spatially biased.** Measured over the CMC:
   mandatory-QA class 0 is only **3.7%** of daytime observations, so the
   CLAUDE.md policy discards ~96% of them on that field alone — and it fails
   hardest exactly where the UHI lives. The CMC keeps 13–23 of 40 possible 1 km
   pixels per year (32–58%); Colombo District keeps 594–611 of ~643 (92–95%).
   **Phase 3 must run a relaxed sensitivity** (`modis.annual_lst(mandatory_qa_max=1)`,
   now supported) and report both, rather than treat the strict CMC daytime
   series as authoritative.
2. **Night is accepted at ≤3 K uncertainty, day at ≤1 K** (decision 7). Night-time
   SUHII is weaker evidence than daytime SUHII and must never be presented as an
   equal-confidence pair.
3. **`obs_count` has geometric structure, not just meteorological.** The 2025
   observation-count map shows clear WRS-2 **path side-lap banding** — coverage
   roughly doubles in the overlap strips, one of which runs through the CMC.
   Phase 4 trend confidence therefore varies with orbit geometry as well as
   cloud, and any map of trend significance will inherit those stripes.
4. **Day and night trends diverge.** Terra night rises ~+2 °C over 2000–2025 in
   both zones while Terra day CMC falls. Before reading anything into that,
   Phase 4 must account for **Terra's orbital drift after ~2020**, which moves
   its overpass time and contaminates end-of-series daytime trends. Aqua and
   Landsat do not share that drift and can be used to test it.
5. **The ~40 °C Landsat annual means are not air temperature and not corrected
   for clear-sky sampling bias.** They are clear-sky, ~10:30, 30 m land-surface
   values over a dense urban core. Both facts belong in the report.
6. **Landsat 5 has zero 2012 scenes** — it stopped acquiring in November 2011,
   so the L7-only gap is **2012-01 to 2013-03**, wider than CLAUDE.md's
   "2012-05" implies. That stretch rests on SLC-off ETM+ alone.

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

- **2026-08-08 — Phase 2 SIGNED OFF**: run 6 confirmed both rev-6 fixes. The QC
  histogram proved the night diagnosis exactly — mandatory-QA class 0 is **0.0%** at
  night — and incidentally exposed that the strict DAY policy keeps only 3.7% of
  observations and fails hardest over the coastal core, which is now a Phase 3
  sensitivity requirement. Night LST returned 22.5-25.2 °C with full pixel coverage,
  and CMC night runs ~2 °C above district night — a nocturnal UHI signal already
  visible. The offset decomposition attributed the ~6 °C CMC gap to scope (2.7),
  resolution (0.8) and a ~3.2 °C genuine sensor residual. All four figures reviewed.
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
