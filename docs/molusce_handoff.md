# MOLUSCE handoff — CA-Markov land-cover projection in QGIS

**Phase 6, Track B.** This document is the contract between the Earth Engine
export step (which Claude Code wrote and you run in Colab) and the cellular
automata step (which you run in QGIS). It exists so the two halves of Track B
can be compared rather than merely both existing.

---

## 0. What this is for, and what it is not

Everything MOLUSCE produces here is a **conditional scenario projection**, not a
forecast. It answers: *if the class-to-class transition rates observed between
2018 and 2024 continued, and if change kept clustering the way it has, where
would the land-cover classes be in 2030?* It says nothing about whether those
rates will continue — planning decisions, the 2026 economy, and the port
expansion are all outside the model.

**`src/colombo_uhi/prediction.py` implements the same Markov chain and a simpler
cellular automaton in pure Python.** That is deliberate. MOLUSCE's transition
potential (an ANN/MLP over the driver rasters) is richer than the Python
neighbourhood filter, so the two will not agree exactly. Where they *disagree*
is informative; where they agree, the result does not depend on one tool's
defaults. Run both. Report both.

---

## 1. Software status (checked 2026-08-13)

| Item | Value |
|---|---|
| Plugin | NextGIS MOLUSCE, from the official QGIS plugin repository |
| Latest version | **5.3.0**, released 2026-04-14 |
| Minimum QGIS | **3.32.0** |
| Maintainer | NextGIS (`github.com/nextgis/qgis_molusce`) |
| Install | QGIS → Plugins → Manage and Install Plugins → search "MOLUSCE" |
| Where it appears | Raster menu, after install |

Older write-ups describe MOLUSCE as a QGIS 2.x-only plugin. That has not been
true since the 4.x line; it is actively maintained. Use a current QGIS (3.32 or
newer) and install from the plugin manager rather than hunting for a fork.

---

## 2. Inputs Colab hands you

Notebook `06_prediction.ipynb`, Part 1, Step 4 exports these to your Drive
folder (`exports.drive_folder`, default `colombo_uhi_exports`). Filenames come
from `exports.export_name`, so they follow
`{product}_{aoi}_{startyear}_{endyear}_{res}m_{suffix}`:

| File | Contents | Role in MOLUSCE |
|---|---|---|
| `lulc_district_..._100m_2017.tif` | band 1 `label`, band 2 `observed` | **Initial** state map |
| `lulc_district_..._100m_2021.tif` | same | **Final** state map for calibration |
| `lulc_district_..._100m_2025.tif` | same | the reference for validation |
| `lulc_district_..._100m_2018.tif`, `..._2024.tif` | same | the 3-year sensitivity interval, and the projection base |
| `rf_training_sample_district_..._100m_2020s_landsat_oli_dry.csv` | sampled pixels | not used by MOLUSCE; Track A only |

Every raster is written on **EPSG:32644 at 100 m**, clipped to **Colombo District**
(`prediction.work_region`, ~699 km²), so they already satisfy MOLUSCE's requirement
that every input share resolution, extent and pixel dimensions. If QGIS's *Check
geometry* still complains, the cause is almost always that one file was re-exported
with a different region — re-run the whole Step 4 loop rather than patching one file.

100 m, not 30 m: the projected land cover feeds a random forest **fitted** at 100 m,
and running the automaton finer would apply that model at a scale it was never fitted
at. One grid for everything.

> **Two earlier runs were discarded over the region.** Run 1 used Western Province
> plus a 25 km ring (18,090 km²) as the analysis unit; run 2 used the district's
> *bounding box* (1,256 km², 46 % of it outside the district) because nothing clipped
> to the geometry. If your rasters do not measure ~699 km² of classified cells,
> stop and re-export.

### The `observed` band is not optional

Each raster carries **two** bands:

* **band 1, `label`** — the Dynamic World modal class code for that year;
* **band 2, `observed`** — 1 where Dynamic World actually classified the cell,
  0 where it never did.

A single-band 0-to-8 GeoTIFF cannot distinguish "class 0, water" from "never
classified", because masked pixels are written as 0. Phase 5 lost a full Colab
run to exactly that: Dynamic World "green" appeared to grow 5.4× from 2016 to
2024, and almost all of the growth was Sentinel-2 coverage rather than land
cover. **Before loading anything into MOLUSCE, mask `label` to `observed == 1`**
(Raster → Raster Calculator, or `gdal_calc`), or the automaton will treat
unclassified coastline as reclaimed water.

---

## 3. Class codes

Dynamic World's legend, and which codes Phase 6 retains
(`prediction.ca_markov.classes`):

| Code | Class | Retained | Colour |
|---|---|---|---|
| 0 | Water | yes — **immutable** | `#419bdf` |
| 1 | Trees | yes | `#397d49` |
| 2 | Grass | yes | `#88b053` |
| 3 | Flooded vegetation | **no** | `#7a87c6` |
| 4 | Crops | yes | `#e49635` |
| 5 | Shrub and scrub | yes | `#dfc35a` |
| 6 | Built | yes | `#c4281b` |
| 7 | Bare | yes | `#a59b8f` |
| 8 | Snow and ice | **no** | `#b39fe1` |

Class 8 does not occur at 6.93° N. Class 3 is a handful of unstable pixels whose
transition row would be estimated from almost nothing. Both are dropped so the
Markov chain does not carry a row fitted to noise.

Water (0) is held **immutable** in the Python CA. MOLUSCE has no equivalent
switch, so if its output "reclaims" harbour cells, that is a known difference
between the two implementations — record it rather than editing the raster.

Apply the colours above via a Paletted/Unique-values style so your QGIS panels
and the notebook's figures are directly comparable by eye.

---

## 4. Spatial variables (factor layers)

MOLUSCE's ANN needs driver rasters. Export these from the same notebook if you
want them (Part 1, Step 6 exports the GHSL layer; the rest come from the Phase 5
covariate stack):

* `elevation_m` — SRTM, static;
* `dist_coast_km` — static;
* `built_fraction` — GHSL, the 2020 epoch for calibration;
* `pop_density` — WorldPop 2020.

Two warnings that apply to both implementations:

* **WorldPop stops at 2020** (`datasets.worldpop.availability`). Population is
  held constant, and `prediction.held_constant` records that. A projected
  population would be a second unvalidated model stacked on the first.
* **Resample every factor to the 100 m CA grid before loading**, on EPSG:32644,
  matching extent and pixel count exactly. A factor on an inherited grid is what
  made a Phase 5 export serialise at 159 MB.

Run MOLUSCE's *Evaluating correlation* tab before training. Phase 5 measured
severe collinearity in this study area — VIF 28.4 for NDBI and 14.3 for NDVI at
GN level — so redundant factors are the expected case, not a surprise.

---

## 5. The run, tab by tab

### 5.1 Inputs
* **Initial**: masked `label` from **2018**.
* **Final**: masked `label` from **2024**.
* **Spatial variables**: the factor layers from §4.
* Press **Check geometry**. It must pass before any other tab unlocks.

### 5.2 Evaluating correlation
Optional but run it. Record the Cramer / Joint Information Uncertainty values
for the categorical factors and Pearson for the continuous ones. Drop a factor
only if you say so in `PROGRESS.md`.

### 5.3 Area changes
Produce **Class statistics** and the **Transition matrix**.

> **Export the transition matrix and keep it.** This is the single number set
> that lets you check MOLUSCE against Python. `prediction.transition_matrix` and
> `prediction.transition_probabilities` compute the same thing from the same two
> rasters, and the notebook prints them. Row-normalised, the two should agree to
> rounding. **If they do not, stop** — it means the two tools are reading
> different pixels, and every result downstream of that is meaningless.

### 5.4 Transition potential modelling
* **Method**: Artificial Neural Network (multi-layer perceptron).
* **Sampling mode**: stratified. Random sampling under-represents the rare
  transitions, which are the only ones that matter for a change model.
* **Sample count**: start at 10 000 and raise it if the training error plateaus
  high.
* **Neighbourhood**: 1 pixel. At 100 m this is a 3×3 window, comparable to the
  Python CA's `neighbourhood_radius_cells: 2` (5×5) — not identical, which is
  another expected source of divergence.
* **Learning rate**: 0.01 to start.
* **Max iterations**: raise until the validation error stops falling, then stop.
  MOLUSCE will happily overfit if you keep going.
* **Save the trained model.** Without it the run is not reproducible.

### 5.5 Cellular automata simulation
* **Iterations**: one iteration is one calibrated interval. The calibration pair
  is 2018 → 2024, so **one interval is 6 years**:
  * **1 iteration → 2030**
  * **2 iterations → 2036**
* 2035 is 1.83 intervals from 2024. A fractional power of a transition matrix is
  not guaranteed to be a valid transition matrix, so Phase 6 does not compute
  one. `prediction.resolve_projection_steps` rounds to the nearest whole step and
  returns the **effective year** beside the requested one — 2035 becomes 2036,
  with `offset_years = 1`. Every figure and table quotes the effective year. Do
  the same in QGIS: label the two-iteration output **2036**, not 2035.
* Save the **simulated map**, the **transition potential map** and the
  **certainty function**. The certainty raster is the closest thing MOLUSCE gives
  you to an uncertainty estimate; it belongs in the report.

### 5.6 Validation
Do this on the **2021 → 2025** hold-out, not on the projection pair.

That is the PRIMARY interval since run 3: calibrate **2017 → 2021** (4 years),
simulate one iteration to **2025**, validate against observed 2025. The 3-year
`2018 → 2021 → 2024` triplet is the sensitivity — run it too if you have the
patience, since the difference between them is the most interesting thing Track B
measured.

Recalibrate with Initial = 2018, Final = 2021, simulate one iteration (3 years)
to 2024, and validate against the **observed 2024** raster. Using the same pair
to calibrate and validate measures memorisation, not skill.

MOLUSCE reports Kappa and an error map (persistent / correct / error).

> **Kappa alone is not enough here, and the notebook will say so.** Over a
> three-year interval most cells do not change, so a null projection that copies
> the 2021 map already scores a high Kappa.
> `prediction.persistence_baseline_kappa` computes exactly that null, and
> `prediction.figure_of_merit` scores only the cells that **changed** — the one
> metric a no-change projection cannot game. Report all three, plus the Pontius
> quantity/allocation split from
> `prediction.quantity_allocation_disagreement`.
>
> Floors, from `prediction.ca_markov.validation`: `min_kappa: 0.60`,
> `min_figure_of_merit: 0.10`. These are **floors, not targets**. Failing one is
> a result to report, never something to tune away.

---

## 6. What to bring back to Colab

Put these in `data/interim/` before running notebook 06, Part 3:

| File | Name it | Used for |
|---|---|---|
| Simulated 2025 map (from the 2017→2021 calibration) | `molusce_projected_2025.tif` | scored with the same Kappa / FoM / Pontius functions as the Python CA |
| Simulated 2030 map | `molusce_projected_2030.tif` | compared cell-by-cell against the Python CA |
| Simulated 2036 map | `molusce_projected_2036.tif` | same |
| Transition matrix export | `molusce_transition_matrix.csv` | checked against `prediction.transition_probabilities` |
| Certainty raster (2030) | `molusce_certainty_2030.tif` | uncertainty discussion in the report |

They must be single-band integer rasters on **EPSG:32644 at 100 m**, on the same
grid as the exports in §2. Part 3 reads them with
`prediction.read_lulc_raster`, which **raises** if the grid does not match —
that is intentional, because a silently misaligned comparison would produce a
confusion matrix full of spurious "change".

**If you skip the MOLUSCE round trip**, Part 3 says so and stops there. The
Python CA-Markov result stands on its own, is fully validated, and Phase 6 signs
off without QGIS. Nothing downstream is blocked.

---

## 7. Recording the outcome

In `PROGRESS.md`, under the Phase 6 record, note:

1. MOLUSCE version and QGIS version actually used;
2. whether the transition matrices agreed (§5.3) and to what tolerance;
3. Kappa, persistence-baseline Kappa, figure of merit and the Pontius split, for
   **both** implementations on the 2024 hold-out;
4. the cell-level agreement between the two 2030 maps;
5. every ANN setting you changed from §5.4, and why.

A disagreement between the two implementations is a finding. Do not reconcile
them by editing one until it matches the other.

---

## 8. What you are now comparing MOLUSCE against

Colab runs 1–3 turned the Python CA-Markov into a **known-negative baseline**, and that
makes Part 3 more interesting rather than less.

Measured over Colombo District, on land only, across two class schemes and both calibration
intervals Dynamic World's record supports:

| interval | step | figure of merit | Kappa − no-change baseline |
|---|---|---|---|
| 2018 → 2021, validated 2024 | 3 yr | 0.0022 | −0.0010 |
| 2017 → 2021, validated 2025 | 4 yr | **0.0740** | −0.0344 |

Quantity disagreement 0.003–0.032 against allocation disagreement 0.059–0.071. In plain
terms: **the automaton gets the amount of each class nearly right and puts it in the wrong
place.** A longer step helps enormously with locating change (34× the figure of merit) but
buys its hits with false alarms, so nothing beats a map that says "nothing changes". Four
years is the longest equal-step triplet the record allows, so there is no further interval
to try.

### Why MOLUSCE might do better, and what to look for

The Python CA allocates **net demand** — the difference between the Markov-projected class
totals and the current ones — and places it by a neighbourhood filter. MOLUSCE's ANN learns
a **transition potential per class from the driver rasters**, which is a genuinely different
mechanism and the obvious candidate for why allocation failed here.

So the comparison to run is not "which is right" but:

1. **Does MOLUSCE's figure of merit clear 0.10 where the Python CA cannot?** Score it on the
   **2017 → 2021 → 2025** interval — that is the primary now. Part 3 scores it through the
   same `prediction.validate_projection` call, on the same mask, so the numbers are directly
   comparable.
2. **Does its Kappa beat the no-change baseline?** Part 3 prints `kappa_above_null` for both.
   That is the number to read first; the baseline is identical for each, so whichever
   implementation sits closer to it has located less of the change.
3. **Does the quantity/allocation split move?** If MOLUSCE's allocation disagreement drops
   while its quantity disagreement stays put, the ANN transition potential is the difference,
   and that is a publishable finding about method choice.

If MOLUSCE also fails to beat the baseline, that is a stronger result than either
implementation alone: two independent allocators, one neighbourhood-based and one
ANN-based, cannot place Dynamic World's land-cover change over Colombo at these intervals.
Report it that way.

**Do not tune either implementation until it clears the floors.** A model fitted to its own
validation set has no validation set, and the whole value of this comparison is that neither
was.
