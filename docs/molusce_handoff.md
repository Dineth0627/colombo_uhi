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
| `lulc_district_..._30m_2018.tif` | band 1 `label`, band 2 `observed` | **Initial** state map |
| `lulc_district_..._30m_2021.tif` | same | calibration end / validation start |
| `lulc_district_..._30m_2024.tif` | same | **Final** state map, and the reference for validation |
| `rf_training_sample_district_..._100m_2020s_landsat_oli_dry.csv` | sampled pixels | not used by MOLUSCE; Track A only |

All three rasters are written on **EPSG:32644 at 30 m**, over the same region
(`aoi.analysis_region`), so they already satisfy MOLUSCE's requirement that every
input share resolution, extent and pixel dimensions. If QGIS's *Check geometry*
still complains, the cause is almost always that one file was re-exported with a
different region — re-run the whole Step 4 loop rather than patching one file.

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
* **Resample every factor to the 30 m CA grid before loading**, on EPSG:32644,
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
* **Neighbourhood**: 1 pixel. At 30 m this is a 3×3 window, comparable to the
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
Do this on the **2021 → 2024** hold-out, not on 2018 → 2024.

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
| Simulated 2024 map (from the 2018→2021 calibration) | `molusce_projected_2024.tif` | scored with the same Kappa / FoM / Pontius functions as the Python CA |
| Simulated 2030 map | `molusce_projected_2030.tif` | compared cell-by-cell against the Python CA |
| Simulated 2036 map | `molusce_projected_2036.tif` | same |
| Transition matrix export | `molusce_transition_matrix.csv` | checked against `prediction.transition_probabilities` |
| Certainty raster (2030) | `molusce_certainty_2030.tif` | uncertainty discussion in the report |

They must be single-band integer rasters on **EPSG:32644 at 30 m**, on the same
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
