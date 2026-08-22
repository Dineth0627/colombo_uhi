# Limitations

> **This file is generated.** Every parameter below is read from
> `config/params.yaml` and from the committed products in `data/outputs/`
> by `colombo_uhi.reporting`. Do not edit it by hand - edit the config, then
> re-run `reporting.write_docs(params)`. `tests/test_reporting.py` fails if
> the committed copy disagrees with what the code produces, which is what
> stops a threshold being changed in one place and left stale in another.


Read this document beside every figure and every number in the report. Nothing here is a disclaimer in the legal sense: each entry changes how a specific result may be read, and several of them were found only by measuring something the documentation said would be fine.

## Part 1 - standing caveats

These are reproduced verbatim from `config/params.yaml`, where they are stamped onto figures and exported metadata by the code itself.

### LST not air temp

`caveats.lst_not_air_temp`

> Values are LAND SURFACE TEMPERATURE (LST), not air temperature. Never label outputs as air temperature or 'temperature felt by residents'; surface UHI can be roughly 2x the canopy-air UHI.

### Valid obs required

`caveats.valid_obs_required`

> Every composite or trend product must be interpreted alongside its per-pixel valid-observation count; tropical cloud cover means only a minority of scenes are usable.

### Scenario not forecast

`caveats.scenario_not_forecast`

> Predictive outputs are conditional scenario projections, NOT forecasts. They must ship with validation metrics (RMSE, R2, Kappa) and explicit uncertainty language.

### Single overpass

`caveats.single_overpass`

> Landsat observes once near ~10:30 local solar time; daytime-only. Night UHI comes only from MODIS (Terra ~22:30, Aqua ~01:30 local).

### Sensitivity reporting

`caveats.sensitivity_reporting`

> Report rural-reference and aggregation-unit (MAUP) sensitivity, never a single unqualified number.

### Trend not causal

`caveats.trend_not_causal`

> A Sen's slope is a rate of change, not an explanation for it. Trend maps stratified by land cover show where warming occurred by TODAY'S land cover; they do not attribute warming to land-cover change.

### FDR dependence

`caveats.fdr_dependence`

> Benjamini-Hochberg controls the false discovery rate under independence or positive regression dependency. Land surface temperature is strongly spatially autocorrelated, so the realised false-discovery proportion varies far more than its controlled expectation; significant-area figures are a screening result, reported alongside Benjamini-Yekutieli.

### Within epoch only

`caveats.within_epoch_only`

> Epoch cluster maps are computed on the POOLED Landsat series, whose inter-sensor offsets over Colombo are material (L5-L7 +1.78 degC, L7-L8 -2.48 degC). Moran's I, LISA and Gi* are within-epoch statistics, so a spatially uniform sensor step cancels and the cluster GEOGRAPHY is valid. No epoch-to-epoch temperature magnitude may be read off these maps; the Phase 4 Mann-Kendall and Sen's slope products are what measure change.

### Zonal not pixel

`caveats.zonal_not_pixel`

> These statistics describe administrative POLYGONS, not pixels and not people. A coefficient fitted across 557 GN divisions is a property of that aggregation; reading it as an individual-level or pixel-level relationship is the ecological fallacy. The same analysis at DS level gives different numbers by construction, which is why both are reported.

### Euclidean not network

`caveats.euclidean_not_network`

> The 300 m of the 3-30-300 rule is a WALKING distance. These service areas are straight-line, so they ignore the Kelani River, Beira Lake, the coastal railway and walled compounds, and therefore OVERSTATE access everywhere by an unknown amount. Each result is reported beside a 231 m variant (a 1.3 detour ratio); the gap between the two is the size of the uncertainty. Green space is also counted whether it is public or private, so compliance is an UPPER BOUND, and the '3 trees from every window' component cannot be measured from satellite data at all and is reported as unmeasured.

### MCDA weights are judgements

`caveats.mcda_weights_are_judgements`

> The AHP criterion weights are JUDGEMENTS, not measurements. They were argued by the analyst from the literature and from this project's own measurements, not elicited from stakeholders, and a different defensible set of judgements gives a different ranking. The consistency ratio tests only whether the judgements are self-consistent, never whether they are right. The criteria are also strongly intercorrelated over Colombo, so the leave-one-out ablation - not the weights - is what says how much the method adds over ranking by land surface temperature alone.

### Trend power floor

`caveats.trend_power_floor`

> A zero significant-pixel count is not the same as an absence of warming. Mann-Kendall on n observations is bounded, so there is a smallest p-value the test can ever return; when that floor lies above the Benjamini-Hochberg threshold for the number of pixels tested, NO pixel can be reported significant whatever the temperature did. Over Colombo the single-sensor 12-year Landsat series is exactly this case. Where it holds, the Sen's slope field is an unconstrained estimate and its sign must NOT be read as cooling or warming; quote the detection limit instead, and take the trend evidence from the 26-year MODIS night series.

### Colour is not the only channel

`caveats.colour_is_not_the_only_channel`

> Colour alone never carries a class on these figures. Every categorical map also states its classes in an ordered legend, and any category the Phase 8 colour-vision check could not separate is additionally distinguished by hatching or by direct labelling. Readers with a colour-vision deficiency, and readers looking at a greyscale photocopy, must be able to recover the same classes.

### Figures are derived not authored

`caveats.figures_are_derived_not_authored`

> Every number on a report figure, and every parameter quoted in docs/methods.md, is read from config/params.yaml or from a committed product in data/outputs/ at render time. Nothing is typed into a caption by hand. A test regenerates both documents and fails if the committed copy disagrees, so a threshold cannot be changed in one place and left stale in another.

## Part 2 - limitations found during implementation

### 1. Collection 2 inter-calibration fails over Colombo

The Landsat Collection 2 documentation states that the archive is inter-calibrated across TM, ETM+ and OLI, which would mean no manual harmonisation is needed. Tested empirically on dry-season CMC means, it is not true here:

| pair | mean offset (°C) | t | overlap years | window | verdict |
|---|---|---|---|---|---|
| landsat5 - landsat7 | +1.78 | +2.72 | 8 | 2004-2011 | **material** |
| landsat7 - landsat8 | -2.48 | -3.60 | 10 | 2014-2023 | **material** |
| landsat8 - landsat9 | -0.40 | -0.67 | 4 | 2022-2025 | **negligible** |
| landsat5 - landsat8 | - | - | 0 | - | **insufficient_overlap** |
| landsat5 - landsat9 | - | - | 0 | - | **insufficient_overlap** |
| landsat7 - landsat9 | - | - | 2 | - | **insufficient_overlap** |

Only three of the six pairs overlap enough to be tested at all, and two of those three step by several times the entire 26-year trend signal. **Consequence:** the headline trend is fitted within a single sensor family only, over a shorter record and therefore with less statistical power. Any multi-sensor product in this project is a geography statement, never a magnitude one.

### 2. The land-cover projection is a measured negative result

The CA-Markov component reproduces the quantity of land-cover change but cannot allocate it, scoring Kappa 0.810 against a persistence Kappa of 0.844 - worse than a no-change map - with a figure of merit of 0.074, across two class schemes and both calibration intervals Dynamic World's record supports. **Consequence:** no projected land-cover map exists, and none of the 2030 or 2036 horizons may be mapped. Deliverable 2 carries a stated, quantified limitation instead of an unvalidated map. The greening counterfactual is unaffected, because it rests on the regression alone.

### 3. The multi-criteria ranking reproduces a ranking by heat alone

The five greening criteria are near-collinear over Colombo. A leave-one-out ablation gives a rank correlation of about 0.98 between the full five-criterion ranking and a ranking on land surface temperature alone, the first principal component carries about 91% of the variance, and the effective dimensionality is about 1.5. **Consequence:** this is a finding about Colombo, not a fault in the method, and the ranking is still the right product - but it is not adding what a five-criterion MCDA is normally assumed to add, and must not be presented as if it were.

### 4. The AHP weights are judgements

They were argued by the analyst from the literature and from this project's own measurements. They were **not** elicited from stakeholders, residents or the municipality. The consistency ratio tests only whether the judgements are self-consistent, never whether they are right, and a different defensible set gives a different ranking.

### 5. The CMC's area is scale-dependent

The administrative polygon measures about 47 km² because it encloses the Colombo Port outer harbour; masked to land it measures about 40 km² at 30 m and less at coarser reduction scales, against a gazetted 37.31 km². **Consequence:** any CMC area, and any per-area figure derived from one, must be quoted with its reduction scale.

### 6. Two boundary sources that do not agree

The GAUL district polygon and the uploaded COD-AB GN polygons differ by roughly 13 km². Exports clipped to the former while zones were burned from the latter left GN area with no exported data, and a fill-with-zero read made it indistinguishable from land the classifier failed to classify. It flagged 23 coastal and edge divisions as unobserved and, for three Colab runs, deleted Pettah - one of the hottest, densest, most treeless divisions in the district - from the greening deliverable. **Consequence:** the export now carries an explicit in-region band, and the guard message that had been printing the discrepancy all along is no longer advisory.

### 7. JRC Global Surface Water does not map the ocean

The cheap static water mask, whose docstring claimed it was authoritative for the ocean and the Colombo Port outer harbour, is not: the water JRC finds across the district is inland. **Consequence:** any coastal product must use the combined mask, which ORs MNDWI, the `QA_PIXEL` water-bit frequency and JRC. Anything that used the cheap mask on a coastal geometry was treating the sea as land.

### 8. Strict MODIS daytime QC biases the sample toward the core

The strict day policy retains only a few per cent of daytime observations and fails disproportionately over the dense coastal core - exactly where the heat island signal lives. **Consequence:** the strict Terra day series is never quoted alone; a relaxed-QC variant is run beside it and both are reported.

### 9. Green-space access is straight-line and an upper bound

Service areas are Euclidean, so they ignore the Kelani River, Beira Lake, the coastal railway and walled compounds, and overstate access by an unknown amount; a shorter detour-ratio variant bounds the uncertainty. Green space is counted whether public or private, so gardens, cantonments and golf courses all count toward compliance. **Consequence:** every compliance figure is an upper bound.

### 10. No official wetland boundary exists

There is no authoritative Colombo Wetland Complex boundary in any free dataset. The wetland layer is three earth-observation proxies plus the legally designated sites in the World Database on Protected Areas. **Consequence:** wetland status is a constructed indicator, not a legal determination.

### 11. Coverage flags travel; they do not delete

A division whose land-cover coverage falls below the floor is flagged in every output row, on the map as a distinct hatch, and in the metadata sidecar - but it is **not** removed from the ranking. The floor gates nothing that enters the score, and across three runs it removed the densest and hottest divisions in the district. **Consequence:** a reader may drop those divisions; the pipeline does not do it for them.

### 12. Agreement between phases is not validation

The greening ranking correlates strongly with the Phase 5 hot-spot result, but the two share inputs, so the agreement is not independent corroboration and is labelled as such in the output.

### 13. Every zonal statistic is a property of its aggregation

Coefficients and rankings computed across GN divisions describe those polygons, not pixels and not people. The DS-level re-run gives different numbers by construction, which is why both levels are always reported.

### 14. Configured datasets that were never used

The following are configured but referenced by no analysis step: `worldcover_2020`, `viirs_nightlights`, `aster_ged`, `era5_land`. The consequential one is `era5_land`: the project specification names ERA5-Land `temperature_2m` for reanalysis air-temperature validation, and **that comparison was never carried out**. **Consequence:** nothing in this project independently corroborates the satellite land surface temperatures against an air-temperature record, which sharpens the standing caveat that LST is not air temperature - the gap between them is unquantified here, not merely unquoted.

### Colour and legibility

Two palettes failed the Phase 8 colour-vision check and were replaced; the measured before-and-after numbers are recorded in `config/params.yaml` beside each one, and the full result is in `data/outputs/palette_cvd_check.csv`. Two palettes are exempt with stated reasons: the Dynamic World legend colours, which are fixed by the catalog, and the 17-category emerging-hot-spot scheme, which cannot be separated by colour even for a reader with normal colour vision. Both are given redundant encoding instead.
