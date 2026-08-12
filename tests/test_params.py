"""Guard config/params.yaml against typos and drift from CLAUDE.md constants.

These tests pin the numeric constants and dataset ids that were verified
against the GEE catalog (2026-08-08). If one fails after an edit, either the
edit is a typo or CLAUDE.md/PROGRESS.md needs a deliberate, documented update.
"""

from __future__ import annotations

from typing import Any

import pytest

from colombo_uhi import load_params, repo_root


@pytest.fixture(scope="module")
def params() -> dict[str, Any]:
    return load_params()


def test_params_lives_in_config_dir() -> None:
    assert (repo_root() / "config" / "params.yaml").is_file()


def test_required_top_level_sections(params: dict[str, Any]) -> None:
    for section in (
        "project",
        "gcp",
        "crs",
        "time",
        "aoi",
        "datasets",
        "non_ee_sources",
        "landsat_c2l2",
        "modis_lst",
        "uhi",
        "trends",
        "spatial_stats",
        "prediction",
        "greening",
        "exports",
        "caveats",
    ):
        assert section in params, f"missing top-level section: {section}"


def test_crs_and_grid(params: dict[str, Any]) -> None:
    assert params["crs"]["analysis_epsg"] == "EPSG:32644"
    assert params["crs"]["analysis_scale_m"] == 30


def test_study_period(params: dict[str, Any]) -> None:
    assert params["time"]["start_year"] == 2000
    assert params["time"]["end_year"] == 2025
    assert params["time"]["seasons"]["dry_window"]["months"] == [1, 2, 3]
    assert params["time"]["seasons"]["sw_monsoon"]["months"] == [5, 6, 7, 8, 9]
    assert params["time"]["seasons"]["ne_monsoon"]["months"] == [12, 1, 2]


def test_aoi_centre(params: dict[str, Any]) -> None:
    assert params["aoi"]["centre"] == {"lat": 6.93, "lon": 79.85}
    assert params["aoi"]["district"]["gn_divisions"] == 557
    assert params["aoi"]["district"]["ds_divisions"] == 13


def test_dataset_ids_match_claude_md(params: dict[str, Any]) -> None:
    expected = {
        "landsat5": "LANDSAT/LT05/C02/T1_L2",
        "landsat7": "LANDSAT/LE07/C02/T1_L2",
        "landsat8": "LANDSAT/LC08/C02/T1_L2",
        "landsat9": "LANDSAT/LC09/C02/T1_L2",
        "modis_terra_lst": "MODIS/061/MOD11A2",
        "modis_aqua_lst": "MODIS/061/MYD11A2",
        "dynamic_world": "GOOGLE/DYNAMICWORLD/V1",
        "worldcover_2020": "ESA/WorldCover/v100",
        "worldcover_2021": "ESA/WorldCover/v200",
        "ghsl_built": "JRC/GHSL/P2023A/GHS_BUILT_S",
        "lcz": "RUB/RUBCLIM/LCZ/global_lcz_map/latest",
        "srtm": "USGS/SRTMGL1_003",
        "worldpop": "WorldPop/GP/100m/pop",
        "viirs_nightlights": "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG",
        "surface_water": "JRC/GSW1_4/GlobalSurfaceWater",
        "aster_ged": "NASA/ASTER_GED/AG100_003",
        "era5_land": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
        "gaul_level2": "FAO/GAUL_SIMPLIFIED_500m/2015/level2",
    }
    for key, dataset_id in expected.items():
        assert params["datasets"][key]["id"] == dataset_id, key


def test_gsod_is_not_an_ee_dataset(params: dict[str, Any]) -> None:
    # NOAA/GSOD is not in the EE catalog (verified 2026-08-08); it must stay
    # out of `datasets` until the user decides on an alternative.
    assert "gsod" not in params["datasets"]
    assert params["non_ee_sources"]["gsod"]["wmo_station_id"] == "43450"


def test_landsat_st_bands(params: dict[str, Any]) -> None:
    assert params["datasets"]["landsat5"]["st_band"] == "ST_B6"
    assert params["datasets"]["landsat7"]["st_band"] == "ST_B6"
    assert params["datasets"]["landsat8"]["st_band"] == "ST_B10"
    assert params["datasets"]["landsat9"]["st_band"] == "ST_B10"


def test_landsat_c2l2_constants(params: dict[str, Any]) -> None:
    c = params["landsat_c2l2"]
    assert c["st_scale"] == 0.00341802
    assert c["st_offset"] == 149.0
    assert c["kelvin_to_celsius_offset"] == 273.15
    assert c["sr_scale"] == 0.0000275
    assert c["sr_offset"] == -0.2
    assert c["st_emis_scale"] == 0.0001
    assert c["st_qa_scale"] == 0.01
    assert c["st_valid_dn_range"] == [293, 65535]
    assert c["st_fill_dn"] == 0
    assert c["processing_level_required"] == "L2SP"


def test_qa_pixel_bits(params: dict[str, Any]) -> None:
    bits = params["landsat_c2l2"]["qa_pixel_bits"]
    assert bits["fill"] == 0
    assert bits["dilated_cloud"] == 1
    assert bits["cirrus"] == 2
    assert bits["cloud"] == 3
    assert bits["cloud_shadow"] == 4
    assert bits["snow"] == 5
    assert bits["clear"] == 6
    assert bits["water"] == 7
    assert bits["cloud_confidence"] == [8, 9]
    assert bits["cloud_shadow_confidence"] == [10, 11]
    assert bits["cirrus_confidence"] == [14, 15]


def test_landsat_standard_mask(params: dict[str, Any]) -> None:
    mask = params["landsat_c2l2"]["standard_mask"]
    assert mask["require_zero_bits"] == [0, 1, 2, 3, 4]
    assert mask["require_qa_radsat_value"] == 0


def test_modis_constants(params: dict[str, Any]) -> None:
    m = params["modis_lst"]
    assert m["lst_scale"] == 0.02
    assert m["qc_filter"]["mandatory_qa_bits"] == [0, 1]
    assert m["qc_filter"]["lst_error_bits"] == [6, 7]
    # Thresholds moved under per-overpass policies in rev 6; see test_modis.py.
    assert m["qc_filter"]["day"]["mandatory_qa_max"] == 0
    assert m["qc_filter"]["day"]["lst_error_max"] == 0
    assert params["datasets"]["modis_terra_lst"]["day_band"] == "LST_Day_1km"
    assert params["datasets"]["modis_terra_lst"]["night_band"] == "LST_Night_1km"


def test_utfvi_classes(params: dict[str, Any]) -> None:
    utfvi = params["uhi"]["utfvi"]
    breaks = utfvi["breaks"]
    labels = utfvi["labels"]
    assert breaks == [0.0, 0.005, 0.010, 0.015, 0.020]
    assert all(b2 > b1 for b1, b2 in zip(breaks, breaks[1:])), "breaks must increase"
    assert labels == ["Excellent", "Good", "Normal", "Bad", "Worse", "Worst"]
    assert len(labels) == len(breaks) + 1, "six classes from five thresholds"


def test_suhii_needs_two_rural_definitions(params: dict[str, Any]) -> None:
    rural = params["uhi"]["suhii"]["rural_definitions"]
    assert len(rural) >= 2
    assert "buffer_ring" in rural and "lcz_based" in rural


def test_phase1_aoi_sections(params: dict[str, Any]) -> None:
    aoi = params["aoi"]
    # Sub-district boundary assets: nullable slots must exist (GAUL stops at
    # district level for Sri Lanka — decision recorded 2026-08-08).
    assert "ds_divisions" in aoi["assets"]
    assert "gn_divisions" in aoi["assets"]
    # CMC = union of the 55 GN divisions the CMC GIS Unit lists (authoritative);
    # the DS pair is kept only as a sensitivity variant (it measures 46.87 km2).
    assert aoi["cmc"]["definition"] == "gn_union"
    assert len(aoi["cmc"]["gn_division_names"]) == aoi["cmc"]["expected_gn_count"] == 55
    assert aoi["cmc"]["ds_division_names"] == ["Colombo", "Thimbirigasyaya"]
    # null = auto-resolve from the matching aoi.assets.*_name_property_candidates.
    for key in ("ds_name_property", "gn_name_property"):
        assert aoi["cmc"][key] is None or isinstance(aoi["cmc"][key], str)
    assert aoi["assets"]["district_value"] == "Colombo"
    # Three distinct CMC numbers (Colab run 5): gazetted 37.31, the raw polygon
    # 47.07 (includes the Colombo Port harbour), and polygon-minus-water 40.18 at
    # the 30 m grid. Ordering must hold: gazetted < land < administrative.
    expected = aoi["expected_areas_km2"]
    assert expected["cmc"] == 37
    assert expected["cmc_land_at_30m"] == 40
    assert expected["cmc_administrative"] == 47
    assert expected["cmc"] < expected["cmc_land_at_30m"] < expected["cmc_administrative"]
    assert expected["district"] == 699
    assert expected["western_province"] == 3684
    # GHSL urban-extent thresholding (asset-free buffer-ring base).
    ue = aoi["urban_extent"]
    assert ue["ghsl_epoch"] == 2020
    assert 0 < ue["built_surface_threshold_m2"] <= 10000  # m2 per 100 m cell
    assert ue["vectorize_scale_m"] == 100


def test_phase1_water_mask_section(params: dict[str, Any]) -> None:
    wm = params["aoi"]["water_mask"]
    assert -1.0 <= wm["mndwi_threshold"] <= 1.0
    assert 0.0 <= wm["qa_water_freq_threshold"] <= 1.0
    assert 0 <= wm["jrc_occurrence_threshold_pct"] <= 100
    assert wm["shoreline_buffer_m"] >= 0
    comp = wm["composite"]
    assert comp["landsat_sources"] == ["landsat8", "landsat9"]
    assert comp["start_year"] <= comp["end_year"]
    assert comp["months"] == params["time"]["seasons"]["dry_window"]["months"]
    assert comp["reducer"] in ("median", "mean")


def test_phase1_rural_reference_settings(params: dict[str, Any]) -> None:
    suhii = params["uhi"]["suhii"]
    assert suhii["buffer_ring"]["base"] in ("urban_extent", "cmc")
    assert suhii["lcz_based"]["urban_classes"] == list(range(1, 11))
    # Rural = LCZ A-G (11-17), user decision 2026-08-08; water mask removes G.
    assert suhii["lcz_based"]["rural_classes"] == list(range(11, 18))
    # Both LCZ masks are clipped to this geometry (user decision 2026-08-08).
    assert suhii["lcz_based"]["scope"] in ("district", "cmc", "analysis_region")


def test_trend_settings(params: dict[str, Any]) -> None:
    t = params["trends"]
    assert t["series_basis"] == "annual_composite"
    assert t["mk_reducer"] == "kendallsCorrelation"
    assert t["sen_reducer"] == "sensSlope"
    assert t["slope_units"] == "degC_per_year"
    assert t["fdr"]["method"] == "benjamini_hochberg"
    assert 0 < t["fdr"]["alpha"] < 1


# --- Phase 4 additions -------------------------------------------------------
def test_trend_bands_cover_the_export_order(params: dict[str, Any]) -> None:
    # Band identity in a GeoTIFF is POSITIONAL. If export_band_order names a band
    # that trends.bands does not define, the exported file cannot be read back.
    bands = set(params["trends"]["bands"].values())
    assert set(params["trends"]["export_band_order"]) <= bands


def test_trend_export_order_has_no_duplicates(params: dict[str, Any]) -> None:
    order = params["trends"]["export_band_order"]
    assert len(order) == len(set(order))


def test_trend_reducer_outputs_are_recorded(params: dict[str, Any]) -> None:
    # Measured in Colab run 11 and pinned here so the product path can select
    # them directly. Reading them back from Earth Engine cost a getInfo on a
    # reduce over all 26 composites, which the interactive memory ceiling
    # cannot support.
    outputs = params["trends"]["reducer_outputs"]
    assert outputs["sen"] == ["slope", "offset"]
    assert outputs["kendall"] == ["tau", "p-value"]


def test_trend_carries_both_p_value_bands(params: dict[str, Any]) -> None:
    # mk_p_two_sided is what FDR consumes; mk_p_ee is the reducer's own p-value,
    # exported beside it so their ratio settles empirically whether the reducer
    # returns a one- or two-sided p (a ratio near 2 means one-sided).
    order = params["trends"]["export_band_order"]
    assert params["trends"]["bands"]["mk_p_two_sided"] in order
    assert params["trends"]["bands"]["mk_p_ee"] in order


def test_trend_always_exports_the_valid_year_count(params: dict[str, Any]) -> None:
    # CLAUDE.md caveat 2: no trend product without its observation count.
    assert params["trends"]["bands"]["n_years"] in params["trends"]["export_band_order"]


def test_trend_pixel_sources_exist_in_the_suhii_sources(
    params: dict[str, Any],
) -> None:
    configured = {source["key"] for source in params["uhi"]["suhii"]["sources"]}
    assert set(params["trends"]["pixel_sources"]) <= configured


def test_trend_pixel_sources_exclude_modis_daytime(params: dict[str, Any]) -> None:
    # Terra's orbital drift after ~2020 moves the overpass time and contaminates
    # end-of-series DAYTIME trends (Phase 2 finding). Night is unaffected, and is
    # the part Landsat structurally cannot provide.
    assert "terra_day" not in params["trends"]["pixel_sources"]
    assert "aqua_day" not in params["trends"]["pixel_sources"]


def test_trend_decades_are_contiguous_and_unequal_by_design(
    params: dict[str, Any],
) -> None:
    windows = sorted(
        (int(v[0]), int(v[1])) for v in params["trends"]["decades"].values()
    )
    assert windows[0][0] == params["time"]["start_year"]
    assert windows[-1][1] == params["time"]["end_year"]
    for earlier, later in zip(windows, windows[1:]):
        assert later[0] == earlier[1] + 1
    # 11 / 10 / 5. Pinned so an "equalise the decades" tidy-up is a visible edit.
    assert [end - start + 1 for start, end in windows] == [11, 10, 5]


def test_trend_decades_are_not_the_utfvi_epochs(params: dict[str, Any]) -> None:
    trend = {(int(v[0]), int(v[1])) for v in params["trends"]["decades"].values()}
    utfvi = {(int(v[0]), int(v[1])) for v in params["uhi"]["utfvi"]["epochs"].values()}
    assert trend != utfvi


def test_trend_slope_palette_is_diverging_and_zero_centred(
    params: dict[str, Any],
) -> None:
    vis = params["trends"]["slope_vis"]
    # An odd-length palette puts its middle colour at the midpoint, and a
    # symmetric range puts that midpoint at zero - without both, a reader cannot
    # tell warming from cooling.
    assert len(vis["palette"]) % 2 == 1
    assert vis["min"] == -vis["max"]


def test_trend_fdr_sensitivity_includes_the_headline_method(
    params: dict[str, Any],
) -> None:
    fdr = params["trends"]["fdr"]
    assert fdr["method"] in fdr["sensitivity_methods"]
    assert "benjamini_yekutieli" in fdr["sensitivity_methods"]


def test_trend_min_years_supports_the_normal_approximation(
    params: dict[str, Any],
) -> None:
    assert params["trends"]["min_years"] >= 8


def test_trend_sets_its_own_valid_obs_floor(params: dict[str, Any]) -> None:
    # composites.min_valid_obs is deliberately null (flag, never mask); the
    # params comment says Phase 4 sets its own floor, and this is it.
    assert params["composites"]["min_valid_obs"] is None
    assert params["trends"]["min_valid_obs"] >= 1


def test_composites_expose_the_series_basis_marker(params: dict[str, Any]) -> None:
    # The property annual_composites stamps and require_annual_series checks.
    assert params["composites"]["series_basis_property"]
    assert params["composites"]["window_months_property"]


def test_landcover_schemes_have_non_empty_class_maps(params: dict[str, Any]) -> None:
    for scheme in ("worldcover", "lcz", "dynamic_world"):
        classes = params["landcover"][scheme]["classes"]
        assert classes
        assert all(isinstance(code, int) for code in classes)
        assert params["landcover"][scheme]["band"]


def test_lcz_class_map_covers_every_configured_suhii_class(
    params: dict[str, Any],
) -> None:
    labelled = set(params["landcover"]["lcz"]["classes"])
    lcz = params["uhi"]["suhii"]["lcz_based"]
    assert set(lcz["urban_classes"]) <= labelled
    assert set(lcz["rural_classes"]) <= labelled


def test_export_limits_are_sane(params: dict[str, Any]) -> None:
    exports_cfg = params["exports"]
    # Earth Engine caps task descriptions at 100 characters.
    assert 0 < exports_cfg["max_name_chars"] <= 100
    assert exports_cfg["poll_seconds"] > 0
    assert exports_cfg["timeout_seconds"] > exports_cfg["poll_seconds"]


def test_trend_caveats_present(params: dict[str, Any]) -> None:
    for key in ("trend_not_causal", "fdr_dependence"):
        assert params["caveats"].get(key), f"missing caveat text: {key}"


def test_prediction_ships_validation_metrics(params: dict[str, Any]) -> None:
    metrics = params["prediction"]["validation_metrics"]
    assert set(metrics) == {"rmse", "r2", "kappa"}


def test_export_naming_template(params: dict[str, Any]) -> None:
    template = params["exports"]["name_template"]
    assert template == "{product}_{aoi}_{startyear}_{endyear}_{res}m"
    rendered = template.format(
        product="lst_annual", aoi="cmc", startyear=2000, endyear=2025, res=30
    )
    assert rendered == "lst_annual_cmc_2000_2025_30m"


def test_caveats_present(params: dict[str, Any]) -> None:
    caveats = params["caveats"]
    for key in (
        "lst_not_air_temp",
        "valid_obs_required",
        "scenario_not_forecast",
        "single_overpass",
        "sensitivity_reporting",
    ):
        assert caveats.get(key), f"missing caveat text: {key}"


# =============================================================================
# Phase 2 additions (LST pipeline) — verified against the GEE catalog 2026-08-08
# =============================================================================
def test_phase2_sections_present(params: dict[str, Any]) -> None:
    for section in ("indices", "composites"):
        assert section in params, f"missing top-level section: {section}"


def test_landsat_phase2_band_names(params: dict[str, Any]) -> None:
    c = params["landsat_c2l2"]
    # ST_QA exists on ALL FOUR collections, L5 included (catalog-verified).
    assert c["st_qa_band"] == "ST_QA"
    assert c["lst_band_name"] == "LST_C"
    assert c["st_qa_band_name"] == "ST_QA_K"
    assert c["harmonised_sr_bands"] == [
        "blue",
        "green",
        "red",
        "nir",
        "swir1",
        "swir2",
    ]


def test_landsat_sensor_keys_cover_all_four(params: dict[str, Any]) -> None:
    keys = params["landsat_c2l2"]["sensor_keys"]
    assert keys == ["landsat5", "landsat7", "landsat8", "landsat9"]
    for key in keys:
        dataset = params["datasets"][key]
        assert dataset["st_band"] in ("ST_B6", "ST_B10")
        assert set(dataset["sr_bands"]) >= set(
            params["landsat_c2l2"]["harmonised_sr_bands"]
        )


def test_processing_level_filter_fails_open(params: dict[str, Any]) -> None:
    # Implemented as neq(L2SR), not eq(L2SP): a missing property must let the
    # scene through rather than silently emptying the collection.
    c = params["landsat_c2l2"]
    assert c["processing_level_exclude"] == "L2SR"
    assert c["processing_level_required"] == "L2SP"
    assert isinstance(c["processing_level_filter_enabled"], bool)


def test_landsat_phase2_defaults_are_the_user_decisions(
    params: dict[str, Any],
) -> None:
    c = params["landsat_c2l2"]
    # SLC-off included: excluding it would empty 2012-05..2013-03 entirely.
    assert c["include_l7_slc_off"] is True
    # ST_QA filter off by default; the ST_QA_K band is still always emitted.
    assert c["st_qa_max_kelvin"] is None
    low, high = c["lst_plausible_range_c"]
    assert low < high


def test_landsat7_slc_off_date(params: dict[str, Any]) -> None:
    # filterDate's end is exclusive, so this is the last SLC-ON day and the
    # code must advance one day past it.
    assert params["datasets"]["landsat7"]["slc_off_after"] == "2003-05-31"


def test_season_partition_is_total_and_disjoint(params: dict[str, Any]) -> None:
    seen: dict[int, int] = {month: 0 for month in range(1, 13)}
    for key in params["time"]["season_partition"]:
        for month in params["time"]["seasons"][key]["months"]:
            seen[month] += 1
    assert all(count == 1 for count in seen.values()), seen


def test_inter_monsoon_fills_the_gap_claude_md_leaves(params: dict[str, Any]) -> None:
    assert params["time"]["seasons"]["inter_monsoon"]["months"] == [3, 4, 10, 11]


def test_composites_defaults(params: dict[str, Any]) -> None:
    c = params["composites"]
    assert c["obs_count_band"] == "obs_count"
    assert c["annual_reducer"] in ("median", "mean")
    assert isinstance(c["percentile"], int) and 0 <= c["percentile"] <= 100
    # User decision: flag with obs_count, never mask pixels away.
    assert c["min_valid_obs"] is None
    assert c["reduce_max_pixels"] > 0
    assert c["tile_scale"] >= 1


def test_modis_phase2_additions(params: dict[str, Any]) -> None:
    m = params["modis_lst"]
    assert m["products"] == {"terra": "modis_terra_lst", "aqua": "modis_aqua_lst"}
    assert m["clear_sky_day_band"] == "Clear_sky_days"
    assert m["clear_sky_night_band"] == "Clear_sky_nights"
    # Bitmask, NOT a count — 8 bits, one per day of the 8-day window.
    assert m["clear_sky_is_bitmask"] is True
    assert m["clear_sky_bits"] == 8
    assert m["valid_dn_range"] == [7500, 65535]
    assert m["reduction_scale_m"] == 1000


def test_modis_aqua_launch_date_is_recorded(params: dict[str, Any]) -> None:
    # No Aqua data for the first 2.5 years of the study period.
    assert params["datasets"]["modis_aqua_lst"]["availability"][0] == "2002-07-04"
    assert params["datasets"]["modis_terra_lst"]["availability"][0] == "2000-02-18"


def test_albedo_uses_one_coefficient_set(params: dict[str, Any]) -> None:
    # Decision reversal recorded in PROGRESS.md: a single set across all four
    # sensors preserves continuity at the 2013 OLI transition; swapping in a
    # different published fit there would CREATE the step change.
    albedo = params["indices"]["albedo"]
    assert albedo["active_set"] == "liang2001"
    assert "liang2001" in albedo["sets"]
    for entry in albedo["sets"].values():
        assert entry["weights"], "an albedo set needs weights"
        assert "source" in entry, "every albedo set must cite its source"


# =============================================================================
# Phase 3 additions (UHI metrics)
# =============================================================================
def test_phase3_sections_present(params: dict[str, Any]) -> None:
    uhi = params["uhi"]
    for section in ("zscore", "drivers"):
        assert section in uhi, f"missing uhi section: {section}"
    for key in ("sources", "batch_years"):
        assert key in uhi["suhii"], f"missing uhi.suhii key: {key}"


def test_utfvi_units_are_celsius(params: dict[str, Any]) -> None:
    # LOAD-BEARING. UTFVI is a ratio, so the class breaks depend on the
    # temperature scale: on Kelvin the 0.005 break would mean ~1.5 K instead of
    # ~0.15 degC. uhi_metrics.utfvi() refuses anything else.
    assert params["uhi"]["utfvi"]["units"] == "celsius"


def test_utfvi_reference_is_the_per_year_mean(params: dict[str, Any]) -> None:
    # User decision 2026-08-09. The consequence must travel with every UTFVI
    # output: a uniformly warming city shows NO class change, so epoch-to-epoch
    # drift is redistribution of heat, never evidence of warming.
    assert params["uhi"]["utfvi"]["reference"] == "per_year_aoi_mean"


def test_utfvi_palette_has_one_colour_per_class(params: dict[str, Any]) -> None:
    utfvi = params["uhi"]["utfvi"]
    assert len(utfvi["palette"]) == len(utfvi["labels"]) == 6
    for colour in utfvi["palette"]:
        assert len(colour) == 6, f"{colour} is not a bare 6-digit hex colour"
        int(colour, 16)  # raises if it is not hex


def test_utfvi_epochs_tile_the_study_period(params: dict[str, Any]) -> None:
    # No gap and no overlap, or a year silently belongs to two epoch maps (or to
    # none) and the three-epoch comparison stops being a partition of the record.
    epochs = params["uhi"]["utfvi"]["epochs"]
    spans = sorted((int(s), int(e)) for s, e in epochs.values())
    assert spans[0][0] == params["time"]["start_year"]
    assert spans[-1][1] == params["time"]["end_year"]
    for (_, earlier_end), (later_start, _) in zip(spans, spans[1:]):
        assert later_start == earlier_end + 1, "epochs must be contiguous"


def test_zscore_defaults(params: dict[str, Any]) -> None:
    zscore = params["uhi"]["zscore"]
    assert zscore["default_sigma"] == 1.0
    assert 1.0 in zscore["sigma_options"] and 2.0 in zscore["sigma_options"]
    assert zscore["ddof"] in (0, 1)


def test_suhii_sources_are_well_formed(params: dict[str, Any]) -> None:
    sources = params["uhi"]["suhii"]["sources"]
    keys = [entry["key"] for entry in sources]
    assert len(keys) == len(set(keys)), "SUHII source keys must be unique"

    products = params["modis_lst"]["products"]
    for entry in sources:
        assert entry["kind"] in ("landsat", "modis")
        assert entry["reducer"] in ("median", "mean")
        assert entry["scale_m"] > 0
        if entry["kind"] == "modis":
            assert entry["product"] in products
            assert entry["daynight"] in ("day", "night")
            # Reduce MODIS at its own 1 km grid, never at the 30 m Landsat one.
            assert entry["scale_m"] == params["modis_lst"]["reduction_scale_m"]


def test_suhii_sources_cover_day_and_night(params: dict[str, Any]) -> None:
    # CLAUDE.md caveat 4: Landsat is a single ~10:30 overpass, so night-time UHI
    # is obtainable ONLY from MODIS. Losing the night entries would silently
    # reduce the study to a daytime one.
    sources = params["uhi"]["suhii"]["sources"]
    modis = [entry for entry in sources if entry["kind"] == "modis"]
    assert any(entry["daynight"] == "night" for entry in modis)
    assert any(entry["daynight"] == "day" for entry in modis)
    assert any(entry["kind"] == "landsat" for entry in sources)


def test_suhii_sources_include_the_relaxed_qc_sensitivity(
    params: dict[str, Any],
) -> None:
    # REQUIRED BY PHASE 2 SIGN-OFF, not optional. Strict day QC keeps only 3.7%
    # of daytime observations and fails hardest over the dense coastal core
    # (CMC retains 13-23 of 40 pixels, District 92-95%), so the strict daytime
    # series must never stand alone.
    sources = {entry["key"]: entry for entry in params["uhi"]["suhii"]["sources"]}
    relaxed = sources["terra_day_relaxed"]
    strict = params["modis_lst"]["qc_filter"]["day"]
    assert relaxed["daynight"] == "day"
    assert relaxed["mandatory_qa_max"] > strict["mandatory_qa_max"]
    assert relaxed["lst_error_max"] >= strict["lst_error_max"]


def test_landsat_suhii_source_uses_the_dry_window(params: dict[str, Any]) -> None:
    sources = {entry["key"]: entry for entry in params["uhi"]["suhii"]["sources"]}
    assert sources["landsat_dry"]["months_key"] == "dry_window"
    # Reducer discipline from Phase 2: Landsat median, MODIS mean. Mixing them
    # makes part of any cross-sensor difference a reducer artefact.
    assert sources["landsat_dry"]["reducer"] == params["composites"]["annual_reducer"]
    for entry in sources.values():
        if entry["kind"] == "modis":
            assert entry["reducer"] == params["composites"]["modis_reducer"]


def test_driver_settings(params: dict[str, Any]) -> None:
    drivers = params["uhi"]["drivers"]
    assert drivers["response"] == params["landsat_c2l2"]["lst_band_name"]
    assert drivers["predictors"] == ["NDVI", "NDBI", "MNDWI", "built_fraction"]
    assert drivers["sample_pixels"] > 0
    assert drivers["min_sample_rows"] > 0
    # Deliberately coarser than the analysis grid: adjacent 30 m LST pixels are
    # near-duplicates, so sampling at 30 m buys autocorrelation, not information.
    assert drivers["sample_scale_m"] > params["crs"]["analysis_scale_m"]


def test_driver_predictors_are_computable(params: dict[str, Any]) -> None:
    # Every predictor must be either a band indices.py can produce or the GHSL
    # built fraction; a typo here would fail only after a Colab round trip.
    from colombo_uhi.indices import INDEX_BAND_NAMES

    available = set(INDEX_BAND_NAMES.values()) | {"built_fraction"}
    for predictor in params["uhi"]["drivers"]["predictors"]:
        assert predictor in available, f"no way to compute predictor {predictor}"


def test_ghsl_cell_area_matches_its_resolution(params: dict[str, Any]) -> None:
    # built_surface is m^2 of built area per cell, so the built FRACTION divides
    # by the cell area. 100 m x 100 m = 10 000 m^2.
    ghsl = params["datasets"]["ghsl_built"]
    assert ghsl["cell_area_m2"] == ghsl["scale_m"] ** 2 == 10000
    assert ghsl["epoch_interval_years"] == 5
    # The Phase 1 urban-extent threshold is expressed in the same units.
    assert params["aoi"]["urban_extent"]["built_surface_threshold_m2"] < ghsl["cell_area_m2"]


# --- Phase 5: spatial statistics ---------------------------------------------
def test_phase5_sections_present(params: dict[str, Any]) -> None:
    spatial = params["spatial_stats"]
    for key in (
        "levels", "epochs_source", "response_band", "permutations", "random_seed",
        "geometry", "weights", "lisa", "gi_star", "ehsa", "regression",
        "covariates", "landscape", "palettes",
    ):
        assert key in spatial, f"spatial_stats.{key} missing"


def test_spatial_levels_are_the_two_aggregation_units(params: dict[str, Any]) -> None:
    # The MAUP pair CLAUDE.md requires. Both must be levels aoi.py can load.
    assert params["spatial_stats"]["levels"] == ["gn", "ds"]
    assert params["aoi"]["expected_counts"]["gn_divisions"] == 557
    assert params["aoi"]["expected_counts"]["ds_divisions"] == 13


def test_spatial_epoch_sources_exist_in_the_suhii_sources(
    params: dict[str, Any],
) -> None:
    keys = {source["key"] for source in params["uhi"]["suhii"]["sources"]}
    spatial = params["spatial_stats"]
    assert spatial["epochs_source"] in keys
    assert spatial["epoch_sensitivity_source"] in keys
    # The sensitivity source must be a DIFFERENT series, or the check is vacuous.
    assert spatial["epochs_source"] != spatial["epoch_sensitivity_source"]


def test_spatial_response_band_is_the_celsius_lst_band(params: dict[str, Any]) -> None:
    # Gi* is undefined for negative values, so the response must be the degC
    # band and never a z-score or an anomaly.
    assert params["spatial_stats"]["response_band"] == params["landsat_c2l2"]["lst_band_name"]
    assert params["spatial_stats"]["regression"]["response"] == params["spatial_stats"]["response_band"]


def test_permutation_count_can_reach_the_finest_confidence_break(
    params: dict[str, Any],
) -> None:
    # The smallest obtainable pseudo p is 1/(permutations + 1). If that exceeds
    # the tightest confidence break, the 99% class is unreachable no matter how
    # extreme a zone is - and nobody would notice.
    spatial = params["spatial_stats"]
    finest = min(spatial["gi_star"]["confidence_breaks"])
    assert 1.0 / (spatial["permutations"] + 1) <= finest


def test_spatial_seed_matches_the_other_reproducibility_seeds(
    params: dict[str, Any],
) -> None:
    assert params["spatial_stats"]["random_seed"] == params["uhi"]["drivers"]["sample_seed"]
    assert params["spatial_stats"]["random_seed"] == params["prediction"]["rf"]["random_seed"]


def test_confidence_breaks_are_strictly_increasing(params: dict[str, Any]) -> None:
    breaks = params["spatial_stats"]["gi_star"]["confidence_breaks"]
    assert all(low < high for low, high in zip(breaks, breaks[1:]))


def test_moran_and_gi_star_use_different_weight_transforms(
    params: dict[str, Any],
) -> None:
    # Row-standardising Gi* forces every neighbourhood sum to 1 and collapses
    # the variance term that lets a large neighbourhood outweigh a small one.
    spatial = params["spatial_stats"]
    assert spatial["weights"]["transform"] == "r"
    assert spatial["gi_star"]["weights_binary"] is True


def test_island_policy_is_one_the_code_implements(params: dict[str, Any]) -> None:
    from colombo_uhi.spatial_stats import ISLAND_POLICIES, WEIGHTS_SCHEMES

    weights = params["spatial_stats"]["weights"]
    assert weights["island_policy"] in ISLAND_POLICIES
    assert weights["scheme"] in WEIGHTS_SCHEMES


def test_geometry_export_leads_with_the_pcode_at_both_levels(
    params: dict[str, Any],
) -> None:
    # The pcode is the ONLY legitimate join key: GN names repeat across
    # Dehiwala, Moratuwa and Kolonnawa.
    properties = params["spatial_stats"]["geometry"]["export_properties"]
    assert properties["gn"][0] == "adm4_pcode"
    assert properties["ds"][0] == "adm3_pcode"


def test_ehsa_thresholds_are_coherent(params: dict[str, Any]) -> None:
    ehsa = params["spatial_stats"]["ehsa"]
    assert ehsa["hot_z"] > 0 > ehsa["cold_z"]
    assert 0.0 < ehsa["persistent_share"] <= 1.0
    assert ehsa["sporadic_max_share"] <= ehsa["persistent_share"]
    assert ehsa["consecutive_min_run"] > ehsa["new_tail_bins"] >= 1
    assert ehsa["bin_years"] >= 1


def test_ehsa_historical_category_is_arithmetically_reachable(
    params: dict[str, Any],
) -> None:
    # "Historical" needs >= persistent_share of bins hot AND the last
    # historical_recent_bins quiet. With share 0.90 and a 10-bin series only one
    # bin may be quiet, so demanding three quiet bins makes the category
    # impossible to reach - a rule that can never fire is a silent bug.
    ehsa = params["spatial_stats"]["ehsa"]
    shortest_series = 10
    max_quiet = shortest_series * (1.0 - ehsa["persistent_share"])
    assert ehsa["historical_recent_bins"] <= max_quiet + 1e-9


def test_ehsa_sources_are_single_sensor_series(params: dict[str, Any]) -> None:
    # Phase 4 established that a trend fitted across a Landsat changeover
    # measures the sensor step. EHSA runs Mann-Kendall over time, so its sources
    # must not be the pooled series.
    sources = {source["key"]: source for source in params["uhi"]["suhii"]["sources"]}
    for key in params["spatial_stats"]["ehsa"]["sources"]:
        assert key in sources, f"ehsa source {key} is not a configured LST source"
        source = sources[key]
        if source["kind"] == "landsat":
            assert "sensors" in source, (
                f"ehsa source {key} pools Landsat sensors; Phase 4 measured "
                "offsets of up to 2.48 degC across changeovers"
            )


def test_ehsa_requires_the_final_bin_by_default(params: dict[str, Any]) -> None:
    # [DECISION 2026-08-12, after Colab run 1] Without it "sporadic" fires for
    # any zone ever significant on one side, regardless of what it is doing now,
    # and it absorbed 329 of 557 GN divisions on the 26-bin MODIS series. An
    # EMERGING hot-spot analysis is about the end of the record.
    assert params["spatial_stats"]["ehsa"]["require_final_bin"] is True


def test_ehsa_power_check_is_on(params: dict[str, Any]) -> None:
    # Required by the Phase 4 sign-off: over a short panel "no pattern" must be
    # separable from "the series could not have resolved it".
    assert params["spatial_stats"]["ehsa"]["power_check"] is True


def test_regression_predictors_are_the_six_claude_md_drivers(
    params: dict[str, Any],
) -> None:
    predictors = params["spatial_stats"]["regression"]["predictors"]
    assert set(predictors) == {
        "NDVI", "NDBI", "built_fraction", "pop_density", "elevation_m",
        "dist_coast_km",
    }
    assert params["spatial_stats"]["regression"]["response"] not in predictors


def test_ds_level_fails_the_estimability_gate_by_design(
    params: dict[str, Any],
) -> None:
    # This is the mechanism behind the Phase 5 decision to record GWR/MGWR as
    # not estimable at DS rather than fitting them to 13 points.
    regression = params["spatial_stats"]["regression"]
    required = regression["min_obs_per_predictor"] * len(regression["predictors"])
    assert params["aoi"]["expected_counts"]["ds_divisions"] < required
    assert params["aoi"]["expected_counts"]["gn_divisions"] >= required


def test_gwr_corrects_its_local_multiple_testing(params: dict[str, Any]) -> None:
    # A GWR fits one model per zone; without the adjustment the local
    # significance map is inflated exactly like an uncorrected p-raster.
    assert params["spatial_stats"]["regression"]["gwr"]["adjust_alpha"] is True


def test_population_year_is_inside_worldpop_coverage(params: dict[str, Any]) -> None:
    # WorldPop ends in 2020. A 2025 population layer does not exist, and
    # silently substituting one would mislabel the column.
    year = params["spatial_stats"]["covariates"]["population"]["year"]
    first, last = params["datasets"]["worldpop"]["availability"]
    assert int(str(first)[:4]) <= year <= int(str(last)[:4])


def test_distance_to_coast_floor_excludes_the_inland_lakes(
    params: dict[str, Any],
) -> None:
    # Beira Lake (~0.65 km2) and Diyawanna (~0.4 km2) must fall BELOW the
    # connected-component floor, or "distance to coast" becomes "distance to the
    # nearest water of any kind" and every division around Beira reads coastal.
    coast = params["spatial_stats"]["covariates"]["dist_coast"]
    cell_area_km2 = (coast["scale_m"] / 1000.0) ** 2
    floor_km2 = coast["min_ocean_pixels"] * cell_area_km2
    assert floor_km2 > 4.0, "the floor must exceed Bolgoda Lake's ~3.7 km2"


def test_landscape_green_classes_exist_in_their_legends(
    params: dict[str, Any],
) -> None:
    # A class code outside the legend matches no pixels, so every metric would
    # come back zero with no error anywhere.
    landscape = params["spatial_stats"]["landscape"]
    for scheme, codes in landscape["green_classes"].items():
        legend = params["landcover"][scheme]["classes"]
        assert codes, f"green_classes.{scheme} is empty"
        for code in codes:
            assert code in legend, f"{scheme} has no class {code}"


def test_landscape_green_classes_exclude_cropland(params: dict[str, Any]) -> None:
    # Agricultural land is not urban green space for a greening-priority
    # purpose, and including it would inflate every fragmentation metric.
    green = params["spatial_stats"]["landscape"]["green_classes"]
    assert 40 not in green["worldcover"]  # WorldCover cropland
    assert 4 not in green["dynamic_world"]  # Dynamic World crops


def test_landscape_dates_are_within_dynamic_world_coverage(
    params: dict[str, Any],
) -> None:
    landscape = params["spatial_stats"]["landscape"]
    first = int(str(params["datasets"]["dynamic_world"]["availability"][0])[:4])
    for year in landscape["dynamic_world_years"]:
        # Strictly AFTER the launch year: Dynamic World opens 2015-06-27, so
        # 2015 is a half year and not comparable with a full one.
        assert year > first
    assert len(set(landscape["dynamic_world_years"])) >= 2, (
        "two dates are needed for a fragmentation CHANGE result"
    )


def test_landscape_connectivity_is_a_real_rule(params: dict[str, Any]) -> None:
    assert params["spatial_stats"]["landscape"]["connectivity"] in (4, 8)
    assert params["spatial_stats"]["landscape"]["raster_scale_m"] > 0


def test_spatial_palettes_cover_every_class_the_code_can_emit(
    params: dict[str, Any],
) -> None:
    from colombo_uhi.spatial_stats import EHSA_CATEGORIES, LISA_QUADRANTS, NOT_SIGNIFICANT

    palettes = params["spatial_stats"]["palettes"]
    for label in (*LISA_QUADRANTS.values(), NOT_SIGNIFICANT):
        assert label in palettes["lisa"], f"no LISA colour for {label}"
    for category in EHSA_CATEGORIES:
        assert category in palettes["ehsa"], f"no EHSA colour for {category}"
    for side in ("hot", "cold"):
        for level in (99, 95, 90):
            assert f"{side}_{level}" in palettes["gi_star"]
    assert NOT_SIGNIFICANT in palettes["gi_star"]


def test_gwr_palette_is_diverging_and_zero_centred(params: dict[str, Any]) -> None:
    # Local coefficients cross zero, so the ramp must have an odd number of
    # stops with a neutral middle; a sequential ramp would hide the sign change.
    palette = params["spatial_stats"]["palettes"]["gwr_diverging"]
    assert len(palette) % 2 == 1
    assert len(palette) >= 5


def test_phase5_caveats_present(params: dict[str, Any]) -> None:
    for key in ("within_epoch_only", "zonal_not_pixel"):
        assert key in params["caveats"], f"caveats.{key} missing"
        assert params["caveats"][key].strip()
