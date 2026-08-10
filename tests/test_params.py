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
