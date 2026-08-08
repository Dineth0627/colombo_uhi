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
    assert m["qc_filter"]["mandatory_qa_required_value"] == 0
    assert m["qc_filter"]["lst_error_bits"] == [6, 7]
    assert m["qc_filter"]["lst_error_required_value"] == 0
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


def test_trend_settings(params: dict[str, Any]) -> None:
    t = params["trends"]
    assert t["series_basis"] == "annual_composite"
    assert t["mk_reducer"] == "kendallsCorrelation"
    assert t["sen_reducer"] == "sensSlope"
    assert t["slope_units"] == "degC_per_year"
    assert t["fdr"]["method"] == "benjamini_hochberg"
    assert 0 < t["fdr"]["alpha"] < 1


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
