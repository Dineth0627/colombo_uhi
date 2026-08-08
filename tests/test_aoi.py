"""Unit tests for the pure-Python parts of aoi.py and landsat.py (Phase 1).

Earth Engine-dependent functions are exercised only in Colab (notebook 01);
these tests cover validation logic, the QA bitmask helper, and the rule that
both modules import cleanly without ``earthengine-api`` installed.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from colombo_uhi import aoi, landsat, load_params


@pytest.fixture(scope="module")
def params() -> dict[str, Any]:
    return load_params()


@pytest.fixture()
def params_copy(params: dict[str, Any]) -> dict[str, Any]:
    """Deep copy for tests that mutate the config."""
    return copy.deepcopy(params)


# --- landsat.bits_to_mask -----------------------------------------------------
def test_bits_to_mask_standard_cloud_mask() -> None:
    # QA_PIXEL bits 0-4 (Fill, Dilated Cloud, Cirrus, Cloud, Shadow) -> 0b11111.
    assert landsat.bits_to_mask([0, 1, 2, 3, 4]) == 31


def test_bits_to_mask_single_bits() -> None:
    assert landsat.bits_to_mask([0]) == 1
    assert landsat.bits_to_mask([7]) == 128  # QA_PIXEL water bit
    assert landsat.bits_to_mask([15]) == 32768


def test_bits_to_mask_empty_and_duplicates() -> None:
    assert landsat.bits_to_mask([]) == 0
    assert landsat.bits_to_mask([3, 3, 3]) == 8


def test_bits_to_mask_rejects_negative() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        landsat.bits_to_mask([2, -1])


def test_bits_to_mask_matches_params_standard_mask(params: dict[str, Any]) -> None:
    bits = params["landsat_c2l2"]["standard_mask"]["require_zero_bits"]
    assert landsat.bits_to_mask(bits) == 31


# --- aoi.validate_buffer_ring_km ------------------------------------------------
def test_buffer_ring_valid_pair() -> None:
    assert aoi.validate_buffer_ring_km(15, 25) == (15.0, 25.0)


def test_buffer_ring_params_default_is_valid(params: dict[str, Any]) -> None:
    cfg = params["uhi"]["suhii"]["buffer_ring"]
    aoi.validate_buffer_ring_km(cfg["inner_km"], cfg["outer_km"])


@pytest.mark.parametrize(
    ("inner", "outer"),
    [(25, 15), (15, 15), (0, 25), (-5, 25), (15, 0), (15, -1)],
)
def test_buffer_ring_rejects_bad_pairs(inner: float, outer: float) -> None:
    with pytest.raises(ValueError):
        aoi.validate_buffer_ring_km(inner, outer)


# --- aoi.resolve_rural_method ---------------------------------------------------
def test_resolve_rural_method_accepts_configured(params: dict[str, Any]) -> None:
    valid = params["uhi"]["suhii"]["rural_definitions"]
    for name in ("buffer_ring", "lcz_based"):
        assert aoi.resolve_rural_method(name, valid) == name


def test_resolve_rural_method_rejects_unknown(params: dict[str, Any]) -> None:
    valid = params["uhi"]["suhii"]["rural_definitions"]
    with pytest.raises(ValueError) as excinfo:
        aoi.resolve_rural_method("nearest_village", valid)
    # The error must name the valid options (usability requirement).
    assert "buffer_ring" in str(excinfo.value)
    assert "lcz_based" in str(excinfo.value)


# --- aoi.validate_lcz_classes -----------------------------------------------------
def test_lcz_defaults_from_params_are_valid(params: dict[str, Any]) -> None:
    lcz = params["uhi"]["suhii"]["lcz_based"]
    urban, rural = aoi.validate_lcz_classes(lcz["urban_classes"], lcz["rural_classes"])
    assert urban == list(range(1, 11))       # built classes 1-10
    assert rural == list(range(11, 18))      # A-G = 11-17 (user decision 2026-08-08)


@pytest.mark.parametrize(
    ("urban", "rural"),
    [
        ([], [11]),               # empty urban
        ([1], []),                # empty rural
        ([0, 1], [11]),           # urban out of range (low)
        ([1, 11], [12]),          # urban out of range (high)
        ([1], [10]),              # rural out of range (low)
        ([1], [17, 18]),          # rural out of range (high)
    ],
)
def test_lcz_rejects_bad_lists(urban: list[int], rural: list[int]) -> None:
    with pytest.raises(ValueError):
        aoi.validate_lcz_classes(urban, rural)


# --- aoi.validate_water_mask_params ---------------------------------------------
def test_water_mask_params_default_config_is_valid(params: dict[str, Any]) -> None:
    cfg = aoi.validate_water_mask_params(params)
    assert cfg is params["aoi"]["water_mask"]


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("mndwi_threshold",), 1.5),
        (("mndwi_threshold",), -2),
        (("qa_water_freq_threshold",), -0.1),
        (("qa_water_freq_threshold",), 1.1),
        (("jrc_occurrence_threshold_pct",), 101),
        (("jrc_occurrence_threshold_pct",), -1),
        (("shoreline_buffer_m",), -60),
        (("composite", "start_year"), 2030),          # > end_year 2025
        (("composite", "months"), []),
        (("composite", "months"), [0, 1]),
        (("composite", "months"), [13]),
        (("composite", "reducer"), "mode"),
        (("composite", "landsat_sources"), []),
        (("composite", "landsat_sources"), ["sentinel2"]),  # not a `datasets` key
    ],
)
def test_water_mask_params_rejects_bad_values(
    params_copy: dict[str, Any], path: tuple[str, ...], bad_value: Any
) -> None:
    node = params_copy["aoi"]["water_mask"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = bad_value
    with pytest.raises(ValueError):
        aoi.validate_water_mask_params(params_copy)


# --- import hygiene -------------------------------------------------------------
def test_modules_import_without_earthengine() -> None:
    # Deferred `import ee` rule: importing the modules must never require
    # earthengine-api. Reaching this point means the top-level imports above
    # succeeded; double-check the modules expose their public API.
    for name in (
        "colombo_district",
        "western_province",
        "ds_divisions",
        "gn_divisions",
        "cmc_boundary",
        "urban_extent",
        "analysis_region",
        "water_mask",
        "water_exclusion_mask",
        "buffer_ring",
        "urban_mask",
        "rural_mask",
        "rural_reference",
        "area_km2",
    ):
        assert callable(getattr(aoi, name)), name
    for name in ("bits_to_mask", "qa_clear_mask", "qa_water_flag", "scale_sr"):
        assert callable(getattr(landsat, name)), name
