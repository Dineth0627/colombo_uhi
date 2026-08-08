"""Unit tests for the pure-Python parts of aoi.py and landsat.py (Phase 1).

Earth Engine-dependent functions are exercised only in Colab (notebook 01);
these tests cover validation logic, the QA bitmask helper, and the rule that
both modules import cleanly without ``earthengine-api`` installed.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from colombo_uhi import aoi, landsat, load_params, viz


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


# --- aoi.validate_property_candidates -------------------------------------------
def test_property_candidates_from_params_are_valid(params: dict[str, Any]) -> None:
    assets = params["aoi"]["assets"]
    for key in (
        "ds_name_property_candidates",
        "gn_name_property_candidates",
        "district_property_candidates",
    ):
        resolved = aoi.validate_property_candidates(assets[key], key)
        assert resolved == assets[key]


def test_codab_lowercase_names_are_first_candidates(params: dict[str, Any]) -> None:
    # The uploaded assets are OCHA COD-AB v03 with LOWERCASE fields (verified by
    # describe_asset in Colab, 2026-08-08). Uppercase-only candidates matched
    # nothing and silently produced 0 DS / 0 GN, so pin the ordering.
    assets = params["aoi"]["assets"]
    assert assets["ds_name_property_candidates"][0] == "adm3_name"
    assert assets["gn_name_property_candidates"][0] == "adm4_name"
    assert assets["district_property_candidates"][0] == "adm2_name"
    assert assets["area_property"] == "area_sqkm"


@pytest.mark.parametrize(
    "candidates",
    [
        [],                          # empty
        ["ADM3_EN", "ADM3_EN"],      # duplicate
        ["ADM3_EN", ""],             # blank
        ["ADM3_EN", "   "],          # whitespace only
        ["ADM3_EN", None],           # non-string
        ["ADM3_EN", 3],              # non-string
    ],
)
def test_property_candidates_rejects_bad_lists(candidates: list[Any]) -> None:
    with pytest.raises(ValueError):
        aoi.validate_property_candidates(candidates, "test")


def test_property_candidates_error_names_the_label() -> None:
    with pytest.raises(ValueError, match="DS-division name"):
        aoi.validate_property_candidates([], "DS-division name")


# --- Phase 1b config expectations -------------------------------------------------
def test_expected_counts_match_claude_md(params: dict[str, Any]) -> None:
    counts = params["aoi"]["expected_counts"]
    assert counts["ds_divisions"] == params["aoi"]["district"]["ds_divisions"] == 13
    assert counts["gn_divisions"] == params["aoi"]["district"]["gn_divisions"] == 557


def test_buffer_ring_base_is_cmc(params: dict[str, Any]) -> None:
    # Set 2026-08-08: basing the ring on the province-wide GHSL urban extent
    # produced a 4049 km2 ring that was not centred on Colombo.
    assert params["uhi"]["suhii"]["buffer_ring"]["base"] == "cmc"


def test_rural_elevation_cap(params: dict[str, Any]) -> None:
    max_elev = params["uhi"]["suhii"]["rural_filters"]["max_elevation_m"]
    assert max_elev is None or max_elev > 0


def test_cmc_definition_is_gn_union(params: dict[str, Any]) -> None:
    # The DS pair measures 46.87 km2 with COD-AB polygons (its Colombo DS
    # encloses the Port's outer harbour), so no DS union can give the gazetted
    # 37.31 km2. The CMC's own GIS Unit GN list is authoritative.
    cmc = params["aoi"]["cmc"]
    assert cmc["definition"] in aoi.CMC_DEFINITIONS
    assert cmc["definition"] == "gn_union"
    assert params["aoi"]["expected_areas_km2"]["cmc"] == 37


def test_cmc_gn_list_is_55_clean_unique_names(params: dict[str, Any]) -> None:
    cmc = params["aoi"]["cmc"]
    names = cmc["gn_division_names"]
    assert len(names) == cmc["expected_gn_count"] == 55
    assert len(set(names)) == len(names), "duplicate GN names"
    for name in names:
        assert isinstance(name, str) and name, f"bad GN name: {name!r}"
        # A stray space silently breaks ee.Filter.inList and undersizes the CMC.
        assert name == name.strip(), f"whitespace around GN name: {name!r}"
        assert "  " not in name, f"double space in GN name: {name!r}"


def test_cmc_gn_list_contains_known_landmarks(params: dict[str, Any]) -> None:
    # Spot-check against the CMC GIS Unit map + the COD-AB samples seen in Colab.
    names = set(params["aoi"]["cmc"]["gn_division_names"])
    for expected in ("Fort", "Pettah", "Sammanthranapura", "Mattakkuliya",
                     "Bambalapitiya", "Slave Island"):
        assert expected in names, expected


def test_cmc_gn_names_use_codab_spellings(params: dict[str, Any]) -> None:
    """Five names differ between the CMC map and COD-AB's adm4_name.

    Colab run 4 matched only 50/55 and printed the asset's spellings. These are
    the corrections; reverting any of them silently undersizes the CMC.
    """
    names = set(params["aoi"]["cmc"]["gn_division_names"])
    corrections = {
        "Ibanwala": "Ibbanwala",
        "Kettarama": "Khettarama",
        "Kirulapona": "Kirulapone",
        "Kotehena East": "Kotahena East",
        "Kotehena West": "Kotahena West",
    }
    for map_spelling, codab_spelling in corrections.items():
        assert codab_spelling in names, f"missing COD-AB spelling {codab_spelling!r}"
        assert map_spelling not in names, f"stale CMC-map spelling {map_spelling!r}"


def test_cmc_gn_selection_is_scoped_to_parent_ds(params: dict[str, Any]) -> None:
    # GN names are NOT unique within Colombo District: unscoped matching pulled in
    # same-named divisions elsewhere and made 50 units measure 47.50 km2, more than
    # both parent DS divisions combined (46.87). Colab run 4.
    assert params["aoi"]["cmc"]["parent_ds_scope"] is True


def test_cmc_expected_areas_separate_land_from_administrative(
    params: dict[str, Any],
) -> None:
    expected = params["aoi"]["expected_areas_km2"]
    # 37 = gazetted, checked against the LAND area; ~47 = the raw polygon, which
    # legitimately includes the Colombo Port outer harbour.
    assert expected["cmc"] == 37
    assert expected["cmc_administrative"] > expected["cmc"]


def test_lcz_scope_is_valid(params: dict[str, Any]) -> None:
    # Set 2026-08-08: unscoped LCZ masks spanned Western Province + 25 km, making
    # "urban" 2464 km2 of built-up across Gampaha and Kalutara.
    assert params["uhi"]["suhii"]["lcz_based"]["scope"] == "district"


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
        "mask_area_km2",
        "describe_asset",
        "cmc_name_audit",
        "cmc_land_area_km2",
    ):
        assert callable(getattr(aoi, name)), name
    for name in ("bits_to_mask", "qa_clear_mask", "qa_water_flag", "scale_sr"):
        assert callable(getattr(landsat, name)), name
    for name in ("outline_image", "elevation_backdrop", "save_thumbnail"):
        assert callable(getattr(viz, name)), name
