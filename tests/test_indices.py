"""Unit tests for the pure-Python parts of indices.py (Phase 2).

The index formulas themselves are one-line Earth Engine expressions verified in
Colab. What is worth pinning here is the coefficient plumbing: that the albedo
set actually resolves, that its weights only reference bands that exist under
the harmonised names, and that the dispatch table stays in step with the band
names it advertises.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from colombo_uhi import indices, load_params


@pytest.fixture(scope="module")
def params() -> dict[str, Any]:
    return load_params()


@pytest.fixture()
def params_copy(params: dict[str, Any]) -> dict[str, Any]:
    """Deep copy for tests that mutate the config."""
    return copy.deepcopy(params)


# --- registry consistency --------------------------------------------------------
def test_all_six_indices_are_available() -> None:
    assert set(indices.INDEX_BAND_NAMES) == {
        "ndvi",
        "ndbi",
        "mndwi",
        "evi",
        "savi",
        "albedo",
    }


def test_dispatch_table_matches_band_name_table() -> None:
    # add_indices() looks names up in both; a drift between them would raise a
    # KeyError only for the one index nobody happened to request.
    assert set(indices.INDEX_FUNCTIONS) == set(indices.INDEX_BAND_NAMES)


def test_index_band_names_are_unique() -> None:
    names = list(indices.INDEX_BAND_NAMES.values())
    assert len(names) == len(set(names))


# --- resolve_indices --------------------------------------------------------------
def test_resolve_indices_defaults_to_all() -> None:
    assert indices.resolve_indices(None) == list(indices.INDEX_BAND_NAMES)


def test_resolve_indices_preserves_requested_order() -> None:
    assert indices.resolve_indices(["savi", "ndvi"]) == ["savi", "ndvi"]


def test_resolve_indices_collapses_duplicates() -> None:
    assert indices.resolve_indices(["ndvi", "ndvi", "evi"]) == ["ndvi", "evi"]


def test_resolve_indices_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        indices.resolve_indices([])


def test_resolve_indices_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="ndwi"):
        indices.resolve_indices(["ndvi", "ndwi"])


# --- albedo coefficients ----------------------------------------------------------
def test_albedo_default_set_is_liang(params: dict[str, Any]) -> None:
    assert params["indices"]["albedo"]["active_set"] == "liang2001"
    weights, constant, divisor = indices.albedo_coefficients(params)
    assert weights == {
        "blue": 0.356,
        "red": 0.130,
        "nir": 0.373,
        "swir1": 0.085,
        "swir2": 0.072,
    }
    assert constant == pytest.approx(-0.0018)
    # Liang's published Eq. 8 does not normalise by the coefficient sum.
    assert divisor == 1.0


def test_albedo_single_set_applies_to_every_sensor(params: dict[str, Any]) -> None:
    # Regression guard for the decision recorded in params: one coefficient set
    # across all four sensors, because swapping fits at the 2013 OLI transition
    # would inject a step change into the 2000-2025 trend series.
    assert "sensors" not in params["indices"]["albedo"]["sets"]["liang2001"]
    assert indices.albedo_coefficients(params, "liang2001")[0] == (
        indices.albedo_coefficients(params)[0]
    )


def test_albedo_sensitivity_variant_is_available(params: dict[str, Any]) -> None:
    weights, _, _ = indices.albedo_coefficients(params, "silva2016_oli")
    # The Silva set is genuinely different — it uses green, Liang does not.
    assert "green" in weights
    assert "green" not in indices.albedo_coefficients(params, "liang2001")[0]


def test_albedo_rejects_unknown_set(params: dict[str, Any]) -> None:
    with pytest.raises(KeyError, match="nonexistent"):
        indices.albedo_coefficients(params, "nonexistent")


def test_albedo_weights_reference_only_harmonised_bands(
    params: dict[str, Any],
) -> None:
    harmonised = set(params["landsat_c2l2"]["harmonised_sr_bands"])
    for set_name in params["indices"]["albedo"]["sets"]:
        weights, _, _ = indices.albedo_coefficients(params, set_name)
        assert set(weights) <= harmonised, set_name


def test_albedo_rejects_weights_on_a_raw_sensor_band(
    params_copy: dict[str, Any],
) -> None:
    # A tempting edit that would break sensor-agnosticism: naming SR_B2 instead
    # of "blue" works on OLI and silently means "green" on TM.
    params_copy["indices"]["albedo"]["sets"]["liang2001"]["weights"]["SR_B2"] = 0.1
    with pytest.raises(ValueError, match="SR_B2"):
        indices.albedo_coefficients(params_copy)


def test_albedo_rejects_zero_divisor(params_copy: dict[str, Any]) -> None:
    params_copy["indices"]["albedo"]["sets"]["liang2001"]["divisor"] = 0
    with pytest.raises(ValueError, match="divisor 0"):
        indices.albedo_coefficients(params_copy)


def test_albedo_clamp_is_a_valid_range(params: dict[str, Any]) -> None:
    low, high = params["indices"]["albedo"]["clamp"]
    assert low < high
    assert (low, high) == (0.0, 1.0)


# --- EVI / SAVI coefficients --------------------------------------------------------
def test_evi_coefficients_are_the_modis_formulation(params: dict[str, Any]) -> None:
    cfg = params["indices"]["evi"]
    assert (cfg["gain"], cfg["c1"], cfg["c2"], cfg["soil_l"]) == (2.5, 6.0, 7.5, 1.0)


def test_savi_soil_factor(params: dict[str, Any]) -> None:
    soil_l = params["indices"]["savi"]["soil_l"]
    assert soil_l == 0.5
    assert 0.0 <= soil_l <= 1.0


def test_savi_formula_reduces_to_ndvi_when_l_is_zero() -> None:
    # Sanity check on the algebra the ee.expression encodes:
    # ((nir - red) / (nir + red + L)) * (1 + L) with L = 0 is NDVI.
    nir, red, soil_l = 0.4, 0.1, 0.0
    savi = ((nir - red) / (nir + red + soil_l)) * (1 + soil_l)
    ndvi = (nir - red) / (nir + red)
    assert savi == pytest.approx(ndvi)
