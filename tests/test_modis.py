"""Unit tests for the pure-Python parts of modis.py (Phase 2).

The QC bit arithmetic is the thing worth pinning: MOD11A2 ships no built-in
quality filtering, so reading the wrong bit field would not crash — it would
quietly return cloud-contaminated LST that looks entirely plausible.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from colombo_uhi import load_params, modis


@pytest.fixture(scope="module")
def params() -> dict[str, Any]:
    return load_params()


@pytest.fixture()
def params_copy(params: dict[str, Any]) -> dict[str, Any]:
    """Deep copy for tests that mutate the config."""
    return copy.deepcopy(params)


# --- qc_bit_range ---------------------------------------------------------------
def test_qc_bit_range_mandatory_qa_field(params: dict[str, Any]) -> None:
    # Bits 0-1, so no shift and a 2-bit mask.
    assert modis.qc_bit_range(params["modis_lst"]["qc_filter"]["mandatory_qa_bits"]) == (
        0,
        0b11,
    )


def test_qc_bit_range_lst_error_field(params: dict[str, Any]) -> None:
    # Bits 6-7: shift down 6, then mask 2 bits.
    assert modis.qc_bit_range(params["modis_lst"]["qc_filter"]["lst_error_bits"]) == (
        6,
        0b11,
    )


@pytest.mark.parametrize(
    ("bits", "expected"),
    [
        ([0], (0, 0b1)),
        ([3], (3, 0b1)),
        ([2, 3], (2, 0b11)),
        ([4, 5], (4, 0b11)),
        ([4, 5, 6, 7], (4, 0b1111)),
    ],
)
def test_qc_bit_range_widths(bits: list[int], expected: tuple[int, int]) -> None:
    assert modis.qc_bit_range(bits) == expected


def test_qc_bit_range_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        modis.qc_bit_range([])


def test_qc_bit_range_rejects_negative() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        modis.qc_bit_range([-1, 0])


@pytest.mark.parametrize("bits", [[0, 2], [6, 8], [1, 2, 4]])
def test_qc_bit_range_rejects_non_contiguous(bits: list[int]) -> None:
    with pytest.raises(ValueError, match="contiguous"):
        modis.qc_bit_range(bits)


def test_qc_bit_range_rejects_descending() -> None:
    # [7, 6] would shift by 7 and read the wrong field entirely.
    with pytest.raises(ValueError, match="contiguous"):
        modis.qc_bit_range([7, 6])


def test_qc_bit_extraction_arithmetic_matches_catalog() -> None:
    # Worked example: QC byte with mandatory QA = 1 (unreliable) and LST error
    # = 2 (<= 3 K) is 0b10_00_00_01. Both fields must decode independently.
    qc_value = 0b10000001
    qa_shift, qa_mask = modis.qc_bit_range([0, 1])
    err_shift, err_mask = modis.qc_bit_range([6, 7])
    assert (qc_value >> qa_shift) & qa_mask == 1
    assert (qc_value >> err_shift) & err_mask == 2

    # A fully good pixel: both fields zero.
    assert (0b00000000 >> qa_shift) & qa_mask == 0
    assert (0b00000000 >> err_shift) & err_mask == 0


def test_day_qc_policy_is_claude_md_strict(params: dict[str, Any]) -> None:
    # Day must stay exactly "good quality AND avg LST error <= 1 K".
    assert modis.qc_thresholds(params, "day") == (0, 0)


def test_night_qc_policy_is_the_documented_deviation(params: dict[str, Any]) -> None:
    # MEASURED, Colab run 5: strict thresholds returned ZERO night pixels over
    # the CMC for all 26 years on both satellites. Night is deliberately looser.
    # If this test fails because someone "corrected" night back to 0/0, read the
    # params comment first — night-time UHI disappears entirely.
    qa_max, err_max = modis.qc_thresholds(params, "night")
    assert (qa_max, err_max) == (1, 2)
    day_qa, day_err = modis.qc_thresholds(params, "day")
    assert qa_max >= day_qa and err_max >= day_err


def test_qc_thresholds_honour_explicit_overrides(params: dict[str, Any]) -> None:
    # Phase 3 sensitivity runs must not need a params edit.
    assert modis.qc_thresholds(params, "night", mandatory_qa_max=0) == (0, 2)
    assert modis.qc_thresholds(params, "day", lst_error_max=3) == (0, 3)
    assert modis.qc_thresholds(params, "day", 1, 1) == (1, 1)


@pytest.mark.parametrize("bad", [-1, 4, 99])
def test_qc_thresholds_reject_out_of_range(params: dict[str, Any], bad: int) -> None:
    # Both QC fields are 2-bit, so anything outside 0..3 is a silent no-op filter.
    with pytest.raises(ValueError, match="0..3"):
        modis.qc_thresholds(params, "day", mandatory_qa_max=bad)
    with pytest.raises(ValueError, match="0..3"):
        modis.qc_thresholds(params, "day", lst_error_max=bad)


def test_qc_thresholds_reject_unknown_overpass(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="daynight"):
        modis.qc_thresholds(params, "twilight")


# --- qc_field_labels -------------------------------------------------------------
def test_qc_field_labels_cover_all_four_classes(params: dict[str, Any]) -> None:
    for field in ("mandatory_qa", "lst_error"):
        labels = modis.qc_field_labels(params, field)
        assert set(labels) == {0, 1, 2, 3}
        assert all(text.strip() for text in labels.values())


def test_qc_field_labels_match_the_catalog(params: dict[str, Any]) -> None:
    assert "good quality" in modis.qc_field_labels(params, "mandatory_qa")[0]
    assert "<= 1 K" in modis.qc_field_labels(params, "lst_error")[0]
    assert "> 3 K" in modis.qc_field_labels(params, "lst_error")[3]


def test_qc_field_labels_reject_unknown_field(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="emissivity_error"):
        modis.qc_field_labels(params, "emissivity_error")


def test_qc_field_labels_reject_an_incomplete_map(params_copy: dict[str, Any]) -> None:
    # A gap would print None for exactly the class that explains a data gap.
    del params_copy["modis_lst"]["qc_filter"]["lst_error_labels"][2]
    with pytest.raises(ValueError, match="classes 0-3"):
        modis.qc_field_labels(params_copy, "lst_error")


# --- resolve_product / resolve_daynight ----------------------------------------
def test_resolve_product(params: dict[str, Any]) -> None:
    assert modis.resolve_product("terra", params) == "modis_terra_lst"
    assert modis.resolve_product("aqua", params) == "modis_aqua_lst"


def test_resolve_product_rejects_unknown(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="suomi"):
        modis.resolve_product("suomi", params)


def test_resolve_product_keys_exist_in_datasets(params: dict[str, Any]) -> None:
    for dataset_key in params["modis_lst"]["products"].values():
        assert dataset_key in params["datasets"]


@pytest.mark.parametrize("value", ["day", "night"])
def test_resolve_daynight_accepts(value: str) -> None:
    assert modis.resolve_daynight(value) == value


@pytest.mark.parametrize("value", ["Day", "dusk", "", "daytime"])
def test_resolve_daynight_rejects(value: str) -> None:
    with pytest.raises(ValueError, match="daynight"):
        modis.resolve_daynight(value)


def test_band_keys_derived_from_daynight_exist(params: dict[str, Any]) -> None:
    # lst_collection builds band names by f-string; make sure both spellings
    # resolve for both products.
    for product_key in params["modis_lst"]["products"].values():
        dataset = params["datasets"][product_key]
        for daynight in modis.DAYNIGHT_OPTIONS:
            assert f"{daynight}_band" in dataset
            assert f"qc_{daynight}_band" in dataset
            assert f"clear_sky_{daynight}_band" in params["modis_lst"]


# --- clamp_start_date -----------------------------------------------------------
def test_clamp_start_date_clamps_aqua_to_launch(params: dict[str, Any]) -> None:
    # Aqua did not exist for the first 2.5 years of the study period.
    with pytest.warns(UserWarning, match="2002-07-04"):
        assert modis.clamp_start_date("2000-01-01", "aqua", params) == "2002-07-04"


def test_clamp_start_date_clamps_terra_to_launch(params: dict[str, Any]) -> None:
    with pytest.warns(UserWarning, match="2000-02-18"):
        assert modis.clamp_start_date("2000-01-01", "terra", params) == "2000-02-18"


def test_clamp_start_date_passes_through_later_dates(params: dict[str, Any]) -> None:
    assert modis.clamp_start_date("2010-01-01", "aqua", params) == "2010-01-01"


def test_clamp_start_date_is_a_no_op_on_the_boundary(params: dict[str, Any]) -> None:
    available = params["datasets"]["modis_aqua_lst"]["availability"][0]
    assert modis.clamp_start_date(available, "aqua", params) == available


# --- clear-sky bitmask configuration -------------------------------------------
def test_clear_sky_is_declared_a_bitmask(params: dict[str, Any]) -> None:
    # The single most damaging misreading available in this dataset: treating
    # the bitmask as a count reports 255 clear days instead of 8.
    cfg = params["modis_lst"]
    assert cfg["clear_sky_is_bitmask"] is True
    assert cfg["clear_sky_bits"] == 8


def test_clear_sky_popcount_arithmetic() -> None:
    # Mirrors the summation clear_sky_count() performs server-side.
    def popcount(value: int, n_bits: int = 8) -> int:
        return sum((value >> bit) & 1 for bit in range(n_bits))

    assert popcount(0b11111111) == 8  # all eight days clear -> 8, NOT 255
    assert popcount(0b00000000) == 0
    assert popcount(0b00000101) == 2
    assert popcount(0b10000000) == 1


def test_modis_valid_dn_range_floor_is_physical(params: dict[str, Any]) -> None:
    dn_min, dn_max = params["modis_lst"]["valid_dn_range"]
    assert dn_min == 7500
    assert dn_max == 65535
    # 7500 * 0.02 = 150 K, the same physical floor as the Landsat DN gate.
    assert dn_min * params["modis_lst"]["lst_scale"] == pytest.approx(150.0)
