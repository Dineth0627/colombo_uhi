"""Unit tests for the pure-Python parts of landsat.py (Phase 2).

The Earth Engine parts (``harmonised_collection`` and friends) are verified only
in Colab, via notebook 02. What is testable here is exactly the logic that would
otherwise fail silently: the month -> season partition, the per-sensor band
mapping (TM and OLI number their bands differently), and the fixed output band
schema that makes the four collections safe to merge.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from colombo_uhi import landsat, load_params


@pytest.fixture(scope="module")
def params() -> dict[str, Any]:
    return load_params()


@pytest.fixture()
def params_copy(params: dict[str, Any]) -> dict[str, Any]:
    """Deep copy for tests that mutate the config."""
    return copy.deepcopy(params)


# --- monsoon_season / season_by_month -----------------------------------------
def test_season_partition_covers_every_month_exactly_once(
    params: dict[str, Any],
) -> None:
    # The whole point of time.season_partition: a total, disjoint partition.
    # If this fails, some scenes get no season and others get two.
    assigned = [landsat.monsoon_season(m, params) for m in range(1, 13)]
    assert len(assigned) == 12
    assert set(assigned) == set(params["time"]["season_partition"])


def test_monsoon_season_matches_claude_md_windows(params: dict[str, Any]) -> None:
    for month in (5, 6, 7, 8, 9):
        assert landsat.monsoon_season(month, params) == "sw_monsoon"
    for month in (12, 1, 2):
        assert landsat.monsoon_season(month, params) == "ne_monsoon"
    for month in (3, 4, 10, 11):
        assert landsat.monsoon_season(month, params) == "inter_monsoon"


@pytest.mark.parametrize("month", [0, 13, -1, 100])
def test_monsoon_season_rejects_out_of_range(
    params: dict[str, Any], month: int
) -> None:
    with pytest.raises(ValueError, match="1..12"):
        landsat.monsoon_season(month, params)


def test_monsoon_season_rejects_overlapping_partition(
    params_copy: dict[str, Any],
) -> None:
    # dry_window (Jan-Mar) overlaps ne_monsoon; adding it to the partition must
    # be caught loudly rather than mapping January to two seasons.
    params_copy["time"]["season_partition"].append("dry_window")
    with pytest.raises(ValueError, match="exactly one"):
        landsat.monsoon_season(1, params_copy)


def test_monsoon_season_rejects_incomplete_partition(
    params_copy: dict[str, Any],
) -> None:
    params_copy["time"]["season_partition"].remove("sw_monsoon")
    with pytest.raises(ValueError, match="0 seasons"):
        landsat.monsoon_season(7, params_copy)


def test_dry_window_is_not_in_the_partition(params: dict[str, Any]) -> None:
    # It is a comparison WINDOW that deliberately straddles two seasons.
    assert "dry_window" not in params["time"]["season_partition"]
    assert params["time"]["seasons"]["dry_window"]["months"] == [1, 2, 3]


def test_season_by_month_is_january_first(params: dict[str, Any]) -> None:
    lookup = landsat.season_by_month(params)
    assert len(lookup) == 12
    assert lookup[0] == landsat.monsoon_season(1, params)
    assert lookup[11] == landsat.monsoon_season(12, params)
    assert lookup[6] == "sw_monsoon"  # July


# --- sensor_keys / resolve_sensors ---------------------------------------------
def test_sensor_keys_are_the_four_landsat_collections(params: dict[str, Any]) -> None:
    assert landsat.sensor_keys(params) == [
        "landsat5",
        "landsat7",
        "landsat8",
        "landsat9",
    ]


def test_sensor_keys_rejects_unconfigured_dataset(params_copy: dict[str, Any]) -> None:
    params_copy["landsat_c2l2"]["sensor_keys"].append("landsat42")
    with pytest.raises(KeyError, match="landsat42"):
        landsat.sensor_keys(params_copy)


def test_resolve_sensors_defaults_to_all(params: dict[str, Any]) -> None:
    assert landsat.resolve_sensors(None, params) == landsat.sensor_keys(params)


def test_resolve_sensors_preserves_chronological_order(params: dict[str, Any]) -> None:
    # Requested out of order; result must follow the configured order so the
    # merged collection is built oldest-first.
    assert landsat.resolve_sensors(["landsat9", "landsat5"], params) == [
        "landsat5",
        "landsat9",
    ]


def test_resolve_sensors_rejects_empty(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="at least one"):
        landsat.resolve_sensors([], params)


def test_resolve_sensors_rejects_unknown(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="modis"):
        landsat.resolve_sensors(["modis"], params)


# --- source_sr_bands: the TM-vs-OLI band shift ---------------------------------
def test_source_sr_bands_tm_numbering(params: dict[str, Any]) -> None:
    # TM/ETM+: blue is SR_B1 and there is no SR_B6 in the SR set.
    for key in ("landsat5", "landsat7"):
        assert landsat.source_sr_bands(key, params) == [
            "SR_B1",
            "SR_B2",
            "SR_B3",
            "SR_B4",
            "SR_B5",
            "SR_B7",
        ]


def test_source_sr_bands_oli_numbering(params: dict[str, Any]) -> None:
    # OLI: everything shifts by one, blue is SR_B2 and swir1 is SR_B6.
    for key in ("landsat8", "landsat9"):
        assert landsat.source_sr_bands(key, params) == [
            "SR_B2",
            "SR_B3",
            "SR_B4",
            "SR_B5",
            "SR_B6",
            "SR_B7",
        ]


def test_source_sr_bands_order_matches_harmonised_names(
    params: dict[str, Any],
) -> None:
    # The rename is positional, so a mismatch here would silently swap bands
    # (e.g. call SWIR "nir") on one sensor only.
    harmonised = params["landsat_c2l2"]["harmonised_sr_bands"]
    for key in landsat.sensor_keys(params):
        source = landsat.source_sr_bands(key, params)
        mapping = params["datasets"][key]["sr_bands"]
        assert source == [mapping[name] for name in harmonised]


def test_source_sr_bands_raises_on_missing_mapping(
    params_copy: dict[str, Any],
) -> None:
    del params_copy["datasets"]["landsat8"]["sr_bands"]["nir"]
    with pytest.raises(KeyError, match="nir"):
        landsat.source_sr_bands("landsat8", params_copy)


# --- output_band_names ----------------------------------------------------------
def test_output_band_names_schema(params: dict[str, Any]) -> None:
    assert landsat.output_band_names(params) == [
        "LST_C",
        "blue",
        "green",
        "red",
        "nir",
        "swir1",
        "swir2",
        "ST_QA_K",
    ]


def test_output_band_names_are_unique(params: dict[str, Any]) -> None:
    names = landsat.output_band_names(params)
    assert len(names) == len(set(names))


# --- month_filter validation (runs before the deferred ee import) ---------------
def test_month_filter_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        landsat.month_filter([])


@pytest.mark.parametrize("months", [[0], [13], [1, 2, 99]])
def test_month_filter_rejects_out_of_range(months: list[int]) -> None:
    with pytest.raises(ValueError, match="1..12"):
        landsat.month_filter(months)
