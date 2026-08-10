"""Pin the pure core of the export layer: naming, settings, and task status.

No Earth Engine anywhere - the ``ee.batch.Export`` wrappers are verified only by
running notebook 04 in Colab. What is testable here is the naming, and that is
where the one genuinely destructive failure lives: a template that drops a field
lets two different products render to the SAME filename, and the second export
silently overwrites the first in Drive.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from colombo_uhi import exports, load_params


@pytest.fixture(scope="module")
def params() -> dict[str, Any]:
    return load_params()


@pytest.fixture()
def params_copy(params: dict[str, Any]) -> dict[str, Any]:
    """Deep copy for tests that mutate the config."""
    return copy.deepcopy(params)


# --- export_name -------------------------------------------------------------
def test_export_name_renders_the_pinned_template(params: dict[str, Any]) -> None:
    assert (
        exports.export_name("lst_trend", "district", params, 2000, 2025, 100)
        == "lst_trend_district_2000_2025_100m"
    )


def test_export_name_defaults_years_and_resolution_from_params(
    params: dict[str, Any],
) -> None:
    name = exports.export_name("lst_trend", "cmc", params)
    assert str(params["time"]["start_year"]) in name
    assert str(params["time"]["end_year"]) in name
    assert name.endswith(f"{params['exports']['default_scale_m']}m")


def test_export_name_sanitises_spaces_and_slashes(params: dict[str, Any]) -> None:
    name = exports.export_name("LST Trend/Sen", "CMC core", params, 2000, 2025, 30)
    assert name == "lst_trend_sen_cmc_core_2000_2025_30m"


def test_export_name_appends_a_suffix(params: dict[str, Any]) -> None:
    name = exports.export_name(
        "lst_trend", "cmc", params, 2000, 2025, 30, suffix="terra_night"
    )
    assert name == "lst_trend_cmc_2000_2025_30m_terra_night"


def test_export_name_rejects_an_inverted_year_range(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be >= start_year"):
        exports.export_name("lst_trend", "cmc", params, 2025, 2000, 30)


def test_export_name_rejects_a_non_positive_resolution(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="positive"):
        exports.export_name("lst_trend", "cmc", params, 2000, 2025, 0)


def test_export_name_rejects_a_name_over_the_task_limit(
    params_copy: dict[str, Any],
) -> None:
    # Earth Engine truncates long task descriptions, which can collapse two
    # different products onto one name.
    params_copy["exports"]["max_name_chars"] = 20
    with pytest.raises(ValueError, match="max_name_chars"):
        exports.export_name("a_very_long_product_name", "district", params_copy)


def test_export_name_rejects_an_empty_component(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="empty"):
        exports.export_name("!!!", "cmc", params)


# --- template validation -----------------------------------------------------
def test_template_rejects_an_unknown_placeholder(params: dict[str, Any]) -> None:
    with pytest.raises(KeyError, match="sensor"):
        exports.resolve_name_template("{product}_{sensor}", params)


def test_template_rejects_a_dropped_field(params: dict[str, Any]) -> None:
    # THE destructive one: without {startyear}, a 2000-2010 export and a
    # 2011-2020 export render to the same name and the second overwrites the
    # first in Drive with no warning anywhere.
    with pytest.raises(ValueError, match="startyear"):
        exports.resolve_name_template("{product}_{aoi}_{endyear}_{res}m", params)


def test_template_defaults_to_the_configured_one(params: dict[str, Any]) -> None:
    assert (
        exports.resolve_name_template(None, params)
        == params["exports"]["name_template"]
    )


# --- export settings ---------------------------------------------------------
def test_settings_default_to_the_analysis_crs_and_grid(params: dict[str, Any]) -> None:
    settings = exports.resolve_export_settings(params)
    assert settings["crs"] == params["exports"]["default_crs"]
    assert settings["scale_m"] == params["exports"]["default_scale_m"]
    assert settings["folder"] == params["exports"]["drive_folder"]
    assert settings["max_pixels"] == params["composites"]["reduce_max_pixels"]


def test_settings_reject_a_non_positive_scale(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="positive"):
        exports.resolve_export_settings(params, scale_m=-30)


def test_settings_reject_an_empty_folder(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        exports.resolve_export_settings(params, folder="   ")


# --- task status -------------------------------------------------------------
def test_task_status_frame_has_the_documented_columns() -> None:
    frame = exports.build_task_status_frame(
        [
            {
                "description": "lst_trend_cmc_2000_2025_100m",
                "id": "ABC123",
                "state": "COMPLETED",
                "start_timestamp_ms": 1_000_000,
                "update_timestamp_ms": 1_060_000,
                "destination_uris": ["https://drive.google.com/x"],
            }
        ]
    )
    assert list(frame.columns) == list(exports.TASK_COLUMNS)
    assert frame["runtime_s"].iloc[0] == pytest.approx(60.0)
    assert frame["destination"].iloc[0] == "https://drive.google.com/x"


def test_task_status_frame_tolerates_missing_keys() -> None:
    # A running task has no error_message and a fresh one has no
    # destination_uris; neither may raise.
    frame = exports.build_task_status_frame([{"state": "RUNNING"}])
    assert frame["runtime_s"].iloc[0] is None
    assert frame["error"].iloc[0] is None


def test_task_status_frame_surfaces_the_error_message() -> None:
    frame = exports.build_task_status_frame(
        [{"state": "FAILED", "error_message": "Image.reduceRegion: too many pixels"}]
    )
    assert "too many pixels" in frame["error"].iloc[0]


def test_task_status_frame_of_no_tasks_is_empty_but_shaped() -> None:
    frame = exports.build_task_status_frame([])
    assert frame.empty
    assert list(frame.columns) == list(exports.TASK_COLUMNS)


def test_all_terminal_recognises_the_terminal_states() -> None:
    assert exports.all_terminal([{"state": "COMPLETED"}, {"state": "FAILED"}])
    assert not exports.all_terminal([{"state": "COMPLETED"}, {"state": "RUNNING"}])
