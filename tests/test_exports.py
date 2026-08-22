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


# --- table formats -----------------------------------------------------------
@pytest.mark.parametrize("requested", ["GeoJSON", "geojson", "GEOJSON", " GeoJSON "])
def test_table_format_accepts_any_case_and_returns_the_canonical_spelling(
    requested: str,
) -> None:
    # THE Phase 5 blocker: table_to_drive upper-cased its input and then tested
    # membership against a tuple in which four of six names are MIXED case, so
    # every format except CSV was rejected - with the self-contradicting message
    # "unsupported table format 'GeoJSON'; expected one of [... 'GeoJSON' ...]".
    # The first Colab run of notebook 05 died on exactly this.
    assert exports.resolve_table_format(requested) == "GeoJSON"


@pytest.mark.parametrize("name", exports.TABLE_FORMATS)
def test_every_declared_table_format_actually_resolves(name: str) -> None:
    # The regression guard proper: a format that is advertised but unusable is
    # worse than one that is absent.
    assert exports.resolve_table_format(name) == name
    assert exports.resolve_table_format(name.lower()) == name
    assert exports.resolve_table_format(name.upper()) == name


def test_table_format_rejects_an_unknown_format() -> None:
    with pytest.raises(ValueError, match="unsupported table format"):
        exports.resolve_table_format("parquet")


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


# --- the submit-time band_order guard (Phase 8, Colab run 3) ------------------
class _Bands:
    """Stand-in for ``image.bandNames()``."""

    def __init__(self, names: list[str] | None, fails: bool = False) -> None:
        self._names = names
        self._fails = fails

    def getInfo(self) -> list[str] | None:  # noqa: N802 - Earth Engine's spelling
        if self._fails:
            raise RuntimeError("no network")
        return self._names


class _Image:
    """Stand-in for an ``ee.Image``; records what was selected."""

    def __init__(self, names: list[str] | None, fails: bool = False) -> None:
        self._bands = _Bands(names, fails)
        self.selected: list[str] | None = None

    def bandNames(self) -> _Bands:  # noqa: N802 - Earth Engine's spelling
        return self._bands

    def select(self, names: list[str]) -> "_Image":
        self.selected = list(names)
        return self


def test_missing_export_bands_lists_what_the_image_lacks() -> None:
    image = _Image(["mean_2000_2010", "sd_2000_2010", "n_years_2000_2010"])
    assert exports.missing_export_bands(image, ["mean_2000_2010"]) == []
    # Exactly the Phase 8 case: decadal_band_order names difference bands that
    # only decadal_product creates.
    assert exports.missing_export_bands(
        image, ["mean_2000_2010", "diff_2011_2020_minus_2000_2010", "diff_se"]
    ) == ["diff_2011_2020_minus_2000_2010", "diff_se"]


def test_missing_export_bands_never_masks_a_failure_of_its_own() -> None:
    # The helper is a diagnostic. If it cannot reach Earth Engine it must report
    # nothing missing, so the caller proceeds and the real error surfaces.
    assert exports.missing_export_bands(_Image(None, fails=True), ["x"]) == []


def test_image_to_drive_refuses_a_band_order_the_image_cannot_satisfy(
    params: dict[str, Any],
) -> None:
    """Phase 8 lost two Colab runs to this exact mismatch.

    ``image.select`` is lazy, so Earth Engine accepted the export and failed
    inside the batch task six minutes later with "Band pattern
    'diff_2011_2020_minus_2000_2010' did not match any bands". The guard makes
    it a submit-time refusal instead.
    """
    image = _Image(["mean_2000_2010", "sd_2000_2010", "n_years_2000_2010"])

    with pytest.raises(ValueError) as caught:
        exports.image_to_drive(
            image, product="lst_decadal", aoi="district", params=params,
            region=object(),
            band_order=["mean_2000_2010", "diff_2011_2020_minus_2000_2010"],
            scale_m=1000, suffix="terra_day",
        )
    message = str(caught.value)
    # Both lists, so the reader does not have to go and look either of them up.
    assert "diff_2011_2020_minus_2000_2010" in message
    assert "mean_2000_2010" in message
    assert "decadal_product" in message and "decadal_means" in message
    # It refused BEFORE selecting or submitting anything.
    assert image.selected is None


def test_image_to_drive_can_skip_the_check(params: dict[str, Any]) -> None:
    # Opt-out exists, and taking it means the lazy failure comes back - which is
    # the caller's choice to make, not a silent default.
    image = _Image(["mean_2000_2010"])
    with pytest.raises(Exception):  # noqa: B017 - the fake has no ee.batch
        exports.image_to_drive(
            image, product="lst_decadal", aoi="district", params=params,
            region=object(), band_order=["diff_2011_2020_minus_2000_2010"],
            scale_m=1000, verify_bands=False,
        )
    # The band order was applied unchecked, which is what verify_bands=False means.
    assert image.selected == ["diff_2011_2020_minus_2000_2010"]
