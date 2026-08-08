"""Unit tests for the pure-Python parts of composites.py (Phase 2).

Two things carry real risk and are therefore pinned hard:

* the reducer output band names, because ``ImageCollection.reduce`` derives them
  from the input band and a wrong guess produces a "Pattern did not match any
  bands" failure only after a long Colab round trip;
* the scene-inventory and zonal-table shaping, because both are fed by a single
  Earth Engine round trip whose failure modes (desynced arrays, a fully-masked
  year) are silent unless something checks.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from colombo_uhi import composites, load_params


@pytest.fixture(scope="module")
def params() -> dict[str, Any]:
    return load_params()


@pytest.fixture()
def params_copy(params: dict[str, Any]) -> dict[str, Any]:
    """Deep copy for tests that mutate the config."""
    return copy.deepcopy(params)


# --- reducer / percentile resolution ---------------------------------------------
def test_resolve_reducer_defaults_to_median(params: dict[str, Any]) -> None:
    assert composites.resolve_reducer_name(None, params) == "median"


def test_resolve_reducer_accepts_mean(params: dict[str, Any]) -> None:
    assert composites.resolve_reducer_name("mean", params) == "mean"


def test_resolve_reducer_rejects_unsupported(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="mode"):
        composites.resolve_reducer_name("mode", params)


def test_modis_reducer_is_configured(params: dict[str, Any]) -> None:
    # annual_lst() defaults to it, so it must be a reducer we can build.
    assert (
        composites.resolve_reducer_name(params["composites"]["modis_reducer"], params)
        == "mean"
    )


def test_resolve_percentile_defaults_to_params(params: dict[str, Any]) -> None:
    assert composites.resolve_percentile(None, params) == 90


@pytest.mark.parametrize("value", [0, 50, 95, 100])
def test_resolve_percentile_accepts_valid_ints(
    params: dict[str, Any], value: int
) -> None:
    assert composites.resolve_percentile(value, params) == value


@pytest.mark.parametrize("value", [-1, 101, 1000])
def test_resolve_percentile_rejects_out_of_range(
    params: dict[str, Any], value: int
) -> None:
    with pytest.raises(ValueError, match="0..100"):
        composites.resolve_percentile(value, params)


def test_resolve_percentile_rejects_float(params: dict[str, Any]) -> None:
    # The band name is derived from the value, so 90.0 would give "p90.0".
    with pytest.raises(ValueError, match="int"):
        composites.resolve_percentile(90.5, params)  # type: ignore[arg-type]


def test_resolve_percentile_rejects_bool(params: dict[str, Any]) -> None:
    # bool is an int subclass; True would silently mean the 1st percentile.
    with pytest.raises(ValueError, match="int"):
        composites.resolve_percentile(True, params)  # type: ignore[arg-type]


# --- band naming ------------------------------------------------------------------
def test_reduced_band_names_match_earth_engine_convention() -> None:
    # ImageCollection.reduce prefixes single-input reducer outputs with the
    # input band name.
    assert composites.reduced_band_names("LST_C", "median", 90) == [
        "LST_C_median",
        "LST_C_p90",
        "LST_C_count",
    ]


def test_reduced_band_names_follow_the_reducer(params: dict[str, Any]) -> None:
    assert composites.reduced_band_names("LST_C", "mean", 95)[0] == "LST_C_mean"
    assert composites.reduced_band_names("LST_C", "mean", 95)[1] == "LST_C_p95"


def test_composite_band_names_keep_a_stable_primary_band(
    params: dict[str, Any],
) -> None:
    # The primary band drops the reducer suffix so downstream phases can select
    # "LST_C" regardless of which reducer built it.
    assert composites.composite_band_names("LST_C", 90, params) == [
        "LST_C",
        "LST_C_p90",
        "obs_count",
    ]


def test_composite_band_names_align_with_reduced_names(
    params: dict[str, Any],
) -> None:
    # annual_composites renames positionally, so the two lists must be the same
    # length or bands get silently mislabelled.
    for reducer in composites.CENTRAL_REDUCERS:
        raw = composites.reduced_band_names("LST_C", reducer, 90)
        out = composites.composite_band_names("LST_C", 90, params)
        assert len(raw) == len(out) == 3


# --- zonal batching ----------------------------------------------------------------
def test_zonal_batch_size_is_configured(params: dict[str, Any]) -> None:
    assert params["composites"]["zonal_batch_years"] >= 1


def test_zonal_batching_covers_the_study_period_exactly() -> None:
    # Mirrors the loop in zonal_annual_means_by_year: every year built once,
    # none dropped, none duplicated - a batching bug would silently truncate
    # the series rather than fail.
    for batch in (1, 3, 4, 7, 26, 100):
        first, last = 2000, 2025
        spans = [
            (low, min(low + batch - 1, last))
            for low in range(first, last + 1, batch)
        ]
        covered = [year for low, high in spans for year in range(low, high + 1)]
        assert covered == list(range(first, last + 1)), batch
        assert all(low <= high for low, high in spans)


def test_zonal_by_year_rejects_a_zero_batch(params: dict[str, Any]) -> None:
    # Guard runs before any Earth Engine work; batch 0 would loop forever.
    with pytest.raises(ValueError, match="batch_years"):
        composites.zonal_annual_means_by_year(None, None, params, batch_years=0)


def test_zonal_annual_means_takes_no_batching_argument(
    params: dict[str, Any],
) -> None:
    # The single-request version deliberately has no batch_years: batching it
    # by ee.Filter was worthless, because filtering a collection built from
    # ee.List.map() materialises every element's graph anyway. Year selection
    # must happen at construction time - that is zonal_annual_means_by_year.
    import inspect

    assert "batch_years" not in inspect.signature(composites.zonal_annual_means).parameters
    by_year = inspect.signature(composites.zonal_annual_means_by_year).parameters
    assert "batch_years" in by_year
    # It builds composites itself, so it needs the compositing knobs too.
    for name in ("reducer", "months", "with_percentile", "mask"):
        assert name in by_year, name


def test_zonal_by_year_defaults_to_no_percentile() -> None:
    # A zonal mean never uses a percentile, and the sorted accumulator it needs
    # is the expensive part of a long reduction.
    import inspect

    default = inspect.signature(
        composites.zonal_annual_means_by_year
    ).parameters["with_percentile"].default
    assert default is False


def test_obs_count_band_is_always_present(params: dict[str, Any]) -> None:
    # CLAUDE.md caveat 2: no composite may exist without its observation count.
    for percentile in (90, None):
        names = composites.composite_band_names("LST_C", percentile, params)
        assert params["composites"]["obs_count_band"] in names


def test_band_names_drop_the_percentile_when_it_is_none(
    params: dict[str, Any],
) -> None:
    # with_percentile=False avoids the sorted accumulator a percentile reducer
    # needs, which is what keeps a 26-year series inside the EE memory limit.
    assert composites.reduced_band_names("LST_C", "mean", None) == [
        "LST_C_mean",
        "LST_C_count",
    ]
    assert composites.composite_band_names("LST_C", None, params) == [
        "LST_C",
        "obs_count",
    ]


def test_band_name_lists_stay_the_same_length(params: dict[str, Any]) -> None:
    # annual_composites renames positionally, so raw and output lists must match
    # in length whether or not the percentile band is produced.
    for percentile in (90, None):
        raw = composites.reduced_band_names("LST_C", "mean", percentile)
        out = composites.composite_band_names("LST_C", percentile, params)
        assert len(raw) == len(out)


def test_min_valid_obs_defaults_to_flag_only(params: dict[str, Any]) -> None:
    # User decision: emit the count, never drop pixels.
    assert params["composites"]["min_valid_obs"] is None


# --- dry_season_composites --------------------------------------------------------
def test_dry_season_rejects_an_explicit_months_argument(
    params: dict[str, Any],
) -> None:
    # It sets months from time.seasons.dry_window; accepting an override would
    # silently produce something that is not the dry season. The guard runs
    # before any Earth Engine import, so it is reachable here.
    with pytest.raises(TypeError, match="dry_window"):
        composites.dry_season_composites(None, params, months=[6, 7])


def test_dry_season_forwards_year_bounds_through_kwargs() -> None:
    # Notebook 02 relies on start_year/end_year reaching annual_composites, to
    # composite ONE year instead of building 26 and filtering down to one.
    import inspect

    signature = inspect.signature(composites.dry_season_composites)
    assert any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    annual = inspect.signature(composites.annual_composites).parameters
    assert "start_year" in annual and "end_year" in annual


# --- build_inventory_frame ----------------------------------------------------------
def test_inventory_frame_counts_by_year_and_sensor(params: dict[str, Any]) -> None:
    years = [2000, 2000, 2001, 2015, 2015, 2015]
    sensors = ["TM", "ETM+", "TM", "OLI/TIRS", "OLI/TIRS", "ETM+"]
    table = composites.build_inventory_frame(years, sensors, params)

    assert table.loc[2000, "TM"] == 1
    assert table.loc[2000, "ETM+"] == 1
    assert table.loc[2001, "TM"] == 1
    assert table.loc[2015, "OLI/TIRS"] == 2
    assert table.loc[2015, "total"] == 3


def test_inventory_frame_spans_the_full_study_period(params: dict[str, Any]) -> None:
    table = composites.build_inventory_frame([2015], ["OLI/TIRS"], params)
    assert table.index[0] == params["time"]["start_year"]
    assert table.index[-1] == params["time"]["end_year"]


def test_inventory_frame_shows_gap_years_as_explicit_zeros(
    params: dict[str, Any],
) -> None:
    # The whole point of the table: a year with no scenes must be a visible 0,
    # not an absent row that reads as "not checked".
    table = composites.build_inventory_frame([2015], ["OLI/TIRS"], params)
    assert table.loc[2001, "total"] == 0
    assert 2001 in table.index


def test_inventory_frame_honours_an_explicit_year_range(
    params: dict[str, Any],
) -> None:
    table = composites.build_inventory_frame(
        [2015], ["OLI/TIRS"], params, start_year=2014, end_year=2016
    )
    assert list(table.index) == [2014, 2015, 2016]


def test_inventory_frame_handles_an_empty_collection(params: dict[str, Any]) -> None:
    table = composites.build_inventory_frame([], [], params)
    assert table["total"].sum() == 0
    assert len(table) == (
        params["time"]["end_year"] - params["time"]["start_year"] + 1
    )


def test_inventory_frame_rejects_desynced_arrays(params: dict[str, Any]) -> None:
    # The failure this guards against: two aggregate_array results of unequal
    # length, which would otherwise pair each year with the wrong sensor.
    with pytest.raises(ValueError, match="parallel"):
        composites.build_inventory_frame([2000, 2001], ["TM"], params)


# --- build_zonal_frame ---------------------------------------------------------------
def test_zonal_frame_renames_reducer_columns(params: dict[str, Any]) -> None:
    rows = [
        {"year": 2001, "LST_C_mean": 30.5, "LST_C_count": 1200},
        {"year": 2000, "LST_C_mean": 29.8, "LST_C_count": 1100},
    ]
    frame = composites.build_zonal_frame(rows, "LST_C", params)

    assert list(frame.columns) == ["year", "mean", "valid_pixels"]
    # Sorted by year, not by input order.
    assert list(frame["year"]) == [2000, 2001]
    assert frame.loc[0, "mean"] == pytest.approx(29.8)


def test_zonal_frame_keeps_fully_masked_years_as_nan(params: dict[str, Any]) -> None:
    # reduceRegion returns the key with a null value for a fully-masked year.
    # The row must survive, so the year axis stays aligned with the others.
    rows = [
        {"year": 2000, "LST_C_mean": 29.8, "LST_C_count": 1100},
        {"year": 2001, "LST_C_mean": None, "LST_C_count": 0},
    ]
    frame = composites.build_zonal_frame(rows, "LST_C", params)

    assert len(frame) == 2
    assert frame.loc[1, "year"] == 2001
    assert frame["mean"].isna().sum() == 1


def test_zonal_frame_carries_a_valid_pixel_count(params: dict[str, Any]) -> None:
    # CLAUDE.md caveat 2 applied to tables, not just rasters.
    rows = [{"year": 2000, "LST_C_mean": 29.8, "LST_C_count": 1100}]
    frame = composites.build_zonal_frame(rows, "LST_C", params)
    assert frame.loc[0, "valid_pixels"] == 1100


def test_zonal_frame_raises_on_unexpected_column_names(
    params: dict[str, Any],
) -> None:
    # Better a loud error than an all-NaN table that looks like missing data.
    rows = [{"year": 2000, "LST_C": 29.8}]
    with pytest.raises(RuntimeError, match="LST_C_mean"):
        composites.build_zonal_frame(rows, "LST_C", params)


def test_zonal_frame_handles_no_rows(params: dict[str, Any]) -> None:
    frame = composites.build_zonal_frame([], "LST_C", params)
    assert frame.empty
    assert list(frame.columns) == ["year", "mean", "valid_pixels"]


def test_zonal_frame_uses_the_configured_year_property(
    params_copy: dict[str, Any],
) -> None:
    params_copy["composites"]["year_property"] = "yr"
    rows = [{"yr": 2000, "LST_C_mean": 29.8, "LST_C_count": 1100}]
    frame = composites.build_zonal_frame(rows, "LST_C", params_copy)
    assert list(frame.columns) == ["yr", "mean", "valid_pixels"]
