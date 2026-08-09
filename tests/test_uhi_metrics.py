"""Pin the pure-Python core of Phase 3: UTFVI classification, z-scores, tables.

No Earth Engine anywhere. Everything EE-dependent in ``uhi_metrics`` is verified
only by running notebook 03 in Colab; what is testable here is the arithmetic and
the reshaping, and that is exactly where the silent-wrong-answer risks live.

Three of these carry real risk and are pinned hard:

* **The UTFVI boundary convention.** The classifier and the server-side
  ``sum(gte)`` construction must agree on which class a value exactly equal to a
  break belongs to. If they ever disagree, a map and the table beside it would
  disagree by one class along every boundary, and nothing would error.
* **NaN handling in the classifier.** ``np.digitize`` sorts NaN into the TOP
  bucket, so a missing pixel would silently be reported as the "Worst" heat
  class — the most alarming class in the scheme, invented out of no data.
* **A constant input to the z-score.** Zero spread means every z-score is 0/0.
  Returning inf, or raising, would both be worse than the honest NaN.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from colombo_uhi import load_params, uhi_metrics


@pytest.fixture(scope="module")
def params() -> dict[str, Any]:
    return load_params()


@pytest.fixture()
def params_copy(params: dict[str, Any]) -> dict[str, Any]:
    """Deep copy for tests that mutate the config."""
    return copy.deepcopy(params)


# --- UTFVI scheme validation -------------------------------------------------
def test_utfvi_scheme_matches_claude_md(params: dict[str, Any]) -> None:
    breaks, labels = uhi_metrics.validate_utfvi_scheme(params)
    assert breaks == [0.0, 0.005, 0.010, 0.015, 0.020]
    assert labels == ["Excellent", "Good", "Normal", "Bad", "Worse", "Worst"]
    assert len(labels) == len(breaks) + 1


def test_utfvi_scheme_rejects_non_increasing_breaks(
    params_copy: dict[str, Any],
) -> None:
    params_copy["uhi"]["utfvi"]["breaks"] = [0.0, 0.010, 0.005, 0.015, 0.020]
    with pytest.raises(ValueError, match="strictly increasing"):
        uhi_metrics.validate_utfvi_scheme(params_copy)


def test_utfvi_scheme_rejects_duplicate_breaks(params_copy: dict[str, Any]) -> None:
    params_copy["uhi"]["utfvi"]["breaks"] = [0.0, 0.005, 0.005, 0.015, 0.020]
    with pytest.raises(ValueError, match="strictly increasing"):
        uhi_metrics.validate_utfvi_scheme(params_copy)


def test_utfvi_scheme_rejects_wrong_label_count(params_copy: dict[str, Any]) -> None:
    params_copy["uhi"]["utfvi"]["labels"] = ["Excellent", "Good", "Normal"]
    with pytest.raises(ValueError, match="one more label than break"):
        uhi_metrics.validate_utfvi_scheme(params_copy)


# --- UTFVI formula -----------------------------------------------------------
def test_utfvi_formula(params: dict[str, Any]) -> None:
    # (Ts - Tmean) / Tmean, with a mean that makes the arithmetic checkable
    values = uhi_metrics.utfvi_from_arrays([30.0, 31.0, 29.0], 30.0)
    assert values == pytest.approx([0.0, 1.0 / 30.0, -1.0 / 30.0])


def test_utfvi_formula_preserves_nan() -> None:
    values = uhi_metrics.utfvi_from_arrays([30.0, np.nan], 30.0)
    assert values[0] == pytest.approx(0.0)
    assert np.isnan(values[1])


def test_utfvi_formula_rejects_zero_mean() -> None:
    # Physically reachable on the Celsius scale, unlike on Kelvin — a real guard.
    with pytest.raises(ValueError, match="t_mean == 0"):
        uhi_metrics.utfvi_from_arrays([1.0, 2.0], 0.0)


# --- UTFVI classifier --------------------------------------------------------
def test_utfvi_classifier_reaches_every_class(params: dict[str, Any]) -> None:
    # One value comfortably inside each of the six classes.
    values = [-0.01, 0.002, 0.007, 0.012, 0.017, 0.030]
    classes = uhi_metrics.utfvi_class_indices(values, params)
    assert list(classes) == [0, 1, 2, 3, 4, 5]


def test_utfvi_classifier_boundaries_are_left_closed(params: dict[str, Any]) -> None:
    # A value EXACTLY on a break belongs to the class ABOVE it. This is the
    # convention the server-side sum(gte) construction produces, and the two must
    # not drift apart: a mismatch would put every boundary pixel in the wrong
    # class on the map while the table said otherwise, with no error anywhere.
    breaks, _ = uhi_metrics.validate_utfvi_scheme(params)
    classes = uhi_metrics.utfvi_class_indices(breaks, params)
    assert list(classes) == [1, 2, 3, 4, 5]


def test_utfvi_classifier_matches_a_sum_of_gte_tests(params: dict[str, Any]) -> None:
    # The exact arithmetic utfvi_class_image() performs server-side, reproduced
    # in numpy. If this test fails, the map and the table have diverged.
    breaks, _ = uhi_metrics.validate_utfvi_scheme(params)
    values = np.array([-0.05, 0.0, 0.004, 0.005, 0.0149, 0.015, 0.02, 0.5])
    expected = sum((values >= threshold).astype("int64") for threshold in breaks)
    classes = uhi_metrics.utfvi_class_indices(values, params)
    assert list(classes) == list(expected)


def test_utfvi_classifier_saturates_at_both_ends(params: dict[str, Any]) -> None:
    classes = uhi_metrics.utfvi_class_indices([-1e6, 1e6], params)
    assert list(classes) == [0, 5]


def test_utfvi_classifier_maps_nan_to_the_nodata_class(
    params: dict[str, Any],
) -> None:
    # np.digitize buckets NaN at the TOP, which would report missing data as the
    # "Worst" heat class. utfvi_class_indices must override that explicitly.
    classes = uhi_metrics.utfvi_class_indices([np.nan, 0.03, np.nan], params)
    assert classes[0] == uhi_metrics.UTFVI_NODATA_CLASS == -1
    assert classes[1] == 5
    assert classes[2] == -1


def test_utfvi_class_labels_are_in_class_order(params: dict[str, Any]) -> None:
    labels = uhi_metrics.utfvi_class_labels(params)
    assert labels[0] == "Excellent"
    assert labels[-1] == "Worst"


# --- sigma / ddof resolution -------------------------------------------------
def test_resolve_sigma_defaults_to_one(params: dict[str, Any]) -> None:
    assert uhi_metrics.resolve_sigma(None, params) == 1.0


def test_resolve_sigma_accepts_the_configured_options(
    params: dict[str, Any],
) -> None:
    for option in params["uhi"]["zscore"]["sigma_options"]:
        assert uhi_metrics.resolve_sigma(option, params) == float(option)


def test_resolve_sigma_warns_on_an_unlisted_threshold(
    params: dict[str, Any],
) -> None:
    # Allowed (sensitivity runs need it) but never silent.
    with pytest.warns(UserWarning, match="not one of the reported thresholds"):
        assert uhi_metrics.resolve_sigma(1.5, params) == 1.5


@pytest.mark.parametrize("bad", [0.0, -1.0, -0.5])
def test_resolve_sigma_rejects_non_positive(params: dict[str, Any], bad: float) -> None:
    with pytest.raises(ValueError, match="sigma must be > 0"):
        uhi_metrics.resolve_sigma(bad, params)


def test_resolve_ddof_defaults_from_params(params: dict[str, Any]) -> None:
    assert uhi_metrics.resolve_ddof(None, params) == params["uhi"]["zscore"]["ddof"]


def test_resolve_ddof_rejects_negative(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="ddof must be >= 0"):
        uhi_metrics.resolve_ddof(-1, params)


# --- z-scores ----------------------------------------------------------------
def test_zscore_on_a_known_array(params: dict[str, Any]) -> None:
    # mean 30, population sd 2 -> z of [-1, 0, 1] at the corresponding values
    values = [28.0, 30.0, 32.0]
    scores = uhi_metrics.zscore_array(values, params, ddof=0)
    assert scores == pytest.approx([-np.sqrt(1.5), 0.0, np.sqrt(1.5)])


def test_zscore_is_standardised(params: dict[str, Any]) -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(30.0, 3.0, size=500)
    scores = uhi_metrics.zscore_array(values, params, ddof=0)
    assert float(np.mean(scores)) == pytest.approx(0.0, abs=1e-9)
    assert float(np.std(scores, ddof=0)) == pytest.approx(1.0, abs=1e-9)


def test_zscore_of_a_constant_array_is_all_nan(params: dict[str, Any]) -> None:
    # Zero spread means every score is 0/0. NaN is the honest answer: "no pixel
    # is anomalous" is a claim about spread that a constant array cannot support.
    # Returning inf, or raising, would both be worse.
    scores = uhi_metrics.zscore_array([25.0] * 8, params)
    assert np.isnan(scores).all()
    assert not np.isinf(scores).any()


def test_zscore_ignores_nan_but_preserves_its_position(
    params: dict[str, Any],
) -> None:
    scores = uhi_metrics.zscore_array([28.0, np.nan, 30.0, 32.0], params, ddof=0)
    assert np.isnan(scores[1])
    # The mean and sd come from the three finite values only.
    finite = uhi_metrics.zscore_array([28.0, 30.0, 32.0], params, ddof=0)
    assert scores[[0, 2, 3]] == pytest.approx(finite)


def test_zscore_of_an_all_nan_array_is_all_nan(params: dict[str, Any]) -> None:
    scores = uhi_metrics.zscore_array([np.nan, np.nan], params)
    assert np.isnan(scores).all()


def test_zscore_of_an_empty_array_is_empty(params: dict[str, Any]) -> None:
    assert uhi_metrics.zscore_array([], params).size == 0


def test_zscore_ddof_changes_the_answer_on_small_n(params: dict[str, Any]) -> None:
    # Documents WHY uhi.zscore.ddof has to be settled against ee.Reducer.stdDev
    # rather than guessed: on small n the two conventions differ visibly.
    values = [10.0, 12.0, 14.0, 16.0]
    population = uhi_metrics.zscore_array(values, params, ddof=0)
    sample = uhi_metrics.zscore_array(values, params, ddof=1)
    assert not np.allclose(population, sample)
    # ddof=1 divides by a smaller n, so the spread is larger and the scores smaller
    assert abs(sample[0]) < abs(population[0])


# --- hot-pixel flags ---------------------------------------------------------
def test_hot_pixel_flags_at_one_and_two_sigma(params: dict[str, Any]) -> None:
    # z = [-1.5, -0.5, 0.5, 1.5] for this array under ddof=0
    values = np.array([25.0, 27.0, 29.0, 31.0])
    scores = uhi_metrics.zscore_array(values, params, ddof=0)
    assert scores == pytest.approx([-1.34164, -0.44721, 0.44721, 1.34164], abs=1e-4)

    one_sigma = uhi_metrics.hot_pixel_flags(values, params, sigma=1.0)
    two_sigma = uhi_metrics.hot_pixel_flags(values, params, sigma=2.0)
    assert list(one_sigma) == [False, False, False, True]
    assert list(two_sigma) == [False, False, False, False]


def test_hot_pixel_flags_include_the_threshold_itself(
    params: dict[str, Any],
) -> None:
    # gte, matching the server-side comparison.
    values = np.array([28.0, 30.0, 32.0])
    scores = uhi_metrics.zscore_array(values, params, ddof=0)
    # An off-list sigma, so the unlisted-threshold warning is expected here too.
    with pytest.warns(UserWarning, match="not one of the reported thresholds"):
        flags = uhi_metrics.hot_pixel_flags(values, params, sigma=float(scores[2]))
    assert bool(flags[2]) is True


def test_hot_pixel_flags_never_flag_missing_data(params: dict[str, Any]) -> None:
    flags = uhi_metrics.hot_pixel_flags([25.0, np.nan, 40.0], params)
    assert bool(flags[1]) is False


def test_hot_pixel_flags_of_a_constant_array_are_all_false(
    params: dict[str, Any],
) -> None:
    flags = uhi_metrics.hot_pixel_flags([30.0] * 5, params)
    assert not flags.any()


# --- method resolution -------------------------------------------------------
def test_resolve_methods_defaults_to_both_definitions(params: dict[str, Any]) -> None:
    assert uhi_metrics.resolve_methods(None, params) == ["buffer_ring", "lcz_based"]


def test_resolve_methods_collapses_duplicates(params: dict[str, Any]) -> None:
    assert uhi_metrics.resolve_methods(
        ["lcz_based", "lcz_based"], params
    ) == ["lcz_based"]


def test_resolve_methods_rejects_an_empty_list(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="at least one"):
        uhi_metrics.resolve_methods([], params)


def test_resolve_methods_rejects_an_unknown_method(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        uhi_metrics.resolve_methods(["ring_buffer"], params)


# --- source resolution -------------------------------------------------------
def test_every_configured_source_resolves(params: dict[str, Any]) -> None:
    for entry in params["uhi"]["suhii"]["sources"]:
        resolved = uhi_metrics.resolve_source(entry["key"], params)
        assert resolved["key"] == entry["key"]
        assert resolved["kind"] in ("landsat", "modis")


def test_resolve_source_rejects_an_unknown_key(params: dict[str, Any]) -> None:
    with pytest.raises(KeyError, match="unknown SUHII source"):
        uhi_metrics.resolve_source("sentinel_day", params)


def test_resolve_source_requires_product_for_modis() -> None:
    with pytest.raises(ValueError, match="missing 'product'"):
        uhi_metrics.resolve_source(
            {"key": "x", "kind": "modis", "reducer": "mean", "scale_m": 1000},
            {"uhi": {"suhii": {"sources": []}}},
        )


def test_resolve_source_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown SUHII source kind"):
        uhi_metrics.resolve_source(
            {"key": "x", "kind": "sentinel", "reducer": "mean", "scale_m": 30},
            {"uhi": {"suhii": {"sources": []}}},
        )


def test_landsat_source_uses_the_dry_window(params: dict[str, Any]) -> None:
    source = uhi_metrics.resolve_source("landsat_dry", params)
    assert uhi_metrics.source_months(source, params) == [1, 2, 3]


def test_modis_sources_have_no_month_restriction(params: dict[str, Any]) -> None:
    source = uhi_metrics.resolve_source("terra_night", params)
    assert uhi_metrics.source_months(source, params) is None


# --- build_suhii_frame -------------------------------------------------------
def _suhii_row(year: int, source: str, **overrides: float) -> dict[str, Any]:
    """One wide reduceRegion row covering both rural definitions."""
    row: dict[str, Any] = {
        "year": year,
        "source": source,
        "buffer_ring_urban_mean": 34.0,
        "buffer_ring_urban_count": 4200.0,
        "buffer_ring_rural_mean": 31.0,
        "buffer_ring_rural_count": 18000.0,
        "lcz_based_urban_mean": 33.5,
        "lcz_based_urban_count": 50000.0,
        "lcz_based_rural_mean": 32.0,
        "lcz_based_rural_count": 16000.0,
    }
    row.update(overrides)
    return row


def test_suhii_frame_has_the_documented_columns(params: dict[str, Any]) -> None:
    frame = uhi_metrics.build_suhii_frame(
        [_suhii_row(2000, "terra_day")], ["buffer_ring", "lcz_based"], params
    )
    assert list(frame.columns) == list(uhi_metrics.SUHII_COLUMNS)


def test_suhii_frame_unpivots_one_row_per_method(params: dict[str, Any]) -> None:
    frame = uhi_metrics.build_suhii_frame(
        [_suhii_row(2000, "terra_day"), _suhii_row(2001, "terra_day")],
        ["buffer_ring", "lcz_based"],
        params,
    )
    assert len(frame) == 4
    assert sorted(frame["rural_definition"].unique()) == ["buffer_ring", "lcz_based"]


def test_suhii_is_urban_minus_rural(params: dict[str, Any]) -> None:
    frame = uhi_metrics.build_suhii_frame(
        [_suhii_row(2000, "terra_day")], ["buffer_ring", "lcz_based"], params
    )
    buffer_row = frame[frame["rural_definition"] == "buffer_ring"].iloc[0]
    lcz_row = frame[frame["rural_definition"] == "lcz_based"].iloc[0]
    assert buffer_row["suhii"] == pytest.approx(3.0)
    assert lcz_row["suhii"] == pytest.approx(1.5)


def test_suhii_is_nan_when_either_side_is_missing(params: dict[str, Any]) -> None:
    # A year with no valid urban pixels has NO SUHII. It must not be reported as
    # one, and it must not silently become the rural mean's negative.
    rows = [
        _suhii_row(
            2000,
            "terra_day",
            buffer_ring_urban_mean=None,
            buffer_ring_urban_count=0.0,
        )
    ]
    # A zero-pixel year also trips the caveat-2 warning; that is the point.
    with pytest.warns(UserWarning, match="COMPLETELY EMPTY"):
        frame = uhi_metrics.build_suhii_frame(rows, ["buffer_ring"], params)
    assert np.isnan(frame.iloc[0]["suhii"])
    assert frame.iloc[0]["urban_pixels"] == 0.0


def test_suhii_frame_warns_when_a_series_is_completely_empty(
    params: dict[str, Any],
) -> None:
    # The Colab run 5 defect, transplanted: a table of Nones that looks plausible
    # until someone checks the count column.
    rows = [
        _suhii_row(
            year,
            "terra_night",
            buffer_ring_urban_mean=None,
            buffer_ring_urban_count=0.0,
        )
        for year in (2000, 2001, 2002)
    ]
    with pytest.warns(UserWarning, match="COMPLETELY EMPTY"):
        uhi_metrics.build_suhii_frame(rows, ["buffer_ring"], params)


def test_suhii_frame_raises_when_a_method_is_absent(params: dict[str, Any]) -> None:
    with pytest.raises(RuntimeError, match="lcz_based"):
        uhi_metrics.build_suhii_frame(
            [{"year": 2000, "source": "terra_day"}], ["lcz_based"], params
        )


def test_suhii_frame_of_no_rows_is_empty_but_shaped(params: dict[str, Any]) -> None:
    frame = uhi_metrics.build_suhii_frame([], ["buffer_ring"], params)
    assert frame.empty
    assert list(frame.columns) == list(uhi_metrics.SUHII_COLUMNS)


def test_suhii_frame_is_sorted_by_source_then_method_then_year(
    params: dict[str, Any],
) -> None:
    rows = [_suhii_row(2001, "terra_day"), _suhii_row(2000, "aqua_day")]
    frame = uhi_metrics.build_suhii_frame(rows, ["buffer_ring"], params)
    assert list(frame["source"]) == ["aqua_day", "terra_day"]


# --- build_class_share_frame -------------------------------------------------
def test_class_shares_sum_to_one_hundred(params: dict[str, Any]) -> None:
    rows = [{"year": 2020, "histogram": {"0": 25, "1": 25, "5": 50}}]
    frame = uhi_metrics.build_class_share_frame(rows, params)
    labels = uhi_metrics.utfvi_class_labels(params)
    assert float(frame[labels].sum(axis=1).iloc[0]) == pytest.approx(100.0)
    assert frame.iloc[0]["Excellent"] == pytest.approx(25.0)
    assert frame.iloc[0]["Worst"] == pytest.approx(50.0)


def test_class_shares_fill_absent_classes_with_zero(params: dict[str, Any]) -> None:
    # Earth Engine's frequencyHistogram omits classes with no pixels entirely.
    rows = [{"year": 2020, "histogram": {"0": 10}}]
    frame = uhi_metrics.build_class_share_frame(rows, params)
    assert frame.iloc[0]["Excellent"] == pytest.approx(100.0)
    assert frame.iloc[0]["Worst"] == pytest.approx(0.0)


def test_class_shares_of_an_empty_year_are_nan_not_zero(
    params: dict[str, Any],
) -> None:
    # A year with no classified pixels is missing data, not "0% everywhere".
    rows = [{"year": 2020, "histogram": {}}]
    frame = uhi_metrics.build_class_share_frame(rows, params)
    labels = uhi_metrics.utfvi_class_labels(params)
    assert frame[labels].iloc[0].isna().all()
    assert frame.iloc[0][uhi_metrics.PIXEL_COUNT_COLUMN] == 0


def test_class_shares_carry_the_pixel_count(params: dict[str, Any]) -> None:
    rows = [{"year": 2020, "histogram": {"2": 30, "3": 70}}]
    frame = uhi_metrics.build_class_share_frame(rows, params)
    assert frame.iloc[0][uhi_metrics.PIXEL_COUNT_COLUMN] == 100


def test_class_shares_are_sorted_by_year(params: dict[str, Any]) -> None:
    rows = [
        {"year": 2021, "histogram": {"0": 1}},
        {"year": 2019, "histogram": {"0": 1}},
    ]
    frame = uhi_metrics.build_class_share_frame(rows, params)
    assert list(frame["year"]) == [2019, 2021]


def test_class_shares_of_no_rows_is_empty_but_shaped(params: dict[str, Any]) -> None:
    frame = uhi_metrics.build_class_share_frame([], params)
    labels = uhi_metrics.utfvi_class_labels(params)
    assert frame.empty
    assert list(frame.columns) == ["year", *labels, uhi_metrics.PIXEL_COUNT_COLUMN]


# --- build_division_frame ----------------------------------------------------
def _division_feature(pcode: str, name: str, **stats: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "adm4_pcode": pcode,
        "adm4_name": name,
        "adm3_name": "Thimbirigasyaya",
        "LST_C_mean": 33.0,
        "LST_C_median": 32.8,
        "LST_C_stdDev": 1.4,
        "LST_C_count": 900,
    }
    row.update(stats)
    return row


def test_division_frame_keeps_the_pcode_as_well_as_the_name(
    params: dict[str, Any],
) -> None:
    # CLAUDE.md: GN names are NOT unique within Colombo District, so the pcode is
    # the key and the name is for reading only. Losing the pcode here would make
    # every downstream join in Phases 5 and 7 quietly wrong.
    frame = uhi_metrics.build_division_frame(
        [_division_feature("LK1103005", "Kirula")],
        params,
        "gn",
        "LST_C",
        ["mean", "median", "stdDev"],
    )
    assert "adm4_pcode" in frame.columns
    assert "adm4_name" in frame.columns
    assert list(frame.columns)[0] == "adm4_pcode"


def test_division_frame_renames_stats_and_the_count(params: dict[str, Any]) -> None:
    frame = uhi_metrics.build_division_frame(
        [_division_feature("LK1103005", "Kirula")],
        params,
        "gn",
        "LST_C",
        ["mean", "median", "stdDev"],
    )
    for column in ("mean", "median", "stdDev", uhi_metrics.PIXEL_COUNT_COLUMN):
        assert column in frame.columns
    assert frame.iloc[0]["mean"] == pytest.approx(33.0)
    assert frame.iloc[0][uhi_metrics.PIXEL_COUNT_COLUMN] == 900


def test_division_frame_accepts_bare_reducer_names(params: dict[str, Any]) -> None:
    # reduceRegions names outputs bare for a single-band image and band-prefixed
    # otherwise; both spellings must work or the helper breaks on band count.
    feature = {
        "adm4_pcode": "LK1103005",
        "adm4_name": "Kirula",
        "mean": 33.0,
        "count": 900,
    }
    frame = uhi_metrics.build_division_frame(
        [feature], params, "gn", "LST_C", ["mean"]
    )
    assert frame.iloc[0]["mean"] == pytest.approx(33.0)


def test_division_frame_raises_on_a_missing_statistic(
    params: dict[str, Any],
) -> None:
    feature = {"adm4_pcode": "LK1103005", "LST_C_mean": 33.0, "LST_C_count": 900}
    with pytest.raises(RuntimeError, match="stdDev"):
        uhi_metrics.build_division_frame(
            [feature], params, "gn", "LST_C", ["mean", "stdDev"]
        )


def test_division_frame_rejects_an_unknown_level(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="level must be"):
        uhi_metrics.build_division_frame(
            [_division_feature("LK1103005", "Kirula")],
            params,
            "province",
            "LST_C",
            ["mean"],
        )


def test_division_frame_of_no_features_is_empty(params: dict[str, Any]) -> None:
    frame = uhi_metrics.build_division_frame([], params, "gn", "LST_C", ["mean"])
    assert frame.empty


# --- fit_drivers -------------------------------------------------------------
def _driver_sample(n: int = 400, noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Synthetic pixels with a KNOWN relationship, so coefficients are checkable."""
    rng = np.random.default_rng(seed)
    ndvi = rng.uniform(0.0, 0.8, size=n)
    ndbi = rng.uniform(-0.3, 0.4, size=n)
    mndwi = rng.uniform(-0.6, 0.2, size=n)
    built = rng.uniform(0.0, 1.0, size=n)
    lst = (
        30.0
        - 6.0 * ndvi
        + 4.0 * ndbi
        - 1.0 * mndwi
        + 2.0 * built
        + rng.normal(0.0, noise, size=n) if noise else
        30.0 - 6.0 * ndvi + 4.0 * ndbi - 1.0 * mndwi + 2.0 * built
    )
    return pd.DataFrame(
        {
            "LST_C": lst,
            "NDVI": ndvi,
            "NDBI": ndbi,
            "MNDWI": mndwi,
            "built_fraction": built,
        }
    )


def test_fit_drivers_recovers_known_coefficients(params: dict[str, Any]) -> None:
    result = uhi_metrics.fit_drivers(_driver_sample(), params)
    coefficients = dict(zip(result["term"], result["coefficient"]))
    assert coefficients["const"] == pytest.approx(30.0, abs=1e-6)
    assert coefficients["NDVI"] == pytest.approx(-6.0, abs=1e-6)
    assert coefficients["NDBI"] == pytest.approx(4.0, abs=1e-6)
    assert coefficients["MNDWI"] == pytest.approx(-1.0, abs=1e-6)
    assert coefficients["built_fraction"] == pytest.approx(2.0, abs=1e-6)


def test_fit_drivers_reports_a_perfect_fit_as_such(params: dict[str, Any]) -> None:
    result = uhi_metrics.fit_drivers(_driver_sample(), params)
    assert float(result["r_squared"].iloc[0]) == pytest.approx(1.0, abs=1e-9)
    assert int(result["n_obs"].iloc[0]) == 400


def test_fit_drivers_with_noise_stays_close_and_significant(
    params: dict[str, Any],
) -> None:
    result = uhi_metrics.fit_drivers(_driver_sample(noise=0.5), params)
    coefficients = dict(zip(result["term"], result["coefficient"]))
    p_values = dict(zip(result["term"], result["p_value"]))
    assert coefficients["NDVI"] == pytest.approx(-6.0, abs=0.3)
    assert coefficients["NDBI"] == pytest.approx(4.0, abs=0.3)
    assert p_values["NDVI"] < 1e-6
    assert 0.9 < float(result["r_squared"].iloc[0]) <= 1.0


def test_fit_drivers_emits_the_documented_columns(params: dict[str, Any]) -> None:
    result = uhi_metrics.fit_drivers(_driver_sample(noise=0.4), params)
    for column in (
        "term",
        "coefficient",
        "std_err",
        "t_stat",
        "p_value",
        "r_squared",
        "adj_r_squared",
        "n_obs",
    ):
        assert column in result.columns
    assert result["term"].iloc[0] == "const"


def test_fit_drivers_refuses_too_few_rows(params: dict[str, Any]) -> None:
    # An R-squared from a handful of pixels is not a result; refusing beats
    # returning a number somebody will quote.
    minimum = params["uhi"]["drivers"]["min_sample_rows"]
    with pytest.raises(ValueError, match="min_sample_rows"):
        uhi_metrics.fit_drivers(_driver_sample(n=minimum - 1), params)


def test_fit_drivers_drops_a_constant_predictor_with_a_warning(
    params: dict[str, Any],
) -> None:
    # Left in, it makes the design matrix singular and statsmodels returns a NaN
    # row that reads exactly like a computed coefficient.
    sample = _driver_sample(noise=0.3)
    sample["MNDWI"] = -0.2
    with pytest.warns(UserWarning, match="'MNDWI' is constant"):
        result = uhi_metrics.fit_drivers(sample, params)
    assert "MNDWI" not in list(result["term"])
    assert not result["coefficient"].isna().any()


@pytest.mark.parametrize("constant", [-0.2, 0.0, 1.0, 0.1, 33.0])
def test_fit_drivers_detects_constancy_by_distinct_values_not_variance(
    params: dict[str, Any], constant: float
) -> None:
    # Regression. The guard used to be `std(ddof=0) == 0.0`, but pandas does not
    # reliably return exactly zero for a constant column — depending on how the
    # frame was constructed it can come back as 2.8e-17, which passes `== 0` and
    # lets a singular column into the design matrix. Counting distinct values is
    # exact. Parametrised over several constants because the floating-point
    # residue depends on the value.
    sample = _driver_sample(noise=0.3)
    sample["MNDWI"] = constant
    with pytest.warns(UserWarning, match="'MNDWI' is constant"):
        result = uhi_metrics.fit_drivers(sample, params)
    assert "MNDWI" not in list(result["term"])


@pytest.mark.parametrize("constant", [-0.2, 0.0, 33.0])
def test_driver_correlations_detect_constancy_the_same_way(
    params: dict[str, Any], constant: float
) -> None:
    sample = _driver_sample(noise=0.3)
    sample["MNDWI"] = constant
    result = uhi_metrics.driver_correlations(sample, params)
    row = result[result["predictor"] == "MNDWI"].iloc[0]
    assert np.isnan(row["pearson_r"]), "a constant predictor has no correlation"


def test_fit_drivers_rejects_a_missing_column(params: dict[str, Any]) -> None:
    sample = _driver_sample().drop(columns=["NDBI"])
    with pytest.raises(ValueError, match="missing column"):
        uhi_metrics.fit_drivers(sample, params)


def test_fit_drivers_ignores_incomplete_rows(params: dict[str, Any]) -> None:
    sample = _driver_sample(noise=0.2)
    sample.loc[:19, "NDVI"] = np.nan
    result = uhi_metrics.fit_drivers(sample, params)
    assert int(result["n_obs"].iloc[0]) == 380


# --- driver_correlations -----------------------------------------------------
def test_driver_correlations_recover_the_expected_signs(
    params: dict[str, Any],
) -> None:
    result = uhi_metrics.driver_correlations(_driver_sample(noise=0.2), params)
    by_predictor = dict(zip(result["predictor"], result["pearson_r"]))
    # Greener is cooler, more built-up is hotter — the whole premise of the study.
    assert by_predictor["NDVI"] < 0
    assert by_predictor["NDBI"] > 0
    assert by_predictor["built_fraction"] > 0


def test_driver_correlations_cover_every_predictor(params: dict[str, Any]) -> None:
    result = uhi_metrics.driver_correlations(_driver_sample(noise=0.2), params)
    assert list(result["predictor"]) == params["uhi"]["drivers"]["predictors"]
    assert list(result.columns) == ["predictor", "pearson_r", "p_value", "n_obs"]


def test_driver_correlations_return_nan_for_a_constant_predictor(
    params: dict[str, Any],
) -> None:
    # One degenerate index must not destroy the whole year's table.
    sample = _driver_sample(noise=0.2)
    sample["MNDWI"] = -0.2
    result = uhi_metrics.driver_correlations(sample, params)
    row = result[result["predictor"] == "MNDWI"].iloc[0]
    assert np.isnan(row["pearson_r"])
    assert not result[result["predictor"] == "NDVI"]["pearson_r"].isna().any()


def test_driver_correlations_reject_a_missing_column(params: dict[str, Any]) -> None:
    sample = _driver_sample().drop(columns=["MNDWI"])
    with pytest.raises(ValueError, match="missing column"):
        uhi_metrics.driver_correlations(sample, params)


# --- predictors and epochs ---------------------------------------------------
def test_resolve_predictors_defaults_from_params(params: dict[str, Any]) -> None:
    assert uhi_metrics.resolve_predictors(None, params) == [
        "NDVI",
        "NDBI",
        "MNDWI",
        "built_fraction",
    ]


def test_resolve_predictors_rejects_an_empty_list(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="at least one"):
        uhi_metrics.resolve_predictors([], params)


def test_epoch_years_resolve(params: dict[str, Any]) -> None:
    assert uhi_metrics.epoch_years(params, "2000s") == (2000, 2009)
    assert uhi_metrics.epoch_years(params, "2020s") == (2020, 2025)


def test_epoch_years_reject_an_unknown_epoch(params: dict[str, Any]) -> None:
    with pytest.raises(KeyError, match="unknown epoch"):
        uhi_metrics.epoch_years(params, "1990s")


def test_epoch_years_reject_an_inverted_range(params_copy: dict[str, Any]) -> None:
    params_copy["uhi"]["utfvi"]["epochs"]["2010s"] = [2019, 2010]
    with pytest.raises(ValueError, match="end_year"):
        uhi_metrics.epoch_years(params_copy, "2010s")


# --- import hygiene ----------------------------------------------------------
def test_uhi_metrics_imports_with_earthengine_unavailable(
    monkeypatch: pytest.MonkeyPatch, params: dict[str, Any]
) -> None:
    # The deferred-import rule, actually enforced. earthengine-api happens to be
    # installed in this environment, so merely importing the module proves
    # nothing — `import ee` has to be BLOCKED for the test to mean anything.
    # Colab always has it; a marker's laptop and CI may not.
    import builtins
    import importlib
    import sys

    real_import = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "ee" or name.startswith("ee."):
            raise ImportError("earthengine-api is not installed (simulated)")
        return real_import(name, *args, **kwargs)

    for cached in [
        m for m in list(sys.modules)
        if m == "colombo_uhi" or m.startswith("colombo_uhi.")
    ]:
        monkeypatch.delitem(sys.modules, cached, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)

    module = importlib.import_module("colombo_uhi.uhi_metrics")

    # The pure surface must be fully usable with no Earth Engine at all.
    assert module.utfvi_class_labels(params)[0] == "Excellent"
    assert list(module.utfvi_class_indices([-0.01, 0.03], params)) == [0, 5]
    assert module.resolve_sigma(None, params) == 1.0
    frame = module.build_suhii_frame(
        [_suhii_row(2000, "terra_day")], ["buffer_ring"], params
    )
    assert frame.iloc[0]["suhii"] == pytest.approx(3.0)


def test_earthengine_backed_functions_fail_loudly_without_it(
    monkeypatch: pytest.MonkeyPatch, params: dict[str, Any]
) -> None:
    # The other half of the contract: deferring the import must not turn a
    # missing dependency into a silent no-op. A function that genuinely needs
    # Earth Engine has to raise when it is absent.
    import builtins
    import importlib
    import sys

    real_import = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "ee" or name.startswith("ee."):
            raise ImportError("earthengine-api is not installed (simulated)")
        return real_import(name, *args, **kwargs)

    for cached in [
        m for m in list(sys.modules)
        if m == "colombo_uhi" or m.startswith("colombo_uhi.")
    ]:
        monkeypatch.delitem(sys.modules, cached, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)

    module = importlib.import_module("colombo_uhi.uhi_metrics")
    with pytest.raises(ImportError):
        module.built_up_fraction(params, 2020)
