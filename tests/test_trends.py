"""Pin the pure-Python core of Phase 4: MK arithmetic, FDR, and the series guard.

No Earth Engine anywhere. Everything EE-dependent in ``trends`` is verified only
by running notebook 04 in Colab; what is testable here is the arithmetic, the
reshaping and the guard logic, and that is exactly where the silent-wrong-answer
risks live.

Four things carry real risk and are therefore pinned hard:

1. **The community tutorial's sign function truncates floats to zero.** Its
   ``clamp(-1, 1).int()`` maps a +0.3 degC year-to-year difference to 0. The
   tests below write that buggy reference out explicitly and assert our answer
   differs, so nobody "simplifies" the implementation back into the bug.
2. **The tutorial's p-value is one-sided.** Feeding the one-sided form to
   Benjamini-Hochberg halves every p and roughly doubles the area reported as
   significantly warming. The p-value tests assert two-sidedness AND assert
   inequality with the one-sided form.
3. **NaN accounting in FDR.** A pixel that was never tested is not a
   non-significant result. NaNs must be excluded from the test count ``m`` and
   must never come back as significant.
4. **The structural guard.** ``validate_series_metadata`` is the only thing
   standing between a user and a Mann-Kendall test fitted to 1674 irregularly
   spaced scenes, so every rejection path has its own test.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any

import numpy as np
import pytest

from colombo_uhi import load_params, trends


@pytest.fixture(scope="module")
def params() -> dict[str, Any]:
    return load_params()


@pytest.fixture()
def params_copy(params: dict[str, Any]) -> dict[str, Any]:
    """Deep copy for tests that mutate the config."""
    return copy.deepcopy(params)


def _annual_metadata(
    years: list[int],
    params: dict[str, Any],
    months: list[int] | None = None,
    n_scenes: list[int] | None = None,
    basis: list[str] | None = None,
    size: int | None = None,
    probed: int | None = None,
) -> dict[str, Any]:
    """A well-formed metadata payload, as require_annual_series would fetch it.

    ``size`` is the whole collection; ``probed`` is how many images had their
    properties read. They differ whenever the guard samples rather than
    inspecting every image.
    """
    count = len(years) if size is None else size
    return {
        "size": count,
        "probed": len(years) if probed is None else probed,
        "series_basis": (
            basis
            if basis is not None
            else [params["trends"]["series_basis"]] * len(years)
        ),
        "years": years,
        "months": months or [],
        "n_scenes": n_scenes if n_scenes is not None else [5] * len(years),
    }


# --- parameter resolution ----------------------------------------------------
def test_resolve_alpha_defaults_to_the_configured_level(params: dict[str, Any]) -> None:
    assert trends.resolve_alpha(None, params) == params["trends"]["fdr"]["alpha"]


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5])
def test_resolve_alpha_rejects_degenerate_levels(
    params: dict[str, Any], value: float
) -> None:
    # At 0 no pixel can ever be significant and at 1 every pixel is; either turns
    # the significance map into a constant without anything looking wrong.
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        trends.resolve_alpha(value, params)


def test_resolve_fdr_method_rejects_an_unknown_method(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="benjamini_hochberg"):
        trends.resolve_fdr_method("bonferroni", params)


def test_resolve_mk_method_rejects_an_unknown_route(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="tau_derived"):
        trends.resolve_mk_method("theil_sen", params)


def test_resolve_min_years_rejects_fewer_than_three(params: dict[str, Any]) -> None:
    # Var(S) and the normal approximation behind Z are not defined on 2 points,
    # so an unmasked Z there is fabricated rather than merely weak.
    with pytest.raises(ValueError, match="Var\\(S\\)"):
        trends.resolve_min_years(2, params)


def test_resolve_min_valid_obs_accepts_null_as_no_floor(
    params_copy: dict[str, Any],
) -> None:
    params_copy["trends"]["min_valid_obs"] = None
    assert trends.resolve_min_valid_obs(None, params_copy) is None


def test_resolve_min_valid_obs_rejects_zero(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        trends.resolve_min_valid_obs(0, params)


def test_resolve_x_origin_defaults_to_the_series_start(params: dict[str, Any]) -> None:
    assert trends.resolve_x_origin(None, params) == params["time"]["start_year"]
    assert trends.resolve_x_origin(None, params, start_year=2005) == 2005


def test_resolve_x_origin_absolute_mode_is_year_zero(
    params_copy: dict[str, Any],
) -> None:
    params_copy["trends"]["x_origin"] = "absolute"
    assert trends.resolve_x_origin(None, params_copy) == 0


def test_resolve_x_origin_rejects_an_unknown_mode(params_copy: dict[str, Any]) -> None:
    params_copy["trends"]["x_origin"] = "midpoint"
    with pytest.raises(ValueError, match="start_year"):
        trends.resolve_x_origin(None, params_copy)


def test_tie_correction_is_off_by_default(params: dict[str, Any]) -> None:
    assert trends.resolve_tie_correction(None, params) is False


def test_tie_correction_requires_the_pairwise_route(params: dict[str, Any]) -> None:
    # The correction only has a meaning inside the explicit pairwise sum; the
    # tau_derived route gets Var(S) from the closed form.
    with pytest.raises(ValueError, match="pairwise"):
        trends.resolve_tie_correction(True, params)


# --- the signum helper (the tutorial's .int() truncation) --------------------
def test_signum_of_a_sub_degree_difference_is_plus_one_not_zero() -> None:
    # THE test. The GEE community tutorial computes the sign as
    # clamp(-1, 1).int(), and .int() truncates toward zero, so +0.3 degC becomes
    # 0. Most annual LST differences in Colombo are well under 1 degC.
    assert trends.signum_array(28.3, 28.0) == pytest.approx(1.0)


def test_the_tutorials_truncating_reference_disagrees_on_a_sub_degree_difference() -> None:
    # The buggy reference, written out so a later "simplification" back to it is
    # a visible test failure rather than a silent collapse of every trend.
    difference = 0.3
    tutorial = int(np.clip(difference, -1, 1))
    assert tutorial == 0
    assert trends.signum_array(28.3, 28.0) != tutorial


@pytest.mark.parametrize(
    ("later", "earlier", "expected"),
    [
        (1.0, 0.0, 1.0),
        (0.0, 1.0, -1.0),
        (5.0, 5.0, 0.0),
        (1e-12, 0.0, 1.0),
        (0.0, 1e-12, -1.0),
        (100.0, 0.0, 1.0),
    ],
)
def test_signum_is_exactly_minus_one_zero_or_plus_one(
    later: float, earlier: float, expected: float
) -> None:
    assert trends.signum_array(later, earlier) == pytest.approx(expected)


def test_signum_preserves_nan() -> None:
    # A missing year must drop the pair, not contribute a spurious 0 to S.
    result = trends.signum_array([1.0, np.nan, 3.0], [0.0, 0.0, np.nan])
    assert result[0] == 1.0
    assert np.isnan(result[1])
    assert np.isnan(result[2])


def test_signum_matches_numpy_sign_on_finite_input() -> None:
    rng = np.random.default_rng(0)
    later, earlier = rng.normal(size=200), rng.normal(size=200)
    assert np.array_equal(
        trends.signum_array(later, earlier), np.sign(later - earlier)
    )


# --- Mann-Kendall arithmetic -------------------------------------------------
def test_kendall_variance_matches_the_closed_form() -> None:
    n = np.array([3, 10, 26, 100], dtype="float64")
    expected = n * (n - 1) * (2 * n + 5) / 18
    assert np.allclose(trends.kendall_variance(n), expected)


def test_kendall_variance_is_nan_below_three_points() -> None:
    assert np.isnan(trends.kendall_variance(2))


def test_s_from_tau_matches_the_pairwise_sign_sum(params: dict[str, Any]) -> None:
    # The naive O(n^2) definition of S, written out here as the oracle.
    del params
    rng = np.random.default_rng(7)
    values = rng.normal(size=15)
    naive = sum(
        np.sign(values[j] - values[i])
        for i in range(len(values))
        for j in range(i + 1, len(values))
    )

    from scipy import stats

    tau = stats.kendalltau(np.arange(values.size), values).statistic
    assert trends.s_from_tau(tau, values.size) == pytest.approx(naive)


def test_s_from_tau_is_nan_below_three_points() -> None:
    assert np.isnan(trends.s_from_tau(1.0, 2))


def test_z_applies_the_continuity_correction_in_both_directions() -> None:
    variance = trends.kendall_variance(20)
    assert trends.z_from_s(50, variance) == pytest.approx(49 / np.sqrt(variance))
    assert trends.z_from_s(-50, variance) == pytest.approx(-49 / np.sqrt(variance))


def test_z_is_exactly_zero_when_s_is_zero() -> None:
    # Built with an explicit branch rather than a division that happens to land
    # near zero, so this is exact.
    assert trends.z_from_s(0, trends.kendall_variance(20)) == 0.0


def test_two_sided_p_matches_scipy() -> None:
    from scipy import stats

    z = np.linspace(-6, 6, 1001)
    assert np.allclose(trends.two_sided_p(z), 2 * stats.norm.sf(np.abs(z)))


def test_two_sided_p_is_not_the_tutorials_one_sided_form() -> None:
    # The tutorial emits 1 - Phi(|Z|) and compensates by thresholding at 0.025.
    # Benjamini-Hochberg needs two-sided input; the one-sided form would halve
    # every p and roughly double the significant area.
    from scipy import stats

    z = 2.5
    one_sided = stats.norm.sf(abs(z))
    assert trends.two_sided_p(z) == pytest.approx(2 * one_sided)
    assert trends.two_sided_p(z) != pytest.approx(one_sided)


def test_two_sided_p_is_bounded_and_nan_safe() -> None:
    p = trends.two_sided_p([0.0, 40.0, np.nan])
    assert p[0] == pytest.approx(1.0)
    assert 0.0 <= p[1] <= 1.0
    assert np.isnan(p[2])


def test_mk_statistics_from_tau_match_pymannkendall() -> None:
    pmk = pytest.importorskip("pymannkendall")
    from scipy import stats

    rng = np.random.default_rng(11)
    for _ in range(25):
        values = np.cumsum(rng.normal(0.05, 1.0, size=26))
        reference = pmk.original_test(values)
        tau = stats.kendalltau(np.arange(values.size), values).statistic
        derived = trends.mk_statistics_from_tau(tau, values.size)

        assert derived["s"] == pytest.approx(reference.s)
        assert derived["var_s"] == pytest.approx(reference.var_s)
        assert derived["z"] == pytest.approx(reference.z)
        assert derived["p"] == pytest.approx(reference.p)


# --- Sen's slope -------------------------------------------------------------
def test_sens_slope_recovers_an_exact_line() -> None:
    years = np.arange(2000, 2026)
    slope, intercept = trends.sens_slope_array(years, 20.0 + 0.05 * (years - 2000))
    assert slope == pytest.approx(0.05)
    assert intercept + slope * 2000 == pytest.approx(20.0)


def test_sens_slope_is_robust_to_a_single_outlier() -> None:
    years = np.arange(2000, 2026, dtype="float64")
    values = 20.0 + 0.05 * (years - 2000)
    values[13] += 50.0

    slope, _ = trends.sens_slope_array(years, values)
    ols = np.polyfit(years, values, 1)[0]
    assert slope == pytest.approx(0.05)
    assert abs(ols - 0.05) > abs(slope - 0.05)


def test_sens_slope_uses_pairwise_deletion_for_nan() -> None:
    years = np.arange(2000, 2010, dtype="float64")
    values = 10.0 + 0.2 * (years - 2000)
    values[3] = np.nan
    slope, _ = trends.sens_slope_array(years, values)
    assert slope == pytest.approx(0.2)


def test_sens_slope_is_per_year_not_per_observation_when_years_are_missing() -> None:
    # THE units bug. pymannkendall's sens_slope uses the array INDEX as x, so on
    # a gapped series it silently reports degC per OBSERVATION. Ours uses the
    # real years and is unchanged by the gap.
    pmk = pytest.importorskip("pymannkendall")

    years = np.arange(2000, 2020, dtype="float64")
    values = 10.0 + 0.1 * (years - 2000)
    keep = np.ones(years.size, dtype=bool)
    keep[2:16] = False  # a wide gap, so most surviving pairs span it

    slope, _ = trends.sens_slope_array(years[keep], values[keep])
    assert slope == pytest.approx(0.1)

    # 0.38 degC per OBSERVATION against 0.10 degC per YEAR - the same series,
    # reported as almost four times the warming.
    index_based = pmk.sens_slope(values[keep]).slope
    assert index_based == pytest.approx(0.38)

    # Worth knowing: the divergence is not guaranteed to be visible. A narrow,
    # contiguous mid-series gap leaves enough within-block pairs that the median
    # is unchanged, so this bug can hide in a spot check and still corrupt the
    # 557 GN divisions that do have scattered gaps.
    narrow = np.ones(years.size, dtype=bool)
    narrow[5:9] = False
    assert pmk.sens_slope(values[narrow]).slope == pytest.approx(0.1)


def test_sens_slope_of_fewer_than_two_points_is_nan() -> None:
    slope, intercept = trends.sens_slope_array([2000.0], [20.0])
    assert np.isnan(slope) and np.isnan(intercept)


def test_sens_slope_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        trends.sens_slope_array([2000, 2001], [10.0])


# --- Benjamini-Hochberg FDR --------------------------------------------------
def _naive_bh(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """The textbook step-up procedure, written out as an independent oracle."""
    m = p_values.size
    order = np.argsort(p_values)
    ranked = p_values[order]
    largest = 0
    for k in range(1, m + 1):
        if ranked[k - 1] <= k / m * alpha:
            largest = k
    reject = np.zeros(m, dtype=bool)
    reject[order[:largest]] = True
    return reject


def test_bh_matches_a_naive_step_up_reference(params: dict[str, Any]) -> None:
    rng = np.random.default_rng(3)
    p_values = np.clip(rng.beta(0.4, 4.0, size=500), 0.0, 1.0)
    alpha = trends.resolve_alpha(None, params)

    reject, _ = trends.benjamini_hochberg(p_values, params=params)
    assert np.array_equal(reject, _naive_bh(p_values, alpha))


def test_bh_matches_statsmodels(params: dict[str, Any]) -> None:
    multipletests = pytest.importorskip("statsmodels.stats.multitest").multipletests

    rng = np.random.default_rng(4)
    p_values = np.clip(rng.beta(0.5, 3.0, size=800), 0.0, 1.0)
    alpha = trends.resolve_alpha(None, params)

    reject, adjusted = trends.benjamini_hochberg(p_values, params=params)
    expected_reject, expected_adjusted, _, _ = multipletests(
        p_values, alpha=alpha, method="fdr_bh"
    )
    assert np.array_equal(reject, expected_reject)
    assert np.allclose(adjusted, expected_adjusted)


def test_benjamini_yekutieli_matches_statsmodels(params: dict[str, Any]) -> None:
    multipletests = pytest.importorskip("statsmodels.stats.multitest").multipletests

    rng = np.random.default_rng(5)
    p_values = np.clip(rng.beta(0.5, 3.0, size=400), 0.0, 1.0)
    alpha = trends.resolve_alpha(None, params)

    reject, adjusted = trends.benjamini_hochberg(
        p_values, params=params, method="benjamini_yekutieli"
    )
    expected_reject, expected_adjusted, _, _ = multipletests(
        p_values, alpha=alpha, method="fdr_by"
    )
    assert np.array_equal(reject, expected_reject)
    assert np.allclose(adjusted, expected_adjusted)


def test_by_is_more_conservative_than_bh(params: dict[str, Any]) -> None:
    rng = np.random.default_rng(6)
    p_values = np.clip(rng.beta(0.4, 5.0, size=600), 0.0, 1.0)

    bh, _ = trends.benjamini_hochberg(p_values, params=params)
    by, _ = trends.benjamini_hochberg(
        p_values, params=params, method="benjamini_yekutieli"
    )
    assert by.sum() <= bh.sum()


def test_bh_adjusted_values_are_monotone_in_the_raw_values(
    params: dict[str, Any],
) -> None:
    # Without the reverse cumulative minimum, a pixel with a smaller p can end up
    # with a larger adjusted value than its neighbour.
    rng = np.random.default_rng(8)
    p_values = np.clip(rng.uniform(size=300), 0.0, 1.0)
    _, adjusted = trends.benjamini_hochberg(p_values, params=params)

    order = np.argsort(p_values)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)


def test_bh_adjusted_never_below_raw_and_never_above_one(
    params: dict[str, Any],
) -> None:
    rng = np.random.default_rng(9)
    p_values = np.clip(rng.uniform(size=300), 0.0, 1.0)
    _, adjusted = trends.benjamini_hochberg(p_values, params=params)
    assert np.all(adjusted >= p_values - 1e-12)
    assert np.all(adjusted <= 1.0)


def test_bh_gives_tied_values_identical_decisions(params: dict[str, Any]) -> None:
    p_values = np.array([0.01, 0.01, 0.01, 0.9, 0.9])
    reject, adjusted = trends.benjamini_hochberg(p_values, params=params)
    assert reject[0] == reject[1] == reject[2]
    assert adjusted[0] == pytest.approx(adjusted[1]) == pytest.approx(adjusted[2])


def test_bh_rejects_everything_when_every_p_is_zero(params: dict[str, Any]) -> None:
    reject, adjusted = trends.benjamini_hochberg(np.zeros(50), params=params)
    assert reject.all()
    assert np.allclose(adjusted, 0.0)


def test_bh_rejects_nothing_when_every_p_is_one(params: dict[str, Any]) -> None:
    reject, adjusted = trends.benjamini_hochberg(np.ones(50), params=params)
    assert not reject.any()
    assert np.allclose(adjusted, 1.0)


def test_bh_of_an_empty_array_is_empty_not_an_error(params: dict[str, Any]) -> None:
    reject, adjusted = trends.benjamini_hochberg(np.array([]), params=params)
    assert reject.size == 0 and adjusted.size == 0


def test_bh_of_an_all_nan_array_returns_all_false_and_warns(
    params: dict[str, Any],
) -> None:
    with pytest.warns(UserWarning, match="COMPLETELY EMPTY"):
        reject, adjusted = trends.benjamini_hochberg(
            np.full(20, np.nan), params=params
        )
    assert not reject.any()
    assert np.isnan(adjusted).all()


def test_bh_excludes_nan_from_the_test_count(params: dict[str, Any]) -> None:
    # NaN pixels were never tested, so they are not tests: the finite entries'
    # adjusted values must be identical with and without them.
    finite = np.array([0.001, 0.02, 0.3, 0.7])
    padded = np.array([0.001, np.nan, 0.02, np.nan, 0.3, np.nan, 0.7])

    _, adjusted_finite = trends.benjamini_hochberg(finite, params=params)
    _, adjusted_padded = trends.benjamini_hochberg(padded, params=params)
    assert np.allclose(adjusted_padded[[0, 2, 4, 6]], adjusted_finite)


def test_bh_never_reports_a_nan_pixel_as_significant(params: dict[str, Any]) -> None:
    reject, _ = trends.benjamini_hochberg(
        np.array([0.0001, np.nan, 0.0002]), params=params
    )
    assert not reject[1]


def test_bh_preserves_a_two_dimensional_raster_shape(params: dict[str, Any]) -> None:
    rng = np.random.default_rng(12)
    grid = np.clip(rng.uniform(size=(17, 23)), 0.0, 1.0)
    reject, adjusted = trends.benjamini_hochberg(grid, params=params)
    assert reject.shape == grid.shape
    assert adjusted.shape == grid.shape


def test_bh_rejects_a_nodata_fill_masquerading_as_a_p_value(
    params: dict[str, Any],
) -> None:
    # The real trap: an exported GeoTIFF carries a nodata fill, and an unmasked
    # -9999 sorts FIRST and becomes the most significant pixel in the AOI.
    with pytest.raises(ValueError, match="-9999"):
        trends.benjamini_hochberg(
            np.array([0.01, -9999.0, 0.5]), params=params
        )


def test_bh_requires_alpha_and_method_when_params_is_absent() -> None:
    with pytest.raises(ValueError, match="params"):
        trends.benjamini_hochberg(np.array([0.1, 0.2]))


def test_bh_is_less_conservative_than_bonferroni(params: dict[str, Any]) -> None:
    rng = np.random.default_rng(13)
    p_values = np.clip(rng.beta(0.3, 6.0, size=500), 0.0, 1.0)
    alpha = trends.resolve_alpha(None, params)

    reject, _ = trends.benjamini_hochberg(p_values, params=params)
    bonferroni = p_values <= alpha / p_values.size
    assert reject.sum() >= bonferroni.sum()


# --- FDR summary and validity ------------------------------------------------
def test_fdr_fraction_reports_both_denominators(params: dict[str, Any]) -> None:
    p_values = np.array([0.0001, 0.0002, 0.5, np.nan, np.nan])
    summary = trends.fdr_significant_fraction(p_values, params)

    assert summary["n_total"] == 5
    assert summary["n_tested"] == 3
    # Quoting the wrong denominator overstates the result; both must be present.
    assert summary["fraction_of_tested"] > summary["fraction_of_total"]


def test_fdr_fraction_splits_warming_from_cooling(params: dict[str, Any]) -> None:
    p_values = np.array([0.0001, 0.0001, 0.9])
    slope = np.array([0.05, -0.04, 0.01])
    summary = trends.fdr_significant_fraction(p_values, params, slope=slope)

    assert summary["n_warming"] == 1
    assert summary["n_cooling"] == 1


def test_fdr_fraction_converts_pixels_to_area(params: dict[str, Any]) -> None:
    p_values = np.zeros(400)
    summary = trends.fdr_significant_fraction(
        p_values, params, pixel_area_m2=100 * 100
    )
    assert summary["area_km2_significant"] == pytest.approx(4.0)


def test_trend_validity_mask_excludes_pixels_below_the_year_floor(
    params: dict[str, Any],
) -> None:
    floor = trends.resolve_min_years(None, params)
    p_values = np.array([0.01, 0.01, 0.01, np.nan])
    slope = np.array([0.1, 0.1, np.nan, 0.1])
    n_years = np.array([floor, floor - 1, floor, floor], dtype="float64")

    mask = trends.trend_validity_mask(p_values, slope, n_years, params)
    assert list(mask) == [True, False, False, False]


# --- the structural guard ----------------------------------------------------
def test_export_shaped_twins_exist_and_are_not_getinfo_shaped() -> None:
    # No interactive question about the trend graph is affordable (Colab runs
    # 10-13), so the grouped reduction has to be able to run inside a batch
    # Export task. These are the unevaluated twins that make that possible.
    import inspect

    from colombo_uhi import landcover

    assert hasattr(trends, "trend_by_class_collection")
    assert hasattr(landcover, "stratified_stats_collection")

    signature = inspect.signature(trends.trend_by_class_collection)
    assert list(signature.parameters)[0] == "image"
    # The evaluated form must still exist for cheap regions.
    assert hasattr(trends, "trend_by_class")


def test_decadal_product_takes_a_source_key() -> None:
    # Same door as trend_image: it builds its own series, so a scene stack
    # cannot reach it.
    import inspect

    parameters = list(inspect.signature(trends.decadal_product).parameters)
    assert parameters[0] == "source"


def test_guard_summary_labels_an_unmeasured_size(params: dict[str, Any]) -> None:
    # The default guard does not measure the collection size, and a bare None in
    # the notebook's printed summary reads like a failure rather than a
    # deliberate omission.
    metadata = {
        "size": None,
        "probed": 2,
        "property_names": ["year", params["composites"]["series_basis_property"]],
        "series_basis": [params["trends"]["series_basis"]] * 2,
        "years": [2023, 2025],
        "months": [],
        "n_scenes": [],
    }
    summary = trends.validate_series_metadata(metadata, params)
    assert summary["n_years"] == "not measured"
    assert summary["n_probed"] == 2


def test_guard_accepts_a_well_formed_annual_series(params: dict[str, Any]) -> None:
    years = list(range(2000, 2026))
    summary = trends.validate_series_metadata(
        _annual_metadata(years, params), params, start_year=2000, end_year=2025
    )
    assert summary["n_years"] == 26
    assert summary["first_year"] == 2000
    assert summary["last_year"] == 2025
    assert summary["empty_years"] == []


def test_guard_accepts_a_dry_season_restricted_series(params: dict[str, Any]) -> None:
    # A Jan-Mar series is still ONE image per year and is the project's primary
    # comparison window; the guard must not confuse a season with a scene stack.
    years = list(range(2000, 2026))
    trends.validate_series_metadata(_annual_metadata(years, params), params)


def test_guard_rejects_a_scene_collection(params: dict[str, Any]) -> None:
    # Scene-level collections carry a `month` property; composites do not. This
    # is the signal that catches a raw sub-annual stack.
    metadata = _annual_metadata([2000, 2001, 2002], params, months=[1, 2, 3])
    with pytest.raises(ValueError, match="SCENE collection"):
        trends.validate_series_metadata(metadata, params)


def test_guard_rejects_a_missing_series_basis(params: dict[str, Any]) -> None:
    metadata = _annual_metadata([2000, 2001, 2002], params, basis=[])
    with pytest.raises(ValueError, match="composites.py"):
        trends.validate_series_metadata(metadata, params)


def test_guard_rejects_a_partially_marked_collection(params: dict[str, Any]) -> None:
    # aggregate_array on a property only SOME images carry returns a shorter
    # array, which is exactly why the probed count is tracked and length-checked.
    metadata = _annual_metadata(
        [2000, 2001, 2002], params, basis=[params["trends"]["series_basis"]]
    )
    with pytest.raises(ValueError, match="only 1 of 3"):
        trends.validate_series_metadata(metadata, params)


def test_guard_accepts_a_sampled_series(params: dict[str, Any]) -> None:
    # The default guard inspects a SAMPLE plus the collection size, because
    # reading properties materialises one full composite graph per image and 26
    # of those exceed the Earth Engine memory limit regardless of region.
    metadata = _annual_metadata(
        [2000, 2001, 2002, 2003], params, size=26, probed=4
    )
    summary = trends.validate_series_metadata(
        metadata, params, start_year=2000, end_year=2025
    )
    assert summary["n_years"] == 26
    assert summary["n_probed"] == 4


def test_guard_rejects_a_size_that_is_not_one_image_per_year(
    params: dict[str, Any],
) -> None:
    # THE check that catches a scene stack even when only a few images are
    # inspected: 1674 scenes against 26 expected years. Distinct years here so
    # the duplicate check does not preempt the one under test.
    metadata = _annual_metadata(
        [2000, 2001, 2002, 2003], params, size=1674, probed=4, months=[]
    )
    with pytest.raises(ValueError, match="holds 1674 image"):
        trends.validate_series_metadata(
            metadata, params, start_year=2000, end_year=2025
        )


def test_guard_rejects_a_probed_year_outside_the_requested_range(
    params: dict[str, Any],
) -> None:
    metadata = _annual_metadata([1998, 1999], params, size=26, probed=2)
    with pytest.raises(ValueError, match="outside the requested range"):
        trends.validate_series_metadata(
            metadata, params, start_year=2000, end_year=2025
        )


def test_a_sampled_guard_still_covers_the_whole_series_via_size(
    params: dict[str, Any],
) -> None:
    # A sample of 4 images out of 26 cannot see a gap in the middle directly -
    # but it does not need to. annual_composites emits one image per calendar
    # year including empty ones, so ANY gap changes the size, and the size is
    # cheap to read. This is what makes sampling sound rather than a shortcut.
    sampled = _annual_metadata([2000, 2001, 2002, 2003], params, size=26, probed=4)
    trends.validate_series_metadata(sampled, params, start_year=2000, end_year=2025)

    gapped = _annual_metadata([2000, 2001, 2002, 2003], params, size=25, probed=4)
    with pytest.raises(ValueError, match="holds 25 image"):
        trends.validate_series_metadata(
            gapped, params, start_year=2000, end_year=2025
        )


def test_guard_rejects_the_wrong_basis_value(params: dict[str, Any]) -> None:
    metadata = _annual_metadata([2000, 2001], params, basis=["monthly", "monthly"])
    with pytest.raises(ValueError, match="expected"):
        trends.validate_series_metadata(metadata, params)


def test_guard_rejects_duplicate_years(params: dict[str, Any]) -> None:
    metadata = _annual_metadata([2000, 2000, 2001], params)
    with pytest.raises(ValueError, match="not unique"):
        trends.validate_series_metadata(metadata, params)


def test_guard_rejects_non_ascending_years(params: dict[str, Any]) -> None:
    metadata = _annual_metadata([2000, 2002, 2001], params)
    with pytest.raises(ValueError, match="ascending"):
        trends.validate_series_metadata(metadata, params)


def test_guard_rejects_a_single_year(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="at least two years"):
        trends.validate_series_metadata(_annual_metadata([2000], params), params)


def test_guard_rejects_a_gap_against_an_explicit_range(params: dict[str, Any]) -> None:
    # annual_composites emits empty years with obs_count == 0, so the axis is
    # always complete. A gap therefore shows up as a SIZE mismatch - 3 images
    # against a 4-year range - which is detectable without inspecting every
    # image, and is why the guard can afford to sample.
    metadata = _annual_metadata([2000, 2001, 2003], params)
    with pytest.raises(ValueError, match="holds 3 image"):
        trends.validate_series_metadata(
            metadata, params, start_year=2000, end_year=2003
        )


def test_guard_warns_about_years_with_no_scenes(params: dict[str, Any]) -> None:
    metadata = _annual_metadata(
        [2000, 2001, 2002], params, n_scenes=[5, 0, 7]
    )
    with pytest.warns(UserWarning, match="ZERO scenes"):
        summary = trends.validate_series_metadata(metadata, params)
    assert summary["empty_years"] == [2001]


def test_guard_does_not_warn_on_a_healthy_series(params: dict[str, Any]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        trends.validate_series_metadata(
            _annual_metadata([2000, 2001, 2002], params), params
        )


# --- structural guard, layer 1 (signature pin) -------------------------------
def test_trend_image_takes_a_source_key_not_a_collection() -> None:
    # Layer 1 of the guard: the primary entry point builds its own annual series
    # from a source KEY, so there is no parameter through which a sub-annual
    # scene stack can reach the Mann-Kendall code. A refactor that reopened this
    # hole would pass every other test in this file.
    import inspect

    parameters = list(inspect.signature(trends.trend_image).parameters)
    assert parameters[0] == "source"
    assert "series" not in parameters
    assert "stack" not in parameters


def test_fit_stack_validates_by_default() -> None:
    # fit_stack is the lower-level door into the reducers - the one a caller can
    # reach with a collection of unknown provenance - so its guard must be
    # opt-OUT, never opt-in.
    import inspect

    assert inspect.signature(trends.fit_stack).parameters["validate"].default is True


def test_require_annual_series_defaults_to_the_cheap_checks() -> None:
    # An annual composite's property dictionary contains n_scenes, which is a
    # COMPUTED collection count. Anything that forces the dictionary to be built
    # - aggregate_array, or sizing the collection - evaluates that count too,
    # which is what exceeded the Earth Engine memory limit in Colab run 11 on
    # only four images. Both expensive paths must stay opt-in.
    import inspect

    signature = inspect.signature(trends.require_annual_series)
    assert signature.parameters["check_size"].default is False
    assert signature.parameters["full"].default is False
    assert "check_scenes" not in signature.parameters


def test_guard_detects_a_scene_from_its_property_names_alone(
    params: dict[str, Any],
) -> None:
    # propertyNames() lists KEYS without evaluating their values, which is the
    # only affordable way to ask "is this a scene?" of a composite whose
    # property dict is expensive to build.
    metadata = {
        "size": None,
        "probed": 1,
        "property_names": ["year", "month", "season", "sensor"],
        "series_basis": [],
        "years": [2000],
        "months": [],
        "n_scenes": [],
    }
    with pytest.raises(ValueError, match="SCENE collection"):
        trends.validate_series_metadata(metadata, params)


def test_guard_rejects_property_names_without_the_marker(
    params: dict[str, Any],
) -> None:
    metadata = {
        "size": None,
        "probed": 1,
        "property_names": ["year", "reducer", "system:time_start"],
        "series_basis": [],
        "years": [2000],
        "months": [],
        "n_scenes": [],
    }
    with pytest.raises(ValueError, match="composites.py"):
        trends.validate_series_metadata(metadata, params)


def test_guard_checks_the_endpoints_when_the_size_is_unknown(
    params: dict[str, Any],
) -> None:
    # With size unknown, the endpoints stand in for it: image 0 must be the
    # start year and image (n-1) must be the end year, which cannot both hold
    # unless there is exactly one image per calendar year.
    def _endpoints(years: list[int]) -> dict[str, Any]:
        # Built inline rather than through _annual_metadata, because `size` here
        # must be genuinely UNKNOWN (None), which is the default guard's state.
        return {
            "size": None,
            "probed": 2,
            "property_names": [
                "year",
                params["composites"]["series_basis_property"],
            ],
            "series_basis": [params["trends"]["series_basis"]] * 2,
            "years": years,
            "months": [],
            "n_scenes": [],
        }

    trends.validate_series_metadata(
        _endpoints([2000, 2025]), params, start_year=2000, end_year=2025
    )

    with pytest.raises(ValueError, match="expected 2025"):
        trends.validate_series_metadata(
            _endpoints([2000, 2019]), params, start_year=2000, end_year=2025
        )


def test_validate_series_metadata_tolerates_absent_scene_counts(
    params: dict[str, Any],
) -> None:
    # The cheap path omits n_scenes entirely; the validator must still work and
    # must not claim any year is empty.
    metadata = _annual_metadata([2000, 2001, 2002], params, n_scenes=[])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        summary = trends.validate_series_metadata(metadata, params)
    assert summary["empty_years"] == []


def test_selftest_annual_series_takes_a_source_key() -> None:
    # The constructor self-test is what Phase 4's structural guarantee rests on
    # in practice, so it must go through the same source-key door as
    # trend_image - not accept a collection.
    import inspect

    parameters = list(inspect.signature(trends.selftest_annual_series).parameters)
    assert parameters[0] == "source"
    assert "collection" not in parameters


def test_selftest_annual_series_rejects_a_single_year(params: dict[str, Any]) -> None:
    # A one-image series cannot demonstrate a year axis at all.
    with pytest.raises(ValueError, match="years must be >= 2"):
        trends.selftest_annual_series("landsat_dry", params, region=None, years=1)


# --- reducer band-name resolution --------------------------------------------
def test_reducer_outputs_match_the_module_constants(params: dict[str, Any]) -> None:
    # The product path selects these names directly, with no getInfo, because
    # they were MEASURED in Colab run 11. If the params drift from the constants
    # the probe helper checks against, one of the two is wrong.
    outputs = params["trends"]["reducer_outputs"]
    assert tuple(outputs["sen"]) == trends.SEN_OUTPUTS
    assert tuple(outputs["kendall"]) == trends.KENDALL_OUTPUTS
    # The hyphen is real and easy to "correct" into an underscore.
    assert "p-value" in outputs["kendall"]


def test_band_names_resolve_from_exact_suffixes() -> None:
    assert trends.resolve_reduced_band_names(
        ["slope", "offset"], trends.SEN_OUTPUTS, "sensSlope"
    ) == ["slope", "offset"]


def test_band_names_resolve_from_band_prefixed_outputs() -> None:
    assert trends.resolve_reduced_band_names(
        ["fit_y_tau", "fit_y_p-value"],
        trends.KENDALL_OUTPUTS,
        "kendallsCorrelation",
        band="fit_y",
    ) == ["fit_y_tau", "fit_y_p-value"]


def test_band_names_fall_back_to_position_with_a_warning() -> None:
    with pytest.warns(UserWarning, match="POSITION"):
        resolved = trends.resolve_reduced_band_names(
            ["a", "b"], trends.SEN_OUTPUTS, "sensSlope"
        )
    assert resolved == ["a", "b"]


def test_band_names_raise_and_name_every_observed_band() -> None:
    with pytest.raises(RuntimeError, match="unexpected_band"):
        trends.resolve_reduced_band_names(
            ["unexpected_band"], trends.SEN_OUTPUTS, "sensSlope"
        )


# --- decades -----------------------------------------------------------------
def test_decades_tile_the_study_period_without_gap_or_overlap(
    params: dict[str, Any],
) -> None:
    windows = trends.resolve_decades(None, params)
    assert windows[0][1] == params["time"]["start_year"]
    assert windows[-1][2] == params["time"]["end_year"]
    for earlier, later in zip(windows, windows[1:]):
        assert later[1] == earlier[2] + 1


def test_decades_are_eleven_ten_and_five_years_by_design(
    params: dict[str, Any],
) -> None:
    # Pinned so an "equalise the decades" tidy-up is a visible edit. The study
    # period ends in 2025, so the last window rests on half the sample of the
    # others and dominates the uncertainty of any difference map it appears in.
    lengths = [end - start + 1 for _, start, end in trends.resolve_decades(None, params)]
    assert lengths == [11, 10, 5]


def test_decades_are_not_the_utfvi_epochs(params: dict[str, Any]) -> None:
    # They answer different questions - UTFVI is within-epoch spatial
    # redistribution against a moving reference, this is absolute temperature
    # level - so one key must never be made to serve both.
    trend_windows = {
        (start, end) for _, start, end in trends.resolve_decades(None, params)
    }
    utfvi = {
        (int(bounds[0]), int(bounds[1]))
        for bounds in params["uhi"]["utfvi"]["epochs"].values()
    }
    assert trend_windows != utfvi


def test_decades_reject_an_inverted_window(params_copy: dict[str, Any]) -> None:
    params_copy["trends"]["decades"] = {"bad": [2020, 2010]}
    with pytest.raises(ValueError, match="inverted"):
        trends.resolve_decades(None, params_copy)


def test_decades_reject_overlapping_windows(params_copy: dict[str, Any]) -> None:
    params_copy["trends"]["decades"] = {"a": [2000, 2012], "b": [2010, 2020]}
    with pytest.raises(ValueError, match="overlap"):
        trends.resolve_decades(None, params_copy)


def test_decade_years_reject_an_unknown_label(params: dict[str, Any]) -> None:
    with pytest.raises(KeyError, match="unknown decade"):
        trends.decade_years(params, "1990_1999")


def test_decadal_band_order_covers_every_window(params: dict[str, Any]) -> None:
    order = trends.decadal_band_order(params)
    for label, _, _ in trends.resolve_decades(None, params):
        # CLAUDE.md caveat 2: the count travels with the statistic.
        assert f"mean_{label}" in order
        assert f"sd_{label}" in order
        assert f"n_years_{label}" in order


def test_decadal_band_order_has_no_duplicates(params: dict[str, Any]) -> None:
    # Band identity in a GeoTIFF is POSITIONAL, so a duplicated name means the
    # reader silently maps two different bands onto one.
    order = trends.decadal_band_order(params)
    assert len(order) == len(set(order))


def test_decadal_band_order_suffixes_every_difference(
    params: dict[str, Any],
) -> None:
    # Several differences share one image, so diff_se and diff_z must carry the
    # window pair or the second difference overwrites the first.
    order = trends.decadal_band_order(params)
    ses = [name for name in order if name.startswith("diff_se")]
    assert len(ses) == 2
    assert all(name != "diff_se" for name in ses)


def test_decadal_band_order_follows_the_configured_windows(
    params_copy: dict[str, Any],
) -> None:
    # Derived from params rather than hardcoded, so adding a window cannot
    # desync the writer from the reader.
    params_copy["trends"]["decades"] = {"a": [2000, 2009], "b": [2010, 2019]}
    order = trends.decadal_band_order(None or params_copy)
    assert "mean_a" in order and "mean_b" in order
    assert len([n for n in order if n.startswith("diff_se")]) == 1


# --- modified Mann-Kendall wrapper -------------------------------------------
def test_mk_comparison_emits_both_tests_with_the_documented_columns(
    params: dict[str, Any],
) -> None:
    pytest.importorskip("pymannkendall")

    years = np.arange(2000, 2026)
    rng = np.random.default_rng(31)
    # Jittered rather than perfectly linear: a deterministic series drives the
    # Hamed-Rao correction into its degenerate branch, which is exercised on
    # purpose by test_mk_comparison_labels_a_degenerate_hamed_rao_variance.
    values = 20.0 + 0.05 * (years - 2000) + rng.normal(0, 0.15, size=years.size)

    frame = trends.mk_comparison(
        values, params, years=years, label="cmc", series="lst"
    )
    assert list(frame.columns) == list(trends.MK_COLUMNS)
    assert set(frame["test"]) == {"original", params["trends"]["mmk"]["method"]}


def test_mk_comparison_recovers_a_planted_slope(params: dict[str, Any]) -> None:
    pytest.importorskip("pymannkendall")

    years = np.arange(2000, 2026)
    frame = trends.mk_comparison(
        20.0 + 0.04 * (years - 2000), params, years=years
    )
    assert frame["slope"].iloc[0] == pytest.approx(0.04)


def test_modified_variance_exceeds_the_original_on_autocorrelated_data(
    params: dict[str, Any],
) -> None:
    # The whole point of running MMK: annual LST is positively autocorrelated,
    # which inflates the true variance of S, so the plain test is
    # anti-conservative.
    pytest.importorskip("pymannkendall")

    rng = np.random.default_rng(21)
    values = np.zeros(40)
    for index in range(1, values.size):
        values[index] = 0.85 * values[index - 1] + rng.normal()

    frame = trends.mk_comparison(values, params, years=np.arange(2000, 2040))
    inflation = frame.loc[frame["test"] == "hamed_rao", "var_inflation"].iloc[0]
    assert inflation > 1.0


def test_mk_comparison_returns_insufficient_data_instead_of_raising(
    params: dict[str, Any],
) -> None:
    # One degenerate GN division must not cost the other 556.
    frame = trends.mk_comparison([1.0, 2.0, 3.0], params, years=[2000, 2001, 2002])
    assert set(frame["status"]) == {trends.TREND_INSUFFICIENT}


def test_mk_comparison_returns_no_variance_for_a_constant_series(
    params: dict[str, Any],
) -> None:
    frame = trends.mk_comparison(
        np.full(20, 28.5), params, years=np.arange(2000, 2020)
    )
    assert set(frame["status"]) == {trends.TREND_NO_VARIANCE}
    assert frame["slope"].iloc[0] == 0.0


def test_mk_comparison_counts_only_the_finite_observations(
    params: dict[str, Any],
) -> None:
    pytest.importorskip("pymannkendall")

    values = np.arange(26, dtype="float64")
    values[[3, 9]] = np.nan
    frame = trends.mk_comparison(values, params, years=np.arange(2000, 2026))
    assert frame["n_years"].iloc[0] == 24
    assert frame["n_missing"].iloc[0] == 2


def test_mk_comparison_flags_a_gapped_series(params: dict[str, Any]) -> None:
    pytest.importorskip("pymannkendall")

    values = np.arange(26, dtype="float64")
    values[:10] = np.nan
    with pytest.warns(UserWarning, match="gapped|missing"):
        frame = trends.mk_comparison(values, params, years=np.arange(2000, 2026))
    assert bool(frame["gapped"].iloc[0]) is True


def test_mk_comparison_labels_a_degenerate_hamed_rao_variance(
    params: dict[str, Any],
) -> None:
    # A perfect arithmetic progression detrends to EXACTLY zero residuals, so the
    # lag-0 autocovariance is zero and the Hamed-Rao correction divides by it,
    # returning a non-finite Var(S) and hence NaN z and p. That is a limitation
    # of the method, not a missing value, and it must be labelled rather than
    # land in a report table as a silent blank.
    pytest.importorskip("pymannkendall")

    years = np.arange(2000, 2026)
    with pytest.warns(UserWarning, match="degenerate"):
        frame = trends.mk_comparison(
            np.arange(20.0, 46.0), params, years=years, label="linear"
        )

    modified = frame[frame["test"] == params["trends"]["mmk"]["method"]].iloc[0]
    assert modified["status"] == "degenerate_variance"
    # The uncorrected test is still usable and is what should be quoted.
    original = frame[frame["test"] == "original"].iloc[0]
    assert original["status"] == trends.TREND_OK
    assert np.isfinite(original["p"])


def test_mk_comparison_rejects_an_unknown_method(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="hamed_rao"):
        trends.mk_comparison([1.0, 2.0], params, method="ols")


def test_mk_comparison_rejects_mismatched_year_and_value_lengths(
    params: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="same length"):
        trends.mk_comparison([1.0, 2.0, 3.0], params, years=[2000, 2001])


def test_build_mk_frame_of_no_rows_is_empty_but_shaped() -> None:
    frame = trends.build_mk_frame([])
    assert frame.empty
    assert list(frame.columns) == list(trends.MK_COLUMNS)


# --- SUHII trend table -------------------------------------------------------
def _suhii_frame() -> "Any":
    import pandas as pd

    # Jittered: a perfectly deterministic series pushes Hamed-Rao into its
    # degenerate branch, which is not what these tests are about.
    rng = np.random.default_rng(41)
    rows = []
    for source in ("landsat_dry", "terra_night"):
        for definition in ("buffer_ring", "lcz_based"):
            for offset, year in enumerate(range(2000, 2026)):
                noise = rng.normal(0, 0.1, size=3)
                rows.append(
                    {
                        "year": year,
                        "source": source,
                        "rural_definition": definition,
                        "urban_mean": 30.0 + 0.03 * offset + noise[0],
                        "rural_mean": 26.0 + 0.01 * offset + noise[1],
                        "suhii": 4.0 + 0.02 * offset + noise[2],
                        "urban_pixels": 4206,
                        "rural_pixels": 9000,
                    }
                )
    return pd.DataFrame(rows)


def test_suhii_trends_cover_every_source_and_rural_definition(
    params: dict[str, Any],
) -> None:
    pytest.importorskip("pymannkendall")

    frame = trends.suhii_trends(_suhii_frame(), params)
    assert set(frame["label"]) == {
        "landsat_dry|buffer_ring",
        "landsat_dry|lcz_based",
        "terra_night|buffer_ring",
        "terra_night|lcz_based",
    }


def test_suhii_trends_decompose_into_urban_and_rural(params: dict[str, Any]) -> None:
    # Answers what the SUHII trend alone cannot: did SUHII rise because the city
    # warmed, or because the countryside warmed less?
    pytest.importorskip("pymannkendall")

    frame = trends.suhii_trends(_suhii_frame(), params)
    assert set(frame["series"]) == {"suhii", "urban_mean", "rural_mean"}


def test_suhii_trends_drop_years_with_no_valid_pixels(params: dict[str, Any]) -> None:
    pytest.importorskip("pymannkendall")

    frame = _suhii_frame()
    frame.loc[frame["year"].isin([2003, 2004]), "urban_pixels"] = 0
    result = trends.suhii_trends(frame, params)
    assert result["n_years"].max() == 24


def test_suhii_trends_raise_on_a_missing_column(params: dict[str, Any]) -> None:
    frame = _suhii_frame().drop(columns=["rural_pixels"])
    with pytest.raises(KeyError, match="rural_pixels"):
        trends.suhii_trends(frame, params)


# --- the exported-raster FDR path --------------------------------------------
#
# This is the AUTHORITATIVE route CLAUDE.md specifies - the correction applied in
# Python, on the exported p-value raster. It is pure Python, so it is testable
# without Earth Engine; it needs rasterio, which is a Colab dependency and may
# not be installed locally, hence the importorskip.
NODATA = -9999.0


def _write_synthetic_trend_raster(path: Any, params: dict[str, Any]) -> None:
    """Write a GeoTIFF shaped exactly like an exported trend product."""
    import rasterio
    from rasterio.transform import from_origin

    bands = params["trends"]["bands"]
    order = list(params["trends"]["export_band_order"])

    rng = np.random.default_rng(5)
    height, width = 40, 50
    slope = rng.normal(0.03, 0.04, (height, width))
    z = slope / 0.012

    data = {
        bands["sen_slope"]: slope,
        bands["sen_offset"]: np.full((height, width), 29.0),
        bands["mk_tau"]: np.clip(z / 6.0, -1, 1),
        bands["mk_z"]: z,
        bands["mk_s"]: z * 10.0,
        bands["mk_var_s"]: np.full((height, width), 2058.3),
        bands["mk_p_two_sided"]: trends.two_sided_p(z),
        bands["mk_p_ee"]: trends.two_sided_p(z) / 2.0,
        bands["n_years"]: rng.integers(4, 27, (height, width)).astype("float64"),
    }

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(order),
        "dtype": "float32",
        "crs": "EPSG:32644",
        "transform": from_origin(400000, 700000, 100, 100),
        "nodata": NODATA,
    }
    with rasterio.open(path, "w", **profile) as handle:
        for index, name in enumerate(order, start=1):
            array = data[name].astype("float32").copy()
            array[:6, :6] = NODATA  # a block that was never observed
            handle.write(array, index)
            handle.set_band_description(index, name)


def test_read_trend_raster_converts_nodata_to_nan(
    params: dict[str, Any], tmp_path: Path
) -> None:
    # THE trap this guards: an unmasked -9999 sorts FIRST and becomes the most
    # significant pixel in the AOI.
    pytest.importorskip("rasterio")

    source = tmp_path / "trend.tif"
    _write_synthetic_trend_raster(source, params)

    arrays, profile = trends.read_trend_raster(source, params)
    assert set(arrays) == set(params["trends"]["export_band_order"])
    assert np.isnan(arrays[params["trends"]["bands"]["sen_slope"]][0, 0])
    assert profile["crs"] is not None


def test_read_trend_raster_rejects_a_band_count_mismatch(
    params: dict[str, Any], tmp_path: Path
) -> None:
    # Band identity in a GeoTIFF is POSITIONAL, so reading under the wrong order
    # would map the p-value band onto the slope and produce a plausible-looking,
    # entirely wrong significance map.
    pytest.importorskip("rasterio")

    source = tmp_path / "trend.tif"
    _write_synthetic_trend_raster(source, params)

    with pytest.raises(RuntimeError, match="export_band_order"):
        trends.read_trend_raster(source, params, band_order=["sen_slope", "mk_z"])


def test_apply_fdr_to_raster_writes_the_masked_slope(
    params: dict[str, Any], tmp_path: Path
) -> None:
    rasterio = pytest.importorskip("rasterio")

    source = tmp_path / "trend.tif"
    destination = tmp_path / "trend_fdr.tif"
    _write_synthetic_trend_raster(source, params)

    summary = trends.apply_fdr_to_raster(source, destination, params)

    assert destination.exists()
    assert summary["n_tested"] < summary["n_total"]  # the nodata block was excluded
    assert 0 <= summary["n_significant"] <= summary["n_tested"]
    # 100 m pixels, so each is 0.01 km2.
    assert summary["area_km2_significant"] == pytest.approx(
        summary["n_significant"] * 0.01
    )

    with rasterio.open(destination) as handle:
        assert list(handle.descriptions) == [
            "sen_slope",
            "sen_slope_fdr",
            "p_two_sided",
            "p_adjusted",
            "significant",
            "n_years",
        ]
        significant = handle.read(5)
        masked_slope = handle.read(2)

    # "not tested" and "tested but not significant" must be DIFFERENT values.
    assert np.isnan(significant[0, 0])
    assert set(np.unique(significant[np.isfinite(significant)])) <= {0.0, 1.0}
    # The masked slope is finite exactly where the trend is significant.
    assert np.array_equal(np.isfinite(masked_slope), significant == 1.0)


def test_apply_fdr_to_raster_excludes_pixels_below_the_year_floor(
    params: dict[str, Any], tmp_path: Path
) -> None:
    pytest.importorskip("rasterio")

    source = tmp_path / "trend.tif"
    _write_synthetic_trend_raster(source, params)

    lenient = trends.apply_fdr_to_raster(
        source, tmp_path / "lenient.tif", params, min_years=3
    )
    strict = trends.apply_fdr_to_raster(
        source, tmp_path / "strict.tif", params, min_years=24
    )
    assert strict["n_tested"] < lenient["n_tested"]


def test_apply_fdr_to_raster_reports_a_more_conservative_by_result(
    params: dict[str, Any], tmp_path: Path
) -> None:
    # The BH/BY pair is the dependence sensitivity that must travel with every
    # significant-area figure.
    pytest.importorskip("rasterio")

    source = tmp_path / "trend.tif"
    _write_synthetic_trend_raster(source, params)

    bh = trends.apply_fdr_to_raster(source, tmp_path / "bh.tif", params)
    by = trends.apply_fdr_to_raster(
        source, tmp_path / "by.tif", params, method="benjamini_yekutieli"
    )
    assert by["n_significant"] <= bh["n_significant"]


# --- cross-sensor continuity -------------------------------------------------
def _sensor_series(rows):
    import pandas as pd

    return pd.DataFrame(
        rows, columns=["sensor_key", "year", "mean", "valid_pixels"]
    )


def test_sensor_offset_summary_recovers_a_planted_step(
    params: dict[str, Any],
) -> None:
    # THE case this exists for: two sensors observing the same years, one
    # reading 0.8 degC hotter. Over 26 years a 0.8 degC step dwarfs a
    # ~0.03 degC/yr trend, and Mann-Kendall would measure the step.
    rows = []
    for year in range(2005, 2012):
        rows.append(["landsat5", year, 30.0, 4000])
        rows.append(["landsat7", year, 30.8, 4000])
    summary = trends.build_sensor_offset_summary(_sensor_series(rows), params)

    assert len(summary) == 1
    record = summary.iloc[0]
    assert record["n_overlap_years"] == 7
    assert record["mean_offset"] == pytest.approx(-0.8)
    assert record["verdict"] == trends.SENSOR_OFFSET_MATERIAL


def test_sensor_offset_summary_calls_a_small_offset_negligible(
    params: dict[str, Any],
) -> None:
    rows = []
    for year in range(2005, 2012):
        rows.append(["landsat5", year, 30.0, 4000])
        rows.append(["landsat7", year, 30.05, 4000])
    summary = trends.build_sensor_offset_summary(_sensor_series(rows), params)
    assert summary.iloc[0]["verdict"] == trends.SENSOR_OFFSET_NEGLIGIBLE


def test_sensor_offset_summary_flags_too_little_overlap(
    params: dict[str, Any],
) -> None:
    # Two sensors that barely coexist cannot be compared; saying so beats
    # reporting an offset from two years of weather.
    rows = [
        ["landsat5", 2011, 30.0, 4000],
        ["landsat7", 2011, 30.9, 4000],
        ["landsat5", 2012, 30.2, 4000],
        ["landsat7", 2012, 31.0, 4000],
    ]
    summary = trends.build_sensor_offset_summary(_sensor_series(rows), params)
    assert summary.iloc[0]["verdict"] == trends.SENSOR_OFFSET_INSUFFICIENT
    assert np.isnan(summary.iloc[0]["mean_offset"])


def test_sensor_offset_summary_ignores_years_with_no_valid_pixels(
    params: dict[str, Any],
) -> None:
    # A year with zero valid pixels is not an observation, so it must not count
    # toward the overlap.
    rows = []
    for year in range(2005, 2012):
        rows.append(["landsat5", year, 30.0, 4000])
        rows.append(["landsat7", year, 30.5, 4000])
    rows.append(["landsat5", 2013, 99.0, 0])
    rows.append(["landsat7", 2013, 99.0, 0])
    summary = trends.build_sensor_offset_summary(_sensor_series(rows), params)
    assert summary.iloc[0]["n_overlap_years"] == 7


def test_sensor_offset_summary_reports_the_spread_not_just_the_offset(
    params: dict[str, Any],
) -> None:
    # A consistent offset and a noisy one mean different things, so the standard
    # error and t statistic travel with the mean.
    rng = np.random.default_rng(31)
    rows = []
    for year in range(2000, 2012):
        rows.append(["landsat5", year, 30.0, 4000])
        rows.append(["landsat7", year, 30.0 + rng.normal(0.6, 0.05), 4000])
    summary = trends.build_sensor_offset_summary(_sensor_series(rows), params)
    record = summary.iloc[0]
    assert record["sd_offset"] > 0
    assert abs(record["t_statistic"]) > 5      # consistent, not sampling noise


def test_sensor_offset_summary_orders_by_absolute_offset(
    params: dict[str, Any],
) -> None:
    rows = []
    for year in range(2005, 2012):
        rows.append(["landsat5", year, 30.0, 4000])
        rows.append(["landsat7", year, 30.1, 4000])
        rows.append(["landsat8", year, 31.5, 4000])
    summary = trends.build_sensor_offset_summary(_sensor_series(rows), params)
    magnitudes = summary["mean_offset"].abs().dropna().tolist()
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_sensor_offset_summary_of_an_empty_frame_is_shaped(
    params: dict[str, Any],
) -> None:
    summary = trends.build_sensor_offset_summary(_sensor_series([]), params)
    assert summary.empty
    assert list(summary.columns) == list(trends.SENSOR_OFFSET_COLUMNS)


def test_sensor_offset_summary_raises_on_a_missing_column(
    params: dict[str, Any],
) -> None:
    frame = _sensor_series([["landsat5", 2005, 30.0, 4000]]).drop(columns=["mean"])
    with pytest.raises(KeyError, match="mean"):
        trends.build_sensor_offset_summary(frame, params)


def test_sensor_annual_means_takes_params_first() -> None:
    # It builds its own per-sensor collections, so it must not accept one.
    import inspect

    parameters = list(inspect.signature(trends.sensor_annual_means).parameters)
    assert parameters[0] == "params"
    assert "collection" not in parameters


def test_source_years_honour_a_sensor_restricted_floor(params: dict[str, Any]) -> None:
    # THE bug from Colab run 17: a 2013 source built 2000-2025, wasting seven
    # round trips on years that cannot contain data and then warning that 54%
    # of every division's series was "missing".
    source = {"key": "landsat_oli_dry", "start_year": 2013}
    first, last = trends.resolve_source_years(source, params)
    assert first == 2013
    assert last == params["time"]["end_year"]


def test_source_years_fall_back_to_the_project_range(params: dict[str, Any]) -> None:
    first, last = trends.resolve_source_years({"key": "landsat_dry"}, params)
    assert first == params["time"]["start_year"]
    assert last == params["time"]["end_year"]


def test_source_years_let_an_explicit_argument_win(params: dict[str, Any]) -> None:
    # Batching passes an explicit window, and that must override the floor.
    source = {"key": "landsat_oli_dry", "start_year": 2013}
    assert trends.resolve_source_years(source, params, start_year=2018)[0] == 2018


def test_source_years_reject_an_inverted_range(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be >= start_year"):
        trends.resolve_source_years({"key": "x"}, params, start_year=2020, end_year=2010)


def test_configured_oli_source_starts_when_landsat8_does(
    params: dict[str, Any],
) -> None:
    # Landsat 8 opens 2013-03; a source restricted to OLI cannot start earlier.
    source = next(
        s for s in params["uhi"]["suhii"]["sources"] if s["key"] == "landsat_oli_dry"
    )
    assert source["start_year"] == 2013
    assert set(source["sensors"]) == {"landsat8", "landsat9"}
    # L8-L9 measured negligible (-0.40 degC, t=-0.67), so they pool; L5 and L7
    # measured MATERIAL against L8 and must never join this source.
    assert "landsat5" not in source["sensors"]
    assert "landsat7" not in source["sensors"]
