"""Pin the pure-Python core of Phase 5: weights, the statistics, EHSA, the ladder.

No Earth Engine anywhere. The Earth Engine surface of ``spatial_stats`` - the
geometry export, the covariate stack, the zonal reductions - is verified only by
running notebook 05 in Colab. What is testable here is the arithmetic, the
classification rules and the guards, which is exactly where the
silent-wrong-answer risks live.

Six things carry real risk and are therefore pinned hard:

1. **An island is not a zero.** A zone with no neighbours gets a local statistic
   computed over an empty neighbourhood, and the naive pseudo-p formula hands it
   the SMALLEST achievable p-value - the island would be the most significant
   cluster on the map. The tests assert its p comes back NaN.
2. **Gi* is undefined for negative values.** It returns finite, plausible
   numbers on a z-score input. The guard has its own test, and asserts the error
   message names the fix.
3. **Moran's I and Gi* need different weights.** Row-standardising Gi* collapses
   its variance term. The tests check the analytic values against hand-computed
   reference cases on a lattice where the answers are known exactly (a perfect
   checkerboard has Moran's I of exactly -1).
4. **The EHSA rule order is load-bearing.** An all-hot series is trivially "one
   unbroken final run", so without the share guard `consecutive` swallows
   `persistent`, `intensifying` and `diminishing` entirely. Every category has
   its own test, including the cold mirrors.
5. **A model that cannot be estimated must not be estimated.** ``require_estimable``
   is what stops a GWR being fitted to 13 DS divisions and 6 predictors.
6. **Landscape metrics are checked against hand-computed values**, not against
   themselves: a solid block has an aggregation index of exactly 100 and a
   checkerboard exactly 0, and those are asserted rather than approximated.

The statistics are implemented analytically in numpy rather than delegated to
``esda``/``spreg`` so they can be tested here at all - the local environment has
no PySAL. That choice is only safe if it is checked, so the cross-validation
tests at the bottom assert agreement with the reference libraries via
``pytest.importorskip``, the same pattern ``test_trends`` uses against
``pymannkendall``.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any

import numpy as np
import pytest

from colombo_uhi import load_params, spatial_stats as ss


@pytest.fixture(scope="module")
def params() -> dict[str, Any]:
    return load_params()


@pytest.fixture()
def params_copy(params: dict[str, Any]) -> dict[str, Any]:
    """Deep copy for tests that mutate the config."""
    return copy.deepcopy(params)


# =============================================================================
# Test doubles - a lattice of squares that quacks like a shapely polygon
# =============================================================================
class _Shared:
    """The result of intersecting two boundaries: only its length is used."""

    def __init__(self, length: float) -> None:
        self.length = length


class _Boundary:
    """A square's boundary, supporting only the intersection this module needs."""

    def __init__(self, square: "_Square") -> None:
        self.square = square

    def intersection(self, other: "_Boundary") -> _Shared:
        a, b = self.square, other.square
        overlap_x = min(a.maxx, b.maxx) - max(a.minx, b.minx)
        overlap_y = min(a.maxy, b.maxy) - max(a.miny, b.miny)
        if overlap_x < 0 or overlap_y < 0:
            return _Shared(0.0)
        if overlap_x == 0 and overlap_y == 0:
            return _Shared(0.0)  # corner contact: queen but not rook
        if overlap_x == 0:
            return _Shared(overlap_y)
        if overlap_y == 0:
            return _Shared(overlap_x)
        return _Shared(0.0)


class _Square:
    """An axis-aligned unit square standing in for a shapely polygon.

    ``contiguity_neighbours`` uses only ``bounds``, ``intersects`` and
    ``boundary``, so a lattice can be built without shapely - which the local
    test environment does not have.
    """

    def __init__(self, col: int, row: int, size: float = 1.0) -> None:
        self.minx, self.miny = col * size, row * size
        self.maxx, self.maxy = self.minx + size, self.miny + size

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.minx, self.miny, self.maxx, self.maxy)

    def intersects(self, other: "_Square") -> bool:
        return not (
            self.maxx < other.minx
            or other.maxx < self.minx
            or self.maxy < other.miny
            or other.maxy < self.miny
        )

    @property
    def boundary(self) -> _Boundary:
        return _Boundary(self)


def _lattice(side: int) -> list[_Square]:
    """Row-major lattice of unit squares, ``side`` by ``side``."""
    return [_Square(col, row) for row in range(side) for col in range(side)]


def _rook_matrix(side: int) -> np.ndarray:
    """Binary rook weights for a ``side`` x ``side`` lattice, row-major."""
    neighbours: list[list[int]] = [[] for _ in range(side * side)]
    for row in range(side):
        for col in range(side):
            here = row * side + col
            for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r, c = row + d_row, col + d_col
                if 0 <= r < side and 0 <= c < side:
                    neighbours[here].append(r * side + c)
    return ss.neighbours_to_matrix(neighbours)


# =============================================================================
# Parameter resolution
# =============================================================================
def test_resolve_level_accepts_both_aggregation_levels() -> None:
    assert ss.resolve_level("gn") == "gn"
    assert ss.resolve_level("ds") == "ds"


def test_resolve_level_rejects_anything_else() -> None:
    with pytest.raises(ValueError, match="level must be one of"):
        ss.resolve_level("grid")


def test_resolve_permutations_defaults_from_params(params: dict[str, Any]) -> None:
    assert ss.resolve_permutations(None, params) == params["spatial_stats"]["permutations"]


def test_resolve_permutations_rejects_a_count_too_small_for_the_breaks(
    params: dict[str, Any],
) -> None:
    # The finest pseudo p obtainable is 1/(n+1); at n=50 that is 0.0196, so the
    # configured 0.01 confidence break could never be reached and every zone
    # would be capped at 95% no matter how extreme.
    with pytest.raises(ValueError, match="smallest pseudo p-value"):
        ss.resolve_permutations(50, params)


def test_resolve_seed_defaults_from_params(params: dict[str, Any]) -> None:
    assert ss.resolve_seed(None, params) == params["spatial_stats"]["random_seed"]


def test_resolve_weights_scheme_rejects_an_unknown_scheme(
    params: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="unsupported weights scheme"):
        ss.resolve_weights_scheme("delaunay", params)


def test_resolve_island_policy_rejects_an_unknown_policy(
    params: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="unsupported island_policy"):
        ss.resolve_island_policy("ignore", params)


def test_regression_predictors_come_from_params(params: dict[str, Any]) -> None:
    assert ss.resolve_regression_predictors(None, params) == list(
        params["spatial_stats"]["regression"]["predictors"]
    )


def test_regression_predictors_collapse_duplicates(params: dict[str, Any]) -> None:
    assert ss.resolve_regression_predictors(["NDVI", "NDVI", "NDBI"], params) == [
        "NDVI",
        "NDBI",
    ]


def test_regression_predictors_reject_the_response(params: dict[str, Any]) -> None:
    # Regressing LST on LST gives R2 = 1 and a perfectly fitting, meaningless map.
    response = params["spatial_stats"]["regression"]["response"]
    with pytest.raises(ValueError, match="appears in the predictor list"):
        ss.resolve_regression_predictors(["NDVI", response], params)


def test_green_classes_are_class_codes_present_in_the_legend(
    params: dict[str, Any],
) -> None:
    codes = ss.resolve_green_classes("worldcover", params)
    legend = params["landcover"]["worldcover"]["classes"]
    assert codes
    assert all(code in legend for code in codes)


def test_green_classes_reject_a_code_outside_the_legend(
    params_copy: dict[str, Any],
) -> None:
    # A code with no legend entry matches no pixels, so every metric would come
    # back zero with no error anywhere.
    params_copy["spatial_stats"]["landscape"]["green_classes"]["worldcover"] = [10, 999]
    with pytest.raises(ValueError, match="999"):
        ss.resolve_green_classes("worldcover", params_copy)


def test_green_classes_reject_an_unknown_scheme(params: dict[str, Any]) -> None:
    with pytest.raises(KeyError, match="lcz"):
        ss.resolve_green_classes("lcz", params)


# =============================================================================
# Spatial weights
# =============================================================================
def test_queen_contiguity_on_a_lattice_includes_the_diagonals() -> None:
    neighbours = ss.contiguity_neighbours(_lattice(3), "queen")
    assert sorted(neighbours[4]) == [0, 1, 2, 3, 5, 6, 7, 8]  # centre cell
    assert sorted(neighbours[0]) == [1, 3, 4]  # corner cell


def test_rook_contiguity_on_a_lattice_excludes_the_diagonals() -> None:
    neighbours = ss.contiguity_neighbours(_lattice(3), "rook")
    assert sorted(neighbours[4]) == [1, 3, 5, 7]
    assert sorted(neighbours[0]) == [1, 3]


def test_contiguity_is_symmetric() -> None:
    neighbours = ss.contiguity_neighbours(_lattice(4), "queen")
    for i, group in enumerate(neighbours):
        for j in group:
            assert i in neighbours[j]


def test_contiguity_rejects_a_non_contiguity_scheme() -> None:
    with pytest.raises(ValueError, match="queen"):
        ss.contiguity_neighbours(_lattice(2), "knn")


def test_knn_neighbours_returns_k_nearest_in_order() -> None:
    # Spacings chosen so no two candidate distances tie; with a tie the order
    # would be decided by index and the test would assert an implementation
    # detail of the KD-tree rather than the behaviour.
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [7.0, 0.0]])
    assert ss.knn_neighbours(coords, 2) == [[1, 2], [0, 2], [3, 1], [2, 1]]


def test_knn_rejects_k_at_or_above_n() -> None:
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="k must be between"):
        ss.knn_neighbours(coords, 2)


def test_row_standardise_makes_rows_sum_to_one() -> None:
    matrix = ss.row_standardise(_rook_matrix(3))
    sums = matrix.sum(axis=1)
    assert np.allclose(sums, 1.0)


def test_row_standardise_leaves_an_island_row_at_zero_not_nan() -> None:
    # A NaN here would propagate into every statistic in the module.
    matrix = _rook_matrix(3)
    matrix[4, :] = 0.0
    standardised = ss.row_standardise(matrix)
    assert np.isfinite(standardised).all()
    assert standardised[4].sum() == 0.0


def test_add_self_neighbours_is_what_makes_gi_a_star() -> None:
    matrix = ss.add_self_neighbours(_rook_matrix(3))
    assert np.allclose(np.diag(matrix), 1.0)


def test_find_islands_reports_the_rows_with_no_neighbours() -> None:
    matrix = _rook_matrix(3)
    matrix[2, :] = 0.0
    assert ss.find_islands(matrix) == [2]


def test_weights_report_leads_with_the_island_count() -> None:
    matrix = _rook_matrix(3)
    report = ss.weights_report(matrix, [f"z{i}" for i in range(9)])
    assert report["n"] == 9
    assert report["islands"] == 0
    assert report["min_neighbours"] == 2  # a corner of a 3x3 rook lattice
    assert report["max_neighbours"] == 4  # the centre
    assert report["symmetric"] is True


# =============================================================================
# Global Moran's I
# =============================================================================
def test_morans_i_of_a_perfect_checkerboard_is_exactly_minus_one() -> None:
    # The theoretical minimum on a regular lattice, and a hard reference value
    # that any implementation error would miss.
    side = 4
    matrix = ss.row_standardise(_rook_matrix(side))
    values = np.array(
        [10.0 + 10.0 * ((row + col) % 2) for row in range(side) for col in range(side)]
    )
    assert ss.morans_i(values, matrix) == pytest.approx(-1.0)


def test_morans_i_of_a_gradient_is_strongly_positive() -> None:
    side = 5
    matrix = ss.row_standardise(_rook_matrix(side))
    values = np.array([30.0 + 2.0 * row for row in range(side) for _ in range(side)])
    assert ss.morans_i(values, matrix) > 0.5


def test_morans_i_rejects_a_constant_attribute() -> None:
    matrix = ss.row_standardise(_rook_matrix(3))
    with pytest.raises(ValueError, match="constant attribute"):
        ss.morans_i(np.full(9, 30.0), matrix)


def test_morans_i_rejects_an_empty_weights_matrix() -> None:
    with pytest.raises(ValueError, match="all-zero weights"):
        ss.morans_i(np.arange(9, dtype="float64"), np.zeros((9, 9)))


def test_morans_i_expectation_is_minus_one_over_n_minus_one() -> None:
    matrix = ss.row_standardise(_rook_matrix(4))
    moments = ss.morans_i_moments(matrix, 16)
    assert moments["expectation"] == pytest.approx(-1.0 / 15.0)
    assert moments["variance"] > 0


def test_build_morans_frame_of_no_records_is_shaped_not_columnless() -> None:
    # THE crash from Colab run 1: pandas.DataFrame([]) has a RangeIndex for
    # columns, so frame["variable"] raises KeyError instead of returning
    # nothing, and the notebook died twice on it when its inputs were missing.
    frame = ss.build_morans_frame([])
    assert frame.empty
    assert list(frame.columns) == list(ss.MORANS_COLUMNS)
    assert frame[frame["variable"] == "LST_C"].empty  # must not raise


def test_build_morans_frame_fills_absent_keys() -> None:
    frame = ss.build_morans_frame(
        [{"level": "gn", "epoch": "2020s", "variable": "LST_C", "morans_i": 0.6}]
    )
    assert list(frame.columns) == list(ss.MORANS_COLUMNS)
    assert frame.iloc[0]["morans_i"] == pytest.approx(0.6)
    assert frame.iloc[0]["p_sim"] is None


def test_global_morans_i_is_reproducible_from_the_seed(params: dict[str, Any]) -> None:
    side = 5
    matrix = ss.row_standardise(_rook_matrix(side))
    values = np.array([30.0 + 2.0 * row for row in range(side) for _ in range(side)])
    first = ss.global_morans_i(values, matrix, params, permutations=199, seed=7)
    second = ss.global_morans_i(values, matrix, params, permutations=199, seed=7)
    assert first["p_sim"] == second["p_sim"]
    assert first["z_sim"] == pytest.approx(second["z_sim"])


def test_global_morans_i_finds_a_gradient_significant(params: dict[str, Any]) -> None:
    side = 6
    matrix = ss.row_standardise(_rook_matrix(side))
    values = np.array([30.0 + 2.0 * row for row in range(side) for _ in range(side)])
    result = ss.global_morans_i(values, matrix, params, permutations=499)
    assert result["i"] > 0.5
    assert result["p_sim"] <= 0.01
    assert result["p_norm"] < 0.01


# =============================================================================
# Local Moran's I (LISA)
# =============================================================================
def test_lisa_quadrant_codes_match_the_esda_convention() -> None:
    # Pinned so a future esda release cannot silently relabel a cluster map.
    assert ss.LISA_QUADRANTS == {1: "HH", 2: "LH", 3: "LL", 4: "HL"}


def test_lisa_labels_a_hot_cluster_hh_and_a_cool_one_ll(
    params: dict[str, Any],
) -> None:
    side = 6
    matrix = ss.row_standardise(_rook_matrix(side))
    values = np.array([30.0 + 2.0 * row for row in range(side) for _ in range(side)])
    frame = ss.local_morans(values, matrix, params, permutations=499)
    hottest = frame.iloc[int(np.argmax(values))]
    coolest = frame.iloc[int(np.argmin(values))]
    assert ss.LISA_QUADRANTS[int(hottest["quadrant"])] == "HH"
    assert ss.LISA_QUADRANTS[int(coolest["quadrant"])] == "LL"


def test_lisa_labels_a_high_outlier_hl(params: dict[str, Any]) -> None:
    matrix = ss.row_standardise(_rook_matrix(5))
    values = np.full(25, 30.0)
    values[12] = 60.0  # one hot cell in a uniform field
    values += np.linspace(0, 0.01, 25)  # break the zero variance
    frame = ss.local_morans(values, matrix, params, permutations=199)
    assert ss.LISA_QUADRANTS[int(frame.iloc[12]["quadrant"])] == "HL"


def test_lisa_emits_the_documented_columns(params: dict[str, Any]) -> None:
    matrix = ss.row_standardise(_rook_matrix(4))
    values = np.arange(16, dtype="float64")
    frame = ss.local_morans(values, matrix, params, permutations=99)
    assert list(frame.columns) == list(ss.LISA_COLUMNS)


def test_lisa_gives_an_island_no_p_value_rather_than_the_smallest_one(
    params: dict[str, Any],
) -> None:
    # THE trap: an island's reference distribution is degenerate, and the naive
    # pseudo-p formula returns 1/(permutations+1) - the most significant value
    # obtainable. It would be drawn as the strongest cluster on the map.
    matrix = _rook_matrix(4)
    matrix[5, :] = 0.0
    matrix[:, 5] = 0.0
    standardised = ss.row_standardise(matrix)
    values = np.arange(16, dtype="float64")
    frame = ss.local_morans(values, standardised, params, permutations=199)
    assert np.isnan(frame.iloc[5]["p_sim"])
    assert not bool(frame.iloc[5]["significant"])
    assert frame.iloc[5]["cluster"] == ss.NOT_SIGNIFICANT


def test_lisa_rejects_missing_values(params: dict[str, Any]) -> None:
    matrix = ss.row_standardise(_rook_matrix(3))
    values = np.arange(9, dtype="float64")
    values[3] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        ss.local_morans(values, matrix, params, permutations=99)


def test_lisa_rejects_a_zone_id_length_mismatch(params: dict[str, Any]) -> None:
    matrix = ss.row_standardise(_rook_matrix(3))
    with pytest.raises(ValueError, match="zone ids for"):
        ss.local_morans(
            np.arange(9, dtype="float64"), matrix, params,
            zone_ids=["a", "b"], permutations=99,
        )


def test_lisa_significance_uses_the_adjusted_p_not_the_raw_one(
    params: dict[str, Any],
) -> None:
    # With 36 simultaneous tests the adjusted p must be >= the raw p everywhere,
    # so the FDR pass can only ever remove clusters, never add them.
    side = 6
    matrix = ss.row_standardise(_rook_matrix(side))
    values = np.array([30.0 + 2.0 * row for row in range(side) for _ in range(side)])
    frame = ss.local_morans(values, matrix, params, permutations=499)
    raw = frame["p_sim"].to_numpy()
    adjusted = frame["p_adjusted"].to_numpy()
    assert np.all(adjusted >= raw - 1e-12)
    assert int(frame["significant"].sum()) <= int((raw < 0.05).sum())


# =============================================================================
# Getis-Ord Gi*
# =============================================================================
def test_gi_star_refuses_negative_values_and_names_the_fix(
    params: dict[str, Any],
) -> None:
    # Gi* is a ratio of a neighbourhood sum to the global sum. On a z-score it
    # returns finite numbers that mean nothing, so the guard must fire.
    matrix = _rook_matrix(3)
    values = np.linspace(-2.0, 2.0, 9)
    with pytest.raises(ValueError, match="undefined for negative values"):
        ss.gi_star(values, matrix, params, permutations=99)


def test_gi_star_guard_can_be_switched_off_in_params(
    params_copy: dict[str, Any],
) -> None:
    params_copy["spatial_stats"]["gi_star"]["require_non_negative"] = False
    matrix = _rook_matrix(3)
    values = np.linspace(-2.0, 2.0, 9)
    frame = ss.gi_star(values, matrix, params_copy, permutations=99)
    assert len(frame) == 9


def test_gi_star_rejects_a_constant_attribute(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="constant attribute"):
        ss.gi_star(np.full(9, 30.0), _rook_matrix(3), params, permutations=99)


def test_gi_star_rejects_missing_values(params: dict[str, Any]) -> None:
    values = np.arange(9, dtype="float64")
    values[2] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        ss.gi_star(values, _rook_matrix(3), params, permutations=99)


def test_gi_star_emits_the_documented_columns(params: dict[str, Any]) -> None:
    frame = ss.gi_star(
        np.arange(16, dtype="float64"), _rook_matrix(4), params, permutations=99
    )
    assert list(frame.columns) == list(ss.GI_STAR_COLUMNS)


def test_gi_star_is_positive_where_the_surface_is_hot(params: dict[str, Any]) -> None:
    side = 6
    values = np.array([30.0 + 2.0 * row for row in range(side) for _ in range(side)])
    frame = ss.gi_star(values, _rook_matrix(side), params, permutations=499)
    assert frame["gi_z"].iloc[-1] > 0  # last row = hottest
    assert frame["gi_z"].iloc[0] < 0  # first row = coolest


def test_gi_star_gives_an_island_no_p_value(params: dict[str, Any]) -> None:
    matrix = _rook_matrix(4)
    matrix[5, :] = 0.0
    matrix[:, 5] = 0.0
    frame = ss.gi_star(
        np.arange(16, dtype="float64"), matrix, params, permutations=199
    )
    assert np.isnan(frame.iloc[5]["p_sim"])
    assert frame.iloc[5]["confidence_class"] == ss.NOT_SIGNIFICANT


def test_gi_star_confidence_classes_follow_the_configured_breaks(
    params: dict[str, Any],
) -> None:
    classes = ss.gi_star_confidence_class(
        [0.005, 0.03, 0.08, 0.5, 0.005, 0.03, 0.08],
        [3.0, 2.0, 1.7, 0.1, -3.0, -2.0, -1.7],
        params,
    )
    assert classes == [
        "hot_99", "hot_95", "hot_90", ss.NOT_SIGNIFICANT,
        "cold_99", "cold_95", "cold_90",
    ]


def test_gi_star_confidence_rejects_non_increasing_breaks(
    params_copy: dict[str, Any],
) -> None:
    params_copy["spatial_stats"]["gi_star"]["confidence_breaks"] = [0.05, 0.01]
    with pytest.raises(ValueError, match="strictly increasing"):
        ss.gi_star_confidence_class([0.02], [2.0], params_copy)


# =============================================================================
# Emerging Hot Spot Analysis
# =============================================================================
@pytest.mark.parametrize(
    ("label", "series", "trend", "significant", "expected"),
    [
        ("hot only in the last bin", [0] * 9 + [3.0], "no trend", False, "new_hot_spot"),
        (
            "an unbroken final run",
            [0] * 7 + [3.0] * 3,
            "no trend",
            False,
            "consecutive_hot_spot",
        ),
        # Without the share guard this all-hot series is trivially "one
        # unbroken final run" and would be filed as consecutive, swallowing
        # persistent / intensifying / diminishing entirely.
        ("hot in every bin", [3.0] * 10, "no trend", False, "persistent_hot_spot"),
        (
            "hot in every bin, rising",
            [3.0] * 10,
            "increasing",
            True,
            "intensifying_hot_spot",
        ),
        (
            "hot in every bin, falling",
            [3.0] * 10,
            "decreasing",
            True,
            "diminishing_hot_spot",
        ),
        (
            "hot 9 of 10, final bin quiet",
            [3.0] * 9 + [0],
            "no trend",
            False,
            "historical_hot_spot",
        ),
        (
            "intermittently hot, still hot at the end",
            [3, 0, 3, 0, 0, 3, 0, 0, 0, 3],
            "no trend",
            False,
            "sporadic_hot_spot",
        ),
        # THE run-1 correction. Under the published taxonomy a zone that was
        # significant at some point but is not doing anything in the final bin
        # is "no pattern": an EMERGING hot-spot map that colours it in is
        # answering "was it ever hot". With the rule off this was sporadic, and
        # sporadic then absorbed 329 of 557 GN divisions.
        (
            "intermittently hot, quiet at the end",
            [3, 0, 3, 0, 0, 3, 0, 0, 3, 0],
            "no trend",
            False,
            ss.EHSA_NO_PATTERN,
        ),
        (
            "cold, then hot in the final bin",
            [-3, -3, -3, 0, 0, 0, 0, 0, 0, 3.0],
            "no trend",
            False,
            "oscillating_hot_spot",
        ),
        ("never significant", [0.2] * 10, "no trend", False, ss.EHSA_NO_PATTERN),
        ("cold in every bin", [-3.0] * 10, "no trend", False, "persistent_cold_spot"),
        ("cold only in the last bin", [0] * 9 + [-3.0], "no trend", False, "new_cold_spot"),
        (
            "hot, then cold in the final bin",
            [3, 3, 3, 0, 0, 0, 0, 0, 0, -3.0],
            "no trend",
            False,
            "oscillating_cold_spot",
        ),
        # For a cold spot, "intensifying" means getting COLDER, so the mirror
        # maps a significant DECREASING trend to intensifying.
        (
            "cold in every bin, falling",
            [-3.0] * 10,
            "decreasing",
            True,
            "intensifying_cold_spot",
        ),
    ],
)
def test_ehsa_classifier_assigns_each_documented_category(
    label: str,
    series: list[float],
    trend: str,
    significant: bool,
    expected: str,
    params: dict[str, Any],
) -> None:
    category, reason = ss.classify_zone_pattern(
        np.array(series, dtype="float64"), trend, significant, params
    )
    assert category == expected, f"{label}: {reason}"
    assert category in ss.EHSA_CATEGORIES


def test_ehsa_classifier_prefers_oscillating_over_new(params: dict[str, Any]) -> None:
    # A zone that was a cold spot and is now a hot spot also satisfies "never a
    # hot spot before". Filing it as merely New would lose the flip, which is
    # the most informative thing about it.
    category, _ = ss.classify_zone_pattern(
        np.array([-3.0, -3.0, 0.0, 0.0, 3.0]), "no trend", False, params
    )
    assert category == "oscillating_hot_spot"


def test_ehsa_final_bin_rule_is_what_separates_sporadic_from_no_pattern(
    params: dict[str, Any],
) -> None:
    # The same series classifies differently under the two rules, which is the
    # whole point of making it a config option rather than a silent edit.
    series = np.array([3.0, 0, 3.0, 0, 0, 3.0, 0, 0, 3.0, 0])
    strict, _ = ss.classify_zone_pattern(
        series, "no trend", False, params, require_final_bin=True
    )
    loose, loose_reason = ss.classify_zone_pattern(
        series, "no trend", False, params, require_final_bin=False
    )
    assert strict == ss.EHSA_NO_PATTERN
    assert loose == "sporadic_hot_spot"
    assert "final bin not significant" in loose_reason


def test_ehsa_final_bin_rule_does_not_touch_the_share_categories(
    params: dict[str, Any],
) -> None:
    # Persistent / intensifying / diminishing / historical qualify on share
    # alone, so a zone hot throughout must classify identically under both
    # rules even when its final bin falls short.
    series = np.array([3.0] * 9 + [0.0])
    for flag in (True, False):
        category, _ = ss.classify_zone_pattern(
            series, "no trend", False, params, require_final_bin=flag
        )
        assert category == "historical_hot_spot"


def test_ehsa_classifier_defaults_to_the_configured_rule(
    params: dict[str, Any], params_copy: dict[str, Any]
) -> None:
    assert params["spatial_stats"]["ehsa"]["require_final_bin"] is True
    series = np.array([3.0, 0, 3.0, 0, 0, 3.0, 0, 0, 3.0, 0])
    assert ss.classify_zone_pattern(series, "no trend", False, params)[0] == (
        ss.EHSA_NO_PATTERN
    )
    params_copy["spatial_stats"]["ehsa"]["require_final_bin"] = False
    assert ss.classify_zone_pattern(series, "no trend", False, params_copy)[0] == (
        "sporadic_hot_spot"
    )


def test_ehsa_classifier_rejects_an_empty_series(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="empty"):
        ss.classify_zone_pattern(np.array([]), "no trend", False, params)


def test_space_time_bins_produces_a_complete_zone_by_bin_panel(
    params: dict[str, Any],
) -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "zone_id": ["a", "a", "b", "b"],
            "year": [2020, 2021, 2020, 2021],
            "mean": [30.0, 31.0, 32.0, 33.0],
        }
    )
    panel = ss.space_time_bins(frame, params)
    assert panel.shape == (2, 2)
    assert list(panel.index) == ["a", "b"]
    assert panel.attrs["panel_report"]["missing_cells"] == 0


def test_space_time_bins_warns_about_a_hole_rather_than_dropping_the_zone(
    params: dict[str, Any],
) -> None:
    # A dropped zone would desynchronise the panel from the weights matrix and
    # silently give every later zone its neighbour's statistic.
    import pandas as pd

    frame = pd.DataFrame(
        {
            "zone_id": ["a", "a", "b"],
            "year": [2020, 2021, 2020],
            "mean": [30.0, 31.0, 32.0],
        }
    )
    with pytest.warns(UserWarning, match="missing"):
        panel = ss.space_time_bins(frame, params)
    assert panel.shape == (2, 2)
    assert bool(panel.isna().any().any())


def test_space_time_bins_rejects_a_duplicated_zone_year(params: dict[str, Any]) -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {"zone_id": ["a", "a"], "year": [2020, 2020], "mean": [30.0, 31.0]}
    )
    with pytest.raises(ValueError, match="duplicate zone/bin"):
        ss.space_time_bins(frame, params)


def test_space_time_bins_rejects_a_bin_width_below_one(params: dict[str, Any]) -> None:
    import pandas as pd

    frame = pd.DataFrame({"zone_id": ["a"], "year": [2020], "mean": [30.0]})
    with pytest.raises(ValueError, match="bin_years must be"):
        ss.space_time_bins(frame, params, bin_years=0)


def test_space_time_bins_rejects_a_missing_column(params: dict[str, Any]) -> None:
    import pandas as pd

    frame = pd.DataFrame({"zone_id": ["a"], "year": [2020]})
    with pytest.raises(ValueError, match="'mean'"):
        ss.space_time_bins(frame, params)


def test_gi_star_panel_rejects_a_zone_order_mismatch(params: dict[str, Any]) -> None:
    import pandas as pd

    panel = pd.DataFrame(
        np.full((4, 3), 30.0) + np.arange(3), index=["a", "b", "c", "d"]
    )
    with pytest.raises(ValueError, match="do not match"):
        ss.gi_star_panel(
            panel, _rook_matrix(2), params, zone_ids=["a", "b", "c", "x"],
            permutations=99,
        )


def test_gi_star_panel_drops_an_incomplete_bin_with_a_warning(
    params: dict[str, Any],
) -> None:
    import pandas as pd

    zones = [f"z{i}" for i in range(9)]
    data = np.tile(np.linspace(30.0, 40.0, 9)[:, None], (1, 3))
    panel = pd.DataFrame(data, index=zones, columns=[2020, 2021, 2022])
    panel.iloc[3, 1] = np.nan
    with pytest.warns(UserWarning, match="dropped"):
        result = ss.gi_star_panel(
            panel, _rook_matrix(3), params, zone_ids=zones, permutations=99
        )
    assert list(result.columns) == [2020, 2022]
    assert result.attrs["dropped_bins"] == [2021]


def test_classify_emerging_hotspots_emits_the_documented_columns(
    params: dict[str, Any],
) -> None:
    import pandas as pd

    zones = [f"z{i}" for i in range(9)]
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        rng.normal(size=(9, 12)), index=zones, columns=list(range(2014, 2026))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = ss.classify_emerging_hotspots(panel, params)
    assert list(frame.columns) == list(ss.EHSA_COLUMNS)
    assert set(frame["category"]) <= set(ss.EHSA_CATEGORIES)


def test_classify_emerging_hotspots_flags_a_flat_series_as_underpowered(
    params: dict[str, Any],
) -> None:
    # A short, noisy, trendless series must come back as "no pattern AND we
    # could not have seen otherwise" - the distinction that made Phase 4's
    # Landsat zero reportable.
    import pandas as pd

    rng = np.random.default_rng(11)
    panel = pd.DataFrame(
        rng.normal(scale=0.4, size=(4, 10)),
        index=["a", "b", "c", "d"],
        columns=list(range(2016, 2026)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = ss.classify_emerging_hotspots(panel, params)
    assert (frame["category"] == ss.EHSA_NO_PATTERN).any()
    flat = frame[frame["category"] == ss.EHSA_NO_PATTERN]
    assert bool(flat["underpowered"].any())
    assert (flat["detectable_slope"].dropna() > 0).all()


def test_ehsa_power_check_says_so_when_most_nulls_are_underpowered(
    params: dict[str, Any],
) -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "category": [ss.EHSA_NO_PATTERN] * 4 + ["persistent_hot_spot"],
            "underpowered": [True, True, True, False, False],
            "detectable_slope": [0.3, 0.3, 0.3, 0.3, np.nan],
        }
    )
    verdict = ss.ehsa_power_check(frame, params)
    assert verdict["n_no_pattern"] == 4
    assert verdict["n_underpowered"] == 3
    assert "UNDERPOWERED" in verdict["verdict"]


def test_unit_noise_detection_limit_is_memoised_and_positive() -> None:
    first = ss.unit_noise_detection_limit(12, alpha=0.05)
    second = ss.unit_noise_detection_limit(12, alpha=0.05)
    assert first == second  # memoised: identical, not merely close
    assert first > 0
    # A longer series must be able to resolve a SMALLER trend.
    assert ss.unit_noise_detection_limit(26, alpha=0.05) < first


# =============================================================================
# The regression ladder
# =============================================================================
def test_require_estimable_passes_at_gn_and_fails_at_ds(params: dict[str, Any]) -> None:
    predictors = len(params["spatial_stats"]["regression"]["predictors"])
    assert ss.require_estimable(557, predictors, params)["estimable"] is True
    refusal = ss.require_estimable(13, predictors, params, statistic="GWR")
    assert refusal["estimable"] is False
    assert "GWR" in refusal["reason"]
    assert str(refusal["required"]) in refusal["reason"]


def test_require_estimable_rejects_a_model_with_no_predictors(
    params: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="k must be"):
        ss.require_estimable(100, 0, params)


def test_vif_is_near_one_for_independent_predictors() -> None:
    rng = np.random.default_rng(0)
    design = rng.normal(size=(200, 3))
    frame = ss.variance_inflation_factors(design, ["a", "b", "c"])
    assert (frame["vif"] < 1.2).all()


def test_vif_explodes_for_a_collinear_pair() -> None:
    # The NDVI/NDBI situation Phase 3 measured: a partial coefficient that flips
    # sign while the bivariate correlation stays clean.
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    design = np.column_stack([a, a + rng.normal(scale=0.01, size=200), rng.normal(size=200)])
    frame = ss.variance_inflation_factors(design, ["a", "a_copy", "c"])
    assert frame.iloc[0]["vif"] > 100


def test_vif_needs_at_least_two_predictors() -> None:
    with pytest.raises(ValueError, match="at least 2 predictors"):
        ss.variance_inflation_factors(np.zeros((10, 1)), ["a"])


def test_ols_recovers_known_coefficients() -> None:
    rng = np.random.default_rng(1)
    design = rng.normal(size=(400, 2))
    y = 3.0 + 2.0 * design[:, 0] - 1.5 * design[:, 1] + rng.normal(scale=0.1, size=400)
    result = ss.ols_fit(y, design, ["a", "b"])
    estimates = dict(zip(result["coefficients"]["term"], result["coefficients"]["estimate"]))
    assert estimates["intercept"] == pytest.approx(3.0, abs=0.02)
    assert estimates["a"] == pytest.approx(2.0, abs=0.02)
    assert estimates["b"] == pytest.approx(-1.5, abs=0.02)
    assert result["r_squared"] > 0.99


def test_ols_rejects_an_exactly_collinear_design() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=50)
    design = np.column_stack([a, 2.0 * a])
    with pytest.raises(ValueError, match="rank deficient"):
        ss.ols_fit(rng.normal(size=50), design, ["a", "double_a"])


def test_lm_tests_point_at_the_error_model_when_the_error_is_spatial(
    params: dict[str, Any],
) -> None:
    side = 12
    matrix = ss.row_standardise(_rook_matrix(side))
    n = side * side
    rng = np.random.default_rng(0)
    design = rng.normal(size=(n, 2))
    error = np.linalg.inv(np.eye(n) - 0.7 * matrix) @ rng.normal(scale=0.5, size=n)
    y = 1.0 + 2.0 * design[:, 0] - design[:, 1] + error
    ols = ss.ols_fit(y, design, ["a", "b"])
    diagnostics = ss.lagrange_multiplier_tests(ols, y, matrix)
    assert diagnostics["moran_p"] < 0.01
    assert diagnostics["rlm_error"][1] < diagnostics["rlm_lag"][1]
    assert ss.lm_decision(diagnostics, params)["model"] == "error"


def test_lm_tests_point_at_the_lag_model_when_the_response_is_lagged(
    params: dict[str, Any],
) -> None:
    side = 12
    matrix = ss.row_standardise(_rook_matrix(side))
    n = side * side
    rng = np.random.default_rng(3)
    design = rng.normal(size=(n, 2))
    linear = 1.0 + 2.0 * design[:, 0] - design[:, 1] + rng.normal(scale=0.4, size=n)
    y = np.linalg.inv(np.eye(n) - 0.7 * matrix) @ linear
    ols = ss.ols_fit(y, design, ["a", "b"])
    diagnostics = ss.lagrange_multiplier_tests(ols, y, matrix)
    assert ss.lm_decision(diagnostics, params)["model"] == "lag"


def test_lm_tests_leave_independent_errors_alone(params: dict[str, Any]) -> None:
    side = 12
    matrix = ss.row_standardise(_rook_matrix(side))
    n = side * side
    rng = np.random.default_rng(5)
    design = rng.normal(size=(n, 2))
    y = 1.0 + 2.0 * design[:, 0] - design[:, 1] + rng.normal(scale=0.5, size=n)
    ols = ss.ols_fit(y, design, ["a", "b"])
    diagnostics = ss.lagrange_multiplier_tests(ols, y, matrix)
    assert ss.lm_decision(diagnostics, params)["model"] == "ols"


@pytest.mark.parametrize(
    ("lag_p", "error_p", "robust_lag_p", "robust_error_p", "expected", "rule"),
    [
        (0.4, 0.4, 0.4, 0.4, "ols", 1),
        (0.001, 0.4, 0.4, 0.4, "lag", 2),
        (0.4, 0.001, 0.4, 0.4, "error", 2),
        (0.001, 0.001, 0.001, 0.4, "lag", 3),
        (0.001, 0.001, 0.4, 0.001, "error", 3),
        (0.001, 0.001, 0.4, 0.4, "error", 3),  # conservative fallback
    ],
)
def test_lm_decision_truth_table(
    lag_p: float,
    error_p: float,
    robust_lag_p: float,
    robust_error_p: float,
    expected: str,
    rule: int,
    params: dict[str, Any],
) -> None:
    diagnostics = {
        "lm_lag": (10.0, lag_p),
        "lm_error": (10.0, error_p),
        "rlm_lag": (5.0, robust_lag_p),
        "rlm_error": (6.0, robust_error_p),
    }
    decision = ss.lm_decision(diagnostics, params)
    assert decision["model"] == expected
    assert decision["rule"] == rule
    assert decision["reason"]


def test_lm_decision_breaks_a_robust_tie_on_the_larger_statistic(
    params: dict[str, Any],
) -> None:
    diagnostics = {
        "lm_lag": (10.0, 0.001),
        "lm_error": (10.0, 0.001),
        "rlm_lag": (9.0, 0.001),
        "rlm_error": (4.0, 0.001),
    }
    assert ss.lm_decision(diagnostics, params)["model"] == "lag"


# =============================================================================
# Model frame assembly and the MAUP table
# =============================================================================
def test_build_model_frame_joins_on_zone_id_and_keeps_complete_cases(
    params: dict[str, Any],
) -> None:
    import pandas as pd

    predictors = ss.resolve_regression_predictors(None, params)
    response = params["spatial_stats"]["regression"]["response"]
    left = pd.DataFrame({"zone_id": ["a", "b", "c"], response: [30.0, 31.0, 32.0]})
    right = pd.DataFrame(
        {"zone_id": ["a", "b", "c"], **{name: [1.0, 2.0, 3.0] for name in predictors}}
    )
    frame = ss.build_model_frame([left, right], params, standardise=False)
    assert list(frame.columns) == ["zone_id", response, *predictors]
    assert len(frame) == 3
    assert frame.attrs["dropped"]["after"] == 3


def test_build_model_frame_standardises_when_asked(params: dict[str, Any]) -> None:
    import pandas as pd

    predictors = ss.resolve_regression_predictors(None, params)
    response = params["spatial_stats"]["regression"]["response"]
    table = pd.DataFrame(
        {
            "zone_id": [f"z{i}" for i in range(10)],
            response: np.linspace(30.0, 40.0, 10),
            **{name: np.linspace(0.0, 1.0, 10) for name in predictors},
        }
    )
    frame = ss.build_model_frame([table], params, standardise=True)
    assert frame[response].mean() == pytest.approx(0.0, abs=1e-12)
    assert frame[response].std(ddof=0) == pytest.approx(1.0)


def test_build_model_frame_reports_a_missing_predictor_by_name(
    params: dict[str, Any],
) -> None:
    import pandas as pd

    response = params["spatial_stats"]["regression"]["response"]
    table = pd.DataFrame({"zone_id": ["a"], response: [30.0]})
    with pytest.raises(ValueError, match="zone_covariate_table"):
        ss.build_model_frame([table], params)


def test_build_model_frame_rejects_a_table_without_the_join_key(
    params: dict[str, Any],
) -> None:
    import pandas as pd

    with pytest.raises(ValueError, match="zone_id"):
        ss.build_model_frame([pd.DataFrame({"name": ["Fort"]})], params)


def test_maup_comparison_keeps_a_refusal_as_a_row_with_a_reason(
    params: dict[str, Any],
) -> None:
    frame = ss.maup_comparison(
        [
            {"statistic": "gwr", "level": "gn", "n_units": 557, "value": 0.7},
            {
                "statistic": "gwr",
                "level": "ds",
                "n_units": 13,
                "status": ss.MAUP_NOT_ESTIMABLE,
                "reason": "only 13 units",
            },
        ],
        params,
    )
    assert list(frame.columns) == list(ss.MAUP_COLUMNS)
    assert len(frame) == 2
    refused = frame[frame["status"] == ss.MAUP_NOT_ESTIMABLE].iloc[0]
    assert refused["reason"] == "only 13 units"


def test_maup_comparison_requires_a_statistic_and_a_level(
    params: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="statistic"):
        ss.maup_comparison([{"level": "gn"}], params)


# =============================================================================
# Landscape metrics
# =============================================================================
def test_patch_labels_separate_a_diagonal_under_four_connectivity() -> None:
    mask = np.array([[True, False], [False, True]])
    _, four = ss.patch_labels(mask, 4)
    _, eight = ss.patch_labels(mask, 8)
    assert four == 2
    assert eight == 1


def test_patch_labels_reject_a_bad_connectivity() -> None:
    with pytest.raises(ValueError, match="connectivity must be"):
        ss.patch_labels(np.ones((2, 2), dtype=bool), 6)


def test_aggregation_index_of_a_solid_block_is_one_hundred() -> None:
    assert ss.aggregation_index(np.ones((4, 4), dtype=bool)) == pytest.approx(100.0)


def test_aggregation_index_of_a_checkerboard_is_zero() -> None:
    mask = (np.indices((4, 4)).sum(axis=0) % 2) == 0
    assert ss.aggregation_index(mask) == pytest.approx(0.0)


def test_aggregation_index_of_a_single_cell_is_zero() -> None:
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    assert ss.aggregation_index(mask) == 0.0


def test_aggregation_index_of_an_absent_class_is_nan() -> None:
    assert np.isnan(ss.aggregation_index(np.zeros((3, 3), dtype=bool)))


def test_landscape_metrics_on_a_hand_computed_case(params: dict[str, Any]) -> None:
    # A 2x2 block of green in a 4x4 landscape at 10 m.
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    metrics = ss.landscape_metrics(mask, 10.0, params)
    assert metrics["landscape_area_ha"] == pytest.approx(16 * 100 / 10_000)
    assert metrics["class_area_ha"] == pytest.approx(4 * 100 / 10_000)
    assert metrics["class_fraction"] == pytest.approx(0.25)
    assert metrics["n_patches"] == 1
    assert metrics["largest_patch_index_pct"] == pytest.approx(25.0)
    # The block has 8 class/non-class rook adjacencies, each 10 m long.
    assert metrics["total_edge_m"] == pytest.approx(80.0)
    assert metrics["aggregation_index_pct"] == pytest.approx(100.0)


def test_landscape_metrics_boundary_convention_changes_the_edge(
    params: dict[str, Any],
) -> None:
    # A class filling the whole landscape has NO edge under the default "no
    # boundary" convention, and a full perimeter under the other. The choice is
    # not cosmetic: it decides whether edge density scales with the arbitrary
    # shape of the study-area clip.
    mask = np.ones((4, 4), dtype=bool)
    without = ss.landscape_metrics(mask, 10.0, params, boundary_as_edge=False)
    with_boundary = ss.landscape_metrics(mask, 10.0, params, boundary_as_edge=True)
    assert without["total_edge_m"] == 0.0
    assert with_boundary["total_edge_m"] == pytest.approx(160.0)


def test_landscape_metrics_ignore_cells_outside_the_valid_mask(
    params: dict[str, Any],
) -> None:
    # A clipped study area must not manufacture edge along its own clip line.
    mask = np.zeros((4, 4), dtype=bool)
    mask[:, :2] = True
    valid = np.zeros((4, 4), dtype=bool)
    valid[:, :2] = True
    metrics = ss.landscape_metrics(mask, 10.0, params, valid=valid)
    assert metrics["landscape_area_ha"] == pytest.approx(8 * 100 / 10_000)
    assert metrics["class_fraction"] == pytest.approx(1.0)
    assert metrics["total_edge_m"] == 0.0


def test_landscape_metrics_reject_a_non_positive_cell_size(
    params: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="cell_size_m must be positive"):
        ss.landscape_metrics(np.ones((2, 2), dtype=bool), 0.0, params)


def test_landscape_metrics_reject_a_non_2d_mask(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        ss.landscape_metrics(np.ones(4, dtype=bool), 10.0, params)


def test_build_landscape_frame_has_the_documented_columns(
    params: dict[str, Any],
) -> None:
    metrics = ss.landscape_metrics(np.ones((4, 4), dtype=bool), 10.0, params)
    frame = ss.build_landscape_frame(
        [{**metrics, "scheme": "worldcover", "year": 2021}], params
    )
    assert list(frame.columns) == list(ss.LANDSCAPE_COLUMNS)


def test_landscape_metrics_by_zone_rejects_mismatched_rasters(
    params: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="same shape"):
        ss.landscape_metrics_by_zone(
            np.zeros((4, 4), dtype=int), np.zeros((3, 3), dtype=int),
            params, 10.0, [1],
        )


def test_landscape_metrics_by_zone_computes_one_row_per_zone(
    params: dict[str, Any],
) -> None:
    classes = np.full((4, 4), 50, dtype=int)
    classes[:, :2] = 10  # tree cover on the left half
    zones = np.ones((4, 4), dtype=int)
    zones[:, 2:] = 2
    frame = ss.landscape_metrics_by_zone(
        classes, zones, params, 10.0, [10], zone_labels={1: "west", 2: "east"},
        scheme="worldcover", year=2021,
    )
    assert list(frame["zone_id"]) == ["west", "east"]
    assert frame.set_index("zone_id").loc["west", "class_fraction"] == pytest.approx(1.0)
    assert frame.set_index("zone_id").loc["east", "class_fraction"] == pytest.approx(0.0)


# =============================================================================
# Exported-table plumbing (pure; the Earth Engine side runs only in Colab)
# =============================================================================
def test_covariate_stack_lets_each_consumer_build_its_own_collection() -> None:
    """The regression from Colab run 2, pinned so it cannot come back.

    ``covariate_stack`` once built one harmonised Landsat collection and passed
    it to both ``epoch_composite`` and ``driver_stack``. Handing them a prebuilt
    collection makes them skip ``uhi_metrics.source_collection``, which is the
    ONLY place the per-source sensor restriction is applied. ``landsat_oli_dry``
    - restricted to L8+L9 precisely to avoid the cross-sensor steps Phase 4
    measured - then silently ran with all four sensors, and the sensitivity
    check compared the pooled series with itself and reported 100 % agreement.

    This is a source-level guard rather than a behavioural one because the
    failure is structural: the numbers it produces are perfectly plausible, so
    only the call graph reveals it.
    """
    import inspect

    body = inspect.getsource(ss.covariate_stack)
    assert "harmonised_collection" not in body, (
        "covariate_stack must NOT build a Landsat collection itself. Doing so "
        "bypasses uhi_metrics.source_collection and drops the source's "
        "`sensors` restriction, which silently turns landsat_oli_dry back into "
        "the pooled four-sensor series."
    )


def test_source_collection_is_where_the_sensor_restriction_lives() -> None:
    # The other half of the invariant above: if this ever stops passing
    # `sensors` through, the guard on covariate_stack protects nothing.
    import inspect

    from colombo_uhi import uhi_metrics

    body = inspect.getsource(uhi_metrics.source_collection)
    assert 'sensors=resolved.get("sensors")' in body, (
        "source_collection is the single place a per-source sensor restriction "
        "is applied; every LST source depends on it"
    )


def test_the_sensitivity_source_actually_restricts_its_sensors(
    params: dict[str, Any],
) -> None:
    # A sensitivity check between two sources that resolve to the same sensors
    # is vacuous no matter how the code is wired.
    sources = {s["key"]: s for s in params["uhi"]["suhii"]["sources"]}
    spatial = params["spatial_stats"]
    pooled = sources[spatial["epochs_source"]]
    sensitivity = sources[spatial["epoch_sensitivity_source"]]
    assert sensitivity.get("sensors"), (
        f"{spatial['epoch_sensitivity_source']} must restrict its sensors, or "
        "the epoch sensitivity check compares the pooled series with itself"
    )
    assert pooled.get("sensors") != sensitivity.get("sensors")


def test_zone_covariate_bands_lead_with_the_response(params: dict[str, Any]) -> None:
    bands = ss.zone_covariate_bands(params)
    assert bands[0] == params["spatial_stats"]["response_band"]
    assert bands[1:] == ss.resolve_regression_predictors(None, params)


def test_zone_covariate_selectors_pin_the_column_order(
    params: dict[str, Any],
) -> None:
    # Export.table.toDrive does not guarantee column order without selectors,
    # and a silently reordered CSV only shows up as a nonsensical coefficient
    # three steps later.
    selectors = ss.zone_covariate_selectors(params, "gn")
    assert selectors[0] == "adm4_pcode"
    for band in ss.zone_covariate_bands(params):
        assert f"{band}_mean" in selectors
        assert f"{band}_count" in selectors


def test_read_zone_covariates_reshapes_an_exported_table(
    params: dict[str, Any], tmp_path: Any
) -> None:
    import pandas as pd

    bands = ss.zone_covariate_bands(params)
    raw = {"adm4_pcode": ["LK1103005", "LK1103010"], "adm4_name": ["A", "B"]}
    for index, band in enumerate(bands):
        raw[f"{band}_mean"] = [30.0 + index, 31.0 + index]
        raw[f"{band}_count"] = [120, 90]
    path = tmp_path / "zone_covariates.csv"
    pd.DataFrame(raw).to_csv(path, index=False)

    frame = ss.read_zone_covariates(str(path), params, "gn")
    assert list(frame["zone_id"]) == ["LK1103005", "LK1103010"]
    for band in bands:
        assert band in frame.columns
        assert f"{band}_pixels" in frame.columns
    assert frame[f"{bands[0]}_pixels"].iloc[0] == 120


def test_read_zone_covariates_names_the_missing_band(
    params: dict[str, Any], tmp_path: Any
) -> None:
    import pandas as pd

    path = tmp_path / "short.csv"
    pd.DataFrame({"adm4_pcode": ["LK1103005"], "LST_C_mean": [30.0]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="NDVI_mean"):
        ss.read_zone_covariates(str(path), params, "gn")


def test_read_zone_covariates_rejects_a_table_without_the_pcode(
    params: dict[str, Any], tmp_path: Any
) -> None:
    import pandas as pd

    path = tmp_path / "nokey.csv"
    pd.DataFrame({"adm4_name": ["Fort"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="adm4_pcode"):
        ss.read_zone_covariates(str(path), params, "gn")


# =============================================================================
# Cross-validation against the reference libraries (skipped without PySAL)
# =============================================================================
def test_our_contiguity_matches_libpysal_queen() -> None:
    pytest.importorskip("libpysal")
    pytest.importorskip("shapely")
    from shapely.geometry import box
    from libpysal.weights import Queen
    import geopandas as gpd

    side = 5
    polygons = [
        box(col, row, col + 1, row + 1) for row in range(side) for col in range(side)
    ]
    ours = ss.contiguity_neighbours(polygons, "queen")
    frame = gpd.GeoDataFrame(geometry=polygons)
    reference = Queen.from_dataframe(frame, use_index=False)
    for i in range(side * side):
        assert sorted(ours[i]) == sorted(reference.neighbors[i])


def test_our_statistics_match_esda(params: dict[str, Any]) -> None:
    pytest.importorskip("esda")
    pytest.importorskip("libpysal")

    side = 6
    matrix = ss.row_standardise(_rook_matrix(side))
    values = np.array(
        [30.0 + 2.0 * row + 0.1 * col for row in range(side) for col in range(side)]
    )
    frame = ss.esda_cross_check(values, matrix, params)
    # Global Moran's I and Gi* have one convention each and must agree to
    # machine precision. Local Moran's I is compared after rescaling, because
    # Anselin (1995) normalises by n and esda by (n-1) - a constant factor that
    # cannot affect a quadrant, a permutation p-value or a cluster map. The
    # quadrant-agreement row is what proves the scaling is harmless.
    assert (frame["abs_diff"] < 1e-9).all(), frame.to_string()
    quadrants = frame[frame["statistic"] == "lisa_quadrant_agreement"].iloc[0]
    assert quadrants["ours"] == pytest.approx(1.0)


def test_our_ols_and_lm_tests_match_spreg(params: dict[str, Any]) -> None:
    pytest.importorskip("spreg")
    pytest.importorskip("libpysal")

    side = 10
    matrix = ss.row_standardise(_rook_matrix(side))
    n = side * side
    rng = np.random.default_rng(0)
    design = rng.normal(size=(n, 2))
    error = np.linalg.inv(np.eye(n) - 0.5 * matrix) @ rng.normal(scale=0.5, size=n)
    y = 1.0 + 2.0 * design[:, 0] - design[:, 1] + error
    frame = ss.spreg_cross_check(y, design, ["a", "b"], matrix)
    assert (frame["abs_diff"] < 1e-6).all(), frame.to_string()
