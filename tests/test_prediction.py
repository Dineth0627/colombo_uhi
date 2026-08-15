"""Unit tests for :mod:`colombo_uhi.prediction`.

What is NOT tested here, and why:

* Anything that calls Earth Engine. There are no credentials in the local test
  environment, and a mocked ``ee`` would test the mock. Those functions are
  exercised by ``notebooks/06_prediction.ipynb`` in Colab, and its Step 1 probe
  settles the server-side band names empirically before anything is built on
  them. The **export guard** is tested here, though, precisely because it must
  fire *before* any Earth Engine call - so it can be, and is, proved to raise
  without a network.
* The random forest's accuracy. That is a measurement, not a property; it is
  reported by the notebook and recorded in PROGRESS.md.

The one genuinely destructive failure mode this module has is a predictive
product escaping without validation metrics, because a scenario projection that
looks like a forecast and carries no error bars is exactly what CLAUDE.md
caveat 3 forbids. :func:`colombo_uhi.prediction.require_validated` is therefore
tested against every way it can be given something inadequate.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from colombo_uhi import load_params, prediction


@pytest.fixture(scope="module")
def params() -> dict[str, Any]:
    return load_params()


@pytest.fixture()
def params_copy(params: dict[str, Any]) -> dict[str, Any]:
    """Deep copy for tests that mutate the config."""
    return copy.deepcopy(params)


@pytest.fixture(scope="module")
def classes(params: dict[str, Any]) -> list[int]:
    return prediction.resolve_ca_classes(params)


# --- resolvers ---------------------------------------------------------------


def test_predictors_come_from_params_and_keep_their_order(
    params: dict[str, Any],
) -> None:
    assert prediction.resolve_predictors(None, params) == list(
        params["prediction"]["rf"]["predictors"]
    )


def test_predictors_collapse_duplicates(params: dict[str, Any]) -> None:
    assert prediction.resolve_predictors(["NDVI", "NDBI", "NDVI"], params) == [
        "NDVI",
        "NDBI",
    ]


def test_predictors_reject_an_empty_list(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="at least one predictor"):
        prediction.resolve_predictors([], params)


def test_predictors_reject_the_response(params: dict[str, Any]) -> None:
    response = params["prediction"]["rf"]["response"]
    with pytest.raises(ValueError, match="R2 = 1"):
        prediction.resolve_predictors(["NDVI", response], params)


def test_categorical_predictors_are_a_subset_of_the_predictors(
    params: dict[str, Any],
) -> None:
    # A drift between these two lists would declare an encoding for a variable
    # the model never sees, and nothing downstream would notice.
    declared = set(params["prediction"]["rf"]["categorical"])
    assert declared <= set(prediction.resolve_predictors(None, params))


def test_categorical_rejects_a_name_the_model_never_sees(
    params_copy: dict[str, Any],
) -> None:
    params_copy["prediction"]["rf"]["categorical"] = ["worldcover_class"]
    with pytest.raises(ValueError, match="worldcover_class"):
        prediction.resolve_categorical(params_copy)


def test_rf_settings_resolve_every_hyperparameter(params: dict[str, Any]) -> None:
    settings = prediction.resolve_rf_settings(params)
    for key in (
        "n_trees",
        "variables_per_split",
        "min_leaf_population",
        "bag_fraction",
        "max_nodes",
        "random_seed",
        "response",
        "epoch",
        "source",
        "scale_m",
        "predictors",
        "categorical",
    ):
        assert key in settings


def test_rf_settings_reject_an_unknown_override(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="n_tress"):
        prediction.resolve_rf_settings(params, n_tress=100)


@pytest.mark.parametrize("bag", [0.0, -0.1, 1.5])
def test_rf_settings_reject_a_bag_fraction_outside_the_unit_interval(
    params: dict[str, Any], bag: float
) -> None:
    with pytest.raises(ValueError, match="bag_fraction"):
        prediction.resolve_rf_settings(params, bag_fraction=bag)


def test_rf_settings_reject_a_non_positive_tree_count(
    params: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="n_trees"):
        prediction.resolve_rf_settings(params, n_trees=0)


def test_split_method_is_never_random(params: dict[str, Any]) -> None:
    assert params["prediction"]["split"]["method"] != prediction.RANDOM_SPLIT
    assert prediction.RANDOM_SPLIT not in prediction.SPLIT_METHODS


def test_resolve_split_refuses_a_random_split(params: dict[str, Any]) -> None:
    # THE methodological point of this phase. A random split of a raster sample
    # puts pixels 100 m apart into train AND test, so the reported R2 measures
    # interpolation between neighbours. The refusal names the escape hatch.
    with pytest.raises(ValueError, match="compare_split_strategies"):
        prediction.resolve_split(params, method=prediction.RANDOM_SPLIT)


def test_resolve_split_refuses_an_unknown_method(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="stratified"):
        prediction.resolve_split(params, method="stratified")


def test_ca_classes_are_all_in_the_scheme_legend(
    params: dict[str, Any], classes: list[int]
) -> None:
    legend = params["landcover"][params["prediction"]["ca_markov"]["scheme"]][
        "classes"
    ]
    assert classes
    assert all(code in legend for code in classes)


def test_ca_classes_reject_a_code_outside_the_legend(
    params_copy: dict[str, Any],
) -> None:
    params_copy["prediction"]["ca_markov"]["classes"] = [0, 1, 99]
    with pytest.raises(ValueError, match="99"):
        prediction.resolve_ca_classes(params_copy)


def test_ca_classes_reject_duplicates(params_copy: dict[str, Any]) -> None:
    params_copy["prediction"]["ca_markov"]["classes"] = [0, 1, 1]
    with pytest.raises(ValueError, match="duplicates"):
        prediction.resolve_ca_classes(params_copy)


def test_immutable_classes_are_retained_classes(
    params: dict[str, Any], classes: list[int]
) -> None:
    immutable = params["prediction"]["ca_markov"]["immutable_classes"]
    assert set(immutable) <= set(classes)


# --- scenarios ---------------------------------------------------------------


def test_every_configured_scenario_resolves(params: dict[str, Any]) -> None:
    for name in params["prediction"]["scenarios"]:
        resolved = prediction.resolve_scenario(name, params)
        assert resolved["key"] == name
        assert 0.0 <= resolved["canopy_increase_fraction"] <= 1.0


def test_business_as_usual_converts_nothing(params: dict[str, Any]) -> None:
    resolved = prediction.resolve_scenario("business_as_usual", params)
    assert resolved["canopy_increase_fraction"] == 0.0


def test_scenario_class_lists_stay_inside_the_ca_classes(
    params: dict[str, Any], classes: list[int]
) -> None:
    resolved = prediction.resolve_scenario("greening", params)
    assert set(resolved["eligible_classes"]) <= set(classes)
    assert set(resolved["protect_classes"]) <= set(classes)
    assert resolved["target_class"] in classes


def test_scenario_eligible_and_protected_never_overlap(
    params: dict[str, Any],
) -> None:
    resolved = prediction.resolve_scenario("greening", params)
    assert not set(resolved["eligible_classes"]) & set(resolved["protect_classes"])


def test_scenario_rejects_a_class_outside_the_ca_classes(
    params_copy: dict[str, Any],
) -> None:
    params_copy["prediction"]["scenarios"]["greening"]["eligible_classes"] = [3]
    with pytest.raises(ValueError, match=r"\[3\]"):
        prediction.resolve_scenario("greening", params_copy)


def test_scenario_rejects_a_class_that_is_both_eligible_and_protected(
    params_copy: dict[str, Any],
) -> None:
    # A contradiction that would otherwise resolve silently, in whichever order
    # the code happened to test the two masks.
    params_copy["prediction"]["scenarios"]["greening"]["protect_classes"] = [0, 2]
    with pytest.raises(ValueError, match="BOTH eligible"):
        prediction.resolve_scenario("greening", params_copy)


def test_scenario_rejects_protecting_its_own_target(
    params_copy: dict[str, Any],
) -> None:
    params_copy["prediction"]["scenarios"]["greening"]["protect_classes"] = [1]
    with pytest.raises(ValueError, match="protects its own target"):
        prediction.resolve_scenario("greening", params_copy)


def test_unknown_scenario_names_the_configured_ones(
    params: dict[str, Any],
) -> None:
    with pytest.raises(KeyError, match="business_as_usual"):
        prediction.resolve_scenario("densification", params)


# --- projection horizons -----------------------------------------------------


def test_projection_steps_are_whole_and_report_the_effective_year(
    params: dict[str, Any],
) -> None:
    frame = prediction.resolve_projection_steps(params)
    assert list(frame.columns) == list(prediction.HORIZON_COLUMNS)
    assert (frame["steps"] >= 1).all()
    # The whole point: a horizon that is not a whole number of steps lands on a
    # DIFFERENT year, and the table says so rather than relabelling it.
    assert (
        frame["effective_year"]
        == frame["base_year"] + frame["steps"] * frame["interval_years"]
    ).all()
    assert (
        frame["offset_years"] == frame["effective_year"] - frame["requested_year"]
    ).all()


def test_a_horizon_on_a_whole_step_has_no_offset(params: dict[str, Any]) -> None:
    frame = prediction.resolve_projection_steps(params)
    base, interval = int(frame["base_year"][0]), int(frame["interval_years"][0])
    exact = frame[frame["requested_year"] == base + interval]
    assert not exact.empty
    assert int(exact["offset_years"].iloc[0]) == 0


def test_projection_steps_refuse_rounding_when_it_is_disabled(
    params_copy: dict[str, Any],
) -> None:
    params_copy["prediction"]["ca_markov"]["round_projection_steps"] = False
    with pytest.raises(ValueError, match="whole number"):
        prediction.resolve_projection_steps(params_copy)


def test_projection_steps_refuse_a_year_before_the_base(
    params_copy: dict[str, Any],
) -> None:
    params_copy["prediction"]["ca_markov"]["projection_years"] = [2020]
    with pytest.raises(ValueError, match="not after the base year"):
        prediction.resolve_projection_steps(params_copy)


def test_projection_steps_refuse_a_rounding_that_moves_too_far(
    params_copy: dict[str, Any],
) -> None:
    params_copy["prediction"]["ca_markov"]["max_step_offset_years"] = 0
    with pytest.raises(ValueError, match="max_step_offset_years"):
        prediction.resolve_projection_steps(params_copy)


# --- spatial blocking --------------------------------------------------------


def test_block_ids_group_coordinates_inside_one_block() -> None:
    x = np.array([10.0, 500.0, 999.0, 1001.0])
    y = np.zeros(4)
    ids = prediction.spatial_block_ids(x, y, 1000.0)
    assert ids[0] == ids[1] == ids[2]
    assert ids[3] != ids[0]


def test_block_ids_are_stable_for_the_same_coordinates() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 1e4, 200), rng.uniform(0, 1e4, 200)
    assert np.array_equal(
        prediction.spatial_block_ids(x, y, 1000.0),
        prediction.spatial_block_ids(x, y, 1000.0),
    )


def test_block_ids_handle_negative_coordinates() -> None:
    ids = prediction.spatial_block_ids([-1.0, -1001.0], [0.0, 0.0], 1000.0)
    assert ids[0] != ids[1]


@pytest.mark.parametrize("size", [0.0, -100.0])
def test_block_ids_reject_a_non_positive_block(size: float) -> None:
    with pytest.raises(ValueError, match="block_size_m"):
        prediction.spatial_block_ids([0.0], [0.0], size)


def test_block_ids_reject_a_nan_coordinate() -> None:
    # A NaN easting would land every affected sample in one shared block, which
    # is the leak the blocking exists to prevent, wearing the shape of a fix.
    with pytest.raises(ValueError, match="finite"):
        prediction.spatial_block_ids([0.0, np.nan], [0.0, 0.0], 1000.0)


def test_block_ids_reject_mismatched_arrays() -> None:
    with pytest.raises(ValueError, match="same shape"):
        prediction.spatial_block_ids([0.0, 1.0], [0.0], 1000.0)


def test_blocked_split_never_puts_one_block_on_both_sides() -> None:
    rng = np.random.default_rng(1)
    ids = prediction.spatial_block_ids(
        rng.uniform(0, 4e4, 2000), rng.uniform(0, 4e4, 2000), 2000.0
    )
    train, test = prediction.blocked_split(ids, 0.25, 42)
    assert not set(ids[train]) & set(ids[test])
    assert train.sum() + test.sum() == ids.size


def test_blocked_split_lands_near_the_requested_fraction() -> None:
    rng = np.random.default_rng(2)
    ids = prediction.spatial_block_ids(
        rng.uniform(0, 4e4, 4000), rng.uniform(0, 4e4, 4000), 2000.0
    )
    _, test = prediction.blocked_split(ids, 0.25, 7)
    assert 0.20 <= test.mean() <= 0.32


def test_blocked_split_is_reproducible_from_the_seed() -> None:
    ids = np.repeat(np.arange(20), 10)
    first = prediction.blocked_split(ids, 0.25, 42)
    second = prediction.blocked_split(ids, 0.25, 42)
    assert np.array_equal(first[1], second[1])


def test_blocked_split_refuses_a_single_block() -> None:
    with pytest.raises(ValueError, match="whole blocks"):
        prediction.blocked_split(np.zeros(100, dtype=int), 0.25, 42)


@pytest.mark.parametrize("fraction", [0.0, 1.0, 1.5, -0.1])
def test_blocked_split_refuses_a_fraction_outside_the_unit_interval(
    fraction: float,
) -> None:
    with pytest.raises(ValueError, match="test_fraction"):
        prediction.blocked_split(np.repeat(np.arange(10), 5), fraction, 42)


def test_blocked_kfold_tests_every_row_exactly_once() -> None:
    ids = np.repeat(np.arange(25), 8)
    counted = np.zeros(ids.size, dtype=int)
    for _, test in prediction.blocked_kfold(ids, 5, 42):
        counted += test.astype(int)
    assert (counted == 1).all()


def test_blocked_kfold_keeps_whole_blocks_in_one_fold() -> None:
    ids = np.repeat(np.arange(25), 8)
    for train, test in prediction.blocked_kfold(ids, 5, 42):
        assert not set(ids[train]) & set(ids[test])


def test_blocked_kfold_refuses_more_folds_than_blocks() -> None:
    with pytest.raises(ValueError, match="exceeds the number of spatial blocks"):
        prediction.blocked_kfold(np.repeat(np.arange(3), 10), 5, 42)


def test_blocked_kfold_refuses_fewer_than_two_folds() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        prediction.blocked_kfold(np.repeat(np.arange(10), 5), 1, 42)


def test_require_enough_blocks_refuses_a_thin_sample(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="min_blocks"):
        prediction.require_enough_blocks(np.repeat(np.arange(3), 10), params)


def test_require_enough_blocks_reports_what_it_found(params: dict[str, Any]) -> None:
    report = prediction.require_enough_blocks(np.repeat(np.arange(30), 10), params)
    assert report["n_blocks"] == 30
    assert report["n_rows"] == 300
    assert report["rows_per_block_median"] == pytest.approx(10.0)


def test_random_row_split_covers_every_row() -> None:
    train, test = prediction.random_row_split(100, 0.25, 42)
    assert train.sum() + test.sum() == 100
    assert test.sum() == 25


# --- regression metrics ------------------------------------------------------


def test_rmse_of_a_perfect_prediction_is_zero() -> None:
    values = np.array([1.0, 2.0, 3.0])
    assert prediction.rmse(values, values) == pytest.approx(0.0)


def test_rmse_matches_the_hand_computed_value() -> None:
    assert prediction.rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(
        math.sqrt((9 + 16) / 2)
    )


def test_r_squared_of_a_perfect_prediction_is_one() -> None:
    values = np.array([1.0, 5.0, 9.0])
    assert prediction.r_squared(values, values) == pytest.approx(1.0)


def test_r_squared_of_the_mean_predictor_is_zero() -> None:
    values = np.array([1.0, 5.0, 9.0])
    assert prediction.r_squared(
        values, np.full_like(values, values.mean())
    ) == pytest.approx(0.0)


def test_held_out_r_squared_can_go_negative_and_is_not_clamped() -> None:
    # A negative held-out R2 is the correct signal that the model does worse
    # than predicting the held-out mean. Clamping it to 0 would hide that.
    assert prediction.r_squared([1.0, 2.0, 3.0], [9.0, 9.0, 9.0]) < 0.0


def test_r_squared_of_a_constant_truth_is_nan() -> None:
    assert math.isnan(prediction.r_squared([2.0, 2.0], [1.0, 3.0]))


@pytest.mark.parametrize("metric", [prediction.rmse, prediction.r_squared])
def test_regression_metrics_reject_mismatched_arrays(metric: Any) -> None:
    with pytest.raises(ValueError, match="must match"):
        metric([1.0, 2.0], [1.0])


# --- land-cover agreement ----------------------------------------------------


def test_confusion_matrix_is_observed_by_projected() -> None:
    matrix = prediction.confusion_matrix([0, 0, 1], [0, 1, 1], [0, 1])
    # Row 0 (observed 0): one projected 0, one projected 1.
    assert matrix.tolist() == [[1, 1], [0, 1]]


def test_confusion_matrix_excludes_codes_outside_the_legend() -> None:
    with pytest.warns(UserWarning, match="outside"):
        matrix = prediction.confusion_matrix([0, 9], [0, 9], [0, 1])
    assert matrix.sum() == 1


def test_confusion_matrix_rejects_duplicate_classes() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        prediction.confusion_matrix([0], [0], [0, 0])


def test_kappa_of_perfect_agreement_is_one() -> None:
    matrix = prediction.confusion_matrix([0, 1, 2, 1], [0, 1, 2, 1], [0, 1, 2])
    assert prediction.cohen_kappa(matrix) == pytest.approx(1.0)


def test_kappa_of_chance_agreement_is_about_zero() -> None:
    # Marginals identical, agreement exactly what chance predicts.
    matrix = np.array([[25, 25], [25, 25]], dtype=float)
    assert prediction.cohen_kappa(matrix) == pytest.approx(0.0)


def test_kappa_matches_the_textbook_worked_example() -> None:
    # Landis & Koch's canonical 2x2: po = 0.70, pe = 0.50, kappa = 0.40.
    matrix = np.array([[20, 5], [10, 15]], dtype=float)
    assert prediction.cohen_kappa(matrix) == pytest.approx(0.4)


def test_kappa_of_a_single_class_map_is_nan_not_one() -> None:
    # Expected agreement is 1, so chance-corrected agreement is undefined. NaN
    # fails require_validated; returning 1.0 would pass as a perfect score.
    assert math.isnan(prediction.cohen_kappa(np.array([[10.0]])))


def test_kappa_rejects_a_non_square_matrix() -> None:
    with pytest.raises(ValueError, match="square"):
        prediction.cohen_kappa(np.zeros((2, 3)))


def test_kappa_rejects_an_empty_matrix() -> None:
    with pytest.raises(ValueError, match="sums to zero"):
        prediction.cohen_kappa(np.zeros((2, 2)))


def test_pontius_components_sum_to_one() -> None:
    # The identity that makes the decomposition worth reporting: proportion
    # correct + quantity + allocation is the whole map.
    rng = np.random.default_rng(5)
    matrix = rng.integers(0, 40, size=(4, 4)).astype(float)
    parts = prediction.quantity_allocation_disagreement(matrix)
    assert (
        parts["proportion_correct"]
        + parts["quantity_disagreement"]
        + parts["allocation_disagreement"]
    ) == pytest.approx(1.0)


def test_pure_allocation_error_has_no_quantity_disagreement() -> None:
    # Same marginals, wrong places: quantity 0, all the error is allocation.
    matrix = np.array([[0.0, 10.0], [10.0, 0.0]])
    parts = prediction.quantity_allocation_disagreement(matrix)
    assert parts["quantity_disagreement"] == pytest.approx(0.0)
    assert parts["allocation_disagreement"] == pytest.approx(1.0)


def test_pure_quantity_error_has_no_allocation_disagreement() -> None:
    # Every observed class-0 cell projected as class 1: nothing is in the right
    # place AND the totals differ, so the error is entirely quantity.
    matrix = np.array([[0.0, 10.0], [0.0, 10.0]])
    parts = prediction.quantity_allocation_disagreement(matrix)
    assert parts["allocation_disagreement"] == pytest.approx(0.0)
    assert parts["quantity_disagreement"] == pytest.approx(0.5)


def test_a_no_change_projection_scores_zero_figure_of_merit() -> None:
    # THE reason the figure of merit is reported beside Kappa. A projection
    # that copies the initial map scores 0 here however high its Kappa is.
    initial = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    observed = np.array([0, 1, 1, 2, 2, 2, 2, 2])
    scores = prediction.figure_of_merit(initial, observed, initial)
    assert scores["figure_of_merit"] == pytest.approx(0.0)
    assert scores["hits"] == 0
    assert scores["misses"] == 2


def test_a_no_change_projection_still_scores_a_high_kappa() -> None:
    initial = np.array([0] * 90 + [1] * 10)
    observed = initial.copy()
    observed[:3] = 1
    kappa = prediction.persistence_baseline_kappa(initial, observed, [0, 1])
    assert kappa > 0.7  # ...which is exactly why it must not be read alone


def test_a_perfect_projection_scores_a_figure_of_merit_of_one() -> None:
    initial = np.array([0, 0, 1, 1])
    observed = np.array([0, 1, 1, 2])
    scores = prediction.figure_of_merit(initial, observed, observed)
    assert scores["figure_of_merit"] == pytest.approx(1.0)
    assert scores["false_alarms"] == 0


def test_figure_of_merit_counts_a_wrong_class_as_a_wrong_hit() -> None:
    initial = np.array([0])
    observed = np.array([1])
    projected = np.array([2])
    scores = prediction.figure_of_merit(initial, observed, projected)
    assert scores["wrong_hits"] == 1
    assert scores["hits"] == 0


def test_figure_of_merit_counts_change_where_none_happened_as_a_false_alarm() -> None:
    scores = prediction.figure_of_merit([0, 0], [0, 0], [0, 1])
    assert scores["false_alarms"] == 1
    assert scores["figure_of_merit"] == pytest.approx(0.0)


def test_figure_of_merit_is_nan_when_nothing_changed_at_all() -> None:
    # No skill to measure is not the same as zero skill.
    scores = prediction.figure_of_merit([0, 0], [0, 0], [0, 0])
    assert math.isnan(scores["figure_of_merit"])


def test_figure_of_merit_rejects_mismatched_arrays() -> None:
    with pytest.raises(ValueError, match="must match"):
        prediction.figure_of_merit([0, 1], [0], [0, 1])


# --- Markov core -------------------------------------------------------------


def test_transition_probability_rows_sum_to_one() -> None:
    counts = np.array([[5.0, 5.0], [2.0, 8.0]])
    assert np.allclose(prediction.transition_probabilities(counts).sum(axis=1), 1.0)


def test_an_unobserved_class_row_becomes_identity_not_nan() -> None:
    # 0/0 is nan, and one nan row poisons every subsequent matrix power. The
    # identity row says "a class we never saw leave, we do not model leaving".
    counts = np.array([[5.0, 5.0], [0.0, 0.0]])
    with pytest.warns(UserWarning, match="never observed"):
        probabilities = prediction.transition_probabilities(counts)
    assert probabilities[1].tolist() == [0.0, 1.0]
    assert np.isfinite(probabilities).all()


def test_transition_probabilities_reject_negative_counts() -> None:
    with pytest.raises(ValueError, match="not be negative"):
        prediction.transition_probabilities(np.array([[1.0, -1.0], [0.0, 1.0]]))


def test_one_markov_step_is_the_matrix_itself() -> None:
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9]])
    assert np.allclose(prediction.markov_project(probabilities, 1), probabilities)


def test_markov_steps_compose() -> None:
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9]])
    two = prediction.markov_project(probabilities, 2)
    assert np.allclose(two, probabilities @ probabilities)
    assert np.allclose(two.sum(axis=1), 1.0)


def test_a_doubly_stochastic_chain_converges_to_uniform() -> None:
    probabilities = np.array([[0.6, 0.4], [0.4, 0.6]])
    assert np.allclose(prediction.markov_project(probabilities, 200), 0.5)


def test_markov_project_refuses_a_fractional_step() -> None:
    # A fractional power of a transition matrix is not guaranteed to BE a
    # transition matrix, so this refuses rather than approximating.
    with pytest.raises(TypeError, match="whole number"):
        prediction.markov_project(np.array([[0.5, 0.5], [0.5, 0.5]]), 1.5)


def test_markov_project_refuses_rows_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        prediction.markov_project(np.array([[3.0, 1.0], [1.0, 3.0]]), 2)


def test_markov_project_refuses_zero_steps() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        prediction.markov_project(np.array([[1.0, 0.0], [0.0, 1.0]]), 0)


def test_projected_areas_conserve_the_total_cell_count(
    params: dict[str, Any], classes: list[int]
) -> None:
    counts = np.full(len(classes), 100, dtype=float)
    probabilities = np.full((len(classes), len(classes)), 1.0 / len(classes))
    frame = prediction.projected_class_areas(
        counts, probabilities, 2, 900.0, classes, params
    )
    assert list(frame.columns) == list(prediction.AREA_COLUMNS)
    assert int(frame["cells"].sum()) == int(counts.sum())
    assert frame["share"].sum() == pytest.approx(1.0)


def test_projected_areas_reject_counts_that_do_not_match_the_matrix(
    params: dict[str, Any], classes: list[int]
) -> None:
    probabilities = np.eye(len(classes))
    with pytest.raises(ValueError, match="expected"):
        prediction.projected_class_areas(
            [1.0, 2.0], probabilities, 1, 900.0, classes, params
        )


# --- cellular automaton ------------------------------------------------------


def test_neighbourhood_potential_of_a_uniform_map_is_one() -> None:
    labels = np.full((5, 5), 3)
    assert prediction.neighbourhood_potential(labels, 3, 1).min() == pytest.approx(
        1.0
    )


def test_neighbourhood_potential_excludes_the_centre_cell() -> None:
    labels = np.zeros((3, 3), dtype=int)
    labels[1, 1] = 1
    # The centre is the only cell of class 1, and it does not count itself.
    assert prediction.neighbourhood_potential(labels, 1, 1)[1, 1] == pytest.approx(
        0.0
    )


def test_neighbourhood_potential_does_not_wrap_around_the_edges() -> None:
    # Wrapping would give a coastal cell neighbours from the far side of the
    # district, which is how a CA invents a patch in the middle of the ocean.
    labels = np.zeros((1, 5), dtype=int)
    labels[0, 0] = 1
    potential = prediction.neighbourhood_potential(labels, 1, 1)
    assert potential[0, -1] == pytest.approx(0.0)


def test_neighbourhood_potential_rejects_a_non_2d_array() -> None:
    with pytest.raises(ValueError, match="2-D"):
        prediction.neighbourhood_potential(np.zeros(5), 1, 1)


def test_allocation_meets_the_demand_it_can_satisfy(
    params: dict[str, Any], classes: list[int]
) -> None:
    rng = np.random.default_rng(3)
    labels = rng.choice(classes, size=(40, 40))
    demand = {code: int(np.sum(labels == code)) for code in classes}
    demand[6] += 100
    demand[2] -= 100
    result, report = prediction.ca_allocate(labels, demand, params)
    assert (report["after"] == report["demand"]).all()
    assert int(report["shortfall"].sum()) == 0
    assert result.shape == labels.shape


def test_allocation_never_moves_an_immutable_class(
    params: dict[str, Any], classes: list[int]
) -> None:
    rng = np.random.default_rng(4)
    labels = rng.choice(classes, size=(30, 30))
    immutable = params["prediction"]["ca_markov"]["immutable_classes"]
    demand = {code: int(np.sum(labels == code)) for code in classes}
    demand[6] += 50
    demand[immutable[0]] -= 50
    result, _ = prediction.ca_allocate(labels, demand, params)
    for code in immutable:
        assert int(np.sum(result == code)) == int(np.sum(labels == code))


def test_allocation_is_deterministic_for_a_seed(
    params: dict[str, Any], classes: list[int]
) -> None:
    rng = np.random.default_rng(6)
    labels = rng.choice(classes, size=(25, 25))
    demand = {code: int(np.sum(labels == code)) for code in classes}
    demand[6] += 30
    demand[7] -= 30
    first, _ = prediction.ca_allocate(labels, demand, params, seed=11)
    second, _ = prediction.ca_allocate(labels, demand, params, seed=11)
    assert np.array_equal(first, second)


def test_allocation_reports_a_shortfall_rather_than_raising(
    params: dict[str, Any], classes: list[int]
) -> None:
    # Not enough changeable cells is a RESULT to report, not an error - so it
    # is a column, and the caller can see how far short the demand fell. Here
    # every cell is water, which is immutable, so there is no donor at all.
    immutable = int(params["prediction"]["ca_markov"]["immutable_classes"][0])
    labels = np.full((10, 10), immutable)
    demand = {code: 0 for code in classes}
    demand[6] = 50
    _, report = prediction.ca_allocate(labels, demand, params)
    shortfall = report.set_index("class_code")["shortfall"]
    assert int(shortfall.loc[6]) == 50
    assert int(np.sum(_ == immutable)) == labels.size


def test_allocation_rejects_a_class_outside_the_ca_classes(
    params: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="outside"):
        prediction.ca_allocate(np.zeros((3, 3), dtype=int), {99: 1}, params)


def test_allocation_rejects_a_mismatched_changeable_mask(
    params: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="changeable"):
        prediction.ca_allocate(
            np.zeros((3, 3), dtype=int), {}, params,
            changeable=np.ones((2, 2), dtype=bool),
        )


def test_ca_markov_project_returns_every_traceable_piece(
    params: dict[str, Any], classes: list[int]
) -> None:
    rng = np.random.default_rng(7)
    early = rng.choice(classes, size=(30, 30))
    late = early.copy()
    late[rng.random(early.shape) < 0.1] = 6
    result = prediction.ca_markov_project(early, late, params, steps=2)
    assert set(result) >= {
        "labels", "counts", "probabilities", "areas", "allocation", "framing"
    }
    assert result["labels"].shape == early.shape
    assert "NOT_forecast" in result["framing"]


def test_ca_markov_project_rejects_mismatched_dates(
    params: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="must match"):
        prediction.ca_markov_project(
            np.zeros((3, 3), dtype=int), np.zeros((4, 4), dtype=int), params, 1
        )


# --- scenarios on a grid -----------------------------------------------------


def test_business_as_usual_leaves_the_map_untouched(
    params: dict[str, Any], classes: list[int]
) -> None:
    rng = np.random.default_rng(8)
    labels = rng.choice(classes, size=(20, 20))
    mask = np.ones(labels.shape, dtype=bool)
    result, report = prediction.apply_greening_scenario(
        labels, mask, params, "business_as_usual"
    )
    assert np.array_equal(result, labels)
    assert report["n_converted"] == 0


def test_greening_converts_only_eligible_cells_inside_priority_zones(
    params: dict[str, Any], classes: list[int]
) -> None:
    rng = np.random.default_rng(9)
    labels = rng.choice(classes, size=(30, 30))
    mask = np.zeros(labels.shape, dtype=bool)
    mask[:15] = True
    result, report = prediction.apply_greening_scenario(
        labels, mask, params, "greening"
    )
    changed = result != labels
    scenario = prediction.resolve_scenario("greening", params)
    assert changed[~mask].sum() == 0  # nothing outside a priority zone
    assert set(np.unique(labels[changed]).tolist()) <= set(
        scenario["eligible_classes"]
    )
    assert set(np.unique(result[changed]).tolist()) == {scenario["target_class"]}
    assert report["n_converted"] <= report["n_eligible"]


def test_greening_never_touches_a_protected_class(
    params: dict[str, Any], classes: list[int]
) -> None:
    rng = np.random.default_rng(10)
    labels = rng.choice(classes, size=(30, 30))
    mask = np.ones(labels.shape, dtype=bool)
    result, _ = prediction.apply_greening_scenario(labels, mask, params, "greening")
    for code in prediction.resolve_scenario("greening", params)["protect_classes"]:
        assert int(np.sum(result == code)) == int(np.sum(labels == code))


def test_greening_converts_the_configured_fraction(
    params_copy: dict[str, Any]
) -> None:
    params_copy["prediction"]["scenarios"]["greening"][
        "canopy_increase_fraction"
    ] = 0.5
    classes = prediction.resolve_ca_classes(params_copy)
    rng = np.random.default_rng(11)
    labels = rng.choice(classes, size=(40, 40))
    mask = np.ones(labels.shape, dtype=bool)
    _, report = prediction.apply_greening_scenario(
        labels, mask, params_copy, "greening"
    )
    assert report["n_converted"] == math.floor(0.5 * report["n_eligible"])


def test_greening_rejects_a_mismatched_priority_mask(
    params: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="priority_mask"):
        prediction.apply_greening_scenario(
            np.zeros((3, 3), dtype=int), np.ones((2, 2), dtype=bool),
            params, "greening",
        )


# --- priority zones ----------------------------------------------------------


def _zone_frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(12)
    return pd.DataFrame(
        {
            "zone_id": [f"LK1101{i:03d}" for i in range(n)],
            "gi_z": rng.normal(size=n),
            "LST_C": rng.normal(31, 2, size=n),
            "NDVI": rng.uniform(0, 0.6, size=n),
        }
    )


def test_priority_zones_flag_the_configured_count(params: dict[str, Any]) -> None:
    ranked = prediction.interim_priority_zones(_zone_frame(), params)
    expected = min(int(params["prediction"]["priority_zones"]["top_n"]), 100)
    assert int(ranked["priority"].sum()) == expected
    assert ranked["rank"].tolist() == list(range(1, 101))


def test_priority_ranking_inverts_an_inverse_criterion(
    params: dict[str, Any]
) -> None:
    # ndvi_inverse means LOW NDVI is high priority; a greening rule that ranked
    # the greenest divisions first would be exactly backwards.
    frame = _zone_frame()
    ranked = prediction.interim_priority_zones(frame, params)
    merged = ranked.merge(frame, on="zone_id")
    assert merged["ndvi_inverse_rank"].corr(merged["NDVI"]) < -0.9


def test_priority_zones_reject_duplicate_zone_ids(params: dict[str, Any]) -> None:
    frame = _zone_frame(4)
    frame.loc[3, "zone_id"] = frame.loc[0, "zone_id"]
    with pytest.raises(ValueError, match="duplicate zone_id"):
        prediction.interim_priority_zones(frame, params)


def test_priority_zones_reject_a_missing_criterion_column(
    params: dict[str, Any]
) -> None:
    frame = _zone_frame(10).drop(columns=["gi_z"])
    with pytest.raises(ValueError, match="gi_z"):
        prediction.interim_priority_zones(frame, params)


def test_priority_weights_match_the_criteria_and_sum_to_one(
    params: dict[str, Any]
) -> None:
    cfg = params["prediction"]["priority_zones"]
    assert len(cfg["weights"]) == len(cfg["rank_by"])
    assert sum(cfg["weights"]) == pytest.approx(1.0)


def test_priority_zones_reject_weights_that_do_not_sum_to_one(
    params_copy: dict[str, Any]
) -> None:
    params_copy["prediction"]["priority_zones"]["weights"] = [0.5, 0.4, 0.2]
    with pytest.raises(ValueError, match="sum to"):
        prediction.interim_priority_zones(_zone_frame(10), params_copy)


# --- class-conditional predictors --------------------------------------------


def test_class_conditional_predictors_take_the_within_class_median() -> None:
    frame = pd.DataFrame(
        {"label": [1, 1, 1, 6, 6], "NDVI": [0.5, 0.6, 0.7, 0.1, 0.3]}
    )
    table = prediction.class_conditional_predictors(frame, "label", ["NDVI"], 1)
    assert table.set_index("class_code").loc[1, "NDVI"] == pytest.approx(0.6)
    assert table.set_index("class_code").loc[6, "NDVI"] == pytest.approx(0.2)


def test_class_conditional_predictors_flag_a_thin_class() -> None:
    frame = pd.DataFrame({"label": [1, 1, 1, 6], "NDVI": [0.5, 0.6, 0.7, 0.1]})
    table = prediction.class_conditional_predictors(frame, "label", ["NDVI"], 3)
    flags = table.set_index("class_code")["thin"]
    assert not bool(flags.loc[1])
    assert bool(flags.loc[6])


def test_painting_leaves_an_unknown_class_as_nan_not_zero() -> None:
    # A zero NDVI is a real value. Filling an unknown class with 0 would be
    # indistinguishable from a genuinely bare pixel.
    table = pd.DataFrame({"class_code": [1], "n": [10], "thin": [False], "NDVI": [0.6]})
    painted = prediction.paint_class_predictors(
        np.array([[1, 6], [1, 6]]), table, ["NDVI"]
    )
    assert painted["NDVI"][0, 0] == pytest.approx(0.6)
    assert math.isnan(painted["NDVI"][0, 1])


def test_painting_rejects_a_predictor_the_table_does_not_carry() -> None:
    table = pd.DataFrame({"class_code": [1], "n": [1], "thin": [False], "NDVI": [0.6]})
    with pytest.raises(ValueError, match="NDBI"):
        prediction.paint_class_predictors(np.zeros((2, 2), dtype=int), table, ["NDBI"])


# --- extrapolation -----------------------------------------------------------


def _predictor_frame(params: dict[str, Any], values: float, n: int = 50) -> pd.DataFrame:
    names = prediction.resolve_predictors(None, params)
    return pd.DataFrame({name: np.full(n, values) for name in names})


def test_a_target_inside_the_training_range_is_not_flagged(
    params: dict[str, Any]
) -> None:
    names = prediction.resolve_predictors(None, params)
    training = pd.DataFrame({name: np.linspace(0, 10, 100) for name in names})
    target = pd.DataFrame({name: np.linspace(2, 8, 40) for name in names})
    flags = prediction.extrapolation_flags(training, target, params)
    assert flags["fraction"] == pytest.approx(0.0)
    assert flags["within_tolerance"]


def test_a_target_beyond_the_training_range_is_flagged(
    params: dict[str, Any]
) -> None:
    names = prediction.resolve_predictors(None, params)
    training = pd.DataFrame({name: np.linspace(0, 10, 100) for name in names})
    target = pd.DataFrame({name: np.linspace(20, 30, 40) for name in names})
    flags = prediction.extrapolation_flags(training, target, params)
    assert flags["fraction"] == pytest.approx(1.0)
    assert not flags["within_tolerance"]
    assert all(value == pytest.approx(1.0) for value in flags["by_predictor"].values())


def test_extrapolation_rejects_an_empty_target(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="empty target"):
        prediction.extrapolation_flags(
            _predictor_frame(params, 1.0), _predictor_frame(params, 1.0, 0), params
        )


def test_extrapolation_rejects_a_frame_missing_a_predictor(
    params: dict[str, Any]
) -> None:
    training = _predictor_frame(params, 1.0)
    with pytest.raises(ValueError, match="missing"):
        prediction.extrapolation_flags(training, pd.DataFrame({"NDVI": [1.0]}), params)


# --- the validation guard ----------------------------------------------------


def _report(params: dict[str, Any], kind: str, **overrides: Any) -> dict[str, Any]:
    metrics = {
        "lst_fit": {"rmse": 1.2, "r2": 0.7},
        "lst_projection": {"rmse": 1.2, "r2": 0.7, "kappa": 0.65},
        "lulc_projection": {"kappa": 0.65},
    }[kind]
    kwargs: dict[str, Any] = {"held_out": True}
    kwargs.update(overrides)
    supplied = kwargs.pop("metrics", metrics)
    return prediction.build_validation_report(kind, supplied, params, **kwargs)


def test_required_metrics_cover_every_product_kind() -> None:
    assert set(prediction.REQUIRED_METRICS) == set(prediction.PRODUCT_KINDS)


def test_a_projected_lst_product_requires_the_land_cover_kappa_too(
    params: dict[str, Any]
) -> None:
    # CLAUDE.md caveat 3 asks for RMSE, R2 AND Kappa on a predictive output.
    # A projected LST surface sits on top of a projected land cover, so it is
    # not validated by its regression metrics alone.
    assert set(prediction.REQUIRED_METRICS["lst_projection"]) == {
        "rmse", "r2", "kappa"
    }
    assert "kappa" not in prediction.REQUIRED_METRICS["lst_fit"]


def test_every_required_metric_is_declared_in_params(params: dict[str, Any]) -> None:
    declared = set(params["prediction"]["validation_metrics"])
    for required in prediction.REQUIRED_METRICS.values():
        assert set(required) <= declared


def test_a_complete_report_passes(params: dict[str, Any]) -> None:
    report = _report(params, "lst_projection", n_blocks=180, block_size_m=2000)
    assert prediction.require_validated(report, params)["kind"] == "lst_projection"


def test_no_report_at_all_is_refused(params: dict[str, Any]) -> None:
    with pytest.raises(prediction.ValidationMissing, match="no validation report"):
        prediction.require_validated(None, params)


def test_an_empty_report_is_refused(params: dict[str, Any]) -> None:
    with pytest.raises(prediction.ValidationMissing, match="no validation report"):
        prediction.require_validated({}, params)


def test_a_missing_metric_is_refused_and_named(params: dict[str, Any]) -> None:
    report = _report(params, "lst_projection", metrics={"rmse": 1.2, "r2": 0.7})
    with pytest.raises(prediction.ValidationMissing, match="kappa"):
        prediction.require_validated(report, params)


def test_a_nan_metric_is_refused_rather_than_passing(params: dict[str, Any]) -> None:
    # A NaN Kappa comes from a single-class map. It is an absence of evidence,
    # and must not slide through as though it were a computed number.
    report = _report(
        params, "lulc_projection", metrics={"kappa": float("nan")}
    )
    with pytest.raises(prediction.ValidationMissing, match="non-finite"):
        prediction.require_validated(report, params)


def test_training_set_metrics_are_refused(params: dict[str, Any]) -> None:
    report = _report(params, "lst_fit", held_out=False)
    with pytest.raises(prediction.ValidationMissing, match="held-out"):
        prediction.require_validated(report, params)


def test_too_much_extrapolation_is_refused(params: dict[str, Any]) -> None:
    report = _report(
        params,
        "lst_fit",
        extrapolation={"fraction": 0.4, "tolerance": 0.05, "within_tolerance": False},
    )
    with pytest.raises(prediction.ValidationMissing, match="extrapolate"):
        prediction.require_validated(report, params)


def test_an_unknown_product_kind_is_refused(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="unknown product kind"):
        prediction.build_validation_report("suhii_map", {}, params, held_out=True)


def test_a_report_with_a_tampered_kind_is_refused(params: dict[str, Any]) -> None:
    report = _report(params, "lst_fit")
    report["kind"] = "something_else"
    with pytest.raises(prediction.ValidationMissing, match="unknown kind"):
        prediction.require_validated(report, params)


def test_export_projection_refuses_before_touching_earth_engine(
    params: dict[str, Any]
) -> None:
    # No ee import, no network, no credentials: the guard must fire first. If
    # this test ever fails with something other than ValidationMissing, the
    # guard has moved below an Earth Engine call.
    with pytest.raises(prediction.ValidationMissing):
        prediction.export_projection(None, params, None, None)


def test_write_lulc_projection_refuses_before_touching_rasterio(
    params: dict[str, Any], tmp_path: Any
) -> None:
    destination = tmp_path / "must_not_exist.tif"
    with pytest.raises(prediction.ValidationMissing):
        prediction.write_lulc_projection(
            np.zeros((2, 2), dtype=int), {}, destination, params, None
        )
    assert not destination.exists()


# --- the figure caption ------------------------------------------------------


def test_the_caption_says_it_is_not_a_forecast(params: dict[str, Any]) -> None:
    caption = prediction.validation_caption(
        _report(params, "lst_projection", n_blocks=180, block_size_m=2000), params
    )
    # Wrapping inserts newlines, so compare against the unwrapped text.
    flat = " ".join(caption.split())
    assert "NOT A FORECAST" in flat
    assert "Land Surface Temperature, not air temperature" in flat


def test_the_caption_carries_every_computed_metric(params: dict[str, Any]) -> None:
    report = _report(
        params,
        "lst_projection",
        metrics={
            # Kappa must beat its own no-change baseline, or require_validated
            # refuses the product and the caption is never built.
            "rmse": 1.2, "r2": 0.7, "kappa": 0.94,
            "persistence_kappa": 0.65, "figure_of_merit": 0.21,
        },
    )
    caption = prediction.validation_caption(report, params)
    for expected in ("RMSE=1.200", "R2=0.700", "Kappa=0.940", "0.650", "0.210"):
        assert expected in caption


def test_the_caption_wraps_to_the_shared_footer_width(
    params: dict[str, Any]
) -> None:
    from colombo_uhi import viz

    assert prediction.CAPTION_WRAP_CHARS == viz.CAVEAT_WRAP_CHARS
    caption = prediction.validation_caption(_report(params, "lst_fit"), params)
    assert all(len(line) <= viz.CAVEAT_WRAP_CHARS for line in caption.splitlines())


def test_the_caption_refuses_an_unvalidated_product(params: dict[str, Any]) -> None:
    with pytest.raises(prediction.ValidationMissing):
        prediction.validation_caption(None, params)


# --- scikit-learn cross-checks (skipped without sklearn) ---------------------


def _autocorrelated_sample(params: dict[str, Any], seed: int = 0) -> pd.DataFrame:
    """Predictors and a response that are each smooth in space but unrelated.

    This is what spatial leakage looks like. Nothing here connects the
    predictors to the response, so the only skill a model can show is
    recognising a near-duplicate neighbour - which a random split hands it and
    a blocked split does not.
    """
    rng = np.random.default_rng(seed)
    n = 1800
    x = rng.uniform(0, 40000, n)
    y = rng.uniform(0, 40000, n)
    frame = pd.DataFrame({"x": x, "y": y})
    for index, name in enumerate(prediction.resolve_predictors(None, params)):
        frame[name] = np.sin((x + 700 * index) / 900.0) + np.cos(
            (y - 400 * index) / 1100.0
        )
    frame[params["prediction"]["rf"]["response"]] = np.sin(x / 850.0) * np.cos(
        y / 1050.0
    )
    frame["block_id"] = prediction.spatial_block_ids(x, y, 4000.0)
    return frame


def test_a_random_split_reports_a_better_score_than_a_blocked_one(
    params_copy: dict[str, Any]
) -> None:
    # THE reason prediction.split.method may not be "random". Both splits see
    # the same rows and the same model; the random one scores higher purely
    # because its test rows have training neighbours.
    pytest.importorskip("sklearn")
    params_copy["prediction"]["rf"]["n_trees"] = 60
    params_copy["prediction"]["rf"]["min_sample_rows"] = 100
    params_copy["prediction"]["split"]["block_size_m"] = 4000

    frame = _autocorrelated_sample(params_copy)
    comparison = prediction.compare_split_strategies(
        frame, params_copy, n_repeats=3
    ).set_index("method")
    # Compared as MEANS over repeated splits. Run 2 showed a single pair of
    # splits cannot resolve a gap smaller than the fold-to-fold spread.
    assert comparison.loc[prediction.RANDOM_SPLIT, "r2_mean"] > comparison.loc[
        "spatial_block", "r2_mean"
    ]
    assert not bool(comparison.loc[prediction.RANDOM_SPLIT, "reportable"])
    assert bool(comparison.loc["spatial_block", "reportable"])


def test_blocked_cross_validation_returns_one_row_per_fold(
    params_copy: dict[str, Any]
) -> None:
    pytest.importorskip("sklearn")
    params_copy["prediction"]["rf"]["n_trees"] = 40
    params_copy["prediction"]["rf"]["min_sample_rows"] = 100
    params_copy["prediction"]["split"]["block_size_m"] = 4000

    frame = _autocorrelated_sample(params_copy, seed=1)
    scores = prediction.blocked_cv_scores(frame, params_copy)
    assert list(scores.columns) == list(prediction.CV_COLUMNS)
    assert len(scores) == params_copy["prediction"]["split"]["n_folds"]
    assert (scores["n_test"] > 0).all()
    assert (scores["n_train_blocks"] > 0).all()


def test_permutation_importance_covers_every_predictor(
    params_copy: dict[str, Any]
) -> None:
    pytest.importorskip("sklearn")
    params_copy["prediction"]["rf"]["n_trees"] = 30
    params_copy["prediction"]["rf"]["min_sample_rows"] = 100

    frame = _autocorrelated_sample(params_copy, seed=2)
    train, test = prediction.blocked_split(frame["block_id"].to_numpy(), 0.25, 42)
    fitted = prediction.fit_sklearn_rf(frame, params_copy, train_mask=train)
    importance = prediction.permutation_importance_frame(
        fitted, frame, params_copy, mask=test, n_repeats=3
    )
    assert list(importance.columns) == list(prediction.IMPORTANCE_COLUMNS)
    assert set(importance["predictor"]) == set(fitted["predictors"])
    assert importance["rank"].tolist() == list(range(1, len(importance) + 1))


def test_fitting_refuses_a_sample_below_the_configured_floor(
    params_copy: dict[str, Any]
) -> None:
    pytest.importorskip("sklearn")
    frame = _autocorrelated_sample(params_copy, seed=3).head(20)
    with pytest.raises(ValueError, match="min_sample_rows"):
        prediction.fit_sklearn_rf(frame, params_copy)


def test_blocked_cv_refuses_a_frame_without_block_ids(
    params: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="spatial_block_ids"):
        prediction.blocked_cv_scores(pd.DataFrame({"NDVI": [1.0]}), params)


# --- the predictor raster seam ------------------------------------------------


def test_the_predictor_band_order_leads_with_the_response(
    params: dict[str, Any]
) -> None:
    order = prediction.predictor_band_order(params)
    assert order[0] == params["prediction"]["rf"]["response"]
    assert order[1:] == prediction.resolve_predictors(None, params)


def test_reading_a_predictor_raster_rejects_a_band_count_mismatch(
    params: dict[str, Any], tmp_path: Any
) -> None:
    # Band order is not stored in a GeoTIFF, so a mismatch would silently
    # rename every predictor and produce a plausible, wrong map.
    rasterio = pytest.importorskip("rasterio")

    path = tmp_path / "too_few_bands.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=4, width=4, count=2,
        dtype="float32", crs="EPSG:32644",
        transform=rasterio.transform.from_origin(0, 0, 100, 100),
    ) as handle:
        handle.write(np.zeros((2, 4, 4), dtype="float32"))
    with pytest.raises(ValueError, match="band"):
        prediction.read_predictor_raster(path, params)


def test_a_predictor_raster_round_trips_through_its_band_order(
    params: dict[str, Any], tmp_path: Any
) -> None:
    rasterio = pytest.importorskip("rasterio")

    names = prediction.predictor_band_order(params)
    stack = np.stack(
        [np.full((3, 5), float(index)) for index in range(len(names))]
    ).astype("float32")
    path = tmp_path / "predictors.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=3, width=5, count=len(names),
        dtype="float32", crs="EPSG:32644",
        transform=rasterio.transform.from_origin(0, 0, 100, 100),
    ) as handle:
        handle.write(stack)

    arrays, profile = prediction.read_predictor_raster(path, params)
    assert list(arrays) == names
    assert profile["width"] == 5
    for index, name in enumerate(names):
        assert arrays[name][0, 0] == pytest.approx(float(index))


def test_flattening_rasters_is_reversible() -> None:
    grid = np.arange(12, dtype=float).reshape(3, 4)
    frame = prediction.raster_sample_frame({"NDVI": grid})
    assert np.array_equal(frame["NDVI"].to_numpy().reshape(3, 4), grid)


def test_flattening_attaches_the_land_cover_label() -> None:
    frame = prediction.raster_sample_frame(
        {"NDVI": np.zeros((2, 2))}, labels=np.array([[1, 6], [6, 1]])
    )
    assert frame["lulc_class"].tolist() == [1, 6, 6, 1]


def test_flattening_rejects_arrays_of_different_shapes() -> None:
    with pytest.raises(ValueError, match="share one shape"):
        prediction.raster_sample_frame(
            {"NDVI": np.zeros((2, 2)), "NDBI": np.zeros((3, 3))}
        )


def test_flattening_rejects_a_mismatched_label_grid() -> None:
    with pytest.raises(ValueError, match="labels has shape"):
        prediction.raster_sample_frame(
            {"NDVI": np.zeros((2, 2))}, labels=np.zeros((3, 3), dtype=int)
        )


def test_predicting_a_surface_leaves_incomplete_pixels_as_nan(
    params_copy: dict[str, Any]
) -> None:
    # A forest handed an imputed predictor returns a confident number built on
    # an invention. NaN is the honest answer for a pixel we do not have.
    pytest.importorskip("sklearn")
    params_copy["prediction"]["rf"]["n_trees"] = 20
    params_copy["prediction"]["rf"]["min_sample_rows"] = 50

    frame = _autocorrelated_sample(params_copy, seed=4)
    fitted = prediction.fit_sklearn_rf(frame, params_copy)

    names = fitted["predictors"]
    arrays = {name: np.full((4, 4), 0.5) for name in names}
    arrays[names[0]] = arrays[names[0]].copy()
    arrays[names[0]][0, 0] = np.nan

    surface = prediction.predict_surface(fitted, arrays, params_copy)
    assert surface.shape == (4, 4)
    assert math.isnan(surface[0, 0])
    assert np.isfinite(surface[1:, 1:]).all()


def test_predicting_a_surface_rejects_a_missing_predictor(
    params_copy: dict[str, Any]
) -> None:
    pytest.importorskip("sklearn")
    params_copy["prediction"]["rf"]["n_trees"] = 20
    params_copy["prediction"]["rf"]["min_sample_rows"] = 50

    frame = _autocorrelated_sample(params_copy, seed=5)
    fitted = prediction.fit_sklearn_rf(frame, params_copy)
    with pytest.raises(ValueError, match="missing"):
        prediction.predict_surface(fitted, {"NDVI": np.zeros((2, 2))}, params_copy)


# --- registry consistency ----------------------------------------------------


def test_training_sample_selectors_lead_with_the_block_coordinates(
    params: dict[str, Any]
) -> None:
    # Without x and y there is no blocked split, only a random one.
    selectors = prediction.training_sample_selectors(params)
    assert selectors[:2] == ["x", "y"]
    assert selectors[2] == params["prediction"]["rf"]["response"]
    assert selectors[3:] == prediction.resolve_predictors(None, params)


def test_every_predictor_is_either_a_phase_5_covariate_or_added_here(
    params: dict[str, Any]
) -> None:
    from colombo_uhi import spatial_stats

    covariates = set(spatial_stats.resolve_regression_predictors(None, params))
    added = {"lcz_class"}
    assert set(prediction.resolve_predictors(None, params)) <= covariates | added


def test_the_response_is_the_landsat_lst_band_name(params: dict[str, Any]) -> None:
    assert (
        params["prediction"]["rf"]["response"]
        == params["landsat_c2l2"]["lst_band_name"]
    )


def test_the_training_source_is_a_single_sensor_family(
    params: dict[str, Any]
) -> None:
    # CLAUDE.md: the Collection 2 inter-calibration was tested over Colombo and
    # FAILED. A model trained on a pooled series would learn the L7->L8 step
    # (-2.48 degC, 3.4x the whole trend signal) as if it were geography.
    key = params["prediction"]["rf"]["source"]
    source = next(
        entry for entry in params["uhi"]["suhii"]["sources"] if entry["key"] == key
    )
    assert source.get("sensors"), (
        f"prediction.rf.source {key!r} does not restrict its sensors; only a "
        "single-family source may train the projection model"
    )


# =============================================================================
# Colab run 1 regressions
# =============================================================================
# Every test below pins a defect the first Colab run actually produced. They are
# grouped here rather than filed under the feature they touch, because what they
# have in common is more useful than what they test: each one is a way a
# confident, plausible, WRONG product reached the end of a pipeline.


# --- D1: the analysis region -------------------------------------------------


def test_no_region_default_resolves_to_the_compositing_bounding_box() -> None:
    # THE run-1 defect. aoi.analysis_region is Western Province buffered by the
    # 25 km SUHII ring - a bounding box for masks and composites, 18,090 km2
    # against the district's 699. Using it as the analysis unit made elevation
    # the top predictor (a regional lapse gradient), inflated R2, collapsed the
    # blocked-vs-random gap to +0.009, and had the CA projecting land-cover
    # transitions on the Indian Ocean.
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(prediction))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "analysis_region"
    ]
    assert not offenders, (
        f"prediction.py calls aoi.analysis_region at line(s) {offenders}. "
        "Phase 6's analysis unit is Colombo District - use work_region()."
    )


def test_work_region_docstring_names_the_helper_it_is_not() -> None:
    # A future reader reaching for analysis_region must hit the reason not to.
    doc = prediction.work_region.__doc__ or ""
    assert "analysis_region" in doc
    assert "18,090" in doc or "699" in doc


def test_the_district_area_is_configured_for_the_notebook_guard(
    params: dict[str, Any]
) -> None:
    # Notebook 06 Step 0 measures the region and compares against this. Without
    # it the guard has nothing to check.
    assert params["aoi"]["expected_areas_km2"]["district"] == 699


# --- D2: water ----------------------------------------------------------------


def test_the_forest_masks_water_by_default(params: dict[str, Any]) -> None:
    # LST over open water is not driven by NDVI, NDBI or built fraction, so
    # water pixels teach a relationship that does not exist.
    assert prediction.resolve_rf_settings(params)["mask_water"] is True


def test_scoring_excludes_a_cell_that_is_immutable_in_either_year(
    params: dict[str, Any]
) -> None:
    # A cell that was water at either end of the interval is dropped. Scoring a
    # class the CA is forbidden from changing pads the diagonal.
    early = np.array([[0, 1], [6, 6]])
    late = np.array([[1, 1], [6, 0]])
    mask, report = prediction.scoring_mask(params, early, late)
    assert mask.tolist() == [[False, True], [True, False]]
    assert report["n_excluded"] == 2
    assert report["excluded_fraction"] == pytest.approx(0.5)


def test_scoring_can_be_told_to_keep_everything(params: dict[str, Any]) -> None:
    mask, report = prediction.scoring_mask(
        params, np.array([[0, 6]]), exclude_immutable=False
    )
    assert mask.all()
    assert report["n_excluded"] == 0
    assert report["exclude_immutable"] is False


def test_scoring_mask_rejects_arrays_of_different_shapes(
    params: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="share a shape"):
        prediction.scoring_mask(params, np.zeros((2, 2)), np.zeros((3, 3)))


def test_scoring_mask_needs_at_least_one_array(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="at least one"):
        prediction.scoring_mask(params)


def test_excluding_water_changes_the_headline_kappa(
    params: dict[str, Any], classes: list[int]
) -> None:
    # Run 1's Kappa of 0.928 was built on 32% water. Make the difference
    # visible: a map that is mostly immutable water scores very differently
    # once the water is dropped.
    rng = np.random.default_rng(21)
    initial = np.where(rng.random((60, 60)) < 0.6, 0, 6)
    observed = initial.copy()
    land = initial != 0
    flip = land & (rng.random(initial.shape) < 0.4)
    observed[flip] = 1
    projected = initial.copy()

    everything = prediction.validate_projection(
        initial, observed, projected, params, classes
    )
    mask, _ = prediction.scoring_mask(params, initial, observed)
    land_only = prediction.validate_projection(
        initial, observed, projected, params, classes, mask=mask
    )
    assert everything["kappa"] > land_only["kappa"]
    assert land_only["n_scored"] < everything["n_scored"]


# --- validate_projection ------------------------------------------------------


def test_the_validation_bundle_agrees_with_the_individual_metrics(
    params: dict[str, Any], classes: list[int]
) -> None:
    rng = np.random.default_rng(22)
    initial = rng.choice(classes, (40, 40))
    observed = initial.copy()
    observed[rng.random(initial.shape) < 0.15] = 6
    projected = initial.copy()
    projected[rng.random(initial.shape) < 0.10] = 6

    bundle = prediction.validate_projection(
        initial, observed, projected, params, classes
    )
    matrix = prediction.confusion_matrix(
        observed.ravel(), projected.ravel(), classes
    )
    assert bundle["kappa"] == pytest.approx(prediction.cohen_kappa(matrix))
    assert bundle["persistence_kappa"] == pytest.approx(
        prediction.persistence_baseline_kappa(
            initial.ravel(), observed.ravel(), classes
        )
    )
    assert bundle["kappa_above_null"] == pytest.approx(
        bundle["kappa"] - bundle["persistence_kappa"]
    )


def test_the_validation_bundle_carries_every_metric_that_must_be_read_together(
    params: dict[str, Any], classes: list[int]
) -> None:
    initial = np.array([[0, 1], [6, 6]])
    bundle = prediction.validate_projection(
        initial, initial, initial, params, classes
    )
    for name in (
        "kappa", "persistence_kappa", "kappa_above_null", "figure_of_merit",
        "quantity_disagreement", "allocation_disagreement", "hits", "misses",
    ):
        assert name in bundle


def test_the_validation_bundle_refuses_an_empty_scoring_set(
    params: dict[str, Any], classes: list[int]
) -> None:
    water = np.zeros((4, 4), dtype=int)
    mask, _ = prediction.scoring_mask(params, water)
    with pytest.raises(ValueError, match="nothing"):
        prediction.validate_projection(
            water, water, water, params, classes, mask=mask
        )


def test_the_validation_bundle_rejects_a_mismatched_mask(
    params: dict[str, Any], classes: list[int]
) -> None:
    grid = np.zeros((3, 3), dtype=int)
    with pytest.raises(ValueError, match="mask has shape"):
        prediction.validate_projection(
            grid, grid, grid, params, classes, mask=np.ones((2, 2), dtype=bool)
        )


# --- D3: class grouping -------------------------------------------------------


def test_the_grouping_covers_every_retained_class(
    params: dict[str, Any], classes: list[int]
) -> None:
    # A class the grouping misses would be silently dropped from the grouped
    # run, and the two schemes would stop being comparable.
    assert set(prediction.resolve_class_grouping(params)) == set(classes)


def test_the_grouping_collapses_the_unstable_vegetation_classes(
    params: dict[str, Any]
) -> None:
    # Run 1: grass persisted 0.399 with 0.425 going to trees; shrub persisted
    # 0.364 with 0.484 going to trees. Grouping exists to suppress exactly that.
    grouping = prediction.resolve_class_grouping(params)
    assert grouping[1] == grouping[2] == grouping[5]  # trees, grass, shrub
    assert grouping[6] != grouping[1]                 # built stays separate
    assert grouping[0] != grouping[1]                 # water stays separate
    assert grouping[4] != grouping[1]                 # crops stay separate


def test_grouping_rejects_a_class_it_does_not_cover(
    params_copy: dict[str, Any]
) -> None:
    params_copy["prediction"]["ca_markov"]["class_grouping"].pop(5)
    with pytest.raises(ValueError, match=r"\[5\]"):
        prediction.resolve_class_grouping(params_copy)


def test_grouping_rejects_a_target_with_no_label(
    params_copy: dict[str, Any]
) -> None:
    params_copy["prediction"]["ca_markov"]["class_grouping"][7] = 99
    with pytest.raises(ValueError, match="99"):
        prediction.resolve_class_grouping(params_copy)


def test_grouped_classes_are_the_distinct_targets(params: dict[str, Any]) -> None:
    grouping = prediction.resolve_class_grouping(params)
    assert prediction.resolve_grouped_classes(params) == sorted(set(grouping.values()))


def test_group_classes_recodes_the_array(params: dict[str, Any]) -> None:
    grouped = prediction.group_classes(np.array([[1, 2, 5], [0, 6, 4]]), params)
    assert grouped[0].tolist() == [1, 1, 1]      # all three become green
    assert grouped[1].tolist() == [0, 6, 4]      # water, built, crops unchanged


def test_group_classes_passes_a_nodata_sentinel_through(
    params: dict[str, Any]
) -> None:
    # Folding -1 into a real class would put nodata on the map as land cover.
    assert prediction.group_classes(np.array([-1, 2]), params).tolist() == [-1, 1]


def test_group_classes_is_idempotent(params: dict[str, Any]) -> None:
    once = prediction.group_classes(np.array([[1, 2, 5, 6]]), params)
    twice = prediction.group_classes(once, params)
    assert np.array_equal(once, twice)


def test_grouping_reduces_off_diagonal_churn(params: dict[str, Any]) -> None:
    # The point of the whole sensitivity, as a property: merging trees, grass
    # and shrub converts grass->trees transitions into persistence.
    classes = prediction.resolve_ca_classes(params)
    grouped_codes = prediction.resolve_grouped_classes(params)
    early = np.array([2, 2, 5, 5, 1, 6, 6, 0])
    late = np.array([1, 1, 1, 2, 1, 6, 6, 0])  # heavy vegetation churn

    # Both matrices carry classes absent at the earlier date, which warn by
    # design; that behaviour has its own test.
    with pytest.warns(UserWarning, match="never observed"):
        raw = prediction.transition_probabilities(
            prediction.transition_matrix(early, late, classes)
        )
    with pytest.warns(UserWarning, match="never observed"):
        grouped = prediction.transition_probabilities(
            prediction.transition_matrix(
                prediction.group_classes(early, params),
                prediction.group_classes(late, params),
                grouped_codes,
            )
        )
    green = grouped_codes.index(prediction.resolve_class_grouping(params)[2])
    grass = classes.index(2)
    assert grouped[green, green] > raw[grass, grass]


# --- D3: grouped scenarios ----------------------------------------------------


def test_the_grouped_scenario_reads_its_own_class_lists(
    params: dict[str, Any]
) -> None:
    ungrouped = prediction.resolve_scenario("greening", params)
    grouped = prediction.resolve_scenario("greening", params, grouped=True)
    assert grouped["grouped"] is True
    assert grouped["eligible_classes"] != ungrouped["eligible_classes"]
    assert set(grouped["eligible_classes"]) <= set(
        prediction.resolve_grouped_classes(params)
    )


def test_the_grouped_lever_is_weaker_and_the_config_says_so(
    params: dict[str, Any]
) -> None:
    # Grass and shrub already count as green once grouped, so only bare is left
    # to convert. That is a real consequence of suppressing the churn, and the
    # report has to state it rather than quote the two schemes as equivalent.
    grouped = prediction.resolve_scenario("greening", params, grouped=True)
    ungrouped = prediction.resolve_scenario("greening", params)
    assert len(grouped["eligible_classes"]) < len(ungrouped["eligible_classes"])


def test_the_grouped_scenario_paints_converted_cells_as_trees(
    params: dict[str, Any]
) -> None:
    # The grouped "green" median blends canopy and lawn, which differ
    # thermally. paint_as_class keeps the canopy assumption explicit.
    grouped = prediction.resolve_scenario("greening", params, grouped=True)
    assert grouped["paint_as_class"] == 1


def test_a_scenario_without_a_grouped_block_refuses_to_be_grouped(
    params_copy: dict[str, Any]
) -> None:
    params_copy["prediction"]["scenarios"]["greening"].pop("grouped")
    with pytest.raises(KeyError, match="NOT interchangeable"):
        prediction.resolve_scenario("greening", params_copy, grouped=True)


def test_business_as_usual_needs_no_grouped_block(params: dict[str, Any]) -> None:
    resolved = prediction.resolve_scenario(
        "business_as_usual", params, grouped=True
    )
    assert resolved["canopy_increase_fraction"] == 0.0


def test_a_grouped_scenario_rejects_an_ungrouped_class_code(
    params_copy: dict[str, Any]
) -> None:
    # Class 2 (grass) does not exist once grouped; naming it would silently
    # convert nothing.
    params_copy["prediction"]["scenarios"]["greening"]["grouped"][
        "eligible_classes"
    ] = [2]
    with pytest.raises(ValueError, match="grouped codes"):
        prediction.resolve_scenario("greening", params_copy, grouped=True)


# --- D4: the GHSL cross-check -------------------------------------------------


def test_ghsl_and_the_ca_agree_when_the_ca_mirrors_the_threshold(
    params: dict[str, Any]
) -> None:
    rng = np.random.default_rng(23)
    fraction = rng.random((30, 30))
    threshold = float(params["prediction"]["ghsl_cross_check_threshold"])
    labels = np.where(fraction >= threshold, 6, 1)
    out = prediction.ghsl_built_comparison(
        fraction, labels, params, built_class=6, cell_area_m2=1e4
    )
    assert out["cell_agreement"] == pytest.approx(1.0)
    assert out["cell_difference"] == 0


def test_the_fraction_weighted_area_is_reported_but_not_the_comparison(
    params: dict[str, Any]
) -> None:
    # Run 1 compared 489.5 km2 of fraction-weighted built SURFACE against
    # 1,625 km2 of built-DOMINANT cells and called the 232% gap a
    # disagreement. The two quantities are now reported separately.
    fraction = np.full((10, 10), 0.6)
    labels = np.full((10, 10), 6)
    out = prediction.ghsl_built_comparison(
        fraction, labels, params, built_class=6, cell_area_m2=1e4
    )
    assert out["ghsl_dominant_km2"] == pytest.approx(1.0)
    assert out["ghsl_fraction_weighted_km2"] == pytest.approx(0.6)
    assert out["ghsl_dominant_km2"] != out["ghsl_fraction_weighted_km2"]


def test_the_ghsl_comparison_honours_a_mask(params: dict[str, Any]) -> None:
    fraction = np.full((4, 4), 0.9)
    labels = np.full((4, 4), 6)
    mask = np.zeros((4, 4), dtype=bool)
    mask[0] = True
    out = prediction.ghsl_built_comparison(
        fraction, labels, params, built_class=6, cell_area_m2=1e4, mask=mask
    )
    assert out["ca_built_cells"] == 4
    assert out["ghsl_dominant_cells"] == 4


def test_the_ghsl_comparison_rejects_a_grid_mismatch(
    params: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="one grid"):
        prediction.ghsl_built_comparison(
            np.zeros((3, 3)), np.zeros((4, 4)), params,
            built_class=6, cell_area_m2=1e4,
        )


# --- D5: the priority geometry reader -----------------------------------------


def test_the_committed_zone_geometry_is_the_processed_layout() -> None:
    # This is the file a fresh clone has, and the one run 1 died on: Phase 5
    # already renamed adm4_pcode to zone_id when it wrote it to data/outputs/.
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "outputs" / (
        "gn_divisions_colombo.geojson"
    )
    if not path.exists():  # pragma: no cover - present in the repo
        pytest.skip("the committed GN geometry is absent")
    with path.open(encoding="utf-8") as handle:
        properties = json.load(handle)["features"][0]["properties"]
    assert "zone_id" in properties
    assert "adm4_pcode" not in properties


def test_the_priority_reader_accepts_the_processed_layout(
    params: dict[str, Any], tmp_path: Any
) -> None:
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    frame = gpd.GeoDataFrame(
        {"zone_id": ["LK1103005", "LK1103006"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs=params["crs"]["analysis_epsg"],
    )
    path = tmp_path / "processed.geojson"
    frame.to_file(path, driver="GeoJSON")

    out = prediction.read_priority_geometry(path, params)
    assert list(out["zone_id"]) == ["LK1103005", "LK1103006"]


def test_the_priority_reader_accepts_the_raw_export_layout(
    params: dict[str, Any], tmp_path: Any
) -> None:
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    key = params["spatial_stats"]["geometry"]["export_properties"]["gn"][0]
    frame = gpd.GeoDataFrame(
        {key: ["LK1103005"]},
        geometry=[box(0, 0, 1, 1)],
        crs=params["crs"]["analysis_epsg"],
    )
    path = tmp_path / "raw.geojson"
    frame.to_file(path, driver="GeoJSON")

    out = prediction.read_priority_geometry(path, params)
    assert list(out["zone_id"]) == ["LK1103005"]


def test_the_priority_reader_names_both_layouts_when_it_gets_neither(
    params: dict[str, Any], tmp_path: Any
) -> None:
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    frame = gpd.GeoDataFrame(
        {"nonsense": ["x"]}, geometry=[box(0, 0, 1, 1)],
        crs=params["crs"]["analysis_epsg"],
    )
    path = tmp_path / "wrong.geojson"
    frame.to_file(path, driver="GeoJSON")

    with pytest.raises(ValueError, match="adm4_pcode"):
        prediction.read_priority_geometry(path, params)


def test_the_priority_reader_rejects_duplicate_zone_ids(
    params: dict[str, Any], tmp_path: Any
) -> None:
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    frame = gpd.GeoDataFrame(
        {"zone_id": ["A", "A"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs=params["crs"]["analysis_epsg"],
    )
    path = tmp_path / "dupes.geojson"
    frame.to_file(path, driver="GeoJSON")

    with pytest.raises(ValueError, match="duplicate zone_id"):
        prediction.read_priority_geometry(path, params)


def test_immutable_classes_are_pinned_so_demand_matches_the_constraint(
    params: dict[str, Any], classes: list[int]
) -> None:
    # An immutable class cannot supply or absorb cells. A Markov chain that
    # projects water shrinking therefore hands the allocator a demand it can
    # never meet, and the deficit surfaces against some OTHER class as a
    # shortfall that reads like an allocation failure. Pinning removes the
    # contradiction at source.
    counts = np.full(len(classes), 100.0)
    probabilities = np.full((len(classes), len(classes)), 1.0 / len(classes))
    frame = prediction.projected_class_areas(
        counts, probabilities, 1, 1e4, classes, params
    )
    water = int(params["prediction"]["ca_markov"]["immutable_classes"][0])
    assert int(frame.set_index("class_code").loc[water, "cells"]) == 100
    assert int(frame["cells"].sum()) == int(counts.sum())


def test_pinning_can_be_switched_off(
    params: dict[str, Any], classes: list[int]
) -> None:
    counts = np.array([200.0] + [50.0] * (len(classes) - 1))
    probabilities = np.full((len(classes), len(classes)), 1.0 / len(classes))
    unpinned = prediction.projected_class_areas(
        counts, probabilities, 1, 1e4, classes, params, pin_immutable=False
    )
    water = int(params["prediction"]["ca_markov"]["immutable_classes"][0])
    assert int(unpinned.set_index("class_code").loc[water, "cells"]) != 200


def test_the_ca_leaves_no_shortfall_under_either_scheme(
    params: dict[str, Any], classes: list[int]
) -> None:
    rng = np.random.default_rng(24)
    early = rng.choice(classes, (40, 40))
    late = early.copy()
    late[rng.random(early.shape) < 0.10] = 6

    for grouped in (False, True):
        a, b = (
            (prediction.group_classes(early, params),
             prediction.group_classes(late, params))
            if grouped else (early, late)
        )
        result = prediction.ca_markov_project(
            a, b, params, steps=1, grouped=grouped
        )
        assert int(result["allocation"]["shortfall"].sum()) == 0
        assert set(np.unique(result["labels"]).tolist()) <= set(result["classes"])


def test_the_ca_reports_which_scheme_it_ran(
    params: dict[str, Any], classes: list[int]
) -> None:
    grid = np.array([[0, 1], [6, 6]])
    # Crops and bare are absent from this 4-cell grid, so their transition rows
    # warn by design; that behaviour has its own test.
    with pytest.warns(UserWarning, match="never observed"):
        grouped = prediction.ca_markov_project(
            prediction.group_classes(grid, params),
            prediction.group_classes(grid, params),
            params, steps=1, grouped=True,
        )
    assert grouped["grouped"] is True
    assert grouped["classes"] == prediction.resolve_grouped_classes(params)


def test_the_scheme_resolver_returns_matching_codes_and_labels(
    params: dict[str, Any]
) -> None:
    for grouped in (False, True):
        codes, labels = prediction.resolve_scheme(params, grouped)
        assert set(codes) <= set(labels), grouped


# =============================================================================
# Colab run 2 regressions
# =============================================================================
# Run 2 fixed the region and produced a genuine urban-heat model - NDBI, built
# fraction and LCZ at the top of the importance list, elevation down from first
# to fifth. It then failed for four new reasons, all pinned below.


# --- the analysis grid is the district, not its bounding box -----------------


def test_the_image_builders_clip_to_the_region() -> None:
    # Export.image.toDrive writes the region's BOUNDING BOX. Run 2 was handed a
    # 1,256 km2 raster of which 46 % lay outside Colombo District, and every
    # Track B number was computed over it. Track A escaped only because
    # .sample() clips to the geometry.
    import ast
    import inspect

    source = inspect.getsource(prediction)
    tree = ast.parse(source)
    clipping = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "clip"
            for inner in ast.walk(node)
        )
    }
    for name in ("prediction_stack", "lulc_class_image"):
        assert name in clipping, (
            f"{name} does not clip to its region; an unclipped export hands "
            "Part 2 a rectangle rather than a district"
        )


def test_the_area_check_accepts_the_district(params: dict[str, Any]) -> None:
    scale = float(params["prediction"]["ca_markov"]["raster_scale_m"])
    cells = int(params["aoi"]["expected_areas_km2"]["district"] * 1e6 / scale ** 2)
    report = prediction.require_expected_area(cells, params)
    assert abs(report["departure"]) < 0.01


def test_the_area_check_refuses_run_2s_bounding_box(
    params: dict[str, Any]
) -> None:
    # 297 x 423 cells at 100 m - the exact raster run 2 analysed.
    with pytest.raises(ValueError, match="1,256 km2"):
        prediction.require_expected_area(297 * 423, params)


def test_the_area_check_names_clipping_as_the_likely_cause(
    params: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="not clipped"):
        prediction.require_expected_area(297 * 423, params)


# --- a Kappa below its own baseline is not evidence --------------------------


def test_a_kappa_below_the_persistence_null_is_refused(
    params: dict[str, Any]
) -> None:
    # Run 2's actual numbers. Kappa 0.854 is finite, so the old guard passed it;
    # the no-change baseline was 0.861, so the projection had demonstrated no
    # skill at locating change, and the LST surface resting on it was not a
    # validated product however good its own RMSE was.
    report = prediction.build_validation_report(
        "lst_projection",
        {"rmse": 1.13, "r2": 0.894, "kappa": 0.8537, "persistence_kappa": 0.8609},
        params, held_out=True,
    )
    with pytest.raises(prediction.ValidationMissing, match="no-change baseline"):
        prediction.require_validated(report, params)


def test_a_kappa_above_the_null_still_passes(params: dict[str, Any]) -> None:
    report = prediction.build_validation_report(
        "lst_projection",
        {"rmse": 1.13, "r2": 0.894, "kappa": 0.90, "persistence_kappa": 0.80},
        params, held_out=True,
    )
    assert prediction.require_validated(report, params)["kind"] == "lst_projection"


def test_the_null_check_is_skipped_when_no_baseline_was_computed(
    params: dict[str, Any]
) -> None:
    # A present-day fitted surface involves no CA, so it carries neither Kappa
    # nor a baseline and must not be caught by this check.
    report = prediction.build_validation_report(
        "lst_fit", {"rmse": 1.13, "r2": 0.894}, params, held_out=True
    )
    assert prediction.require_validated(report, params)


def test_the_null_margin_floor_is_configurable(
    params_copy: dict[str, Any]
) -> None:
    # Demanding the model beat the null by a margin, rather than merely tie it.
    params_copy["prediction"]["ca_markov"]["validation"][
        "min_kappa_above_null"
    ] = 0.10
    report = prediction.build_validation_report(
        "lulc_projection", {"kappa": 0.85, "persistence_kappa": 0.80},
        params_copy, held_out=True,
    )
    with pytest.raises(prediction.ValidationMissing, match="margin"):
        prediction.require_validated(report, params_copy)


# --- more than one validation interval ---------------------------------------


def test_the_validation_intervals_resolve(params: dict[str, Any]) -> None:
    intervals = prediction.resolve_validation_intervals(params)
    assert len(intervals) >= 2, (
        "run 2 showed one interval cannot separate classifier churn from a "
        "structural failure of net-demand allocation; a second is the test"
    )
    for early, late, target in intervals:
        assert early < late < target


def test_every_validation_gap_is_a_whole_number_of_steps(
    params: dict[str, Any]
) -> None:
    for early, late, target in prediction.resolve_validation_intervals(params):
        assert (target - late) % (late - early) == 0


def test_a_second_interval_is_longer_than_the_first(
    params: dict[str, Any]
) -> None:
    # The point of the second interval: a longer step lets real change
    # accumulate against a roughly fixed level of classifier noise.
    intervals = prediction.resolve_validation_intervals(params)
    steps = [late - early for early, late, _ in intervals]
    assert len(set(steps)) > 1, f"all intervals use the same step: {steps}"


def test_intervals_reject_a_non_increasing_triplet(
    params_copy: dict[str, Any]
) -> None:
    params_copy["prediction"]["ca_markov"]["validation_intervals"] = [
        [2021, 2018, 2024]
    ]
    with pytest.raises(ValueError, match="strictly increase"):
        prediction.resolve_validation_intervals(params_copy)


def test_intervals_reject_a_gap_the_chain_cannot_reach(
    params_copy: dict[str, Any]
) -> None:
    # A 3-year step cannot land on a date 4 years away.
    params_copy["prediction"]["ca_markov"]["validation_intervals"] = [
        [2018, 2021, 2025]
    ]
    with pytest.raises(ValueError, match="whole number"):
        prediction.resolve_validation_intervals(params_copy)


def test_intervals_fall_back_to_the_single_configured_triplet(
    params_copy: dict[str, Any]
) -> None:
    params_copy["prediction"]["ca_markov"].pop("validation_intervals")
    assert prediction.resolve_validation_intervals(params_copy) == [
        (2018, 2021, 2024)
    ]


# --- the split comparison must compare distributions -------------------------


def test_the_split_comparison_reports_a_spread(params_copy: dict[str, Any]) -> None:
    # Run 2 fitted one split of each and got a NEGATIVE gap of -0.037 - the
    # blocked split scoring higher - while the blocked CV's own fold R2 ranged
    # 0.761 to 0.887. A single pair cannot resolve a difference several times
    # smaller than the spread it is drawn from.
    pytest.importorskip("sklearn")
    params_copy["prediction"]["rf"]["n_trees"] = 30
    params_copy["prediction"]["rf"]["min_sample_rows"] = 100
    params_copy["prediction"]["split"]["block_size_m"] = 4000

    frame = _autocorrelated_sample(params_copy, seed=6)
    out = prediction.compare_split_strategies(frame, params_copy, n_repeats=3)
    assert set(out.columns) >= {
        "method", "n_repeats", "r2_mean", "r2_sd", "r2_min", "r2_max",
    }
    assert (out["n_repeats"] == 3).all()
    assert out["r2_sd"].notna().all()


def test_the_split_comparison_still_marks_only_blocked_reportable(
    params_copy: dict[str, Any]
) -> None:
    pytest.importorskip("sklearn")
    params_copy["prediction"]["rf"]["n_trees"] = 20
    params_copy["prediction"]["rf"]["min_sample_rows"] = 100

    frame = _autocorrelated_sample(params_copy, seed=7)
    out = prediction.compare_split_strategies(
        frame, params_copy, n_repeats=2
    ).set_index("method")
    assert bool(out.loc["spatial_block", "reportable"])
    assert not bool(out.loc[prediction.RANDOM_SPLIT, "reportable"])


def test_the_split_comparison_rejects_zero_repeats(
    params: dict[str, Any]
) -> None:
    frame = pd.DataFrame({"block_id": [0, 1]})
    with pytest.raises(ValueError, match="n_repeats"):
        prediction.compare_split_strategies(frame, params, n_repeats=0)


# =============================================================================
# Colab run 3 regressions
# =============================================================================
# Run 3 confirmed the clip (688.9 km2, -1.5 % from the district) and settled
# Track B. It then stopped dead at the end of Step 9, because a MEASURED
# NEGATIVE RESULT raised the same exception as "nobody ran a validation" - and
# took Track A's valid products with it.


# --- an absence is not a failure ---------------------------------------------


def test_a_failed_validation_is_a_subclass_of_a_missing_one() -> None:
    # So a caller that already excepts ValidationMissing keeps working, while a
    # caller that wants to tell them apart can.
    assert issubclass(prediction.ValidationFailed, prediction.ValidationMissing)


@pytest.mark.parametrize(
    "metrics, held_out",
    [
        ({"rmse": 1.0, "r2": 0.7}, True),                       # kappa absent
        ({"rmse": 1.0, "r2": 0.7, "kappa": float("nan")}, True),  # kappa nan
    ],
)
def test_no_evidence_reads_as_absent_not_failed(
    params: dict[str, Any], metrics: dict[str, float], held_out: bool
) -> None:
    report = prediction.build_validation_report(
        "lst_projection", metrics, params, held_out=held_out
    )
    verdict = prediction.assess_validation(report, params)
    assert verdict["present"] is False
    assert verdict["valid"] is False


def test_an_absent_report_reads_as_absent(params: dict[str, Any]) -> None:
    verdict = prediction.assess_validation(None, params)
    assert verdict["present"] is False
    assert "NO VALIDATION" in verdict["headline"]


def test_run_3s_below_null_kappa_reads_as_a_computed_failure(
    params: dict[str, Any]
) -> None:
    # Run 3's actual primary-interval numbers. This is a MEASUREMENT, and it
    # belongs in the report - not a crash that discards the rest of the run.
    report = prediction.build_validation_report(
        "lulc_projection",
        {"kappa": 0.8566, "persistence_kappa": 0.8576, "figure_of_merit": 0.0022},
        params, held_out=True,
    )
    verdict = prediction.assess_validation(report, params)
    assert verdict["present"] is True
    assert verdict["valid"] is False
    assert "FAILED VALIDATION" in verdict["headline"]


def test_training_set_metrics_read_as_a_computed_failure(
    params: dict[str, Any]
) -> None:
    report = prediction.build_validation_report(
        "lst_fit", {"rmse": 1.0, "r2": 0.7}, params, held_out=False
    )
    verdict = prediction.assess_validation(report, params)
    assert verdict["present"] is True
    assert verdict["valid"] is False


def test_a_passing_report_is_present_and_valid(params: dict[str, Any]) -> None:
    report = prediction.build_validation_report(
        "lulc_projection", {"kappa": 0.90, "persistence_kappa": 0.80},
        params, held_out=True,
    )
    verdict = prediction.assess_validation(report, params)
    assert verdict == {
        "present": True, "valid": True, "reason": None, "headline": None
    }


def test_require_validated_raises_the_specific_type(
    params: dict[str, Any]
) -> None:
    failed = prediction.build_validation_report(
        "lulc_projection", {"kappa": 0.85, "persistence_kappa": 0.86},
        params, held_out=True,
    )
    with pytest.raises(prediction.ValidationFailed):
        prediction.require_validated(failed, params)
    with pytest.raises(prediction.ValidationMissing):
        prediction.require_validated(None, params)


# --- the caption stamps a failure rather than refusing it --------------------


def test_the_caption_stamps_a_measured_failure(params: dict[str, Any]) -> None:
    # A figure showing that a projection FAILED is exactly what the report
    # needs. Refusing to draw it throws the evidence away.
    report = prediction.build_validation_report(
        "lulc_projection",
        {"kappa": 0.8566, "persistence_kappa": 0.8576, "figure_of_merit": 0.0022},
        params, held_out=True,
    )
    caption = " ".join(prediction.validation_caption(report, params).split())
    assert "FAILED VALIDATION" in caption
    assert "must not be quoted as a projection" in caption
    # ...and the metrics are still on it, so the reader can see the numbers.
    assert "Kappa=0.857" in caption


def test_the_caption_still_refuses_when_nothing_was_computed(
    params: dict[str, Any]
) -> None:
    with pytest.raises(prediction.ValidationMissing):
        prediction.validation_caption(None, params)


def test_a_passing_caption_carries_no_failure_banner(
    params: dict[str, Any]
) -> None:
    report = prediction.build_validation_report(
        "lulc_projection", {"kappa": 0.90, "persistence_kappa": 0.80},
        params, held_out=True,
    )
    assert "FAILED VALIDATION" not in prediction.validation_caption(report, params)


def test_exports_still_refuse_a_measured_failure(params: dict[str, Any]) -> None:
    # The guard has to keep holding where it matters: nothing unvalidated is
    # ever WRITTEN, however freely it may be plotted.
    report = prediction.build_validation_report(
        "lulc_projection", {"kappa": 0.85, "persistence_kappa": 0.86},
        params, held_out=True,
    )
    with pytest.raises(prediction.ValidationFailed):
        prediction.export_projection(None, params, report, None)


def test_writing_a_raster_still_refuses_a_measured_failure(
    params: dict[str, Any], tmp_path: Any
) -> None:
    report = prediction.build_validation_report(
        "lulc_projection", {"kappa": 0.85, "persistence_kappa": 0.86},
        params, held_out=True,
    )
    destination = tmp_path / "must_not_exist.tif"
    with pytest.raises(prediction.ValidationFailed):
        prediction.write_lulc_projection(
            np.zeros((2, 2), dtype=int), {}, destination, params, report
        )
    assert not destination.exists()


# --- nodata must never masquerade as a class ---------------------------------


def test_the_nodata_sentinel_is_not_a_real_class(params: dict[str, Any]) -> None:
    # 0 is WATER. Clipping leaves the bounding box's corners unwritten, and an
    # unmasked read fills them with 0 - so run 3 reported 583 km2 of water in a
    # 689 km2 district.
    assert prediction.LULC_NODATA not in prediction.resolve_ca_classes(params)
    assert prediction.LULC_NODATA < 0


def test_reading_a_clipped_raster_fills_nodata_with_the_sentinel(
    params: dict[str, Any], tmp_path: Any
) -> None:
    rasterio = pytest.importorskip("rasterio")

    labels = np.full((4, 4), 6, dtype="int16")
    observed = np.ones((4, 4), dtype="int16")
    labels[0] = 0          # a row that is genuinely water
    observed[3] = 0        # a row outside the district

    path = tmp_path / "lulc.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=4, width=4, count=2, dtype="int16",
        crs="EPSG:32644", nodata=0,
        transform=rasterio.transform.from_origin(0, 0, 100, 100),
    ) as handle:
        handle.write(labels, 1)
        handle.write(observed, 2)

    read_labels, read_observed, _ = prediction.read_lulc_raster(path, params)
    # nodata=0 means the water row reads back as the sentinel, not as water -
    # which is the honest answer for a file that cannot tell them apart.
    assert (read_labels[0] == prediction.LULC_NODATA).all()
    assert not read_observed[3].any()


# --- the exclusion count belongs inside the analysis grid --------------------


def test_scoring_counts_are_taken_inside_the_analysis_grid(
    params: dict[str, Any]
) -> None:
    # Run 3 printed "58,366 cells excluded as immutable" - 583 km2 of water in a
    # 689 km2 district - because the count spanned the whole raster.
    grid = np.full((100, 100), 6, dtype=np.int16)
    inside = np.zeros((100, 100), dtype=bool)
    inside[:55] = True
    grid[~inside] = prediction.LULC_NODATA
    grid[:2] = np.where(inside[:2], 0, prediction.LULC_NODATA)

    _, wide = prediction.scoring_mask(params, grid)
    _, narrow = prediction.scoring_mask(params, grid, within=inside)

    assert wide["n_total"] == 10_000
    assert narrow["n_total"] == 5_500
    # Both find the same 200 water cells; only the denominator differs.
    assert wide["n_excluded"] == narrow["n_excluded"] == 200
    assert narrow["excluded_fraction"] > wide["excluded_fraction"]


def test_the_scoring_mask_is_confined_to_the_analysis_grid(
    params: dict[str, Any]
) -> None:
    grid = np.full((10, 10), 6, dtype=np.int16)
    inside = np.zeros((10, 10), dtype=bool)
    inside[:4] = True
    mask, _ = prediction.scoring_mask(params, grid, within=inside)
    assert not mask[4:].any()
    assert mask[:4].all()


def test_scoring_rejects_a_mismatched_within_mask(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="within has shape"):
        prediction.scoring_mask(
            params, np.zeros((3, 3), dtype=int),
            within=np.ones((2, 2), dtype=bool),
        )


# --- the primary interval is the longest one --------------------------------


def test_the_primary_interval_is_the_longest_step(params: dict[str, Any]) -> None:
    # Chosen on figure of merit: 0.074 at a 4-year step against 0.002 at three,
    # the one metric a no-change map cannot game. A future edit must not quietly
    # demote it back.
    intervals = prediction.resolve_validation_intervals(params)
    steps = [late - early for early, late, _ in intervals]
    assert steps[0] == max(steps), (
        f"the primary interval has a {steps[0]}-year step but a longer one "
        f"({max(steps)}) is configured; run 3 measured 34x the figure of merit "
        "at the longer step"
    )


def test_the_primary_interval_is_the_longest_the_record_supports(
    params: dict[str, Any]
) -> None:
    # Dynamic World is only reliable from 2017 (2016 measured 56 % coverage),
    # so 2017 -> 2021 -> 2025 is the last equal-step triplet available. That is
    # what makes Track B's failure settled rather than open.
    early, late, target = prediction.resolve_validation_intervals(params)[0]
    step = late - early
    assert early - step < 2017, (
        "a longer step would still fit inside Dynamic World's usable record; "
        "try it before calling Track B settled"
    )
    assert target <= int(params["time"]["end_year"])
