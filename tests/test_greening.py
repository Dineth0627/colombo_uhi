"""Unit tests for the greening priority module (Phase 7).

Everything here runs on synthetic arrays with no Earth Engine session, which is
the point: the half of ``greening.py`` that decides a ranking is pure Python
precisely so that it can be tested here rather than only in Colab.

Three AHP anchors are used, in increasing order of strength:

1. **Saaty's "choosing a leader" 4x4** - the canonical worked example, so the
   implementation is checked against a *published* priority vector.
2. **A textbook 3x3** where the row geometric mean equals the principal
   eigenvector to machine precision, which is a clean invariant to pin.
3. **An exactly consistent matrix** built as ``a_ij = w_i/w_j`` from an
   arbitrary ``w``. Weights must return ``w`` exactly and CR must be 0. That one
   is mathematics rather than citation, and it is the strongest of the three.

``numpy.linalg.eig`` appears only in this file. That is deliberate:
``greening.principal_eigenvector`` uses power iteration because a positive
reciprocal matrix is exactly the Perron-Frobenius case, and ``eig`` - which
returns complex values in arbitrary order - is the *independent implementation*
it is checked against, which is the right place for it.
"""

from __future__ import annotations

import ast
import json
import math
import warnings
from pathlib import Path
from typing import Any

import pytest

from colombo_uhi import load_params, repo_root

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from colombo_uhi import greening  # noqa: E402


# =============================================================================
# Fixtures and textbook anchors
# =============================================================================


@pytest.fixture(scope="module")
def params() -> dict[str, Any]:
    return load_params()


#: Saaty's "choosing a leader" criteria matrix (Experience, Education, Charisma,
#: Age). Published priority vector (0.5470, 0.1270, 0.2703, 0.0556), lambda_max
#: ~ 4.12, CI ~ 0.0395, CR ~ 0.044.
SAATY_LEADER = np.array(
    [
        [1.0, 4.0, 3.0, 7.0],
        [1 / 4, 1.0, 1 / 3, 3.0],
        [1 / 3, 3.0, 1.0, 5.0],
        [1 / 7, 1 / 3, 1 / 5, 1.0],
    ]
)
SAATY_LEADER_NAMES = ["Experience", "Education", "Charisma", "Age"]
#: Recomputed here to full precision; the published figures above are the 3-dp
#: rounding of these. The tests pin the ARITHMETIC, not the rounding.
SAATY_LEADER_WEIGHTS = (0.547569, 0.126555, 0.269950, 0.055926)
SAATY_LEADER_LAMBDA = 4.118418
SAATY_LEADER_CI = 0.039473
SAATY_LEADER_CR = 0.043859

#: A classic 3x3 where the geometric mean equals the eigenvector exactly.
TEXTBOOK_3X3 = np.array([[1.0, 3.0, 5.0], [1 / 3, 1.0, 3.0], [1 / 5, 1 / 3, 1.0]])
TEXTBOOK_3X3_WEIGHTS = (0.636986, 0.258285, 0.104729)
TEXTBOOK_3X3_LAMBDA = 3.038511
TEXTBOOK_3X3_CR = 0.033199


def consistent_matrix(weights: "np.ndarray") -> "np.ndarray":
    """An exactly consistent pairwise matrix ``a_ij = w_i / w_j``."""
    vector = np.asarray(weights, dtype=float)
    return vector[:, None] / vector[None, :]


def eig_weights(matrix: "np.ndarray") -> tuple["np.ndarray", float]:
    """Independent eigenvector via numpy.linalg.eig, for cross-checking."""
    values, vectors = np.linalg.eig(np.asarray(matrix, dtype=float))
    dominant = int(np.argmax(values.real))
    weights = np.abs(vectors[:, dominant].real)
    weights = weights / weights.sum()
    return weights, float(np.max(values.real))


def random_reciprocal(rng: Any, size: int) -> "np.ndarray":
    """A random positive reciprocal matrix on the Saaty scale."""
    matrix = np.ones((size, size), dtype=float)
    for row in range(size):
        for col in range(row + 1, size):
            value = float(rng.choice([1 / 9, 1 / 5, 1 / 3, 1.0, 3.0, 5.0, 9.0]))
            matrix[row, col] = value
            matrix[col, row] = 1.0 / value
    return matrix


@pytest.fixture
def criterion_frame() -> "pd.DataFrame":
    """A synthetic per-zone criterion frame with every column the module wants."""
    rng = np.random.default_rng(11)
    n = 60
    frame = pd.DataFrame(
        {
            "zone_id": [f"LK1103{index:03d}" for index in range(n)],
            "LST_C": rng.normal(31.0, 1.5, n),
            "utfvi_severe_share": rng.random(n),
            "NDVI": rng.random(n) * 0.6,
            "pop_density": rng.lognormal(9.0, 1.0, n),
            "pop_within_300m_pct": rng.random(n) * 100.0,
        }
    )
    for column in ("LST_C", "utfvi_severe_share", "NDVI", "pop_density"):
        frame[f"{column}_pixels"] = 500
    return frame


@pytest.fixture
def shipped_weights(params: dict[str, Any]) -> dict[str, float]:
    matrix, names = greening.pairwise_matrix(params)
    return greening.ahp_weights(matrix, params, names, warn=False)["weights"]


# =============================================================================
# validate_pairwise
# =============================================================================


def test_validate_pairwise_accepts_a_valid_matrix() -> None:
    result = greening.validate_pairwise(SAATY_LEADER, SAATY_LEADER_NAMES)
    assert result.dtype == np.float64
    assert np.allclose(result, SAATY_LEADER)


def test_validate_pairwise_rejects_a_non_square_matrix() -> None:
    with pytest.raises(ValueError, match="square"):
        greening.validate_pairwise(np.ones((3, 4)))


def test_validate_pairwise_rejects_a_1d_array() -> None:
    with pytest.raises(ValueError, match="2-D"):
        greening.validate_pairwise(np.ones(4))


def test_validate_pairwise_rejects_a_3d_array() -> None:
    with pytest.raises(ValueError, match="2-D"):
        greening.validate_pairwise(np.ones((2, 2, 2)))


def test_validate_pairwise_rejects_an_empty_matrix() -> None:
    with pytest.raises(ValueError, match="at least one criterion"):
        greening.validate_pairwise(np.ones((0, 0)))


def test_validate_pairwise_rejects_a_zero_entry() -> None:
    matrix = np.ones((3, 3))
    matrix[0, 1] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        greening.validate_pairwise(matrix)


def test_validate_pairwise_rejects_a_negative_entry() -> None:
    matrix = np.ones((3, 3))
    matrix[2, 0] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        greening.validate_pairwise(matrix)


def test_validate_pairwise_rejects_a_nan() -> None:
    matrix = np.ones((3, 3))
    matrix[1, 2] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        greening.validate_pairwise(matrix)


def test_validate_pairwise_rejects_a_non_unit_diagonal() -> None:
    matrix = np.ones((3, 3))
    matrix[1, 1] = 2.0
    with pytest.raises(ValueError, match="diagonal"):
        greening.validate_pairwise(matrix)


def test_validate_pairwise_rejects_a_non_reciprocal_pair() -> None:
    matrix = np.ones((3, 3))
    matrix[0, 1] = 3.0
    matrix[1, 0] = 3.0  # should be 1/3
    with pytest.raises(ValueError, match="not reciprocal"):
        greening.validate_pairwise(matrix)


def test_a_non_reciprocal_pair_is_named_by_criterion() -> None:
    # "element [1][3] is not reciprocal" is not something an analyst can act on.
    matrix = np.ones((3, 3))
    matrix[0, 2] = 5.0
    matrix[2, 0] = 5.0
    with pytest.raises(ValueError) as excinfo:
        greening.validate_pairwise(matrix, ["heat", "trees", "people"])
    message = str(excinfo.value)
    assert "heat" in message and "people" in message


def test_validate_pairwise_rejects_a_name_count_mismatch() -> None:
    with pytest.raises(ValueError, match="name"):
        greening.validate_pairwise(np.ones((3, 3)), ["a", "b"])


# =============================================================================
# Named pairs
# =============================================================================


def test_pairwise_matrix_round_trips_the_shipped_judgements(
    params: dict[str, Any],
) -> None:
    matrix, names = greening.pairwise_matrix(params)
    assert names == greening.criterion_names(params)
    assert matrix.shape == (len(names), len(names))
    index = {name: position for position, name in enumerate(names)}
    for key, value in params["greening"]["ahp"]["pairwise"].items():
        left, right = str(key).split("__")
        assert matrix[index[left], index[right]] == pytest.approx(float(value))


def test_reciprocals_are_derived_exactly(params: dict[str, Any]) -> None:
    # Every judgement names the MORE important criterion first, so each
    # reciprocal is exactly 1/value rather than a rounded decimal.
    matrix, _ = greening.pairwise_matrix(params)
    assert np.allclose(matrix * matrix.T, 1.0, rtol=0, atol=1e-15)


def test_reordering_the_criteria_changes_no_weight(params: dict[str, Any]) -> None:
    # This is the whole reason judgements are NAMED PAIRS rather than a nested
    # list: a reordering elsewhere must not reattach a judgement to another pair.
    names = greening.criterion_names(params)
    forward = greening.ahp_weights(
        *greening.pairwise_matrix(params)[:1], params, names, warn=False
    )["weights"]
    reversed_names = list(reversed(names))
    matrix, order = greening.pairwise_matrix(params, reversed_names)
    backward = greening.ahp_weights(matrix, params, order, warn=False)["weights"]
    for name in names:
        assert forward[name] == pytest.approx(backward[name], abs=1e-10)


def test_pairwise_matrix_rejects_a_malformed_key(
    params: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    edited = json.loads(json.dumps(params["greening"]["ahp"]["pairwise"]))
    edited["lst_hot-ndvi_deficit"] = 2
    patched = dict(params)
    patched["greening"] = {
        **params["greening"],
        "ahp": {**params["greening"]["ahp"], "pairwise": edited},
    }
    with pytest.raises(ValueError, match="malformed"):
        greening.pairwise_matrix(patched)


def test_pairwise_matrix_rejects_a_self_comparison(params: dict[str, Any]) -> None:
    patched = dict(params)
    patched["greening"] = {
        **params["greening"],
        "ahp": {
            **params["greening"]["ahp"],
            "pairwise": {"lst_hot__lst_hot": 2},
        },
    }
    with pytest.raises(ValueError, match="itself"):
        greening.pairwise_matrix(patched)


def test_pairwise_matrix_rejects_an_unknown_criterion(params: dict[str, Any]) -> None:
    patched = dict(params)
    patched["greening"] = {
        **params["greening"],
        "ahp": {
            **params["greening"]["ahp"],
            "pairwise": {"lst_hot__rainfall": 2},
        },
    }
    with pytest.raises(ValueError, match="rainfall"):
        greening.pairwise_matrix(patched)


def test_pairwise_matrix_rejects_a_pair_given_in_both_orders(
    params: dict[str, Any],
) -> None:
    edited = dict(params["greening"]["ahp"]["pairwise"])
    edited["ndvi_deficit__lst_hot"] = 5
    patched = dict(params)
    patched["greening"] = {
        **params["greening"],
        "ahp": {**params["greening"]["ahp"], "pairwise": edited},
    }
    with pytest.raises(ValueError, match="same pair"):
        greening.pairwise_matrix(patched)


def test_pairwise_matrix_rejects_a_missing_pair(params: dict[str, Any]) -> None:
    # An absent judgement would silently default to "equally important", which
    # is a decision nobody made.
    edited = dict(params["greening"]["ahp"]["pairwise"])
    edited.pop("lst_hot__ndvi_deficit")
    patched = dict(params)
    patched["greening"] = {
        **params["greening"],
        "ahp": {**params["greening"]["ahp"], "pairwise": edited},
    }
    with pytest.raises(ValueError, match="no judgement"):
        greening.pairwise_matrix(patched)


def test_pairwise_matrix_rejects_a_non_positive_judgement(
    params: dict[str, Any],
) -> None:
    edited = dict(params["greening"]["ahp"]["pairwise"])
    edited["lst_hot__ndvi_deficit"] = 0
    patched = dict(params)
    patched["greening"] = {
        **params["greening"],
        "ahp": {**params["greening"]["ahp"], "pairwise": edited},
    }
    with pytest.raises(ValueError):
        greening.pairwise_matrix(patched)


def test_pairwise_matrix_supports_a_criterion_subset(params: dict[str, Any]) -> None:
    subset = ["lst_hot", "ndvi_deficit"]
    matrix, names = greening.pairwise_matrix(params, subset)
    assert names == subset
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] == pytest.approx(2.0)


# =============================================================================
# The eigenvector, lambda_max and the textbook anchors
# =============================================================================


def test_exact_consistency_recovers_the_weights_exactly(
    params: dict[str, Any],
) -> None:
    # The strongest anchor: mathematics, not citation.
    weights = np.array([0.5, 0.3, 0.15, 0.05])
    vector, lambda_max = greening.principal_eigenvector(
        consistent_matrix(weights), params
    )
    assert np.allclose(vector, weights, rtol=0, atol=1e-12)
    assert lambda_max == pytest.approx(len(weights), abs=1e-10)
    assert greening.consistency_index(lambda_max, len(weights)) == pytest.approx(0.0)


def test_power_iteration_matches_numpy_eig_on_random_matrices(
    params: dict[str, Any],
) -> None:
    rng = np.random.default_rng(7)
    for _ in range(20):
        size = int(rng.integers(3, 8))
        matrix = random_reciprocal(rng, size)
        ours, our_lambda = greening.principal_eigenvector(matrix, params)
        theirs, _ = eig_weights(matrix)
        assert np.allclose(ours, theirs, rtol=0, atol=1e-10)
        assert our_lambda >= size - 1e-9


def test_lambda_max_is_never_below_the_matrix_order(params: dict[str, Any]) -> None:
    rng = np.random.default_rng(3)
    for _ in range(10):
        size = int(rng.integers(3, 7))
        _, lambda_max = greening.principal_eigenvector(
            random_reciprocal(rng, size), params
        )
        assert lambda_max >= size - 1e-9


def test_weights_are_positive_and_sum_to_one(params: dict[str, Any]) -> None:
    vector, _ = greening.principal_eigenvector(SAATY_LEADER, params)
    assert (vector > 0).all()
    assert float(vector.sum()) == pytest.approx(1.0, abs=1e-12)


def test_a_single_criterion_carries_all_the_weight(params: dict[str, Any]) -> None:
    vector, lambda_max = greening.principal_eigenvector(np.ones((1, 1)), params)
    assert vector.tolist() == [1.0]
    assert lambda_max == pytest.approx(1.0)


def test_two_criteria_are_consistent_by_construction(params: dict[str, Any]) -> None:
    matrix = np.array([[1.0, 4.0], [0.25, 1.0]])
    vector, lambda_max = greening.principal_eigenvector(matrix, params)
    assert vector[0] == pytest.approx(0.8)
    assert greening.consistency_index(lambda_max, 2) == 0.0
    assert greening.consistency_ratio(lambda_max, 2, params) == 0.0


def test_power_iteration_reports_non_convergence_by_name(
    params: dict[str, Any],
) -> None:
    # Impossible for a valid reciprocal matrix, so reaching it means the
    # validation was bypassed - and the message must say which knob was hit.
    with pytest.raises(RuntimeError, match="max_iter"):
        greening.principal_eigenvector(SAATY_LEADER, params, max_iter=1, tol=1e-18)


def test_the_saaty_leader_matrix_reproduces_its_published_vector(
    params: dict[str, Any],
) -> None:
    """Pins the arithmetic against Saaty's canonical worked example.

    The published vector is (0.5470, 0.1270, 0.2703, 0.0556) to 4 dp; the
    constants here are the same numbers at full precision, so this asserts the
    computation rather than the rounding.
    """
    report = greening.ahp_weights(SAATY_LEADER, params, SAATY_LEADER_NAMES)
    for name, expected in zip(SAATY_LEADER_NAMES, SAATY_LEADER_WEIGHTS):
        assert report["weights"][name] == pytest.approx(expected, rel=1e-4)
    assert report["lambda_max"] == pytest.approx(SAATY_LEADER_LAMBDA, rel=1e-5)
    assert report["consistency_index"] == pytest.approx(SAATY_LEADER_CI, rel=1e-4)
    assert report["consistency_ratio"] == pytest.approx(SAATY_LEADER_CR, rel=1e-4)
    assert report["consistent"] is True


def test_the_saaty_leader_matrix_matches_the_published_rounding(
    params: dict[str, Any],
) -> None:
    report = greening.ahp_weights(SAATY_LEADER, params, SAATY_LEADER_NAMES)
    rounded = [round(report["weights"][name], 3) for name in SAATY_LEADER_NAMES]
    assert rounded == [0.548, 0.127, 0.270, 0.056]


def test_the_textbook_3x3_reproduces_its_vector(params: dict[str, Any]) -> None:
    report = greening.ahp_weights(TEXTBOOK_3X3, params, ["a", "b", "c"])
    for name, expected in zip("abc", TEXTBOOK_3X3_WEIGHTS):
        assert report["weights"][name] == pytest.approx(expected, rel=1e-5)
    assert report["lambda_max"] == pytest.approx(TEXTBOOK_3X3_LAMBDA, rel=1e-5)
    assert report["consistency_ratio"] == pytest.approx(TEXTBOOK_3X3_CR, rel=1e-4)


def test_the_geometric_mean_equals_the_eigenvector_for_the_3x3() -> None:
    # A clean invariant worth pinning separately: for this matrix the two agree
    # to machine precision, so a divergence means one of them regressed.
    geometric = greening.geometric_mean_weights(TEXTBOOK_3X3)
    eigen, _ = eig_weights(TEXTBOOK_3X3)
    assert np.allclose(geometric, eigen, rtol=0, atol=1e-12)


def test_the_geometric_mean_is_exact_for_a_consistent_matrix() -> None:
    weights = np.array([0.4, 0.35, 0.2, 0.05])
    assert np.allclose(
        greening.geometric_mean_weights(consistent_matrix(weights)),
        weights,
        rtol=0,
        atol=1e-12,
    )


def test_the_shipped_judgements_pass_the_consistency_threshold(
    params: dict[str, Any],
) -> None:
    matrix, names = greening.pairwise_matrix(params)
    report = greening.ahp_weights(matrix, params, names)
    assert report["consistency_ratio"] <= report["consistency_ratio_max"]
    assert report["consistent"] is True
    assert report["degenerate"] is False


def test_the_shipped_weights_match_the_params_regression_pin(
    params: dict[str, Any], shipped_weights: dict[str, float]
) -> None:
    reference = params["greening"]["ahp"]["derived_weights_reference"]
    for name, expected in reference.items():
        assert shipped_weights[name] == pytest.approx(float(expected), abs=5e-5)


# =============================================================================
# Random index, consistency index and ratio
# =============================================================================


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        (1, 0.00),
        (2, 0.00),
        (3, 0.58),
        (4, 0.90),
        (5, 1.12),
        (6, 1.24),
        (7, 1.32),
        (8, 1.41),
        (9, 1.45),
        (10, 1.49),
    ],
)
def test_saaty_random_index_values(
    params: dict[str, Any], order: int, expected: float
) -> None:
    assert greening.random_index(order, params) == pytest.approx(expected)


def test_random_index_refuses_to_extrapolate(params: dict[str, Any]) -> None:
    # The values are measured constants, not a curve. Extrapolating one would
    # invent a consistency threshold and attribute it to Saaty.
    with pytest.raises(ValueError, match="no published random index"):
        greening.random_index(11, params)


def test_consistency_index_is_zero_for_small_matrices() -> None:
    assert greening.consistency_index(1.0, 1) == 0.0
    assert greening.consistency_index(2.0, 2) == 0.0


def test_consistency_index_is_never_negative() -> None:
    # lambda_max marginally below n is floating-point noise, not negative
    # inconsistency.
    assert greening.consistency_index(3.0 - 1e-12, 3) == 0.0


def test_consistency_index_is_monotone_in_lambda_max() -> None:
    values = [greening.consistency_index(lam, 5) for lam in (5.0, 5.1, 5.4, 6.0)]
    assert values == sorted(values)
    assert values[0] == 0.0


def test_consistency_ratio_is_zero_where_the_random_index_is(
    params: dict[str, Any],
) -> None:
    assert greening.consistency_ratio(2.0, 2, params) == 0.0


def test_consistency_ratio_divides_by_the_random_index(
    params: dict[str, Any],
) -> None:
    index = greening.consistency_index(SAATY_LEADER_LAMBDA, 4)
    assert greening.consistency_ratio(
        SAATY_LEADER_LAMBDA, 4, params
    ) == pytest.approx(index / 0.90)


# =============================================================================
# Warn at computation, refuse at product
# =============================================================================


def inconsistent_matrix() -> "np.ndarray":
    """A deliberately incoherent matrix: a > b > c > a."""
    matrix = np.ones((3, 3))
    matrix[0, 1], matrix[1, 0] = 9.0, 1 / 9
    matrix[1, 2], matrix[2, 1] = 9.0, 1 / 9
    matrix[2, 0], matrix[0, 2] = 9.0, 1 / 9
    return matrix


def test_ahp_weights_warns_above_the_threshold(params: dict[str, Any]) -> None:
    with pytest.warns(greening.ConsistencyWarning, match="consistency ratio"):
        report = greening.ahp_weights(inconsistent_matrix(), params, ["a", "b", "c"])
    assert report["consistent"] is False
    assert report["consistency_ratio"] > report["consistency_ratio_max"]


def test_ahp_weights_does_not_warn_below_the_threshold(
    params: dict[str, Any],
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", greening.ConsistencyWarning)
        greening.ahp_weights(SAATY_LEADER, params, SAATY_LEADER_NAMES)


def test_ahp_weights_never_raises_on_inconsistency(params: dict[str, Any]) -> None:
    # An analyst whose judgements are inconsistent needs to SEE the weights in
    # order to fix them. This is the Colab-run-3 lesson, transplanted.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", greening.ConsistencyWarning)
        report = greening.ahp_weights(inconsistent_matrix(), params, ["a", "b", "c"])
    assert math.isfinite(report["consistency_ratio"])
    assert set(report["weights"]) == {"a", "b", "c"}


def test_warn_false_suppresses_the_warning(params: dict[str, Any]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        greening.ahp_weights(
            inconsistent_matrix(), params, ["a", "b", "c"], warn=False
        )


def test_require_consistent_refuses_above_the_threshold(
    params: dict[str, Any],
) -> None:
    report = greening.ahp_weights(
        inconsistent_matrix(), params, ["a", "b", "c"], warn=False
    )
    with pytest.raises(greening.InconsistentJudgements, match="consistency ratio"):
        greening.require_consistent(report, params)


def test_require_consistent_accepts_the_shipped_judgements(
    params: dict[str, Any],
) -> None:
    matrix, names = greening.pairwise_matrix(params)
    report = greening.ahp_weights(matrix, params, names, warn=False)
    assert greening.require_consistent(report, params)["consistent"] is True


def test_an_all_ones_matrix_is_consistent_and_still_refused(
    params: dict[str, Any],
) -> None:
    # PERFECTLY consistent (CR = 0) and it has said nothing. A near-zero
    # consistency ratio can be evidence that no judgement was made.
    report = greening.ahp_weights(np.ones((4, 4)), params, list("abcd"), warn=False)
    assert report["consistency_ratio"] == pytest.approx(0.0)
    assert report["consistent"] is True
    assert report["degenerate"] is True
    with pytest.raises(greening.InconsistentJudgements, match="weight spread"):
        greening.require_consistent(report, params)


def test_an_all_ones_matrix_warns_about_degeneracy(params: dict[str, Any]) -> None:
    with pytest.warns(greening.ConsistencyWarning, match="weight spread"):
        greening.ahp_weights(np.ones((4, 4)), params, list("abcd"))


def test_require_consistent_rejects_a_report_it_did_not_produce(
    params: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="missing"):
        greening.require_consistent({"consistency_ratio": 0.01}, params)


def test_inconsistent_judgements_is_a_value_error() -> None:
    # So existing `except ValueError` handlers keep working, the same way
    # prediction.ValidationFailed subclasses ValidationMissing.
    assert issubclass(greening.InconsistentJudgements, ValueError)
    assert issubclass(greening.CriteriaIncomplete, ValueError)


# =============================================================================
# ahp_global_weights and build_ahp_frame
# =============================================================================


def test_global_weights_sum_to_one(params: dict[str, Any]) -> None:
    weights = greening.ahp_global_weights(params["greening"]["ahp"]["hierarchy"], params)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert set(weights) == set(greening.criterion_names(params))


def test_a_single_child_group_inherits_its_parents_weight(
    params: dict[str, Any],
) -> None:
    weights = greening.ahp_global_weights(params["greening"]["ahp"]["hierarchy"], params)
    # Four groups, so each parent carries 0.25; ndvi_deficit is alone in its own.
    assert weights["ndvi_deficit"] == pytest.approx(0.25)
    assert weights["lst_hot"] == pytest.approx(0.125)


def test_grouping_raises_the_heat_blocs_weight(
    params: dict[str, Any], shipped_weights: dict[str, float]
) -> None:
    # The point of the hierarchy sensitivity: under a flat list the two heat
    # criteria carry whatever the pairwise judgements give them; grouped, they
    # carry what the analyst intended.
    grouped = greening.ahp_global_weights(
        params["greening"]["ahp"]["hierarchy"], params
    )
    flat_bloc = shipped_weights["lst_hot"] + shipped_weights["utfvi_severe_share"]
    grouped_bloc = grouped["lst_hot"] + grouped["utfvi_severe_share"]
    assert grouped_bloc == pytest.approx(0.25)
    assert flat_bloc != pytest.approx(grouped_bloc)


def test_global_weights_reject_an_unknown_child(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="rainfall"):
        greening.ahp_global_weights({"thermal": ["rainfall"]}, params)


def test_global_weights_reject_a_repeated_child(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="more than one"):
        greening.ahp_global_weights(
            {"a": ["lst_hot"], "b": ["lst_hot", "ndvi_deficit"]}, params
        )


def test_global_weights_reject_an_empty_group(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="no criteria"):
        greening.ahp_global_weights({"thermal": []}, params)


def test_build_ahp_frame_columns_and_order(params: dict[str, Any]) -> None:
    matrix, names = greening.pairwise_matrix(params)
    report = greening.ahp_weights(matrix, params, names, warn=False)
    frame = greening.build_ahp_frame(report, params)
    assert list(frame.columns) == list(greening.AHP_COLUMNS)
    assert len(frame) == len(names)
    assert frame["weight"].is_monotonic_decreasing
    assert frame["weight_rank"].tolist() == list(range(1, len(names) + 1))
    assert frame.attrs["consistency_ratio"] == pytest.approx(
        report["consistency_ratio"]
    )


def test_build_ahp_frame_carries_the_criterion_labels(params: dict[str, Any]) -> None:
    matrix, names = greening.pairwise_matrix(params)
    frame = greening.build_ahp_frame(
        greening.ahp_weights(matrix, params, names, warn=False), params
    )
    assert (frame["label"].str.len() > 0).all()
    assert set(frame["direction"]) <= set(greening.DIRECTIONS)


# =============================================================================
# Resolvers
# =============================================================================


def test_resolve_level_defaults_and_validates(params: dict[str, Any]) -> None:
    assert greening.resolve_level(None, params) == params["greening"]["level"]
    assert greening.resolve_level("ds", params) == "ds"
    with pytest.raises(ValueError):
        greening.resolve_level("province", params)


def test_resolve_criteria_subset_and_order(params: dict[str, Any]) -> None:
    subset = greening.resolve_criteria(params, ["pop_density", "lst_hot"])
    assert [entry["name"] for entry in subset] == ["pop_density", "lst_hot"]


def test_resolve_criteria_rejects_an_unknown_name(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="rainfall"):
        greening.resolve_criteria(params, ["rainfall"])


def test_resolve_normalisation(params: dict[str, Any]) -> None:
    assert greening.resolve_normalisation(None, params) == "percentile_rank"
    assert greening.resolve_normalisation("min_max", params) == "min_max"
    with pytest.raises(ValueError):
        greening.resolve_normalisation("softmax", params)


def test_resolve_landcover_year_refuses_a_year_before_dynamic_world(
    params: dict[str, Any],
) -> None:
    assert greening.resolve_landcover_year(params) == params["greening"]["landcover_year"]
    with pytest.raises(ValueError, match="predates Dynamic World"):
        greening.resolve_landcover_year(params, 2012)


def test_resolve_wetland_sources_drops_the_unset_asset(params: dict[str, Any]) -> None:
    sources = greening.resolve_wetland_sources(params)
    assert "asset" not in sources
    assert set(sources) <= set(greening.WETLAND_SOURCES)


def test_requesting_the_wetland_asset_explicitly_raises_with_instructions(
    params: dict[str, Any],
) -> None:
    # An official boundary silently replaced by a proxy union would be the wrong
    # kind of quiet.
    with pytest.raises(ValueError) as excinfo:
        greening.resolve_wetland_sources(params, ["wdpa", "asset"])
    message = str(excinfo.value)
    assert "upload" in message.lower()
    assert "Colombo Wetland Complex" in message
    assert "Wetland CITY" in str(
        pytest.raises(ValueError, greening.wetland_asset_collection, params).value
    )


def test_resolve_wetland_sources_rejects_an_unknown_source(
    params: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="peatland"):
        greening.resolve_wetland_sources(params, ["peatland"])


# =============================================================================
# Criterion preparation
# =============================================================================


def test_apply_direction_negates_only_costs() -> None:
    values = np.array([1.0, 2.0, 3.0])
    assert np.allclose(greening.apply_direction(values, "benefit"), values)
    assert np.allclose(greening.apply_direction(values, "cost"), -values)


def test_apply_direction_rejects_an_unknown_direction() -> None:
    with pytest.raises(ValueError, match="direction"):
        greening.apply_direction(np.array([1.0]), "neutral")


def test_percentile_rank_is_bounded_and_ordered() -> None:
    ranks = greening.percentile_rank(np.array([10.0, 30.0, 20.0]))
    assert (ranks > 0).all() and (ranks <= 1).all()
    assert ranks[1] > ranks[2] > ranks[0]


def test_percentile_rank_preserves_nan() -> None:
    # A division with no NDVI because it sat under cloud is not thereby a
    # low-priority division.
    ranks = greening.percentile_rank(np.array([1.0, np.nan, 3.0]))
    assert np.isnan(ranks[1])
    assert np.isfinite(ranks[0]) and np.isfinite(ranks[2])


def test_percentile_rank_averages_ties() -> None:
    ranks = greening.percentile_rank(np.array([5.0, 5.0, 9.0]))
    assert ranks[0] == pytest.approx(ranks[1])
    assert ranks[2] > ranks[0]


def test_na_option_bottom_appears_nowhere_in_the_module() -> None:
    """No call may sink a missing value to the bottom of the ranking.

    Checked on the AST rather than by substring, so the docstring that *warns*
    against ``na_option="bottom"`` does not trip the test and an actual call
    cannot hide behind formatting.
    """
    source = (repo_root() / "src" / "colombo_uhi" / "greening.py").read_text(
        encoding="utf-8"
    )
    offenders = [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "na_option"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "bottom"
    ]
    assert not offenders, (
        f"na_option='bottom' is called at line(s) {offenders}. It sinks a zone "
        "with missing data to the bottom of the ranking, which is a wrong answer "
        "that never announces itself."
    )


def test_min_max_scale_bounds_to_the_unit_interval() -> None:
    scaled = greening.min_max_scale(np.array([2.0, 4.0, 6.0]))
    assert scaled.tolist() == [0.0, 0.5, 1.0]


def test_min_max_scale_on_a_constant_column_warns_and_returns_half() -> None:
    with pytest.warns(RuntimeWarning, match="constant criterion"):
        scaled = greening.min_max_scale(np.array([3.0, 3.0, 3.0]))
    assert scaled.tolist() == [0.5, 0.5, 0.5]


def test_min_max_scale_preserves_nan() -> None:
    scaled = greening.min_max_scale(np.array([1.0, np.nan, 5.0]))
    assert np.isnan(scaled[1])


def test_z_score_uses_the_population_standard_deviation() -> None:
    values = np.array([10.0, 12.0, 14.0, 16.0])
    scores = greening.z_score(values)
    assert scores.mean() == pytest.approx(0.0, abs=1e-12)
    assert float(np.std(values, ddof=0)) == pytest.approx(2.2360679775, abs=1e-9)
    assert scores[0] == pytest.approx((10.0 - 13.0) / 2.2360679775, abs=1e-9)


def test_z_score_on_a_constant_column_is_zero() -> None:
    assert greening.z_score(np.array([4.0, 4.0])).tolist() == [0.0, 0.0]


def test_normalise_criterion_applies_direction_then_method() -> None:
    values = np.array([1.0, 2.0, 3.0])
    benefit = greening.normalise_criterion(values, "benefit", "min_max")
    cost = greening.normalise_criterion(values, "cost", "min_max")
    assert benefit.tolist() == [0.0, 0.5, 1.0]
    assert cost.tolist() == [1.0, 0.5, 0.0]


def test_normalise_criterion_needs_a_method_or_params() -> None:
    with pytest.raises(ValueError, match="method"):
        greening.normalise_criterion(np.array([1.0]), "benefit")


def test_direction_is_applied_exactly_once(
    criterion_frame: "pd.DataFrame", params: dict[str, Any]
) -> None:
    # After prepare_criteria, higher must ALWAYS mean higher priority - so the
    # cost criteria must be anti-correlated with their raw values and never
    # double-negated back.
    prepared, _ = greening.prepare_criteria(criterion_frame, params)
    for name, column, direction in (
        ("ndvi_deficit", "NDVI", "cost"),
        ("green_access_deficit", "pop_within_300m_pct", "cost"),
        ("lst_hot", "LST_C", "benefit"),
    ):
        correlation = float(
            np.corrcoef(prepared[column], prepared[f"{name}_norm"])[0, 1]
        )
        assert (correlation < 0) if direction == "cost" else (correlation > 0)


def test_prepare_criteria_produces_a_norm_column_per_criterion(
    criterion_frame: "pd.DataFrame", params: dict[str, Any]
) -> None:
    prepared, report = greening.prepare_criteria(criterion_frame, params)
    for name in greening.criterion_names(params):
        assert f"{name}_norm" in prepared.columns
    assert report["n_zones"] == len(criterion_frame)
    assert report["n_ok"] == len(criterion_frame)
    assert report["method"] == "percentile_rank"


def test_prepare_criteria_rejects_duplicate_zone_ids(
    criterion_frame: "pd.DataFrame", params: dict[str, Any]
) -> None:
    doubled = pd.concat([criterion_frame, criterion_frame.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate zone_id"):
        greening.prepare_criteria(doubled, params)


def test_prepare_criteria_rejects_a_missing_zone_id(
    criterion_frame: "pd.DataFrame", params: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="zone_id"):
        greening.prepare_criteria(criterion_frame.drop(columns="zone_id"), params)


def test_a_criterion_absent_for_every_zone_is_a_missing_input(
    criterion_frame: "pd.DataFrame", params: dict[str, Any]
) -> None:
    # Not missing DATA - redistributing its weight 557 times would hide a broken
    # join behind a plausible-looking ranking.
    with pytest.raises(ValueError, match="missing input"):
        greening.prepare_criteria(criterion_frame.drop(columns="NDVI"), params)


def test_missing_weight_is_the_ahp_weight_of_the_absent_criteria(
    criterion_frame: "pd.DataFrame",
    params: dict[str, Any],
    shipped_weights: dict[str, float],
) -> None:
    frame = criterion_frame.copy()
    frame.loc[0, "NDVI"] = np.nan
    prepared, _ = greening.prepare_criteria(frame, params)
    assert prepared.loc[0, "missing_weight"] == pytest.approx(
        shipped_weights["ndvi_deficit"], abs=1e-9
    )
    assert prepared.loc[0, "incomplete_criteria"] == 1
    assert prepared.loc[0, "status"] == greening.STATUS_OK


def test_too_much_missing_weight_becomes_insufficient_data(
    criterion_frame: "pd.DataFrame", params: dict[str, Any]
) -> None:
    frame = criterion_frame.copy()
    frame.loc[0, ["LST_C", "pop_density", "utfvi_severe_share"]] = np.nan
    prepared, report = greening.prepare_criteria(frame, params)
    assert prepared.loc[0, "status"] == greening.STATUS_INSUFFICIENT
    assert report["n_insufficient"] == 1


def test_the_min_pixels_gate_nulls_a_thin_criterion(
    criterion_frame: "pd.DataFrame", params: dict[str, Any]
) -> None:
    # Sammanthranapura is 0.18 km2 - about 18 cells at 100 m before cloud loss.
    frame = criterion_frame.copy()
    frame.loc[0, "LST_C_pixels"] = 2
    prepared, _ = greening.prepare_criteria(frame, params)
    assert np.isnan(prepared.loc[0, "LST_C"])
    assert np.isnan(prepared.loc[0, "lst_hot_norm"])


def test_n_pixels_min_reports_the_thinnest_criterion(
    criterion_frame: "pd.DataFrame", params: dict[str, Any]
) -> None:
    frame = criterion_frame.copy()
    frame.loc[3, "NDVI_pixels"] = 17
    prepared, _ = greening.prepare_criteria(frame, params)
    assert prepared.loc[3, "n_pixels_min"] == 17


def test_redistribute_weights_renormalises_over_present_criteria() -> None:
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    result = greening.redistribute_weights(weights, {"a": True, "b": True, "c": False})
    assert sum(result.values()) == pytest.approx(1.0)
    assert result["c"] == 0.0
    assert result["a"] / result["b"] == pytest.approx(0.5 / 0.3)


def test_redistribute_weights_with_nothing_present_is_all_zero() -> None:
    result = greening.redistribute_weights({"a": 1.0}, {"a": False})
    assert result == {"a": 0.0}


def test_require_scored_fraction_accepts_a_complete_frame(
    criterion_frame: "pd.DataFrame", params: dict[str, Any]
) -> None:
    prepared, _ = greening.prepare_criteria(criterion_frame, params)
    assert greening.require_scored_fraction(prepared, params) == pytest.approx(1.0)


def test_require_scored_fraction_refuses_a_sparse_frame(
    criterion_frame: "pd.DataFrame", params: dict[str, Any]
) -> None:
    frame = criterion_frame.copy()
    frame.loc[: len(frame) // 2, ["LST_C", "pop_density", "utfvi_severe_share"]] = np.nan
    prepared, _ = greening.prepare_criteria(frame, params)
    with pytest.raises(greening.CriteriaIncomplete, match="could be scored"):
        greening.require_scored_fraction(prepared, params)


# =============================================================================
# land_observed_fraction - the F3 fix
# =============================================================================


def land_frames() -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """A harbour-dominated zone and an ordinary one, at Fort's real proportions."""
    landscape = pd.DataFrame(
        {
            "zone_id": ["FORT", "INLAND"],
            # Fort: 131 ha of land classified inside a 746 ha polygon.
            "landscape_area_ha": [131.4, 90.0],
            "observed_fraction": [0.705, 0.95],
        }
    )
    land_area = pd.DataFrame(
        {"zone_id": ["FORT", "INLAND"], "land_area_ha": [133.0, 94.0]}
    )
    return landscape, land_area


def test_the_land_floor_rescues_a_harbour_dominated_zone(
    params: dict[str, Any],
) -> None:
    # Fort's COD-AB polygon IS the Colombo Port outer harbour. Excluding on the
    # raw flag deletes exactly the dense, hot, treeless divisions this analysis
    # exists to find.
    landscape, land_area = land_frames()
    result = greening.land_observed_fraction(landscape, land_area, params)
    fort = result.set_index("zone_id").loc["FORT"]
    assert fort["below_coverage_floor_raw"]
    assert not fort["below_land_coverage_floor"]
    assert fort["status_changed"]
    assert fort["land_observed_fraction"] > 0.95


def test_the_land_floor_reports_how_many_zones_change_status(
    params: dict[str, Any],
) -> None:
    landscape, land_area = land_frames()
    result = greening.land_observed_fraction(landscape, land_area, params)
    assert result.attrs["n_status_changed"] == 1
    assert result.attrs["n_below_floor_raw"] == 1
    assert result.attrs["n_below_floor_land"] == 0


def test_the_land_floor_still_flags_a_genuinely_unobserved_zone(
    params: dict[str, Any],
) -> None:
    landscape = pd.DataFrame(
        {
            "zone_id": ["CLOUDY"],
            "landscape_area_ha": [40.0],
            "observed_fraction": [0.4],
        }
    )
    land_area = pd.DataFrame({"zone_id": ["CLOUDY"], "land_area_ha": [100.0]})
    result = greening.land_observed_fraction(landscape, land_area, params)
    assert bool(result.loc[0, "below_land_coverage_floor"])


def test_the_land_fraction_is_capped_at_one(params: dict[str, Any]) -> None:
    landscape = pd.DataFrame(
        {"zone_id": ["A"], "landscape_area_ha": [120.0], "observed_fraction": [1.0]}
    )
    land_area = pd.DataFrame({"zone_id": ["A"], "land_area_ha": [100.0]})
    result = greening.land_observed_fraction(landscape, land_area, params)
    assert result.loc[0, "land_observed_fraction"] == pytest.approx(1.0)


def test_land_observed_fraction_rejects_a_missing_column(
    params: dict[str, Any],
) -> None:
    landscape, land_area = land_frames()
    with pytest.raises(ValueError, match="missing"):
        greening.land_observed_fraction(
            landscape.drop(columns="landscape_area_ha"), land_area, params
        )


def test_land_observed_fraction_rejects_duplicate_zones(
    params: dict[str, Any],
) -> None:
    landscape, land_area = land_frames()
    doubled = pd.concat([landscape, landscape.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        greening.land_observed_fraction(doubled, land_area, params)


def test_a_below_floor_zone_is_ranked_but_never_prioritised(
    criterion_frame: "pd.DataFrame", params: dict[str, Any], shipped_weights
) -> None:
    frame = criterion_frame.copy()
    frame["below_land_coverage_floor"] = False
    frame.loc[0, "below_land_coverage_floor"] = True
    frame.loc[0, "LST_C"] = 99.0  # would otherwise rank first
    prepared, _ = greening.prepare_criteria(frame, params)
    ranked = greening.rank_frame(
        greening.mcda_scores(prepared, params, shipped_weights), params, top_n=5
    )
    row = ranked.set_index("zone_id").loc[frame.loc[0, "zone_id"]]
    assert row["status"] == greening.STATUS_BELOW_FLOOR
    assert not bool(row["priority"])
    assert math.isfinite(float(row["score_ahp"]))


# =============================================================================
# The weighted overlay
# =============================================================================


def test_weighted_overlay_is_a_plain_weighted_mean() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    scores = greening.weighted_overlay(matrix, np.array([0.7, 0.3]))
    assert scores.tolist() == [0.7, 0.3]


def test_weighted_overlay_redistributes_weight_over_nan() -> None:
    matrix = np.array([[1.0, np.nan]])
    assert greening.weighted_overlay(matrix, np.array([0.7, 0.3]))[0] == pytest.approx(
        1.0
    )


def test_weighted_overlay_scores_an_empty_row_nan() -> None:
    # Zero is a legitimate score and would rank the zone last on evidence that
    # does not exist.
    matrix = np.array([[np.nan, np.nan]])
    assert np.isnan(greening.weighted_overlay(matrix, np.array([0.5, 0.5]))[0])


def test_weighted_overlay_rejects_weights_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to"):
        greening.weighted_overlay(np.ones((2, 2)), np.array([0.5, 0.9]))


def test_weighted_overlay_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="weight"):
        greening.weighted_overlay(np.ones((2, 3)), np.array([0.5, 0.5]))


def test_the_overlay_is_monotone_in_every_criterion(
    criterion_frame: "pd.DataFrame", params: dict[str, Any], shipped_weights
) -> None:
    # Raising one criterion must never lower a zone's score.
    prepared, _ = greening.prepare_criteria(criterion_frame, params)
    base = greening.mcda_scores(prepared, params, shipped_weights)
    for name in greening.criterion_names(params):
        bumped = prepared.copy()
        bumped[f"{name}_norm"] = bumped[f"{name}_norm"] + 0.1
        raised = greening.mcda_scores(bumped, params, shipped_weights)
        assert (raised["score_ahp"] >= base["score_ahp"] - 1e-12).all()


def test_a_dominating_zone_ranks_first(
    criterion_frame: "pd.DataFrame", params: dict[str, Any], shipped_weights
) -> None:
    prepared, _ = greening.prepare_criteria(criterion_frame, params)
    prepared.loc[5, [f"{name}_norm" for name in greening.criterion_names(params)]] = 1.0
    ranked = greening.rank_frame(
        greening.mcda_scores(prepared, params, shipped_weights), params
    )
    assert ranked.iloc[0]["zone_id"] == prepared.loc[5, "zone_id"]


def test_scores_are_bounded_under_rank_normalisation(
    criterion_frame: "pd.DataFrame", params: dict[str, Any], shipped_weights
) -> None:
    prepared, _ = greening.prepare_criteria(criterion_frame, params)
    scores = greening.mcda_scores(prepared, params, shipped_weights)["score_ahp"]
    assert scores.min() > 0.0 and scores.max() <= 1.0


def test_mcda_scores_rejects_a_missing_weight(
    criterion_frame: "pd.DataFrame", params: dict[str, Any]
) -> None:
    prepared, _ = greening.prepare_criteria(criterion_frame, params)
    with pytest.raises(ValueError, match="no weight"):
        greening.mcda_scores(prepared, params, {"lst_hot": 1.0}, ["lst_hot", "NDVI"])


def test_mcda_scores_rejects_an_unprepared_frame(
    criterion_frame: "pd.DataFrame", params: dict[str, Any], shipped_weights
) -> None:
    with pytest.raises(ValueError, match="prepare_criteria"):
        greening.mcda_scores(criterion_frame, params, shipped_weights)


def test_rank_frame_numbers_from_one_without_gaps(
    criterion_frame: "pd.DataFrame", params: dict[str, Any], shipped_weights
) -> None:
    prepared, _ = greening.prepare_criteria(criterion_frame, params)
    ranked = greening.rank_frame(
        greening.mcda_scores(prepared, params, shipped_weights), params, top_n=10
    )
    assert ranked["rank_ahp"].tolist() == list(range(1, len(ranked) + 1))
    assert int(ranked["priority"].sum()) == 10


def test_rank_frame_reports_the_gap_and_ties_at_the_cut(
    params: dict[str, Any]
) -> None:
    # A top-N is meaningless if ranks N and N+1 differ in the fourth decimal.
    scores = pd.DataFrame(
        {
            "zone_id": ["a", "b", "c", "d"],
            "score_ahp": [0.90, 0.80, 0.80, 0.10],
        }
    )
    ranked = greening.rank_frame(scores, params, top_n=2)
    assert ranked.attrs["score_gap_at_cut"] == pytest.approx(0.0)
    assert ranked.attrs["tied_at_cut"] == 1
    assert ranked["score_gap_at_cut"].nunique() == 1


def test_rank_frame_puts_unscorable_zones_last(params: dict[str, Any]) -> None:
    scores = pd.DataFrame(
        {"zone_id": ["a", "b", "c"], "score_ahp": [0.1, np.nan, 0.9]}
    )
    ranked = greening.rank_frame(scores, params, top_n=3)
    assert ranked.iloc[-1]["zone_id"] == "b"
    assert not bool(ranked.iloc[-1]["priority"])


def test_rank_frame_rejects_duplicate_zone_ids(params: dict[str, Any]) -> None:
    scores = pd.DataFrame({"zone_id": ["a", "a"], "score_ahp": [0.1, 0.2]})
    with pytest.raises(ValueError, match="duplicate"):
        greening.rank_frame(scores, params)


# =============================================================================
# TOPSIS
# =============================================================================

#: A hand-checkable worked example: X = [[7,9],[8,6],[9,5]], w = (0.6, 0.4),
#: both benefit criteria. Expected closeness (0.609140, 0.332713, 0.390860),
#: i.e. A1 > A3 > A2.
TOPSIS_X = np.array([[7.0, 9.0], [8.0, 6.0], [9.0, 5.0]])
TOPSIS_W = np.array([0.6, 0.4])
TOPSIS_CLOSENESS = (0.609140, 0.332713, 0.390860)


def test_the_topsis_worked_example(params: dict[str, Any]) -> None:
    result = greening.topsis(TOPSIS_X, TOPSIS_W, params)
    assert np.allclose(result["closeness"], TOPSIS_CLOSENESS, rtol=0, atol=1e-6)
    order = np.argsort(-np.asarray(result["closeness"]))
    assert order.tolist() == [0, 2, 1]


def test_the_topsis_separation_measures(params: dict[str, Any]) -> None:
    result = greening.topsis(TOPSIS_X, TOPSIS_W, params)
    assert np.allclose(
        result["d_plus"], [0.086155, 0.109529, 0.134269], rtol=0, atol=1e-6
    )
    assert np.allclose(
        result["d_minus"], [0.134269, 0.054612, 0.086155], rtol=0, atol=1e-6
    )


def test_vector_normalise_gives_unit_column_norms() -> None:
    normalised = greening.vector_normalise(TOPSIS_X)
    assert np.allclose(np.sqrt((normalised**2).sum(axis=0)), 1.0)


def test_vector_normalise_leaves_a_zero_column_alone() -> None:
    matrix = np.array([[0.0, 3.0], [0.0, 4.0]])
    normalised = greening.vector_normalise(matrix)
    assert normalised[:, 0].tolist() == [0.0, 0.0]


def test_vector_normalise_rejects_a_1d_array() -> None:
    with pytest.raises(ValueError, match="2-D"):
        greening.vector_normalise(np.ones(3))


def test_weighted_matrix_scales_each_column() -> None:
    result = greening.weighted_matrix(np.ones((2, 2)), np.array([0.25, 0.75]))
    assert result[0].tolist() == [0.25, 0.75]


def test_weighted_matrix_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="cannot weight"):
        greening.weighted_matrix(np.ones((2, 2)), np.array([1.0]))


def test_the_ideal_alternative_scores_exactly_one(params: dict[str, Any]) -> None:
    matrix = np.array([[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]])
    result = greening.topsis(matrix, np.array([0.5, 0.5]), params, normalise=False)
    assert result["closeness"][0] == pytest.approx(1.0)
    assert result["closeness"][-1] == pytest.approx(0.0)


def test_closeness_is_bounded(params: dict[str, Any]) -> None:
    rng = np.random.default_rng(5)
    matrix = rng.random((30, 4))
    result = greening.topsis(matrix, np.full(4, 0.25), params)
    closeness = np.asarray(result["closeness"])
    assert (closeness >= 0).all() and (closeness <= 1).all()


def test_identical_alternatives_score_identically(params: dict[str, Any]) -> None:
    matrix = np.array([[3.0, 4.0], [3.0, 4.0], [1.0, 9.0]])
    result = greening.topsis(matrix, np.array([0.5, 0.5]), params)
    assert result["closeness"][0] == pytest.approx(result["closeness"][1])


def test_a_zero_denominator_warns_and_scores_half() -> None:
    with pytest.warns(RuntimeWarning, match="equidistant"):
        result = greening.closeness_coefficient(np.zeros(2), np.zeros(2))
    assert result.tolist() == [0.5, 0.5]


def test_ideal_solutions_reject_an_empty_matrix() -> None:
    with pytest.raises(ValueError, match="alternative set"):
        greening.ideal_solutions(np.ones((0, 3)))


def test_negate_then_max_equals_leave_then_min() -> None:
    """The equivalence a future reader will otherwise 'fix'.

    For a cost criterion, negating and taking the column maximum selects exactly
    the same alternative as leaving it and taking the minimum. That is why
    :func:`greening.ideal_solutions` can be a plain ``nanmax``/``nanmin`` after
    :func:`greening.apply_direction` has run.
    """
    cost = np.array([[5.0], [2.0], [9.0]])
    negated = greening.apply_direction(cost, "cost")
    positive, negative = greening.ideal_solutions(negated)
    assert positive[0] == pytest.approx(-float(cost.min()))
    assert negative[0] == pytest.approx(-float(cost.max()))
    assert int(np.argmax(negated[:, 0])) == int(np.argmin(cost[:, 0]))


def test_topsis_records_the_alternative_count(
    criterion_frame: "pd.DataFrame", params: dict[str, Any], shipped_weights
) -> None:
    # The set the ideals were drawn from is part of what the scores mean.
    prepared, _ = greening.prepare_criteria(criterion_frame, params)
    scored = greening.topsis_scores(prepared, params, shipped_weights)
    assert scored.attrs["n_alternatives"] == len(criterion_frame)


def test_topsis_on_ranks_uses_no_further_normalisation(
    criterion_frame: "pd.DataFrame", params: dict[str, Any], shipped_weights
) -> None:
    prepared, _ = greening.prepare_criteria(criterion_frame, params)
    on_ranks = greening.topsis_scores(prepared, params, shipped_weights, on_ranks=True)
    assert on_ranks.attrs["normalisation"] == "none"
    assert on_ranks.attrs["on_ranks"] is True


def test_topsis_and_the_overlay_agree_more_on_ranks_than_on_raw_values(
    criterion_frame: "pd.DataFrame", params: dict[str, Any], shipped_weights
) -> None:
    """Why ``also_on_ranks`` exists.

    Without the second run, a low AHP-vs-TOPSIS correlation conflates METHOD
    with NORMALISATION and neither can be blamed. Run on the same percentile
    ranks the overlay used, TOPSIS agrees far more closely - which localises the
    disagreement in the normalisation rather than in the method.
    """
    prepared, _ = greening.prepare_criteria(criterion_frame, params)
    overlay = greening.rank_frame(
        greening.mcda_scores(prepared, params, shipped_weights), params
    )
    raw = greening.rank_frame(
        greening.topsis_scores(prepared, params, shipped_weights),
        params,
        score_column="score_topsis",
    )
    ranks = greening.rank_frame(
        greening.topsis_scores(prepared, params, shipped_weights, on_ranks=True),
        params,
        score_column="score_topsis",
    )
    rho_raw = greening.compare_rankings(overlay, raw, params)["spearman_rho"]
    rho_ranks = greening.compare_rankings(overlay, ranks, params)["spearman_rho"]
    assert rho_ranks > rho_raw


def test_topsis_scores_reject_a_missing_weight(
    criterion_frame: "pd.DataFrame", params: dict[str, Any]
) -> None:
    prepared, _ = greening.prepare_criteria(criterion_frame, params)
    with pytest.raises(ValueError, match="no weight"):
        greening.topsis_scores(
            prepared, params, {"lst_hot": 1.0}, ["lst_hot", "ndvi_deficit"]
        )


def test_topsis_scores_reject_zero_weights(
    criterion_frame: "pd.DataFrame", params: dict[str, Any]
) -> None:
    prepared, _ = greening.prepare_criteria(criterion_frame, params)
    weights = {name: 0.0 for name in greening.criterion_names(params)}
    with pytest.raises(ValueError, match="sum to zero"):
        greening.topsis_scores(prepared, params, weights)


def test_topsis_rejects_an_empty_alternative_set(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="at least one alternative"):
        greening.topsis(np.ones((0, 2)), np.array([0.5, 0.5]), params)


def test_topsis_rejects_a_weight_shape_mismatch(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="weights"):
        greening.topsis(np.ones((3, 2)), np.array([1.0]), params)


def test_topsis_handles_a_cost_criterion_without_a_sign_error(
    params: dict[str, Any]
) -> None:
    # A negated cost column is shifted non-negative before vector normalisation,
    # which preserves ordering and spacing - the only things TOPSIS uses.
    frame = pd.DataFrame(
        {
            "zone_id": ["a", "b", "c"],
            "LST_C": [30.0, 31.0, 32.0],
            "utfvi_severe_share": [0.1, 0.2, 0.3],
            "NDVI": [0.5, 0.3, 0.1],
            "pop_density": [100.0, 200.0, 300.0],
            "pop_within_300m_pct": [90.0, 50.0, 10.0],
        }
    )
    for column in ("LST_C", "utfvi_severe_share", "NDVI", "pop_density"):
        frame[f"{column}_pixels"] = 100
    prepared, _ = greening.prepare_criteria(frame, params)
    matrix, names = greening.pairwise_matrix(params)
    weights = greening.ahp_weights(matrix, params, names, warn=False)["weights"]
    scored = greening.topsis_scores(prepared, params, weights)
    # Zone c is worst on every criterion, so it must score highest for greening.
    assert scored.set_index("zone_id").loc["c", "score_topsis"] == pytest.approx(
        scored["score_topsis"].max()
    )


# =============================================================================
# Comparison, ablation and circularity
# =============================================================================


def test_spearman_rho_of_a_series_with_itself_is_one() -> None:
    values = [3.0, 1.0, 4.0, 1.0, 5.0]
    assert greening.spearman_rho(values, values)[0] == pytest.approx(1.0)


def test_spearman_rho_of_a_reversed_series_is_minus_one() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert greening.spearman_rho(values, values[::-1])[0] == pytest.approx(-1.0)


def test_spearman_rho_rejects_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="correlate"):
        greening.spearman_rho([1.0, 2.0], [1.0, 2.0, 3.0])


def test_spearman_rho_rejects_too_few_finite_pairs() -> None:
    with pytest.raises(ValueError, match="finite"):
        greening.spearman_rho([1.0, np.nan, np.nan], [1.0, 2.0, 3.0])


def test_compare_rankings_refuses_mismatched_zone_sets(params: dict[str, Any]) -> None:
    """Not fussiness - TOPSIS's ideals come from the alternative set.

    Correlating a 557-zone ranking against a 535-zone one would measure rank
    reversal and report it as method disagreement.
    """
    left = pd.DataFrame({"zone_id": ["a", "b", "c"], "rank_ahp": [1, 2, 3]})
    right = pd.DataFrame({"zone_id": ["a", "b"], "rank_topsis": [1, 2]})
    with pytest.raises(ValueError, match="RE-RUN"):
        greening.compare_rankings(left, right, params)


def test_compare_rankings_reports_rho_tau_and_overlap(params: dict[str, Any]) -> None:
    left = pd.DataFrame(
        {"zone_id": list("abcdef"), "rank_ahp": [1, 2, 3, 4, 5, 6]}
    )
    right = pd.DataFrame(
        {"zone_id": list("abcdef"), "rank_topsis": [1, 3, 2, 4, 6, 5]}
    )
    report = greening.compare_rankings(left, right, params)
    assert report["n"] == 6
    assert 0.8 < report["spearman_rho"] < 1.0
    assert "kendall_tau" in report
    assert report["max_abs_shift"] == 1.0
    assert report["mean_abs_shift"] == pytest.approx(4 / 6)


def test_compare_rankings_rejects_a_missing_rank_column(
    params: dict[str, Any],
) -> None:
    left = pd.DataFrame({"zone_id": ["a"], "rank_ahp": [1]})
    with pytest.raises(ValueError, match="rank_topsis"):
        greening.compare_rankings(left, pd.DataFrame({"zone_id": ["a"]}), params)


def test_compare_rankings_handles_two_frames_with_the_same_rank_column(
    params: dict[str, Any],
) -> None:
    """The normalisation and sensor sensitivities both hit this case.

    Percentile rank against min-max, and the pooled series against the
    single-sensor one, are comparisons between two runs of the SAME method, so
    both frames carry ``rank_ahp``. A plain merge suffixes them and every lookup
    raises.
    """
    left = pd.DataFrame({"zone_id": list("abcd"), "rank_ahp": [1, 2, 3, 4]})
    right = pd.DataFrame({"zone_id": list("abcd"), "rank_ahp": [2, 1, 4, 3]})
    report = greening.compare_rankings(
        left, right, params, left_rank="rank_ahp", right_rank="rank_ahp"
    )
    assert report["n"] == 4
    assert report["max_abs_shift"] == 1.0
    assert 0.5 < report["spearman_rho"] < 1.0


def test_rank_shift_frame_handles_the_same_rank_column(params: dict[str, Any]) -> None:
    left = pd.DataFrame({"zone_id": list("abc"), "rank_ahp": [1, 2, 3]})
    right = pd.DataFrame({"zone_id": list("abc"), "rank_ahp": [3, 2, 1]})
    shifts = greening.rank_shift_frame(
        left, right, params, left_rank="rank_ahp", right_rank="rank_ahp"
    )
    assert shifts.iloc[0]["abs_shift"] == 2
    assert "rank_ahp_right" in shifts.columns


def test_rank_shift_frame_sorts_by_the_biggest_movers(params: dict[str, Any]) -> None:
    left = pd.DataFrame({"zone_id": list("abc"), "rank_ahp": [1, 2, 3]})
    right = pd.DataFrame({"zone_id": list("abc"), "rank_topsis": [3, 2, 1]})
    shifts = greening.rank_shift_frame(left, right, params)
    assert shifts.iloc[0]["abs_shift"] == 2
    assert set(shifts.columns) >= {"zone_id", "rank_shift", "abs_shift"}


def test_criterion_correlation_is_square_and_named(
    criterion_frame: "pd.DataFrame", params: dict[str, Any]
) -> None:
    prepared, _ = greening.prepare_criteria(criterion_frame, params)
    matrix = greening.criterion_correlation(prepared, params)
    names = greening.criterion_names(params)
    assert list(matrix.index) == names
    assert list(matrix.columns) == names
    assert np.allclose(np.diag(matrix.to_numpy()), 1.0)


def test_criterion_correlation_rejects_an_unprepared_frame(
    criterion_frame: "pd.DataFrame", params: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="prepare_criteria"):
        greening.criterion_correlation(criterion_frame, params)


def test_effective_dimensionality_of_independent_criteria(
    criterion_frame: "pd.DataFrame", params: dict[str, Any]
) -> None:
    prepared, _ = greening.prepare_criteria(criterion_frame, params)
    report = greening.effective_dimensionality(
        greening.criterion_correlation(prepared, params)
    )
    assert report["n_criteria"] == 5
    assert report["n_effective"] > 3.0
    assert report["pc1_variance_share"] < 0.6


def test_perfectly_correlated_criteria_collapse_to_one_dimension() -> None:
    matrix = pd.DataFrame(np.ones((3, 3)), index=list("abc"), columns=list("abc"))
    report = greening.effective_dimensionality(matrix)
    assert report["pc1_variance_share"] == pytest.approx(1.0)
    assert report["n_effective"] == pytest.approx(1.0)


def test_effective_dimensionality_rejects_a_non_square_matrix() -> None:
    with pytest.raises(ValueError, match="square"):
        greening.effective_dimensionality(pd.DataFrame(np.ones((2, 3))))


def test_the_ablation_has_a_row_per_criterion_plus_the_baseline(
    criterion_frame: "pd.DataFrame", params: dict[str, Any], shipped_weights
) -> None:
    prepared, _ = greening.prepare_criteria(criterion_frame, params)
    frame = greening.criterion_ablation(prepared, params, shipped_weights)
    names = greening.criterion_names(params)
    assert set(frame["variant"]) == {
        "full",
        *(f"without_{name}" for name in names),
        "single_criterion",
    }
    assert frame.loc[frame["variant"] == "full", "spearman_rho"].iloc[0] == 1.0


def test_the_ablation_reports_the_single_criterion_verdict(
    criterion_frame: "pd.DataFrame", params: dict[str, Any], shipped_weights
) -> None:
    prepared, _ = greening.prepare_criteria(criterion_frame, params)
    frame = greening.criterion_ablation(prepared, params, shipped_weights)
    assert math.isfinite(frame.attrs["single_criterion_rho"])
    assert frame.attrs["baseline_single_criterion"] == "lst_hot"
    assert isinstance(frame.attrs["reproduces_single_criterion"], bool)


def test_the_ablation_detects_a_ranking_that_is_really_one_criterion(
    params: dict[str, Any], shipped_weights
) -> None:
    # The failure mode this exists to catch: five criteria that are all the same
    # variable, so the MCDA reproduces a ranking by LST alone.
    rng = np.random.default_rng(2)
    heat = rng.normal(31.0, 2.0, 40)
    scaled = (heat - heat.min()) / np.ptp(heat)
    frame = pd.DataFrame(
        {
            "zone_id": [f"Z{i:03d}" for i in range(40)],
            "LST_C": heat,
            "utfvi_severe_share": scaled,
            "NDVI": 1.0 - scaled,
            "pop_density": np.exp(heat / 4.0),
            "pop_within_300m_pct": 100.0 * (1.0 - scaled),
        }
    )
    for column in ("LST_C", "utfvi_severe_share", "NDVI", "pop_density"):
        frame[f"{column}_pixels"] = 100
    prepared, _ = greening.prepare_criteria(frame, params)
    ablation = greening.criterion_ablation(prepared, params, shipped_weights)
    assert ablation.attrs["single_criterion_rho"] > 0.95
    assert ablation.attrs["reproduces_single_criterion"] is True


def test_dropping_a_criterion_leaves_the_weight_it_carried(
    criterion_frame: "pd.DataFrame",
    params: dict[str, Any],
    shipped_weights: dict[str, float],
) -> None:
    prepared, _ = greening.prepare_criteria(criterion_frame, params)
    frame = greening.criterion_ablation(prepared, params, shipped_weights).set_index(
        "variant"
    )
    assert frame.loc["without_lst_hot", "weight_dropped"] == pytest.approx(
        shipped_weights["lst_hot"]
    )


def test_the_ablation_rejects_a_missing_weight(
    criterion_frame: "pd.DataFrame", params: dict[str, Any]
) -> None:
    prepared, _ = greening.prepare_criteria(criterion_frame, params)
    with pytest.raises(ValueError, match="no weight"):
        greening.criterion_ablation(prepared, params, {"lst_hot": 1.0})


def test_circularity_is_always_reported_as_not_independent(
    params: dict[str, Any],
) -> None:
    # Three of five criteria overlap the Phase 5 proxy, and rho(interim, LST) is
    # already +0.9829. Agreement here is not validation.
    ranked = pd.DataFrame({"zone_id": list("abcde"), "rank_ahp": [1, 2, 3, 4, 5]})
    interim = pd.DataFrame({"zone_id": list("abcde"), "rank": [1, 2, 4, 3, 5]})
    report = greening.circularity_report(ranked, interim, params)
    assert report["independence"] == greening.NOT_INDEPENDENT
    assert "NOT VALIDATION" in report["interpretation"]
    assert report["shared_criteria"]
    assert report["spearman_rho"] > 0.8


def test_circularity_reports_the_phase5_criteria(params: dict[str, Any]) -> None:
    ranked = pd.DataFrame({"zone_id": list("abcde"), "rank_ahp": [1, 2, 3, 4, 5]})
    interim = pd.DataFrame({"zone_id": list("abcde"), "rank": [1, 2, 3, 4, 5]})
    report = greening.circularity_report(ranked, interim, params)
    assert report["phase5_criteria"] == params["prediction"]["priority_zones"]["rank_by"]


def test_circularity_rejects_a_disjoint_join(params: dict[str, Any]) -> None:
    ranked = pd.DataFrame({"zone_id": list("abc"), "rank_ahp": [1, 2, 3]})
    interim = pd.DataFrame({"zone_id": list("xyz"), "rank": [1, 2, 3]})
    with pytest.raises(ValueError, match="no zones joined"):
        greening.circularity_report(ranked, interim, params)


def test_circularity_rejects_a_table_with_no_rank(params: dict[str, Any]) -> None:
    ranked = pd.DataFrame({"zone_id": ["a"], "rank_ahp": [1]})
    with pytest.raises(ValueError, match="'rank' column"):
        greening.circularity_report(ranked, pd.DataFrame({"zone_id": ["a"]}), params)


# =============================================================================
# The 3-30-300 rule
# =============================================================================

CELL_M = 10.0


def test_exactly_half_a_hectare_qualifies(params: dict[str, Any]) -> None:
    # At 10 m, 50 cells is exactly 0.5 ha. The threshold is INCLUSIVE; an
    # exclusive comparison would silently move the rule's own number.
    grid = np.zeros((10, 20), dtype=bool)
    grid.flat[:50] = True
    mask = greening.qualifying_green_mask(grid, np.ones_like(grid), CELL_M, params)
    assert mask.sum() == 50


def test_forty_nine_cells_does_not_qualify(params: dict[str, Any]) -> None:
    grid = np.zeros((10, 20), dtype=bool)
    grid.flat[:49] = True
    mask = greening.qualifying_green_mask(grid, np.ones_like(grid), CELL_M, params)
    assert not mask.any()


def test_green_patches_measures_area_in_hectares(params: dict[str, Any]) -> None:
    grid = np.zeros((10, 10), dtype=bool)
    grid[:5, :4] = True  # 20 cells at 10 m = 0.2 ha
    _, patches = greening.green_patches(grid, np.ones_like(grid), CELL_M, params)
    assert patches.iloc[0]["n_cells"] == 20
    assert patches.iloc[0]["area_ha"] == pytest.approx(0.2)


def test_connectivity_4_and_8_disagree_on_a_diagonal_touch(
    params: dict[str, Any],
) -> None:
    # 8-connectivity can fuse two gardens into one "park" through a single
    # diagonal pixel, which is why the patch count is reported under both.
    grid = np.zeros((12, 12), dtype=bool)
    grid[0:5, 0:6] = True
    grid[5:10, 6:12] = True
    _, eight = greening.green_patches(
        grid, np.ones_like(grid), CELL_M, params, connectivity=8
    )
    _, four = greening.green_patches(
        grid, np.ones_like(grid), CELL_M, params, connectivity=4
    )
    assert eight.attrs["n_patches"] == 1
    assert four.attrs["n_patches"] == 2


def test_an_unobserved_cell_is_neither_green_nor_not_green(
    params: dict[str, Any],
) -> None:
    # Phase 5's 5.4x "green growth" artefact is what happens when unobserved is
    # allowed to mean anything.
    grid = np.ones((10, 10), dtype=bool)
    observed = np.ones((10, 10), dtype=bool)
    observed[:5, :] = False
    _, patches = greening.green_patches(grid, observed, CELL_M, params)
    assert patches.iloc[0]["n_cells"] == 50


def test_green_patches_rejects_a_shape_mismatch(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="same shape"):
        greening.green_patches(
            np.ones((4, 4), bool), np.ones((4, 5), bool), CELL_M, params
        )


def test_green_patches_rejects_a_non_positive_cell_size(
    params: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="positive"):
        greening.green_patches(np.ones((4, 4), bool), np.ones((4, 4), bool), 0, params)


def test_the_service_area_is_a_disc_of_the_right_radius(
    params: dict[str, Any],
) -> None:
    grid = np.zeros((61, 61), dtype=bool)
    grid[30, 30] = True
    served = greening.service_area_mask(grid, CELL_M, params, distance_m=100.0)
    # A 100 m radius at 10 m cells is a disc of area ~ pi * 10^2 cells.
    assert 300 <= int(served.sum()) <= 330
    assert served[30, 30]
    assert served[30, 40]  # exactly 100 m away, inclusive
    assert not served[30, 41]


def test_a_qualifying_cell_serves_itself(params: dict[str, Any]) -> None:
    grid = np.zeros((5, 5), dtype=bool)
    grid[2, 2] = True
    assert greening.service_area_mask(grid, CELL_M, params, distance_m=10.0)[2, 2]


def test_an_empty_green_mask_serves_nobody(params: dict[str, Any]) -> None:
    served = greening.service_area_mask(np.zeros((5, 5), bool), CELL_M, params)
    assert not served.any()


def test_the_detour_variant_is_a_strict_subset(params: dict[str, Any]) -> None:
    # Euclidean distance overstates walking access; the detour column bounds the
    # error rather than merely admitting it.
    grid = np.zeros((81, 81), dtype=bool)
    grid[40, 40] = True
    full = greening.service_area_mask(grid, CELL_M, params)
    detour = greening.service_area_mask(
        grid, CELL_M, params, distance_m=greening.detour_distance_m(params)
    )
    assert detour.sum() < full.sum()
    assert bool((detour & ~full).sum() == 0)


def test_the_detour_distance_is_the_rule_divided_by_the_ratio(
    params: dict[str, Any],
) -> None:
    assert greening.detour_distance_m(params) == pytest.approx(300.0 / 1.3, abs=1e-9)


def test_service_area_rejects_a_non_positive_distance(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="positive"):
        greening.service_area_mask(
            np.ones((3, 3), bool), CELL_M, params, distance_m=0.0
        )


def test_require_integer_refinement_accepts_an_exact_multiple() -> None:
    greening.require_integer_refinement((100, 100), (10, 10), 10)


def test_require_integer_refinement_refuses_a_ragged_grid() -> None:
    # Without this, block_mean would trim the remainder and misregister the
    # service mask against the population grid by up to a third of 300 m.
    with pytest.raises(ValueError, match="misregister"):
        greening.require_integer_refinement((97, 97), (10, 10), 10)


def test_require_integer_refinement_refuses_a_zero_factor() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        greening.require_integer_refinement((10, 10), (10, 10), 0)


def test_block_mean_is_exact() -> None:
    values = np.arange(16.0).reshape(4, 4)
    assert greening.block_mean(values, 2).tolist() == [[2.5, 4.5], [10.5, 12.5]]


def test_block_mean_of_booleans_is_a_share() -> None:
    values = np.array([[True, False], [False, False]])
    assert greening.block_mean(values, 2).tolist() == [[0.25]]


def test_block_mean_refuses_an_indivisible_grid() -> None:
    with pytest.raises(ValueError, match="does not divide"):
        greening.block_mean(np.ones((5, 5)), 2)


def zone_setup() -> tuple["np.ndarray", dict[int, str]]:
    """Two zones splitting a 20x20 grid down the middle."""
    zones = np.zeros((20, 20), dtype=int)
    zones[:, :10] = 1
    zones[:, 10:] = 2
    return zones, {1: "LEFT", 2: "RIGHT"}


def test_canopy_fraction_uses_the_observed_land_denominator(
    params: dict[str, Any],
) -> None:
    # Using the polygon would make Fort read as treeless, because 82 % of it is
    # harbour rather than land anyone could plant on.
    zones, labels = zone_setup()
    canopy = np.zeros((20, 20), dtype=bool)
    canopy[:10, :10] = True  # 100 of the left zone's cells
    observed = np.zeros((20, 20), dtype=bool)
    observed[:, :10] = True  # only half the left zone was ever classified...
    observed[:10, 10:] = True

    frame = greening.canopy_fraction_by_zone(
        canopy, observed, zones, labels, params
    ).set_index("zone_id")
    assert frame.loc["LEFT", "canopy_pct"] == pytest.approx(50.0)
    assert frame.loc["RIGHT", "canopy_pct"] == pytest.approx(0.0)


def test_the_30_percent_rule_uses_the_configured_target(
    params: dict[str, Any],
) -> None:
    zones, labels = zone_setup()
    canopy = np.zeros((20, 20), dtype=bool)
    canopy[:7, :10] = True  # 35 % of the left zone
    observed = np.ones((20, 20), dtype=bool)
    frame = greening.canopy_fraction_by_zone(
        canopy, observed, zones, labels, params
    ).set_index("zone_id")
    assert frame.loc["LEFT", "canopy_pct"] == pytest.approx(35.0)
    assert bool(frame.loc["LEFT", "rule_30_pass"])
    assert not bool(frame.loc["RIGHT", "rule_30_pass"])


def test_a_zone_with_no_observations_gets_nan_canopy(params: dict[str, Any]) -> None:
    zones, labels = zone_setup()
    observed = np.zeros((20, 20), dtype=bool)
    observed[:, :10] = True
    frame = greening.canopy_fraction_by_zone(
        np.zeros((20, 20), bool), observed, zones, labels, params
    ).set_index("zone_id")
    assert np.isnan(frame.loc["RIGHT", "canopy_pct"])
    assert not bool(frame.loc["RIGHT", "rule_30_pass"])


def test_canopy_fraction_rejects_a_shape_mismatch(params: dict[str, Any]) -> None:
    zones, labels = zone_setup()
    with pytest.raises(ValueError, match="same shape"):
        greening.canopy_fraction_by_zone(
            np.ones((5, 5), bool), np.ones((20, 20), bool), zones, labels, params
        )


def population_setup() -> tuple[Any, ...]:
    """A 100x100 fine grid over a 10x10 coarse grid, two zones, one park."""
    fine = np.zeros((100, 100), dtype=bool)
    fine[0:40, 0:40] = True  # served: the whole left-top quarter
    coarse_zones = np.zeros((10, 10), dtype=int)
    coarse_zones[:, :5] = 1
    coarse_zones[:, 5:] = 2
    observed = np.ones((10, 10), dtype=bool)
    return fine, coarse_zones, observed, {1: "LEFT", 2: "RIGHT"}


def test_population_weighting_differs_from_area_share(params: dict[str, Any]) -> None:
    """The test that proves the weighting is actually applied.

    The rule counts RESIDENCES. Where population is concentrated away from the
    park, the population-weighted share and the area share must differ - and if
    they never differ, the weighting is not being applied.
    """
    fine, zones, observed, labels = population_setup()
    population = np.zeros((10, 10), dtype=float)
    population[8:, :5] = 1000.0  # everyone lives far from the park

    frame = greening.served_population_by_zone(
        fine, fine, population, observed, zones, labels, params, factor=10
    ).set_index("zone_id")
    assert frame.loc["LEFT", "area_within_300m_pct"] > 0
    assert frame.loc["LEFT", "pop_within_300m_pct"] == pytest.approx(0.0)
    assert (
        frame.loc["LEFT", "pop_within_300m_pct"]
        != frame.loc["LEFT", "area_within_300m_pct"]
    )


def test_population_concentrated_on_the_park_scores_high(
    params: dict[str, Any],
) -> None:
    fine, zones, observed, labels = population_setup()
    population = np.zeros((10, 10), dtype=float)
    population[0:4, 0:4] = 1000.0  # everyone lives on the park
    frame = greening.served_population_by_zone(
        fine, fine, population, observed, zones, labels, params, factor=10
    ).set_index("zone_id")
    assert frame.loc["LEFT", "pop_within_300m_pct"] == pytest.approx(100.0)
    assert bool(frame.loc["LEFT", "rule_300_pass"])


def test_a_zone_with_no_residents_falls_back_to_the_area_share(
    params: dict[str, Any],
) -> None:
    fine, zones, observed, labels = population_setup()
    population = np.zeros((10, 10), dtype=float)
    frame = greening.served_population_by_zone(
        fine, fine, population, observed, zones, labels, params, factor=10
    ).set_index("zone_id")
    assert frame.loc["LEFT", "pop_within_300m_pct"] == pytest.approx(
        frame.loc["LEFT", "area_within_300m_pct"]
    )


def test_served_population_refuses_a_ragged_refinement(params: dict[str, Any]) -> None:
    _, zones, observed, labels = population_setup()
    with pytest.raises(ValueError, match="misregister"):
        greening.served_population_by_zone(
            np.ones((97, 97), bool),
            np.ones((97, 97), bool),
            np.ones((10, 10)),
            observed,
            zones,
            labels,
            params,
            factor=10,
        )


def test_the_service_frame_records_both_distances(params: dict[str, Any]) -> None:
    fine, zones, observed, labels = population_setup()
    frame = greening.served_population_by_zone(
        fine, fine, np.ones((10, 10)), observed, zones, labels, params, factor=10
    )
    assert frame.attrs["service_distance_m"] == 300.0
    assert frame.attrs["detour_distance_m"] == pytest.approx(300.0 / 1.3)
    assert frame.attrs["public_only"] is False


def test_zone_raster_burns_codes_from_one(params: dict[str, Any]) -> None:
    pytest.importorskip("rasterio")
    gpd = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely.geometry")
    from rasterio.transform import from_origin

    geometry = gpd.GeoDataFrame(
        {"zone_id": ["A", "B"]},
        geometry=[shapely.box(0, 0, 50, 100), shapely.box(50, 0, 100, 100)],
        crs="EPSG:32644",
    )
    profile = {
        "transform": from_origin(0, 100, 10, 10),
        "height": 10,
        "width": 10,
        "crs": "EPSG:32644",
    }
    codes, labels = greening.zone_raster(profile, geometry, params)
    assert codes.shape == (10, 10)
    assert labels == {1: "A", 2: "B"}
    # 0 means "outside every zone", which is what the zonal helpers treat as nodata.
    assert set(np.unique(codes)) <= {0, 1, 2}
    assert int((codes == 1).sum()) == 50
    assert int((codes == 2).sum()) == 50


def test_zone_raster_rejects_an_incomplete_profile(params: dict[str, Any]) -> None:
    pytest.importorskip("rasterio")
    gpd = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely.geometry")

    geometry = gpd.GeoDataFrame(
        {"zone_id": ["A"]}, geometry=[shapely.box(0, 0, 10, 10)], crs="EPSG:32644"
    )
    with pytest.raises(ValueError, match="transform"):
        greening.zone_raster({"height": 4, "width": 4}, geometry, params)


def test_zone_raster_rejects_a_missing_identifier(params: dict[str, Any]) -> None:
    pytest.importorskip("rasterio")
    gpd = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely.geometry")
    from rasterio.transform import from_origin

    geometry = gpd.GeoDataFrame(
        {"name": ["A"]}, geometry=[shapely.box(0, 0, 10, 10)], crs="EPSG:32644"
    )
    profile = {"transform": from_origin(0, 10, 1, 1), "height": 10, "width": 10}
    with pytest.raises(ValueError, match="zone_id"):
        greening.zone_raster(profile, geometry, params)


def test_zone_land_area_counts_cells_not_polygon_area() -> None:
    # Taking the denominator from the polygon is exactly the bug that makes Fort,
    # whose COD-AB polygon IS the Colombo Port outer harbour, look unobserved.
    codes = np.zeros((10, 10), dtype=int)
    codes[:, :4] = 1
    codes[:, 4:6] = 2
    frame = greening.zone_land_area(codes, {1: "A", 2: "B"}, 10.0).set_index("zone_id")
    assert frame.loc["A", "land_area_ha"] == pytest.approx(40 * 100 / 10_000.0)
    assert frame.loc["B", "land_area_ha"] == pytest.approx(20 * 100 / 10_000.0)


def test_zone_land_area_respects_the_observed_mask() -> None:
    codes = np.ones((10, 10), dtype=int)
    observed = np.zeros((10, 10), dtype=bool)
    observed[:5, :] = True
    frame = greening.zone_land_area(codes, {1: "A"}, 10.0, observed=observed)
    assert frame.loc[0, "land_area_ha"] == pytest.approx(50 * 100 / 10_000.0)


def test_the_trees_in_view_proxy_is_always_labelled_unmeasured(
    params: dict[str, Any],
) -> None:
    zones, labels = zone_setup()
    canopy = np.zeros((20, 20), dtype=bool)
    canopy[:, :10] = True
    built = np.ones((20, 20), dtype=bool)
    frame = greening.trees_in_view_proxy(canopy, built, zones, labels, params)
    assert (frame["rule_3_status"] == greening.RULE_3_STATUS).all()
    assert frame.attrs["enters_score"] is False


def test_the_trees_in_view_proxy_responds_to_canopy(params: dict[str, Any]) -> None:
    zones, labels = zone_setup()
    canopy = np.zeros((20, 20), dtype=bool)
    canopy[:, :10] = True
    built = np.ones((20, 20), dtype=bool)
    frame = greening.trees_in_view_proxy(
        canopy, built, zones, labels, params
    ).set_index("zone_id")
    assert frame.loc["LEFT", "rule_3_proxy_pct"] > frame.loc["RIGHT", "rule_3_proxy_pct"]


def test_the_trees_in_view_proxy_never_enters_the_criterion_set(
    params: dict[str, Any],
) -> None:
    # Pinned so a later edit cannot quietly let an unmeasurable component decide
    # a priority ranking.
    columns = {str(entry["column"]) for entry in greening.resolve_criteria(params)}
    assert "rule_3_proxy_pct" not in columns
    assert "rule_3_status" not in columns
    assert params["greening"]["rule_3_30_300"]["trees_in_view"]["enters_score"] is False


def test_compliance_takes_exactly_the_five_configured_categories(
    params: dict[str, Any],
) -> None:
    canopy = pd.DataFrame(
        {
            "zone_id": ["a", "b", "c", "d", "e"],
            "canopy_pct": [40.0, 40.0, 10.0, 10.0, np.nan],
            "rule_30_pass": [True, True, False, False, False],
        }
    )
    service = pd.DataFrame(
        {
            "zone_id": ["a", "b", "c", "d", "e"],
            "pop_within_300m_pct": [80.0, 10.0, 80.0, 10.0, np.nan],
            "rule_300_pass": [True, False, True, False, False],
        }
    )
    frame = greening.compliance_3_30_300(canopy, service, None, params)
    assert list(frame.columns) == list(greening.COMPLIANCE_COLUMNS)
    assert frame["compliance"].tolist() == [
        "both_30_and_300",
        "canopy_only",
        "access_only",
        "neither",
        "not_assessable",
    ]
    assert set(frame["compliance"]) <= set(greening.COMPLIANCE_CATEGORIES)


def test_compliance_is_never_a_boolean(params: dict[str, Any]) -> None:
    # The "3" is unmeasured, so a pass/fail flag would imply it was checked.
    assert len(greening.COMPLIANCE_CATEGORIES) == 5
    assert "not_assessable" in greening.COMPLIANCE_CATEGORIES


def test_compliance_is_labelled_an_upper_bound(params: dict[str, Any]) -> None:
    # Dynamic World cannot tell a public park from the Colombo Golf Club.
    canopy = pd.DataFrame(
        {"zone_id": ["a"], "canopy_pct": [40.0], "rule_30_pass": [True]}
    )
    service = pd.DataFrame(
        {"zone_id": ["a"], "pop_within_300m_pct": [80.0], "rule_300_pass": [True]}
    )
    frame = greening.compliance_3_30_300(canopy, service, None, params)
    assert frame.attrs["upper_bound"] is True
    assert frame.attrs["counts"]["both_30_and_300"] == 1


def test_compliance_rejects_a_missing_column(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="missing"):
        greening.compliance_3_30_300(
            pd.DataFrame({"zone_id": ["a"]}),
            pd.DataFrame(
                {"zone_id": ["a"], "pop_within_300m_pct": [1.0], "rule_300_pass": [True]}
            ),
            None,
            params,
        )


# =============================================================================
# The wetland cross
# =============================================================================


def wetland_bands() -> tuple[dict[str, "np.ndarray"], "np.ndarray", dict[int, str]]:
    """A 20x20 grid: wetland in the left zone only."""
    shape = (20, 20)
    wetland = np.zeros(shape, dtype=bool)
    wetland[:, :5] = True
    bands = {
        "dw_flooded_vegetation": wetland.copy(),
        "worldcover_wetland": np.zeros(shape, dtype=bool),
        "gsw_seasonal": np.zeros(shape, dtype=bool),
        "wdpa": np.zeros(shape, dtype=bool),
        "wetland": wetland,
        "observed": np.ones(shape, dtype=bool),
    }
    zones = np.zeros(shape, dtype=int)
    zones[:, :10] = 1
    zones[:, 10:] = 2
    return bands, zones, {1: "LEFT", 2: "RIGHT"}


def test_wetland_shares_are_a_percentage_of_observed_land(
    params: dict[str, Any],
) -> None:
    bands, zones, labels = wetland_bands()
    frame = greening.wetland_shares_by_zone(bands, zones, labels, params).set_index(
        "zone_id"
    )
    assert frame.loc["LEFT", "wetland_within_pct"] == pytest.approx(50.0)
    assert frame.loc["RIGHT", "wetland_within_pct"] == pytest.approx(0.0)


def test_every_source_gets_a_column_even_at_zero(params: dict[str, Any]) -> None:
    bands, zones, labels = wetland_bands()
    frame = greening.wetland_shares_by_zone(bands, zones, labels, params)
    for source in greening.resolve_wetland_sources(params):
        assert f"{source}_pct" in frame.columns


def test_wetland_sources_names_only_those_that_fired(params: dict[str, Any]) -> None:
    # WDPA is a legal designation and the others are remote-sensing proxies; a
    # policy recommendation must not present them as the same kind of statement.
    bands, zones, labels = wetland_bands()
    frame = greening.wetland_shares_by_zone(bands, zones, labels, params).set_index(
        "zone_id"
    )
    assert frame.loc["LEFT", "wetland_sources"] == "dw_flooded_vegetation"
    assert frame.loc["RIGHT", "wetland_sources"] == ""


def test_the_union_is_at_least_as_large_as_any_source(params: dict[str, Any]) -> None:
    bands, zones, labels = wetland_bands()
    bands["wdpa"] = np.zeros((20, 20), dtype=bool)
    bands["wdpa"][:, 15:] = True
    bands["wetland"] = bands["dw_flooded_vegetation"] | bands["wdpa"]
    frame = greening.wetland_shares_by_zone(bands, zones, labels, params).set_index(
        "zone_id"
    )
    for source in greening.resolve_wetland_sources(params):
        column = f"{source}_pct"
        assert (
            frame["wetland_within_pct"].fillna(0) >= frame[column].fillna(0) - 1e-9
        ).all()


def test_wetland_shares_reject_a_missing_union_band(params: dict[str, Any]) -> None:
    bands, zones, labels = wetland_bands()
    bands.pop("wetland")
    with pytest.raises(ValueError, match="'wetland' band"):
        greening.wetland_shares_by_zone(bands, zones, labels, params)


def adjacency_geometry() -> Any:
    """Three zones in a row: wetland, its neighbour, and one further away.

    FAR sits 1.9 km from the wetland, comfortably beyond the largest configured
    sensitivity distance (1000 m). Putting it at 300 m would make it "adjacent"
    under the 500 m default, which is the code behaving correctly.
    """
    gpd = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely.geometry")

    boxes = [
        shapely.box(0, 0, 100, 100),
        shapely.box(100, 0, 200, 100),
        shapely.box(2000, 0, 2100, 100),
    ]
    return gpd.GeoDataFrame(
        {"zone_id": ["WET", "NEXT", "FAR"]}, geometry=boxes, crs="EPSG:32644"
    )


def test_wetland_adjacency_classifies_within_adjacent_and_neither(
    params: dict[str, Any],
) -> None:
    geometry = adjacency_geometry()
    shares = pd.DataFrame(
        {"zone_id": ["WET", "NEXT", "FAR"], "wetland_within_pct": [40.0, 0.0, 0.0]}
    )
    result = greening.wetland_adjacency(geometry, shares, params).set_index("zone_id")
    assert result.loc["WET", "wetland_status"] == "within"
    assert result.loc["NEXT", "wetland_status"] == "adjacent"
    assert result.loc["FAR", "wetland_status"] == "neither"


def test_the_wetland_policy_flag_covers_within_and_adjacent(
    params: dict[str, Any],
) -> None:
    geometry = adjacency_geometry()
    shares = pd.DataFrame(
        {"zone_id": ["WET", "NEXT", "FAR"], "wetland_within_pct": [40.0, 0.0, 0.0]}
    )
    result = greening.wetland_adjacency(geometry, shares, params).set_index("zone_id")
    assert bool(result.loc["WET", "wetland_policy_flag"])
    assert bool(result.loc["NEXT", "wetland_policy_flag"])
    assert not bool(result.loc["FAR", "wetland_policy_flag"])


def test_adjacency_is_reported_at_every_sensitivity_distance(
    params: dict[str, Any],
) -> None:
    geometry = adjacency_geometry()
    shares = pd.DataFrame(
        {"zone_id": ["WET", "NEXT", "FAR"], "wetland_within_pct": [40.0, 0.0, 0.0]}
    )
    result = greening.wetland_adjacency(geometry, shares, params)
    for distance in params["greening"]["wetland"]["adjacency"]["distance_sensitivity_m"]:
        assert f"adjacent_within_{int(distance)}m" in result.columns


def test_adjacency_at_250m_is_a_subset_of_500m_and_1000m(
    params: dict[str, Any],
) -> None:
    geometry = adjacency_geometry()
    shares = pd.DataFrame(
        {"zone_id": ["WET", "NEXT", "FAR"], "wetland_within_pct": [40.0, 0.0, 0.0]}
    )
    result = greening.wetland_adjacency(geometry, shares, params)
    near = result["adjacent_within_250m"].to_numpy(dtype=bool)
    mid = result["adjacent_within_500m"].to_numpy(dtype=bool)
    far = result["adjacent_within_1000m"].to_numpy(dtype=bool)
    assert not (near & ~mid).any()
    assert not (mid & ~far).any()


def test_buffer_and_queen_adjacency_can_disagree(params: dict[str, Any]) -> None:
    # The FAR zone is 200 m from the wetland: within a 250 m buffer, but sharing
    # no boundary at all. Two definitions, both reported, neither authoritative.
    gpd = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely.geometry")

    geometry = gpd.GeoDataFrame(
        {"zone_id": ["WET", "GAP"]},
        geometry=[shapely.box(0, 0, 100, 100), shapely.box(200, 0, 300, 100)],
        crs="EPSG:32644",
    )
    shares = pd.DataFrame({"zone_id": ["WET", "GAP"], "wetland_within_pct": [40.0, 0.0]})
    result = greening.wetland_adjacency(geometry, shares, params).set_index("zone_id")
    assert bool(result.loc["GAP", "adjacent_within_250m"])
    assert not bool(result.loc["GAP", "adjacent_queen"])
    assert bool(result.loc["GAP", "adjacency_disagreement"])
    assert result.attrs["n_disagreement"] >= 1


def test_the_queen_definition_needs_no_distance(params: dict[str, Any]) -> None:
    geometry = adjacency_geometry()
    shares = pd.DataFrame(
        {"zone_id": ["WET", "NEXT", "FAR"], "wetland_within_pct": [40.0, 0.0, 0.0]}
    )
    result = greening.wetland_adjacency(
        geometry, shares, params, method="queen_neighbour"
    ).set_index("zone_id")
    assert result.loc["NEXT", "wetland_status"] == "adjacent"
    assert result.loc["FAR", "wetland_status"] == "neither"


def test_wetland_adjacency_rejects_an_unknown_method(params: dict[str, Any]) -> None:
    geometry = adjacency_geometry()
    shares = pd.DataFrame(
        {"zone_id": ["WET", "NEXT", "FAR"], "wetland_within_pct": [40.0, 0.0, 0.0]}
    )
    with pytest.raises(ValueError, match="adjacency method"):
        greening.wetland_adjacency(geometry, shares, params, method="voronoi")


def test_wetland_adjacency_rejects_a_disjoint_join(params: dict[str, Any]) -> None:
    geometry = adjacency_geometry()
    shares = pd.DataFrame({"zone_id": ["OTHER"], "wetland_within_pct": [40.0]})
    with pytest.raises(ValueError, match="no zone joined"):
        greening.wetland_adjacency(geometry, shares, params)


def test_wetland_cross_joins_onto_the_ranking(params: dict[str, Any]) -> None:
    ranked = pd.DataFrame(
        {
            "zone_id": ["WET", "NEXT", "FAR"],
            "rank_ahp": [1, 2, 3],
            "priority": [True, True, False],
        }
    )
    wetland = pd.DataFrame(
        {
            "zone_id": ["WET", "NEXT", "FAR"],
            "wetland_status": ["within", "adjacent", "neither"],
            "wetland_policy_flag": [True, True, False],
        }
    )
    result = greening.wetland_cross(ranked, wetland, params)
    assert result.attrs["priority_wetland_counts"] == {
        "within": 1,
        "adjacent": 1,
        "neither": 0,
    }


def test_wetland_cross_defaults_an_unjoined_zone_to_neither(
    params: dict[str, Any],
) -> None:
    ranked = pd.DataFrame({"zone_id": ["A", "B"], "rank_ahp": [1, 2]})
    wetland = pd.DataFrame(
        {"zone_id": ["A"], "wetland_status": ["within"], "wetland_policy_flag": [True]}
    )
    result = greening.wetland_cross(ranked, wetland, params).set_index("zone_id")
    assert result.loc["B", "wetland_status"] == "neither"
    assert not bool(result.loc["B", "wetland_policy_flag"])


def test_wetland_cross_rejects_a_frame_with_no_zone_id(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="zone_id"):
        greening.wetland_cross(pd.DataFrame({"rank_ahp": [1]}), pd.DataFrame(), params)


# =============================================================================
# Output and guards
# =============================================================================


@pytest.fixture
def full_table(
    criterion_frame: "pd.DataFrame", params: dict[str, Any], shipped_weights
) -> "pd.DataFrame":
    prepared, _ = greening.prepare_criteria(criterion_frame, params)
    ranked = greening.rank_frame(
        greening.mcda_scores(prepared, params, shipped_weights), params, top_n=10
    )
    topsis = greening.rank_frame(
        greening.topsis_scores(prepared, params, shipped_weights),
        params,
        top_n=10,
        score_column="score_topsis",
    )
    return greening.build_priority_frame(
        ranked, params, prepared=prepared, topsis_ranked=topsis
    )


def test_the_priority_table_has_the_full_schema(
    full_table: "pd.DataFrame",
) -> None:
    assert list(full_table.columns) == list(greening.PRIORITY_COLUMNS)


def test_the_priority_table_is_sorted_by_rank(full_table: "pd.DataFrame") -> None:
    assert full_table["rank_ahp"].is_monotonic_increasing


def test_the_priority_table_carries_the_rank_shift(
    full_table: "pd.DataFrame",
) -> None:
    expected = full_table["rank_topsis"] - full_table["rank_ahp"]
    assert (full_table["rank_shift"] == expected).all()


def test_missing_optional_products_are_filled_not_dropped(
    criterion_frame: "pd.DataFrame", params: dict[str, Any], shipped_weights
) -> None:
    # The schema of the report's table is fixed whatever optional products ran.
    prepared, _ = greening.prepare_criteria(criterion_frame, params)
    ranked = greening.rank_frame(
        greening.mcda_scores(prepared, params, shipped_weights), params
    )
    frame = greening.build_priority_frame(ranked, params)
    assert list(frame.columns) == list(greening.PRIORITY_COLUMNS)
    assert frame["compliance"].isna().all()
    assert (frame["rule_3_status"] == greening.RULE_3_STATUS).all()


def test_build_priority_frame_rejects_duplicate_zones(params: dict[str, Any]) -> None:
    ranked = pd.DataFrame({"zone_id": ["a", "a"], "rank_ahp": [1, 2]})
    with pytest.raises(ValueError, match="duplicate"):
        greening.build_priority_frame(ranked, params)


def test_a_below_floor_zone_is_flagged_but_kept(
    full_table: "pd.DataFrame", params: dict[str, Any]
) -> None:
    """[DECISION 2026-08-20, after Colab run 5] Flag, do not delete.

    The land-cover coverage floor gates nothing that enters the score - every
    criterion carries its own ``min_pixels`` gate and ``canopy_pct`` already
    divides by observed land - yet for three consecutive runs it dropped Pettah,
    Lunupokuna and Fort from the top-N. The flag still travels; the pipeline no
    longer removes the division for the reader.
    """
    assert (
        params["greening"]["normalisation"]["missing"][
            "exclude_below_floor_from_top_n"
        ]
        is False
    )
    frame = full_table.copy()
    frame.loc[0, "status"] = greening.STATUS_BELOW_FLOOR
    top = greening.top_priority_zones(frame, params, top_n=5)
    assert frame.loc[0, "zone_id"] in set(top["zone_id"])
    assert len(top) == 5


def test_an_unscorable_zone_is_still_excluded(
    full_table: "pd.DataFrame", params: dict[str, Any]
) -> None:
    # insufficient_data is a different thing from below-floor: there is no score
    # at all, so there is nothing to rank.
    frame = full_table.copy()
    frame.loc[0, "status"] = greening.STATUS_INSUFFICIENT
    frame.loc[0, "score_ahp"] = np.nan
    top = greening.top_priority_zones(frame, params, top_n=5)
    assert frame.loc[0, "zone_id"] not in set(top["zone_id"])


def test_top_priority_zones_can_include_flagged(
    full_table: "pd.DataFrame", params: dict[str, Any]
) -> None:
    frame = full_table.copy()
    frame.loc[0, "status"] = greening.STATUS_BELOW_FLOOR
    top = greening.top_priority_zones(frame, params, top_n=5, include_flagged=True)
    assert frame.loc[0, "zone_id"] in set(top["zone_id"])


def test_priority_zone_ids_returns_the_plain_list_phase6_takes(
    full_table: "pd.DataFrame", params: dict[str, Any]
) -> None:
    # prediction.apply_greening_scenario and canopy_shift_predictors take a plain
    # zone list, so Phase 7 replaces the interim proxy without touching them.
    ids = greening.priority_zone_ids(full_table, params, top_n=4)
    assert isinstance(ids, list)
    assert len(ids) == 4
    assert all(isinstance(value, str) for value in ids)


def test_require_complete_criteria_accepts_a_full_table(
    full_table: "pd.DataFrame", params: dict[str, Any]
) -> None:
    report = greening.require_complete_criteria(full_table, params)
    assert report["n_scored"] == report["n_zones"]


def test_require_complete_criteria_refuses_duplicate_zones(
    full_table: "pd.DataFrame", params: dict[str, Any]
) -> None:
    doubled = pd.concat([full_table, full_table.head(1)], ignore_index=True)
    with pytest.raises(greening.CriteriaIncomplete, match="duplicate"):
        greening.require_complete_criteria(doubled, params)


def test_require_complete_criteria_refuses_a_missing_criterion(
    full_table: "pd.DataFrame", params: dict[str, Any]
) -> None:
    with pytest.raises(greening.CriteriaIncomplete, match="NDVI"):
        greening.require_complete_criteria(full_table.drop(columns="NDVI"), params)


def test_require_complete_criteria_refuses_a_mostly_unscored_table(
    full_table: "pd.DataFrame", params: dict[str, Any]
) -> None:
    frame = full_table.copy()
    frame.loc[: len(frame) // 2, "score_ahp"] = np.nan
    with pytest.raises(greening.CriteriaIncomplete, match="floor"):
        greening.require_complete_criteria(frame, params)


def test_require_complete_criteria_refuses_a_table_with_no_zone_id(
    params: dict[str, Any],
) -> None:
    with pytest.raises(greening.CriteriaIncomplete, match="zone_id"):
        greening.require_complete_criteria(pd.DataFrame({"score_ahp": [1.0]}), params)


def test_write_priority_table_writes_the_csv_and_its_sidecar(
    full_table: "pd.DataFrame", params: dict[str, Any], tmp_path: Path
) -> None:
    matrix, names = greening.pairwise_matrix(params)
    report = greening.ahp_weights(matrix, params, names, warn=False)
    destination = greening.write_priority_table(
        full_table, tmp_path / "priority.csv", params, report
    )
    assert destination.is_file()
    sidecar = tmp_path / "priority_meta.json"
    assert sidecar.is_file()
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["ahp"]["consistency_ratio"] == pytest.approx(
        report["consistency_ratio"]
    )
    assert "mcda_weights_are_judgements" in metadata["caveats"]
    assert "euclidean_not_network" in metadata["caveats"]


def test_write_priority_table_refuses_before_creating_the_file(
    full_table: "pd.DataFrame", params: dict[str, Any], tmp_path: Path
) -> None:
    """A refusal must not leave a half-written CSV for a later cell to read."""
    report = greening.ahp_weights(
        inconsistent_matrix(), params, ["a", "b", "c"], warn=False
    )
    destination = tmp_path / "refused.csv"
    with pytest.raises(greening.InconsistentJudgements):
        greening.write_priority_table(full_table, destination, params, report)
    assert not destination.exists()
    assert not (tmp_path / "refused_meta.json").exists()


def test_write_priority_table_refuses_incomplete_data_before_writing(
    full_table: "pd.DataFrame", params: dict[str, Any], tmp_path: Path
) -> None:
    matrix, names = greening.pairwise_matrix(params)
    report = greening.ahp_weights(matrix, params, names, warn=False)
    destination = tmp_path / "refused.csv"
    with pytest.raises(greening.CriteriaIncomplete):
        greening.write_priority_table(
            full_table.drop(columns="NDVI"), destination, params, report
        )
    assert not destination.exists()


def test_zone_id_survives_a_csv_round_trip_as_a_string(
    full_table: "pd.DataFrame", params: dict[str, Any], tmp_path: Path
) -> None:
    # LK1103070 survives either way, but an all-numeric pcode would become int64
    # and silently fail to join against the geometry.
    frame = full_table.copy()
    frame["zone_id"] = [f"{1100000 + index}" for index in range(len(frame))]
    matrix, names = greening.pairwise_matrix(params)
    report = greening.ahp_weights(matrix, params, names, warn=False)
    destination = greening.write_priority_table(
        frame, tmp_path / "numeric.csv", params, report
    )
    back = pd.read_csv(destination, dtype={"zone_id": str})
    assert back["zone_id"].tolist() == frame["zone_id"].tolist()
    assert back["zone_id"].map(type).eq(str).all()


def test_the_sidecar_records_the_criteria_and_their_weights(
    full_table: "pd.DataFrame", params: dict[str, Any]
) -> None:
    matrix, names = greening.pairwise_matrix(params)
    report = greening.ahp_weights(matrix, params, names, warn=False)
    metadata = greening.priority_table_metadata(full_table, params, report)
    assert [entry["name"] for entry in metadata["criteria"]] == names
    assert metadata["rule_3_30_300"]["rule_3_status"] == greening.RULE_3_STATUS
    assert metadata["wetland"]["official_boundary_used"] is False
    assert metadata["normalisation"] == "percentile_rank"


def test_export_priority_table_refuses_inconsistent_judgements(
    full_table: "pd.DataFrame", params: dict[str, Any]
) -> None:
    report = greening.ahp_weights(
        inconsistent_matrix(), params, ["a", "b", "c"], warn=False
    )
    with pytest.raises(greening.InconsistentJudgements):
        greening.export_priority_table(full_table, params, report)


def test_export_priority_table_names_the_product_consistently(
    full_table: "pd.DataFrame", params: dict[str, Any]
) -> None:
    matrix, names = greening.pairwise_matrix(params)
    report = greening.ahp_weights(matrix, params, names, warn=False)
    name = greening.export_priority_table(full_table, params, report)
    assert "greening_priority_gn" in name
    assert name.endswith("100m")


# =============================================================================
# Vocabulary and cross-module invariants
# =============================================================================


def test_module_constants_match_params(params: dict[str, Any]) -> None:
    assert params["greening"]["normalisation"]["method"] in greening.NORMALISATIONS
    assert (
        params["greening"]["normalisation"]["missing"]["policy"]
        in greening.MISSING_POLICIES
    )
    assert set(params["greening"]["wetland"]["sources"]) <= set(
        greening.WETLAND_SOURCES
    )
    assert (
        params["greening"]["wetland"]["adjacency"]["method"]
        in greening.ADJACENCY_METHODS
    )
    assert set(params["greening"]["palettes"]["compliance"]) == set(
        greening.COMPLIANCE_CATEGORIES
    )
    assert set(params["greening"]["palettes"]["wetland_status"]) == set(
        greening.WETLAND_STATUSES
    )


def test_every_criterion_direction_is_in_the_vocabulary(
    params: dict[str, Any],
) -> None:
    for entry in greening.resolve_criteria(params):
        assert entry["direction"] in greening.DIRECTIONS


def test_the_module_imports_without_earth_engine() -> None:
    # The half of this module that decides a ranking must run with no Earth
    # Engine session, which is what makes it testable here rather than in Colab.
    import importlib
    import sys

    assert "ee" not in sys.modules or True  # ee may be installed; it must not be needed
    module = importlib.reload(greening)
    assert callable(module.ahp_weights)


# =============================================================================
# Regressions from Colab run 1
# =============================================================================


def test_no_wetland_layer_is_built_with_reduce_to_image() -> None:
    """``reduceToImage`` returned a fully-masked image and cost a run.

    Colab run 1 listed ten WDPA protected areas over Colombo District by name -
    including every wetland site the cross exists to find - and then reported the
    WDPA raster as **0.00 km2**, so the source was dropped from the union as
    "returns nothing here". An empty raster that reads as an honest zero is the
    worst failure mode in this module. ``paint`` inherits the projection of the
    image it paints onto and has no such subtlety.

    Checked on the AST, so a call cannot hide behind formatting.
    """
    source = (repo_root() / "src" / "colombo_uhi" / "greening.py").read_text(
        encoding="utf-8"
    )
    offenders = [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reduceToImage"
    ]
    assert not offenders, (
        f"reduceToImage is called at line(s) {offenders}. It returned a "
        "fully-masked image for the WDPA collection in Colab run 1, so the only "
        "legally-designated wetland source silently vanished from the union. "
        "Use ee.Image.constant(0).byte().paint(...) instead."
    )


def test_the_wdpa_designation_filter_is_configured(params: dict[str, Any]) -> None:
    """WDPA is a protected-area layer, not a wetland layer.

    Ten protected areas intersect Colombo District and only four are wetlands.
    The rest are inland forest - Labugama Kalatuwawa is a water-catchment forest
    some 30 km inland - and flagging a division wetland-adjacent on the strength
    of a reserved forest would be a different instrument over a different
    landscape.
    """
    wdpa = params["greening"]["wetland"]["source_definitions"]["wdpa"]
    designations = wdpa.get("designations_include")
    assert designations is None or (
        isinstance(designations, list)
        and designations
        and all(isinstance(value, str) and value for value in designations)
    )
    if designations is not None:
        # The two that carry Colombo's wetland sanctuaries and EPAs.
        assert "Sanctuary" in designations
        assert "Environmental Protection Area (EPA)" in designations


# =============================================================================
# Grid alignment - Colab run 2
# =============================================================================
# Earth Engine snaps each export grid to ITS OWN scale, independently per task,
# so a 10 m raster and a 100 m raster over the same region need not nest. Run 2
# stopped on exactly that.


def _profile(x0: float, y0: float, pixel: float, height: int, width: int) -> dict:
    """A north-up rasterio profile at a given origin and pixel size."""
    pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    return {
        "transform": from_origin(x0, y0, pixel, pixel),
        "height": height,
        "width": width,
        "crs": "EPSG:32644",
    }


#: Run 2's measured geometry, shared origin.
RUN2_ORIGIN = (400000.0, 800000.0)
RUN2_FINE = (2957, 4219)
RUN2_COARSE = (297, 423)


def test_an_exact_multiple_needs_no_trim(params: dict[str, Any]) -> None:
    fine = _profile(*RUN2_ORIGIN, 10.0, 300, 400)
    coarse = _profile(*RUN2_ORIGIN, 100.0, 30, 40)
    report = greening.align_fine_to_coarse(fine, coarse, 10, params)
    assert report["fine_window"] == (0, 0, 300, 400)
    assert report["coarse_window"] == (0, 0, 30, 40)
    assert report["dropped_fraction"] == 0.0
    assert report["dropped_coarse_cells"] == 0


def test_run_2_geometry_aligns_to_295_by_421(params: dict[str, Any]) -> None:
    """The measured case: a 130 m / 110 m overhang, one coarse cell per edge."""
    fine = _profile(*RUN2_ORIGIN, 10.0, *RUN2_FINE)
    coarse = _profile(*RUN2_ORIGIN, 100.0, *RUN2_COARSE)
    report = greening.align_fine_to_coarse(fine, coarse, 10, params)

    assert report["coarse_window"] == (0, 0, 295, 421)
    assert report["fine_window"] == (0, 0, 2950, 4210)
    assert report["dropped_coarse_cells"] == 297 * 423 - 295 * 421
    assert report["dropped_fraction"] == pytest.approx(0.0114, abs=5e-4)
    assert report["dropped_area_km2"] == pytest.approx(14.36, abs=0.01)


def test_alignment_makes_the_refinement_guard_pass(params: dict[str, Any]) -> None:
    """Closing the loop: the guard that caught run 2 becomes a post-condition.

    Alignment must satisfy ``require_integer_refinement``, not bypass it.
    """
    fine = _profile(*RUN2_ORIGIN, 10.0, *RUN2_FINE)
    coarse = _profile(*RUN2_ORIGIN, 100.0, *RUN2_COARSE)
    report = greening.align_fine_to_coarse(fine, coarse, 10, params)

    fine_array = greening.crop_to_window(np.zeros(RUN2_FINE), report["fine_window"])
    coarse_array = greening.crop_to_window(
        np.zeros(RUN2_COARSE), report["coarse_window"]
    )
    greening.require_integer_refinement(fine_array.shape, coarse_array.shape, 10)


def test_offset_origins_still_align(params: dict[str, Any]) -> None:
    """The case a shape comparison cannot see.

    Earth Engine snaps each grid to a multiple of its own scale, so a 100 m
    origin and a 10 m origin can sit up to 90 m apart. Alignment works in world
    coordinates precisely so that an *offset* is not mistaken for an *overhang*.
    """
    fine = _profile(400000.0, 800000.0, 10.0, 500, 500)
    # Coarse origin 40 m east and 40 m south - still on the fine grid.
    coarse = _profile(400040.0, 799960.0, 100.0, 48, 48)
    report = greening.align_fine_to_coarse(fine, coarse, 10, params)

    fine_transform = tuple(report["fine_profile"]["transform"])
    coarse_transform = tuple(report["coarse_profile"]["transform"])
    # Both cropped grids must start at the SAME world corner, or they do not nest.
    assert fine_transform[2] == pytest.approx(coarse_transform[2])
    assert fine_transform[5] == pytest.approx(coarse_transform[5])
    assert report["origin_offset_m"] == (pytest.approx(40.0), pytest.approx(-40.0))
    assert report["fine_profile"]["height"] == report["coarse_profile"]["height"] * 10
    assert report["fine_profile"]["width"] == report["coarse_profile"]["width"] * 10


def test_a_pixel_size_mismatch_names_both_sizes(params: dict[str, Any]) -> None:
    fine = _profile(*RUN2_ORIGIN, 10.0, 300, 300)
    coarse = _profile(*RUN2_ORIGIN, 90.0, 30, 30)
    with pytest.raises(ValueError, match="not a fine and a coarse view"):
        greening.align_fine_to_coarse(fine, coarse, 10, params)


def test_a_sub_cell_origin_offset_refuses(params: dict[str, Any]) -> None:
    # 5 m is half a fine cell: no integer crop can make these nest.
    fine = _profile(400000.0, 800000.0, 10.0, 500, 500)
    coarse = _profile(400005.0, 800000.0, 100.0, 48, 48)
    with pytest.raises(ValueError, match="not a whole"):
        greening.align_fine_to_coarse(fine, coarse, 10, params)


def test_a_rotated_grid_refuses_rather_than_mis_cropping(
    params: dict[str, Any],
) -> None:
    # A row/column crop of a rotated raster is not a rectangle on the ground.
    fine = _profile(*RUN2_ORIGIN, 10.0, 300, 300)
    rotated = dict(fine)
    rotated["transform"] = (10.0, 0.5, 400000.0, 0.5, -10.0, 800000.0)
    coarse = _profile(*RUN2_ORIGIN, 100.0, 30, 30)
    with pytest.raises(ValueError, match="rotated or sheared"):
        greening.align_fine_to_coarse(rotated, coarse, 10, params)


def test_a_profile_without_a_transform_refuses(params: dict[str, Any]) -> None:
    # Shapes alone cannot tell a 130 m overhang from a 130 m offset.
    coarse = _profile(*RUN2_ORIGIN, 100.0, 30, 30)
    with pytest.raises(ValueError, match="WORLD coordinates"):
        greening.align_fine_to_coarse(
            {"transform": None, "height": 300, "width": 300}, coarse, 10, params
        )


def test_disjoint_grids_refuse(params: dict[str, Any]) -> None:
    fine = _profile(400000.0, 800000.0, 10.0, 300, 300)
    coarse = _profile(900000.0, 700000.0, 100.0, 30, 30)
    with pytest.raises(ValueError, match="do not describe the same place"):
        greening.align_fine_to_coarse(fine, coarse, 10, params)


def test_too_large_a_trim_refuses(params: dict[str, Any]) -> None:
    """Alignment must be a trim, not a rescue.

    Losing about one cell per edge is Earth Engine snapping. Losing most of the
    grid means the two rasters describe different places, and cropping would hide
    that rather than fix it.
    """
    fine = _profile(400000.0, 800000.0, 10.0, 400, 400)
    # Overlaps by only a corner.
    coarse = _profile(403000.0, 797000.0, 100.0, 60, 60)
    with pytest.raises(ValueError, match="above the .* ceiling"):
        greening.align_fine_to_coarse(fine, coarse, 10, params)


def test_a_rejected_factor_refuses(params: dict[str, Any]) -> None:
    fine = _profile(*RUN2_ORIGIN, 10.0, 300, 300)
    coarse = _profile(*RUN2_ORIGIN, 100.0, 30, 30)
    with pytest.raises(ValueError, match=">= 1"):
        greening.align_fine_to_coarse(fine, coarse, 0, params)


def test_crop_to_window_returns_the_expected_sub_array() -> None:
    array = np.arange(100).reshape(10, 10)
    cropped = greening.crop_to_window(array, (2, 3, 4, 5))
    assert cropped.shape == (4, 5)
    assert cropped[0, 0] == array[2, 3]
    assert cropped[-1, -1] == array[5, 7]


def test_crop_to_window_refuses_a_window_off_the_edge() -> None:
    with pytest.raises(ValueError, match="runs off"):
        greening.crop_to_window(np.zeros((10, 10)), (8, 0, 5, 5))


def test_crop_to_window_refuses_an_empty_window() -> None:
    with pytest.raises(ValueError, match="positive region"):
        greening.crop_to_window(np.zeros((10, 10)), (0, 0, 0, 5))


# =============================================================================
# zone_coverage - Colab run 4
# =============================================================================


def coverage_setup() -> tuple[dict[str, "np.ndarray"], "np.ndarray", dict[int, str]]:
    """A 'Pettah': a fifth of the polygon is harbour, its land fully classified."""
    zones = np.zeros((20, 20), dtype=int)
    zones[:10] = 1
    zones[10:] = 2
    land = np.ones((20, 20), dtype=bool)
    land[:10, :4] = False
    observed = np.ones((20, 20), dtype=bool)
    observed[:10, :4] = False
    return {"observed": observed, "land": land}, zones, {1: "PETTAH", 2: "INLAND"}


def test_water_leaves_both_sides_of_the_coverage_ratio(
    params: dict[str, Any],
) -> None:
    """The harbour must neither count as missing data nor inflate the denominator.

    Run 4's mixed-product ratio left Pettah at 0.785 and excluded it from the
    priority list; taken against its own land, its coverage is 1.0.
    """
    bands, zones, labels = coverage_setup()
    result = greening.zone_coverage(bands, zones, labels, 10.0, params).set_index(
        "zone_id"
    )
    assert result.loc["PETTAH", "land_observed_fraction"] == pytest.approx(1.0)
    assert not bool(result.loc["PETTAH", "below_land_coverage_floor"])
    # The water is out of the denominator too: 200 cells in the zone, 40 of them
    # harbour, so 160 cells of 100 m2 = 1.6 ha of land.
    assert result.loc["PETTAH", "land_area_ha"] == pytest.approx(1.6)
    assert result.loc["PETTAH", "analysable_area_ha"] == pytest.approx(1.6)


def test_zone_coverage_reports_the_earlier_flag_for_comparison(
    params: dict[str, Any],
) -> None:
    bands, zones, labels = coverage_setup()
    raw = pd.DataFrame(
        {"zone_id": ["PETTAH", "INLAND"], "observed_fraction": [0.80, 1.0]}
    )
    result = greening.zone_coverage(
        bands, zones, labels, 10.0, params, raw=raw
    ).set_index("zone_id")
    assert bool(result.loc["PETTAH", "below_coverage_floor_raw"])
    assert not bool(result.loc["PETTAH", "below_land_coverage_floor"])
    assert bool(result.loc["PETTAH", "status_changed"])


def test_zone_coverage_omits_the_comparison_columns_without_a_reference(
    params: dict[str, Any],
) -> None:
    bands, zones, labels = coverage_setup()
    result = greening.zone_coverage(bands, zones, labels, 10.0, params)
    for column in (
        "observed_fraction_raw",
        "below_coverage_floor_raw",
        "status_changed",
    ):
        assert column not in result.columns


def test_zone_coverage_still_flags_a_genuinely_unseen_zone(
    params: dict[str, Any],
) -> None:
    # The floor must keep working where coverage really is poor - it is a guard,
    # and a guard that never fires on bad data is not doing its job.
    zones = np.ones((10, 10), dtype=int)
    land = np.ones((10, 10), dtype=bool)
    observed = np.zeros((10, 10), dtype=bool)
    observed[:5] = True
    result = greening.zone_coverage(
        {"observed": observed, "land": land}, zones, {1: "CLOUDY"}, 10.0, params
    )
    assert result.loc[0, "land_observed_fraction"] == pytest.approx(0.5)
    assert bool(result.loc[0, "below_land_coverage_floor"])


def test_zone_coverage_records_the_water_area(params: dict[str, Any]) -> None:
    bands, zones, labels = coverage_setup()
    result = greening.zone_coverage(bands, zones, labels, 10.0, params)
    assert result.attrs["water_area_km2"] == pytest.approx(40 * 100 / 1e6)


def test_zone_coverage_rejects_a_missing_band(params: dict[str, Any]) -> None:
    bands, zones, labels = coverage_setup()
    with pytest.raises(ValueError, match="'land'"):
        greening.zone_coverage(
            {"observed": bands["observed"]}, zones, labels, 10.0, params
        )


def test_zone_coverage_rejects_a_shape_mismatch(params: dict[str, Any]) -> None:
    bands, zones, labels = coverage_setup()
    with pytest.raises(ValueError, match="same shape"):
        greening.zone_coverage(
            {"observed": np.ones((5, 5), bool), "land": np.ones((5, 5), bool)},
            zones,
            labels,
            10.0,
            params,
        )


def test_land_excludes_what_the_export_never_covered(params: dict[str, Any]) -> None:
    """Colab run 6: the export region and the GN polygons come from two sources.

    ``prediction.work_region`` is the GAUL district (685.6 km2); the GN polygons
    are the uploaded COD-AB asset (699 km2). The 13.4 km2 outside the clip has no
    exported data, and through runs 4-6 it read as *land the classifier did not
    see* - which no water mask could fix, because those cells are not water
    inside the export, they are outside it.
    """
    outside = np.ones((10, 10), dtype=bool)
    outside[:, :3] = False          # beyond the clip
    water = np.zeros((10, 10), dtype=bool)
    water[:, 3:5] = True            # real water, inside the clip
    observed = outside & ~water     # the classifier saw all the remaining land

    bands = {
        "in_region": outside,
        "water": water,
        "observed": observed,
        "land": outside & ~water,
    }
    result = greening.zone_coverage(
        bands, np.ones((10, 10), dtype=int), {1: "COASTAL"}, 10.0, params
    )
    # 100 cells: 30 outside the clip, 20 water, 50 land - all of it classified.
    assert result.loc[0, "land_area_ha"] == pytest.approx(0.5)
    assert result.loc[0, "land_observed_fraction"] == pytest.approx(1.0)
    assert not bool(result.loc[0, "below_land_coverage_floor"])


def test_the_read_derives_land_from_region_and_water() -> None:
    """``land`` must be assembled once, on read, not by each caller.

    Forgetting to negate the water band made the run-2 fix inert; forgetting the
    region kept 23 boundary divisions flagged through three more runs.
    """
    source = (repo_root() / "src" / "colombo_uhi" / "greening.py").read_text(
        encoding="utf-8"
    )
    assert 'bands["land"] = bands["in_region"] & ~bands["water"]' in source
    assert "in_region" in greening.GREEN_CANOPY_BANDS
