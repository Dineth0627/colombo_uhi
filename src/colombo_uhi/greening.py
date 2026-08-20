"""Urban greening priority: MCDA / AHP weighted overlay, TOPSIS, and 3-30-300.

Deliverable (3) of the practicum. This module ranks Colombo's GN divisions for
greening investment on five observed criteria - surface heat, the share of the
zone in severe UTFVI classes, vegetation deficit, population density and
green-space access deficit - weighted by an Analytic Hierarchy Process, and
cross-checked against an independent TOPSIS ranking.

**The weights are judgements, not measurements.** The consistency ratio tests
only whether a set of pairwise judgements is *self-consistent*; it says nothing
about whether they are *right*. :func:`ahp_weights` reports the ratio,
:func:`require_consistent` refuses to let a product ship above the threshold, and
``caveats.mcda_weights_are_judgements`` travels on every figure.

**Nothing here is a prediction.** Every criterion is an observed 2020s quantity.
Phase 6's land-cover projection never beat a no-change map, so a criterion drawn
from a projected surface would inherit a model with no demonstrated allocation
skill. ``tests/test_notebook07.py`` enforces that on the notebook's AST rather
than trusting the convention.

Six things in here are easy to get wrong, produce a confident wrong answer, and
are therefore pinned by unit tests and stated in the relevant docstrings:

1. **The criteria are near-collinear over Colombo.** Measured on the committed
   Phase 5 outputs across all 557 GN divisions: ``rho(LST, green fraction) =
   -0.9147``, ``rho(LST, Gi* z) = +0.9576``. WorldPop is itself modelled partly
   from built-up area, so ``pop_density`` is a third correlate. A five-criterion
   MCDA over five near-identical variables reproduces a ranking by LST while
   wearing the authority of a multi-criteria method.
   :func:`criterion_correlation`, :func:`effective_dimensionality` and above all
   :func:`criterion_ablation` exist to *measure* that, and the notebook prints
   the answer whether or not it flatters the method.
2. **Zone-mean UTFVI is LST wearing a hat.** ``UTFVI = (Ts - Tmean)/Tmean`` with
   a scalar ``Tmean``, so a zone mean is an affine transform of a zone-mean LST:
   ``rho = +1.000000`` exactly. The criterion here is therefore the *severe-class
   share* (:func:`utfvi_shares_by_zone`), a within-zone distributional property
   the mean cannot reproduce.
3. **The Phase 5 coverage floor fires on water, not on cloud.** The identical
   ``observed_fraction`` appears for three classifiers across three dates in 552
   of 557 zones, because it measures the polygon enclosing water. Fort's COD-AB
   polygon *is* the Colombo Port outer harbour. Excluding on the raw flag deletes
   Pettah and Lunupokuna - dense, hot, treeless CMC-core divisions - from the
   priority list. :func:`land_observed_fraction` recomputes it against land.
4. **Missing data must not sink a division.** A zone with no NDVI because it sat
   under cloud is not thereby low-priority. :func:`prepare_criteria`
   redistributes the absent weight and flags the zone; it never uses
   ``na_option="bottom"``, and a test asserts that string appears nowhere here.
5. **TOPSIS reverses ranks when the alternative set changes.** The ideal and
   anti-ideal are drawn from the set, so :func:`topsis_scores` is re-run on the
   retained zones rather than sub-selected, and :func:`compare_rankings` refuses
   mismatched zone sets.
6. **A one-band GeoTIFF loses the coverage mask.** Phase 5's 5.4x "green growth"
   artefact and Phase 6's nodata-reads-as-water bug have the same cause.
   :func:`read_green_canopy_raster` raises on a band count other than three and
   :func:`read_population_raster` on other than two.

Products:
    * :func:`pairwise_matrix` / :func:`principal_eigenvector` /
      :func:`consistency_ratio` / :func:`ahp_weights` - the AHP, with the
      warn-at-computation / refuse-at-product split;
    * :func:`prepare_criteria` / :func:`mcda_scores` / :func:`rank_frame` - the
      weighted overlay;
    * :func:`topsis_scores` / :func:`compare_rankings` - the alternative ranking
      and the robustness check;
    * :func:`criterion_ablation` / :func:`circularity_report` - what the method
      actually adds, and what it merely repeats;
    * :func:`qualifying_green_mask` / :func:`service_area_mask` /
      :func:`compliance_3_30_300` - the 3-30-300 rule;
    * :func:`wetland_image` / :func:`wetland_cross` - the Ramsar Wetland City
      policy lever;
    * :func:`write_priority_table` - the guarded writer.

Earth Engine, numpy, pandas, scipy, rasterio and geopandas are imported *inside*
the functions that need them. The pure-Python half of this module - which is
everything that decides a ranking - must import and run with no Earth Engine
session and no geospatial stack at all, because that is what makes it testable
here rather than only in Colab.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import ee
    import geopandas as gpd
    import numpy as np
    import pandas as pd

# --- Vocabulary --------------------------------------------------------------

#: Normalisation methods :func:`normalise_criterion` accepts.
NORMALISATIONS: tuple[str, ...] = ("percentile_rank", "min_max", "z_score")

#: Criterion directions. ``benefit`` = higher is higher priority; ``cost`` = lower is.
DIRECTIONS: tuple[str, ...] = ("benefit", "cost")

#: Missing-data policies :func:`prepare_criteria` accepts.
MISSING_POLICIES: tuple[str, ...] = ("redistribute_and_flag", "insufficient")

#: What the "3" of 3-30-300 is reported as. It cannot be measured from space.
RULE_3_STATUS: str = "not_remotely_sensable"

#: Wetland layers :func:`wetland_source_image` knows how to build.
WETLAND_SOURCES: tuple[str, ...] = (
    "dw_flooded_vegetation",
    "worldcover_wetland",
    "gsw_seasonal",
    "wdpa",
    "asset",
)

#: Wetland adjacency definitions. Both are reported; neither is authoritative.
ADJACENCY_METHODS: tuple[str, ...] = ("buffer", "queen_neighbour")

#: Per-zone status values.
STATUS_OK: str = "ok"
STATUS_INSUFFICIENT: str = "insufficient_data"
STATUS_BELOW_FLOOR: str = "below_land_coverage_floor"

#: What :func:`circularity_report` returns for ``independence``. There is no
#: value of this field that means "independent" - the overlap is structural.
NOT_INDEPENDENT: str = "not_independent"

#: The five 3-30-300 compliance categories. Never a boolean: the "3" is
#: unmeasured, and a pass/fail flag would imply it had been checked.
COMPLIANCE_CATEGORIES: tuple[str, ...] = (
    "both_30_and_300",
    "canopy_only",
    "access_only",
    "neither",
    "not_assessable",
)

#: Wetland relationship of a zone to the wetland layer.
WETLAND_STATUSES: tuple[str, ...] = ("within", "adjacent", "neither")

#: Band order :func:`green_canopy_image` produces, and the order the exported
#: GeoTIFF must be read back in.
#:
#: ``water`` is not decoration. Colab run 3 measured what its absence costs: the
#: land-coverage floor was computed against the whole rasterised polygon, so it
#: reproduced the Phase 5 fraction to 16 decimal places for 555 of 557 zones and
#: excluded Pettah and Lunupokuna from the priority list on the strength of the
#: harbour inside their polygons.
GREEN_CANOPY_BANDS: tuple[str, ...] = ("green", "canopy", "water", "observed")

#: Band order :func:`population_image` produces.
POPULATION_BANDS: tuple[str, ...] = ("population", "observed")

#: Column order of the frame :func:`build_ahp_frame` returns.
AHP_COLUMNS: tuple[str, ...] = (
    "criterion",
    "label",
    "direction",
    "weight",
    "weight_geometric",
    "weight_rank",
)

#: Column order of the frame :func:`compliance_3_30_300` returns.
COMPLIANCE_COLUMNS: tuple[str, ...] = (
    "zone_id",
    "canopy_pct",
    "rule_30_pass",
    "pop_within_300m_pct",
    "area_within_300m_pct",
    "rule_300_pass",
    "pop_within_300m_detour_pct",
    "rule_300_pass_detour",
    "rule_3_proxy_pct",
    "rule_3_status",
    "compliance",
)


class ConsistencyWarning(UserWarning):
    """Raised as a warning when AHP judgements exceed the consistency threshold.

    A warning rather than an exception, so that an analyst can *see* the weights
    their judgements imply and correct them. :func:`require_consistent` is what
    refuses at the point a product would be written.
    """


class InconsistentJudgements(ValueError):
    """Raised when a product is requested from judgements that cannot support it.

    Three distinct failures raise this, and the message says which:
    the consistency ratio exceeds ``greening.ahp.consistency_ratio_max``; the
    matrix is degenerate (every judgement equal, so ``CR = 0`` and nothing was
    actually decided); or the weight spread is below
    ``greening.ahp.min_weight_spread``.
    """


class CriteriaIncomplete(ValueError):
    """Raised when a frame cannot support a published ranking.

    Distinct from :class:`InconsistentJudgements`: the judgements may be fine
    and the *data* inadequate - a criterion column absent, duplicate zone ids, or
    too few zones scored.
    """


# =============================================================================
# Group A - resolvers
# =============================================================================


def resolve_level(level: str | None, params: dict[str, Any]) -> str:
    """Normalise an aggregation-level key, defaulting to ``greening.level``.

    Args:
        level: ``"gn"``, ``"ds"``, or ``None`` for the configured default.
        params: Parsed params mapping.

    Returns:
        The canonical level key.

    Raises:
        ValueError: If the level is not one of ``greening.levels``.
    """
    from colombo_uhi import spatial_stats

    resolved = spatial_stats.resolve_level(
        str(params["greening"]["level"] if level is None else level)
    )
    allowed = [str(value) for value in params["greening"]["levels"]]
    if resolved not in allowed:
        raise ValueError(
            f"level {resolved!r} is not in greening.levels {allowed}"
        )
    return resolved


def resolve_criteria(
    params: dict[str, Any], names: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Return the criterion definitions, optionally restricted and reordered.

    Args:
        params: Parsed params mapping.
        names: Optional subset of criterion names, in the order wanted.

    Returns:
        A list of criterion mappings, each carrying at least ``name``,
        ``direction``, ``column``, ``label`` and ``provenance``.

    Raises:
        ValueError: If a configured criterion is malformed, if names repeat, or
            if a requested name is not configured.
    """
    configured = params["greening"]["criteria"]
    if not configured:
        raise ValueError("greening.criteria is empty; there is nothing to rank on")

    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in configured:
        name = str(entry["name"])
        if name in seen:
            raise ValueError(f"greening.criteria lists {name!r} twice")
        seen.add(name)
        direction = str(entry["direction"])
        if direction not in DIRECTIONS:
            raise ValueError(
                f"criterion {name!r} has direction {direction!r}; expected one "
                f"of {list(DIRECTIONS)}"
            )
        resolved.append(dict(entry))

    if names is None:
        return resolved

    index = {str(entry["name"]): entry for entry in resolved}
    wanted: list[dict[str, Any]] = []
    for name in names:
        key = str(name)
        if key not in index:
            raise ValueError(
                f"criterion {key!r} is not configured; greening.criteria has "
                f"{sorted(index)}"
            )
        wanted.append(index[key])
    return wanted


def criterion_names(params: dict[str, Any]) -> list[str]:
    """Return the configured criterion names, in configuration order.

    Args:
        params: Parsed params mapping.

    Returns:
        The criterion names.
    """
    return [str(entry["name"]) for entry in resolve_criteria(params)]


def criterion_columns(
    params: dict[str, Any], names: Sequence[str] | None = None
) -> dict[str, str]:
    """Map criterion name to the frame column it is read from.

    Args:
        params: Parsed params mapping.
        names: Optional subset of criterion names.

    Returns:
        Mapping of criterion name to source column name.
    """
    return {
        str(entry["name"]): str(entry["column"])
        for entry in resolve_criteria(params, names)
    }


def resolve_normalisation(method: str | None, params: dict[str, Any]) -> str:
    """Normalise a normalisation-method key, defaulting to the configured one.

    Args:
        method: One of :data:`NORMALISATIONS`, or ``None`` for the default.
        params: Parsed params mapping.

    Returns:
        The canonical method name.

    Raises:
        ValueError: If the method is unknown.
    """
    resolved = str(
        params["greening"]["normalisation"]["method"] if method is None else method
    )
    if resolved not in NORMALISATIONS:
        raise ValueError(
            f"normalisation {resolved!r} is not one of {list(NORMALISATIONS)}"
        )
    return resolved


def resolve_landcover_year(params: dict[str, Any], year: int | None = None) -> int:
    """Resolve the land-cover year, defaulting to ``greening.landcover_year``.

    Args:
        params: Parsed params mapping.
        year: Optional explicit year.

    Returns:
        The year to build land-cover products for.

    Raises:
        ValueError: If the year predates the configured Dynamic World start,
            which is the year Phase 5 measured at 10.5 % coverage.
    """
    resolved = int(params["greening"]["landcover_year"] if year is None else year)
    availability = params["datasets"]["dynamic_world"]["availability"]
    start = int(str(availability[0])[:4])
    if resolved < start:
        raise ValueError(
            f"landcover year {resolved} predates Dynamic World, which opens "
            f"{availability[0]}. Phase 5 measured 2016 at 10.5 % coverage over "
            "this district; a greening recommendation cannot rest on a year the "
            "classifier barely saw."
        )
    return resolved


def resolve_wetland_sources(
    params: dict[str, Any], sources: Sequence[str] | None = None
) -> list[str]:
    """Resolve the wetland source list, dropping the asset hook when unset.

    Args:
        params: Parsed params mapping.
        sources: Optional explicit source list.

    Returns:
        The source keys to build, in order.

    Raises:
        ValueError: If a source is unknown, or if ``"asset"`` is requested
            explicitly while ``greening.wetland.asset`` is null. Requesting the
            official boundary and silently getting a proxy union instead would
            be the wrong kind of quiet.
    """
    config = params["greening"]["wetland"]
    explicit = sources is not None
    resolved = [
        str(value) for value in (config["sources"] if sources is None else sources)
    ]

    for source in resolved:
        if source not in WETLAND_SOURCES:
            raise ValueError(
                f"wetland source {source!r} is not one of {list(WETLAND_SOURCES)}"
            )

    if "asset" in resolved and not config.get("asset"):
        if explicit:
            raise ValueError(
                "wetland source 'asset' was requested but greening.wetland.asset "
                "is null. Upload the official Colombo Wetland Complex boundary "
                "(Colombo Wetland Management Strategy / the 2018 Ramsar Wetland "
                "City accreditation - SLLRDC, UDA, or the Metro Colombo Urban "
                "Development Project) as an Earth Engine table asset and set its "
                "id there. Without it the wetland cross runs on a union of "
                "remote-sensing proxies, which is the default and is reported "
                "as such."
            )
        resolved = [source for source in resolved if source != "asset"]

    if not resolved:
        raise ValueError("no usable wetland sources; greening.wetland.sources is empty")
    return resolved


# =============================================================================
# Group B - the Analytic Hierarchy Process
# =============================================================================


def random_index(n: int, params: dict[str, Any]) -> float:
    """Return Saaty's random consistency index for a matrix of order ``n``.

    The table is the mean consistency index of 500 randomly generated reciprocal
    matrices of each order (Saaty 1980), read from
    ``greening.ahp.random_index`` so no constant lives in ``src/``.

    Args:
        n: Matrix order.
        params: Parsed params mapping.

    Returns:
        The random index. ``0.0`` for ``n <= 2``, where a reciprocal matrix is
        consistent by construction.

    Raises:
        ValueError: If ``n`` is outside the published table. The values are
            measured constants, not a curve; extrapolating one would invent a
            threshold and present it as Saaty's.
    """
    table = params["greening"]["ahp"]["random_index"]
    order = int(n)
    if order not in table:
        raise ValueError(
            f"no published random index for n={order}. Saaty's table covers "
            f"{min(table)}-{max(table)}; the values are measured constants "
            "rather than a curve, so extrapolating one would invent a "
            "consistency threshold and attribute it to Saaty."
        )
    return float(table[order])


def validate_pairwise(
    matrix: "np.ndarray",
    names: Sequence[str] | None = None,
    tol: float = 1e-9,
) -> "np.ndarray":
    """Check that a matrix is a valid AHP pairwise-comparison matrix.

    Every failure names the offending pair by criterion name where names are
    supplied, because "element [1][3] is not reciprocal" is not something an
    analyst can act on.

    Args:
        matrix: Square array of pairwise judgements.
        names: Optional criterion names, used in error messages.
        tol: Absolute tolerance for the reciprocity and unit-diagonal checks.

    Returns:
        The matrix as a float64 array.

    Raises:
        ValueError: If the matrix is not 2-D and square, contains a non-finite
            or non-positive entry, has a diagonal element other than 1, is not
            reciprocal, or does not match the length of ``names``.
    """
    import numpy as np  # Deferred: see module docstring.

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ValueError(
            f"a pairwise matrix must be 2-D; got {array.ndim} dimension(s) with "
            f"shape {array.shape}"
        )
    if array.shape[0] != array.shape[1]:
        raise ValueError(f"a pairwise matrix must be square; got {array.shape}")

    size = array.shape[0]
    if size == 0:
        raise ValueError("a pairwise matrix must have at least one criterion")

    labels = [str(name) for name in names] if names is not None else None
    if labels is not None and len(labels) != size:
        raise ValueError(
            f"{len(labels)} name(s) for a {size}x{size} matrix: {labels}"
        )

    def _label(position: int) -> str:
        return labels[position] if labels is not None else f"[{position}]"

    if not np.isfinite(array).all():
        rows, cols = np.nonzero(~np.isfinite(array))
        raise ValueError(
            f"pairwise judgement {_label(int(rows[0]))} vs {_label(int(cols[0]))} "
            f"is not finite ({array[rows[0], cols[0]]})"
        )
    if (array <= 0).any():
        rows, cols = np.nonzero(array <= 0)
        raise ValueError(
            f"pairwise judgement {_label(int(rows[0]))} vs {_label(int(cols[0]))} "
            f"is {array[rows[0], cols[0]]}; judgements must be strictly positive "
            "(Saaty 1-9 and their reciprocals)"
        )

    diagonal = np.diag(array)
    if not np.allclose(diagonal, 1.0, rtol=0, atol=tol):
        bad = int(np.argmax(np.abs(diagonal - 1.0)))
        raise ValueError(
            f"the diagonal entry for {_label(bad)} is {diagonal[bad]}, not 1; a "
            "criterion is always exactly as important as itself"
        )

    product = array * array.T
    off_diagonal = ~np.eye(size, dtype=bool)
    if not np.allclose(product[off_diagonal], 1.0, rtol=0, atol=tol):
        offset = np.abs(product - 1.0)
        offset[~off_diagonal] = -1.0
        row, col = np.unravel_index(int(np.argmax(offset)), offset.shape)
        raise ValueError(
            f"pairwise judgements for {_label(int(row))} and {_label(int(col))} "
            f"are not reciprocal: a_ij = {array[row, col]}, a_ji = "
            f"{array[col, row]}, product {product[row, col]} != 1"
        )

    return array


def pairwise_matrix(
    params: dict[str, Any], criteria: Sequence[str] | None = None
) -> tuple["np.ndarray", list[str]]:
    """Build the AHP matrix from the named pairs in params.

    Judgements are configured as named pairs ``i__j`` rather than as a nested
    list, so that reordering ``greening.criteria`` cannot silently reattach a
    judgement to the wrong pair. Reciprocals are derived as exactly ``1/value``
    and are never configured, which makes the matrix reciprocal by construction
    rather than by rounding.

    Args:
        params: Parsed params mapping.
        criteria: Optional criterion subset/order. Defaults to all configured.

    Returns:
        ``(matrix, names)`` - the float64 matrix and the criterion names in the
        row/column order used.

    Raises:
        ValueError: If a key is malformed, names an unknown criterion, gives the
            same pair in both orders, duplicates a pair, or if any pair of the
            requested criteria has no judgement at all.
    """
    import numpy as np  # Deferred: see module docstring.

    names = [str(entry["name"]) for entry in resolve_criteria(params, criteria)]
    index = {name: position for position, name in enumerate(names)}
    size = len(names)
    matrix = np.ones((size, size), dtype=float)

    given: dict[frozenset[str], str] = {}
    for key, value in params["greening"]["ahp"]["pairwise"].items():
        text = str(key)
        parts = text.split("__")
        if len(parts) != 2:
            raise ValueError(
                f"pairwise key {text!r} is malformed; expected 'left__right' "
                "naming the MORE important criterion first"
            )
        left, right = parts
        if left == right:
            raise ValueError(f"pairwise key {text!r} compares a criterion with itself")
        if left not in index or right not in index:
            unknown = left if left not in index else right
            # A pair naming a criterion outside the requested subset is skipped
            # rather than fatal - `criteria=` exists for the ablation, which runs
            # the AHP on subsets of the configured list.
            if criteria is not None and unknown in criterion_names(params):
                continue
            raise ValueError(
                f"pairwise key {text!r} names {unknown!r}, which is not a "
                f"configured criterion; greening.criteria has "
                f"{sorted(criterion_names(params))}"
            )

        pair = frozenset((left, right))
        if pair in given:
            raise ValueError(
                f"pairwise judgements {given[pair]!r} and {text!r} describe the "
                "same pair. Reciprocals are implied and must never be "
                "configured; giving a pair in both orders lets the two disagree."
            )
        given[pair] = text

        judgement = float(value)
        if judgement <= 0:
            raise ValueError(f"pairwise judgement {text!r} is {judgement}; must be > 0")
        matrix[index[left], index[right]] = judgement
        matrix[index[right], index[left]] = 1.0 / judgement

    missing = [
        f"{a}__{b}"
        for position, a in enumerate(names)
        for b in names[position + 1 :]
        if frozenset((a, b)) not in given
    ]
    if missing:
        raise ValueError(
            f"greening.ahp.pairwise has no judgement for {missing}. Every pair "
            "needs one: an absent judgement would silently default to 'equally "
            "important', which is a decision nobody made."
        )

    return validate_pairwise(matrix, names), names


def principal_eigenvector(
    matrix: "np.ndarray",
    params: dict[str, Any] | None = None,
    max_iter: int | None = None,
    tol: float | None = None,
) -> tuple["np.ndarray", float]:
    """Principal eigenvector and lambda-max of a pairwise matrix, by power iteration.

    A positive reciprocal matrix is exactly the Perron-Frobenius case: a simple,
    real, strictly dominant eigenvalue with a strictly positive eigenvector. Power
    iteration converges to it unconditionally, so there is nothing to
    disambiguate. ``numpy.linalg.eig`` returns complex values in arbitrary order
    and needs an ``argmax``, an ``abs`` and a sign fix to select from - three
    places where a silently wrong eigenvector can be chosen, none of which raise.
    ``eig`` is used as the independent cross-check in ``tests/test_greening.py``,
    which is its right place.

    Args:
        matrix: Validated pairwise matrix.
        params: Parsed params mapping, for the iteration settings.
        max_iter: Override for ``greening.ahp.power_iteration.max_iter``.
        tol: Override for ``greening.ahp.power_iteration.tol``.

    Returns:
        ``(weights, lambda_max)`` with weights positive and summing to 1.

    Raises:
        ValueError: If the matrix is not a valid pairwise matrix.
        RuntimeError: On non-convergence, which is impossible for a valid
            reciprocal matrix and therefore means the validation was bypassed.
    """
    import numpy as np  # Deferred: see module docstring.

    array = validate_pairwise(matrix)
    size = array.shape[0]

    settings: Mapping[str, Any] = {}
    if params is not None:
        settings = params["greening"]["ahp"]["power_iteration"]
    iterations = int(settings.get("max_iter", 1000) if max_iter is None else max_iter)
    epsilon = float(settings.get("tol", 1e-12) if tol is None else tol)

    if size == 1:
        return np.ones(1, dtype=float), 1.0

    vector = np.full(size, 1.0 / size, dtype=float)
    for _ in range(iterations):
        product = array @ vector
        total = float(product.sum())
        candidate = product / total
        if float(np.abs(candidate - vector).max()) < epsilon:
            vector = candidate
            break
        vector = candidate
    else:
        raise RuntimeError(
            f"power iteration did not converge in {iterations} steps. A positive "
            "reciprocal matrix always converges, so this means validate_pairwise "
            "was bypassed - check the matrix rather than raising max_iter."
        )

    # The AHP textbook form: lambda_max as the mean of the row ratios
    # (A @ w)_i / w_i, which equals the eigenvalue exactly at the eigenvector
    # and degrades gracefully to its average when iteration stopped early.
    lambda_max = float(np.mean((array @ vector) / vector))
    return vector, lambda_max


def geometric_mean_weights(matrix: "np.ndarray") -> "np.ndarray":
    """Saaty's row-geometric-mean approximation to the priority vector.

    Reported beside the eigenvector rather than instead of it. The two agreeing
    is itself evidence of consistency; the two diverging says the judgements are
    incoherent in a way the consistency ratio may not have caught.

    Args:
        matrix: Validated pairwise matrix.

    Returns:
        Positive weights summing to 1.

    Raises:
        ValueError: If the matrix is not a valid pairwise matrix.
    """
    import numpy as np  # Deferred: see module docstring.

    array = validate_pairwise(matrix)
    # In log space, so a 9x9 matrix of Saaty ratios cannot overflow the product.
    weights = np.exp(np.log(array).mean(axis=1))
    return weights / weights.sum()


def consistency_index(lambda_max: float, n: int) -> float:
    """Saaty's consistency index ``(lambda_max - n) / (n - 1)``.

    Args:
        lambda_max: Principal eigenvalue.
        n: Matrix order.

    Returns:
        The consistency index; ``0.0`` for ``n <= 2``, where a reciprocal matrix
        is consistent by construction. Clamped at zero: a lambda-max marginally
        below ``n`` is floating-point noise, not negative inconsistency.
    """
    size = int(n)
    if size <= 2:
        return 0.0
    return max(0.0, (float(lambda_max) - size) / (size - 1))


def consistency_ratio(lambda_max: float, n: int, params: dict[str, Any]) -> float:
    """Saaty's consistency ratio ``CI / RI``.

    Args:
        lambda_max: Principal eigenvalue.
        n: Matrix order.
        params: Parsed params mapping.

    Returns:
        The consistency ratio; ``0.0`` where the random index is zero.

    Raises:
        ValueError: If ``n`` is outside Saaty's published table.
    """
    index = consistency_index(lambda_max, n)
    reference = random_index(n, params)
    if reference <= 0:
        return 0.0
    return index / reference


def ahp_weights(
    matrix: "np.ndarray",
    params: dict[str, Any],
    names: Sequence[str] | None = None,
    warn: bool = True,
) -> dict[str, Any]:
    """Derive criterion weights from a pairwise matrix, and report on them.

    **This function never raises on inconsistency.** It computes, records
    ``consistent: False``, and warns. That split is the same one Colab run 3
    forced on Phase 6: ``require_validated`` turned a *measured* negative result
    into a traceback and destroyed every valid product beside it. An analyst
    whose judgements are inconsistent needs to *see* the weights in order to fix
    them. :func:`require_consistent` is what refuses, at the point a product
    would be written.

    Args:
        matrix: Pairwise comparison matrix.
        params: Parsed params mapping.
        names: Optional criterion names, in row/column order.
        warn: Emit a :class:`ConsistencyWarning` above the threshold.

    Returns:
        Mapping with ``weights`` (name -> weight), ``weights_geometric``,
        ``max_geometric_departure``, ``lambda_max``, ``consistency_index``,
        ``consistency_ratio``, ``consistency_ratio_max``, ``random_index``,
        ``n``, ``names``, ``consistent``, ``weight_spread`` and ``degenerate``.

    Raises:
        ValueError: If the matrix is not a valid pairwise matrix.
    """
    import numpy as np  # Deferred: see module docstring.

    array = validate_pairwise(matrix, names)
    size = array.shape[0]
    labels = (
        [str(name) for name in names]
        if names is not None
        else [str(position) for position in range(size)]
    )

    weights, lambda_max = principal_eigenvector(array, params)
    geometric = geometric_mean_weights(array)
    index = consistency_index(lambda_max, size)
    ratio = consistency_ratio(lambda_max, size, params)
    maximum = float(params["greening"]["ahp"]["consistency_ratio_max"])
    spread = float(weights.max() - weights.min())
    minimum_spread = float(params["greening"]["ahp"]["min_weight_spread"])

    # A matrix of all 1s is PERFECTLY consistent and has said nothing. A near-zero
    # consistency ratio can be evidence that no judgement was made rather than
    # that a good one was, which is why this is reported separately.
    degenerate = bool(spread < minimum_spread)
    consistent = bool(ratio <= maximum)

    report: dict[str, Any] = {
        "weights": {name: float(value) for name, value in zip(labels, weights)},
        "weights_geometric": {
            name: float(value) for name, value in zip(labels, geometric)
        },
        "max_geometric_departure": float(np.abs(weights - geometric).max()),
        "lambda_max": float(lambda_max),
        "consistency_index": float(index),
        "consistency_ratio": float(ratio),
        "consistency_ratio_max": maximum,
        "random_index": random_index(size, params),
        "n": int(size),
        "names": labels,
        "consistent": consistent,
        "weight_spread": spread,
        "min_weight_spread": minimum_spread,
        "degenerate": degenerate,
    }

    if warn and not consistent:
        warnings.warn(
            f"AHP consistency ratio {ratio:.4f} exceeds {maximum:.2f}. The "
            "judgements are not self-consistent, so these weights should not be "
            "published. Revise the pairwise comparisons in greening.ahp.pairwise "
            "- require_consistent will refuse to write a product from them.",
            ConsistencyWarning,
            stacklevel=2,
        )
    if warn and degenerate:
        warnings.warn(
            f"AHP weight spread {spread:.4f} is below {minimum_spread:.2f}: the "
            "judgements are nearly all 'equally important', which is perfectly "
            "consistent (CR near 0) and has decided nothing. A low consistency "
            "ratio here is evidence that no judgement was made.",
            ConsistencyWarning,
            stacklevel=2,
        )

    return report


def ahp_global_weights(
    levels: Mapping[str, Any], params: dict[str, Any]
) -> dict[str, float]:
    """Compose a two-level AHP hierarchy into global criterion weights.

    Reported as a sensitivity, never as the headline. It is the textbook response
    to the collinearity this module warns about: grouping the two heat criteria
    under one parent gives the heat bloc the weight the analyst *intended* rather
    than the weight a flat list happens to give it.

    Args:
        levels: Mapping of group name to the criteria in it, as configured in
            ``greening.ahp.hierarchy``.
        params: Parsed params mapping.

    Returns:
        Mapping of criterion name to global weight, summing to 1. Groups are
        weighted equally and each group's weight is split equally among its
        members, so this measures the effect of *grouping* alone rather than
        introducing a second set of unstated judgements.

    Raises:
        ValueError: If a group is empty, a criterion appears in two groups, or a
            member is not a configured criterion.
    """
    configured = set(criterion_names(params))
    groups = {str(name): [str(member) for member in members] for name, members in levels.items()}
    if not groups:
        raise ValueError("the hierarchy is empty; there is nothing to compose")

    seen: set[str] = set()
    for group, members in groups.items():
        if not members:
            raise ValueError(f"hierarchy group {group!r} has no criteria")
        for member in members:
            if member not in configured:
                raise ValueError(
                    f"hierarchy group {group!r} names {member!r}, which is not a "
                    f"configured criterion; greening.criteria has {sorted(configured)}"
                )
            if member in seen:
                raise ValueError(
                    f"criterion {member!r} appears in more than one hierarchy group"
                )
            seen.add(member)

    parent = 1.0 / len(groups)
    return {
        member: parent / len(members)
        for members in groups.values()
        for member in members
    }


def build_ahp_frame(
    report: Mapping[str, Any], params: dict[str, Any]
) -> "pd.DataFrame":
    """Tabulate an AHP report, one row per criterion.

    Args:
        report: The mapping :func:`ahp_weights` returned.
        params: Parsed params mapping.

    Returns:
        ``pandas.DataFrame`` with :data:`AHP_COLUMNS`, sorted by descending
        weight, carrying the consistency figures in ``.attrs``.
    """
    import pandas as pd  # Deferred: see module docstring.

    definitions = {
        str(entry["name"]): entry for entry in resolve_criteria(params)
    }
    rows = [
        {
            "criterion": name,
            "label": str(definitions.get(name, {}).get("label", name)),
            "direction": str(definitions.get(name, {}).get("direction", "")),
            "weight": float(weight),
            "weight_geometric": float(
                report["weights_geometric"].get(name, float("nan"))
            ),
        }
        for name, weight in report["weights"].items()
    ]
    frame = pd.DataFrame(rows, columns=list(AHP_COLUMNS[:-1]))
    frame = frame.sort_values("weight", ascending=False).reset_index(drop=True)
    frame["weight_rank"] = frame.index + 1
    frame = frame[list(AHP_COLUMNS)]
    for key in (
        "lambda_max",
        "consistency_index",
        "consistency_ratio",
        "consistency_ratio_max",
        "random_index",
        "n",
        "consistent",
        "weight_spread",
        "degenerate",
        "max_geometric_departure",
    ):
        frame.attrs[key] = report[key]
    return frame


def require_consistent(
    report: Mapping[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """Refuse to proceed unless the AHP judgements can support a product.

    Called by every writer and export in this module, **before** anything touches
    disk or ``ee.batch``, and by every Phase 7 figure builder - which catches the
    exception and draws a stamped figure rather than refusing, because a figure
    showing that judgements failed is exactly what the report needs.

    Args:
        report: The mapping :func:`ahp_weights` returned.
        params: Parsed params mapping.

    Returns:
        The report, unchanged, when it passes.

    Raises:
        InconsistentJudgements: If the consistency ratio exceeds the threshold,
            if the matrix is degenerate, or if the weight spread is too small.
        ValueError: If the report is missing the fields a judgement needs.
    """
    required = ("consistency_ratio", "consistency_ratio_max", "consistent", "weight_spread")
    absent = [key for key in required if key not in report]
    if absent:
        raise ValueError(
            f"the AHP report is missing {absent}; it was not produced by "
            "ahp_weights, so there is no judgement to enforce"
        )

    ratio = float(report["consistency_ratio"])
    maximum = float(report["consistency_ratio_max"])
    if not math.isfinite(ratio):
        raise InconsistentJudgements(
            f"the consistency ratio is {ratio}; the pairwise matrix is degenerate"
        )
    if ratio > maximum:
        raise InconsistentJudgements(
            f"AHP consistency ratio {ratio:.4f} exceeds the {maximum:.2f} "
            "threshold, so no priority product may be written from these "
            "judgements. Revise greening.ahp.pairwise until the ratio passes; "
            "ahp_weights() will show you the weights meanwhile."
        )

    minimum_spread = float(
        report.get("min_weight_spread", params["greening"]["ahp"]["min_weight_spread"])
    )
    spread = float(report["weight_spread"])
    if spread < minimum_spread:
        raise InconsistentJudgements(
            f"AHP weight spread {spread:.4f} is below {minimum_spread:.2f}. The "
            "judgements are effectively all 'equally important', which scores a "
            f"consistency ratio of {ratio:.4f} precisely because no judgement "
            "was made. A passing CR here means nothing was decided, not that it "
            "was decided well."
        )
    return dict(report)


# =============================================================================
# Group C - criterion preparation
# =============================================================================


def apply_direction(values: "np.ndarray", direction: str) -> "np.ndarray":
    """Orient a criterion so that higher always means higher priority.

    Applied **exactly once**, in :func:`prepare_criteria`. Everything downstream
    - the weighted overlay, TOPSIS's ideal solutions, the ablation - then treats
    every column as a benefit. That kills a whole class of sign bugs, and it is
    what makes :func:`ideal_solutions` correct with a plain column maximum.

    Args:
        values: Raw criterion values.
        direction: ``"benefit"`` (higher is better) or ``"cost"`` (lower is).

    Returns:
        The values, negated for a cost criterion.

    Raises:
        ValueError: If the direction is unknown.
    """
    import numpy as np  # Deferred: see module docstring.

    if direction not in DIRECTIONS:
        raise ValueError(f"direction {direction!r} is not one of {list(DIRECTIONS)}")
    array = np.asarray(values, dtype=float)
    return -array if direction == "cost" else array


def percentile_rank(values: "np.ndarray") -> "np.ndarray":
    """Percentile rank in ``(0, 1]``, **preserving NaN**.

    Never ``na_option="bottom"``. ``prediction.interim_priority_zones`` used it,
    which was tolerable for an interim proxy and is not for a published
    recommendation: a division with no NDVI because it sat under cloud is not
    thereby a low-priority division, and sinking it silently is a wrong answer
    that never announces itself.

    Args:
        values: Criterion values, already direction-corrected.

    Returns:
        Percentile ranks, NaN wherever the input was NaN. Ties take their
        average rank.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd

    series = pd.Series(np.asarray(values, dtype=float))
    return series.rank(pct=True, na_option="keep").to_numpy(dtype=float)


def min_max_scale(values: "np.ndarray") -> "np.ndarray":
    """Scale to ``[0, 1]``, preserving NaN.

    Args:
        values: Criterion values, already direction-corrected.

    Returns:
        Scaled values. A constant column returns 0.5 everywhere with a warning
        rather than dividing by zero - it carries no information, and 0.5 says
        so without ranking anything.
    """
    import numpy as np  # Deferred: see module docstring.

    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.full(array.shape, np.nan)

    low = float(finite.min())
    high = float(finite.max())
    if math.isclose(high, low, rel_tol=0, abs_tol=0.0):
        warnings.warn(
            "min_max_scale received a constant criterion; returning 0.5 "
            "everywhere. A constant column carries no information and cannot "
            "discriminate between zones, whatever weight it is given.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.where(np.isfinite(array), 0.5, np.nan)
    return (array - low) / (high - low)


def z_score(values: "np.ndarray") -> "np.ndarray":
    """Standardise to zero mean and unit variance, preserving NaN.

    Uses the population standard deviation (``ddof=0``), matching
    ``uhi.zscore.ddof`` - which Phase 3 settled empirically against
    ``ee.Reducer.stdDev``, so the two agree exactly rather than to ``O(1/n)``.

    Args:
        values: Criterion values, already direction-corrected.

    Returns:
        Standardised values; 0.0 for a constant column.
    """
    import numpy as np  # Deferred: see module docstring.

    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if not finite.any():
        return np.full(array.shape, np.nan)

    mean = float(array[finite].mean())
    deviation = float(array[finite].std(ddof=0))
    if math.isclose(deviation, 0.0, rel_tol=0, abs_tol=1e-15):
        return np.where(finite, 0.0, np.nan)
    return (array - mean) / deviation


def normalise_criterion(
    values: "np.ndarray",
    direction: str,
    method: str | None = None,
    params: dict[str, Any] | None = None,
) -> "np.ndarray":
    """Direction-correct and normalise one criterion.

    Args:
        values: Raw criterion values.
        direction: ``"benefit"`` or ``"cost"``.
        method: One of :data:`NORMALISATIONS`, or ``None`` for the configured one.
        params: Parsed params mapping, required when ``method`` is ``None``.

    Returns:
        Normalised values in which higher always means higher priority.

    Raises:
        ValueError: If the method or direction is unknown, or if neither
            ``method`` nor ``params`` was supplied.
    """
    if method is None:
        if params is None:
            raise ValueError("normalise_criterion needs either method= or params=")
        resolved = resolve_normalisation(None, params)
    else:
        resolved = str(method)
        if resolved not in NORMALISATIONS:
            raise ValueError(
                f"normalisation {resolved!r} is not one of {list(NORMALISATIONS)}"
            )

    oriented = apply_direction(values, direction)
    if resolved == "percentile_rank":
        return percentile_rank(oriented)
    if resolved == "min_max":
        return min_max_scale(oriented)
    return z_score(oriented)


def land_observed_fraction(
    landscape: "pd.DataFrame",
    land_area: "pd.DataFrame",
    params: dict[str, Any],
    landscape_area_column: str = "landscape_area_ha",
    land_column: str = "land_area_ha",
) -> "pd.DataFrame":
    """Recompute the land-cover coverage floor against LAND, not polygon, area.

    .. warning::
        **The Phase 5 floor fires on water, not on cloud, and excluding on it
        deletes the CMC core.** ``spatial_stats.landscape.min_observed_fraction``
        compares classified area against the *polygon*. Measured on the committed
        outputs: the identical ``observed_fraction`` appears for Dynamic World
        2018, Dynamic World 2024 *and* WorldCover 2021 in 552 of 557 zones -
        three classifiers, three dates, one number. It is not measuring
        classifier coverage. Fort's COD-AB polygon is 7.46 km2 enclosing 131 ha
        of land, because it *is* the Colombo Port outer harbour (CLAUDE.md's
        6.89 km2). Excluding on the raw flag drops Pettah and Lunupokuna from the
        priority list: dense, hot, treeless CMC-core divisions, which is exactly
        what a greening priority list exists to find.

    Both flags travel onward, and the count of zones whose status changes is
    returned so the notebook can print it rather than quietly benefiting from
    the fix.

    Args:
        landscape: Per-zone landscape metrics, with ``zone_id``, the classified
            area column and the raw ``observed_fraction``.
        land_area: Per-zone land area, with ``zone_id`` and ``land_column``.
        params: Parsed params mapping.
        landscape_area_column: Column holding the analysable (classified) area.
        land_column: Column holding the land area of the polygon.

    Returns:
        ``pandas.DataFrame`` with ``zone_id``, ``observed_fraction_raw``,
        ``land_observed_fraction``, ``below_coverage_floor_raw``,
        ``below_land_coverage_floor`` and ``status_changed``.

    Raises:
        ValueError: If a required column is absent or ``zone_id`` duplicates.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd

    for frame, name, needed in (
        (landscape, "landscape", ("zone_id", landscape_area_column)),
        (land_area, "land_area", ("zone_id", land_column)),
    ):
        missing = [column for column in needed if column not in frame.columns]
        if missing:
            raise ValueError(
                f"{name} frame is missing {missing}; it has {sorted(frame.columns)}"
            )
        if frame["zone_id"].astype(str).duplicated().any():
            raise ValueError(f"{name} frame has duplicate zone_id values")

    floor = float(
        params["greening"]["normalisation"]["missing"]["min_land_observed_fraction"]
    )
    raw_floor = float(params["spatial_stats"]["landscape"]["min_observed_fraction"])

    left = landscape.copy()
    left["zone_id"] = left["zone_id"].astype(str)
    right = land_area.copy()
    right["zone_id"] = right["zone_id"].astype(str)

    merged = left.merge(right[["zone_id", land_column]], on="zone_id", how="left")
    classified = pd.to_numeric(merged[landscape_area_column], errors="coerce")
    land = pd.to_numeric(merged[land_column], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(land > 0, classified / land, np.nan)
    # A polygon can be classified over marginally more land than the coarser land
    # mask reports; this is a coverage measure, not an area, so cap it at 1.
    fraction = np.clip(fraction, 0.0, 1.0)

    if "observed_fraction" in merged.columns:
        raw = pd.to_numeric(merged["observed_fraction"], errors="coerce").to_numpy()
    else:
        raw = np.full(len(merged), np.nan)

    result = pd.DataFrame(
        {
            "zone_id": merged["zone_id"].astype(str),
            "observed_fraction_raw": raw,
            "land_observed_fraction": fraction,
            "below_coverage_floor_raw": raw < raw_floor,
            "below_land_coverage_floor": fraction < floor,
        }
    )
    result["status_changed"] = result["below_coverage_floor_raw"].fillna(
        False
    ) != result["below_land_coverage_floor"].fillna(False)
    result.attrs["min_land_observed_fraction"] = floor
    result.attrs["min_observed_fraction_raw"] = raw_floor
    result.attrs["n_status_changed"] = int(result["status_changed"].sum())
    result.attrs["n_below_floor_raw"] = int(
        result["below_coverage_floor_raw"].fillna(False).sum()
    )
    result.attrs["n_below_floor_land"] = int(
        result["below_land_coverage_floor"].fillna(False).sum()
    )
    return result


def criterion_quality_flags(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    criteria: Sequence[str] | None = None,
) -> "pd.DataFrame":
    """Flag zones whose criterion values cannot support a ranking.

    CLAUDE.md caveat 2 at the zone level: a division mean over four cloud-free
    100 m pixels is not the same datum as one over four thousand, and the
    difference must be visible in the output rather than buried in it.

    Args:
        frame: One row per zone with ``zone_id``, the criterion columns and, for
            each, its ``_pixels`` count where one exists.
        params: Parsed params mapping.
        criteria: Optional criterion subset.

    Returns:
        ``pandas.DataFrame`` with ``zone_id``, ``n_pixels_min``, any coverage
        columns carried through, and a boolean ``<name>_usable`` per criterion.

    Raises:
        ValueError: If ``zone_id`` is absent or duplicated.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd

    if "zone_id" not in frame.columns:
        raise ValueError(f"frame has no 'zone_id' column; it has {sorted(frame.columns)}")
    if frame["zone_id"].astype(str).duplicated().any():
        raise ValueError(
            "frame has duplicate zone_id values; merge the criterion tables "
            "before flagging, or the same division competes with itself"
        )

    definitions = resolve_criteria(params, criteria)
    minimum = int(params["greening"]["normalisation"]["missing"]["min_pixels"])

    out = pd.DataFrame({"zone_id": frame["zone_id"].astype(str)})
    counts: list[Any] = []
    for entry in definitions:
        name = str(entry["name"])
        column = str(entry["column"])
        values = (
            pd.to_numeric(frame[column], errors="coerce")
            if column in frame.columns
            else pd.Series(np.nan, index=frame.index)
        )
        usable = values.notna().to_numpy()

        pixels_column = entry.get("pixels_column")
        if pixels_column and str(pixels_column) in frame.columns:
            pixels = pd.to_numeric(frame[str(pixels_column)], errors="coerce").to_numpy()
            counts.append(pixels)
            usable = usable & (pixels >= minimum)
        out[f"{name}_usable"] = usable

    if counts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out["n_pixels_min"] = np.nanmin(np.vstack(counts), axis=0)
    else:
        out["n_pixels_min"] = np.full(len(frame), np.nan)

    for column in ("below_land_coverage_floor", "land_observed_fraction"):
        if column in frame.columns:
            out[column] = frame[column].to_numpy()
    out.attrs["min_pixels"] = minimum
    return out


def redistribute_weights(
    weights: Mapping[str, float], present: Mapping[str, bool]
) -> dict[str, float]:
    """Renormalise weights over the criteria a zone actually has.

    Args:
        weights: Criterion name to weight.
        present: Criterion name to whether the zone has a usable value.

    Returns:
        Weights over the present criteria, summing to 1. All zero when nothing
        is present - the caller flags the zone rather than scoring it from
        nothing.
    """
    total = sum(
        float(weight) for name, weight in weights.items() if present.get(name, False)
    )
    if total <= 0:
        return {name: 0.0 for name in weights}
    return {
        name: (float(weight) / total if present.get(name, False) else 0.0)
        for name, weight in weights.items()
    }


def prepare_criteria(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    criteria: Sequence[str] | None = None,
    method: str | None = None,
) -> tuple["pd.DataFrame", dict[str, Any]]:
    """Build the decision matrix: direction-corrected, normalised, flagged.

    After this, **higher always means higher priority** in every ``_norm``
    column, which is what lets the overlay be a plain weighted sum and TOPSIS
    take a plain column maximum as its ideal.

    Args:
        frame: One row per zone with ``zone_id`` and the criterion columns.
        params: Parsed params mapping.
        criteria: Optional criterion subset/order.
        method: Override for ``greening.normalisation.method``.

    Returns:
        ``(prepared, report)``. ``prepared`` carries ``zone_id``, the gated raw
        criterion columns, a ``<name>_norm`` column each, plus ``n_pixels_min``,
        ``incomplete_criteria``, ``missing_weight`` and ``status``. ``report``
        records the method, criteria, weights, per-criterion missing counts and
        the status tallies.

    Raises:
        ValueError: If ``zone_id`` is absent or duplicated, or a criterion column
            is absent from the frame entirely.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd

    if "zone_id" not in frame.columns:
        raise ValueError(f"frame has no 'zone_id' column; it has {sorted(frame.columns)}")
    if frame["zone_id"].astype(str).duplicated().any():
        raise ValueError(
            "frame has duplicate zone_id values; merge the criterion tables "
            "before preparing, or the same division competes with itself"
        )

    definitions = resolve_criteria(params, criteria)
    names = [str(entry["name"]) for entry in definitions]
    resolved_method = resolve_normalisation(method, params)
    policy = str(params["greening"]["normalisation"]["missing"]["policy"])
    if policy not in MISSING_POLICIES:
        raise ValueError(
            f"missing policy {policy!r} is not one of {list(MISSING_POLICIES)}"
        )
    max_missing = float(
        params["greening"]["normalisation"]["missing"]["max_missing_weight"]
    )

    absent = [
        str(entry["column"])
        for entry in definitions
        if str(entry["column"]) not in frame.columns
    ]
    if absent:
        raise ValueError(
            f"criterion column(s) {absent} are not in the frame, which has "
            f"{sorted(frame.columns)}. A criterion absent for EVERY zone is a "
            "missing input, not missing data - fix the join rather than letting "
            "its weight be redistributed 557 times."
        )

    quality = criterion_quality_flags(frame, params, criteria)
    prepared = pd.DataFrame({"zone_id": frame["zone_id"].astype(str)})

    missing_counts: dict[str, int] = {}
    for entry in definitions:
        name = str(entry["name"])
        column = str(entry["column"])
        raw = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        usable = quality[f"{name}_usable"].to_numpy(dtype=bool)
        gated = np.where(usable, raw, np.nan)
        prepared[column] = gated
        prepared[f"{name}_norm"] = normalise_criterion(
            gated, str(entry["direction"]), resolved_method
        )
        missing_counts[name] = int((~usable).sum())

    prepared["n_pixels_min"] = quality["n_pixels_min"].to_numpy()
    for column in ("land_observed_fraction", "below_land_coverage_floor"):
        if column in quality.columns:
            prepared[column] = quality[column].to_numpy()

    # The AHP weights decide how much of a zone's evidence is actually missing.
    # A zone lacking only the 0.068-weighted UTFVI share is in a different
    # position from one lacking the 0.319-weighted heat criterion, and a plain
    # count of absent criteria cannot tell the two apart.
    matrix, matrix_names = pairwise_matrix(params, names)
    report = ahp_weights(matrix, params, matrix_names, warn=False)
    weights = report["weights"]

    norms = prepared[[f"{name}_norm" for name in names]].to_numpy(dtype=float)
    present = np.isfinite(norms)
    weight_vector = np.array([float(weights[name]) for name in names], dtype=float)
    missing_weight = ((~present) * weight_vector).sum(axis=1)

    prepared["incomplete_criteria"] = (~present).sum(axis=1)
    prepared["missing_weight"] = missing_weight

    status = np.where(
        missing_weight > max_missing, STATUS_INSUFFICIENT, STATUS_OK
    ).astype(object)
    if "below_land_coverage_floor" in prepared.columns:
        floored = (
            prepared["below_land_coverage_floor"].fillna(False).to_numpy(dtype=bool)
        )
        status = np.where((status == STATUS_OK) & floored, STATUS_BELOW_FLOOR, status)
    prepared["status"] = status

    summary: dict[str, Any] = {
        "method": resolved_method,
        "criteria": names,
        "weights": {name: float(weights[name]) for name in names},
        "policy": policy,
        "max_missing_weight": max_missing,
        "missing_per_criterion": missing_counts,
        "n_zones": int(len(prepared)),
        "n_ok": int((prepared["status"] == STATUS_OK).sum()),
        "n_insufficient": int((prepared["status"] == STATUS_INSUFFICIENT).sum()),
        "n_below_floor": int((prepared["status"] == STATUS_BELOW_FLOOR).sum()),
        "min_pixels": int(quality.attrs["min_pixels"]),
    }
    prepared.attrs.update(summary)
    return prepared, summary


def require_scored_fraction(
    prepared: "pd.DataFrame", params: dict[str, Any]
) -> float:
    """Refuse to publish a ranking built from too few scored zones.

    Args:
        prepared: The frame :func:`prepare_criteria` returned.
        params: Parsed params mapping.

    Returns:
        The share of zones that produced a score.

    Raises:
        CriteriaIncomplete: Below ``greening.min_scored_fraction``.
    """
    floor = float(params["greening"]["min_scored_fraction"])
    if not len(prepared):
        raise CriteriaIncomplete("no zones at all; there is nothing to rank")
    scored = float((prepared["status"] != STATUS_INSUFFICIENT).mean())
    if scored < floor:
        raise CriteriaIncomplete(
            f"only {scored:.1%} of zones could be scored, below the "
            f"{floor:.0%} floor in greening.min_scored_fraction. A ranking over "
            "this many zones describes the coverage of the input rasters more "
            "than it describes Colombo."
        )
    return scored


# =============================================================================
# Group D - the weighted overlay
# =============================================================================


def weighted_overlay(matrix: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    """Weighted sum of a decision matrix, redistributing weight over NaN.

    Args:
        matrix: ``(n_alternatives, n_criteria)`` of normalised, direction-
            corrected values. NaN marks a criterion the alternative lacks.
        weights: Criterion weights, summing to 1.

    Returns:
        One score per alternative. A row with no usable criterion scores NaN
        rather than zero - zero is a legitimate score and would rank the zone
        last on evidence that does not exist.

    Raises:
        ValueError: If the shapes disagree or the weights do not sum to 1.
    """
    import numpy as np  # Deferred: see module docstring.

    values = np.asarray(matrix, dtype=float)
    vector = np.asarray(weights, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"the decision matrix must be 2-D; got shape {values.shape}")
    if vector.shape != (values.shape[1],):
        raise ValueError(
            f"{vector.size} weight(s) for {values.shape[1]} criteria"
        )
    if not math.isclose(float(vector.sum()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"weights sum to {vector.sum()}, not 1")

    present = np.isfinite(values)
    available = present * vector
    total = available.sum(axis=1)
    contribution = np.where(present, values, 0.0) * available
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, contribution.sum(axis=1) / total, np.nan)


def mcda_scores(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    weights: Mapping[str, float],
    criteria: Sequence[str] | None = None,
) -> "pd.DataFrame":
    """Score prepared zones by the AHP-weighted overlay.

    ``weights`` is a mapping keyed by criterion **name**, never a positional
    array, for the same reason the judgements are named pairs: a reordering
    somewhere else cannot silently reattach a weight to the wrong criterion.

    Args:
        frame: The frame :func:`prepare_criteria` returned.
        params: Parsed params mapping.
        weights: Criterion name to weight.
        criteria: Optional criterion subset; defaults to the keys of ``weights``.

    Returns:
        ``pandas.DataFrame`` with ``zone_id``, ``score_ahp`` and the
        pass-through flag columns.

    Raises:
        ValueError: If a normalised column is absent or a weight is missing.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd

    names = (
        [str(name) for name in criteria]
        if criteria is not None
        else [str(name) for name in weights]
    )
    unknown = [name for name in names if name not in weights]
    if unknown:
        raise ValueError(f"no weight supplied for criteria {unknown}")

    columns = [f"{name}_norm" for name in names]
    absent = [column for column in columns if column not in frame.columns]
    if absent:
        raise ValueError(
            f"{absent} not in the frame; pass the output of prepare_criteria, "
            f"which has {sorted(frame.columns)}"
        )

    vector = np.array([float(weights[name]) for name in names], dtype=float)
    total = float(vector.sum())
    if total <= 0:
        raise ValueError("the supplied weights sum to zero")
    vector = vector / total

    scores = weighted_overlay(frame[columns].to_numpy(dtype=float), vector)

    out = pd.DataFrame({"zone_id": frame["zone_id"].astype(str), "score_ahp": scores})
    for column in (
        "n_pixels_min",
        "land_observed_fraction",
        "below_land_coverage_floor",
        "incomplete_criteria",
        "missing_weight",
        "status",
    ):
        if column in frame.columns:
            out[column] = frame[column].to_numpy()

    # A zone the pipeline refused to score must not carry a number that ranks.
    if "status" in out.columns:
        out.loc[out["status"] == STATUS_INSUFFICIENT, "score_ahp"] = np.nan

    out.attrs["weights"] = {name: float(weights[name]) for name in names}
    out.attrs["criteria"] = names
    return out


def rank_frame(
    scores: "pd.DataFrame",
    params: dict[str, Any],
    top_n: int | None = None,
    score_column: str = "score_ahp",
    rank_column: str | None = None,
) -> "pd.DataFrame":
    """Rank scored zones, mark the top N, and report the gap at the cut.

    .. note::
        **A top-60 is meaningless if ranks 60 and 61 differ in the fourth
        decimal.** With five rank-normalised criteria the score distribution is
        smooth, so ``score_gap_at_cut`` and ``tied_at_cut`` are computed and
        carried both in ``.attrs`` and as columns. The report quotes the score,
        not only the rank.

    Args:
        scores: Frame with ``zone_id`` and ``score_column``.
        params: Parsed params mapping.
        top_n: Override for ``greening.top_n``.
        score_column: Which score to rank on.
        rank_column: Name for the rank column. Defaults to ``score_ahp`` ->
            ``rank_ahp``.

    Returns:
        ``pandas.DataFrame`` sorted by descending score, with the rank column and
        a boolean ``priority``. Unscorable and below-floor zones are ranked but
        never flagged priority.

    Raises:
        ValueError: If the score column is absent or ``zone_id`` duplicates.
    """
    if score_column not in scores.columns:
        raise ValueError(
            f"no {score_column!r} column; the frame has {sorted(scores.columns)}"
        )
    if scores["zone_id"].astype(str).duplicated().any():
        raise ValueError("scores frame has duplicate zone_id values")

    name = rank_column or score_column.replace("score", "rank")
    limit = int(params["greening"]["top_n"] if top_n is None else top_n)
    exclude_floored = bool(
        params["greening"]["normalisation"]["missing"]["exclude_below_floor_from_top_n"]
    )

    out = scores.copy()
    out["zone_id"] = out["zone_id"].astype(str)
    # NaN scores sort last deterministically rather than by input order.
    out = out.sort_values(
        score_column, ascending=False, na_position="last", kind="mergesort"
    ).reset_index(drop=True)
    out[name] = out.index + 1

    eligible = out[score_column].notna()
    if exclude_floored and "status" in out.columns:
        eligible = eligible & (out["status"] != STATUS_BELOW_FLOOR)
    if exclude_floored and "below_land_coverage_floor" in out.columns:
        eligible = eligible & ~out["below_land_coverage_floor"].fillna(False).astype(
            bool
        )

    out["priority"] = eligible & (eligible.cumsum() <= limit)

    selected = out.loc[out["priority"], score_column]
    remainder = out.loc[eligible & ~out["priority"], score_column]
    if len(selected) and len(remainder):
        gap = float(selected.min() - remainder.max())
        tied = int((remainder >= selected.min() - 1e-12).sum())
    else:
        gap = float("nan")
        tied = 0

    out.attrs["top_n"] = limit
    out.attrs["n_priority"] = int(out["priority"].sum())
    out.attrs["n_eligible"] = int(eligible.sum())
    out.attrs["score_gap_at_cut"] = gap
    out.attrs["tied_at_cut"] = tied
    out.attrs["score_column"] = score_column
    out.attrs["rank_column"] = name
    out["score_gap_at_cut"] = gap
    out["tied_at_cut"] = tied
    return out


# =============================================================================
# Group E - TOPSIS
# =============================================================================


def vector_normalise(matrix: "np.ndarray") -> "np.ndarray":
    """TOPSIS vector normalisation: ``x_ij / sqrt(sum_i x_ij^2)``.

    This is the method's own definition (Hwang & Yoon 1981). Feeding TOPSIS
    percentile ranks instead would make it a different method wearing the same
    name - which is why ``greening.topsis.also_on_ranks`` runs *both* and the
    notebook reports the three-way decomposition rather than one number.

    Args:
        matrix: ``(n_alternatives, n_criteria)`` of direction-corrected values.

    Returns:
        Column-normalised matrix. A zero column is returned unchanged rather
        than divided by zero.

    Raises:
        ValueError: If the matrix is not 2-D.
    """
    import numpy as np  # Deferred: see module docstring.

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"the decision matrix must be 2-D; got shape {values.shape}")

    norms = np.sqrt(np.nansum(values**2, axis=0))
    return values / np.where(norms > 0, norms, 1.0)


def weighted_matrix(matrix: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    """Multiply each normalised column by its criterion weight.

    Args:
        matrix: Normalised decision matrix.
        weights: Criterion weights.

    Returns:
        The weighted normalised matrix.

    Raises:
        ValueError: If the shapes disagree.
    """
    import numpy as np  # Deferred: see module docstring.

    values = np.asarray(matrix, dtype=float)
    vector = np.asarray(weights, dtype=float)
    if values.ndim != 2 or vector.shape != (values.shape[1],):
        raise ValueError(
            f"cannot weight a {values.shape} matrix with {vector.shape} weights"
        )
    return values * vector


def ideal_solutions(matrix: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    """Positive and negative ideal solutions of a weighted matrix.

    Every column is a benefit here, because :func:`apply_direction` already
    negated the cost criteria. So the positive ideal is a plain column maximum
    and the negative a plain minimum.

    That equivalence is worth stating because a future reader will otherwise
    "fix" it: for a cost criterion, negating and then taking the maximum selects
    exactly the same alternative as leaving it and taking the minimum. A test
    pins that the two routes agree.

    Args:
        matrix: Weighted normalised decision matrix.

    Returns:
        ``(positive_ideal, negative_ideal)``, one value per criterion.

    Raises:
        ValueError: If the matrix is not 2-D or has no rows.
    """
    import numpy as np  # Deferred: see module docstring.

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(
            f"cannot derive ideal solutions from a matrix of shape {values.shape}; "
            "the ideals are drawn from the alternative set, so there must be one"
        )
    return np.nanmax(values, axis=0), np.nanmin(values, axis=0)


def separation_measures(
    matrix: "np.ndarray", positive: "np.ndarray", negative: "np.ndarray"
) -> tuple["np.ndarray", "np.ndarray"]:
    """Euclidean distance from each alternative to the two ideal solutions.

    Args:
        matrix: Weighted normalised decision matrix.
        positive: Positive ideal solution.
        negative: Negative ideal solution.

    Returns:
        ``(d_plus, d_minus)``, one distance per alternative. NaN criteria are
        skipped rather than propagating, so a zone missing one criterion is
        compared on the ones it has.
    """
    import numpy as np  # Deferred: see module docstring.

    values = np.asarray(matrix, dtype=float)
    d_plus = np.sqrt(np.nansum((values - np.asarray(positive, float)) ** 2, axis=1))
    d_minus = np.sqrt(np.nansum((values - np.asarray(negative, float)) ** 2, axis=1))
    return d_plus, d_minus


def closeness_coefficient(
    d_plus: "np.ndarray", d_minus: "np.ndarray"
) -> "np.ndarray":
    """TOPSIS closeness ``d- / (d+ + d-)``, in ``[0, 1]``.

    Args:
        d_plus: Distance to the positive ideal.
        d_minus: Distance to the negative ideal.

    Returns:
        Closeness coefficients. A zero denominator - every alternative identical
        on every criterion - returns 0.5 with a warning rather than NaN or a
        divide-by-zero, because then no alternative is closer to either ideal.
    """
    import numpy as np  # Deferred: see module docstring.

    plus = np.asarray(d_plus, dtype=float)
    minus = np.asarray(d_minus, dtype=float)
    total = plus + minus
    if bool(np.any(total <= 0)):
        warnings.warn(
            "TOPSIS found alternatives equidistant from both ideals (d+ + d- = "
            "0), which means they are identical on every criterion; scoring them "
            "0.5. Check that the criteria discriminate at all before reading the "
            "ranking.",
            RuntimeWarning,
            stacklevel=2,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, minus / total, 0.5)


def topsis(
    matrix: "np.ndarray",
    weights: "np.ndarray",
    params: dict[str, Any],
    normalise: bool = True,
) -> dict[str, Any]:
    """Run TOPSIS over a direction-corrected decision matrix.

    Args:
        matrix: ``(n_alternatives, n_criteria)``, every column a benefit.
        weights: Criterion weights.
        params: Parsed params mapping.
        normalise: Apply :func:`vector_normalise` first. ``False`` when the
            matrix is already on a common scale (the run on percentile ranks).

    Returns:
        Mapping with ``closeness``, ``d_plus``, ``d_minus``, ``positive_ideal``,
        ``negative_ideal``, ``normalised``, ``weighted``, ``n_alternatives`` and
        ``normalisation``.

    Raises:
        ValueError: If the matrix is empty or the shapes disagree.
    """
    import numpy as np  # Deferred: see module docstring.

    values = np.asarray(matrix, dtype=float)
    vector = np.asarray(weights, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(
            f"TOPSIS needs at least one alternative; got shape {values.shape}"
        )
    if vector.shape != (values.shape[1],):
        raise ValueError(f"{vector.shape} weights for a {values.shape} decision matrix")

    normalised = vector_normalise(values) if normalise else values
    weighted = weighted_matrix(normalised, vector)
    positive, negative = ideal_solutions(weighted)
    d_plus, d_minus = separation_measures(weighted, positive, negative)

    return {
        "closeness": closeness_coefficient(d_plus, d_minus),
        "d_plus": d_plus,
        "d_minus": d_minus,
        "positive_ideal": positive,
        "negative_ideal": negative,
        "normalised": normalised,
        "weighted": weighted,
        "n_alternatives": int(values.shape[0]),
        "normalisation": (
            str(params["greening"]["topsis"]["normalisation"]) if normalise else "none"
        ),
    }


def topsis_scores(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    weights: Mapping[str, float],
    criteria: Sequence[str] | None = None,
    on_ranks: bool = False,
) -> "pd.DataFrame":
    """Score prepared zones by TOPSIS.

    .. warning::
        **TOPSIS reverses ranks when the alternative set changes.** The ideal and
        anti-ideal are drawn from the set, so removing one division can reverse
        the order of two others. Whenever zones are dropped - below-floor
        exclusions, ``insufficient_data`` - this must be **re-run** on the
        retained set, never sub-selected from an earlier run.
        :func:`compare_rankings` refuses mismatched zone sets and names this as
        the reason.

    Args:
        frame: The frame :func:`prepare_criteria` returned.
        params: Parsed params mapping.
        weights: Criterion name to weight.
        criteria: Optional criterion subset.
        on_ranks: Run on the percentile-rank columns with no further
            normalisation, rather than on the raw direction-corrected values.
            This second run is what lets METHOD be told apart from NORMALISATION.

    Returns:
        ``pandas.DataFrame`` with ``zone_id``, ``score_topsis``, ``d_plus`` and
        ``d_minus``, carrying ``n_alternatives`` in ``.attrs``.

    Raises:
        ValueError: If a required column is absent or the weights sum to zero.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd

    definitions = resolve_criteria(
        params, criteria if criteria is not None else list(weights)
    )
    names = [str(entry["name"]) for entry in definitions]

    if on_ranks:
        columns = [f"{name}_norm" for name in names]
        absent = [column for column in columns if column not in frame.columns]
        if absent:
            raise ValueError(f"{absent} not in the frame; pass prepare_criteria output")
        values = frame[columns].to_numpy(dtype=float)
        normalise = False
    else:
        columns = [str(entry["column"]) for entry in definitions]
        absent = [column for column in columns if column not in frame.columns]
        if absent:
            raise ValueError(f"{absent} not in the frame; pass prepare_criteria output")
        # Direction correction must happen here too, so TOPSIS sees the same
        # orientation the overlay did and the two are genuinely comparable.
        values = np.column_stack(
            [
                apply_direction(
                    pd.to_numeric(frame[str(entry["column"])], errors="coerce").to_numpy(
                        dtype=float
                    ),
                    str(entry["direction"]),
                )
                for entry in definitions
            ]
        )
        # Vector normalisation is not translation invariant, so a negated cost
        # column would otherwise contribute a sign the method never intended.
        # Shifting each column to be non-negative preserves the ordering and the
        # spacing, which is all TOPSIS uses.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            floors = np.nanmin(values, axis=0)
        floors = np.where(np.isfinite(floors), floors, 0.0)
        values = values - np.where(floors < 0, floors, 0.0)
        normalise = True

    missing = [name for name in names if name not in weights]
    if missing:
        raise ValueError(f"no weight supplied for criteria {missing}")
    vector = np.array([float(weights[name]) for name in names], dtype=float)
    total = float(vector.sum())
    if total <= 0:
        raise ValueError("the supplied weights sum to zero")
    vector = vector / total

    result = topsis(values, vector, params, normalise=normalise)

    out = pd.DataFrame(
        {
            "zone_id": frame["zone_id"].astype(str),
            "score_topsis": np.asarray(result["closeness"], dtype=float),
            "d_plus": result["d_plus"],
            "d_minus": result["d_minus"],
        }
    )
    for column in ("status", "below_land_coverage_floor", "n_pixels_min"):
        if column in frame.columns:
            out[column] = frame[column].to_numpy()
    if "status" in out.columns:
        out.loc[out["status"] == STATUS_INSUFFICIENT, "score_topsis"] = np.nan

    out.attrs["n_alternatives"] = int(result["n_alternatives"])
    out.attrs["normalisation"] = result["normalisation"]
    out.attrs["on_ranks"] = bool(on_ranks)
    out.attrs["criteria"] = names
    return out


# =============================================================================
# Group F - comparison and robustness
# =============================================================================


def spearman_rho(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Spearman rank correlation and its p-value.

    Args:
        a: First ranking or score.
        b: Second, in the same order.

    Returns:
        ``(rho, p_value)``.

    Raises:
        ValueError: If the lengths differ or fewer than three pairs are finite.
    """
    import numpy as np  # Deferred: see module docstring.
    from scipy import stats

    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.shape != right.shape:
        raise ValueError(f"cannot correlate {left.shape} against {right.shape}")

    usable = np.isfinite(left) & np.isfinite(right)
    if int(usable.sum()) < 3:
        raise ValueError(
            f"only {int(usable.sum())} pair(s) are finite; a rank correlation "
            "over fewer than three is not a statistic"
        )
    result = stats.spearmanr(left[usable], right[usable])
    return float(result.statistic), float(result.pvalue)


def compare_rankings(
    left: "pd.DataFrame",
    right: "pd.DataFrame",
    params: dict[str, Any],
    left_name: str = "AHP weighted overlay",
    right_name: str = "TOPSIS",
    left_rank: str = "rank_ahp",
    right_rank: str = "rank_topsis",
) -> dict[str, Any]:
    """Compare two rankings of the same zones.

    .. warning::
        Refuses mismatched zone sets. That is not fussiness: TOPSIS draws its
        ideal and anti-ideal from the alternative *set*, so a ranking computed
        over 557 zones and one computed over 535 are not comparable objects, and
        correlating them would silently measure rank reversal as method
        disagreement.

    Args:
        left: First ranking, with ``zone_id`` and ``left_rank``.
        right: Second ranking, with ``zone_id`` and ``right_rank``.
        params: Parsed params mapping.
        left_name: Label for the first method.
        right_name: Label for the second.
        left_rank: Rank column in ``left``.
        right_rank: Rank column in ``right``.

    Returns:
        Mapping with ``spearman_rho``, ``spearman_p``, optionally
        ``kendall_tau`` and ``kendall_p``, ``n``, ``top_n``,
        ``top_n_overlap``, ``top_n_jaccard``, ``mean_abs_shift``,
        ``median_abs_shift``, ``max_abs_shift`` and the method labels.

    Raises:
        ValueError: If a rank column is absent or the zone sets differ.
    """
    import numpy as np  # Deferred: see module docstring.
    from scipy import stats

    for frame, column, label in (
        (left, left_rank, left_name),
        (right, right_rank, right_name),
    ):
        if column not in frame.columns:
            raise ValueError(
                f"{label} ranking has no {column!r} column; it has "
                f"{sorted(frame.columns)}"
            )

    left_ids = set(left["zone_id"].astype(str))
    right_ids = set(right["zone_id"].astype(str))
    if left_ids != right_ids:
        only_left = sorted(left_ids - right_ids)[:5]
        only_right = sorted(right_ids - left_ids)[:5]
        raise ValueError(
            f"{left_name} ranks {len(left_ids)} zones and {right_name} ranks "
            f"{len(right_ids)}; only in the first: {only_left}, only in the "
            f"second: {only_right}. TOPSIS draws its ideal and anti-ideal from "
            "the alternative SET, so a ranking must be RE-RUN on the retained "
            "zones rather than sub-selected - otherwise this comparison "
            "measures rank reversal and calls it method disagreement."
        )

    # Both frames legitimately carry the SAME rank column when the comparison is
    # between two runs of one method - percentile rank against min-max, or the
    # pooled series against the single-sensor one. A plain merge would suffix
    # them to `rank_ahp_x`/`rank_ahp_y` and every lookup below would raise, so
    # they are renamed to fixed internal names first.
    merged = left[["zone_id", left_rank]].rename(columns={left_rank: "_left"}).merge(
        right[["zone_id", right_rank]].rename(columns={right_rank: "_right"}),
        on="zone_id",
        how="inner",
    )
    rho, p_value = spearman_rho(merged["_left"], merged["_right"])

    config = params["greening"]["comparison"]
    report: dict[str, Any] = {
        "left_name": left_name,
        "right_name": right_name,
        "n": int(len(merged)),
        "spearman_rho": rho,
        "spearman_p": p_value,
    }

    if bool(config.get("also_kendall", False)):
        tau = stats.kendalltau(merged["_left"], merged["_right"])
        report["kendall_tau"] = float(tau.statistic)
        report["kendall_p"] = float(tau.pvalue)

    if bool(config.get("top_n_overlap", False)):
        limit = int(params["greening"]["top_n"])
        top_left = set(merged.nsmallest(limit, "_left")["zone_id"].astype(str))
        top_right = set(merged.nsmallest(limit, "_right")["zone_id"].astype(str))
        union = top_left | top_right
        report["top_n"] = limit
        report["top_n_overlap"] = int(len(top_left & top_right))
        report["top_n_jaccard"] = (
            float(len(top_left & top_right) / len(union)) if union else float("nan")
        )

    shift = (merged["_right"] - merged["_left"]).to_numpy(dtype=float)
    report["mean_abs_shift"] = float(np.abs(shift).mean())
    report["median_abs_shift"] = float(np.median(np.abs(shift)))
    report["max_abs_shift"] = float(np.abs(shift).max())
    return report


def rank_shift_frame(
    left: "pd.DataFrame",
    right: "pd.DataFrame",
    params: dict[str, Any],
    left_rank: str = "rank_ahp",
    right_rank: str = "rank_topsis",
    top_n: int | None = None,
) -> "pd.DataFrame":
    """Tabulate the zones the two methods disagree about most.

    Args:
        left: First ranking.
        right: Second ranking.
        params: Parsed params mapping.
        left_rank: Rank column in ``left``.
        right_rank: Rank column in ``right``.
        top_n: How many movers to return; defaults to
            ``greening.comparison.max_rank_shift_rows``.

    Returns:
        ``pandas.DataFrame`` with ``zone_id``, both ranks, ``rank_shift`` and
        ``abs_shift``, sorted by descending absolute shift.
    """
    limit = int(
        params["greening"]["comparison"]["max_rank_shift_rows"]
        if top_n is None
        else top_n
    )
    # Distinct output names even when both inputs use the same rank column; see
    # the note in `compare_rankings`.
    left_name = left_rank
    right_name = right_rank if right_rank != left_rank else f"{right_rank}_right"
    merged = left[["zone_id", left_rank]].merge(
        right[["zone_id", right_rank]].rename(columns={right_rank: right_name}),
        on="zone_id",
        how="inner",
    )
    merged["rank_shift"] = merged[right_name] - merged[left_name]
    merged["abs_shift"] = merged["rank_shift"].abs()
    return (
        merged.sort_values("abs_shift", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )


def criterion_correlation(
    prepared: "pd.DataFrame",
    params: dict[str, Any],
    criteria: Sequence[str] | None = None,
    method: str = "spearman",
) -> "pd.DataFrame":
    """Correlation matrix of the normalised criteria.

    This is the figure that makes the collinearity visible. Measured over the
    committed Phase 5 outputs across all 557 GN divisions, ``rho(LST, green
    fraction) = -0.9147``: "high LST" and "low vegetation" are very nearly one
    variable in this city, and a multi-criteria method over near-duplicate
    criteria gives the same latent factor its weight several times.

    Args:
        prepared: The frame :func:`prepare_criteria` returned.
        params: Parsed params mapping.
        criteria: Optional criterion subset.
        method: Correlation method passed to ``DataFrame.corr``.

    Returns:
        Square ``pandas.DataFrame`` indexed and columned by criterion name.

    Raises:
        ValueError: If a normalised column is absent.
    """
    names = [str(entry["name"]) for entry in resolve_criteria(params, criteria)]
    columns = [f"{name}_norm" for name in names]
    absent = [column for column in columns if column not in prepared.columns]
    if absent:
        raise ValueError(f"{absent} not in the frame; pass prepare_criteria output")

    matrix = prepared[columns].corr(method=method)
    matrix.index = names
    matrix.columns = names
    matrix.attrs["method"] = method
    return matrix


def effective_dimensionality(matrix: "pd.DataFrame") -> dict[str, Any]:
    """How many independent dimensions a criterion correlation matrix really has.

    Args:
        matrix: The square matrix :func:`criterion_correlation` returned.

    Returns:
        Mapping with ``pc1_variance_share``, ``n_effective`` (the exponential of
        the eigenvalue entropy), ``n_criteria`` and ``eigenvalues``. When
        ``pc1_variance_share`` is high the MCDA is effectively one-dimensional
        and the weights are largely decorative, whatever their consistency ratio.

    Raises:
        ValueError: If the matrix is not square.
    """
    import numpy as np  # Deferred: see module docstring.

    values = np.asarray(matrix.to_numpy(), dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"expected a square correlation matrix; got {values.shape}")

    size = values.shape[0]
    # Symmetric by construction, so eigvalsh rather than eigvals: real, ordered,
    # and no complex part to discard.
    eigenvalues = np.sort(np.linalg.eigvalsh(np.nan_to_num(values, nan=0.0)))[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 0:
        return {
            "pc1_variance_share": float("nan"),
            "n_effective": float("nan"),
            "n_criteria": size,
            "eigenvalues": eigenvalues.tolist(),
        }

    shares = eigenvalues / total
    positive = shares[shares > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    return {
        "pc1_variance_share": float(shares[0]),
        "n_effective": float(np.exp(entropy)),
        "n_criteria": size,
        "eigenvalues": eigenvalues.tolist(),
    }


def criterion_ablation(
    prepared: "pd.DataFrame",
    params: dict[str, Any],
    weights: Mapping[str, float],
    criteria: Sequence[str] | None = None,
) -> "pd.DataFrame":
    """Drop each criterion in turn and measure how far the ranking moves.

    **This is the most valuable diagnostic in the phase.** The criteria over
    Colombo are near-collinear, so a five-criterion MCDA can reproduce a ranking
    by land surface temperature alone while wearing the authority of a
    multi-criteria method. The single-criterion baseline row is what detects
    that, and the notebook prints the verdict whether or not it flatters the
    method.

    Args:
        prepared: The frame :func:`prepare_criteria` returned.
        params: Parsed params mapping.
        weights: The full criterion weights.
        criteria: Optional criterion subset.

    Returns:
        ``pandas.DataFrame``, one row per variant, with ``variant``,
        ``dropped``, ``n_criteria``, ``spearman_rho`` against the full ranking,
        ``top_n_overlap`` and ``max_abs_shift``. The last row is
        ``single_criterion``: a ranking on ``ablation.baseline_single_criterion``
        alone.

    Raises:
        ValueError: If a weight is missing.
    """
    import pandas as pd  # Deferred: see module docstring.

    names = [str(entry["name"]) for entry in resolve_criteria(params, criteria)]
    missing = [name for name in names if name not in weights]
    if missing:
        raise ValueError(f"no weight supplied for criteria {missing}")

    limit = int(params["greening"]["top_n"])
    full = rank_frame(mcda_scores(prepared, params, weights, names), params)
    full_top = set(full.loc[full["priority"], "zone_id"].astype(str))

    def _row(variant: str, dropped: str, subset: Sequence[str]) -> dict[str, Any]:
        subset_weights = {name: float(weights[name]) for name in subset}
        ranking = rank_frame(
            mcda_scores(prepared, params, subset_weights, list(subset)), params
        )
        merged = full[["zone_id", "rank_ahp"]].merge(
            ranking[["zone_id", "rank_ahp"]], on="zone_id", suffixes=("_full", "_var")
        )
        rho, _ = spearman_rho(merged["rank_ahp_full"], merged["rank_ahp_var"])
        variant_top = set(ranking.loc[ranking["priority"], "zone_id"].astype(str))
        shift = (merged["rank_ahp_var"] - merged["rank_ahp_full"]).abs()
        return {
            "variant": variant,
            "dropped": dropped,
            "n_criteria": len(subset),
            "weight_dropped": float(
                sum(float(weights[name]) for name in names if name not in subset)
            ),
            "spearman_rho": rho,
            "top_n_overlap": int(len(full_top & variant_top)),
            "top_n": limit,
            "max_abs_shift": float(shift.max()),
        }

    rows = [
        {
            "variant": "full",
            "dropped": "",
            "n_criteria": len(names),
            "weight_dropped": 0.0,
            "spearman_rho": 1.0,
            "top_n_overlap": int(len(full_top)),
            "top_n": limit,
            "max_abs_shift": 0.0,
        }
    ]
    for name in names:
        subset = [other for other in names if other != name]
        if subset:
            rows.append(_row(f"without_{name}", name, subset))

    baseline = str(params["greening"]["ablation"]["baseline_single_criterion"])
    if baseline in names:
        rows.append(
            _row(
                "single_criterion",
                ", ".join(name for name in names if name != baseline),
                [baseline],
            )
        )

    frame = pd.DataFrame(rows)
    threshold = float(params["greening"]["ablation"]["warn_rho_above"])
    single = frame.loc[frame["variant"] == "single_criterion", "spearman_rho"]
    frame.attrs["warn_rho_above"] = threshold
    frame.attrs["baseline_single_criterion"] = baseline
    frame.attrs["single_criterion_rho"] = (
        float(single.iloc[0]) if len(single) else float("nan")
    )
    frame.attrs["reproduces_single_criterion"] = bool(
        len(single) and float(single.iloc[0]) > threshold
    )
    return frame


def circularity_report(
    ranked: "pd.DataFrame",
    interim: "pd.DataFrame",
    params: dict[str, Any],
    rank_column: str = "rank_ahp",
) -> dict[str, Any]:
    """Measure agreement with the Phase 5 interim proxy, and label it honestly.

    .. warning::
        **Agreement here is not validation.** Phase 5's proxy ranks on
        ``gi_star_hot`` / ``lst_2020s`` / ``ndvi_inverse``; three of Phase 7's
        five criteria overlap it, and ``rho(interim score, LST)`` is already
        +0.9829 over this district. The two rankings will agree because they are
        largely the same inputs reweighted. ``independence`` is therefore always
        :data:`NOT_INDEPENDENT`; there is no value of that field meaning
        otherwise, because the overlap is structural rather than empirical.

    Args:
        ranked: The Phase 7 ranking.
        interim: The Phase 5 interim table, with ``zone_id`` and ``rank``.
        params: Parsed params mapping.
        rank_column: The Phase 7 rank column.

    Returns:
        Mapping with ``spearman_rho``, ``spearman_p``, ``n``,
        ``top_n_overlap``, ``shared_criteria``, ``independence`` and
        ``interpretation``.

    Raises:
        ValueError: If a rank column is absent or no zones join.
    """
    if rank_column not in ranked.columns:
        raise ValueError(f"the ranking has no {rank_column!r} column")
    interim_rank = "rank" if "rank" in interim.columns else None
    if interim_rank is None:
        raise ValueError(
            f"the interim table has no 'rank' column; it has {sorted(interim.columns)}"
        )

    left = ranked[["zone_id", rank_column]].copy()
    left["zone_id"] = left["zone_id"].astype(str)
    right = interim[["zone_id", interim_rank]].copy()
    right["zone_id"] = right["zone_id"].astype(str)
    merged = left.merge(right, on="zone_id", how="inner", suffixes=("", "_interim"))
    if merged.empty:
        raise ValueError(
            "no zones joined between the Phase 7 ranking and the Phase 5 interim "
            "table; check that both are keyed on the same pcode"
        )

    rho, p_value = spearman_rho(merged[rank_column], merged[interim_rank])
    limit = int(params["greening"]["top_n"])
    overlap = int(
        len(
            set(merged.nsmallest(limit, rank_column)["zone_id"])
            & set(merged.nsmallest(limit, interim_rank)["zone_id"])
        )
    )

    phase5 = [
        str(name) for name in params["prediction"]["priority_zones"]["rank_by"]
    ]
    shared = ["lst (Gi* z and the zone mean)", "ndvi (inverted)"]
    return {
        "spearman_rho": rho,
        "spearman_p": p_value,
        "n": int(len(merged)),
        "top_n": limit,
        "top_n_overlap": overlap,
        "phase5_criteria": phase5,
        "shared_criteria": shared,
        "independence": NOT_INDEPENDENT,
        "interpretation": (
            "AGREEMENT HERE IS NOT VALIDATION. Phase 5's interim proxy ranks on "
            f"{phase5}, which overlaps Phase 7's criteria in {shared}. The two "
            "rankings agree because they are largely the same inputs reweighted, "
            "not because one confirms the other. Phase 6's greening "
            "counterfactual was also computed INSIDE the Phase 5 zones, so "
            "quoting it as evidence for these zones would close the loop "
            "entirely."
        ),
    }


# =============================================================================
# Group G - the 3-30-300 rule
# =============================================================================
# Konijnendijk's rule: 3 trees visible from every home, 30 % canopy in every
# neighbourhood, 300 m to the nearest public green space >= 0.5 ha.
#
# Earth Engine supplies the source rasters and nothing else. Patch labelling, the
# >= 0.5 ha filter, the service areas and every zonal statistic are pure Python,
# so all of it is unit-testable with no Earth Engine session. An Earth Engine
# route via fastDistanceTransform exists and is deliberately not used: the
# spatial_stats.covariates.dist_coast params comment records what an inherited
# projection cost there - a 160 MB request and distances wrong by 3.3x. At 10 m,
# 300 m is 30 pixels, and the numpy EDT is exact.


def green_patches(
    green: "np.ndarray",
    observed: "np.ndarray",
    cell_size_m: float,
    params: dict[str, Any],
    connectivity: int | None = None,
) -> tuple["np.ndarray", "pd.DataFrame"]:
    """Label contiguous green patches and measure each one's area.

    Reuses :func:`colombo_uhi.spatial_stats.patch_labels`, which Phase 5 already
    tested, rather than growing a second connected-components implementation
    that could disagree with the landscape metrics.

    Args:
        green: Boolean array, ``True`` where the cell is green.
        observed: Boolean array, ``True`` where the classifier saw the cell.
        cell_size_m: Cell size in metres.
        params: Parsed params mapping.
        connectivity: 4 or 8; defaults to
            ``greening.rule_3_30_300.green_space.connectivity``.

    Returns:
        ``(labels, patches)`` - the labelled array and a frame with ``patch``,
        ``n_cells`` and ``area_ha``, sorted by descending area.

    Raises:
        ValueError: If the arrays disagree in shape or the cell size is not
            positive.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd

    from colombo_uhi import spatial_stats

    mask = np.asarray(green, dtype=bool)
    seen = np.asarray(observed, dtype=bool)
    if mask.shape != seen.shape:
        raise ValueError(
            f"green {mask.shape} and observed {seen.shape} must be the same shape"
        )
    if float(cell_size_m) <= 0:
        raise ValueError(f"cell_size_m must be positive; got {cell_size_m}")

    scheme = int(
        params["greening"]["rule_3_30_300"]["green_space"]["connectivity"]
        if connectivity is None
        else connectivity
    )
    # A cell the classifier never saw is not green. Phase 5's 5.4x "green growth"
    # artefact is what happens when unobserved is allowed to mean anything.
    labels, count = spatial_stats.patch_labels(mask & seen, connectivity=scheme)

    cell_ha = (float(cell_size_m) ** 2) / 10_000.0
    rows = []
    if count:
        sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
        rows = [
            {"patch": index + 1, "n_cells": int(size), "area_ha": float(size) * cell_ha}
            for index, size in enumerate(sizes)
        ]

    patches = pd.DataFrame(rows, columns=["patch", "n_cells", "area_ha"])
    patches = patches.sort_values("area_ha", ascending=False).reset_index(drop=True)
    patches.attrs["connectivity"] = scheme
    patches.attrs["cell_size_m"] = float(cell_size_m)
    patches.attrs["n_patches"] = int(count)
    return labels, patches


def qualifying_green_mask(
    green: "np.ndarray",
    observed: "np.ndarray",
    cell_size_m: float,
    params: dict[str, Any],
    min_patch_ha: float | None = None,
    connectivity: int | None = None,
) -> "np.ndarray":
    """Green patches large enough to count under the 300 rule.

    The area threshold is **inclusive**: at 10 m, 50 cells is exactly 0.5 ha and
    qualifies. An exclusive comparison would silently move the rule's own number.

    Args:
        green: Boolean green mask.
        observed: Boolean coverage mask.
        cell_size_m: Cell size in metres.
        params: Parsed params mapping.
        min_patch_ha: Override for ``green_space.min_patch_ha``.
        connectivity: Override for ``green_space.connectivity``.

    Returns:
        Boolean array, ``True`` in cells belonging to a qualifying patch.
    """
    import numpy as np  # Deferred: see module docstring.

    threshold = float(
        params["greening"]["rule_3_30_300"]["green_space"]["min_patch_ha"]
        if min_patch_ha is None
        else min_patch_ha
    )
    labels, patches = green_patches(
        green, observed, cell_size_m, params, connectivity=connectivity
    )
    if patches.empty:
        return np.zeros(labels.shape, dtype=bool)

    # Inclusive, with a tolerance so 50 cells at 10 m is not excluded by the
    # floating-point representation of 0.5.
    keep = patches.loc[patches["area_ha"] >= threshold - 1e-9, "patch"].to_numpy()
    if keep.size == 0:
        return np.zeros(labels.shape, dtype=bool)

    lookup = np.zeros(int(labels.max()) + 1, dtype=bool)
    lookup[keep.astype(int)] = True
    return lookup[labels]


def service_area_mask(
    qualifying: "np.ndarray",
    cell_size_m: float,
    params: dict[str, Any],
    distance_m: float | None = None,
) -> "np.ndarray":
    """Cells within the service distance of a qualifying green space.

    .. warning::
        **This is straight-line distance, and the rule is about walking.**
        Colombo puts the Kelani River, Beira Lake, the coastal railway and long
        walled compounds between residents and parks, so a Euclidean service area
        OVERSTATES access everywhere by an unknown amount. Every headline result
        is reported beside a ``detour_ratio``-shrunk variant, and the gap between
        the two columns is the size of the problem. See
        ``caveats.euclidean_not_network``.

    Args:
        qualifying: Boolean mask of qualifying green cells.
        cell_size_m: Cell size in metres.
        params: Parsed params mapping.
        distance_m: Override for ``green_space.service_distance_m``.

    Returns:
        Boolean array, ``True`` where a qualifying green space is within the
        distance. Qualifying cells are themselves served.

    Raises:
        ValueError: If the cell size or distance is not positive.
    """
    import numpy as np  # Deferred: see module docstring.
    from scipy import ndimage

    mask = np.asarray(qualifying, dtype=bool)
    size = float(cell_size_m)
    radius = float(
        params["greening"]["rule_3_30_300"]["green_space"]["service_distance_m"]
        if distance_m is None
        else distance_m
    )
    if size <= 0:
        raise ValueError(f"cell_size_m must be positive; got {size}")
    if radius <= 0:
        raise ValueError(f"the service distance must be positive; got {radius}")
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)

    # distance_transform_edt measures distance to the nearest ZERO, so the mask
    # is inverted: distance from each cell to the nearest qualifying green cell.
    distance = ndimage.distance_transform_edt(~mask, sampling=(size, size))
    # Inclusive at the boundary, and tolerant of the floating-point
    # representation of an exact multiple of the cell size.
    return np.asarray(distance) <= radius + 1e-9


def detour_distance_m(params: dict[str, Any], distance_m: float | None = None) -> float:
    """The straight-line radius that a walking distance actually buys.

    Args:
        params: Parsed params mapping.
        distance_m: Override for ``green_space.service_distance_m``.

    Returns:
        ``service_distance_m / detour_ratio``. With the shipped 300 m and 1.3
        this is 231 m.
    """
    config = params["greening"]["rule_3_30_300"]["green_space"]
    radius = float(config["service_distance_m"] if distance_m is None else distance_m)
    return radius / float(config["detour_ratio"])


def require_integer_refinement(
    fine_shape: tuple[int, ...], coarse_shape: tuple[int, ...], factor: int
) -> None:
    """Refuse to block-average grids that are not an exact integer refinement.

    Without this, :func:`block_mean` would silently trim the remainder and
    misregister the 10 m service mask against the 100 m population grid by up to
    90 m - which is a third of the 300 m the rule is about.

    Args:
        fine_shape: Shape of the fine grid.
        coarse_shape: Shape of the coarse grid.
        factor: Expected refinement factor.

    Raises:
        ValueError: If the factor is not positive, or the fine grid is not
            exactly ``factor`` times the coarse one in both dimensions.
    """
    if int(factor) < 1:
        raise ValueError(f"the refinement factor must be >= 1; got {factor}")
    expected = tuple(int(value) * int(factor) for value in coarse_shape[:2])
    if tuple(int(value) for value in fine_shape[:2]) != expected:
        raise ValueError(
            f"the fine grid is {tuple(fine_shape[:2])} but a {factor}x refinement "
            f"of the coarse grid {tuple(coarse_shape[:2])} would be {expected}. "
            "Block-averaging these would misregister the service mask against "
            "the population grid by up to one coarse cell - a third of the 300 m "
            "the rule is about. Re-export both on the same grid."
        )


def _grid_geometry(
    profile: Mapping[str, Any], label: str
) -> tuple[float, float, float, float, int, int]:
    """Origin, pixel size and shape of a north-up, axis-aligned raster profile.

    Args:
        profile: A rasterio profile.
        label: Name used in error messages.

    Returns:
        ``(x0, y0, pixel_x, pixel_y, height, width)``, where ``(x0, y0)`` is the
        upper-left corner in world coordinates and both pixel sizes are positive.

    Raises:
        ValueError: If the profile is incomplete, or the transform is rotated or
            sheared - in which case a row/column crop is not a rectangle on the
            ground and cropping would silently misplace the data.
    """
    for key in ("transform", "height", "width"):
        if key not in profile or profile[key] is None:
            raise ValueError(
                f"the {label} profile has no usable {key!r}. Grid alignment works "
                "in WORLD coordinates, not on shapes, so it needs the affine "
                "transform - shapes alone cannot tell a 130 m overhang from a "
                "130 m offset."
            )

    values = [float(value) for value in tuple(profile["transform"])[:6]]
    pixel_x, row_rotation, x0, col_rotation, pixel_y, y0 = values

    if abs(row_rotation) > 1e-9 or abs(col_rotation) > 1e-9:
        raise ValueError(
            f"the {label} grid is rotated or sheared (rotation terms "
            f"{row_rotation}, {col_rotation}). A row/column crop of a rotated "
            "raster is not a rectangle on the ground, so alignment refuses "
            "rather than silently misplacing the data."
        )
    if pixel_x <= 0 or pixel_y >= 0:
        raise ValueError(
            f"the {label} grid is not north-up (pixel size {pixel_x}, {pixel_y}); "
            "alignment assumes x increases with column and y decreases with row"
        )
    return x0, y0, pixel_x, -pixel_y, int(profile["height"]), int(profile["width"])


def crop_to_window(
    array: "np.ndarray", window: tuple[int, int, int, int]
) -> "np.ndarray":
    """Crop an array to ``(row_off, col_off, height, width)``.

    Args:
        array: The array to crop; trailing dimensions are preserved.
        window: Offsets and size, as :func:`align_fine_to_coarse` returns.

    Returns:
        The cropped view.

    Raises:
        ValueError: If the window is not positive or runs off the array.
    """
    import numpy as np  # Deferred: see module docstring.

    values = np.asarray(array)
    row_off, col_off, height, width = (int(value) for value in window)
    if row_off < 0 or col_off < 0 or height <= 0 or width <= 0:
        raise ValueError(f"window {tuple(window)} is not a positive region")
    if row_off + height > values.shape[0] or col_off + width > values.shape[1]:
        raise ValueError(
            f"window {tuple(window)} runs off an array of shape {values.shape[:2]}"
        )
    return values[row_off : row_off + height, col_off : col_off + width]


def align_fine_to_coarse(
    fine_profile: Mapping[str, Any],
    coarse_profile: Mapping[str, Any],
    factor: int,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Crop two grids to a common extent on which the fine one nests in the coarse.

    .. warning::
        **Earth Engine snaps each export grid to its own scale, independently per
        task.** A 10 m raster and a 100 m raster exported over the *same* region
        therefore need not nest. Colab run 2 measured it over Colombo: a
        ``2957 x 4219`` green/canopy grid at 10 m against a ``297 x 423``
        population grid at 100 m, the coarse one overhanging by 130 m in Y and
        110 m in X - one coarse cell per edge, 1.14 % of cells.

    The work is done in **world coordinates**, from each profile's affine
    transform, rather than from shapes. That is what makes it correct when the
    two *origins* differ rather than merely the extents: Earth Engine snaps each
    grid to a multiple of its own scale, so a 100 m origin and a 10 m origin can
    sit up to 90 m apart, and no comparison of shapes can see that.

    The common extent is snapped **inward to whole coarse cells**, so the cropped
    grids nest by construction and :func:`require_integer_refinement` passes as a
    post-condition rather than being bypassed.

    Args:
        fine_profile: Rasterio profile of the fine grid.
        coarse_profile: Rasterio profile of the coarse grid.
        factor: Expected refinement, e.g. 10 for 10 m against 100 m.
        params: Parsed params mapping, for
            ``greening.grid_alignment.max_dropped_fraction``.

    Returns:
        Mapping with ``fine_window`` and ``coarse_window`` as
        ``(row_off, col_off, height, width)``, the cropped ``fine_profile`` and
        ``coarse_profile``, and ``dropped_coarse_cells``, ``dropped_fraction``,
        ``dropped_area_km2``, ``factor`` and ``origin_offset_m``.

    Raises:
        ValueError: If a profile is unusable, the coarse pixel size is not
            exactly ``factor`` times the fine one, the grids do not overlap, the
            coarse origin does not lie on the fine grid, or the trim would
            discard more than ``max_dropped_fraction`` of the coarse cells -
            which would make this a rescue of two rasters describing different
            places rather than a boundary trim.
    """
    step = int(factor)
    if step < 1:
        raise ValueError(f"the refinement factor must be >= 1; got {factor}")

    fine_x0, fine_y0, fine_px, fine_py, fine_h, fine_w = _grid_geometry(
        fine_profile, "fine"
    )
    coarse_x0, coarse_y0, coarse_px, coarse_py, coarse_h, coarse_w = _grid_geometry(
        coarse_profile, "coarse"
    )

    tolerance = 1e-6
    for axis, fine_size, coarse_size in (
        ("x", fine_px, coarse_px),
        ("y", fine_py, coarse_py),
    ):
        if abs(coarse_size - fine_size * step) > tolerance:
            raise ValueError(
                f"the coarse {axis} pixel is {coarse_size} m but {step}x the fine "
                f"{axis} pixel ({fine_size} m) is {fine_size * step} m. These are "
                "not a fine and a coarse view of one grid, and no crop makes them "
                "nest."
            )

    # World extents. Origin is upper-left, so y decreases with row.
    fine_right = fine_x0 + fine_w * fine_px
    fine_bottom = fine_y0 - fine_h * fine_py
    coarse_right = coarse_x0 + coarse_w * coarse_px
    coarse_bottom = coarse_y0 - coarse_h * coarse_py

    left, right = max(fine_x0, coarse_x0), min(fine_right, coarse_right)
    top, bottom = min(fine_y0, coarse_y0), max(fine_bottom, coarse_bottom)
    if right - left < coarse_px or top - bottom < coarse_py:
        raise ValueError(
            f"the two grids overlap by {max(right - left, 0.0):.1f} x "
            f"{max(top - bottom, 0.0):.1f} m, less than one coarse cell "
            f"({coarse_px} x {coarse_py} m). They do not describe the same place."
        )

    # Snap the overlap inward to whole COARSE cells, measured from the coarse
    # origin, so the crop lands on real coarse-cell boundaries.
    col_start = math.ceil((left - coarse_x0) / coarse_px - tolerance)
    row_start = math.ceil((coarse_y0 - top) / coarse_py - tolerance)
    col_stop = math.floor((right - coarse_x0) / coarse_px + tolerance)
    row_stop = math.floor((coarse_y0 - bottom) / coarse_py + tolerance)

    coarse_cols, coarse_rows = col_stop - col_start, row_stop - row_start
    if coarse_cols < 1 or coarse_rows < 1:
        raise ValueError(
            "the common extent holds no whole coarse cell once snapped to the "
            f"coarse grid ({coarse_rows} x {coarse_cols})"
        )

    # The same corner, expressed on the fine grid. The coarse origin must land on
    # a fine cell boundary or no integer crop can make the two nest.
    crop_x = coarse_x0 + col_start * coarse_px
    crop_y = coarse_y0 - row_start * coarse_py
    fine_col = (crop_x - fine_x0) / fine_px
    fine_row = (fine_y0 - crop_y) / fine_py
    if abs(fine_col - round(fine_col)) > 1e-6 or abs(fine_row - round(fine_row)) > 1e-6:
        raise ValueError(
            "the coarse grid origin is offset from the fine grid by "
            f"{(fine_col - round(fine_col)) * fine_px:.3f} x "
            f"{(fine_row - round(fine_row)) * fine_py:.3f} m, which is not a whole "
            "number of fine cells. No integer crop makes these nest; re-export "
            "both with an explicit crsTransform."
        )

    fine_window = (
        int(round(fine_row)),
        int(round(fine_col)),
        int(coarse_rows * step),
        int(coarse_cols * step),
    )
    coarse_window = (int(row_start), int(col_start), int(coarse_rows), int(coarse_cols))

    total = coarse_h * coarse_w
    kept = coarse_rows * coarse_cols
    dropped_fraction = float((total - kept) / total) if total else 1.0

    ceiling = (
        float(params["greening"]["grid_alignment"]["max_dropped_fraction"])
        if params is not None
        else 1.0
    )
    if dropped_fraction > ceiling:
        raise ValueError(
            f"aligning these grids would discard {dropped_fraction:.1%} of the "
            f"coarse cells, above the {ceiling:.0%} ceiling in "
            "greening.grid_alignment.max_dropped_fraction. A trim of about one "
            "cell per edge is Earth Engine snapping each scale to its own grid; "
            "a loss this large means the two rasters describe different places, "
            "and cropping would hide that rather than fix it."
        )

    def _cropped(
        profile: Mapping[str, Any],
        x: float,
        y: float,
        rows: int,
        cols: int,
        px: float,
        py: float,
    ) -> dict[str, Any]:
        out = dict(profile)
        out["height"], out["width"] = int(rows), int(cols)
        transform = profile["transform"]
        # Rebuild through the transform's own class where it has one (rasterio's
        # Affine), so the cropped profile stays a drop-in for the original.
        try:
            out["transform"] = type(transform)(px, 0.0, x, 0.0, -py, y)
        except Exception:  # pragma: no cover - a plain tuple in a test fixture
            out["transform"] = (px, 0.0, x, 0.0, -py, y)
        return out

    return {
        "fine_window": fine_window,
        "coarse_window": coarse_window,
        "fine_profile": _cropped(
            fine_profile, crop_x, crop_y, fine_window[2], fine_window[3], fine_px, fine_py
        ),
        "coarse_profile": _cropped(
            coarse_profile, crop_x, crop_y, coarse_rows, coarse_cols, coarse_px, coarse_py
        ),
        "factor": step,
        "dropped_coarse_cells": int(total - kept),
        "dropped_fraction": dropped_fraction,
        "dropped_area_km2": float((total - kept) * coarse_px * coarse_py / 1e6),
        "origin_offset_m": (float(coarse_x0 - fine_x0), float(coarse_y0 - fine_y0)),
    }


def block_mean(array: "np.ndarray", factor: int) -> "np.ndarray":
    """Exact block mean, aggregating a fine grid onto a coarse one.

    Args:
        array: Fine-grid array; booleans are averaged as 0/1 shares.
        factor: Block size. The array's dimensions must be exact multiples.

    Returns:
        The block-averaged array.

    Raises:
        ValueError: If the factor is not positive or does not divide the shape.
    """
    import numpy as np  # Deferred: see module docstring.

    values = np.asarray(array, dtype=float)
    step = int(factor)
    if step < 1:
        raise ValueError(f"the block factor must be >= 1; got {factor}")
    rows, cols = values.shape[:2]
    if rows % step or cols % step:
        raise ValueError(
            f"a {rows}x{cols} grid does not divide into {step}x{step} blocks; "
            "trimming the remainder would shift every downstream zonal statistic"
        )
    return values.reshape(rows // step, step, cols // step, step).mean(axis=(1, 3))


def _zone_sums(
    zones: "np.ndarray", values: "np.ndarray", weights: "np.ndarray | None" = None
) -> dict[int, tuple[float, float]]:
    """Sum ``values`` (optionally weighted) and the weight itself, per zone code."""
    import numpy as np  # Deferred: see module docstring.

    codes = np.asarray(zones)
    flat = codes.ravel()
    valid = flat > 0
    if not valid.any():
        return {}

    weight = (
        np.ones(flat.shape, dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float).ravel()
    )
    numerator = np.asarray(values, dtype=float).ravel() * weight
    length = int(flat.max()) + 1
    total = np.bincount(flat[valid], weights=numerator[valid], minlength=length)
    denominator = np.bincount(flat[valid], weights=weight[valid], minlength=length)
    return {
        code: (float(total[code]), float(denominator[code]))
        for code in range(1, length)
        if denominator[code] > 0 or total[code] > 0
    }


def canopy_fraction_by_zone(
    canopy: "np.ndarray",
    observed: "np.ndarray",
    zones: "np.ndarray",
    zone_labels: Mapping[int, Any],
    params: dict[str, Any],
) -> "pd.DataFrame":
    """Tree-class share per zone, over OBSERVED LAND rather than the polygon.

    The denominator matters more than the numerator here. Using the polygon would
    make Fort read as treeless, because 82 % of its COD-AB polygon is the Colombo
    Port outer harbour rather than land anyone could plant on.

    .. note::
        This is the **tree-class share of a 10 m modal classification**, not crown
        cover from a canopy-height model: it counts a 10 m cell as fully canopy or
        not at all. Quote it as "Dynamic World tree-class share", never as
        "canopy cover".

    Args:
        canopy: Boolean canopy mask.
        observed: Boolean coverage mask.
        zones: Integer zone-code raster; 0 is outside every zone.
        zone_labels: Zone code to ``zone_id``.
        params: Parsed params mapping.

    Returns:
        ``pandas.DataFrame`` with ``zone_id``, ``canopy_pct``,
        ``observed_cells``, ``canopy_cells`` and ``rule_30_pass``.

    Raises:
        ValueError: If the arrays disagree in shape.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd

    tree = np.asarray(canopy, dtype=bool)
    seen = np.asarray(observed, dtype=bool)
    codes = np.asarray(zones)
    if not (tree.shape == seen.shape == codes.shape):
        raise ValueError(
            f"canopy {tree.shape}, observed {seen.shape} and zones {codes.shape} "
            "must be the same shape"
        )

    target = float(params["greening"]["rule_3_30_300"]["canopy"]["target_pct"])
    sums = _zone_sums(codes, (tree & seen).astype(float), seen.astype(float))

    rows = []
    for code, zone_id in zone_labels.items():
        numerator, denominator = sums.get(int(code), (0.0, 0.0))
        share = 100.0 * numerator / denominator if denominator > 0 else np.nan
        rows.append(
            {
                "zone_id": str(zone_id),
                "canopy_pct": share,
                "observed_cells": denominator,
                "canopy_cells": numerator,
                "rule_30_pass": bool(share >= target) if denominator > 0 else False,
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "zone_id",
            "canopy_pct",
            "observed_cells",
            "canopy_cells",
            "rule_30_pass",
        ],
    )
    frame.attrs["target_pct"] = target
    frame.attrs["denominator"] = "observed_land"
    return frame


def served_population_by_zone(
    service_fine: "np.ndarray",
    service_fine_detour: "np.ndarray",
    population_coarse: "np.ndarray",
    observed_coarse: "np.ndarray",
    zones_coarse: "np.ndarray",
    zone_labels: Mapping[int, Any],
    params: dict[str, Any],
    factor: int,
) -> "pd.DataFrame":
    """Share of each zone's residents within 300 m of a qualifying green space.

    The rule counts **residences**, so the headline is population-weighted. The
    area share is reported beside it (CLAUDE.md caveat 5), and the two differ
    wherever population is concentrated away from the green space - which is the
    whole reason the weighting exists.

    Args:
        service_fine: Boolean 10 m service mask at the full distance.
        service_fine_detour: Boolean 10 m service mask at the detour-shrunk
            distance.
        population_coarse: People per coarse cell.
        observed_coarse: Boolean coverage mask on the coarse grid.
        zones_coarse: Integer zone codes on the coarse grid.
        zone_labels: Zone code to ``zone_id``.
        params: Parsed params mapping.
        factor: Refinement factor from the coarse grid to the fine one.

    Returns:
        ``pandas.DataFrame`` with ``zone_id``, ``pop_within_300m_pct``,
        ``area_within_300m_pct``, ``pop_within_300m_detour_pct``,
        ``rule_300_pass``, ``rule_300_pass_detour`` and ``population``.

    Raises:
        ValueError: If the grids are not an exact integer refinement.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd

    require_integer_refinement(service_fine.shape, population_coarse.shape, factor)
    require_integer_refinement(
        service_fine_detour.shape, population_coarse.shape, factor
    )

    served = block_mean(np.asarray(service_fine, dtype=float), factor)
    served_detour = block_mean(np.asarray(service_fine_detour, dtype=float), factor)
    people = np.asarray(population_coarse, dtype=float)
    seen = np.asarray(observed_coarse, dtype=bool)
    codes = np.asarray(zones_coarse)

    people = np.where(seen & np.isfinite(people), people, 0.0)
    by_population = _zone_sums(codes, served, people)
    by_area = _zone_sums(codes, served, seen.astype(float))
    by_population_detour = _zone_sums(codes, served_detour, people)

    threshold = 50.0  # a zone "passes" when the majority of residents are served
    rows = []
    for code, zone_id in zone_labels.items():
        key = int(code)
        pop_num, pop_den = by_population.get(key, (0.0, 0.0))
        area_num, area_den = by_area.get(key, (0.0, 0.0))
        detour_num, detour_den = by_population_detour.get(key, (0.0, 0.0))

        pop_pct = 100.0 * pop_num / pop_den if pop_den > 0 else np.nan
        area_pct = 100.0 * area_num / area_den if area_den > 0 else np.nan
        detour_pct = 100.0 * detour_num / detour_den if detour_den > 0 else np.nan
        # With no residents at all, the population share is undefined; fall back
        # to the area share rather than reporting a zone as unserved.
        headline = pop_pct if pop_den > 0 else area_pct
        headline_detour = detour_pct if detour_den > 0 else area_pct

        rows.append(
            {
                "zone_id": str(zone_id),
                "pop_within_300m_pct": headline,
                "area_within_300m_pct": area_pct,
                "pop_within_300m_detour_pct": headline_detour,
                "population": pop_den,
                "rule_300_pass": bool(headline >= threshold)
                if np.isfinite(headline)
                else False,
                "rule_300_pass_detour": bool(headline_detour >= threshold)
                if np.isfinite(headline_detour)
                else False,
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "zone_id",
            "pop_within_300m_pct",
            "area_within_300m_pct",
            "pop_within_300m_detour_pct",
            "population",
            "rule_300_pass",
            "rule_300_pass_detour",
        ],
    )
    frame.attrs["service_distance_m"] = float(
        params["greening"]["rule_3_30_300"]["green_space"]["service_distance_m"]
    )
    frame.attrs["detour_distance_m"] = detour_distance_m(params)
    frame.attrs["pass_threshold_pct"] = threshold
    frame.attrs["weight_by"] = str(
        params["greening"]["rule_3_30_300"]["green_space"]["weight_by"]
    )
    frame.attrs["public_only"] = bool(
        params["greening"]["rule_3_30_300"]["green_space"]["public_only"]
    )
    return frame


def trees_in_view_proxy(
    canopy: "np.ndarray",
    built: "np.ndarray",
    zones: "np.ndarray",
    zone_labels: Mapping[int, Any],
    params: dict[str, Any],
) -> "pd.DataFrame":
    """A proxy for the "3 trees from every window" component - never a verdict.

    .. warning::
        **The "3" of 3-30-300 cannot be measured from space.** It needs window
        orientations and individual tree stems, and no free dataset carries
        either. Reporting it as passed, as failed, or dropping it silently would
        all be wrong. Every row carries ``rule_3_status =
        "not_remotely_sensable"``, ``greening.rule_3_30_300.trees_in_view
        .enters_score`` is ``False``, and a test pins that this column never
        enters the criterion set or the compliance verdict.

    Args:
        canopy: Boolean canopy mask.
        built: Boolean built-cell mask - the cells where people live.
        zones: Integer zone-code raster.
        zone_labels: Zone code to ``zone_id``.
        params: Parsed params mapping.

    Returns:
        ``pandas.DataFrame`` with ``zone_id``, ``rule_3_proxy_pct`` and
        ``rule_3_status``.

    Raises:
        ValueError: If the arrays disagree in shape.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd
    from scipy import ndimage

    tree = np.asarray(canopy, dtype=bool)
    homes = np.asarray(built, dtype=bool)
    codes = np.asarray(zones)
    if not (tree.shape == homes.shape == codes.shape):
        raise ValueError(
            f"canopy {tree.shape}, built {homes.shape} and zones {codes.shape} "
            "must be the same shape"
        )

    config = params["greening"]["rule_3_30_300"]["trees_in_view"]
    cell = float(params["greening"]["landcover_scale_m"])
    radius = float(config["proxy_radius_m"])
    minimum = int(config["proxy_min_tree_cells"])

    if tree.any():
        # Count canopy cells within the radius of each built cell, rather than
        # merely testing whether one exists, so proxy_min_tree_cells means what
        # it says.
        span = max(1, int(round(radius / cell)))
        offsets = np.arange(-span, span + 1) * cell
        grid_y, grid_x = np.meshgrid(offsets, offsets, indexing="ij")
        footprint = (grid_y**2 + grid_x**2) <= radius**2 + 1e-9
        neighbours = ndimage.convolve(
            tree.astype(np.int32), footprint.astype(np.int32), mode="constant", cval=0
        )
        in_view = homes & (neighbours >= minimum)
    else:
        in_view = np.zeros(tree.shape, dtype=bool)

    sums = _zone_sums(codes, in_view.astype(float), homes.astype(float))
    rows = []
    for code, zone_id in zone_labels.items():
        numerator, denominator = sums.get(int(code), (0.0, 0.0))
        rows.append(
            {
                "zone_id": str(zone_id),
                "rule_3_proxy_pct": (
                    100.0 * numerator / denominator if denominator > 0 else np.nan
                ),
                "rule_3_status": RULE_3_STATUS,
            }
        )

    frame = pd.DataFrame(
        rows, columns=["zone_id", "rule_3_proxy_pct", "rule_3_status"]
    )
    frame.attrs["proxy_radius_m"] = radius
    frame.attrs["proxy_min_tree_cells"] = minimum
    frame.attrs["enters_score"] = bool(config["enters_score"])
    return frame


def compliance_3_30_300(
    canopy_frame: "pd.DataFrame",
    service_frame: "pd.DataFrame",
    proxy_frame: "pd.DataFrame | None",
    params: dict[str, Any],
) -> "pd.DataFrame":
    """Join the three components into a five-category per-zone verdict.

    **Five categories, never a boolean.** The "3" is unmeasured, so a pass/fail
    flag would imply it had been checked. ``not_assessable`` is what a zone gets
    when neither component could be computed - which is a result, not a gap.

    Args:
        canopy_frame: Output of :func:`canopy_fraction_by_zone`.
        service_frame: Output of :func:`served_population_by_zone`.
        proxy_frame: Optional output of :func:`trees_in_view_proxy`.
        params: Parsed params mapping.

    Returns:
        ``pandas.DataFrame`` with :data:`COMPLIANCE_COLUMNS`.

    Raises:
        ValueError: If a required column is absent.
    """
    import numpy as np  # Deferred: see module docstring.

    for frame, name, needed in (
        (canopy_frame, "canopy", ("zone_id", "canopy_pct", "rule_30_pass")),
        (
            service_frame,
            "service",
            ("zone_id", "pop_within_300m_pct", "rule_300_pass"),
        ),
    ):
        missing = [column for column in needed if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} frame is missing {missing}")

    merged = canopy_frame.merge(service_frame, on="zone_id", how="outer")
    if proxy_frame is not None:
        merged = merged.merge(proxy_frame, on="zone_id", how="left")
    else:
        merged["rule_3_proxy_pct"] = np.nan
        merged["rule_3_status"] = RULE_3_STATUS

    canopy_known = merged["canopy_pct"].notna()
    access_known = merged["pop_within_300m_pct"].notna()
    canopy_pass = merged["rule_30_pass"].fillna(False).astype(bool)
    access_pass = merged["rule_300_pass"].fillna(False).astype(bool)

    verdict = np.full(len(merged), "not_assessable", dtype=object)
    assessable = (canopy_known & access_known).to_numpy()
    verdict = np.where(
        assessable & canopy_pass & access_pass, "both_30_and_300", verdict
    )
    verdict = np.where(assessable & canopy_pass & ~access_pass, "canopy_only", verdict)
    verdict = np.where(assessable & ~canopy_pass & access_pass, "access_only", verdict)
    verdict = np.where(assessable & ~canopy_pass & ~access_pass, "neither", verdict)
    merged["compliance"] = verdict
    merged["rule_3_status"] = merged["rule_3_status"].fillna(RULE_3_STATUS)

    for column in COMPLIANCE_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan
    out = merged[list(COMPLIANCE_COLUMNS)].copy()
    out["zone_id"] = out["zone_id"].astype(str)

    out.attrs["categories"] = list(COMPLIANCE_CATEGORIES)
    out.attrs["counts"] = {
        category: int((out["compliance"] == category).sum())
        for category in COMPLIANCE_CATEGORIES
    }
    out.attrs["rule_3_status"] = RULE_3_STATUS
    out.attrs["public_only"] = bool(
        params["greening"]["rule_3_30_300"]["green_space"]["public_only"]
    )
    out.attrs["upper_bound"] = not out.attrs["public_only"]
    return out.reset_index(drop=True)


# --- Earth Engine side of the 3-30-300 inputs --------------------------------


def green_canopy_image(
    params: dict[str, Any],
    year: int | None = None,
    region: "ee.Geometry | None" = None,
    scheme: str | None = None,
) -> "ee.Image":
    """Green, canopy and coverage as one three-band image.

    ``green`` is the 3-30-300 green-space classes; ``canopy`` is the tree class
    alone, because the 30 % target is a *canopy* target and grass is not canopy.
    ``observed`` follows ``spatial_stats.green_class_image``'s pattern exactly -
    it is 1 only where the classifier produced a value AND the cell is inside the
    region, which is the analysable set.

    The third band is not bookkeeping. A 0/1 raster written to GeoTIFF cannot
    distinguish "classified, not green" from "never classified", and masked
    pixels are written as 0. That is the bug that made Phase 5's Dynamic World
    green appear to grow 5.4x between 2016 and 2024.

    Args:
        params: Parsed params mapping.
        year: Land-cover year; defaults to ``greening.landcover_year``.
        region: Optional region to clip to.
        scheme: Override for ``greening.landcover_scheme``.

    Returns:
        Four-band ``ee.Image`` in :data:`GREEN_CANOPY_BANDS` order - ``green``,
        ``canopy``, ``water`` and ``observed`` - carrying ``scheme`` and ``year``
        properties.
    """
    import ee  # Deferred: see module docstring.

    from colombo_uhi import aoi, landcover

    config = params["greening"]["rule_3_30_300"]
    resolved_scheme = str(
        params["greening"]["landcover_scheme"] if scheme is None else scheme
    )
    resolved_year = resolve_landcover_year(params, year)

    classes = landcover.class_image(
        resolved_scheme, params, year=resolved_year, region=region
    )
    if region is not None:
        classes = classes.clip(region)

    # .mask() BEFORE unmasking: 1 wherever the classifier produced a value and 0
    # elsewhere, which after the clip means "inside the study area and actually
    # classified".
    observed = classes.mask().reduce("min").gt(0).rename("observed").toByte()

    green_codes = [int(code) for code in config["green_space"]["class_codes"]]
    canopy_codes = [int(code) for code in config["canopy"]["class_codes"]]
    green = (
        classes.remap(green_codes, [1] * len(green_codes), 0)
        .unmask(0)
        .rename("green")
        .toByte()
    )
    canopy = (
        classes.remap(canopy_codes, [1] * len(canopy_codes), 0)
        .unmask(0)
        .rename("canopy")
        .toByte()
    )

    # *** THE LAND DENOMINATOR, AND IT MUST SEE THE SEA. ***
    # `observed` says the classifier produced a value; it does NOT say the cell
    # is land. Over the Colombo Port outer harbour Dynamic World produces
    # nothing at all, so a coverage fraction taken against the whole polygon
    # counts open water as missing data - and excludes Pettah, Lunupokuna and
    # Fort from the priority list.
    #
    # Run 4 tried aoi.static_water_mask here and it did not work: JRC Global
    # Surface Water maps INLAND water and does not map the ocean, so the harbour
    # still counted as land. Run 5 measured it - Fort came out at 0.767 against
    # 0.705 on the raw polygon, and only NINE of 557 divisions moved by more
    # than 0.01.
    #
    # aoi.water_mask ORs MNDWI, the QA_PIXEL water-bit frequency AND JRC, and
    # the MNDWI detector sees the sea. Its docstring warns that it embeds a
    # Landsat composite into every image it masks, which is why the cheap
    # variant exists for long series - but this is ONE static export, which is
    # the case that docstring explicitly recommends it for.
    water_source = str(params["greening"]["land_mask"])
    if water_source == "combined":
        water = aoi.water_mask(params, region=region)
    elif water_source == "static_jrc":
        water = aoi.static_water_mask(params, region=region)
    else:
        raise ValueError(
            f"greening.land_mask is {water_source!r}; expected 'combined' "
            "(MNDWI + QA + JRC, sees the ocean) or 'static_jrc' (JRC only, "
            "inland water, cheap enough for a long series)"
        )
    water = water.rename("water").toByte()

    return ee.Image.cat([green, canopy, water, observed]).set(
        {"scheme": resolved_scheme, "year": int(resolved_year)}
    )


def population_image(
    params: dict[str, Any],
    year: int | None = None,
    region: "ee.Geometry | None" = None,
) -> "ee.Image":
    """People per cell, plus a coverage band.

    ``spatial_stats.population_density`` returns people per km2, which is the
    right unit for a regression covariate and the wrong one for the 300 rule -
    that rule counts *residents*, so a share must be weighted by a count. This
    converts to people per cell on the configured grid.

    Args:
        params: Parsed params mapping.
        year: WorldPop year; defaults to the Phase 5 covariate year.
        region: Optional region to clip to.

    Returns:
        Two-band ``ee.Image`` in :data:`POPULATION_BANDS` order, carrying the
        WorldPop year it actually used as a property. WorldPop ends 2020, and the
        year must travel rather than being silently relabelled.
    """
    import ee  # Deferred: see module docstring.

    from colombo_uhi import spatial_stats

    resolved_year = int(
        params["spatial_stats"]["covariates"]["population"]["year"]
        if year is None
        else year
    )
    scale = float(params["greening"]["population_scale_m"])

    density = spatial_stats.population_density(params, year=resolved_year)
    if region is not None:
        density = density.clip(region)

    # people/km2 -> people per cell of the analysis grid
    cell_km2 = (scale * scale) / 1_000_000.0
    population = density.multiply(cell_km2).rename("population")
    observed = density.mask().reduce("min").gt(0).rename("observed")

    # *** BOTH BANDS MUST SHARE ONE DTYPE. *** Colab run 1 lost this export to
    # "Exported bands must have compatible data types; found inconsistent types:
    # Float32 and Byte" - a Float32 count beside a Byte mask. Export.image.toDrive
    # refuses a mixed-type stack outright, so the cast goes on the CAT rather than
    # on each band, where the two could drift apart again.
    return ee.Image.cat([population.unmask(0), observed]).toFloat().set(
        {"population_year": resolved_year, "scale_m": scale}
    )


def export_green_canopy_raster(
    params: dict[str, Any],
    region: "ee.Geometry",
    year: int | None = None,
    folder: str | None = None,
    suffix: str | None = None,
    start: bool = True,
) -> Any:
    """Submit the 10 m green/canopy/coverage raster to Drive.

    Args:
        params: Parsed params mapping.
        region: Region to export.
        year: Land-cover year.
        folder: Drive folder override.
        suffix: Export-name suffix.
        start: Start the task immediately.

    Returns:
        The ``ee.batch.Task``.
    """
    from colombo_uhi import exports

    resolved_year = resolve_landcover_year(params, year)
    scale = int(params["greening"]["landcover_scale_m"])
    return exports.image_to_drive(
        green_canopy_image(params, year=resolved_year, region=region),
        product="greening_green_canopy",
        aoi="district",
        params=params,
        region=region,
        band_order=list(GREEN_CANOPY_BANDS),
        scale_m=scale,
        folder=folder,
        start_year=resolved_year,
        end_year=resolved_year,
        suffix=suffix,
        start=start,
    )


def export_population_raster(
    params: dict[str, Any],
    region: "ee.Geometry",
    year: int | None = None,
    folder: str | None = None,
    suffix: str | None = None,
    start: bool = True,
) -> Any:
    """Submit the population raster to Drive.

    Args:
        params: Parsed params mapping.
        region: Region to export.
        year: WorldPop year.
        folder: Drive folder override.
        suffix: Export-name suffix.
        start: Start the task immediately.

    Returns:
        The ``ee.batch.Task``.
    """
    from colombo_uhi import exports

    resolved_year = int(
        params["spatial_stats"]["covariates"]["population"]["year"]
        if year is None
        else year
    )
    return exports.image_to_drive(
        population_image(params, year=resolved_year, region=region),
        product="greening_population",
        aoi="district",
        params=params,
        region=region,
        band_order=list(POPULATION_BANDS),
        scale_m=int(params["greening"]["population_scale_m"]),
        folder=folder,
        start_year=resolved_year,
        end_year=resolved_year,
        suffix=suffix,
        start=start,
    )


def _read_bands(
    path: str | Path, expected: Sequence[str], label: str
) -> tuple[list[Any], dict[str, Any]]:
    """Read exactly ``len(expected)`` bands, refusing anything else."""
    import rasterio  # Deferred: see module docstring.

    with rasterio.open(str(path)) as handle:
        if handle.count != len(expected):
            raise ValueError(
                f"{path} has {handle.count} band(s); a {label} raster must carry "
                f"exactly {len(expected)}: {list(expected)}. Without the coverage "
                "band, cells the classifier never saw are written as 0 and read "
                "back as a real value - the bug that made Phase 5's green cover "
                "appear to grow 5.4x."
            )
        # masked reads throughout: the rasters are clipped to the district, so
        # the bounding box corners carry no data, and an unmasked read returns
        # the GeoTIFF fill rather than an absence.
        bands = [handle.read(index + 1, masked=True) for index in range(handle.count)]
        profile = dict(handle.profile)
    return bands, profile


def read_green_canopy_raster(
    path: str | Path, params: dict[str, Any]
) -> tuple[dict[str, "np.ndarray"], dict[str, Any]]:
    """Read the exported green/canopy raster, keeping coverage apart from class.

    Args:
        path: Path to the downloaded ``.tif``.
        params: Parsed params mapping.

    Returns:
        ``(bands, profile)`` where ``bands`` maps each of
        :data:`GREEN_CANOPY_BANDS` to a boolean array, plus a derived ``land``
        band (``not water``) - the denominator the coverage floor must be taken
        against.

    Raises:
        ValueError: If the file does not carry exactly three bands.
    """
    import numpy as np  # Deferred: see module docstring.

    raw, profile = _read_bands(path, GREEN_CANOPY_BANDS, "green/canopy")
    bands = {
        name: np.asarray(values.filled(0)).astype(bool)
        for name, values in zip(GREEN_CANOPY_BANDS, raw)
    }
    # A cell the classifier never saw is neither green nor not-green.
    bands["green"] = bands["green"] & bands["observed"]
    bands["canopy"] = bands["canopy"] & bands["observed"]
    # The land denominator the coverage floor must be taken against. Derived here
    # so no caller has to remember to negate the water band - forgetting exactly
    # that is what made the run-2 fix inert.
    bands["land"] = ~bands["water"]
    return bands, profile


def zone_raster(
    profile: Mapping[str, Any],
    geometry: "gpd.GeoDataFrame",
    params: dict[str, Any],
    id_column: str = "zone_id",
) -> tuple["np.ndarray", dict[int, str]]:
    """Burn zone polygons onto a raster grid, returning codes and their labels.

    Phase 5 did this inline in the notebook, twice. Phase 7 needs it at two
    different resolutions - 10 m for the green/canopy products and 100 m for
    population - and a zone raster that silently disagrees with the grid it is
    paired with misregisters every zonal statistic downstream. So it lives here,
    where it is tested, rather than in a notebook cell.

    Codes start at 1; 0 means "outside every zone", which is what the ``_zone_sums``
    helpers treat as nodata.

    Args:
        profile: A rasterio profile, carrying ``transform``, ``crs``, ``height``
            and ``width``.
        geometry: Zone polygons with ``id_column``.
        params: Parsed params mapping.
        id_column: Column holding the zone identifier.

    Returns:
        ``(codes, labels)`` - an int32 array on the profile's grid, and a mapping
        of code to zone id.

    Raises:
        ValueError: If the profile is incomplete or the identifier is absent.
    """
    import numpy as np  # Deferred: see module docstring.
    from rasterio import features as rio_features

    for key in ("transform", "height", "width"):
        if key not in profile:
            raise ValueError(
                f"the raster profile has no {key!r}; pass the profile returned by "
                "read_green_canopy_raster or read_population_raster"
            )
    if id_column not in geometry.columns:
        raise ValueError(
            f"the geometry has no {id_column!r} column; it has "
            f"{sorted(geometry.columns)}"
        )

    frame = geometry
    target = profile.get("crs")
    if target is not None and getattr(frame, "crs", None) is not None:
        try:
            if str(frame.crs) != str(target):
                frame = frame.to_crs(target)
        except Exception:  # pragma: no cover - a CRS-less test fixture
            pass

    labels = {
        index + 1: str(value)
        for index, value in enumerate(frame[id_column].astype(str))
    }
    codes = rio_features.rasterize(
        (
            (geom, code)
            for code, geom in zip(labels, frame.geometry)
            if geom is not None and not geom.is_empty
        ),
        out_shape=(int(profile["height"]), int(profile["width"])),
        transform=profile["transform"],
        fill=0,
        dtype="int32",
    )
    return np.asarray(codes), labels


def zone_land_area(
    codes: "np.ndarray",
    labels: Mapping[int, Any],
    cell_size_m: float,
    observed: "np.ndarray | None" = None,
) -> "pd.DataFrame":
    """Land area per zone, from the zone raster rather than from the polygon.

    This is the denominator :func:`land_observed_fraction` needs. Taking it from
    the polygon instead is precisely the bug that makes Fort - whose COD-AB
    polygon *is* the Colombo Port outer harbour - look like an unobserved zone.

    Args:
        codes: Zone-code raster.
        labels: Zone code to ``zone_id``.
        cell_size_m: Cell size in metres.
        observed: Optional coverage mask restricting the count to land the
            classifier saw.

    Returns:
        ``pandas.DataFrame`` with ``zone_id`` and ``land_area_ha``.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd

    array = np.asarray(codes)
    mask = (
        np.ones(array.shape, dtype=bool)
        if observed is None
        else np.asarray(observed, dtype=bool)
    )
    cell_ha = (float(cell_size_m) ** 2) / 10_000.0
    counts = np.bincount(
        array[mask].ravel(), minlength=(int(array.max()) + 1 if array.size else 1)
    )
    return pd.DataFrame(
        {
            "zone_id": [str(value) for value in labels.values()],
            "land_area_ha": [
                float(counts[code]) * cell_ha if code < len(counts) else 0.0
                for code in labels
            ],
        }
    )


def zone_coverage(
    bands: Mapping[str, "np.ndarray"],
    zones: "np.ndarray",
    zone_labels: Mapping[int, Any],
    cell_size_m: float,
    params: dict[str, Any],
    raw: "pd.DataFrame | None" = None,
) -> "pd.DataFrame":
    """Per-zone land-cover coverage, numerator and denominator from ONE raster.

    .. warning::
        **Both sides must come from the same product.** Colab run 4 took the
        classified area from Phase 5's committed
        ``landscape_metrics_green_by_gn.csv`` and the land area from the current
        10 m raster - two different exports, from two different phases - so the
        ratio measured the difference between them rather than coverage. Step 2
        of the same run had just measured Dynamic World 2024 covering **99.96 %**
        of the district, while the mixed ratio put 15 zones below a 90 % floor.
        Pettah stayed excluded from the priority list on the strength of it.

        This function exists so a caller cannot mix the two. It takes one
        ``bands`` mapping and derives both sides from it.

    Coverage is ``(observed AND land) / land``: the share of a zone's **land**
    that the classifier actually saw. Permanent water is removed from both sides,
    so a harbour inside a polygon neither counts as missing data nor inflates the
    denominator.

    Args:
        bands: The mapping :func:`read_green_canopy_raster` returned, carrying at
            least ``observed`` and ``land``.
        zones: Integer zone-code raster on the same grid as ``bands``.
        zone_labels: Zone code to ``zone_id``.
        cell_size_m: Cell size in metres.
        params: Parsed params mapping.
        raw: Optional frame with ``zone_id`` and ``observed_fraction`` - an
            earlier phase's flag, carried through for comparison only and never
            used as a numerator.

    Returns:
        ``pandas.DataFrame`` with ``zone_id``, ``land_area_ha``,
        ``analysable_area_ha``, ``land_observed_fraction`` and
        ``below_land_coverage_floor``; plus ``observed_fraction_raw``,
        ``below_coverage_floor_raw`` and ``status_changed`` when ``raw`` is given.

    Raises:
        ValueError: If a required band is absent or the shapes disagree.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd

    for name in ("observed", "land"):
        if name not in bands:
            raise ValueError(
                f"no {name!r} band; zone_coverage needs both 'observed' and "
                f"'land' from one raster, and got {sorted(bands)}"
            )
    observed = np.asarray(bands["observed"], dtype=bool)
    land = np.asarray(bands["land"], dtype=bool)
    codes = np.asarray(zones)
    if not (observed.shape == land.shape == codes.shape):
        raise ValueError(
            f"observed {observed.shape}, land {land.shape} and zones "
            f"{codes.shape} must be the same shape"
        )

    land_area = zone_land_area(codes, zone_labels, cell_size_m, observed=land)
    analysable = zone_land_area(
        codes, zone_labels, cell_size_m, observed=observed & land
    ).rename(columns={"land_area_ha": "analysable_area_ha"})

    landscape = analysable.rename(
        columns={"analysable_area_ha": "landscape_area_ha"}
    )
    if raw is not None and "observed_fraction" in raw.columns:
        reference = raw[["zone_id", "observed_fraction"]].copy()
        reference["zone_id"] = reference["zone_id"].astype(str)
        landscape = landscape.merge(reference, on="zone_id", how="left")

    fractions = land_observed_fraction(landscape, land_area, params)
    result = fractions.merge(
        analysable[["zone_id", "analysable_area_ha"]], on="zone_id", how="left"
    ).merge(land_area[["zone_id", "land_area_ha"]], on="zone_id", how="left")

    if raw is None or "observed_fraction" not in raw.columns:
        result = result.drop(
            columns=[
                "observed_fraction_raw",
                "below_coverage_floor_raw",
                "status_changed",
            ],
            errors="ignore",
        )

    # DataFrame.merge does NOT carry .attrs, so the counts land_observed_fraction
    # recorded would be silently dropped - and the notebook prints them.
    result.attrs.update(fractions.attrs)
    result.attrs["water_area_km2"] = float(
        (~land).sum() * (float(cell_size_m) ** 2) / 1e6
    )
    return result


def read_population_raster(
    path: str | Path, params: dict[str, Any]
) -> tuple["np.ndarray", "np.ndarray", dict[str, Any]]:
    """Read the exported population raster.

    Args:
        path: Path to the downloaded ``.tif``.
        params: Parsed params mapping.

    Returns:
        ``(population, observed, profile)``.

    Raises:
        ValueError: If the file does not carry exactly two bands.
    """
    import numpy as np  # Deferred: see module docstring.

    raw, profile = _read_bands(path, POPULATION_BANDS, "population")
    population = np.asarray(raw[0].filled(0.0)).astype(float)
    observed = np.asarray(raw[1].filled(0)).astype(bool)
    return np.where(observed, population, 0.0), observed, profile


# =============================================================================
# Group H - the UTFVI severe-class share
# =============================================================================


def utfvi_severe_classes(params: dict[str, Any]) -> list[int]:
    """Class indices counted as severe, resolved through the configured labels.

    Resolving by *label* rather than by hardcoded index means that editing
    ``uhi.utfvi.labels`` cannot silently redefine what "severe" means - it raises
    instead.

    Args:
        params: Parsed params mapping.

    Returns:
        Zero-based class indices of the severe UTFVI classes.

    Raises:
        ValueError: If a configured severe label is not a UTFVI class.
    """
    from colombo_uhi import uhi_metrics

    labels = uhi_metrics.utfvi_class_labels(params)
    entry = next(
        item
        for item in resolve_criteria(params)
        if str(item["name"]) == "utfvi_severe_share"
    )
    severe = [str(label) for label in entry["severe_labels"]]

    indices = []
    for label in severe:
        if label not in labels:
            raise ValueError(
                f"UTFVI class {label!r} is configured as severe but is not one of "
                f"{labels}. Editing uhi.utfvi.labels changes what the greening "
                "criterion measures, so it raises rather than quietly redefining "
                "the criterion."
            )
        indices.append(labels.index(label))
    return sorted(indices)


def utfvi_severe_image(
    params: dict[str, Any],
    epoch: str | None = None,
    region: "ee.Geometry | None" = None,
    source: str | None = None,
    collection: "ee.ImageCollection | None" = None,
) -> "ee.Image":
    """A 0/1 severe-UTFVI mask plus its coverage band, for one epoch.

    Args:
        params: Parsed params mapping.
        epoch: Epoch key; defaults to ``greening.epoch``.
        region: Region the composite is built over. Required.
        source: Override for ``greening.source``.
        collection: Optional pre-built scene collection.

    Returns:
        Two-band ``ee.Image``: ``utfvi_severe_share`` (0/1) and ``observed``.

    Raises:
        ValueError: If no region is supplied - UTFVI's reference is that year's
            own spatial mean over the AOI, so the region is part of the
            definition rather than an optimisation.
    """
    import ee  # Deferred: see module docstring.

    from colombo_uhi import uhi_metrics

    if region is None:
        raise ValueError(
            "utfvi_severe_image needs a region: uhi.utfvi.reference is "
            "'per_year_aoi_mean', so the AOI is part of the definition of the "
            "index rather than a way to bound the computation"
        )

    resolved_epoch = str(params["greening"]["epoch"] if epoch is None else epoch)
    resolved_source = str(params["greening"]["source"] if source is None else source)
    scale = int(params["greening"]["scale_m"])

    composite = uhi_metrics.epoch_composite(
        resolved_source, params, resolved_epoch, collection=collection, region=region
    )
    index = uhi_metrics.utfvi(composite, params, region, scale_m=scale)
    classes = uhi_metrics.utfvi_class_image(index, params)

    severe = utfvi_severe_classes(params)
    observed = classes.mask().reduce("min").gt(0).rename("observed").toByte()
    share = (
        classes.remap(severe, [1] * len(severe), 0)
        .unmask(0)
        .rename("utfvi_severe_share")
        .toFloat()
    )
    return ee.Image.cat([share, observed]).set(
        {
            "epoch": resolved_epoch,
            "source": resolved_source,
            "severe_classes": severe,
        }
    )


def utfvi_shares_by_zone(
    params: dict[str, Any],
    level: str | None = None,
    epoch: str | None = None,
    region: "ee.Geometry | None" = None,
    source: str | None = None,
    scale_m: int | None = None,
    collection: "ee.ImageCollection | None" = None,
) -> "pd.DataFrame":
    """Per-zone share of pixels in the severe UTFVI classes.

    The mean of a 0/1 mask over a zone *is* the share, which is why this reuses
    ``uhi_metrics.zonal_by_division`` with a mean reducer rather than growing a
    second zonal-statistics path.

    Args:
        params: Parsed params mapping.
        level: ``"gn"`` or ``"ds"``.
        epoch: Epoch key.
        region: Region the composite is built over.
        source: Override for ``greening.source``.
        scale_m: Override for ``greening.scale_m``.
        collection: Optional pre-built scene collection.

    Returns:
        ``pandas.DataFrame`` with ``zone_id``, ``utfvi_severe_share`` and
        ``utfvi_severe_share_pixels``.
    """
    from colombo_uhi import uhi_metrics

    resolved_level = resolve_level(level, params)
    scale = int(params["greening"]["scale_m"] if scale_m is None else scale_m)
    image = utfvi_severe_image(
        params, epoch=epoch, region=region, source=source, collection=collection
    )
    table = uhi_metrics.zonal_by_division(
        image,
        params,
        level=resolved_level,
        band="utfvi_severe_share",
        scale_m=scale,
        reducers=("mean",),
    )
    identifier = next(
        column for column in table.columns if column not in ("mean", "pixel_count")
    )
    out = table.rename(
        columns={
            identifier: "zone_id",
            "mean": "utfvi_severe_share",
            "pixel_count": "utfvi_severe_share_pixels",
        }
    )
    out["zone_id"] = out["zone_id"].astype(str)
    out.attrs["level"] = resolved_level
    out.attrs["scale_m"] = scale
    return out.sort_values("zone_id").reset_index(drop=True)


# =============================================================================
# Group I - the Colombo Wetland Complex cross
# =============================================================================


def wetland_asset_collection(params: dict[str, Any]) -> "ee.FeatureCollection":
    """Load the user-uploaded official wetland boundary.

    Args:
        params: Parsed params mapping.

    Returns:
        The ``ee.FeatureCollection`` at ``greening.wetland.asset``.

    Raises:
        ValueError: If the asset is not configured, with upload instructions.
            The same pattern as ``aoi.assets.gn_divisions``: an official boundary
            silently replaced by a proxy union would be the wrong kind of quiet.
    """
    import ee  # Deferred: see module docstring.

    asset = params["greening"]["wetland"].get("asset")
    if not asset:
        raise ValueError(
            "greening.wetland.asset is not set. No official Colombo Wetland "
            "Complex boundary exists in any free dataset - Colombo's 2018 Ramsar "
            "accreditation is a Wetland CITY accreditation, not a Ramsar Site "
            "designation, so there is no polygon to download. To use the official "
            "boundary: obtain it from the Colombo Wetland Management Strategy "
            "(SLLRDC, UDA, or the Metro Colombo Urban Development Project), "
            "upload it as an Earth Engine table asset, and set its id here. "
            "Otherwise leave it null and the wetland cross runs on the "
            "four-source proxy union, reported as such."
        )
    return ee.FeatureCollection(str(asset))


def wdpa_collection(
    params: dict[str, Any], region: "ee.Geometry | None" = None
) -> "ee.FeatureCollection":
    """Sri Lanka's protected areas from the World Database on Protected Areas.

    .. note::
        WDPA captures **legally declared** protected areas - the
        Bellanwila-Attidiya sanctuary among them - and **not** the full Colombo
        Wetland Complex. Treat it as the strongest single line of evidence here
        and simultaneously the least complete.

    Args:
        params: Parsed params mapping.
        region: Optional region to filter to.

    Returns:
        The filtered ``ee.FeatureCollection``.
    """
    import ee  # Deferred: see module docstring.

    config = params["greening"]["wetland"]["source_definitions"]["wdpa"]
    collection = ee.FeatureCollection(str(config["id"])).filter(
        ee.Filter.eq(str(config["iso3_property"]), str(config["iso3_value"]))
    )

    # *** WDPA IS A PROTECTED-AREA LAYER, NOT A WETLAND LAYER. ***
    # [MEASURED - Colab run 1] Ten protected areas intersect Colombo District,
    # and only four of them are wetlands: the Bellanwila-Attidiya and Sri
    # Jayawardanapura sanctuaries, and the Thalangama and Bolgoda Environmental
    # Protection Areas. The other six are inland forest - Labugama Kalatuwawa,
    # Indikada Mukalana, Mitirigala, Kananpella, Miriyagalla and Kurana Madakada,
    # the first of which is a water-catchment forest some 30 km inland. Folding
    # those into a "wetland" union would flag divisions as wetland-adjacent on
    # the strength of a forest reserve, which is a different policy instrument
    # over a different landscape.
    designations = config.get("designations_include")
    if designations:
        collection = collection.filter(
            ee.Filter.inList(
                str(config["designation_property"]),
                [str(value) for value in designations],
            )
        )
    if region is not None:
        collection = collection.filterBounds(region)
    return collection


def wetland_source_image(
    source: str,
    params: dict[str, Any],
    year: int | None = None,
    region: "ee.Geometry | None" = None,
) -> "ee.Image":
    """Build one wetland layer as a 0/1 image.

    Args:
        source: One of :data:`WETLAND_SOURCES`.
        params: Parsed params mapping.
        year: Land-cover year, where the source needs one.
        region: Optional region to clip to.

    Returns:
        Single-band ``ee.Image`` named after the source.

    Raises:
        ValueError: If the source is unknown or not configured.
    """
    import ee  # Deferred: see module docstring.

    from colombo_uhi import landcover

    if source not in WETLAND_SOURCES:
        raise ValueError(f"wetland source {source!r} is not one of {list(WETLAND_SOURCES)}")

    definitions = params["greening"]["wetland"]["source_definitions"]
    if source != "asset" and source not in definitions:
        raise ValueError(
            f"wetland source {source!r} has no entry in "
            "greening.wetland.source_definitions"
        )

    if source == "asset":
        # Same `paint` idiom as the WDPA branch below, and for the same reason.
        # `reduceToImage(["system:index"], count)` would be worse here still:
        # system:index is a STRING, which a count reducer cannot sum over.
        image = ee.Image.constant(0).byte().paint(
            featureCollection=wetland_asset_collection(params), color=1
        )
    elif source == "wdpa":
        # `paint` rather than `reduceToImage`. Colab run 1 measured what the
        # difference costs: reduceToImage returned a fully-masked image, so WDPA
        # summed to 0.00 km2 and was dropped from the union as "returns nothing
        # here" - while the probe beside it was listing ten protected areas by
        # name, including every wetland site the cross exists to find. An empty
        # raster that reads as an honest zero is the worst failure mode in this
        # module. `paint` inherits the projection of the image it paints onto and
        # has no such subtlety.
        image = ee.Image.constant(0).byte().paint(
            featureCollection=wdpa_collection(params, region=region), color=1
        )
    elif source == "gsw_seasonal":
        config = definitions[source]
        dataset = params["datasets"][str(config["dataset"])]
        seasonality = ee.Image(str(dataset["id"])).select(str(config["band"]))
        # Seasonally inundated but NOT permanent: excluding 12 months is what
        # makes this a wetland proxy rather than a second water mask.
        image = seasonality.gte(int(config["min_months"])).And(
            seasonality.lte(int(config["max_months"]))
        )
    else:
        config = definitions[source]
        dataset_key = str(config["dataset"])
        codes = [int(code) for code in config["class_codes"]]
        scheme = (
            "dynamic_world" if dataset_key == "dynamic_world" else "worldcover"
        )
        classes = landcover.class_image(
            scheme,
            params,
            year=resolve_landcover_year(params, year) if scheme == "dynamic_world" else None,
            region=region,
        )
        image = classes.remap(codes, [1] * len(codes), 0)

    image = image.unmask(0).rename(source).toByte()
    if region is not None:
        image = image.clip(region)
    return image.set({"wetland_source": source})


def wetland_image(
    params: dict[str, Any],
    sources: Sequence[str] | None = None,
    year: int | None = None,
    region: "ee.Geometry | None" = None,
) -> "ee.Image":
    """Union of every configured wetland layer, with per-source provenance.

    Every source keeps its own band, so a zone's status can say *which* evidence
    fired for it. That matters: WDPA is a legal designation and the other three
    are remote-sensing proxies, and a policy recommendation should not present
    them as the same kind of statement.

    Args:
        params: Parsed params mapping.
        sources: Optional explicit source list.
        year: Land-cover year, where a source needs one.
        region: Optional region to clip to.

    Returns:
        ``ee.Image`` with one band per source, plus ``wetland`` (the union),
        ``n_sources`` (how many fired per pixel) and ``observed``.
    """
    import ee  # Deferred: see module docstring.

    resolved = resolve_wetland_sources(params, sources)
    layers = [
        wetland_source_image(source, params, year=year, region=region)
        for source in resolved
    ]
    stack = ee.Image.cat(layers)

    count = stack.reduce(ee.Reducer.sum()).rename("n_sources").toByte()
    union = count.gt(0).rename("wetland").toByte()
    observed = ee.Image.constant(1).rename("observed").toByte()
    if region is not None:
        observed = observed.clip(region)

    return ee.Image.cat([stack, union, count, observed]).set(
        {"wetland_sources": resolved, "n_wetland_sources": len(resolved)}
    )


def export_wetland_raster(
    params: dict[str, Any],
    region: "ee.Geometry",
    sources: Sequence[str] | None = None,
    year: int | None = None,
    folder: str | None = None,
    suffix: str | None = None,
    start: bool = True,
) -> Any:
    """Submit the wetland raster to Drive.

    Args:
        params: Parsed params mapping.
        region: Region to export.
        sources: Optional explicit source list.
        year: Land-cover year.
        folder: Drive folder override.
        suffix: Export-name suffix.
        start: Start the task immediately.

    Returns:
        The ``ee.batch.Task``.
    """
    from colombo_uhi import exports

    resolved = resolve_wetland_sources(params, sources)
    bands = [*resolved, "wetland", "n_sources", "observed"]
    resolved_year = resolve_landcover_year(params, year)
    return exports.image_to_drive(
        wetland_image(params, sources=resolved, year=resolved_year, region=region),
        product="greening_wetland",
        aoi="district",
        params=params,
        region=region,
        band_order=bands,
        scale_m=int(params["greening"]["landcover_scale_m"]),
        folder=folder,
        start_year=resolved_year,
        end_year=resolved_year,
        suffix=suffix,
        start=start,
    )


def read_wetland_raster(
    path: str | Path, params: dict[str, Any], sources: Sequence[str] | None = None
) -> tuple[dict[str, "np.ndarray"], dict[str, Any]]:
    """Read the exported wetland raster, one array per source plus the union.

    Args:
        path: Path to the downloaded ``.tif``.
        params: Parsed params mapping.
        sources: Optional explicit source list, in export order.

    Returns:
        ``(bands, profile)``.

    Raises:
        ValueError: If the band count does not match the expected layout.
    """
    import numpy as np  # Deferred: see module docstring.

    resolved = resolve_wetland_sources(params, sources)
    expected = (*resolved, "wetland", "n_sources", "observed")
    raw, profile = _read_bands(path, expected, "wetland")
    bands = {
        name: np.asarray(values.filled(0))
        for name, values in zip(expected, raw)
    }
    for name in expected:
        if name != "n_sources":
            bands[name] = bands[name].astype(bool)
    return bands, profile


def wetland_shares_by_zone(
    bands: Mapping[str, "np.ndarray"],
    zones: "np.ndarray",
    zone_labels: Mapping[int, Any],
    params: dict[str, Any],
    sources: Sequence[str] | None = None,
) -> "pd.DataFrame":
    """Per-zone wetland share, and which sources produced it.

    Args:
        bands: The mapping :func:`read_wetland_raster` returned.
        zones: Integer zone-code raster.
        zone_labels: Zone code to ``zone_id``.
        params: Parsed params mapping.
        sources: Optional explicit source list.

    Returns:
        ``pandas.DataFrame`` with ``zone_id``, ``wetland_within_pct``, a
        ``<source>_pct`` column per source, and ``wetland_sources`` naming the
        sources that actually fired in that zone.

    Raises:
        ValueError: If a band is absent or shapes disagree.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd

    resolved = resolve_wetland_sources(params, sources)
    if "wetland" not in bands:
        raise ValueError(f"no 'wetland' band; got {sorted(bands)}")

    codes = np.asarray(zones)
    observed = np.asarray(bands.get("observed", np.ones(codes.shape, dtype=bool)), bool)
    if observed.shape != codes.shape:
        raise ValueError(
            f"the wetland bands {observed.shape} and zones {codes.shape} must "
            "be the same shape"
        )

    threshold = float(params["greening"]["wetland"]["within_pct_threshold"])
    union = _zone_sums(
        codes, np.asarray(bands["wetland"], bool).astype(float), observed.astype(float)
    )
    per_source = {
        source: _zone_sums(
            codes, np.asarray(bands[source], bool).astype(float), observed.astype(float)
        )
        for source in resolved
        if source in bands
    }

    rows = []
    for code, zone_id in zone_labels.items():
        key = int(code)
        numerator, denominator = union.get(key, (0.0, 0.0))
        row: dict[str, Any] = {
            "zone_id": str(zone_id),
            "wetland_within_pct": (
                100.0 * numerator / denominator if denominator > 0 else np.nan
            ),
        }
        fired = []
        for source, sums in per_source.items():
            source_numerator, source_denominator = sums.get(key, (0.0, 0.0))
            share = (
                100.0 * source_numerator / source_denominator
                if source_denominator > 0
                else np.nan
            )
            row[f"{source}_pct"] = share
            if np.isfinite(share) and share >= threshold:
                fired.append(source)
        row["wetland_sources"] = ";".join(fired)
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.attrs["sources"] = resolved
    frame.attrs["within_pct_threshold"] = threshold
    return frame


def wetland_adjacency(
    geometry: "gpd.GeoDataFrame",
    shares: "pd.DataFrame",
    params: dict[str, Any],
    method: str | None = None,
    distance_m: float | None = None,
) -> "pd.DataFrame":
    """Classify each zone as within, adjacent to, or away from wetland.

    Two definitions, both reported, because neither is authoritative. ``buffer``
    is metric and needs a distance threshold that is frankly arbitrary - so it is
    computed at all three of ``adjacency.distance_sensitivity_m``.
    ``queen_neighbour`` is topological and needs no threshold at all, reusing the
    already-tested ``spatial_stats.contiguity_neighbours``. Where the two
    disagree is itself a reportable result.

    Args:
        geometry: Zone polygons with ``zone_id``, in the analysis CRS.
        shares: The frame :func:`wetland_shares_by_zone` returned.
        params: Parsed params mapping.
        method: Override for ``adjacency.method``.
        distance_m: Override for ``adjacency.distance_m``.

    Returns:
        ``pandas.DataFrame`` with ``zone_id``, ``wetland_status``,
        ``wetland_within_pct``, an ``adjacent_within_<d>m`` column per
        sensitivity distance, ``adjacent_queen`` and ``wetland_policy_flag``.

    Raises:
        ValueError: If the method is unknown or ``zone_id`` does not join.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd

    from colombo_uhi import spatial_stats

    config = params["greening"]["wetland"]["adjacency"]
    resolved_method = str(config["method"] if method is None else method)
    if resolved_method not in ADJACENCY_METHODS:
        raise ValueError(
            f"adjacency method {resolved_method!r} is not one of "
            f"{list(ADJACENCY_METHODS)}"
        )
    primary = float(config["distance_m"] if distance_m is None else distance_m)
    distances = sorted(
        {float(value) for value in config["distance_sensitivity_m"]} | {primary}
    )
    threshold = float(params["greening"]["wetland"]["within_pct_threshold"])

    frame = geometry.copy()
    frame["zone_id"] = frame["zone_id"].astype(str)
    merged = frame.merge(
        shares[["zone_id", "wetland_within_pct"]].assign(
            zone_id=lambda table: table["zone_id"].astype(str)
        ),
        on="zone_id",
        how="left",
    )
    if merged["wetland_within_pct"].isna().all():
        raise ValueError(
            "no zone joined between the geometry and the wetland shares; check "
            "that both are keyed on the same pcode"
        )

    within = (merged["wetland_within_pct"].fillna(0.0) >= threshold).to_numpy()
    wetland_geometry = merged.loc[within, "geometry"]

    out = pd.DataFrame(
        {
            "zone_id": merged["zone_id"].astype(str),
            "wetland_within_pct": merged["wetland_within_pct"].to_numpy(),
        }
    )

    for distance in distances:
        if wetland_geometry.empty:
            out[f"adjacent_within_{int(distance)}m"] = False
            continue
        buffered = wetland_geometry.buffer(distance).union_all()
        out[f"adjacent_within_{int(distance)}m"] = (
            merged["geometry"].intersects(buffered).to_numpy() & ~within
        )

    # The threshold-free second opinion.
    neighbours = spatial_stats.contiguity_neighbours(
        list(merged["geometry"]), scheme="queen"
    )
    adjacent_queen = np.array(
        [
            (not within[position])
            and any(bool(within[other]) for other in neighbours[position])
            for position in range(len(merged))
        ],
        dtype=bool,
    )
    out["adjacent_queen"] = adjacent_queen

    primary_column = f"adjacent_within_{int(primary)}m"
    adjacent = (
        out[primary_column].to_numpy(dtype=bool)
        if resolved_method == "buffer"
        else adjacent_queen
    )

    status = np.where(within, "within", np.where(adjacent, "adjacent", "neither"))
    out["wetland_status"] = status
    out["wetland_policy_flag"] = np.isin(status, ("within", "adjacent"))
    out["adjacency_disagreement"] = (
        out[primary_column].to_numpy(dtype=bool) != adjacent_queen
    ) & ~within

    out.attrs["method"] = resolved_method
    out.attrs["distance_m"] = primary
    out.attrs["distance_sensitivity_m"] = distances
    out.attrs["n_within"] = int(within.sum())
    out.attrs["n_adjacent"] = int(adjacent.sum())
    out.attrs["n_disagreement"] = int(out["adjacency_disagreement"].sum())
    return out


def wetland_cross(
    ranked: "pd.DataFrame", wetland: "pd.DataFrame", params: dict[str, Any]
) -> "pd.DataFrame":
    """Join the wetland status onto the priority ranking.

    Colombo is a Ramsar Wetland City, so wetland protection and expansion is the
    strongest local policy lever available: a high-priority zone that is inside
    or beside an existing wetland can be acted on through an instrument that
    already exists.

    Args:
        ranked: The priority ranking.
        wetland: The frame :func:`wetland_adjacency` returned.
        params: Parsed params mapping.

    Returns:
        ``ranked`` with the wetland columns joined on ``zone_id``, carrying the
        priority-zone tallies in ``.attrs``.

    Raises:
        ValueError: If ``zone_id`` is absent from either frame.
    """
    for frame, name in ((ranked, "ranked"), (wetland, "wetland")):
        if "zone_id" not in frame.columns:
            raise ValueError(f"the {name} frame has no 'zone_id' column")

    columns = [
        column
        for column in wetland.columns
        if column == "zone_id" or column not in ranked.columns
    ]
    left = ranked.copy()
    left["zone_id"] = left["zone_id"].astype(str)
    right = wetland[columns].copy()
    right["zone_id"] = right["zone_id"].astype(str)

    out = left.merge(right, on="zone_id", how="left")
    if "wetland_status" in out.columns:
        out["wetland_status"] = out["wetland_status"].fillna("neither")
    if "wetland_policy_flag" in out.columns:
        out["wetland_policy_flag"] = (
            out["wetland_policy_flag"].fillna(False).astype(bool)
        )

    if "priority" in out.columns and "wetland_status" in out.columns:
        priority = out.loc[out["priority"].fillna(False).astype(bool)]
        out.attrs["priority_wetland_counts"] = {
            status: int((priority["wetland_status"] == status).sum())
            for status in WETLAND_STATUSES
        }
        out.attrs["n_priority"] = int(len(priority))
    return out


# =============================================================================
# Group J - output and guards
# =============================================================================

#: Column order of the published priority table. Anything absent is filled with
#: NaN rather than dropped, so the schema of the report's table is fixed whatever
#: optional products were computed.
PRIORITY_COLUMNS: tuple[str, ...] = (
    "rank_ahp",
    "zone_id",
    "adm4_name",
    "adm3_name",
    "area_sqkm",
    "score_ahp",
    "score_topsis",
    "rank_topsis",
    "rank_shift",
    "LST_C",
    "utfvi_severe_share",
    "NDVI",
    "pop_density",
    "pop_within_300m_pct",
    "lst_hot_norm",
    "utfvi_severe_share_norm",
    "ndvi_deficit_norm",
    "pop_density_norm",
    "green_access_deficit_norm",
    "canopy_pct",
    "rule_30_pass",
    "area_within_300m_pct",
    "rule_300_pass",
    "pop_within_300m_detour_pct",
    "rule_300_pass_detour",
    "rule_3_proxy_pct",
    "rule_3_status",
    "compliance",
    "wetland_within_pct",
    "wetland_status",
    "wetland_sources",
    "wetland_policy_flag",
    "priority",
    "score_gap_at_cut",
    "tied_at_cut",
    "n_pixels_min",
    "land_observed_fraction",
    "below_land_coverage_floor",
    "incomplete_criteria",
    "missing_weight",
    "status",
)


def build_priority_frame(
    ranked: "pd.DataFrame",
    params: dict[str, Any],
    prepared: "pd.DataFrame | None" = None,
    topsis_ranked: "pd.DataFrame | None" = None,
    compliance: "pd.DataFrame | None" = None,
    wetland: "pd.DataFrame | None" = None,
    geometry: "gpd.GeoDataFrame | pd.DataFrame | None" = None,
) -> "pd.DataFrame":
    """Assemble the published priority table from its parts.

    Every join is on ``zone_id`` and on nothing else. GN names are **not** unique
    within Colombo District - Dehiwala, Moratuwa and Kolonnawa carry divisions
    with the same names as CMC ones - so joining on a name would silently merge
    unrelated divisions.

    Args:
        ranked: The AHP ranking.
        params: Parsed params mapping.
        prepared: Optional prepared criterion frame, for the raw and normalised
            criterion columns.
        topsis_ranked: Optional TOPSIS ranking, for ``score_topsis`` and
            ``rank_topsis``.
        compliance: Optional 3-30-300 compliance frame.
        wetland: Optional wetland adjacency frame.
        geometry: Optional zone attributes, for the division names.

    Returns:
        ``pandas.DataFrame`` with :data:`PRIORITY_COLUMNS`, sorted by rank.
        ``zone_id`` is a string - ``LK1103070`` survives either way, but an
        all-numeric pcode would become ``int64`` and silently fail to join.

    Raises:
        ValueError: If ``zone_id`` is absent or duplicated in any part.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd

    def _keyed(frame: Any, name: str) -> "pd.DataFrame":
        if "zone_id" not in frame.columns:
            raise ValueError(f"the {name} frame has no 'zone_id' column")
        out = pd.DataFrame(frame.drop(columns="geometry", errors="ignore"))
        out["zone_id"] = out["zone_id"].astype(str)
        if out["zone_id"].duplicated().any():
            raise ValueError(
                f"the {name} frame has duplicate zone_id values; the same "
                "division would appear twice in the priority table"
            )
        return out

    merged = _keyed(ranked, "ranked")
    for frame, name in (
        (prepared, "prepared"),
        (topsis_ranked, "topsis"),
        (compliance, "compliance"),
        (wetland, "wetland"),
        (geometry, "geometry"),
    ):
        if frame is None:
            continue
        right = _keyed(frame, name)
        new = ["zone_id"] + [
            column for column in right.columns if column not in merged.columns
        ]
        merged = merged.merge(right[new], on="zone_id", how="left")

    if "rank_ahp" in merged.columns and "rank_topsis" in merged.columns:
        merged["rank_shift"] = merged["rank_topsis"] - merged["rank_ahp"]

    for column in PRIORITY_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan
    if merged["rule_3_status"].isna().all():
        merged["rule_3_status"] = RULE_3_STATUS

    out = merged[list(PRIORITY_COLUMNS)].copy()
    out["zone_id"] = out["zone_id"].astype(str)
    out = out.sort_values("rank_ahp", na_position="last").reset_index(drop=True)
    for key, value in ranked.attrs.items():
        out.attrs.setdefault(key, value)
    return out


def top_priority_zones(
    ranked: "pd.DataFrame",
    params: dict[str, Any],
    top_n: int | None = None,
    include_flagged: bool = False,
) -> "pd.DataFrame":
    """The top N divisions, excluding flagged zones by default.

    Args:
        ranked: The priority table.
        params: Parsed params mapping.
        top_n: Override for ``greening.top_n``.
        include_flagged: Keep below-floor and insufficient-data zones.

    Returns:
        The top rows, in rank order.

    Raises:
        ValueError: If the rank column is absent.
    """
    if "rank_ahp" not in ranked.columns:
        raise ValueError(f"no 'rank_ahp' column; the frame has {sorted(ranked.columns)}")

    limit = int(params["greening"]["top_n"] if top_n is None else top_n)
    out = ranked.sort_values("rank_ahp", na_position="last")

    if not include_flagged:
        keep = out["score_ahp"].notna()
        if "status" in out.columns:
            keep = keep & (out["status"] != STATUS_INSUFFICIENT)
            if bool(
                params["greening"]["normalisation"]["missing"][
                    "exclude_below_floor_from_top_n"
                ]
            ):
                keep = keep & (out["status"] != STATUS_BELOW_FLOOR)
        out = out.loc[keep]

    result = out.head(limit).reset_index(drop=True)
    result.attrs["top_n"] = limit
    result.attrs["include_flagged"] = bool(include_flagged)
    result.attrs["n_excluded"] = int(len(ranked) - len(out))
    return result


def priority_zone_ids(
    ranked: "pd.DataFrame", params: dict[str, Any], top_n: int | None = None
) -> list[str]:
    """The priority zone ids, as the plain list Phase 6's scenario code takes.

    ``prediction.apply_greening_scenario`` and
    ``prediction.canopy_shift_predictors`` accept a plain zone list, so Phase 7
    replaces the Phase 5 interim proxy by handing this in - no change to the
    scenario code at all, which is what PROGRESS.md's Phase 6 sign-off promised.

    Args:
        ranked: The priority table.
        params: Parsed params mapping.
        top_n: Override for ``greening.top_n``.

    Returns:
        Zone ids in rank order, as strings.
    """
    return [
        str(value)
        for value in top_priority_zones(ranked, params, top_n=top_n)["zone_id"]
    ]


def require_complete_criteria(
    frame: "pd.DataFrame", params: dict[str, Any]
) -> dict[str, Any]:
    """Refuse to write a priority table the data cannot support.

    Args:
        frame: The priority table.
        params: Parsed params mapping.

    Returns:
        Mapping with ``n_zones``, ``n_scored`` and ``scored_fraction``.

    Raises:
        CriteriaIncomplete: If ``zone_id`` is absent or duplicated, a criterion
            column is missing entirely, or too few zones were scored.
    """
    if "zone_id" not in frame.columns:
        raise CriteriaIncomplete(
            f"no 'zone_id' column; the frame has {sorted(frame.columns)}"
        )
    if frame["zone_id"].astype(str).duplicated().any():
        duplicated = sorted(
            frame.loc[frame["zone_id"].astype(str).duplicated(), "zone_id"]
            .astype(str)
            .unique()
        )[:5]
        raise CriteriaIncomplete(
            f"duplicate zone_id values {duplicated}; the same division would be "
            "published twice with two different ranks"
        )

    absent = [
        str(entry["column"])
        for entry in resolve_criteria(params)
        if str(entry["column"]) not in frame.columns
    ]
    if absent:
        raise CriteriaIncomplete(
            f"criterion column(s) {absent} are not in the table. A published "
            "ranking must carry the values it was computed from, so a reader can "
            "see why a division ranks where it does."
        )

    if "score_ahp" not in frame.columns:
        raise CriteriaIncomplete("no 'score_ahp' column; there is no ranking to write")

    floor = float(params["greening"]["min_scored_fraction"])
    scored = int(frame["score_ahp"].notna().sum())
    fraction = scored / len(frame) if len(frame) else 0.0
    if fraction < floor:
        raise CriteriaIncomplete(
            f"only {scored} of {len(frame)} zones ({fraction:.1%}) carry a score, "
            f"below the {floor:.0%} floor in greening.min_scored_fraction. A "
            "ranking over this many zones describes the coverage of the input "
            "rasters more than it describes Colombo."
        )
    return {
        "n_zones": int(len(frame)),
        "n_scored": scored,
        "scored_fraction": float(fraction),
    }


def priority_table_metadata(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    ahp_report: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The sidecar a published priority table must travel with.

    A ranked CSV with no record of the judgements that produced it is not
    reproducible, and the weights are the part a reader most needs and is least
    able to reconstruct.

    Args:
        frame: The priority table.
        params: Parsed params mapping.
        ahp_report: The mapping :func:`ahp_weights` returned.
        extra: Anything else worth recording.

    Returns:
        A JSON-serialisable mapping.
    """
    greening = params["greening"]
    metadata: dict[str, Any] = {
        "method": str(greening["method"]),
        "level": str(greening["level"]),
        "epoch": str(greening["epoch"]),
        "source": str(greening["source"]),
        "sensitivity_source": str(greening["sensitivity_source"]),
        "scale_m": int(greening["scale_m"]),
        "landcover_scheme": str(greening["landcover_scheme"]),
        "landcover_year": int(greening["landcover_year"]),
        "normalisation": str(greening["normalisation"]["method"]),
        "top_n": int(greening["top_n"]),
        "n_zones": int(len(frame)),
        "criteria": [
            {
                "name": str(entry["name"]),
                "direction": str(entry["direction"]),
                "label": str(entry["label"]),
                "provenance": str(entry["provenance"]),
                "weight": float(ahp_report["weights"].get(str(entry["name"]), float("nan"))),
            }
            for entry in resolve_criteria(params)
        ],
        "ahp": {
            key: ahp_report[key]
            for key in (
                "weights",
                "weights_geometric",
                "lambda_max",
                "consistency_index",
                "consistency_ratio",
                "consistency_ratio_max",
                "random_index",
                "consistent",
                "weight_spread",
                "degenerate",
            )
            if key in ahp_report
        },
        "rule_3_30_300": {
            "canopy_target_pct": float(
                greening["rule_3_30_300"]["canopy"]["target_pct"]
            ),
            "min_patch_ha": float(
                greening["rule_3_30_300"]["green_space"]["min_patch_ha"]
            ),
            "service_distance_m": float(
                greening["rule_3_30_300"]["green_space"]["service_distance_m"]
            ),
            "detour_distance_m": detour_distance_m(params),
            "public_only": bool(
                greening["rule_3_30_300"]["green_space"]["public_only"]
            ),
            "rule_3_status": RULE_3_STATUS,
        },
        "wetland": {
            "sources": resolve_wetland_sources(params),
            "official_boundary_used": bool(greening["wetland"].get("asset")),
        },
        "caveats": {
            key: " ".join(str(params["caveats"][key]).split())
            for key in (
                "lst_not_air_temp",
                "zonal_not_pixel",
                "sensitivity_reporting",
                "within_epoch_only",
                "euclidean_not_network",
                "mcda_weights_are_judgements",
            )
            if key in params["caveats"]
        },
    }
    for key in ("score_gap_at_cut", "tied_at_cut", "n_priority", "n_eligible"):
        if key in frame.attrs:
            metadata[key] = frame.attrs[key]
    if extra:
        metadata.update(dict(extra))
    return metadata


def write_priority_table(
    frame: "pd.DataFrame",
    path: str | Path,
    params: dict[str, Any],
    ahp_report: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a priority table, under the guard, with its metadata sidecar.

    Both guards run **before** anything touches disk, so a refusal never leaves a
    half-written CSV behind for a later cell to read as if it were valid.

    Args:
        frame: The priority table.
        path: Destination ``.csv`` path.
        params: Parsed params mapping.
        ahp_report: The mapping :func:`ahp_weights` returned.
        extra: Anything else for the sidecar.

    Returns:
        The path written.

    Raises:
        InconsistentJudgements: If the judgements cannot support a product.
        CriteriaIncomplete: If the data cannot support a published ranking.
    """
    require_consistent(ahp_report, params)
    require_complete_criteria(frame, params)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    out = frame.copy()
    out["zone_id"] = out["zone_id"].astype(str)
    out.to_csv(destination, index=False)

    sidecar = destination.with_name(f"{destination.stem}_meta.json")
    metadata = priority_table_metadata(frame, params, ahp_report, extra)
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    return destination


def export_priority_table(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    ahp_report: Mapping[str, Any],
    product: str | None = None,
    folder: str | None = None,
    suffix: str | None = None,
) -> str:
    """Resolve the Drive-consistent export name for a priority table.

    The naming goes through ``exports.export_name`` so that Phase 7's products
    are named the same way every other phase's are, and so a name too long for an
    Earth Engine task description is refused rather than truncated into a
    collision.

    Args:
        frame: The priority table.
        params: Parsed params mapping.
        ahp_report: The mapping :func:`ahp_weights` returned.
        product: Override for ``greening.outputs.ranked_table``.
        folder: Unused; accepted so the signature matches the export helpers.
        suffix: Export-name suffix.

    Returns:
        The sanitised export name.

    Raises:
        InconsistentJudgements: If the judgements cannot support a product.
        CriteriaIncomplete: If the data cannot support a published ranking.
    """
    from colombo_uhi import exports

    require_consistent(ahp_report, params)
    require_complete_criteria(frame, params)

    name = str(
        params["greening"]["outputs"]["ranked_table"] if product is None else product
    )
    start, end = params["uhi"]["utfvi"]["epochs"][str(params["greening"]["epoch"])]
    return exports.export_name(
        product=name,
        aoi="district",
        params=params,
        start_year=int(start),
        end_year=int(end),
        res_m=int(params["greening"]["scale_m"]),
        suffix=suffix,
    )
