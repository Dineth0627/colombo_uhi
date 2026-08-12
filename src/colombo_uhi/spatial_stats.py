"""Spatial statistics: Moran's I, Getis-Ord Gi*, EHSA, and the regression ladder.

This is the layer that answers *where* the heat is, whether the pattern is
moving, and what explains it once spatial dependence is modelled rather than
ignored. It is also where Phase 3's honest limitation is discharged: the driver
OLS there was fitted to 5 000 spatially autocorrelated sampled pixels, so its
standard errors are anti-conservative by construction (``uhi.drivers`` says so
in the config). Nothing in this module inherits that flaw, and no output here
should be described as "confirming" the Phase 3 coefficients.

Products:
    * :func:`build_weights` / :func:`weights_report` - the spatial weights matrix
      the whole module runs on, with **islands detected and repaired** rather
      than silently mis-analysed;
    * :func:`global_morans_i` / :func:`global_morans_table` - is the surface
      clustered at all, with permutation inference;
    * :func:`local_morans` - LISA cluster membership (HH/LL/HL/LH) with FDR
      across the 557 simultaneous local tests;
    * :func:`gi_star` - Getis-Ord Gi* hot and cold spots at 99/95/90 %;
    * :func:`space_time_bins` / :func:`gi_star_panel` /
      :func:`classify_emerging_hotspots` / :func:`ehsa_power_check` - Emerging
      Hot Spot Analysis, implemented here in Python (CLAUDE.md forbids assuming
      ArcGIS), each zone shipping its own detection limit;
    * :func:`ols_fit` / :func:`lagrange_multiplier_tests` / :func:`lm_decision` /
      :func:`spatial_lag_model` / :func:`spatial_error_model` / :func:`gwr_model` /
      :func:`mgwr_model` - the CLAUDE.md escalation ladder;
    * :func:`maup_comparison` - what survives coarsening from 557 GN units to 13
      DS units, including what stops being estimable at all;
    * :func:`landscape_metrics` - patch density, edge density, mean patch size
      and the aggregation index on the green-space class.

Six things in here are easy to get wrong and are therefore pinned by unit tests
and stated loudly in the relevant docstrings:

1. **An island is not a zero.** A polygon with no contiguity neighbour does not
   raise anywhere in the PySAL stack; it produces a statistic computed from an
   empty neighbourhood that lands on the map looking like every other value.
   Colombo's ragged coast and the COD-AB port polygon make this a real risk, so
   :func:`build_weights` counts islands, repairs them per
   ``spatial_stats.weights.island_policy``, and reports the count every time.
2. **Gi* is undefined for negative attributes.** It is a ratio of a
   neighbourhood sum to the global sum. Feed it ``LST_z`` or an anomaly and it
   returns finite, plausible numbers that mean nothing. :func:`gi_star` refuses.
3. **Moran's I and Gi* want DIFFERENT weights.** Moran's I is row-standardised
   so it reads as a correlation with the spatial lag; Gi* is binary-with-self so
   neighbourhood size still counts. Using one matrix for both is a silent error.
4. **557 local tests are 557 tests.** Uncorrected at alpha 0.05 they manufacture
   roughly 28 clusters from noise. Every local statistic here carries a
   Benjamini-Hochberg adjusted p beside the raw one, and both counts are
   reported - the same discipline Phase 4 applied to pixel-wise significance.
5. **"No pattern" is ambiguous.** Over 12 annual bins a Mann-Kendall test cannot
   resolve most real trends, so an EHSA "no pattern" may mean *stable* or *too
   short to tell*. :func:`ehsa_power_check` separates them, reusing the
   detection-limit machinery that made Phase 4's Landsat zero reportable.
6. **A statistic that cannot be estimated must not be estimated.** At DS level
   n=13 against 6 predictors, GWR local coefficients are noise wearing a
   colour ramp. :func:`require_estimable` gates it and the refusal, with its
   reason, becomes a row in the MAUP table.

Design notes:
    * The statistics are computed **analytically in numpy**, not delegated to
      ``esda``. That is deliberate: it keeps the numerical core unit-testable in
      the local pytest environment (which has no PySAL), makes the permutation
      p-values reproducible from ``spatial_stats.random_seed``, and removes a
      dependency on API details that vary between ``esda`` releases.
      :func:`esda_cross_check` and :func:`spreg_cross_check` exist to verify the
      implementations against the reference libraries in Colab, and the tests
      assert agreement via ``pytest.importorskip`` - the same pattern
      :mod:`colombo_uhi.trends` uses against ``pymannkendall``.
    * ``spreg`` and ``mgwr`` ARE used for what they uniquely provide: maximum
      likelihood spatial lag/error estimation, and GWR/MGWR bandwidth search.
    * ``import ee`` (and numpy/pandas/geopandas/scipy/spreg/mgwr) is deferred
      into function bodies so this module, and the local pytest suite, import
      cleanly without ``earthengine-api``.
    * Every constant comes from ``config/params.yaml`` (``spatial_stats``).
    * Validation runs BEFORE the deferred import, so it stays unit-testable.
    * Weights are held as a **dense** ``numpy`` array. At n=557 that is a 2.5 MB
      matrix and every operation is a BLAS call; sparsity would buy nothing and
      cost clarity.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, never at runtime
    import ee
    import geopandas as gpd
    import numpy as np
    import pandas as pd

#: Contiguity schemes :func:`build_weights` knows how to build.
WEIGHTS_SCHEMES: tuple[str, ...] = ("queen", "rook", "knn")

#: What to do about a polygon with no neighbours. See the module docstring.
ISLAND_POLICIES: tuple[str, ...] = ("attach_knn1", "error", "allow")

#: Aggregation levels, matching :func:`colombo_uhi.uhi_metrics.zonal_by_division`.
LEVELS: tuple[str, ...] = ("gn", "ds")

#: Local Moran quadrant codes. Matches ``esda.Moran_Local.q``; pinned by a test
#: so a future ``esda`` release cannot silently relabel a cluster map.
LISA_QUADRANTS: dict[int, str] = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}

#: Label used for a local statistic that did not survive FDR correction.
NOT_SIGNIFICANT = "ns"

#: Column order of the global Moran's I table.
MORANS_COLUMNS: tuple[str, ...] = (
    "level",
    "epoch",
    "variable",
    "status",
    "n",
    "n_missing",
    "morans_i",
    "expectation",
    "z_norm",
    "p_norm",
    "z_sim",
    "p_sim",
    "permutations",
)

#: Column order of the LISA table.
LISA_COLUMNS: tuple[str, ...] = (
    "zone_id",
    "value",
    "spatial_lag",
    "local_i",
    "z_score",
    "p_norm",
    "p_sim",
    "p_adjusted",
    "significant",
    "quadrant",
    "cluster",
)

#: Column order of the Gi* table.
GI_STAR_COLUMNS: tuple[str, ...] = (
    "zone_id",
    "value",
    "gi_z",
    "p_norm",
    "p_sim",
    "p_adjusted",
    "significant",
    "confidence_class",
)

#: Emerging-hot-spot categories this module assigns. The hot forms and their
#: cold mirrors, plus the null. Definitions live in :func:`classify_zone_pattern`
#: and every threshold comes from ``spatial_stats.ehsa``.
EHSA_HOT_CATEGORIES: tuple[str, ...] = (
    "new_hot_spot",
    "consecutive_hot_spot",
    "intensifying_hot_spot",
    "persistent_hot_spot",
    "diminishing_hot_spot",
    "sporadic_hot_spot",
    "oscillating_hot_spot",
    "historical_hot_spot",
)

#: Cold-spot mirrors of :data:`EHSA_HOT_CATEGORIES`.
EHSA_COLD_CATEGORIES: tuple[str, ...] = tuple(
    name.replace("hot_spot", "cold_spot") for name in EHSA_HOT_CATEGORIES
)

#: Assigned when no rule fires: the zone is never significantly hot or cold.
EHSA_NO_PATTERN = "no_pattern"

#: Every category :func:`classify_zone_pattern` can return.
EHSA_CATEGORIES: tuple[str, ...] = (
    *EHSA_HOT_CATEGORIES,
    *EHSA_COLD_CATEGORIES,
    EHSA_NO_PATTERN,
)

#: Column order of the EHSA table.
EHSA_COLUMNS: tuple[str, ...] = (
    "zone_id",
    "name",
    "n_bins",
    "n_hot",
    "n_cold",
    "hot_share",
    "cold_share",
    "final_run",
    "mk_tau",
    "mk_z",
    "mk_p",
    "mk_slope",
    "trend",
    "category",
    "reason",
    "noise_sd",
    "detectable_slope",
    "underpowered",
    "status",
)

#: Column order of the landscape-metrics table.
LANDSCAPE_COLUMNS: tuple[str, ...] = (
    "scheme",
    "year",
    "zone_id",
    "cell_size_m",
    "landscape_area_ha",
    "class_area_ha",
    "class_fraction",
    "n_patches",
    "patch_density_per_100ha",
    "total_edge_m",
    "edge_density_m_per_ha",
    "mean_patch_area_ha",
    "largest_patch_index_pct",
    "aggregation_index_pct",
    # Share of the unit the classifier actually reached. Without it, a date with
    # sparser satellite coverage looks like a date with less green space - which
    # is exactly what Dynamic World 2016 did in Colab run 3.
    "observed_fraction",
)

#: Column order of the MAUP comparison table.
MAUP_COLUMNS: tuple[str, ...] = (
    "statistic",
    "level",
    "n_units",
    "status",
    "value",
    "detail",
    "reason",
)

#: Status values a MAUP row can carry.
MAUP_OK = "ok"
MAUP_NOT_ESTIMABLE = "not_estimable"

#: Sentinel statuses shared with :mod:`colombo_uhi.trends`.
STATUS_OK = "ok"
STATUS_INSUFFICIENT = "insufficient_data"


# =============================================================================
# Pure helpers - parameter resolution (no Earth Engine; unit-tested)
# =============================================================================
def resolve_level(level: str) -> str:
    """Validate an aggregation level.

    Args:
        level: ``"gn"`` (557 divisions) or ``"ds"`` (13).

    Returns:
        The validated level.

    Raises:
        ValueError: If the level is not one of :data:`LEVELS`.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {list(LEVELS)}, got {level!r}")
    return level


def resolve_permutations(
    permutations: int | None, params: dict[str, Any]
) -> int:
    """Validate the permutation count for conditional randomisation.

    Args:
        permutations: Override; ``None`` reads ``spatial_stats.permutations``.
        params: Parsed params mapping.

    Returns:
        The permutation count.

    Raises:
        ValueError: If it is below 99. The finest achievable pseudo p-value is
            ``1 / (permutations + 1)``, so 99 draws cannot resolve anything
            below 0.01 and would make the 99 % confidence class unreachable.
    """
    resolved = int(
        params["spatial_stats"]["permutations"]
        if permutations is None
        else permutations
    )
    if resolved < 99:
        raise ValueError(
            f"permutations must be >= 99, got {resolved}: the smallest pseudo "
            f"p-value obtainable is 1/(n+1) = {1.0 / (resolved + 1):.4f}, which "
            "would make the configured confidence breaks unreachable"
        )
    return resolved


def resolve_seed(seed: int | None, params: dict[str, Any]) -> int:
    """Resolve the random seed for permutation inference.

    Args:
        seed: Override; ``None`` reads ``spatial_stats.random_seed``.
        params: Parsed params mapping.

    Returns:
        The seed. Without one, permutation p-values differ between runs and a
        "significant" cluster can disappear on a re-run with no code change.
    """
    return int(
        params["spatial_stats"]["random_seed"] if seed is None else seed
    )


def resolve_weights_scheme(scheme: str | None, params: dict[str, Any]) -> str:
    """Validate the spatial-weights scheme.

    Args:
        scheme: Override; ``None`` reads ``spatial_stats.weights.scheme``.
        params: Parsed params mapping.

    Returns:
        One of :data:`WEIGHTS_SCHEMES`.

    Raises:
        ValueError: If the scheme is unknown.
    """
    resolved = str(
        params["spatial_stats"]["weights"]["scheme"] if scheme is None else scheme
    )
    if resolved not in WEIGHTS_SCHEMES:
        raise ValueError(
            f"unsupported weights scheme {resolved!r}; expected one of "
            f"{list(WEIGHTS_SCHEMES)}"
        )
    return resolved


def resolve_island_policy(policy: str | None, params: dict[str, Any]) -> str:
    """Validate the island policy.

    Args:
        policy: Override; ``None`` reads
            ``spatial_stats.weights.island_policy``.
        params: Parsed params mapping.

    Returns:
        One of :data:`ISLAND_POLICIES`.

    Raises:
        ValueError: If the policy is unknown.
    """
    resolved = str(
        params["spatial_stats"]["weights"]["island_policy"]
        if policy is None
        else policy
    )
    if resolved not in ISLAND_POLICIES:
        raise ValueError(
            f"unsupported island_policy {resolved!r}; expected one of "
            f"{list(ISLAND_POLICIES)}"
        )
    return resolved


def resolve_regression_predictors(
    predictors: Sequence[str] | None, params: dict[str, Any]
) -> list[str]:
    """Validate the regression predictor list.

    Args:
        predictors: Override; ``None`` reads
            ``spatial_stats.regression.predictors``.
        params: Parsed params mapping.

    Returns:
        Predictor names, order preserved, duplicates collapsed.

    Raises:
        ValueError: If the list is empty, or a predictor repeats the response.
    """
    cfg = params["spatial_stats"]["regression"]
    names = list(cfg["predictors"] if predictors is None else predictors)
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(str(name))
    if not seen:
        raise ValueError("at least one predictor is required")
    response = str(cfg["response"])
    if response in seen:
        raise ValueError(
            f"the response {response!r} appears in the predictor list; a "
            "regression of a variable on itself has R2 = 1 and means nothing"
        )
    return seen


def resolve_green_classes(scheme: str, params: dict[str, Any]) -> list[int]:
    """Class codes counted as green space for a land-cover scheme.

    .. note::
        These are class **codes**, not band indices. WorldCover's legend is
        non-contiguous (10, 20, 30, ... 95, 100), so indexing a list by class
        code would silently select the wrong classes.

    Args:
        scheme: ``"dynamic_world"`` or ``"worldcover"``.
        params: Parsed params mapping.

    Returns:
        Sorted class codes.

    Raises:
        KeyError: If the scheme has no green-class list.
        ValueError: If a listed code is not in that scheme's legend, which would
            silently contribute zero pixels to every metric.
    """
    cfg = params["spatial_stats"]["landscape"]["green_classes"]
    if scheme not in cfg:
        raise KeyError(
            f"no green classes configured for scheme {scheme!r}; "
            f"spatial_stats.landscape.green_classes defines {sorted(cfg)}"
        )
    codes = [int(code) for code in cfg[scheme]]
    if not codes:
        raise ValueError(f"green_classes.{scheme} is empty")

    legend = params["landcover"].get(scheme, {}).get("classes", {})
    if legend:
        unknown = [code for code in codes if code not in legend]
        if unknown:
            raise ValueError(
                f"green_classes.{scheme} lists code(s) {unknown} that are not in "
                f"the {scheme} legend {sorted(legend)}. A code outside the legend "
                "matches no pixels, so every metric would come back zero without "
                "an error anywhere."
            )
    return sorted(codes)


# =============================================================================
# Pure helpers - spatial weights (no Earth Engine; unit-tested)
# =============================================================================
def contiguity_neighbours(
    geometries: Sequence[Any], scheme: str = "queen"
) -> list[list[int]]:
    """Contiguity neighbour lists from polygon geometries.

    Implemented directly on shapely predicates rather than through
    ``libpysal.weights.Queen`` so the result does not depend on a constructor
    signature that has changed between libpysal releases, and so it can be
    tested on a synthetic lattice without PySAL installed. A test cross-checks
    it against libpysal when that is available.

    * **queen** - polygons sharing at least a single boundary point.
    * **rook** - polygons sharing a boundary of non-zero length. On a regular
      lattice this excludes the diagonals; on real administrative polygons the
      distinction is small but not empty.

    Args:
        geometries: Shapely polygon geometries, indexed positionally.
        scheme: ``"queen"`` or ``"rook"``.

    Returns:
        One sorted neighbour index list per input geometry.

    Raises:
        ValueError: If ``scheme`` is not a contiguity scheme.
    """
    if scheme not in ("queen", "rook"):
        raise ValueError(
            f"contiguity_neighbours handles 'queen' and 'rook', got {scheme!r}"
        )

    geoms = list(geometries)
    n = len(geoms)
    neighbours: list[set[int]] = [set() for _ in range(n)]

    # Bounding-box prefilter. O(n^2) on boxes is 310k cheap comparisons at
    # n=557; the expensive predicate then runs only on real candidates.
    boxes = [geometry.bounds for geometry in geoms]
    for i in range(n):
        min_x_i, min_y_i, max_x_i, max_y_i = boxes[i]
        for j in range(i + 1, n):
            min_x_j, min_y_j, max_x_j, max_y_j = boxes[j]
            if max_x_i < min_x_j or max_x_j < min_x_i:
                continue
            if max_y_i < min_y_j or max_y_j < min_y_i:
                continue
            if not geoms[i].intersects(geoms[j]):
                continue
            if scheme == "rook":
                shared = geoms[i].boundary.intersection(geoms[j].boundary)
                if getattr(shared, "length", 0.0) <= 0.0:
                    continue
            neighbours[i].add(j)
            neighbours[j].add(i)

    return [sorted(group) for group in neighbours]


def knn_neighbours(coords: "np.ndarray", k: int) -> list[list[int]]:
    """K-nearest-neighbour lists from projected coordinates.

    .. warning::
        The coordinates must be **projected** (metres). Nearest neighbours
        computed on degrees at 6.9 degrees N are stretched by about 1 % in
        latitude against longitude, and any bandwidth derived from them is in
        degrees, which is not a distance.

    Args:
        coords: ``(n, 2)`` array of projected coordinates.
        k: Neighbours per unit, excluding the unit itself.

    Returns:
        One neighbour index list per row, nearest first.

    Raises:
        ValueError: If ``k`` is not in ``1 .. n - 1``.
    """
    import numpy as np  # Deferred: see module docstring.
    from scipy.spatial import cKDTree  # Deferred: see module docstring.

    array = np.asarray(coords, dtype="float64")
    n = array.shape[0]
    if not 1 <= int(k) <= n - 1:
        raise ValueError(
            f"k must be between 1 and n-1 = {n - 1}, got {k}"
        )
    tree = cKDTree(array)
    # k+1 because the first hit is always the point itself.
    _, indices = tree.query(array, k=int(k) + 1)
    return [[int(j) for j in row[1:]] for row in np.atleast_2d(indices)]


def neighbours_to_matrix(neighbours: Sequence[Sequence[int]]) -> "np.ndarray":
    """Dense binary weights matrix from neighbour lists.

    Args:
        neighbours: One neighbour index list per unit.

    Returns:
        ``(n, n)`` float64 array of 0/1, zero on the diagonal.
    """
    import numpy as np  # Deferred: see module docstring.

    n = len(neighbours)
    matrix = np.zeros((n, n), dtype="float64")
    for i, group in enumerate(neighbours):
        for j in group:
            if int(j) != i:
                matrix[i, int(j)] = 1.0
    return matrix


def row_standardise(matrix: "np.ndarray") -> "np.ndarray":
    """Row-standardise a weights matrix, leaving all-zero rows alone.

    Args:
        matrix: ``(n, n)`` weights.

    Returns:
        A new array whose non-empty rows sum to 1. An island's row stays all
        zero rather than becoming NaN - :func:`build_weights` is responsible for
        islands, and a NaN here would propagate into every statistic.
    """
    import numpy as np  # Deferred: see module docstring.

    array = np.asarray(matrix, dtype="float64").copy()
    sums = array.sum(axis=1)
    nonzero = sums > 0
    array[nonzero] = array[nonzero] / sums[nonzero, None]
    return array


def add_self_neighbours(matrix: "np.ndarray") -> "np.ndarray":
    """Put the focal unit into its own neighbourhood - the Gi* "star".

    Args:
        matrix: ``(n, n)`` binary weights.

    Returns:
        A copy with 1.0 on the diagonal. This is what distinguishes Gi* from Gi;
        without it a hot unit surrounded by cool ones can read as a cold spot.
    """
    import numpy as np  # Deferred: see module docstring.

    array = np.asarray(matrix, dtype="float64").copy()
    np.fill_diagonal(array, 1.0)
    return array


def find_islands(matrix: "np.ndarray") -> list[int]:
    """Indices of units with no neighbours.

    Args:
        matrix: ``(n, n)`` weights.

    Returns:
        Sorted island indices.
    """
    import numpy as np  # Deferred: see module docstring.

    array = np.asarray(matrix, dtype="float64")
    return [int(i) for i in np.flatnonzero(array.sum(axis=1) <= 0)]


def weights_report(matrix: "np.ndarray", ids: Sequence[Any]) -> dict[str, Any]:
    """Summarise a weights matrix, island count first.

    Args:
        matrix: ``(n, n)`` weights.
        ids: Unit identifiers, positionally aligned with the matrix.

    Returns:
        Mapping with ``n``, ``islands``, ``island_ids``, ``min_neighbours``,
        ``mean_neighbours``, ``max_neighbours``, ``pct_nonzero`` and
        ``symmetric``. Carried into every output table's provenance, because a
        cluster map is only as trustworthy as the adjacency behind it.
    """
    import numpy as np  # Deferred: see module docstring.

    array = np.asarray(matrix, dtype="float64")
    counts = (array > 0).sum(axis=1)
    islands = find_islands(array)
    n = int(array.shape[0])
    return {
        "n": n,
        "islands": len(islands),
        "island_ids": [ids[i] for i in islands],
        "min_neighbours": int(counts.min()) if n else 0,
        "mean_neighbours": float(counts.mean()) if n else 0.0,
        "max_neighbours": int(counts.max()) if n else 0,
        "pct_nonzero": float(100.0 * (array > 0).sum() / (n * n)) if n else 0.0,
        "symmetric": bool(np.allclose(array, array.T)),
    }


def build_weights(
    frame: "gpd.GeoDataFrame",
    params: dict[str, Any],
    scheme: str | None = None,
    knn_k: int | None = None,
    island_policy: str | None = None,
    transform: str | None = None,
    id_column: str = "zone_id",
) -> tuple["np.ndarray", list[Any], dict[str, Any]]:
    """The spatial weights matrix every statistic in this module runs on.

    .. warning::
        **Islands are the failure mode that does not announce itself.** A GN
        division with no queen neighbour - Colombo's coast is ragged and the
        COD-AB polygon encloses the port outer harbour - gets a local statistic
        computed over an empty neighbourhood. Nothing raises; the value simply
        appears on the map. Under the default ``attach_knn1`` policy every
        island is given its single nearest neighbour and the count is returned
        in the report, so the repair is visible rather than assumed.

    Args:
        frame: Zone polygons. **Must be in a projected CRS** - see
            :func:`read_zone_geodataframe`, which enforces that.
        params: Parsed params mapping.
        scheme: Override for ``spatial_stats.weights.scheme``.
        knn_k: Override for ``spatial_stats.weights.knn_k``; only used by the
            ``"knn"`` scheme and by the island repair.
        island_policy: Override for ``spatial_stats.weights.island_policy``.
        transform: ``"r"`` for row-standardised, ``"b"`` for binary; defaults to
            ``spatial_stats.weights.transform``.
        id_column: Column holding the zone identifier.

    Returns:
        ``(matrix, ids, report)``. ``matrix`` is dense ``(n, n)`` float64,
        ``ids`` is positionally aligned with it, ``report`` is
        :func:`weights_report` plus the resolved settings.

    Raises:
        ValueError: If the frame lacks ``id_column``, is not projected, or if
            islands remain under ``island_policy="error"``.
    """
    import numpy as np  # Deferred: see module docstring.

    cfg = params["spatial_stats"]["weights"]
    resolved_scheme = resolve_weights_scheme(scheme, params)
    resolved_policy = resolve_island_policy(island_policy, params)
    resolved_transform = str(cfg["transform"] if transform is None else transform)
    k = int(cfg["knn_k"] if knn_k is None else knn_k)

    if id_column not in frame.columns:
        raise ValueError(
            f"the zone frame has no {id_column!r} column; it has "
            f"{sorted(frame.columns)}. Zones are keyed on the pcode - GN names "
            "are NOT unique within Colombo District."
        )
    crs = getattr(frame, "crs", None)
    if crs is not None and getattr(crs, "is_geographic", False):
        raise ValueError(
            f"the zone frame is in a geographic CRS ({crs.to_string()}). "
            "Contiguity would survive but centroid distances, KNN and every GWR "
            "bandwidth would be in degrees, which is not a distance. Reproject "
            f"to {params['crs']['analysis_epsg']} first - "
            "read_zone_geodataframe does this."
        )

    ids = list(frame[id_column])
    centroids = np.column_stack(
        [frame.geometry.centroid.x.to_numpy(), frame.geometry.centroid.y.to_numpy()]
    )

    if resolved_scheme == "knn":
        neighbours = knn_neighbours(centroids, k)
    else:
        neighbours = contiguity_neighbours(list(frame.geometry), resolved_scheme)
    matrix = neighbours_to_matrix(neighbours)

    islands_before = find_islands(matrix)
    repaired = 0
    if islands_before and resolved_policy == "error":
        raise ValueError(
            f"{len(islands_before)} zone(s) have no {resolved_scheme} neighbour: "
            f"{[ids[i] for i in islands_before]}. Every local statistic for them "
            "would be computed over an empty neighbourhood. Set "
            "spatial_stats.weights.island_policy to 'attach_knn1' to link each "
            "to its nearest neighbour, or 'allow' to accept the consequence."
        )
    if islands_before and resolved_policy == "attach_knn1":
        nearest = knn_neighbours(centroids, 1)
        for i in islands_before:
            j = nearest[i][0]
            matrix[i, j] = 1.0
            matrix[j, i] = 1.0  # keep the matrix symmetric
            repaired += 1
    if islands_before and resolved_policy == "allow":
        warnings.warn(
            f"{len(islands_before)} island zone(s) kept with NO neighbours "
            f"({[ids[i] for i in islands_before]}). Their local statistics are "
            "computed over an empty neighbourhood and must not be mapped.",
            stacklevel=2,
        )

    binary = matrix.copy()
    if resolved_transform == "r":
        matrix = row_standardise(matrix)
    elif resolved_transform != "b":
        raise ValueError(
            f"weights transform must be 'r' or 'b', got {resolved_transform!r}"
        )

    report = weights_report(matrix, ids)
    report.update(
        {
            "scheme": resolved_scheme,
            "transform": resolved_transform,
            "island_policy": resolved_policy,
            "islands_before_repair": len(islands_before),
            "islands_repaired": repaired,
            "knn_k": k,
        }
    )
    # The binary form is what Gi* needs; hand it back so callers never have to
    # rebuild the adjacency (and never accidentally feed Gi* a row-standardised
    # matrix, which collapses its variance term).
    report["binary_matrix"] = binary
    return matrix, ids, report


# =============================================================================
# Pure helpers - the statistics themselves (no Earth Engine; unit-tested)
# =============================================================================
def _two_sided_normal_p(z: "np.ndarray") -> "np.ndarray":
    """Two-sided normal p-value, NaN-preserving.

    Args:
        z: Standard normal deviates.

    Returns:
        ``erfc(|z| / sqrt(2))``, with NaN passed through rather than becoming 0.
    """
    import numpy as np  # Deferred: see module docstring.
    from scipy.special import erfc  # Deferred: see module docstring.

    array = np.asarray(z, dtype="float64")
    out = np.full(array.shape, np.nan, dtype="float64")
    finite = np.isfinite(array)
    out[finite] = erfc(np.abs(array[finite]) / math.sqrt(2.0))
    return out


def _pseudo_p(observed: "np.ndarray", simulated: "np.ndarray") -> "np.ndarray":
    """Two-tailed pseudo p-value from a conditional-randomisation reference set.

    Follows the PySAL convention exactly, so our numbers are comparable with
    ``esda``'s: count how many simulated values equal or exceed the observed,
    fold the count to the smaller tail, then apply the ``(count + 1) /
    (permutations + 1)`` correction that keeps the p-value strictly positive.

    Args:
        observed: ``(n,)`` observed statistics.
        simulated: ``(permutations, n)`` simulated statistics.

    Returns:
        ``(n,)`` pseudo p-values in ``(0, 1]``.
    """
    import numpy as np  # Deferred: see module docstring.

    permutations = int(simulated.shape[0])
    larger = (simulated >= observed[None, :]).sum(axis=0).astype("float64")
    larger = np.minimum(larger, permutations - larger)
    return (larger + 1.0) / (permutations + 1.0)


def _conditional_samples(
    n: int, cardinalities: "np.ndarray", permutations: int, seed: int
) -> "np.ndarray":
    """Index matrix for conditional randomisation, built once and reused.

    For a local statistic the null holds unit ``i``'s own value fixed and
    reshuffles the other ``n - 1``. Drawing a fresh permutation per unit per
    replicate would be ``O(permutations * n)`` **per unit**; drawing one
    ``(permutations, n - 1)`` index matrix and re-mapping it around each unit is
    ``O(permutations * n)`` in total. That is the difference between an EHSA
    panel taking seconds and taking a quarter of an hour.

    Args:
        n: Number of units.
        cardinalities: Neighbour count per unit (only the maximum is used).
        permutations: Replicates.
        seed: RNG seed.

    Returns:
        ``(permutations, max_cardinality)`` array of indices into ``0..n-2``.
    """
    import numpy as np  # Deferred: see module docstring.

    rng = np.random.default_rng(seed)
    width = int(cardinalities.max()) if len(cardinalities) else 0
    width = max(width, 1)
    draws = np.empty((permutations, width), dtype="int64")
    for replicate in range(permutations):
        draws[replicate] = rng.permutation(n - 1)[:width]
    return draws


def morans_i(values: "np.ndarray | Sequence[float]", matrix: "np.ndarray") -> float:
    """Global Moran's I, computed analytically.

    ``I = (n / S0) * (z' W z) / (z' z)`` with ``z`` the deviations from the mean
    and ``S0`` the sum of all weights.

    Args:
        values: ``(n,)`` attribute values, no NaN.
        matrix: ``(n, n)`` weights.

    Returns:
        Moran's I.

    Raises:
        ValueError: If the attribute has no variance, or every weight is zero.
    """
    import numpy as np  # Deferred: see module docstring.

    y = np.asarray(values, dtype="float64")
    w = np.asarray(matrix, dtype="float64")
    n = y.size
    z = y - y.mean()
    denominator = float(z @ z)
    if denominator <= 0.0:
        raise ValueError(
            "Moran's I is undefined for a constant attribute (zero variance)"
        )
    s0 = float(w.sum())
    if s0 <= 0.0:
        raise ValueError("Moran's I is undefined for an all-zero weights matrix")
    return float(n / s0 * (z @ (w @ z)) / denominator)


def morans_i_moments(matrix: "np.ndarray", n: int) -> dict[str, float]:
    """Expectation and normality variance of Moran's I.

    The randomisation-free (normality) moments from Cliff & Ord (1981). They are
    emitted beside the permutation p-value so a reader can see whether the two
    inferential routes agree; when they disagree, trust the permutation.

    Args:
        matrix: ``(n, n)`` weights.
        n: Number of units.

    Returns:
        Mapping with ``expectation``, ``variance``, ``s0``, ``s1``, ``s2``.
    """
    import numpy as np  # Deferred: see module docstring.

    w = np.asarray(matrix, dtype="float64")
    s0 = float(w.sum())
    s1 = float(0.5 * ((w + w.T) ** 2).sum())
    s2 = float(((w.sum(axis=1) + w.sum(axis=0)) ** 2).sum())
    expectation = -1.0 / (n - 1)
    numerator = n * n * s1 - n * s2 + 3.0 * s0 * s0
    denominator = (n * n - 1.0) * s0 * s0
    variance = numerator / denominator - expectation * expectation
    return {
        "expectation": expectation,
        "variance": float(variance),
        "s0": s0,
        "s1": s1,
        "s2": s2,
    }


def global_morans_i(
    values: "np.ndarray | Sequence[float]",
    matrix: "np.ndarray",
    params: dict[str, Any],
    permutations: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Global Moran's I with both normality and permutation inference.

    Args:
        values: ``(n,)`` attribute values, no NaN.
        matrix: ``(n, n)`` weights, normally row-standardised.
        params: Parsed params mapping.
        permutations: Override for ``spatial_stats.permutations``.
        seed: Override for ``spatial_stats.random_seed``.

    Returns:
        Mapping with ``i``, ``expectation``, ``variance``, ``z_norm``,
        ``p_norm``, ``z_sim``, ``p_sim``, ``n`` and ``permutations``.
    """
    import numpy as np  # Deferred: see module docstring.

    draws = resolve_permutations(permutations, params)
    rng_seed = resolve_seed(seed, params)

    y = np.asarray(values, dtype="float64")
    w = np.asarray(matrix, dtype="float64")
    n = int(y.size)

    observed = morans_i(y, w)
    moments = morans_i_moments(w, n)
    sd = math.sqrt(moments["variance"]) if moments["variance"] > 0 else float("nan")
    z_norm = (observed - moments["expectation"]) / sd if sd == sd else float("nan")
    p_norm = (
        math.erfc(abs(z_norm) / math.sqrt(2.0)) if z_norm == z_norm else float("nan")
    )

    rng = np.random.default_rng(rng_seed)
    simulated = np.empty(draws, dtype="float64")
    for replicate in range(draws):
        simulated[replicate] = morans_i(rng.permutation(y), w)
    p_sim = float(
        _pseudo_p(np.array([observed]), simulated[:, None])[0]
    )
    sim_sd = float(simulated.std(ddof=1))
    z_sim = (
        float((observed - simulated.mean()) / sim_sd) if sim_sd > 0 else float("nan")
    )

    return {
        "i": observed,
        "expectation": moments["expectation"],
        "variance": moments["variance"],
        "z_norm": float(z_norm),
        "p_norm": float(p_norm),
        "z_sim": z_sim,
        "p_sim": p_sim,
        "n": n,
        "permutations": draws,
    }


def build_morans_frame(
    records: Sequence[Mapping[str, Any]],
) -> "pd.DataFrame":
    """Shape global Moran's I results into the reporting table.

    .. note::
        **An empty result is a SHAPED empty frame, never a column-less one.**
        ``pandas.DataFrame([])`` has a ``RangeIndex`` for columns, so every
        later ``frame["variable"]`` raises ``KeyError`` rather than returning
        nothing - which is exactly how the first Colab run of notebook 05 died
        twice when its covariate exports had not been downloaded. Same reason
        :func:`colombo_uhi.trends.build_mk_frame` exists.

    Args:
        records: One mapping per level x epoch x variable. Missing keys are
            filled with ``None``; unknown keys are dropped.

    Returns:
        ``pandas.DataFrame`` with :data:`MORANS_COLUMNS`.
    """
    import pandas as pd  # Deferred: see module docstring.

    if not records:
        return pd.DataFrame(columns=list(MORANS_COLUMNS))
    frame = pd.DataFrame(list(records))
    for column in MORANS_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[list(MORANS_COLUMNS)]


def local_morans(
    values: "np.ndarray | Sequence[float]",
    matrix: "np.ndarray",
    params: dict[str, Any],
    zone_ids: Sequence[Any] | None = None,
    permutations: int | None = None,
    seed: int | None = None,
    alpha: float | None = None,
    fdr_method: str | None = None,
) -> "pd.DataFrame":
    """Local Moran's I (LISA) with quadrant labels and FDR across zones.

    .. warning::
        **These are ``n`` simultaneous hypothesis tests.** At n=557 and
        alpha 0.05, roughly 28 zones are expected to look like clusters purely
        by chance. The ``significant`` column is therefore driven by the
        Benjamini-Hochberg adjusted p, and ``p_sim`` is kept beside it so both
        counts can be reported as a pair - the same discipline Phase 4 applied
        to pixel-wise versus per-GN significance.

    .. note::
        Quadrants use :data:`LISA_QUADRANTS`, which matches ``esda``'s ``.q``
        coding. ``HL`` means a high value surrounded by low ones - a spatial
        outlier, not a cluster - and must never be drawn in the same colour
        family as ``HH``.

    Args:
        values: ``(n,)`` attribute values, no NaN.
        matrix: ``(n, n)`` weights, normally row-standardised.
        params: Parsed params mapping.
        zone_ids: Identifiers aligned with ``values``; defaults to positions.
        permutations: Override for ``spatial_stats.permutations``.
        seed: Override for ``spatial_stats.random_seed``.
        alpha: Override for ``spatial_stats.lisa.alpha``.
        fdr_method: Override for ``spatial_stats.lisa.fdr_method``.

    Returns:
        ``pandas.DataFrame`` with :data:`LISA_COLUMNS`, one row per zone.

    Raises:
        ValueError: If ``values`` contains NaN, or lengths disagree.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd  # Deferred: see module docstring.

    from colombo_uhi import trends

    cfg = params["spatial_stats"]["lisa"]
    draws = resolve_permutations(permutations, params)
    rng_seed = resolve_seed(seed, params)
    level = float(cfg["alpha"] if alpha is None else alpha)
    method = str(cfg["fdr_method"] if fdr_method is None else fdr_method)

    y = np.asarray(values, dtype="float64")
    w = np.asarray(matrix, dtype="float64")
    n = int(y.size)
    if w.shape != (n, n):
        raise ValueError(f"weights are {w.shape}, expected ({n}, {n})")
    if not np.isfinite(y).all():
        raise ValueError(
            "local Moran's I cannot be computed with missing values: a NaN zone "
            "would silently drop out of its neighbours' lags too. Fill or "
            "exclude the zone (and rebuild the weights) before calling."
        )
    identifiers = list(zone_ids) if zone_ids is not None else list(range(n))
    if len(identifiers) != n:
        raise ValueError(
            f"{len(identifiers)} zone ids for {n} values"
        )

    z = y - y.mean()
    # Anselin (1995): I_i = z_i * sum_j(w_ij z_j) / m2, with m2 the POPULATION
    # second moment. GeoDa and `esda` divide by (n-1) instead, so their Is are
    # ours times n/(n-1) - a constant, identical for every zone. That constant
    # cancels out of the quadrant, the permutation p-value and therefore the
    # cluster map; only the printed magnitude of `local_i` differs.
    # :func:`esda_cross_check` reports the ratio explicitly rather than
    # pretending the two conventions are the same number.
    m2 = float((z * z).sum() / n)
    lag = w @ z
    local = z * lag / m2

    # Conditional randomisation: hold z_i fixed, reshuffle the rest.
    cardinalities = (w > 0).sum(axis=1)
    sample_index = _conditional_samples(n, cardinalities, draws, rng_seed)
    simulated = np.empty((draws, n), dtype="float64")
    all_index = np.arange(n)
    for i in range(n):
        k = int(cardinalities[i])
        if k == 0:
            simulated[:, i] = 0.0
            continue
        others = np.delete(all_index, i)
        picked = others[sample_index[:, :k]]
        weights_i = w[i, w[i] > 0]
        simulated[:, i] = z[i] * (z[picked] @ weights_i) / m2

    p_sim = _pseudo_p(local, simulated)
    # A zone with no neighbours has an all-constant reference distribution, and
    # the pseudo-p formula would hand it the SMALLEST achievable p-value - the
    # island would come out as the most significant cluster on the map. Mark it
    # untested instead; benjamini_hochberg excludes NaN from m and never returns
    # it as significant.
    p_sim = np.where(cardinalities > 0, p_sim, np.nan)

    sim_sd = simulated.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_score = np.where(
            (sim_sd > 0) & (cardinalities > 0),
            (local - simulated.mean(axis=0)) / sim_sd,
            np.nan,
        )
    p_norm = _two_sided_normal_p(z_score)

    reject, adjusted = trends.benjamini_hochberg(
        p_sim, alpha=level, params=params, method=method
    )

    # Quadrants compare the zone and its neighbourhood against the MEAN, so both
    # tests are on the deviations and the threshold is zero.
    quadrant = np.where(
        z > 0,
        np.where(lag > 0, 1, 4),   # HH else HL
        np.where(lag > 0, 2, 3),   # LH else LL
    ).astype("int64")
    cluster = [
        LISA_QUADRANTS[int(q)] if bool(sig) else NOT_SIGNIFICANT
        for q, sig in zip(quadrant, reject)
    ]

    frame = pd.DataFrame(
        {
            "zone_id": identifiers,
            "value": y,
            "spatial_lag": w @ y,
            "local_i": local,
            "z_score": z_score,
            "p_norm": p_norm,
            "p_sim": p_sim,
            "p_adjusted": adjusted,
            "significant": reject,
            "quadrant": quadrant,
            "cluster": cluster,
        }
    )
    return frame[list(LISA_COLUMNS)]


def gi_star(
    values: "np.ndarray | Sequence[float]",
    matrix: "np.ndarray",
    params: dict[str, Any],
    zone_ids: Sequence[Any] | None = None,
    permutations: int | None = None,
    seed: int | None = None,
    alpha: float | None = None,
    fdr_method: str | None = None,
) -> "pd.DataFrame":
    """Getis-Ord Gi* hot- and cold-spot z-scores.

    .. warning::
        **Gi* requires non-negative values.** It is a ratio of a neighbourhood
        sum to the global sum, so a variable that can go negative - an anomaly,
        a z-score, a Sen's slope - produces finite numbers with no meaning.
        Pass raw land surface temperature in degrees Celsius
        (``spatial_stats.response_band``). The guard is on by default via
        ``spatial_stats.gi_star.require_non_negative``.

    .. note::
        The weights must be **binary with the focal unit included** (the
        "star"). ``build_weights`` returns that form in
        ``report["binary_matrix"]``; this function adds the diagonal itself. A
        row-standardised matrix would force every neighbourhood sum to 1 and
        collapse the variance term that lets a large, well-connected
        neighbourhood outweigh a small one.

    Args:
        values: ``(n,)`` non-negative attribute values, no NaN.
        matrix: ``(n, n)`` binary weights. The diagonal is set here.
        params: Parsed params mapping.
        zone_ids: Identifiers aligned with ``values``; defaults to positions.
        permutations: Override for ``spatial_stats.permutations``.
        seed: Override for ``spatial_stats.random_seed``.
        alpha: Override for ``spatial_stats.gi_star.alpha``.
        fdr_method: Override for ``spatial_stats.gi_star.fdr_method``.

    Returns:
        ``pandas.DataFrame`` with :data:`GI_STAR_COLUMNS`.

    Raises:
        ValueError: If a value is negative (and the guard is on), if a value is
            NaN, or if the shapes disagree.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd  # Deferred: see module docstring.

    from colombo_uhi import trends

    cfg = params["spatial_stats"]["gi_star"]
    draws = resolve_permutations(permutations, params)
    rng_seed = resolve_seed(seed, params)
    level = float(cfg["alpha"] if alpha is None else alpha)
    method = str(cfg["fdr_method"] if fdr_method is None else fdr_method)

    y = np.asarray(values, dtype="float64")
    n = int(y.size)
    if not np.isfinite(y).all():
        raise ValueError(
            "Gi* cannot be computed with missing values; fill or exclude the "
            "zone (and rebuild the weights) before calling"
        )
    if bool(cfg.get("require_non_negative", True)) and (y < 0).any():
        negative = int((y < 0).sum())
        raise ValueError(
            f"Gi* is undefined for negative values and {negative} of {n} are "
            "negative. Gi* is a ratio of a neighbourhood sum to the global sum, "
            "so a z-score, an anomaly or a Sen's slope produces finite numbers "
            "that mean nothing. Pass raw LST in degC "
            f"('{params['spatial_stats']['response_band']}'), not a "
            "standardised variable."
        )

    w = add_self_neighbours(np.asarray(matrix, dtype="float64"))
    if w.shape != (n, n):
        raise ValueError(f"weights are {w.shape}, expected ({n}, {n})")
    identifiers = list(zone_ids) if zone_ids is not None else list(range(n))

    mean = float(y.mean())
    # Population standard deviation, per Getis & Ord (1992).
    sd = float(math.sqrt((y * y).sum() / n - mean * mean))
    if sd <= 0.0:
        raise ValueError("Gi* is undefined for a constant attribute")

    w_sum = w.sum(axis=1)
    w_sq_sum = (w * w).sum(axis=1)
    numerator = (w @ y) - mean * w_sum
    denominator = sd * np.sqrt((n * w_sq_sum - w_sum * w_sum) / (n - 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        observed = np.where(denominator > 0, numerator / denominator, np.nan)

    # Conditional randomisation. The focal unit is inside its own neighbourhood,
    # so its own value is held fixed and the other k-1 members are resampled.
    cardinalities = (w > 0).sum(axis=1) - 1  # exclude self
    cardinalities = np.maximum(cardinalities, 0)
    sample_index = _conditional_samples(n, cardinalities, draws, rng_seed)
    simulated = np.empty((draws, n), dtype="float64")
    all_index = np.arange(n)
    for i in range(n):
        k = int(cardinalities[i])
        if k == 0 or not np.isfinite(observed[i]):
            simulated[:, i] = np.nan
            continue
        others = np.delete(all_index, i)
        picked = others[sample_index[:, :k]]
        # The focal unit's own value is held FIXED (that is what conditional
        # randomisation means) and only its neighbours are resampled. Take the
        # self weight off the diagonal explicitly rather than assuming it is the
        # first non-zero entry of the row - it is at column i, which is only
        # position 0 for i = 0.
        row = w[i, :].copy()
        self_term = row[i] * y[i]
        row[i] = 0.0
        neighbour_weights = row[row > 0]
        sums = self_term + (y[picked] @ neighbour_weights)
        simulated[:, i] = (sums - mean * w_sum[i]) / denominator[i]

    p_sim = _pseudo_p(np.nan_to_num(observed), simulated)
    # See local_morans: an island's degenerate reference set would otherwise be
    # handed the smallest achievable p-value and become the hottest spot.
    p_sim = np.where((cardinalities > 0) & np.isfinite(observed), p_sim, np.nan)
    p_norm = _two_sided_normal_p(observed)

    reject, adjusted = trends.benjamini_hochberg(
        p_sim, alpha=level, params=params, method=method
    )
    classes = gi_star_confidence_class(adjusted, observed, params)

    frame = pd.DataFrame(
        {
            "zone_id": identifiers,
            "value": y,
            "gi_z": observed,
            "p_norm": p_norm,
            "p_sim": p_sim,
            "p_adjusted": adjusted,
            "significant": reject,
            "confidence_class": classes,
        }
    )
    return frame[list(GI_STAR_COLUMNS)]


def gi_star_confidence_class(
    p_adjusted: "np.ndarray | Sequence[float]",
    gi_z: "np.ndarray | Sequence[float]",
    params: dict[str, Any],
) -> list[str]:
    """Bin Gi* results into the conventional hot/cold confidence classes.

    The breaks come from ``spatial_stats.gi_star.confidence_breaks`` and are
    applied to the **FDR-adjusted** p-value, not the raw one, so the legend
    means what it says once multiple testing is accounted for.

    Args:
        p_adjusted: Adjusted p-values.
        gi_z: Gi* z-scores, whose sign chooses hot versus cold.
        params: Parsed params mapping.

    Returns:
        One label per unit: ``"hot_99"``, ``"hot_95"``, ``"hot_90"``, the cold
        equivalents, or :data:`NOT_SIGNIFICANT`.

    Raises:
        ValueError: If the breaks are not strictly increasing.
    """
    import numpy as np  # Deferred: see module docstring.

    breaks = [float(b) for b in params["spatial_stats"]["gi_star"]["confidence_breaks"]]
    if any(hi <= lo for lo, hi in zip(breaks, breaks[1:])):
        raise ValueError(
            f"gi_star.confidence_breaks must be strictly increasing, got {breaks}"
        )
    labels = [f"{int(round((1 - b) * 100))}" for b in breaks]

    p = np.asarray(p_adjusted, dtype="float64")
    z = np.asarray(gi_z, dtype="float64")
    out: list[str] = []
    for value, score in zip(p, z):
        if not np.isfinite(value) or not np.isfinite(score) or score == 0.0:
            out.append(NOT_SIGNIFICANT)
            continue
        side = "hot" if score > 0 else "cold"
        label = NOT_SIGNIFICANT
        for threshold, name in zip(breaks, labels):
            if value <= threshold:
                label = f"{side}_{name}"
                break
        out.append(label)
    return out


# =============================================================================
# Pure helpers - Emerging Hot Spot Analysis (no Earth Engine; unit-tested)
# =============================================================================
def space_time_bins(
    long_frame: "pd.DataFrame",
    params: dict[str, Any],
    value_column: str = "mean",
    zone_column: str = "zone_id",
    time_column: str = "year",
    bin_years: int | None = None,
) -> "pd.DataFrame":
    """Reshape a long per-zone series into a zone x time-bin panel.

    .. note::
        **Every zone must appear in every bin.** The panel is later multiplied
        by a weights matrix whose rows are positional, so a zone missing from
        one bin would silently shift every subsequent zone's neighbours. Missing
        cells are therefore filled with NaN and counted, never dropped.

    Args:
        long_frame: Output of
            :func:`colombo_uhi.trends.zonal_annual_series` - one row per zone
            per year.
        params: Parsed params mapping.
        value_column: Column holding the value.
        zone_column: Column holding the zone identifier.
        time_column: Column holding the time index.
        bin_years: Years per bin; defaults to ``spatial_stats.ehsa.bin_years``.

    Returns:
        Wide ``pandas.DataFrame`` indexed by zone id, one column per bin, sorted
        by bin. A ``pandas`` attribute ``panel_report`` carries the missing-cell
        accounting.

    Raises:
        ValueError: If a required column is absent, ``bin_years`` is below 1, or
            a zone/bin pair is duplicated.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd  # Deferred: see module docstring.

    width = int(
        params["spatial_stats"]["ehsa"]["bin_years"] if bin_years is None else bin_years
    )
    if width < 1:
        raise ValueError(f"bin_years must be >= 1, got {width}")

    for column in (zone_column, time_column, value_column):
        if column not in long_frame.columns:
            raise ValueError(
                f"column {column!r} is not in the frame; it has "
                f"{sorted(long_frame.columns)}"
            )
    if long_frame.empty:
        return pd.DataFrame()

    work = long_frame[[zone_column, time_column, value_column]].copy()
    work[time_column] = pd.to_numeric(work[time_column], errors="coerce")
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work = work.dropna(subset=[time_column])

    origin = int(work[time_column].min())
    work["_bin"] = origin + ((work[time_column] - origin) // width).astype("int64") * width

    duplicated = work.duplicated([zone_column, "_bin"]).sum()
    if duplicated and width == 1:
        raise ValueError(
            f"{duplicated} duplicate zone/bin pair(s) at bin_years=1. Two rows "
            "for the same zone and year mean the input was concatenated twice, "
            "or the zone identifier is not unique - check that the join key is "
            "the pcode and not the GN name."
        )

    panel = work.pivot_table(
        index=zone_column, columns="_bin", values=value_column, aggfunc="mean"
    ).sort_index(axis=1)

    missing = int(panel.isna().sum().sum())
    total = int(panel.size)
    if missing:
        warnings.warn(
            f"{missing} of {total} zone/bin cells ({missing / max(total, 1):.1%}) "
            "are missing and were filled with NaN. Gi* cannot be computed for a "
            "bin with missing zones, so those bins are dropped by "
            "gi_star_panel with a warning.",
            stacklevel=2,
        )
    panel.attrs["panel_report"] = {
        "zones": int(panel.shape[0]),
        "bins": int(panel.shape[1]),
        "bin_years": width,
        "missing_cells": missing,
        "complete_bins": int((~panel.isna().any(axis=0)).sum()),
    }
    panel.attrs["bins"] = [int(b) for b in panel.columns]
    return panel


def gi_star_panel(
    panel: "pd.DataFrame",
    matrix: "np.ndarray",
    params: dict[str, Any],
    zone_ids: Sequence[Any] | None = None,
    permutations: int | None = None,
    seed: int | None = None,
) -> "pd.DataFrame":
    """Gi* z-score per zone per time bin - the EHSA input.

    Each bin is standardised against **its own** spatial mean and standard
    deviation, which is exactly why EHSA is robust to a common-mode shift: a
    sensor step or a hot year raises every zone together and cancels out of the
    z-score. What survives is the change in spatial *pattern*.

    Args:
        panel: Zone x bin frame from :func:`space_time_bins`.
        matrix: ``(n, n)`` **binary** weights; the diagonal is added internally.
        params: Parsed params mapping.
        zone_ids: Weights row order. Defaults to the panel index, which is only
            correct if the panel was reindexed onto the weights order first.
        permutations: Override for ``spatial_stats.permutations``.
        seed: Override for ``spatial_stats.random_seed``.

    Returns:
        Zone x bin ``pandas.DataFrame`` of Gi* z-scores. Bins with any missing
        zone are dropped and reported in ``attrs["dropped_bins"]``.

    Raises:
        ValueError: If the panel index does not match ``zone_ids``.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd  # Deferred: see module docstring.

    if panel.empty:
        return pd.DataFrame()

    order = list(zone_ids) if zone_ids is not None else list(panel.index)
    if sorted(map(str, order)) != sorted(map(str, panel.index)):
        raise ValueError(
            "the panel's zones do not match the weights' zones. Reindex the "
            "panel onto the weights order before calling; a positional "
            "mismatch silently gives every zone its neighbour's statistic."
        )
    aligned = panel.reindex(order)

    columns: dict[Any, "np.ndarray"] = {}
    dropped: list[Any] = []
    for column in aligned.columns:
        values = aligned[column].to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            dropped.append(column)
            continue
        if float(values.min()) < 0.0:
            raise ValueError(
                f"bin {column} contains negative values, which Gi* cannot use. "
                "The EHSA panel must carry raw LST in degC, not anomalies."
            )
        result = gi_star(
            values,
            matrix,
            params,
            zone_ids=order,
            permutations=permutations,
            seed=seed,
        )
        columns[column] = result["gi_z"].to_numpy(dtype="float64")

    if dropped:
        warnings.warn(
            f"{len(dropped)} time bin(s) dropped because at least one zone had "
            f"no value: {dropped}. EHSA is computed on the remaining "
            f"{len(columns)} bin(s), and the shorter series has correspondingly "
            "less power - see ehsa_power_check.",
            stacklevel=2,
        )

    frame = pd.DataFrame(columns, index=order)
    frame.attrs["dropped_bins"] = dropped
    frame.attrs["bins"] = [int(c) for c in frame.columns]
    return frame


def classify_zone_pattern(
    z_series: "np.ndarray | Sequence[float]",
    trend: str,
    trend_significant: bool,
    params: dict[str, Any],
    require_final_bin: bool | None = None,
) -> tuple[str, str]:
    """Assign one zone's Gi* z-series to an emerging-hot-spot category.

    The rules, in the order they are tested. ``hot`` means the bin's Gi* z-score
    exceeds ``ehsa.hot_z``; ``share`` is the fraction of bins that are hot.

    ==============================  ====================================================
    Category                        Rule (tested in this order)
    ==============================  ====================================================
    ``oscillating_hot_spot``        **final bin hot**, having been significantly COLD at
                                    some earlier bin
    ``new_hot_spot``                hot only in the final ``new_tail_bins`` bin(s),
                                    never before
    ``consecutive_hot_spot``        an unbroken final run of >= ``consecutive_min_run``
                                    hot bins, never hot before it, and still below
                                    ``persistent_share`` of all bins
    ``sporadic_hot_spot``           **final bin hot**, on-and-off hot below
                                    ``sporadic_max_share``, never cold
    ------------------------------  ----------------------------------------------------
    ``historical_hot_spot``         hot in >= ``persistent_share`` of bins but NOT in
                                    the last ``historical_recent_bins``
    ``intensifying_hot_spot``       hot in >= ``persistent_share`` of bins AND a
                                    significant increasing Mann-Kendall trend
    ``diminishing_hot_spot``        hot in >= ``persistent_share`` of bins AND a
                                    significant decreasing trend
    ``persistent_hot_spot``         hot in >= ``persistent_share`` of bins, no
                                    significant trend
    ==============================  ====================================================

    The four above the rule are the "what is it doing NOW" categories and need
    the final bin to be significant; the four below need only the share, so a
    zone that has been hot throughout still classifies even if the last bin
    happens to fall short.

    The cold-spot categories are the exact mirror. A zone that is never
    significantly hot or cold is :data:`EHSA_NO_PATTERN` - which, over a short
    series, may mean *stable* or *unresolvable*. :func:`ehsa_power_check` is what
    tells those apart; this function deliberately does not guess.

    .. note::
        The hot/cold side is decided by the **final** bin whenever that bin is
        significant, and only falls back to the majority when it is not. Taking
        the majority first would misfile exactly the zones that matter most: a
        division that was a cold spot for a decade and is a hot spot now is an
        *oscillating hot spot*, not a cold spot.

    .. warning::
        Under ``ehsa.require_final_bin`` (**on** by default) the first four
        categories in the table can only fire when the final bin is itself
        significant; everything else falls through to
        :data:`EHSA_NO_PATTERN`. That is the published definition, and it is
        what makes this an *emerging* hot-spot analysis rather than an
        "was ever hot" map. With the flag off, ``sporadic`` alone absorbed 329
        of 557 GN divisions on the 26-bin MODIS series in Colab run 1 - a
        category covering most of the study area has stopped discriminating.

    Args:
        z_series: Gi* z-scores in bin order, oldest first.
        trend: ``"increasing"``, ``"decreasing"`` or anything else. These are
            the labels :func:`colombo_uhi.trends.mk_comparison` emits.
        trend_significant: Whether the Mann-Kendall test rejected.
        params: Parsed params mapping.
        require_final_bin: Override for ``spatial_stats.ehsa.require_final_bin``.

    Returns:
        ``(category, reason)``. The reason is a short human-readable string that
        travels into the output table so a map's legend can be audited.

    Raises:
        ValueError: If the series is empty.
    """
    import numpy as np  # Deferred: see module docstring.

    cfg = params["spatial_stats"]["ehsa"]
    hot_z = float(cfg["hot_z"])
    cold_z = float(cfg["cold_z"])
    tail = int(cfg["new_tail_bins"])
    min_run = int(cfg["consecutive_min_run"])
    persistent = float(cfg["persistent_share"])
    sporadic_max = float(cfg["sporadic_max_share"])
    recent = int(cfg["historical_recent_bins"])
    final_required = bool(
        cfg["require_final_bin"] if require_final_bin is None else require_final_bin
    )

    z = np.asarray(z_series, dtype="float64")
    n = int(z.size)
    if n == 0:
        raise ValueError("cannot classify an empty Gi* series")

    hot = z > hot_z
    cold = z < cold_z
    n_hot, n_cold = int(hot.sum()), int(cold.sum())

    if n_hot == 0 and n_cold == 0:
        return EHSA_NO_PATTERN, "never significantly hot or cold"

    # Work in the "hot" frame, then mirror. THE FINAL BIN DECIDES: a zone that
    # was a cold spot for a decade and is a hot spot now is an oscillating HOT
    # spot, and choosing the side by majority would file it as a cold spot and
    # lose the very transition the analysis exists to find. Only when the final
    # bin is not significant does the majority break the tie.
    if bool(hot[-1]):
        side_hot = True
    elif bool(cold[-1]):
        side_hot = False
    else:
        side_hot = n_hot >= n_cold

    flag = hot if side_hot else cold
    opposite = cold if side_hot else hot
    suffix = "hot_spot" if side_hot else "cold_spot"
    word = "hot" if side_hot else "cold"
    share = float(flag.sum()) / n
    increasing = "increasing" if side_hot else "decreasing"
    decreasing = "decreasing" if side_hot else "increasing"

    # Final consecutive run of significant bins.
    run = 0
    for value in flag[::-1]:
        if not value:
            break
        run += 1

    # Oscillating is tested FIRST among the "significant in the final bin" rules.
    # A zone that was a cold spot and is now a hot spot also satisfies "never a
    # hot spot before", so it would otherwise be filed as merely New - and the
    # flip, which is the most informative thing about it, would be lost.
    if flag[-1] and opposite.any():
        return (
            f"oscillating_{suffix}",
            f"{word} in the final bin, having been the opposite earlier",
        )

    if run and run <= tail and int(flag.sum()) == run:
        return f"new_{suffix}", f"{word} only in the final {run} bin(s)"

    # `share < persistent` matters: without it a series that is hot in EVERY bin
    # is trivially "one unbroken final run" and would be filed as consecutive,
    # swallowing the persistent / intensifying / diminishing cases entirely.
    if run >= min_run and int(flag.sum()) == run and share < persistent:
        return (
            f"consecutive_{suffix}",
            f"an unbroken final run of {run} {word} bin(s), never before",
        )

    if share >= persistent:
        if not flag[-recent:].any():
            return (
                f"historical_{suffix}",
                f"{word} in {share:.0%} of bins but not in the last "
                f"{recent} bin(s)",
            )
        if trend_significant and trend == increasing:
            return (
                f"intensifying_{suffix}",
                f"{word} in {share:.0%} of bins and significantly {increasing}",
            )
        if trend_significant and trend == decreasing:
            return (
                f"diminishing_{suffix}",
                f"{word} in {share:.0%} of bins but significantly {decreasing}",
            )
        return (
            f"persistent_{suffix}",
            f"{word} in {share:.0%} of bins with no significant trend",
        )

    if share < sporadic_max and not opposite.any() and (flag[-1] or not final_required):
        return (
            f"sporadic_{suffix}",
            f"on and off {word} in {share:.0%} of bins, never the opposite"
            + ("" if flag[-1] else " (final bin not significant)"),
        )

    # Below the persistent share and NOT significant in the final bin. Under the
    # published taxonomy that is no pattern: the zone has been significant at
    # some point but is not doing anything now, and an EMERGING hot-spot map
    # that colours it in is answering "was it ever hot" instead.
    if final_required and not flag[-1]:
        return (
            EHSA_NO_PATTERN,
            f"{word} in only {share:.0%} of bins and not in the final bin",
        )

    if trend_significant and trend == increasing:
        return (
            f"intensifying_{suffix}",
            f"{word} in {share:.0%} of bins and significantly {increasing}",
        )
    if trend_significant and trend == decreasing:
        return (
            f"diminishing_{suffix}",
            f"{word} in {share:.0%} of bins but significantly {decreasing}",
        )
    return (
        f"sporadic_{suffix}",
        f"intermittently {word} in {share:.0%} of bins",
    )


#: Memo for :func:`unit_noise_detection_limit`, keyed on ``(n_bins, alpha)``.
_UNIT_LIMIT_CACHE: dict[tuple[int, float], float] = {}


def unit_noise_detection_limit(n_bins: int, alpha: float = 0.05) -> float:
    """Smallest Mann-Kendall-detectable slope for UNIT noise, memoised.

    :func:`colombo_uhi.trends.minimum_detectable_slope` bisects over a Monte
    Carlo power simulation, which costs thousands of Mann-Kendall evaluations.
    Calling it once per zone would be hundreds of times that, for no
    information: **Mann-Kendall is scale invariant**, so detecting a slope
    ``s`` in noise of standard deviation ``sigma`` is exactly the problem of
    detecting ``s / sigma`` in unit noise. The limit therefore scales linearly
    with the noise, and only the unit-noise value has to be simulated - once per
    series length.

    Args:
        n_bins: Number of time bins.
        alpha: Two-sided significance level.

    Returns:
        The detectable slope per bin at unit noise; multiply by a zone's own
        noise standard deviation to get its limit. NaN if ``n_bins`` is below 3.
    """
    from colombo_uhi import trends

    key = (int(n_bins), float(alpha))
    if key in _UNIT_LIMIT_CACHE:
        return _UNIT_LIMIT_CACHE[key]
    if int(n_bins) < 3:
        _UNIT_LIMIT_CACHE[key] = float("nan")
        return float("nan")
    limit = float(
        trends.minimum_detectable_slope(int(n_bins), 1.0, alpha=float(alpha))[
            "slope_per_year"
        ]
    )
    _UNIT_LIMIT_CACHE[key] = limit
    return limit


def classify_emerging_hotspots(
    gi_panel: "pd.DataFrame",
    params: dict[str, Any],
    names: Mapping[Any, str] | None = None,
    mk_alpha: float | None = None,
    power_check: bool | None = None,
    require_final_bin: bool | None = None,
) -> "pd.DataFrame":
    """Emerging Hot Spot Analysis over a Gi* space-time panel.

    Runs Mann-Kendall on each zone's Gi* z-series with
    :func:`colombo_uhi.trends.mk_comparison` - the same tested implementation
    Phase 4 used, including its two corrections to the community tutorial - then
    applies :func:`classify_zone_pattern`.

    .. warning::
        Over a short panel most zones will land in :data:`EHSA_NO_PATTERN`, and
        that is **not** evidence of stability. The ``underpowered`` column says
        whether the series could have detected a trend at all. Report the two
        together, exactly as Phase 4 reports its Landsat zero beside a detection
        limit of 0.34 degC/yr.

    Args:
        gi_panel: Zone x bin Gi* z-scores from :func:`gi_star_panel`.
        params: Parsed params mapping.
        names: Optional zone id -> display name mapping.
        mk_alpha: Override for ``spatial_stats.ehsa.mk_alpha``.
        power_check: Override for ``spatial_stats.ehsa.power_check``.
        require_final_bin: Override for
            ``spatial_stats.ehsa.require_final_bin``. Pass ``False`` to
            reproduce the looser pre-run-1 classification as a sensitivity.

    Returns:
        ``pandas.DataFrame`` with :data:`EHSA_COLUMNS`, one row per zone.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd  # Deferred: see module docstring.

    from colombo_uhi import trends

    cfg = params["spatial_stats"]["ehsa"]
    level = float(cfg["mk_alpha"] if mk_alpha is None else mk_alpha)
    check_power = bool(cfg["power_check"] if power_check is None else power_check)
    hot_z = float(cfg["hot_z"])
    cold_z = float(cfg["cold_z"])

    if gi_panel.empty:
        return pd.DataFrame(columns=list(EHSA_COLUMNS))

    bins = [int(c) for c in gi_panel.columns]
    rows: list[dict[str, Any]] = []
    for zone_id, series in gi_panel.iterrows():
        z = series.to_numpy(dtype="float64")
        result = trends.mk_comparison(
            z, params, years=np.array(bins, dtype="int64"),
            label=str(zone_id), series="gi_star_z",
        )
        original = result[result["test"] == "original"]
        record = original.iloc[0] if not original.empty else None

        p_value = float(record["p"]) if record is not None else float("nan")
        trend_word = str(record["trend"]) if record is not None else "no trend"
        significant = bool(p_value == p_value and p_value < level)
        status = str(record["status"]) if record is not None else STATUS_INSUFFICIENT

        category, reason = classify_zone_pattern(
            z, trend_word, significant, params, require_final_bin=require_final_bin
        )

        hot = z > hot_z
        cold = z < cold_z
        run = 0
        final = hot if hot.sum() >= cold.sum() else cold
        for value in final[::-1]:
            if not value:
                break
            run += 1

        row: dict[str, Any] = {
            "zone_id": zone_id,
            "name": (names or {}).get(zone_id),
            "n_bins": int(z.size),
            "n_hot": int(hot.sum()),
            "n_cold": int(cold.sum()),
            "hot_share": float(hot.sum()) / z.size,
            "cold_share": float(cold.sum()) / z.size,
            "final_run": run,
            "mk_tau": float(record["tau"]) if record is not None else float("nan"),
            "mk_z": float(record["z"]) if record is not None else float("nan"),
            "mk_p": p_value,
            "mk_slope": float(record["slope"]) if record is not None else float("nan"),
            "trend": trend_word,
            "category": category,
            "reason": reason,
            "status": status,
        }
        if check_power:
            # Noise estimated from first differences: sd(diff)/sqrt(2) is the
            # residual scale of a series about its own trend, and unlike a plain
            # sd it is not inflated by the trend the test is looking for.
            noise_sd = (
                float(np.nanstd(np.diff(z)) / math.sqrt(2.0))
                if z.size > 1
                else float("nan")
            )
            row["noise_sd"] = noise_sd
            unit_limit = unit_noise_detection_limit(int(z.size), alpha=level)
            detectable = (
                unit_limit * noise_sd
                if unit_limit == unit_limit and noise_sd == noise_sd and noise_sd > 0
                else float("nan")
            )
            row["detectable_slope"] = detectable
            observed = abs(row["mk_slope"]) if row["mk_slope"] == row["mk_slope"] else 0.0
            row["underpowered"] = bool(
                category == EHSA_NO_PATTERN
                and detectable == detectable
                and observed < detectable
            )
        else:
            row["noise_sd"] = float("nan")
            row["detectable_slope"] = float("nan")
            row["underpowered"] = False
        rows.append(row)

    frame = pd.DataFrame(rows)
    return frame[list(EHSA_COLUMNS)]


def ehsa_power_check(
    ehsa_frame: "pd.DataFrame", params: dict[str, Any]
) -> dict[str, Any]:
    """Summarise how much of an EHSA result is a power limitation.

    Args:
        ehsa_frame: Output of :func:`classify_emerging_hotspots`.
        params: Parsed params mapping.

    Returns:
        Mapping with ``n_zones``, ``n_no_pattern``, ``n_underpowered``,
        ``underpowered_share_of_no_pattern``, ``median_detectable_slope`` and a
        ready-to-print ``verdict`` sentence.
    """
    import numpy as np  # Deferred: see module docstring.

    if ehsa_frame.empty:
        return {
            "n_zones": 0,
            "n_no_pattern": 0,
            "n_underpowered": 0,
            "underpowered_share_of_no_pattern": float("nan"),
            "median_detectable_slope": float("nan"),
            "verdict": "no zones to assess",
        }

    n_zones = int(len(ehsa_frame))
    no_pattern = ehsa_frame["category"] == EHSA_NO_PATTERN
    n_no_pattern = int(no_pattern.sum())
    n_under = int(ehsa_frame["underpowered"].fillna(False).astype(bool).sum())
    share = float(n_under / n_no_pattern) if n_no_pattern else float("nan")
    median = float(np.nanmedian(ehsa_frame["detectable_slope"].to_numpy("float64")))

    if n_no_pattern == 0:
        verdict = "every zone carries a pattern; power is not the binding constraint"
    elif share == share and share >= 0.5:
        verdict = (
            f"{n_under} of the {n_no_pattern} 'no pattern' zones are UNDERPOWERED "
            f"(median detectable Gi* trend {median:.3f}/bin). Report this as a "
            "limit of the series, NOT as spatial stability."
        )
    else:
        verdict = (
            f"{n_under} of {n_no_pattern} 'no pattern' zones are underpowered; "
            "the remainder are genuinely without a detectable pattern"
        )
    return {
        "n_zones": n_zones,
        "n_no_pattern": n_no_pattern,
        "n_underpowered": n_under,
        "underpowered_share_of_no_pattern": share,
        "median_detectable_slope": median,
        "verdict": verdict,
    }


# =============================================================================
# Pure helpers - the regression ladder (no Earth Engine; unit-tested)
# =============================================================================
def require_estimable(
    n: int, k: int, params: dict[str, Any], statistic: str = "model"
) -> dict[str, Any]:
    """Decide whether a local model may be fitted at all.

    .. warning::
        This is the guard behind the DS-level decision. Thirteen Divisional
        Secretariat divisions against six predictors leaves seven residual
        degrees of freedom for a **global** model, and a GWR - which fits one
        weighted regression per unit - has effectively none. It still returns
        numbers, and those numbers still make a colourful map. Refusing to fit
        it, and saying why, is the result.

    Args:
        n: Number of observations.
        k: Number of predictors, excluding the intercept.
        params: Parsed params mapping.
        statistic: Name used in the refusal message.

    Returns:
        Mapping with ``estimable``, ``n``, ``k``, ``required`` and ``reason``.

    Raises:
        ValueError: If ``k`` is below 1.
    """
    if int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    per = float(params["spatial_stats"]["regression"]["min_obs_per_predictor"])
    required = int(math.ceil(per * int(k)))
    ok = int(n) >= required
    reason = (
        ""
        if ok
        else (
            f"{statistic} needs at least {required} observations for {k} "
            f"predictors ({per:g} per predictor) and this level has {n}. "
            "Fitting it would return coefficients indistinguishable from noise."
        )
    )
    return {
        "estimable": ok,
        "n": int(n),
        "k": int(k),
        "required": required,
        "reason": reason,
    }


def variance_inflation_factors(
    design: "np.ndarray", names: Sequence[str]
) -> "pd.DataFrame":
    """Variance inflation factors for a predictor matrix.

    .. note::
        Expect this to fire on NDVI against NDBI. Phase 3 measured the symptom
        already: NDVI's partial coefficient flipped sign in 5 of 26 years while
        its bivariate correlation stayed a clean -0.51. That is collinearity in
        a dense city, and it is the reason CLAUDE.md's ladder ends at MGWR
        rather than at a single multivariate coefficient.

    Args:
        design: ``(n, k)`` predictor matrix, **without** an intercept column.
        names: Predictor names, aligned with the columns.

    Returns:
        ``pandas.DataFrame`` with ``predictor``, ``vif`` and ``r_squared``,
        sorted by VIF descending.

    Raises:
        ValueError: If the shapes disagree, or there are fewer than 2 predictors.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd  # Deferred: see module docstring.

    x = np.asarray(design, dtype="float64")
    if x.ndim != 2:
        raise ValueError(f"design must be 2-D, got shape {x.shape}")
    if x.shape[1] != len(names):
        raise ValueError(
            f"{x.shape[1]} design columns for {len(names)} names"
        )
    if x.shape[1] < 2:
        raise ValueError("VIF needs at least 2 predictors")

    rows: list[dict[str, Any]] = []
    for j, name in enumerate(names):
        y = x[:, j]
        others = np.delete(x, j, axis=1)
        design_j = np.column_stack([np.ones(others.shape[0]), others])
        beta, *_ = np.linalg.lstsq(design_j, y, rcond=None)
        residual = y - design_j @ beta
        total = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float((residual**2).sum()) / total if total > 0 else float("nan")
        vif = 1.0 / (1.0 - r2) if r2 == r2 and r2 < 1.0 else float("inf")
        rows.append({"predictor": name, "vif": vif, "r_squared": r2})
    return (
        pd.DataFrame(rows)
        .sort_values("vif", ascending=False)
        .reset_index(drop=True)
    )


def ols_fit(
    y: "np.ndarray", design: "np.ndarray", names: Sequence[str]
) -> dict[str, Any]:
    """Ordinary least squares with the diagnostics the ladder needs.

    Implemented in numpy so the first rung of CLAUDE.md's escalation path, and
    the residuals every later rung is tested against, exist without ``spreg``.
    :func:`spreg_cross_check` verifies the agreement in Colab.

    Args:
        y: ``(n,)`` response.
        design: ``(n, k)`` predictors, **without** an intercept column.
        names: Predictor names.

    Returns:
        Mapping with ``coefficients`` (a frame of estimate/se/t/p), ``residuals``,
        ``fitted``, ``r_squared``, ``adj_r_squared``, ``sigma2``, ``aic``,
        ``n``, ``k``.

    Raises:
        ValueError: If the shapes disagree or the system is rank deficient.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd  # Deferred: see module docstring.

    response = np.asarray(y, dtype="float64")
    x = np.asarray(design, dtype="float64")
    n, k = x.shape
    if response.size != n:
        raise ValueError(f"{response.size} responses for {n} design rows")
    if x.shape[1] != len(names):
        raise ValueError(f"{x.shape[1]} design columns for {len(names)} names")

    full = np.column_stack([np.ones(n), x])
    if np.linalg.matrix_rank(full) < full.shape[1]:
        raise ValueError(
            "the design matrix is rank deficient - two predictors are exactly "
            "collinear. Check variance_inflation_factors before fitting."
        )

    xtx_inv = np.linalg.inv(full.T @ full)
    beta = xtx_inv @ full.T @ response
    fitted = full @ beta
    residual = response - fitted
    rss = float(residual @ residual)
    dof = n - full.shape[1]
    sigma2 = rss / dof
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    t = beta / se

    from scipy import stats  # Deferred: see module docstring.

    p = 2.0 * stats.t.sf(np.abs(t), dof)
    tss = float(((response - response.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    adj = 1.0 - (1.0 - r2) * (n - 1) / dof if dof > 0 else float("nan")
    aic = n * math.log(rss / n) + 2.0 * full.shape[1]

    coefficients = pd.DataFrame(
        {
            "term": ["intercept", *names],
            "estimate": beta,
            "std_error": se,
            "t_statistic": t,
            "p_value": p,
        }
    )
    return {
        "coefficients": coefficients,
        "residuals": residual,
        "fitted": fitted,
        "r_squared": float(r2),
        "adj_r_squared": float(adj),
        "sigma2": float(sigma2),
        "aic": float(aic),
        "n": int(n),
        "k": int(k),
        "design": full,
    }


def lagrange_multiplier_tests(
    ols: Mapping[str, Any], y: "np.ndarray", matrix: "np.ndarray"
) -> dict[str, Any]:
    """Anselin's Lagrange Multiplier tests for spatial lag and spatial error.

    Implements the classic and robust forms (Anselin 1988; Anselin, Bera, Florax
    & Yoon 1996) in numpy, so the decision that selects the next model is
    reproducible and unit-tested rather than read off a printed ``spreg``
    summary. :func:`spreg_cross_check` verifies agreement.

    Args:
        ols: Output of :func:`ols_fit`.
        y: ``(n,)`` response.
        matrix: ``(n, n)`` **row-standardised** weights.

    Returns:
        Mapping with ``lm_error``, ``lm_lag``, ``rlm_error``, ``rlm_lag``,
        ``lm_sarma``, each a ``(statistic, p_value)`` pair, plus ``moran_i`` of
        the residuals and its permutation-free z.
    """
    import numpy as np  # Deferred: see module docstring.
    from scipy import stats  # Deferred: see module docstring.

    w = np.asarray(matrix, dtype="float64")
    response = np.asarray(y, dtype="float64")
    e = np.asarray(ols["residuals"], dtype="float64")
    full = np.asarray(ols["design"], dtype="float64")
    n = int(e.size)

    s2 = float(e @ e) / n
    t_trace = float(np.trace((w.T + w) @ w))

    # M = I - X (X'X)^-1 X', the OLS residual-maker.
    xtx_inv = np.linalg.inv(full.T @ full)
    hat = full @ xtx_inv @ full.T
    m = np.eye(n) - hat

    wy = w @ response
    wxb = w @ ols["fitted"]
    d = float(wxb @ m @ wxb) / s2 + t_trace

    e_we = float(e @ (w @ e)) / s2
    e_wy = float(e @ wy) / s2

    lm_error = e_we**2 / t_trace if t_trace > 0 else float("nan")
    lm_lag = e_wy**2 / d if d > 0 else float("nan")

    ratio = t_trace / d if d > 0 else float("nan")
    rlm_error_den = t_trace * (1.0 - ratio)
    rlm_error = (
        (e_we - ratio * e_wy) ** 2 / rlm_error_den
        if rlm_error_den > 0
        else float("nan")
    )
    rlm_lag_den = d - t_trace
    rlm_lag = (e_wy - e_we) ** 2 / rlm_lag_den if rlm_lag_den > 0 else float("nan")
    lm_sarma = (
        lm_lag + rlm_error
        if lm_lag == lm_lag and rlm_error == rlm_error
        else float("nan")
    )

    def _pair(statistic: float, df: int) -> tuple[float, float]:
        if statistic != statistic:
            return float("nan"), float("nan")
        return float(statistic), float(stats.chi2.sf(statistic, df))

    residual_i = morans_i(e, w)
    moments = morans_i_moments(w, n)
    sd = math.sqrt(moments["variance"]) if moments["variance"] > 0 else float("nan")
    residual_z = (
        (residual_i - moments["expectation"]) / sd if sd == sd else float("nan")
    )

    return {
        "lm_error": _pair(lm_error, 1),
        "lm_lag": _pair(lm_lag, 1),
        "rlm_error": _pair(rlm_error, 1),
        "rlm_lag": _pair(rlm_lag, 1),
        "lm_sarma": _pair(lm_sarma, 2),
        "moran_i": float(residual_i),
        "moran_z": float(residual_z),
        "moran_p": float(
            math.erfc(abs(residual_z) / math.sqrt(2.0))
            if residual_z == residual_z
            else float("nan")
        ),
        "n": n,
    }


def lm_decision(
    diagnostics: Mapping[str, Any],
    params: dict[str, Any],
    alpha: float | None = None,
) -> dict[str, Any]:
    """Anselin's decision rule: which spatial model the LM tests point to.

    The rule, applied in order:

    1. Neither ``lm_lag`` nor ``lm_error`` significant -> **keep OLS**. The
       residual dependence is not strong enough to justify a spatial model.
    2. Exactly one significant -> fit **that** model.
    3. Both significant -> consult the robust forms, which are each constructed
       to be insensitive to the other alternative. Fit whichever robust test is
       significant; if both are, fit the one with the larger statistic.

    Encoding it here rather than eyeballing a summary table means the model
    choice is reproducible and can be stated in the report as a rule rather than
    as a judgement.

    Args:
        diagnostics: Output of :func:`lagrange_multiplier_tests`.
        params: Parsed params mapping.
        alpha: Override for ``spatial_stats.regression.lm_alpha``.

    Returns:
        Mapping with ``model`` (``"ols"``, ``"lag"`` or ``"error"``),
        ``rule`` (which numbered branch fired) and ``reason``.
    """
    level = float(
        params["spatial_stats"]["regression"]["lm_alpha"] if alpha is None else alpha
    )

    def _p(key: str) -> float:
        value = diagnostics.get(key)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return float(value[1])
        return float("nan")

    def _stat(key: str) -> float:
        value = diagnostics.get(key)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return float(value[0])
        return float("nan")

    lag_sig = _p("lm_lag") < level
    error_sig = _p("lm_error") < level

    if not lag_sig and not error_sig:
        return {
            "model": "ols",
            "rule": 1,
            "reason": (
                f"neither LM-lag (p={_p('lm_lag'):.4g}) nor LM-error "
                f"(p={_p('lm_error'):.4g}) is significant at {level}; OLS stands"
            ),
        }
    if lag_sig and not error_sig:
        return {
            "model": "lag",
            "rule": 2,
            "reason": f"only LM-lag is significant (p={_p('lm_lag'):.4g})",
        }
    if error_sig and not lag_sig:
        return {
            "model": "error",
            "rule": 2,
            "reason": f"only LM-error is significant (p={_p('lm_error'):.4g})",
        }

    r_lag_sig = _p("rlm_lag") < level
    r_error_sig = _p("rlm_error") < level
    if r_lag_sig and not r_error_sig:
        return {
            "model": "lag",
            "rule": 3,
            "reason": (
                "both classic LM tests are significant; only robust LM-lag is "
                f"(p={_p('rlm_lag'):.4g})"
            ),
        }
    if r_error_sig and not r_lag_sig:
        return {
            "model": "error",
            "rule": 3,
            "reason": (
                "both classic LM tests are significant; only robust LM-error is "
                f"(p={_p('rlm_error'):.4g})"
            ),
        }
    if r_lag_sig and r_error_sig:
        larger = "lag" if _stat("rlm_lag") >= _stat("rlm_error") else "error"
        return {
            "model": larger,
            "rule": 3,
            "reason": (
                "both robust LM tests are significant; taking the larger "
                f"statistic (robust LM-{larger})"
            ),
        }
    return {
        "model": "error",
        "rule": 3,
        "reason": (
            "both classic LM tests are significant but neither robust form is; "
            "defaulting to the error model, which is the conservative choice "
            "because it treats the dependence as a nuisance rather than as a "
            "substantive spillover"
        ),
    }


def build_model_frame(
    tables: Sequence["pd.DataFrame"],
    params: dict[str, Any],
    response: str | None = None,
    predictors: Sequence[str] | None = None,
    zone_column: str = "zone_id",
    standardise: bool | None = None,
) -> "pd.DataFrame":
    """Join per-zone covariate tables into one complete-case model frame.

    .. note::
        Joins are on ``zone_id`` (the pcode) and only on that. GN names repeat
        across Dehiwala, Moratuwa and Kolonnawa, so a name join silently merges
        unrelated divisions.

    Args:
        tables: Per-zone frames, each carrying ``zone_column`` and one or more
            of the modelled variables.
        params: Parsed params mapping.
        response: Override for ``spatial_stats.regression.response``.
        predictors: Override for ``spatial_stats.regression.predictors``.
        zone_column: Join key.
        standardise: Override for ``spatial_stats.regression.standardise``.

    Returns:
        ``pandas.DataFrame`` with ``zone_id``, the response and the predictors,
        complete cases only, sorted by ``zone_id``. ``attrs["dropped"]`` records
        how many zones were lost and to which column.

    Raises:
        ValueError: If a required column is missing from every table, or if
            fewer than two complete cases survive.
    """
    import pandas as pd  # Deferred: see module docstring.

    cfg = params["spatial_stats"]["regression"]
    target = str(cfg["response"] if response is None else response)
    names = resolve_regression_predictors(predictors, params)
    scale = bool(cfg["standardise"] if standardise is None else standardise)

    merged: "pd.DataFrame | None" = None
    for table in tables:
        if zone_column not in table.columns:
            raise ValueError(
                f"a covariate table has no {zone_column!r} column; it has "
                f"{sorted(table.columns)}"
            )
        frame = table.copy()
        frame[zone_column] = frame[zone_column].astype(str)
        if merged is None:
            merged = frame
            continue
        overlap = [
            c for c in frame.columns if c in merged.columns and c != zone_column
        ]
        merged = merged.merge(
            frame.drop(columns=overlap), on=zone_column, how="outer"
        )
    if merged is None:
        raise ValueError("no covariate tables were supplied")

    wanted = [target, *names]
    missing = [column for column in wanted if column not in merged.columns]
    if missing:
        raise ValueError(
            f"the joined covariate table is missing {missing}; it has "
            f"{sorted(merged.columns)}. Every predictor in "
            "spatial_stats.regression.predictors must be exported by "
            "zone_covariate_table before the ladder can run."
        )

    before = len(merged)
    dropped_by = {
        column: int(merged[column].isna().sum()) for column in wanted
    }
    complete = merged.dropna(subset=wanted).copy()
    if len(complete) < 2:
        raise ValueError(
            f"only {len(complete)} zone(s) have every modelled variable "
            f"(of {before}). Missing counts per column: {dropped_by}"
        )

    if scale:
        for column in wanted:
            values = complete[column].astype("float64")
            sd = float(values.std(ddof=0))
            complete[column] = (
                (values - values.mean()) / sd if sd > 0 else values - values.mean()
            )

    result = complete[[zone_column, *wanted]].sort_values(zone_column)
    result = result.reset_index(drop=True)
    result.attrs["dropped"] = {
        "before": before,
        "after": int(len(result)),
        "per_column": dropped_by,
        "standardised": scale,
    }
    return result


def maup_comparison(
    records: Sequence[Mapping[str, Any]], params: dict[str, Any]
) -> "pd.DataFrame":
    """Assemble the aggregation-unit sensitivity table.

    A statistic that could not be estimated at a level is a **row with a
    reason**, never an omission. Which statistics survive the coarsening from
    557 GN units to 13 DS units is itself the MAUP finding CLAUDE.md requires.

    Args:
        records: Mappings with at least ``statistic`` and ``level``; optionally
            ``n_units``, ``status``, ``value``, ``detail`` and ``reason``.
        params: Parsed params mapping.

    Returns:
        ``pandas.DataFrame`` with :data:`MAUP_COLUMNS`, sorted by statistic then
        level.

    Raises:
        ValueError: If a record omits ``statistic`` or ``level``.
    """
    import pandas as pd  # Deferred: see module docstring.

    rows: list[dict[str, Any]] = []
    for record in records:
        if "statistic" not in record or "level" not in record:
            raise ValueError(
                f"every MAUP record needs 'statistic' and 'level'; got "
                f"{sorted(record)}"
            )
        rows.append(
            {
                "statistic": record["statistic"],
                "level": resolve_level(str(record["level"])),
                "n_units": record.get("n_units"),
                "status": record.get("status", MAUP_OK),
                "value": record.get("value"),
                "detail": record.get("detail"),
                "reason": record.get("reason", ""),
            }
        )
    if not rows:
        return pd.DataFrame(columns=list(MAUP_COLUMNS))
    frame = pd.DataFrame(rows)[list(MAUP_COLUMNS)]
    return frame.sort_values(["statistic", "level"]).reset_index(drop=True)


# =============================================================================
# Pure helpers - landscape metrics (no Earth Engine; unit-tested)
# =============================================================================
def patch_labels(
    mask: "np.ndarray", connectivity: int = 8
) -> tuple["np.ndarray", int]:
    """Label connected patches of a boolean class mask.

    Args:
        mask: 2-D boolean array; ``True`` is the focal class.
        connectivity: 4 (rook) or 8 (queen, the FRAGSTATS default).

    Returns:
        ``(labels, n_patches)``. ``labels`` is 0 for background.

    Raises:
        ValueError: If the mask is not 2-D or the connectivity is not 4 or 8.
    """
    import numpy as np  # Deferred: see module docstring.
    from scipy import ndimage  # Deferred: see module docstring.

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {array.shape}")
    if int(connectivity) not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    structure = (
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        if int(connectivity) == 4
        else np.ones((3, 3), dtype=int)
    )
    labels, count = ndimage.label(array, structure=structure)
    return labels, int(count)


def aggregation_index(mask: "np.ndarray", valid: "np.ndarray | None" = None) -> float:
    """Aggregation index of a class mask, as a percentage (He et al. 2000).

    ``AI = g_ii / max(g_ii) * 100`` where ``g_ii`` is the number of like
    adjacencies between focal-class cells, counted **once per shared edge**, and
    the maximum is the value a maximally compact patch of the same area would
    achieve.

    .. note::
        Like adjacencies are counted with **4-neighbour** adjacency regardless of
        the patch connectivity setting. That is the FRAGSTATS definition; using
        8 here would make the index exceed 100 for a solid block.

    Args:
        mask: 2-D boolean class mask.
        valid: Optional 2-D boolean mask of analysable cells. Adjacencies that
            cross into an invalid cell are ignored entirely, so a clipped study
            area does not fake a boundary.

    Returns:
        Aggregation index in ``[0, 100]``; NaN if the class is absent.
    """
    import numpy as np  # Deferred: see module docstring.

    array = np.asarray(mask, dtype=bool)
    if valid is not None:
        array = array & np.asarray(valid, dtype=bool)
    area = int(array.sum())
    if area == 0:
        return float("nan")
    if area == 1:
        return 0.0

    horizontal = int((array[:, :-1] & array[:, 1:]).sum())
    vertical = int((array[:-1, :] & array[1:, :]).sum())
    g = horizontal + vertical

    n = int(math.floor(math.sqrt(area)))
    m = area - n * n
    if m == 0:
        max_g = 2 * n * (n - 1)
    elif m <= n:
        max_g = 2 * n * (n - 1) + 2 * m - 1
    else:
        max_g = 2 * n * (n - 1) + 2 * m - 2
    if max_g <= 0:
        return 0.0
    return float(100.0 * g / max_g)


def landscape_metrics(
    mask: "np.ndarray",
    cell_size_m: float,
    params: dict[str, Any],
    valid: "np.ndarray | None" = None,
    connectivity: int | None = None,
    boundary_as_edge: bool | None = None,
) -> dict[str, float]:
    """Class-level landscape metrics for a binary green-space mask.

    .. warning::
        **Every one of these is scale dependent.** The same city classified at
        10 m and at 30 m gives different patch counts, different edge densities
        and a different aggregation index. ``cell_size_m`` is returned in the
        result and must be quoted with any figure.

    Args:
        mask: 2-D boolean array; ``True`` is green space.
        cell_size_m: Ground size of one cell, in metres.
        params: Parsed params mapping.
        valid: Optional analysable-cell mask. Cells outside it count neither as
            class nor as background, so the study-area clip cannot manufacture
            edge.
        connectivity: Override for ``spatial_stats.landscape.connectivity``.
        boundary_as_edge: Override for
            ``spatial_stats.landscape.boundary_as_edge``. When ``False`` (the
            default), the landscape's own outer boundary is not class edge, so
            edge density does not scale with the arbitrary shape of the clip.

    Returns:
        Mapping with ``cell_size_m``, ``landscape_area_ha``, ``class_area_ha``,
        ``class_fraction``, ``n_patches``, ``patch_density_per_100ha``,
        ``total_edge_m``, ``edge_density_m_per_ha``, ``mean_patch_area_ha``,
        ``largest_patch_index_pct`` and ``aggregation_index_pct``.

    Raises:
        ValueError: If the mask is not 2-D or the cell size is not positive.
    """
    import numpy as np  # Deferred: see module docstring.

    cfg = params["spatial_stats"]["landscape"]
    neighbours = int(cfg["connectivity"] if connectivity is None else connectivity)
    count_boundary = bool(
        cfg["boundary_as_edge"] if boundary_as_edge is None else boundary_as_edge
    )
    minimum_cells = int(cfg.get("min_patch_cells", 1))

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {array.shape}")
    size = float(cell_size_m)
    if size <= 0:
        raise ValueError(f"cell_size_m must be positive, got {cell_size_m}")

    valid_mask = (
        np.ones_like(array, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    )
    array = array & valid_mask

    cell_area_ha = size * size / 10_000.0
    landscape_area_ha = float(valid_mask.sum()) * cell_area_ha
    class_area_ha = float(array.sum()) * cell_area_ha

    labels, n_patches = patch_labels(array, neighbours)
    if minimum_cells > 1 and n_patches:
        sizes = np.bincount(labels.ravel())
        keep = np.flatnonzero(sizes >= minimum_cells)
        keep = keep[keep != 0]
        array = np.isin(labels, keep)
        labels, n_patches = patch_labels(array, neighbours)

    if n_patches:
        patch_sizes = np.bincount(labels.ravel())[1:]
        largest = float(patch_sizes.max()) * cell_area_ha
        mean_patch = float(patch_sizes.mean()) * cell_area_ha
    else:
        largest = 0.0
        mean_patch = float("nan")

    # Edge = a 4-neighbour adjacency between a class cell and a NON-class cell
    # that is inside the analysable area. Adjacencies to an invalid cell are
    # ignored, and the array's own rim is counted only if boundary_as_edge.
    def _edges(a: "np.ndarray", axis: int) -> int:
        if axis == 0:
            left, right = a[:-1, :], a[1:, :]
            left_valid, right_valid = valid_mask[:-1, :], valid_mask[1:, :]
        else:
            left, right = a[:, :-1], a[:, 1:]
            left_valid, right_valid = valid_mask[:, :-1], valid_mask[:, 1:]
        both_valid = left_valid & right_valid
        return int(((left != right) & both_valid).sum())

    edge_cells = _edges(array, 0) + _edges(array, 1)
    if count_boundary:
        edge_cells += int(array[0, :].sum() + array[-1, :].sum())
        edge_cells += int(array[:, 0].sum() + array[:, -1].sum())
    total_edge_m = float(edge_cells) * size

    return {
        "cell_size_m": size,
        "landscape_area_ha": landscape_area_ha,
        "class_area_ha": class_area_ha,
        "class_fraction": (
            class_area_ha / landscape_area_ha if landscape_area_ha > 0 else float("nan")
        ),
        "n_patches": float(n_patches),
        "patch_density_per_100ha": (
            100.0 * n_patches / landscape_area_ha
            if landscape_area_ha > 0
            else float("nan")
        ),
        "total_edge_m": total_edge_m,
        "edge_density_m_per_ha": (
            total_edge_m / landscape_area_ha if landscape_area_ha > 0 else float("nan")
        ),
        "mean_patch_area_ha": mean_patch,
        "largest_patch_index_pct": (
            100.0 * largest / landscape_area_ha
            if landscape_area_ha > 0
            else float("nan")
        ),
        "aggregation_index_pct": aggregation_index(array, valid_mask),
    }


def build_landscape_frame(
    records: Sequence[Mapping[str, Any]], params: dict[str, Any]
) -> "pd.DataFrame":
    """Shape landscape-metric results into the reporting table.

    Args:
        records: Mappings from :func:`landscape_metrics`, each additionally
            carrying ``scheme``, ``year`` and optionally ``zone_id``.
        params: Parsed params mapping.

    Returns:
        ``pandas.DataFrame`` with :data:`LANDSCAPE_COLUMNS`.
    """
    import pandas as pd  # Deferred: see module docstring.

    if not records:
        return pd.DataFrame(columns=list(LANDSCAPE_COLUMNS))
    frame = pd.DataFrame(list(records))
    for column in LANDSCAPE_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[list(LANDSCAPE_COLUMNS)]


def landscape_metrics_by_zone(
    class_raster: "np.ndarray",
    zone_raster: "np.ndarray",
    params: dict[str, Any],
    cell_size_m: float,
    green_codes: Sequence[int],
    zone_labels: Mapping[int, Any] | None = None,
    scheme: str = "",
    year: int | None = None,
    valid: "np.ndarray | None" = None,
) -> "pd.DataFrame":
    """Landscape metrics computed separately inside each zone.

    Feeds Phase 7: a division with the same green *fraction* but a far more
    fragmented one is a different greening proposition.

    Args:
        class_raster: 2-D integer class raster.
        zone_raster: 2-D integer zone raster, 0 for outside.
        params: Parsed params mapping.
        cell_size_m: Ground size of one cell.
        green_codes: Class codes counted as green.
        zone_labels: Zone raster code -> ``zone_id``.
        scheme: Recorded in the output.
        year: Recorded in the output.
        valid: Optional analysable-pixel mask, normally the ``observed`` band
            from :func:`green_class_image`. Without it a zone's unclassified
            pixels count as "not green" and its green fraction is understated by
            however much of it the classifier missed.

    Returns:
        ``pandas.DataFrame`` with :data:`LANDSCAPE_COLUMNS`, one row per zone,
        plus ``observed_fraction`` - the share of the zone that was classified
        at all.

    Raises:
        ValueError: If the rasters have different shapes.
    """
    import numpy as np  # Deferred: see module docstring.

    classes = np.asarray(class_raster)
    zones = np.asarray(zone_raster)
    if classes.shape != zones.shape:
        raise ValueError(
            f"class raster {classes.shape} and zone raster {zones.shape} must "
            "have the same shape; they must be exported on the same grid"
        )
    observed = (
        np.ones_like(zones, dtype=bool) if valid is None
        else np.asarray(valid, dtype=bool)
    )
    if observed.shape != zones.shape:
        raise ValueError(
            f"valid mask {observed.shape} and zone raster {zones.shape} must "
            "have the same shape"
        )

    green = np.isin(classes, list(green_codes))
    records: list[dict[str, Any]] = []
    for code in sorted(int(c) for c in np.unique(zones) if int(c) != 0):
        inside = zones == code
        analysable = inside & observed
        metrics = landscape_metrics(green, cell_size_m, params, valid=analysable)
        cells = int(inside.sum())
        metrics.update(
            {
                "scheme": scheme,
                "year": year,
                "zone_id": (zone_labels or {}).get(code, code),
                "observed_fraction": (
                    float(analysable.sum()) / cells if cells else float("nan")
                ),
            }
        )
        records.append(metrics)
    return build_landscape_frame(records, params)


# =============================================================================
# Cross-checks against the reference libraries (run in Colab, not locally)
# =============================================================================
def esda_cross_check(
    values: "np.ndarray",
    matrix: "np.ndarray",
    params: dict[str, Any],
) -> "pd.DataFrame":
    """Verify this module's statistics against ``esda``.

    The statistics here are computed analytically in numpy so they can be
    unit-tested without PySAL. That choice is only safe if it is checked, so
    this runs ``esda``'s implementations on the same inputs and reports the
    differences. Notebook 05 calls it in its probe step, and a test asserts
    agreement via ``pytest.importorskip``.

    .. note::
        **Local Moran's I is compared after rescaling, deliberately.** Anselin
        (1995) normalises by the population second moment (divide by ``n``);
        GeoDa and ``esda`` divide by ``(n-1)``. ``esda``'s values are therefore
        ours times ``(n-1)/n`` - a constant identical for every zone, so
        quadrants, permutation p-values and the cluster map are unaffected. The
        raw ratio is reported as its own row so the difference is visible rather
        than hidden, and the rescaled residual is what must be zero. Global
        Moran's I and Gi\\* have one convention each and are compared exactly.

    .. warning::
        Moran's I and Gi\\* are compared under **different weights**, because
        they are defined under different weights: row-standardised for Moran's
        I, binary-with-self for Gi\\*. Handing ``esda.G_Local`` the
        row-standardised matrix - which this function did until the first Colab
        run - compares two different statistics and reports the mismatch as
        though the implementation were wrong.

    Args:
        values: ``(n,)`` non-negative attribute values.
        matrix: ``(n, n)`` row-standardised weights.
        params: Parsed params mapping.

    Returns:
        ``pandas.DataFrame`` with ``statistic``, ``ours``, ``esda``,
        ``abs_diff`` and ``note``.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd  # Deferred: see module docstring.
    import esda  # Deferred: see module docstring.
    from libpysal.weights import W  # Deferred: see module docstring.

    array = np.asarray(matrix, dtype="float64")
    n = array.shape[0]
    neighbours = {
        i: [int(j) for j in np.flatnonzero(array[i] > 0)] for i in range(n)
    }
    weights = {i: [float(array[i, j]) for j in neighbours[i]] for i in range(n)}
    w = W(neighbours, weights, silence_warnings=True)

    y = np.asarray(values, dtype="float64")
    ours_global = morans_i(y, array)
    theirs_global = esda.Moran(y, w, permutations=0).I

    ours_lisa = local_morans(y, array, params, permutations=99)
    ours_local = ours_lisa["local_i"].to_numpy()
    reference_local = esda.Moran_Local(y, w, permutations=0)
    theirs_local = np.asarray(reference_local.Is, dtype="float64")

    # Gi* needs its OWN weights object, and getting this wrong is what the first
    # Colab run caught. `w` above is row-standardised, which is right for
    # Moran's I and wrong for Gi*; passing it to G_Local made the two sides
    # compute the statistic under different weightings and produced a 1.48
    # "disagreement" that was entirely an artefact of the comparison. esda even
    # warned about it: "Gi* requested, but (a) weights are already
    # row-standardized, (b) no weights are on the diagonal".
    #
    # So: binary weights WITH the self-weight on the diagonal - exactly what
    # gi_star builds internally - `transform="B"` so esda does not
    # row-standardise it back (its default is "R"), and `star=None`, which is
    # what esda's own warning prescribes once the diagonal is set.
    binary = (array > 0).astype("float64")
    star_matrix = add_self_neighbours(binary)
    star_neighbours = {
        i: [int(j) for j in np.flatnonzero(star_matrix[i] > 0)] for i in range(n)
    }
    w_star = W(
        star_neighbours,
        {
            i: [float(star_matrix[i, j]) for j in star_neighbours[i]]
            for i in range(n)
        },
        silence_warnings=True,
    )
    ours_gi = gi_star(y, binary, params, permutations=99)["gi_z"].to_numpy()
    theirs_gi = np.asarray(
        esda.G_Local(y, w_star, transform="B", star=None, permutations=0).Zs,
        dtype="float64",
    )

    # The constant that separates the two local-Moran normalisations. If the two
    # implementations really are the same statistic, this is one number, not a
    # spread - so its own spread is the thing worth testing.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ours_local != 0, theirs_local / ours_local, np.nan)
    scale = float(np.nanmedian(ratio))
    rescaled_diff = float(np.nanmax(np.abs(theirs_local - scale * ours_local)))
    quadrant_match = float(
        (ours_lisa["quadrant"].to_numpy() == np.asarray(reference_local.q)).mean()
    )

    rows = [
        {
            "statistic": "global_morans_i",
            "ours": float(ours_global),
            "esda": float(theirs_global),
            "abs_diff": abs(float(ours_global) - float(theirs_global)),
            "note": "one convention; must match exactly",
        },
        {
            "statistic": "gi_star_z (max abs diff)",
            "ours": float(np.abs(ours_gi).max()),
            "esda": float(np.abs(theirs_gi).max()),
            "abs_diff": float(np.abs(ours_gi - theirs_gi).max()),
            "note": (
                "one convention; must match exactly. BOTH sides use binary "
                "weights with the self-weight on the diagonal - a "
                "row-standardised W here is a different statistic"
            ),
        },
        {
            "statistic": "local_morans_i (after rescaling)",
            "ours": float(np.abs(ours_local).max()),
            "esda": float(np.abs(theirs_local).max()),
            "abs_diff": rescaled_diff,
            "note": (
                f"esda/ours = {scale:.6f} for every zone "
                f"(expected (n-1)/n = {(n - 1) / n:.6f}); spread "
                f"{float(np.nanmax(ratio) - np.nanmin(ratio)):.2e}"
            ),
        },
        {
            "statistic": "lisa_quadrant_agreement",
            "ours": quadrant_match,
            "esda": 1.0,
            "abs_diff": abs(1.0 - quadrant_match),
            "note": "cluster labels must agree exactly; scaling cannot affect them",
        },
    ]
    return pd.DataFrame(rows)


def spreg_cross_check(
    y: "np.ndarray",
    design: "np.ndarray",
    names: Sequence[str],
    matrix: "np.ndarray",
) -> "pd.DataFrame":
    """Verify :func:`ols_fit` and :func:`lagrange_multiplier_tests` vs ``spreg``.

    Args:
        y: ``(n,)`` response.
        design: ``(n, k)`` predictors without an intercept.
        names: Predictor names.
        matrix: ``(n, n)`` row-standardised weights.

    Returns:
        ``pandas.DataFrame`` with ``statistic``, ``ours``, ``spreg`` and
        ``abs_diff``.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd  # Deferred: see module docstring.
    from libpysal.weights import W  # Deferred: see module docstring.
    from spreg import OLS  # Deferred: see module docstring.

    array = np.asarray(matrix, dtype="float64")
    n = array.shape[0]
    neighbours = {
        i: [int(j) for j in np.flatnonzero(array[i] > 0)] for i in range(n)
    }
    weights = {i: [float(array[i, j]) for j in neighbours[i]] for i in range(n)}
    w = W(neighbours, weights, silence_warnings=True)

    response = np.asarray(y, dtype="float64").reshape(-1, 1)
    x = np.asarray(design, dtype="float64")
    reference = OLS(
        response, x, w=w, spat_diag=True, moran=True, name_x=list(names)
    )

    ours = ols_fit(np.asarray(y, dtype="float64"), x, names)
    diagnostics = lagrange_multiplier_tests(ours, np.asarray(y, "float64"), array)

    rows = [
        {
            "statistic": "r_squared",
            "ours": ours["r_squared"],
            "spreg": float(reference.r2),
        },
        {
            "statistic": "lm_error",
            "ours": diagnostics["lm_error"][0],
            "spreg": float(reference.lm_error[0]),
        },
        {
            "statistic": "lm_lag",
            "ours": diagnostics["lm_lag"][0],
            "spreg": float(reference.lm_lag[0]),
        },
        {
            "statistic": "rlm_error",
            "ours": diagnostics["rlm_error"][0],
            "spreg": float(reference.rlm_error[0]),
        },
        {
            "statistic": "rlm_lag",
            "ours": diagnostics["rlm_lag"][0],
            "spreg": float(reference.rlm_lag[0]),
        },
    ]
    frame = pd.DataFrame(rows)
    frame["abs_diff"] = (frame["ours"] - frame["spreg"]).abs()
    return frame


# =============================================================================
# Spatial models that genuinely need spreg / mgwr
# =============================================================================
def _libpysal_weights(matrix: "np.ndarray") -> Any:
    """Convert the dense matrix into a ``libpysal`` W, for spreg and mgwr only.

    Args:
        matrix: ``(n, n)`` weights.

    Returns:
        ``libpysal.weights.W``.
    """
    import numpy as np  # Deferred: see module docstring.
    from libpysal.weights import W  # Deferred: see module docstring.

    array = np.asarray(matrix, dtype="float64")
    n = array.shape[0]
    neighbours = {
        i: [int(j) for j in np.flatnonzero(array[i] > 0)] for i in range(n)
    }
    weights = {i: [float(array[i, j]) for j in neighbours[i]] for i in range(n)}
    return W(neighbours, weights, silence_warnings=True)


def spatial_lag_model(
    frame: "pd.DataFrame",
    matrix: "np.ndarray",
    params: dict[str, Any],
    response: str | None = None,
    predictors: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit the spatial lag (SAR) model.

    The lag model says a division's temperature depends on its neighbours'
    temperature - a substantive spillover. Choose it only when
    :func:`lm_decision` points there.

    Args:
        frame: Model frame from :func:`build_model_frame`.
        matrix: ``(n, n)`` row-standardised weights, aligned with the frame.
        params: Parsed params mapping.
        response: Override for ``spatial_stats.regression.response``.
        predictors: Override for ``spatial_stats.regression.predictors``.

    Returns:
        Mapping with ``coefficients``, ``rho``, ``pseudo_r_squared``, ``aic``,
        ``log_likelihood`` and ``estimator``.
    """
    return _fit_spreg_model(
        frame, matrix, params, "lag", response=response, predictors=predictors
    )


def spatial_error_model(
    frame: "pd.DataFrame",
    matrix: "np.ndarray",
    params: dict[str, Any],
    response: str | None = None,
    predictors: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit the spatial error model.

    The error model says the dependence is in what the model *omits* - a
    nuisance, not a spillover. It is the conservative reading of a significant
    residual Moran's I.

    Args:
        frame: Model frame from :func:`build_model_frame`.
        matrix: ``(n, n)`` row-standardised weights, aligned with the frame.
        params: Parsed params mapping.
        response: Override for ``spatial_stats.regression.response``.
        predictors: Override for ``spatial_stats.regression.predictors``.

    Returns:
        Mapping with ``coefficients``, ``lambda``, ``pseudo_r_squared``,
        ``aic``, ``log_likelihood`` and ``estimator``.
    """
    return _fit_spreg_model(
        frame, matrix, params, "error", response=response, predictors=predictors
    )


def _fit_spreg_model(
    frame: "pd.DataFrame",
    matrix: "np.ndarray",
    params: dict[str, Any],
    kind: str,
    response: str | None = None,
    predictors: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Shared body of :func:`spatial_lag_model` and :func:`spatial_error_model`.

    Args:
        frame: Model frame.
        matrix: Row-standardised weights.
        params: Parsed params mapping.
        kind: ``"lag"`` or ``"error"``.
        response: Response override.
        predictors: Predictor override.

    Returns:
        The tidy result mapping described by the two public wrappers.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd  # Deferred: see module docstring.

    if kind not in ("lag", "error"):
        raise ValueError(f"kind must be 'lag' or 'error', got {kind!r}")

    cfg = params["spatial_stats"]["regression"]
    target = str(cfg["response"] if response is None else response)
    names = resolve_regression_predictors(predictors, params)
    estimator = str(cfg["lag_estimator"]).lower()

    y = frame[target].to_numpy(dtype="float64").reshape(-1, 1)
    x = frame[names].to_numpy(dtype="float64")
    w = _libpysal_weights(matrix)

    import spreg  # Deferred: see module docstring.

    if kind == "lag":
        model = (
            spreg.ML_Lag(y, x, w=w, name_x=names, name_y=target)
            if estimator == "ml"
            else spreg.GM_Lag(y, x, w=w, name_x=names, name_y=target)
        )
        rho = float(model.betas[-1][0])
        spatial = {"rho": rho}
    else:
        model = (
            spreg.ML_Error(y, x, w=w, name_x=names, name_y=target)
            if estimator == "ml"
            else spreg.GM_Error(y, x, w=w, name_x=names, name_y=target)
        )
        spatial = {"lambda": float(model.betas[-1][0])}

    betas = np.asarray(model.betas).ravel()
    try:
        std_err = np.asarray(model.std_err).ravel()
    except AttributeError:  # pragma: no cover - estimator dependent
        std_err = np.full(betas.shape, np.nan)
    terms = ["intercept", *names]
    if len(betas) == len(terms) + 1:
        terms = [*terms, "rho" if kind == "lag" else "lambda"]

    coefficients = pd.DataFrame(
        {
            "term": terms[: len(betas)],
            "estimate": betas,
            "std_error": std_err[: len(betas)],
        }
    )
    coefficients["z_statistic"] = (
        coefficients["estimate"] / coefficients["std_error"]
    )

    result: dict[str, Any] = {
        "model": kind,
        "estimator": estimator,
        "coefficients": coefficients,
        "pseudo_r_squared": float(getattr(model, "pr2", float("nan"))),
        "log_likelihood": float(getattr(model, "logll", float("nan"))),
        "aic": float(getattr(model, "aic", float("nan"))),
        "n": int(frame.shape[0]),
    }
    result.update(spatial)
    return result


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute that may raise rather than merely be absent.

    ``getattr(obj, name, default)`` falls back only on ``AttributeError``.
    ``mgwr`` raises ``NotImplementedError("Not yet implemented for multiple
    bandwidths")`` from several ``MGWRResults`` properties, which propagates and
    - in Colab run 3 - discarded an entire successful MGWR fit, **including the
    per-covariate bandwidths that are its whole point**. One unimplemented
    diagnostic must not cost the result.

    Args:
        obj: Object to read from.
        name: Attribute name.
        default: Returned if the attribute is missing OR raises.

    Returns:
        The attribute value, or ``default``.
    """
    try:
        return getattr(obj, name)
    except Exception:  # noqa: BLE001 - see docstring; any failure means "absent"
        return default


def gwr_model(
    frame: "pd.DataFrame",
    coords: "np.ndarray",
    params: dict[str, Any],
    response: str | None = None,
    predictors: Sequence[str] | None = None,
    multiscale: bool = False,
) -> dict[str, Any]:
    """Fit GWR (or MGWR) and return per-zone local coefficients.

    .. warning::
        A GWR fits one regression per zone, so its local t-tests are ``n``
        simultaneous tests. ``spatial_stats.regression.gwr.adjust_alpha`` turns
        on the da Silva & Fotheringham correction to the critical t value;
        without it the map of "significant" local coefficients is inflated the
        same way an uncorrected pixel-wise p-map is. The corrected critical
        value is returned, and the ``significant`` columns use it.

    .. note::
        MGWR's headline output is the **per-covariate bandwidth**, not the
        coefficients: it says which relationships operate locally and which are
        effectively global. If every bandwidth comes back identical, the
        multiscale search collapsed and the result is just GWR.

    Args:
        frame: Model frame from :func:`build_model_frame`.
        coords: ``(n, 2)`` **projected** zone centroids, aligned with the frame.
        params: Parsed params mapping.
        response: Override for ``spatial_stats.regression.response``.
        predictors: Override for ``spatial_stats.regression.predictors``.
        multiscale: Fit MGWR instead of GWR.

    Returns:
        Mapping with ``local_coefficients`` (a frame of one row per zone),
        ``bandwidth`` (scalar for GWR, per-term list for MGWR), ``aicc``,
        ``r_squared``, ``critical_t``, ``adj_alpha`` and ``estimable``.

    Raises:
        ValueError: If the model is not estimable at this ``n``, or the shapes
            disagree.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd  # Deferred: see module docstring.

    cfg = params["spatial_stats"]["regression"]
    gwr_cfg = cfg["gwr"]
    target = str(cfg["response"] if response is None else response)
    names = resolve_regression_predictors(predictors, params)

    n = int(frame.shape[0])
    gate = require_estimable(
        n, len(names), params, statistic="MGWR" if multiscale else "GWR"
    )
    if not gate["estimable"]:
        raise ValueError(gate["reason"])

    points = np.asarray(coords, dtype="float64")
    if points.shape != (n, 2):
        raise ValueError(
            f"coords is {points.shape}, expected ({n}, 2) aligned with the frame"
        )

    y = frame[target].to_numpy(dtype="float64").reshape(-1, 1)
    x = frame[names].to_numpy(dtype="float64")

    from mgwr.gwr import GWR, MGWR  # Deferred: see module docstring.
    from mgwr.sel_bw import Sel_BW  # Deferred: see module docstring.

    kernel = str(gwr_cfg["kernel"])
    fixed = bool(gwr_cfg["fixed"])
    criterion = str(gwr_cfg["criterion"])

    selector = Sel_BW(points, y, x, kernel=kernel, fixed=fixed, multi=multiscale)
    if multiscale:
        bandwidth = selector.search(
            criterion=criterion, multi_bw_min=[int(gwr_cfg["mgwr_bw_min"])]
        )
        results = MGWR(points, y, x, selector, kernel=kernel, fixed=fixed).fit()
    else:
        bandwidth = selector.search(criterion=criterion)
        results = GWR(points, y, x, bandwidth, kernel=kernel, fixed=fixed).fit()

    terms = ["intercept", *names]
    betas = np.asarray(results.params)
    t_values = np.asarray(_safe_attr(results, "tvalues", np.full(betas.shape, np.nan)))

    adj_alpha = float("nan")
    critical_t = float("nan")
    if bool(gwr_cfg["adjust_alpha"]):
        try:
            adj_alpha = float(np.asarray(results.adj_alpha)[1])
            critical_t = float(results.critical_tval(adj_alpha))
        except Exception as error:  # pragma: no cover - mgwr version dependent
            warnings.warn(
                f"could not compute the adjusted critical t value ({error}); "
                "falling back to 1.96. The local significance map is then "
                "UNCORRECTED for multiple testing and must be labelled as such.",
                stacklevel=2,
            )
            critical_t = 1.96
    else:
        critical_t = 1.96

    local = pd.DataFrame({"zone_id": frame["zone_id"].to_numpy()})
    for index, term in enumerate(terms):
        local[f"beta_{term}"] = betas[:, index]
        local[f"t_{term}"] = t_values[:, index]
        local[f"sig_{term}"] = np.abs(t_values[:, index]) > critical_t
    local["local_r2"] = np.asarray(
        _safe_attr(results, "localR2", np.full((n, 1), np.nan))
    ).ravel()

    def _scalar(name: str) -> float:
        value = _safe_attr(results, name)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    return {
        "model": "mgwr" if multiscale else "gwr",
        "local_coefficients": local,
        "terms": terms,
        "bandwidth": (
            [float(b) for b in np.asarray(bandwidth).ravel()]
            if multiscale
            else float(bandwidth)
        ),
        "aicc": _scalar("aicc"),
        "r_squared": _scalar("R2"),
        "critical_t": critical_t,
        "adj_alpha": adj_alpha,
        "estimable": True,
        "n": n,
    }


def mgwr_model(
    frame: "pd.DataFrame",
    coords: "np.ndarray",
    params: dict[str, Any],
    response: str | None = None,
    predictors: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit MGWR - GWR with a separate bandwidth per covariate.

    Args:
        frame: Model frame from :func:`build_model_frame`.
        coords: ``(n, 2)`` projected zone centroids.
        params: Parsed params mapping.
        response: Override for ``spatial_stats.regression.response``.
        predictors: Override for ``spatial_stats.regression.predictors``.

    Returns:
        As :func:`gwr_model`, with ``bandwidth`` a list, one per term.
    """
    return gwr_model(
        frame,
        coords,
        params,
        response=response,
        predictors=predictors,
        multiscale=True,
    )


# =============================================================================
# Earth Engine - zone geometry and covariates
# =============================================================================
def division_geometry_collection(
    params: dict[str, Any], level: str, simplify_m: float | None = None
) -> "ee.FeatureCollection":
    """Zone polygons carrying only the properties Phase 5 needs.

    Simplifying and dropping properties is not tidiness: ``data/outputs/`` is
    committed, and an unsimplified 557-polygon COD-AB export with every COD-AB
    attribute is far larger than a repository should carry.

    Args:
        params: Parsed params mapping.
        level: ``"gn"`` or ``"ds"``.
        simplify_m: Override for ``spatial_stats.geometry.simplify_m``.

    Returns:
        ``ee.FeatureCollection`` of simplified polygons.
    """
    import ee  # Deferred: see module docstring.

    from colombo_uhi import aoi

    resolved = resolve_level(level)
    cfg = params["spatial_stats"]["geometry"]
    tolerance = float(cfg["simplify_m"] if simplify_m is None else simplify_m)
    keep = list(cfg["export_properties"][resolved])

    divisions = (
        aoi.gn_divisions(params) if resolved == "gn" else aoi.ds_divisions(params)
    )

    def _shrink(feature: "ee.Feature") -> "ee.Feature":
        return ee.Feature(
            feature.geometry().simplify(tolerance), feature.toDictionary(keep)
        )

    return divisions.map(_shrink)


def export_division_geojson(
    params: dict[str, Any],
    level: str,
    simplify_m: float | None = None,
    folder: str | None = None,
    start: bool = True,
) -> "ee.batch.Task":
    """Export the zone polygons to Drive as GeoJSON.

    This closes the one structural gap Phase 5 inherited: no zone geometry had
    ever left Earth Engine, so nothing could build a spatial weights matrix.

    Args:
        params: Parsed params mapping.
        level: ``"gn"`` or ``"ds"``.
        simplify_m: Override for ``spatial_stats.geometry.simplify_m``.
        folder: Drive folder; defaults to ``exports.drive_folder``.
        start: Submit the task.

    Returns:
        The ``ee.batch.Task``.
    """
    from colombo_uhi import exports

    resolved = resolve_level(level)
    collection = division_geometry_collection(
        params, resolved, simplify_m=simplify_m
    )
    return exports.table_to_drive(
        collection,
        product=f"{resolved}_divisions",
        aoi="district",
        params=params,
        file_format="GeoJSON",
        folder=folder,
        res_m=int(params["spatial_stats"]["epoch_scale_m"]),
        start=start,
    )


def read_zone_geodataframe(
    path: str,
    params: dict[str, Any],
    level: str,
    id_column: str | None = None,
) -> "gpd.GeoDataFrame":
    """Read exported zone polygons and put them in the analysis CRS.

    .. warning::
        Earth Engine writes GeoJSON in **EPSG:4326**. Contiguity would survive
        that, but centroid distances, KNN and every GWR bandwidth would be in
        degrees, which is not a distance. This function reprojects to
        ``crs.analysis_epsg`` and every downstream function assumes it has.

    Args:
        path: Path to the exported ``.geojson``.
        params: Parsed params mapping.
        level: ``"gn"`` or ``"ds"``.
        id_column: Override for the identifier property; defaults to the first
            entry of ``spatial_stats.geometry.export_properties[level]``, which
            is the pcode.

    Returns:
        ``geopandas.GeoDataFrame`` in the analysis CRS with a ``zone_id``
        column, sorted by it.

    Raises:
        ValueError: If the identifier column is absent, if identifiers repeat,
            or if the feature count does not match ``aoi.expected_counts``.
    """
    import geopandas as gpd  # Deferred: see module docstring.

    resolved = resolve_level(level)
    cfg = params["spatial_stats"]["geometry"]
    key = str(id_column or cfg["export_properties"][resolved][0])

    frame = gpd.read_file(path)
    if key not in frame.columns:
        raise ValueError(
            f"the exported {resolved.upper()} geometry has no {key!r} property; "
            f"it has {sorted(frame.columns)}. Check "
            f"spatial_stats.geometry.export_properties.{resolved}."
        )

    frame = frame.rename(columns={key: "zone_id"})
    frame["zone_id"] = frame["zone_id"].astype(str)
    duplicated = frame["zone_id"].duplicated().sum()
    if duplicated:
        raise ValueError(
            f"{duplicated} duplicate zone_id(s) in the exported geometry. The "
            "identifier must be the pcode; GN names are NOT unique within "
            "Colombo District and a name-keyed frame silently merges divisions "
            "from Dehiwala, Moratuwa and Kolonnawa."
        )

    expected_key = "gn_divisions" if resolved == "gn" else "ds_divisions"
    expected = int(params["aoi"]["expected_counts"][expected_key])
    if len(frame) != expected:
        warnings.warn(
            f"the exported {resolved.upper()} geometry has {len(frame)} features, "
            f"expected {expected}. A short export leaves holes in the weights "
            "matrix; a long one means the district filter did not apply.",
            stacklevel=2,
        )

    invalid = int((~frame.geometry.is_valid).sum())
    if invalid:
        warnings.warn(
            f"{invalid} invalid polygon(s) repaired with buffer(0). Simplifying "
            "can self-intersect a ragged coastal boundary; the repair changes "
            "area by a negligible amount but is recorded here.",
            stacklevel=2,
        )
        frame["geometry"] = frame.geometry.buffer(0)

    target_crs = params["crs"]["analysis_epsg"]
    if frame.crs is None:
        warnings.warn(
            "the exported geometry declares no CRS; assuming EPSG:4326, which "
            "is what Earth Engine writes.",
            stacklevel=2,
        )
        frame = frame.set_crs("EPSG:4326")
    frame = frame.to_crs(target_crs)

    return frame.sort_values("zone_id").reset_index(drop=True)


def distance_to_coast(
    params: dict[str, Any], region: "ee.Geometry | None" = None
) -> "ee.Image":
    """Distance from every pixel to the ocean, in kilometres.

    .. note::
        The ocean is isolated from inland water by a **connected-component size
        floor** (``covariates.dist_coast.min_ocean_pixels``). Without it this
        would be "distance to the nearest water of any kind": Beira Lake sits in
        the middle of the CMC, and every division around it would read as
        coastal.

    Args:
        params: Parsed params mapping.
        region: Optional region to clip to.

    Returns:
        Single-band ``ee.Image`` named ``dist_coast_km``.
    """
    import ee  # Deferred: see module docstring.

    cfg = params["spatial_stats"]["covariates"]["dist_coast"]
    water_cfg = params["datasets"]["surface_water"]
    scale = int(cfg["scale_m"])
    floor = int(cfg["min_ocean_pixels"])
    reach = int(cfg["max_distance_px"])

    water = (
        ee.Image(water_cfg["id"])
        .select(water_cfg["band_occurrence"])
        .gte(int(water_cfg["occurrence_threshold_pct"]))
        .unmask(0)
    )
    # *** PIN THE GRID BEFORE ANYTHING NEIGHBOURHOOD-BASED RUNS. ***
    # connectedPixelCount and fastDistanceTransform both work in the input
    # image's projection, and the input here is JRC GSW at 30 m. Leaving that
    # implicit cost Colab run 1 every covariate export: the neighbourhood needed
    # 1333 px to span 40 km at 30 m, and its square serialised to ~160 MB. It
    # also silently made "pixels" mean 30 m while the conversion below assumed
    # `scale`, and made the component floor count 30 m cells. Reprojecting first
    # makes one pixel exactly `scale` metres, so all three agree by construction.
    ocean_grid = water.reproject(crs=params["crs"]["analysis_epsg"], scale=scale)

    # connectedPixelCount saturates at maxSize, so `gte(floor)` reads as "part of
    # a component at least this large" - the ocean and nothing else here.
    component = ocean_grid.selfMask().connectedPixelCount(floor, True)
    ocean = component.gte(floor).unmask(0)

    # fastDistanceTransform returns SQUARED distance in PIXELS of the pinned grid.
    distance = (
        ocean.fastDistanceTransform(reach)
        .sqrt()
        .multiply(scale)
        .divide(1000.0)
        .rename("dist_coast_km")
        .toFloat()
    )
    return distance.clip(region) if region is not None else distance


def population_density(
    params: dict[str, Any], year: int | None = None
) -> "ee.Image":
    """WorldPop population density in people per square kilometre.

    .. note::
        **WorldPop ends in 2020** (``datasets.worldpop.availability``). A "2025"
        population density does not exist; the configured
        ``covariates.population.year`` is used and travels as a ``year``
        property so the table can be labelled honestly.

    Args:
        params: Parsed params mapping.
        year: Override for ``spatial_stats.covariates.population.year``.

    Returns:
        Single-band ``ee.Image`` named ``pop_density``, with a ``year`` property.

    Raises:
        ValueError: If the requested year is outside WorldPop's coverage.
    """
    import ee  # Deferred: see module docstring.

    cfg = params["datasets"]["worldpop"]
    requested = int(
        params["spatial_stats"]["covariates"]["population"]["year"]
        if year is None
        else year
    )
    first = int(str(cfg["availability"][0])[:4])
    last = int(str(cfg["availability"][1])[:4])
    if not first <= requested <= last:
        raise ValueError(
            f"WorldPop covers {first}-{last}; {requested} is outside it. Set "
            "spatial_stats.covariates.population.year to a covered year and "
            "label the output with that year - do not silently substitute one."
        )

    image = (
        ee.ImageCollection(cfg["id"])
        .filter(
            ee.Filter.eq(cfg["country_filter_property"], cfg["country_filter_value"])
        )
        .filterDate(f"{requested}-01-01", f"{requested + 1}-01-01")
        .first()
        .select([cfg["band"]])
    )
    # people per cell -> people per km2, using the true pixel area.
    per_km2 = image.divide(ee.Image.pixelArea()).multiply(1e6)
    return per_km2.rename("pop_density").toFloat().set("year", requested)


def elevation_image(params: dict[str, Any]) -> "ee.Image":
    """SRTM elevation in metres, named for the regression.

    Args:
        params: Parsed params mapping.

    Returns:
        Single-band ``ee.Image`` named ``elevation_m``.
    """
    import ee  # Deferred: see module docstring.

    cfg = params["datasets"]["srtm"]
    return ee.Image(cfg["id"]).select([cfg["band"]], ["elevation_m"]).toFloat()


def covariate_stack(
    params: dict[str, Any],
    epoch: str,
    region: "ee.Geometry",
    source: str | None = None,
    collection: "ee.ImageCollection | None" = None,
) -> "ee.Image":
    """One image carrying the response and every regression predictor.

    Built on :func:`colombo_uhi.uhi_metrics.driver_stack` for the response and
    the spectral/built-up predictors, then extended with the three Phase 5
    additions - population density, elevation and distance to coast - so the
    Phase 3 driver definitions are reused rather than re-derived.

    Args:
        params: Parsed params mapping.
        epoch: Epoch key from ``uhi.utfvi.epochs``.
        region: Region the collection is filtered to.
        source: Override for ``spatial_stats.epochs_source``.
        collection: Optional pre-built scene collection.

    Returns:
        ``ee.Image`` with the response band and every configured predictor.
    """
    import ee  # Deferred: see module docstring.

    from colombo_uhi import uhi_metrics

    cfg = params["spatial_stats"]
    key = str(cfg["epochs_source"] if source is None else source)
    start, _ = uhi_metrics.epoch_years(params, epoch)

    # *** DO NOT "OPTIMISE" THIS BY BUILDING ONE SHARED COLLECTION HERE. ***
    # That was tried (commit 2ac7934) and silently broke the science. Passing a
    # prebuilt collection makes epoch_composite and driver_stack skip
    # uhi_metrics.source_collection, which is the ONLY place the per-source
    # sensor restriction is applied (`sensors=resolved.get("sensors")`). So
    # `landsat_oli_dry` - L8+L9 only, the whole point of which is to avoid the
    # cross-sensor steps Phase 4 measured - quietly ran with all four sensors,
    # and the run-2 sensitivity check compared the pooled series WITH ITSELF and
    # reported a perfect 100% agreement.
    #
    # It saved about 12 KB of a 38 KB graph against a ~10 MB limit. The graph was
    # never the constraint; a distance transform on an inherited 30 m grid was.
    # Leave each consumer to build the collection its own source asks for.
    epoch_image = uhi_metrics.epoch_composite(
        key, params, epoch, collection=collection, region=region
    )
    # The spectral predictors come from the epoch's FIRST year rather than the
    # whole epoch: driver_stack composites one year, and matching its window to
    # the response's whole epoch would be a false precision. The year used
    # travels as a property.
    drivers = uhi_metrics.driver_stack(
        key, params, int(start), region, collection=collection
    )

    response = params["spatial_stats"]["response_band"]
    predictors = resolve_regression_predictors(None, params)
    from_drivers = [
        name for name in predictors if name in ("NDVI", "NDBI", "MNDWI", "built_fraction")
    ]

    layers = [epoch_image.select([response])]
    if from_drivers:
        layers.append(drivers.select(from_drivers))
    if "pop_density" in predictors:
        layers.append(population_density(params))
    if "elevation_m" in predictors:
        layers.append(elevation_image(params))
    if "dist_coast_km" in predictors:
        layers.append(distance_to_coast(params, region=None))

    return (
        ee.Image.cat(layers)
        .toFloat()
        .set(
            {
                "epoch": epoch,
                "source": key,
                "driver_year": int(start),
                "population_year": int(
                    params["spatial_stats"]["covariates"]["population"]["year"]
                ),
            }
        )
    )


def zone_covariate_table(
    params: dict[str, Any],
    level: str,
    epoch: str,
    region: "ee.Geometry",
    source: str | None = None,
    scale_m: int | None = None,
    collection: "ee.ImageCollection | None" = None,
) -> "pd.DataFrame":
    """Per-zone means of the response and every predictor, for one epoch.

    One ``reduceRegions`` per band, reusing
    :func:`colombo_uhi.uhi_metrics.zonal_by_division` so the identifier
    resolution, the pixel-count column (CLAUDE.md caveat 2) and the
    reduceRegions output-naming quirk are handled exactly once in the project.

    Args:
        params: Parsed params mapping.
        level: ``"gn"`` or ``"ds"``.
        epoch: Epoch key from ``uhi.utfvi.epochs``.
        region: Region the collection is filtered to.
        source: Override for ``spatial_stats.epochs_source``.
        scale_m: Override for ``spatial_stats.epoch_scale_m``.
        collection: Optional pre-built scene collection.

    Returns:
        ``pandas.DataFrame`` with ``zone_id``, one column per modelled variable,
        and a ``pixel_count`` column per variable suffixed ``_pixels``.
    """
    import pandas as pd  # Deferred: see module docstring.

    from colombo_uhi import uhi_metrics

    resolved = resolve_level(level)
    scale = int(
        params["spatial_stats"]["epoch_scale_m"] if scale_m is None else scale_m
    )
    stack = covariate_stack(
        params, epoch, region, source=source, collection=collection
    )
    wanted = [
        params["spatial_stats"]["response_band"],
        *resolve_regression_predictors(None, params),
    ]

    merged: "pd.DataFrame | None" = None
    for band in wanted:
        table = uhi_metrics.zonal_by_division(
            stack, params, level=resolved, band=band, scale_m=scale,
            reducers=("mean",),
        )
        id_columns = [c for c in table.columns if c not in ("mean", "pixel_count")]
        renamed = table.rename(
            columns={"mean": band, "pixel_count": f"{band}_pixels"}
        )
        if merged is None:
            merged = renamed
            key = id_columns[0]
        else:
            merged = merged.merge(
                renamed[[key, band, f"{band}_pixels"]], on=key, how="outer"
            )

    assert merged is not None  # `wanted` is never empty
    result = merged.rename(columns={key: "zone_id"})
    result["zone_id"] = result["zone_id"].astype(str)
    result.attrs["epoch"] = epoch
    result.attrs["level"] = resolved
    result.attrs["scale_m"] = scale
    return result.sort_values("zone_id").reset_index(drop=True)


def serialised_size(obj: Any) -> int:
    """Bytes an Earth Engine object occupies in the request that carries it.

    Purely client-side - it serialises the computation graph exactly as
    ``ee`` would and measures it, without contacting the server. That makes it
    the cheapest possible check for the failure mode that killed every covariate
    export in Colab run 1: ``Object too large (159861536 bytes)``, where the
    payload was three orders of magnitude over the limit and nothing in the
    submission path had measured it.

    .. note::
        Graph size is not compute cost. A cheap operation can serialise huge
        (a large ``fastDistanceTransform`` neighbourhood is the example that bit
        us) and an expensive one can serialise small. This measures only what
        travels in the request.

    Args:
        obj: Any ``ee`` object with a computation graph.

    Returns:
        Length in bytes of the serialised graph.
    """
    import json

    import ee  # Deferred: see module docstring.

    return len(json.dumps(ee.serializer.encode(obj)))


def graph_size_report(
    params: dict[str, Any],
    epoch: str,
    region: "ee.Geometry",
    source: str | None = None,
) -> "pd.DataFrame":
    """Serialised size of every band the covariate stack is built from.

    Run this BEFORE submitting an export. One oversized component makes the
    whole request unsendable, and the error names only the total, so without a
    per-component breakdown the search is guesswork.

    Args:
        params: Parsed params mapping.
        epoch: Epoch key from ``uhi.utfvi.epochs``.
        region: Region the source collection is filtered to.
        source: Override for ``spatial_stats.epochs_source``.

    Returns:
        ``pandas.DataFrame`` with ``component``, ``bytes`` and ``mb``, largest
        first, plus a ``covariate_stack (all)`` row for the assembled image.
    """
    import pandas as pd  # Deferred: see module docstring.

    from colombo_uhi import uhi_metrics

    key = str(
        params["spatial_stats"]["epochs_source"] if source is None else source
    )
    start, _ = uhi_metrics.epoch_years(params, epoch)

    components: dict[str, Any] = {
        "epoch_composite": uhi_metrics.epoch_composite(
            key, params, epoch, region=region
        ),
        "driver_stack": uhi_metrics.driver_stack(
            key, params, int(start), region
        ),
        "population_density": population_density(params),
        "elevation": elevation_image(params),
        "distance_to_coast": distance_to_coast(params),
        "covariate_stack (all)": covariate_stack(
            params, epoch, region, source=key
        ),
    }
    rows = []
    for name, obj in components.items():
        try:
            size = serialised_size(obj)
        except Exception as error:  # pragma: no cover - diagnostic only
            rows.append({"component": name, "bytes": None, "mb": None,
                         "note": f"could not serialise: {error}"})
            continue
        rows.append(
            {
                "component": name,
                "bytes": size,
                "mb": round(size / 1_048_576, 3),
                "note": "",
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("bytes", ascending=False, na_position="last").reset_index(
        drop=True
    )


def zone_covariate_bands(params: dict[str, Any]) -> list[str]:
    """The band order :func:`covariate_stack` produces: response then predictors.

    Args:
        params: Parsed params mapping.

    Returns:
        Band names in stack order.
    """
    return [
        params["spatial_stats"]["response_band"],
        *resolve_regression_predictors(None, params),
    ]


def zone_covariate_selectors(params: dict[str, Any], level: str) -> list[str]:
    """Property names to select when exporting a zone covariate table.

    ``Export.table.toDrive`` does not guarantee column order unless ``selectors``
    is passed, and a silently reordered CSV is the kind of error that only shows
    up as a nonsensical coefficient three steps later.

    Args:
        params: Parsed params mapping.
        level: ``"gn"`` or ``"ds"``.

    Returns:
        The identifier properties followed by ``<band>_mean`` and
        ``<band>_count`` for every modelled variable.
    """
    resolved = resolve_level(level)
    identifiers = list(params["spatial_stats"]["geometry"]["export_properties"][resolved])
    statistics: list[str] = []
    for band in zone_covariate_bands(params):
        statistics.extend([f"{band}_mean", f"{band}_count"])
    return [*identifiers, *statistics]


def zone_covariate_collection(
    params: dict[str, Any],
    level: str,
    epoch: str,
    region: "ee.Geometry",
    source: str | None = None,
    scale_m: int | None = None,
    collection: "ee.ImageCollection | None" = None,
) -> "ee.FeatureCollection":
    """The per-zone covariate reduction as an EXPORTABLE FeatureCollection.

    The batch twin of :func:`zone_covariate_table`, and the one the notebook
    uses. Phase 4's Colab runs 10-13 established that a district-wide reduction
    over a composite graph is not affordable interactively - down to and
    including ``bandNames()`` - while a batch ``Export`` task has no such
    ceiling. One ``reduceRegions`` over the whole multi-band stack replaces one
    call per band, so this is also seven times fewer graph evaluations.

    Args:
        params: Parsed params mapping.
        level: ``"gn"`` or ``"ds"``.
        epoch: Epoch key from ``uhi.utfvi.epochs``.
        region: Region the source collection is filtered to.
        source: Override for ``spatial_stats.epochs_source``.
        scale_m: Override for ``spatial_stats.epoch_scale_m``.
        collection: Optional pre-built scene collection.

    Returns:
        ``ee.FeatureCollection``, one feature per zone, carrying
        :func:`zone_covariate_selectors`.
    """
    import ee  # Deferred: see module docstring.

    from colombo_uhi import aoi

    resolved = resolve_level(level)
    comp = params["composites"]
    scale = int(
        params["spatial_stats"]["epoch_scale_m"] if scale_m is None else scale_m
    )
    stack = covariate_stack(
        params, epoch, region, source=source, collection=collection
    ).select(zone_covariate_bands(params))

    divisions = (
        aoi.gn_divisions(params) if resolved == "gn" else aoi.ds_divisions(params)
    )
    # count() alongside mean() is not optional: CLAUDE.md caveat 2 requires a
    # valid-observation count beside every aggregate, and a division mean over
    # four cloud-free pixels is not the same datum as one over four thousand.
    return stack.reduceRegions(
        collection=divisions,
        reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
        scale=scale,
        tileScale=comp["tile_scale"],
    )


def export_zone_covariates(
    params: dict[str, Any],
    level: str,
    epoch: str,
    region: "ee.Geometry",
    source: str | None = None,
    scale_m: int | None = None,
    collection: "ee.ImageCollection | None" = None,
    folder: str | None = None,
    suffix: str | None = None,
    start: bool = True,
) -> "ee.batch.Task":
    """Submit the per-zone covariate table as a batch export.

    .. note::
        ``suffix`` defaults to ``"{level}_{epoch}"``. **Override it whenever the
        same level and epoch are exported twice** - the sensitivity run on the
        single-sensor series is the case that matters - or the second task
        renders to the same Drive filename and silently overwrites the first.

    Args:
        params: Parsed params mapping.
        level: ``"gn"`` or ``"ds"``.
        epoch: Epoch key from ``uhi.utfvi.epochs``.
        region: Region the source collection is filtered to.
        source: Override for ``spatial_stats.epochs_source``.
        scale_m: Override for ``spatial_stats.epoch_scale_m``.
        collection: Optional pre-built scene collection.
        folder: Drive folder; defaults to ``exports.drive_folder``.
        suffix: Name discriminator; defaults to ``"{level}_{epoch}"``.
        start: Submit the task.

    Returns:
        The ``ee.batch.Task``.
    """
    from colombo_uhi import exports

    resolved = resolve_level(level)
    scale = int(
        params["spatial_stats"]["epoch_scale_m"] if scale_m is None else scale_m
    )
    features = zone_covariate_collection(
        params, resolved, epoch, region, source=source, scale_m=scale,
        collection=collection,
    )
    return exports.table_to_drive(
        features,
        product="zone_covariates",
        aoi="district",
        params=params,
        file_format="CSV",
        selectors=zone_covariate_selectors(params, resolved),
        folder=folder,
        res_m=scale,
        suffix=suffix if suffix is not None else f"{resolved}_{epoch}",
        start=start,
    )


def read_zone_covariates(
    path: str, params: dict[str, Any], level: str
) -> "pd.DataFrame":
    """Read an exported zone covariate CSV into the model-frame shape.

    Args:
        path: Path to the downloaded ``.csv``.
        params: Parsed params mapping.
        level: ``"gn"`` or ``"ds"``.

    Returns:
        ``pandas.DataFrame`` with ``zone_id``, one column per modelled variable,
        and a ``<variable>_pixels`` count column beside each.

    Raises:
        ValueError: If the identifier column or a modelled band is absent.
    """
    import pandas as pd  # Deferred: see module docstring.

    resolved = resolve_level(level)
    identifiers = list(
        params["spatial_stats"]["geometry"]["export_properties"][resolved]
    )
    key = identifiers[0]

    frame = pd.read_csv(path)
    if key not in frame.columns:
        raise ValueError(
            f"the exported covariate table has no {key!r} column; it has "
            f"{sorted(frame.columns)}. Zones are keyed on the pcode."
        )

    out = pd.DataFrame({"zone_id": frame[key].astype(str)})
    for name in identifiers[1:]:
        if name in frame.columns:
            out[name] = frame[name]
    missing: list[str] = []
    for band in zone_covariate_bands(params):
        mean_column = f"{band}_mean"
        count_column = f"{band}_count"
        if mean_column not in frame.columns:
            missing.append(mean_column)
            continue
        out[band] = pd.to_numeric(frame[mean_column], errors="coerce")
        out[f"{band}_pixels"] = pd.to_numeric(
            frame.get(count_column), errors="coerce"
        )
    if missing:
        raise ValueError(
            f"the exported covariate table is missing {missing}; it has "
            f"{sorted(frame.columns)}. Check that export_zone_covariates ran "
            "with the same spatial_stats.regression.predictors list."
        )
    return out.sort_values("zone_id").reset_index(drop=True)


def green_class_image(
    scheme: str,
    params: dict[str, Any],
    year: int | None = None,
    region: "ee.Geometry | None" = None,
) -> "ee.Image":
    """Green-space mask and its observation mask, as a two-band image.

    .. warning::
        **The second band is not optional bookkeeping.** A 0/1 raster written to
        GeoTIFF cannot distinguish "classified, not green" from "never
        classified" or "outside the study area" - masked pixels are written as
        0. Colab run 3 shows what that costs twice over:

        * landscape area came back as **125,259 ha** for a district of
          **69,900 ha**, because every pixel of the export's bounding box
          counted as analysable land;
        * Dynamic World green appeared to grow from 5,669 ha in 2016 to
          30,423 ha in 2024 - a 5.4x rise that is mostly Sentinel-2 coverage.
          Dynamic World begins mid-2015 and 2016 has far fewer scenes, so
          unclassified pixels were silently counted as *not green*.

        ``observed`` is 1 only where the scheme actually classified the pixel
        AND it is inside ``region``, which is exactly the analysable set
        (CLAUDE.md caveat 2 applied to land cover).

    Args:
        scheme: ``"dynamic_world"`` or ``"worldcover"``.
        params: Parsed params mapping.
        year: Required for Dynamic World.
        region: Optional region to bound the work.

    Returns:
        Two-band ``ee.Image`` - ``green`` (0/1) and ``observed`` (0/1) - with
        ``scheme`` and ``year`` properties.
    """
    import ee  # Deferred: see module docstring.

    from colombo_uhi import landcover

    codes = resolve_green_classes(scheme, params)
    classes = landcover.class_image(scheme, params, year=year, region=region)
    if region is not None:
        classes = classes.clip(region)

    # `.mask()` BEFORE unmasking: it is 1 wherever the classifier produced a
    # value and 0 everywhere else, which after the clip means "inside the study
    # area and actually classified".
    observed = classes.mask().reduce("min").gt(0).rename("observed").toByte()
    green = (
        classes.remap(codes, [1] * len(codes), 0)
        .unmask(0)
        .rename("green")
        .toByte()
    )
    return ee.Image.cat([green, observed]).set(
        {"scheme": scheme, "year": year if year is not None else -1}
    )


#: Band order :func:`green_class_image` produces, and the order the exported
#: GeoTIFF must be read back in.
GREEN_BANDS: tuple[str, ...] = ("green", "observed")
