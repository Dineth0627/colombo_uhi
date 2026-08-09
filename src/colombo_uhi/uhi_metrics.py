"""UHI metrics: SUHII, UTFVI, LST z-scores, zonal statistics and driver OLS.

This is the layer that turns land surface temperature into *urban heat island*
quantities. Everything it produces is LST — never air temperature (CLAUDE.md
caveat 1) — and every table it produces carries the pixel counts its numbers
rest on (caveat 2).

Products:
    * :func:`suhii_all_sources` - the tidy SUHII table, every year x every LST
      source x BOTH rural definitions, because a single SUHII number is not a
      result (caveat 5);
    * :func:`utfvi` / :func:`utfvi_class_image` / :func:`utfvi_class_series` -
      the six-class Urban Thermal Field Variance Index;
    * :func:`lst_zscore` / :func:`hot_pixel_mask` - per-scene standardisation
      and a hot-pixel mask at a configurable sigma;
    * :func:`zonal_by_division` - any image reduced to per-GN or per-DS
      statistics, the table Phase 5's spatial statistics consume;
    * :func:`driver_series` - per-year OLS and correlation of LST against NDVI,
      NDBI, MNDWI and built-up fraction.

Three things in here are easy to get wrong and are therefore pinned by unit
tests and stated loudly in the relevant docstrings:

1. **UTFVI is a ratio, so it depends on the temperature scale.** The published
   class breaks assume degrees Celsius. Feeding Kelvin silently reinterprets
   every class boundary by a factor of ten. :func:`utfvi` refuses anything but
   the configured Celsius band.
2. **UTFVI's reference is the year's own mean** (``uhi.utfvi.reference``), so it
   measures where heat sits *within* a year. A uniformly warming city produces
   no class change at all. Epoch-to-epoch drift is redistribution, not warming.
3. **The class boundary convention is left-closed** — a value exactly on a break
   belongs to the class above. The pure classifier and the server-side one are
   written to agree on this, and a test pins it.

Design notes:
    * ``import ee`` (and pandas/numpy/statsmodels) is deferred into function
      bodies so this module, and the local pytest suite, import cleanly without
      ``earthengine-api``.
    * Every constant comes from ``config/params.yaml`` (``uhi`` section).
    * Server-side only: no ``.getInfo()`` inside a loop over images. The client
      round trips are one per year-batch, exactly as in
      :mod:`colombo_uhi.composites`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, never at runtime
    import ee
    import numpy as np
    import pandas as pd

#: The two roles a SUHII mask can play. Band names are ``f"{method}_{role}"``.
MASK_ROLES: tuple[str, ...] = ("urban", "rural")

#: Column order of the tidy SUHII table produced by :func:`build_suhii_frame`.
SUHII_COLUMNS: tuple[str, ...] = (
    "year",
    "source",
    "rural_definition",
    "urban_mean",
    "rural_mean",
    "suhii",
    "urban_pixels",
    "rural_pixels",
)

#: Class index assigned to a missing (NaN) UTFVI value. Deliberately outside
#: 0..5 so it can never be mistaken for "Excellent".
UTFVI_NODATA_CLASS = -1

#: Column holding the number of pixels a per-division statistic rests on.
PIXEL_COUNT_COLUMN = "pixel_count"

#: Reducers :func:`zonal_by_division` knows how to build, mapped to the
#: ``ee.Reducer`` factory name. ``count`` is always added on top.
DIVISION_REDUCERS: tuple[str, ...] = ("mean", "median", "stdDev", "min", "max")


# =============================================================================
# Pure helpers - UTFVI (no Earth Engine; unit-tested)
# =============================================================================
def validate_utfvi_scheme(params: dict[str, Any]) -> tuple[list[float], list[str]]:
    """Validate and return the UTFVI class scheme.

    Args:
        params: Parsed params mapping (``uhi.utfvi``).

    Returns:
        ``(breaks, labels)`` with ``len(labels) == len(breaks) + 1``.

    Raises:
        ValueError: If the breaks are not strictly increasing, or if the label
            count does not match. Both would silently mislabel every map, so
            they fail here rather than in a figure caption.
    """
    cfg = params["uhi"]["utfvi"]
    breaks = [float(b) for b in cfg["breaks"]]
    labels = list(cfg["labels"])

    if not breaks:
        raise ValueError("uhi.utfvi.breaks must not be empty")
    if any(hi <= lo for lo, hi in zip(breaks, breaks[1:])):
        raise ValueError(
            f"uhi.utfvi.breaks must be strictly increasing, got {breaks}"
        )
    if len(labels) != len(breaks) + 1:
        raise ValueError(
            f"uhi.utfvi needs one more label than break: {len(breaks)} breaks "
            f"imply {len(breaks) + 1} classes but {len(labels)} labels are "
            f"configured ({labels})"
        )
    return breaks, labels


def utfvi_class_labels(params: dict[str, Any]) -> list[str]:
    """The six UTFVI class labels, in class order (index 0 = coolest).

    Args:
        params: Parsed params mapping.

    Returns:
        Validated list of labels.
    """
    return validate_utfvi_scheme(params)[1]


def utfvi_from_arrays(
    ts: "np.ndarray | Sequence[float]", t_mean: float
) -> "np.ndarray":
    """UTFVI = ``(Ts - Tmean) / Tmean`` on plain arrays.

    The reference implementation of the formula, so the server-side
    :func:`utfvi` has something to be tested against.

    .. warning::
        ``ts`` and ``t_mean`` must be in the SAME units, and per
        ``uhi.utfvi.units`` those units must be degrees Celsius. UTFVI is a
        ratio: with Tmean around 30 degC the 0.005 class break is about
        0.15 degC, but on the Kelvin scale (Tmean about 303 K) the same break is
        about 1.5 K. Passing Kelvin does not error — it silently reinterprets
        every class boundary.

    Args:
        ts: Surface temperatures in degrees Celsius.
        t_mean: Reference mean temperature in degrees Celsius.

    Returns:
        ``numpy`` array of UTFVI values, NaN-preserving.

    Raises:
        ValueError: If ``t_mean`` is zero, which would divide by zero. On the
            Celsius scale a zero mean is physically possible in principle, so
            this is a real guard and not a formality.
    """
    import numpy as np

    if float(t_mean) == 0.0:
        raise ValueError(
            "UTFVI is undefined for t_mean == 0 (division by zero). This is a "
            "hazard of the Celsius scale; check that the reference mean was "
            "computed over a non-empty region."
        )
    values = np.asarray(ts, dtype="float64")
    return (values - float(t_mean)) / float(t_mean)


def utfvi_class_indices(
    values: "np.ndarray | Sequence[float]", params: dict[str, Any]
) -> "np.ndarray":
    """Classify UTFVI values into the six classes, as integer indices.

    Boundaries are **left-closed**: a value exactly equal to a break belongs to
    the class ABOVE it, so 0.005 is "Normal", not "Good". The server-side
    :func:`utfvi_class_image` is built as a sum of ``gte`` tests, which produces
    the identical convention — that agreement is what a unit test pins, because
    a silent disagreement between the two would only show up as a map that
    disagrees with its own table.

    Args:
        values: UTFVI values.
        params: Parsed params mapping.

    Returns:
        Integer array of class indices 0..5, with :data:`UTFVI_NODATA_CLASS`
        (-1) wherever the input is NaN.
    """
    import numpy as np

    breaks, _ = validate_utfvi_scheme(params)
    array = np.asarray(values, dtype="float64")
    classes = np.digitize(array, breaks, right=False).astype("int64")
    # np.digitize sorts NaN into the TOP bucket, which would report missing data
    # as the "Worst" heat class. Overwrite it explicitly.
    return np.where(np.isnan(array), UTFVI_NODATA_CLASS, classes)


# =============================================================================
# Pure helpers - z-scores (no Earth Engine; unit-tested)
# =============================================================================
def resolve_sigma(sigma: float | None, params: dict[str, Any]) -> float:
    """Validate a hot-pixel threshold in standard deviations.

    Args:
        sigma: Requested threshold, or ``None`` for
            ``uhi.zscore.default_sigma``.
        params: Parsed params mapping.

    Returns:
        The threshold as a float.

    Raises:
        ValueError: If the threshold is not strictly positive. A zero or
            negative sigma would flag half the study area (or all of it) as a
            hot spot.

    Warns:
        UserWarning: If the threshold is not one of ``uhi.zscore.sigma_options``.
            Ad-hoc thresholds are allowed — they are sometimes exactly what a
            sensitivity run needs — but they must not slip into a figure
            unnoticed.
    """
    cfg = params["uhi"]["zscore"]
    resolved = float(cfg["default_sigma"] if sigma is None else sigma)
    if resolved <= 0:
        raise ValueError(
            f"sigma must be > 0 (it is a threshold in standard deviations), "
            f"got {resolved}"
        )
    options = [float(value) for value in cfg["sigma_options"]]
    if resolved not in options:
        warnings.warn(
            f"sigma={resolved} is not one of the reported thresholds {options}; "
            "label any figure or table built from it explicitly so it is not "
            "read as the standard 1-sigma or 2-sigma product.",
            stacklevel=2,
        )
    return resolved


def resolve_ddof(ddof: int | None, params: dict[str, Any]) -> int:
    """Resolve the delta-degrees-of-freedom used by the z-score standard deviation.

    .. warning::
        ``uhi.zscore.ddof`` is **not yet verified against Earth Engine**. The
        Earth Engine documentation does not state whether ``ee.Reducer.stdDev()``
        is the population (ddof 0) or sample (ddof 1) standard deviation, and a
        separate ``ee.Reducer.sampleStdDev()`` exists — which suggests, but does
        not prove, that ``stdDev`` is the population form. Notebook 03 settles it
        empirically. Until it does, :func:`zscore_array` and :func:`lst_zscore`
        are only guaranteed to agree to order 1/n.

    Args:
        ddof: Requested value, or ``None`` for ``uhi.zscore.ddof``.
        params: Parsed params mapping.

    Returns:
        The delta degrees of freedom as an int.

    Raises:
        ValueError: If negative.
    """
    resolved = int(params["uhi"]["zscore"]["ddof"] if ddof is None else ddof)
    if resolved < 0:
        raise ValueError(f"ddof must be >= 0, got {resolved}")
    return resolved


def zscore_array(
    values: "np.ndarray | Sequence[float]",
    params: dict[str, Any],
    ddof: int | None = None,
) -> "np.ndarray":
    """Standardise values against their own mean and standard deviation.

    The reference implementation of :func:`lst_zscore`. NaN-aware: missing
    values are excluded from the mean and standard deviation and stay missing in
    the output, rather than poisoning the whole array.

    A constant input has zero spread, so every z-score is 0/0. This returns all
    NaN for that case rather than raising or emitting infinities: a uniform
    scene is a legitimate (if degenerate) input, and NaN is the honest answer —
    "no pixel is anomalous relative to the others" is a statement about spread
    that a constant array cannot support.

    Args:
        values: Input values.
        params: Parsed params mapping.
        ddof: Delta degrees of freedom; defaults to ``uhi.zscore.ddof``. See
            :func:`resolve_ddof` for why that value is provisional.

    Returns:
        ``numpy`` array of z-scores, same shape as the input.
    """
    import numpy as np

    delta = resolve_ddof(ddof, params)
    array = np.asarray(values, dtype="float64")
    if array.size == 0:
        return array.copy()

    finite = np.isfinite(array)
    if not finite.any():
        return np.full(array.shape, np.nan)

    mean = np.nanmean(array[finite])
    spread = np.nanstd(array[finite], ddof=delta)
    if not np.isfinite(spread) or spread == 0:
        return np.full(array.shape, np.nan)
    return (array - mean) / spread


def hot_pixel_flags(
    values: "np.ndarray | Sequence[float]",
    params: dict[str, Any],
    sigma: float | None = None,
    ddof: int | None = None,
) -> "np.ndarray":
    """Boolean hot-pixel flags at ``z >= sigma``.

    The reference implementation of :func:`hot_pixel_mask`. The comparison is
    ``>=`` so the threshold value itself counts as hot, matching the ``gte``
    used server-side.

    Args:
        values: Input values (LST, in any consistent unit — a z-score is scale
            invariant, so this one is safe on Kelvin as well as Celsius).
        params: Parsed params mapping.
        sigma: Threshold in standard deviations; defaults to
            ``uhi.zscore.default_sigma`` (1.0). Pass 2.0 for the stricter
            product.
        ddof: Forwarded to :func:`zscore_array`.

    Returns:
        Boolean array; ``False`` wherever the z-score is NaN, so missing data is
        never reported as a hot spot.
    """
    import numpy as np

    threshold = resolve_sigma(sigma, params)
    scores = zscore_array(values, params, ddof=ddof)
    return np.where(np.isnan(scores), False, scores >= threshold)


# =============================================================================
# Pure helpers - table shaping (no Earth Engine; unit-tested)
# =============================================================================
def resolve_methods(
    methods: Sequence[str] | None, params: dict[str, Any]
) -> list[str]:
    """Validate the rural-reference methods a SUHII run covers.

    Args:
        methods: Requested methods, or ``None`` for
            ``uhi.suhii.rural_definitions``.
        params: Parsed params mapping.

    Returns:
        Validated list, order preserved, duplicates collapsed.

    Raises:
        ValueError: If empty, or if a name is not a configured rural definition.
    """
    from colombo_uhi import aoi

    valid = list(params["uhi"]["suhii"]["rural_definitions"])
    requested = list(valid if methods is None else methods)
    if not requested:
        raise ValueError(
            f"methods must name at least one of {valid}; CLAUDE.md requires "
            "SUHII under at least two rural definitions, so an empty list is "
            "never what you want"
        )
    seen: dict[str, None] = {}
    for name in requested:
        seen.setdefault(aoi.resolve_rural_method(name, valid), None)
    return list(seen)


def build_suhii_frame(
    rows: Sequence[dict[str, Any]],
    methods: Sequence[str],
    params: dict[str, Any],
) -> "pd.DataFrame":
    """Shape raw ``reduceRegion`` outputs into the tidy SUHII table.

    Split out from :func:`suhii_series` so the reshaping — and its failure
    modes — are unit-testable without Earth Engine.

    Each input row is one year of one source, holding every method's urban and
    rural statistics side by side (that is the point of
    :func:`masked_pair_image`: all of them come back from a single reduction).
    This unpivots that wide row into one output row per method.

    Args:
        rows: Property mappings, each carrying ``year``, ``source`` and the keys
            ``f"{method}_{role}_mean"`` / ``f"{method}_{role}_count"`` for every
            method and both roles.
        methods: The rural-reference methods present in ``rows``.
        params: Parsed params mapping.

    Returns:
        ``pandas.DataFrame`` with columns :data:`SUHII_COLUMNS`, sorted by
        source, then rural definition, then year. ``suhii`` is
        ``urban_mean - rural_mean`` and is NaN wherever either side is missing —
        a year with no valid urban pixels has no SUHII, and must not be
        reported as one.

    Raises:
        RuntimeError: If an expected key is absent, rather than quietly
            producing an all-NaN column.
    """
    import pandas as pd

    from colombo_uhi import composites

    year_prop = params["composites"]["year_property"]
    resolved = list(methods)

    if not len(rows):
        return pd.DataFrame(columns=list(SUHII_COLUMNS))

    wide = pd.DataFrame(list(rows))
    records: list[dict[str, Any]] = []
    for method in resolved:
        keys = {
            f"{role}_{stat}": f"{method}_{role}_{stat}"
            for role in MASK_ROLES
            for stat in ("mean", "count")
        }
        missing = [key for key in keys.values() if key not in wide.columns]
        if missing:
            raise RuntimeError(
                f"reduceRegion did not return {missing} for rural definition "
                f"'{method}'; it returned {sorted(wide.columns)}. Check that "
                "masked_pair_image() was built with the same methods that "
                "build_suhii_frame() was given."
            )
        block = pd.DataFrame(
            {
                "year": wide[year_prop],
                "source": wide["source"],
                "rural_definition": method,
                "urban_mean": pd.to_numeric(wide[keys["urban_mean"]], errors="coerce"),
                "rural_mean": pd.to_numeric(wide[keys["rural_mean"]], errors="coerce"),
                "urban_pixels": pd.to_numeric(
                    wide[keys["urban_count"]], errors="coerce"
                ),
                "rural_pixels": pd.to_numeric(
                    wide[keys["rural_count"]], errors="coerce"
                ),
            }
        )
        block["suhii"] = block["urban_mean"] - block["rural_mean"]
        records.append(block)

    frame = (
        pd.concat(records, ignore_index=True)[list(SUHII_COLUMNS)]
        .sort_values(["source", "rural_definition", "year"])
        .reset_index(drop=True)
    )

    # CLAUDE.md caveat 2, through the one shared implementation. Both sides get
    # their own check: a SUHII can be destroyed by an empty rural reference just
    # as easily as by an empty urban one, and the two fail for different reasons
    # (QC filtering over the core vs an elevation cap over the ring).
    for column in ("urban_pixels", "rural_pixels"):
        composites.warn_if_counts_are_empty(frame[column], column, stacklevel=3)
    return frame


def build_class_share_frame(
    rows: Sequence[dict[str, Any]], params: dict[str, Any]
) -> "pd.DataFrame":
    """Shape per-year UTFVI class histograms into a share-of-AOI table.

    Args:
        rows: One mapping per year, carrying the year property and a
            ``histogram`` mapping of class index (as a string, the way Earth
            Engine's ``frequencyHistogram`` returns it) to pixel count.
        params: Parsed params mapping.

    Returns:
        ``pandas.DataFrame`` with the year column, one percentage column per
        class label, and :data:`PIXEL_COUNT_COLUMN`. Percentages sum to 100
        across the class columns of each row; a year with no classified pixels
        has NaN percentages and a zero count, so an empty year is visible rather
        than drawn as a flat 0% everywhere.
    """
    import pandas as pd

    _, labels = validate_utfvi_scheme(params)
    year_prop = params["composites"]["year_property"]
    columns = [year_prop, *labels, PIXEL_COUNT_COLUMN]

    if not len(rows):
        return pd.DataFrame(columns=columns)

    records: list[dict[str, Any]] = []
    for row in rows:
        histogram = row.get("histogram") or {}
        counts = []
        for index in range(len(labels)):
            # Earth Engine keys frequencyHistogram by the STRING form of the
            # pixel value, and drops classes with no pixels entirely.
            raw = histogram.get(str(index), histogram.get(index, 0))
            counts.append(float(raw or 0))
        total = sum(counts)
        record: dict[str, Any] = {year_prop: row.get(year_prop)}
        for label, count in zip(labels, counts):
            record[label] = (count / total * 100.0) if total > 0 else float("nan")
        record[PIXEL_COUNT_COLUMN] = total
        records.append(record)

    return (
        pd.DataFrame(records)[columns]
        .sort_values(year_prop)
        .reset_index(drop=True)
    )


def build_division_frame(
    features: Sequence[dict[str, Any]],
    params: dict[str, Any],
    level: str,
    band: str,
    reducers: Sequence[str],
) -> "pd.DataFrame":
    """Shape ``reduceRegions`` output into a per-division statistics table.

    Split out from :func:`zonal_by_division` so the property resolution and its
    failure mode are unit-testable without Earth Engine.

    ``reduceRegions`` names its outputs by the bare reducer (``mean``) for a
    single-band image but by ``<band>_<reducer>`` in other cases, so both spellings
    are accepted and the band-prefixed one wins when present.

    .. note::
        **GN division names are not unique within Colombo District** (CLAUDE.md).
        The identifier column is therefore the pcode, and the name is carried
        alongside it for readability only. Never join two tables from this
        function on the name.

    Args:
        features: ``feature["properties"]`` mappings from a ``getInfo`` call.
        params: Parsed params mapping.
        level: ``"gn"`` or ``"ds"``.
        band: The band that was reduced, used to derive the expected keys.
        reducers: Reducer names that were combined, excluding ``count``.

    Returns:
        ``pandas.DataFrame`` with the identifier columns present in the input,
        one column per reducer, and :data:`PIXEL_COUNT_COLUMN`.

    Raises:
        RuntimeError: If an expected statistic is absent from every feature.
    """
    import pandas as pd

    wanted = [*reducers, "count"]
    if not len(features):
        return pd.DataFrame(columns=[*wanted[:-1], PIXEL_COUNT_COLUMN])

    frame = pd.DataFrame(list(features))
    rename: dict[str, str] = {}
    missing: list[str] = []
    for name in wanted:
        prefixed, bare = f"{band}_{name}", name
        if prefixed in frame.columns:
            rename[prefixed] = name
        elif bare in frame.columns:
            pass  # already correctly named
        else:
            missing.append(f"{prefixed} (or {bare})")
    if missing:
        raise RuntimeError(
            f"reduceRegions did not return {missing}; it returned "
            f"{sorted(frame.columns)}. Check that band '{band}' exists on the "
            "image and that the reducer list matches."
        )
    frame = frame.rename(columns=rename)

    # Identifier columns, in the order a reader wants them: pcode first because
    # it is the actual key, then the human-readable names.
    id_candidates = _division_id_columns(params, level)
    id_columns = [name for name in id_candidates if name in frame.columns]
    stat_columns = list(reducers)
    frame = frame.rename(columns={"count": PIXEL_COUNT_COLUMN})

    ordered = [*id_columns, *stat_columns, PIXEL_COUNT_COLUMN]
    result = frame[ordered].copy()
    for column in [*stat_columns, PIXEL_COUNT_COLUMN]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if id_columns:
        result = result.sort_values(id_columns[0]).reset_index(drop=True)
    return result


def _division_id_columns(params: dict[str, Any], level: str) -> list[str]:
    """Candidate identifier property names for a division level, best first.

    Args:
        params: Parsed params mapping.
        level: ``"gn"`` or ``"ds"``.

    Returns:
        Candidate column names: pcode, own name, then parent names.

    Raises:
        ValueError: If ``level`` is not ``"gn"`` or ``"ds"``.
    """
    if level not in ("gn", "ds"):
        raise ValueError(f"level must be 'gn' or 'ds', got {level!r}")
    assets = params["aoi"]["assets"]
    if level == "gn":
        return [
            "adm4_pcode",
            *assets["gn_name_property_candidates"],
            "adm3_pcode",
            *assets["ds_name_property_candidates"],
            *assets["district_property_candidates"],
        ]
    return [
        "adm3_pcode",
        *assets["ds_name_property_candidates"],
        *assets["district_property_candidates"],
    ]


# =============================================================================
# Pure helpers - driver regression (no Earth Engine; unit-tested)
# =============================================================================
def resolve_predictors(
    predictors: Sequence[str] | None, params: dict[str, Any]
) -> list[str]:
    """Validate the driver predictor list.

    Args:
        predictors: Requested predictors, or ``None`` for
            ``uhi.drivers.predictors``.
        params: Parsed params mapping.

    Returns:
        Validated list, order preserved, duplicates collapsed.

    Raises:
        ValueError: If empty.
    """
    requested = list(
        params["uhi"]["drivers"]["predictors"] if predictors is None else predictors
    )
    if not requested:
        raise ValueError("predictors must name at least one driver")
    seen: dict[str, None] = {}
    for name in requested:
        seen.setdefault(name, None)
    return list(seen)


def fit_drivers(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    response: str | None = None,
    predictors: Sequence[str] | None = None,
) -> "pd.DataFrame":
    """Ordinary least squares of LST against its candidate drivers.

    .. warning::
        **These p-values are anti-conservative.** The rows are sampled pixels
        from a contiguous surface, so they are spatially autocorrelated; OLS
        assumes independence, so the standard errors come out too small and
        significance is overstated. Read this as a screening device, not as
        inference. CLAUDE.md's attribution workflow exists precisely for this:
        Phase 5 tests the residual Moran's I and escalates to a spatial
        lag/error model and then GWR/MGWR.

    Args:
        frame: One row per sampled pixel, with the response and predictor
            columns.
        params: Parsed params mapping (``uhi.drivers``).
        response: Response column; defaults to ``uhi.drivers.response``.
        predictors: Predictor columns; defaults to ``uhi.drivers.predictors``.

    Returns:
        ``pandas.DataFrame`` with one row per model term (``const`` first),
        columns ``term``, ``coefficient``, ``std_err``, ``t_stat``, ``p_value``,
        plus the model-level ``r_squared``, ``adj_r_squared`` and ``n_obs``
        repeated on every row so the table stays tidy when concatenated across
        years.

    Raises:
        ValueError: If a named column is absent, or if fewer than
            ``uhi.drivers.min_sample_rows`` complete rows survive. Returning an
            R-squared computed from a handful of pixels would be worse than
            refusing.

    Warns:
        UserWarning: If a predictor is constant across the sample and is
            therefore dropped. Left in, it would make the design matrix
            singular and statsmodels would return a row of NaNs that looks like
            a computed result.
    """
    import pandas as pd
    import statsmodels.api as sm

    cfg = params["uhi"]["drivers"]
    y_name = response or cfg["response"]
    x_names = resolve_predictors(predictors, params)

    missing = [c for c in (y_name, *x_names) if c not in frame.columns]
    if missing:
        raise ValueError(
            f"frame is missing column(s) {missing}; it has {sorted(frame.columns)}"
        )

    data = frame[[y_name, *x_names]].apply(pd.to_numeric, errors="coerce").dropna()

    kept: list[str] = []
    for name in x_names:
        # nunique(), not std() == 0. pandas' standard deviation of a constant
        # column is not reliably exactly zero — depending on how the frame was
        # built it can come back as 2.8e-17, which passes a `== 0` test and lets
        # a singular column through into the design matrix. Counting distinct
        # values is exact and says what is actually meant.
        if len(data) and int(data[name].nunique(dropna=True)) <= 1:
            warnings.warn(
                f"predictor '{name}' is constant across the {len(data)} sampled "
                "rows and has been dropped; keeping it would make the design "
                "matrix singular and produce a NaN row that reads like a "
                "result. A constant index usually means the sample fell "
                "entirely inside one land-cover type.",
                stacklevel=2,
            )
            continue
        kept.append(name)

    minimum = int(cfg["min_sample_rows"])
    if len(data) < minimum:
        raise ValueError(
            f"only {len(data)} complete rows after dropping missing values, "
            f"below uhi.drivers.min_sample_rows ({minimum}). An R-squared from "
            "this few pixels is not a result. Check the sample region, the "
            "cloud masking, and whether the year has any usable scenes at all."
        )
    if not kept:
        raise ValueError(
            "every predictor was constant across the sample; there is nothing "
            "to regress against"
        )

    design = sm.add_constant(data[kept], has_constant="add")
    model = sm.OLS(data[y_name], design).fit()

    result = pd.DataFrame(
        {
            "term": list(model.params.index),
            "coefficient": model.params.to_numpy(),
            "std_err": model.bse.to_numpy(),
            "t_stat": model.tvalues.to_numpy(),
            "p_value": model.pvalues.to_numpy(),
        }
    )
    result["r_squared"] = float(model.rsquared)
    result["adj_r_squared"] = float(model.rsquared_adj)
    result["n_obs"] = int(model.nobs)
    return result


def driver_correlations(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    response: str | None = None,
    predictors: Sequence[str] | None = None,
) -> "pd.DataFrame":
    """Pairwise Pearson correlation of the response against each driver.

    Reported alongside :func:`fit_drivers` because a multivariate coefficient
    and a bivariate correlation answer different questions, and NDVI/NDBI are
    strongly collinear in a dense city — a driver can have a large simple
    correlation and a small partial coefficient without either being wrong.

    The same autocorrelation caveat as :func:`fit_drivers` applies to these
    p-values.

    Args:
        frame: One row per sampled pixel.
        params: Parsed params mapping.
        response: Response column; defaults to ``uhi.drivers.response``.
        predictors: Predictor columns; defaults to ``uhi.drivers.predictors``.

    Returns:
        ``pandas.DataFrame`` with columns ``predictor``, ``pearson_r``,
        ``p_value``, ``n_obs``. A predictor with too few complete pairs, or no
        variance, yields NaN rather than raising, so one degenerate index does
        not destroy the whole year's table.

    Raises:
        ValueError: If a named column is absent.
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    cfg = params["uhi"]["drivers"]
    y_name = response or cfg["response"]
    x_names = resolve_predictors(predictors, params)

    missing = [c for c in (y_name, *x_names) if c not in frame.columns]
    if missing:
        raise ValueError(
            f"frame is missing column(s) {missing}; it has {sorted(frame.columns)}"
        )

    records: list[dict[str, Any]] = []
    for name in x_names:
        pair = (
            frame[[y_name, name]].apply(pd.to_numeric, errors="coerce").dropna()
        )
        # See fit_drivers: distinct-value counting, not std() == 0, because a
        # constant column's standard deviation is not reliably exactly zero.
        usable = (
            len(pair) >= 3
            and int(pair[y_name].nunique(dropna=True)) > 1
            and int(pair[name].nunique(dropna=True)) > 1
        )
        if usable:
            r_value, p_value = stats.pearsonr(pair[y_name], pair[name])
        else:
            r_value, p_value = np.nan, np.nan
        records.append(
            {
                "predictor": name,
                "pearson_r": float(r_value),
                "p_value": float(p_value),
                "n_obs": int(len(pair)),
            }
        )
    return pd.DataFrame(records)


# =============================================================================
# SUHII (Earth Engine)
# =============================================================================
def resolve_source(source: str | Mapping[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Resolve and validate one entry of ``uhi.suhii.sources``.

    Args:
        source: A source key (looked up in ``uhi.suhii.sources``) or the source
            mapping itself.
        params: Parsed params mapping.

    Returns:
        A copy of the source mapping, with ``key``, ``kind``, ``reducer`` and
        ``scale_m`` guaranteed present.

    Raises:
        KeyError: If a source key is not configured.
        ValueError: If ``kind`` is unknown, or a required field for that kind is
            missing.
    """
    configured = {entry["key"]: entry for entry in params["uhi"]["suhii"]["sources"]}
    if isinstance(source, str):
        if source not in configured:
            raise KeyError(
                f"unknown SUHII source '{source}'; uhi.suhii.sources defines "
                f"{sorted(configured)}"
            )
        resolved = dict(configured[source])
    else:
        resolved = dict(source)

    for field in ("key", "kind", "reducer", "scale_m"):
        if field not in resolved:
            raise ValueError(
                f"SUHII source {resolved.get('key', resolved)!r} is missing "
                f"required field '{field}'"
            )
    kind = resolved["kind"]
    if kind == "modis":
        for field in ("product", "daynight"):
            if field not in resolved:
                raise ValueError(
                    f"MODIS SUHII source '{resolved['key']}' is missing '{field}'"
                )
    elif kind != "landsat":
        raise ValueError(
            f"unknown SUHII source kind '{kind}' for '{resolved['key']}'; "
            "valid kinds: ['landsat', 'modis']"
        )
    return resolved


def source_months(source: Mapping[str, Any], params: dict[str, Any]) -> list[int] | None:
    """Calendar-month restriction for a source, or ``None`` for the whole year.

    Args:
        source: A resolved source mapping.
        params: Parsed params mapping.

    Returns:
        The months of ``time.seasons[<months_key>]``, or ``None``.
    """
    months_key = source.get("months_key")
    if not months_key:
        return None
    return [int(month) for month in params["time"]["seasons"][months_key]["months"]]


def source_collection(
    source: str | Mapping[str, Any],
    params: dict[str, Any],
    region: "ee.Geometry | None" = None,
) -> "ee.ImageCollection":
    """The SCENE-level collection behind one SUHII source.

    Scene level, not composites, is deliberate and load-bearing: every batched
    consumer rebuilds the composites it needs from this. Filtering an
    already-computed composite collection forces Earth Engine to materialise
    every composite graph just to evaluate the predicate — the Colab run 4
    failure that :func:`colombo_uhi.composites.zonal_annual_means_by_year`
    documents at length.

    Args:
        source: Source key or mapping from ``uhi.suhii.sources``.
        params: Parsed params mapping.
        region: Geometry the source is filtered to; ``None`` lets each loader
            fall back to :func:`colombo_uhi.aoi.analysis_region`. Passing the
            tight :func:`suhii_region` instead is the main memory lever.

    Returns:
        ``ee.ImageCollection`` of scenes/granules carrying the ``LST_C`` band.
    """
    from colombo_uhi import landsat, modis

    resolved = resolve_source(source, params)
    if resolved["kind"] == "landsat":
        return landsat.harmonised_collection(
            params,
            region=region,
            include_sr=False,   # SUHII needs LST only; the SR bands would be
            include_st_qa=False,  # carried through 26 years of compositing unused
        )
    return modis.lst_collection(
        resolved["product"],
        resolved["daynight"],
        params,
        region=region,
        mandatory_qa_max=resolved.get("mandatory_qa_max"),
        lst_error_max=resolved.get("lst_error_max"),
    )


def suhii_region(method: str, params: dict[str, Any]) -> "ee.Geometry":
    """Tight reduction envelope enclosing both masks of one rural definition.

    This is the memory lever for the whole phase. The masks themselves are
    clipped, so any enclosing region gives identical numbers — but
    :func:`colombo_uhi.aoi.analysis_region` is Western Province plus 25 km, and
    reducing over that costs roughly twenty times the pixels for no gain.

    * ``"buffer_ring"``: the bounds of the ring, which (the ring being an
      annulus around the urban core) also enclose the core itself.
    * ``"lcz_based"``: the bounds of the configured LCZ scope.

    Args:
        method: One of ``uhi.suhii.rural_definitions``.
        params: Parsed params mapping.

    Returns:
        ``ee.Geometry`` rectangle enclosing the method's urban and rural masks.
    """
    from colombo_uhi import aoi

    resolved = aoi.resolve_rural_method(
        method, params["uhi"]["suhii"]["rural_definitions"]
    )
    if resolved == "buffer_ring":
        return aoi.buffer_ring(params).bounds(aoi._MAX_ERROR_M)
    return aoi.lcz_scope_geometry(params).bounds(aoi._MAX_ERROR_M)


def suhii_work_region(
    params: dict[str, Any], methods: Sequence[str] | None = None
) -> "ee.Geometry":
    """One region enclosing every method's :func:`suhii_region`.

    Use it to bound :func:`source_collection` and the static water mask, so the
    whole phase shares a single work envelope.

    Args:
        params: Parsed params mapping.
        methods: Methods to cover; defaults to all rural definitions.

    Returns:
        ``ee.Geometry`` enclosing every requested method's reduction region.
    """
    from colombo_uhi import aoi

    resolved = resolve_methods(methods, params)
    region = suhii_region(resolved[0], params)
    for method in resolved[1:]:
        region = region.union(suhii_region(method, params), aoi._MAX_ERROR_M)
    return region.bounds(aoi._MAX_ERROR_M)


def mask_pairs(
    params: dict[str, Any],
    methods: Sequence[str] | None = None,
    water: "ee.Image | None" = None,
) -> dict[str, tuple["ee.Image", "ee.Image"]]:
    """Build the urban/rural mask pair for every rural definition, once.

    Building these once and reusing them across years and sources matters: each
    pair carries an LCZ mosaic, an SRTM threshold and a water mask, and
    rebuilding them per year would multiply those graphs by 26.

    Args:
        params: Parsed params mapping.
        methods: Methods to build; defaults to all rural definitions.
        water: Optional 0/1 water image forwarded to
            :func:`colombo_uhi.aoi.rural_reference`. **Pass
            ``aoi.static_water_mask(params, region=suhii_work_region(params))``
            for any multi-year run** — the default water mask composites Landsat
            internally and that composite is re-instantiated for every image it
            masks.

    Returns:
        Mapping of method name to ``(urban, rural)``.
    """
    from colombo_uhi import aoi

    return {
        method: aoi.rural_reference(method, params, water=water)
        for method in resolve_methods(methods, params)
    }


def masked_pair_image(
    image: "ee.Image",
    band: str,
    pairs: Mapping[str, tuple["ee.Image", "ee.Image"]],
) -> "ee.Image":
    """Stack one LST band masked to every urban and rural mask.

    The result has ``2 * len(pairs)`` bands, named ``f"{method}_{role}"``. A
    single ``reduceRegion`` over it then returns every urban and rural mean AND
    their pixel counts in one request, which is what keeps the round-trip count
    for a six-source, two-definition, 26-year table down to a few dozen rather
    than a few hundred.

    Args:
        image: An annual composite carrying ``band``.
        band: The LST band to mask.
        pairs: Mapping of method name to ``(urban, rural)`` from
            :func:`mask_pairs`.

    Returns:
        Multi-band ``ee.Image``.
    """
    import ee  # Deferred: see module docstring.

    layer = image.select([band])
    bands = [
        layer.updateMask(mask).rename(f"{method}_{role}")
        for method, masks in pairs.items()
        for role, mask in zip(MASK_ROLES, masks)
    ]
    return ee.Image.cat(bands)


def _suhii_feature(
    image: "ee.Image",
    band: str,
    pairs: Mapping[str, tuple["ee.Image", "ee.Image"]],
    region: "ee.Geometry",
    scale: int,
    params: dict[str, Any],
) -> "ee.Feature":
    """Reduce one composite to every method's urban/rural mean and count.

    Args:
        image: An annual composite.
        band: The LST band to reduce.
        pairs: Mapping of method name to ``(urban, rural)``.
        region: Region to reduce over.
        scale: Reduction scale in metres.
        params: Parsed params mapping.

    Returns:
        Geometry-less ``ee.Feature`` carrying the year plus every statistic.
    """
    import ee  # Deferred: see module docstring.

    comp = params["composites"]
    # sharedInputs=True keeps the output names band-prefixed (buffer_ring_urban_mean);
    # without it the combined reducer reports multiple inputs and the names come
    # back bare, colliding across methods.
    reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)
    stats = masked_pair_image(image, band, pairs).reduceRegion(
        reducer=reducer,
        geometry=region,
        scale=scale,
        maxPixels=comp["reduce_max_pixels"],
        tileScale=comp["tile_scale"],
    )
    return ee.Feature(
        None, stats.set(comp["year_property"], image.get(comp["year_property"]))
    )


def suhii_series(
    source: str | Mapping[str, Any],
    params: dict[str, Any],
    pairs: Mapping[str, tuple["ee.Image", "ee.Image"]],
    collection: "ee.ImageCollection | None" = None,
    region: "ee.Geometry | None" = None,
    batch_years: int | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    progress: bool = False,
) -> "pd.DataFrame":
    """SUHII for one LST source, every year, under every supplied rural definition.

    SUHII = mean urban LST - mean rural LST (``uhi.suhii.definition``). Both
    means come from a single reduction per year (see :func:`masked_pair_image`),
    and the composites are built a few years at a time from the scene-level
    collection, exactly as
    :func:`colombo_uhi.composites.zonal_annual_means_by_year` does and for the
    same reason.

    .. warning::
        At MODIS's 1 km reduction scale the 100 m LCZ masks and the rasterised
        CMC polygon are resampled, so a 1 km pixel with *any* urban fraction is
        counted as urban. The CMC holds only about 40 MODIS pixels (measured,
        Phase 2), so the MODIS urban mean is a coarse-unit, edge-contaminated
        statistic. Read it against the ``urban_pixels`` column, and remember
        that night LST is accepted at up to 3 K stated uncertainty against 1 K
        for day — night SUHII is weaker evidence than day SUHII and must never
        be presented as an equal-confidence pair.

    Args:
        source: Source key or mapping from ``uhi.suhii.sources``.
        params: Parsed params mapping.
        pairs: Mapping of method name to ``(urban, rural)`` from
            :func:`mask_pairs`. Built once by the caller and reused across
            sources.
        collection: The scene-level collection; ``None`` builds it with
            :func:`source_collection`. Pass one in to reuse a single Landsat
            collection across several runs.
        region: Region to reduce over; defaults to :func:`suhii_work_region`.
        batch_years: Years built per request; defaults to
            ``uhi.suhii.batch_years``.
        start_year: First year; defaults to ``time.start_year``.
        end_year: Last year, inclusive; defaults to ``time.end_year``.
        progress: Print a line per batch, so a slow run visibly advances.

    Returns:
        Tidy ``pandas.DataFrame`` with columns :data:`SUHII_COLUMNS`.

    Raises:
        ValueError: If ``batch_years`` is less than 1.
    """
    from colombo_uhi import composites

    resolved = resolve_source(source, params)
    suhii_cfg = params["uhi"]["suhii"]
    band = params["landsat_c2l2"]["lst_band_name"]
    scale = int(resolved["scale_m"])
    months = source_months(resolved, params)

    # Validation before the deferred import, so it stays unit-testable.
    batch = int(suhii_cfg["batch_years"] if batch_years is None else batch_years)
    if batch < 1:
        raise ValueError(f"batch_years must be >= 1, got {batch}")
    first_year = int(params["time"]["start_year"] if start_year is None else start_year)
    last_year = int(params["time"]["end_year"] if end_year is None else end_year)

    import ee  # Deferred: see module docstring.

    if region is None:
        region = suhii_work_region(params, list(pairs))
    if collection is None:
        collection = source_collection(resolved, params, region=region)

    rows: list[dict[str, Any]] = []
    for low in range(first_year, last_year + 1, batch):
        high = min(low + batch - 1, last_year)
        annual = composites.annual_composites(
            collection,
            params,
            band=band,
            reducer=resolved["reducer"],
            months=months,
            with_percentile=False,  # a percentile reducer must retain every
            start_year=low,         # observation per pixel to sort them, and no
            end_year=high,          # zonal mean ever uses the result
        )
        features = ee.FeatureCollection(
            annual.map(
                lambda image: _suhii_feature(
                    image, band, pairs, region, scale, params
                )
            )
        ).getInfo()["features"]
        for feature in features:
            properties = dict(feature["properties"])
            properties["source"] = resolved["key"]
            rows.append(properties)
        if progress:
            print(f"  {resolved['key']}: years {low}-{high} done ({len(rows)} rows)")

    return build_suhii_frame(rows, list(pairs), params)


def suhii_all_sources(
    params: dict[str, Any],
    pairs: Mapping[str, tuple["ee.Image", "ee.Image"]] | None = None,
    sources: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    water: "ee.Image | None" = None,
    region: "ee.Geometry | None" = None,
    progress: bool = False,
    **kwargs: Any,
) -> "pd.DataFrame":
    """The full tidy SUHII table: every source x every rural definition x year.

    This is the Phase 3 headline product. CLAUDE.md requires SUHII under at
    least two rural definitions with the sensitivity reported, and the two
    configured definitions differ **by design** — buffer_ring contrasts the CMC
    with a 15-25 km annulus, lcz_based contrasts built with vegetated LCZ
    classes inside Colombo District. Their SUHII values will differ. That spread
    is the result; it is not a discrepancy to reconcile.

    Args:
        params: Parsed params mapping.
        pairs: Pre-built mask pairs; ``None`` builds them with
            :func:`mask_pairs` (forwarding ``methods`` and ``water``).
        sources: Source keys to run; defaults to every entry of
            ``uhi.suhii.sources``.
        methods: Rural definitions; defaults to all of them.
        water: Optional water image forwarded to :func:`mask_pairs`; ignored
            when ``pairs`` is supplied.
        region: Region to reduce over; defaults to :func:`suhii_work_region`.
        progress: Print a line per batch.
        **kwargs: Forwarded to :func:`suhii_series`.

    Returns:
        Tidy ``pandas.DataFrame`` with columns :data:`SUHII_COLUMNS`, one block
        per source.
    """
    import pandas as pd

    resolved_methods = resolve_methods(methods, params)
    if pairs is None:
        pairs = mask_pairs(params, methods=resolved_methods, water=water)
    if region is None:
        region = suhii_work_region(params, resolved_methods)

    keys = (
        [entry["key"] for entry in params["uhi"]["suhii"]["sources"]]
        if sources is None
        else list(sources)
    )
    frames = [
        suhii_series(
            key, params, pairs, region=region, progress=progress, **kwargs
        )
        for key in keys
    ]
    if not frames:
        return pd.DataFrame(columns=list(SUHII_COLUMNS))
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# UTFVI (Earth Engine)
# =============================================================================
def aoi_mean_lst(
    image: "ee.Image",
    params: dict[str, Any],
    region: "ee.Geometry",
    band: str | None = None,
    scale_m: int | None = None,
) -> "ee.Number":
    """Spatial mean LST over a region — the ``Tmean`` of the UTFVI formula.

    Args:
        image: Image carrying the LST band.
        params: Parsed params mapping.
        region: Region to average over.
        band: LST band; defaults to ``landsat_c2l2.lst_band_name``.
        scale_m: Reduction scale; defaults to ``crs.analysis_scale_m``.

    Returns:
        ``ee.Number`` mean, in the band's own units (degrees Celsius).
    """
    import ee  # Deferred: see module docstring.

    comp = params["composites"]
    target = band or params["landsat_c2l2"]["lst_band_name"]
    scale = int(scale_m or params["crs"]["analysis_scale_m"])
    stats = image.select([target]).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=scale,
        maxPixels=comp["reduce_max_pixels"],
        tileScale=comp["tile_scale"],
    )
    return ee.Number(stats.get(target))


def utfvi(
    image: "ee.Image",
    params: dict[str, Any],
    region: "ee.Geometry",
    band: str | None = None,
    scale_m: int | None = None,
) -> "ee.Image":
    """Urban Thermal Field Variance Index, ``(Ts - Tmean) / Tmean``.

    ``Tmean`` is the spatial mean of this same image over ``region``
    (``uhi.utfvi.reference`` = ``per_year_aoi_mean``), which is the standard
    formulation and the one the published class breaks were calibrated on.

    .. warning::
        **Two things about this index are routinely misread.**

        First, it is a RATIO, so it depends on the temperature scale. The class
        breaks assume degrees Celsius; on Kelvin every boundary means something
        ten times larger. This function therefore only accepts the configured
        Celsius LST band, and ``uhi.utfvi.units`` records the requirement.

        Second, because ``Tmean`` is recomputed from the data itself, UTFVI
        describes where heat sits WITHIN a scene, not how hot the scene is. A
        city that warms uniformly by 2 degC produces identical UTFVI classes
        before and after. Never read epoch-to-epoch class drift as warming —
        that is what the Phase 4 Mann-Kendall and Sen's slope products measure.

    Args:
        image: Image carrying the LST band in degrees Celsius.
        params: Parsed params mapping.
        region: Region defining ``Tmean``, e.g. the CMC or the district.
        band: LST band; defaults to ``landsat_c2l2.lst_band_name``.
        scale_m: Reduction scale for ``Tmean``; defaults to
            ``crs.analysis_scale_m``.

    Returns:
        Single-band ``ee.Image`` named by ``uhi.utfvi.band_name`` (``UTFVI``).

    Raises:
        ValueError: If ``uhi.utfvi.units`` is not ``"celsius"``, or if ``band``
            is not the configured Celsius band. Both are guards against feeding
            it Kelvin.
    """
    cfg = params["uhi"]["utfvi"]
    units = str(cfg["units"]).lower()
    if units != "celsius":
        raise ValueError(
            f"uhi.utfvi.units is '{units}', but the configured class breaks "
            f"{cfg['breaks']} are only meaningful on the Celsius scale. UTFVI "
            "is a ratio; changing the temperature scale rescales every class "
            "boundary."
        )
    celsius_band = params["landsat_c2l2"]["lst_band_name"]
    target = band or celsius_band
    if target != celsius_band:
        raise ValueError(
            f"UTFVI must be computed on the Celsius LST band '{celsius_band}', "
            f"got '{target}'. If that band really is in Celsius, rename it "
            "before calling; the check exists because a Kelvin input fails "
            "silently rather than loudly."
        )

    t_mean = aoi_mean_lst(image, params, region, band=target, scale_m=scale_m)
    return (
        image.select([target])
        .subtract(t_mean)
        .divide(t_mean)
        .rename(cfg["band_name"])
    )


def utfvi_class_image(
    utfvi_image: "ee.Image", params: dict[str, Any]
) -> "ee.Image":
    """Classify a UTFVI image into the six classes, as an integer 0-5 band.

    Built as a sum of ``gte`` tests, one per break. That construction is not an
    optimisation — it is what makes the server-side classification agree with
    :func:`utfvi_class_indices` exactly, including the left-closed boundary
    convention where a value equal to a break falls into the class above.

    Args:
        utfvi_image: Output of :func:`utfvi`.
        params: Parsed params mapping.

    Returns:
        Single-band integer ``ee.Image`` named by ``uhi.utfvi.class_band_name``.
        Masked input pixels stay masked — missing data is never classified.
    """
    import ee  # Deferred: see module docstring.

    breaks, _ = validate_utfvi_scheme(params)
    band = params["uhi"]["utfvi"]["band_name"]
    source = utfvi_image.select([band])

    total = ee.Image.constant(0).toInt()
    for threshold in breaks:
        total = total.add(source.gte(float(threshold)))
    return (
        total.toInt()
        .updateMask(source.mask())
        .rename(params["uhi"]["utfvi"]["class_band_name"])
    )


def utfvi_class_shares(
    class_image: "ee.Image",
    params: dict[str, Any],
    region: "ee.Geometry",
    scale_m: int | None = None,
) -> "pd.DataFrame":
    """Percentage of a region in each UTFVI class, for one image.

    Args:
        class_image: Output of :func:`utfvi_class_image`.
        params: Parsed params mapping.
        region: Region to tabulate over.
        scale_m: Reduction scale; defaults to ``crs.analysis_scale_m``.

    Returns:
        Single-row ``pandas.DataFrame`` from :func:`build_class_share_frame`.
    """
    import ee  # Deferred: see module docstring.

    comp = params["composites"]
    band = params["uhi"]["utfvi"]["class_band_name"]
    scale = int(scale_m or params["crs"]["analysis_scale_m"])
    histogram = (
        class_image.select([band])
        .reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=region,
            scale=scale,
            maxPixels=comp["reduce_max_pixels"],
            tileScale=comp["tile_scale"],
        )
        .get(band)
        .getInfo()
    )
    year = class_image.get(comp["year_property"]).getInfo()
    return build_class_share_frame(
        [{comp["year_property"]: year, "histogram": histogram}], params
    )


def _class_share_feature(
    image: "ee.Image",
    params: dict[str, Any],
    region: "ee.Geometry",
    scale: int,
) -> "ee.Feature":
    """Compute UTFVI and tabulate its class histogram for one composite.

    Args:
        image: An annual composite carrying the Celsius LST band.
        params: Parsed params mapping.
        region: Region defining ``Tmean`` and the tabulation area.
        scale: Reduction scale in metres.

    Returns:
        Geometry-less ``ee.Feature`` carrying the year and the histogram.
    """
    import ee  # Deferred: see module docstring.

    comp = params["composites"]
    class_band = params["uhi"]["utfvi"]["class_band_name"]
    classes = utfvi_class_image(
        utfvi(image, params, region, scale_m=scale), params
    )
    histogram = classes.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=region,
        scale=scale,
        maxPixels=comp["reduce_max_pixels"],
        tileScale=comp["tile_scale"],
    ).get(class_band)
    return ee.Feature(
        None,
        {
            comp["year_property"]: image.get(comp["year_property"]),
            "histogram": histogram,
        },
    )


def utfvi_class_series(
    source: str | Mapping[str, Any],
    params: dict[str, Any],
    region: "ee.Geometry",
    collection: "ee.ImageCollection | None" = None,
    scale_m: int | None = None,
    batch_years: int | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    progress: bool = False,
) -> "pd.DataFrame":
    """Percentage of the AOI in each UTFVI class, per year.

    Each year is classified against ITS OWN mean (``per_year_aoi_mean``), so the
    shares describe how concentrated that year's heat was, not how hot the year
    was. A year in which every pixel warms equally has unchanged shares.

    Args:
        source: Source key or mapping from ``uhi.suhii.sources``.
        params: Parsed params mapping.
        region: AOI defining ``Tmean`` and the tabulation area.
        collection: Scene-level collection; ``None`` builds it with
            :func:`source_collection`.
        scale_m: Reduction scale; defaults to the source's own ``scale_m``.
        batch_years: Years per request; defaults to ``uhi.suhii.batch_years``.
        start_year: First year; defaults to ``time.start_year``.
        end_year: Last year, inclusive; defaults to ``time.end_year``.
        progress: Print a line per batch.

    Returns:
        ``pandas.DataFrame`` from :func:`build_class_share_frame`: year, one
        percentage column per class label, and a pixel count.

    Raises:
        ValueError: If ``batch_years`` is less than 1.
    """
    from colombo_uhi import composites

    resolved = resolve_source(source, params)
    band = params["landsat_c2l2"]["lst_band_name"]
    scale = int(scale_m or resolved["scale_m"])
    months = source_months(resolved, params)

    batch = int(
        params["uhi"]["suhii"]["batch_years"] if batch_years is None else batch_years
    )
    if batch < 1:
        raise ValueError(f"batch_years must be >= 1, got {batch}")
    first_year = int(params["time"]["start_year"] if start_year is None else start_year)
    last_year = int(params["time"]["end_year"] if end_year is None else end_year)

    import ee  # Deferred: see module docstring.

    if collection is None:
        collection = source_collection(resolved, params, region=region)

    rows: list[dict[str, Any]] = []
    for low in range(first_year, last_year + 1, batch):
        high = min(low + batch - 1, last_year)
        annual = composites.annual_composites(
            collection,
            params,
            band=band,
            reducer=resolved["reducer"],
            months=months,
            with_percentile=False,
            start_year=low,
            end_year=high,
        )
        features = ee.FeatureCollection(
            annual.map(
                lambda image: _class_share_feature(image, params, region, scale)
            )
        ).getInfo()["features"]
        rows.extend(feature["properties"] for feature in features)
        if progress:
            print(f"  UTFVI shares: years {low}-{high} done ({len(rows)} rows)")

    return build_class_share_frame(rows, params)


def epoch_years(params: dict[str, Any], epoch: str) -> tuple[int, int]:
    """Resolve one named epoch from ``uhi.utfvi.epochs``.

    Args:
        params: Parsed params mapping.
        epoch: Epoch key, e.g. ``"2010s"``.

    Returns:
        ``(start_year, end_year)``, both inclusive.

    Raises:
        KeyError: If the epoch is not configured.
        ValueError: If the range is inverted.
    """
    epochs = params["uhi"]["utfvi"]["epochs"]
    if epoch not in epochs:
        raise KeyError(
            f"unknown epoch '{epoch}'; uhi.utfvi.epochs defines {sorted(epochs)}"
        )
    start, end = (int(value) for value in epochs[epoch])
    if end < start:
        raise ValueError(f"epoch '{epoch}' has end_year {end} < start_year {start}")
    return start, end


def epoch_composite(
    source: str | Mapping[str, Any],
    params: dict[str, Any],
    epoch: str,
    collection: "ee.ImageCollection | None" = None,
    region: "ee.Geometry | None" = None,
) -> "ee.Image":
    """One composite covering a whole named epoch.

    Used for the three-epoch UTFVI maps. The epoch's years are composited
    together rather than averaged from annual composites, so a year with few
    usable scenes contributes in proportion to the observations it actually has.

    Args:
        source: Source key or mapping from ``uhi.suhii.sources``.
        params: Parsed params mapping.
        epoch: Epoch key from ``uhi.utfvi.epochs``.
        collection: Scene-level collection; ``None`` builds it with
            :func:`source_collection`.
        region: Forwarded to :func:`source_collection`.

    Returns:
        ``ee.Image`` carrying the LST band and ``obs_count``, with ``year`` set
        to the epoch's first year and an ``epoch`` property naming it.
    """
    from colombo_uhi import composites

    resolved = resolve_source(source, params)
    start, end = epoch_years(params, epoch)
    band = params["landsat_c2l2"]["lst_band_name"]

    if collection is None:
        collection = source_collection(resolved, params, region=region)

    scenes = collection.filterDate(f"{start}-01-01", f"{end + 1}-01-01")
    months = source_months(resolved, params)
    if months:
        from colombo_uhi import landsat

        scenes = scenes.filter(landsat.month_filter(months))

    # annual_composites over a single "year" spanning the epoch is not possible
    # (it slices by calendar year), so the epoch is reduced directly with the
    # same combined reducer, keeping obs_count alongside (CLAUDE.md caveat 2).
    reducer_name = composites.resolve_reducer_name(resolved["reducer"], params)
    raw_names = composites.reduced_band_names(band, reducer_name, None)
    out_names = composites.composite_band_names(band, None, params)
    obs_band = params["composites"]["obs_count_band"]

    import ee  # Deferred: see module docstring.

    reduced = (
        scenes.select([band])
        .merge(composites._padding_collection(band))
        .reduce(composites._composite_reducer(reducer_name, None))
        .select(raw_names, out_names)
    )
    data_bands = [name for name in out_names if name != obs_band]
    return (
        ee.Image.cat(
            [reduced.select(data_bands), reduced.select(obs_band).unmask(0)]
        )
        .select(out_names)
        .toFloat()
        .set(
            {
                params["composites"]["year_property"]: start,
                "epoch": epoch,
                "epoch_start_year": start,
                "epoch_end_year": end,
                "reducer": reducer_name,
            }
        )
    )


# =============================================================================
# Z-scores and hot pixels (Earth Engine)
# =============================================================================
def lst_zscore(
    image: "ee.Image",
    params: dict[str, Any],
    region: "ee.Geometry",
    band: str | None = None,
    scale_m: int | None = None,
    ddof: int | None = None,
) -> "ee.Image":
    """Standardise an LST image against its own spatial mean and spread.

    Per scene (or per composite): both the mean and the standard deviation come
    from this image over ``region``, so the result says how unusual each pixel
    is *within that scene*. That makes z-scores comparable between dates in a
    way raw LST is not — but for the same reason it cannot show that a whole
    scene was hotter than another.

    .. note::
        The standard deviation comes from ``ee.Reducer.stdDev()``, whose degrees
        of freedom the Earth Engine documentation does not state. See
        :func:`resolve_ddof`; ``uhi.zscore.ddof`` is provisional until notebook
        03 measures it. The difference is of order 1/n and is immaterial over a
        scene of thousands of pixels, but it matters for exact agreement with
        :func:`zscore_array`.

    Args:
        image: Image carrying the LST band.
        params: Parsed params mapping.
        region: Region defining the mean and standard deviation.
        band: LST band; defaults to ``landsat_c2l2.lst_band_name``.
        scale_m: Reduction scale; defaults to ``crs.analysis_scale_m``.
        ddof: Accepted for signature parity with :func:`zscore_array` and
            validated, but Earth Engine's reducer choice is fixed — a non-default
            value raises rather than silently doing nothing.

    Returns:
        Single-band ``ee.Image`` named by ``uhi.zscore.band_name`` (``LST_z``).

    Raises:
        ValueError: If ``ddof`` differs from ``uhi.zscore.ddof``.
    """
    import ee  # Deferred: see module docstring.

    configured = resolve_ddof(None, params)
    if resolve_ddof(ddof, params) != configured:
        raise ValueError(
            f"lst_zscore cannot honour ddof={ddof}: the degrees of freedom are "
            "whatever ee.Reducer.stdDev() uses, which is why uhi.zscore.ddof "
            "must be set to match it rather than chosen freely."
        )

    comp = params["composites"]
    target = band or params["landsat_c2l2"]["lst_band_name"]
    scale = int(scale_m or params["crs"]["analysis_scale_m"])
    layer = image.select([target])

    stats = layer.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
        geometry=region,
        scale=scale,
        maxPixels=comp["reduce_max_pixels"],
        tileScale=comp["tile_scale"],
    )
    mean = ee.Number(stats.get(f"{target}_mean"))
    spread = ee.Number(stats.get(f"{target}_stdDev"))
    return (
        layer.subtract(mean)
        .divide(spread)
        .rename(params["uhi"]["zscore"]["band_name"])
    )


def hot_pixel_mask(
    image: "ee.Image",
    params: dict[str, Any],
    region: "ee.Geometry",
    sigma: float | None = None,
    band: str | None = None,
    scale_m: int | None = None,
) -> "ee.Image":
    """0/1 mask of pixels at or above ``sigma`` standard deviations.

    The comparison is ``gte``, matching :func:`hot_pixel_flags`.

    A hot pixel here is hot *relative to the rest of this scene*. It is not a
    heat-health threshold, and it is still land surface temperature — a hot
    pixel is a hot roof or road surface, not an air temperature a resident
    experiences (CLAUDE.md caveat 1).

    Args:
        image: Image carrying the LST band.
        params: Parsed params mapping.
        region: Region defining the mean and standard deviation.
        sigma: Threshold in standard deviations; defaults to
            ``uhi.zscore.default_sigma`` (1.0). Pass 2.0 for the stricter mask.
        band: LST band; defaults to ``landsat_c2l2.lst_band_name``.
        scale_m: Reduction scale; defaults to ``crs.analysis_scale_m``.

    Returns:
        Single-band 0/1 ``ee.Image`` named ``hot_pixels``, carrying a ``sigma``
        property so a saved product can never lose which threshold made it.
    """
    threshold = resolve_sigma(sigma, params)
    scores = lst_zscore(image, params, region, band=band, scale_m=scale_m)
    return scores.gte(threshold).rename("hot_pixels").set("sigma", threshold)


# =============================================================================
# Zonal statistics by administrative division
# =============================================================================
def _division_reducer(reducers: Sequence[str]) -> "ee.Reducer":
    """Combine the named reducers, plus ``count``, into one reducer.

    Args:
        reducers: Names from :data:`DIVISION_REDUCERS`.

    Returns:
        The combined ``ee.Reducer``.

    Raises:
        ValueError: If ``reducers`` is empty or names an unsupported reducer.
    """
    import ee  # Deferred: see module docstring.

    names = list(reducers)
    if not names:
        raise ValueError(
            f"reducers must name at least one of {list(DIVISION_REDUCERS)}"
        )
    unknown = [name for name in names if name not in DIVISION_REDUCERS]
    if unknown:
        raise ValueError(
            f"unsupported reducer(s) {unknown}; available: "
            f"{list(DIVISION_REDUCERS)}"
        )

    factories = {
        "mean": ee.Reducer.mean,
        "median": ee.Reducer.median,
        "stdDev": ee.Reducer.stdDev,
        "min": ee.Reducer.min,
        "max": ee.Reducer.max,
    }
    combined = factories[names[0]]()
    for name in names[1:]:
        combined = combined.combine(factories[name](), sharedInputs=True)
    return combined.combine(ee.Reducer.count(), sharedInputs=True)


def zonal_by_division(
    image: "ee.Image",
    params: dict[str, Any],
    level: str = "gn",
    band: str | None = None,
    scale_m: int | None = None,
    reducers: Sequence[str] = ("mean", "median", "stdDev"),
) -> "pd.DataFrame":
    """Reduce any image to per-GN or per-DS statistics, in one request.

    The general-purpose aggregation helper for the rest of the project: Phase 5
    consumes its output as the input to Getis-Ord Gi* and Local Moran's I, and
    Phase 7 uses it to rank divisions for greening priority.

    Aggregating to divisions is itself a methodological choice, not a neutral
    summary — the same surface reduced to DS divisions rather than GN divisions
    gives different hot spots. That is the MAUP sensitivity CLAUDE.md requires
    to be reported, which is why ``level`` is a first-class argument rather than
    a hardcoded GN.

    .. note::
        **GN names are not unique within Colombo District.** The returned table
        leads with the pcode for that reason. Never join on the name alone.

    Args:
        image: Any single-band-selectable image.
        params: Parsed params mapping.
        level: ``"gn"`` (557 divisions) or ``"ds"`` (13).
        band: Band to reduce; defaults to ``landsat_c2l2.lst_band_name``.
        scale_m: Reduction scale; defaults to ``crs.analysis_scale_m``.
        reducers: Statistics to compute; ``count`` is always added, and is
            emitted as :data:`PIXEL_COUNT_COLUMN` (CLAUDE.md caveat 2 — a
            division mean over four cloud-free pixels is not the same datum as
            one over four thousand).

    Returns:
        ``pandas.DataFrame``, one row per division.

    Raises:
        ValueError: If ``level`` or a reducer name is invalid.
    """
    from colombo_uhi import aoi

    if level not in ("gn", "ds"):
        raise ValueError(f"level must be 'gn' or 'ds', got {level!r}")

    comp = params["composites"]
    target = band or params["landsat_c2l2"]["lst_band_name"]
    scale = int(scale_m or params["crs"]["analysis_scale_m"])
    names = list(reducers)
    combined = _division_reducer(names)

    divisions = (
        aoi.gn_divisions(params) if level == "gn" else aoi.ds_divisions(params)
    )
    reduced = image.select([target]).reduceRegions(
        collection=divisions,
        reducer=combined,
        scale=scale,
        tileScale=comp["tile_scale"],
    )
    features = reduced.getInfo()["features"]
    return build_division_frame(
        [feature["properties"] for feature in features],
        params,
        level,
        target,
        names,
    )


# =============================================================================
# Driver attribution (Earth Engine sampling + Python regression)
# =============================================================================
def built_up_fraction(params: dict[str, Any], year: int) -> "ee.Image":
    """Built-up surface fraction from GHSL, for the epoch covering a year.

    ``built_surface`` is square metres of built area per 100 m cell, so the
    fraction is that divided by ``datasets.ghsl_built.cell_area_m2``.

    .. note::
        GHSL is published in five-year epochs, so the requested year is SNAPPED
        DOWN to the epoch containing it — a 2023 "built fraction" is really the
        2020 layer. The chosen epoch is set as an ``epoch`` property on the
        result, and any table built from it should carry that number rather than
        the requested year.

    Args:
        params: Parsed params mapping.
        year: Calendar year of interest.

    Returns:
        Single-band ``ee.Image`` named ``built_fraction`` (0-1), with an
        ``epoch`` property.
    """
    import ee  # Deferred: see module docstring.

    cfg = params["datasets"]["ghsl_built"]
    interval = int(cfg["epoch_interval_years"])
    first = int(str(cfg["availability"][0])[:4])
    last = int(str(cfg["availability"][1])[:4])

    epoch = int(year) - (int(year) % interval)
    epoch = min(max(epoch, first), last)

    built = (
        ee.ImageCollection(cfg["id"])
        .filterDate(f"{epoch}-01-01", f"{epoch}-12-31")
        .first()
        .select(cfg["band"])
    )
    return (
        built.divide(float(cfg["cell_area_m2"]))
        .rename("built_fraction")
        .toFloat()
        .set("epoch", epoch)
    )


def driver_stack(
    source: str | Mapping[str, Any],
    params: dict[str, Any],
    year: int,
    region: "ee.Geometry",
    collection: "ee.ImageCollection | None" = None,
) -> "ee.Image":
    """One year's LST alongside every candidate driver, as a single image.

    The spectral indices are computed on that year's own composite, so the
    predictors and the response describe the same window. The built-up fraction
    comes from the nearest GHSL epoch — see :func:`built_up_fraction`.

    Args:
        source: Source key or mapping; must be a Landsat source, since the
            indices need surface reflectance.
        params: Parsed params mapping.
        year: Calendar year.
        region: Region the source collection is filtered to.
        collection: Scene-level collection; ``None`` builds one WITH surface
            reflectance (unlike :func:`source_collection`, which drops it).

    Returns:
        ``ee.Image`` carrying the response band, the index bands and
        ``built_fraction``.

    Raises:
        ValueError: If the source is not a Landsat source.
    """
    from colombo_uhi import composites, indices, landsat

    resolved = resolve_source(source, params)
    if resolved["kind"] != "landsat":
        raise ValueError(
            f"driver_stack needs surface reflectance for the spectral indices, "
            f"so it requires a Landsat source; '{resolved['key']}' is "
            f"'{resolved['kind']}'"
        )

    band = params["landsat_c2l2"]["lst_band_name"]
    predictors = resolve_predictors(None, params)
    index_names = [
        name
        for name, out in indices.INDEX_BAND_NAMES.items()
        if out in predictors
    ]

    if collection is None:
        collection = landsat.harmonised_collection(
            params, region=region, include_st_qa=False
        )

    annual = composites.annual_composites(
        collection,
        params,
        band=band,
        reducer=resolved["reducer"],
        months=source_months(resolved, params),
        with_percentile=False,
        start_year=int(year),
        end_year=int(year),
    ).first()

    # The indices need the harmonised reflectance bands, which annual_composites
    # drops (it composites one band at a time). Composite them separately with
    # the same window and reducer, then join.
    reflectance = _reflectance_composite(collection, params, resolved, int(year))
    stacked = indices.add_indices(reflectance, params, which=index_names)

    import ee  # Deferred: see module docstring.

    return ee.Image.cat(
        [
            annual.select([band]),
            stacked.select([indices.INDEX_BAND_NAMES[name] for name in index_names]),
            built_up_fraction(params, int(year)),
        ]
    ).set(params["composites"]["year_property"], int(year))


def _reflectance_composite(
    collection: "ee.ImageCollection",
    params: dict[str, Any],
    source: Mapping[str, Any],
    year: int,
) -> "ee.Image":
    """Composite the harmonised reflectance bands for one year.

    Args:
        collection: A harmonised Landsat collection carrying the SR bands.
        params: Parsed params mapping.
        source: Resolved source mapping (for the reducer and month window).
        year: Calendar year.

    Returns:
        ``ee.Image`` with the harmonised reflectance band names.
    """
    import ee  # Deferred: see module docstring.

    from colombo_uhi import composites, landsat

    sr_bands = list(params["landsat_c2l2"]["harmonised_sr_bands"])
    reducer_name = composites.resolve_reducer_name(source["reducer"], params)
    reducer = {"median": ee.Reducer.median, "mean": ee.Reducer.mean}[reducer_name]()

    scenes = collection.select(sr_bands).filterDate(
        f"{year}-01-01", f"{year + 1}-01-01"
    )
    months = source_months(source, params)
    if months:
        scenes = scenes.filter(landsat.month_filter(months))
    return scenes.reduce(reducer).rename(sr_bands)


def sample_drivers(
    source: str | Mapping[str, Any],
    params: dict[str, Any],
    year: int,
    region: "ee.Geometry",
    collection: "ee.ImageCollection | None" = None,
    n_pixels: int | None = None,
    seed: int | None = None,
    scale_m: int | None = None,
) -> "pd.DataFrame":
    """Sample pixels of one year's LST-and-drivers stack into a DataFrame.

    Args:
        source: Landsat source key or mapping.
        params: Parsed params mapping.
        year: Calendar year.
        region: Region to sample within.
        collection: Scene-level collection; ``None`` builds one.
        n_pixels: Sample size; defaults to ``uhi.drivers.sample_pixels``.
        seed: Random seed; defaults to ``uhi.drivers.sample_seed``, so a rerun
            reproduces the same sample and the same coefficients.
        scale_m: Sampling scale; defaults to ``uhi.drivers.sample_scale_m``,
            deliberately coarser than the 30 m grid because adjacent 30 m
            pixels are near-duplicates.

    Returns:
        ``pandas.DataFrame``, one row per sampled pixel, with the response,
        the index columns, ``built_fraction`` and ``year``.
    """
    import pandas as pd

    cfg = params["uhi"]["drivers"]
    count = int(cfg["sample_pixels"] if n_pixels is None else n_pixels)
    random_seed = int(cfg["sample_seed"] if seed is None else seed)
    scale = int(cfg["sample_scale_m"] if scale_m is None else scale_m)

    stack = driver_stack(source, params, year, region, collection=collection)
    features = stack.sample(
        region=region,
        scale=scale,
        numPixels=count,
        seed=random_seed,
        dropNulls=True,
        geometries=False,
        tileScale=params["composites"]["tile_scale"],
    ).getInfo()["features"]

    frame = pd.DataFrame([feature["properties"] for feature in features])
    frame[params["composites"]["year_property"]] = int(year)
    return frame


def driver_series(
    source: str | Mapping[str, Any],
    params: dict[str, Any],
    region: "ee.Geometry",
    years: Sequence[int] | None = None,
    collection: "ee.ImageCollection | None" = None,
    progress: bool = False,
    **kwargs: Any,
) -> tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
    """Per-year OLS and correlation of LST against its drivers.

    One Earth Engine round trip per year (the sample), then the regression runs
    locally. Years whose sample is too small to fit are reported and skipped
    rather than aborting the series — a single cloud-wrecked year should not
    cost the other twenty-five.

    Args:
        source: Landsat source key or mapping.
        params: Parsed params mapping.
        region: Region to sample within.
        years: Years to fit; defaults to ``time.start_year`` through
            ``time.end_year``.
        collection: Scene-level collection; ``None`` builds one, reused across
            years.
        progress: Print a line per year.
        **kwargs: Forwarded to :func:`sample_drivers`.

    Returns:
        ``(coefficients, correlations, samples)``: the per-year OLS table, the
        per-year Pearson table, and the pooled sampled pixels (kept so the
        notebook can draw the scatter plots from the very rows the models were
        fitted on).
    """
    import pandas as pd

    from colombo_uhi import landsat

    year_prop = params["composites"]["year_property"]
    requested = (
        list(range(int(params["time"]["start_year"]), int(params["time"]["end_year"]) + 1))
        if years is None
        else [int(year) for year in years]
    )
    if collection is None:
        collection = landsat.harmonised_collection(
            params, region=region, include_st_qa=False
        )

    coefficient_frames: list["pd.DataFrame"] = []
    correlation_frames: list["pd.DataFrame"] = []
    sample_frames: list["pd.DataFrame"] = []

    for year in requested:
        sample = sample_drivers(
            source, params, year, region, collection=collection, **kwargs
        )
        sample_frames.append(sample)
        try:
            fit = fit_drivers(sample, params)
            correlation = driver_correlations(sample, params)
        except ValueError as error:
            # Not fatal: a year can legitimately have too few clear pixels.
            # Say so, keep the sample, and continue with the rest of the series.
            warnings.warn(
                f"skipping driver fit for {year}: {error}",
                stacklevel=2,
            )
            if progress:
                print(f"  {year}: SKIPPED ({len(sample)} sampled rows)")
            continue
        fit[year_prop] = year
        correlation[year_prop] = year
        coefficient_frames.append(fit)
        correlation_frames.append(correlation)
        if progress:
            r_squared = float(fit["r_squared"].iloc[0])
            print(f"  {year}: n={len(sample)}, R2={r_squared:.3f}")

    coefficients = (
        pd.concat(coefficient_frames, ignore_index=True)
        if coefficient_frames
        else pd.DataFrame()
    )
    correlations = (
        pd.concat(correlation_frames, ignore_index=True)
        if correlation_frames
        else pd.DataFrame()
    )
    samples = (
        pd.concat(sample_frames, ignore_index=True)
        if sample_frames
        else pd.DataFrame()
    )
    return coefficients, correlations, samples
