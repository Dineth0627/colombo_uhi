"""Trend analysis: Mann-Kendall, Sen's slope, and Benjamini-Hochberg FDR.

This is the only layer in the project that measures whether Colombo is actually
getting *hotter*. Phase 3's UTFVI epoch maps cannot answer that: UTFVI's
reference is the year's own mean, so a uniformly warming city produces no class
change at all. Epoch-to-epoch UTFVI drift is redistribution; the Sen's slope
here is warming. Nothing in this module "confirms" the UTFVI maps, and no
output may be described that way.

Products:
    * :func:`trend_image` - the pixel-wise product: Sen's slope in degC/yr,
      Kendall's tau, S, Var(S), Z, a TWO-SIDED p-value, and the per-pixel valid
      year count (CLAUDE.md caveat 2);
    * :func:`benjamini_hochberg` / :func:`fdr_significant_fraction` - pure-numpy
      FDR control and the "how much of the AOI is significantly warming" summary;
    * :func:`apply_fdr_to_raster` - the authoritative path: read the exported
      p-value GeoTIFF back, apply FDR, write the masked slope alongside it;
    * :func:`decadal_means` / :func:`decadal_difference` - the 2000-2010 /
      2011-2020 / 2021-2025 comparison and its difference maps;
    * :func:`mk_comparison` / :func:`suhii_trends` / :func:`zonal_trend_table` -
      modified Mann-Kendall (Hamed & Rao) on aggregate and per-division series,
      always beside the uncorrected test so the autocorrelation effect is visible.

Four things in here are easy to get wrong and are therefore pinned by unit tests
and stated loudly in the relevant docstrings:

1. **Mann-Kendall must never touch the sub-annual stack.** Scene-level LST is
   dominated by the seasonal cycle and by an irregular observation calendar, so
   MK on it measures the sampling, not the climate. Two structural layers
   prevent it: :func:`trend_image` takes a source KEY and builds its own annual
   series, and :func:`require_annual_series` refuses anything that is not one.
2. **The community tutorial's sign function truncates floats to zero.** See
   :func:`signum_array`. Following it verbatim would collapse most of S.
3. **The community tutorial's p-value is ONE-SIDED.** Benjamini-Hochberg needs
   two-sided input. See :func:`two_sided_p`.
4. **A nodata fill in a downloaded p-value raster sorts first.** A ``-9999``
   read as a p-value becomes the most significant pixel in the AOI.
   :func:`benjamini_hochberg` refuses any value outside [0, 1].

Design notes:
    * ``import ee`` (and numpy/pandas/scipy/pymannkendall/rasterio) is deferred
      into function bodies so this module, and the local pytest suite, import
      cleanly without ``earthengine-api``.
    * Every constant comes from ``config/params.yaml`` (``trends`` section).
    * Validation runs BEFORE the deferred import, so it stays unit-testable.
    * Server-side only: no ``.getInfo()`` inside a loop over images. Client round
      trips are one per year-batch, exactly as in :mod:`colombo_uhi.composites`.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, never at runtime
    import ee
    import numpy as np
    import pandas as pd

#: Band names of the two-band fit stack, in the order the reducers consume them.
#: ``ee.Reducer.sensSlope`` is a TWO-input reducer taking x THEN y; reversing
#: them returns the RECIPROCAL slope rather than an error.
FIT_X_BAND = "fit_x"
FIT_Y_BAND = "fit_y"

#: Output names ``ee.Reducer.sensSlope`` produces, in order.
SEN_OUTPUTS: tuple[str, ...] = ("slope", "offset")

#: Output names ``ee.Reducer.kendallsCorrelation`` produces, in order. Note the
#: HYPHEN in ``p-value`` - it is not a valid Python identifier and not a band
#: name we ever keep.
KENDALL_OUTPUTS: tuple[str, ...] = ("tau", "p-value")

#: Mann-Kendall statistic routes. See ``trends.mk_method`` in params.
MK_METHODS: tuple[str, ...] = ("tau_derived", "pairwise")

#: FDR procedures :func:`benjamini_hochberg` implements.
FDR_METHODS: tuple[str, ...] = ("benjamini_hochberg", "benjamini_yekutieli")

#: Modified Mann-Kendall variants :func:`mk_comparison` can dispatch to.
MMK_METHODS: tuple[str, ...] = (
    "hamed_rao",
    "yue_wang",
    "prewhitening",
    "trend_free_prewhitening",
)

#: Sentinel trend labels returned instead of raising on degenerate series.
TREND_INSUFFICIENT = "insufficient_data"
TREND_NO_VARIANCE = "no_variance"
TREND_OK = "ok"

#: Column order of the tidy Mann-Kendall comparison table.
MK_COLUMNS: tuple[str, ...] = (
    "label",
    "series",
    "test",
    "status",
    "n_years",
    "n_missing",
    "gapped",
    "trend",
    "h",
    "p",
    "z",
    "tau",
    "s",
    "var_s",
    "slope",
    "intercept",
    "slope_index_based",
    "var_inflation",
)

#: Column order of the per-division trend table.
ZONAL_TREND_COLUMNS: tuple[str, ...] = (
    "zone_id",
    "name",
    "n_years",
    "slope",
    "tau",
    "s",
    "var_s",
    "z",
    "p",
    "p_adjusted",
    "significant",
    "status",
)


# =============================================================================
# Pure helpers - parameter resolution (no Earth Engine; unit-tested)
# =============================================================================
def resolve_alpha(alpha: float | None, params: dict[str, Any]) -> float:
    """Validate the FDR significance level.

    Args:
        alpha: Explicit level, or ``None`` to read ``trends.fdr.alpha``.
        params: Parsed params mapping.

    Returns:
        The validated level.

    Raises:
        ValueError: Unless ``0 < alpha < 1``. Both endpoints are degenerate:
            ``0`` can never reject and ``1`` always does, so either silently
            turns the significance map into a constant.
    """
    resolved = float(params["trends"]["fdr"]["alpha"] if alpha is None else alpha)
    if not 0.0 < resolved < 1.0:
        raise ValueError(
            f"alpha must be strictly between 0 and 1, got {resolved}; at 0 no "
            "pixel can ever be significant and at 1 every pixel is"
        )
    return resolved


def resolve_fdr_method(method: str | None, params: dict[str, Any]) -> str:
    """Validate the FDR procedure name.

    Args:
        method: Explicit name, or ``None`` to read ``trends.fdr.method``.
        params: Parsed params mapping.

    Returns:
        One of :data:`FDR_METHODS`.

    Raises:
        ValueError: If the name is not implemented.
    """
    resolved = str(params["trends"]["fdr"]["method"] if method is None else method)
    if resolved not in FDR_METHODS:
        raise ValueError(
            f"unsupported FDR method '{resolved}'; expected one of "
            f"{list(FDR_METHODS)}"
        )
    return resolved


def resolve_mk_method(method: str | None, params: dict[str, Any]) -> str:
    """Validate the Mann-Kendall statistic route.

    Args:
        method: Explicit name, or ``None`` to read ``trends.mk_method``.
        params: Parsed params mapping.

    Returns:
        One of :data:`MK_METHODS`.

    Raises:
        ValueError: If the name is unknown.
    """
    resolved = str(params["trends"]["mk_method"] if method is None else method)
    if resolved not in MK_METHODS:
        raise ValueError(
            f"unsupported mk_method '{resolved}'; expected one of "
            f"{list(MK_METHODS)}"
        )
    return resolved


def resolve_min_valid_obs(value: int | None, params: dict[str, Any]) -> int | None:
    """Resolve the per-pixel-per-year observation floor for trend fitting.

    ``composites.min_valid_obs`` is deliberately ``null`` - Phase 2 flags but
    never masks, so nothing is silently hidden. Phase 4 is where a floor is
    finally applied, because a "year" resting on one cloud-edge scene is a
    number, not an observation, and Mann-Kendall cannot tell the difference.

    Args:
        value: Explicit floor, or ``None`` to read ``trends.min_valid_obs``.
        params: Parsed params mapping.

    Returns:
        The floor, or ``None`` for no floor.

    Raises:
        ValueError: If the floor is not a positive integer.
    """
    resolved = params["trends"]["min_valid_obs"] if value is None else value
    if resolved is None:
        return None
    if isinstance(resolved, bool) or not isinstance(resolved, (int, float)):
        raise ValueError(f"min_valid_obs must be an integer or null, got {resolved!r}")
    resolved = int(resolved)
    if resolved < 1:
        raise ValueError(
            f"min_valid_obs must be >= 1 (it counts observations), got {resolved}; "
            "use null to disable the floor"
        )
    return resolved


def resolve_min_years(value: int | None, params: dict[str, Any]) -> int:
    """Resolve the per-pixel valid-year floor below which MK statistics are masked.

    Args:
        value: Explicit floor, or ``None`` to read ``trends.min_years``.
        params: Parsed params mapping.

    Returns:
        The floor.

    Raises:
        ValueError: If fewer than 3. Var(S) and the normal approximation behind
            Z are meaningless below that, so an unmasked Z there is not a weak
            result but a fabricated one.
    """
    resolved = int(params["trends"]["min_years"] if value is None else value)
    if resolved < 3:
        raise ValueError(
            f"min_years must be >= 3, got {resolved}; Var(S) and the normal "
            "approximation behind Z are not defined on fewer points, so the "
            "resulting Z would be fabricated rather than merely weak"
        )
    return resolved


def resolve_x_origin(
    x_origin_year: int | None,
    params: dict[str, Any],
    start_year: int | None = None,
) -> int:
    """Resolve the year the Sen's-slope x axis is measured from.

    The SLOPE is identical whichever origin is chosen, and is degC/yr either
    way. Only the offset differs: centred on ``start_year`` it is the fitted LST
    in that year, which is directly interpretable; absolute, it is the fitted
    LST at year 0 AD - a 2000-year extrapolation with roughly 1e-5 relative
    precision in float32 and no meaning in a caption.

    Args:
        x_origin_year: Explicit origin year, or ``None`` to read
            ``trends.x_origin``.
        params: Parsed params mapping.
        start_year: The series start, used when the mode is ``"start_year"``.

    Returns:
        The origin year (``0`` for absolute years).

    Raises:
        ValueError: If ``trends.x_origin`` is not a recognised mode.
    """
    if x_origin_year is not None:
        return int(x_origin_year)

    mode = str(params["trends"]["x_origin"])
    if mode == "absolute":
        return 0
    if mode == "start_year":
        return int(
            params["time"]["start_year"] if start_year is None else start_year
        )
    raise ValueError(
        f"unsupported x_origin '{mode}'; expected 'start_year' or 'absolute'"
    )


def resolve_decades(
    decades: Mapping[str, Sequence[int]] | None,
    params: dict[str, Any],
) -> list[tuple[str, int, int]]:
    """Resolve and validate the decadal comparison windows.

    .. note::
        These are deliberately SEPARATE from ``uhi.utfvi.epochs``. The
        boundaries differ, ``test_params`` pins that the UTFVI epochs tile the
        study period, and the two answer different questions: UTFVI is
        within-epoch spatial redistribution against a moving reference, this is
        absolute temperature level.

    .. warning::
        The configured windows are 11 / 10 / 5 years - UNEQUAL by construction,
        because the study period ends in 2025. The last window rests on roughly
        half the sample of the others, so its mean carries about sqrt(2) the
        standard error and dominates the uncertainty of any difference map it
        appears in. This is not fixable by re-slicing; it is reported.

    Args:
        decades: Explicit mapping of label to ``[start, end]``, or ``None`` to
            read ``trends.decades``.
        params: Parsed params mapping.

    Returns:
        ``(label, start_year, end_year)`` triples, ascending by start year.

    Raises:
        ValueError: If a window is malformed, inverted, or overlaps another.
    """
    resolved = params["trends"]["decades"] if decades is None else decades
    if not resolved:
        raise ValueError("trends.decades is empty; at least one window is required")

    windows: list[tuple[str, int, int]] = []
    for label, bounds in resolved.items():
        if len(bounds) != 2:
            raise ValueError(
                f"decade '{label}' must be [start_year, end_year], got {list(bounds)}"
            )
        start, end = int(bounds[0]), int(bounds[1])
        if end < start:
            raise ValueError(
                f"decade '{label}' is inverted: end_year ({end}) < start_year ({start})"
            )
        windows.append((str(label), start, end))

    windows.sort(key=lambda window: window[1])
    for earlier, later in zip(windows, windows[1:]):
        if later[1] <= earlier[2]:
            raise ValueError(
                f"decades '{earlier[0]}' ({earlier[1]}-{earlier[2]}) and "
                f"'{later[0]}' ({later[1]}-{later[2]}) overlap; a year counted "
                "in two windows appears on both sides of a difference map"
            )
    return windows


def decade_years(params: dict[str, Any], decade: str) -> tuple[int, int]:
    """Look up one decadal window's year bounds.

    Args:
        params: Parsed params mapping.
        decade: Window label, e.g. ``"2011_2020"``.

    Returns:
        ``(start_year, end_year)``.

    Raises:
        KeyError: If the label is not configured, listing those that are.
    """
    for label, start, end in resolve_decades(None, params):
        if label == decade:
            return start, end
    raise KeyError(
        f"unknown decade '{decade}'; trends.decades defines "
        f"{[label for label, _, _ in resolve_decades(None, params)]}"
    )


def resolve_tie_correction(enabled: bool | None, params: dict[str, Any]) -> bool:
    """Resolve whether the Kendall tie correction is applied.

    Defaults off, with cause. Ties are detected by EXACT float equality, and
    ``LST_C`` is a float in degrees Celsius produced by a median composite - two
    different years returning bit-identical values is effectively impossible, so
    the correction term is measurably zero while the ``arraySort``/``arrayMask``
    machinery it needs is expensive on a 26-image stack.

    Args:
        enabled: Explicit flag, or ``None`` to read ``trends.tie_correction``.
        params: Parsed params mapping.

    Returns:
        Whether to apply the correction.

    Raises:
        ValueError: If enabled without ``mk_method: "pairwise"``, since the
            correction only has a meaning inside the explicit pairwise route.
    """
    resolved = bool(
        params["trends"]["tie_correction"] if enabled is None else enabled
    )
    if resolved and params["trends"]["mk_method"] != "pairwise":
        raise ValueError(
            "tie_correction is only meaningful with mk_method: 'pairwise'; the "
            "tau_derived route obtains Var(S) from the closed form. Set "
            "trends.mk_method to 'pairwise' or turn tie_correction off."
        )
    return resolved


# =============================================================================
# Pure helpers - Mann-Kendall arithmetic (no Earth Engine; unit-tested)
# =============================================================================
def signum_array(
    later: "np.ndarray | Sequence[float] | float",
    earlier: "np.ndarray | Sequence[float] | float",
) -> "np.ndarray":
    """Exact -1 / 0 / +1 sign of ``later - earlier``, without integer truncation.

    .. warning::
        The Earth Engine community tutorial "Non-Parametric Trend Analysis"
        computes this as ``diff.clamp(-1, 1).int()``. ``.int()`` TRUNCATES
        TOWARD ZERO, so a year-to-year difference of +0.3 degC becomes a sign of
        ``0`` instead of ``+1``. Most annual LST differences in Colombo are well
        under 1 degC, so following the tutorial verbatim zeroes out most of the
        Mann-Kendall statistic and collapses every trend toward "no change". The
        tutorial states its own scope: it is for discrete, non-floating-point
        data. ``gt(0) - lt(0)`` is exact for any float and needs no clamp.

    Args:
        later: The later observation(s).
        earlier: The earlier observation(s).

    Returns:
        Float array of -1.0, 0.0 or +1.0; ``NaN`` wherever either input is NaN,
        so a missing year drops the pair rather than contributing a spurious 0.
    """
    import numpy as np  # Deferred: see module docstring.

    diff = np.asarray(later, dtype="float64") - np.asarray(earlier, dtype="float64")
    signs = (diff > 0).astype("float64") - (diff < 0).astype("float64")
    return np.where(np.isnan(diff), np.nan, signs)


def kendall_variance(n: "np.ndarray | Sequence[float] | float") -> "np.ndarray":
    """Var(S) for Mann-Kendall with no ties: ``n(n-1)(2n+5)/18``.

    This is the community tutorial's ``factors()`` expression with the tie term
    zero. See :func:`resolve_tie_correction` for why the tie term is zero on
    continuous LST.

    Args:
        n: Number of observations per pixel or series.

    Returns:
        Float array of Var(S); ``NaN`` where ``n < 3``, below which the variance
        is not usable.
    """
    import numpy as np  # Deferred: see module docstring.

    count = np.asarray(n, dtype="float64")
    variance = count * (count - 1.0) * (2.0 * count + 5.0) / 18.0
    return np.where(count >= 3, variance, np.nan)


def s_from_tau(
    tau: "np.ndarray | Sequence[float] | float",
    n: "np.ndarray | Sequence[float] | float",
) -> "np.ndarray":
    """Recover the Mann-Kendall S statistic from Kendall's tau-b.

    ``tau_b = S / sqrt((n0 - n1)(n0 - n2))`` with ``n0 = n(n-1)/2``. Our x is the
    calendar year, so ties in x are impossible by construction, and ties in float
    LST are effectively impossible - therefore ``n1 = n2 = 0`` and
    ``tau_b = S / n0`` exactly.

    .. note::
        Where ties DO exist this is conservative, not wrong: tau_b understates
        ``S / n0``, so the recovered S is too small, |Z| too small and p too
        large. Erring toward fewer significant pixels is the safe direction.

    Args:
        tau: Kendall's tau-b.
        n: Number of observations.

    Returns:
        S, rounded to the nearest integer (S is a sum of +/-1 terms); ``NaN``
        where ``n < 3``.
    """
    import numpy as np  # Deferred: see module docstring.

    correlation = np.asarray(tau, dtype="float64")
    count = np.asarray(n, dtype="float64")
    pairs = count * (count - 1.0) / 2.0
    statistic = np.rint(correlation * pairs)
    return np.where(count >= 3, statistic, np.nan)


def z_from_s(
    s: "np.ndarray | Sequence[float] | float",
    var_s: "np.ndarray | Sequence[float] | float",
) -> "np.ndarray":
    """Continuity-corrected Mann-Kendall Z statistic.

    ``Z = (S-1)/sqrt(Var)`` for ``S > 0``, ``0`` for ``S == 0``, and
    ``(S+1)/sqrt(Var)`` for ``S < 0`` - the standard correction, and the form the
    community tutorial uses. The correction shrinks |Z| toward zero, so omitting
    it would overstate significance on short series.

    Args:
        s: Mann-Kendall S.
        var_s: Var(S).

    Returns:
        Float array of Z; ``NaN`` where Var(S) is NaN or non-positive.
    """
    import numpy as np  # Deferred: see module docstring.

    statistic = np.asarray(s, dtype="float64")
    variance = np.asarray(var_s, dtype="float64")

    with np.errstate(invalid="ignore", divide="ignore"):
        spread = np.sqrt(variance)
        adjusted = statistic - np.sign(statistic)
        z = np.where(statistic == 0.0, 0.0, adjusted / spread)
    return np.where(np.isfinite(spread) & (spread > 0), z, np.nan)


def two_sided_p(z: "np.ndarray | Sequence[float] | float") -> "np.ndarray":
    """Two-sided normal p-value: ``2 * (1 - Phi(|Z|))``, i.e. ``erfc(|Z|/sqrt(2))``.

    .. warning::
        The Earth Engine community tutorial emits the ONE-SIDED ``1 - Phi(|Z|)``
        and compensates by thresholding at 0.025 for a two-sided 95% test.
        Benjamini-Hochberg must be fed two-sided p-values; feeding it the
        one-sided form halves every p and roughly doubles the area reported as
        significantly warming.

    Args:
        z: Z statistic.

    Returns:
        Float array in [0, 1]; ``NaN`` where Z is NaN.
    """
    import numpy as np  # Deferred: see module docstring.
    from scipy import special  # Deferred: see module docstring.

    statistic = np.asarray(z, dtype="float64")
    with np.errstate(invalid="ignore"):
        p = special.erfc(np.abs(statistic) / math.sqrt(2.0))
    return np.where(np.isnan(statistic), np.nan, np.clip(p, 0.0, 1.0))


def mk_statistics_from_tau(
    tau: "np.ndarray | Sequence[float] | float",
    n: "np.ndarray | Sequence[float] | float",
) -> dict[str, "np.ndarray"]:
    """S, Var(S), Z and the two-sided p, from Kendall's tau and the sample size.

    The pure reference implementation of the server-side ``tau_derived`` route,
    and the only way the Earth Engine result can be checked against a number
    computed from first principles.

    Args:
        tau: Kendall's tau-b.
        n: Number of observations.

    Returns:
        Mapping with keys ``s``, ``var_s``, ``z`` and ``p``.
    """
    statistic = s_from_tau(tau, n)
    variance = kendall_variance(n)
    z = z_from_s(statistic, variance)
    return {"s": statistic, "var_s": variance, "z": z, "p": two_sided_p(z)}


def sens_slope_array(
    years: "np.ndarray | Sequence[float]",
    values: "np.ndarray | Sequence[float]",
) -> tuple[float, float]:
    """Theil-Sen slope and intercept: the median of all pairwise slopes.

    The reference implementation of the server-side Sen's slope. NaN-safe by
    PAIRWISE DELETION - a pair contributes only if both endpoints are finite,
    which is exactly the rule the masked fit stack enforces server-side.

    .. warning::
        This uses the ACTUAL years, so the slope is degC per YEAR even when
        years are missing. ``pymannkendall.sens_slope`` uses the array INDEX as
        x, so on a gapped series it silently reports degC per OBSERVATION. The
        two agree only when no year is missing.

    Args:
        years: The x values, in calendar years.
        values: The y values.

    Returns:
        ``(slope, intercept)``; ``(nan, nan)`` when fewer than two usable pairs
        exist. The intercept is ``median(y - slope*x)``, the Theil-Sen
        convention, so it is the fitted value at x = 0.

    Raises:
        ValueError: If the two inputs have different lengths.
    """
    import numpy as np  # Deferred: see module docstring.

    x = np.asarray(years, dtype="float64")
    y = np.asarray(values, dtype="float64")
    if x.shape != y.shape:
        raise ValueError(
            f"years and values must be the same length, got {x.shape} and {y.shape}"
        )

    usable = np.isfinite(x) & np.isfinite(y)
    x, y = x[usable], y[usable]
    if x.size < 2:
        return (float("nan"), float("nan"))

    rows, cols = np.triu_indices(x.size, k=1)
    run = x[cols] - x[rows]
    with np.errstate(invalid="ignore", divide="ignore"):
        slopes = (y[cols] - y[rows]) / run
    # A zero run (two observations in the same year) yields inf or NaN; both are
    # dropped here rather than allowed to dominate the median.
    slopes = slopes[np.isfinite(slopes)]
    if slopes.size == 0:
        return (float("nan"), float("nan"))

    slope = float(np.median(slopes))
    return (slope, float(np.median(y - slope * x)))


# =============================================================================
# Pure helpers - Benjamini-Hochberg FDR (no Earth Engine; unit-tested)
# =============================================================================
def benjamini_hochberg(
    p_values: "np.ndarray | Sequence[float]",
    alpha: float | None = None,
    params: dict[str, Any] | None = None,
    method: str | None = None,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Step-up false-discovery-rate control, NaN-safe and shape-preserving.

    .. warning::
        Benjamini-Hochberg controls the FDR under independence or positive
        regression dependency. A land-surface-temperature raster is strongly
        spatially autocorrelated over hundreds of metres, so the REALISED
        false-discovery proportion varies far more than its controlled
        expectation even though that expectation holds. Report the
        Benjamini-Yekutieli figure - valid under arbitrary dependence - beside
        it, and read the significant area as a screening result.

    Args:
        p_values: Raw p-values, any shape. ``NaN`` marks a pixel that was never
            tested.
        alpha: Significance level; defaults to ``trends.fdr.alpha``.
        params: Parsed params mapping. Required unless both ``alpha`` and
            ``method`` are given explicitly.
        method: One of :data:`FDR_METHODS`; defaults to ``trends.fdr.method``.

    Returns:
        ``(reject, p_adjusted)``, both the shape of the input. ``reject`` is
        boolean and is ``False`` wherever the input was NaN - a pixel that was
        never tested is not a non-significant result and must never be drawn as
        one. ``p_adjusted`` is float64 with NaN preserved in place.

    Raises:
        ValueError: If a finite p-value lies outside [0, 1]. This is a real
            guard, not a formality: a GeoTIFF read back from Drive carries a
            nodata fill, and a ``-9999`` sorts first and becomes the most
            significant pixel in the AOI.
    """
    import numpy as np  # Deferred: see module docstring.

    if params is None:
        if alpha is None or method is None:
            raise ValueError(
                "benjamini_hochberg needs either `params`, or both `alpha` and "
                "`method` given explicitly"
            )
        level = float(alpha)
        if not 0.0 < level < 1.0:
            raise ValueError(f"alpha must be strictly between 0 and 1, got {level}")
        procedure = str(method)
        if procedure not in FDR_METHODS:
            raise ValueError(
                f"unsupported FDR method '{procedure}'; expected one of "
                f"{list(FDR_METHODS)}"
            )
    else:
        level = resolve_alpha(alpha, params)
        procedure = resolve_fdr_method(method, params)

    array = np.asarray(p_values, dtype="float64")
    shape = array.shape
    flat = array.ravel()

    finite = np.isfinite(flat)
    if finite.any():
        offending = flat[finite][(flat[finite] < 0.0) | (flat[finite] > 1.0)]
        if offending.size:
            raise ValueError(
                f"{offending.size} p-value(s) lie outside [0, 1], first is "
                f"{offending[0]}. A raster read back from Drive carries a nodata "
                "fill (commonly -9999 or 0); an unmasked fill sorts first and "
                "becomes the most significant pixel in the AOI. Apply the "
                "raster's nodata value, or pass a validity mask, before "
                "correcting."
            )

    reject = np.zeros(flat.shape, dtype=bool)
    adjusted = np.full(flat.shape, np.nan, dtype="float64")

    m = int(finite.sum())
    if m == 0:
        if flat.size:
            warnings.warn(
                f"the p-value array is COMPLETELY EMPTY: all {flat.size} values "
                "are NaN, so nothing was tested. Do not map or interpret the "
                "result. Usual causes, in order: the wrong band was read, the "
                "raster's nodata value was not applied, or min_years masked "
                "every pixel.",
                stacklevel=2,
            )
        return reject.reshape(shape), adjusted.reshape(shape)

    values = flat[finite]
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    ranks = np.arange(1, m + 1, dtype="float64")

    scale = float(m)
    if procedure == "benjamini_yekutieli":
        # Valid under arbitrary dependence, at the cost of a log-factor penalty.
        scale *= float(np.sum(1.0 / ranks))

    q = ranked * scale / ranks
    # Reverse cumulative minimum: without it the adjusted values are not
    # monotone in the raw p-values, and a pixel with a smaller p can end up with
    # a larger q than its neighbour.
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    sorted_reject = q <= level

    scattered_q = np.empty(m, dtype="float64")
    scattered_q[order] = q
    scattered_reject = np.zeros(m, dtype=bool)
    scattered_reject[order] = sorted_reject

    adjusted[finite] = scattered_q
    reject[finite] = scattered_reject
    return reject.reshape(shape), adjusted.reshape(shape)


def trend_validity_mask(
    p_values: "np.ndarray",
    slope: "np.ndarray",
    n_years: "np.ndarray",
    params: dict[str, Any],
    min_years: int | None = None,
) -> "np.ndarray":
    """Boolean mask of the pixels that were genuinely TESTED.

    A pixel counts as tested only when its p-value and slope are both finite AND
    its valid-year count meets the floor. Getting this wrong is an error in
    either direction: too permissive and nodata fill inflates ``m``, dragging the
    Benjamini-Hochberg threshold down for every real pixel; too strict and
    warming that was actually measured is thrown away.

    Args:
        p_values: Raw p-value array.
        slope: Sen's slope array, same shape.
        n_years: Valid-year count array, same shape.
        params: Parsed params mapping.
        min_years: Override for ``trends.min_years``.

    Returns:
        Boolean array, ``True`` where the pixel was tested.
    """
    import numpy as np  # Deferred: see module docstring.

    floor = resolve_min_years(min_years, params)
    return (
        np.isfinite(p_values)
        & np.isfinite(slope)
        & np.isfinite(n_years)
        & (np.asarray(n_years, dtype="float64") >= floor)
    )


def fdr_significant_fraction(
    p_values: "np.ndarray | Sequence[float]",
    params: dict[str, Any],
    alpha: float | None = None,
    method: str | None = None,
    slope: "np.ndarray | Sequence[float] | None" = None,
    pixel_area_m2: float | None = None,
) -> dict[str, Any]:
    """Summarise an FDR result: how much of the AOI is significantly trending.

    .. note::
        Both denominators are returned because they are different numbers and
        quoting the wrong one overstates the result. Untested pixels - cloud
        starved, water masked, or below ``min_years`` - are neither significant
        nor non-significant, so ``fraction_of_tested`` and ``fraction_of_total``
        must be reported together with their counts.

    Args:
        p_values: Raw p-values.
        params: Parsed params mapping.
        alpha: Override for ``trends.fdr.alpha``.
        method: Override for ``trends.fdr.method``.
        slope: Sen's slope, same shape, used to split warming from cooling.
        pixel_area_m2: Area of one pixel, used to report areas in km2.

    Returns:
        Mapping with the method and level, the pixel counts, both fractions, the
        warming/cooling split and, when ``pixel_area_m2`` is given, areas in km2.
    """
    import numpy as np  # Deferred: see module docstring.

    level = resolve_alpha(alpha, params)
    procedure = resolve_fdr_method(method, params)
    reject, adjusted = benjamini_hochberg(
        p_values, alpha=level, params=params, method=procedure
    )

    array = np.asarray(p_values, dtype="float64")
    tested = np.isfinite(array)
    n_total = int(array.size)
    n_tested = int(tested.sum())
    n_significant = int(reject.sum())

    summary: dict[str, Any] = {
        "method": procedure,
        "alpha": level,
        "n_total": n_total,
        "n_tested": n_tested,
        "n_significant": n_significant,
        "fraction_of_tested": (n_significant / n_tested) if n_tested else float("nan"),
        "fraction_of_total": (n_significant / n_total) if n_total else float("nan"),
        "max_adjusted_p_rejected": (
            float(np.nanmax(adjusted[reject])) if n_significant else float("nan")
        ),
    }

    if slope is not None:
        slopes = np.asarray(slope, dtype="float64")
        warming = int(np.sum(reject & (slopes > 0)))
        cooling = int(np.sum(reject & (slopes < 0)))
        summary.update(
            {
                "n_warming": warming,
                "n_cooling": cooling,
                "fraction_warming_of_tested": (
                    warming / n_tested if n_tested else float("nan")
                ),
                "fraction_cooling_of_tested": (
                    cooling / n_tested if n_tested else float("nan")
                ),
            }
        )

    if pixel_area_m2:
        km2 = float(pixel_area_m2) / 1e6
        summary["area_km2_tested"] = n_tested * km2
        summary["area_km2_significant"] = n_significant * km2
        if slope is not None:
            summary["area_km2_warming"] = summary["n_warming"] * km2
            summary["area_km2_cooling"] = summary["n_cooling"] * km2

    return summary


# =============================================================================
# Pure helpers - series validation (no Earth Engine; unit-tested)
# =============================================================================
def validate_series_metadata(
    metadata: Mapping[str, Any],
    params: dict[str, Any],
    start_year: int | None = None,
    end_year: int | None = None,
) -> dict[str, Any]:
    """Raise unless the fetched metadata describes a complete annual series.

    Split out from :func:`require_annual_series` so the guard's LOGIC is testable
    without Earth Engine - which matters more here than anywhere else in the
    project, because this function is the only thing standing between a user and
    a Mann-Kendall test fitted to 1674 irregularly spaced scenes.

    Args:
        metadata: Mapping with keys ``size``, ``series_basis``, ``years``,
            ``months`` and (optionally) ``n_scenes``, as fetched in one round
            trip by :func:`require_annual_series`.
        params: Parsed params mapping.
        start_year: If given with ``end_year``, require exactly this year range.
        end_year: See ``start_year``.

    Returns:
        Summary mapping with ``n_years``, ``first_year``, ``last_year`` and
        ``empty_years`` - printed by the notebook, so the guard doubles as the
        series inventory and nobody is tempted to skip it.

    Raises:
        ValueError: If the collection is not a complete, strictly ascending
            annual composite series.
    """
    comp = params["composites"]
    expected_basis = params["trends"]["series_basis"]

    size = int(metadata.get("size") or 0)
    if size < 2:
        raise ValueError(
            f"a trend needs at least two years, got a collection of {size} "
            "image(s); widen start_year/end_year"
        )

    basis = list(metadata.get("series_basis") or [])
    if len(basis) != size:
        raise ValueError(
            f"only {len(basis)} of {size} images carry a "
            f"'{comp['series_basis_property']}' property. Either this is not an "
            "annual composite series, or your composites.py predates Phase 4 - "
            "re-run the clone cell and the module-purge cell, then rebuild the "
            "series with trends.annual_series()."
        )
    distinct = sorted(set(basis))
    if distinct != [expected_basis]:
        raise ValueError(
            f"series_basis is {distinct}, expected ['{expected_basis}']. "
            "Mann-Kendall is only defined here for the annual composite series."
        )

    months = list(metadata.get("months") or [])
    if months:
        raise ValueError(
            f"{len(months)} images carry a '{comp['season_property']}'-level "
            "`month` property, so this is a SCENE collection, not an annual "
            "series. Mann-Kendall on sub-annual data measures the seasonal cycle "
            "and the observation calendar, not a climate trend. Build the series "
            "with trends.annual_series(source, params) and pass that instead."
        )

    years = [int(year) for year in (metadata.get("years") or [])]
    if len(years) != size:
        raise ValueError(
            f"only {len(years)} of {size} images carry a "
            f"'{comp['year_property']}' property; the series is not usable for "
            "trend fitting"
        )
    duplicates = sorted({year for year in years if years.count(year) > 1})
    if duplicates:
        raise ValueError(
            f"years are not unique: {duplicates} appear more than once. More than "
            "one image per year means this is not an annual series - "
            "annual_composites() emits exactly one composite per calendar year."
        )
    for earlier, later in zip(years, years[1:]):
        if later <= earlier:
            raise ValueError(
                f"years are not strictly ascending: {earlier} is followed by "
                f"{later}. Sen's slope and Kendall's tau are computed over the "
                "collection in order, so an unsorted series silently fits the "
                "wrong pairs."
            )

    if start_year is not None and end_year is not None:
        expected = list(range(int(start_year), int(end_year) + 1))
        missing = sorted(set(expected) - set(years))
        if missing:
            raise ValueError(
                f"the series is missing year(s) {missing}. annual_composites() "
                "emits empty years with obs_count == 0 so the axis stays "
                "complete, so a gap here means the series was filtered after it "
                "was built - rebuild it with start_year/end_year instead of "
                "calling .filter()."
            )

    scenes = list(metadata.get("n_scenes") or [])
    empty_years = [
        year for year, count in zip(years, scenes) if count is not None and int(count) == 0
    ]
    if empty_years:
        warnings.warn(
            f"{len(empty_years)} year(s) in the series rest on ZERO scenes: "
            f"{empty_years}. They are still on the year axis with obs_count == 0, "
            "so the trend will be fitted over the remaining years - but read the "
            "n_years band before interpreting any pixel in those regions.",
            stacklevel=3,
        )

    return {
        "n_years": size,
        "first_year": years[0],
        "last_year": years[-1],
        "empty_years": empty_years,
    }


def resolve_reduced_band_names(
    observed: Sequence[str],
    suffixes: Sequence[str],
    label: str,
    band: str | None = None,
) -> list[str]:
    """Map a reducer's observed output bands onto its expected outputs, in order.

    Earth Engine names reducer outputs differently depending on whether the
    reducer reports one input or several: ``composites._composite_reducer``
    already records that a multi-input combined reducer makes
    ``ImageCollection.reduce`` emit BARE output names, while a single-input one
    emits ``<band>_<output>``. Rather than depend on which case applies, this
    resolves both and falls back to position.

    Args:
        observed: Band names the reduce actually produced.
        suffixes: Expected output names, in order, e.g. ``("slope", "offset")``.
        label: Reducer name, used in the error message.
        band: Input band name, for the ``<band>_<output>`` spelling.

    Returns:
        The observed band names corresponding to ``suffixes``, in order.

    Raises:
        RuntimeError: If they cannot be matched, naming EVERY observed band and
            the fix.
    """
    names = list(observed)

    exact = [suffix for suffix in suffixes if suffix in names]
    if len(exact) == len(suffixes):
        return exact

    if band:
        prefixed = [f"{band}_{suffix}" for suffix in suffixes]
        if all(name in names for name in prefixed):
            return prefixed

    if len(names) == len(suffixes):
        warnings.warn(
            f"{label} returned bands {names}, which match neither {list(suffixes)} "
            f"nor the '<band>_<output>' spelling; falling back to POSITION. Run "
            "the band-name probe cell in notebook 04 and record the real names in "
            "trends.bands so this stops being a guess.",
            stacklevel=2,
        )
        return names

    raise RuntimeError(
        f"{label} returned {len(names)} band(s) {names}, but "
        f"{len(suffixes)} were expected {list(suffixes)}. The reducer's output "
        "naming has changed. Run the band-name probe cell in notebook 04 "
        "(Step 2), then update trends.bands in config/params.yaml."
    )


# =============================================================================
# Pure helpers - table shaping (no Earth Engine; unit-tested)
# =============================================================================
def build_mk_frame(rows: Sequence[Mapping[str, Any]]) -> "pd.DataFrame":
    """Shape Mann-Kendall result rows into the :data:`MK_COLUMNS` table.

    Args:
        rows: Result mappings, one per (series, test).

    Returns:
        ``pd.DataFrame`` with columns :data:`MK_COLUMNS`; empty but correctly
        shaped when no rows are given.
    """
    import pandas as pd  # Deferred: see module docstring.

    if not rows:
        return pd.DataFrame(columns=list(MK_COLUMNS))
    frame = pd.DataFrame(list(rows))
    for column in MK_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[list(MK_COLUMNS)]


# =============================================================================
# Earth Engine products - the annual series and its guard
# =============================================================================
def annual_series(
    source: str | Mapping[str, Any],
    params: dict[str, Any],
    region: "ee.Geometry | None" = None,
    collection: "ee.ImageCollection | None" = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> "ee.ImageCollection":
    """Build the annual composite series a trend may legitimately be fitted to.

    This is layer 1 of the structural guard. It takes a source KEY, not a
    collection, and builds the series itself - so there is no parameter through
    which a sub-annual scene stack can reach the Mann-Kendall code.

    Args:
        source: Key into ``uhi.suhii.sources``, or an explicit source mapping.
        params: Parsed params mapping.
        region: Region bounding the work; passed to the scene collection.
        collection: Optional pre-built SCENE-level collection, used purely as a
            cache to avoid rebuilding it per call. It is never treated as a
            series - it is composited here exactly as if it had been built here.
        start_year: Defaults to ``time.start_year``.
        end_year: Defaults to ``time.end_year``.

    Returns:
        ``ee.ImageCollection`` with one composite per calendar year, ascending.
    """
    from colombo_uhi import composites, uhi_metrics

    resolved = uhi_metrics.resolve_source(source, params)
    scenes = (
        collection
        if collection is not None
        else uhi_metrics.source_collection(resolved, params, region=region)
    )
    return composites.annual_composites(
        scenes,
        params,
        reducer=resolved["reducer"],
        months=uhi_metrics.source_months(resolved, params),
        # No trend product ever reads LST_C_p90, and a percentile reducer must
        # retain every observation per pixel in order to sort them, whereas mean
        # and count stream. Turning it off is free here and is the cheapest
        # memory lever available.
        with_percentile=False,
        start_year=start_year,
        end_year=end_year,
    )


def require_annual_series(
    collection: "ee.ImageCollection",
    params: dict[str, Any],
    start_year: int | None = None,
    end_year: int | None = None,
) -> dict[str, Any]:
    """Refuse to fit a trend to anything that is not an annual composite series.

    Layer 2 of the structural guard. Costs exactly ONE ``getInfo``, with every
    property array wrapped into a single ``ee.List`` - separate calls would be
    independent evaluations whose element order is not guaranteed to align, and
    which desync in length when a property is missing on some images.

    ``aggregate_array`` on a property only some images carry returns a SHORTER
    array, which is precisely why ``size`` is fetched alongside and length
    checked: "half of these are scenes" becomes a loud error rather than a
    silent pass.

    Args:
        collection: The collection to check.
        params: Parsed params mapping.
        start_year: If given with ``end_year``, require exactly this year range.
        end_year: See ``start_year``.

    Returns:
        The summary from :func:`validate_series_metadata`.

    Raises:
        ValueError: If the collection is not a complete annual composite series.
    """
    import ee  # Deferred: see module docstring.

    comp = params["composites"]
    size, basis, years, months, n_scenes = ee.List(
        [
            collection.size(),
            collection.aggregate_array(comp["series_basis_property"]),
            collection.aggregate_array(comp["year_property"]),
            collection.aggregate_array("month"),
            collection.aggregate_array(comp["n_scenes_property"]),
        ]
    ).getInfo()

    return validate_series_metadata(
        {
            "size": size,
            "series_basis": basis,
            "years": years,
            "months": months,
            "n_scenes": n_scenes,
        },
        params,
        start_year=start_year,
        end_year=end_year,
    )


def fit_stack(
    series: "ee.ImageCollection",
    params: dict[str, Any],
    band: str | None = None,
    min_valid_obs: int | None = None,
    x_origin_year: int | None = None,
    start_year: int | None = None,
    validate: bool = True,
) -> "ee.ImageCollection":
    """Two-band ``[x, y]`` collection ready for sensSlope and kendallsCorrelation.

    .. note::
        The x band is explicitly masked to the y band's mask.
        ``ee.Image.constant`` is unmasked everywhere, so without this a pixel-year
        with no valid LST would still carry a valid x. Masking them together
        gives the pairwise-deletion semantics both Sen's slope and Kendall's tau
        require, without depending on undocumented multi-input reducer behaviour.

    Args:
        series: An annual composite series.
        params: Parsed params mapping.
        band: LST band; defaults to ``landsat_c2l2.lst_band_name``.
        min_valid_obs: Override for ``trends.min_valid_obs``.
        x_origin_year: Override for the x-axis origin.
        start_year: Series start, used to resolve a ``start_year`` origin.
        validate: Run :func:`require_annual_series` first. Leave this on unless
            the caller has already validated the same collection.

    Returns:
        ``ee.ImageCollection`` of two-band images, ``x`` then ``y``.
    """
    import ee  # Deferred: see module docstring.

    comp = params["composites"]
    target = band or params["landsat_c2l2"]["lst_band_name"]
    floor = resolve_min_valid_obs(min_valid_obs, params)
    origin = resolve_x_origin(x_origin_year, params, start_year=start_year)
    obs_band = comp["obs_count_band"]
    year_prop = comp["year_property"]

    if validate:
        require_annual_series(series, params)

    def to_pair(image: "ee.Image") -> "ee.Image":
        year = ee.Number(image.get(year_prop))
        lst = image.select([target])
        if floor is not None:
            lst = lst.updateMask(image.select([obs_band]).gte(floor))
        x = (
            ee.Image.constant(year.subtract(origin))
            .toDouble()
            .rename(FIT_X_BAND)
            .updateMask(lst.mask())
        )
        return ee.Image.cat([x, lst.toDouble().rename(FIT_Y_BAND)]).set(
            {year_prop: year, "x_origin_year": origin}
        )

    return series.map(to_pair)


def valid_year_count(
    stack: "ee.ImageCollection",
    params: dict[str, Any],
) -> "ee.Image":
    """Per-pixel count of years contributing to the fit (CLAUDE.md caveat 2).

    Args:
        stack: A fit stack from :func:`fit_stack`.
        params: Parsed params mapping.

    Returns:
        Single-band ``ee.Image`` named ``trends.bands.n_years``, unmasked to 0 so
        a pixel with no usable year reads as 0 rather than as a hole.
    """
    name = params["trends"]["bands"]["n_years"]
    return stack.select([FIT_Y_BAND]).count().rename(name).unmask(0).toFloat()


def sen_slope_image(
    stack: "ee.ImageCollection",
    params: dict[str, Any],
) -> "ee.Image":
    """Sen's slope and offset from ``ee.Reducer.sensSlope``.

    .. warning::
        ``ee.Reducer.sensSlope`` is a TWO-input reducer taking x THEN y.
        Reversing the inputs returns the RECIPROCAL slope, not an error. The band
        order is driven by ``trends.sen_input_order`` and probed in notebook 04
        against a collection of known slope 2.0.

    Args:
        stack: A fit stack from :func:`fit_stack`.
        params: Parsed params mapping.

    Returns:
        Two-band ``ee.Image``: ``trends.bands.sen_slope`` in degC/yr, and
        ``trends.bands.sen_offset`` in degC at the x origin.
    """
    import ee  # Deferred: see module docstring.

    bands = params["trends"]["bands"]
    order = list(params["trends"]["sen_input_order"])
    inputs = [FIT_X_BAND if role == "x" else FIT_Y_BAND for role in order]

    reduced = stack.select(inputs).reduce(ee.Reducer.sensSlope())
    names = resolve_reduced_band_names(
        reduced.bandNames().getInfo(), SEN_OUTPUTS, "ee.Reducer.sensSlope"
    )
    return reduced.select(names, [bands["sen_slope"], bands["sen_offset"]]).toFloat()


def kendall_image(
    stack: "ee.ImageCollection",
    params: dict[str, Any],
) -> "ee.Image":
    """Kendall's tau-b and the reducer's own p-value.

    .. note::
        ``trends.bands.mk_p_ee`` is exported for COMPARISON ONLY and is never fed
        to the FDR correction: the sidedness and tie convention of the reducer's
        p-value are undocumented. Its ratio to ``mk_p_two_sided`` over the AOI is
        the empirical answer - a ratio near 2 means the reducer's p is one-sided.

    Args:
        stack: A fit stack from :func:`fit_stack`.
        params: Parsed params mapping.

    Returns:
        Two-band ``ee.Image``: ``trends.bands.mk_tau`` and
        ``trends.bands.mk_p_ee``.
    """
    import ee  # Deferred: see module docstring.

    bands = params["trends"]["bands"]
    num_inputs = int(params["trends"]["mk_num_inputs"])

    if num_inputs == 2:
        source = stack.select([FIT_X_BAND, FIT_Y_BAND])
        probe_band = None
    else:
        source = stack.select([FIT_Y_BAND])
        probe_band = FIT_Y_BAND

    reduced = source.reduce(ee.Reducer.kendallsCorrelation(num_inputs))
    names = resolve_reduced_band_names(
        reduced.bandNames().getInfo(),
        KENDALL_OUTPUTS,
        "ee.Reducer.kendallsCorrelation",
        band=probe_band,
    )
    return reduced.select(names, [bands["mk_tau"], bands["mk_p_ee"]]).toFloat()


def normal_cdf(z: "ee.Image") -> "ee.Image":
    """Standard normal CDF: ``0.5 * (1 + erf(z / sqrt(2)))``."""
    import ee  # Deferred: see module docstring.

    return ee.Image(0.5).multiply(
        ee.Image(1).add(z.divide(ee.Image(2).sqrt()).erf())
    )


def mk_statistics(
    tau: "ee.Image",
    n_years: "ee.Image",
    params: dict[str, Any],
) -> "ee.Image":
    """S, Var(S), Z and the TWO-SIDED p, derived from tau and the year count.

    The server-side twin of :func:`mk_statistics_from_tau`. See that function for
    why ``S = tau * n(n-1)/2`` is exact with no ties and conservative with them,
    and :func:`two_sided_p` for why the p-value here is not the tutorial's.

    Args:
        tau: Kendall's tau-b image.
        n_years: Valid-year count image.
        params: Parsed params mapping.

    Returns:
        Four-band ``ee.Image``: ``mk_s``, ``mk_var_s``, ``mk_z`` and
        ``mk_p_two_sided``.
    """
    import ee  # Deferred: see module docstring.

    bands = params["trends"]["bands"]
    n = n_years.toDouble()

    pairs = n.multiply(n.subtract(1)).divide(2)
    s = tau.toDouble().multiply(pairs).round().rename(bands["mk_s"])
    var_s = (
        n.multiply(n.subtract(1))
        .multiply(n.multiply(2).add(5))
        .divide(18)
        .rename(bands["mk_var_s"])
    )

    # Continuity correction: (S-1)/sqrt(var) for S>0, 0 for S==0, (S+1)/sqrt(var)
    # for S<0. Built with where() so the S==0 case is exact rather than a
    # division that happens to land near zero.
    spread = var_s.sqrt()
    adjusted = s.subtract(s.signum())
    z = adjusted.divide(spread).where(s.eq(0), 0).rename(bands["mk_z"])

    # p = 2*(1 - Phi(|Z|)) = erfc(|Z|/sqrt(2)) = 1 - erf(|Z|/sqrt(2)).
    p = (
        ee.Image(1)
        .subtract(z.abs().divide(ee.Image(2).sqrt()).erf())
        .clamp(0, 1)
        .rename(bands["mk_p_two_sided"])
    )

    return ee.Image.cat([s, var_s, z, p]).toFloat()


def trend_image(
    source: str | Mapping[str, Any],
    params: dict[str, Any],
    region: "ee.Geometry | None" = None,
    collection: "ee.ImageCollection | None" = None,
    band: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    min_valid_obs: int | None = None,
    min_years: int | None = None,
    method: str | None = None,
) -> "ee.Image":
    """The pixel-wise trend product, from a source key.

    This is the phase's headline product and the primary entry point. It takes a
    source KEY rather than a collection precisely so that a sub-annual stack
    cannot be handed to Mann-Kendall - see the module docstring, point 1.

    Args:
        source: Key into ``uhi.suhii.sources``, e.g. ``"landsat_dry"``.
        params: Parsed params mapping.
        region: Region bounding the work.
        collection: Optional pre-built SCENE-level collection, used as a cache.
        band: LST band; defaults to ``landsat_c2l2.lst_band_name``.
        start_year: Defaults to ``time.start_year``.
        end_year: Defaults to ``time.end_year``.
        min_valid_obs: Override for ``trends.min_valid_obs``.
        min_years: Override for ``trends.min_years``.
        method: Override for ``trends.mk_method``.

    Returns:
        ``ee.Image`` carrying every band in ``trends.export_band_order``.
        Statistical bands are masked where ``n_years`` is below the floor;
        ``n_years`` itself is never masked, so an excluded pixel reads as
        "excluded" rather than "missing".

    Raises:
        NotImplementedError: If ``method`` is ``"pairwise"``. That route builds
            325 pairwise images from a 26-year series, each carrying two full
            composite graphs, and is not implemented as a production path - use
            ``tau_derived``, which is exact with no ties.
    """
    routing = resolve_mk_method(method, params)
    if routing == "pairwise":
        raise NotImplementedError(
            "mk_method 'pairwise' is not implemented as a production route: a "
            "26-year series produces 325 pairwise sign images, each carrying two "
            "full annual-composite graphs, and Earth Engine's user memory limit "
            "is exceeded by 26 of them alone. Use 'tau_derived', which recovers "
            "S exactly from tau when there are no ties (x is the calendar year, "
            "so ties in x are impossible) and conservatively when there are."
        )

    floor_years = resolve_min_years(min_years, params)
    first = int(params["time"]["start_year"] if start_year is None else start_year)

    series = annual_series(
        source,
        params,
        region=region,
        collection=collection,
        start_year=start_year,
        end_year=end_year,
    )
    stack = fit_stack(
        series,
        params,
        band=band,
        min_valid_obs=min_valid_obs,
        start_year=first,
        validate=True,
    )

    n_years = valid_year_count(stack, params)
    sen = sen_slope_image(stack, params)
    kendall = kendall_image(stack, params)
    statistics = mk_statistics(kendall.select([params["trends"]["bands"]["mk_tau"]]), n_years, params)

    # Mask the STATISTICS, never the count: a pixel dropped for too few years
    # must remain visibly excluded (CLAUDE.md caveat 2).
    enough = n_years.gte(floor_years)
    statistical = sen.addBands(kendall).addBands(statistics).updateMask(enough)

    return statistical.addBands(n_years).select(
        list(params["trends"]["export_band_order"])
    )


def verify_trend_bands(
    image: "ee.Image",
    params: dict[str, Any],
    region: "ee.Geometry",
    scale_m: int | None = None,
) -> dict[str, Any]:
    """Check the trend image's bands are numerically what their names claim.

    A name check cannot catch an output ORDER swap, because tau and p are both
    dimensionless and both small. This can: **tau is negative about half the
    time and a p-value never is**, and a p-value is bounded by 1 while Z is not.

    Args:
        image: A :func:`trend_image` result.
        params: Parsed params mapping.
        region: Region to check over - use a small box, not the full AOI.
        scale_m: Reduction scale; defaults to ``trends.fit_scale_m``.

    Returns:
        Mapping of band name to its observed ``(min, max)``.

    Raises:
        RuntimeError: If a band is outside its mathematically possible range,
            naming the band and the observed values.
    """
    import ee  # Deferred: see module docstring.

    bands = params["trends"]["bands"]
    comp = params["composites"]
    scale = int(params["trends"]["fit_scale_m"] if scale_m is None else scale_m)

    stats = image.reduceRegion(
        reducer=ee.Reducer.min().combine(ee.Reducer.max(), sharedInputs=True),
        geometry=region,
        scale=scale,
        maxPixels=comp["reduce_max_pixels"],
        tileScale=comp["tile_scale"],
    ).getInfo()

    observed = {
        name: (stats.get(f"{name}_min"), stats.get(f"{name}_max"))
        for name in image.bandNames().getInfo()
    }

    checks = [
        (bands["mk_p_two_sided"], 0.0, 1.0),
        (bands["mk_p_ee"], 0.0, 1.0),
        (bands["mk_tau"], -1.0, 1.0),
    ]
    for name, low, high in checks:
        band_min, band_max = observed.get(name, (None, None))
        if band_min is None or band_max is None:
            continue
        if band_min < low - 1e-6 or band_max > high + 1e-6:
            raise RuntimeError(
                f"band '{name}' spans [{band_min}, {band_max}], outside its "
                f"possible range [{low}, {high}]. The reducer's outputs are most "
                "likely in a different ORDER than assumed - a swap that correct "
                "band NAMES cannot reveal, because tau and p are both small and "
                "dimensionless. Run the band-name probe cell in notebook 04 "
                "(Step 2) and correct trends.bands."
            )
    return observed


# =============================================================================
# Earth Engine products - decadal means
# =============================================================================
def decadal_means(
    source: str | Mapping[str, Any],
    params: dict[str, Any],
    region: "ee.Geometry | None" = None,
    collection: "ee.ImageCollection | None" = None,
    decades: Mapping[str, Sequence[int]] | None = None,
    band: str | None = None,
    min_valid_obs: int | None = None,
) -> "ee.Image":
    """Mean LST per decadal window, with its spread and contributing year count.

    .. note::
        The mean is over the ANNUAL COMPOSITES, so every year counts once. This
        is deliberately the opposite of :func:`uhi_metrics.epoch_composite`,
        which composites all the window's scenes and so weights years by scene
        count. That is right for UTFVI, which wants the observation-weighted
        thermal field, and wrong here: it would make part of any decade-to-decade
        difference a change in the Landsat constellation rather than in
        temperature.

    Args:
        source: Key into ``uhi.suhii.sources``.
        params: Parsed params mapping.
        region: Region bounding the work.
        collection: Optional pre-built SCENE-level collection, used as a cache.
        decades: Override for ``trends.decades``.
        band: LST band; defaults to ``landsat_c2l2.lst_band_name``.
        min_valid_obs: Override for ``trends.min_valid_obs``.

    Returns:
        ``ee.Image`` with ``mean_<label>``, ``sd_<label>`` and ``n_years_<label>``
        for every window. Means are masked below ``trends.decades_min_years``;
        the counts are not.
    """
    import ee  # Deferred: see module docstring.

    target = band or params["landsat_c2l2"]["lst_band_name"]
    floor = resolve_min_valid_obs(min_valid_obs, params)
    min_window_years = int(params["trends"]["decades_min_years"])
    obs_band = params["composites"]["obs_count_band"]

    layers: list["ee.Image"] = []
    for label, start, end in resolve_decades(decades, params):
        series = annual_series(
            source,
            params,
            region=region,
            collection=collection,
            start_year=start,
            end_year=end,
        )

        def usable(image: "ee.Image") -> "ee.Image":
            lst = image.select([target])
            if floor is not None:
                lst = lst.updateMask(image.select([obs_band]).gte(floor))
            return lst

        annual = series.map(usable)
        counts = annual.count().rename(f"n_years_{label}").unmask(0)
        enough = counts.gte(min_window_years)
        layers.append(
            ee.Image.cat(
                [
                    annual.mean().rename(f"mean_{label}").updateMask(enough),
                    annual.reduce(ee.Reducer.stdDev())
                    .rename(f"sd_{label}")
                    .updateMask(enough),
                    counts,
                ]
            )
        )

    return ee.Image.cat(layers).toFloat()


def decadal_difference(
    means: "ee.Image",
    params: dict[str, Any],
    later: str,
    earlier: str,
) -> "ee.Image":
    """Difference between two decadal means, with its standard error.

    .. warning::
        The configured windows are 11 / 10 / 5 years. Any difference involving
        ``2021_2025`` has its uncertainty dominated by that window's smaller
        sample, which is why ``diff_se`` and ``diff_z`` are emitted rather than
        the bare difference. A +0.4 degC difference against the 5-year window may
        be indistinguishable from interannual noise while the same difference
        between the 11- and 10-year windows is not.

    Args:
        means: A :func:`decadal_means` result.
        params: Parsed params mapping.
        later: Label of the later window.
        earlier: Label of the earlier window.

    Returns:
        ``ee.Image`` with ``diff_<later>_minus_<earlier>``, ``diff_se``,
        ``diff_z`` and ``n_years_min``.

    Raises:
        KeyError: If either label is not configured.
    """
    import ee  # Deferred: see module docstring.

    decade_years(params, later)
    decade_years(params, earlier)

    mean_a = means.select([f"mean_{later}"])
    mean_b = means.select([f"mean_{earlier}"])
    sd_a = means.select([f"sd_{later}"])
    sd_b = means.select([f"sd_{earlier}"])
    n_a = means.select([f"n_years_{later}"])
    n_b = means.select([f"n_years_{earlier}"])

    difference = mean_a.subtract(mean_b).rename(f"diff_{later}_minus_{earlier}")
    standard_error = (
        sd_a.pow(2).divide(n_a).add(sd_b.pow(2).divide(n_b)).sqrt().rename("diff_se")
    )
    return ee.Image.cat(
        [
            difference,
            standard_error,
            difference.divide(standard_error).rename("diff_z"),
            n_a.min(n_b).rename("n_years_min"),
        ]
    ).toFloat()


# =============================================================================
# Earth Engine products - stratification by class
# =============================================================================
def trend_by_class(
    image: "ee.Image",
    params: dict[str, Any],
    region: "ee.Geometry",
    scheme: str,
    band: str | None = None,
    scale_m: int | None = None,
    reducers: Sequence[str] | None = None,
) -> "pd.DataFrame":
    """Sen's slope summarised per land-cover or LCZ class, in one round trip.

    A thin wrapper over :func:`colombo_uhi.landcover.stratified_stats` that
    defaults the band to the Sen's slope and the scale to the fit scale.

    .. warning::
        The class layers are fixed snapshots - WorldCover is 2021, LCZ is derived
        from 2018-2019 imagery. This is therefore a PRESENT-DAY stratification of
        a 2000-2025 trend: "where the warming is, by today's land cover". It is
        NOT land-cover-change attribution. A pixel that was paddy in 2002 and is
        built now sits in the built class for its whole history. Attributing
        trend TO land-cover change is Phase 6.

    Args:
        image: A :func:`trend_image` result.
        params: Parsed params mapping.
        region: Region to summarise over.
        scheme: Class scheme key, e.g. ``"worldcover"`` or ``"lcz"``.
        band: Band to summarise; defaults to ``trends.bands.sen_slope``.
        scale_m: Reduction scale; defaults to ``trends.fit_scale_m``.
        reducers: Statistics to compute; defaults to ``("mean", "stdDev")``.

    Returns:
        ``pd.DataFrame`` with one row per class, carrying the pixel count each
        class statistic rests on.
    """
    from colombo_uhi import landcover

    return landcover.stratified_stats(
        image,
        scheme,
        params,
        region,
        band=band or params["trends"]["bands"]["sen_slope"],
        scale_m=int(params["trends"]["fit_scale_m"] if scale_m is None else scale_m),
        reducers=reducers,
    )


# =============================================================================
# Client-side summaries
# =============================================================================
def sample_trend_arrays(
    image: "ee.Image",
    params: dict[str, Any],
    region: "ee.Geometry",
    scale_m: int | None = None,
    bands: Sequence[str] | None = None,
) -> dict[str, "np.ndarray"]:
    """Pull trend bands into local numpy arrays for the in-session FDR preview.

    .. note::
        This is the PREVIEW path. It exists so the significance map can be seen
        without waiting on a Drive export, and it runs at
        ``trends.fdr.preview_scale_m``, which is coarser than the fit scale on
        purpose. The authoritative significant-area figure comes from
        :func:`apply_fdr_to_raster` on the exported GeoTIFF.

    Args:
        image: A :func:`trend_image` result.
        params: Parsed params mapping.
        region: Region to sample - a rectangle, since the result is a grid.
        scale_m: Sampling scale; defaults to ``trends.fdr.preview_scale_m``.
        bands: Bands to pull; defaults to slope, two-sided p and the year count.

    Returns:
        Mapping of band name to a 2-D float64 array, masked pixels as ``NaN``.

    Raises:
        RuntimeError: If a requested band is absent from the response.
    """
    import ee  # Deferred: see module docstring.
    import numpy as np  # Deferred: see module docstring.

    band_cfg = params["trends"]["bands"]
    scale = int(
        params["trends"]["fdr"]["preview_scale_m"] if scale_m is None else scale_m
    )
    wanted = list(
        bands
        or [band_cfg["sen_slope"], band_cfg["mk_p_two_sided"], band_cfg["n_years"]]
    )

    # sampleRectangle returns each band as a nested list property. A masked pixel
    # needs a sentinel, since JSON has no NaN: -9999 is outside every band's
    # plausible range and is converted back to NaN below.
    sentinel = -9999.0
    response = (
        image.select(wanted)
        .unmask(sentinel)
        .sampleRectangle(region=region, defaultValue=sentinel)
        .getInfo()
    )
    properties = response.get("properties", {})

    missing = [name for name in wanted if name not in properties]
    if missing:
        raise RuntimeError(
            f"sampleRectangle did not return {missing}; it returned "
            f"{sorted(properties)}. Either the band names in trends.bands are "
            "wrong, or the region falls entirely outside the image."
        )

    arrays: dict[str, "np.ndarray"] = {}
    for name in wanted:
        array = np.asarray(properties[name], dtype="float64")
        arrays[name] = np.where(array <= sentinel + 1.0, np.nan, array)
    return arrays


def mk_comparison(
    values: "np.ndarray | Sequence[float]",
    params: dict[str, Any],
    years: Sequence[int] | None = None,
    label: str = "",
    series: str = "",
    method: str | None = None,
) -> "pd.DataFrame":
    """Plain Mann-Kendall AND the autocorrelation-corrected variant, side by side.

    The point of running both is the ``var_inflation`` column - modified Var(S)
    divided by original Var(S). Annual LST is positively autocorrelated, which
    inflates the true variance of S, so the plain test's p-value is
    ANTI-CONSERVATIVE. A single number tells the reader how much that mattered
    here; and if it comes out near 1, the plain test was fine and saying so is
    itself a result.

    .. warning::
        The test statistics come from ``pymannkendall`` on the NaN-dropped
        series, which is correct because S and tau depend only on the ORDER of
        the observations. The slope does NOT: ``pymannkendall`` uses the array
        index as x, so on a gapped series its slope is degC per OBSERVATION.
        ``slope`` and ``intercept`` here therefore come from
        :func:`sens_slope_array` on the real years, and the index-based value is
        carried alongside for comparison.

    Args:
        values: The series, in year order. ``NaN`` marks a missing year.
        params: Parsed params mapping.
        years: Calendar years matching ``values``; defaults to a dense range
            starting at ``time.start_year``.
        label: Identifier for the rows, e.g. a GN pcode or a source key.
        series: What the values are, e.g. ``"suhii"``.
        method: Override for ``trends.mmk.method``.

    Returns:
        ``pd.DataFrame`` with columns :data:`MK_COLUMNS`, one row per test.

    Raises:
        ValueError: If ``method`` is not one of :data:`MMK_METHODS`, or the year
            and value lengths differ.
    """
    import numpy as np  # Deferred: see module docstring.

    cfg = params["trends"]["mmk"]
    variant = str(cfg["method"] if method is None else method)
    if variant not in MMK_METHODS:
        raise ValueError(
            f"unsupported modified Mann-Kendall method '{variant}'; expected one "
            f"of {list(MMK_METHODS)}"
        )

    array = np.asarray(list(values), dtype="float64")
    if years is None:
        start = int(params["time"]["start_year"])
        year_values = np.arange(start, start + array.size, dtype="float64")
    else:
        year_values = np.asarray(list(years), dtype="float64")
        if year_values.shape != array.shape:
            raise ValueError(
                f"years and values must be the same length, got "
                f"{year_values.shape} and {array.shape}"
            )

    finite = np.isfinite(array)
    n_finite = int(finite.sum())
    n_missing = int(array.size - n_finite)
    missing_fraction = (n_missing / array.size) if array.size else 1.0
    gapped = missing_fraction > float(cfg["max_missing_fraction"])

    base: dict[str, Any] = {
        "label": label,
        "series": series,
        "n_years": n_finite,
        "n_missing": n_missing,
        "gapped": gapped,
    }

    if n_finite < int(cfg["min_years"]):
        return build_mk_frame(
            [
                {**base, "test": test, "status": TREND_INSUFFICIENT, "trend": TREND_INSUFFICIENT}
                for test in ("original", variant)
            ]
        )

    clean = array[finite]
    clean_years = year_values[finite]

    if float(np.ptp(clean)) == 0.0:
        return build_mk_frame(
            [
                {
                    **base,
                    "test": test,
                    "status": TREND_NO_VARIANCE,
                    "trend": "no trend",
                    "h": False,
                    "p": 1.0,
                    "z": 0.0,
                    "tau": 0.0,
                    "s": 0.0,
                    "slope": 0.0,
                    "intercept": float(clean[0]),
                }
                for test in ("original", variant)
            ]
        )

    if gapped:
        warnings.warn(
            f"series '{label}/{series}' is missing {n_missing} of {array.size} "
            f"years ({missing_fraction:.0%}). The Hamed-Rao variance correction "
            "estimates lag-k autocorrelation, which assumes regular spacing, so "
            "on a gapped series the correction is biased. The row is flagged "
            "`gapped`; read it as indicative.",
            stacklevel=2,
        )

    import pymannkendall as pmk  # Deferred: see module docstring.

    runners = {
        "original": pmk.original_test,
        "hamed_rao": pmk.hamed_rao_modification_test,
        "yue_wang": pmk.yue_wang_modification_test,
        "prewhitening": pmk.pre_whitening_modification_test,
        "trend_free_prewhitening": pmk.trend_free_pre_whitening_modification_test,
    }

    slope, intercept = sens_slope_array(clean_years, clean)

    rows: list[dict[str, Any]] = []
    original_var: float | None = None
    for test in ("original", variant):
        try:
            # pymannkendall divides by the detrended series' autocovariance and
            # takes the square root of the corrected variance, both of which can
            # be degenerate; numpy would otherwise emit a bare RuntimeWarning
            # from inside the library with no indication of which series it came
            # from.
            with np.errstate(invalid="ignore", divide="ignore"):
                result = runners[test](clean)
        except Exception as error:  # pragma: no cover - defensive, see docstring
            rows.append(
                {**base, "test": test, "status": f"failed: {error}", "trend": None}
            )
            continue

        variance = float(result.var_s)
        if test == "original":
            original_var = variance

        # The autocorrelation correction can return a non-finite or negative
        # Var(S), which makes z and p NaN. The clearest trigger is a series whose
        # detrended residuals are EXACTLY zero - a perfect arithmetic
        # progression - because the lag-0 autocovariance is then zero and the
        # correction divides by it. Real LST is never that clean, but a
        # constant-increment synthetic series or a heavily interpolated zonal
        # series can be, and the result is a limitation of the method rather
        # than a missing value. It is labelled rather than passed through as a
        # silent blank in a report table.
        degenerate = not (
            math.isfinite(variance)
            and variance > 0
            and math.isfinite(float(result.z))
            and math.isfinite(float(result.p))
        )
        if degenerate:
            warnings.warn(
                f"series '{label}/{series}': the '{test}' test returned a "
                f"degenerate result (var_s={variance}, z={result.z}, "
                f"p={result.p}). The autocorrelation correction is undefined "
                "when the detrended residuals have zero variance, and unstable "
                "when they are nearly deterministic. Quote the 'original' test "
                "for this series and say why.",
                stacklevel=2,
            )

        rows.append(
            {
                **base,
                "test": test,
                "status": TREND_OK if not degenerate else "degenerate_variance",
                "trend": result.trend,
                "h": bool(result.h),
                "p": float(result.p),
                "z": float(result.z),
                "tau": float(result.Tau),
                "s": float(result.s),
                "var_s": variance,
                "slope": slope,
                "intercept": intercept,
                "slope_index_based": float(result.slope),
                "var_inflation": (
                    variance / original_var
                    if original_var not in (None, 0.0)
                    and math.isfinite(variance)
                    and original_var is not None
                    and math.isfinite(original_var)
                    else None
                ),
            }
        )

    tolerance = float(cfg["slope_tolerance"])
    index_based = [row.get("slope_index_based") for row in rows if row.get("slope_index_based") is not None]
    if index_based and math.isfinite(slope):
        if abs(index_based[0] - slope) > tolerance:
            warnings.warn(
                f"series '{label}/{series}': year-based Sen's slope is "
                f"{slope:.4f} degC/yr but pymannkendall's INDEX-based slope is "
                f"{index_based[0]:.4f}, a difference of "
                f"{abs(index_based[0] - slope):.4f} over the "
                f"{tolerance} tolerance. This is expected when years are missing "
                "(pymannkendall's x is the array index, so its slope is degC per "
                "OBSERVATION). Quote the `slope` column, not "
                "`slope_index_based`.",
                stacklevel=2,
            )

    return build_mk_frame(rows)


def suhii_trends(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    value_columns: Sequence[str] = ("suhii", "urban_mean", "rural_mean"),
    method: str | None = None,
) -> "pd.DataFrame":
    """Mann-Kendall on the Phase 3 SUHII table, per source x rural definition.

    .. note::
        Running this on ``urban_mean`` and ``rural_mean`` as well as ``suhii``
        costs nothing and answers what the SUHII trend alone cannot: did SUHII
        rise because the city warmed, or because the countryside warmed less?
        Those are different findings with different implications, and the
        decomposition is one extra grouping.

    Args:
        frame: The Phase 3 SUHII table (``uhi_metrics.SUHII_COLUMNS``).
        params: Parsed params mapping.
        value_columns: Series to test.
        method: Override for ``trends.mmk.method``.

    Returns:
        ``pd.DataFrame`` with columns :data:`MK_COLUMNS`.

    Raises:
        KeyError: If a required column is absent, naming what the frame has.
    """
    import pandas as pd  # Deferred: see module docstring.

    required = {"year", "source", "rural_definition", "urban_pixels", "rural_pixels"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(
            f"the SUHII frame is missing {missing}; it has {sorted(frame.columns)}. "
            "Pass the table produced by uhi_metrics.suhii_all_sources()."
        )

    # Drop years with no valid pixels BEFORE fitting, exactly as the Phase 3
    # figure does, so the trend table and the figure can never disagree about
    # which years exist.
    usable = frame[(frame["urban_pixels"] > 0) & (frame["rural_pixels"] > 0)]

    results: list["pd.DataFrame"] = []
    for (source, definition), group in usable.groupby(
        ["source", "rural_definition"], sort=True
    ):
        ordered = group.sort_values("year")
        for column in value_columns:
            if column not in ordered.columns:
                continue
            results.append(
                mk_comparison(
                    ordered[column].to_numpy(dtype="float64"),
                    params,
                    years=ordered["year"].to_numpy(dtype="int64"),
                    label=f"{source}|{definition}",
                    series=column,
                    method=method,
                )
            )

    if not results:
        return build_mk_frame([])
    return pd.concat(results, ignore_index=True)


def zonal_annual_series(
    source: str | Mapping[str, Any],
    params: dict[str, Any],
    level: str = "gn",
    region: "ee.Geometry | None" = None,
    collection: "ee.ImageCollection | None" = None,
    band: str | None = None,
    scale_m: int | None = None,
    batch_years: int | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    progress: bool = False,
) -> "pd.DataFrame":
    """Long table of per-division annual mean LST: one row per division per year.

    .. note::
        Batching is by REBUILDING the series for a narrower year range, never by
        filtering a computed collection. An annual series is built with
        ``ee.ImageCollection.fromImages(ee.List.map(...))``, so a ``.filter()``
        forces Earth Engine to materialise every composite graph in order to
        evaluate the predicate - the opposite of the intended saving.

    Args:
        source: Key into ``uhi.suhii.sources``.
        params: Parsed params mapping.
        level: ``"gn"`` or ``"ds"``.
        region: Region bounding the work.
        collection: Optional pre-built SCENE-level collection, used as a cache.
        band: LST band; defaults to ``landsat_c2l2.lst_band_name``.
        scale_m: Reduction scale; defaults to ``trends.fit_scale_m``.
        batch_years: Years per request; defaults to ``trends.zonal_batch_years``.
        start_year: Defaults to ``time.start_year``.
        end_year: Defaults to ``time.end_year``.
        progress: Print a line per batch.

    Returns:
        ``pd.DataFrame`` with columns ``zone_id``, ``name``, ``year``, ``mean``
        and ``pixel_count``.

    Raises:
        ValueError: If ``batch_years`` is less than 1, or ``level`` is unknown.
    """
    import ee  # Deferred: see module docstring.
    import pandas as pd  # Deferred: see module docstring.

    from colombo_uhi import aoi

    # Validation before the deferred work, so it stays cheap to get wrong.
    if level not in ("gn", "ds"):
        raise ValueError(f"level must be 'gn' or 'ds', got {level!r}")
    batch = int(
        params["trends"]["zonal_batch_years"] if batch_years is None else batch_years
    )
    if batch < 1:
        raise ValueError(f"batch_years must be >= 1, got {batch}")

    comp = params["composites"]
    target = band or params["landsat_c2l2"]["lst_band_name"]
    scale = int(params["trends"]["fit_scale_m"] if scale_m is None else scale_m)
    first = int(params["time"]["start_year"] if start_year is None else start_year)
    last = int(params["time"]["end_year"] if end_year is None else end_year)

    divisions = (
        aoi.gn_divisions(params) if level == "gn" else aoi.ds_divisions(params)
    )
    # Candidates come from the same params keys uhi_metrics uses, because the
    # uploaded assets' schema is not knowable in advance (COD-AB uses ADM4_EN,
    # geoBoundaries uses shapeName). The pcode is FIRST and is what identifies a
    # division: GN names are NOT unique within Colombo District.
    assets = params["aoi"]["assets"]
    id_candidates = (
        ["adm4_pcode", *assets["gn_name_property_candidates"]]
        if level == "gn"
        else ["adm3_pcode", *assets["ds_name_property_candidates"]]
    )

    rows: list[dict[str, Any]] = []
    for low in range(first, last + 1, batch):
        high = min(low + batch - 1, last)
        years = list(range(low, high + 1))
        series = annual_series(
            source,
            params,
            region=region,
            collection=collection,
            start_year=low,
            end_year=high,
        )
        # toBands() rather than a per-year .filter(): the series is built with
        # ImageCollection.fromImages(List.map(...)), so filtering it forces Earth
        # Engine to materialise every composite graph just to evaluate the
        # predicate - the opposite of the intended saving. toBands keeps the
        # collection order, which annual_composites guarantees is ascending by
        # year, so position i is year low+i.
        stack = series.select([target]).toBands().rename([f"y{year}" for year in years])
        reduced = stack.reduceRegions(
            collection=divisions,
            reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
            scale=scale,
            tileScale=comp["tile_scale"],
        ).getInfo()["features"]

        for feature in reduced:
            properties = feature["properties"]
            identifier = next(
                (properties[name] for name in id_candidates if name in properties),
                None,
            )
            display = next(
                (
                    properties[name]
                    for name in id_candidates[1:]
                    if name in properties
                ),
                None,
            )
            for year in years:
                rows.append(
                    {
                        "zone_id": identifier,
                        "name": display,
                        "year": year,
                        "mean": properties.get(f"y{year}_mean"),
                        "pixel_count": properties.get(f"y{year}_count"),
                    }
                )
        if progress:
            print(f"  years {low}-{high} done ({len(rows)} rows)")

    frame = pd.DataFrame(rows)
    if not frame.empty:
        from colombo_uhi import composites

        composites.warn_if_counts_are_empty(
            frame["pixel_count"].fillna(0), f"{level} zonal series"
        )
    return frame


def zonal_trend_table(
    long_frame: "pd.DataFrame",
    params: dict[str, Any],
    value_column: str = "mean",
    alpha: float | None = None,
    method: str | None = None,
) -> "pd.DataFrame":
    """One trend row per division, with FDR applied ACROSS divisions.

    This is the tractable companion to the pixel-wise raster FDR: ``m`` is about
    557 rather than a hundred thousand, and the tests are far less mutually
    dependent. The two significant fractions will differ, and that spread is the
    MAUP sensitivity CLAUDE.md requires, applied to significance rather than to
    means.

    .. note::
        Divisions are keyed on ``zone_id`` (the pcode), never on the name. GN
        names are NOT unique within Colombo District, so a name join silently
        merges unrelated divisions from Dehiwala, Moratuwa or Kolonnawa.

    Args:
        long_frame: Output of :func:`zonal_annual_series`.
        params: Parsed params mapping.
        value_column: Column holding the annual value.
        alpha: Override for ``trends.fdr.alpha``.
        method: Override for ``trends.fdr.method``.

    Returns:
        ``pd.DataFrame`` with columns :data:`ZONAL_TREND_COLUMNS`. Divisions with
        too short a series get a flagged row, never a dropped one - a missing
        division leaves a hole in Phase 5's spatial weights matrix.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd  # Deferred: see module docstring.

    if long_frame.empty:
        return pd.DataFrame(columns=list(ZONAL_TREND_COLUMNS))

    rows: list[dict[str, Any]] = []
    for zone_id, group in long_frame.groupby("zone_id", sort=True):
        ordered = group.sort_values("year")
        result = mk_comparison(
            ordered[value_column].to_numpy(dtype="float64"),
            params,
            years=ordered["year"].to_numpy(dtype="int64"),
            label=str(zone_id),
            series=value_column,
        )
        original = result[result["test"] == "original"]
        if original.empty:
            continue
        record = original.iloc[0]
        rows.append(
            {
                "zone_id": zone_id,
                "name": ordered["name"].iloc[0] if "name" in ordered else None,
                "n_years": record["n_years"],
                "slope": record["slope"],
                "tau": record["tau"],
                "s": record["s"],
                "var_s": record["var_s"],
                "z": record["z"],
                "p": record["p"],
                "status": record["status"],
            }
        )

    frame = pd.DataFrame(rows)
    p_values = pd.to_numeric(frame["p"], errors="coerce").to_numpy(dtype="float64")
    reject, adjusted = benjamini_hochberg(
        p_values, alpha=alpha, params=params, method=method
    )
    frame["p_adjusted"] = adjusted
    frame["significant"] = np.where(np.isnan(p_values), False, reject)
    return frame[list(ZONAL_TREND_COLUMNS)]


# =============================================================================
# Raster post-processing (rasterio; no Earth Engine)
# =============================================================================
def read_trend_raster(
    path: str | Path,
    params: dict[str, Any],
    band_order: Sequence[str] | None = None,
) -> tuple[dict[str, "np.ndarray"], dict[str, Any]]:
    """Read an exported trend GeoTIFF into named float64 arrays plus its profile.

    Band names come from the file's own band descriptions when it has them, and
    otherwise from ``trends.export_band_order`` - which is exactly why that key
    exists, and why :func:`colombo_uhi.exports.image_to_drive` selects the image
    into that order before exporting. If the two disagree in COUNT this raises
    rather than guessing: a silent off-by-one would map the p-value band onto
    the slope and produce a plausible-looking, entirely wrong significance map.

    Args:
        path: Path to the GeoTIFF.
        params: Parsed params mapping.
        band_order: Override for ``trends.export_band_order``.

    Returns:
        ``(arrays, profile)``. Arrays are float64 with the raster's nodata
        converted to ``NaN``; the profile is rasterio's, for writing alongside.

    Raises:
        RuntimeError: If the band count does not match the expected order.
    """
    import numpy as np  # Deferred: see module docstring.
    import rasterio  # Deferred: see module docstring.

    expected = list(band_order or params["trends"]["export_band_order"])

    with rasterio.open(str(path)) as handle:
        profile = dict(handle.profile)
        descriptions = [name for name in (handle.descriptions or []) if name]

        if handle.count != len(expected):
            raise RuntimeError(
                f"{path} has {handle.count} band(s) but trends.export_band_order "
                f"lists {len(expected)}: {expected}. Band identity in a GeoTIFF "
                "is positional, so reading it under the wrong order maps the "
                "p-value band onto the slope. Re-export with the current "
                "export_band_order, or pass band_order explicitly."
            )

        if len(descriptions) == handle.count:
            names = descriptions
        else:
            names = expected
            warnings.warn(
                f"{path} carries no usable band descriptions, so band identity "
                f"falls back to POSITION via trends.export_band_order: {expected}. "
                "Confirm this matches how the file was exported.",
                stacklevel=2,
            )

        arrays: dict[str, "np.ndarray"] = {}
        for index, name in enumerate(names, start=1):
            data = handle.read(index).astype("float64")
            nodata = handle.nodatavals[index - 1]
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
            arrays[name] = data

    return arrays, profile


def apply_fdr_to_raster(
    in_path: str | Path,
    out_path: str | Path,
    params: dict[str, Any],
    alpha: float | None = None,
    method: str | None = None,
    band_order: Sequence[str] | None = None,
    min_years: int | None = None,
) -> dict[str, Any]:
    """Apply FDR to an exported p-value raster and write the masked slope beside it.

    This is the authoritative path CLAUDE.md specifies: the correction is applied
    in Python, on the exported raster, and only FDR-significant trends are
    mapped.

    Output bands, in order: ``sen_slope`` (unmasked, for reference),
    ``sen_slope_fdr`` (NaN wherever the trend is not FDR-significant),
    ``p_two_sided``, ``p_adjusted``, ``significant`` (1/0, NaN where untested),
    and ``n_years``. CRS, transform and compression carry over from the input.

    Args:
        in_path: The downloaded trend GeoTIFF.
        out_path: Where to write the FDR product.
        params: Parsed params mapping.
        alpha: Override for ``trends.fdr.alpha``.
        method: Override for ``trends.fdr.method``.
        band_order: Override for ``trends.export_band_order``.
        min_years: Override for ``trends.min_years``.

    Returns:
        The :func:`fdr_significant_fraction` summary, including areas in km2
        derived from the raster's pixel size.
    """
    import numpy as np  # Deferred: see module docstring.
    import rasterio  # Deferred: see module docstring.

    bands = params["trends"]["bands"]
    arrays, profile = read_trend_raster(in_path, params, band_order=band_order)

    slope = arrays[bands["sen_slope"]]
    p_values = arrays[bands["mk_p_two_sided"]]
    n_years = arrays[bands["n_years"]]

    tested = trend_validity_mask(p_values, slope, n_years, params, min_years=min_years)
    masked_p = np.where(tested, p_values, np.nan)

    reject, adjusted = benjamini_hochberg(
        masked_p, alpha=alpha, params=params, method=method
    )

    transform = profile.get("transform")
    pixel_area = abs(transform.a * transform.e) if transform is not None else None
    summary = fdr_significant_fraction(
        masked_p,
        params,
        alpha=alpha,
        method=method,
        slope=slope,
        pixel_area_m2=pixel_area,
    )

    layers = {
        "sen_slope": slope,
        "sen_slope_fdr": np.where(reject, slope, np.nan),
        "p_two_sided": masked_p,
        "p_adjusted": adjusted,
        # "not tested" and "tested but not significant" are different statements
        # and must be different values, so untested stays NaN rather than 0.
        "significant": np.where(tested, reject.astype("float64"), np.nan),
        "n_years": n_years,
    }

    profile.update(
        count=len(layers), dtype="float32", nodata=float("nan"), compress="deflate"
    )
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", **profile) as handle:
        for index, (name, data) in enumerate(layers.items(), start=1):
            handle.write(data.astype("float32"), index)
            handle.set_band_description(index, name)

    summary["output_path"] = str(destination)
    return summary
