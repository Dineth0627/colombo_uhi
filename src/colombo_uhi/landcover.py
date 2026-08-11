"""Land-cover and Local Climate Zone class images, and stratified statistics.

Phase 4 needs to answer "where is the warming, by land cover?", and nothing in
the project read a land-cover product before this module: ``datasets.
dynamic_world``, ``datasets.worldcover_*`` and ``datasets.lcz`` were configured
and unused. This is also the layer Phases 5-7 need - hot-spot context, random
forest predictors, and the greening MCDA all stratify by land cover - so it
lives in its own module rather than inside :mod:`colombo_uhi.trends`.

Products:
    * :func:`worldcover` / :func:`lcz_class_image` / :func:`dynamic_world_mode` -
      single-band integer class images;
    * :func:`stratified_stats` - any image reduced to per-class statistics in ONE
      round trip, via a grouped reducer.

.. warning::
    Every class layer here is a fixed snapshot. ESA WorldCover is 2020/2021, the
    LCZ map is derived from 2018-2019 imagery, and Dynamic World does not begin
    until 2015-06-27. Stratifying a 2000-2025 trend by any of them yields
    "where the warming is, by TODAY'S land cover" - it is NOT an attribution of
    warming to land-cover change. A pixel that was paddy in 2002 and is built
    now sits in the built class for its whole history. Attribution to land-cover
    change is Phase 6.

Design notes:
    * ``import ee`` (and pandas) is deferred into function bodies so this module,
      and the local pytest suite, import cleanly without ``earthengine-api``.
    * Every constant comes from ``config/params.yaml`` (``landcover`` and
      ``datasets`` sections).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, never at runtime
    import ee
    import pandas as pd

#: Class schemes this module knows how to build an image for.
SCHEMES: tuple[str, ...] = ("worldcover", "lcz", "dynamic_world")

#: Reducers :func:`stratified_stats` knows how to build. ``count`` is always
#: appended, so a class mean never travels without the pixels it rests on.
STRATIFIED_REDUCERS: tuple[str, ...] = ("mean", "median", "stdDev", "min", "max")

#: Column order of the table :func:`build_stratified_frame` produces.
STRATIFIED_COLUMNS: tuple[str, ...] = (
    "scheme",
    "class_code",
    "class_label",
    "pixel_count",
    "below_pixel_floor",
)

#: The first day Dynamic World has any imagery. Requesting a year before this
#: yields an empty collection and a band-less mosaic.
DYNAMIC_WORLD_START = "2015-06-27"


# =============================================================================
# Pure helpers (no Earth Engine; unit-tested)
# =============================================================================
def resolve_scheme(scheme: str, params: dict[str, Any]) -> str:
    """Validate a class scheme name against what is configured and implemented.

    Args:
        scheme: Scheme key, e.g. ``"worldcover"``.
        params: Parsed params mapping.

    Returns:
        The validated scheme name.

    Raises:
        KeyError: If the scheme is not configured under ``landcover``.
        ValueError: If it is configured but this module cannot build an image
            for it.
    """
    if scheme not in params["landcover"]:
        raise KeyError(
            f"unknown land-cover scheme '{scheme}'; params.landcover defines "
            f"{sorted(params['landcover'])}"
        )
    if scheme not in SCHEMES:
        raise ValueError(
            f"land-cover scheme '{scheme}' is configured but not implemented; "
            f"this module builds images for {list(SCHEMES)}"
        )
    return scheme


def class_labels(params: dict[str, Any], scheme: str) -> dict[int, str]:
    """Class code to human label for one scheme.

    .. note::
        WorldCover codes are NOT contiguous (10, 20, ... 95, 100), so never index
        a palette or a list by class code.

    Args:
        params: Parsed params mapping.
        scheme: Scheme key.

    Returns:
        Mapping of integer class code to label.

    Raises:
        KeyError: If the scheme is not configured.
    """
    if scheme not in params["landcover"]:
        raise KeyError(
            f"unknown land-cover scheme '{scheme}'; params.landcover defines "
            f"{sorted(params['landcover'])}"
        )
    return {
        int(code): str(label)
        for code, label in params["landcover"][scheme]["classes"].items()
    }


def resolve_stratified_reducers(names: Sequence[str] | None) -> list[str]:
    """Validate the reducer list for :func:`stratified_stats`.

    Args:
        names: Requested reducers, or ``None`` for ``("mean", "stdDev")``.

    Returns:
        The validated names, in order.

    Raises:
        ValueError: If the list is empty or names an unsupported reducer.
    """
    resolved = list(names) if names else ["mean", "stdDev"]
    if not resolved:
        raise ValueError("at least one reducer is required")
    unsupported = [name for name in resolved if name not in STRATIFIED_REDUCERS]
    if unsupported:
        raise ValueError(
            f"unsupported reducer(s) {unsupported}; expected any of "
            f"{list(STRATIFIED_REDUCERS)}"
        )
    return resolved


def build_stratified_frame(
    groups: Sequence[dict[str, Any]],
    params: dict[str, Any],
    scheme: str,
    reducers: Sequence[str],
) -> "pd.DataFrame":
    """Shape grouped-reducer output into a per-class table.

    Args:
        groups: Group dictionaries from a grouped ``reduceRegion``.
        params: Parsed params mapping.
        scheme: Scheme key, for labels.
        reducers: Reducer names present in each group, in order.

    Returns:
        ``pd.DataFrame`` with :data:`STRATIFIED_COLUMNS` followed by one column
        per reducer; empty but correctly shaped when no groups are given.
    """
    import pandas as pd  # Deferred: see module docstring.

    labels = class_labels(params, scheme)
    floor = int(params["trends"]["stratify"]["min_pixels_per_class"])
    columns = list(STRATIFIED_COLUMNS) + list(reducers)

    rows: list[dict[str, Any]] = []
    for group in groups:
        code = int(group.get("class", group.get("group", -1)))
        pixels = int(group.get("count") or 0)
        row: dict[str, Any] = {
            "scheme": scheme,
            "class_code": code,
            "class_label": labels.get(code, f"unknown ({code})"),
            "pixel_count": pixels,
            "below_pixel_floor": pixels < floor,
        }
        for name in reducers:
            row[name] = group.get(name)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns].sort_values("class_code").reset_index(drop=True)


# =============================================================================
# Earth Engine class images
# =============================================================================
def worldcover(
    params: dict[str, Any],
    asset_key: str | None = None,
) -> "ee.Image":
    """ESA WorldCover class image, mosaicked to a single band.

    Args:
        params: Parsed params mapping.
        asset_key: Key into ``datasets``; defaults to
            ``trends.stratify.worldcover_asset``.

    Returns:
        Single-band integer ``ee.Image`` named ``landcover.worldcover.band``.

    Raises:
        KeyError: If the dataset key is not configured.
    """
    import ee  # Deferred: see module docstring.

    key = asset_key or params["trends"]["stratify"]["worldcover_asset"]
    if key not in params["datasets"]:
        raise KeyError(
            f"unknown WorldCover dataset key '{key}'; params.datasets defines "
            f"{sorted(params['datasets'])}"
        )
    band = params["landcover"]["worldcover"]["band"]
    return ee.ImageCollection(params["datasets"][key]["id"]).mosaic().select([band])


def lcz_class_image(params: dict[str, Any]) -> "ee.Image":
    """Local Climate Zone class image (the recommended, filtered band).

    Delegates to :mod:`colombo_uhi.aoi`, which already loads this map for the
    LCZ-based rural reference, so both routes see the same pixels.

    Args:
        params: Parsed params mapping.

    Returns:
        Single-band integer ``ee.Image`` of LCZ classes 1-17.
    """
    from colombo_uhi import aoi

    return aoi._lcz_image(params)


def dynamic_world_mode(
    params: dict[str, Any],
    year: int,
    region: "ee.Geometry | None" = None,
) -> "ee.Image":
    """Modal Dynamic World class over one calendar year.

    .. note::
        Exposed for Phase 6, not used by Phase 4: Dynamic World begins on
        2015-06-27 and so cannot classify the early part of the 2000-2025
        series.

    Args:
        params: Parsed params mapping.
        year: Calendar year to reduce over.
        region: Optional region to bound the work.

    Returns:
        Single-band integer ``ee.Image`` of the modal class.

    Raises:
        ValueError: If the year ends before Dynamic World begins, which would
            otherwise yield a band-less mosaic and a confusing downstream error.
    """
    import ee  # Deferred: see module docstring.

    band = params["landcover"]["dynamic_world"]["band"]
    start_year = int(DYNAMIC_WORLD_START[:4])
    if int(year) < start_year:
        raise ValueError(
            f"Dynamic World starts {DYNAMIC_WORLD_START}, so it cannot classify "
            f"{year}. Use worldcover() or lcz_class_image() for a fixed "
            "reference classification of the early series."
        )

    collection = (
        ee.ImageCollection(params["datasets"]["dynamic_world"]["id"])
        .filterDate(f"{int(year)}-01-01", f"{int(year) + 1}-01-01")
        .select([band])
    )
    if region is not None:
        collection = collection.filterBounds(region)
    return collection.mode().rename([band]).toInt()


def class_image(
    scheme: str,
    params: dict[str, Any],
    year: int | None = None,
    region: "ee.Geometry | None" = None,
) -> "ee.Image":
    """Dispatch to the class image for a scheme.

    Args:
        scheme: One of :data:`SCHEMES`.
        params: Parsed params mapping.
        year: Required for ``"dynamic_world"``; ignored otherwise.
        region: Optional region to bound the work.

    Returns:
        Single-band integer class ``ee.Image``.

    Raises:
        KeyError: If the scheme is not configured.
        ValueError: If the scheme is not implemented, or a year is needed and
            not given.
    """
    resolved = resolve_scheme(scheme, params)
    if resolved == "worldcover":
        return worldcover(params)
    if resolved == "lcz":
        return lcz_class_image(params)
    if year is None:
        raise ValueError(
            "dynamic_world needs an explicit `year`; it is a per-year modal "
            "classification, not a fixed reference layer"
        )
    return dynamic_world_mode(params, year, region=region)


# =============================================================================
# Stratified statistics
# =============================================================================
def _grouped_reduction(
    image: "ee.Image",
    scheme: str,
    params: dict[str, Any],
    region: "ee.Geometry",
    band: str | None = None,
    scale_m: int | None = None,
    reducers: Sequence[str] | None = None,
    year: int | None = None,
) -> tuple["ee.Dictionary", list[str], str]:
    """Build the grouped reduction, unevaluated. Shared by both public callers.

    The class band is stacked LAST and ``groupField`` is its index, made explicit
    here rather than left for the caller to infer - a mis-grouped reduction
    returns a plausible-looking table rather than an error.
    """
    import ee  # Deferred: see module docstring.

    # Validation before the deferred work, so a bad argument is cheap.
    resolved = resolve_scheme(scheme, params)
    names = resolve_stratified_reducers(reducers)

    comp = params["composites"]
    scale = int(params["crs"]["analysis_scale_m"] if scale_m is None else scale_m)

    target = band or image.bandNames().get(0)
    classes = class_image(resolved, params, year=year, region=region)
    stacked = image.select([target]).addBands(classes.rename("class").toInt())

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
    combined = combined.combine(ee.Reducer.count(), sharedInputs=True)

    reduction = stacked.reduceRegion(
        reducer=combined.group(groupField=1, groupName="class"),
        geometry=region,
        scale=scale,
        maxPixels=comp["reduce_max_pixels"],
        tileScale=comp["tile_scale"],
    )
    return reduction, names, resolved


def stratified_stats_collection(
    image: "ee.Image",
    scheme: str,
    params: dict[str, Any],
    region: "ee.Geometry",
    band: str | None = None,
    scale_m: int | None = None,
    reducers: Sequence[str] | None = None,
    year: int | None = None,
) -> "ee.FeatureCollection":
    """The grouped per-class reduction as an EXPORTABLE FeatureCollection.

    Identical statistics to :func:`stratified_stats`, but nothing is evaluated:
    the ``groups`` list is wrapped into geometry-less features so the whole
    reduction can be handed to
    :func:`colombo_uhi.exports.table_to_drive` and run inside a batch task.

    .. note::
        This exists because a grouped ``reduceRegion`` over a district-sized
        trend surface cannot be evaluated interactively - Colab runs 10-13
        established that **no** interactive question about the trend graph is
        affordable, down to and including ``bandNames()``. A batch ``Export``
        task has no such ceiling, so the same reduction that fails in-session
        succeeds as a submitted task.

    Args:
        image: Image to summarise.
        scheme: Class scheme key.
        params: Parsed params mapping.
        region: Region to summarise over.
        band: Band to summarise; defaults to the image's first band.
        scale_m: Reduction scale; defaults to ``crs.analysis_scale_m``.
        reducers: Statistics to compute; defaults to ``("mean", "stdDev")``.
        year: Passed through to :func:`class_image` for ``"dynamic_world"``.

    Returns:
        ``ee.FeatureCollection``, one geometry-less feature per class.
    """
    import ee  # Deferred: see module docstring.

    reduction, _, _ = _grouped_reduction(
        image, scheme, params, region,
        band=band, scale_m=scale_m, reducers=reducers, year=year,
    )
    groups = ee.List(reduction.get("groups"))
    return ee.FeatureCollection(
        groups.map(lambda group: ee.Feature(None, ee.Dictionary(group)))
    )


def stratified_stats(
    image: "ee.Image",
    scheme: str,
    params: dict[str, Any],
    region: "ee.Geometry",
    band: str | None = None,
    scale_m: int | None = None,
    reducers: Sequence[str] | None = None,
    year: int | None = None,
) -> "pd.DataFrame":
    """Reduce one image band to per-class statistics in a single round trip.

    .. warning::
        This EVALUATES the reduction. Over a district-sized trend surface that
        exceeds the Earth Engine interactive memory limit; use
        :func:`stratified_stats_collection` with
        :func:`colombo_uhi.exports.table_to_drive` instead, and read the
        resulting CSV. This form remains correct and convenient for small
        regions and cheap source images.

    Args:
        image: Image to summarise.
        scheme: Class scheme key.
        params: Parsed params mapping.
        region: Region to summarise over.
        band: Band to summarise; defaults to the image's first band.
        scale_m: Reduction scale; defaults to ``crs.analysis_scale_m``.
        reducers: Statistics to compute; defaults to ``("mean", "stdDev")``.
            ``count`` is always appended.
        year: Passed through to :func:`class_image` for ``"dynamic_world"``.

    Returns:
        ``pd.DataFrame`` with :data:`STRATIFIED_COLUMNS` plus one column per
        reducer.

    Raises:
        RuntimeError: If the grouped reduction returns no ``groups`` key, naming
            what it did return.
    """
    reduction, names, resolved = _grouped_reduction(
        image, scheme, params, region,
        band=band, scale_m=scale_m, reducers=reducers, year=year,
    )
    result = reduction.getInfo()

    groups = result.get("groups")
    if groups is None:
        raise RuntimeError(
            "the grouped reduceRegion returned no 'groups' key; it returned "
            f"{sorted(result)}. Check that the class band is the LAST band of "
            "the stack and that groupField matches its index."
        )
    return build_stratified_frame(groups, params, resolved, names)
