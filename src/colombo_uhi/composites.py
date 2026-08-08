"""Annual and seasonal composites, each shipping a per-pixel valid-observation count.

CLAUDE.md caveat 2 is non-negotiable: *every* composite or trend product must be
interpreted alongside its per-pixel valid-observation count, because tropical
cloud cover makes only a minority of scenes usable. This module therefore has no
way to build a composite WITHOUT an ``obs_count`` band — the count is produced by
the same reducer call as the composite itself.

Products:
    * :func:`annual_composites` - one image per calendar year, carrying the
      central-tendency composite, a configurable upper percentile, and
      ``obs_count``;
    * :func:`dry_season_composites` - the same over the Jan-Mar dry window, the
      project's primary comparison window;
    * :func:`scene_inventory` - year x sensor scene counts, so data gaps are
      visible rather than inferred;
    * :func:`zonal_annual_means` - a year -> mean table over one geometry, used
      for the Landsat-vs-MODIS comparison; carries a valid-pixel count per row so
      caveat 2 applies to tables as well as rasters.

Design notes:
    * ``import ee`` is deferred into function bodies so this module (and the
      local pytest suite) imports cleanly without ``earthengine-api``.
    * Composites are built entirely server-side (``ee.List.sequence().map()``);
      the only ``getInfo`` calls are the single round trip each in
      :func:`scene_inventory` and :func:`zonal_annual_means`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, never at runtime
    import ee
    import pandas as pd

#: Reducers accepted for the central-tendency composite.
CENTRAL_REDUCERS: tuple[str, ...] = ("median", "mean")

#: Column names produced by :func:`zonal_annual_means`.
MEAN_COLUMN = "mean"
VALID_PIXELS_COLUMN = "valid_pixels"


# =============================================================================
# Pure helpers (no Earth Engine; unit-tested)
# =============================================================================
def resolve_reducer_name(name: str | None, params: dict[str, Any]) -> str:
    """Validate the central-tendency reducer name.

    Args:
        name: Requested reducer, or ``None`` for ``composites.annual_reducer``.
        params: Parsed params mapping.

    Returns:
        A name from :data:`CENTRAL_REDUCERS`.

    Raises:
        ValueError: If the reducer is not supported.
    """
    resolved = name or params["composites"]["annual_reducer"]
    if resolved not in CENTRAL_REDUCERS:
        raise ValueError(
            f"unsupported reducer '{resolved}'; expected one of {list(CENTRAL_REDUCERS)}"
        )
    return resolved


def resolve_percentile(percentile: int | None, params: dict[str, Any]) -> int:
    """Validate the upper-percentile setting.

    Args:
        percentile: Requested percentile, or ``None`` for
            ``composites.percentile``.
        params: Parsed params mapping.

    Returns:
        An integer percentile in 0-100.

    Raises:
        ValueError: If the percentile is not an integer in 0-100. Integers are
            required because the Earth Engine output band name is derived from
            the value (``p90``), and a float would not round-trip.
    """
    resolved = params["composites"]["percentile"] if percentile is None else percentile
    if isinstance(resolved, bool) or not isinstance(resolved, int):
        raise ValueError(
            f"percentile must be an int (the band name is derived from it), "
            f"got {resolved!r}"
        )
    if not 0 <= resolved <= 100:
        raise ValueError(f"percentile must be in 0..100, got {resolved}")
    return resolved


def reduced_band_names(
    band: str, reducer_name: str, percentile: int | None
) -> list[str]:
    """Band names Earth Engine produces from the combined reducer.

    ``ImageCollection.reduce`` prefixes single-input reducer outputs with the
    input band name, so a ``median``/``percentile``/``count`` combination over
    band ``LST_C`` yields ``LST_C_median``, ``LST_C_p90``, ``LST_C_count``.

    Args:
        band: Input band name.
        reducer_name: A name from :data:`CENTRAL_REDUCERS`.
        percentile: Integer percentile, or ``None`` to omit the percentile band.

    Returns:
        The raw output band names, in reducer order.
    """
    names = [f"{band}_{reducer_name}"]
    if percentile is not None:
        names.append(f"{band}_p{percentile}")
    names.append(f"{band}_count")
    return names


def composite_band_names(
    band: str, percentile: int | None, params: dict[str, Any]
) -> list[str]:
    """Band names a composite is renamed to, in order.

    The central-tendency band keeps the plain input name (``LST_C``) rather than
    a reducer-suffixed one, so downstream phases have a stable band to select
    regardless of which reducer produced it; the reducer is recorded as an image
    property instead. ``obs_count`` is always present (CLAUDE.md caveat 2).

    Args:
        band: Input band name.
        percentile: Integer percentile, or ``None`` to omit the percentile band.
        params: Parsed params mapping (``composites.obs_count_band``).

    Returns:
        ``[band, f"{band}_p{percentile}", obs_count_band]``, with the middle
        entry omitted when ``percentile`` is ``None``.
    """
    names = [band]
    if percentile is not None:
        names.append(f"{band}_p{percentile}")
    names.append(params["composites"]["obs_count_band"])
    return names


# =============================================================================
# Earth Engine products
# =============================================================================
def valid_obs_count(
    collection: "ee.ImageCollection", params: dict[str, Any], band: str | None = None
) -> "ee.Image":
    """Per-pixel count of valid (unmasked) observations in a collection.

    CLAUDE.md caveat 2. Unmasked pixels with no valid observation come back as
    0 rather than masked, so a coverage map reads as "zero scenes here" instead
    of a hole indistinguishable from a missing tile.

    Args:
        collection: Any masked ``ee.ImageCollection``.
        params: Parsed params mapping.
        band: Band to count; defaults to ``landsat_c2l2.lst_band_name``.

    Returns:
        Single-band ``ee.Image`` named ``obs_count``.
    """
    import ee  # Deferred: see module docstring.

    target = band or params["landsat_c2l2"]["lst_band_name"]
    return (
        collection.select([target])
        .reduce(ee.Reducer.count())
        .unmask(0)
        .rename(params["composites"]["obs_count_band"])
        .toFloat()
    )


def _composite_reducer(
    reducer_name: str, percentile: int | None
) -> "ee.Reducer":
    """Combined central-tendency (+ percentile) + count reducer.

    One pass over a shared accumulator, cheaper than separate reduce calls.

    Omitting the percentile is a large saving, not a cosmetic one: a percentile
    reducer must retain every observation per pixel to sort them, whereas mean
    and count are streaming accumulators. Over a 26-year series that difference
    is what decides whether the request fits in the Earth Engine memory limit.

    Args:
        reducer_name: A name from :data:`CENTRAL_REDUCERS`.
        percentile: Integer percentile, or ``None`` to omit it.

    Returns:
        The combined ``ee.Reducer``.
    """
    import ee  # Deferred: see module docstring.

    reducer = {
        "median": ee.Reducer.median,
        "mean": ee.Reducer.mean,
    }[reducer_name]()
    # sharedInputs=True is load-bearing: without it the combined reducer reports
    # multiple INPUTS, ImageCollection.reduce switches to using output names
    # directly, and the bands come back as bare median/p90/count (and error
    # unless the collection happens to have exactly that many bands).
    if percentile is not None:
        reducer = reducer.combine(
            ee.Reducer.percentile([percentile]), sharedInputs=True
        )
    return reducer.combine(ee.Reducer.count(), sharedInputs=True)


def _padding_collection(band: str) -> "ee.ImageCollection":
    """A one-image collection holding a fully-masked scene, merged into every year.

    Reducing a genuinely EMPTY collection yields a BAND-LESS image, which dies at
    the next ``select()`` ("Pattern did not match any bands"). Merging one
    fully-masked image into each year's collection guarantees the reducer always
    emits the full band set, while contributing nothing to any statistic: masked
    pixels are skipped by median, percentile and count alike, so ``obs_count``
    stays honest at 0 for a year with no real scenes.

    This replaces an earlier ``ee.Algorithms.If`` placeholder. ``If`` builds and
    evaluates BOTH branches regardless of the condition, which made every
    composite carry the placeholder graph as well as the real one and blew the
    Earth Engine user memory limit in Colab run 1.

    Args:
        band: Band name the collection being padded carries.

    Returns:
        Single-image ``ee.ImageCollection``, fully masked.
    """
    import ee  # Deferred: see module docstring.

    masked = (
        ee.Image.constant(0)
        .rename(band)
        .toFloat()
        .updateMask(ee.Image.constant(0))
    )
    return ee.ImageCollection([masked])


def annual_composites(
    collection: "ee.ImageCollection",
    params: dict[str, Any],
    band: str | None = None,
    reducer: str | None = None,
    months: Sequence[int] | None = None,
    percentile: int | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    with_percentile: bool = True,
) -> "ee.ImageCollection":
    """One composite per calendar year, each with a valid-observation count.

    Bands, in order: the central-tendency composite under the plain band name
    (``LST_C``), the upper percentile (``LST_C_p90``), and ``obs_count``.
    Properties: ``year``, ``n_scenes``, ``reducer``, ``percentile`` and
    ``system:time_start`` (1 January of that year).

    Years with no usable scene still appear, with masked data bands and
    ``obs_count == 0`` — the series stays complete for trend fitting, and the
    gap is visible rather than silently dropped.

    .. warning::
        Year slicing is by CALENDAR year. For a window that wraps the new year
        (e.g. the NE monsoon, Dec-Feb) this groups December with the *January
        and February of the same calendar year*, which is not one continuous
        monsoon. The Jan-Mar dry window used by
        :func:`dry_season_composites` is unaffected.

    .. note::
        With only a handful of cloud-free scenes per pixel per year, a high
        percentile is close to the maximum of a very small sample. Treat
        ``LST_C_p90`` as "hot-tail behaviour", not a stable statistic, and read
        it against ``obs_count``. Phase 4 trend fitting should use the
        central-tendency band under an ``obs_count`` floor.

    Args:
        collection: A harmonised collection from
            :func:`colombo_uhi.landsat.harmonised_collection`.
        params: Parsed params mapping.
        band: Band to composite; defaults to ``landsat_c2l2.lst_band_name``.
        reducer: Central-tendency reducer; defaults to
            ``composites.annual_reducer``.
        months: Optional calendar-month restriction applied before compositing.
        percentile: Upper percentile; defaults to ``composites.percentile``.
        start_year: First year; defaults to ``time.start_year``.
        end_year: Last year, inclusive; defaults to ``time.end_year``.
        with_percentile: Set ``False`` to omit the percentile band. A percentile
            reducer must retain every observation per pixel in order to sort
            them; mean and count are streaming. Turning it off is the cheapest
            way to fit a long series inside the Earth Engine memory limit, and
            it costs nothing when only the central-tendency band is wanted (as
            in the Landsat-vs-MODIS comparison).

    Returns:
        ``ee.ImageCollection`` with one image per year, ascending.
    """
    import ee  # Deferred: see module docstring.

    from colombo_uhi import landsat

    comp = params["composites"]
    target = band or params["landsat_c2l2"]["lst_band_name"]
    reducer_name = resolve_reducer_name(reducer, params)
    pct = resolve_percentile(percentile, params) if with_percentile else None

    first_year = int(params["time"]["start_year"] if start_year is None else start_year)
    last_year = int(params["time"]["end_year"] if end_year is None else end_year)
    if last_year < first_year:
        raise ValueError(
            f"end_year ({last_year}) must be >= start_year ({first_year})"
        )

    # select() BEFORE reduce(): a single-input reducer is applied to every band,
    # so reducing the full 8-band scene would produce 24 output bands.
    source = collection.select([target])
    if months:
        source = source.filter(landsat.month_filter(months))

    combined = _composite_reducer(reducer_name, pct)
    raw_names = reduced_band_names(target, reducer_name, pct)
    out_names = composite_band_names(target, pct, params)
    obs_band = comp["obs_count_band"]
    padding = _padding_collection(target)

    def per_year(year: Any) -> "ee.Image":
        year_number = ee.Number(year).toInt()
        start = ee.Date.fromYMD(year_number, 1, 1)
        end = ee.Date.fromYMD(year_number.add(1), 1, 1)
        # filterDate hits the date index; calendarRange would evaluate metadata
        # for every element of the collection.
        yearly = source.filterDate(start, end)

        renamed = yearly.merge(padding).reduce(combined).select(raw_names, out_names)
        # An all-masked pixel must read as 0 observations, not as a hole.
        # Rebuilt with an explicit select() rather than addBands(overwrite=True),
        # which does not guarantee band position.
        data_bands = [name for name in out_names if name != obs_band]
        composite = (
            ee.Image.cat(
                [renamed.select(data_bands), renamed.select(obs_band).unmask(0)]
            )
            .select(out_names)
            .toFloat()
        )

        # n_scenes counts REAL scenes, so the padding image must not be included.
        # `percentile` is omitted rather than set to None when the band is not
        # produced; ee.Image.set rejects a null value.
        properties: dict[str, Any] = {
            comp["year_property"]: year_number,
            comp["n_scenes_property"]: yearly.size(),
            "reducer": reducer_name,
            "system:time_start": start.millis(),
        }
        if pct is not None:
            properties["percentile"] = pct
        return composite.set(properties)

    years = ee.List.sequence(first_year, last_year)
    return ee.ImageCollection.fromImages(years.map(per_year))


def dry_season_composites(
    collection: "ee.ImageCollection",
    params: dict[str, Any],
    **kwargs: Any,
) -> "ee.ImageCollection":
    """Annual composites restricted to the Jan-Mar dry window.

    This is the project's primary comparison window (CLAUDE.md): the driest,
    clearest skies of the year, so the composites rest on the most observations
    and the between-year comparison is least confounded by cloud.

    Args:
        collection: A harmonised collection.
        params: Parsed params mapping.
        **kwargs: Forwarded to :func:`annual_composites` (``months`` is set
            here and must not be passed).

    Returns:
        ``ee.ImageCollection`` with one dry-season composite per year.

    Raises:
        TypeError: If ``months`` is passed, which would defeat the purpose.
    """
    if "months" in kwargs:
        raise TypeError(
            "dry_season_composites() sets months from "
            "time.seasons.dry_window.months; use annual_composites() directly "
            "for a different window"
        )
    months = params["time"]["seasons"]["dry_window"]["months"]
    return annual_composites(collection, params, months=months, **kwargs)


# =============================================================================
# Client-side summaries (exactly one round trip each)
# =============================================================================
def scene_inventory(
    collection: "ee.ImageCollection",
    params: dict[str, Any],
    start_year: int | None = None,
    end_year: int | None = None,
) -> "pd.DataFrame":
    """Year x sensor table of scenes surviving filtering — the data-gap view.

    Expect visible, explainable gaps: no Landsat 5 after 2012-05, no Landsat 8
    before 2013-03, and Landsat 7 striped from 2003-06.

    Both property arrays are pulled in a SINGLE ``getInfo`` (wrapped in one
    ``ee.List``). Two separate calls would be two independent evaluations whose
    element order is not guaranteed to align, and which would desync by length
    if a property were missing on some scenes.

    Args:
        collection: A harmonised collection (before compositing).
        params: Parsed params mapping.
        start_year: First row; defaults to ``time.start_year``.
        end_year: Last row, inclusive; defaults to ``time.end_year``.

    Returns:
        ``pandas.DataFrame`` indexed by year, one column per sensor plus
        ``total``. Years with no scenes are present as zero rows.
    """
    import ee  # Deferred: see module docstring.

    comp = params["composites"]
    years_list, sensors_list = ee.List(
        [
            collection.aggregate_array(comp["year_property"]),
            collection.aggregate_array(comp["sensor_property"]),
        ]
    ).getInfo()
    return build_inventory_frame(
        years_list, sensors_list, params, start_year=start_year, end_year=end_year
    )


def build_inventory_frame(
    years: Sequence[int],
    sensors: Sequence[str],
    params: dict[str, Any],
    start_year: int | None = None,
    end_year: int | None = None,
) -> "pd.DataFrame":
    """Shape parallel year/sensor arrays into the scene-inventory table.

    Split out from :func:`scene_inventory` so the table logic — reindexing to
    the full study period so gap years show as explicit zeros — is unit-testable
    without Earth Engine.

    Args:
        years: Acquisition year of each scene.
        sensors: Sensor label of each scene, parallel to ``years``.
        params: Parsed params mapping.
        start_year: First row; defaults to ``time.start_year``.
        end_year: Last row, inclusive; defaults to ``time.end_year``.

    Returns:
        ``pandas.DataFrame`` indexed by year with one column per sensor plus
        ``total``.

    Raises:
        ValueError: If ``years`` and ``sensors`` have different lengths, which
            would mean the two property arrays desynced.
    """
    import pandas as pd

    if len(years) != len(sensors):
        raise ValueError(
            f"years and sensors must be parallel arrays; got {len(years)} years "
            f"and {len(sensors)} sensors. A length mismatch means some scenes "
            "are missing one of the properties."
        )

    comp = params["composites"]
    year_key, sensor_key = comp["year_property"], comp["sensor_property"]
    first_year = int(params["time"]["start_year"] if start_year is None else start_year)
    last_year = int(params["time"]["end_year"] if end_year is None else end_year)
    index = pd.RangeIndex(first_year, last_year + 1, name=year_key)

    if not len(years):
        return pd.DataFrame(0, index=index, columns=["total"])

    scenes = pd.DataFrame({year_key: list(years), sensor_key: list(sensors)})
    table = pd.crosstab(scenes[year_key], scenes[sensor_key])
    table = table.reindex(index, fill_value=0)
    table.columns.name = None
    table["total"] = table.sum(axis=1)
    return table


def _zonal_feature(
    image: "ee.Image",
    geometry: "ee.Geometry",
    band: str,
    scale: int,
    params: dict[str, Any],
) -> "ee.Feature":
    """Reduce one image over a geometry into a property-only feature.

    Args:
        image: Image to reduce.
        geometry: Region to reduce over.
        band: Band to average.
        scale: Reduction scale in metres.
        params: Parsed params mapping.

    Returns:
        ``ee.Feature`` with no geometry, carrying the year plus the reducer
        outputs.
    """
    import ee  # Deferred: see module docstring.

    comp = params["composites"]
    reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)
    stats = image.select([band]).reduceRegion(
        reducer=reducer,
        geometry=geometry,
        scale=scale,
        maxPixels=comp["reduce_max_pixels"],
        tileScale=comp["tile_scale"],
    )
    # ee.Feature(None, ...) keeps the payload to properties only.
    return ee.Feature(
        None, stats.set(comp["year_property"], image.get(comp["year_property"]))
    )


def zonal_annual_means(
    collection: "ee.ImageCollection",
    geometry: "ee.Geometry",
    params: dict[str, Any],
    band: str | None = None,
    scale_m: int | None = None,
) -> "pd.DataFrame":
    """Year -> spatial mean table for an ALREADY-BUILT collection, in one request.

    Each row carries the number of valid pixels the mean rests on, so CLAUDE.md
    caveat 2 applies to the table and not only to the rasters.

    Uses ``FeatureCollection.getInfo()`` rather than ``aggregate_array``: when a
    year is fully masked, ``reduceRegion`` still returns the key with a null
    value, so the row survives as ``NaN`` instead of silently shortening the
    array and misaligning every later year.

    .. warning::
        This evaluates the WHOLE collection in one request. For a multi-decade
        annual series that exceeds the Earth Engine user memory limit — use
        :func:`zonal_annual_means_by_year`, which builds the composites a few
        years at a time. This function is for collections that are already small.

    Args:
        collection: Composites (or any collection) carrying the year property.
        geometry: Region to average over, e.g. ``aoi.cmc_boundary(params)``.
        params: Parsed params mapping.
        band: Band to average; defaults to ``landsat_c2l2.lst_band_name``.
        scale_m: Reduction scale; defaults to ``crs.analysis_scale_m``. Pass
            the sensor's native scale for MODIS (``modis_lst.reduction_scale_m``).

    Returns:
        ``pandas.DataFrame`` with columns ``year``, ``mean``, ``valid_pixels``,
        sorted by year.

    Raises:
        RuntimeError: If Earth Engine returns unexpected property names, rather
            than quietly producing an all-NaN table.
    """
    import ee  # Deferred: see module docstring.

    target = band or params["landsat_c2l2"]["lst_band_name"]
    scale = int(scale_m or params["crs"]["analysis_scale_m"])

    features = ee.FeatureCollection(
        collection.map(
            lambda image: _zonal_feature(image, geometry, target, scale, params)
        )
    ).getInfo()["features"]
    return build_zonal_frame(
        [feature["properties"] for feature in features], target, params
    )


def zonal_annual_means_by_year(
    source: "ee.ImageCollection",
    geometry: "ee.Geometry",
    params: dict[str, Any],
    band: str | None = None,
    scale_m: int | None = None,
    reducer: str | None = None,
    months: Sequence[int] | None = None,
    with_percentile: bool = False,
    mask: "ee.Image | None" = None,
    batch_years: int | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    progress: bool = False,
) -> "pd.DataFrame":
    """Year -> spatial mean table, building the composites a few years at a time.

    This is the memory-safe way to pull a multi-decade series out of Earth
    Engine, and the distinction from :func:`zonal_annual_means` is the whole
    point of it:

    **Filtering a computed collection does not reduce the work.** An annual
    series is built with ``ee.ImageCollection.fromImages(ee.List.map(...))``, so
    ``.filter(year == 2000)`` on it forces Earth Engine to materialise all 26
    composite graphs just to evaluate the predicate on each. Batching by filter
    is therefore worthless — it was tried, and still exceeded the memory limit at
    one year per request (Colab run 4, 2026-08-08).

    So this takes the SCENE collection and calls :func:`annual_composites`
    separately per batch with ``start_year``/``end_year`` set, which constructs
    only those years' graphs. The numbers are identical to an unbatched fetch.

    Args:
        source: The scene-level collection — a harmonised Landsat collection or
            MODIS granules — NOT pre-built annual composites.
        geometry: Region to average over, e.g. ``aoi.cmc_boundary(params)``.
        params: Parsed params mapping.
        band: Band to average; defaults to ``landsat_c2l2.lst_band_name``.
        scale_m: Reduction scale; defaults to ``crs.analysis_scale_m``.
        reducer: Central-tendency reducer for the composites. Use the SAME one
            for every series being compared, or part of the difference between
            them is a reducer artefact rather than a sensor difference.
        months: Optional calendar-month restriction.
        with_percentile: Defaults to ``False`` here — a percentile reducer must
            retain every observation per pixel in order to sort them, and a
            zonal mean never uses the result.
        mask: Optional 0/1 image applied to each composite, e.g. a land mask.
            Prefer :func:`colombo_uhi.aoi.static_water_mask` over
            :func:`colombo_uhi.aoi.water_mask`: the latter composites Landsat
            internally, and that composite is instantiated once per image masked.
        batch_years: Years built per request; defaults to
            ``composites.zonal_batch_years``. 1 is valid.
        start_year: First year; defaults to ``time.start_year``.
        end_year: Last year, inclusive; defaults to ``time.end_year``.
        progress: Print a line per batch, so a slow run visibly advances.

    Returns:
        ``pandas.DataFrame`` with columns ``year``, ``mean``, ``valid_pixels``,
        sorted by year.

    Raises:
        ValueError: If ``batch_years`` is less than 1.
    """
    comp = params["composites"]
    target = band or params["landsat_c2l2"]["lst_band_name"]
    scale = int(scale_m or params["crs"]["analysis_scale_m"])

    # Validation before the deferred import, so it stays unit-testable.
    batch = int(comp["zonal_batch_years"] if batch_years is None else batch_years)
    if batch < 1:
        raise ValueError(f"batch_years must be >= 1, got {batch}")
    first_year = int(params["time"]["start_year"] if start_year is None else start_year)
    last_year = int(params["time"]["end_year"] if end_year is None else end_year)

    import ee  # Deferred: see module docstring.

    rows: list[dict[str, Any]] = []
    for low in range(first_year, last_year + 1, batch):
        high = min(low + batch - 1, last_year)
        annual = annual_composites(
            source,
            params,
            band=target,
            reducer=reducer,
            months=months,
            with_percentile=with_percentile,
            start_year=low,
            end_year=high,
        )
        if mask is not None:
            annual = annual.map(lambda image: image.updateMask(mask))
        features = ee.FeatureCollection(
            annual.map(
                lambda image: _zonal_feature(image, geometry, target, scale, params)
            )
        ).getInfo()["features"]
        rows.extend(feature["properties"] for feature in features)
        if progress:
            print(f"  years {low}-{high} done ({len(rows)} rows)")

    return build_zonal_frame(rows, target, params)


def build_zonal_frame(
    rows: Sequence[dict[str, Any]], band: str, params: dict[str, Any]
) -> "pd.DataFrame":
    """Shape ``reduceRegion`` property dicts into the year/mean/count table.

    Split out from :func:`zonal_annual_means` so the column resolution and its
    failure mode are unit-testable without Earth Engine.

    Args:
        rows: One property mapping per image, as returned by
            ``FeatureCollection.getInfo()``.
        band: The band that was reduced, used to derive the expected keys.
        params: Parsed params mapping.

    Returns:
        ``pandas.DataFrame`` with columns ``year``, ``mean``, ``valid_pixels``,
        sorted by year. A fully-masked year keeps its row, with ``NaN``.

    Raises:
        RuntimeError: If the expected ``<band>_mean``/``<band>_count`` keys are
            absent, rather than quietly producing an all-NaN table.
    """
    import pandas as pd

    year_prop = params["composites"]["year_property"]
    mean_key, count_key = f"{band}_mean", f"{band}_count"

    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame(columns=[year_prop, MEAN_COLUMN, VALID_PIXELS_COLUMN])

    missing = [key for key in (mean_key, count_key) if key not in frame.columns]
    if missing:
        raise RuntimeError(
            f"reduceRegion did not return {missing}; it returned "
            f"{sorted(frame.columns)}. Check that band '{band}' exists on "
            "every image in the collection."
        )

    frame = frame.rename(
        columns={mean_key: MEAN_COLUMN, count_key: VALID_PIXELS_COLUMN}
    )
    result = (
        frame[[year_prop, MEAN_COLUMN, VALID_PIXELS_COLUMN]]
        .sort_values(year_prop)
        .reset_index(drop=True)
    )
    _warn_if_series_is_empty(result, band)
    return result


def _warn_if_series_is_empty(frame: "pd.DataFrame", band: str) -> None:
    """Complain loudly about a series with no, or almost no, valid pixels.

    CLAUDE.md caveat 2 is not satisfied by merely *emitting* a count that nobody
    reads. In Colab run 5 both MODIS night series returned 26 rows of ``None``
    with ``valid_pixels == 0`` and the pipeline said nothing — the table looked
    like a plausible product until someone checked the column. An all-empty
    result is a defect, not a datum, and must announce itself.

    The frame is still returned by the caller: downstream code needs the shape,
    and suppressing it would trade one silent failure for another.

    Args:
        frame: The assembled year/mean/valid_pixels table.
        band: Band name, for a message that identifies the culprit.
    """
    if frame.empty:
        return

    counts = frame[VALID_PIXELS_COLUMN].fillna(0)
    empty_years = int((counts <= 0).sum())
    total_years = len(frame)
    if empty_years == 0:
        return

    if empty_years == total_years:
        warnings.warn(
            f"zonal series for band '{band}' is COMPLETELY EMPTY: all "
            f"{total_years} years returned 0 valid pixels. Do not plot or "
            "interpret it. Usual causes, in order: a QC filter that rejects "
            "everything (run modis.qc_class_histogram to see which class), a "
            "mask that excludes the whole geometry, or a reduction scale finer "
            "than the data.",
            stacklevel=4,
        )
    elif empty_years > total_years / 2:
        warnings.warn(
            f"zonal series for band '{band}' is mostly empty: {empty_years} of "
            f"{total_years} years returned 0 valid pixels. The remaining years "
            "may rest on very few pixels — check the valid_pixels column before "
            "reading any trend into this series.",
            stacklevel=4,
        )
