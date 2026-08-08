"""Landsat Collection 2 Level-2 masking, scaling and the harmonised LST collection.

Phase 1 shipped the pieces the study-area layer needed (QA_PIXEL clear mask,
surface-reflectance scaling). Phase 2 adds the main product of this module:
:func:`harmonised_collection`, one ``ee.ImageCollection`` merging L5/L7/L8/L9 in
which every scene carries

* ``LST_C``   - land surface temperature in degrees CELSIUS,
* ``blue`` ... ``swir2`` - surface reflectance under sensor-agnostic names, so
  :mod:`colombo_uhi.indices` never has to know which satellite it is looking at,
* ``ST_QA_K`` - the product's own per-pixel LST uncertainty, in Kelvin,

plus ``system:time_start`` and the ``sensor`` / ``sensor_key`` / ``year`` /
``month`` / ``season`` properties that :mod:`colombo_uhi.composites` groups on.
Band order and dtype are identical across all four sensors (see
:func:`output_band_names`) so the merge is safe and downstream reducers see one
consistent schema.

CAVEAT (CLAUDE.md #1): ``LST_C`` is LAND SURFACE TEMPERATURE, not air
temperature, and Landsat sees it at a single ~10:30 local overpass (caveat #4).

Design notes:
    * ``import ee`` is deferred into function bodies so this module (and the
      local pytest suite) imports cleanly without ``earthengine-api``.
    * Every constant (band names, scale factors, QA bit positions, sensor list)
      comes from ``config/params.yaml`` — no magic numbers.
    * :mod:`colombo_uhi.aoi` imports this module, so this module must NEVER
      import ``aoi`` at module scope. :func:`harmonised_collection` needs it for
      one default and does a deferred import inside the function body.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, never at runtime
    import ee

#: Months in a calendar year, used to build the month -> season lookup.
_MONTHS_IN_YEAR = 12


# =============================================================================
# Pure helpers (no Earth Engine; unit-tested)
# =============================================================================
def bits_to_mask(bits: Sequence[int]) -> int:
    """Build an integer bitmask with the given bit positions set.

    Pure helper for QA-band masking, e.g. the standard Landsat C2 L2 cloud
    mask requires QA_PIXEL bits 0-4 to all be zero:
    ``bits_to_mask([0, 1, 2, 3, 4]) == 0b11111 == 31``.

    Args:
        bits: Bit positions (0-based, each >= 0). Duplicates are allowed and
            have no extra effect.

    Returns:
        Integer with exactly the given bits set (0 for an empty sequence).

    Raises:
        ValueError: If any bit position is negative.
    """
    mask = 0
    for bit in bits:
        if bit < 0:
            raise ValueError(f"bit positions must be >= 0, got {bit}")
        mask |= 1 << bit
    return mask


def sensor_keys(params: dict[str, Any]) -> list[str]:
    """The Landsat collections to use, in chronological order.

    Read from ``landsat_c2l2.sensor_keys`` rather than hardcoded, so an
    OLI-only or TM-only sensitivity run is a config change.

    Args:
        params: Parsed params mapping.

    Returns:
        Ordered list of keys into ``params["datasets"]``.

    Raises:
        KeyError: If a configured key is absent from ``datasets``.
    """
    keys: list[str] = list(params["landsat_c2l2"]["sensor_keys"])
    unknown = [k for k in keys if k not in params["datasets"]]
    if unknown:
        raise KeyError(
            f"landsat_c2l2.sensor_keys refers to datasets that do not exist: "
            f"{unknown}. Known datasets: {sorted(params['datasets'])}"
        )
    return keys


def resolve_sensors(
    sensors: Sequence[str] | None, params: dict[str, Any]
) -> list[str]:
    """Validate a sensor subset, preserving the configured chronological order.

    Args:
        sensors: Requested subset of sensor keys, or ``None`` for all of them.
        params: Parsed params mapping.

    Returns:
        Ordered list of sensor keys.

    Raises:
        ValueError: If ``sensors`` is empty or names an unconfigured sensor.
    """
    configured = sensor_keys(params)
    if sensors is None:
        return configured
    requested = list(sensors)
    if not requested:
        raise ValueError(
            f"sensors must name at least one of {configured}, got an empty sequence"
        )
    unknown = [s for s in requested if s not in configured]
    if unknown:
        raise ValueError(
            f"unknown sensor keys {unknown}; configured sensors are {configured}"
        )
    return [key for key in configured if key in set(requested)]


def source_sr_bands(sensor_key: str, params: dict[str, Any]) -> list[str]:
    """Per-sensor surface-reflectance band names, ordered to match the harmonised names.

    TM/ETM+ and OLI number their bands differently (TM blue is ``SR_B1``, OLI
    blue is ``SR_B2``). This returns the source band names in the same order as
    ``landsat_c2l2.harmonised_sr_bands``, which is what makes one ``select()``
    call rename every sensor onto a common schema.

    Args:
        sensor_key: Key into ``params["datasets"]``, e.g. ``"landsat8"``.
        params: Parsed params mapping.

    Returns:
        Source band names, e.g. ``["SR_B2", "SR_B3", ..., "SR_B7"]`` for OLI.

    Raises:
        KeyError: If the sensor lacks a mapping for a harmonised band name.
    """
    harmonised = params["landsat_c2l2"]["harmonised_sr_bands"]
    mapping = params["datasets"][sensor_key]["sr_bands"]
    missing = [name for name in harmonised if name not in mapping]
    if missing:
        raise KeyError(
            f"datasets.{sensor_key}.sr_bands has no entry for {missing}; "
            f"it defines {sorted(mapping)}"
        )
    return [mapping[name] for name in harmonised]


def output_band_names(
    params: dict[str, Any],
    include_sr: bool = True,
    include_st_qa: bool = True,
) -> list[str]:
    """Band names every scene in the harmonised collection carries, in order.

    Fixed order and identical dtype across sensors is what lets the four
    collections merge cleanly and lets :mod:`colombo_uhi.composites` build a
    correctly-shaped placeholder for years with no usable scenes.

    Args:
        params: Parsed params mapping.
        include_sr: Whether the six harmonised reflectance bands are present.
        include_st_qa: Whether the scaled ``ST_QA_K`` band is present.

    Returns:
        ``["LST_C", "blue", "green", "red", "nir", "swir1", "swir2", "ST_QA_K"]``
        with the actual configured names, minus any group switched off.
    """
    c2l2 = params["landsat_c2l2"]
    names = [c2l2["lst_band_name"]]
    if include_sr:
        names.extend(c2l2["harmonised_sr_bands"])
    if include_st_qa:
        names.append(c2l2["st_qa_band_name"])
    return names


def monsoon_season(month: int, params: dict[str, Any]) -> str:
    """Name the monsoon season a calendar month falls in.

    Only the seasons listed in ``time.season_partition`` are considered. That
    list exists because ``time.seasons`` also holds ``dry_window`` (Jan-Mar),
    which is a comparison WINDOW overlapping two seasons — iterating
    ``seasons.keys()`` would map January to both ``ne_monsoon`` and
    ``dry_window``.

    Args:
        month: Calendar month, 1-12.
        params: Parsed params mapping.

    Returns:
        The season key, e.g. ``"sw_monsoon"``.

    Raises:
        ValueError: If ``month`` is out of range, or if the configured partition
            does not assign the month exactly one season (a params error).
    """
    if not 1 <= int(month) <= _MONTHS_IN_YEAR:
        raise ValueError(f"month must be in 1..{_MONTHS_IN_YEAR}, got {month}")

    time_cfg = params["time"]
    matches = [
        key
        for key in time_cfg["season_partition"]
        if int(month) in time_cfg["seasons"][key]["months"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"time.season_partition assigns month {month} to {len(matches)} "
            f"seasons ({matches}); it must assign exactly one. Check that "
            f"{list(time_cfg['season_partition'])} partitions months 1-12."
        )
    return matches[0]


def season_by_month(params: dict[str, Any]) -> list[str]:
    """Month -> season lookup as a 12-element list, index 0 = January.

    Built client-side from the pure :func:`monsoon_season` so the season
    property can be assigned inside a server-side ``map()`` with a single
    ``ee.List.get()`` — no server-side branching, and the partition is
    validated once, up front, where the error message is readable.

    Args:
        params: Parsed params mapping.

    Returns:
        Season keys ordered January..December.
    """
    return [monsoon_season(m, params) for m in range(1, _MONTHS_IN_YEAR + 1)]


# =============================================================================
# Earth Engine helpers
# =============================================================================
def month_filter(months: Sequence[int]) -> "ee.Filter":
    """Calendar filter matching any of the given months.

    Handles wrapping windows such as the NE monsoon (Dec, Jan, Feb) because it
    ORs single-month ranges rather than using one ``calendarRange(start, end)``.

    Args:
        months: Calendar months, 1-12.

    Returns:
        An ``ee.Filter`` matching images acquired in any of those months.

    Raises:
        ValueError: If ``months`` is empty or holds an out-of-range value.
    """
    # Validation runs BEFORE the deferred import so it stays unit-testable on a
    # machine without earthengine-api.
    if not list(months):
        raise ValueError("months must contain at least one month")
    bad = [m for m in months if not 1 <= int(m) <= _MONTHS_IN_YEAR]
    if bad:
        raise ValueError(f"months must be in 1..{_MONTHS_IN_YEAR}, got {bad}")

    import ee  # Deferred: see module docstring.

    return ee.Filter.Or(*[ee.Filter.calendarRange(m, m, "month") for m in months])


def qa_clear_mask(image: "ee.Image", params: dict[str, Any]) -> "ee.Image":
    """Compute the standard clear-observation mask for a C2 L2 scene.

    Standard mask per CLAUDE.md: QA_PIXEL bits 0-4 (Fill, Dilated Cloud,
    Cirrus, Cloud, Cloud Shadow) all zero AND ``QA_RADSAT == 0`` (no saturated
    bands). Bit 2 is *Unused* (always 0) on TM/ETM+, so the same mask is valid
    across L5/L7/L8/L9.

    Note that requiring the whole ``QA_RADSAT`` word to be zero also excludes
    terrain-occluded pixels (bit 11) — marginally stricter than CLAUDE.md's
    wording, and immaterial on flat coastal Colombo. See params.

    Args:
        image: A single Landsat C2 L2 ``ee.Image``.
        params: Parsed params mapping (``landsat_c2l2`` section is used).

    Returns:
        Single-band 0/1 ``ee.Image`` (1 = clear, unsaturated observation).
    """
    import ee  # Deferred: see module docstring.

    c2l2 = params["landsat_c2l2"]
    qa_bits = bits_to_mask(c2l2["standard_mask"]["require_zero_bits"])
    qa_pixel = image.select(c2l2["qa_pixel_band"])
    qa_radsat = image.select(c2l2["qa_radsat_band"])

    clear = qa_pixel.bitwiseAnd(qa_bits).eq(0)
    unsaturated = qa_radsat.eq(c2l2["standard_mask"]["require_qa_radsat_value"])
    return clear.And(unsaturated).rename("clear")


def qa_water_flag(image: "ee.Image", params: dict[str, Any]) -> "ee.Image":
    """Extract the QA_PIXEL water bit (bit 7) as a 0/1 image.

    Args:
        image: A single Landsat C2 L2 ``ee.Image``.
        params: Parsed params mapping (``landsat_c2l2.qa_pixel_bits.water``).

    Returns:
        Single-band 0/1 ``ee.Image`` named ``water`` (1 = flagged as water).
    """
    import ee  # Deferred: see module docstring.

    c2l2 = params["landsat_c2l2"]
    water_bit = bits_to_mask([c2l2["qa_pixel_bits"]["water"]])
    return (
        image.select(c2l2["qa_pixel_band"]).bitwiseAnd(water_bit).gt(0).rename("water")
    )


def scale_sr(image: "ee.Image", params: dict[str, Any]) -> "ee.Image":
    """Convert surface-reflectance DN bands (``SR_B*``) to reflectance.

    Applies ``DN * 0.0000275 - 0.2`` (constants from params) to every SR band
    while leaving all other bands untouched.

    Args:
        image: A single Landsat C2 L2 ``ee.Image``.
        params: Parsed params mapping (``landsat_c2l2.sr_scale``/``sr_offset``).

    Returns:
        The input ``ee.Image`` with its ``SR_B.*`` bands replaced by scaled
        reflectance values (band names preserved).
    """
    import ee  # Deferred: see module docstring.

    c2l2 = params["landsat_c2l2"]
    scaled = (
        image.select("SR_B.*")
        .multiply(c2l2["sr_scale"])
        .add(c2l2["sr_offset"])
    )
    return image.addBands(scaled, overwrite=True)


def scale_st(image: "ee.Image", sensor_key: str, params: dict[str, Any]) -> "ee.Image":
    """Convert a scene's thermal DN band to land surface temperature in Celsius.

    ``DN * 0.00341802 + 149.0`` gives Kelvin; subtracting 273.15 gives Celsius.
    Invalid retrievals are masked using two independent gates:

    1. raw DN outside ``landsat_c2l2.st_valid_dn_range`` (this is what removes
       fill and in-footprint retrieval failures that QA_PIXEL bit 0 misses);
    2. optionally, Celsius outside ``landsat_c2l2.lst_plausible_range_c``.

    Both gates are applied to the thermal band ONLY. Applying them to a whole
    multi-band image would delete perfectly good surface reflectance wherever
    the thermal retrieval failed (``updateMask`` with a single-band mask applies
    it to every band).

    Args:
        image: A single Landsat C2 L2 ``ee.Image``.
        sensor_key: Key into ``params["datasets"]`` (selects ``ST_B6`` vs ``ST_B10``).
        params: Parsed params mapping.

    Returns:
        Single-band ``ee.Image`` named ``LST_C`` (degrees Celsius, masked).
    """
    import ee  # Deferred: see module docstring.

    c2l2 = params["landsat_c2l2"]
    st_dn = image.select(params["datasets"][sensor_key]["st_band"])

    dn_min, dn_max = c2l2["st_valid_dn_range"]
    valid = st_dn.gte(dn_min).And(st_dn.lte(dn_max))

    lst_c = (
        st_dn.multiply(c2l2["st_scale"])
        .add(c2l2["st_offset"])
        .subtract(c2l2["kelvin_to_celsius_offset"])
        .rename(c2l2["lst_band_name"])
    )

    plausible = c2l2.get("lst_plausible_range_c")
    if plausible is not None:
        lo, hi = plausible
        valid = valid.And(lst_c.gte(lo)).And(lst_c.lte(hi))

    return lst_c.updateMask(valid)


def scale_st_qa(image: "ee.Image", params: dict[str, Any]) -> "ee.Image":
    """Scale the ``ST_QA`` uncertainty band to Kelvin.

    Args:
        image: A single Landsat C2 L2 ``ee.Image``.
        params: Parsed params mapping (``landsat_c2l2.st_qa_scale``).

    Returns:
        Single-band ``ee.Image`` named ``ST_QA_K`` (Kelvin).
    """
    import ee  # Deferred: see module docstring.

    c2l2 = params["landsat_c2l2"]
    return (
        image.select(c2l2["st_qa_band"])
        .multiply(c2l2["st_qa_scale"])
        .rename(c2l2["st_qa_band_name"])
    )


def _prepare_scene(
    image: "ee.Image",
    sensor_key: str,
    params: dict[str, Any],
    st_qa_max_kelvin: float | None = None,
    include_sr: bool = True,
    include_st_qa: bool = True,
) -> "ee.Image":
    """Mask, scale and rename one raw C2 L2 scene onto the harmonised schema.

    Masking order is deliberate and load-bearing:

    1. thermal-specific gates (DN range, plausible Celsius, optional ST_QA
       threshold) are applied to ``LST_C`` alone;
    2. the bands are concatenated, so each keeps its own mask;
    3. the cloud/saturation mask is applied last, to ALL bands.

    Collapsing 1 and 3 into a single whole-image ``updateMask`` would throw
    away usable reflectance wherever the thermal retrieval failed.

    Args:
        image: A single raw Landsat C2 L2 ``ee.Image``.
        sensor_key: Key into ``params["datasets"]``.
        params: Parsed params mapping.
        st_qa_max_kelvin: Per-pixel LST uncertainty ceiling in Kelvin; ``None``
            falls back to ``landsat_c2l2.st_qa_max_kelvin`` (itself ``null`` =
            no filtering by default).
        include_sr: Emit the six harmonised reflectance bands. Set ``False`` for
            LST-only work: scaling and renaming six bands on every scene is real
            graph weight, and over a 26-year series it is weight Earth Engine
            charges against the user memory limit for nothing. ``indices.py``
            needs them; trends and zonal LST means do not.
        include_st_qa: Emit the scaled ``ST_QA_K`` band. The uncertainty filter
            still works when this is ``False`` — the band is computed for the
            gate and simply not returned.

    Returns:
        ``ee.Image`` with the bands from :func:`output_band_names`, as float,
        carrying the source scene's properties and ``system:time_start``.
    """
    import ee  # Deferred: see module docstring.

    c2l2 = params["landsat_c2l2"]

    clear = qa_clear_mask(image, params)
    lst_c = scale_st(image, sensor_key, params)

    threshold = (
        c2l2["st_qa_max_kelvin"] if st_qa_max_kelvin is None else st_qa_max_kelvin
    )
    st_qa_k = (
        scale_st_qa(image, params) if include_st_qa or threshold is not None else None
    )
    if threshold is not None:
        assert st_qa_k is not None
        # Thermal-only gate, same reasoning as the gates inside scale_st().
        lst_c = lst_c.updateMask(st_qa_k.lte(float(threshold)))

    bands: list["ee.Image"] = [lst_c]
    if include_sr:
        bands.append(
            scale_sr(image, params).select(
                source_sr_bands(sensor_key, params), c2l2["harmonised_sr_bands"]
            )
        )
    if include_st_qa:
        assert st_qa_k is not None
        bands.append(st_qa_k)

    stacked = (
        ee.Image.cat(bands)
        .toFloat()  # identical dtype across sensors, so merge + reduce stay valid
        .updateMask(clear)
    )

    # copyProperties returns an ee.Element; a mapped function must return an
    # ee.Image, hence the cast. system:time_start is then set explicitly.
    return ee.Image(stacked.copyProperties(image, image.propertyNames())).set(
        "system:time_start", image.get("system:time_start")
    )


def _tag_scene(
    image: "ee.Image",
    sensor_key: str,
    params: dict[str, Any],
    season_lookup: Sequence[str],
) -> "ee.Image":
    """Attach the sensor/year/month/season properties composites group on.

    Args:
        image: A prepared scene.
        sensor_key: Key into ``params["datasets"]``.
        params: Parsed params mapping.
        season_lookup: 12-element month -> season list from
            :func:`season_by_month`, resolved client-side so no branching is
            needed server-side.

    Returns:
        The same ``ee.Image`` with extra properties set.
    """
    import ee  # Deferred: see module docstring.

    comp = params["composites"]
    date = ee.Date(image.get("system:time_start"))
    month = ee.Number(date.get("month"))
    return image.set(
        {
            comp["year_property"]: date.get("year"),
            "month": month,
            comp["season_property"]: ee.List(list(season_lookup)).get(
                month.subtract(1)
            ),
            comp["sensor_property"]: params["datasets"][sensor_key]["sensor"],
            "sensor_key": sensor_key,
        }
    )


def _sensor_collection(
    sensor_key: str,
    params: dict[str, Any],
    region: "ee.Geometry",
    start_date: str,
    end_date: str,
    months: Sequence[int] | None = None,
    include_l7_slc_off: bool | None = None,
    st_qa_max_kelvin: float | None = None,
    include_sr: bool = True,
    include_st_qa: bool = True,
) -> "ee.ImageCollection":
    """Build one sensor's filtered, masked, harmonised collection.

    Args:
        sensor_key: Key into ``params["datasets"]``.
        params: Parsed params mapping.
        region: Geometry to intersect scenes with.
        start_date: Inclusive start date, ``YYYY-MM-DD``.
        end_date: **Exclusive** end date, ``YYYY-MM-DD`` (Earth Engine semantics).
        months: Optional calendar-month restriction.
        include_l7_slc_off: Landsat 7 only; ``None`` uses
            ``landsat_c2l2.include_l7_slc_off``.
        st_qa_max_kelvin: See :func:`_prepare_scene`.
        include_sr: See :func:`_prepare_scene`.
        include_st_qa: See :func:`_prepare_scene`.

    Returns:
        ``ee.ImageCollection`` of prepared, tagged scenes.
    """
    import ee  # Deferred: see module docstring.

    c2l2 = params["landsat_c2l2"]
    dataset = params["datasets"][sensor_key]

    collection = (
        ee.ImageCollection(dataset["id"])
        .filterBounds(region)
        .filterDate(start_date, end_date)
    )

    if c2l2["processing_level_filter_enabled"]:
        # neq, not eq: ee.Filter.neq is eq().Not(), so a missing or renamed
        # PROCESSING_LEVEL property lets the scene THROUGH rather than silently
        # emptying the collection. On L2SR scenes the ST bands exist but are
        # fully masked, so this filter is a speed optimisation, not correctness.
        collection = collection.filter(
            ee.Filter.neq(
                c2l2["processing_level_property"], c2l2["processing_level_exclude"]
            )
        )

    if months:
        collection = collection.filter(month_filter(months))

    keep_slc_off = (
        c2l2["include_l7_slc_off"]
        if include_l7_slc_off is None
        else include_l7_slc_off
    )
    slc_off_after = dataset.get("slc_off_after")
    if slc_off_after and not keep_slc_off:
        # filterDate's end is EXCLUSIVE, and slc_off_after is the last SLC-on
        # day, so advance one day to keep that day's scenes.
        collection = collection.filterDate(
            start_date, ee.Date(slc_off_after).advance(1, "day")
        )

    season_lookup = season_by_month(params)

    def prepare(image: "ee.Image") -> "ee.Image":
        return _tag_scene(
            _prepare_scene(
                image,
                sensor_key,
                params,
                st_qa_max_kelvin,
                include_sr=include_sr,
                include_st_qa=include_st_qa,
            ),
            sensor_key,
            params,
            season_lookup,
        )

    return collection.map(prepare)


def harmonised_collection(
    params: dict[str, Any],
    region: "ee.Geometry | None" = None,
    start_date: str | None = None,
    end_date: str | None = None,
    months: Sequence[int] | None = None,
    sensors: Sequence[str] | None = None,
    include_l7_slc_off: bool | None = None,
    st_qa_max_kelvin: float | None = None,
    include_sr: bool = True,
    include_st_qa: bool = True,
) -> "ee.ImageCollection":
    """One harmonised L5/L7/L8/L9 C2 L2 collection, sorted by acquisition time.

    Every scene carries the bands from :func:`output_band_names` and the
    ``sensor`` / ``sensor_key`` / ``year`` / ``month`` / ``season`` properties.
    Collection 2 is already inter-calibrated across TM/ETM+/OLI, so no manual
    harmonisation coefficients are applied (``landsat_c2l2.harmonisation:
    none``) — but that should still be checked empirically on the overlap years
    (2000-2012 for L5/L7, 2021-2024 for L8/L9).

    Landsat 7 SLC-off scenes are INCLUDED by default. Rationale: median
    compositing over many scenes dilutes the striping, and excluding them would
    empty 2012-05 to 2013-03 entirely (Landsat 5 had ended, Landsat 8 had not
    launched). Pass ``include_l7_slc_off=False`` for a sensitivity run.

    Args:
        params: Parsed params mapping.
        region: Geometry to filter scenes to; defaults to
            :func:`colombo_uhi.aoi.analysis_region`.
        start_date: Inclusive start, ``YYYY-MM-DD``; defaults to
            ``time.start_year``-01-01.
        end_date: **Exclusive** end, ``YYYY-MM-DD``; defaults to
            ``time.end_year`` + 1 -01-01, so the whole final year is included.
        months: Optional calendar-month restriction, e.g.
            ``time.seasons.dry_window.months``.
        sensors: Optional subset of ``landsat_c2l2.sensor_keys``.
        include_l7_slc_off: Override the params default described above.
        st_qa_max_kelvin: Optional per-pixel LST uncertainty ceiling in Kelvin;
            ``None`` uses ``landsat_c2l2.st_qa_max_kelvin`` (``null`` = off).
        include_sr: Emit the six harmonised reflectance bands. Set ``False`` for
            LST-only work (trends, zonal means): it strips six band scalings per
            scene from the graph, which is the cheapest way to keep a long
            series inside the Earth Engine user memory limit. Leave it ``True``
            whenever :mod:`colombo_uhi.indices` will run on the result.
        include_st_qa: Emit the scaled ``ST_QA_K`` band. The optional
            uncertainty filter still applies when this is ``False``.

    Returns:
        ``ee.ImageCollection`` sorted ascending by ``system:time_start``.
    """
    import ee  # Deferred: see module docstring.

    if region is None:
        # Deferred import: aoi imports this module, so importing it at module
        # scope would be circular. By call time both modules are fully loaded.
        from colombo_uhi import aoi

        region = aoi.analysis_region(params)

    time_cfg = params["time"]
    start = start_date or f"{time_cfg['start_year']}-01-01"
    end = end_date or f"{int(time_cfg['end_year']) + 1}-01-01"

    merged: "ee.ImageCollection | None" = None
    for key in resolve_sensors(sensors, params):
        collection = _sensor_collection(
            key,
            params,
            region,
            start,
            end,
            months=months,
            include_l7_slc_off=include_l7_slc_off,
            st_qa_max_kelvin=st_qa_max_kelvin,
            include_sr=include_sr,
            include_st_qa=include_st_qa,
        )
        merged = collection if merged is None else merged.merge(collection)

    assert merged is not None  # resolve_sensors guarantees a non-empty list
    return ee.ImageCollection(merged).sort("system:time_start")
