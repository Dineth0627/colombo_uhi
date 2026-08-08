"""Study-area geometries: boundaries, rural references, water mask (Phase 1).

Provides:
    * Colombo District and Western Province from FAO GAUL level 2.
    * DS/GN division loaders backed by user-uploaded EE assets
      (``aoi.assets.*`` in params), with a prominent warning + GAUL-district
      fallback while the assets are missing. GAUL for Sri Lanka stops at
      district level — there is NO DS/GN/CMC boundary in any GEE dataset
      (decision recorded in PROGRESS.md, 2026-08-08).
    * ``cmc_boundary`` — Colombo Municipal Council (~37 km2) dissolved from
      named DS divisions of the DS asset; actionable error while missing.
    * ``urban_extent`` — asset-free urban-core polygon from GHSL built surface.
    * Combined water mask (MNDWI OR QA_PIXEL water frequency OR JRC GSW
      occurrence) with optional shoreline-buffer dilation.
    * Two SUHII rural-reference definitions behind one interface
      (:func:`rural_reference`): ``"buffer_ring"`` and ``"lcz_based"``.

Design notes:
    * ``import ee`` is deferred into function bodies so this module (and the
      local pytest suite) imports cleanly without ``earthengine-api``.
    * Every constant comes from ``config/params.yaml``; validation helpers at
      the top are pure Python and unit-tested.
    * All geometry buffers/areas use ``_MAX_ERROR_M`` metres of tolerance —
      GAUL is simplified at 500 m, so sub-metre precision would be fake.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Sequence

from colombo_uhi import landsat

if TYPE_CHECKING:  # pragma: no cover - typing only, never at runtime
    import ee

#: Geometry tolerance (m) for buffer/difference/area ops (GAUL is 500 m-simplified).
_MAX_ERROR_M = 10

#: Valid LCZ class code ranges (Demuzere et al. 2022): built 1-10, natural 11-17.
LCZ_BUILT_RANGE = (1, 10)
LCZ_NATURAL_RANGE = (11, 17)


# =============================================================================
# Pure-Python validation helpers (unit-tested; no Earth Engine)
# =============================================================================
def validate_buffer_ring_km(inner_km: float, outer_km: float) -> tuple[float, float]:
    """Validate the rural buffer-ring distances.

    Args:
        inner_km: Ring inner edge, km beyond the urban-core boundary.
        outer_km: Ring outer edge, km beyond the urban-core boundary.

    Returns:
        ``(inner_km, outer_km)`` as floats.

    Raises:
        ValueError: If either distance is not positive or inner >= outer.
    """
    inner, outer = float(inner_km), float(outer_km)
    if inner <= 0 or outer <= 0:
        raise ValueError(
            f"buffer ring distances must be positive, got inner={inner_km}, "
            f"outer={outer_km}"
        )
    if inner >= outer:
        raise ValueError(
            f"buffer ring inner edge must be < outer edge, got inner={inner_km}, "
            f"outer={outer_km}"
        )
    return inner, outer


def resolve_rural_method(name: str, valid: Sequence[str]) -> str:
    """Validate a rural-reference method name against the configured list.

    Args:
        name: Requested method, e.g. ``"buffer_ring"`` or ``"lcz_based"``.
        valid: Allowed method names (``uhi.suhii.rural_definitions`` in params).

    Returns:
        The validated method name.

    Raises:
        ValueError: If ``name`` is not in ``valid``, listing the valid options.
    """
    if name not in valid:
        raise ValueError(
            f"unknown rural-reference method '{name}'; valid options: "
            f"{sorted(valid)}"
        )
    return name


def validate_lcz_classes(
    urban: Sequence[int], rural: Sequence[int]
) -> tuple[list[int], list[int]]:
    """Validate LCZ urban/rural class lists.

    Args:
        urban: LCZ built classes (must be within 1-10).
        rural: LCZ natural classes (must be within 11-17).

    Returns:
        ``(urban, rural)`` as lists of ints.

    Raises:
        ValueError: If either list is empty, out of its valid range, or the
            two lists overlap.
    """
    urban_list, rural_list = [int(c) for c in urban], [int(c) for c in rural]
    if not urban_list or not rural_list:
        raise ValueError("LCZ urban and rural class lists must both be non-empty")
    lo, hi = LCZ_BUILT_RANGE
    bad = [c for c in urban_list if not lo <= c <= hi]
    if bad:
        raise ValueError(f"LCZ urban classes must be in {lo}-{hi}, got {bad}")
    lo, hi = LCZ_NATURAL_RANGE
    bad = [c for c in rural_list if not lo <= c <= hi]
    if bad:
        raise ValueError(f"LCZ rural classes must be in {lo}-{hi}, got {bad}")
    overlap = set(urban_list) & set(rural_list)
    if overlap:
        raise ValueError(f"LCZ urban and rural classes overlap: {sorted(overlap)}")
    return urban_list, rural_list


def validate_water_mask_params(params: dict[str, Any]) -> dict[str, Any]:
    """Validate the ``aoi.water_mask`` configuration.

    Args:
        params: Full parsed params mapping.

    Returns:
        The validated ``aoi.water_mask`` sub-mapping.

    Raises:
        ValueError: On any out-of-range threshold, bad year/month window,
            unknown reducer, or unknown Landsat source key.
    """
    cfg = params["aoi"]["water_mask"]
    if not -1.0 <= cfg["mndwi_threshold"] <= 1.0:
        raise ValueError(
            f"mndwi_threshold must be in [-1, 1], got {cfg['mndwi_threshold']}"
        )
    if not 0.0 <= cfg["qa_water_freq_threshold"] <= 1.0:
        raise ValueError(
            "qa_water_freq_threshold must be in [0, 1], got "
            f"{cfg['qa_water_freq_threshold']}"
        )
    if not 0 <= cfg["jrc_occurrence_threshold_pct"] <= 100:
        raise ValueError(
            "jrc_occurrence_threshold_pct must be in [0, 100], got "
            f"{cfg['jrc_occurrence_threshold_pct']}"
        )
    if cfg["shoreline_buffer_m"] < 0:
        raise ValueError(
            f"shoreline_buffer_m must be >= 0, got {cfg['shoreline_buffer_m']}"
        )

    comp = cfg["composite"]
    if comp["start_year"] > comp["end_year"]:
        raise ValueError(
            f"composite start_year {comp['start_year']} > end_year "
            f"{comp['end_year']}"
        )
    months = comp["months"]
    if not months or not all(1 <= m <= 12 for m in months):
        raise ValueError(f"composite months must be a non-empty subset of 1-12, got {months}")
    if comp["reducer"] not in ("median", "mean"):
        raise ValueError(f"composite reducer must be 'median' or 'mean', got {comp['reducer']!r}")
    unknown = [k for k in comp["landsat_sources"] if k not in params["datasets"]]
    if not comp["landsat_sources"] or unknown:
        raise ValueError(
            f"composite landsat_sources must be non-empty keys of `datasets`; unknown: {unknown}"
        )
    return cfg


# =============================================================================
# Administrative boundaries
# =============================================================================
def _gaul_level2(params: dict[str, Any]) -> "ee.FeatureCollection":
    """Load FAO GAUL level 2 filtered to Sri Lanka."""
    import ee  # Deferred: see module docstring.

    gaul_cfg = params["aoi"]["gaul"]
    return ee.FeatureCollection(params["datasets"]["gaul_level2"]["id"]).filter(
        ee.Filter.eq(gaul_cfg["country_property"], gaul_cfg["country_value"])
    )


def colombo_district(params: dict[str, Any]) -> "ee.FeatureCollection":
    """Colombo District boundary from FAO GAUL level 2.

    Args:
        params: Parsed params mapping.

    Returns:
        ``ee.FeatureCollection`` that should contain exactly one feature
        (``ADM2_NAME == 'Colombo'``); notebook 01 verifies the count.
    """
    import ee  # Deferred: see module docstring.

    gaul_cfg = params["aoi"]["gaul"]
    return _gaul_level2(params).filter(
        ee.Filter.eq(gaul_cfg["district_property"], gaul_cfg["district_value"])
    )


def western_province(params: dict[str, Any]) -> "ee.FeatureCollection":
    """Western Province = Colombo + Gampaha + Kalutara districts (GAUL level 2).

    Built from the three verified ADM2 district names rather than an ADM1
    filter, because the exact GAUL ADM1_NAME string for the province is
    unverified while the district names are.

    Args:
        params: Parsed params mapping.

    Returns:
        ``ee.FeatureCollection`` with one feature per district (3 expected).
    """
    import ee  # Deferred: see module docstring.

    gaul_cfg = params["aoi"]["gaul"]
    return _gaul_level2(params).filter(
        ee.Filter.inList(
            gaul_cfg["district_property"], params["aoi"]["western_province_districts"]
        )
    )


def _missing_asset_fallback(kind: str, params: dict[str, Any]) -> "ee.FeatureCollection":
    """Warn prominently and return the GAUL district as a 1-feature fallback."""
    banner = (
        "\n" + "!" * 78 + "\n"
        f"!! {kind} asset NOT CONFIGURED (aoi.assets in config/params.yaml is null).\n"
        "!! FALLING BACK to the FAO GAUL *district* boundary (1 coarse feature).\n"
        "!! Sub-district statistics are IMPOSSIBLE until you upload boundaries:\n"
        "!!   1. Download 'Sri Lanka - Subnational Administrative Boundaries'\n"
        "!!      from OCHA/HDX (admin3 = DS divisions, admin4 = GN divisions).\n"
        "!!   2. Upload the shapefile(s) as EE table assets (Code Editor > Assets).\n"
        "!!   3. Paste the asset ids into aoi.assets.ds_divisions / gn_divisions.\n"
        "!! Notebook 01 has the step-by-step instructions.\n" + "!" * 78
    )
    print(banner)
    warnings.warn(f"{kind} asset missing - using GAUL district fallback", stacklevel=3)
    return colombo_district(params)


def ds_divisions(params: dict[str, Any]) -> "ee.FeatureCollection":
    """Divisional Secretariat (DS) divisions of Colombo District (13 expected).

    Loads the user-uploaded EE asset at ``aoi.assets.ds_divisions``. While
    that is null, warns prominently and falls back to the GAUL district
    (a single coarse feature — unusable for sub-district statistics).

    Args:
        params: Parsed params mapping.

    Returns:
        ``ee.FeatureCollection`` of DS divisions, or the 1-feature fallback.
    """
    import ee  # Deferred: see module docstring.

    asset_id = params["aoi"]["assets"]["ds_divisions"]
    if not asset_id:
        return _missing_asset_fallback("DS-division", params)
    return ee.FeatureCollection(asset_id)


def gn_divisions(params: dict[str, Any]) -> "ee.FeatureCollection":
    """Grama Niladhari (GN) divisions of Colombo District (557 expected).

    Loads the user-uploaded EE asset at ``aoi.assets.gn_divisions``. While
    that is null, falls back (with a prominent warning) to the DS-division
    asset if configured, else to the GAUL district.

    Args:
        params: Parsed params mapping.

    Returns:
        ``ee.FeatureCollection`` of GN divisions, or the best available fallback.
    """
    import ee  # Deferred: see module docstring.

    asset_id = params["aoi"]["assets"]["gn_divisions"]
    if not asset_id:
        if params["aoi"]["assets"]["ds_divisions"]:
            warnings.warn(
                "GN-division asset missing - falling back to DS divisions "
                "(13 units instead of 557; MAUP sensitivity applies)",
                stacklevel=2,
            )
            return ds_divisions(params)
        return _missing_asset_fallback("GN-division", params)
    return ee.FeatureCollection(asset_id)


def cmc_boundary(params: dict[str, Any]) -> "ee.Geometry":
    """Colombo Municipal Council (~37 km2) boundary.

    CMC exists in no GEE dataset; it is dissolved from the DS divisions named
    in ``aoi.cmc.ds_division_names`` (Colombo + Thimbirigasyaya) of the
    user-uploaded DS asset. Notebook 01 sanity-checks the area against
    ``aoi.expected_areas_km2.cmc``. If the name filter matches nothing the
    result is an empty geometry — the notebook's area print (0 km2) catches
    that; check ``aoi.cmc.ds_name_property`` against the asset's schema.

    Args:
        params: Parsed params mapping.

    Returns:
        Dissolved ``ee.Geometry`` of the CMC.

    Raises:
        RuntimeError: While ``aoi.assets.ds_divisions`` is null, with the
            exact steps to fix it.
    """
    import ee  # Deferred: see module docstring.

    cmc_cfg = params["aoi"]["cmc"]
    asset_id = params["aoi"]["assets"]["ds_divisions"]
    if not asset_id:
        raise RuntimeError(
            "Cannot build the CMC boundary: aoi.assets.ds_divisions is null in "
            "config/params.yaml and no GEE dataset contains CMC/DS boundaries "
            "(GAUL stops at district level for Sri Lanka). Fix:\n"
            "  1. Download 'Sri Lanka - Subnational Administrative Boundaries' "
            "(admin3 layer) from OCHA/HDX.\n"
            "  2. Upload the shapefile as an EE table asset "
            "(Code Editor > Assets > New > Shape files).\n"
            "  3. Set aoi.assets.ds_divisions to the asset id in "
            "config/params.yaml.\n"
            f"CMC will then be the dissolve of {cmc_cfg['ds_division_names']} "
            f"(matched on property '{cmc_cfg['ds_name_property']}')."
        )
    dissolved = (
        ee.FeatureCollection(asset_id)
        .filter(
            ee.Filter.inList(
                cmc_cfg["ds_name_property"], cmc_cfg["ds_division_names"]
            )
        )
        .union(_MAX_ERROR_M)
    )
    return dissolved.geometry(_MAX_ERROR_M)


# =============================================================================
# Urban extent + analysis region
# =============================================================================
def urban_extent(params: dict[str, Any]) -> "ee.Geometry":
    """Asset-free urban-core polygon from GHSL built surface.

    Thresholds ``built_surface`` (m2 of built area per 100 m cell) of the
    configured GHSL epoch inside Western Province and vectorises the result.
    This is a PHYSICAL built-up extent, not the administrative CMC.

    Args:
        params: Parsed params mapping (``aoi.urban_extent`` section).

    Returns:
        Dissolved ``ee.Geometry`` of the built-up extent.
    """
    import ee  # Deferred: see module docstring.

    cfg = params["aoi"]["urban_extent"]
    ghsl_cfg = params["datasets"]["ghsl_built"]
    region = western_province(params).geometry(_MAX_ERROR_M)

    built = (
        ee.ImageCollection(ghsl_cfg["id"])
        .filterDate(f"{cfg['ghsl_epoch']}-01-01", f"{cfg['ghsl_epoch']}-12-31")
        .first()
        .select(ghsl_cfg["band"])
    )
    mask = built.gte(cfg["built_surface_threshold_m2"]).selfMask()
    vectors = mask.reduceToVectors(
        geometry=region,
        scale=cfg["vectorize_scale_m"],
        geometryType="polygon",
        eightConnected=True,
        maxPixels=1e10,
    )
    return vectors.union(_MAX_ERROR_M).geometry(_MAX_ERROR_M)


def analysis_region(params: dict[str, Any]) -> "ee.Geometry":
    """Bounding region for masks/composites: Western Province + ring headroom.

    Western Province buffered outward by the rural ring's outer distance, so
    every product (including the buffer-ring rural reference) fits inside it.

    Args:
        params: Parsed params mapping.

    Returns:
        Buffered ``ee.Geometry``.
    """
    outer_km = float(params["uhi"]["suhii"]["buffer_ring"]["outer_km"])
    return western_province(params).geometry(_MAX_ERROR_M).buffer(
        outer_km * 1000.0, _MAX_ERROR_M
    )


# =============================================================================
# Water mask
# =============================================================================
def _month_filter(months: Sequence[int]) -> "ee.Filter":
    """Calendar filter matching any of the given months (handles Dec-Feb wrap)."""
    import ee  # Deferred: see module docstring.

    return ee.Filter.Or(*[ee.Filter.calendarRange(m, m, "month") for m in months])


def _water_composite_collection(
    params: dict[str, Any], region: "ee.Geometry"
) -> "ee.ImageCollection":
    """Merged, QA-masked, scaled Landsat collection with green/swir1/water bands."""
    import ee  # Deferred: see module docstring.

    comp = params["aoi"]["water_mask"]["composite"]

    merged: "ee.ImageCollection | None" = None
    for source_key in comp["landsat_sources"]:
        ds = params["datasets"][source_key]
        sr_bands = ds["sr_bands"]

        def prep(img: "ee.Image", _sr_bands: dict[str, str] = sr_bands) -> "ee.Image":
            clear = landsat.qa_clear_mask(img, params)
            scaled = landsat.scale_sr(img, params)
            green = scaled.select(_sr_bands["green"]).rename("green")
            swir1 = scaled.select(_sr_bands["swir1"]).rename("swir1")
            water_flag = landsat.qa_water_flag(img, params)
            return ee.Image.cat([green, swir1, water_flag]).updateMask(clear)

        col = (
            ee.ImageCollection(ds["id"])
            .filterBounds(region)
            .filterDate(f"{comp['start_year']}-01-01", f"{comp['end_year']}-12-31")
            .filter(_month_filter(comp["months"]))
            .map(prep)
        )
        merged = col if merged is None else merged.merge(col)
    assert merged is not None  # landsat_sources validated non-empty
    return merged


def water_mask(params: dict[str, Any]) -> "ee.Image":
    """Combined 0/1 water mask (1 = water) for the analysis region.

    OR of three independent detectors, so a miss by any one source does not
    punch a hole in the mask:

    1. MNDWI (green - swir1)/(green + swir1) above ``mndwi_threshold`` on a
       QA-masked dry-season Landsat reflectance composite;
    2. QA_PIXEL water-bit frequency over the same collection above
       ``qa_water_freq_threshold``;
    3. JRC Global Surface Water ``occurrence`` >=
       ``jrc_occurrence_threshold_pct``.

    Args:
        params: Parsed params mapping (``aoi.water_mask`` section; validated
            via :func:`validate_water_mask_params`).

    Returns:
        Single-band 0/1 ``ee.Image`` named ``water``, clipped to
        :func:`analysis_region`.
    """
    import ee  # Deferred: see module docstring.

    cfg = validate_water_mask_params(params)
    region = analysis_region(params)
    collection = _water_composite_collection(params, region)

    reducer = {"median": ee.Reducer.median(), "mean": ee.Reducer.mean()}[
        cfg["composite"]["reducer"]
    ]
    composite = collection.select(["green", "swir1"]).reduce(reducer)
    mndwi = composite.normalizedDifference(
        [f"green_{cfg['composite']['reducer']}", f"swir1_{cfg['composite']['reducer']}"]
    )
    mndwi_water = mndwi.gt(cfg["mndwi_threshold"]).unmask(0)

    qa_freq = collection.select("water").mean()
    qa_water = qa_freq.gt(cfg["qa_water_freq_threshold"]).unmask(0)

    gsw_cfg = params["datasets"]["surface_water"]
    jrc_water = (
        ee.Image(gsw_cfg["id"])
        .select(gsw_cfg["band_occurrence"])
        .gte(cfg["jrc_occurrence_threshold_pct"])
        .unmask(0)
    )

    return (
        mndwi_water.Or(qa_water).Or(jrc_water).rename("water").clip(region)
    )


def water_exclusion_mask(
    params: dict[str, Any], shoreline_buffer_m: float | None = None
) -> "ee.Image":
    """Water mask dilated by an optional shoreline buffer (1 = exclude).

    Coastal/lakeside Landsat pixels mix land and water signals; dilating the
    water mask by 1-2 pixel widths (e.g. 60 m) removes that contaminated
    fringe from any land-only statistic.

    Args:
        params: Parsed params mapping.
        shoreline_buffer_m: Dilation radius in metres; ``None`` uses
            ``aoi.water_mask.shoreline_buffer_m`` (default 0 = plain water mask).

    Returns:
        Single-band 0/1 ``ee.Image`` named ``water_excluded``.
    """
    buffer_m = (
        float(params["aoi"]["water_mask"]["shoreline_buffer_m"])
        if shoreline_buffer_m is None
        else float(shoreline_buffer_m)
    )
    if buffer_m < 0:
        raise ValueError(f"shoreline buffer must be >= 0 m, got {buffer_m}")
    mask = water_mask(params)
    if buffer_m > 0:
        mask = mask.focalMax(radius=buffer_m, kernelType="circle", units="meters")
    return mask.rename("water_excluded")


# =============================================================================
# Rural-reference interface (SUHII urban/rural definitions)
# =============================================================================
def _paint(geometry: "ee.Geometry", region: "ee.Geometry") -> "ee.Image":
    """Rasterise a geometry to a 0/1 image over ``region``."""
    import ee  # Deferred: see module docstring.

    return (
        ee.Image(0)
        .byte()
        .paint(ee.FeatureCollection([ee.Feature(geometry)]), 1)
        .clip(region)
    )


def _lcz_image(params: dict[str, Any]) -> "ee.Image":
    """Load the recommended (filtered) band of the global LCZ map."""
    import ee  # Deferred: see module docstring.

    lcz_cfg = params["datasets"]["lcz"]
    return ee.ImageCollection(lcz_cfg["id"]).mosaic().select(lcz_cfg["band"])


def _lcz_class_mask(params: dict[str, Any], classes: Sequence[int]) -> "ee.Image":
    """0/1 image of pixels whose LCZ class is in ``classes``."""
    classes = [int(c) for c in classes]
    return _lcz_image(params).remap(classes, [1] * len(classes), 0)


def _buffer_ring_base(params: dict[str, Any]) -> "ee.Geometry":
    """Resolve the buffer ring's base geometry (``urban_extent`` or ``cmc``)."""
    base = params["uhi"]["suhii"]["buffer_ring"]["base"]
    if base == "urban_extent":
        return urban_extent(params)
    if base == "cmc":
        return cmc_boundary(params)
    raise ValueError(
        f"unknown buffer_ring base '{base}'; valid options: ['cmc', 'urban_extent']"
    )


def buffer_ring(params: dict[str, Any]) -> "ee.Geometry":
    """Rural-reference ring geometry around the urban core.

    Ring = base buffered by ``outer_km`` minus base buffered by ``inner_km``,
    where base is set by ``uhi.suhii.buffer_ring.base``.

    Args:
        params: Parsed params mapping.

    Returns:
        ``ee.Geometry`` of the ring (water/built-up NOT yet excluded — that
        happens in :func:`rural_mask`).
    """
    cfg = params["uhi"]["suhii"]["buffer_ring"]
    inner_km, outer_km = validate_buffer_ring_km(cfg["inner_km"], cfg["outer_km"])
    base = _buffer_ring_base(params)
    outer = base.buffer(outer_km * 1000.0, _MAX_ERROR_M)
    inner = base.buffer(inner_km * 1000.0, _MAX_ERROR_M)
    return outer.difference(inner, _MAX_ERROR_M)


def urban_mask(method: str, params: dict[str, Any]) -> "ee.Image":
    """0/1 urban mask for SUHII under the given rural-reference method.

    * ``"buffer_ring"``: rasterised base geometry (urban extent or CMC).
    * ``"lcz_based"``: LCZ built classes (1-10 by default).

    Args:
        method: One of ``uhi.suhii.rural_definitions``.
        params: Parsed params mapping.

    Returns:
        Single-band 0/1 ``ee.Image`` named ``urban``, clipped to
        :func:`analysis_region`.
    """
    suhii = params["uhi"]["suhii"]
    method = resolve_rural_method(method, suhii["rural_definitions"])
    region = analysis_region(params)

    if method == "buffer_ring":
        mask = _paint(_buffer_ring_base(params), region)
    else:  # lcz_based
        urban_classes, _ = validate_lcz_classes(
            suhii["lcz_based"]["urban_classes"], suhii["lcz_based"]["rural_classes"]
        )
        mask = _lcz_class_mask(params, urban_classes).clip(region)
    return mask.rename("urban")


def rural_mask(method: str, params: dict[str, Any]) -> "ee.Image":
    """0/1 rural-reference mask for SUHII under the given method.

    * ``"buffer_ring"``: ring around the urban core, minus the exclusions in
      ``buffer_ring.exclude`` (water via :func:`water_exclusion_mask`,
      built-up via the LCZ built classes).
    * ``"lcz_based"``: LCZ natural classes (A-G = 11-17 by default) minus the
      water exclusion (which removes class G plus any shoreline buffer).

    Args:
        method: One of ``uhi.suhii.rural_definitions``.
        params: Parsed params mapping.

    Returns:
        Single-band 0/1 ``ee.Image`` named ``rural``, clipped to
        :func:`analysis_region`.
    """
    suhii = params["uhi"]["suhii"]
    method = resolve_rural_method(method, suhii["rural_definitions"])
    region = analysis_region(params)
    urban_classes, rural_classes = validate_lcz_classes(
        suhii["lcz_based"]["urban_classes"], suhii["lcz_based"]["rural_classes"]
    )
    not_water = water_exclusion_mask(params).Not()

    if method == "buffer_ring":
        mask = _paint(buffer_ring(params), region)
        exclude = suhii["buffer_ring"]["exclude"]
        if "water" in exclude:
            mask = mask.And(not_water)
        if "built_up" in exclude:
            mask = mask.And(_lcz_class_mask(params, urban_classes).Not())
    else:  # lcz_based
        mask = _lcz_class_mask(params, rural_classes).clip(region).And(not_water)
    return mask.rename("rural")


def rural_reference(
    method: str, params: dict[str, Any]
) -> tuple["ee.Image", "ee.Image"]:
    """Urban and rural 0/1 masks for SUHII under one method — the common interface.

    Run the whole SUHII pipeline once per method in
    ``uhi.suhii.rural_definitions`` and report the sensitivity (CLAUDE.md
    caveat 5): ``rural_reference("buffer_ring", p)`` and
    ``rural_reference("lcz_based", p)``.

    Args:
        method: ``"buffer_ring"`` or ``"lcz_based"``.
        params: Parsed params mapping.

    Returns:
        ``(urban_mask, rural_mask)`` pair of 0/1 ``ee.Image``s.
    """
    return urban_mask(method, params), rural_mask(method, params)


# =============================================================================
# Convenience
# =============================================================================
def area_km2(geometry: "ee.Geometry") -> "ee.Number":
    """Geodesic area of a geometry in km2 (server-side; call .getInfo() to print).

    Args:
        geometry: Any ``ee.Geometry``.

    Returns:
        ``ee.Number`` area in km2.
    """
    return geometry.area(_MAX_ERROR_M).divide(1e6)
