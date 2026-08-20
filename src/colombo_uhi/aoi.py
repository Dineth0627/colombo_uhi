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

#: Cache of {asset_id: [property names]} so schema resolution costs one
#: round-trip per asset per session (see :func:`_asset_property_names`).
_PROPERTY_CACHE: dict[str, list[str]] = {}


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


def validate_property_candidates(candidates: Sequence[str], label: str) -> list[str]:
    """Validate a list of candidate attribute names for an uploaded asset.

    Args:
        candidates: Candidate property names, tried in order.
        label: Human-readable name of what is being resolved (for errors).

    Returns:
        The candidates as a list of strings.

    Raises:
        ValueError: If the list is empty, holds a non-string or blank entry,
            or repeats a name.
    """
    if not candidates:
        raise ValueError(f"{label}: candidate property list must not be empty")
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError(
                f"{label}: candidate property names must be non-empty strings, "
                f"got {candidate!r}"
            )
    duplicates = {c for c in candidates if list(candidates).count(c) > 1}
    if duplicates:
        raise ValueError(f"{label}: duplicate candidate properties {sorted(duplicates)}")
    return list(candidates)


# =============================================================================
# Uploaded-asset schema resolution (the only client-side calls in this module)
# =============================================================================
# The DS/GN assets are user-uploaded, so their attribute schema is not knowable
# in advance: OCHA COD-AB uses ADM3_EN/ADM2_EN, geoBoundaries uses shapeName and
# carries no parent-district column at all. Guessing wrong once already cost a
# Colab round trip and produced a silent 0 km2 CMC (2026-08-08), so these
# helpers read the asset's real schema and fail loudly with it. Each does ONE
# getInfo per asset per session (cached) — never inside a loop.
def _asset_property_names(asset_id: str) -> list[str]:
    """Return (and cache) the property names of an uploaded FeatureCollection."""
    import ee  # Deferred: see module docstring.

    if asset_id not in _PROPERTY_CACHE:
        first = ee.FeatureCollection(asset_id).first()
        _PROPERTY_CACHE[asset_id] = list(first.propertyNames().getInfo())
    return _PROPERTY_CACHE[asset_id]


def _resolve_property(
    asset_id: str,
    candidates: Sequence[str],
    label: str,
    override: str | None = None,
    required: bool = True,
) -> str | None:
    """Resolve an attribute name on an uploaded asset from candidates.

    Args:
        asset_id: Earth Engine table asset id.
        candidates: Property names to try, in order.
        label: What is being resolved, e.g. ``"DS-division name"``.
        override: Explicit property name from params; used verbatim if set.
        required: When True, raise if nothing matches; when False, return None
            (used for the optional parent-district column).

    Returns:
        The resolved property name, or None when ``required`` is False and no
        candidate is present.

    Raises:
        ValueError: If no candidate matches and ``required`` is True. The
            message lists the asset's ACTUAL property names so the fix is a
            one-line params edit rather than another guess.
    """
    validate_property_candidates(candidates, label)
    available = _asset_property_names(asset_id)

    if override:
        if override not in available:
            raise ValueError(
                f"{label}: property '{override}' (set in config/params.yaml) is "
                f"not in asset '{asset_id}'. Available properties: {available}"
            )
        return override

    for candidate in candidates:
        if candidate in available:
            return candidate

    if not required:
        return None
    raise ValueError(
        f"{label}: none of the candidate properties {list(candidates)} exist in "
        f"asset '{asset_id}'. Available properties: {available}\n"
        "Fix: add the correct name to the matching *_candidates list under "
        "aoi.assets in config/params.yaml (or set the explicit override)."
    )


def describe_asset(asset_id: str, n_samples: int = 3) -> dict[str, Any]:
    """Diagnostic: report an uploaded asset's size, schema and sample values.

    Call this from a notebook when a filter unexpectedly matches nothing — it
    shows the attribute names and example values to filter on. Unlike the rest
    of this module it is *meant* to evaluate client-side.

    Args:
        asset_id: Earth Engine table asset id.
        n_samples: Number of features to sample property values from.

    Returns:
        Dict with ``asset_id``, ``count``, ``properties`` and ``samples``
        (a list of per-feature property dicts).
    """
    import ee  # Deferred: see module docstring.

    fc = ee.FeatureCollection(asset_id)
    properties = _asset_property_names(asset_id)
    samples = (
        fc.limit(n_samples)
        .toList(n_samples)
        .map(lambda f: ee.Feature(f).toDictionary())
        .getInfo()
    )
    return {
        "asset_id": asset_id,
        "count": fc.size().getInfo(),
        "properties": properties,
        "samples": samples,
    }


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


def _filter_to_district(
    fc: "ee.FeatureCollection", asset_id: str, params: dict[str, Any]
) -> "ee.FeatureCollection":
    """Restrict a nationwide boundary asset to Colombo District.

    Prefers an attribute filter on the asset's parent-district column. The
    geoBoundaries products carry no such column, so the fallback is spatial:
    keep features whose CENTROID falls inside the GAUL district (plain
    ``filterBounds`` would also pull in neighbouring districts' units that
    merely touch the 500 m-simplified GAUL edge).
    """
    import ee  # Deferred: see module docstring.

    assets_cfg = params["aoi"]["assets"]
    district_property = _resolve_property(
        asset_id,
        assets_cfg["district_property_candidates"],
        "parent-district",
        required=False,
    )
    if district_property is not None:
        return fc.filter(
            ee.Filter.eq(district_property, assets_cfg["district_value"])
        )

    warnings.warn(
        f"asset '{asset_id}' has no parent-district column "
        f"(tried {assets_cfg['district_property_candidates']}); falling back to "
        "a centroid-within-GAUL-district spatial filter. Counts may differ by a "
        "unit or two along the district edge - verify against the expected "
        "13 DS / 557 GN.",
        stacklevel=3,
    )
    district_geom = colombo_district(params).geometry(_MAX_ERROR_M)

    def tag_centroid(feature: "ee.Feature") -> "ee.Feature":
        centroid = feature.geometry().centroid(_MAX_ERROR_M)
        # ee.Geometry.contains() returns an ee.Boolean, which round-trips as JSON
        # `true`/`false` — ee.Filter.eq(prop, 1) does NOT match that (it silently
        # matched nothing in the 2026-08-08 Colab run). Cast to ee.Number so the
        # stored value is unambiguously 1/0. Do not "simplify" this back.
        return feature.set(
            "_centroid_in_district",
            ee.Number(district_geom.contains(centroid, _MAX_ERROR_M)),
        )

    return (
        fc.filterBounds(district_geom)          # cheap pre-filter
        .map(tag_centroid)
        .filter(ee.Filter.eq("_centroid_in_district", 1))
    )


def ds_divisions(
    params: dict[str, Any], district_only: bool = True
) -> "ee.FeatureCollection":
    """Divisional Secretariat (DS) divisions of Colombo District (13 expected).

    Loads the user-uploaded EE asset at ``aoi.assets.ds_divisions``. The
    uploaded layers cover all of Sri Lanka (339 DS features), so the result is
    filtered to Colombo District unless ``district_only`` is False. While the
    asset id is null, warns prominently and falls back to the GAUL district
    (a single coarse feature — unusable for sub-district statistics).

    Args:
        params: Parsed params mapping.
        district_only: Restrict to Colombo District (default) or return the
            whole uploaded layer.

    Returns:
        ``ee.FeatureCollection`` of DS divisions, or the 1-feature fallback.
    """
    import ee  # Deferred: see module docstring.

    asset_id = params["aoi"]["assets"]["ds_divisions"]
    if not asset_id:
        return _missing_asset_fallback("DS-division", params)
    fc = ee.FeatureCollection(asset_id)
    return _filter_to_district(fc, asset_id, params) if district_only else fc


def gn_divisions(
    params: dict[str, Any], district_only: bool = True
) -> "ee.FeatureCollection":
    """Grama Niladhari (GN) divisions of Colombo District (557 expected).

    Loads the user-uploaded EE asset at ``aoi.assets.gn_divisions`` (14 043
    features nationwide) and filters it to Colombo District unless
    ``district_only`` is False. While the asset id is null, falls back (with a
    prominent warning) to the DS-division asset if configured, else to GAUL.

    Args:
        params: Parsed params mapping.
        district_only: Restrict to Colombo District (default) or return the
            whole uploaded layer.

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
            return ds_divisions(params, district_only=district_only)
        return _missing_asset_fallback("GN-division", params)
    fc = ee.FeatureCollection(asset_id)
    return _filter_to_district(fc, asset_id, params) if district_only else fc


#: Valid values for ``aoi.cmc.definition``.
CMC_DEFINITIONS = ("gn_union", "ds_union")


def _cmc_source(params: dict[str, Any]) -> tuple[str, str, str, list[str], str]:
    """Resolve everything ``cmc_boundary``/``cmc_name_audit`` need.

    Returns:
        ``(definition, asset_key, name_property, configured_names, level)``
        where ``level`` is a human label ("GN"/"DS") for messages.

    Raises:
        RuntimeError: If the required asset id is null, with fix steps.
        ValueError: On an unknown ``definition``, or if the name attribute
            cannot be resolved.
    """
    cmc_cfg = params["aoi"]["cmc"]
    assets_cfg = params["aoi"]["assets"]
    definition = cmc_cfg["definition"]
    if definition not in CMC_DEFINITIONS:
        raise ValueError(
            f"unknown aoi.cmc.definition '{definition}'; valid options: "
            f"{list(CMC_DEFINITIONS)}"
        )

    if definition == "gn_union":
        asset_key, level = "gn_divisions", "GN"
        candidates = assets_cfg["gn_name_property_candidates"]
        override = cmc_cfg["gn_name_property"]
        names = cmc_cfg["gn_division_names"]
    else:  # ds_union
        asset_key, level = "ds_divisions", "DS"
        candidates = assets_cfg["ds_name_property_candidates"]
        override = cmc_cfg["ds_name_property"]
        names = cmc_cfg["ds_division_names"]

    asset_id = assets_cfg[asset_key]
    if not asset_id:
        raise RuntimeError(
            f"Cannot build the CMC boundary: aoi.assets.{asset_key} is null in "
            "config/params.yaml and no GEE dataset contains CMC/DS/GN boundaries "
            "(GAUL stops at district level for Sri Lanka). Fix:\n"
            "  1. Download 'Sri Lanka - Subnational Administrative Boundaries' "
            "from OCHA/HDX (admin3 = DS divisions, admin4 = GN divisions).\n"
            "  2. Upload the shapefile as an EE table asset "
            "(Code Editor > Assets > New > Shape files).\n"
            f"  3. Set aoi.assets.{asset_key} to the asset id in "
            "config/params.yaml.\n"
            f"CMC will then be the union of {len(names)} {level} divisions."
        )

    name_property = _resolve_property(
        asset_id, candidates, f"{level}-division name", override=override
    )
    return definition, asset_id, name_property, list(names), level


def _cmc_units(params: dict[str, Any]) -> "ee.FeatureCollection":
    """The candidate divisions the CMC is dissolved from, before name matching.

    For ``gn_union`` this is scoped to the CMC's parent DS divisions when
    ``aoi.cmc.parent_ds_scope`` is set, because **GN names are not unique within
    Colombo District**: matching names across all 557 divisions also picked up
    same-named divisions in Dehiwala/Moratuwa/Kolonnawa, which inflated the CMC
    to 47.50 km2 for only 50 of 55 units — larger than both parent DS divisions
    combined (Colab run 4, 2026-08-08). Do not remove this scoping.
    """
    import ee  # Deferred: see module docstring.

    definition, asset_id, _, _, _ = _cmc_source(params)
    if definition != "gn_union":
        return ds_divisions(params, district_only=True)

    units = gn_divisions(params, district_only=True)
    cmc_cfg = params["aoi"]["cmc"]
    if not cmc_cfg.get("parent_ds_scope", True):
        return units

    ds_property = _resolve_property(
        asset_id,
        params["aoi"]["assets"]["ds_name_property_candidates"],
        "DS-division name (on the GN asset)",
    )
    return units.filter(
        ee.Filter.inList(ds_property, cmc_cfg["ds_division_names"])
    )


def cmc_name_audit(params: dict[str, Any]) -> dict[str, Any]:
    """Diagnostic: check the configured CMC division names against the asset.

    A *partial* name match is the dangerous case — it yields a plausible-looking
    but undersized CMC. This reports both directions so spelling drift between
    the CMC GIS Unit map and COD-AB (e.g. Wellawatta/Wellawatte) is visible
    rather than silently absorbed. Like :func:`describe_asset`, evaluating
    client-side is the point of this function.

    Args:
        params: Parsed params mapping.

    Returns:
        Dict with ``definition``, ``name_property``, ``expected`` (configured
        count), ``matched`` / ``missing`` (configured names present / absent in
        the candidate units), ``extra`` (candidate division names NOT on the
        configured list — where a misspelling surfaces, since the candidates are
        scoped to the CMC's parent DS divisions), ``matched_features`` (feature
        count; a value above ``expected`` means duplicate names slipped in) and
        ``stated_area_km2`` (sum of the asset's own area field over the matches).
    """
    import ee  # Deferred: see module docstring.

    definition, _, name_property, configured, _ = _cmc_source(params)
    assets_cfg = params["aoi"]["assets"]

    units = _cmc_units(params)   # already scoped to the parent DS divisions
    matched_fc = units.filter(ee.Filter.inList(name_property, configured))
    matched_names = matched_fc.aggregate_array(name_property).getInfo()
    stated = matched_fc.aggregate_sum(assets_cfg["area_property"]).getInfo()
    available = set(units.aggregate_array(name_property).getInfo())

    return {
        "definition": definition,
        "name_property": name_property,
        "expected": len(configured),
        "matched": sorted(set(matched_names)),
        "matched_features": len(matched_names),
        "missing": sorted(set(configured) - set(matched_names)),
        "extra": sorted(available - set(configured)),
        "stated_area_km2": stated,
    }


def cmc_boundary(params: dict[str, Any]) -> "ee.Geometry":
    """Colombo Municipal Council (~37 km2) boundary.

    CMC exists in no GEE dataset. It is dissolved from the user-uploaded admin
    assets according to ``aoi.cmc.definition``:

    * ``"gn_union"`` (default, authoritative) — the 55 GN divisions the Colombo
      Municipal Council's own GIS Unit lists as inside the CMC.
    * ``"ds_union"`` — Colombo + Thimbirigasyaya DS divisions. Measures
      46.87 km2 with COD-AB polygons because its Colombo DS polygon encloses the
      Port's outer harbour; kept as a sensitivity variant only.

    The name attribute is auto-resolved from the matching ``*_candidates`` list.
    Notebook 01 checks the area against ``aoi.expected_areas_km2.cmc``.

    Neither an empty nor a partial name match is allowed to pass silently: zero
    matches raise, and a partial match warns with the missing names. The earlier
    silent 0 km2 CMC (Colab run 2026-08-08) was indistinguishable from a real
    answer, and an undersized one would be worse.

    Args:
        params: Parsed params mapping.

    Returns:
        Dissolved ``ee.Geometry`` of the CMC.

    Raises:
        RuntimeError: If the required asset id is null (with fix steps), or when
            no division matches the configured names (listing what is present).
        ValueError: On an unknown definition or an unresolvable name attribute.
    """
    import ee  # Deferred: see module docstring.

    definition, asset_id, name_property, configured, level = _cmc_source(params)
    cmc_cfg = params["aoi"]["cmc"]
    units = _cmc_units(params)
    matched = units.filter(ee.Filter.inList(name_property, configured))

    # One client-side check; see the docstring for why it is worth the round trip.
    matched_names = matched.aggregate_array(name_property).getInfo()
    if not matched_names:
        available = sorted(units.aggregate_array(name_property).getInfo())
        raise RuntimeError(
            f"No {level} division matched aoi.cmc "
            f"({len(configured)} configured names) on property "
            f"'{name_property}' in asset '{asset_id}' (within Colombo District).\n"
            f"{level} names actually present: {available}\n"
            f"Fix: correct the name list under aoi.cmc in config/params.yaml."
        )

    missing = sorted(set(configured) - set(matched_names))
    if missing:
        warnings.warn(
            f"CMC is being built from only {len(matched_names)}/{len(configured)} "
            f"{level} divisions - these configured names are NOT in the asset and "
            f"the CMC is therefore UNDERSIZED: {missing}. Run "
            "aoi.cmc_name_audit(params) to see the asset's actual spellings, then "
            "fix aoi.cmc in config/params.yaml.",
            stacklevel=2,
        )

    expected_n = cmc_cfg.get("expected_gn_count") if definition == "gn_union" else None
    if expected_n is not None:
        if len(configured) != expected_n:
            warnings.warn(
                f"aoi.cmc.gn_division_names holds {len(configured)} names but "
                f"expected_gn_count is {expected_n} - update whichever is stale.",
                stacklevel=2,
            )
        # More features than names = duplicate names matched (GN names are not
        # unique within Colombo District; see _cmc_units). That silently enlarges
        # the CMC with polygons from other DS divisions.
        if len(matched_names) > expected_n:
            warnings.warn(
                f"CMC matched {len(matched_names)} features for {expected_n} "
                "expected divisions - duplicate names have been included, so the "
                "CMC is TOO LARGE and scattered. Check that "
                "aoi.cmc.parent_ds_scope is true in config/params.yaml.",
                stacklevel=2,
            )
    return matched.union(_MAX_ERROR_M).geometry(_MAX_ERROR_M)


def cmc_land_area_km2(
    params: dict[str, Any], scale_m: int | None = None
) -> "ee.Number":
    """CMC area in km2 with water excluded — the land-only figure.

    The administrative polygon measures ~47 km2 against a gazetted CMC of
    37.31 km2 because COD-AB's Colombo DS polygon encloses the Colombo Port
    outer harbour (6.89 km2 of water sits inside it, measured in Colab run 5).
    Masking water yields the land area that LST statistics actually cover
    (CLAUDE.md requires water masked before any LST statistic).

    **The result is scale-dependent and must be quoted with its scale**: run 5
    gave 40.18 km2 at the 30 m analysis grid and 37.70 km2 at 300 m — a 6.6%
    spread from raster discretisation of a ragged coastal mask, not from a
    different definition. Quote the 30 m figure by default. The residual over
    37.31 km2 comes from COD-AB polygon generalisation plus the water-mask
    thresholds in ``aoi.water_mask``; it is a sensitivity to report, not an error
    to tune away.

    Implemented by rasterising the CMC rather than differencing geometries: the
    water mask is a raster, and a vectorised difference would be both expensive
    and needlessly ragged.

    Args:
        params: Parsed params mapping.
        scale_m: Reduction scale; defaults to ``crs.analysis_scale_m`` (30 m).

    Returns:
        ``ee.Number`` land area in km2 (call ``.getInfo()`` to print).
    """
    cmc_mask = _paint(cmc_boundary(params), analysis_region(params)).And(
        water_exclusion_mask(params).Not()
    )
    return mask_area_km2(cmc_mask, params, scale_m=scale_m)


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
    """Calendar filter matching any of the given months (handles Dec-Feb wrap).

    Delegates to :func:`colombo_uhi.landsat.month_filter`. The implementation
    lives in ``landsat`` because Phase 2 needs it too and this module already
    depends on ``landsat`` — putting the shared helper the other way round
    would make the imports circular.
    """
    return landsat.month_filter(months)


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


def water_mask(
    params: dict[str, Any], region: "ee.Geometry | None" = None
) -> "ee.Image":
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
        region: Geometry to build and clip the mask over; defaults to
            :func:`analysis_region` (the Phase 1 behaviour, which the verified
            Phase 1 areas depend on — do not change that default). Pass a
            smaller geometry, e.g. the district, when the mask is only needed
            there: the source reflectance composite is built over this region,
            so narrowing it is the single biggest saving available when Earth
            Engine reports "User memory limit exceeded".

    Returns:
        Single-band 0/1 ``ee.Image`` named ``water``, clipped to ``region``.
    """
    import ee  # Deferred: see module docstring.

    cfg = validate_water_mask_params(params)
    if region is None:
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
    params: dict[str, Any],
    shoreline_buffer_m: float | None = None,
    region: "ee.Geometry | None" = None,
) -> "ee.Image":
    """Water mask dilated by an optional shoreline buffer (1 = exclude).

    Coastal/lakeside Landsat pixels mix land and water signals; dilating the
    water mask by 1-2 pixel widths (e.g. 60 m) removes that contaminated
    fringe from any land-only statistic.

    Args:
        params: Parsed params mapping.
        shoreline_buffer_m: Dilation radius in metres; ``None`` uses
            ``aoi.water_mask.shoreline_buffer_m`` (default 0 = plain water mask).
        region: Forwarded to :func:`water_mask`.

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
    mask = water_mask(params, region=region)
    if buffer_m > 0:
        mask = mask.focalMax(radius=buffer_m, kernelType="circle", units="meters")
    return mask.rename("water_excluded")


def static_water_mask(
    params: dict[str, Any], region: "ee.Geometry | None" = None
) -> "ee.Image":
    """Cheap 0/1 water mask from JRC Global Surface Water alone (1 = water).

    :func:`water_mask` ORs three detectors, two of which require compositing a
    Landsat reflectance collection. That composite then gets embedded into
    *every* image it masks, so applying it across a 26-year annual series
    multiplies a heavy graph 26 times and exceeds the Earth Engine user memory
    limit (Colab run 3, 2026-08-08). This variant is a single static image:
    essentially free, and safe to apply to a long series.

    What it does and does not capture, so the trade is explicit:

    * JRC ``occurrence`` is itself derived from 38 years of Landsat, so for
      permanent INLAND water — Beira, Bolgoda and Diyawanna lakes, the Kelani —
      it agrees closely with the combined mask.
    * *** IT DOES NOT MAP THE OCEAN, AND IT DOES NOT MAP THE COLOMBO PORT OUTER
      HARBOUR. *** [MEASURED — Phase 7, Colab run 5] An earlier version of this
      docstring claimed it was authoritative for both. It is not. Used as a land
      mask over the coastal divisions it left the harbour counting as land:
      Fort's land-coverage fraction came out at 0.767 against 0.705 on the raw
      polygon, when removing the harbour from a 7.46 km2 polygon holding 6.89 km2
      of it should have put Fort near 1.0. Across all 557 GN divisions only NINE
      differed from the polygon-based fraction by more than 0.01, and the 8.3 km2
      it did find over Colombo District is inland water.
      **For anything coastal, use :func:`water_mask` — its MNDWI detector sees
      the sea.**
    * It will miss seasonal, shallow or newly-impounded water that the MNDWI and
      QA_PIXEL detectors catch, and it carries no shoreline dilation.

    Use :func:`water_mask` for maps and for any single-date product. Use this one
    for long series, and report the difference rather than assuming it is nil —
    notebook 02 measures it on one year.

    Args:
        params: Parsed params mapping.
        region: Geometry to clip to; defaults to :func:`analysis_region`.

    Returns:
        Single-band 0/1 ``ee.Image`` named ``water``.
    """
    import ee  # Deferred: see module docstring.

    if region is None:
        region = analysis_region(params)
    gsw_cfg = params["datasets"]["surface_water"]
    threshold = params["aoi"]["water_mask"]["jrc_occurrence_threshold_pct"]
    return (
        ee.Image(gsw_cfg["id"])
        .select(gsw_cfg["band_occurrence"])
        .gte(threshold)
        .unmask(0)
        .rename("water")
        .clip(region)
    )


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


def _lcz_scope_geometry(params: dict[str, Any]) -> "ee.Geometry":
    """Resolve the geometry both LCZ masks are clipped to.

    Set by ``uhi.suhii.lcz_based.scope``. Without a scope the LCZ masks span the
    whole analysis region, which made "urban" 2464 km2 of built-up LCZ across
    Gampaha and Kalutara — a regional statistic rather than Colombo's.

    Note the trade-off of the district scope: the rural reference then sits much
    closer to the urban core than the buffer method's 15-25 km ring, so
    advection may damp its SUHII. That divergence between the two definitions is
    precisely the sensitivity CLAUDE.md requires to be reported, not a defect.

    Raises:
        ValueError: On an unknown scope value, listing the valid options.
    """
    scope = params["uhi"]["suhii"]["lcz_based"]["scope"]
    if scope == "district":
        return colombo_district(params).geometry(_MAX_ERROR_M)
    if scope == "cmc":
        return cmc_boundary(params)
    if scope == "analysis_region":
        return analysis_region(params)
    raise ValueError(
        f"unknown lcz_based scope '{scope}'; valid options: "
        "['district', 'cmc', 'analysis_region']"
    )


#: Public alias for :func:`_lcz_scope_geometry`. Phase 3 sizes its SUHII
#: reduction region from the LCZ scope, and reaching into a private for that
#: would either couple the modules through an underscore name or duplicate the
#: scope dispatch — both worse than exporting the one function.
lcz_scope_geometry = _lcz_scope_geometry


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


def _resolve_water(
    params: dict[str, Any], water: "ee.Image | None"
) -> "ee.Image":
    """Resolve the water layer the SUHII masks exclude.

    Args:
        params: Parsed params mapping.
        water: Caller-supplied 0/1 water image, or ``None`` for the default
            :func:`water_exclusion_mask`.

    Returns:
        A 0/1 ``ee.Image`` where 1 means "exclude as water".
    """
    return water_exclusion_mask(params) if water is None else water


def urban_mask(
    method: str, params: dict[str, Any], water: "ee.Image | None" = None
) -> "ee.Image":
    """0/1 urban mask for SUHII under the given rural-reference method.

    * ``"buffer_ring"``: rasterised base geometry (urban extent or CMC), over
      :func:`analysis_region`.
    * ``"lcz_based"``: LCZ built classes (1-10 by default), clipped to
      :func:`_lcz_scope_geometry` (Colombo District by default).

    Water is excluded under both methods (CLAUDE.md: water bodies must be masked
    before any LST statistic). This matters concretely here: the administrative
    CMC encloses the Colombo Port outer harbour, and open water would drag the
    urban LST mean down and inflate SUHII. Unlike :func:`rural_mask` there is no
    elevation cap — the urban core is not elevation-matched by construction.

    Args:
        method: One of ``uhi.suhii.rural_definitions``.
        params: Parsed params mapping.
        water: Optional 0/1 water image (1 = exclude) replacing the default
            :func:`water_exclusion_mask`. **Pass
            :func:`static_water_mask` for anything that masks a multi-year
            series.** The default composites Landsat internally and that
            composite is instantiated once per image masked, so a 26-year
            SUHII run over six sources would multiply the heavy graph past the
            Earth Engine memory limit — the Colab run 3 failure. Notebook 02
            measured the cost of the substitution at **-0.074 degC** on the CMC
            2025 mean; quote that figure rather than re-deriving it. Leave
            ``None`` for single-date maps, where the combined mask is better.

    Returns:
        Single-band 0/1 ``ee.Image`` named ``urban``.
    """
    suhii = params["uhi"]["suhii"]
    method = resolve_rural_method(method, suhii["rural_definitions"])

    if method == "buffer_ring":
        mask = _paint(_buffer_ring_base(params), analysis_region(params))
    else:  # lcz_based
        urban_classes, _ = validate_lcz_classes(
            suhii["lcz_based"]["urban_classes"], suhii["lcz_based"]["rural_classes"]
        )
        mask = _lcz_class_mask(params, urban_classes).clip(_lcz_scope_geometry(params))
    return mask.And(_resolve_water(params, water).Not()).rename("urban")


def _elevation_cap_mask(params: dict[str, Any]) -> "ee.Image | None":
    """0/1 mask of pixels at or below ``rural_filters.max_elevation_m``.

    Colombo sits on the coastal lowland, but the 15-25 km rural ring reaches
    inland relief (~50-150 m). At a ~6.5 degC/km environmental lapse rate that
    is up to ~0.65 degC of elevation-driven cooling — a large fraction of a
    typical SUHII — so the rural reference is capped to comparable elevations.

    Returns:
        The 0/1 ``ee.Image``, or None when the cap is disabled (null in params).
    """
    import ee  # Deferred: see module docstring.

    max_elev = params["uhi"]["suhii"].get("rural_filters", {}).get("max_elevation_m")
    if max_elev is None:
        return None
    srtm_cfg = params["datasets"]["srtm"]
    return ee.Image(srtm_cfg["id"]).select(srtm_cfg["band"]).lte(max_elev)


def rural_mask(
    method: str, params: dict[str, Any], water: "ee.Image | None" = None
) -> "ee.Image":
    """0/1 rural-reference mask for SUHII under the given method.

    * ``"buffer_ring"``: ring around the urban core over
      :func:`analysis_region`, minus the exclusions in ``buffer_ring.exclude``
      (water via :func:`water_exclusion_mask`, built-up via the LCZ built
      classes).
    * ``"lcz_based"``: LCZ natural classes (A-G = 11-17 by default) within
      :func:`_lcz_scope_geometry` (Colombo District by default), minus the water
      exclusion (which removes class G plus any shoreline buffer).

    Both definitions then get the SRTM elevation cap from
    ``uhi.suhii.rural_filters.max_elevation_m`` (see
    :func:`_elevation_cap_mask`). Urban masks are deliberately NOT capped.

    Args:
        method: One of ``uhi.suhii.rural_definitions``.
        params: Parsed params mapping.
        water: Optional 0/1 water image (1 = exclude) replacing the default
            :func:`water_exclusion_mask`. See :func:`urban_mask` for why a
            multi-year series must pass :func:`static_water_mask` here, and
            pass the SAME image to both masks so the pair stays consistent.

    Returns:
        Single-band 0/1 ``ee.Image`` named ``rural``.
    """
    suhii = params["uhi"]["suhii"]
    method = resolve_rural_method(method, suhii["rural_definitions"])
    urban_classes, rural_classes = validate_lcz_classes(
        suhii["lcz_based"]["urban_classes"], suhii["lcz_based"]["rural_classes"]
    )
    not_water = _resolve_water(params, water).Not()

    if method == "buffer_ring":
        mask = _paint(buffer_ring(params), analysis_region(params))
        exclude = suhii["buffer_ring"]["exclude"]
        if "water" in exclude:
            mask = mask.And(not_water)
        if "built_up" in exclude:
            mask = mask.And(_lcz_class_mask(params, urban_classes).Not())
    else:  # lcz_based
        mask = (
            _lcz_class_mask(params, rural_classes)
            .clip(_lcz_scope_geometry(params))
            .And(not_water)
        )

    elevation_ok = _elevation_cap_mask(params)
    if elevation_ok is not None:
        mask = mask.And(elevation_ok)
    return mask.rename("rural")


def rural_reference(
    method: str, params: dict[str, Any], water: "ee.Image | None" = None
) -> tuple["ee.Image", "ee.Image"]:
    """Urban and rural 0/1 masks for SUHII under one method — the common interface.

    Run the whole SUHII pipeline once per method in
    ``uhi.suhii.rural_definitions`` and report the sensitivity (CLAUDE.md
    caveat 5): ``rural_reference("buffer_ring", p)`` and
    ``rural_reference("lcz_based", p)``.

    Args:
        method: ``"buffer_ring"`` or ``"lcz_based"``.
        params: Parsed params mapping.
        water: Optional 0/1 water image applied to BOTH masks; see
            :func:`urban_mask`. Phase 3 passes
            ``static_water_mask(params, region=...)`` here, because the default
            embeds a Landsat composite into every masked image.

    Returns:
        ``(urban_mask, rural_mask)`` pair of 0/1 ``ee.Image``s.
    """
    return urban_mask(method, params, water), rural_mask(method, params, water)


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


def mask_area_km2(
    mask: "ee.Image", params: dict[str, Any], scale_m: int | None = None
) -> "ee.Number":
    """Area covered by a 0/1 mask, in km2 (server-side).

    Use it to check that a mask is not accidentally empty — e.g. that the
    elevation cap has not eliminated the rural reference.

    Args:
        mask: 0/1 ``ee.Image``.
        params: Parsed params mapping (analysis scale and region).
        scale_m: Reduction scale; defaults to ``crs.analysis_scale_m``. Pass a
            coarser value for a faster approximate check.

    Returns:
        ``ee.Number`` area in km2.
    """
    import ee  # Deferred: see module docstring.

    scale = scale_m or params["crs"]["analysis_scale_m"]
    total = (
        ee.Image.pixelArea()
        .updateMask(mask)
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=analysis_region(params),
            scale=scale,
            maxPixels=1e10,
            bestEffort=True,
        )
        .get("area")
    )
    return ee.Number(total).divide(1e6)
