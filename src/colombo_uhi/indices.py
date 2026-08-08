"""Spectral indices: NDVI, NDBI, MNDWI, EVI, SAVI and shortwave albedo.

Every function here operates on the HARMONISED band names produced by
:func:`colombo_uhi.landsat.harmonised_collection` (``blue``, ``green``, ``red``,
``nir``, ``swir1``, ``swir2``), never on ``SR_B*``. That rename is what makes
this module sensor-agnostic: TM/ETM+ and OLI number their bands differently, but
by the time an image reaches these functions the difference is gone.

Albedo deserves a note, because the obvious intuition is backwards. Liang (2001)
fitted his coefficients for TM/ETM+ and predates OLI by twelve years; the
universally-cited Landsat 8 form is the SAME coefficients applied to the
wavelength-matched OLI bands — which our rename already performs. Substituting a
genuinely different published fit (e.g. Silva 2016) at the 2013 sensor change
would INJECT a step into the 2000-2025 series exactly where the Mann-Kendall
trend is least able to absorb it. So one coefficient set is used throughout, and
the alternative ships in params as a labelled sensitivity variant.

Design notes:
    * ``import ee`` is deferred into function bodies so this module (and the
      local pytest suite) imports cleanly without ``earthengine-api``.
    * Every coefficient comes from ``config/params.yaml`` (``indices`` section).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, never at runtime
    import ee

#: Output band names, one per index. Keys are the names accepted by
#: :func:`add_indices` and :func:`resolve_indices`.
INDEX_BAND_NAMES: dict[str, str] = {
    "ndvi": "NDVI",
    "ndbi": "NDBI",
    "mndwi": "MNDWI",
    "evi": "EVI",
    "savi": "SAVI",
    "albedo": "albedo",
}


# =============================================================================
# Pure helpers (no Earth Engine; unit-tested)
# =============================================================================
def resolve_indices(which: Sequence[str] | None) -> list[str]:
    """Validate a requested set of index names.

    Args:
        which: Index names, or ``None`` for all of them. Order is preserved,
            duplicates are collapsed.

    Returns:
        Validated list of index names.

    Raises:
        ValueError: If ``which`` is empty or names an unknown index.
    """
    if which is None:
        return list(INDEX_BAND_NAMES)
    requested = list(which)
    if not requested:
        raise ValueError(
            f"which must name at least one of {sorted(INDEX_BAND_NAMES)}, "
            "got an empty sequence"
        )
    unknown = [name for name in requested if name not in INDEX_BAND_NAMES]
    if unknown:
        raise ValueError(
            f"unknown index names {unknown}; available: {sorted(INDEX_BAND_NAMES)}"
        )
    seen: dict[str, None] = {}
    for name in requested:
        seen.setdefault(name, None)
    return list(seen)


def albedo_coefficients(
    params: dict[str, Any], set_name: str | None = None
) -> tuple[dict[str, float], float, float]:
    """Resolve and validate one albedo coefficient set.

    Args:
        params: Parsed params mapping.
        set_name: Key under ``indices.albedo.sets``; ``None`` uses
            ``indices.albedo.active_set``.

    Returns:
        ``(weights, constant, divisor)`` where ``weights`` maps harmonised band
        names to coefficients.

    Raises:
        KeyError: If the named set does not exist.
        ValueError: If the set references a band that is not a harmonised
            surface-reflectance band, or if the divisor is zero.
    """
    cfg = params["indices"]["albedo"]
    name = set_name or cfg["active_set"]
    if name not in cfg["sets"]:
        raise KeyError(
            f"albedo coefficient set '{name}' is not defined; "
            f"indices.albedo.sets has {sorted(cfg['sets'])}"
        )
    entry = cfg["sets"][name]
    weights = {band: float(w) for band, w in entry["weights"].items()}

    harmonised = set(params["landsat_c2l2"]["harmonised_sr_bands"])
    unknown = sorted(set(weights) - harmonised)
    if unknown:
        raise ValueError(
            f"albedo set '{name}' weights reference non-reflectance bands "
            f"{unknown}; harmonised bands are {sorted(harmonised)}"
        )

    divisor = float(entry.get("divisor", 1.0))
    if divisor == 0:
        raise ValueError(f"albedo set '{name}' has divisor 0")
    return weights, float(entry["constant"]), divisor


# =============================================================================
# Indices
# =============================================================================
def ndvi(image: "ee.Image", params: dict[str, Any]) -> "ee.Image":
    """Normalised Difference Vegetation Index, ``(nir - red) / (nir + red)``.

    Args:
        image: Image carrying the harmonised reflectance bands.
        params: Parsed params mapping (unused; kept for a uniform signature).

    Returns:
        Single-band ``ee.Image`` named ``NDVI``.
    """
    return image.normalizedDifference(["nir", "red"]).rename(
        INDEX_BAND_NAMES["ndvi"]
    )


def ndbi(image: "ee.Image", params: dict[str, Any]) -> "ee.Image":
    """Normalised Difference Built-up Index, ``(swir1 - nir) / (swir1 + nir)``.

    Args:
        image: Image carrying the harmonised reflectance bands.
        params: Parsed params mapping (unused; kept for a uniform signature).

    Returns:
        Single-band ``ee.Image`` named ``NDBI``.
    """
    return image.normalizedDifference(["swir1", "nir"]).rename(
        INDEX_BAND_NAMES["ndbi"]
    )


def mndwi(image: "ee.Image", params: dict[str, Any]) -> "ee.Image":
    """Modified Normalised Difference Water Index, ``(green - swir1) / (green + swir1)``.

    Xu (2006). Positive values indicate water; ``aoi.water_mask`` thresholds
    this at ``aoi.water_mask.mndwi_threshold``.

    Args:
        image: Image carrying the harmonised reflectance bands.
        params: Parsed params mapping (unused; kept for a uniform signature).

    Returns:
        Single-band ``ee.Image`` named ``MNDWI``.
    """
    return image.normalizedDifference(["green", "swir1"]).rename(
        INDEX_BAND_NAMES["mndwi"]
    )


def evi(image: "ee.Image", params: dict[str, Any]) -> "ee.Image":
    """Enhanced Vegetation Index (Huete et al. 2002).

    ``G * (nir - red) / (nir + C1 * red - C2 * blue + L)``, coefficients from
    ``indices.evi``. Unlike NDVI, EVI does not saturate over dense tropical
    canopy, which matters for Colombo's remaining green cover.

    Args:
        image: Image carrying the harmonised reflectance bands.
        params: Parsed params mapping (``indices.evi``).

    Returns:
        Single-band ``ee.Image`` named ``EVI``.
    """
    cfg = params["indices"]["evi"]
    return image.expression(
        "G * (nir - red) / (nir + C1 * red - C2 * blue + L)",
        {
            "nir": image.select("nir"),
            "red": image.select("red"),
            "blue": image.select("blue"),
            "G": float(cfg["gain"]),
            "C1": float(cfg["c1"]),
            "C2": float(cfg["c2"]),
            "L": float(cfg["soil_l"]),
        },
    ).rename(INDEX_BAND_NAMES["evi"])


def savi(image: "ee.Image", params: dict[str, Any]) -> "ee.Image":
    """Soil-Adjusted Vegetation Index (Huete 1988).

    ``((nir - red) / (nir + red + L)) * (1 + L)`` with ``L`` from
    ``indices.savi.soil_l``.

    Args:
        image: Image carrying the harmonised reflectance bands.
        params: Parsed params mapping (``indices.savi``).

    Returns:
        Single-band ``ee.Image`` named ``SAVI``.
    """
    soil_l = float(params["indices"]["savi"]["soil_l"])
    return image.expression(
        "((nir - red) / (nir + red + L)) * (1 + L)",
        {
            "nir": image.select("nir"),
            "red": image.select("red"),
            "L": soil_l,
        },
    ).rename(INDEX_BAND_NAMES["savi"])


def albedo(
    image: "ee.Image", params: dict[str, Any], set_name: str | None = None
) -> "ee.Image":
    """Broadband shortwave albedo as a weighted sum of reflectance bands.

    One coefficient set is applied to all four sensors — see the module
    docstring for why that preserves, rather than breaks, continuity across the
    2013 TM/ETM+ to OLI transition.

    The result is clamped to ``indices.albedo.clamp``: Collection 2 surface
    reflectance legitimately goes slightly negative over water and deep shadow,
    where a linear narrowband-to-broadband fit is undefined.

    Args:
        image: Image carrying the harmonised reflectance bands.
        params: Parsed params mapping (``indices.albedo``).
        set_name: Optional coefficient-set override for sensitivity runs, e.g.
            ``"silva2016_oli"``.

    Returns:
        Single-band ``ee.Image`` named ``albedo`` (dimensionless, 0-1).
    """
    weights, constant, divisor = albedo_coefficients(params, set_name)

    total: "ee.Image | None" = None
    for band, weight in weights.items():
        term = image.select(band).multiply(weight)
        total = term if total is None else total.add(term)
    assert total is not None  # albedo_coefficients rejects an empty weight map

    result = total.add(constant).divide(divisor).rename(INDEX_BAND_NAMES["albedo"])

    clamp = params["indices"]["albedo"].get("clamp")
    if clamp is not None:
        result = result.clamp(float(clamp[0]), float(clamp[1]))
    return result


#: Dispatch table for :func:`add_indices`. Defined after the functions it names.
INDEX_FUNCTIONS: dict[str, Callable[["ee.Image", dict[str, Any]], "ee.Image"]] = {
    "ndvi": ndvi,
    "ndbi": ndbi,
    "mndwi": mndwi,
    "evi": evi,
    "savi": savi,
    "albedo": albedo,
}


def add_indices(
    image: "ee.Image", params: dict[str, Any], which: Sequence[str] | None = None
) -> "ee.Image":
    """Append one or more index bands to an image.

    Args:
        image: Image carrying the harmonised reflectance bands (a scene from
            :func:`colombo_uhi.landsat.harmonised_collection`, or a composite
            whose bands have been renamed back to the harmonised names).
        params: Parsed params mapping.
        which: Index names to add; ``None`` adds all of
            :data:`INDEX_BAND_NAMES`.

    Returns:
        The input image with the requested index bands appended.

    Raises:
        ValueError: If ``which`` names an unknown index.
    """
    import ee  # Deferred: see module docstring.

    names = resolve_indices(which)
    bands = [INDEX_FUNCTIONS[name](image, params) for name in names]
    return image.addBands(ee.Image.cat(bands))
