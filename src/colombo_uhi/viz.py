"""Visualisation helpers: static PNG thumbnails now, report figures in Phase 8.

Phase 1b adds :func:`save_thumbnail` because an interactive ``geemap.Map``
renders nothing once a notebook is saved — visual verification needs an artefact
that survives in ``figures/`` and can be shared for review. Phase 8 extends this
module with matplotlib report figures, each stamped with the applicable
``caveats`` strings from ``config/params.yaml``.

Design notes:
    * ``import ee`` is deferred into function bodies so this module (and the
      local pytest suite) imports cleanly without ``earthengine-api``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, never at runtime
    import ee

#: Default long edge (px) for exported thumbnails.
DEFAULT_THUMBNAIL_PX = 900


def outline_image(
    features: "ee.FeatureCollection | ee.Geometry", color: str, width: int = 2
) -> "ee.Image":
    """Render a boundary as a transparent-filled coloured outline.

    Args:
        features: Geometry or FeatureCollection to outline.
        color: Hex colour without ``#``, e.g. ``"d62728"``.
        width: Stroke width in pixels.

    Returns:
        An RGBA ``ee.Image`` suitable for ``ee.ImageCollection.mosaic``.
    """
    import ee  # Deferred: see module docstring.

    collection = features
    if isinstance(features, ee.Geometry):
        collection = ee.FeatureCollection([ee.Feature(features)])
    return collection.style(color=color, fillColor="00000000", width=width)


def elevation_backdrop(
    params: dict[str, Any], max_elevation_m: int = 300
) -> "ee.Image":
    """Greyscale SRTM backdrop giving terrain context to mask overlays.

    Elevation (rather than a flat colour) is deliberate: it shows at a glance
    the inland relief that ``uhi.suhii.rural_filters.max_elevation_m`` guards
    against.

    Args:
        params: Parsed params mapping.
        max_elevation_m: Upper end of the greyscale stretch.

    Returns:
        Visualised RGB ``ee.Image``.
    """
    import ee  # Deferred: see module docstring.

    srtm_cfg = params["datasets"]["srtm"]
    return (
        ee.Image(srtm_cfg["id"])
        .select(srtm_cfg["band"])
        .unmask(0)
        .visualize(min=0, max=max_elevation_m, palette=["f7f7f7", "969696", "252525"])
    )


def save_thumbnail(
    layers: Sequence["ee.Image"],
    region: "ee.Geometry",
    out_path: str | Path,
    dimensions: int = DEFAULT_THUMBNAIL_PX,
) -> Path:
    """Mosaic visualised layers into a PNG on disk.

    Args:
        layers: Visualised (RGB/RGBA) images, drawn back-to-front — index 0 is
            the bottom layer.
        region: Geometry to render.
        out_path: Destination ``.png`` path; parent directories are created.
        dimensions: Long-edge size in pixels.

    Returns:
        The path written.

    Raises:
        RuntimeError: If Earth Engine's thumbnail endpoint returns an error.
    """
    import ee  # Deferred: see module docstring.
    import requests

    mosaic = ee.ImageCollection(list(layers)).mosaic()
    url = mosaic.getThumbURL(
        {"region": region, "dimensions": dimensions, "format": "png"}
    )
    response = requests.get(url, timeout=300)
    if response.status_code != 200:
        raise RuntimeError(
            f"Earth Engine thumbnail request failed ({response.status_code}): "
            f"{response.text[:500]}"
        )

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return destination
