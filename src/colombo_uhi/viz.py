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

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, never at runtime
    import ee
    import pandas as pd

#: Default long edge (px) for exported thumbnails.
DEFAULT_THUMBNAIL_PX = 900

#: Width (characters) the caveat footer is wrapped to on matplotlib figures.
CAVEAT_WRAP_CHARS = 110


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


def caveat_footer(params: dict[str, Any], keys: Sequence[str]) -> str:
    """Assemble a wrapped caveat string for stamping onto a figure.

    Args:
        params: Parsed params mapping (``caveats`` section).
        keys: Caveat keys to include, e.g. ``["lst_not_air_temp"]``.

    Returns:
        Newline-wrapped text, one bulleted caveat per entry.

    Raises:
        KeyError: If a key is not defined under ``caveats``.
    """
    lines: list[str] = []
    for key in keys:
        if key not in params["caveats"]:
            raise KeyError(
                f"unknown caveat '{key}'; params.caveats defines "
                f"{sorted(params['caveats'])}"
            )
        text = " ".join(params["caveats"][key].split())
        lines.extend(
            textwrap.wrap(
                f"- {text}",
                width=CAVEAT_WRAP_CHARS,
                subsequent_indent="  ",
            )
        )
    return "\n".join(lines)


def plot_annual_lst_comparison(
    series: Mapping[str, "pd.DataFrame"],
    out_path: str | Path,
    params: dict[str, Any],
    title: str = "Annual mean land surface temperature",
    value_column: str = "mean",
    count_column: str | None = "valid_pixels",
) -> Path:
    """Plot several annual LST series together and stamp the caveats on it.

    Built for the Landsat-vs-MODIS comparison, but it takes any number of
    labelled series from :func:`colombo_uhi.composites.zonal_annual_means`.

    The two sensors are NOT expected to agree in absolute terms - 30 m versus
    1 km, a single ~10:30 overpass versus an 8-day clear-sky average. Agreement
    in SHAPE is the meaningful result; the offset is a finding to report, not
    an error to tune away.

    Args:
        series: Mapping of label -> DataFrame with a year column plus
            ``value_column``.
        out_path: Destination ``.png`` path; parent directories are created.
        params: Parsed params mapping.
        title: Axes title.
        value_column: Column holding the annual mean.
        count_column: Optional column of per-row valid-pixel counts; rows where
            it is zero or missing are dropped, so an empty year cannot be drawn
            as a real value. Pass ``None`` to skip that check.

    Returns:
        The path written.

    Raises:
        ValueError: If ``series`` is empty or a frame lacks the needed columns.
    """
    # The object-oriented API, deliberately NOT pyplot: pyplot would attach this
    # figure to the global state machine, and forcing a backend would break
    # Colab's inline rendering for every later cell. The figure is written to
    # disk and displayed from there.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if not series:
        raise ValueError("series must contain at least one labelled DataFrame")

    year_column = params["composites"]["year_property"]
    figure = Figure(figsize=(10, 5.5))
    FigureCanvasAgg(figure)
    axes = figure.subplots()

    for label, frame in series.items():
        missing = [c for c in (year_column, value_column) if c not in frame.columns]
        if missing:
            raise ValueError(
                f"series '{label}' is missing column(s) {missing}; it has "
                f"{sorted(frame.columns)}"
            )
        plotted = frame.dropna(subset=[value_column])
        if count_column and count_column in plotted.columns:
            plotted = plotted[plotted[count_column].fillna(0) > 0]
        axes.plot(
            plotted[year_column],
            plotted[value_column],
            marker="o",
            markersize=4,
            linewidth=1.6,
            label=label,
        )

    axes.set_xlabel("Year")
    axes.set_ylabel("Land surface temperature (degC)")
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend(frameon=False)

    figure.text(
        0.01,
        0.01,
        caveat_footer(params, ["lst_not_air_temp", "single_overpass"]),
        fontsize=7,
        va="bottom",
        ha="left",
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.13, 1, 1))

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    return destination
