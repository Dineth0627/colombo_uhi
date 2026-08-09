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
    labelled series from :func:`colombo_uhi.composites.zonal_annual_means` or
    :func:`colombo_uhi.composites.zonal_annual_means_by_year`.

    When the frames carry ``count_column``, the figure gains a second panel
    plotting it on a log scale. That panel is not decoration: a MODIS annual
    mean over the CMC can rest on as few as 2 one-kilometre pixels while the
    Landsat mean beside it rests on ~4200, and nothing in the top panel reveals
    that. CLAUDE.md caveat 2 applies to figures, not only to tables.

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
    show_counts = bool(count_column) and any(
        count_column in frame.columns for frame in series.values()
    )

    figure = Figure(figsize=(10, 7.0 if show_counts else 5.5))
    FigureCanvasAgg(figure)
    if show_counts:
        # Counts get their own panel rather than an annotation: a reader cannot
        # otherwise see that one series' point rests on 2 pixels and another's
        # on 4200. CLAUDE.md caveat 2 applied to the figure, not just the table.
        axes, count_axes = figure.subplots(2, 1, sharex=True, height_ratios=[3, 1])
    else:
        axes, count_axes = figure.subplots(), None

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
        line, = axes.plot(
            plotted[year_column],
            plotted[value_column],
            marker="o",
            markersize=4,
            linewidth=1.6,
            label=label,
        )
        if count_axes is not None and count_column in plotted.columns:
            count_axes.plot(
                plotted[year_column],
                plotted[count_column],
                marker=".",
                markersize=4,
                linewidth=1.0,
                color=line.get_color(),
            )

    axes.set_ylabel("Land surface temperature (degC)")
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend(frameon=False, fontsize=8)

    if count_axes is not None:
        count_axes.set_yscale("log")
        count_axes.set_ylabel("valid\npixels", fontsize=8)
        count_axes.set_xlabel("Year")
        count_axes.grid(True, alpha=0.3)
        count_axes.tick_params(labelsize=8)
    else:
        axes.set_xlabel("Year")

    figure.text(
        0.01,
        0.01,
        caveat_footer(params, ["lst_not_air_temp", "single_overpass"]),
        fontsize=7,
        va="bottom",
        ha="left",
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.11, 1, 1))

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    return destination


# =============================================================================
# Phase 3 figures
# =============================================================================
def utfvi_vis_params(params: dict[str, Any]) -> dict[str, Any]:
    """Visualisation dictionary for a UTFVI class image.

    Keeps the palette in ``config/params.yaml`` rather than in a notebook cell,
    so the map and the stacked-share figure cannot drift apart.

    Args:
        params: Parsed params mapping (``uhi.utfvi``).

    Returns:
        ``{"min": 0, "max": n_classes - 1, "palette": [...]}`` for
        ``ee.Image.visualize``.

    Raises:
        ValueError: If the palette length does not match the class count.
    """
    from colombo_uhi import uhi_metrics

    _, labels = uhi_metrics.validate_utfvi_scheme(params)
    palette = list(params["uhi"]["utfvi"]["palette"])
    if len(palette) != len(labels):
        raise ValueError(
            f"uhi.utfvi.palette has {len(palette)} colours but there are "
            f"{len(labels)} classes ({labels}); a short palette silently "
            "recycles colours and makes two classes indistinguishable"
        )
    return {"min": 0, "max": len(labels) - 1, "palette": palette}


def plot_suhii_sensitivity(
    frame: "pd.DataFrame",
    out_path: str | Path,
    params: dict[str, Any],
    sources: Sequence[str] | None = None,
    title: str = "Surface UHI intensity by source and rural reference",
) -> Path:
    """Plot the SUHII time series with both rural definitions overlaid.

    The whole point of the figure is the SPREAD between the two rural
    definitions, not either line on its own (CLAUDE.md: report rural-reference
    sensitivity, never a single unqualified number). So each source gets one
    colour and each rural definition one line style — the vertical gap between a
    solid and a dashed line of the same colour is the sensitivity, readable
    directly off the page.

    The two definitions differ by construction (CMC versus a 15-25 km annulus,
    against built versus vegetated LCZ classes inside the district). They are
    not two estimates of one number.

    Args:
        frame: Tidy SUHII table from
            :func:`colombo_uhi.uhi_metrics.suhii_all_sources`.
        out_path: Destination ``.png`` path; parent directories are created.
        params: Parsed params mapping.
        sources: Sources to draw, in order; ``None`` draws every source present.
        title: Axes title.

    Returns:
        The path written.

    Raises:
        ValueError: If the frame is empty or lacks the expected columns.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    required = ["year", "source", "rural_definition", "suhii"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"SUHII frame is missing column(s) {missing}; it has "
            f"{sorted(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("SUHII frame is empty; there is nothing to plot")

    keys = list(sources) if sources else list(dict.fromkeys(frame["source"]))
    methods = list(dict.fromkeys(frame["rural_definition"]))
    # Solid first, so the default rural definition reads as the primary line.
    styles = ["-", "--", ":", "-."]

    show_counts = "urban_pixels" in frame.columns
    figure = Figure(figsize=(11, 7.5 if show_counts else 6.0))
    FigureCanvasAgg(figure)
    if show_counts:
        axes, count_axes = figure.subplots(2, 1, sharex=True, height_ratios=[3, 1])
    else:
        axes, count_axes = figure.subplots(), None

    palette = [f"C{index % 10}" for index in range(len(keys))]

    for colour, key in zip(palette, keys):
        for style, method in zip(styles, methods):
            block = frame[
                (frame["source"] == key) & (frame["rural_definition"] == method)
            ].dropna(subset=["suhii"])
            if block.empty:
                continue
            if "urban_pixels" in block.columns:
                block = block[block["urban_pixels"].fillna(0) > 0]
            axes.plot(
                block["year"],
                block["suhii"],
                linestyle=style,
                color=colour,
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=f"{key} / {method}",
            )
            if count_axes is not None and "urban_pixels" in block.columns:
                count_axes.plot(
                    block["year"],
                    block["urban_pixels"],
                    linestyle=style,
                    color=colour,
                    linewidth=1.0,
                )

    # Zero is the line that decides whether there is an island at all.
    axes.axhline(0.0, color="#888888", linewidth=1.0, zorder=0)
    axes.set_ylabel("SUHII (degC): mean urban LST - mean rural LST")
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend(frameon=False, fontsize=7, ncol=2)

    if count_axes is not None:
        count_axes.set_yscale("log")
        count_axes.set_ylabel("urban\npixels", fontsize=8)
        count_axes.set_xlabel("Year")
        count_axes.grid(True, alpha=0.3)
        count_axes.tick_params(labelsize=8)
    else:
        axes.set_xlabel("Year")

    definitions = ", ".join(methods)
    figure.text(
        0.01,
        0.01,
        caveat_footer(
            params, ["lst_not_air_temp", "sensitivity_reporting", "single_overpass"]
        )
        + f"\n- Rural definitions shown ({definitions}) differ by construction; "
        "the gap between line styles IS the sensitivity.",
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


def plot_utfvi_class_shares(
    frame: "pd.DataFrame",
    out_path: str | Path,
    params: dict[str, Any],
    title: str = "Share of AOI in each UTFVI class, per year",
) -> Path:
    """Stacked-area plot of the six UTFVI class shares over time.

    Args:
        frame: Table from
            :func:`colombo_uhi.uhi_metrics.utfvi_class_series`.
        out_path: Destination ``.png`` path; parent directories are created.
        params: Parsed params mapping.
        title: Axes title.

    Returns:
        The path written.

    Raises:
        ValueError: If the frame is empty or a class column is missing.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from colombo_uhi import uhi_metrics

    _, labels = uhi_metrics.validate_utfvi_scheme(params)
    palette = [f"#{colour}" for colour in params["uhi"]["utfvi"]["palette"]]
    year_column = params["composites"]["year_property"]

    missing = [c for c in (year_column, *labels) if c not in frame.columns]
    if missing:
        raise ValueError(
            f"UTFVI share frame is missing column(s) {missing}; it has "
            f"{sorted(frame.columns)}"
        )
    plotted = frame.dropna(subset=list(labels), how="all")
    if plotted.empty:
        raise ValueError("UTFVI share frame has no year with classified pixels")

    figure = Figure(figsize=(10, 6))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.stackplot(
        plotted[year_column],
        *[plotted[label].fillna(0) for label in labels],
        labels=labels,
        colors=palette,
    )
    axes.set_ylim(0, 100)
    axes.set_ylabel("Share of AOI (%)")
    axes.set_xlabel("Year")
    axes.set_title(title)
    axes.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)

    figure.text(
        0.01,
        0.01,
        caveat_footer(params, ["lst_not_air_temp", "valid_obs_required"])
        + "\n- UTFVI is referenced to EACH YEAR'S OWN mean LST, so these shares "
        "show how heat was DISTRIBUTED that year,\n  not how hot it was. "
        "Uniform warming leaves them unchanged; use the Phase 4 trend products "
        "for warming.",
        fontsize=7,
        va="bottom",
        ha="left",
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.15, 1, 1))

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    return destination


def plot_lst_vs_index(
    frame: "pd.DataFrame",
    out_path: str | Path,
    params: dict[str, Any],
    index_columns: Sequence[str] | None = None,
    response: str | None = None,
    max_points: int = 4000,
    title: str | None = None,
) -> Path:
    """Scatter sampled pixels of LST against each driver, with a fitted line.

    Args:
        frame: Sampled-pixel table from
            :func:`colombo_uhi.uhi_metrics.sample_drivers` or the pooled samples
            returned by :func:`colombo_uhi.uhi_metrics.driver_series`.
        out_path: Destination ``.png`` path; parent directories are created.
        params: Parsed params mapping.
        index_columns: Drivers to plot, one panel each; ``None`` uses
            ``uhi.drivers.predictors``.
        response: Response column; defaults to ``uhi.drivers.response``.
        max_points: Points drawn per panel. A 26-year pooled sample is 130 000
            rows, which renders as a solid block; thinning is a DRAWING choice
            only and the fitted line is computed from every row.
        title: Figure title; ``None`` builds one naming the response.

    Returns:
        The path written.

    Raises:
        ValueError: If the frame is empty or a named column is missing.
    """
    import numpy as np
    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from colombo_uhi import uhi_metrics

    y_name = response or params["uhi"]["drivers"]["response"]
    columns = uhi_metrics.resolve_predictors(index_columns, params)

    missing = [c for c in (y_name, *columns) if c not in frame.columns]
    if missing:
        raise ValueError(
            f"sample frame is missing column(s) {missing}; it has "
            f"{sorted(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("sample frame is empty; there is nothing to plot")

    figure = Figure(figsize=(5.2 * len(columns), 5.0))
    FigureCanvasAgg(figure)
    panels = figure.subplots(1, len(columns), squeeze=False)[0]

    for axes, column in zip(panels, columns):
        pair = frame[[y_name, column]].apply(pd.to_numeric, errors="coerce").dropna()
        if pair.empty:
            axes.set_title(f"{column}: no data")
            continue

        shown = pair
        if len(pair) > max_points:
            shown = pair.sample(
                max_points, random_state=params["uhi"]["drivers"]["sample_seed"]
            )
        axes.scatter(shown[column], shown[y_name], s=4, alpha=0.25, edgecolors="none")

        # The line is fitted on EVERY row, not on the thinned draw.
        # Distinct-value counting, not std() == 0: pandas' standard deviation of
        # a constant column can come back as 2.8e-17 rather than exactly zero,
        # which would slip a singular fit past the guard and into np.polyfit.
        if int(pair[column].nunique(dropna=True)) > 1:
            slope, intercept = np.polyfit(pair[column], pair[y_name], 1)
            grid = np.linspace(pair[column].min(), pair[column].max(), 50)
            axes.plot(grid, slope * grid + intercept, color="#d7191c", linewidth=1.8)
            r_value = float(np.corrcoef(pair[column], pair[y_name])[0, 1])
            axes.set_title(
                f"{column}\nslope {slope:.2f} degC per unit, "
                f"r = {r_value:.2f}, R2 = {r_value ** 2:.2f}, n = {len(pair)}",
                fontsize=9,
            )
        else:
            axes.set_title(f"{column}: constant, no fit", fontsize=9)

        axes.set_xlabel(column)
        axes.grid(True, alpha=0.3)
    panels[0].set_ylabel("Land surface temperature (degC)")

    figure.suptitle(
        title or f"{y_name} against candidate drivers (sampled pixels)",
        fontsize=11,
    )
    figure.text(
        0.01,
        0.01,
        caveat_footer(params, ["lst_not_air_temp", "single_overpass"])
        + "\n- Sampled pixels are spatially autocorrelated, so the fitted "
        "relationships are a screening device:\n  OLS standard errors are too "
        "small and p-values overstate significance until Phase 5 corrects them.",
        fontsize=7,
        va="bottom",
        ha="left",
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.15, 1, 0.94))

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    return destination
