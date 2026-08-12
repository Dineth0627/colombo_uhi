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


#: Per-rural-definition line styling for :func:`plot_suhii_sensitivity`.
#: Colour, dash pattern AND marker all change together. That redundancy is
#: deliberate: the figure stays readable in greyscale and under the common
#: colour-vision deficiencies, neither of which a colour-only encoding survives.
SUHII_METHOD_STYLES: dict[str, dict[str, Any]] = {
    "buffer_ring": {"color": "#1f78b4", "linestyle": "-", "marker": "o"},
    "lcz_based": {"color": "#e66101", "linestyle": "--", "marker": "s"},
}

#: Fallback styling for a rural definition not named in
#: :data:`SUHII_METHOD_STYLES` (a sensitivity run may add one).
SUHII_FALLBACK_STYLES: tuple[dict[str, Any], ...] = (
    {"color": "#4daf4a", "linestyle": "-.", "marker": "^"},
    {"color": "#984ea3", "linestyle": ":", "marker": "v"},
)

#: Panels per row in the SUHII small-multiples grid.
SUHII_FACET_COLUMNS = 3


def plot_suhii_sensitivity(
    frame: "pd.DataFrame",
    out_path: str | Path,
    params: dict[str, Any],
    sources: Sequence[str] | None = None,
    title: str = "Surface UHI intensity: rural-reference sensitivity by source",
) -> Path:
    """Plot SUHII as small multiples, one panel per LST source.

    The figure exists to show the SPREAD between the rural definitions, not
    either line on its own (CLAUDE.md: report rural-reference sensitivity, never
    a single unqualified number). Faceting is what makes that spread legible:
    every panel carries the SAME two lines, so the sensitivity is a vertical gap
    inside one panel and the legend collapses to two unambiguous entries.

    This replaced a single-axes overlay of every source and definition together.
    With six sources that was twelve lines sharing six colours — one colour per
    source, two dash patterns per colour — and a legend swatch is far too short
    to show a dash pattern, so the legend read as six duplicated pairs. Faceting
    removes the ambiguity structurally rather than patching the legend.

    The shaded band between the two lines is the sensitivity itself, drawn so it
    does not have to be measured by eye.

    .. note::
        The y-axis is SHARED across panels, and that is a deliberate scientific
        choice rather than a layout convenience. With free axes, ``aqua_night``'s
        roughly 0.5 degC gap would be drawn the same size as ``landsat_dry``'s
        roughly 3 degC one — exactly the misreading this figure exists to
        prevent. Each panel's mean gap is printed in its title so a visually
        compressed panel is still quantified.

    The definitions differ by construction (CMC against a 15-25 km annulus,
    versus built against vegetated LCZ classes inside the district). They are not
    two estimates of one number, and the gap is not an error bar.

    Args:
        frame: Tidy SUHII table from
            :func:`colombo_uhi.uhi_metrics.suhii_all_sources`.
        out_path: Destination ``.png`` path; parent directories are created.
        params: Parsed params mapping.
        sources: Sources to draw, in order, one panel each; ``None`` draws every
            source present in the frame. A named source with no rows is skipped
            rather than drawn as an empty panel.
        title: Figure title.

    Returns:
        The path written.

    Raises:
        ValueError: If the frame is empty, lacks the expected columns, or leaves
            no source with plottable rows.
    """
    figure = build_suhii_figure(frame, params, sources=sources, title=title)
    return _save_figure(figure, out_path)


def _save_figure(figure: Any, out_path: str | Path) -> Path:
    """Write a figure to disk, creating parent directories.

    Args:
        figure: A ``matplotlib.figure.Figure``.
        out_path: Destination ``.png`` path.

    Returns:
        The path written.
    """
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    return destination


def build_suhii_figure(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    sources: Sequence[str] | None = None,
    title: str = "Surface UHI intensity: rural-reference sensitivity by source",
) -> Any:
    """Build the SUHII small-multiples figure without writing it.

    Split out from :func:`plot_suhii_sensitivity` so the figure's STRUCTURE is
    testable — a saved PNG is pixels, and the property that matters most here
    (the legend carries exactly one entry per rural definition, never one per
    source-and-definition pair) cannot be asserted on a file.

    Args:
        frame: Tidy SUHII table.
        params: Parsed params mapping.
        sources: Sources to draw, one panel each.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: See :func:`plot_suhii_sensitivity`.
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

    requested = list(sources) if sources else list(dict.fromkeys(frame["source"]))
    keys = [key for key in requested if (frame["source"] == key).any()]
    if not keys:
        raise ValueError(
            f"none of the requested sources {requested} appear in the frame, "
            f"which has {sorted(set(frame['source']))}"
        )
    methods = list(dict.fromkeys(frame["rural_definition"]))
    styles = _suhii_styles(methods)

    columns = min(SUHII_FACET_COLUMNS, len(keys))
    rows = -(-len(keys) // columns)  # ceiling division
    figure = Figure(figsize=(4.2 * columns, 3.1 * rows + 2.0))
    FigureCanvasAgg(figure)
    panels = figure.subplots(rows, columns, sharex=True, sharey=True, squeeze=False)
    flat = [panels[r][c] for r in range(rows) for c in range(columns)]

    for axes, key in zip(flat, keys):
        drawn = _draw_suhii_panel(axes, frame, key, methods, styles)
        axes.set_title(_suhii_panel_title(frame, key, methods), fontsize=9)
        axes.axhline(0.0, color="#888888", linewidth=0.9, zorder=0)
        axes.grid(True, alpha=0.3)
        axes.tick_params(labelsize=8)
        if not drawn:
            axes.text(
                0.5,
                0.5,
                "no plottable years",
                transform=axes.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#888888",
            )

    # Hide the unused cells of a partial last row rather than leaving empty
    # boxes, which read as missing data.
    for axes in flat[len(keys):]:
        axes.set_visible(False)

    for index, axes in enumerate(flat[: len(keys)]):
        if index % columns == 0:
            axes.set_ylabel("SUHII (degC)", fontsize=9)
        if index >= len(keys) - columns:
            axes.set_xlabel("Year", fontsize=9)

    handles = [
        _legend_handle(styles[method], f"rural reference: {method}")
        for method in methods
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        handlelength=3.2,   # long enough that the dash pattern is actually visible
    )
    figure.suptitle(title, fontsize=12, y=0.995)

    definitions = ", ".join(methods)
    figure.text(
        0.01,
        0.01,
        caveat_footer(
            params, ["lst_not_air_temp", "sensitivity_reporting", "single_overpass"]
        )
        + f"\n- The shaded band between the two lines IS the rural-reference "
        f"sensitivity. The definitions ({definitions}) differ by\n  construction, "
        "so the band is not an error bar and the two lines are not two estimates "
        "of one number.\n- Panels share a y-axis so gaps are comparable between "
        "sources; 'urban px' is the median pixel count each mean rests on.",
        fontsize=7,
        va="bottom",
        ha="left",
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.10 + 0.02 * rows, 1, 0.93))
    return figure


def _suhii_styles(methods: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Resolve a line style for every rural definition present.

    Args:
        methods: Rural-definition names, in draw order.

    Returns:
        Mapping of method name to matplotlib line kwargs.
    """
    styles: dict[str, dict[str, Any]] = {}
    spare = list(SUHII_FALLBACK_STYLES)
    for method in methods:
        if method in SUHII_METHOD_STYLES:
            styles[method] = dict(SUHII_METHOD_STYLES[method])
        else:
            styles[method] = dict(
                spare.pop(0) if spare else {"color": "#666666", "linestyle": "-"}
            )
    return styles


def _suhii_plottable(
    frame: "pd.DataFrame", key: str, method: str
) -> "pd.DataFrame":
    """Rows of one source and rural definition that may honestly be drawn.

    Drops years with no SUHII and years whose mean rests on zero valid pixels —
    a SUHII computed over no urban pixels is not a measurement (CLAUDE.md
    caveat 2), and plotting it would draw a line through a hole in the data.

    Args:
        frame: Tidy SUHII table.
        key: Source key.
        method: Rural-definition name.

    Returns:
        The plottable rows, sorted by year.
    """
    block = frame[
        (frame["source"] == key) & (frame["rural_definition"] == method)
    ].dropna(subset=["suhii"])
    for column in ("urban_pixels", "rural_pixels"):
        if column in block.columns:
            block = block[block[column].fillna(0) > 0]
    return block.sort_values("year")


def _draw_suhii_panel(
    axes: Any,
    frame: "pd.DataFrame",
    key: str,
    methods: Sequence[str],
    styles: dict[str, dict[str, Any]],
) -> bool:
    """Draw one source's panel: both definitions plus the shaded gap between them.

    Args:
        axes: Target axes.
        frame: Tidy SUHII table.
        key: Source key.
        methods: Rural-definition names.
        styles: Mapping from :func:`_suhii_styles`.

    Returns:
        ``True`` if anything was drawn.
    """
    drawn: dict[str, "pd.DataFrame"] = {}
    for method in methods:
        block = _suhii_plottable(frame, key, method)
        if block.empty:
            continue
        drawn[method] = block
        axes.plot(
            block["year"],
            block["suhii"],
            markersize=3.5,
            linewidth=1.6,
            **styles[method],
        )

    # Shade the sensitivity, but only where both definitions actually have a
    # value for the same year - shading across a gap would invent one.
    if len(drawn) == 2:
        import pandas as pd

        first, second = (drawn[method] for method in drawn)
        merged = pd.merge(
            first[["year", "suhii"]],
            second[["year", "suhii"]],
            on="year",
            suffixes=("_a", "_b"),
        )
        if not merged.empty:
            axes.fill_between(
                merged["year"],
                merged["suhii_a"],
                merged["suhii_b"],
                color="#9e9e9e",
                alpha=0.20,
                linewidth=0,
                zorder=0,
            )
    return bool(drawn)


def _suhii_panel_title(
    frame: "pd.DataFrame", key: str, methods: Sequence[str]
) -> str:
    """Panel title: the source, its pixel counts, and the mean sensitivity gap.

    The pixel counts are not decoration. A MODIS SUHII over the CMC can rest on a
    median of 17 one-kilometre pixels while the Landsat one beside it rests on
    4101, and nothing in the lines themselves reveals that (CLAUDE.md caveat 2).

    Args:
        frame: Tidy SUHII table.
        key: Source key.
        methods: Rural-definition names.

    Returns:
        A two-line title string.
    """
    counts: list[str] = []
    means: dict[str, float] = {}
    for method in methods:
        block = _suhii_plottable(frame, key, method)
        if block.empty:
            continue
        means[method] = float(block["suhii"].mean())
        if "urban_pixels" in block.columns:
            counts.append(f"{float(block['urban_pixels'].median()):.0f}")

    detail: list[str] = []
    if counts:
        detail.append("urban px " + " / ".join(counts))
    if len(means) == 2:
        gap = abs(means[methods[0]] - means[methods[1]])
        detail.append(f"gap {gap:.2f} degC")
    return key if not detail else f"{key}\n{'   '.join(detail)}"


def _legend_handle(style: dict[str, Any], label: str) -> Any:
    """Build a standalone legend handle for one rural definition.

    Args:
        style: Matplotlib line kwargs.
        label: Legend text.

    Returns:
        A ``matplotlib.lines.Line2D`` proxy artist.
    """
    from matplotlib.lines import Line2D

    return Line2D([], [], label=label, linewidth=1.6, markersize=5, **style)


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
    return _save_figure(
        build_utfvi_shares_figure(frame, params, title=title), out_path
    )


def build_utfvi_shares_figure(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    title: str = "Share of AOI in each UTFVI class, per year",
) -> Any:
    """Build the UTFVI stacked-share figure without writing it.

    Split out so the legend's PLACEMENT is testable. The bug this guards against
    was a legend drawn inside the axes, where the red "Worst" swatch landed on
    the red "Worst" band and became invisible — a defect no pixel comparison
    would catch but an anchor assertion will.

    Args:
        frame: Per-year class-share table.
        params: Parsed params mapping.
        title: Axes title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: See :func:`plot_utfvi_class_shares`.
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

    figure = Figure(figsize=(10, 5.8))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.stackplot(
        plotted[year_column],
        *[plotted[label].fillna(0) for label in labels],
        labels=labels,
        colors=palette,
        # Thin white separators: Good, Normal and Bad are only a few percent each
        # in the real series and blur into one band without them.
        edgecolor="white",
        linewidth=0.4,
    )
    axes.set_ylim(0, 100)
    axes.set_xlim(plotted[year_column].min(), plotted[year_column].max())
    axes.set_ylabel("Share of AOI (%)")
    axes.set_xlabel("Year")
    axes.set_title(title)
    # BELOW the axes, not inside it. Placed inside (the previous `loc="upper
    # left"`), the legend sat on top of the stack, and the "Worst" swatch — red
    # on the red Worst band — was invisible. A legend must never be drawn over
    # the thing it names.
    axes.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=len(labels),
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        handlelength=1.6,
    )

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
    # Reserve just enough for the legend strip plus the five-line caveat footer.
    figure.tight_layout(rect=(0, 0.17, 1, 1))
    return figure


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
    figure = build_lst_vs_index_figure(
        frame,
        params,
        index_columns=index_columns,
        response=response,
        max_points=max_points,
        title=title,
    )
    return _save_figure(figure, out_path)


def build_lst_vs_index_figure(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    index_columns: Sequence[str] | None = None,
    response: str | None = None,
    max_points: int = 4000,
    title: str | None = None,
) -> Any:
    """Build the LST-vs-driver scatter figure without writing it.

    Split out so the fitted line's DRAWN EXTENT is testable: the line is computed
    from every row but drawn only across the 1st-99th percentile of x, and that
    clipping is invisible in a saved PNG.

    Args:
        frame: Sampled-pixel table.
        params: Parsed params mapping.
        index_columns: Drivers to plot, one panel each.
        response: Response column.
        max_points: Points drawn per panel.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: See :func:`plot_lst_vs_index`.
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
        axes.scatter(
            shown[column],
            shown[y_name],
            s=4,
            alpha=0.25,
            edgecolors="none",
            label="sampled pixels",
        )

        # The line is fitted on EVERY row, not on the thinned draw.
        # Distinct-value counting, not std() == 0: pandas' standard deviation of
        # a constant column can come back as 2.8e-17 rather than exactly zero,
        # which would slip a singular fit past the guard and into np.polyfit.
        if int(pair[column].nunique(dropna=True)) > 1:
            slope, intercept = np.polyfit(pair[column], pair[y_name], 1)
            # Drawn across the 1st-99th percentile, NOT the full min-max. The fit
            # still uses every row; only the drawn extent is clipped. Extending
            # the line into a tail holding a handful of pixels makes the
            # relationship look better supported out there than it is.
            low, high = np.percentile(pair[column], [1, 99])
            if high <= low:  # a near-degenerate spread; fall back to the range
                low, high = float(pair[column].min()), float(pair[column].max())
            grid = np.linspace(low, high, 50)
            axes.plot(
                grid,
                slope * grid + intercept,
                color="#d7191c",
                linewidth=1.8,
                label="OLS fit (1st-99th pct)",
            )
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
        # The red line was previously unlabelled and a reader had to guess it.
        axes.legend(loc="best", fontsize=7, framealpha=0.85, markerscale=2.5)
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
    return figure


# =============================================================================
# Phase 4 - trend figures
# =============================================================================
def trend_vis_params(params: dict[str, Any]) -> dict[str, Any]:
    """Visualisation parameters for a Sen's-slope map.

    .. note::
        The palette MUST be diverging and the range symmetric about zero. On a
        sequential ramp a reader cannot tell warming from cooling, which is the
        one thing a trend map exists to show.

    Args:
        params: Parsed params mapping (``trends.slope_vis``).

    Returns:
        Mapping with ``min``, ``max`` and ``palette``, ready for
        :func:`save_thumbnail` or ``geemap``.

    Raises:
        ValueError: If the configured range is not symmetric about zero, or the
            palette has an even number of colours (so no colour sits at zero).
    """
    vis = params["trends"]["slope_vis"]
    if vis["min"] != -vis["max"]:
        raise ValueError(
            f"trends.slope_vis must be symmetric about zero so the middle colour "
            f"marks 'no trend', got min={vis['min']} max={vis['max']}"
        )
    if len(vis["palette"]) % 2 == 0:
        raise ValueError(
            f"trends.slope_vis.palette must have an ODD number of colours so one "
            f"sits at zero, got {len(vis['palette'])}"
        )
    return {"min": vis["min"], "max": vis["max"], "palette": list(vis["palette"])}


def build_trend_map_figure(
    arrays: Mapping[str, Any],
    params: dict[str, Any],
    slope_key: str = "sen_slope",
    significant_key: str = "significant",
    title: str = "Sen's slope of land surface temperature, 2000-2025",
) -> Any:
    """Build the two-panel trend map: all slopes, then FDR-significant only.

    Both panels are drawn because they answer different questions, and the pair
    is the honest presentation: the left shows the estimated rate everywhere,
    the right only where that rate survives multiple-testing correction. The
    left alone overstates confidence; the right alone hides how much of the map
    was never testable at all.

    Args:
        arrays: Mapping of band name to 2-D array, as returned by
            :func:`colombo_uhi.trends.sample_trend_arrays` or
            :func:`colombo_uhi.trends.read_trend_raster`.
        params: Parsed params mapping.
        slope_key: Key holding the Sen's slope.
        significant_key: Key holding the significance mask (1/0/NaN).
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If a required array is missing, naming what is present.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.figure import Figure

    missing = [key for key in (slope_key, significant_key) if key not in arrays]
    if missing:
        raise ValueError(
            f"trend arrays are missing {missing}; they hold {sorted(arrays)}"
        )

    vis = trend_vis_params(params)
    colours = ["#" + str(colour).lstrip("#") for colour in vis["palette"]]
    cmap = LinearSegmentedColormap.from_list("trend_diverging", colours)
    cmap = cmap.with_extremes(bad="#dcdcdc")
    norm = TwoSlopeNorm(vmin=vis["min"], vcenter=0.0, vmax=vis["max"])

    slope = np.asarray(arrays[slope_key], dtype="float64")
    significant = np.asarray(arrays[significant_key], dtype="float64")
    masked = np.where(significant == 1.0, slope, np.nan)

    figure = Figure(figsize=(11.0, 6.4))
    FigureCanvasAgg(figure)
    panels = figure.subplots(1, 2)
    # A colorbar attached to BOTH panels is not compatible with tight_layout, so
    # this figure is spaced explicitly rather than fitted. The generous bottom
    # margin is for the four-caveat footer, which is long by design.
    figure.subplots_adjust(left=0.02, right=0.90, top=0.91, bottom=0.28, wspace=0.05)

    image = None
    for axes, data, panel_title in (
        (panels[0], slope, "Sen's slope (all fitted pixels)"),
        (panels[1], masked, "FDR-significant trends only"),
    ):
        image = axes.imshow(
            np.ma.masked_invalid(data), cmap=cmap, norm=norm, interpolation="nearest"
        )
        axes.set_title(panel_title, fontsize=10)
        axes.set_xticks([])
        axes.set_yticks([])

    bar = figure.colorbar(
        image, ax=list(panels), orientation="vertical", fraction=0.035, pad=0.02
    )
    bar.set_label("degC per year", fontsize=9)
    bar.ax.tick_params(labelsize=8)

    # The denominator is the TESTED set, taken from the significance array rather
    # than from the slope. A pixel can carry a finite slope and still not have
    # been tested (below the minimum-year floor), and counting those would give a
    # figure caption that disagrees with trends.fdr_significant_fraction.
    tested = int(np.isfinite(significant).sum())
    n_significant = int(np.nansum(significant == 1.0))
    share = (n_significant / tested) if tested else float("nan")

    figure.suptitle(title, fontsize=12)
    figure.text(
        0.01,
        0.01,
        caveat_footer(
            params,
            [
                "lst_not_air_temp",
                "valid_obs_required",
                "single_overpass",
                "fdr_dependence",
            ],
        )
        + f"\n- {n_significant} of {tested} fitted pixels ({share:.1%}) are "
        "FDR-significant. Grey is NOT 'no trend': it is a pixel that was never\n"
        "  tested, because it fell below the valid-observation or minimum-year "
        "floor. Read this map against the n_years band.",
        fontsize=7,
        va="bottom",
        ha="left",
        color="#444444",
    )
    return figure


def plot_trend_map(
    arrays: Mapping[str, Any],
    out_path: str | Path,
    params: dict[str, Any],
    slope_key: str = "sen_slope",
    significant_key: str = "significant",
    title: str = "Sen's slope of land surface temperature, 2000-2025",
) -> Path:
    """Write the two-panel trend map. See :func:`build_trend_map_figure`."""
    return _save_figure(
        build_trend_map_figure(
            arrays,
            params,
            slope_key=slope_key,
            significant_key=significant_key,
            title=title,
        ),
        out_path,
    )


def build_mk_comparison_figure(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    title: str = "Mann-Kendall: effect of the autocorrelation correction",
) -> Any:
    """Build the plain-versus-modified Mann-Kendall comparison figure.

    The left panel pairs each series' uncorrected and corrected p-value against
    the significance level; the right shows the variance inflation. Together
    they answer the only question the modified test is run to answer: **how much
    did serial autocorrelation inflate our confidence?**

    Args:
        frame: Output of :func:`colombo_uhi.trends.mk_comparison` or
            :func:`colombo_uhi.trends.suhii_trends`.
        params: Parsed params mapping.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the frame is empty or lacks the required columns.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    required = ["label", "series", "test", "p", "var_inflation"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"MK frame is missing column(s) {missing}; it has "
            f"{sorted(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("MK frame is empty; there is nothing to plot")

    alpha = params["trends"]["fdr"]["alpha"]
    modified_name = params["trends"]["mmk"]["method"]

    original = frame[frame["test"] == "original"]
    modified = frame[frame["test"] == modified_name]
    keys = [f"{row.label}|{row.series}" for row in original.itertuples()]
    lookup = {f"{row.label}|{row.series}": row for row in modified.itertuples()}

    # The footer is a fixed number of INCHES tall, so the fraction it occupies
    # shrinks as rows are added. Reserving a constant fraction instead lets the
    # x-axis label collide with it on a short figure.
    footer_inches = 1.45
    height = max(4.6, 0.32 * len(keys) + 2.6) + footer_inches
    figure = Figure(figsize=(11.5, height))
    FigureCanvasAgg(figure)
    panels = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.6, 1.0]})

    positions = np.arange(len(keys))

    def _value(row: Any, field: str) -> float:
        value = getattr(row, field, None)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    original_p = [_value(row, "p") for row in original.itertuples()]
    modified_p = [
        _value(lookup[key], "p") if key in lookup else float("nan") for key in keys
    ]

    # Colour AND marker vary together, so the pair still reads in greyscale.
    for low, high, index in zip(original_p, modified_p, positions):
        if np.isfinite(low) and np.isfinite(high):
            panels[0].plot(
                [low, high], [index, index], color="#999999", linewidth=1.0, zorder=2
            )
    panels[0].scatter(
        original_p, positions, color="#1f77b4", marker="o", s=42,
        label="uncorrected", zorder=3,
    )
    panels[0].scatter(
        modified_p, positions, color="#d62728", marker="s", s=42,
        label=f"corrected ({modified_name})", zorder=3,
    )
    panels[0].axvline(
        alpha, color="#444444", linestyle="--", linewidth=1.1,
        label=f"alpha = {alpha}",
    )
    panels[0].set_xscale("log")
    panels[0].set_yticks(positions)
    panels[0].set_yticklabels(keys, fontsize=7)
    panels[0].set_xlabel("p-value (log scale)", fontsize=9)
    panels[0].grid(True, axis="x", alpha=0.3)
    panels[0].legend(loc="lower right", fontsize=8, framealpha=0.9)

    inflation = [
        _value(lookup[key], "var_inflation") if key in lookup else float("nan")
        for key in keys
    ]
    panels[1].barh(positions, inflation, color="#7f7f7f")
    panels[1].axvline(1.0, color="#444444", linestyle="--", linewidth=1.1)
    panels[1].set_yticks(positions)
    panels[1].set_yticklabels([])
    panels[1].set_xlabel("Var(S) inflation (corrected / uncorrected)", fontsize=9)
    panels[1].grid(True, axis="x", alpha=0.3)

    figure.suptitle(title, fontsize=12)
    figure.text(
        0.01,
        0.01,
        caveat_footer(params, ["lst_not_air_temp", "sensitivity_reporting"])
        + "\n- Annual LST is positively autocorrelated, so the UNCORRECTED "
        "p-value is anti-conservative. An inflation bar above 1 means the\n"
        "  uncorrected test overstated significance by that factor in the "
        "variance; a bar near 1 means the correction changed nothing,\n  which is "
        "itself a reportable result. Quote the corrected p-value.",
        fontsize=7,
        va="bottom",
        ha="left",
        color="#444444",
    )
    figure.tight_layout(rect=(0, footer_inches / height, 1, 1 - 0.35 / height))
    return figure


def plot_mk_comparison(
    frame: "pd.DataFrame",
    out_path: str | Path,
    params: dict[str, Any],
    title: str = "Mann-Kendall: effect of the autocorrelation correction",
) -> Path:
    """Write the MK comparison figure. See :func:`build_mk_comparison_figure`."""
    return _save_figure(
        build_mk_comparison_figure(frame, params, title=title), out_path
    )


def build_trend_by_class_figure(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    value_column: str = "mean",
    title: str | None = None,
) -> Any:
    """Build the trend-magnitude-by-class bar figure.

    Classes below ``trends.stratify.min_pixels_per_class`` are hatched rather
    than dropped: a class mean resting on twelve pixels is not an estimate, but
    silently removing it hides that the class exists at all.

    Args:
        frame: Output of :func:`colombo_uhi.trends.trend_by_class`.
        params: Parsed params mapping.
        value_column: Column holding the mean slope.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the frame is empty or lacks the required columns.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    required = ["class_label", "pixel_count", "below_pixel_floor", value_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"class trend frame is missing column(s) {missing}; it has "
            f"{sorted(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("class trend frame is empty; there is nothing to plot")

    ordered = frame.sort_values(value_column, ascending=True)
    labels = list(ordered["class_label"])
    values = ordered[value_column].astype("float64").to_numpy()
    counts = ordered["pixel_count"].astype("float64").to_numpy()
    sparse = ordered["below_pixel_floor"].astype(bool).to_numpy()
    errors = (
        ordered["stdDev"].astype("float64").to_numpy()
        if "stdDev" in ordered.columns
        else None
    )

    # As in build_mk_comparison_figure: the footer is a fixed height in inches,
    # so reserve it in inches rather than as a fraction.
    footer_inches = 1.25
    height = max(3.4, 0.42 * len(labels) + 1.9) + footer_inches
    figure = Figure(figsize=(9.5, height))
    FigureCanvasAgg(figure)
    axes = figure.subplots()

    positions = np.arange(len(labels))
    colours = ["#b2182b" if value > 0 else "#2166ac" for value in values]
    bars = axes.barh(
        positions, values, xerr=errors, color=colours, error_kw={"elinewidth": 0.8}
    )
    # Hatch AND edge colour together, so the "too few pixels" flag survives
    # greyscale printing.
    for bar, is_sparse in zip(bars, sparse):
        if is_sparse:
            bar.set_hatch("///")
            bar.set_edgecolor("#333333")

    axes.axvline(0.0, color="#444444", linewidth=1.0)
    axes.set_yticks(positions)
    axes.set_yticklabels(
        [f"{label}  (n={int(count):,})" for label, count in zip(labels, counts)],
        fontsize=8,
    )
    axes.set_xlabel("Mean Sen's slope (degC per year)", fontsize=9)
    axes.grid(True, axis="x", alpha=0.3)

    scheme = str(frame["scheme"].iloc[0]) if "scheme" in frame.columns else "class"
    axes.set_title(title or f"Trend magnitude by {scheme} class", fontsize=12)

    floor = params["trends"]["stratify"]["min_pixels_per_class"]
    figure.text(
        0.01,
        0.01,
        caveat_footer(
            params, ["lst_not_air_temp", "valid_obs_required", "trend_not_causal"]
        )
        + f"\n- Hatched bars rest on fewer than {floor} pixels and are shown for "
        "completeness, not as estimates. Error bars are the WITHIN-CLASS\n  "
        "standard deviation of the slope, not a standard error of the mean.",
        fontsize=7,
        va="bottom",
        ha="left",
        color="#444444",
    )
    figure.tight_layout(rect=(0, footer_inches / height, 1, 1 - 0.2 / height))
    return figure


def plot_trend_by_class(
    frame: "pd.DataFrame",
    out_path: str | Path,
    params: dict[str, Any],
    value_column: str = "mean",
    title: str | None = None,
) -> Path:
    """Write the trend-by-class figure. See :func:`build_trend_by_class_figure`."""
    return _save_figure(
        build_trend_by_class_figure(
            frame, params, value_column=value_column, title=title
        ),
        out_path,
    )


def build_decadal_difference_figure(
    arrays: Mapping[str, Any],
    params: dict[str, Any],
    difference_key: str,
    se_key: str | None = None,
    max_degc: float = 2.0,
    title: str | None = None,
) -> Any:
    """Build the decadal difference map, beside its signal-to-noise panel.

    Two panels, because the difference alone is not interpretable. The
    configured windows are 11 / 10 / 5 years - unequal by construction, since
    the study period ends in 2025 - so a difference involving the short window
    rests on roughly half the sample and carries about sqrt(2) the standard
    error. The right panel divides by that standard error so a reader can see
    which parts of the difference are distinguishable from interannual noise.

    .. warning::
        A decadal difference is NOT the warming rate. It conflates trend with
        interannual variability (a hot year at either end moves it) and with
        changing observation counts. Sen's slope is the warming number; these
        maps exist to show WHERE it is concentrated.

    Args:
        arrays: Mapping of band name to 2-D array, from
            :func:`colombo_uhi.trends.read_trend_raster`.
        params: Parsed params mapping.
        difference_key: Band holding the difference in degC.
        se_key: Band holding its standard error. ``None`` draws one panel.
        max_degc: Symmetric colour limit for the difference panel.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If a required array is missing, naming what is present.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.figure import Figure

    wanted = [difference_key] + ([se_key] if se_key else [])
    missing = [key for key in wanted if key not in arrays]
    if missing:
        raise ValueError(
            f"decadal arrays are missing {missing}; they hold {sorted(arrays)}"
        )
    if max_degc <= 0:
        raise ValueError(f"max_degc must be positive, got {max_degc}")

    colours = [
        "#" + str(colour).lstrip("#")
        for colour in params["trends"]["slope_vis"]["palette"]
    ]
    cmap = LinearSegmentedColormap.from_list("decadal_diverging", colours)
    cmap = cmap.with_extremes(bad="#dcdcdc")

    difference = np.asarray(arrays[difference_key], dtype="float64")
    panels_needed = 2 if se_key else 1

    figure = Figure(figsize=(5.9 * panels_needed, 5.9))
    FigureCanvasAgg(figure)
    axes_list = figure.subplots(1, panels_needed, squeeze=False)[0]
    # Each panel carries its own colorbar, so wspace must leave room for the
    # left panel's bar AND its label without either landing on the right panel.
    figure.subplots_adjust(
        left=0.03, right=0.92, top=0.90, bottom=0.27, wspace=0.30
    )

    image = axes_list[0].imshow(
        np.ma.masked_invalid(difference),
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-max_degc, vcenter=0.0, vmax=max_degc),
        interpolation="nearest",
    )
    axes_list[0].set_title("Difference in mean LST (degC)", fontsize=10)
    axes_list[0].set_xticks([])
    axes_list[0].set_yticks([])

    bar = figure.colorbar(
        image, ax=axes_list[0], orientation="vertical", fraction=0.04, pad=0.02
    )
    bar.set_label("degC", fontsize=9)
    bar.ax.tick_params(labelsize=8)

    if se_key:
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = difference / np.asarray(arrays[se_key], dtype="float64")
        ratio_image = axes_list[1].imshow(
            np.ma.masked_invalid(ratio),
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=3.0),
            interpolation="nearest",
        )
        axes_list[1].set_title(
            "Difference / standard error", fontsize=10
        )
        axes_list[1].set_xticks([])
        axes_list[1].set_yticks([])
        ratio_bar = figure.colorbar(
            ratio_image, ax=axes_list[1], orientation="vertical",
            fraction=0.04, pad=0.02,
        )
        ratio_bar.set_label("standard errors", fontsize=9)
        ratio_bar.ax.tick_params(labelsize=8)

    figure.suptitle(title or difference_key.replace("_", " "), fontsize=12)
    figure.text(
        0.01,
        0.01,
        caveat_footer(
            params,
            ["lst_not_air_temp", "valid_obs_required", "sensitivity_reporting"],
        )
        + "\n- A decadal difference is NOT the warming rate: it conflates trend "
        "with interannual variability and with changing observation\n  counts. "
        "Sen's slope is the warming number; this map shows WHERE it is "
        "concentrated.\n- The windows are 11 / 10 / 5 years - UNEQUAL, because "
        "the study period ends in 2025. Any difference involving 2021-2025 "
        "rests on\n  about half the sample of the others, which is what the "
        "right-hand panel exists to show. Roughly |ratio| < 2 is not "
        "distinguishable\n  from noise.",
        fontsize=7,
        va="bottom",
        ha="left",
        color="#444444",
    )
    return figure


def plot_decadal_difference(
    arrays: Mapping[str, Any],
    out_path: str | Path,
    params: dict[str, Any],
    difference_key: str,
    se_key: str | None = None,
    max_degc: float = 2.0,
    title: str | None = None,
) -> Path:
    """Write the decadal difference map. See :func:`build_decadal_difference_figure`."""
    return _save_figure(
        build_decadal_difference_figure(
            arrays,
            params,
            difference_key=difference_key,
            se_key=se_key,
            max_degc=max_degc,
            title=title,
        ),
        out_path,
    )


# =============================================================================
# Phase 5 - spatial statistics figures
# =============================================================================
def spatial_palette(params: dict[str, Any], name: str) -> dict[str, str]:
    """Return one of the Phase 5 categorical palettes, hex prefixed with ``#``.

    Args:
        params: Parsed params mapping.
        name: Palette key under ``spatial_stats.palettes``.

    Returns:
        Mapping of class label to a matplotlib-ready colour string.

    Raises:
        KeyError: If the palette is not configured, or is a list rather than a
            class mapping.
    """
    palettes = params["spatial_stats"]["palettes"]
    if name not in palettes:
        raise KeyError(
            f"unknown palette '{name}'; spatial_stats.palettes defines "
            f"{sorted(palettes)}"
        )
    entry = palettes[name]
    if not isinstance(entry, dict):
        raise KeyError(
            f"palette '{name}' is a sequence, not a class mapping; use it directly"
        )
    return {str(key): f"#{value}" for key, value in entry.items()}


def map_aspect_ratio(zones: Any, default: float = 1.0) -> float:
    """Height-to-width ratio of a zone layer's bounding box.

    Choropleths are drawn with ``set_aspect("equal")``, so a canvas whose shape
    ignores the data leaves the difference as blank paper. Colombo District is
    roughly twice as wide as it is tall, and the first Colab run produced maps
    occupying the top 60 % of the figure with the legend floating in the void
    below.

    Args:
        zones: ``geopandas.GeoDataFrame``.
        default: Returned when the bounds are degenerate.

    Returns:
        ``dy / dx``, clamped to ``[0.25, 3.0]`` so one pathological geometry
        cannot produce a figure thousands of inches tall.
    """
    try:
        min_x, min_y, max_x, max_y = (float(v) for v in zones.total_bounds)
    except Exception:  # pragma: no cover - defensive, non-geo input
        return default
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        return default
    return float(min(max(height / width, 0.25), 3.0))


def _map_figure(
    zones: Any,
    footer_inches: float,
    legend_inches: float = 0.0,
    width_inches: float = 9.5,
    title_inches: float = 0.45,
) -> tuple[Any, Any, float]:
    """Create a figure whose map axes matches the data's shape.

    Args:
        zones: ``geopandas.GeoDataFrame`` the map will draw.
        footer_inches: Vertical space reserved for the caveat footer.
        legend_inches: Vertical space reserved for a legend below the map.
        width_inches: Figure width.
        title_inches: Vertical space reserved for the title.

    Returns:
        ``(figure, axes, height_inches)``.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    map_inches = width_inches * map_aspect_ratio(zones)
    height = map_inches + legend_inches + footer_inches + title_inches
    figure = Figure(figsize=(width_inches, height))
    FigureCanvasAgg(figure)
    axes = figure.add_axes(
        (
            0.02,
            (footer_inches + legend_inches) / height,
            0.96,
            map_inches / height,
        )
    )
    return figure, axes, height


def _zone_choropleth(
    axes: Any,
    frame: Any,
    column: str,
    colours: Mapping[str, str],
    default: str = "#f0f0f0",
    hatch_mask: Any = None,
    hatch: str = "xxx",
) -> None:
    """Draw a categorical choropleth onto an axes.

    Args:
        axes: Target matplotlib axes.
        frame: ``geopandas.GeoDataFrame`` carrying ``column``.
        column: Categorical column to colour by.
        colours: Class label to colour.
        default: Colour for a class with no palette entry.
        hatch_mask: Optional boolean series; ``True`` rows are hatched.
        hatch: Hatch pattern.
    """
    face = [colours.get(str(value), default) for value in frame[column]]
    frame.plot(ax=axes, color=face, edgecolor="#ffffff", linewidth=0.15)
    if hatch_mask is not None and bool(hatch_mask.any()):
        frame[hatch_mask.to_numpy(dtype=bool)].plot(
            ax=axes,
            facecolor="none",
            edgecolor="#333333",
            linewidth=0.25,
            hatch=hatch,
        )
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)


def _category_legend(
    axes: Any, colours: Mapping[str, str], order: Sequence[str], ncol: int = 4
) -> None:
    """Attach a categorical legend BELOW the map axes.

    Placing it inside the axes only works when the map leaves empty corners.
    Once the canvas is sized to the data (see :func:`_map_figure`) there are no
    empty corners, so the legend goes underneath where it cannot cover a
    division.

    Args:
        axes: Target axes.
        colours: Class label to colour.
        order: Labels to show, in legend order.
        ncol: Legend columns.
    """
    from matplotlib.patches import Patch

    handles = [
        Patch(
            facecolor=colours.get(str(label), "#f0f0f0"),
            edgecolor="#666666",
            label=str(label).replace("_", " "),
        )
        for label in order
    ]
    axes.legend(
        handles=handles,
        fontsize=7.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
        ncol=ncol,
        handlelength=1.3,
        columnspacing=1.4,
    )


def build_cluster_map_figure(
    zones: Any,
    lisa: "pd.DataFrame",
    params: dict[str, Any],
    title: str | None = None,
) -> Any:
    """Build a Local Moran's I (LISA) cluster map.

    .. note::
        ``HL`` and ``LH`` are spatial **outliers**, not clusters: a hot division
        surrounded by cool ones, or the reverse. The palette gives them pale
        mid-tones on purpose so they cannot be read as an extension of the HH
        core, which is the most common way a LISA map is misread.

    Args:
        zones: ``geopandas.GeoDataFrame`` with ``zone_id`` and geometry.
        lisa: Output of :func:`colombo_uhi.spatial_stats.local_morans`.
        params: Parsed params mapping.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the join leaves no rows.
    """
    merged = zones.merge(
        lisa.assign(zone_id=lisa["zone_id"].astype(str)), on="zone_id", how="left"
    )
    if merged.empty:
        raise ValueError(
            "the LISA table and the zone geometry share no zone_id; both must "
            "be keyed on the pcode"
        )
    merged["cluster"] = merged["cluster"].fillna("ns")

    colours = spatial_palette(params, "lisa")
    footer_inches = 1.15
    figure, axes, height = _map_figure(
        merged, footer_inches, legend_inches=0.45
    )

    _zone_choropleth(axes, merged, "cluster", colours)
    present = set(merged["cluster"])
    _category_legend(
        axes,
        colours,
        [label for label in ("HH", "LL", "HL", "LH", "ns") if label in present],
        ncol=5,
    )
    significant = int((merged["cluster"] != "ns").sum())
    axes.set_title(
        title
        or (
            f"Local Moran's I clusters - {significant} of {len(merged)} zones "
            "significant after FDR"
        ),
        fontsize=12,
    )
    figure.text(
        0.01,
        0.01,
        caveat_footer(
            params,
            [
                "lst_not_air_temp",
                "within_epoch_only",
                "zonal_not_pixel",
                "fdr_dependence",
            ],
        )
        + "\n- HL and LH are spatial OUTLIERS, not clusters. Significance is the "
        "Benjamini-Hochberg adjusted permutation p-value.",
        fontsize=7,
        va="bottom",
        ha="left",
        color="#444444",
    )
    return figure


def plot_cluster_map(
    zones: Any,
    lisa: "pd.DataFrame",
    out_path: str | Path,
    params: dict[str, Any],
    title: str | None = None,
) -> Path:
    """Write the LISA cluster map. See :func:`build_cluster_map_figure`."""
    return _save_figure(
        build_cluster_map_figure(zones, lisa, params, title=title), out_path
    )


def build_hotspot_map_figure(
    zones: Any,
    hotspots: "pd.DataFrame",
    params: dict[str, Any],
    title: str | None = None,
) -> Any:
    """Build a Getis-Ord Gi* hot- and cold-spot map.

    Args:
        zones: ``geopandas.GeoDataFrame`` with ``zone_id`` and geometry.
        hotspots: Output of :func:`colombo_uhi.spatial_stats.gi_star`.
        params: Parsed params mapping.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the join leaves no rows.
    """
    merged = zones.merge(
        hotspots.assign(zone_id=hotspots["zone_id"].astype(str)),
        on="zone_id",
        how="left",
    )
    if merged.empty:
        raise ValueError("the Gi* table and the zone geometry share no zone_id")
    merged["confidence_class"] = merged["confidence_class"].fillna("ns")

    colours = spatial_palette(params, "gi_star")
    order = ["hot_99", "hot_95", "hot_90", "ns", "cold_90", "cold_95", "cold_99"]

    footer_inches = 1.05
    figure, axes, height = _map_figure(
        merged, footer_inches, legend_inches=0.45
    )

    _zone_choropleth(axes, merged, "confidence_class", colours)
    _category_legend(axes, colours, order, ncol=7)

    hot = int(merged["confidence_class"].astype(str).str.startswith("hot").sum())
    cold = int(merged["confidence_class"].astype(str).str.startswith("cold").sum())
    axes.set_title(
        title or f"Getis-Ord Gi* - {hot} hot and {cold} cold zones (FDR adjusted)",
        fontsize=12,
    )
    figure.text(
        0.01,
        0.01,
        caveat_footer(
            params, ["lst_not_air_temp", "within_epoch_only", "zonal_not_pixel"]
        )
        + "\n- Confidence classes are bins of the FDR-adjusted permutation "
        "p-value, so the legend means what it says after multiple testing.",
        fontsize=7,
        va="bottom",
        ha="left",
        color="#444444",
    )
    return figure


def plot_hotspot_map(
    zones: Any,
    hotspots: "pd.DataFrame",
    out_path: str | Path,
    params: dict[str, Any],
    title: str | None = None,
) -> Path:
    """Write the Gi* hot-spot map. See :func:`build_hotspot_map_figure`."""
    return _save_figure(
        build_hotspot_map_figure(zones, hotspots, params, title=title), out_path
    )


def build_ehsa_map_figure(
    zones: Any,
    ehsa: "pd.DataFrame",
    params: dict[str, Any],
    title: str | None = None,
) -> Any:
    """Build the emerging-hot-spot map.

    .. warning::
        Zones whose series **could not have resolved** a trend are hatched. Over
        a 12-bin Landsat panel that will be most of the "no pattern" area, and
        without the hatch the map reads as a finding of spatial stability when
        it is really a statement about series length.

    Args:
        zones: ``geopandas.GeoDataFrame`` with ``zone_id`` and geometry.
        ehsa: Output of
            :func:`colombo_uhi.spatial_stats.classify_emerging_hotspots`.
        params: Parsed params mapping.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the join leaves no rows.
    """
    merged = zones.merge(
        ehsa.assign(zone_id=ehsa["zone_id"].astype(str)), on="zone_id", how="left"
    )
    if merged.empty:
        raise ValueError("the EHSA table and the zone geometry share no zone_id")
    merged["category"] = merged["category"].fillna("no_pattern")
    merged["underpowered"] = merged["underpowered"].fillna(False).astype(bool)

    colours = spatial_palette(params, "ehsa")
    present = [name for name in colours if name in set(merged["category"])]

    footer_inches = 1.35
    # The EHSA legend can run to a dozen categories, so it needs more than one
    # row underneath the map.
    legend_inches = 0.34 * max(1, -(-len(present) // 4))
    figure, axes, height = _map_figure(
        merged, footer_inches, legend_inches=legend_inches
    )

    _zone_choropleth(
        axes,
        merged,
        "category",
        colours,
        hatch_mask=merged["underpowered"],
        hatch=str(params["spatial_stats"]["palettes"]["underpowered_hatch"]),
    )
    _category_legend(axes, colours, present, ncol=4)

    under = int(merged["underpowered"].sum())
    axes.set_title(
        title or f"Emerging hot spots - {under} zone(s) hatched as underpowered",
        fontsize=12,
    )
    figure.text(
        0.01,
        0.01,
        caveat_footer(
            params, ["lst_not_air_temp", "zonal_not_pixel", "sensitivity_reporting"]
        )
        + "\n- HATCHED zones are those whose series could not have detected a "
        "trend of the size sought. For them 'no pattern' means UNRESOLVABLE,\n"
        "  not stable. Gi* is standardised within each time bin, so a "
        "common-mode shift between bins cancels and only pattern change remains.",
        fontsize=7,
        va="bottom",
        ha="left",
        color="#444444",
    )
    return figure


def plot_ehsa_map(
    zones: Any,
    ehsa: "pd.DataFrame",
    out_path: str | Path,
    params: dict[str, Any],
    title: str | None = None,
) -> Path:
    """Write the emerging-hot-spot map. See :func:`build_ehsa_map_figure`."""
    return _save_figure(
        build_ehsa_map_figure(zones, ehsa, params, title=title), out_path
    )


def build_gwr_coefficient_figure(
    zones: Any,
    local_coefficients: "pd.DataFrame",
    params: dict[str, Any],
    terms: Sequence[str] | None = None,
    title: str | None = None,
) -> Any:
    """Build small multiples of GWR/MGWR local coefficients, one panel per term.

    .. warning::
        Zones where the local coefficient is not significant at the
        **multiple-testing-adjusted** critical t are hatched. A GWR fits one
        regression per zone, so an unadjusted local t-map overstates
        significance exactly the way an uncorrected pixel-wise p-map does.

    Args:
        zones: ``geopandas.GeoDataFrame`` with ``zone_id`` and geometry.
        local_coefficients: The ``local_coefficients`` frame from
            :func:`colombo_uhi.spatial_stats.gwr_model`.
        params: Parsed params mapping.
        terms: Terms to draw; defaults to every ``beta_*`` column except the
            intercept.
        title: Figure suptitle.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If no plottable term is found.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.figure import Figure

    if terms is None:
        terms = [
            column[len("beta_") :]
            for column in local_coefficients.columns
            if column.startswith("beta_") and column != "beta_intercept"
        ]
    terms = list(terms)
    if not terms:
        raise ValueError(
            "no local coefficient columns found; expected columns named "
            f"beta_<term>, got {sorted(local_coefficients.columns)}"
        )

    merged = zones.merge(
        local_coefficients.assign(
            zone_id=local_coefficients["zone_id"].astype(str)
        ),
        on="zone_id",
        how="left",
    )
    palette = [
        f"#{value}" for value in params["spatial_stats"]["palettes"]["gwr_diverging"]
    ]
    cmap = LinearSegmentedColormap.from_list("gwr", palette)

    columns = min(3, len(terms))
    rows = int(np.ceil(len(terms) / columns))
    footer_inches = 1.0
    panel_width = 4.6
    # Same reason as the single maps: with set_aspect("equal") a panel taller
    # than the data leaves blank paper under every subplot, multiplied by the
    # number of covariates. +0.5 in per row for the title and colour bar.
    panel_height = panel_width * map_aspect_ratio(zones) + 0.5
    height = panel_height * rows + footer_inches
    figure = Figure(figsize=(panel_width * columns, height))
    FigureCanvasAgg(figure)
    axes_grid = figure.subplots(rows, columns, squeeze=False)

    for index, term in enumerate(terms):
        axes = axes_grid[index // columns][index % columns]
        values = merged[f"beta_{term}"].astype("float64").to_numpy()
        limit = float(np.nanmax(np.abs(values))) or 1.0
        merged.plot(
            ax=axes,
            column=f"beta_{term}",
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            edgecolor="#ffffff",
            linewidth=0.1,
            legend=True,
            legend_kwds={"shrink": 0.6},
            missing_kwds={"color": "#dddddd"},
        )
        flag = f"sig_{term}"
        if flag in merged.columns:
            insignificant = ~merged[flag].fillna(False).astype(bool)
            if bool(insignificant.any()):
                merged[insignificant.to_numpy()].plot(
                    ax=axes,
                    facecolor="none",
                    edgecolor="#555555",
                    linewidth=0.2,
                    hatch="////",
                )
        axes.set_title(f"beta({term})", fontsize=10)
        axes.set_aspect("equal")
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in axes.spines.values():
            spine.set_visible(False)

    for index in range(len(terms), rows * columns):
        axes_grid[index // columns][index % columns].axis("off")

    figure.suptitle(title or "GWR local coefficients", fontsize=13)
    figure.text(
        0.01,
        0.01,
        caveat_footer(params, ["lst_not_air_temp", "zonal_not_pixel"])
        + "\n- HATCHED zones are NOT significant at the multiple-testing "
        "adjusted critical t. Predictors are standardised, so coefficients are\n"
        "  comparable across panels but are per standard deviation, not per "
        "native unit.",
        fontsize=7,
        va="bottom",
        ha="left",
        color="#444444",
    )
    figure.tight_layout(rect=(0, footer_inches / height, 1, 0.97))
    return figure


def plot_gwr_coefficients(
    zones: Any,
    local_coefficients: "pd.DataFrame",
    out_path: str | Path,
    params: dict[str, Any],
    terms: Sequence[str] | None = None,
    title: str | None = None,
) -> Path:
    """Write the GWR coefficient panels. See :func:`build_gwr_coefficient_figure`."""
    return _save_figure(
        build_gwr_coefficient_figure(
            zones, local_coefficients, params, terms=terms, title=title
        ),
        out_path,
    )


def build_maup_table_figure(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    title: str | None = None,
) -> Any:
    """Render the MAUP comparison as a figure-ready table.

    Rows that could not be estimated are shaded and keep their reason, because
    "this statistic stops existing at 13 units" is the finding, not a gap.

    Args:
        frame: Output of :func:`colombo_uhi.spatial_stats.maup_comparison`.
        params: Parsed params mapping.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the frame is empty.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if frame.empty:
        raise ValueError("the MAUP frame is empty; there is nothing to render")

    def _text(value: Any) -> str:
        if value is None or value != value:
            return ""
        return f"{value:.4g}" if isinstance(value, float) else str(value)

    display = frame.copy()
    display["value"] = [_text(value) for value in display["value"]]
    display["n_units"] = [_text(value) for value in display["n_units"]]
    # A refused statistic has no detail but does have a reason, and the reason is
    # the entire point of the row. Merge them into one column and wrap it, rather
    # than printing an empty cell beside a NaN.
    merged_detail = [
        _text(detail) if str(status) == "ok" else _text(reason)
        for status, detail, reason in zip(
            display["status"], display["detail"], display.get("reason", display["detail"])
        )
    ]
    display["detail"] = ["\n".join(textwrap.wrap(text, 62)) for text in merged_detail]
    wanted = ["statistic", "level", "n_units", "status", "value", "detail"]
    display = display[[column for column in wanted if column in display.columns]]

    # Row height must follow the WRAPPED detail, or a long refusal reason
    # overflows its cell and covers the row beneath it.
    line_counts = [max(1, text.count("\n") + 1) for text in display["detail"]]
    total_lines = sum(line_counts) + 1
    footer_inches = 1.05
    table_inches = 0.19 * total_lines + 0.35
    height = table_inches + 0.5 + footer_inches
    figure = Figure(figsize=(11.5, height))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.axis("off")

    table = axes.table(
        cellText=display.astype(str).values,
        colLabels=[column.replace("_", " ") for column in display.columns],
        cellLoc="left",
        # bbox pins the table to the axes, so the figure height set above is the
        # height it actually occupies instead of leaving a blank lower half.
        bbox=[0.0, 0.0, 1.0, 1.0],
        colWidths=[0.16, 0.06, 0.07, 0.11, 0.09, 0.51],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)

    status_index = list(display.columns).index("status")
    for row in range(len(display)):
        shaded = str(display.iloc[row, status_index]) != "ok"
        for column in range(len(display.columns)):
            cell = table[row + 1, column]
            cell.set_edgecolor("#cccccc")
            cell.set_height(line_counts[row] / total_lines)
            if shaded:
                cell.set_facecolor("#f2e6e6")
    for column in range(len(display.columns)):
        table[0, column].set_height(1.0 / total_lines)

    axes.set_title(
        title or "Aggregation-unit (MAUP) sensitivity", fontsize=12, pad=14
    )
    figure.text(
        0.01,
        0.01,
        caveat_footer(params, ["sensitivity_reporting", "zonal_not_pixel"])
        + "\n- Shaded rows are statistics that are NOT ESTIMABLE at that "
        "aggregation level. That is a reported result, not a missing value.",
        fontsize=7,
        va="bottom",
        ha="left",
        color="#444444",
    )
    figure.tight_layout(rect=(0, footer_inches / height, 1, 1))
    return figure


def plot_maup_table(
    frame: "pd.DataFrame",
    out_path: str | Path,
    params: dict[str, Any],
    title: str | None = None,
) -> Path:
    """Write the MAUP comparison table. See :func:`build_maup_table_figure`."""
    return _save_figure(build_maup_table_figure(frame, params, title=title), out_path)


def build_landscape_change_figure(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    metrics: Sequence[str] | None = None,
    title: str | None = None,
) -> Any:
    """Build the green-space fragmentation comparison across dates and schemes.

    Args:
        frame: Output of
            :func:`colombo_uhi.spatial_stats.build_landscape_frame`, one row per
            scheme and date.
        params: Parsed params mapping.
        metrics: Metric columns to draw; defaults to the four CLAUDE.md names
            plus the aggregation index.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the frame is empty or a requested metric is absent.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if frame.empty:
        raise ValueError("the landscape frame is empty; there is nothing to plot")
    chosen = list(
        metrics
        or (
            "class_area_ha",
            "patch_density_per_100ha",
            "edge_density_m_per_ha",
            "mean_patch_area_ha",
            "aggregation_index_pct",
        )
    )
    missing = [name for name in chosen if name not in frame.columns]
    if missing:
        raise ValueError(
            f"landscape frame is missing metric(s) {missing}; it has "
            f"{sorted(frame.columns)}"
        )

    work = frame.copy()
    work["label"] = [
        f"{scheme}\n{'' if year is None or year != year else int(year)}"
        for scheme, year in zip(work["scheme"], work["year"])
    ]

    footer_inches = 1.0
    columns = min(3, len(chosen))
    rows = int(np.ceil(len(chosen) / columns))
    height = 3.1 * rows + footer_inches
    figure = Figure(figsize=(4.2 * columns, height))
    FigureCanvasAgg(figure)
    axes_grid = figure.subplots(rows, columns, squeeze=False)

    for index, metric in enumerate(chosen):
        axes = axes_grid[index // columns][index % columns]
        values = work[metric].astype("float64").to_numpy()
        positions = np.arange(len(values))
        axes.bar(positions, values, color="#4d9221")
        axes.set_xticks(positions)
        axes.set_xticklabels(list(work["label"]), fontsize=7)
        axes.set_title(metric.replace("_", " "), fontsize=9)
        axes.grid(True, axis="y", alpha=0.3)

    for index in range(len(chosen), rows * columns):
        axes_grid[index // columns][index % columns].axis("off")

    scale = (
        float(frame["cell_size_m"].iloc[0])
        if "cell_size_m" in frame.columns
        else float(params["spatial_stats"]["landscape"]["raster_scale_m"])
    )
    figure.suptitle(title or "Green-space landscape metrics", fontsize=13)
    figure.text(
        0.01,
        0.01,
        caveat_footer(params, ["sensitivity_reporting"])
        + f"\n- EVERY metric here is scale dependent and was computed at "
        f"{scale:g} m. The same city at 30 m gives different patch counts, edge "
        "density and\n  aggregation index, so the grid size must be quoted with "
        "any figure. Cross-scheme differences are partly classifier differences.",
        fontsize=7,
        va="bottom",
        ha="left",
        color="#444444",
    )
    figure.tight_layout(rect=(0, footer_inches / height, 1, 0.95))
    return figure


def plot_landscape_change(
    frame: "pd.DataFrame",
    out_path: str | Path,
    params: dict[str, Any],
    metrics: Sequence[str] | None = None,
    title: str | None = None,
) -> Path:
    """Write the landscape-metrics figure. See :func:`build_landscape_change_figure`."""
    return _save_figure(
        build_landscape_change_figure(frame, params, metrics=metrics, title=title),
        out_path,
    )
