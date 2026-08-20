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

import math
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


def _readable_on(colour: Any) -> str:
    """Black or white, whichever is legible on ``colour``.

    Annotating a heatmap in one fixed colour leaves the darkest cells unreadable
    - the correlation panel's ``1.00`` diagonal disappeared into its own red in
    Colab run 3. Uses the ITU-R BT.601 luma the eye actually responds to, not a
    mean of the channels.

    Args:
        colour: An RGB or RGBA tuple in 0-1.

    Returns:
        ``"#111111"`` or ``"#ffffff"``.
    """
    red, green, blue = (float(channel) for channel in tuple(colour)[:3])
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    return "#111111" if luma > 0.55 else "#ffffff"


def _wrap_bullets(text: str | Sequence[str]) -> str:
    """Wrap free footer text to the same width and shape as :func:`caveat_footer`.

    Appending figure-specific footer text verbatim is what pushed the last line
    of two Phase 7 figures off the right edge of the canvas in Colab run 3, while
    every caveat above it wrapped correctly.

    **Pass a sequence, one string per bullet.** An earlier version took one
    concatenated string and split it on ``" - "``, which cannot tell a bullet
    boundary from a dash used as punctuation - and duly tore
    ``"the leave-one-out ablation - not the AHP weights - are what say..."`` into
    three bullets. A sequence says where the boundaries are instead of guessing.

    Args:
        text: One bullet per element, or a single string treated as one bullet
            (newlines still separate bullets).

    Returns:
        Newline-wrapped text, one wrapped bullet per line group.
    """
    raw = [text] if isinstance(text, str) else list(text)
    bullets: list[str] = []
    for entry in raw:
        for line in str(entry).split("\n"):
            collapsed = " ".join(line.split())
            if not collapsed:
                continue
            body = collapsed[2:].strip() if collapsed.startswith("- ") else collapsed
            bullets.append(f"- {body}")

    lines: list[str] = []
    for bullet in bullets:
        lines.extend(
            textwrap.wrap(bullet, width=CAVEAT_WRAP_CHARS, subsequent_indent="  ")
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


#: Raster density for the per-phase diagnostic figures in ``figures/``.
DIAGNOSTIC_DPI = 150


def _save_figure(figure: Any, out_path: str | Path, dpi: int = DIAGNOSTIC_DPI) -> Path:
    """Write a figure to disk, creating parent directories.

    Args:
        figure: A ``matplotlib.figure.Figure``.
        out_path: Destination ``.png`` path.
        dpi: Raster density. Defaults to the phase-diagnostic 150; Phase 8
            report figures pass :func:`report_dpi`, which reads ``report.dpi``.

    Returns:
        The path written.
    """
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=int(dpi))
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


#: Vertical inches one line of 7 pt footer text occupies, including leading.
FOOTER_LINE_INCHES = 0.125


def footer_inches(text: str, pad_inches: float = 0.22) -> float:
    """Vertical space a caveat footer actually needs.

    Reserving a fixed constant is what put the legend on top of the footer in
    Colab run 2: the LISA footer carries four caveats plus an explanatory line -
    about thirteen wrapped lines, well over the 1.15 in it had been given - so it
    overflowed upward into the legend band.

    Args:
        text: The assembled footer, as :func:`caveat_footer` returns it.
        pad_inches: Breathing room above and below.

    Returns:
        Height in inches.
    """
    lines = text.count("\n") + 1 if text else 0
    return lines * FOOTER_LINE_INCHES + pad_inches


def _map_figure(
    zones: Any,
    footer_text: str,
    legend_inches: float = 0.0,
    width_inches: float = 9.5,
    title_inches: float = 0.45,
) -> tuple[Any, Any, float, float]:
    """Create a figure whose map axes matches the data's shape.

    Args:
        zones: ``geopandas.GeoDataFrame`` the map will draw.
        footer_text: The caveat footer; its line count sizes the reserved band.
        legend_inches: Vertical space reserved for a legend below the map.
        width_inches: Figure width.
        title_inches: Vertical space reserved for the title.

    Returns:
        ``(figure, axes, height_inches, footer_inches)``.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    reserved = footer_inches(footer_text)
    map_inches = width_inches * map_aspect_ratio(zones)
    height = map_inches + legend_inches + reserved + title_inches
    figure = Figure(figsize=(width_inches, height))
    FigureCanvasAgg(figure)
    axes = figure.add_axes(
        (0.02, (reserved + legend_inches) / height, 0.96, map_inches / height)
    )
    return figure, axes, height, reserved


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
    footer = (
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
        "Benjamini-Hochberg adjusted permutation p-value."
    )
    figure, axes, height, reserved = _map_figure(
        merged, footer, legend_inches=0.45
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
    figure.text(0.01, 0.01, footer, fontsize=7, va="bottom", ha="left",
                color="#444444")
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

    footer = (
        caveat_footer(
            params, ["lst_not_air_temp", "within_epoch_only", "zonal_not_pixel"]
        )
        + "\n- Confidence classes are bins of the FDR-adjusted permutation "
        "p-value, so the legend means what it says after multiple testing."
    )
    figure, axes, height, reserved = _map_figure(
        merged, footer, legend_inches=0.45
    )

    _zone_choropleth(axes, merged, "confidence_class", colours)
    _category_legend(axes, colours, order, ncol=7)

    hot = int(merged["confidence_class"].astype(str).str.startswith("hot").sum())
    cold = int(merged["confidence_class"].astype(str).str.startswith("cold").sum())
    axes.set_title(
        title or f"Getis-Ord Gi* - {hot} hot and {cold} cold zones (FDR adjusted)",
        fontsize=12,
    )
    figure.text(0.01, 0.01, footer, fontsize=7, va="bottom", ha="left",
                color="#444444")
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

    footer = (
        caveat_footer(
            params, ["lst_not_air_temp", "zonal_not_pixel", "sensitivity_reporting"]
        )
        + "\n- HATCHED zones are those whose series could not have detected a "
        "trend of the size sought. For them 'no pattern' means UNRESOLVABLE,\n"
        "  not stable. Gi* is standardised within each time bin, so a "
        "common-mode shift between bins cancels and only pattern change remains."
    )
    # The EHSA legend can run to a dozen categories, so it needs more than one
    # row underneath the map.
    legend_inches = 0.34 * max(1, -(-len(present) // 4))
    figure, axes, height, reserved = _map_figure(
        merged, footer, legend_inches=legend_inches
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
    figure.text(0.01, 0.01, footer, fontsize=7, va="bottom", ha="left",
                color="#444444")
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
    footer = (
        caveat_footer(params, ["lst_not_air_temp", "zonal_not_pixel"])
        + "\n- HATCHED zones are NOT significant at the multiple-testing "
        "adjusted critical t. Predictors are standardised, so coefficients are\n"
        "  comparable across panels but are per standard deviation, not per "
        "native unit."
    )
    reserved = footer_inches(footer)
    panel_width = 4.6
    # Same reason as the single maps: with set_aspect("equal") a panel taller
    # than the data leaves blank paper under every subplot, multiplied by the
    # number of covariates. +0.5 in per row for the title and colour bar.
    panel_height = panel_width * map_aspect_ratio(zones) + 0.5
    height = panel_height * rows + reserved
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
    figure.text(0.01, 0.01, footer, fontsize=7, va="bottom", ha="left",
                color="#444444")
    figure.tight_layout(rect=(0, reserved / height, 1, 0.97))
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
    footer = (
        caveat_footer(params, ["sensitivity_reporting", "zonal_not_pixel"])
        + "\n- Shaded rows are statistics that are NOT ESTIMABLE at that "
        "aggregation level. That is a reported result, not a missing value."
    )
    reserved = footer_inches(footer)
    table_inches = 0.19 * total_lines + 0.35
    height = table_inches + 0.5 + reserved
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
    figure.text(0.01, 0.01, footer, fontsize=7, va="bottom", ha="left",
                color="#444444")
    figure.tight_layout(rect=(0, reserved / height, 1, 1))
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

    scale = (
        float(frame["cell_size_m"].iloc[0])
        if "cell_size_m" in frame.columns
        else float(params["spatial_stats"]["landscape"]["raster_scale_m"])
    )
    footer = (
        caveat_footer(params, ["sensitivity_reporting"])
        + f"\n- EVERY metric here is scale dependent and was computed at "
        f"{scale:g} m. The same city at 30 m gives different patch counts, edge "
        "density and\n  aggregation index, so the grid size must be quoted with "
        "any figure. Cross-scheme differences are partly classifier differences."
    )
    reserved = footer_inches(footer)
    columns = min(3, len(chosen))
    rows = int(np.ceil(len(chosen) / columns))
    height = 3.1 * rows + reserved
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

    figure.suptitle(title or "Green-space landscape metrics", fontsize=13)
    figure.text(0.01, 0.01, footer, fontsize=7, va="bottom", ha="left",
                color="#444444")
    figure.tight_layout(rect=(0, reserved / height, 1, 0.95))
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


# =============================================================================
# Phase 6 - conditional scenario projection figures
# =============================================================================
# Every predictive figure here takes a validation report and passes it through
# colombo_uhi.prediction.require_validated BEFORE drawing anything. A predictive
# figure without its metrics on it is not a figure this project ships, and
# refusing at the top of the BUILDER is the only way to make that true of the
# build_* functions the tests exercise as well as the plot_* wrappers.


def projection_caption(
    report: Mapping[str, Any] | None,
    params: dict[str, Any],
    keys: Sequence[str] = ("scenario_not_forecast", "lst_not_air_temp"),
    extra: str = "",
) -> str:
    """Assemble the footer for a predictive figure.

    Args:
        report: Validation report from
            :func:`colombo_uhi.prediction.build_validation_report`.
        params: Parsed params mapping.
        keys: Caveat keys from ``caveats``.
        extra: Figure-specific lines appended below.

    Returns:
        The caveat footer, the validation caption and ``extra``, joined. When
        validation was computed and FAILED, the caption's banner leads.

    Raises:
        colombo_uhi.prediction.ValidationMissing: If validation was never
            computed. A measured failure is stamped, not refused.
    """
    from colombo_uhi import prediction

    caption = prediction.validation_caption(report, params)
    caveats = caveat_footer(params, list(keys))

    # When validation FAILED, its banner leads the whole footer. The standing
    # caveats are true of every predictive figure; this one is true of THIS
    # product, and a reader has to meet it first.
    parts = (
        [caption, caveats] if failure_headline(report, params)
        else [caveats, caption]
    )
    if extra:
        parts.append(extra.strip("\n"))
    return "\n".join(part for part in parts if part)


def failure_headline(
    report: Mapping[str, Any] | None, params: dict[str, Any]
) -> str | None:
    """One-line banner when a product's validation was computed and FAILED.

    Args:
        report: Validation report.
        params: Parsed params mapping.

    Returns:
        The headline, or ``None`` when validation passed or was never computed.
        An absence is the caption's business; it refuses outright.
    """
    from colombo_uhi import prediction

    verdict = prediction.assess_validation(report, params)
    if verdict["present"] and not verdict["valid"]:
        return str(verdict["headline"])
    return None


def _suptitle(
    figure: Any,
    title: str,
    report: Mapping[str, Any] | None,
    params: dict[str, Any],
    fontsize: int = 12,
) -> None:
    """Set the figure title, marking a failed validation inside it.

    The footer leads with the failure too, but a reader skims titles first, and
    a product that did not pass validation has to announce that before anything
    else on the figure is read.
    """
    if failure_headline(report, params) is None:
        figure.suptitle(title, fontsize=fontsize)
        return
    # A SHORT single-line marker, and the whole title in red. The full reason
    # leads the footer. Anything longer wraps, and a wrapped suptitle lands on
    # the panel titles of the multi-panel figures - which is worse than terse.
    figure.suptitle(
        f"{title}   [FAILED VALIDATION]", fontsize=fontsize, color="#b2182b"
    )


def landcover_palette(params: dict[str, Any], scheme: str) -> dict[int, str]:
    """Class-code to colour mapping for a land-cover scheme.

    Args:
        params: Parsed params mapping.
        scheme: Key under ``prediction.palettes``.

    Returns:
        Mapping of integer class code to ``#rrggbb``.

    Raises:
        KeyError: If the scheme has no palette, naming those that do.
    """
    palettes = params["prediction"]["palettes"]
    entry = palettes.get(scheme)
    if not isinstance(entry, Mapping):
        raise KeyError(
            f"no class palette for scheme {scheme!r}; prediction.palettes "
            f"defines {sorted(k for k, v in palettes.items() if isinstance(v, Mapping))}"
        )
    return {
        int(code): "#" + str(colour).lstrip("#") for code, colour in entry.items()
    }


def build_observed_vs_predicted_figure(
    observed: Any,
    predicted: Any,
    params: dict[str, Any],
    report: Mapping[str, Any] | None,
    title: str | None = None,
    max_points: int = 6000,
) -> Any:
    """Held-out observed against predicted LST, with the metrics on the axes.

    The 1:1 line is drawn, not a fitted line. A regression line through this
    cloud would make a compressed prediction range look like a good fit; the
    1:1 line shows the compression, which a random forest does at both tails by
    construction.

    Args:
        observed: Held-out measured values.
        predicted: Model values for the same rows.
        params: Parsed params mapping.
        report: Validation report; stamped onto the figure.
        title: Figure title.
        max_points: Thinning cap for the scatter.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the arrays differ in length or are empty.
        colombo_uhi.prediction.ValidationMissing: If the report is incomplete.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    footer = projection_caption(
        report,
        params,
        keys=("scenario_not_forecast", "lst_not_air_temp"),
        extra=(
            "- The line is 1:1, NOT a fitted line. A random forest averages the "
            "training rows in each leaf, so it compresses\n  both tails: the "
            "coolest pixels come back too warm and the warmest too cool. That "
            "compression is visible against\n  1:1 and would be hidden by a "
            "regression line."
        ),
    )

    truth = np.asarray(observed, dtype="float64").ravel()
    guess = np.asarray(predicted, dtype="float64").ravel()
    if truth.shape != guess.shape:
        raise ValueError(
            f"observed and predicted must match, got {truth.shape} and "
            f"{guess.shape}"
        )
    if truth.size == 0:
        raise ValueError("cannot plot an empty prediction")

    keep = np.isfinite(truth) & np.isfinite(guess)
    truth, guess = truth[keep], guess[keep]
    if truth.size == 0:
        raise ValueError("every observed/predicted pair is non-finite")
    if truth.size > int(max_points):
        rng = np.random.default_rng(int(params["prediction"]["rf"]["random_seed"]))
        pick = rng.choice(truth.size, int(max_points), replace=False)
        truth, guess = truth[pick], guess[pick]

    reserved = footer_inches(footer)
    height = 5.6 + reserved
    figure = Figure(figsize=(6.4, height))
    FigureCanvasAgg(figure)
    axes = figure.add_axes((0.13, (reserved + 0.45) / height, 0.82, 4.9 / height))

    axes.scatter(truth, guess, s=6, alpha=0.28, color="#b2182b", linewidths=0)
    low = float(min(truth.min(), guess.min()))
    high = float(max(truth.max(), guess.max()))
    if high <= low:
        high = low + 1.0
    axes.plot([low, high], [low, high], color="#333333", linewidth=1.0, zorder=3)
    axes.set_xlim(low, high)
    axes.set_ylim(low, high)
    axes.set_xlabel("Observed LST (degC), held-out blocks", fontsize=9)
    axes.set_ylabel("Predicted LST (degC)", fontsize=9)
    axes.tick_params(labelsize=8)
    axes.grid(True, linewidth=0.3, color="#dddddd")
    axes.set_axisbelow(True)

    metrics = dict((report or {}).get("metrics") or {})
    shown = [
        f"{name.upper()} = {metrics[name]:.3f}"
        for name in ("rmse", "r2")
        if name in metrics
    ]
    axes.text(
        0.03,
        0.97,
        "\n".join([*shown, f"n = {truth.size:,}"]),
        transform=axes.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox={
            "facecolor": "#ffffff",
            "edgecolor": "#999999",
            "alpha": 0.85,
            "boxstyle": "round,pad=0.35",
        },
    )

    _suptitle(
        figure, title or "Random forest LST fit, held-out spatial blocks",
        report, params,
    )
    figure.text(
        0.01, 0.01, footer, fontsize=7, va="bottom", ha="left", color="#444444"
    )
    return figure


def plot_observed_vs_predicted(
    observed: Any,
    predicted: Any,
    out_path: str | Path,
    params: dict[str, Any],
    report: Mapping[str, Any] | None,
    title: str | None = None,
) -> Path:
    """Write the held-out scatter. See :func:`build_observed_vs_predicted_figure`."""
    return _save_figure(
        build_observed_vs_predicted_figure(
            observed, predicted, params, report, title=title
        ),
        out_path,
    )


def build_feature_importance_figure(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    report: Mapping[str, Any] | None,
    title: str | None = None,
) -> Any:
    """Permutation importance on the held-out blocks, with its spread.

    Args:
        frame: Output of
            :func:`colombo_uhi.prediction.permutation_importance_frame`.
        params: Parsed params mapping.
        report: Validation report; stamped onto the figure.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If a required column is absent or the table is empty.
        colombo_uhi.prediction.ValidationMissing: If the report is incomplete.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    required = ["predictor", "importance_mean", "importance_std"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(
            f"the importance table is missing {missing}; it has "
            f"{sorted(frame.columns)}"
        )
    if len(frame) == 0:
        raise ValueError("cannot plot an empty importance table")

    categorical = list(params["prediction"]["rf"]["categorical"])
    footer = projection_caption(
        report,
        params,
        keys=("scenario_not_forecast", "trend_not_causal"),
        extra=(
            "- PERMUTATION importance on held-out blocks, not impurity "
            "importance. Impurity importance favours predictors\n  with many "
            f"distinct values, which would flatter {categorical} regardless of "
            "what it contributes.\n"
            "- Importance is not causation. A predictor can rank high because "
            "it stands in for something unmeasured."
        ),
    )

    ordered = frame.sort_values("importance_mean")
    reserved = footer_inches(footer)
    panel = max(2.6, 0.34 * len(ordered) + 0.9)
    height = panel + reserved + 0.55
    figure = Figure(figsize=(8.4, height))
    FigureCanvasAgg(figure)
    axes = figure.add_axes(
        (0.20, (reserved + 0.35) / height, 0.76, panel / height)
    )

    colours = [
        "#67a9cf" if str(name) in categorical else "#2166ac"
        for name in ordered["predictor"]
    ]
    axes.barh(
        [str(name) for name in ordered["predictor"]],
        ordered["importance_mean"].to_numpy(dtype="float64"),
        xerr=ordered["importance_std"].to_numpy(dtype="float64"),
        color=colours,
        error_kw={"ecolor": "#555555", "elinewidth": 0.8, "capsize": 2},
    )
    axes.axvline(0.0, color="#333333", linewidth=0.8)
    axes.set_xlabel(
        "Increase in held-out error when the predictor is shuffled", fontsize=9
    )
    axes.tick_params(labelsize=8)
    axes.grid(True, axis="x", linewidth=0.3, color="#dddddd")
    axes.set_axisbelow(True)

    _suptitle(figure, title or "Predictor importance", report, params)
    figure.text(
        0.01, 0.01, footer, fontsize=7, va="bottom", ha="left", color="#444444"
    )
    return figure


def plot_feature_importance(
    frame: "pd.DataFrame",
    out_path: str | Path,
    params: dict[str, Any],
    report: Mapping[str, Any] | None,
    title: str | None = None,
) -> Path:
    """Write the importance figure. See :func:`build_feature_importance_figure`."""
    return _save_figure(
        build_feature_importance_figure(frame, params, report, title=title),
        out_path,
    )


def build_transition_matrix_figure(
    probabilities: Any,
    params: dict[str, Any],
    classes: Sequence[int],
    title: str | None = None,
    class_labels: Mapping[int, str] | None = None,
) -> Any:
    """Heatmap of the calibrated one-step transition probabilities.

    The diagonal is persistence, and it is normally the overwhelming majority of
    every row. That is exactly why a bare Kappa on a short interval flatters a
    projection, and reading this matrix is how it becomes obvious.

    .. note::
        No validation report is required here. A transition matrix is an
        OBSERVATION of what happened between two dates, not a projection.

    Args:
        probabilities: Row-stochastic matrix from
            :func:`colombo_uhi.prediction.transition_probabilities`.
        params: Parsed params mapping.
        classes: Class codes, in the matrix's row and column order.
        title: Figure title.
        class_labels: Override for the class names. Pass
            ``prediction.ca_markov.grouped_labels`` when the matrix was built on
            the grouped scheme, or the axes will be labelled with the raw
            Dynamic World legend and "Green" will read as "Trees".

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the matrix is not square or does not match ``classes``.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    table = np.asarray(probabilities, dtype="float64")
    codes = [int(code) for code in classes]
    if table.ndim != 2 or table.shape[0] != table.shape[1]:
        raise ValueError(f"probabilities must be square, got {table.shape}")
    if table.shape[0] != len(codes):
        raise ValueError(
            f"probabilities is {table.shape[0]}x{table.shape[0]} but "
            f"{len(codes)} class code(s) were given"
        )

    scheme = str(params["prediction"]["ca_markov"]["scheme"])
    names = dict(
        params["landcover"][scheme]["classes"] if class_labels is None
        else class_labels
    )
    labels = [str(names.get(code, code)) for code in codes]

    footer = caveat_footer(params, ["scenario_not_forecast"]) + (
        "\n- Rows are the earlier date, columns the later one; each row sums to "
        "1. The DIAGONAL is persistence.\n- This is an observation of ONE "
        "interval, not a law. Projecting it forward assumes the rates that "
        "produced it continue."
    )
    reserved = footer_inches(footer)
    panel = 0.55 * len(codes) + 2.0
    height = panel + reserved + 0.6
    figure = Figure(figsize=(max(7.0, panel + 2.0), height))
    FigureCanvasAgg(figure)
    axes = figure.add_axes(
        (0.22, (reserved + 0.35) / height, 0.62, panel / height)
    )

    image = axes.imshow(table, cmap="Blues", vmin=0.0, vmax=1.0)
    axes.set_xticks(range(len(codes)))
    axes.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes.set_yticks(range(len(codes)))
    axes.set_yticklabels(labels, fontsize=8)
    axes.set_xlabel("to", fontsize=9)
    axes.set_ylabel("from", fontsize=9)
    for row in range(len(codes)):
        for column in range(len(codes)):
            value = float(table[row, column])
            axes.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="#ffffff" if value > 0.55 else "#333333",
            )
    bar = figure.colorbar(image, ax=axes, fraction=0.04, pad=0.03)
    bar.set_label("transition probability", fontsize=9)
    bar.ax.tick_params(labelsize=8)

    figure.suptitle(title or "One-step land-cover transition matrix", fontsize=12)
    figure.text(
        0.01, 0.01, footer, fontsize=7, va="bottom", ha="left", color="#444444"
    )
    return figure


def plot_transition_matrix(
    probabilities: Any,
    out_path: str | Path,
    params: dict[str, Any],
    classes: Sequence[int],
    title: str | None = None,
    class_labels: Mapping[int, str] | None = None,
) -> Path:
    """Write the transition heatmap. See :func:`build_transition_matrix_figure`."""
    return _save_figure(
        build_transition_matrix_figure(
            probabilities, params, classes, title=title, class_labels=class_labels
        ),
        out_path,
    )


def build_lulc_validation_figure(
    initial: Any,
    observed: Any,
    projected: Any,
    params: dict[str, Any],
    report: Mapping[str, Any] | None,
    title: str | None = None,
    class_labels: Mapping[int, str] | None = None,
) -> Any:
    """Three land-cover panels - start, observed, projected - and the metrics.

    Args:
        initial: Class-code array at the start of the validated interval.
        observed: Observed class codes at the end of it.
        projected: Projected class codes for the same date.
        params: Parsed params mapping.
        report: Validation report; stamped onto the figure.
        title: Figure title.
        class_labels: Override for the legend names. Pass
            ``prediction.ca_markov.grouped_labels`` when the panels carry
            grouped codes, or the legend will read "Trees" for a class that is
            really trees, grass and shrub together.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the three arrays do not share a shape.
        colombo_uhi.prediction.ValidationMissing: If the report is incomplete.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    panels = {
        "Start of interval": np.asarray(initial),
        "Observed": np.asarray(observed),
        "Projected": np.asarray(projected),
    }
    shapes = {name: array.shape for name, array in panels.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"the three panels must share a shape, got {shapes}")

    scheme = str(params["prediction"]["ca_markov"]["scheme"])
    palette = landcover_palette(params, scheme)
    names = dict(
        params["landcover"][scheme]["classes"] if class_labels is None
        else class_labels
    )
    codes = sorted(palette)
    cmap = ListedColormap([palette[code] for code in codes])
    norm = BoundaryNorm([*codes, codes[-1] + 1], cmap.N)

    footer = projection_caption(
        report,
        params,
        keys=("scenario_not_forecast", "sensitivity_reporting"),
        extra=(
            "- Read Kappa against the no-change null, NOT on its own. Over a "
            "short interval most cells persist, so a\n  projection that changes "
            "nothing already scores highly. The figure of merit scores only the "
            "cells that CHANGED,\n  which is the number that says whether the "
            "model located change or merely copied the map."
        ),
    )
    reserved = footer_inches(footer)
    rows, columns = panels["Observed"].shape
    aspect = min(max(rows / max(columns, 1), 0.25), 3.0)
    panel_width = 3.5
    panel_height = panel_width * aspect
    height = panel_height + reserved + 1.4
    figure = Figure(figsize=(panel_width * 3 + 0.6, height))
    FigureCanvasAgg(figure)
    axes_list = figure.subplots(1, 3, squeeze=False)[0]
    figure.subplots_adjust(
        left=0.02,
        right=0.98,
        top=1.0 - 0.45 / height,
        bottom=(reserved + 0.85) / height,
        wspace=0.05,
    )

    for axes, (label, array) in zip(axes_list, panels.items()):
        axes.imshow(array, cmap=cmap, norm=norm, interpolation="nearest")
        axes.set_title(label, fontsize=10)
        axes.set_xticks([])
        axes.set_yticks([])

    present = sorted(set(np.unique(panels["Observed"]).tolist()) & set(codes))
    figure.legend(
        handles=[
            Patch(
                facecolor=palette[code],
                edgecolor="#666666",
                label=str(names.get(code, code)),
            )
            for code in present
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, (reserved + 0.05) / height),
        ncol=min(max(len(present), 1), 5),
        fontsize=7.5,
        frameon=False,
        handlelength=1.3,
    )
    _suptitle(
        figure, title or "Land-cover projection against the held-out year",
        report, params,
    )
    figure.text(
        0.01, 0.01, footer, fontsize=7, va="bottom", ha="left", color="#444444"
    )
    return figure


def plot_lulc_validation(
    initial: Any,
    observed: Any,
    projected: Any,
    out_path: str | Path,
    params: dict[str, Any],
    report: Mapping[str, Any] | None,
    title: str | None = None,
    class_labels: Mapping[int, str] | None = None,
) -> Path:
    """Write the land-cover validation panels. See :func:`build_lulc_validation_figure`."""
    return _save_figure(
        build_lulc_validation_figure(
            initial, observed, projected, params, report, title=title,
            class_labels=class_labels,
        ),
        out_path,
    )


def build_projected_lst_figure(
    surfaces: Mapping[str, Any],
    params: dict[str, Any],
    report: Mapping[str, Any] | None,
    title: str | None = None,
) -> Any:
    """Projected LST under each scenario, on one shared colour scale.

    The shared scale is not cosmetic: two panels with independent scales can
    make a 0.2 degC difference look like a 3 degC one.

    Args:
        surfaces: Mapping of scenario label to 2-D projected LST array.
        params: Parsed params mapping.
        report: Validation report; stamped onto the figure.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If no surfaces are given, they differ in shape, or every
            value is non-finite.
        colombo_uhi.prediction.ValidationMissing: If the report is incomplete.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if not surfaces:
        raise ValueError("at least one scenario surface is required")
    arrays = {
        str(label): np.asarray(array, dtype="float64")
        for label, array in surfaces.items()
    }
    shapes = {label: array.shape for label, array in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"scenario surfaces must share a shape, got {shapes}")

    footer = projection_caption(
        report,
        params,
        keys=("scenario_not_forecast", "lst_not_air_temp", "single_overpass"),
        extra=(
            "- Panels share ONE colour scale. Independent scales would make a "
            "0.2 degC difference look like a 3 degC one.\n"
            "- These are dry-season, ~10:30 local overpass surfaces. They say "
            "nothing about night-time conditions."
        ),
    )
    stacked = np.stack(list(arrays.values()))
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        raise ValueError("every scenario surface is entirely non-finite")
    low = float(np.percentile(finite, 2))
    high = float(np.percentile(finite, 98))
    if high <= low:
        high = low + 1.0

    reserved = footer_inches(footer)
    rows, columns = next(iter(arrays.values())).shape
    aspect = min(max(rows / max(columns, 1), 0.25), 3.0)
    panel_width = 4.4
    height = panel_width * aspect + reserved + 1.0
    figure = Figure(figsize=(panel_width * len(arrays) + 0.9, height))
    FigureCanvasAgg(figure)
    axes_list = figure.subplots(1, len(arrays), squeeze=False)[0]
    figure.subplots_adjust(
        left=0.02,
        right=0.90,
        top=1.0 - 0.45 / height,
        bottom=(reserved + 0.25) / height,
        wspace=0.06,
    )

    image = None
    for axes, (label, array) in zip(axes_list, arrays.items()):
        image = axes.imshow(
            np.ma.masked_invalid(array),
            cmap="inferno",
            vmin=low,
            vmax=high,
            interpolation="nearest",
        )
        axes.set_title(label, fontsize=10)
        axes.set_xticks([])
        axes.set_yticks([])
    bar = figure.colorbar(image, ax=list(axes_list), fraction=0.03, pad=0.02)
    bar.set_label("Projected LST (degC)", fontsize=9)
    bar.ax.tick_params(labelsize=8)

    _suptitle(
        figure, title or "Projected land surface temperature", report, params
    )
    figure.text(
        0.01, 0.01, footer, fontsize=7, va="bottom", ha="left", color="#444444"
    )
    return figure


def plot_projected_lst(
    surfaces: Mapping[str, Any],
    out_path: str | Path,
    params: dict[str, Any],
    report: Mapping[str, Any] | None,
    title: str | None = None,
) -> Path:
    """Write the scenario surfaces. See :func:`build_projected_lst_figure`."""
    return _save_figure(
        build_projected_lst_figure(surfaces, params, report, title=title),
        out_path,
    )


def build_scenario_difference_figure(
    difference: Any,
    params: dict[str, Any],
    report: Mapping[str, Any] | None,
    max_degc: float | None = None,
    title: str | None = None,
) -> Any:
    """Greening minus business as usual, on a symmetric diverging ramp.

    .. warning::
        The difference of two projections carries BOTH projections'
        uncertainty. It is the least certain product in this phase, not the
        most. It only looks clean because the two surfaces share a model and so
        share its errors, which partly cancel - and that cancellation is a
        property of the method, not evidence that the difference is precise.

    Args:
        difference: 2-D array of scenario minus baseline, in degC.
        params: Parsed params mapping.
        report: Validation report; stamped onto the figure.
        max_degc: Symmetric colour limit; defaults to
            ``prediction.palettes.scenario_difference_max_degc``.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the array is not 2-D or the limit is non-positive.
        colombo_uhi.prediction.ValidationMissing: If the report is incomplete.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.figure import Figure

    array = np.asarray(difference, dtype="float64")
    if array.ndim != 2:
        raise ValueError(f"difference must be 2-D, got shape {array.shape}")
    palettes = params["prediction"]["palettes"]
    limit = float(
        palettes["scenario_difference_max_degc"] if max_degc is None else max_degc
    )
    if limit <= 0:
        raise ValueError(f"max_degc must be positive, got {limit}")

    # A counterfactual-minus-observed difference and a projection-minus-
    # projection difference carry different things, and the footer must not
    # claim the wrong one.
    carries = (
        "- A COUNTERFACTUAL MINUS ITS OBSERVED BASELINE still carries the "
        "model's error twice: both surfaces come from the\n  same forest, so "
        "its errors partly cancel. That cancellation is a property of the "
        "method, not evidence that\n  the difference is precise."
        if str((report or {}).get("kind", "")) == "lst_scenario"
        else
        "- A DIFFERENCE OF TWO PROJECTIONS carries both projections' "
        "uncertainty. It looks clean because the two surfaces\n  share a "
        "model and so share its errors, which partly cancel - a property of "
        "the method, not evidence of precision."
    )
    footer = projection_caption(
        report,
        params,
        keys=("scenario_not_forecast", "lst_not_air_temp"),
        extra=(
            carries + "\n"
            "- Blue is cooler under the greening scenario. The magnitude is what "
            "the fitted LST-driver relationship implies for\n  the converted "
            "pixels; it is not a measured cooling."
        ),
    )
    colours = [
        "#" + str(colour).lstrip("#")
        for colour in palettes["scenario_difference"]
    ]
    cmap = LinearSegmentedColormap.from_list("scenario_difference", colours)
    cmap = cmap.with_extremes(bad="#dcdcdc")

    reserved = footer_inches(footer)
    rows, columns = array.shape
    aspect = min(max(rows / max(columns, 1), 0.25), 3.0)
    width = 6.4
    map_inches = width * aspect
    height = map_inches + reserved + 0.9
    figure = Figure(figsize=(width, height))
    FigureCanvasAgg(figure)
    axes = figure.add_axes(
        (0.02, (reserved + 0.20) / height, 0.84, map_inches / height)
    )

    image = axes.imshow(
        np.ma.masked_invalid(array),
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        interpolation="nearest",
    )
    axes.set_xticks([])
    axes.set_yticks([])
    bar = figure.colorbar(image, ax=axes, fraction=0.035, pad=0.02)
    bar.set_label("Scenario minus baseline (degC)", fontsize=9)
    bar.ax.tick_params(labelsize=8)

    _suptitle(
        figure, title or "Greening scenario minus business as usual",
        report, params,
    )
    figure.text(
        0.01, 0.01, footer, fontsize=7, va="bottom", ha="left", color="#444444"
    )
    return figure


def plot_scenario_difference(
    difference: Any,
    out_path: str | Path,
    params: dict[str, Any],
    report: Mapping[str, Any] | None,
    max_degc: float | None = None,
    title: str | None = None,
) -> Path:
    """Write the scenario difference map. See :func:`build_scenario_difference_figure`."""
    return _save_figure(
        build_scenario_difference_figure(
            difference, params, report, max_degc=max_degc, title=title
        ),
        out_path,
    )


# =============================================================================
# Phase 7 - greening priority figures
# =============================================================================
# Every builder here calls `greening.require_consistent` inside a try/except and,
# on failure, STAMPS the figure and draws it anyway. That is Phase 6's
# post-run-3 behaviour: a figure showing that judgements failed is exactly what
# the report needs, and refusing to draw it throws the evidence away. What no
# path allows is a priority figure silent about its consistency ratio.

#: Caveats every Phase 7 figure carries.
GREENING_CAVEATS: tuple[str, ...] = (
    "lst_not_air_temp",
    "zonal_not_pixel",
    "sensitivity_reporting",
    "mcda_weights_are_judgements",
)


def greening_failure_headline(
    report: Mapping[str, Any] | None, params: dict[str, Any]
) -> str | None:
    """One-line banner when AHP judgements were computed and failed.

    The Phase 6 counterpart, :func:`failure_headline`, reads a *validation*
    report. This reads an *AHP* report, and the two failures are different
    things: one says a model has no demonstrated skill, the other says a set of
    judgements is not self-consistent.

    Args:
        report: The mapping :func:`colombo_uhi.greening.ahp_weights` returned.
        params: Parsed params mapping.

    Returns:
        The headline, or ``None`` when the judgements pass or no report exists.
    """
    from colombo_uhi import greening

    if report is None:
        return None
    try:
        greening.require_consistent(report, params)
    except greening.InconsistentJudgements as failure:
        ratio = float(report.get("consistency_ratio", float("nan")))
        maximum = float(report.get("consistency_ratio_max", float("nan")))
        # A failing consistency ratio LEADS, even when the matrix is also
        # degenerate. The two can co-occur - a perfect 3-cycle at equal strength
        # is symmetric, so it scores a huge CR *and* returns equal weights - and
        # of the two, "these judgements contradict each other" is the
        # substantive finding. Degeneracy is the interesting failure only when
        # the ratio PASSES, which is the trap it exists to catch: a CR near zero
        # reading as "the judgements were good" when it means none were made.
        if math.isfinite(ratio) and ratio > maximum:
            return f"*** INCONSISTENT JUDGEMENTS (CR = {ratio:.3f} > {maximum:.3f}) ***"
        if report.get("degenerate"):
            return (
                "*** NO JUDGEMENT WAS MADE: the pairwise weights are effectively "
                f"equal (spread {float(report.get('weight_spread', 0.0)):.4f}) ***"
            )
        return f"*** INCONSISTENT JUDGEMENTS: {failure} ***"
    except ValueError:
        return None
    return None


def greening_caption(
    report: Mapping[str, Any] | None,
    params: dict[str, Any],
    keys: Sequence[str] = GREENING_CAVEATS,
    extra: str | Sequence[str] = (),
) -> str:
    """Assemble the footer for a greening-priority figure.

    Args:
        report: The mapping :func:`colombo_uhi.greening.ahp_weights` returned.
        params: Parsed params mapping.
        keys: Caveat keys from ``caveats``.
        extra: Figure-specific lines appended below.

    Returns:
        The caveat footer, a line stating the consistency ratio and the
        normalisation, and ``extra``. When the judgements failed, the failure
        banner leads the whole footer.
    """
    caveats = caveat_footer(params, list(keys))

    if report is not None:
        ratio = float(report.get("consistency_ratio", float("nan")))
        maximum = float(report.get("consistency_ratio_max", float("nan")))
        verdict = "PASSES" if report.get("consistent") else "FAILS"
        summary = _wrap_bullets(
            f"- AHP consistency ratio {ratio:.4f} against a {maximum:.2f} "
            f"threshold ({verdict}); criteria normalised by "
            f"{params['greening']['normalisation']['method']}."
        )
    else:
        summary = ""

    banner = greening_failure_headline(report, params)
    parts = [banner, summary, caveats] if banner else [caveats, summary]
    if extra:
        # *** WRAP IT. *** Colab run 3 shipped two figures whose final footer
        # line ran off the right edge of the canvas, because `extra` was appended
        # verbatim while caveat_footer wrapped everything above it.
        parts.append(_wrap_bullets(extra))
    return "\n".join(part for part in parts if part)


def _greening_suptitle(
    figure: Any,
    title: str,
    report: Mapping[str, Any] | None,
    params: dict[str, Any],
    fontsize: int = 12,
) -> None:
    """Set the figure title, marking failed judgements inside it."""
    if greening_failure_headline(report, params) is None:
        figure.suptitle(title, fontsize=fontsize)
        return
    figure.suptitle(
        f"{title}   [INCONSISTENT JUDGEMENTS]", fontsize=fontsize, color="#b2182b"
    )


def priority_palette(params: dict[str, Any]) -> list[str]:
    """The greening-priority colour ramp, hex prefixed with ``#``.

    Args:
        params: Parsed params mapping.

    Returns:
        Colours from low to high priority.
    """
    return [
        "#" + str(colour).lstrip("#")
        for colour in params["greening"]["palettes"]["priority"]
    ]


def compliance_palette(params: dict[str, Any]) -> dict[str, str]:
    """The 3-30-300 compliance palette, hex prefixed with ``#``.

    Args:
        params: Parsed params mapping.

    Returns:
        Mapping of compliance category to colour. Five entries, never two: the
        "3" of the rule is unmeasured, and a two-colour pass/fail legend would
        imply it had been checked.
    """
    return {
        str(key): "#" + str(value).lstrip("#")
        for key, value in params["greening"]["palettes"]["compliance"].items()
    }


def build_greening_priority_map_figure(
    zones: Any,
    ranked: "pd.DataFrame",
    params: dict[str, Any],
    ahp_report: Mapping[str, Any] | None = None,
    score_column: str = "score_ahp",
    title: str | None = None,
    label_top_n: int | None = None,
) -> Any:
    """Build the greening-priority choropleth.

    Two hatches, deliberately different, because they say opposite things: a
    wetland-adjacent division is a **policy opportunity**, and a below-coverage
    division is one the data could not see properly. Conflating them on one
    figure would be worse than omitting both.

    Args:
        zones: ``geopandas.GeoDataFrame`` with ``zone_id`` and geometry.
        ranked: The priority table.
        params: Parsed params mapping.
        ahp_report: The mapping :func:`colombo_uhi.greening.ahp_weights` returned.
        score_column: Score to colour by.
        title: Figure title.
        label_top_n: Label this many highest-ranked divisions with leader lines
            into a column beside the map. ``0`` labels none; ``None`` labels none
            too, so the Phase 7 diagnostic keeps the appearance it was signed off
            with, and Phase 8 opts in explicitly with ``report.label_top_n``.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the frame is empty or the score column is absent.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if ranked.empty:
        raise ValueError("the priority frame is empty; there is nothing to map")
    if score_column not in ranked.columns:
        raise ValueError(
            f"no {score_column!r} column; the frame has {sorted(ranked.columns)}"
        )

    columns = [
        column
        for column in (
            "zone_id",
            score_column,
            "priority",
            "wetland_policy_flag",
            "below_land_coverage_floor",
            "status",
            # The rank and the division name are what label_top_n draws. Leaving
            # them out of the merge made annotate_top_zones find no rank column
            # and silently label nothing - the map rendered, correctly in every
            # other respect, with no labels and no error.
            "rank_ahp",
            "rank",
            "adm4_name",
        )
        if column in ranked.columns
    ]
    frame = zones.copy()
    frame["zone_id"] = frame["zone_id"].astype(str)
    # suffixes: the zone layer already carries adm4_name, and an unsuffixed
    # collision would silently rename BOTH columns to adm4_name_x / _y.
    merged = frame.merge(
        ranked[columns].assign(zone_id=ranked["zone_id"].astype(str)),
        on="zone_id",
        how="left",
        suffixes=("", "_ranked"),
    )

    extra = (
        "Outlined divisions are the exported top-N priority set. Divisions "
        "hatched '///' are within or beside mapped wetland, where wetland "
        "protection is an existing policy instrument; divisions hatched 'xxx' "
        "fell below the land-cover coverage floor: less of their land was "
        "classified than the floor requires. They are FLAGGED, not removed, "
        "because that floor gates nothing that enters the score - every "
        "criterion carries its own minimum-pixel gate.",
    )
    footer = greening_caption(ahp_report, params, extra=extra)
    # This map carries TWO legends below the axes - a continuous colourbar for
    # the score and a patch legend for the overlays - and in Colab run 3 they
    # landed on each other, the colourbar's tick labels reading through the
    # legend text. They need a band deep enough for both, and the patch legend
    # has to clear the colourbar AND its axis label.
    # Sized WITH the label gutter when labelling, so the map keeps its size and
    # the labels get their own column rather than eating into it.
    figure, axes, height, reserved = _map_figure(
        _PaddedBounds(merged, LABEL_GUTTER_FRACTION) if label_top_n else merged,
        footer,
        legend_inches=1.35,
    )

    colours = priority_palette(params)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("greening_priority", colours)
    merged.plot(
        ax=axes,
        column=score_column,
        cmap=cmap,
        edgecolor="#ffffff",
        linewidth=0.15,
        missing_kwds={"color": "#f0f0f0", "edgecolor": "#ffffff", "linewidth": 0.15},
        legend=True,
        legend_kwds={
            "label": "Greening priority score (higher = higher priority)",
            "orientation": "horizontal",
            "shrink": 0.6,
            "pad": 0.03,
            "fraction": 0.03,
        },
    )
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)

    handles: list[Any] = []
    if "priority" in merged.columns:
        selected = merged["priority"].fillna(False).astype(bool)
        if bool(selected.any()):
            merged[selected.to_numpy()].plot(
                ax=axes, facecolor="none", edgecolor="#111111", linewidth=0.9
            )
            handles.append(
                Line2D([], [], color="#111111", linewidth=1.2, label="Top-N priority")
            )
    if "wetland_policy_flag" in merged.columns:
        wetland = merged["wetland_policy_flag"].fillna(False).astype(bool)
        if bool(wetland.any()):
            merged[wetland.to_numpy()].plot(
                ax=axes,
                facecolor="none",
                edgecolor="#2c7fb8",
                linewidth=0.3,
                hatch=str(params["greening"]["palettes"]["wetland_hatch"]),
            )
            handles.append(
                Patch(
                    facecolor="none",
                    edgecolor="#2c7fb8",
                    hatch="///",
                    label="Within / beside wetland",
                )
            )
    if "below_land_coverage_floor" in merged.columns:
        floored = merged["below_land_coverage_floor"].fillna(False).astype(bool)
        if bool(floored.any()):
            merged[floored.to_numpy()].plot(
                ax=axes,
                facecolor="none",
                edgecolor="#666666",
                linewidth=0.3,
                hatch="xxx",
            )
            handles.append(
                Patch(
                    facecolor="none",
                    edgecolor="#666666",
                    hatch="xxx",
                    label="Land-cover coverage below floor",
                )
            )
    if handles:
        # Well below the colourbar and its axis label, not level with them.
        axes.legend(
            handles=handles,
            fontsize=7.5,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.19),
            frameon=False,
            ncol=3,
        )

    if label_top_n:
        annotate_top_zones(axes, merged, params, top_n=int(label_top_n))

    _greening_suptitle(
        figure,
        title or "Greening priority by GN division (AHP weighted overlay)",
        ahp_report,
        params,
    )
    figure.text(
        0.01, 0.01, footer, fontsize=7, va="bottom", ha="left", color="#444444"
    )
    return figure


def plot_greening_priority_map(
    zones: Any,
    ranked: "pd.DataFrame",
    out_path: str | Path,
    params: dict[str, Any],
    ahp_report: Mapping[str, Any] | None = None,
    score_column: str = "score_ahp",
    title: str | None = None,
    label_top_n: int | None = None,
) -> Path:
    """Write the priority map. See :func:`build_greening_priority_map_figure`."""
    return _save_figure(
        build_greening_priority_map_figure(
            zones, ranked, params, ahp_report, score_column=score_column,
            title=title, label_top_n=label_top_n,
        ),
        out_path,
    )


def build_ahp_weights_figure(
    ahp_frame: "pd.DataFrame",
    ahp_report: Mapping[str, Any],
    params: dict[str, Any],
    matrix: Any = None,
    names: Sequence[str] | None = None,
    title: str | None = None,
) -> Any:
    """Build the AHP weights figure: the judgements, the weights, and the CR.

    The pairwise matrix is drawn on a **logarithmic** colour scale, because the
    Saaty scale is multiplicative: 9 and 1/9 are equally far from 1, and a linear
    ramp would make "nine times more important" look eight units away from equal
    while "one ninth" looked like almost nothing.

    Args:
        ahp_frame: Output of :func:`colombo_uhi.greening.build_ahp_frame`.
        ahp_report: The mapping :func:`colombo_uhi.greening.ahp_weights` returned.
        params: Parsed params mapping.
        matrix: Optional pairwise matrix to draw as a heatmap.
        names: Criterion names in matrix order.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the weights frame is empty.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LogNorm
    from matplotlib.figure import Figure

    if ahp_frame.empty:
        raise ValueError("the AHP frame is empty; there are no weights to draw")

    extra = (
        "The pairwise matrix is drawn on a LOGARITHMIC colour scale: the Saaty "
        "scale is multiplicative, so 9 and 1/9 are equally far from 1.",
        "Row geometric means are overlaid on the weight bars; the two agreeing "
        "is itself evidence of consistency.",
    )
    footer = greening_caption(ahp_report, params, extra=extra)
    reserved = footer_inches(footer)
    height = 4.4 + reserved
    figure = Figure(figsize=(11.0, height))
    FigureCanvasAgg(figure)

    panels = 2 if matrix is not None else 1
    axes_list = figure.subplots(1, panels, squeeze=False)[0]

    if matrix is not None:
        heat_axes = axes_list[0]
        values = np.asarray(matrix, dtype=float)
        labels = (
            [str(name) for name in names]
            if names is not None
            else list(ahp_report.get("names", range(values.shape[0])))
        )
        image = heat_axes.imshow(
            values, cmap="PuOr_r", norm=LogNorm(vmin=1 / 9, vmax=9.0)
        )
        heat_axes.set_xticks(range(len(labels)))
        heat_axes.set_yticks(range(len(labels)))
        heat_axes.set_xticklabels(
            [label.replace("_", "\n") for label in labels], fontsize=7
        )
        heat_axes.set_yticklabels(
            [label.replace("_", " ") for label in labels], fontsize=7
        )
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                entry = values[row, col]
                text = f"{entry:.0f}" if entry >= 1 else f"1/{1 / entry:.0f}"
                heat_axes.text(
                    col,
                    row,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=_readable_on(image.cmap(image.norm(entry))),
                )
        heat_axes.set_title(
            "Pairwise judgements (row is how many times\nmore important than column)",
            fontsize=9,
        )
        figure.colorbar(image, ax=heat_axes, fraction=0.046, pad=0.04)

    bar_axes = axes_list[-1]
    order = ahp_frame.iloc[::-1]
    positions = np.arange(len(order))
    bar_axes.barh(
        positions, order["weight"], color="#4575b4", edgecolor="#274b6d", height=0.62
    )
    if "weight_geometric" in order.columns:
        bar_axes.scatter(
            order["weight_geometric"],
            positions,
            marker="|",
            s=180,
            color="#d73027",
            zorder=3,
            label="Row geometric mean",
        )
        bar_axes.legend(fontsize=7, frameon=False, loc="lower right")
    bar_axes.set_yticks(positions)
    bar_axes.set_yticklabels(
        [str(name).replace("_", " ") for name in order["criterion"]], fontsize=8
    )
    for position, weight in zip(positions, order["weight"]):
        bar_axes.text(
            float(weight) + 0.006, position, f"{float(weight):.3f}", fontsize=7.5,
            va="center",
        )
    bar_axes.set_xlim(0, float(ahp_frame["weight"].max()) * 1.28)
    bar_axes.set_xlabel("Criterion weight (principal eigenvector)", fontsize=8.5)
    bar_axes.set_title("Derived weights", fontsize=9)
    for spine in ("top", "right"):
        bar_axes.spines[spine].set_visible(False)

    ratio = float(ahp_report["consistency_ratio"])
    maximum = float(ahp_report["consistency_ratio_max"])
    passed = bool(ahp_report.get("consistent")) and not bool(
        ahp_report.get("degenerate")
    )
    banner = (
        f"lambda_max = {float(ahp_report['lambda_max']):.4f}    "
        f"CI = {float(ahp_report['consistency_index']):.4f}    "
        f"RI = {float(ahp_report['random_index']):.2f}    "
        f"CR = {ratio:.4f}  ({'PASS' if passed else 'INCONSISTENT'} at {maximum:.2f})"
    )
    figure.text(
        0.5,
        (reserved + 0.12) / height,
        banner,
        ha="center",
        fontsize=8.5,
        color="#1a9850" if passed else "#b2182b",
        weight="bold" if not passed else "normal",
    )

    _greening_suptitle(
        figure,
        title or "AHP criterion weights and consistency",
        ahp_report,
        params,
    )
    figure.text(
        0.01, 0.01, footer, fontsize=7, va="bottom", ha="left", color="#444444"
    )
    figure.tight_layout(rect=(0, (reserved + 0.3) / height, 1, 0.93))
    return figure


def plot_ahp_weights(
    ahp_frame: "pd.DataFrame",
    ahp_report: Mapping[str, Any],
    out_path: str | Path,
    params: dict[str, Any],
    matrix: Any = None,
    names: Sequence[str] | None = None,
    title: str | None = None,
) -> Path:
    """Write the AHP weights figure. See :func:`build_ahp_weights_figure`."""
    return _save_figure(
        build_ahp_weights_figure(
            ahp_frame, ahp_report, params, matrix=matrix, names=names, title=title
        ),
        out_path,
    )


def build_ranking_comparison_figure(
    left: "pd.DataFrame",
    right: "pd.DataFrame",
    comparison: Mapping[str, Any],
    params: dict[str, Any],
    left_name: str = "AHP weighted overlay",
    right_name: str = "TOPSIS",
    left_rank: str = "rank_ahp",
    right_rank: str = "rank_topsis",
    shifts: "pd.DataFrame | None" = None,
    ahp_report: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> Any:
    """Build the AHP-vs-TOPSIS rank comparison.

    Args:
        left: First ranking.
        right: Second ranking.
        comparison: Output of :func:`colombo_uhi.greening.compare_rankings`.
        params: Parsed params mapping.
        left_name: Label for the first method.
        right_name: Label for the second.
        left_rank: Rank column in ``left``.
        right_rank: Rank column in ``right``.
        shifts: Optional biggest-mover table to label.
        ahp_report: AHP report, for the footer.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If either ranking is empty.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if left.empty or right.empty:
        raise ValueError("a ranking is empty; there is nothing to compare")

    merged = left[["zone_id", left_rank]].merge(
        right[["zone_id", right_rank]], on="zone_id", how="inner"
    )
    top_n = int(comparison.get("top_n", params["greening"]["top_n"]))

    extra = (
        "Rank agreement is a ROBUSTNESS check, not a validation: two methods run "
        "on the same five criteria agreeing tells you the ranking is stable "
        "under the method, not that the criteria are the right ones.",
    )
    footer = greening_caption(ahp_report, params, extra=extra)
    reserved = footer_inches(footer)
    height = 5.2 + reserved
    figure = Figure(figsize=(9.0, height))
    FigureCanvasAgg(figure)
    axes = figure.subplots()

    axes.axvspan(0.5, top_n + 0.5, color="#fdae61", alpha=0.13, zorder=0)
    axes.axhspan(0.5, top_n + 0.5, color="#fdae61", alpha=0.13, zorder=0)
    limit = int(max(merged[left_rank].max(), merged[right_rank].max()))
    axes.plot([1, limit], [1, limit], color="#999999", linewidth=0.8, zorder=1)
    axes.scatter(
        merged[left_rank], merged[right_rank], s=12, color="#2166ac", alpha=0.7, zorder=2
    )

    if shifts is not None and not shifts.empty:
        # Alternate the offset. The biggest movers cluster - they are the zones
        # the two methods disagree about, and those sit together - so a single
        # fixed offset stacked five labels on top of each other in Colab run 3.
        offsets = ((9, 8), (9, -14), (-9, 14), (-9, -20))
        for index, (_, row) in enumerate(shifts.head(6).iterrows()):
            axes.annotate(
                str(row["zone_id"]),
                (float(row[left_rank]), float(row[right_rank])),
                fontsize=6.5,
                color="#b2182b",
                xytext=offsets[index % len(offsets)],
                textcoords="offset points",
                ha="left" if index % 4 < 2 else "right",
            )

    axes.set_xlabel(f"{left_name} rank", fontsize=9)
    axes.set_ylabel(f"{right_name} rank", fontsize=9)
    axes.invert_xaxis()
    axes.invert_yaxis()
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)

    lines = [
        f"Spearman rho = {float(comparison['spearman_rho']):.4f}",
    ]
    if "kendall_tau" in comparison:
        lines.append(f"Kendall tau = {float(comparison['kendall_tau']):.4f}")
    if "top_n_overlap" in comparison:
        lines.append(
            f"Top-{top_n} overlap = {int(comparison['top_n_overlap'])} "
            f"(Jaccard {float(comparison['top_n_jaccard']):.3f})"
        )
    lines.append(f"Median |rank shift| = {float(comparison['median_abs_shift']):.1f}")
    lines.append(f"Max |rank shift| = {float(comparison['max_abs_shift']):.0f}")
    axes.text(
        0.02,
        0.02,
        "\n".join(lines),
        transform=axes.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"facecolor": "#ffffff", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.4"},
    )

    _greening_suptitle(
        figure,
        title or f"{left_name} against {right_name}",
        ahp_report,
        params,
    )
    figure.text(
        0.01, 0.01, footer, fontsize=7, va="bottom", ha="left", color="#444444"
    )
    figure.tight_layout(rect=(0, reserved / height, 1, 0.94))
    return figure


def plot_ranking_comparison(
    left: "pd.DataFrame",
    right: "pd.DataFrame",
    comparison: Mapping[str, Any],
    out_path: str | Path,
    params: dict[str, Any],
    left_name: str = "AHP weighted overlay",
    right_name: str = "TOPSIS",
    left_rank: str = "rank_ahp",
    right_rank: str = "rank_topsis",
    shifts: "pd.DataFrame | None" = None,
    ahp_report: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> Path:
    """Write the ranking comparison. See :func:`build_ranking_comparison_figure`."""
    return _save_figure(
        build_ranking_comparison_figure(
            left,
            right,
            comparison,
            params,
            left_name=left_name,
            right_name=right_name,
            left_rank=left_rank,
            right_rank=right_rank,
            shifts=shifts,
            ahp_report=ahp_report,
            title=title,
        ),
        out_path,
    )


def build_compliance_map_figure(
    zones: Any,
    compliance: "pd.DataFrame",
    params: dict[str, Any],
    ahp_report: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> Any:
    """Build the 3-30-300 compliance map.

    Five categories, never a pass/fail pair. The "3" of the rule - three trees
    visible from every home - cannot be measured from satellite data at all, so a
    two-colour legend would imply it had been checked.

    Args:
        zones: ``geopandas.GeoDataFrame`` with ``zone_id`` and geometry.
        compliance: Output of :func:`colombo_uhi.greening.compliance_3_30_300`.
        params: Parsed params mapping.
        ahp_report: AHP report, for the footer.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the compliance frame is empty or has no verdict column.
    """
    from colombo_uhi import greening as greening_module

    if compliance.empty:
        raise ValueError("the compliance frame is empty; there is nothing to map")
    if "compliance" not in compliance.columns:
        raise ValueError(
            f"no 'compliance' column; the frame has {sorted(compliance.columns)}"
        )

    frame = zones.copy()
    frame["zone_id"] = frame["zone_id"].astype(str)
    columns = [
        column
        for column in ("zone_id", "compliance", "canopy_pct", "pop_within_300m_pct")
        if column in compliance.columns
    ]
    merged = frame.merge(
        compliance[columns].assign(zone_id=compliance["zone_id"].astype(str)),
        on="zone_id",
        how="left",
    )
    merged["compliance"] = merged["compliance"].fillna("not_assessable")

    target = int(params["greening"]["rule_3_30_300"]["canopy"]["target_pct"])
    distance = int(
        params["greening"]["rule_3_30_300"]["green_space"]["service_distance_m"]
    )
    detour = greening_module.detour_distance_m(params)
    extra = (
        f"'{target}' is the tree-class share of a "
        f"{int(params['greening']['landcover_scale_m'])} m modal classification, "
        "NOT crown cover from a canopy-height model.",
        f"'{distance}' counts residents within {distance} m; the detour variant "
        f"at {detour:.0f} m is in the exported table beside it.",
        "The '3' of 3-30-300 is NOT MEASURABLE from satellite data and is "
        "reported as unmeasured, which is why there are five categories rather "
        "than a pass/fail flag.",
    )
    footer = greening_caption(
        ahp_report,
        params,
        keys=(*GREENING_CAVEATS, "euclidean_not_network"),
        extra=extra,
    )
    figure, axes, height, reserved = _map_figure(merged, footer, legend_inches=0.5)

    colours = compliance_palette(params)
    _zone_choropleth(axes, merged, "compliance", colours)
    _category_legend(
        axes, colours, list(greening_module.COMPLIANCE_CATEGORIES), ncol=5
    )

    _greening_suptitle(
        figure,
        title or f"3-30-300 compliance by GN division ({target} % canopy, {distance} m access)",
        ahp_report,
        params,
    )
    figure.text(
        0.01, 0.01, footer, fontsize=7, va="bottom", ha="left", color="#444444"
    )
    return figure


def plot_compliance_map(
    zones: Any,
    compliance: "pd.DataFrame",
    out_path: str | Path,
    params: dict[str, Any],
    ahp_report: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> Path:
    """Write the 3-30-300 compliance map. See :func:`build_compliance_map_figure`."""
    return _save_figure(
        build_compliance_map_figure(
            zones, compliance, params, ahp_report=ahp_report, title=title
        ),
        out_path,
    )


def build_criterion_panel_figure(
    zones: Any,
    prepared: "pd.DataFrame",
    params: dict[str, Any],
    criteria: Sequence[str] | None = None,
    correlation: "pd.DataFrame | None" = None,
    ahp_report: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> Any:
    """Small multiples of the normalised criteria, plus their correlation matrix.

    **This is the figure that makes the collinearity visible**, and it belongs in
    the report. Over Colombo, ``rho(LST, green fraction) = -0.9147``: "high LST"
    and "low vegetation" are very nearly one variable, so five criterion maps
    that all look the same are not a redundancy in the figure - they are the
    finding.

    Args:
        zones: ``geopandas.GeoDataFrame`` with ``zone_id`` and geometry.
        prepared: Output of :func:`colombo_uhi.greening.prepare_criteria`.
        params: Parsed params mapping.
        criteria: Optional criterion subset.
        correlation: Optional matrix from
            :func:`colombo_uhi.greening.criterion_correlation`, drawn as a final
            panel.
        ahp_report: AHP report, for the footer.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If a normalised column is absent.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from colombo_uhi import greening as greening_module

    definitions = greening_module.resolve_criteria(params, criteria)
    names = [str(entry["name"]) for entry in definitions]
    absent = [
        f"{name}_norm" for name in names if f"{name}_norm" not in prepared.columns
    ]
    if absent:
        raise ValueError(f"{absent} not in the frame; pass prepare_criteria output")

    frame = zones.copy()
    frame["zone_id"] = frame["zone_id"].astype(str)
    merged = frame.merge(
        prepared[["zone_id", *[f"{name}_norm" for name in names]]].assign(
            zone_id=prepared["zone_id"].astype(str)
        ),
        on="zone_id",
        how="left",
    )

    extra = (
        "Every panel is on the SAME normalised scale, so panels that look alike "
        "are measuring the same underlying variable. Over Colombo the criteria "
        "are strongly intercorrelated, and the correlation panel plus the "
        "leave-one-out ablation - not the AHP weights - are what say how much "
        "the multi-criteria method adds over ranking by heat alone.",
    )
    footer = greening_caption(ahp_report, params, extra=extra)
    reserved = footer_inches(footer)

    panels = len(names) + (1 if correlation is not None else 0)
    cols = min(3, panels)
    rows = int(np.ceil(panels / cols))
    panel_inches = 3.0
    height = rows * panel_inches + reserved + 0.55
    figure = Figure(figsize=(cols * panel_inches + 0.8, height))
    FigureCanvasAgg(figure)
    axes_grid = figure.subplots(rows, cols, squeeze=False)
    flat = [axes_grid[row][col] for row in range(rows) for col in range(cols)]

    labels = {str(entry["name"]): str(entry["label"]) for entry in definitions}
    for axes, name in zip(flat, names):
        merged.plot(
            ax=axes,
            column=f"{name}_norm",
            cmap="YlOrRd",
            edgecolor="#ffffff",
            linewidth=0.08,
            vmin=0.0,
            vmax=1.0,
            missing_kwds={"color": "#e8e8e8"},
        )
        axes.set_title(
            "\n".join(textwrap.wrap(labels.get(name, name), 34)), fontsize=7.5
        )
        axes.set_aspect("equal")
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in axes.spines.values():
            spine.set_visible(False)

    if correlation is not None:
        axes = flat[len(names)]
        values = np.asarray(correlation.to_numpy(), dtype=float)
        image = axes.imshow(values, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ticks = range(len(correlation.index))
        axes.set_xticks(list(ticks))
        axes.set_yticks(list(ticks))
        axes.set_xticklabels(
            [str(name).replace("_", "\n") for name in correlation.index], fontsize=6
        )
        axes.set_yticklabels(
            [str(name).replace("_", " ") for name in correlation.index], fontsize=6
        )
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                entry = values[row, col]
                axes.text(
                    col,
                    row,
                    f"{entry:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=_readable_on(image.cmap(image.norm(entry))),
                )
        axes.set_title("Criterion correlation", fontsize=7.5)
        figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)

    for axes in flat[panels:]:
        axes.axis("off")

    _greening_suptitle(
        figure,
        title or "Greening criteria, normalised (and how far they differ)",
        ahp_report,
        params,
    )
    figure.text(
        0.01, 0.01, footer, fontsize=7, va="bottom", ha="left", color="#444444"
    )
    figure.tight_layout(rect=(0, reserved / height, 1, 0.95))
    return figure


def plot_criterion_panel(
    zones: Any,
    prepared: "pd.DataFrame",
    out_path: str | Path,
    params: dict[str, Any],
    criteria: Sequence[str] | None = None,
    correlation: "pd.DataFrame | None" = None,
    ahp_report: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> Path:
    """Write the criterion panel. See :func:`build_criterion_panel_figure`."""
    return _save_figure(
        build_criterion_panel_figure(
            zones,
            prepared,
            params,
            criteria=criteria,
            correlation=correlation,
            ahp_report=ahp_report,
            title=title,
        ),
        out_path,
    )


def build_priority_table_figure(
    ranked: "pd.DataFrame",
    params: dict[str, Any],
    ahp_report: Mapping[str, Any] | None = None,
    top_n: int | None = None,
    title: str | None = None,
) -> Any:
    """Render the top-N priority divisions as a figure-ready table.

    Args:
        ranked: The priority table.
        params: Parsed params mapping.
        ahp_report: AHP report, for the footer and the title stamp.
        top_n: How many rows to show; defaults to ``greening.top_n``.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the frame is empty.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if ranked.empty:
        raise ValueError("the priority frame is empty; there is nothing to tabulate")

    limit = int(params["greening"]["top_n"] if top_n is None else top_n)
    display = ranked.head(limit).copy()

    def _text(value: Any, digits: int = 3) -> str:
        if value is None or value != value:
            return ""
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    wanted = [
        ("rank_ahp", "rank", 0),
        ("zone_id", "zone id", 0),
        ("adm4_name", "GN division", 0),
        ("score_ahp", "score", 3),
        ("rank_topsis", "TOPSIS", 0),
        ("LST_C", "LST degC", 2),
        ("NDVI", "NDVI", 3),
        ("canopy_pct", "canopy %", 1),
        ("pop_within_300m_pct", "within 300 m %", 1),
        ("compliance", "3-30-300", 0),
        ("wetland_status", "wetland", 0),
    ]
    columns = [entry for entry in wanted if entry[0] in display.columns]
    table_data = []
    for _, row in display.iterrows():
        cells = []
        for name, _, digits in columns:
            text = _text(row[name], digits)
            if name == "adm4_name":
                text = "\n".join(textwrap.wrap(text, 22)) or text
            cells.append(text.replace("_", " ") if name in ("compliance",) else text)
        table_data.append(cells)

    line_counts = [max(1, max(cell.count("\n") + 1 for cell in row)) for row in table_data]
    total_lines = sum(line_counts) + 1

    extra = (
        "The SCORE matters as much as the rank: with rank-normalised criteria "
        "the distribution is smooth, so adjacent ranks can be separated by very "
        "little. The gap at the cut is in the exported table.",
    )
    footer = greening_caption(ahp_report, params, extra=extra)
    reserved = footer_inches(footer)
    table_inches = 0.17 * total_lines + 0.3
    height = table_inches + 0.6 + reserved
    figure = Figure(figsize=(12.0, height))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.axis("off")

    table = axes.table(
        cellText=table_data,
        colLabels=[label for _, label, _ in columns],
        cellLoc="left",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)

    flagged_index = (
        list(display.columns).index("status") if "status" in display.columns else None
    )
    for row in range(len(table_data)):
        shaded = False
        if flagged_index is not None:
            shaded = str(display.iloc[row, flagged_index]) != "ok"
        for column in range(len(columns)):
            cell = table[row + 1, column]
            cell.set_edgecolor("#dddddd")
            cell.set_height(line_counts[row] / total_lines)
            if shaded:
                cell.set_facecolor("#f2e6e6")
    for column in range(len(columns)):
        table[0, column].set_height(1.0 / total_lines)
        table[0, column].set_facecolor("#eeeeee")

    axes.set_title(
        title or f"Top {min(limit, len(display))} greening-priority GN divisions",
        fontsize=12,
        pad=12,
        color="#b2182b" if greening_failure_headline(ahp_report, params) else "#000000",
    )
    figure.text(
        0.01, 0.01, footer, fontsize=7, va="bottom", ha="left", color="#444444"
    )
    figure.tight_layout(rect=(0, reserved / height, 1, 1))
    return figure


def plot_priority_table(
    ranked: "pd.DataFrame",
    out_path: str | Path,
    params: dict[str, Any],
    ahp_report: Mapping[str, Any] | None = None,
    top_n: int | None = None,
    title: str | None = None,
) -> Path:
    """Write the priority table figure. See :func:`build_priority_table_figure`."""
    return _save_figure(
        build_priority_table_figure(
            ranked, params, ahp_report=ahp_report, top_n=top_n, title=title
        ),
        out_path,
    )


# =============================================================================
# Phase 8 - report output plumbing
# =============================================================================
def report_dpi(params: dict[str, Any]) -> int:
    """Raster density for report figures, from ``report.dpi``.

    Args:
        params: Parsed params mapping.

    Returns:
        Dots per inch.
    """
    return int(params["report"]["dpi"])


def report_figure_path(params: dict[str, Any], index: int, slug: str) -> Path:
    """Destination path for one numbered report figure.

    Report figures live in ``report.figure_dir``, deliberately NOT in
    ``figures/``: that directory holds the per-phase diagnostics, which are
    working products at :data:`DIAGNOSTIC_DPI` and are not regenerated here.
    Mixing the two would leave a reader unable to tell which set a given PNG
    belongs to.

    Args:
        params: Parsed params mapping.
        index: Figure number, 1-based; zero-padded into the filename.
        slug: Short lower-case identifier, e.g. ``"decadal_lst"``.

    Returns:
        Path under ``report.figure_dir``, with a ``.png`` suffix.

    Raises:
        ValueError: If ``index`` is not positive or ``slug`` is empty.
    """
    if int(index) < 1:
        raise ValueError(f"figure index must be 1-based and positive, got {index}")
    cleaned = str(slug).strip().lower().replace(" ", "_")
    if not cleaned:
        raise ValueError("figure slug must not be empty")
    template = params["report"]["name_template"]
    name = template.format(index=int(index), slug=cleaned)
    return Path(params["report"]["figure_dir"]) / f"{name}.png"


def save_report_figure(
    figure: Any, params: dict[str, Any], index: int, slug: str
) -> Path:
    """Write one report figure at ``report.dpi``.

    Args:
        figure: A ``matplotlib.figure.Figure``.
        params: Parsed params mapping.
        index: Figure number, 1-based.
        slug: Short lower-case identifier.

    Returns:
        The path written.
    """
    return _save_figure(
        figure, report_figure_path(params, index, slug), dpi=report_dpi(params)
    )


def saturated_fraction(
    values: Any, low: float, high: float
) -> float:
    """Fraction of finite values falling outside a colour stretch.

    A stretch that clips its data understates the extremes it exists to show -
    the mistake that forced ``trends.slope_vis`` to be widened twice. Every
    continuous report figure reports this against
    ``report.max_saturated_fraction`` rather than leaving it to the eye.

    Args:
        values: Array-like; NaNs are ignored.
        low: Lower end of the stretch.
        high: Upper end.

    Returns:
        Share of finite values outside ``[low, high]``; ``nan`` if none are
        finite.
    """
    import numpy as np

    data = np.asarray(values, dtype="float64").ravel()
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return float("nan")
    outside = (finite < float(low)) | (finite > float(high))
    return float(outside.mean())


# =============================================================================
# Phase 8 - colour-vision-deficiency verification
# =============================================================================
# Vienot, Brettel & Mollon (1999), "Digital video colourmaps for checking the
# legibility of displays by dichromats". Linear RGB -> LMS, project onto the
# dichromat's confusion plane, invert. The matrices below are theirs verbatim;
# they operate on LINEAR light, which is why the sRGB transfer function has to
# be undone first and reapplied afterwards. Skipping the linearisation is the
# commonest way to get a plausible-looking but wrong simulation.

#: Hunt-Pointer-Estevez style linear-RGB to LMS matrix (Vienot et al. 1999).
_RGB_TO_LMS: tuple[tuple[float, float, float], ...] = (
    (17.8824, 43.5161, 4.11935),
    (3.45565, 27.1554, 3.86714),
    (0.0299566, 0.184309, 1.46709),
)

#: Per-deficiency LMS projection matrices (Vienot et al. 1999).
_CVD_MATRICES: dict[str, tuple[tuple[float, float, float], ...]] = {
    "protanopia": ((0.0, 2.02344, -2.52581), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "deuteranopia": ((1.0, 0.0, 0.0), (0.494207, 0.0, 1.24827), (0.0, 0.0, 1.0)),
    "tritanopia": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-0.395913, 0.801109, 0.0)),
}

#: CIE D65 white point, for the Lab conversion behind the dE metric.
_D65 = (0.95047, 1.0, 1.08883)

#: sRGB (linear) to CIE XYZ, D65.
_RGB_TO_XYZ: tuple[tuple[float, float, float], ...] = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)


def normalise_hex(colour: str) -> str:
    """Return ``colour`` as a lower-case ``#rrggbb`` string.

    ``params.yaml`` stores hex WITHOUT the leading ``#`` because Earth Engine
    palettes want it that way, while matplotlib wants it with. Every Phase 8
    colour helper goes through here so neither convention leaks.

    Args:
        colour: Hex colour, with or without ``#``, 3 or 6 digits.

    Returns:
        ``#rrggbb``, lower case.

    Raises:
        ValueError: If the string is not a hex colour.
    """
    text = str(colour).strip().lstrip("#").lower()
    if len(text) == 3:
        text = "".join(character * 2 for character in text)
    if len(text) != 6 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"not a hex colour: {colour!r}")
    return f"#{text}"


def _to_linear_rgb(colour: str) -> list[float]:
    """sRGB hex to linear-light RGB in 0-1."""
    text = normalise_hex(colour)[1:]
    channels = [int(text[index : index + 2], 16) / 255.0 for index in (0, 2, 4)]
    return [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]


def _from_linear_rgb(channels: Sequence[float]) -> str:
    """Linear-light RGB to an sRGB hex string, clipped to gamut."""
    encoded = []
    for value in channels:
        clamped = min(max(float(value), 0.0), 1.0)
        gamma = (
            clamped * 12.92
            if clamped <= 0.0031308
            else 1.055 * clamped ** (1 / 2.4) - 0.055
        )
        encoded.append(int(round(min(max(gamma, 0.0), 1.0) * 255)))
    return "#" + "".join(f"{value:02x}" for value in encoded)


def _apply(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    """3x3 matrix times a 3-vector, in plain Python (no numpy import cost)."""
    return [sum(row[index] * vector[index] for index in range(3)) for row in matrix]


def _invert3(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """Inverse of a 3x3 matrix by cofactors."""
    (a, b, c), (d, e, f), (g, h, i) = (tuple(row) for row in matrix)
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(determinant) < 1e-12:
        raise ValueError("matrix is singular")
    return [
        [(e * i - f * h) / determinant, (c * h - b * i) / determinant,
         (b * f - c * e) / determinant],
        [(f * g - d * i) / determinant, (a * i - c * g) / determinant,
         (c * d - a * f) / determinant],
        [(d * h - e * g) / determinant, (b * g - a * h) / determinant,
         (a * e - b * d) / determinant],
    ]


def simulate_cvd(colour: str, deficiency: str) -> str:
    """Simulate how one colour appears to a dichromat.

    Args:
        colour: Hex colour, with or without ``#``.
        deficiency: ``"protanopia"``, ``"deuteranopia"`` or ``"tritanopia"``.

    Returns:
        The simulated colour as ``#rrggbb``.

    Raises:
        KeyError: If the deficiency is not one of the three simulated.
    """
    if deficiency not in _CVD_MATRICES:
        raise KeyError(
            f"unknown deficiency '{deficiency}'; simulate_cvd handles "
            f"{sorted(_CVD_MATRICES)}"
        )
    lms = _apply(_RGB_TO_LMS, _to_linear_rgb(colour))
    projected = _apply(_CVD_MATRICES[deficiency], lms)
    return _from_linear_rgb(_apply(_invert3(_RGB_TO_LMS), projected))


def lab(colour: str) -> tuple[float, float, float]:
    """CIE L*a*b* coordinates of an sRGB colour, D65.

    Args:
        colour: Hex colour, with or without ``#``.

    Returns:
        ``(L, a, b)``. ``L`` is 0 (black) to 100 (white).
    """
    xyz = _apply(_RGB_TO_XYZ, _to_linear_rgb(colour))
    scaled = [value / white for value, white in zip(xyz, _D65)]
    epsilon = (6 / 29) ** 3
    transformed = [
        value ** (1 / 3) if value > epsilon else value / (3 * (6 / 29) ** 2) + 4 / 29
        for value in scaled
    ]
    return (
        116 * transformed[1] - 16,
        500 * (transformed[0] - transformed[1]),
        200 * (transformed[1] - transformed[2]),
    )


def lightness(colour: str) -> float:
    """CIE L* of a colour, 0-100. See :func:`lab`."""
    return lab(colour)[0]


def delta_e(first: str, second: str) -> float:
    """CIE76 colour difference between two sRGB colours.

    CIE76 rather than CIEDE2000 on purpose: these are large flat map patches,
    not the small-patch, near-threshold judgements CIEDE2000 was fitted to, and
    CIE76's plain Euclidean form is auditable by hand from the recorded L*a*b*.

    Args:
        first: Hex colour.
        second: Hex colour.

    Returns:
        Euclidean distance in Lab.
    """
    return math.dist(lab(first), lab(second))


def palette_separation(
    colours: Sequence[str], deficiency: str | None = None
) -> tuple[float, tuple[str, str] | None]:
    """Smallest pairwise CIE76 difference in a palette, and the pair at fault.

    .. warning::
        This is the test for a **categorical** palette only. Adjacent stops of a
        sequential or diverging ramp are *meant* to be close, so every
        ColorBrewer ramp in ``params.yaml`` scores badly here while being
        perfectly readable. Ramps are judged by :func:`palette_lightness_profile`
        instead.

    Args:
        colours: Hex colours, with or without ``#``.
        deficiency: Simulate this deficiency first; ``None`` for normal vision.

    Returns:
        ``(minimum_delta_e, (colour_a, colour_b))``; the pair is ``None`` when
        fewer than two colours were given, and the distance ``inf``.
    """
    import itertools

    seen = [normalise_hex(colour) for colour in colours]
    shown = [simulate_cvd(colour, deficiency) if deficiency else colour for colour in seen]
    best: tuple[float, tuple[str, str] | None] = (float("inf"), None)
    for (index_a, first), (index_b, second) in itertools.combinations(
        list(enumerate(shown)), 2
    ):
        distance = math.dist(lab(first), lab(second))
        if distance < best[0]:
            best = (distance, (seen[index_a], seen[index_b]))
    return best


def palette_lightness_profile(
    colours: Sequence[str], deficiency: str | None = None
) -> list[float]:
    """L* of every stop in a palette, in order. See :func:`lightness`."""
    return [
        lightness(simulate_cvd(colour, deficiency) if deficiency else colour)
        for colour in (normalise_hex(entry) for entry in colours)
    ]


def is_monotonic(values: Sequence[float]) -> bool:
    """True if a sequence rises throughout or falls throughout."""
    steps = [b - a for a, b in zip(values, values[1:])]
    return bool(steps) and (all(s > 0 for s in steps) or all(s < 0 for s in steps))


def is_diverging_monotonic(values: Sequence[float]) -> bool:
    """True if L* is monotonic on each limb away from the palette's centre.

    A diverging ramp's centre is its neutral, so lightness should peak (or
    trough) there and fall away symmetrically. Requires an odd length, which
    :func:`trend_vis_params` already enforces for the slope palette so that one
    colour sits exactly at zero.

    Args:
        values: L* per stop, in palette order.

    Returns:
        Whether both limbs are monotonic in opposite directions.
    """
    if len(values) < 3 or len(values) % 2 == 0:
        return False
    middle = len(values) // 2
    rising = list(values[: middle + 1])
    falling = list(values[middle:])
    up_then_down = is_monotonic(rising) and is_monotonic(falling) and (
        rising[-1] > rising[0] and falling[-1] < falling[0]
    )
    down_then_up = is_monotonic(rising) and is_monotonic(falling) and (
        rising[-1] < rising[0] and falling[-1] > falling[0]
    )
    return bool(up_then_down or down_then_up)


def resolve_palette(params: dict[str, Any], path: str) -> list[str]:
    """Look up a palette by dotted path, whatever shape it is stored in.

    ``params.yaml`` holds palettes both as ordered lists (ramps) and as
    class-keyed mappings (categories). The CVD check has to walk both, so the
    shape is resolved here rather than at every call site.

    Args:
        params: Parsed params mapping.
        path: Dotted path, e.g. ``"greening.palettes.compliance"``.

    Returns:
        Hex colours as ``#rrggbb``, in the order they are stored. For a mapping
        that is insertion order, which is the order the YAML declares - and for
        every palette in this project that order is the class order.

    Raises:
        KeyError: If the path does not resolve, naming the level that failed.
    """
    node: Any = params
    walked: list[str] = []
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            available = sorted(node) if isinstance(node, Mapping) else type(node).__name__
            raise KeyError(
                f"palette path '{path}' does not resolve: "
                f"'{'.'.join(walked) or '<root>'}' has {available}"
            )
        node = node[part]
        walked.append(part)
    entries = list(node.values()) if isinstance(node, Mapping) else list(node)
    return [normalise_hex(entry) for entry in entries]


def check_palette(
    colours: Sequence[str], kind: str, params: dict[str, Any]
) -> dict[str, Any]:
    """Run the applicable colour-vision test over one palette.

    Args:
        colours: Hex colours in palette order.
        kind: ``"categorical"``, ``"sequential"`` or ``"diverging"``.
        params: Parsed params mapping (``report.cvd``).

    Returns:
        A row mapping with ``kind``, ``n_colours``, the per-vision metric, and a
        boolean ``passed`` plus a human-readable ``verdict``.

    Raises:
        ValueError: If ``kind`` is not one of the three.
    """
    config = params["report"]["cvd"]
    visions: list[str | None] = [None, *config["deficiencies"]]
    row: dict[str, Any] = {"kind": kind, "n_colours": len(list(colours))}

    if kind == "categorical":
        floor = float(config["min_delta_e"])
        worst = float("inf")
        worst_pair: tuple[str, str] | None = None
        for vision in visions:
            distance, pair = palette_separation(colours, vision)
            row[f"min_delta_e_{vision or 'normal'}"] = round(distance, 2)
            if distance < worst:
                worst, worst_pair = distance, pair
        row["metric"] = "min_delta_e"
        row["worst"] = round(worst, 2)
        row["threshold"] = floor
        row["worst_pair"] = "/".join(worst_pair) if worst_pair else ""
        row["passed"] = bool(worst >= floor)
        row["verdict"] = (
            f"min dE {worst:.1f} >= {floor:.1f}"
            if row["passed"]
            else f"min dE {worst:.1f} < {floor:.1f} on {row['worst_pair']}"
        )
        return row

    if kind not in ("sequential", "diverging"):
        raise ValueError(
            f"unknown palette kind '{kind}'; expected categorical, sequential "
            "or diverging"
        )

    span_floor = float(config["min_lightness_span"])
    test = is_monotonic if kind == "sequential" else is_diverging_monotonic
    all_monotonic = True
    smallest_span = float("inf")
    for vision in visions:
        profile = palette_lightness_profile(colours, vision)
        label = vision or "normal"
        row[f"monotonic_{label}"] = bool(test(profile))
        row[f"lightness_span_{label}"] = round(max(profile) - min(profile), 1)
        all_monotonic &= bool(test(profile))
        smallest_span = min(smallest_span, max(profile) - min(profile))
    row["metric"] = "lightness_monotonicity"
    row["worst"] = round(smallest_span, 1)
    row["threshold"] = span_floor
    row["worst_pair"] = ""
    row["passed"] = bool(all_monotonic and smallest_span >= span_floor)
    if not all_monotonic:
        row["verdict"] = "L* is not monotonic under every deficiency"
    elif smallest_span < span_floor:
        row["verdict"] = f"L* span {smallest_span:.1f} < {span_floor:.1f}"
    else:
        row["verdict"] = f"L* monotonic, span {smallest_span:.1f}"
    return row


def cvd_report(params: dict[str, Any]) -> "pd.DataFrame":
    """Run the colour-vision check over every palette ``report.cvd`` names.

    Exempted palettes are still MEASURED and still appear in the table - they
    are simply not required to pass, and the recorded ``reason`` says why. An
    exemption that hides its number would be worthless.

    Args:
        params: Parsed params mapping.

    Returns:
        One row per palette: path, kind, the applicable metric, whether it
        passed, whether it is exempt, and the reason.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for entry in params["report"]["cvd"]["palettes"]:
        colours = resolve_palette(params, entry["path"])
        row = check_palette(colours, entry["kind"], params)
        row["path"] = entry["path"]
        row["exempt"] = bool(entry.get("exempt", False))
        row["reason"] = " ".join(str(entry.get("reason", "")).split())
        row["colours"] = " ".join(colours)
        rows.append(row)

    frame = pd.DataFrame(rows)
    leading = ["path", "kind", "n_colours", "metric", "worst", "threshold",
               "passed", "exempt", "worst_pair", "verdict"]
    ordered = leading + [column for column in frame.columns if column not in leading]
    return frame[ordered]


def cvd_failures(report: "pd.DataFrame") -> "pd.DataFrame":
    """Rows of a :func:`cvd_report` that failed and are NOT exempt."""
    return report[(~report["passed"]) & (~report["exempt"])]


# =============================================================================
# Phase 8 - report figures 1-4 (raster maps)
# =============================================================================
def _sequential_cmap(colours: Sequence[str], name: str) -> Any:
    """Continuous colormap from an ordered palette, grey where masked."""
    from matplotlib.colors import LinearSegmentedColormap

    stops = [normalise_hex(colour) for colour in colours]
    cmap = LinearSegmentedColormap.from_list(name, stops)
    return cmap.with_extremes(bad="#dcdcdc")


def panel_aspect(array: Any, default: float = 0.6) -> float:
    """Height-to-width ratio of a raster, for sizing the canvas it is drawn on.

    ``imshow`` draws with ``aspect="equal"``, so a figure whose height ignores
    the array's shape leaves the difference as blank paper - the same defect
    :func:`map_aspect_ratio` was written for on the Phase 5 choropleths, and it
    reappeared on the first Phase 8 draft as a two-inch void between the two
    rows of the decadal figure.

    Args:
        array: 2-D array-like.
        default: Returned when the shape is degenerate.

    Returns:
        ``rows / columns``, clamped to ``[0.2, 3.0]``.
    """
    import numpy as np

    shape = np.asarray(array).shape
    if len(shape) != 2 or min(shape) < 1:
        return default
    return float(min(max(shape[0] / shape[1], 0.2), 3.0))


def _blank_axes(axes: Any) -> Any:
    """Strip ticks and spines from a map panel."""
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)
    return axes


def sensor_step_banner(
    offsets: "pd.DataFrame | None", params: dict[str, Any]
) -> str:
    """One-line summary of the measured Landsat inter-sensor offsets.

    Derived from ``data/outputs/sensor_offsets_cmc.csv`` rather than typed into
    a caption, so the banner cannot disagree with the measurement it is warning
    about. See ``caveats.figures_are_derived_not_authored``.

    Args:
        offsets: The table :func:`colombo_uhi.trends.build_sensor_offset_summary`
            wrote. ``None`` yields a generic warning naming no numbers.
        params: Parsed params mapping.

    Returns:
        A single line, ready to stamp on a figure.
    """
    generic = (
        "POOLED LANDSAT: this row crosses the L5/L7/L8 changeovers and the "
        "offsets over Colombo are material. Read it for GEOGRAPHY only."
    )
    if offsets is None or getattr(offsets, "empty", True):
        return generic

    material = offsets[offsets["verdict"].astype(str) == "material"]
    if material.empty:
        return generic

    def _short(name: str) -> str:
        text = str(name)
        return "L" + text[-1] if text.startswith("landsat") else text

    parts = [
        f"{_short(row['sensor_a'])}-{_short(row['sensor_b'])} "
        f"{float(row['mean_offset']):+.2f} degC "
        f"(t={float(row['t_statistic']):+.1f}, {int(row['n_overlap_years'])} yr)"
        for _, row in material.iterrows()
    ]
    return (
        "MEASURED INTER-SENSOR OFFSETS over the CMC dry season: "
        + "; ".join(parts)
        + ". Decade-to-decade differences in the top row are dominated by these "
        "steps, NOT by climate."
    )


def build_decadal_lst_panel_figure(
    landsat: Mapping[str, Any],
    modis: Mapping[str, Any],
    params: dict[str, Any],
    sensor_offsets: "pd.DataFrame | None" = None,
    title: str = "Mean dry-season land surface temperature by decade",
) -> Any:
    """Build the six-panel decadal LST figure: pooled Landsat over MODIS Terra.

    **Two rows, and the reason is the figure.** The top row is the pooled
    Landsat decadal product, which spans the L5 -> L7 -> L8 changeovers. Its
    measured offsets over Colombo are 2.4x and 3.4x the entire 26-year trend
    signal, so a decade-to-decade comparison read off it measures the
    constellation, not the city; it is drawn because it is the only product with
    intra-urban detail, and it is labelled as a diagnostic. The bottom row is
    MODIS Terra day - one sensor across all three windows, no changeover - which
    is the comparison that is actually valid, at 1 km.

    All six panels share ONE colour stretch (``report.decadal.shared_vis``).
    Autoscaling each panel would hide exactly the step this figure exists to
    expose.

    Args:
        landsat: Band-name to 2-D array, from
            :func:`colombo_uhi.trends.read_trend_raster` on the pooled decadal
            GeoTIFF, read with :func:`colombo_uhi.trends.decadal_band_order`.
        modis: The same, for the MODIS Terra day decadal export.
        params: Parsed params mapping.
        sensor_offsets: ``sensor_offsets_cmc.csv``; drives the banner text.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If a decade's mean band is missing from either input,
            naming which bands are present.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    from colombo_uhi import trends

    config = params["report"]["decadal"]
    labels = [label for label, _, _ in trends.resolve_decades(None, params)]
    keys = [f"mean_{label}" for label in labels]

    rows = (
        ("Landsat, pooled L5/L7/L8/L9\n(SENSOR-STEP DIAGNOSTIC)", landsat),
        ("MODIS Terra day, one sensor\n(the valid comparison)", modis),
    )
    for row_label, arrays in rows:
        missing = [key for key in keys if key not in arrays]
        if missing:
            raise ValueError(
                f"the '{row_label.splitlines()[0]}' arrays are missing {missing}; "
                f"they hold {sorted(arrays)}"
            )

    low = float(config["shared_vis"]["min"])
    high = float(config["shared_vis"]["max"])
    if high <= low:
        raise ValueError(
            f"report.decadal.shared_vis must have max > min, got {low}..{high}"
        )
    cmap = _sequential_cmap(config["palette"], "report_decadal")
    norm = Normalize(vmin=low, vmax=high)

    footer = caveat_footer(
        params,
        ["lst_not_air_temp", "valid_obs_required", "single_overpass",
         "within_epoch_only"],
    ) + "\n" + _wrap_bullets(
        [
            sensor_step_banner(sensor_offsets, params),
            "All six panels share ONE colour stretch of "
            f"{low:.0f}-{high:.0f} degC. Per-panel autoscaling would hide the "
            "step between the top row's decades, which is the point of drawing "
            "the two rows together.",
            "The bottom row is 1 km: the CMC is about six pixels across, so it "
            "carries the absolute levels honestly and the intra-urban pattern "
            "not at all. Neither row is sufficient alone.",
            "The number under each panel is that panel's mean over its finite "
            "pixels, so the decade-to-decade step can be read as a number "
            "rather than guessed from the colour.",
        ]
    )

    # Height follows the data, not a constant: see panel_aspect.
    width = float(params["report"]["width_inches"])
    left, right = 0.10, 0.87
    aspect = panel_aspect(landsat[keys[0]])
    panel_height = (width * (right - left) / len(labels)) * aspect
    reserved_inches = footer_inches(footer)
    #: title band, per-panel captions and inter-row breathing room.
    chrome = 1.30
    height = 2 * panel_height + chrome + reserved_inches

    figure = Figure(figsize=(width, height))
    FigureCanvasAgg(figure)
    panels = figure.subplots(2, len(labels), squeeze=False)
    figure.subplots_adjust(
        left=left, right=right,
        top=1 - 0.55 / height, bottom=(reserved_inches + 0.30) / height,
        wspace=0.06, hspace=0.42 / max(panel_height, 0.4),
    )

    image = None
    for row_index, (row_label, arrays) in enumerate(rows):
        for column, (label, key) in enumerate(zip(labels, keys)):
            axes = _blank_axes(panels[row_index][column])
            data = np.asarray(arrays[key], dtype="float64")
            image = axes.imshow(
                np.ma.masked_invalid(data), cmap=cmap, norm=norm,
                interpolation="nearest",
            )
            if row_index == 0:
                axes.set_title(label.replace("_", "-"), fontsize=10)
            finite = data[np.isfinite(data)]
            mean_text = f"mean {finite.mean():.2f} degC" if finite.size else "no data"
            clipped = saturated_fraction(data, low, high)
            axes.set_xlabel(
                f"{mean_text}\n{clipped:.1%} outside the stretch", fontsize=7.5
            )
            if column == 0:
                axes.set_ylabel(row_label, fontsize=8.5)

    bar = figure.colorbar(
        image, ax=panels.ravel().tolist(), orientation="vertical",
        fraction=0.028, pad=0.02,
    )
    bar.set_label("Mean dry-season LST (degC)", fontsize=9)
    bar.ax.tick_params(labelsize=8)

    figure.suptitle(title, fontsize=12, y=1 - 0.18 / height)
    figure.text(
        0.01, 0.10 / height, footer, fontsize=7, va="bottom", ha="left",
        color="#444444",
    )
    return figure


def plot_decadal_lst_panel(
    landsat: Mapping[str, Any],
    modis: Mapping[str, Any],
    out_path: str | Path,
    params: dict[str, Any],
    sensor_offsets: "pd.DataFrame | None" = None,
    title: str = "Mean dry-season land surface temperature by decade",
) -> Path:
    """Write the decadal LST figure. See :func:`build_decadal_lst_panel_figure`."""
    return _save_figure(
        build_decadal_lst_panel_figure(
            landsat, modis, params, sensor_offsets=sensor_offsets, title=title
        ),
        out_path,
        dpi=report_dpi(params),
    )


def stipple_coordinates(
    mask: Any, stride: int
) -> tuple[Any, Any]:
    """Column and row indices of every ``stride``-th true cell in a mask.

    Drawing one dot per significant pixel turns a significant region into a
    solid black blob and hides the slope beneath it, which defeats an overlay
    whose whole job is to sit on top of a readable map. Sub-sampling on a fixed
    lattice keeps the density even, so the stipple reads as texture rather than
    as a second, misleading choropleth.

    Args:
        mask: 2-D boolean-ish array; truthy cells are candidates.
        stride: Take every Nth row and Nth column. Must be >= 1.

    Returns:
        ``(x, y)`` index arrays, ready for ``axes.scatter``.

    Raises:
        ValueError: If ``stride`` is below 1.
    """
    import numpy as np

    if int(stride) < 1:
        raise ValueError(f"stipple stride must be >= 1, got {stride}")
    flags = np.asarray(mask)
    lattice = np.zeros(flags.shape, dtype=bool)
    lattice[:: int(stride), :: int(stride)] = True
    rows, columns = np.nonzero(flags & lattice)
    return columns, rows


def build_sen_slope_stipple_figure(
    arrays: Mapping[str, Any],
    params: dict[str, Any],
    slope_key: str = "sen_slope",
    significant_key: str = "significant",
    title: str = "Sen's slope of land surface temperature",
) -> Any:
    """Build the report Sen's-slope map: one panel, significance as stippling.

    The Phase 4 :func:`build_trend_map_figure` draws two panels - all slopes,
    then significant slopes - because that pair is the honest DIAGNOSTIC. For
    the report a single map is stronger: stippling puts the significance and the
    magnitude in the same place, so a reader cannot read a rate off one panel
    and its confidence off another without registering the two by eye.

    Args:
        arrays: Band-name to 2-D array, from
            :func:`colombo_uhi.trends.read_trend_raster` on the FDR-corrected
            raster.
        params: Parsed params mapping.
        slope_key: Band holding the Sen's slope, degC per year.
        significant_key: Band holding the FDR significance mask (1/0/NaN).
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
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    missing = [key for key in (slope_key, significant_key) if key not in arrays]
    if missing:
        raise ValueError(
            f"trend arrays are missing {missing}; they hold {sorted(arrays)}"
        )

    vis = trend_vis_params(params)
    cmap = LinearSegmentedColormap.from_list(
        "report_trend", [normalise_hex(colour) for colour in vis["palette"]]
    ).with_extremes(bad="#dcdcdc")
    norm = TwoSlopeNorm(vmin=vis["min"], vcenter=0.0, vmax=vis["max"])

    slope = np.asarray(arrays[slope_key], dtype="float64")
    significant = np.asarray(arrays[significant_key], dtype="float64")

    tested = int(np.isfinite(significant).sum())
    n_significant = int(np.nansum(significant == 1.0))
    share = (n_significant / tested) if tested else float("nan")
    clipped = saturated_fraction(slope, vis["min"], vis["max"])

    stipple = params["report"]["stipple"]
    footer = caveat_footer(
        params,
        ["lst_not_air_temp", "valid_obs_required", "single_overpass",
         "fdr_dependence", "trend_not_causal"],
    ) + "\n" + _wrap_bullets(
        [
            f"{n_significant:,} of {tested:,} fitted pixels ({share:.1%}) survive "
            f"{params['trends']['fdr']['method'].replace('_', '-')} correction at "
            f"alpha = {params['trends']['fdr']['alpha']}. Only those are stippled.",
            "Grey is NOT 'no trend'. It is a pixel that was never tested, because "
            "it fell below the valid-observation or minimum-year floor. Read this "
            "map against figure 3.",
            f"Stippling is drawn on a 1-in-{stipple['stride']} lattice, so its "
            "density is a fixed texture and NOT proportional to anything. Its "
            "extent is the result; its darkness is not.",
            f"{clipped:.1%} of fitted pixels fall outside the "
            f"{vis['min']:+.2f}..{vis['max']:+.2f} degC/yr colour stretch.",
        ]
    )

    width = float(params["report"]["width_inches"])
    left, span = 0.04, 0.80
    panel_height = width * span * panel_aspect(slope)
    reserved_inches = footer_inches(footer)
    chrome = 0.85                       # title band plus a margin below the map
    height = panel_height + chrome + reserved_inches

    figure = Figure(figsize=(width, height))
    FigureCanvasAgg(figure)
    axes = _blank_axes(
        figure.add_axes(
            (left, (reserved_inches + 0.20) / height, span, panel_height / height)
        )
    )

    image = axes.imshow(
        np.ma.masked_invalid(slope), cmap=cmap, norm=norm, interpolation="nearest"
    )
    x, y = stipple_coordinates(significant == 1.0, int(stipple["stride"]))
    axes.scatter(
        x, y,
        s=float(stipple["marker_size"]),
        marker=str(stipple["marker"]),
        c=normalise_hex(stipple["colour"]),
        linewidths=0,
    )

    bar = figure.colorbar(image, ax=axes, fraction=0.032, pad=0.02)
    bar.set_label("Sen's slope (degC per year)", fontsize=9)
    bar.ax.tick_params(labelsize=8)

    axes.legend(
        handles=[
            Line2D([], [], linestyle="none", marker=str(stipple["marker"]),
                   markersize=4, color=normalise_hex(stipple["colour"]),
                   label=str(stipple["label"])),
            Patch(facecolor="#dcdcdc", edgecolor="#999999",
                  label="not tested (below the observation floor)"),
        ],
        loc="lower left", fontsize=8, frameon=True, facecolor="white",
        edgecolor="#cccccc",
    )

    figure.suptitle(title, fontsize=12, y=1 - 0.22 / height)
    figure.text(
        0.01, 0.08 / height, footer, fontsize=7, va="bottom", ha="left",
        color="#444444",
    )
    return figure


def plot_sen_slope_stipple(
    arrays: Mapping[str, Any],
    out_path: str | Path,
    params: dict[str, Any],
    slope_key: str = "sen_slope",
    significant_key: str = "significant",
    title: str = "Sen's slope of land surface temperature",
) -> Path:
    """Write the stippled slope map. See :func:`build_sen_slope_stipple_figure`."""
    return _save_figure(
        build_sen_slope_stipple_figure(
            arrays, params, slope_key=slope_key,
            significant_key=significant_key, title=title,
        ),
        out_path,
        dpi=report_dpi(params),
    )


def build_obs_count_figure(
    arrays: Mapping[str, Any],
    params: dict[str, Any],
    count_key: str | None = None,
    title: str = "Valid dry-season observations per pixel",
) -> Any:
    """Build the valid-observation-count map beside its cumulative distribution.

    ``CLAUDE.md`` caveat 2 requires a per-pixel valid-observation count beside
    every composite and trend product. A map alone does not discharge that: a
    reader cannot integrate a colour ramp by eye, and "most of the city is
    thin" and "a corner of the city is thin" look similar on one. The ECDF panel
    turns it into a number - what share of the AOI rests on fewer than N usable
    scenes - which is the form the caveat is actually about.

    Args:
        arrays: Band-name to 2-D array holding the count.
        params: Parsed params mapping.
        count_key: Band holding the count; defaults to
            ``composites.obs_count_band``.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the count band is missing, or holds no finite pixel.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    key = count_key or params["composites"]["obs_count_band"]
    if key not in arrays:
        raise ValueError(
            f"observation-count arrays are missing '{key}'; they hold "
            f"{sorted(arrays)}"
        )
    counts = np.asarray(arrays[key], dtype="float64")
    finite = counts[np.isfinite(counts)]
    if finite.size == 0:
        raise ValueError(f"band '{key}' has no finite pixel; nothing to draw")

    vis = params["report"]["obs_count_vis"]
    cmap = _sequential_cmap(vis["palette"], "report_obs_count")
    norm = Normalize(vmin=float(vis["min"]), vmax=float(vis["max"]))
    marks = [int(mark) for mark in vis["ecdf_marks"]]

    shares = {mark: float((finite < mark).mean()) for mark in marks}
    footer = caveat_footer(
        params, ["valid_obs_required", "lst_not_air_temp", "single_overpass"]
    ) + "\n" + _wrap_bullets(
        [
            "Median "
            f"{np.median(finite):.0f} usable observations per pixel; "
            f"1st percentile {np.percentile(finite, 1):.0f}, "
            f"99th {np.percentile(finite, 99):.0f}.",
            "Share of the AOI below each reference count: "
            + ", ".join(f"<{mark}: {shares[mark]:.1%}" for mark in marks)
            + ".",
            "A count of zero is a real tropical-cloud result, not a gap in the "
            "processing: some pixels had no cloud-free scene in that window at "
            "all. Those pixels are masked out of every downstream product, which "
            "is why the trend map has untested grey.",
        ]
    )

    width = float(params["report"]["width_inches"])
    left, right, map_share = 0.04, 0.94, 1.35 / 2.35
    map_height = width * (right - left) * map_share * panel_aspect(counts)
    reserved_inches = footer_inches(footer)
    chrome = 1.55                       # title band, the ECDF's axis labels, and the
                                        # horizontal colorbar under the map
    # Floor the panel height so a very wide raster does not squash the ECDF,
    # whose y axis is a share and needs room to be read.
    height = max(map_height, 2.8) + chrome + reserved_inches

    figure = Figure(figsize=(width, height))
    FigureCanvasAgg(figure)
    panels = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.35, 1.0]})
    figure.subplots_adjust(
        left=left, right=right,
        top=1 - 0.55 / height, bottom=(reserved_inches + 0.60) / height,
        wspace=0.26,
    )

    axes = _blank_axes(panels[0])
    image = axes.imshow(
        np.ma.masked_invalid(counts), cmap=cmap, norm=norm, interpolation="nearest"
    )
    axes.set_title("Per-pixel count", fontsize=10)
    # HORIZONTAL, under the map. A vertical bar here sits in the gap between the
    # two panels and its label lands on top of the ECDF's y-axis label - which is
    # exactly what the first Phase 8 draft did.
    bar = figure.colorbar(
        image, ax=axes, orientation="horizontal", fraction=0.055, pad=0.04
    )
    bar.set_label("usable observations", fontsize=9)
    bar.ax.tick_params(labelsize=8)

    ecdf = panels[1]
    ordered = np.sort(finite)
    ecdf.plot(
        ordered,
        np.arange(1, ordered.size + 1) / ordered.size,
        color="#31688e", linewidth=1.6,
    )
    for mark in marks:
        ecdf.axvline(mark, color="#b2182b", linewidth=0.9, linestyle="--")
        ecdf.annotate(
            f"{mark}: {shares[mark]:.0%}",
            xy=(mark, shares[mark]), xytext=(4, 4), textcoords="offset points",
            fontsize=7.5, color="#b2182b",
        )
    ecdf.set_xlabel("usable observations per pixel", fontsize=9)
    ecdf.set_ylabel("cumulative share of pixels", fontsize=9)
    ecdf.set_ylim(0, 1)
    ecdf.set_title("Cumulative distribution", fontsize=10)
    ecdf.grid(True, alpha=0.3)
    ecdf.tick_params(labelsize=8)

    figure.suptitle(title, fontsize=12, y=1 - 0.18 / height)
    figure.text(
        0.01, 0.10 / height, footer, fontsize=7, va="bottom", ha="left",
        color="#444444",
    )
    return figure


def plot_obs_count(
    arrays: Mapping[str, Any],
    out_path: str | Path,
    params: dict[str, Any],
    count_key: str | None = None,
    title: str = "Valid dry-season observations per pixel",
) -> Path:
    """Write the observation-count figure. See :func:`build_obs_count_figure`."""
    return _save_figure(
        build_obs_count_figure(arrays, params, count_key=count_key, title=title),
        out_path,
        dpi=report_dpi(params),
    )


def build_utfvi_epoch_maps_figure(
    epochs: Mapping[str, Any],
    params: dict[str, Any],
    title: str = "Urban thermal field variance index by epoch",
) -> Any:
    """Build the per-epoch UTFVI six-class maps with a shared class legend.

    .. warning::
        UTFVI is referenced to **each epoch's own mean LST**
        (``uhi.utfvi.reference``), so class drift between panels is a
        REDISTRIBUTION of heat within that epoch, never evidence of warming. A
        city that warms uniformly by 2 degC produces three identical panels.
        The class-share bars beneath the maps exist to make that reading
        available as numbers, and the footer states it in words.

    Args:
        epochs: Epoch label to 2-D class array holding integer codes 0-5, with
            NaN (or a masked array) outside the AOI.
        params: Parsed params mapping.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If no epoch is supplied, or an epoch holds a class code
            outside the configured scheme.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    from colombo_uhi import uhi_metrics

    _, labels = uhi_metrics.validate_utfvi_scheme(params)
    palette = [normalise_hex(colour) for colour in params["uhi"]["utfvi"]["palette"]]
    order = [key for key in params["report"]["facet_epochs"] if key in epochs]
    order += [key for key in epochs if key not in order]
    if not order:
        raise ValueError("no epoch arrays supplied; there is nothing to map")

    cmap = ListedColormap(palette).with_extremes(bad="#dcdcdc")
    norm = BoundaryNorm(list(range(len(labels) + 1)), cmap.N)

    shares: dict[str, list[float]] = {}
    for key in order:
        codes = np.asarray(epochs[key], dtype="float64")
        finite = codes[np.isfinite(codes)]
        if finite.size and (finite.min() < 0 or finite.max() > len(labels) - 1):
            raise ValueError(
                f"epoch '{key}' holds class codes {finite.min():.0f}..{finite.max():.0f}, "
                f"outside the configured 0..{len(labels) - 1} scheme"
            )
        counts = np.array(
            [float((finite == index).sum()) for index in range(len(labels))]
        )
        total = counts.sum()
        shares[key] = list(counts / total * 100.0) if total else [float("nan")] * len(labels)

    footer = caveat_footer(
        params,
        ["lst_not_air_temp", "valid_obs_required", "within_epoch_only",
         "colour_is_not_the_only_channel"],
    ) + "\n" + _wrap_bullets(
        [
            "UTFVI is referenced to EACH EPOCH'S OWN mean LST, so a difference "
            "between these panels is a REDISTRIBUTION of heat within that epoch "
            "and NEVER evidence of warming. Uniform warming leaves all three "
            "panels identical; figure 2 is what measures change.",
            "Class breaks "
            + ", ".join(str(value) for value in params["uhi"]["utfvi"]["breaks"])
            + " are the published thresholds and assume degrees CELSIUS - UTFVI "
            "is a ratio, so on Kelvin every break would mean something else.",
            "The palette runs light to dark with severity, so the ordering "
            "survives greyscale printing and every simulated colour-vision "
            "deficiency; the bars repeat the same information as numbers.",
        ]
    )

    width = float(params["report"]["width_inches"])
    left, right = 0.05, 0.97
    map_height = (width * (right - left) / len(order)) * panel_aspect(epochs[order[0]])
    bar_height = 1.25                   # the class-share row, in inches
    reserved_inches = footer_inches(footer)
    chrome = 1.35                       # title band, legend strip and panel titles
    height = map_height + bar_height + chrome + reserved_inches

    figure = Figure(figsize=(width, height))
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        2, len(order), height_ratios=[map_height, bar_height],
        left=left, right=right,
        top=1 - 0.95 / height, bottom=(reserved_inches + 0.55) / height,
        wspace=0.08, hspace=0.55 / max(map_height, 0.4),
    )

    for column, key in enumerate(order):
        axes = _blank_axes(figure.add_subplot(grid[0, column]))
        codes = np.asarray(epochs[key], dtype="float64")
        axes.imshow(
            np.ma.masked_invalid(codes), cmap=cmap, norm=norm, interpolation="nearest"
        )
        axes.set_title(key, fontsize=10)

        bars = figure.add_subplot(grid[1, column])
        positions = list(range(len(labels)))
        bars.bar(positions, shares[key], color=palette, edgecolor="#666666", linewidth=0.4)
        bars.set_xticks(positions)
        bars.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
        bars.set_ylim(0, 100)
        bars.tick_params(labelsize=7)
        bars.grid(True, axis="y", alpha=0.3)
        if column == 0:
            bars.set_ylabel("share of AOI (%)", fontsize=8)
        else:
            bars.set_yticklabels([])

    figure.legend(
        handles=[
            Patch(facecolor=colour, edgecolor="#666666",
                  label=f"{index}  {label}")
            for index, (label, colour) in enumerate(zip(labels, palette))
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1 - 0.42 / height),
        ncol=len(labels), fontsize=7.5, frameon=True,
        facecolor="white", edgecolor="#cccccc", handlelength=1.4,
    )
    figure.suptitle(title, fontsize=12, y=1 - 0.16 / height)
    figure.text(
        0.01, 0.10 / height, footer, fontsize=7, va="bottom", ha="left",
        color="#444444",
    )
    return figure


def plot_utfvi_epoch_maps(
    epochs: Mapping[str, Any],
    out_path: str | Path,
    params: dict[str, Any],
    title: str = "Urban thermal field variance index by epoch",
) -> Path:
    """Write the UTFVI epoch maps. See :func:`build_utfvi_epoch_maps_figure`."""
    return _save_figure(
        build_utfvi_epoch_maps_figure(epochs, params, title=title),
        out_path,
        dpi=report_dpi(params),
    )


# =============================================================================
# Phase 8 - report figures 7 and 9
# =============================================================================
def build_lst_vs_index_facet_figure(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    epoch_column: str = "epoch",
    index_columns: Sequence[str] | None = None,
    response: str | None = None,
    max_points: int | None = None,
    title: str | None = None,
) -> Any:
    """Build the LST-versus-driver scatter grid, one ROW per epoch.

    The Phase 3 :func:`build_lst_vs_index_figure` pools every year into one row,
    which answers "is LST related to vegetation?" but not "did that relationship
    change?". Faceting by epoch is what makes the second question askable: three
    rows with the same axes let a reader compare slopes by eye, and each panel
    states its own slope, r and n so the comparison does not have to be by eye.

    .. warning::
        The fitted lines are a SCREENING device. Sampled pixels are strongly
        spatially autocorrelated, so the ordinary least-squares standard errors
        are too small and any p-value read off them is anti-conservative. Phase
        5's GWR/MGWR is where this is handled properly.

    Args:
        frame: Sampled-pixel table carrying ``epoch_column``, the response and
            every index column - the pooled output of
            :func:`colombo_uhi.uhi_metrics.sample_drivers` per epoch.
        params: Parsed params mapping.
        epoch_column: Column naming each row's epoch.
        index_columns: Drivers to plot, one column each; ``None`` uses
            ``report.facet_indices``.
        response: Response column; defaults to ``uhi.drivers.response``.
        max_points: Points DRAWN per panel; ``None`` uses
            ``report.facet_max_points``. Every fit uses every row regardless.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the frame is empty, a named column is missing, or no
            configured epoch appears in it.
    """
    import numpy as np
    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    y_name = response or params["uhi"]["drivers"]["response"]
    columns = list(index_columns or params["report"]["facet_indices"])
    drawn = int(params["report"]["facet_max_points"] if max_points is None else max_points)

    if frame.empty:
        raise ValueError("sample frame is empty; there is nothing to plot")
    missing = [
        column
        for column in (epoch_column, y_name, *columns)
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"sample frame is missing column(s) {missing}; it has "
            f"{sorted(frame.columns)}"
        )

    present = set(frame[epoch_column].astype(str))
    epochs = [key for key in params["report"]["facet_epochs"] if key in present]
    epochs += [key for key in dict.fromkeys(frame[epoch_column].astype(str))
               if key not in epochs]
    if not epochs:
        raise ValueError(
            f"no epoch appears in column '{epoch_column}'; it holds {sorted(present)}"
        )

    footer = caveat_footer(
        params,
        ["lst_not_air_temp", "single_overpass", "valid_obs_required",
         "within_epoch_only", "trend_not_causal"],
    ) + "\n" + _wrap_bullets(
        [
            "Each fit uses EVERY sampled row; only the drawing is thinned to "
            f"{drawn:,} points a panel. The line is drawn across the 1st-99th "
            "percentile of x, so it is not extrapolated into a tail holding a "
            "handful of pixels.",
            "Sampled pixels are spatially autocorrelated, so these OLS standard "
            "errors are too small and any p-value from them overstates "
            "significance. Read the slopes as a screening result; Phase 5's "
            "GWR/MGWR is the inferential version.",
            "Rows share axes so the slopes are comparable BY EYE. They are not "
            "comparable in LEVEL: the epochs are composited from different "
            "Landsat sensors, whose offsets over Colombo are material, so a "
            "vertical shift between rows is not a measurement of warming.",
        ]
    )

    rows, cols = len(epochs), len(columns)
    reserved_inches = footer_inches(footer)
    panel = 3.0
    height = rows * panel + 1.25 + reserved_inches
    width = float(params["report"]["width_inches"])

    figure = Figure(figsize=(width, height))
    FigureCanvasAgg(figure)
    panels = figure.subplots(rows, cols, squeeze=False, sharex="col", sharey=True)
    figure.subplots_adjust(
        left=0.09, right=0.98,
        top=1 - 0.62 / height, bottom=(reserved_inches + 0.60) / height,
        wspace=0.10, hspace=0.28,
    )

    seed = int(params["uhi"]["drivers"]["sample_seed"])
    for row, epoch in enumerate(epochs):
        block = frame[frame[epoch_column].astype(str) == epoch]
        for column_index, column in enumerate(columns):
            axes = panels[row][column_index]
            pair = (
                block[[y_name, column]]
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
            )
            if pair.empty:
                axes.set_title(f"{epoch} - {column}: no data", fontsize=8.5)
                continue

            shown = pair.sample(drawn, random_state=seed) if len(pair) > drawn else pair
            axes.scatter(
                shown[column], shown[y_name], s=3, alpha=0.22, edgecolors="none",
                color="#31688e",
            )
            if int(pair[column].nunique(dropna=True)) > 1:
                slope, intercept = np.polyfit(pair[column], pair[y_name], 1)
                low, high = np.percentile(pair[column], [1, 99])
                if high <= low:
                    low, high = float(pair[column].min()), float(pair[column].max())
                grid = np.linspace(low, high, 50)
                axes.plot(grid, slope * grid + intercept, color="#b2182b", linewidth=1.7)
                r_value = float(np.corrcoef(pair[column], pair[y_name])[0, 1])
                axes.set_title(
                    f"{epoch} - {column}:  {slope:+.2f} degC per unit,  "
                    f"r = {r_value:+.2f},  n = {len(pair):,}",
                    fontsize=8.5,
                )
            else:
                axes.set_title(f"{epoch} - {column}: constant, no fit", fontsize=8.5)

            axes.grid(True, alpha=0.3)
            axes.tick_params(labelsize=7.5)
            if row == rows - 1:
                axes.set_xlabel(column, fontsize=9)
            if column_index == 0:
                axes.set_ylabel("LST (degC)", fontsize=9)

    figure.suptitle(
        title or f"{y_name} against NDVI and NDBI, by epoch (sampled pixels)",
        fontsize=12, y=1 - 0.20 / height,
    )
    figure.text(
        0.01, 0.10 / height, footer, fontsize=7, va="bottom", ha="left",
        color="#444444",
    )
    return figure


def plot_lst_vs_index_facet(
    frame: "pd.DataFrame",
    out_path: str | Path,
    params: dict[str, Any],
    epoch_column: str = "epoch",
    index_columns: Sequence[str] | None = None,
    response: str | None = None,
    max_points: int | None = None,
    title: str | None = None,
) -> Path:
    """Write the faceted scatter grid. See :func:`build_lst_vs_index_facet_figure`."""
    return _save_figure(
        build_lst_vs_index_facet_figure(
            frame, params, epoch_column=epoch_column, index_columns=index_columns,
            response=response, max_points=max_points, title=title,
        ),
        out_path,
        dpi=report_dpi(params),
    )


def build_scenario_triptych_figure(
    baseline: Any,
    greened: Any,
    difference: Any,
    params: dict[str, Any],
    report: Mapping[str, Any] | None = None,
    title: str | None = None,
    priority_mask: Any = None,
) -> Any:
    """Build the greening counterfactual triptych: baseline, greened, difference.

    This is a **counterfactual on observed predictors** - "what would the surface
    look like if these zones were greened today" - and NOT a projection to a
    future date. That distinction is why it can be drawn at all: it rests on the
    validated random forest alone, with no land-cover projection underneath it,
    so it carries regression metrics and no Kappa. The 2030 and 2036 horizons DO
    rest on the CA-Markov component, which Phase 6 measured and could not
    validate, so no map of them is produced.

    The two temperature panels share ONE stretch, taken from the pair, so the
    eye compares like with like; the difference panel takes the symmetric
    diverging ramp from ``prediction.palettes``.

    Args:
        baseline: 2-D array of modelled LST under the unmodified predictors.
        greened: 2-D array under the greening lever.
        difference: 2-D array of greened minus baseline, in degC.
        params: Parsed params mapping.
        report: Validation report for the ``lst_scenario`` product; drives the
            caption and, on failure, the banner.
        title: Figure title.
        priority_mask: Optional boolean array marking the greened zones. Supply
            it: the district-wide mean difference is dominated by the ~97% of
            cells nothing was done to, so quoting it as the greening effect
            understates the result by an order of magnitude. When given, the
            footer reports BOTH numbers and says which is which.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the three arrays do not share a shape, or the mask does
            not match them.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
    from matplotlib.figure import Figure

    surfaces = [np.asarray(array, dtype="float64") for array in (baseline, greened, difference)]
    shapes = {surface.shape for surface in surfaces}
    if len(shapes) != 1:
        raise ValueError(
            f"the three scenario surfaces must share a shape, got {sorted(shapes)}"
        )

    palettes = params["prediction"]["palettes"]
    limit = float(palettes["scenario_difference_max_degc"])
    heat = _sequential_cmap(params["report"]["decadal"]["palette"], "report_scenario")
    diverging = LinearSegmentedColormap.from_list(
        "report_scenario_difference",
        [normalise_hex(colour) for colour in palettes["scenario_difference"]],
    ).with_extremes(bad="#dcdcdc")

    both = np.concatenate([surfaces[0].ravel(), surfaces[1].ravel()])
    finite = both[np.isfinite(both)]
    if finite.size:
        low, high = (float(value) for value in np.percentile(finite, [1, 99]))
    else:
        low, high = 0.0, 1.0
    if high <= low:
        low, high = low - 0.5, low + 0.5

    delta = surfaces[2]
    delta_finite = delta[np.isfinite(delta)]
    mean_delta = float(delta_finite.mean()) if delta_finite.size else float("nan")
    cooled = float((delta_finite < 0).mean()) if delta_finite.size else float("nan")
    lever = params["prediction"]["scenarios"]["greening"]["canopy_increase_fraction"]

    inside_line = (
        "The mean above is over the WHOLE mapped area, most of which was not "
        "greened, so it is NOT the greening effect. Quote the priority-zone "
        "mean from greening_counterfactual_by_gn.csv instead."
    )
    if priority_mask is not None:
        flags = np.asarray(priority_mask, dtype=bool)
        if flags.shape != delta.shape:
            raise ValueError(
                f"priority_mask has shape {flags.shape}, but the surfaces are "
                f"{delta.shape}"
            )
        inside = delta[flags & np.isfinite(delta)]
        if inside.size:
            inside_line = (
                f"INSIDE the greened priority zones the mean difference is "
                f"{float(inside.mean()):+.2f} degC over {int(inside.size):,} "
                "cells. That is the result. The district-wide mean above is "
                "diluted by every cell nothing was done to, and must not be "
                "quoted as the greening effect."
            )

    footer = projection_caption(
        report,
        params,
        keys=["scenario_not_forecast", "lst_not_air_temp", "single_overpass",
              "zonal_not_pixel"],
        extra=_wrap_bullets(
            [
                "This is a COUNTERFACTUAL ON OBSERVED PREDICTORS - 'if these "
                "zones were greened today' - not a projection to a future date. "
                "It involves no land-cover projection, which is why it carries "
                "regression metrics and no Kappa, and why it may be mapped at "
                "all while the 2030 and 2036 horizons may not.",
                f"The lever is a {float(lever):.0%} shift of each priority "
                "cell's surface character toward the observed canopy signature. "
                "It assumes the planting happens, and it assumes the fitted "
                "LST-driver relationship holds under a surface the model never "
                "observed.",
                f"Mean difference {mean_delta:+.2f} degC inside the mapped area; "
                f"{cooled:.1%} of finite cells cool. The two temperature panels "
                f"share one stretch of {low:.1f}-{high:.1f} degC (1st-99th "
                "percentile of the pair), so they are directly comparable.",
                inside_line,
            ]
        ),
    )

    width = float(params["report"]["width_inches"])
    left, right = 0.04, 0.97
    panel_height = (width * (right - left) / 3) * panel_aspect(surfaces[0])
    reserved_inches = footer_inches(footer)
    chrome = 1.35                       # title band, panel titles and two colorbars
    height = panel_height + chrome + reserved_inches

    figure = Figure(figsize=(width, height))
    FigureCanvasAgg(figure)
    panels = figure.subplots(1, 3)
    figure.subplots_adjust(
        left=left, right=right,
        top=1 - 0.60 / height, bottom=(reserved_inches + 0.55) / height,
        wspace=0.10,
    )

    heat_image = None
    for axes, data, panel_title in (
        (panels[0], surfaces[0], "Baseline (observed predictors)"),
        (panels[1], surfaces[1], f"Greening scenario ({float(lever):.0%} canopy shift)"),
    ):
        _blank_axes(axes)
        heat_image = axes.imshow(
            np.ma.masked_invalid(data), cmap=heat,
            norm=Normalize(vmin=low, vmax=high), interpolation="nearest",
        )
        axes.set_title(panel_title, fontsize=9.5)

    heat_bar = figure.colorbar(
        heat_image, ax=[panels[0], panels[1]], orientation="horizontal",
        fraction=0.055, pad=0.05,
    )
    heat_bar.set_label("Modelled LST (degC)", fontsize=9)
    heat_bar.ax.tick_params(labelsize=8)

    _blank_axes(panels[2])
    delta_image = panels[2].imshow(
        np.ma.masked_invalid(delta), cmap=diverging,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        interpolation="nearest",
    )
    panels[2].set_title("Difference (greening minus baseline)", fontsize=9.5)
    delta_bar = figure.colorbar(
        delta_image, ax=panels[2], orientation="horizontal",
        fraction=0.055, pad=0.05,
    )
    delta_bar.set_label("degC", fontsize=9)
    delta_bar.ax.tick_params(labelsize=8)

    _suptitle(
        figure,
        title or "Greening counterfactual: modelled land surface temperature",
        report,
        params,
    )
    figure.text(
        0.01, 0.10 / height, footer, fontsize=7, va="bottom", ha="left",
        color="#444444",
    )
    return figure


def plot_scenario_triptych(
    baseline: Any,
    greened: Any,
    difference: Any,
    out_path: str | Path,
    params: dict[str, Any],
    report: Mapping[str, Any] | None = None,
    title: str | None = None,
    priority_mask: Any = None,
) -> Path:
    """Write the scenario triptych. See :func:`build_scenario_triptych_figure`."""
    return _save_figure(
        build_scenario_triptych_figure(
            baseline, greened, difference, params, report=report, title=title,
            priority_mask=priority_mask,
        ),
        out_path,
        dpi=report_dpi(params),
    )


# =============================================================================
# Phase 8 - top-N labelling for the greening priority map
# =============================================================================
def spread_label_positions(
    wanted: Sequence[float],
    min_gap: float,
    low: float | None = None,
    high: float | None = None,
) -> list[float]:
    """Push overlapping label positions apart along one axis, preserving order.

    The top-ten priority divisions are the densest, smallest divisions in the
    CMC core, so their centroids fall within a few hundred metres of each other
    and ten labels drawn at those centroids overlap into an unreadable mat. This
    nudges them apart on a single axis and keeps their relative order, so the
    leader lines joining label to division never cross.

    A single forward sweep then a single backward sweep, not an iterative
    physical relaxation: the result is deterministic, which a test can pin, and
    the ordering guarantee is exact rather than emergent.

    Args:
        wanted: Desired positions, one per label, in data units. Order is
            preserved as given - sort by it beforehand if that matters.
        min_gap: Minimum spacing between adjacent labels.
        low: Optional lower bound the block is kept inside.
        high: Optional upper bound.

    Returns:
        Adjusted positions, in the input order.

    Raises:
        ValueError: If ``min_gap`` is negative, or the labels cannot fit between
            ``low`` and ``high`` - which is a real answer, not a layout hiccup:
            it means too many labels were asked for.
    """
    if float(min_gap) < 0:
        raise ValueError(f"min_gap must be non-negative, got {min_gap}")
    values = [float(value) for value in wanted]
    if not values:
        return []
    if low is not None and high is not None:
        needed = (len(values) - 1) * float(min_gap)
        if needed > float(high) - float(low):
            raise ValueError(
                f"{len(values)} labels need {needed:.3g} of span at a "
                f"{float(min_gap):.3g} gap, but only "
                f"{float(high) - float(low):.3g} is available; label fewer"
            )

    # Forward: nothing may sit closer than min_gap to the label before it.
    for index in range(1, len(values)):
        values[index] = max(values[index], values[index - 1] + float(min_gap))
    # Backward: pull the whole block down if the forward sweep overshot the top.
    if high is not None and values[-1] > float(high):
        values[-1] = float(high)
        for index in range(len(values) - 2, -1, -1):
            values[index] = min(values[index], values[index + 1] - float(min_gap))
    if low is not None and values[0] < float(low):
        values[0] = float(low)
        for index in range(1, len(values)):
            values[index] = max(values[index], values[index - 1] + float(min_gap))
    return values


#: Share of the map's own width reserved to its RIGHT for the label column.
#:
#: The labelled divisions are the smallest and most tightly packed in the
#: district, so labels have to sit outside the map and be joined by leader
#: lines. Widening the axes limits to make room WITHOUT widening the canvas
#: shrinks the map by roughly a third and leaves white bands above and below it,
#: because the figure was already sized to the unpadded bounding box. The
#: builder therefore sizes the canvas with this gutter included and
#: :func:`annotate_top_zones` places labels inside the same gutter.
LABEL_GUTTER_FRACTION = 0.30

#: Inset of the label column from the map's right edge, as a share of width.
LABEL_GUTTER_INSET = 0.03


class _PaddedBounds:
    """Stand-in for a GeoDataFrame carrying widened ``total_bounds``.

    Only :func:`map_aspect_ratio` reads the layer when sizing a map canvas, and
    it reads nothing but ``total_bounds``.
    """

    def __init__(self, zones: Any, right_padding: float) -> None:
        min_x, min_y, max_x, max_y = (float(v) for v in zones.total_bounds)
        width = max_x - min_x
        self.total_bounds = (min_x, min_y, max_x + width * right_padding, max_y)


def label_gutter_bounds(zones: Any) -> tuple[float, float]:
    """``(label_x, right_limit)`` for a map drawn with a label gutter.

    Args:
        zones: ``geopandas.GeoDataFrame``.

    Returns:
        The x at which labels start, and the x the axes should extend to.
    """
    min_x, _, max_x, _ = (float(value) for value in zones.total_bounds)
    width = max_x - min_x
    return (
        max_x + width * LABEL_GUTTER_INSET,
        max_x + width * LABEL_GUTTER_FRACTION,
    )


def annotate_top_zones(
    axes: Any,
    merged: Any,
    params: dict[str, Any],
    top_n: int | None = None,
    rank_column: str = "rank_ahp",
    name_column: str = "adm4_name",
) -> int:
    """Label the highest-ranked divisions in a column beside the map.

    Labels go in a stack to the RIGHT of the map rather than on the polygons
    they name. Ten of the top divisions are sub-square-kilometre and adjacent,
    so on-polygon labels would overlap each other and obscure the choropleth
    they sit on; a leader line to an outside stack costs nothing and stays
    readable at any zoom.

    Args:
        axes: The map axes, already drawn.
        merged: ``geopandas.GeoDataFrame`` of zones joined to the ranking.
        params: Parsed params mapping.
        top_n: How many to label; defaults to ``report.label_top_n``.
        rank_column: Column holding the rank, 1 = highest priority.
        name_column: Column holding the division name.

    Returns:
        The number of labels drawn; zero when the frame carries no rank.
    """
    limit = int(params["report"]["label_top_n"] if top_n is None else top_n)
    if rank_column not in merged.columns or limit < 1:
        return 0

    labelled = merged.dropna(subset=[rank_column]).nsmallest(limit, rank_column)
    if labelled.empty:
        return 0

    centres = labelled.geometry.representative_point()
    frame = labelled.assign(
        _x=[point.x for point in centres], _y=[point.y for point in centres]
    ).sort_values("_y", ascending=False)

    min_x, min_y, max_x, max_y = (float(value) for value in merged.total_bounds)
    span_y = max_y - min_y
    gap = span_y / max(len(frame) + 1, 2)
    try:
        placed = spread_label_positions(
            list(frame["_y"])[::-1], gap, low=min_y, high=max_y
        )[::-1]
    except ValueError:
        placed = list(frame["_y"])

    label_x, right_limit = label_gutter_bounds(merged)
    for (_, row), y_position in zip(frame.iterrows(), placed):
        name = str(row.get(name_column, row.get("zone_id", "")))
        axes.annotate(
            f"{int(row[rank_column])}. {name}",
            xy=(float(row["_x"]), float(row["_y"])),
            xytext=(label_x, y_position),
            fontsize=6.5, va="center", ha="left", color="#111111",
            arrowprops={
                "arrowstyle": "-", "linewidth": 0.5, "color": "#555555",
                "shrinkA": 0, "shrinkB": 1,
            },
        )
    # The canvas was already sized with this gutter (see LABEL_GUTTER_FRACTION),
    # so setting the limits to match reveals the label column rather than
    # squeezing the map into it.
    axes.set_xlim(min_x, right_limit)
    return int(len(frame))


# =============================================================================
# Phase 8 - figure 11: the data-provenance table
# =============================================================================
#: Columns the provenance figure shows, and the width each gets, as a share of
#: the table. The CSV in ``data/outputs`` carries every column; the FIGURE drops
#: the free-text note, which is paragraphs long for some entries and would set
#: the row height for the whole table.
PROVENANCE_FIGURE_COLUMNS: tuple[tuple[str, str, float], ...] = (
    ("collection_id", "Collection ID / asset path", 0.26),
    ("type", "Type", 0.10),
    ("bands", "Bands used", 0.22),
    ("native_scale_m", "Native\nscale (m)", 0.07),
    ("scale_factor", "Scale factor", 0.21),
    ("temporal_coverage", "Coverage", 0.14),
)


def build_provenance_table_figure(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    title: str = "Data provenance: every source, band and scale factor",
) -> Any:
    """Build the data-provenance table as a figure.

    Rendered as a drawn table rather than as ``matplotlib``'s ``table`` helper,
    which cannot wrap a cell: several collection IDs and band lists are long
    enough that a non-wrapping table either overflows the canvas or has to be
    set in unreadably small type.

    Sources the project configures but never uses are drawn in grey and marked,
    because "configured" and "used" are different claims and a provenance table
    that conflates them overstates what the analysis rests on.

    Args:
        frame: The table :func:`colombo_uhi.reporting.provenance_frame` returns.
        params: Parsed params mapping.
        title: Figure title.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the frame is empty or lacks a shown column.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    if frame.empty:
        raise ValueError("the provenance frame is empty; there is nothing to tabulate")
    shown = [(name, header, width) for name, header, width in PROVENANCE_FIGURE_COLUMNS]
    missing = [name for name, _, _ in shown if name not in frame.columns]
    if missing:
        raise ValueError(
            f"provenance frame is missing column(s) {missing}; it has "
            f"{sorted(frame.columns)}"
        )

    width = float(params["report"]["width_inches"])
    #: Characters of 6.5 pt DejaVu Sans that fit in one inch, measured off the
    #: first render. Deliberately CONSERVATIVE: over-estimating it packs more
    #: characters onto a line than fit and one column runs into the next, which
    #: is what the first draft did to the band and scale-factor columns.
    chars_per_inch = 15.5
    left, right = 0.02, 0.99
    table_width = width * (right - left)

    rows: list[tuple[str, list[list[str]], bool]] = []
    for _, record in frame.iterrows():
        cells: list[list[str]] = []
        for name, _, share in shown:
            text = str(record[name]) if record[name] == record[name] else ""
            if name == "type":
                # "image_collection" has no break point, so textwrap splits it
                # mid-word into "image_collecti / on". Spaces give it one.
                text = text.replace("_", " ")
            limit = max(int(table_width * share * chars_per_inch), 8)
            cells.append(textwrap.wrap(text, width=limit) or [""])
        unused = str(record.get("used_by", "")).startswith("not ")
        rows.append((str(record["key"]), cells, unused))

    n_unused = sum(1 for _, _, unused in rows if unused)
    footer = _wrap_bullets(
        [
            "Generated from config/params.yaml by colombo_uhi.reporting, not "
            "typed. The full table, including the module that reads each source "
            "and the free-text notes, is data/outputs/data_provenance.csv.",
            "Every source is free and public. GN-division polygons are in no "
            "public dataset and are uploaded by the project owner as an Earth "
            "Engine asset.",
            f"{n_unused} configured source(s) are shown in grey: they are named "
            "in the configuration but referenced by no analysis step. Their "
            "consequences are in docs/limitations.md."
            if n_unused else
            "Every configured source is referenced by at least one analysis step.",
        ]
    )

    line_height = 0.115                 # inches per wrapped line of 6.5 pt text
    row_pad = 0.07
    # +1 line for the italic config key printed under the collection ID. The
    # first draft drew that key at the row's bottom edge WITHOUT reserving space
    # for it, so on every single-line row it landed on the row below and struck
    # through the next collection ID.
    line_counts = [
        max(len(cell) for cell in cells) + 1 for _, cells, _ in rows
    ]
    heights = [count * line_height + row_pad for count in line_counts]
    header_height = 0.34
    reserved_inches = footer_inches(footer)
    height = sum(heights) + header_height + 0.75 + reserved_inches

    figure = Figure(figsize=(width, height))
    FigureCanvasAgg(figure)
    axes = figure.add_axes((left, reserved_inches / height, right - left,
                            1 - (reserved_inches + 0.55) / height))
    axes.set_axis_off()
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)

    # Column left edges, in axes coordinates. The key column is drawn in the
    # first column's space, above the collection ID, so long IDs get full width.
    edges: list[float] = []
    cursor = 0.0
    for _, _, share in shown:
        edges.append(cursor)
        cursor += share

    total = sum(heights) + header_height
    y = 1.0
    axes.add_patch(
        Rectangle(
            (0, 1 - header_height / total), 1, header_height / total,
            facecolor="#e8e8e8", edgecolor="none", transform=axes.transAxes,
        )
    )
    for (name, header, _), x in zip(shown, edges):
        axes.text(
            x + 0.004, 1 - header_height / total / 2, header,
            fontsize=7, fontweight="bold", va="center", ha="left",
        )
    y -= header_height / total

    for (key, cells, unused), row_height, line_count in zip(rows, heights, line_counts):
        share_height = row_height / total
        colour = "#8a8a8a" if unused else "#111111"
        if unused:
            axes.add_patch(
                Rectangle(
                    (0, y - share_height), 1, share_height,
                    facecolor="#f6f6f6", edgecolor="none", transform=axes.transAxes,
                )
            )
        axes.axhline(y, color="#d0d0d0", linewidth=0.4)
        top = y - 0.4 * line_height / total
        for cell, x in zip(cells, edges):
            for index, line in enumerate(cell):
                axes.text(
                    x + 0.004, top - index * line_height / total, line,
                    fontsize=6.5, va="top", ha="left", color=colour,
                    family="DejaVu Sans",
                )
        # The config key, small and grey, on the line reserved for it.
        axes.text(
            edges[0] + 0.004, top - (line_count - 1) * line_height / total,
            key, fontsize=5.5, va="top", ha="left", color="#999999",
            style="italic",
        )
        y -= share_height

    figure.suptitle(title, fontsize=12, y=1 - 0.18 / height)
    figure.text(
        0.01, 0.10 / height, footer, fontsize=7, va="bottom", ha="left",
        color="#444444",
    )
    return figure


def plot_provenance_table(
    frame: "pd.DataFrame",
    out_path: str | Path,
    params: dict[str, Any],
    title: str = "Data provenance: every source, band and scale factor",
) -> Path:
    """Write the provenance table. See :func:`build_provenance_table_figure`."""
    return _save_figure(
        build_provenance_table_figure(frame, params, title=title),
        out_path,
        dpi=report_dpi(params),
    )


# =============================================================================
# Phase 8 - the report manifest
# =============================================================================
#: The eleven report figures, in the order the report presents them.
#:
#: ``inputs`` names what each figure needs, in the vocabulary notebook 08 uses
#: for its discovery cell, so a missing input is reported once and by name
#: rather than as a traceback partway through a render loop.
REPORT_FIGURES: tuple[dict[str, Any], ...] = (
    {
        "index": 1, "slug": "decadal_lst",
        "title": "Decadal mean dry-season LST, Landsat and MODIS",
        "builder": "build_decadal_lst_panel_figure",
        "inputs": ("decadal_landsat_tif", "decadal_modis_tif", "sensor_offsets_csv"),
    },
    {
        "index": 2, "slug": "sen_slope_fdr",
        "title": "Sen's slope with FDR-significance stippling",
        "builder": "build_sen_slope_stipple_figure",
        "inputs": ("trend_fdr_tif",),
    },
    {
        "index": 3, "slug": "valid_observations",
        "title": "Per-pixel valid-observation count",
        "builder": "build_obs_count_figure",
        "inputs": ("obs_count_tif",),
    },
    {
        "index": 4, "slug": "utfvi_epochs",
        "title": "UTFVI six-class maps by epoch",
        "builder": "build_utfvi_epoch_maps_figure",
        "inputs": ("utfvi_class_tif",),
    },
    {
        "index": 5, "slug": "hotspots",
        "title": "Getis-Ord Gi* and emerging hot spots",
        "builder": "build_hotspot_map_figure",
        "inputs": ("zones_geojson", "gi_star_csv", "ehsa_csv"),
    },
    {
        "index": 6, "slug": "suhii_series",
        "title": "SUHII time series by source and rural definition",
        "builder": "build_suhii_figure",
        "inputs": ("suhii_csv",),
    },
    {
        "index": 7, "slug": "lst_vs_indices",
        "title": "LST against NDVI and NDBI, faceted by epoch",
        "builder": "build_lst_vs_index_facet_figure",
        "inputs": ("driver_samples_csv",),
    },
    {
        "index": 8, "slug": "gwr_coefficients",
        "title": "GWR local coefficients",
        "builder": "build_gwr_coefficient_figure",
        "inputs": ("zones_geojson", "gwr_csv"),
    },
    {
        "index": 9, "slug": "greening_scenario",
        "title": "Greening counterfactual: baseline, greened, difference",
        "builder": "build_scenario_triptych_figure",
        "inputs": ("counterfactual_baseline_tif", "counterfactual_greened_tif",
                   "counterfactual_delta_tif"),
    },
    {
        "index": 10, "slug": "greening_priority",
        "title": "Greening priority by GN division, top zones labelled",
        "builder": "build_greening_priority_map_figure",
        "inputs": ("zones_geojson", "priority_csv"),
    },
    {
        "index": 11, "slug": "data_provenance",
        "title": "Data provenance",
        "builder": "build_provenance_table_figure",
        "inputs": (),
    },
)


def report_manifest(params: dict[str, Any]) -> list[dict[str, Any]]:
    """The eleven report figures, with their output paths resolved.

    Notebook 08 iterates this rather than hardcoding a list, so the notebook and
    the module cannot disagree about how many figures there are or where they
    go. A test pins that every entry names a real builder and that no two claim
    the same path.

    Args:
        params: Parsed params mapping.

    Returns:
        One mapping per figure: ``index``, ``slug``, ``title``, ``builder``,
        ``inputs`` and the resolved ``path``.
    """
    return [
        {**entry, "path": report_figure_path(params, entry["index"], entry["slug"])}
        for entry in REPORT_FIGURES
    ]


def missing_inputs(
    available: Mapping[str, Any], params: dict[str, Any]
) -> dict[int, list[str]]:
    """Which figures cannot be drawn, and what each is missing.

    Called before anything is rendered. A render loop that discovers a missing
    raster on figure 9 has already spent several minutes drawing eight others
    and reports the problem as a traceback; this reports every gap at once, by
    name, before the first pixel.

    Args:
        available: Mapping of input name to whatever was found - a path, a
            frame, an array. A value of ``None`` counts as absent.
        params: Parsed params mapping.

    Returns:
        Mapping of figure index to its missing input names. Figures that can be
        drawn do not appear.
    """
    gaps: dict[int, list[str]] = {}
    for entry in report_manifest(params):
        absent = [
            name for name in entry["inputs"]
            if available.get(name) is None
        ]
        if absent:
            gaps[int(entry["index"])] = absent
    return gaps
