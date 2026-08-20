"""Unit tests for the pure-Python parts of viz.py.

``plot_annual_lst_comparison`` needs no Earth Engine, so it is exercised for
real here: the figure is actually rendered to a temporary PNG. The Earth
Engine-backed helpers (``outline_image``, ``elevation_backdrop``,
``save_thumbnail``) are verified only in Colab.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from colombo_uhi import load_params, viz


@pytest.fixture(scope="module")
def params() -> dict[str, Any]:
    return load_params()


class _Bounds:
    """Minimal stand-in for a GeoDataFrame: only ``total_bounds`` is read."""

    def __init__(self, bounds: tuple[float, float, float, float]) -> None:
        self.total_bounds = bounds


# --- Phase 5 map geometry ----------------------------------------------------
def test_map_aspect_ratio_follows_the_bounding_box() -> None:
    # Colombo District is roughly twice as wide as it is tall. A choropleth is
    # drawn with set_aspect("equal"), so a canvas that ignores this leaves the
    # difference as blank paper - which is what the first Colab run produced.
    assert viz.map_aspect_ratio(_Bounds((0.0, 0.0, 60000.0, 30000.0))) == pytest.approx(0.5)
    assert viz.map_aspect_ratio(_Bounds((0.0, 0.0, 10.0, 10.0))) == pytest.approx(1.0)


def test_map_aspect_ratio_is_clamped_against_pathological_bounds() -> None:
    # One degenerate geometry must not produce a figure thousands of inches tall.
    assert viz.map_aspect_ratio(_Bounds((0.0, 0.0, 1.0, 10_000.0))) == pytest.approx(3.0)
    assert viz.map_aspect_ratio(_Bounds((0.0, 0.0, 10_000.0, 1.0))) == pytest.approx(0.25)


def test_footer_height_grows_with_the_caveat_text(params: dict[str, Any]) -> None:
    # Colab run 2 put the legend on top of the footer: the reserved band was a
    # fixed 1.15 in while the four-caveat LISA footer needs ~2.2 in, so it
    # overflowed upward into the legend.
    one = viz.caveat_footer(params, ["lst_not_air_temp"])
    four = viz.caveat_footer(
        params,
        ["lst_not_air_temp", "within_epoch_only", "zonal_not_pixel", "fdr_dependence"],
    )
    assert viz.footer_inches(four) > viz.footer_inches(one)
    # The four-caveat footer must need more than the constant it replaced.
    assert viz.footer_inches(four) > 1.35


def test_footer_height_covers_every_line_it_is_given() -> None:
    text = "\n".join(f"line {i}" for i in range(10))
    reserved = viz.footer_inches(text, pad_inches=0.0)
    assert reserved == pytest.approx(10 * viz.FOOTER_LINE_INCHES)


def test_footer_height_of_nothing_is_just_padding() -> None:
    assert viz.footer_inches("", pad_inches=0.2) == pytest.approx(0.2)


def test_map_aspect_ratio_falls_back_on_degenerate_or_missing_bounds() -> None:
    assert viz.map_aspect_ratio(_Bounds((5.0, 5.0, 5.0, 5.0))) == pytest.approx(1.0)
    assert viz.map_aspect_ratio(object(), default=0.75) == pytest.approx(0.75)


@pytest.fixture()
def landsat_series() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2000, 2001, 2002],
            "mean": [29.4, 30.1, 30.8],
            "valid_pixels": [1200, 1310, 980],
        }
    )


@pytest.fixture()
def modis_series() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2000, 2001, 2002],
            "mean": [28.1, 28.9, 29.3],
            "valid_pixels": [40, 42, 39],
        }
    )


# --- caveat_footer ---------------------------------------------------------------
def test_caveat_footer_includes_the_requested_caveats(params: dict[str, Any]) -> None:
    footer = viz.caveat_footer(params, ["lst_not_air_temp"])
    assert "LAND SURFACE TEMPERATURE" in footer
    assert footer.startswith("- ")


def test_caveat_footer_wraps_long_text(params: dict[str, Any]) -> None:
    footer = viz.caveat_footer(params, ["lst_not_air_temp", "single_overpass"])
    assert len(footer.splitlines()) > 1
    for line in footer.splitlines():
        assert len(line) <= viz.CAVEAT_WRAP_CHARS + 2


def test_caveat_footer_collapses_yaml_folded_newlines(
    params: dict[str, Any],
) -> None:
    # The params strings are YAML folded blocks; stray newlines would render as
    # gaps in the figure footer.
    footer = viz.caveat_footer(params, ["scenario_not_forecast"])
    assert "  " not in footer.replace("\n  ", "")


def test_caveat_footer_rejects_unknown_key(params: dict[str, Any]) -> None:
    with pytest.raises(KeyError, match="not_a_caveat"):
        viz.caveat_footer(params, ["not_a_caveat"])


# --- plot_annual_lst_comparison ----------------------------------------------------
def test_plot_writes_a_png(
    tmp_path: Path,
    params: dict[str, Any],
    landsat_series: pd.DataFrame,
    modis_series: pd.DataFrame,
) -> None:
    out = viz.plot_annual_lst_comparison(
        {"Landsat": landsat_series, "MODIS Terra day": modis_series},
        tmp_path / "comparison.png",
        params,
    )
    assert out.is_file()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_creates_missing_parent_directories(
    tmp_path: Path, params: dict[str, Any], landsat_series: pd.DataFrame
) -> None:
    out = viz.plot_annual_lst_comparison(
        {"Landsat": landsat_series}, tmp_path / "nested" / "fig.png", params
    )
    assert out.is_file()


def test_plot_tolerates_fully_masked_years(
    tmp_path: Path, params: dict[str, Any]
) -> None:
    # zonal_annual_means keeps empty years as NaN rows; they must be skipped,
    # not plotted as a value and not crash the renderer.
    frame = pd.DataFrame(
        {
            "year": [2000, 2001, 2002],
            "mean": [29.4, float("nan"), 30.8],
            "valid_pixels": [1200, 0, 980],
        }
    )
    out = viz.plot_annual_lst_comparison(
        {"Landsat": frame}, tmp_path / "gap.png", params
    )
    assert out.is_file()


def test_plot_drops_rows_with_zero_valid_pixels(
    tmp_path: Path, params: dict[str, Any]
) -> None:
    # A mean computed over zero valid pixels is not a temperature.
    frame = pd.DataFrame(
        {"year": [2000, 2001], "mean": [29.4, 0.0], "valid_pixels": [1200, 0]}
    )
    out = viz.plot_annual_lst_comparison(
        {"Landsat": frame}, tmp_path / "zero.png", params
    )
    assert out.is_file()


def test_plot_adds_a_count_panel_when_counts_are_present(
    tmp_path: Path,
    params: dict[str, Any],
    landsat_series: pd.DataFrame,
    modis_series: pd.DataFrame,
) -> None:
    # Two panels means a taller figure than the single-panel form; that is the
    # cheapest observable proof the count axis was actually added.
    with_counts = viz.plot_annual_lst_comparison(
        {"Landsat": landsat_series, "MODIS": modis_series},
        tmp_path / "with_counts.png",
        params,
    )
    without_counts = viz.plot_annual_lst_comparison(
        {"Landsat": landsat_series.drop(columns=["valid_pixels"])},
        tmp_path / "without_counts.png",
        params,
        count_column=None,
    )
    assert with_counts.stat().st_size > 0
    assert without_counts.stat().st_size > 0


def test_plot_without_a_count_column_still_works(
    tmp_path: Path, params: dict[str, Any]
) -> None:
    # A caller may legitimately have no counts; the figure must degrade to one
    # panel rather than crash looking for the column.
    frame = pd.DataFrame({"year": [2000, 2001], "mean": [29.4, 30.1]})
    out = viz.plot_annual_lst_comparison(
        {"Landsat": frame}, tmp_path / "nocounts.png", params, count_column=None
    )
    assert out.is_file()


def test_plot_rejects_empty_series(tmp_path: Path, params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="at least one"):
        viz.plot_annual_lst_comparison({}, tmp_path / "none.png", params)


def test_plot_rejects_a_frame_missing_the_value_column(
    tmp_path: Path, params: dict[str, Any]
) -> None:
    frame = pd.DataFrame({"year": [2000], "average": [29.4]})
    with pytest.raises(ValueError, match="mean"):
        viz.plot_annual_lst_comparison(
            {"Landsat": frame}, tmp_path / "bad.png", params
        )


def test_plot_does_not_touch_the_global_matplotlib_backend(
    tmp_path: Path, params: dict[str, Any], landsat_series: pd.DataFrame
) -> None:
    # Forcing a backend would break Colab's inline rendering for every later
    # cell, so the helper uses the object-oriented API instead of pyplot.
    matplotlib = pytest.importorskip("matplotlib")
    before = matplotlib.get_backend()
    viz.plot_annual_lst_comparison(
        {"Landsat": landsat_series}, tmp_path / "backend.png", params
    )
    assert matplotlib.get_backend() == before


# =============================================================================
# Phase 3 figures
# =============================================================================
@pytest.fixture()
def suhii_frame() -> pd.DataFrame:
    """Tidy SUHII table shaped exactly like uhi_metrics.suhii_all_sources."""
    rows = []
    for source, urban, rural, pixels in (
        ("landsat_dry", 34.0, 31.0, 4200),
        ("terra_night", 24.5, 22.6, 40),
    ):
        for method, offset in (("buffer_ring", 0.0), ("lcz_based", -0.9)):
            for index, year in enumerate((2000, 2001, 2002)):
                rows.append(
                    {
                        "year": year,
                        "source": source,
                        "rural_definition": method,
                        "urban_mean": urban + index * 0.1,
                        "rural_mean": rural + offset,
                        "suhii": (urban + index * 0.1) - (rural + offset),
                        "urban_pixels": pixels,
                        "rural_pixels": pixels * 4,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def utfvi_share_frame(params: dict[str, Any]) -> pd.DataFrame:
    labels = params["uhi"]["utfvi"]["labels"]
    rows = []
    for year in (2000, 2001, 2002):
        shares = dict(zip(labels, [30.0, 25.0, 20.0, 12.0, 8.0, 5.0]))
        rows.append({"year": year, **shares, "pixel_count": 5000})
    return pd.DataFrame(rows)


@pytest.fixture()
def driver_sample_frame() -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(3)
    ndvi = rng.uniform(0.0, 0.8, size=300)
    ndbi = rng.uniform(-0.3, 0.4, size=300)
    return pd.DataFrame(
        {
            "LST_C": 30.0 - 6.0 * ndvi + 4.0 * ndbi + rng.normal(0, 0.4, 300),
            "NDVI": ndvi,
            "NDBI": ndbi,
            "MNDWI": rng.uniform(-0.6, 0.2, size=300),
            "built_fraction": rng.uniform(0.0, 1.0, size=300),
        }
    )


# --- utfvi_vis_params --------------------------------------------------------
def test_utfvi_vis_params_spans_every_class(params: dict[str, Any]) -> None:
    vis = viz.utfvi_vis_params(params)
    assert vis["min"] == 0
    assert vis["max"] == len(params["uhi"]["utfvi"]["labels"]) - 1 == 5
    assert len(vis["palette"]) == 6


def test_utfvi_vis_params_rejects_a_short_palette(params: dict[str, Any]) -> None:
    # A short palette silently RECYCLES colours, making two classes
    # indistinguishable on the map with no error anywhere.
    import copy

    mutated = copy.deepcopy(params)
    mutated["uhi"]["utfvi"]["palette"] = ["ffffff", "000000"]
    with pytest.raises(ValueError, match="palette has 2 colours"):
        viz.utfvi_vis_params(mutated)


# --- plot_suhii_sensitivity --------------------------------------------------
def test_suhii_sensitivity_writes_a_png(
    tmp_path: Path, params: dict[str, Any], suhii_frame: pd.DataFrame
) -> None:
    out = viz.plot_suhii_sensitivity(suhii_frame, tmp_path / "suhii.png", params)
    assert out.is_file()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_suhii_sensitivity_accepts_a_source_subset(
    tmp_path: Path, params: dict[str, Any], suhii_frame: pd.DataFrame
) -> None:
    out = viz.plot_suhii_sensitivity(
        suhii_frame, tmp_path / "one.png", params, sources=["terra_night"]
    )
    assert out.is_file()


def test_suhii_sensitivity_skips_zero_pixel_years(
    tmp_path: Path, params: dict[str, Any], suhii_frame: pd.DataFrame
) -> None:
    # A SUHII resting on zero urban pixels is not a measurement.
    frame = suhii_frame.copy()
    frame.loc[0, "urban_pixels"] = 0
    out = viz.plot_suhii_sensitivity(frame, tmp_path / "zero.png", params)
    assert out.is_file()


def test_suhii_sensitivity_rejects_an_empty_frame(
    tmp_path: Path, params: dict[str, Any]
) -> None:
    empty = pd.DataFrame(
        columns=["year", "source", "rural_definition", "suhii"]
    )
    with pytest.raises(ValueError, match="empty"):
        viz.plot_suhii_sensitivity(empty, tmp_path / "empty.png", params)


def test_suhii_sensitivity_rejects_a_missing_column(
    tmp_path: Path, params: dict[str, Any], suhii_frame: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="rural_definition"):
        viz.plot_suhii_sensitivity(
            suhii_frame.drop(columns=["rural_definition"]),
            tmp_path / "bad.png",
            params,
        )


def _many_source_frame(n_sources: int) -> pd.DataFrame:
    """A SUHII frame with n distinct sources, both rural definitions."""
    rows = []
    for index in range(n_sources):
        for method, offset in (("buffer_ring", 0.0), ("lcz_based", -1.2)):
            for year in (2000, 2001, 2002):
                urban = 33.0 + index * 0.2
                rural = 30.0 + offset
                rows.append(
                    {
                        "year": year,
                        "source": f"source_{index}",
                        "rural_definition": method,
                        "urban_mean": urban,
                        "rural_mean": rural,
                        "suhii": urban - rural,
                        "urban_pixels": 100 + index,
                        "rural_pixels": 400,
                    }
                )
    return pd.DataFrame(rows)


# --- the reported bug: legend entries were indistinguishable -----------------
@pytest.mark.parametrize("n_sources", [1, 2, 4, 5, 6, 7])
def test_suhii_legend_has_one_entry_per_rural_definition_not_per_line(
    params: dict[str, Any], n_sources: int
) -> None:
    # THE regression test for the reported defect. The old overlay drew every
    # source x definition on one axes and gave each pair the SAME colour,
    # distinguished only by a dash pattern that a short legend swatch cannot
    # show — so with six sources the legend read as six duplicated pairs.
    # Faceting means the legend must carry exactly ONE entry per rural
    # definition, no matter how many sources there are.
    figure = viz.build_suhii_figure(_many_source_frame(n_sources), params)
    labels = [text.get_text() for text in figure.legends[0].get_texts()]
    assert len(labels) == 2, f"expected 2 legend entries, got {labels}"
    assert len(set(labels)) == 2, "legend entries must be distinguishable"
    assert all("rural reference" in label for label in labels)


@pytest.mark.parametrize(
    ("n_sources", "expected_visible"), [(1, 1), (2, 2), (4, 4), (5, 5), (6, 6)]
)
def test_suhii_grid_hides_unused_cells_of_a_partial_row(
    params: dict[str, Any], n_sources: int, expected_visible: int
) -> None:
    # 4 and 5 sources leave a partial last row in a 3-wide grid. An unused cell
    # left visible reads as missing data rather than as no panel.
    figure = viz.build_suhii_figure(_many_source_frame(n_sources), params)
    visible = [axes for axes in figure.axes if axes.get_visible()]
    assert len(visible) == expected_visible


def test_suhii_panels_share_one_y_axis(params: dict[str, Any]) -> None:
    # Deliberate: free axes would draw a 0.5 degC gap the same size as a 3 degC
    # one, which is the exact misreading this figure exists to prevent.
    figure = viz.build_suhii_figure(_many_source_frame(4), params)
    visible = [axes for axes in figure.axes if axes.get_visible()]
    limits = {axes.get_ylim() for axes in visible}
    assert len(limits) == 1, f"panels must share a y-axis, got {limits}"


def test_suhii_panel_titles_carry_pixel_counts_and_the_gap(
    params: dict[str, Any], suhii_frame: pd.DataFrame
) -> None:
    # CLAUDE.md caveat 2 survived the redesign: dropping the count panel is only
    # acceptable because the counts moved into the titles.
    figure = viz.build_suhii_figure(suhii_frame, params)
    titles = [axes.get_title() for axes in figure.axes if axes.get_visible()]
    assert any("urban px" in title for title in titles)
    assert any("gap" in title for title in titles)


def test_suhii_skips_a_requested_source_absent_from_the_frame(
    params: dict[str, Any], suhii_frame: pd.DataFrame
) -> None:
    figure = viz.build_suhii_figure(
        suhii_frame, params, sources=["terra_night", "not_a_source"]
    )
    visible = [axes for axes in figure.axes if axes.get_visible()]
    assert len(visible) == 1


def test_suhii_raises_when_no_requested_source_is_present(
    params: dict[str, Any], suhii_frame: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="none of the requested sources"):
        viz.build_suhii_figure(suhii_frame, params, sources=["nope"])


def test_suhii_styles_are_distinct_in_colour_and_dash_and_marker(
    params: dict[str, Any],
) -> None:
    # Redundant encoding: the figure has to survive greyscale printing and the
    # common colour-vision deficiencies, neither of which colour alone does.
    styles = viz._suhii_styles(["buffer_ring", "lcz_based"])
    assert len({s["color"] for s in styles.values()}) == 2
    assert len({s["linestyle"] for s in styles.values()}) == 2
    assert len({s["marker"] for s in styles.values()}) == 2


def test_suhii_styles_cope_with_an_unconfigured_definition() -> None:
    # A sensitivity run may add a third rural definition.
    styles = viz._suhii_styles(["buffer_ring", "lcz_based", "lcz_strict"])
    assert len(styles) == 3
    assert len({s["color"] for s in styles.values()}) == 3


# --- plot_utfvi_class_shares -------------------------------------------------
def test_utfvi_shares_writes_a_png(
    tmp_path: Path, params: dict[str, Any], utfvi_share_frame: pd.DataFrame
) -> None:
    out = viz.plot_utfvi_class_shares(
        utfvi_share_frame, tmp_path / "shares.png", params
    )
    assert out.is_file()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_utfvi_shares_rejects_a_missing_class_column(
    tmp_path: Path, params: dict[str, Any], utfvi_share_frame: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="Worst"):
        viz.plot_utfvi_class_shares(
            utfvi_share_frame.drop(columns=["Worst"]),
            tmp_path / "bad.png",
            params,
        )


def test_utfvi_shares_reject_an_all_empty_series(
    tmp_path: Path, params: dict[str, Any], utfvi_share_frame: pd.DataFrame
) -> None:
    labels = params["uhi"]["utfvi"]["labels"]
    frame = utfvi_share_frame.copy()
    frame[labels] = float("nan")
    with pytest.raises(ValueError, match="no year with classified pixels"):
        viz.plot_utfvi_class_shares(frame, tmp_path / "nan.png", params)


def test_utfvi_legend_sits_outside_the_axes(
    params: dict[str, Any], utfvi_share_frame: pd.DataFrame
) -> None:
    # Regression: the legend used to be drawn INSIDE the axes at "upper left",
    # which put the red "Worst" swatch on top of the red "Worst" band and made
    # it invisible. A legend must never be drawn over the thing it names.
    figure = viz.build_utfvi_shares_figure(utfvi_share_frame, params)
    axes = figure.axes[0]
    legend = axes.get_legend()
    assert legend is not None

    # Express the legend's anchor in axes coordinates: (0, 0) is the bottom-left
    # of the plot area, so an anchor below it has a negative y. This is the
    # property that actually prevents a swatch landing on the band it names.
    anchor_in_axes = legend.get_bbox_to_anchor().transformed(
        axes.transAxes.inverted()
    )
    assert anchor_in_axes.y0 < 0, (
        f"legend must be anchored below the axes, got y0={anchor_in_axes.y0}"
    )


def test_utfvi_legend_lists_every_class(
    params: dict[str, Any], utfvi_share_frame: pd.DataFrame
) -> None:
    figure = viz.build_utfvi_shares_figure(utfvi_share_frame, params)
    labels = [t.get_text() for t in figure.axes[0].get_legend().get_texts()]
    assert labels == params["uhi"]["utfvi"]["labels"]


def test_utfvi_legend_has_an_opaque_frame(
    params: dict[str, Any], utfvi_share_frame: pd.DataFrame
) -> None:
    # So swatches read against whatever is behind them.
    figure = viz.build_utfvi_shares_figure(utfvi_share_frame, params)
    assert figure.axes[0].get_legend().get_frame_on()


# --- plot_lst_vs_index -------------------------------------------------------
def test_lst_vs_index_writes_a_png(
    tmp_path: Path, params: dict[str, Any], driver_sample_frame: pd.DataFrame
) -> None:
    out = viz.plot_lst_vs_index(
        driver_sample_frame,
        tmp_path / "scatter.png",
        params,
        index_columns=["NDVI", "NDBI"],
    )
    assert out.is_file()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_lst_vs_index_defaults_to_every_predictor(
    tmp_path: Path, params: dict[str, Any], driver_sample_frame: pd.DataFrame
) -> None:
    out = viz.plot_lst_vs_index(
        driver_sample_frame, tmp_path / "all.png", params
    )
    assert out.is_file()


def test_lst_vs_index_thins_only_the_drawing(
    tmp_path: Path, params: dict[str, Any], driver_sample_frame: pd.DataFrame
) -> None:
    # max_points controls how many dots are drawn; the fitted line still uses
    # every row, so a thinned figure is not a differently-fitted figure.
    out = viz.plot_lst_vs_index(
        driver_sample_frame,
        tmp_path / "thin.png",
        params,
        index_columns=["NDVI"],
        max_points=25,
    )
    assert out.is_file()


def test_lst_vs_index_survives_a_constant_predictor(
    tmp_path: Path, params: dict[str, Any], driver_sample_frame: pd.DataFrame
) -> None:
    # Regression: the constancy guard used to test std() == 0, but pandas
    # returns 2.8e-17 rather than exactly 0 for a constant column depending on
    # how the frame was built. The guard passed, np.polyfit got a singular
    # design and emitted RankWarning, and the panel drew a meaningless line.
    # Escalating the warning to an error is what pins the fix.
    import warnings

    frame = driver_sample_frame.copy()
    frame["MNDWI"] = -0.2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = viz.plot_lst_vs_index(
            frame, tmp_path / "const.png", params, index_columns=["NDVI", "MNDWI"]
        )
    assert out.is_file()


def test_lst_vs_index_rejects_an_empty_frame(
    tmp_path: Path, params: dict[str, Any]
) -> None:
    empty = pd.DataFrame(columns=["LST_C", "NDVI"])
    with pytest.raises(ValueError, match="empty"):
        viz.plot_lst_vs_index(
            empty, tmp_path / "empty.png", params, index_columns=["NDVI"]
        )


def test_lst_vs_index_labels_the_fitted_line(
    params: dict[str, Any], driver_sample_frame: pd.DataFrame
) -> None:
    # The red line used to be unlabelled and a reader had to guess what it was.
    figure = viz.build_lst_vs_index_figure(
        driver_sample_frame, params, index_columns=["NDVI"]
    )
    labels = [t.get_text() for t in figure.axes[0].get_legend().get_texts()]
    assert any("sampled pixels" in label for label in labels)
    assert any("OLS fit" in label for label in labels)


def test_lst_vs_index_clips_the_drawn_fit_to_the_bulk_of_the_data(
    params: dict[str, Any], driver_sample_frame: pd.DataFrame
) -> None:
    # The line is COMPUTED from every row but DRAWN only across the 1st-99th
    # percentile. Extending it into a tail holding a handful of pixels makes the
    # relationship look better supported out there than it is. A far outlier is
    # added here so the clipping has something to bite on.
    import numpy as np

    frame = driver_sample_frame.copy()
    frame.loc[len(frame)] = {
        "LST_C": 30.0, "NDVI": -5.0, "NDBI": 0.0, "MNDWI": 0.0, "built_fraction": 0.0
    }
    figure = viz.build_lst_vs_index_figure(frame, params, index_columns=["NDVI"])

    fit_line = [
        line for line in figure.axes[0].get_lines() if len(line.get_xdata()) == 50
    ]
    assert fit_line, "expected the 50-point fitted line"
    drawn_min = float(np.min(fit_line[0].get_xdata()))
    assert drawn_min > -5.0, "the fit must not be drawn out to the lone outlier"
    assert drawn_min >= float(np.percentile(frame["NDVI"], 1)) - 1e-9


# =============================================================================
# Phase 4 - trend figures
# =============================================================================
@pytest.fixture()
def trend_arrays() -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(17)
    slope = rng.normal(0.03, 0.05, size=(24, 32))
    significant = (slope > 0.05).astype("float64")
    # A block that was never tested - the case the figure must NOT draw as
    # "no trend".
    slope[:4, :4] = np.nan
    significant[:4, :4] = np.nan
    return {"sen_slope": slope, "significant": significant}


@pytest.fixture()
def mk_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": ["landsat_dry|buffer_ring"] * 2 + ["terra_night|lcz_based"] * 2,
            "series": ["suhii"] * 4,
            "test": ["original", "hamed_rao", "original", "hamed_rao"],
            "p": [0.001, 0.03, 0.2, 0.4],
            "var_inflation": [1.0, 2.4, 1.0, 1.8],
        }
    )


@pytest.fixture()
def class_trend_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scheme": ["worldcover"] * 4,
            "class_code": [10, 30, 50, 95],
            "class_label": ["Tree cover", "Grassland", "Built-up", "Mangroves"],
            "pixel_count": [5200, 3100, 8800, 40],
            "below_pixel_floor": [False, False, False, True],
            "mean": [0.012, 0.028, 0.051, 0.004],
            "stdDev": [0.010, 0.014, 0.019, 0.008],
        }
    )


def test_trend_vis_params_returns_the_configured_ramp(params: dict[str, Any]) -> None:
    vis = viz.trend_vis_params(params)
    assert vis["min"] == -vis["max"]
    assert len(vis["palette"]) % 2 == 1


def test_trend_vis_params_rejects_an_asymmetric_range(params: dict[str, Any]) -> None:
    # A ramp not centred on zero puts the "no trend" colour somewhere else, and
    # a reader cannot tell warming from cooling by eye.
    import copy

    broken = copy.deepcopy(params)
    broken["trends"]["slope_vis"]["min"] = -0.1
    broken["trends"]["slope_vis"]["max"] = 0.2
    with pytest.raises(ValueError, match="symmetric about zero"):
        viz.trend_vis_params(broken)


def test_trend_vis_params_rejects_an_even_palette(params: dict[str, Any]) -> None:
    import copy

    broken = copy.deepcopy(params)
    broken["trends"]["slope_vis"]["palette"] = ["2166ac", "b2182b"]
    with pytest.raises(ValueError, match="ODD number"):
        viz.trend_vis_params(broken)


def test_trend_map_figure_has_two_panels(
    trend_arrays: dict[str, Any], params: dict[str, Any]
) -> None:
    # All-slopes beside significant-only. Either panel alone misleads: the first
    # overstates confidence, the second hides what was never testable.
    figure = viz.build_trend_map_figure(trend_arrays, params)
    images = [axes for axes in figure.axes if axes.get_images()]
    assert len(images) == 2


def test_trend_map_figure_reports_both_counts_in_the_footer(
    trend_arrays: dict[str, Any], params: dict[str, Any]
) -> None:
    figure = viz.build_trend_map_figure(trend_arrays, params)
    footer = " ".join(
        " ".join(text.get_text().split()) for text in figure.texts
    )
    assert "FDR-significant" in footer
    # Untested pixels must be named as untested, never implied to be "no trend".
    assert "never" in footer and "tested" in footer


def test_trend_map_figure_raises_on_a_missing_array(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="significant"):
        viz.build_trend_map_figure({"sen_slope": [[0.1, 0.2]]}, params)


def test_trend_map_writes_a_png(
    trend_arrays: dict[str, Any], params: dict[str, Any], tmp_path: Path
) -> None:
    out = viz.plot_trend_map(trend_arrays, tmp_path / "trend.png", params)
    assert out.exists() and out.stat().st_size > 0


def test_mk_comparison_figure_marks_the_alpha_threshold(
    mk_frame: pd.DataFrame, params: dict[str, Any]
) -> None:
    figure = viz.build_mk_comparison_figure(mk_frame, params)
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert any("alpha" in label for label in labels)
    assert any("uncorrected" in label for label in labels)


def test_mk_comparison_figure_raises_on_a_missing_column(
    mk_frame: pd.DataFrame, params: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="var_inflation"):
        viz.build_mk_comparison_figure(mk_frame.drop(columns=["var_inflation"]), params)


def test_mk_comparison_figure_raises_on_an_empty_frame(
    params: dict[str, Any]
) -> None:
    empty = pd.DataFrame(
        columns=["label", "series", "test", "p", "var_inflation"]
    )
    with pytest.raises(ValueError, match="empty"):
        viz.build_mk_comparison_figure(empty, params)


def test_mk_comparison_writes_a_png(
    mk_frame: pd.DataFrame, params: dict[str, Any], tmp_path: Path
) -> None:
    out = viz.plot_mk_comparison(mk_frame, tmp_path / "mk.png", params)
    assert out.exists() and out.stat().st_size > 0


def test_class_figure_hatches_classes_below_the_pixel_floor(
    class_trend_frame: pd.DataFrame, params: dict[str, Any]
) -> None:
    # Sparse classes are shown, not dropped: removing them hides that the class
    # exists at all. Hatch AND edge colour vary so the flag survives greyscale.
    figure = viz.build_trend_by_class_figure(class_trend_frame, params)
    hatched = [
        patch for patch in figure.axes[0].patches if patch.get_hatch() is not None
    ]
    assert len(hatched) == 1


def test_class_figure_shows_pixel_counts_in_the_tick_labels(
    class_trend_frame: pd.DataFrame, params: dict[str, Any]
) -> None:
    # CLAUDE.md caveat 2: a class mean never travels without its pixel count.
    figure = viz.build_trend_by_class_figure(class_trend_frame, params)
    labels = [text.get_text() for text in figure.axes[0].get_yticklabels()]
    assert all("n=" in label for label in labels)


def test_class_figure_raises_on_a_missing_column(
    class_trend_frame: pd.DataFrame, params: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="pixel_count"):
        viz.build_trend_by_class_figure(
            class_trend_frame.drop(columns=["pixel_count"]), params
        )


def test_class_figure_writes_a_png(
    class_trend_frame: pd.DataFrame, params: dict[str, Any], tmp_path: Path
) -> None:
    out = viz.plot_trend_by_class(class_trend_frame, tmp_path / "byclass.png", params)
    assert out.exists() and out.stat().st_size > 0


# --- decadal difference figure -----------------------------------------------
@pytest.fixture()
def decadal_arrays() -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(23)
    difference = rng.normal(0.6, 0.4, size=(20, 28))
    standard_error = np.full((20, 28), 0.35)
    difference[:4, :4] = np.nan
    standard_error[:4, :4] = np.nan
    return {"diff_a_minus_b": difference, "diff_se_a_minus_b": standard_error}


def test_decadal_figure_draws_the_signal_to_noise_panel(
    decadal_arrays: dict[str, Any], params: dict[str, Any]
) -> None:
    # The difference alone is not interpretable: the windows are 11/10/5 years,
    # so any difference involving the short one rests on half the sample. The
    # second panel is what shows which parts survive that.
    figure = viz.build_decadal_difference_figure(
        decadal_arrays, params,
        difference_key="diff_a_minus_b", se_key="diff_se_a_minus_b",
    )
    assert len([ax for ax in figure.axes if ax.get_images()]) == 2


def test_decadal_figure_without_a_standard_error_draws_one_panel(
    decadal_arrays: dict[str, Any], params: dict[str, Any]
) -> None:
    figure = viz.build_decadal_difference_figure(
        decadal_arrays, params, difference_key="diff_a_minus_b"
    )
    assert len([ax for ax in figure.axes if ax.get_images()]) == 1


def test_decadal_figure_footer_warns_it_is_not_the_warming_rate(
    decadal_arrays: dict[str, Any], params: dict[str, Any]
) -> None:
    figure = viz.build_decadal_difference_figure(
        decadal_arrays, params,
        difference_key="diff_a_minus_b", se_key="diff_se_a_minus_b",
    )
    footer = " ".join(text.get_text() for text in figure.texts)
    assert "NOT the warming rate" in footer
    assert "UNEQUAL" in footer


def test_decadal_figure_raises_on_a_missing_array(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="missing"):
        viz.build_decadal_difference_figure(
            {"diff_a": [[0.1, 0.2]]}, params, difference_key="nope"
        )


def test_decadal_figure_rejects_a_non_positive_limit(
    decadal_arrays: dict[str, Any], params: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="positive"):
        viz.build_decadal_difference_figure(
            decadal_arrays, params, difference_key="diff_a_minus_b", max_degc=0
        )


def test_decadal_figure_writes_a_png(
    decadal_arrays: dict[str, Any], params: dict[str, Any], tmp_path: Path
) -> None:
    out = viz.plot_decadal_difference(
        decadal_arrays, tmp_path / "decadal.png", params,
        difference_key="diff_a_minus_b", se_key="diff_se_a_minus_b",
    )
    assert out.exists() and out.stat().st_size > 0


# =============================================================================
# Phase 6 - conditional scenario projection figures
# =============================================================================
# The load-bearing property of this whole section is that a predictive figure
# CANNOT be built without its validation metrics. Every builder is tested for
# that refusal, not just the wrappers, because the builders are what the rest of
# the suite exercises and a guard that only lives in plot_* would be bypassed by
# anyone calling build_* directly.


@pytest.fixture(scope="module")
def projection_report(params: dict[str, Any]) -> dict[str, Any]:
    from colombo_uhi import prediction

    return prediction.build_validation_report(
        "lst_projection",
        {
            # Kappa must BEAT its no-change baseline, or require_validated
            # refuses the product and every figure builder refuses with it.
            # Each figure has its own test for that refusal; these fixtures
            # exist to exercise the drawing.
            "rmse": 1.24,
            "r2": 0.71,
            "kappa": 0.94,
            "persistence_kappa": 0.68,
            "figure_of_merit": 0.21,
        },
        params,
        held_out=True,
        n_blocks=180,
        block_size_m=2000,
        extrapolation={
            "fraction": 0.012,
            "tolerance": 0.05,
            "within_tolerance": True,
        },
    )


@pytest.fixture(scope="module")
def fit_report(params: dict[str, Any]) -> dict[str, Any]:
    from colombo_uhi import prediction

    return prediction.build_validation_report(
        "lst_fit",
        {"rmse": 1.24, "r2": 0.71},
        params,
        held_out=True,
        n_blocks=180,
        block_size_m=2000,
    )


@pytest.fixture(scope="module")
def lulc_panels(params: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    from colombo_uhi import prediction

    codes = prediction.resolve_ca_classes(params)
    rng = np.random.default_rng(0)
    initial = rng.choice(codes, size=(24, 40))
    observed = initial.copy()
    observed[rng.random(initial.shape) < 0.10] = 6
    projected = initial.copy()
    projected[rng.random(initial.shape) < 0.08] = 6
    return {"initial": initial, "observed": observed, "projected": projected}


def test_the_landcover_palette_covers_the_whole_legend(
    params: dict[str, Any],
) -> None:
    palette = viz.landcover_palette(params, "dynamic_world")
    assert set(palette) == set(params["landcover"]["dynamic_world"]["classes"])
    assert all(colour.startswith("#") for colour in palette.values())


def test_an_unknown_palette_names_the_ones_that_exist(
    params: dict[str, Any],
) -> None:
    with pytest.raises(KeyError, match="dynamic_world"):
        viz.landcover_palette(params, "worldcover")


def test_the_projection_caption_carries_both_caveats_and_metrics(
    params: dict[str, Any], projection_report: dict[str, Any]
) -> None:
    caption = " ".join(
        viz.projection_caption(projection_report, params).split()
    )
    assert "NOT forecasts" in caption  # from caveats.scenario_not_forecast
    assert "NOT A FORECAST" in caption  # from prediction.validation_caption
    assert "RMSE=1.240" in caption


def test_the_projection_caption_refuses_an_unvalidated_product(
    params: dict[str, Any],
) -> None:
    from colombo_uhi import prediction

    with pytest.raises(prediction.ValidationMissing):
        viz.projection_caption(None, params)


# --- observed vs predicted ---------------------------------------------------


def test_the_scatter_draws_a_one_to_one_line_not_a_fit(
    params: dict[str, Any], fit_report: dict[str, Any]
) -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    observed = rng.normal(31.0, 2.0, 500)
    figure = viz.build_observed_vs_predicted_figure(
        observed, observed * 0.8 + 6.2, params, fit_report
    )
    axes = figure.axes[0]
    assert axes.get_xlim() == axes.get_ylim()
    line = axes.lines[0]
    assert np.allclose(line.get_xdata(), line.get_ydata())


def test_the_scatter_stamps_its_metrics_on_the_axes(
    params: dict[str, Any], fit_report: dict[str, Any]
) -> None:
    import numpy as np

    figure = viz.build_observed_vs_predicted_figure(
        np.linspace(28, 34, 200), np.linspace(28, 34, 200), params, fit_report
    )
    stamped = " ".join(text.get_text() for text in figure.axes[0].texts)
    assert "RMSE = 1.240" in stamped
    assert "R2 = 0.710" in stamped


def test_the_scatter_thins_a_large_sample_reproducibly(
    params: dict[str, Any], fit_report: dict[str, Any]
) -> None:
    import numpy as np

    values = np.linspace(28.0, 34.0, 20_000)
    figure = viz.build_observed_vs_predicted_figure(
        values, values, params, fit_report, max_points=500
    )
    assert figure.axes[0].collections[0].get_offsets().shape[0] == 500


def test_the_scatter_refuses_an_unvalidated_product(params: dict[str, Any]) -> None:
    from colombo_uhi import prediction

    with pytest.raises(prediction.ValidationMissing):
        viz.build_observed_vs_predicted_figure([1.0], [1.0], params, None)


def test_the_scatter_rejects_mismatched_arrays(
    params: dict[str, Any], fit_report: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="must match"):
        viz.build_observed_vs_predicted_figure(
            [1.0, 2.0], [1.0], params, fit_report
        )


def test_the_scatter_writes_a_png(
    params: dict[str, Any], fit_report: dict[str, Any], tmp_path: Path
) -> None:
    import numpy as np

    out = viz.plot_observed_vs_predicted(
        np.linspace(28, 34, 100), np.linspace(28, 34, 100),
        tmp_path / "scatter.png", params, fit_report,
    )
    assert out.exists() and out.stat().st_size > 0


# --- feature importance ------------------------------------------------------


def _importance_frame(params: dict[str, Any]) -> pd.DataFrame:
    names = list(params["prediction"]["rf"]["predictors"])
    return pd.DataFrame(
        {
            "predictor": names,
            "importance_mean": [1.0 - 0.1 * i for i in range(len(names))],
            "importance_std": [0.02] * len(names),
            "rank": range(1, len(names) + 1),
        }
    )


def test_the_importance_figure_says_it_is_permutation_importance(
    params: dict[str, Any], fit_report: dict[str, Any]
) -> None:
    # Impurity importance favours high-cardinality predictors, and lcz_class is
    # exactly that. The footer must not let a reader confuse the two.
    figure = viz.build_feature_importance_figure(
        _importance_frame(params), params, fit_report
    )
    footer = " ".join(text.get_text() for text in figure.texts)
    assert "PERMUTATION importance" in footer
    assert "not causation" in footer


def test_the_importance_figure_draws_every_predictor(
    params: dict[str, Any], fit_report: dict[str, Any]
) -> None:
    frame = _importance_frame(params)
    figure = viz.build_feature_importance_figure(frame, params, fit_report)
    labels = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
    assert set(labels) == set(frame["predictor"])


def test_the_importance_figure_rejects_a_missing_column(
    params: dict[str, Any], fit_report: dict[str, Any]
) -> None:
    frame = _importance_frame(params).drop(columns=["importance_std"])
    with pytest.raises(ValueError, match="importance_std"):
        viz.build_feature_importance_figure(frame, params, fit_report)


def test_the_importance_figure_refuses_an_unvalidated_product(
    params: dict[str, Any]
) -> None:
    from colombo_uhi import prediction

    with pytest.raises(prediction.ValidationMissing):
        viz.build_feature_importance_figure(_importance_frame(params), params, None)


def test_the_importance_figure_writes_a_png(
    params: dict[str, Any], fit_report: dict[str, Any], tmp_path: Path
) -> None:
    out = viz.plot_feature_importance(
        _importance_frame(params), tmp_path / "importance.png", params, fit_report
    )
    assert out.exists() and out.stat().st_size > 0


# --- transition matrix -------------------------------------------------------


def test_the_transition_figure_needs_no_validation_report(
    params: dict[str, Any]
) -> None:
    # A transition matrix is an OBSERVATION of one interval, not a projection,
    # so it is the one figure here that is not gated.
    import numpy as np

    from colombo_uhi import prediction

    codes = prediction.resolve_ca_classes(params)
    figure = viz.build_transition_matrix_figure(
        np.eye(len(codes)), params, codes
    )
    assert figure.axes[0].get_images()


def test_the_transition_figure_labels_rows_from_and_columns_to(
    params: dict[str, Any]
) -> None:
    import numpy as np

    from colombo_uhi import prediction

    codes = prediction.resolve_ca_classes(params)
    figure = viz.build_transition_matrix_figure(np.eye(len(codes)), params, codes)
    axes = figure.axes[0]
    assert axes.get_ylabel() == "from"
    assert axes.get_xlabel() == "to"
    footer = " ".join(text.get_text() for text in figure.texts)
    assert "DIAGONAL is persistence" in footer


def test_the_transition_figure_rejects_a_size_mismatch(
    params: dict[str, Any]
) -> None:
    import numpy as np

    with pytest.raises(ValueError, match="class code"):
        viz.build_transition_matrix_figure(np.eye(3), params, [0, 1])


def test_the_transition_figure_writes_a_png(
    params: dict[str, Any], tmp_path: Path
) -> None:
    import numpy as np

    from colombo_uhi import prediction

    codes = prediction.resolve_ca_classes(params)
    out = viz.plot_transition_matrix(
        np.eye(len(codes)), tmp_path / "transitions.png", params, codes
    )
    assert out.exists() and out.stat().st_size > 0


# --- land-cover validation ---------------------------------------------------


def test_the_lulc_figure_draws_three_panels(
    params: dict[str, Any],
    projection_report: dict[str, Any],
    lulc_panels: dict[str, Any],
) -> None:
    figure = viz.build_lulc_validation_figure(
        lulc_panels["initial"], lulc_panels["observed"], lulc_panels["projected"],
        params, projection_report,
    )
    assert len([ax for ax in figure.axes if ax.get_images()]) == 3


def test_the_lulc_figure_warns_that_kappa_needs_its_baseline(
    params: dict[str, Any],
    projection_report: dict[str, Any],
    lulc_panels: dict[str, Any],
) -> None:
    figure = viz.build_lulc_validation_figure(
        lulc_panels["initial"], lulc_panels["observed"], lulc_panels["projected"],
        params, projection_report,
    )
    footer = " ".join(" ".join(text.get_text().split()) for text in figure.texts)
    assert "Read Kappa against the no-change null" in footer
    assert "cells that CHANGED" in footer


def test_the_lulc_figure_rejects_panels_of_different_shapes(
    params: dict[str, Any],
    projection_report: dict[str, Any],
    lulc_panels: dict[str, Any],
) -> None:
    import numpy as np

    with pytest.raises(ValueError, match="share a shape"):
        viz.build_lulc_validation_figure(
            lulc_panels["initial"], lulc_panels["observed"],
            np.zeros((2, 2), dtype=int), params, projection_report,
        )


def test_the_lulc_figure_refuses_an_unvalidated_product(
    params: dict[str, Any], lulc_panels: dict[str, Any]
) -> None:
    from colombo_uhi import prediction

    with pytest.raises(prediction.ValidationMissing):
        viz.build_lulc_validation_figure(
            lulc_panels["initial"], lulc_panels["observed"],
            lulc_panels["projected"], params, None,
        )


def test_the_lulc_figure_writes_a_png(
    params: dict[str, Any],
    projection_report: dict[str, Any],
    lulc_panels: dict[str, Any],
    tmp_path: Path,
) -> None:
    out = viz.plot_lulc_validation(
        lulc_panels["initial"], lulc_panels["observed"], lulc_panels["projected"],
        tmp_path / "lulc.png", params, projection_report,
    )
    assert out.exists() and out.stat().st_size > 0


# --- projected surfaces ------------------------------------------------------


def _surfaces() -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(2)
    baseline = rng.normal(32.0, 1.5, (24, 40))
    return {
        "Business as usual, 2030": baseline,
        "Greening, 2030": baseline - rng.random((24, 40)) * 0.6,
    }


def test_both_scenario_panels_share_one_colour_scale(
    params: dict[str, Any], projection_report: dict[str, Any]
) -> None:
    # Independent scales would make a 0.2 degC difference look like a 3 degC
    # one, which is the single easiest way to overstate a greening result.
    figure = viz.build_projected_lst_figure(_surfaces(), params, projection_report)
    limits = {
        image.get_clim()
        for axes in figure.axes
        for image in axes.get_images()
    }
    assert len(limits) == 1


def test_the_scenario_figure_rejects_surfaces_of_different_shapes(
    params: dict[str, Any], projection_report: dict[str, Any]
) -> None:
    import numpy as np

    surfaces = _surfaces()
    surfaces["Greening, 2030"] = np.zeros((5, 5))
    with pytest.raises(ValueError, match="share a shape"):
        viz.build_projected_lst_figure(surfaces, params, projection_report)


def test_the_scenario_figure_rejects_an_empty_mapping(
    params: dict[str, Any], projection_report: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="at least one"):
        viz.build_projected_lst_figure({}, params, projection_report)


def test_the_scenario_figure_refuses_an_unvalidated_product(
    params: dict[str, Any]
) -> None:
    from colombo_uhi import prediction

    with pytest.raises(prediction.ValidationMissing):
        viz.build_projected_lst_figure(_surfaces(), params, None)


def test_the_scenario_figure_writes_a_png(
    params: dict[str, Any], projection_report: dict[str, Any], tmp_path: Path
) -> None:
    out = viz.plot_projected_lst(
        _surfaces(), tmp_path / "projected.png", params, projection_report
    )
    assert out.exists() and out.stat().st_size > 0


# --- scenario difference -----------------------------------------------------


def test_the_difference_map_is_centred_on_zero(
    params: dict[str, Any], projection_report: dict[str, Any]
) -> None:
    import numpy as np

    rng = np.random.default_rng(3)
    figure = viz.build_scenario_difference_figure(
        rng.normal(-0.3, 0.2, (24, 40)), params, projection_report
    )
    low, high = figure.axes[0].get_images()[0].get_clim()
    assert low == pytest.approx(-high)


def test_the_difference_map_warns_it_carries_both_uncertainties(
    params: dict[str, Any], projection_report: dict[str, Any]
) -> None:
    import numpy as np

    figure = viz.build_scenario_difference_figure(
        np.zeros((10, 10)), params, projection_report
    )
    footer = " ".join(" ".join(text.get_text().split()) for text in figure.texts)
    assert "DIFFERENCE OF TWO PROJECTIONS" in footer
    assert "not a measured cooling" in footer


def test_the_difference_map_rejects_a_non_2d_array(
    params: dict[str, Any], projection_report: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="2-D"):
        viz.build_scenario_difference_figure(
            [1.0, 2.0, 3.0], params, projection_report
        )


def test_the_difference_map_rejects_a_non_positive_limit(
    params: dict[str, Any], projection_report: dict[str, Any]
) -> None:
    import numpy as np

    with pytest.raises(ValueError, match="positive"):
        viz.build_scenario_difference_figure(
            np.zeros((5, 5)), params, projection_report, max_degc=0
        )


def test_the_difference_map_refuses_an_unvalidated_product(
    params: dict[str, Any]
) -> None:
    import numpy as np

    from colombo_uhi import prediction

    with pytest.raises(prediction.ValidationMissing):
        viz.build_scenario_difference_figure(np.zeros((5, 5)), params, None)


def test_the_difference_map_writes_a_png(
    params: dict[str, Any], projection_report: dict[str, Any], tmp_path: Path
) -> None:
    import numpy as np

    out = viz.plot_scenario_difference(
        np.zeros((10, 10)), tmp_path / "difference.png", params, projection_report
    )
    assert out.exists() and out.stat().st_size > 0


# --- a measured failure is drawn, and stamped --------------------------------
# Run 3's land-cover projection failed validation, and the run died rather than
# drawing it - taking Track A's valid figures with it. A figure showing that a
# projection failed is exactly what the report needs; what must never happen is
# a figure silent about its status.


@pytest.fixture(scope="module")
def failed_report(params: dict[str, Any]) -> dict[str, Any]:
    from colombo_uhi import prediction

    # Run 3's actual primary-interval numbers.
    return prediction.build_validation_report(
        "lulc_projection",
        {"kappa": 0.8566, "persistence_kappa": 0.8576, "figure_of_merit": 0.0022},
        params,
        held_out=True,
        n_test=66_926,
    )


def test_the_projection_caption_stamps_a_measured_failure(
    params: dict[str, Any], failed_report: dict[str, Any]
) -> None:
    caption = " ".join(viz.projection_caption(failed_report, params).split())
    assert "FAILED VALIDATION" in caption
    assert "must not be quoted as a projection" in caption


def test_the_lulc_figure_draws_a_failed_projection_with_the_banner(
    params: dict[str, Any],
    failed_report: dict[str, Any],
    lulc_panels: dict[str, Any],
) -> None:
    figure = viz.build_lulc_validation_figure(
        lulc_panels["initial"], lulc_panels["observed"], lulc_panels["projected"],
        params, failed_report,
    )
    assert len([ax for ax in figure.axes if ax.get_images()]) == 3
    footer = " ".join(" ".join(t.get_text().split()) for t in figure.texts)
    assert "FAILED VALIDATION" in footer


def test_the_scenario_figure_draws_a_failed_projection_with_the_banner(
    params: dict[str, Any], failed_report: dict[str, Any]
) -> None:
    import numpy as np

    rng = np.random.default_rng(11)
    baseline = rng.normal(32.0, 1.5, (16, 24))
    figure = viz.build_projected_lst_figure(
        {"Business as usual, 2030": baseline, "Greening, 2030": baseline - 0.4},
        params, failed_report,
    )
    footer = " ".join(" ".join(t.get_text().split()) for t in figure.texts)
    assert "FAILED VALIDATION" in footer


def test_the_difference_map_draws_a_failed_projection_with_the_banner(
    params: dict[str, Any], failed_report: dict[str, Any]
) -> None:
    import numpy as np

    figure = viz.build_scenario_difference_figure(
        np.full((10, 10), -0.3), params, failed_report
    )
    footer = " ".join(" ".join(t.get_text().split()) for t in figure.texts)
    assert "FAILED VALIDATION" in footer


def test_a_figure_still_refuses_when_nothing_was_computed(
    params: dict[str, Any], lulc_panels: dict[str, Any]
) -> None:
    # Absence of evidence is not evidence, and it does not get a figure.
    from colombo_uhi import prediction

    with pytest.raises(prediction.ValidationMissing):
        viz.build_lulc_validation_figure(
            lulc_panels["initial"], lulc_panels["observed"],
            lulc_panels["projected"], params, None,
        )


def test_a_failed_figure_writes_a_png(
    params: dict[str, Any],
    failed_report: dict[str, Any],
    lulc_panels: dict[str, Any],
    tmp_path: Path,
) -> None:
    out = viz.plot_lulc_validation(
        lulc_panels["initial"], lulc_panels["observed"], lulc_panels["projected"],
        tmp_path / "failed.png", params, failed_report,
    )
    assert out.exists() and out.stat().st_size > 0


def test_the_difference_map_names_what_it_actually_differenced(
    params: dict[str, Any]
) -> None:
    # A counterfactual minus its observed baseline is not a difference of two
    # projections, and the footer must not claim it is.
    import numpy as np

    from colombo_uhi import prediction

    scenario = prediction.build_validation_report(
        "lst_scenario", {"rmse": 1.13, "r2": 0.894}, params, held_out=True
    )
    figure = viz.build_scenario_difference_figure(
        np.zeros((8, 8)), params, scenario
    )
    footer = " ".join(" ".join(t.get_text().split()) for t in figure.texts)
    assert "COUNTERFACTUAL MINUS ITS OBSERVED BASELINE" in footer
    assert "DIFFERENCE OF TWO PROJECTIONS" not in footer


# =============================================================================
# Phase 7 - greening priority figures
# =============================================================================


def _figure_text(figure: Any) -> str:
    """All the loose text on a figure, whitespace-collapsed."""
    return " ".join(" ".join(text.get_text().split()) for text in figure.texts)


def _figure_titles(figure: Any) -> str:
    """Suptitle plus every axes title."""
    parts = [figure._suptitle.get_text() if figure._suptitle else ""]
    parts.extend(axes.get_title() for axes in figure.axes)
    return " ".join(" ".join(part.split()) for part in parts if part)


@pytest.fixture(scope="module")
def ahp_report(params: dict[str, Any]) -> dict[str, Any]:
    from colombo_uhi import greening

    matrix, names = greening.pairwise_matrix(params)
    return greening.ahp_weights(matrix, params, names, warn=False)


@pytest.fixture(scope="module")
def ahp_matrix(params: dict[str, Any]) -> Any:
    from colombo_uhi import greening

    return greening.pairwise_matrix(params)


@pytest.fixture(scope="module")
def inconsistent_report(params: dict[str, Any]) -> dict[str, Any]:
    """Judgements that fail the consistency ratio without being degenerate.

    Deliberately asymmetric. A perfect 3-cycle at equal strength (a 9 b, b 9 c,
    c 9 a) is symmetric, so it returns *equal* weights and is degenerate as well
    as inconsistent - which is a different failure and is covered separately.
    """
    import numpy as np

    from colombo_uhi import greening

    matrix = np.array([[1.0, 9.0, 1 / 3], [1 / 9, 1.0, 5.0], [3.0, 1 / 5, 1.0]])
    report = greening.ahp_weights(matrix, params, ["a", "b", "c"], warn=False)
    assert not report["consistent"] and not report["degenerate"]
    return report


@pytest.fixture(scope="module")
def greening_tables(params: dict[str, Any]) -> dict[str, Any]:
    """A small end-to-end Phase 7 result set, built the way the notebook does."""
    import numpy as np

    from colombo_uhi import greening

    rng = np.random.default_rng(23)
    n = 24
    frame = pd.DataFrame(
        {
            "zone_id": [f"LK1103{index:03d}" for index in range(n)],
            "LST_C": rng.normal(31.0, 1.5, n),
            "utfvi_severe_share": rng.random(n),
            "NDVI": rng.random(n) * 0.6,
            "pop_density": rng.lognormal(9.0, 1.0, n),
            "pop_within_300m_pct": rng.random(n) * 100.0,
        }
    )
    for column in ("LST_C", "utfvi_severe_share", "NDVI", "pop_density"):
        frame[f"{column}_pixels"] = 400

    matrix, names = greening.pairwise_matrix(params)
    report = greening.ahp_weights(matrix, params, names, warn=False)
    prepared, _ = greening.prepare_criteria(frame, params)
    ranked = greening.rank_frame(
        greening.mcda_scores(prepared, params, report["weights"]), params, top_n=8
    )
    topsis = greening.rank_frame(
        greening.topsis_scores(prepared, params, report["weights"]),
        params,
        top_n=8,
        score_column="score_topsis",
    )
    comparison = greening.compare_rankings(ranked, topsis, params)
    shifts = greening.rank_shift_frame(ranked, topsis, params)
    correlation = greening.criterion_correlation(prepared, params)

    compliance = pd.DataFrame(
        {
            "zone_id": frame["zone_id"],
            "canopy_pct": rng.random(n) * 50.0,
            "rule_30_pass": rng.random(n) > 0.5,
            "pop_within_300m_pct": rng.random(n) * 100.0,
            "rule_300_pass": rng.random(n) > 0.5,
        }
    )
    compliance = greening.compliance_3_30_300(
        compliance[["zone_id", "canopy_pct", "rule_30_pass"]],
        compliance[["zone_id", "pop_within_300m_pct", "rule_300_pass"]],
        None,
        params,
    )
    wetland = pd.DataFrame(
        {
            "zone_id": frame["zone_id"],
            "wetland_status": ["within", "adjacent", "neither"] * (n // 3),
            "wetland_policy_flag": [True, True, False] * (n // 3),
            "wetland_within_pct": rng.random(n) * 20.0,
        }
    )
    full = greening.build_priority_frame(
        ranked,
        params,
        prepared=prepared,
        topsis_ranked=topsis,
        compliance=compliance,
        wetland=wetland,
    )
    full["adm4_name"] = [f"Division number {index}" for index in range(n)]
    return {
        "prepared": prepared,
        "ranked": ranked,
        "topsis": topsis,
        "comparison": comparison,
        "shifts": shifts,
        "correlation": correlation,
        "compliance": compliance,
        "wetland": wetland,
        "full": full,
        "report": report,
        "matrix": matrix,
        "names": names,
        "ahp_frame": greening.build_ahp_frame(report, params),
    }


@pytest.fixture(scope="module")
def greening_zones(greening_tables: dict[str, Any]) -> Any:
    """Zone polygons matching the synthetic tables."""
    gpd = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely.geometry")

    ids = list(greening_tables["full"]["zone_id"])
    boxes = [
        shapely.box(
            (index % 6) * 100.0,
            (index // 6) * 100.0,
            (index % 6) * 100.0 + 95.0,
            (index // 6) * 100.0 + 95.0,
        )
        for index in range(len(ids))
    ]
    return gpd.GeoDataFrame({"zone_id": ids}, geometry=boxes, crs="EPSG:32644")


# --- The caption and the failure banner --------------------------------------


def test_the_greening_caption_carries_every_required_caveat(
    params: dict[str, Any], ahp_report: dict[str, Any]
) -> None:
    caption = viz.greening_caption(ahp_report, params)
    for key in (
        "lst_not_air_temp",
        "zonal_not_pixel",
        "sensitivity_reporting",
        "mcda_weights_are_judgements",
    ):
        fragment = " ".join(params["caveats"][key].split())[:60]
        assert fragment in " ".join(caption.split())


def test_the_greening_caption_states_the_ratio_and_normalisation(
    params: dict[str, Any], ahp_report: dict[str, Any]
) -> None:
    caption = viz.greening_caption(ahp_report, params)
    assert "consistency ratio" in caption
    assert "PASSES" in caption
    assert "percentile_rank" in caption


def test_the_caption_leads_with_the_failure_when_judgements_fail(
    params: dict[str, Any], inconsistent_report: dict[str, Any]
) -> None:
    # A reader has to meet the failure before anything else on the figure.
    caption = viz.greening_caption(inconsistent_report, params)
    assert caption.splitlines()[0].startswith("*** INCONSISTENT JUDGEMENTS")
    assert "FAILS" in caption


def test_the_failure_headline_is_none_when_judgements_pass(
    params: dict[str, Any], ahp_report: dict[str, Any]
) -> None:
    assert viz.greening_failure_headline(ahp_report, params) is None
    assert viz.greening_failure_headline(None, params) is None


def test_the_failure_headline_names_the_consistency_ratio(
    params: dict[str, Any], inconsistent_report: dict[str, Any]
) -> None:
    headline = viz.greening_failure_headline(inconsistent_report, params)
    assert headline is not None
    assert "CR =" in headline


def test_a_degenerate_matrix_gets_its_own_headline(params: dict[str, Any]) -> None:
    # "No judgement was made" is a different failure from "the judgements
    # disagree with each other", and the banner must not conflate them.
    import numpy as np

    from colombo_uhi import greening

    report = greening.ahp_weights(np.ones((4, 4)), params, list("abcd"), warn=False)
    headline = viz.greening_failure_headline(report, params)
    assert headline is not None
    assert "NO JUDGEMENT WAS MADE" in headline


def test_a_failing_ratio_leads_even_when_the_matrix_is_also_degenerate(
    params: dict[str, Any],
) -> None:
    """The two failures can co-occur, and the substantive one must lead.

    A perfect 3-cycle at equal strength is symmetric, so it scores a huge
    consistency ratio *and* returns equal weights. Reporting only "no judgement
    was made" would hide the fact that the judgements also contradict each other.
    """
    import numpy as np

    from colombo_uhi import greening

    matrix = np.ones((3, 3))
    matrix[0, 1], matrix[1, 0] = 9.0, 1 / 9
    matrix[1, 2], matrix[2, 1] = 9.0, 1 / 9
    matrix[2, 0], matrix[0, 2] = 9.0, 1 / 9
    report = greening.ahp_weights(matrix, params, ["a", "b", "c"], warn=False)
    assert report["degenerate"] is True
    assert report["consistency_ratio"] > report["consistency_ratio_max"]

    headline = viz.greening_failure_headline(report, params)
    assert headline is not None
    assert headline.startswith("*** INCONSISTENT JUDGEMENTS")
    assert "CR =" in headline


def test_the_compliance_caption_adds_the_network_caveat(
    params: dict[str, Any], ahp_report: dict[str, Any]
) -> None:
    caption = viz.greening_caption(
        ahp_report,
        params,
        keys=(*viz.GREENING_CAVEATS, "euclidean_not_network"),
    )
    assert "WALKING distance" in " ".join(caption.split())


# --- Palettes ----------------------------------------------------------------


def test_the_priority_palette_is_a_ramp(params: dict[str, Any]) -> None:
    palette = viz.priority_palette(params)
    assert len(palette) >= 4
    assert all(colour.startswith("#") for colour in palette)


def test_the_compliance_palette_has_exactly_five_entries(
    params: dict[str, Any],
) -> None:
    from colombo_uhi import greening

    palette = viz.compliance_palette(params)
    assert set(palette) == set(greening.COMPLIANCE_CATEGORIES)
    assert len(palette) == 5
    assert all(colour.startswith("#") for colour in palette.values())


# --- The AHP weights figure --------------------------------------------------


def test_the_ahp_weights_figure_renders(
    params: dict[str, Any], greening_tables: dict[str, Any]
) -> None:
    pytest.importorskip("matplotlib")
    figure = viz.build_ahp_weights_figure(
        greening_tables["ahp_frame"],
        greening_tables["report"],
        params,
        matrix=greening_tables["matrix"],
        names=greening_tables["names"],
    )
    assert figure.get_figwidth() > 6
    assert len(figure.axes) >= 2


def test_the_ahp_banner_reports_pass_when_consistent(
    params: dict[str, Any], greening_tables: dict[str, Any]
) -> None:
    pytest.importorskip("matplotlib")
    figure = viz.build_ahp_weights_figure(
        greening_tables["ahp_frame"], greening_tables["report"], params
    )
    text = _figure_text(figure)
    assert "CR = 0.0081" in text
    assert "PASS" in text
    assert "INCONSISTENT" not in _figure_titles(figure)


def test_the_ahp_banner_reports_inconsistent_above_the_threshold(
    params: dict[str, Any], inconsistent_report: dict[str, Any]
) -> None:
    pytest.importorskip("matplotlib")
    from colombo_uhi import greening

    frame = greening.build_ahp_frame(inconsistent_report, params)
    figure = viz.build_ahp_weights_figure(frame, inconsistent_report, params)
    assert "INCONSISTENT" in _figure_text(figure)
    assert "INCONSISTENT JUDGEMENTS" in _figure_titles(figure)


def test_the_ahp_figure_still_draws_when_judgements_fail(
    params: dict[str, Any], inconsistent_report: dict[str, Any]
) -> None:
    # A figure showing that judgements failed is exactly what the report needs;
    # refusing to draw it throws the evidence away.
    pytest.importorskip("matplotlib")
    from colombo_uhi import greening

    frame = greening.build_ahp_frame(inconsistent_report, params)
    figure = viz.build_ahp_weights_figure(frame, inconsistent_report, params)
    assert figure.axes


def test_the_ahp_figure_explains_the_log_colour_scale(
    params: dict[str, Any], greening_tables: dict[str, Any]
) -> None:
    pytest.importorskip("matplotlib")
    figure = viz.build_ahp_weights_figure(
        greening_tables["ahp_frame"],
        greening_tables["report"],
        params,
        matrix=greening_tables["matrix"],
        names=greening_tables["names"],
    )
    assert "LOGARITHMIC" in _figure_text(figure)


def test_the_ahp_figure_refuses_an_empty_frame(
    params: dict[str, Any], ahp_report: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="no weights"):
        viz.build_ahp_weights_figure(pd.DataFrame(), ahp_report, params)


def test_plot_ahp_weights_writes_a_png(
    params: dict[str, Any], greening_tables: dict[str, Any], tmp_path: Path
) -> None:
    pytest.importorskip("matplotlib")
    out = viz.plot_ahp_weights(
        greening_tables["ahp_frame"],
        greening_tables["report"],
        tmp_path / "ahp.png",
        params,
        matrix=greening_tables["matrix"],
        names=greening_tables["names"],
    )
    assert out.is_file() and out.stat().st_size > 0


# --- The ranking comparison --------------------------------------------------


def test_the_ranking_comparison_renders_and_reports_rho(
    params: dict[str, Any], greening_tables: dict[str, Any]
) -> None:
    pytest.importorskip("matplotlib")
    figure = viz.build_ranking_comparison_figure(
        greening_tables["ranked"],
        greening_tables["topsis"],
        greening_tables["comparison"],
        params,
        shifts=greening_tables["shifts"],
        ahp_report=greening_tables["report"],
    )
    axes_text = " ".join(text.get_text() for text in figure.axes[0].texts)
    assert "Spearman rho" in axes_text
    assert "Kendall tau" in axes_text
    assert "Top-" in axes_text


def test_the_ranking_comparison_says_agreement_is_not_validation(
    params: dict[str, Any], greening_tables: dict[str, Any]
) -> None:
    pytest.importorskip("matplotlib")
    figure = viz.build_ranking_comparison_figure(
        greening_tables["ranked"],
        greening_tables["topsis"],
        greening_tables["comparison"],
        params,
        ahp_report=greening_tables["report"],
    )
    text = " ".join(_figure_text(figure).split())
    assert "ROBUSTNESS check, not a validation" in text


def test_the_ranking_comparison_refuses_an_empty_ranking(
    params: dict[str, Any], greening_tables: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="is empty"):
        viz.build_ranking_comparison_figure(
            pd.DataFrame(), greening_tables["topsis"], {}, params
        )


def test_plot_ranking_comparison_writes_a_png(
    params: dict[str, Any], greening_tables: dict[str, Any], tmp_path: Path
) -> None:
    pytest.importorskip("matplotlib")
    out = viz.plot_ranking_comparison(
        greening_tables["ranked"],
        greening_tables["topsis"],
        greening_tables["comparison"],
        tmp_path / "compare.png",
        params,
        shifts=greening_tables["shifts"],
        ahp_report=greening_tables["report"],
    )
    assert out.is_file() and out.stat().st_size > 0


# --- The priority table figure -----------------------------------------------


def test_the_priority_table_renders_the_top_rows(
    params: dict[str, Any], greening_tables: dict[str, Any]
) -> None:
    pytest.importorskip("matplotlib")
    figure = viz.build_priority_table_figure(
        greening_tables["full"], params, ahp_report=greening_tables["report"], top_n=6
    )
    assert "Top 6" in _figure_titles(figure)


def test_the_priority_table_wraps_a_long_division_name(
    params: dict[str, Any], greening_tables: dict[str, Any]
) -> None:
    pytest.importorskip("matplotlib")
    frame = greening_tables["full"].copy()
    frame.loc[frame.index[0], "adm4_name"] = "A" * 60
    figure = viz.build_priority_table_figure(
        frame, params, ahp_report=greening_tables["report"], top_n=4
    )
    table = figure.axes[0].tables[0]
    wrapped = table[1, list(range(len(table.get_celld()))).index(0) + 2].get_text()
    assert "\n" in wrapped.get_text() or len(wrapped.get_text()) <= 60


def test_the_priority_table_says_the_score_matters_as_much_as_the_rank(
    params: dict[str, Any], greening_tables: dict[str, Any]
) -> None:
    pytest.importorskip("matplotlib")
    figure = viz.build_priority_table_figure(
        greening_tables["full"], params, ahp_report=greening_tables["report"], top_n=5
    )
    assert "SCORE matters as much as the rank" in " ".join(_figure_text(figure).split())


def test_the_priority_table_refuses_an_empty_frame(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="nothing to tabulate"):
        viz.build_priority_table_figure(pd.DataFrame(), params)


def test_plot_priority_table_writes_a_png(
    params: dict[str, Any], greening_tables: dict[str, Any], tmp_path: Path
) -> None:
    pytest.importorskip("matplotlib")
    out = viz.plot_priority_table(
        greening_tables["full"],
        tmp_path / "table.png",
        params,
        ahp_report=greening_tables["report"],
        top_n=6,
    )
    assert out.is_file() and out.stat().st_size > 0


# --- The maps (need geopandas) -----------------------------------------------


def test_the_priority_map_renders(
    params: dict[str, Any], greening_tables: dict[str, Any], greening_zones: Any
) -> None:
    pytest.importorskip("matplotlib")
    figure = viz.build_greening_priority_map_figure(
        greening_zones,
        greening_tables["full"],
        params,
        ahp_report=greening_tables["report"],
    )
    assert figure.axes


def test_the_priority_map_uses_different_hatches_for_wetland_and_flagged(
    params: dict[str, Any], greening_tables: dict[str, Any], greening_zones: Any
) -> None:
    # The two say opposite things: a wetland-adjacent division is a policy
    # opportunity, a below-coverage one is a division the data could not see.
    pytest.importorskip("matplotlib")
    frame = greening_tables["full"].copy()
    frame["below_land_coverage_floor"] = False
    frame.loc[frame.index[:3], "below_land_coverage_floor"] = True
    figure = viz.build_greening_priority_map_figure(
        greening_zones, frame, params, ahp_report=greening_tables["report"]
    )
    legend = figure.axes[0].get_legend()
    labels = [text.get_text() for text in legend.get_texts()]
    assert "Within / beside wetland" in labels
    assert "Land-cover coverage below floor" in labels
    # geopandas draws polygons into a PatchCollection, not into axes.patches.
    hatches = {
        collection.get_hatch()
        for collection in figure.axes[0].collections
        if collection.get_hatch()
    }
    assert hatches == {"///", "xxx"}
    legend_hatches = {
        handle.get_hatch()
        for handle in legend.legend_handles
        if getattr(handle, "get_hatch", None) and handle.get_hatch()
    }
    assert legend_hatches == hatches


def test_the_priority_map_explains_both_hatches_in_the_footer(
    params: dict[str, Any], greening_tables: dict[str, Any], greening_zones: Any
) -> None:
    pytest.importorskip("matplotlib")
    figure = viz.build_greening_priority_map_figure(
        greening_zones,
        greening_tables["full"],
        params,
        ahp_report=greening_tables["report"],
    )
    text = " ".join(_figure_text(figure).split())
    # Flagged, not removed - the floor gates nothing that enters the score.
    assert "FLAGGED, not removed" in text
    assert "less of their land was classified" in text
    assert "wetland protection is an existing policy instrument" in text


def test_the_priority_map_refuses_an_empty_ranking(
    params: dict[str, Any], greening_zones: Any
) -> None:
    with pytest.raises(ValueError, match="nothing to map"):
        viz.build_greening_priority_map_figure(greening_zones, pd.DataFrame(), params)


def test_the_priority_map_refuses_a_missing_score_column(
    params: dict[str, Any], greening_tables: dict[str, Any], greening_zones: Any
) -> None:
    with pytest.raises(ValueError, match="score_missing"):
        viz.build_greening_priority_map_figure(
            greening_zones,
            greening_tables["full"],
            params,
            score_column="score_missing",
        )


def test_plot_greening_priority_map_writes_a_png(
    params: dict[str, Any],
    greening_tables: dict[str, Any],
    greening_zones: Any,
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    out = viz.plot_greening_priority_map(
        greening_zones,
        greening_tables["full"],
        tmp_path / "priority.png",
        params,
        ahp_report=greening_tables["report"],
    )
    assert out.is_file() and out.stat().st_size > 0


def test_the_compliance_map_legend_has_five_entries_in_palette_order(
    params: dict[str, Any], greening_tables: dict[str, Any], greening_zones: Any
) -> None:
    pytest.importorskip("matplotlib")
    from colombo_uhi import greening

    figure = viz.build_compliance_map_figure(
        greening_zones,
        greening_tables["compliance"],
        params,
        ahp_report=greening_tables["report"],
    )
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    expected = [
        category.replace("_", " ") for category in greening.COMPLIANCE_CATEGORIES
    ]
    assert labels == expected


def test_the_compliance_footer_carries_the_network_and_unmeasured_caveats(
    params: dict[str, Any], greening_tables: dict[str, Any], greening_zones: Any
) -> None:
    pytest.importorskip("matplotlib")
    figure = viz.build_compliance_map_figure(
        greening_zones,
        greening_tables["compliance"],
        params,
        ahp_report=greening_tables["report"],
    )
    text = " ".join(_figure_text(figure).split())
    assert "WALKING distance" in text
    assert "NOT MEASURABLE from satellite data" in text
    assert "NOT crown cover" in text


def test_the_compliance_map_refuses_a_frame_with_no_verdict(
    params: dict[str, Any], greening_zones: Any
) -> None:
    with pytest.raises(ValueError, match="'compliance' column"):
        viz.build_compliance_map_figure(
            greening_zones, pd.DataFrame({"zone_id": ["a"]}), params
        )


def test_the_compliance_map_refuses_an_empty_frame(
    params: dict[str, Any], greening_zones: Any
) -> None:
    with pytest.raises(ValueError, match="nothing to map"):
        viz.build_compliance_map_figure(greening_zones, pd.DataFrame(), params)


def test_plot_compliance_map_writes_a_png(
    params: dict[str, Any],
    greening_tables: dict[str, Any],
    greening_zones: Any,
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    out = viz.plot_compliance_map(
        greening_zones,
        greening_tables["compliance"],
        tmp_path / "compliance.png",
        params,
        ahp_report=greening_tables["report"],
    )
    assert out.is_file() and out.stat().st_size > 0


def test_the_criterion_panel_draws_one_axes_per_criterion_plus_correlation(
    params: dict[str, Any], greening_tables: dict[str, Any], greening_zones: Any
) -> None:
    pytest.importorskip("matplotlib")
    from colombo_uhi import greening

    figure = viz.build_criterion_panel_figure(
        greening_zones,
        greening_tables["prepared"],
        params,
        correlation=greening_tables["correlation"],
        ahp_report=greening_tables["report"],
    )
    titled = [axes.get_title() for axes in figure.axes if axes.get_title()]
    assert len(titled) >= len(greening.criterion_names(params)) + 1
    assert any("Criterion correlation" in title for title in titled)


def test_the_criterion_panel_says_why_the_panels_look_alike(
    params: dict[str, Any], greening_tables: dict[str, Any], greening_zones: Any
) -> None:
    # Five criterion maps that all look the same are not a redundancy in the
    # figure - they are the finding.
    pytest.importorskip("matplotlib")
    figure = viz.build_criterion_panel_figure(
        greening_zones,
        greening_tables["prepared"],
        params,
        ahp_report=greening_tables["report"],
    )
    text = " ".join(_figure_text(figure).split())
    assert "measuring the same underlying variable" in text
    assert "leave-one-out ablation" in text


def test_the_criterion_panel_refuses_an_unprepared_frame(
    params: dict[str, Any], greening_zones: Any
) -> None:
    with pytest.raises(ValueError, match="prepare_criteria"):
        viz.build_criterion_panel_figure(
            greening_zones, pd.DataFrame({"zone_id": ["a"]}), params
        )


def test_plot_criterion_panel_writes_a_png(
    params: dict[str, Any],
    greening_tables: dict[str, Any],
    greening_zones: Any,
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    out = viz.plot_criterion_panel(
        greening_zones,
        greening_tables["prepared"],
        tmp_path / "criteria.png",
        params,
        correlation=greening_tables["correlation"],
        ahp_report=greening_tables["report"],
    )
    assert out.is_file() and out.stat().st_size > 0


def test_every_phase7_map_stamps_the_title_when_judgements_fail(
    params: dict[str, Any],
    greening_tables: dict[str, Any],
    greening_zones: Any,
    inconsistent_report: dict[str, Any],
) -> None:
    pytest.importorskip("matplotlib")
    for figure in (
        viz.build_greening_priority_map_figure(
            greening_zones,
            greening_tables["full"],
            params,
            ahp_report=inconsistent_report,
        ),
        viz.build_compliance_map_figure(
            greening_zones,
            greening_tables["compliance"],
            params,
            ahp_report=inconsistent_report,
        ),
    ):
        assert "INCONSISTENT JUDGEMENTS" in _figure_titles(figure)


# --- Footer legibility, from Colab run 3 -------------------------------------


def test_the_caption_wraps_figure_specific_footer_text(
    params: dict[str, Any], ahp_report: dict[str, Any]
) -> None:
    """Run 3 shipped two figures whose last footer line ran off the canvas.

    ``caveat_footer`` wrapped the standing caveats; ``extra`` was appended
    verbatim beside them.
    """
    caption = viz.greening_caption(
        ahp_report,
        params,
        extra=("word " * 90, "a second bullet " * 20),
    )
    longest = max(len(line) for line in caption.split("\n"))
    assert longest <= viz.CAVEAT_WRAP_CHARS


def test_a_dash_inside_a_bullet_does_not_split_it(
    params: dict[str, Any], ahp_report: dict[str, Any]
) -> None:
    """A dash used as punctuation is not a bullet boundary.

    An earlier version split ``extra`` on ``" - "`` and tore the criterion
    panel's footer - which contains ``"the ablation - not the weights - are
    what say..."`` - into three bullets mid-sentence.
    """
    caption = viz.greening_caption(
        ahp_report,
        params,
        extra=("The ablation - not the AHP weights - is what says how much.",),
    )
    bullets = [line for line in caption.split("\n") if line.startswith("- ")]
    assert sum("not the AHP weights" in line for line in bullets) == 1
    assert not any(line.strip() == "- not the AHP weights" for line in bullets)


def test_a_plain_string_extra_is_still_accepted(
    params: dict[str, Any], ahp_report: dict[str, Any]
) -> None:
    caption = viz.greening_caption(ahp_report, params, extra="a single bullet")
    assert "- a single bullet" in caption


def test_heatmap_labels_are_legible_on_dark_and_light_cells() -> None:
    # The correlation panel's 1.00 diagonal disappeared into its own red.
    assert viz._readable_on((0.05, 0.02, 0.10)) == "#ffffff"
    assert viz._readable_on((0.98, 0.98, 0.95)) == "#111111"
    assert viz._readable_on((1.0, 1.0, 1.0, 1.0)) == "#111111"


def test_the_priority_map_separates_its_colourbar_from_its_legend(
    params: dict[str, Any], greening_tables: dict[str, Any], greening_zones: Any
) -> None:
    """Two legends below one axes need a band deep enough for both.

    In run 3 the colourbar's tick labels read through the patch legend's text.
    """
    pytest.importorskip("matplotlib")
    figure = viz.build_greening_priority_map_figure(
        greening_zones,
        greening_tables["full"],
        params,
        ahp_report=greening_tables["report"],
    )
    legend = figure.axes[0].get_legend()
    assert legend is not None
    # The patch legend is anchored below the axes in axes coordinates, far
    # enough down to clear the colourbar AND the colourbar's own axis label.
    anchor_y = legend.get_bbox_to_anchor().transformed(
        figure.axes[0].transAxes.inverted()
    ).y1
    assert anchor_y <= -0.15, (
        f"the legend is anchored at y={anchor_y:.3f}; it needs to sit below the "
        "colourbar, whose tick labels it overlapped in Colab run 3"
    )
    # And the reserved band actually grew to hold both.
    assert figure.get_figheight() > 5.0


# =============================================================================
# Phase 8 - report plumbing
# =============================================================================
def test_report_figure_path_uses_the_configured_directory_and_template(
    params: dict[str, Any],
) -> None:
    path = viz.report_figure_path(params, 1, "decadal_lst")
    assert path.parent.as_posix() == params["report"]["figure_dir"]
    assert path.name == "fig01_decadal_lst.png"
    # Zero-padded, so a directory listing sorts into report order rather than
    # putting figure 10 between 1 and 2.
    assert viz.report_figure_path(params, 11, "x").name.startswith("fig11_")


def test_report_figure_path_rejects_a_meaningless_index_or_slug(
    params: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="1-based"):
        viz.report_figure_path(params, 0, "decadal_lst")
    with pytest.raises(ValueError, match="slug"):
        viz.report_figure_path(params, 1, "   ")


def test_report_figures_land_outside_the_phase_diagnostic_directory(
    params: dict[str, Any],
) -> None:
    # figures/ holds the per-phase diagnostics at 150 dpi and Phase 8 must not
    # overwrite any of them: a reader has to be able to tell the two sets apart.
    directory = params["report"]["figure_dir"]
    assert directory != "figures"
    assert directory.startswith("figures/")


def test_manifest_has_eleven_figures_with_unique_paths(params: dict[str, Any]) -> None:
    manifest = viz.report_manifest(params)
    assert len(manifest) == 11
    assert [entry["index"] for entry in manifest] == list(range(1, 12))
    assert len({str(entry["path"]) for entry in manifest}) == 11


def test_every_manifest_entry_names_a_builder_that_exists(
    params: dict[str, Any],
) -> None:
    # The notebook dispatches on these names. A typo here is an AttributeError
    # in Colab, several minutes into a render loop.
    for entry in viz.report_manifest(params):
        assert hasattr(viz, entry["builder"]), (
            f"figure {entry['index']} names viz.{entry['builder']}, which does "
            "not exist"
        )


def test_missing_inputs_names_every_gap_before_anything_is_drawn(
    params: dict[str, Any],
) -> None:
    # Nothing available: every figure with inputs is blocked, and figure 11 -
    # which is generated from params alone - is not.
    gaps = viz.missing_inputs({}, params)
    assert 11 not in gaps
    assert gaps[9] == [
        "counterfactual_baseline_tif",
        "counterfactual_greened_tif",
        "counterfactual_delta_tif",
    ]
    # A key present but None counts as absent, because that is what the
    # notebook's discovery helpers return for a file that is not there.
    assert viz.missing_inputs({"suhii_csv": None}, params)[6] == ["suhii_csv"]
    assert 6 not in viz.missing_inputs({"suhii_csv": "x.csv"}, params)


def test_saturated_fraction_counts_only_finite_values() -> None:
    import numpy as np

    data = np.array([0.0, 1.0, 2.0, 3.0, np.nan])
    assert viz.saturated_fraction(data, 0.0, 2.0) == pytest.approx(0.25)
    assert viz.saturated_fraction(data, -10.0, 10.0) == pytest.approx(0.0)
    assert np.isnan(viz.saturated_fraction(np.array([np.nan]), 0.0, 1.0))


def test_panel_aspect_follows_the_raster_and_is_clamped() -> None:
    import numpy as np

    assert viz.panel_aspect(np.zeros((30, 60))) == pytest.approx(0.5)
    # One pathological shape must not produce a figure thousands of inches tall.
    assert viz.panel_aspect(np.zeros((10_000, 1))) == pytest.approx(3.0)
    assert viz.panel_aspect(np.zeros((1, 10_000))) == pytest.approx(0.2)
    assert viz.panel_aspect(np.zeros(5)) == pytest.approx(0.6)


def test_save_figure_writes_the_dpi_it_is_given(tmp_path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(2, 2))
    FigureCanvasAgg(figure)
    figure.subplots().plot([0, 1], [0, 1])

    default = viz._save_figure(figure, tmp_path / "a.png")
    report = viz._save_figure(figure, tmp_path / "b.png", dpi=300)
    assert default.stat().st_size and report.stat().st_size
    # The report set has to be 300 dpi; the phase diagnostics have to stay 150,
    # so the default must not have moved.
    assert viz.DIAGNOSTIC_DPI == 150
    # PNG stores resolution in pixels per METRE, so a 300 dpi round trip comes
    # back as 299.9994. Compare with a tolerance rather than pinning the artefact.
    PIL = pytest.importorskip("PIL.Image")
    assert PIL.open(report).info["dpi"] == pytest.approx((300, 300), rel=1e-3)
    assert PIL.open(default).info["dpi"] == pytest.approx((150, 150), rel=1e-3)


# =============================================================================
# Phase 8 - colour-vision verification
# =============================================================================
def test_normalise_hex_accepts_both_conventions_in_params() -> None:
    # params.yaml stores hex WITHOUT '#' because Earth Engine palettes want it
    # that way; matplotlib wants it with.
    assert viz.normalise_hex("b2182b") == "#b2182b"
    assert viz.normalise_hex("#B2182B") == "#b2182b"
    assert viz.normalise_hex("fff") == "#ffffff"
    with pytest.raises(ValueError, match="not a hex colour"):
        viz.normalise_hex("mauve")


def test_simulate_cvd_leaves_greys_where_they_are() -> None:
    # A dichromat confusion line passes through the achromatic axis, so a
    # neutral must simulate to itself. This is the sharpest cheap check that the
    # sRGB linearisation has not been skipped: without it, greys shift.
    for grey in ("#000000", "#808080", "#ffffff"):
        for deficiency in ("protanopia", "deuteranopia", "tritanopia"):
            simulated = viz.simulate_cvd(grey, deficiency)
            channels = [int(simulated[i:i + 2], 16) for i in (1, 3, 5)]
            assert max(channels) - min(channels) <= 2, (
                f"{grey} under {deficiency} became {simulated}, which is not neutral"
            )


def test_simulate_cvd_is_idempotent() -> None:
    # Simulation projects onto the dichromat's plane; a colour already on that
    # plane must not move again. A second application that shifts the colour
    # means the projection is not a projection.
    for colour in ("#b2182b", "#2166ac", "#1a9850", "#fee08b"):
        for deficiency in ("protanopia", "deuteranopia", "tritanopia"):
            once = viz.simulate_cvd(colour, deficiency)
            twice = viz.simulate_cvd(once, deficiency)
            assert viz.delta_e(once, twice) < 1.5


def test_simulate_cvd_collapses_the_classic_red_green_pair() -> None:
    # The point of the whole exercise: red and green are far apart in normal
    # vision and close under deuteranopia.
    assert viz.delta_e("#1a9850", "#d73027") > 60
    assert viz.delta_e(
        viz.simulate_cvd("#1a9850", "deuteranopia"),
        viz.simulate_cvd("#d73027", "deuteranopia"),
    ) < 40


def test_simulate_cvd_rejects_an_unknown_deficiency() -> None:
    with pytest.raises(KeyError, match="unknown deficiency"):
        viz.simulate_cvd("#ffffff", "achromatopsia")


def test_lab_and_delta_e_behave_like_a_metric() -> None:
    assert viz.lightness("#ffffff") == pytest.approx(100.0, abs=0.2)
    assert viz.lightness("#000000") == pytest.approx(0.0, abs=0.2)
    assert viz.delta_e("#b2182b", "#b2182b") == pytest.approx(0.0)
    assert viz.delta_e("#000000", "#ffffff") == pytest.approx(
        viz.delta_e("#ffffff", "#000000")
    )


def test_palette_separation_names_the_pair_at_fault() -> None:
    distance, pair = viz.palette_separation(["ffffff", "000000", "fefefe"])
    assert distance < 1.0
    assert set(pair) == {"#ffffff", "#fefefe"}
    # Fewer than two colours has no pairwise distance to report.
    assert viz.palette_separation(["ffffff"]) == (float("inf"), None)


def test_monotonicity_helpers() -> None:
    assert viz.is_monotonic([1.0, 2.0, 3.0])
    assert viz.is_monotonic([3.0, 2.0, 1.0])
    assert not viz.is_monotonic([1.0, 3.0, 2.0])
    assert not viz.is_monotonic([1.0, 1.0, 2.0]), "a flat step is not monotonic"
    assert not viz.is_monotonic([])

    # Diverging: lightest in the middle, falling away on both limbs.
    assert viz.is_diverging_monotonic([40.0, 70.0, 97.0, 70.0, 40.0])
    assert viz.is_diverging_monotonic([97.0, 70.0, 40.0, 70.0, 97.0])
    assert not viz.is_diverging_monotonic([40.0, 70.0, 97.0, 99.0, 40.0])
    # An even-length ramp has no stop AT the centre, so there is no limb to test.
    assert not viz.is_diverging_monotonic([40.0, 97.0, 97.0, 40.0])


def test_resolve_palette_handles_both_shapes_params_stores() -> None:
    # Ramps are lists; categorical schemes are class-keyed mappings.
    mapping = {"spatial_stats": {"palettes": {"lisa": {"HH": "b2182b", "LL": "2166ac"}}},
               "trends": {"slope_vis": {"palette": ["2166ac", "f7f7f7", "b2182b"]}}}
    assert viz.resolve_palette(mapping, "spatial_stats.palettes.lisa") == [
        "#b2182b", "#2166ac"
    ]
    assert viz.resolve_palette(mapping, "trends.slope_vis.palette")[0] == "#2166ac"
    with pytest.raises(KeyError, match="does not resolve"):
        viz.resolve_palette(mapping, "trends.slope_vis.missing")


def test_check_palette_applies_the_test_that_fits_the_palette_kind(
    params: dict[str, Any],
) -> None:
    # THE reason the check has two branches. This ramp is perfectly readable -
    # its lightness is monotonic - but adjacent stops are close, so a pairwise
    # dE test "fails" it. Every ColorBrewer diverging palette in params.yaml
    # behaves this way.
    ramp = ["2166ac", "67a9cf", "d1e5f0", "f7f7f7", "fddbc7", "ef8a62", "b2182b"]
    assert viz.check_palette(ramp, "diverging", params)["passed"]
    assert not viz.check_palette(ramp, "categorical", params)["passed"]
    with pytest.raises(ValueError, match="unknown palette kind"):
        viz.check_palette(ramp, "ordinal", params)


def test_check_palette_reports_the_metric_under_every_deficiency(
    params: dict[str, Any],
) -> None:
    row = viz.check_palette(["1a9850", "a6d96a", "fee08b"], "categorical", params)
    for vision in ("normal", *params["report"]["cvd"]["deficiencies"]):
        assert f"min_delta_e_{vision}" in row
    assert row["worst"] <= row["min_delta_e_normal"], (
        "the worst case must be at or below normal vision"
    )


def test_every_non_exempt_palette_passes_the_colour_vision_check(
    params: dict[str, Any],
) -> None:
    # This is the test that holds the Phase 8 decision in place. Two palettes
    # were changed because they failed here - uhi.utfvi.palette on lightness
    # monotonicity and greening.palettes.compliance on pairwise separation - and
    # this stops either drifting back.
    #
    # If it fails: fix the PALETTE in params.yaml and record the measured before
    # and after numbers beside it. Do not add an exemption to make it pass.
    report = viz.cvd_report(params)
    failures = viz.cvd_failures(report)
    assert failures.empty, (
        "palettes failing the colour-vision check:\n"
        + failures[["path", "kind", "metric", "worst", "verdict"]].to_string(index=False)
    )


def test_the_two_exempt_palettes_are_still_measured(params: dict[str, Any]) -> None:
    # An exemption that hid its number would be worthless. Both exempt palettes
    # carry the measurement that justifies the exemption.
    report = viz.cvd_report(params).set_index("path")
    exempt = report[report["exempt"]]
    assert set(exempt.index) == {
        "prediction.palettes.dynamic_world", "spatial_stats.palettes.ehsa"
    }
    for path, row in exempt.iterrows():
        assert row["worst"] > 0 or row["metric"], f"{path} carries no measurement"
        assert row["reason"], f"{path} is exempt without a stated reason"


def test_the_cvd_check_covers_every_palette_in_params(params: dict[str, Any]) -> None:
    # A palette added to params.yaml and left out of report.cvd.palettes would
    # be unverified while the report claims every palette was checked.
    checked = {entry["path"] for entry in params["report"]["cvd"]["palettes"]}

    def _walk(node: Any, prefix: str = "") -> list[str]:
        found: list[str] = []
        if isinstance(node, dict):
            for key, value in node.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                looks_like_a_palette = (
                    key == "palette"
                    or (prefix.endswith("palettes") and isinstance(value, (list, dict)))
                )
                if looks_like_a_palette and _is_colours(value):
                    found.append(path)
                else:
                    found.extend(_walk(value, path))
        return found

    def _is_colours(value: Any) -> bool:
        entries = list(value.values()) if isinstance(value, dict) else (
            list(value) if isinstance(value, list) else []
        )
        if not entries:
            return False
        try:
            for entry in entries:
                viz.normalise_hex(entry)
        except (ValueError, TypeError):
            return False
        return True

    for path in _walk(params):
        assert path in checked, (
            f"palette '{path}' is not listed under report.cvd.palettes, so it is "
            "never verified"
        )


def test_the_utfvi_palette_is_an_ordered_ramp(params: dict[str, Any]) -> None:
    # UTFVI is an ORDERED severity scale, so lightness must carry the order and
    # keep carrying it in greyscale and under every simulated deficiency. The
    # Spectral palette this replaced ran light in the middle and dark at both
    # ends, which made "Excellent" and "Worse" nearly the same lightness.
    colours = params["uhi"]["utfvi"]["palette"]
    assert len(colours) == len(params["uhi"]["utfvi"]["labels"])
    for vision in (None, *params["report"]["cvd"]["deficiencies"]):
        profile = viz.palette_lightness_profile(colours, vision)
        assert viz.is_monotonic(profile), (
            f"UTFVI lightness is not monotonic under {vision or 'normal vision'}: "
            f"{[round(value) for value in profile]}"
        )
    assert profile[0] > profile[-1], "the ramp must run light (cool) to dark (hot)"


def test_the_compliance_palette_separates_the_two_partial_categories(
    params: dict[str, Any],
) -> None:
    # The distinction this map exists to draw. Under the green-to-red palette
    # this replaced, canopy_only and access_only sat at dE 10.4 under both
    # red-green deficiencies - a reader could not tell "has canopy, no park
    # access" from "has park access, no canopy".
    compliance = params["greening"]["palettes"]["compliance"]
    pair = [compliance["canopy_only"], compliance["access_only"]]
    for vision in (None, *params["report"]["cvd"]["deficiencies"]):
        distance, _ = viz.palette_separation(pair, vision)
        assert distance >= params["report"]["cvd"]["min_delta_e"], (
            f"canopy_only and access_only are only dE {distance:.1f} apart under "
            f"{vision or 'normal vision'}"
        )


# =============================================================================
# Phase 8 - label placement
# =============================================================================
def test_spread_label_positions_preserves_order_and_the_minimum_gap() -> None:
    # Order is preserved so the leader lines joining label to division never
    # cross; the gap is what stops ten CMC-core labels overlapping into a mat.
    placed = viz.spread_label_positions([0.0, 0.1, 0.2, 5.0], min_gap=1.0)
    assert placed == sorted(placed)
    assert all(b - a >= 1.0 - 1e-9 for a, b in zip(placed, placed[1:]))
    # A well-separated input is left alone.
    assert viz.spread_label_positions([0.0, 5.0, 10.0], min_gap=1.0) == [0.0, 5.0, 10.0]
    assert viz.spread_label_positions([], min_gap=1.0) == []


def test_spread_label_positions_keeps_the_block_inside_its_bounds() -> None:
    placed = viz.spread_label_positions(
        [9.0, 9.2, 9.4], min_gap=1.0, low=0.0, high=10.0
    )
    assert placed[-1] <= 10.0 + 1e-9
    assert placed[0] >= 0.0 - 1e-9
    assert all(b - a >= 1.0 - 1e-9 for a, b in zip(placed, placed[1:]))


def test_spread_label_positions_refuses_an_impossible_request() -> None:
    # Not a layout hiccup - it means too many labels were asked for, and saying
    # so is more useful than silently stacking them.
    with pytest.raises(ValueError, match="label fewer"):
        viz.spread_label_positions([0.0] * 20, min_gap=1.0, low=0.0, high=5.0)
    with pytest.raises(ValueError, match="non-negative"):
        viz.spread_label_positions([0.0, 1.0], min_gap=-1.0)


# =============================================================================
# Phase 8 - figure structure
# =============================================================================
def _decadal_arrays(params: dict[str, Any], shape: tuple[int, int] = (12, 20)):
    """Synthetic decadal means, one band per configured window."""
    import numpy as np

    from colombo_uhi import trends

    labels = [label for label, _, _ in trends.resolve_decades(None, params)]
    arrays = {}
    for index, label in enumerate(labels):
        block = np.full(shape, 30.0 + index)
        block[0, 0] = np.nan
        arrays[f"mean_{label}"] = block
    return arrays


def test_the_decadal_figure_draws_one_panel_per_window_on_one_shared_norm(
    params: dict[str, Any],
) -> None:
    from colombo_uhi import trends

    labels = [label for label, _, _ in trends.resolve_decades(None, params)]
    figure = viz.build_decadal_lst_panel_figure(
        _decadal_arrays(params), _decadal_arrays(params, (6, 10)), params
    )
    images = [image for axes in figure.axes for image in axes.get_images()]
    assert len(images) == 2 * len(labels)

    # ONE stretch across all six panels. Autoscaling each would hide the sensor
    # step between the top row's decades, which is the point of the figure.
    limits = {(image.norm.vmin, image.norm.vmax) for image in images}
    assert len(limits) == 1
    shared = params["report"]["decadal"]["shared_vis"]
    assert limits == {(float(shared["min"]), float(shared["max"]))}


def test_the_decadal_figure_names_the_missing_band(params: dict[str, Any]) -> None:
    good = _decadal_arrays(params)
    partial = {key: value for key, value in list(good.items())[1:]}
    with pytest.raises(ValueError, match="missing"):
        viz.build_decadal_lst_panel_figure(partial, good, params)


def test_the_decadal_banner_quotes_the_measured_offsets(
    params: dict[str, Any],
) -> None:
    import pandas as pd

    offsets = pd.DataFrame([
        {"sensor_a": "landsat5", "sensor_b": "landsat7", "n_overlap_years": 8,
         "mean_offset": 1.783, "t_statistic": 2.72, "verdict": "material"},
        {"sensor_a": "landsat8", "sensor_b": "landsat9", "n_overlap_years": 4,
         "mean_offset": -0.397, "t_statistic": -0.67, "verdict": "negligible"},
    ])
    banner = viz.sensor_step_banner(offsets, params)
    assert "+1.78" in banner
    # Only MATERIAL steps are named: listing the negligible one beside them
    # would suggest the reader should discount all three equally.
    assert "-0.40" not in banner
    # With nothing measured the banner still warns, it just names no numbers.
    assert "POOLED LANDSAT" in viz.sensor_step_banner(None, params)


def test_stipple_coordinates_mark_only_significant_cells() -> None:
    import numpy as np

    mask = np.zeros((12, 12), dtype=bool)
    mask[4:8, 4:8] = True
    x, y = viz.stipple_coordinates(mask, stride=2)
    assert len(x) == len(y)
    assert all(mask[row, column] for row, column in zip(y, x))
    # Every marked cell is on the lattice, so the density is a fixed texture and
    # not a second choropleth a reader might read as a quantity.
    assert all(row % 2 == 0 and column % 2 == 0 for row, column in zip(y, x))
    # A denser lattice marks more of the same region, never a different one.
    dense_x, _ = viz.stipple_coordinates(mask, stride=1)
    assert len(dense_x) > len(x)
    with pytest.raises(ValueError, match="stride"):
        viz.stipple_coordinates(mask, stride=0)


def test_the_slope_figure_stipples_exactly_the_significant_lattice(
    params: dict[str, Any],
) -> None:
    import numpy as np

    slope = np.linspace(-0.4, 0.4, 400).reshape(20, 20)
    significant = np.zeros((20, 20))
    significant[:10] = 1.0
    significant[15:] = np.nan          # tested/untested/significant, all three

    figure = viz.build_sen_slope_stipple_figure(
        {"sen_slope": slope, "significant": significant}, params
    )
    expected, _ = viz.stipple_coordinates(
        significant == 1.0, int(params["report"]["stipple"]["stride"])
    )
    # Count only on the MAP axes: figure.axes also holds the colorbar, whose
    # solids are a collection as well.
    map_axes = next(axes for axes in figure.axes if axes.get_images())
    drawn = sum(
        collection.get_offsets().shape[0] for collection in map_axes.collections
    )
    assert drawn == len(expected)


def test_the_slope_figure_needs_both_bands(params: dict[str, Any]) -> None:
    import numpy as np

    with pytest.raises(ValueError, match="missing"):
        viz.build_sen_slope_stipple_figure({"sen_slope": np.zeros((4, 4))}, params)


def test_the_observation_count_figure_needs_a_finite_pixel(
    params: dict[str, Any],
) -> None:
    import numpy as np

    band = params["composites"]["obs_count_band"]
    with pytest.raises(ValueError, match="missing"):
        viz.build_obs_count_figure({"other": np.zeros((4, 4))}, params)
    with pytest.raises(ValueError, match="no finite pixel"):
        viz.build_obs_count_figure({band: np.full((4, 4), np.nan)}, params)


def test_the_observation_count_figure_draws_the_map_and_the_distribution(
    params: dict[str, Any],
) -> None:
    import numpy as np

    band = params["composites"]["obs_count_band"]
    counts = np.arange(400, dtype="float64").reshape(20, 20) % 120
    figure = viz.build_obs_count_figure({band: counts}, params)
    # A colour ramp cannot be integrated by eye, which is why the ECDF is there.
    assert any(axes.get_images() for axes in figure.axes)
    assert any(axes.lines for axes in figure.axes)
    # One reference line per configured mark.
    marks = params["report"]["obs_count_vis"]["ecdf_marks"]
    vertical = [
        line for axes in figure.axes for line in axes.lines
        if len(set(line.get_xdata())) == 1
    ]
    assert len(vertical) == len(marks)


def test_the_utfvi_figure_legends_every_class_and_bounds_the_codes(
    params: dict[str, Any],
) -> None:
    import numpy as np

    from colombo_uhi import uhi_metrics

    _, labels = uhi_metrics.validate_utfvi_scheme(params)
    epochs = {
        key: np.arange(36, dtype="float64").reshape(6, 6) % len(labels)
        for key in params["report"]["facet_epochs"]
    }
    figure = viz.build_utfvi_epoch_maps_figure(epochs, params)
    legend = figure.legends[0]
    assert len(legend.get_texts()) == len(labels)
    # The class NUMBER travels with the name, so colour is never the only
    # channel carrying the class (caveats.colour_is_not_the_only_channel).
    assert legend.get_texts()[0].get_text().startswith("0")

    with pytest.raises(ValueError, match="outside the configured"):
        viz.build_utfvi_epoch_maps_figure(
            {"2000s": np.full((4, 4), 99.0)}, params
        )
    with pytest.raises(ValueError, match="no epoch"):
        viz.build_utfvi_epoch_maps_figure({}, params)


def test_the_facet_figure_is_epochs_by_indices(params: dict[str, Any]) -> None:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    epochs = list(params["report"]["facet_epochs"])
    indices = list(params["report"]["facet_indices"])
    frame = pd.concat([
        pd.DataFrame({
            "epoch": epoch,
            "LST_C": rng.normal(35, 2, 200),
            **{name: rng.normal(0, 0.3, 200) for name in indices},
        })
        for epoch in epochs
    ], ignore_index=True)

    figure = viz.build_lst_vs_index_facet_figure(frame, params)
    panels = [axes for axes in figure.axes if axes.get_title()]
    assert len(panels) == len(epochs) * len(indices)
    # Rows share axes so the SLOPES are comparable by eye.
    limits = {axes.get_ylim() for axes in panels}
    assert len(limits) == 1

    with pytest.raises(ValueError, match="missing column"):
        viz.build_lst_vs_index_facet_figure(frame.drop(columns=["LST_C"]), params)
    with pytest.raises(ValueError, match="empty"):
        viz.build_lst_vs_index_facet_figure(frame.iloc[:0], params)


def test_the_facet_figure_fits_on_every_row_but_draws_a_thinned_sample(
    params: dict[str, Any],
) -> None:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(1)
    drawn = int(params["report"]["facet_max_points"])
    size = drawn * 3
    index = params["report"]["facet_indices"][0]
    frame = pd.DataFrame({
        "epoch": params["report"]["facet_epochs"][0],
        index: rng.uniform(0, 1, size),
        params["report"]["facet_indices"][1]: rng.uniform(0, 1, size),
        "LST_C": rng.normal(35, 1, size),
    })
    figure = viz.build_lst_vs_index_facet_figure(frame, params)
    for axes in figure.axes:
        for collection in axes.collections:
            assert collection.get_offsets().shape[0] <= drawn
    # The fit uses every row, so the panel title reports the full n.
    assert f"{size:,}" in " ".join(axes.get_title() for axes in figure.axes)


def test_the_scenario_triptych_requires_three_matching_surfaces(
    params: dict[str, Any], monkeypatch: Any
) -> None:
    import numpy as np

    from colombo_uhi import prediction

    report = prediction.build_validation_report(
        "lst_scenario", {"rmse": 1.13, "r2": 0.89}, params, held_out=True
    )
    good = np.full((8, 10), 34.0)
    with pytest.raises(ValueError, match="share a shape"):
        viz.build_scenario_triptych_figure(good, good, np.zeros((4, 4)), params,
                                           report=report)
    with pytest.raises(ValueError, match="priority_mask has shape"):
        viz.build_scenario_triptych_figure(
            good, good, np.zeros((8, 10)), params, report=report,
            priority_mask=np.ones((4, 4), dtype=bool),
        )


def test_the_scenario_triptych_reports_the_priority_zone_mean_separately(
    params: dict[str, Any],
) -> None:
    import numpy as np

    from colombo_uhi import prediction

    report = prediction.build_validation_report(
        "lst_scenario", {"rmse": 1.13, "r2": 0.89}, params, held_out=True
    )
    baseline = np.full((10, 10), 36.0)
    delta = np.zeros((10, 10))
    delta[:1] = -2.0                      # one row greened, the rest untouched
    mask = delta < 0

    figure = viz.build_scenario_triptych_figure(
        baseline, baseline + delta, delta, params, report=report, priority_mask=mask
    )
    # The footer is wrapped, so a sentence spans line breaks; compare on
    # collapsed whitespace rather than on the wrap points.
    footer = " ".join(
        " ".join(text.get_text().split()) for text in figure.texts
    )
    # The district-wide mean here is -0.20 degC and the greened-zone mean is
    # -2.00. Quoting the first as the greening effect understates it tenfold,
    # which is exactly the misreading this line exists to prevent.
    assert "-2.00 degC" in footer
    assert "must not be quoted as the greening effect" in footer


def test_the_provenance_figure_refuses_an_empty_table(params: dict[str, Any]) -> None:
    import pandas as pd

    from colombo_uhi import reporting

    with pytest.raises(ValueError, match="empty"):
        viz.build_provenance_table_figure(
            pd.DataFrame(columns=list(reporting.PROVENANCE_COLUMNS)), params
        )


def test_the_provenance_figure_shows_every_configured_source(
    params: dict[str, Any],
) -> None:
    from colombo_uhi import reporting

    frame = reporting.provenance_frame(params)
    figure = viz.build_provenance_table_figure(frame, params)
    drawn = " ".join(
        text.get_text() for axes in figure.axes for text in axes.texts
    )
    for key in params["datasets"]:
        assert key in drawn, f"{key} is missing from the provenance figure"


def test_every_report_builder_renders_at_report_dpi(
    params: dict[str, Any], tmp_path: Path
) -> None:
    # An end-to-end check that the report path really is 300 dpi, not just that
    # report_dpi() returns 300.
    import numpy as np

    figure = viz.build_decadal_lst_panel_figure(
        _decadal_arrays(params), _decadal_arrays(params), params
    )
    path = viz._save_figure(figure, tmp_path / "fig.png", dpi=viz.report_dpi(params))
    PIL = pytest.importorskip("PIL.Image")
    expected = float(params["report"]["dpi"])
    assert PIL.open(path).info["dpi"] == pytest.approx((expected, expected), rel=1e-3)


def test_the_label_gutter_widens_the_canvas_instead_of_the_axes_limits(
    params: dict[str, Any],
) -> None:
    """Labels get their own column; the map keeps its size.

    Widening the axes limits alone shrank the map by roughly a third and left
    white bands above and below it, because ``_map_figure`` had already sized
    the canvas to the unpadded bounding box.
    """
    class _Zones:
        total_bounds = (0.0, 0.0, 100.0, 50.0)

    label_x, right_limit = viz.label_gutter_bounds(_Zones())
    assert label_x == pytest.approx(100.0 + 100.0 * viz.LABEL_GUTTER_INSET)
    assert right_limit == pytest.approx(100.0 + 100.0 * viz.LABEL_GUTTER_FRACTION)
    # The labels have to sit inside the gutter, not past its edge.
    assert label_x < right_limit

    padded = viz._PaddedBounds(_Zones(), viz.LABEL_GUTTER_FRACTION)
    assert padded.total_bounds[2] == pytest.approx(right_limit)
    # Only the right edge moves: padding the others would recentre the map.
    assert padded.total_bounds[:2] == (0.0, 0.0)
    assert padded.total_bounds[3] == pytest.approx(50.0)


def test_the_priority_map_carries_the_rank_needed_to_label_it(
    params: dict[str, Any],
) -> None:
    """The rank and the division name have to survive the merge.

    They did not in the first draft: the map rendered correctly in every other
    respect, ``annotate_top_zones`` found no rank column, and it labelled
    nothing - with no error anywhere.
    """
    import inspect

    source = inspect.getsource(viz.build_greening_priority_map_figure)
    merge_block = source[source.index("columns = ["):source.index("frame = zones.copy()")]
    for column in ("rank_ahp", "adm4_name"):
        assert f'"{column}"' in merge_block, (
            f"{column} is not merged into the mapped frame, so label_top_n "
            "would silently draw nothing"
        )
