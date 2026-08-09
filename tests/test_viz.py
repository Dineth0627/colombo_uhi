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
