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
