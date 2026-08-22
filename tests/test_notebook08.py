"""Pin notebook 08's WIRING — the class of bug that only shows up in Colab.

Phase 8 lost two Colab runs to notebook-only mistakes in which every ``src/``
function involved was individually correct:

* run 1 passed :func:`colombo_uhi.uhi_metrics.source_collection` into the driver
  sampler. That collection is built ``include_sr=False`` on purpose, the
  spectral indices need exactly those bands, and all 26 years failed;
* run 3 paired :func:`colombo_uhi.trends.decadal_means` with
  :func:`colombo_uhi.trends.decadal_band_order`. The band order names difference
  bands only ``decadal_product`` creates, ``image.select`` is lazy, and Earth
  Engine failed six minutes into the batch task. Figure 1 was never drawable.

Phase 6 lost three runs the same way and the project answered with
``tests/test_notebook07.py``. Phase 8 had no equivalent; this is it.

Cells are located by CONTENT, not by id. Notebook 08 is generated, so its cell
ids are assigned by Colab on execution and do not survive a regeneration - a
test keyed on them would quietly stop testing anything.
"""

from __future__ import annotations

import ast
import builtins
import json
from typing import Any

import pytest

from colombo_uhi import repo_root

NOTEBOOK = "08_figures_for_report.ipynb"


def _cells() -> list[dict[str, Any]]:
    notebook = json.loads(
        (repo_root() / "notebooks" / NOTEBOOK).read_text(encoding="utf-8")
    )
    return list(notebook["cells"])


def _code_sources() -> list[str]:
    return ["".join(cell["source"]) for cell in _cells() if cell["cell_type"] == "code"]


def _neutralise_magics(source: str) -> str:
    """Replace ``!`` and ``%`` lines with ``pass``, preserving indentation.

    Indentation matters: the clone cell's ``!git clone`` sits inside an ``if``,
    and a bare ``pass`` at column zero turns a valid cell into a syntax error
    that this file would then report as a notebook defect.
    """
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            lines.append(" " * (len(line) - len(stripped)) + "pass")
        else:
            lines.append(line)
    return "\n".join(lines)


def _find(marker: str) -> str:
    """The one code cell containing ``marker``."""
    matches = [source for source in _code_sources() if marker in source]
    assert len(matches) == 1, (
        f"expected exactly one code cell containing {marker!r}, found "
        f"{len(matches)}; this test can no longer find what it checks"
    )
    return matches[0]


def _calls(source: str) -> set[str]:
    """Attribute-call names made in a cell, e.g. ``decadal_product``."""
    return {
        node.func.attr
        for node in ast.walk(ast.parse(_neutralise_magics(source)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def _bound(tree: ast.AST) -> set[str]:
    """Every name a cell binds."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


# --- structure ---------------------------------------------------------------
def test_every_code_cell_parses() -> None:
    for index, source in enumerate(_code_sources()):
        try:
            ast.parse(_neutralise_magics(source))
        except SyntaxError as error:  # pragma: no cover - the failure IS the report
            pytest.fail(f"code cell {index} does not parse: {error}")


def test_every_code_cell_carries_the_colab_marker() -> None:
    """``CLAUDE.md`` requires it on every cell needing an authenticated session."""
    for index, source in enumerate(_code_sources()):
        assert "# COLAB: RUN THIS CELL" in source, (
            f"code cell {index} has no COLAB marker"
        )


# --- run 3: the band order must describe the image it is applied to ----------
def test_the_decadal_export_pairs_the_band_order_with_its_own_builder() -> None:
    """The run-3 failure, pinned.

    ``trends.decadal_band_order`` names ``diff_``/``diff_se_``/``diff_z_``/
    ``n_years_min_`` bands that only ``trends.decadal_product`` creates.
    ``decadal_means`` returns the per-window statistics alone. ``image.select``
    is lazy, so pairing the two is accepted at submit time and fails inside the
    batch task minutes later - which is how figure 1 went undrawn for three
    runs. ``decadal_product``'s docstring says it exists precisely "so that the
    band order the reader assumes and the band order the writer produces come
    from the same place".
    """
    cell = _find('product="lst_decadal"')
    called = _calls(cell)
    assert "decadal_product" in called, (
        "the decadal export does not use trends.decadal_product, which is what "
        "makes the exported bands match trends.decadal_band_order"
    )
    assert "decadal_means" not in called, (
        "the decadal export calls trends.decadal_means while passing "
        "trends.decadal_band_order. The band order names difference bands "
        "decadal_means does not produce; Earth Engine accepts the export and "
        "fails six minutes into the batch task."
    )
    assert "decadal_band_order" in called


def test_figure_one_reads_both_rows_with_the_same_band_order() -> None:
    """One band order for both rows is only sound if both files carry it."""
    # The Step 0 staleness guard also names this function, as a string.
    cell = _find("viz.build_decadal_lst_panel_figure(")
    assert cell.count("band_order=_order") == 2, (
        "figure 1 must read the Landsat and MODIS rasters with the same band "
        "order; if it needs two, the two exports have diverged"
    )


# --- run 1: the driver sample must build its own collection ------------------
def test_the_driver_sample_uses_driver_series_and_builds_its_own_collection() -> None:
    """The run-1 failure, pinned.

    ``uhi_metrics.source_collection`` is built ``include_sr=False`` on purpose -
    SUHII needs LST only. The spectral indices need exactly those reflectance
    bands, so handing that collection to the sampler failed all 26 years
    identically. ``driver_series`` builds its own, correctly, and reuses it.
    """
    cell = _find("DRIVER_SAMPLES")
    called = _calls(cell)
    assert "driver_series" in called, (
        "Step 3 does not call uhi_metrics.driver_series, the Phase 3 path that "
        "builds a collection carrying surface reflectance"
    )
    assert "source_collection" not in called, (
        "Step 3 builds a collection with uhi_metrics.source_collection, which "
        "drops the reflectance bands the spectral indices need"
    )
    tree = ast.parse(_neutralise_magics(cell))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "driver_series"
        ):
            assert not any(kw.arg == "collection" for kw in node.keywords), (
                "driver_series was passed a collection; let it build its own"
            )


# --- exports ------------------------------------------------------------------
def test_every_image_export_states_its_band_order() -> None:
    """Earth Engine does not reliably write band names into a GeoTIFF.

    The reader falls back to position, so an export without an explicit band
    order makes every downstream band index a guess.
    """
    for source in _code_sources():
        tree = ast.parse(_neutralise_magics(source))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "image_to_drive"
            ):
                assert any(kw.arg == "band_order" for kw in node.keywords), (
                    "an image_to_drive call passes no band_order"
                )


def test_part_one_submits_exactly_three_exports() -> None:
    """Phase 8 adds three tasks and reuses everything else.

    A fourth would mean something is being re-exported that Phase 4 already
    produced - which is both slow and a way for two products to disagree.
    """
    total = sum(source.count("image_to_drive(") for source in _code_sources())
    assert total == 3, f"notebook 08 submits {total} image exports, expected 3"


# --- error handling -----------------------------------------------------------
def test_no_handler_discards_the_exception_message() -> None:
    """Run 1's handler printed ``type(_error).__name__`` and threw the rest away.

    Earth Engine names the missing band in its message. Printing only the type
    turned a one-line diagnosis into a Colab round trip, 26 times over.
    """
    for index, source in enumerate(_code_sources()):
        assert "__name__" not in source or "type(" not in source, (
            f"code cell {index} may be printing an exception's type instead of "
            "its message"
        )


# --- namespace ----------------------------------------------------------------
def test_part_two_never_depends_on_a_part_one_export_cell() -> None:
    """Part 2 is routinely re-run alone after a reconnect.

    The user re-runs the clone cell, Step 0 and Step 1, then Part 2. A Part 2
    cell that needs a name bound only while submitting the exports raises
    NameError there - which is how ``MODIS_SCALE_M`` had to be hoisted.
    """
    cells = _cells()
    split = next(
        index for index, cell in enumerate(cells)
        if cell["cell_type"] == "markdown"
        and "# Part 2 - render" in "".join(cell["source"])
    )
    trees = {
        index: ast.parse(_neutralise_magics("".join(cell["source"])))
        for index, cell in enumerate(cells)
        if cell["cell_type"] == "code"
    }
    step0 = next(
        index for index in trees if "_required = {" in "".join(cells[index]["source"])
    )
    step1 = next(
        index for index in trees
        if "WORK_REGION = prediction.work_region" in "".join(cells[index]["source"])
    )
    prelude: set[str] = set()
    for index in trees:
        if index <= step0 or index == step1:
            prelude |= _bound(trees[index])

    problems: list[str] = []
    for index in sorted(trees):
        if index < split:
            continue
        own = _bound(trees[index])
        used = {
            node.id for node in ast.walk(trees[index])
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        for name in sorted(used - own - prelude):
            if hasattr(builtins, name):
                continue
            earlier = [
                other for other in sorted(trees)
                if split <= other < index and name in _bound(trees[other])
            ]
            if not earlier:
                problems.append(f"cell {index} uses {name!r}")
    assert not problems, (
        "these Part 2 names are unavailable in a Part-2-only run: "
        + "; ".join(problems)
    )
