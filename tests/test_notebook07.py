"""Execute notebook 07's own Part 2 / Part 3 cells against synthetic data.

Phase 6 lost three Colab runs to notebook-only bugs - a ``NameError``, a
``KeyError`` and a silently mislabelled table - and none of them was reachable by
a unit test of ``prediction.py``, because every function involved was correct.
The bug was always in the *notebook's* wiring, and the local harness that should
have caught it was a hand-written parallel copy that drifted from the notebook it
was supposed to mirror.

So this loads ``notebooks/07_greening_priority.ipynb`` and runs its real code
cells, verbatim, in one namespace.

What is faked, and only this:

* the four ``greening.read_*`` raster readers and
  ``spatial_stats.read_zone_covariates``, which would otherwise need real Earth
  Engine exports on disk;
* ``prediction.read_priority_geometry``, which reads a committed GeoJSON;
* the files the discovery cell checks for, as empty placeholders.

**geopandas, rasterio, scipy and matplotlib are NOT faked.** They are declared
dependencies, and the figure cell is 400 lines of map-drawing code whose first
execution would otherwise be in Colab - which is precisely the situation that
cost Phase 6 three runs. The zone polygons handed in are a real
``GeoDataFrame`` and the figures are really rendered.

What this asserts is narrow and worth exactly what it says: **the cells run, in
order, without a NameError, KeyError or TypeError, and produce the objects the
later cells expect.** It says nothing about whether the science is right; that is
what the rest of the suite and the Colab sign-off checklist are for.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from colombo_uhi import load_params, repo_root

gpd = pytest.importorskip("geopandas")
pytest.importorskip("rasterio")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")
shapely_geometry = pytest.importorskip("shapely.geometry")

from colombo_uhi import greening, prediction, spatial_stats  # noqa: E402

#: Part 2 and Part 3, minus the two cells needing a live Earth Engine session.
NOTEBOOK_CELLS: tuple[str, ...] = (
    "cell06",  # Step 0 - constants and the staleness guard
    # Step 1 is Part 1 by position and pure Python by nature: the AHP is
    # arithmetic on params alone, touching no data and no Earth Engine. Part 2
    # depends on WEIGHTS / AHP_REPORT throughout, so a fresh Part-2-only session
    # must re-run it - which is what the WAIT HERE cell now says.
    "cell08",  # Step 1  - the AHP, assessed and continued
    "cell26",  # file discovery
    "cell28",  # Step 9  - assemble the criterion frame
    "cell30",  # Step 10 - the 3-30-300 rule
    "cell32",  # Step 11 - coverage floor, prepare, correlation, dimensionality
    "cell34",  # Step 12 - overlay + normalisation and sensor sensitivities
    "cell36",  # Step 13 - TOPSIS and the three-way decomposition
    "cell38",  # Step 14 - ablation and circularity
    "cell40",  # Step 15 - the wetland cross
    "cell42",  # Step 16 - the DS-level MAUP sensitivity
    "cell44",  # Step 17 - the figures
    "cell48",  # Step 18 - the guard, exercised deliberately
    "cell50",  # Step 19 - write every product under the guard
    "cell52",  # bundle
)

#: Cells that assemble or normalise criteria. None of them may draw a criterion
#: from a Phase 6 projected product - Track B never beat a no-change map.
CRITERION_CELLS: tuple[str, ...] = ("cell28", "cell32", "cell34")

# *** THESE GRIDS DELIBERATELY DO NOT NEST. ***
# Earth Engine snaps each export grid to its own scale, so the 10 m and 100 m
# rasters over Colombo came back as 2957 x 4219 against 297 x 423 - the coarse
# grid overhanging by one cell per edge. Colab run 2 stopped on exactly that, and
# this fixture could not have caught it: it carried nesting shapes and
# `"transform": None`. It now mirrors the real failure at a testable size, so
# Step 10's alignment is exercised rather than assumed.
FACTOR = 10
COARSE_ROWS, COARSE_COLS = 40, 50
FINE_ROWS, FINE_COLS = COARSE_ROWS * FACTOR - 3, COARSE_COLS * FACTOR - 3  # 397 x 497
#: Shapes both grids are cropped to once aligned.
ALIGNED_COARSE = (COARSE_ROWS - 1, COARSE_COLS - 1)
ALIGNED_FINE = (ALIGNED_COARSE[0] * FACTOR, ALIGNED_COARSE[1] * FACTOR)
#: Shared world origin, in the analysis CRS.
ORIGIN_X, ORIGIN_Y = 400000.0, 800000.0
N_ZONES = 24


def _profile(pixel: float, height: int, width: int, count: int) -> dict[str, Any]:
    """A real north-up profile, so grid alignment has a transform to work from."""
    from rasterio.transform import from_origin

    return {
        "transform": from_origin(ORIGIN_X, ORIGIN_Y, pixel, pixel),
        "height": height,
        "width": width,
        "crs": "EPSG:32644",
        "count": count,
    }


def _zone_frame(prefix: str, count: int) -> "gpd.GeoDataFrame":
    """Real polygons tiling the synthetic raster grid, in the analysis CRS."""
    cell = 10.0
    per_row = 6
    boxes = []
    ids = []
    for index in range(count):
        col = index % per_row
        row = index // per_row
        width = (FINE_COLS * cell) / per_row
        height = (FINE_ROWS * cell) / max(1, -(-count // per_row))
        boxes.append(
            shapely_geometry.box(
                col * width, row * height, (col + 1) * width, (row + 1) * height
            )
        )
        ids.append(f"{prefix}{index:04d}")
    return gpd.GeoDataFrame(
        {
            "zone_id": ids,
            "adm4_name": [f"Division {index}" for index in range(count)],
            "adm3_name": ["Colombo"] * count,
            "adm3_pcode": [f"LK11{index % 4:02d}" for index in range(count)],
            "area_sqkm": [1.5] * count,
        },
        geometry=boxes,
        crs="EPSG:32644",
    )


def _synthetic(params: dict[str, Any]) -> dict[str, Any]:
    """Every array and table the faked readers hand back."""
    rng = np.random.default_rng(0)
    zones = _zone_frame("LK1103", N_ZONES)
    ds_zones = _zone_frame("LK11", 4)

    # A green/canopy raster with real structure: patches large enough to qualify
    # in some places and speckle too small in others.
    green = np.zeros((FINE_ROWS, FINE_COLS), dtype=bool)
    green[: FINE_ROWS // 3, : FINE_COLS // 2] = True          # a large park
    green[FINE_ROWS // 2 : FINE_ROWS // 2 + 40, -80:] = True  # a smaller one, still >= 0.5 ha
    green[-3:, :3] = True                                     # speckle, well under 0.5 ha
    green |= rng.random((FINE_ROWS, FINE_COLS)) < 0.02
    canopy = green & (rng.random((FINE_ROWS, FINE_COLS)) < 0.7)
    observed = np.ones((FINE_ROWS, FINE_COLS), dtype=bool)
    observed[:4, :] = False  # a strip the classifier never saw
    # Permanent water along one edge, standing in for the Colombo Port outer
    # harbour: the cells that must NOT count against a zone's land coverage.
    water = np.zeros((FINE_ROWS, FINE_COLS), dtype=bool)
    water[:, :20] = True

    population = rng.lognormal(4.0, 0.8, (COARSE_ROWS, COARSE_COLS))
    pop_observed = np.ones((COARSE_ROWS, COARSE_COLS), dtype=bool)

    wetland_sources = greening.resolve_wetland_sources(params)
    wetland_bands: dict[str, Any] = {}
    union = np.zeros((FINE_ROWS, FINE_COLS), dtype=bool)
    for index, source in enumerate(wetland_sources):
        band = np.zeros((FINE_ROWS, FINE_COLS), dtype=bool)
        band[20 + index * 6 : 34 + index * 6, 5:45] = True
        wetland_bands[source] = band
        union |= band
    wetland_bands["wetland"] = union
    wetland_bands["n_sources"] = union.astype(np.int16)
    wetland_bands["observed"] = np.ones((FINE_ROWS, FINE_COLS), dtype=bool)

    def _covariates(ids: list[str], seed: int) -> pd.DataFrame:
        local = np.random.default_rng(seed)
        count = len(ids)
        frame = pd.DataFrame(
            {
                "zone_id": ids,
                "LST_C": local.normal(31.0, 1.6, count),
                "NDVI": local.random(count) * 0.6,
                "NDBI": local.normal(-0.05, 0.1, count),
                "MNDWI": local.normal(-0.2, 0.1, count),
                "built_fraction": local.random(count),
                "pop_density": local.lognormal(9.0, 1.0, count),
                "elevation_m": local.random(count) * 20.0,
                "dist_coast_km": local.random(count) * 12.0,
            }
        )
        for column in list(frame.columns):
            if column != "zone_id":
                frame[f"{column}_pixels"] = 400
        return frame

    zone_ids = list(zones["zone_id"])
    ds_ids = list(ds_zones["zone_id"])
    return {
        "zones": zones,
        "ds_zones": ds_zones,
        "green": {
            "green": green & observed,
            "canopy": canopy & observed,
            "water": water,
            "observed": observed,
            "land": ~water,
        },
        "green_profile": _profile(10.0, FINE_ROWS, FINE_COLS, 4),
        "population": population,
        "pop_observed": pop_observed,
        "pop_profile": _profile(100.0, COARSE_ROWS, COARSE_COLS, 2),
        "wetland_bands": wetland_bands,
        "wetland_profile": _profile(10.0, FINE_ROWS, FINE_COLS, len(wetland_bands)),
        "covariates_gn": _covariates(zone_ids, 1),
        "covariates_ds": _covariates(ds_ids, 2),
        "covariates_oli": _covariates(zone_ids, 3),
        "utfvi_gn": pd.DataFrame(
            {
                "zone_id": zone_ids,
                "utfvi_severe_share": rng.random(len(zone_ids)),
                "utfvi_severe_share_pixels": 400,
            }
        ),
        "utfvi_ds": pd.DataFrame(
            {
                "zone_id": ds_ids,
                "utfvi_severe_share": rng.random(len(ds_ids)),
                "utfvi_severe_share_pixels": 400,
            }
        ),
    }


def _zone_codes(profile: Any, geometry: Any, params: Any, **kwargs: Any) -> Any:
    """Deterministic zone raster: horizontal bands, one per zone.

    Substituted for ``greening.zone_raster`` so the notebook test does not depend
    on real polygon rasterisation; ``zone_raster`` itself is tested for real in
    ``tests/test_greening.py``.

    The bands are laid out by FRACTION of height rather than by a fixed row
    count, so the fine and coarse rasters describe the same geography despite
    having different shapes - which is what makes the population-weighted share
    comparable against the area share.
    """
    height = int(profile["height"])
    width = int(profile["width"])
    ids = [str(value) for value in geometry["zone_id"]]
    codes = np.zeros((height, width), dtype=np.int32)
    per = height / max(len(ids), 1)
    for index in range(len(ids)):
        low, high = int(index * per), int((index + 1) * per)
        if low >= height:
            break
        codes[low : min(max(high, low + 1), height), :] = index + 1
    return codes, {index + 1: value for index, value in enumerate(ids)}


def _load_sources() -> dict[str, str]:
    notebook = json.loads(
        (repo_root() / "notebooks" / "07_greening_priority.ipynb").read_text(
            encoding="utf-8"
        )
    )
    return {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }


def test_notebook_cell_ids_are_stable() -> None:
    """A renumbering must not silently make this file test nothing."""
    sources = _load_sources()
    missing = [name for name in NOTEBOOK_CELLS if name not in sources]
    assert not missing, (
        f"notebook 07 has no cell(s) {missing}; the ids moved, so this test is no "
        "longer running what it claims to run. Fix NOTEBOOK_CELLS deliberately."
    )
    for name in CRITERION_CELLS:
        assert name in sources, f"criterion cell {name} is gone"


def test_every_part_2_notebook_cell_executes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = load_params()
    state = _synthetic(params)
    sources = _load_sources()

    monkeypatch.setattr(
        greening,
        "read_green_canopy_raster",
        lambda path, p: (dict(state["green"]), dict(state["green_profile"])),
    )
    monkeypatch.setattr(
        greening,
        "read_population_raster",
        lambda path, p: (
            state["population"].copy(),
            state["pop_observed"].copy(),
            dict(state["pop_profile"]),
        ),
    )
    monkeypatch.setattr(
        greening,
        "read_wetland_raster",
        lambda path, p, sources=None: (
            dict(state["wetland_bands"]),
            dict(state["wetland_profile"]),
        ),
    )
    monkeypatch.setattr(greening, "zone_raster", _zone_codes)
    monkeypatch.setattr(
        spatial_stats,
        "read_zone_covariates",
        lambda path, p, level: (
            state["covariates_ds"].copy()
            if level == "ds"
            else (
                state["covariates_oli"].copy()
                if "oli" in str(path)
                else state["covariates_gn"].copy()
            )
        ),
    )
    monkeypatch.setattr(
        prediction,
        "read_priority_geometry",
        lambda path, p: (
            state["ds_zones"].copy()
            if "ds_divisions" in str(path)
            else state["zones"].copy()
        ),
    )

    monkeypatch.chdir(tmp_path)
    for folder in ("data/interim", "data/outputs", "figures"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)

    # The files the discovery cell globs for, plus the committed inputs Part 2
    # reads directly.
    interim = tmp_path / "data/interim"
    # THE REAL NAMES Colab run 1 produced. The first three matter: the export
    # template puts the level in the SUFFIX, so they are
    # `zone_covariates_district_..._gn_2020s.csv`, not `zone_covariates_gn_...`.
    # This fixture originally carried the guessed form, which is why the run
    # reached Part 2 with globs that matched nothing.
    for stem in (
        "zone_covariates_district_2000_2025_100m_gn_2020s.csv",
        "zone_covariates_district_2000_2025_100m_ds_2020s.csv",
        "zone_covariates_district_2000_2025_100m_gn_2020s_oli.csv",
        "greening_utfvi_severe_gn.csv",
        "greening_utfvi_severe_ds.csv",
        "greening_green_canopy_district_2024_2024_10m.tif",
        "greening_population_district_2020_2020_100m.tif",
        "greening_wetland_district_2024_2024_10m.tif",
    ):
        (interim / stem).touch()
    state["utfvi_gn"].to_csv(interim / "greening_utfvi_severe_gn.csv", index=False)
    state["utfvi_ds"].to_csv(interim / "greening_utfvi_severe_ds.csv", index=False)

    outputs = tmp_path / "data/outputs"
    (outputs / "gn_divisions_colombo.geojson").touch()
    (outputs / "ds_divisions_colombo.geojson").touch()
    aligned_fine = _profile(10.0, *ALIGNED_FINE, 3)
    aligned_codes, aligned_labels = _zone_codes(aligned_fine, state["zones"], params)
    land_area = greening.zone_land_area(aligned_codes, aligned_labels, 10.0)
    classified = land_area["land_area_ha"].to_numpy() * 0.97
    classified[:3] *= 0.5  # three zones genuinely below the floor
    landscape = pd.DataFrame(
        {
            "scheme": ["dynamic_world"] * N_ZONES,
            "year": [int(params["greening"]["landcover_year"])] * N_ZONES,
            "zone_id": list(land_area["zone_id"]),
            "landscape_area_ha": classified,
            "observed_fraction": np.linspace(0.6, 1.0, N_ZONES),
        }
    )
    landscape.to_csv(outputs / "landscape_metrics_green_by_gn.csv", index=False)
    pd.DataFrame(
        {
            "zone_id": list(state["zones"]["zone_id"]),
            "score": np.linspace(1.0, 0.0, N_ZONES),
            "rank": np.arange(1, N_ZONES + 1),
            "priority": [True] * 8 + [False] * (N_ZONES - 8),
        }
    ).to_csv(
        tmp_path / params["greening"]["ablation"]["circularity_reference"], index=False
    )

    from colombo_uhi import aoi, exports, landcover, uhi_metrics, viz

    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "params": params,
        "greening": greening,
        "prediction": prediction,
        "spatial_stats": spatial_stats,
        "uhi_metrics": uhi_metrics,
        "landcover": landcover,
        "exports": exports,
        "aoi": aoi,
        "viz": viz,
        "np": np,
        "pd": pd,
        "os": os,
        "sys": sys,
        "display": lambda *a, **k: None,
        "Image": lambda **k: None,
        # Part 1 constants Part 2 legitimately depends on. In Colab these are set
        # by cells 10 and 12; here they are supplied so Part 2 can be run as the
        # separate, Earth-Engine-free session the notebook says it is.
        "LC_YEAR": int(params["greening"]["landcover_year"]),
        "USABLE_WETLAND_SOURCES": greening.resolve_wetland_sources(params),
    }

    for name in NOTEBOOK_CELLS:
        source = sources[name]
        if name == "cell06":
            # Step 0 ends with prints; everything before them is the constants and
            # the staleness guard that Part 2 runs on. `import ee` is left in - it
            # is a declared dependency and importing it needs no credentials.
            pass
        try:
            exec(compile(source, f"<notebook07:{name}>", "exec"), namespace)
        except Exception as error:  # noqa: BLE001 - reporting is the whole point
            raise AssertionError(
                f"notebook 07 cell {name} raised {type(error).__name__}: {error}\n"
                "This is a wiring bug in the NOTEBOOK, not in greening.py - a name, "
                "key or argument a rewrite left behind. Three of these reached "
                "Colab in Phase 6 and cost a run each."
            ) from error

    # The cells ran; check they actually did the work rather than no-opping.
    assert np.isfinite(namespace["AHP_REPORT"]["consistency_ratio"])
    ranked = namespace["RANKED"]
    assert len(ranked) == N_ZONES
    assert ranked["rank_ahp"].tolist() == list(range(1, N_ZONES + 1))
    assert "TOPSIS_RANKED" in namespace and len(namespace["TOPSIS_RANKED"]) == N_ZONES
    assert set(namespace["COMPLIANCE"]["compliance"]) <= set(
        greening.COMPLIANCE_CATEGORIES
    )
    # Non-degenerate: a fixture that lands every zone in one category would run
    # the cell without exercising the verdict.
    assert namespace["COMPLIANCE"]["compliance"].nunique() >= 3

    # *** THE RUN-2 FAILURE, REPRODUCED AND FIXED. ***
    # The fixture grids deliberately do not nest, so Step 10 must have trimmed
    # them; if this ever reads 0 the fixture has drifted back to nesting shapes
    # and stopped testing the alignment at all.
    alignment = namespace["ALIGNMENT"]
    assert alignment["dropped_fraction"] > 0
    assert alignment["fine_window"][2:] == ALIGNED_FINE
    assert alignment["coarse_window"][2:] == ALIGNED_COARSE
    assert namespace["SERVICE"].shape == ALIGNED_FINE
    assert namespace["POPULATION"].shape == ALIGNED_COARSE
    # And the guard that caught run 2 passed as a post-condition rather than
    # being bypassed.
    greening.require_integer_refinement(
        namespace["SERVICE"].shape, namespace["POPULATION"].shape, FACTOR
    )

    # Both sides of the land-coverage floor were exercised.
    prep = namespace["PREP_REPORT"]
    assert prep["n_ok"] > 0 and prep["n_below_floor"] > 0
    assert namespace["CIRCULARITY"]["independence"] == greening.NOT_INDEPENDENT
    assert not namespace["ABLATION"].empty
    assert "WETLAND" in namespace and not namespace["WETLAND"].empty
    assert len(namespace["DS_RANKED"]) == 4

    written = namespace["WRITTEN"]
    assert "ranked" in written and "top" in written
    for path in written.values():
        assert Path(path).is_file(), f"{path} was not written"
    assert Path(str(written["ranked"])).with_name(
        Path(str(written["ranked"])).stem + "_meta.json"
    ).is_file(), "the ranked table has no metadata sidecar"

    for path in namespace["FIGURES"].values():
        assert Path(path).is_file() and Path(path).stat().st_size > 0

    assert namespace["BUNDLED"], "the bundle cell collected no files"
    assert namespace["PRIORITY_ZONE_IDS"], "no priority zone ids for Phase 6"
    assert all(isinstance(value, str) for value in namespace["PRIORITY_ZONE_IDS"])


def _neutralise_magics(source: str) -> str:
    """Replace ``!shell`` / ``%magic`` lines with ``pass``, preserving indentation.

    Deleting them instead would empty the clone cell's ``if`` block and turn a
    name-flow check into a syntax error.
    """
    lines = []
    for line in source.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            lines.append(" " * (len(line) - len(stripped)) + "pass")
        else:
            lines.append(line)
    return "\n".join(lines)


def test_no_all_caps_name_is_used_before_it_is_assigned() -> None:
    """Catch a ``NameError`` in Part 1, which no test can execute.

    Part 1 needs a live Earth Engine session, so its cells cannot be run here -
    and that is exactly where Phase 6's run-2 ``NameError`` lived: Step 12 still
    referenced ``KAPPA`` / ``NULL_KAPPA`` / ``FOM`` after Step 9 had been
    rewritten around ``LULC_METRICS``. A static flow check over the ALL-CAPS
    names - the notebook's convention for state that crosses cells - reaches what
    execution cannot.
    """
    notebook = json.loads(
        (repo_root() / "notebooks" / "07_greening_priority.ipynb").read_text(
            encoding="utf-8"
        )
    )
    cells = [
        (cell["id"], "".join(cell["source"]))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]

    defined: set[str] = set()
    problems: list[str] = []
    for cell_id, source in cells:
        try:
            tree = ast.parse(_neutralise_magics(source))
        except SyntaxError as error:  # pragma: no cover - a broken notebook
            problems.append(f"{cell_id}: does not parse - {error}")
            continue

        loaded: set[str] = set()
        assigned: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    assigned.add(node.id)
                elif node.id.isupper() and len(node.id) > 2:
                    loaded.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assigned.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    assigned.add((alias.asname or alias.name).split(".")[0])

        for name in sorted(loaded - defined - assigned):
            problems.append(f"{cell_id}: uses {name} before any cell assigns it")
        defined |= assigned

    assert not problems, "\n".join(problems)


def test_the_ahp_step_assesses_rather_than_requires() -> None:
    """Step 1 must assess and continue, never enforce.

    Phase 6's Colab run 3 wired ``require_validated`` into the step that
    *measured* the result, so a measured negative read as a crash and destroyed
    every valid product beside it. An analyst whose judgements are inconsistent
    needs to see the weights in order to fix them; the refusal belongs at the
    writer. Checked on the AST rather than on a substring.
    """
    tree = ast.parse(_load_sources()["cell08"])
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "ahp_weights" in called, "Step 1 does not compute the AHP weights at all"
    assert "require_consistent" not in called, (
        "Step 1 calls require_consistent, so inconsistent judgements would raise "
        "here and take every later cell with them. Assess and continue; the "
        "refusal belongs in Part 3's writer."
    )


def test_part_3_refuses_before_it_writes() -> None:
    """The guard cell must run before the writing cell, and must actually guard."""
    sources = _load_sources()
    guard = ast.parse(sources["cell48"])
    called = {
        node.func.attr
        for node in ast.walk(guard)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {"require_consistent", "require_complete_criteria"} <= called, (
        "Step 18 does not exercise both guards, so a refusal nobody has seen fire "
        "is a refusal nobody knows is wired up"
    )
    assert NOTEBOOK_CELLS.index("cell48") < NOTEBOOK_CELLS.index("cell50")


def test_the_ranked_product_is_written_through_the_guarded_writer() -> None:
    """The ranked table may not be written with a bare ``to_csv``."""
    tree = ast.parse(_load_sources()["cell50"])
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "write_priority_table" in called, (
        "Step 19 does not use the guarded writer, so an inconsistent set of "
        "judgements could publish a priority table"
    )


def test_no_criterion_is_drawn_from_a_projected_product() -> None:
    """D1's structural guard.

    Phase 6's Track B never beat a no-change map, so a criterion drawn from a
    projected surface would inherit a model with no demonstrated allocation
    skill. This is enforced on the notebook rather than left to convention.
    """
    sources = _load_sources()
    forbidden = (
        "projected_lst_by_gn",
        "greening_counterfactual",
        "lulc_projected",
        "projected_class_areas",
        "scenario_conversions",
    )
    for name in CRITERION_CELLS:
        source = sources[name]
        hits = [token for token in forbidden if token in source]
        assert not hits, (
            f"notebook 07 cell {name} references {hits}. Every criterion must be an "
            "OBSERVED 2020s quantity: Track B never beat a no-change map, so a "
            "criterion drawn from a projected product would inherit a model with "
            "no demonstrated allocation skill."
        )


def test_coverage_is_computed_from_one_raster_not_two_products() -> None:
    """The run-4 lesson, pinned on the notebook.

    Run 4 took the classified area from Phase 5's committed
    ``landscape_metrics_green_by_gn.csv`` and the land area from the current 10 m
    export - two different products - so the ratio measured the difference
    between them rather than coverage, and Pettah stayed out of the priority
    list. ``greening.zone_coverage`` takes ONE bands mapping and derives both
    sides from it; calling ``land_observed_fraction`` directly from the notebook
    re-opens the door to mixing them.
    """
    tree = ast.parse(_load_sources()["cell32"])
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "zone_coverage" in called, (
        "Step 11 does not use greening.zone_coverage, which is what keeps the "
        "coverage numerator and denominator on the same raster"
    )
    assert "land_observed_fraction" not in called, (
        "Step 11 calls land_observed_fraction directly. That takes two separately "
        "built frames and cannot check they came from the same product - which is "
        "exactly how run 4 produced a coverage ratio spanning two phases."
    )
