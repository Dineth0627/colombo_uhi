"""Execute notebook 06's own Part 2 / Part 3 cells against synthetic data.

Three notebook-only bugs reached Colab and cost a run each, because the local
dry-run harness was a hand-written parallel copy of the notebook that drifted
from it:

======  ============  ====================================================
run     failure       cause
======  ============  ====================================================
2       NameError     Step 12 still referenced ``KAPPA`` / ``NULL_KAPPA`` /
                      ``FOM`` after Step 9 was rewritten around
                      ``LULC_METRICS``
2       wrong result  Step 10 never passed ``grouped=`` to
                      ``ca_markov_project``, so grass and shrub became
                      all-zero identity rows and the areas table was
                      mislabelled
4       KeyError      Step 10 still keyed ``VALIDATION`` by scheme after
                      Step 9 moved to a ``(early, late, target, scheme)``
                      tuple
======  ============  ====================================================

None of them was reachable by a copy that had already been written correctly,
and none is the sort of thing a unit test of ``prediction.py`` can see: every
function involved was fine. The bug was always in the *notebook's* wiring.

So this loads ``notebooks/06_prediction.ipynb`` and runs its real code cells,
verbatim, in one namespace.

What is faked, and only this:

* ``rasterio`` and ``geopandas``, which the local test environment lacks;
* the four ``prediction.read_*`` functions, which would otherwise need real
  Earth Engine exports on disk;
* the files those readers look for, as empty placeholders so the discovery
  cell's existence checks pass;
* the analysis-grid area check, which is calibrated to the real 699 km2
  district and would refuse a toy grid.

Everything else - every line of analysis in Part 2 - is the notebook's own. What
this asserts is narrow and worth exactly what it says: **the cells run, in
order, without a NameError, KeyError or TypeError.** It says nothing about
whether the science is right; that is what the rest of the suite and the Colab
sign-off checklist are for.
"""

from __future__ import annotations

import io
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from colombo_uhi import load_params, prediction, repo_root

pytest.importorskip("sklearn")

#: Part 2 and Part 3, minus the one cell needing a live Earth Engine session.
NOTEBOOK_CELLS: tuple[str, ...] = (
    "cell06",  # Step 0 constants
    "cell22",  # file discovery
    "cell24",  # Track A: blocked CV, fit, held-out scatter
    "cell26",  # split comparison + permutation importance
    "cell28",  # LULC read, VALID, area check, transition matrix
    "cell29",  # two intervals x two schemes, assess-and-continue
    "cell31",  # horizons + GHSL cross-check
    "cell33",  # priority zones
    "cell34",  # scenarios
    "cell36",  # class-conditional profile + extrapolation
    "cell36b",  # Step 12a: greening on the observed baseline (validated, exports)
    "cell36c",  # Step 12a figures + per-division table
    "cell37",  # validation reports + projected figures
    "cell38",  # per-division summary
    "cell41",  # MOLUSCE reconciliation (absent -> skip path)
    "cell43",  # the guard, exercised deliberately
    "cell44",  # write the guarded products
    "cell46",  # bundle
)

ROWS, COLS = 90, 140


def _fake_geo_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub rasterio and geopandas so the notebook's imports succeed.

    Registered through ``monkeypatch.setitem`` so the fakes are torn down
    with the test. A bare ``sys.modules`` assignment leaks into the rest of
    the session and turns every other module's ``importorskip("rasterio")``
    into a false positive - which is exactly what it did on first writing.

    Forced even where the real packages ARE installed (Colab), so the test
    behaves identically everywhere: the faked readers hand back synthetic
    arrays that real rasterio and geopandas would reject.
    """

    class _Handle:
        count = 1
        profile = {"width": COLS, "height": ROWS, "transform": None,
                   "crs": "EPSG:32644", "dtype": "float32"}

        def __enter__(self) -> "_Handle":
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

        def read(self, index: int, masked: bool = False) -> Any:
            data = np.random.default_rng(index).random((ROWS, COLS))
            return np.ma.masked_invalid(data) if masked else data

        # Writing is a no-op: this test is about the notebook's wiring, not
        # about GDAL. That the guarded writers are REACHED, with a passing
        # report, is what matters - and the WRITTEN list records it.
        def write(self, *args: Any, **kwargs: Any) -> None:
            return None

        def update_tags(self, *args: Any, **kwargs: Any) -> None:
            return None

    def _rasterize(shapes, out_shape, transform=None, fill=-1, dtype="int32"):
        out = np.full(out_shape, fill, dtype=dtype)
        pairs = list(shapes)
        per = max(out_shape[0] // max(len(pairs), 1), 1)
        for index, (_, value) in enumerate(pairs):
            low = index * per
            if low < out_shape[0]:
                out[low : low + per, : out_shape[1] // 2] = value
        return out

    rasterio = types.ModuleType("rasterio")
    rasterio.open = lambda *a, **k: _Handle()
    features = types.ModuleType("rasterio.features")
    features.rasterize = _rasterize
    rasterio.features = features
    rasterio.transform = types.ModuleType("rasterio.transform")

    geopandas = types.ModuleType("geopandas")
    geopandas.read_file = lambda *a, **k: pd.DataFrame()

    monkeypatch.setitem(sys.modules, "rasterio", rasterio)
    monkeypatch.setitem(sys.modules, "rasterio.features", features)
    monkeypatch.setitem(sys.modules, "rasterio.transform", rasterio.transform)
    monkeypatch.setitem(sys.modules, "geopandas", geopandas)


def _synthetic(params: dict[str, Any]) -> dict[str, Any]:
    settings = prediction.resolve_rf_settings(params)
    classes = prediction.resolve_ca_classes(params)
    intervals = prediction.resolve_validation_intervals(params)
    base = [int(y) for y in params["prediction"]["ca_markov"]["projection_base_years"]]
    years = sorted({*base, *(y for triplet in intervals for y in triplet)})

    rng = np.random.default_rng(0)
    start = rng.choice(
        classes, (ROWS, COLS), p=[0.10, 0.18, 0.14, 0.05, 0.10, 0.33, 0.10]
    )
    lulc, observed = {}, {}
    for index, year in enumerate(years):
        grid = start.copy()
        grid[(rng.random(grid.shape) < 0.04 * index) & (grid != 0)] = 6
        grid[np.isin(grid, [2, 5]) & (rng.random(grid.shape) < 0.3 * min(index, 1))] = 1
        lulc[year] = grid.astype(np.int16)
        seen = np.ones(grid.shape, dtype=bool)
        seen[:index] = False
        observed[year] = seen

    yy, xx = np.mgrid[0:ROWS, 0:COLS].astype(float)
    arrays = {
        name: np.sin(yy / (9 + i)) + np.cos(xx / (13 + i))
        for i, name in enumerate(settings["predictors"])
    }
    arrays["built_fraction"] = (lulc[base[1]] == 6).astype(float)
    arrays["lcz_class"] = rng.integers(1, 18, (ROWS, COLS)).astype(float)
    arrays["NDVI"] = np.where(lulc[base[1]] == 1, 0.7, 0.2) + rng.normal(
        0, 0.05, (ROWS, COLS)
    )
    arrays[settings["response"]] = (
        29.0 + 4.0 * arrays["built_fraction"] - 3.0 * arrays["NDVI"]
        + rng.normal(0, 0.3, (ROWS, COLS))
    )

    count = 3000
    py, px = rng.integers(0, ROWS, count), rng.integers(0, COLS, count)
    sample = pd.DataFrame({
        "x": 400000.0 + px * settings["scale_m"],
        "y": 760000.0 + py * settings["scale_m"],
        settings["response"]: arrays[settings["response"]][py, px],
        **{name: arrays[name][py, px] for name in settings["predictors"]},
    })
    sample["block_id"] = prediction.spatial_block_ids(
        sample["x"].to_numpy(), sample["y"].to_numpy(),
        prediction.resolve_split(params)["block_size_m"],
    )
    return {
        "lulc": lulc, "observed": observed, "arrays": arrays, "sample": sample,
        "zones": pd.DataFrame({
            "zone_id": [f"LK11{i:04d}" for i in range(24)],
            "geometry": [None] * 24,
        }),
        "profile": {"width": COLS, "height": ROWS, "transform": None,
                    "crs": "EPSG:32644", "dtype": "int16", "count": 2},
    }


def test_every_part_2_notebook_cell_executes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = load_params()
    _fake_geo_modules(monkeypatch)
    state = _synthetic(params)

    # A small forest: this test is about wiring, not accuracy.
    real_settings = prediction.resolve_rf_settings
    monkeypatch.setattr(
        prediction, "resolve_rf_settings",
        lambda p, **kw: {**real_settings(p, **kw), "n_trees": 8,
                         "min_sample_rows": 50},
    )
    monkeypatch.setattr(
        prediction, "read_lulc_raster",
        lambda path, p: (
            state["lulc"][int(str(path)[-8:-4])],
            state["observed"][int(str(path)[-8:-4])],
            state["profile"],
        ),
    )
    monkeypatch.setattr(
        prediction, "read_predictor_raster",
        lambda path, p: (state["arrays"], state["profile"]),
    )
    monkeypatch.setattr(
        prediction, "read_training_sample",
        lambda path, p, add_blocks=True: state["sample"].copy(),
    )
    monkeypatch.setattr(
        prediction, "read_priority_geometry",
        lambda path, p: state["zones"].copy(),
    )
    # Calibrated to the real 699 km2 district; this grid is a toy.
    monkeypatch.setattr(
        prediction, "require_expected_area",
        lambda n, p, **kw: {"n_cells": int(n), "area_km2": float(n) / 100.0,
                            "expected_km2": 699.0, "departure": 0.0},
    )

    notebook = json.loads(
        (repo_root() / "notebooks" / "06_prediction.ipynb").read_text(
            encoding="utf-8"
        )
    )
    sources = {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    }
    missing = [name for name in NOTEBOOK_CELLS if name not in sources]
    assert not missing, (
        f"notebook 06 has no cell(s) {missing}; the ids moved, so this test is "
        "no longer running what it claims to run"
    )

    monkeypatch.chdir(tmp_path)
    for folder in ("data/interim", "data/outputs", "figures"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)

    from colombo_uhi import aoi, exports, landcover, spatial_stats, viz

    namespace: dict[str, Any] = {
        "__name__": "__main__", "params": params, "prediction": prediction,
        "np": np, "pd": pd, "os": os, "sys": sys, "aoi": aoi,
        "exports": exports, "landcover": landcover,
        "spatial_stats": spatial_stats, "viz": viz,
        "display": lambda *a, **k: None, "Image": lambda **k: None,
    }

    for name in NOTEBOOK_CELLS:
        source = sources[name]
        if name == "cell06":
            # Step 0's region guard needs a live Earth Engine call; the rest of
            # the cell is the constants Part 2 runs on.
            source = source[: source.index("# ---- THE REGION GUARD")]
            source += "\nWORK_REGION = None\n"
        if name == "cell22":
            for year in state["lulc"]:
                (tmp_path / "data/interim"
                 / f"lulc_district_2000_2025_100m_{year}.tif").touch()
            for stem in (
                "rf_training_sample_district_2000_2025_100m_2020s_landsat_oli_dry.csv",
                "predictor_stack_district_2000_2025_100m_2020s_landsat_oli_dry.tif",
                "ghsl_built_district_2000_2025_100m_2030.tif",
            ):
                (tmp_path / "data/interim" / stem).touch()
        if name == "cell33":
            (tmp_path / "data/outputs/gn_divisions_colombo.geojson").touch()
            pd.DataFrame({
                "zone_id": state["zones"]["zone_id"],
                "gi_z": np.random.default_rng(3).normal(size=len(state["zones"])),
            }).to_csv(tmp_path / "data/outputs/gi_star_gn_2020s.csv", index=False)

        try:
            exec(compile(source, f"<notebook06:{name}>", "exec"), namespace)
        except Exception as error:  # noqa: BLE001 - the whole point is to report
            raise AssertionError(
                f"notebook 06 cell {name} raised "
                f"{type(error).__name__}: {error}\n"
                "This is a wiring bug in the NOTEBOOK, not in prediction.py - "
                "a name, key or argument that Step 9's or Step 10's rewrites "
                "left behind. Three of these reached Colab and cost a run each."
            ) from error

    # The cells ran; check they actually did the work rather than no-opping.
    assert "LULC_VALIDATED" in namespace, "Step 9 never set the validation flag"
    assert "SURFACES" in namespace and namespace["SURFACES"], (
        "Step 12 produced no projected surfaces"
    )
    # Run 5's finding: the class-flip lever converted ONE cell, because grass,
    # shrub and bare are 0.63% of the district. The canopy shift must actually
    # move something, or deliverable 3 is a no-op again.
    assert namespace["CANOPY_REPORT"]["n_shifted"] > 0, (
        "Step 12a shifted no cells - the greening lever does not bite"
    )
    assert namespace["GREENING_VALIDATED"], (
        "the greening counterfactual rests on Track A alone and must validate"
    )
    # ...and being validated, it must actually reach the guarded writer, while
    # the CA-dependent products do not.
    written = namespace.get("WRITTEN", [])
    assert any("greening_counterfactual" in path for path in written), (
        f"the greening counterfactual was not written; WRITTEN holds {written}"
    )
    assert not any("lulc_projected" in path for path in written), (
        "a land-cover projection was written despite failing validation"
    )
    assert (tmp_path / "figures").glob("*.png"), "no figures were written"


def test_step_9_assesses_the_projection_rather_than_requiring_it() -> None:
    """Step 9 must ASSESS, not require - requiring is what killed run 3.

    A land-cover projection that fails validation is a MEASURED RESULT. Calling
    ``require_validated`` on it turns that result into a traceback and takes
    every valid product beside it down too - which is exactly what happened, at
    this exact line.

    Track A's own fit report is a different case: Step 7 may require it, because
    if the present-day surface cannot be validated there is nothing to continue
    to. Only the projection is assessed-and-carried.
    """
    import ast

    notebook = json.loads(
        (repo_root() / "notebooks" / "06_prediction.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = next(
        "".join(cell["source"]) for cell in notebook["cells"]
        if cell.get("id") == "cell29"
    )
    called = {
        node.func.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "require_validated" not in called, (
        "notebook 06 Step 9 calls require_validated on the land-cover "
        "projection. Use assess_validation and carry the verdict - the export "
        "wrappers do the enforcing, and a measured failure must not abort the "
        "run."
    )
    assert "assess_validation" in called, (
        "notebook 06 Step 9 no longer assesses the projection at all"
    )
