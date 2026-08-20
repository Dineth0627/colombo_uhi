"""Unit tests for colombo_uhi.reporting (Phase 8).

Two kinds of test here. The ordinary kind checks that the provenance table and
the reference search say true things. The important kind is the **drift test**:
it regenerates ``docs/methods.md``, ``docs/limitations.md`` and
``data/outputs/data_provenance.csv`` in memory and asserts the committed copies
match byte for byte.

That test is the whole mechanism behind ``caveats.figures_are_derived_not_authored``.
Without it, "generated from params.yaml" is a claim in a docstring; with it, a
threshold cannot be edited in the config and left stale in the write-up, because
the suite goes red until someone re-runs ``reporting.write_docs``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from colombo_uhi import load_params, repo_root, reporting


@pytest.fixture(scope="module")
def params() -> dict[str, Any]:
    return load_params()


# --- the provenance table ----------------------------------------------------
def test_provenance_covers_every_configured_dataset(params: dict[str, Any]) -> None:
    # A dataset added to params.yaml and left out of the table would make the
    # provenance figure a partial account of what the analysis rests on, which
    # is worse than no figure at all.
    frame = reporting.provenance_frame(params)
    for key in params["datasets"]:
        assert key in set(frame["key"]), f"{key} is configured but not in the table"
        row = frame[frame["key"] == key].iloc[0]
        assert row["collection_id"] == params["datasets"][key]["id"]


def test_provenance_columns_are_exactly_the_declared_order(
    params: dict[str, Any],
) -> None:
    frame = reporting.provenance_frame(params)
    assert tuple(frame.columns) == reporting.PROVENANCE_COLUMNS


def test_provenance_lists_the_uploaded_assets_but_not_property_names(
    params: dict[str, Any],
) -> None:
    # aoi.assets holds two asset PATHS alongside several property-name candidate
    # lists and a literal filter value. Only the paths are data sources; the
    # first draft listed all six and claimed "Colombo" was a dataset.
    frame = reporting.provenance_frame(params)
    assets = [key for key in frame["key"] if str(key).startswith("asset:")]
    assert set(assets) == {"asset:ds_divisions", "asset:gn_divisions"}
    for key in assets:
        row = frame[frame["key"] == key].iloc[0]
        assert "/assets/" in row["collection_id"]


def test_provenance_reports_landsat_and_modis_scale_factors(
    params: dict[str, Any],
) -> None:
    # Scale factors live in landsat_c2l2 / modis_lst, NOT in the datasets block,
    # so the table has to resolve them rather than read them off the entry.
    frame = reporting.provenance_frame(params).set_index("key")
    landsat = frame.loc["landsat8", "scale_factor"]
    assert str(params["landsat_c2l2"]["st_scale"]) in landsat
    assert "0.0000275" in landsat, "scientific notation leaked into the table"
    assert "+ -" not in landsat, "a negative offset was rendered as '+ -'"
    assert str(params["modis_lst"]["lst_scale"]) in frame.loc[
        "modis_terra_lst", "scale_factor"
    ]


def test_coverage_renders_an_open_ended_window_as_ongoing(
    params: dict[str, Any],
) -> None:
    frame = reporting.provenance_frame(params).set_index("key")
    assert frame.loc["landsat8", "temporal_coverage"].endswith("ongoing")
    assert frame.loc["landsat5", "temporal_coverage"].endswith("2012-05-05")


# --- the reference search ----------------------------------------------------
def test_references_find_datasets_reached_only_through_the_config(
    params: dict[str, Any],
) -> None:
    # landsat5 appears in NO source file: landsat.resolve_sensors reads the key
    # list out of landsat_c2l2.sensor_keys. A source-only search reported it as
    # unreferenced, which is the opposite of the truth.
    references = reporting.dataset_references(params)
    assert references["landsat5"], "landsat5 reported as unreferenced"
    assert any("sensor_keys" in hit for hit in references["landsat5"])
    assert any("products" in hit for hit in references["modis_aqua_lst"])


def test_references_do_not_count_a_key_definition_as_a_reference(
    params: dict[str, Any],
) -> None:
    # Every key is defined under `datasets`, so including that block would make
    # every dataset trivially "referenced" and the column worthless.
    references = reporting.dataset_references(params)
    for hits in references.values():
        assert not any(hit.startswith("params.yaml:datasets") for hit in hits)


def test_reporting_is_excluded_from_its_own_reference_scan(
    params: dict[str, Any],
) -> None:
    # reporting.py names dataset keys in _scale_factor and in its docstrings.
    # Counting those would list it as a consumer of srtm and surface_water.
    references = reporting.dataset_references(params)
    for hits in references.values():
        assert "reporting.py" not in hits


def test_unused_datasets_are_reported_as_unused(params: dict[str, Any]) -> None:
    # This is a finding, not a bug: CLAUDE.md names era5_land for reanalysis
    # air-temperature validation and that comparison was never carried out. The
    # table has to say so rather than implying every configured source was used.
    references = reporting.dataset_references(params)
    unused = {key for key, hits in references.items() if not hits}
    assert "era5_land" in unused
    frame = reporting.provenance_frame(params).set_index("key")
    assert frame.loc["era5_land", "used_by"] == "not referenced"


# --- rendering helpers -------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0000275, "0.0000275"), (0.00341802, "0.00341802"), (149.0, "149.0"),
     (0.02, "0.02"), (0.0, "0.0")],
)
def test_plain_never_uses_scientific_notation(value: float, expected: str) -> None:
    # "2.75e-05" is not how the Landsat scale factor is written in any USGS
    # document a reader will check the methods section against.
    assert reporting._plain(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"), [(149.0, "+ 149.0"), (-0.2, "- 0.2"), (0.0, "+ 0.0")]
)
def test_signed_offset_never_renders_plus_minus(value: float, expected: str) -> None:
    assert reporting._signed_offset(value) == expected


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("lst_not_air_temp", "LST not air temp"),
        ("fdr_dependence", "FDR dependence"),
        ("mcda_weights_are_judgements", "MCDA weights are judgements"),
        ("single_overpass", "Single overpass"),
    ],
)
def test_titleise_keeps_acronyms_uppercase(key: str, expected: str) -> None:
    assert reporting._titleise(key) == expected


def test_sensor_offset_table_keeps_the_pairs_that_could_not_be_tested() -> None:
    # Three of the six Landsat pairs have too little overlap to test. Dropping
    # them would imply the archive had been checked end to end.
    rows = reporting._sensor_offset_lines()
    assert len(rows) == 6
    assert sum("insufficient_overlap" in row for row in rows) == 3
    # An untestable pair carries dashes, not a crash and not a fabricated zero.
    untestable = [row for row in rows if "insufficient_overlap" in row]
    assert all("| - |" in row or "| - " in row for row in untestable)


# --- the generated prose -----------------------------------------------------
def test_methods_quotes_every_measured_headline(params: dict[str, Any]) -> None:
    text = reporting.methods_markdown(params)
    # Parameters
    assert str(params["prediction"]["rf"]["n_trees"]) in text
    assert str(params["trends"]["fdr"]["alpha"]) in text
    assert str(params["greening"]["top_n"]) in text
    assert str(params["report"]["dpi"]) in text
    # Measurements read from data/outputs, not typed
    assert "1.13" in text and "0.894" in text, "RF held-out metrics absent"
    assert "+1.78" in text and "-2.48" in text, "sensor offsets absent"
    assert f"{params['greening']['ahp']['derived_consistency']['consistency_ratio']:.4f}" in text


def test_limitations_reproduces_every_caveat_verbatim(params: dict[str, Any]) -> None:
    text = reporting.limitations_markdown(params)
    for key, caveat in params["caveats"].items():
        assert f"caveats.{key}" in text, f"{key} is not named"
        # Whitespace is re-flowed; the words are not.
        assert " ".join(str(caveat).split()) in text, f"{key} is not verbatim"


def test_limitations_names_the_missing_air_temperature_comparison(
    params: dict[str, Any],
) -> None:
    text = reporting.limitations_markdown(params)
    assert "era5_land" in text
    assert "air-temperature" in text


def test_generated_documents_end_in_exactly_one_newline(
    params: dict[str, Any],
) -> None:
    for text in (
        reporting.methods_markdown(params),
        reporting.limitations_markdown(params),
    ):
        assert text.endswith("\n") and not text.endswith("\n\n")


# --- the drift test ----------------------------------------------------------
@pytest.mark.parametrize(
    ("relative", "generator"),
    [
        (reporting.METHODS_PATH, "methods_markdown"),
        (reporting.LIMITATIONS_PATH, "limitations_markdown"),
    ],
)
def test_committed_document_matches_what_the_code_generates_now(
    params: dict[str, Any], relative: Path, generator: str
) -> None:
    # THE point of this module. If this fails, something in params.yaml or in a
    # committed measurement changed and the write-up was not regenerated:
    #
    #     python -c "from colombo_uhi import load_params, reporting; \\
    #                reporting.write_docs(load_params())"
    path = repo_root() / relative
    assert path.is_file(), (
        f"{relative} is missing. Generate it with reporting.write_docs(params)."
    )
    committed = path.read_text(encoding="utf-8")
    expected = getattr(reporting, generator)(params)
    assert committed == expected, (
        f"{relative} is stale. Re-run reporting.write_docs(params) and commit "
        "the result; do not edit the file by hand."
    )


def test_committed_provenance_csv_matches_what_the_code_generates_now(
    params: dict[str, Any],
) -> None:
    import pandas as pd

    path = repo_root() / reporting.PROVENANCE_PATH
    assert path.is_file(), "data/outputs/data_provenance.csv is missing"
    committed = pd.read_csv(path, keep_default_na=False)
    expected = reporting.provenance_frame(params).fillna("").astype(str)
    pd.testing.assert_frame_equal(
        committed.astype(str).reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )


# --- the validation report carried inside a raster ---------------------------
def test_scenario_report_reads_the_metrics_stamped_in_the_geotiff(
    params: dict[str, Any],
) -> None:
    rasterio = pytest.importorskip("rasterio")
    path = repo_root() / "data" / "outputs" / "greening_counterfactual_delta.tif"
    if not path.is_file():  # pragma: no cover - product not yet exported
        pytest.skip("the greening counterfactual raster is not committed here")

    import ast

    report = reporting.scenario_report_from_raster(path, params)
    assert report["kind"] == "lst_scenario"
    with rasterio.open(str(path)) as handle:
        stamped = ast.literal_eval(handle.tags()["validation"])
    # The report carries exactly the metrics the guard stamped, unaltered.
    assert set(report["metrics"]) == set(stamped)
    for metric, value in stamped.items():
        assert report["metrics"][metric] == pytest.approx(float(value))
    # The report must satisfy the guard that would let the product be written.
    from colombo_uhi import prediction

    assert prediction.assess_validation(report, params)["valid"]


def test_scenario_report_refuses_a_raster_with_no_validation_tag(
    params: dict[str, Any], tmp_path: Path
) -> None:
    # A surface that reached disk without going through the guard has no
    # provenance for its own accuracy and must not be plotted as a prediction.
    rasterio = pytest.importorskip("rasterio")
    numpy = pytest.importorskip("numpy")

    path = tmp_path / "untagged.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32"
    ) as handle:
        handle.write(numpy.zeros((4, 4), dtype="float32"), 1)

    with pytest.raises(ValueError, match="no 'validation' tag"):
        reporting.scenario_report_from_raster(path, params)
