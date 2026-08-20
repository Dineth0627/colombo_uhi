"""Report artefacts derived from ``config/params.yaml``: provenance and prose.

Phase 8. Everything here is **generated**, never authored:

* :func:`provenance_frame` walks ``datasets`` and ``non_ee_sources`` and reports
  every collection ID, band, scale factor and date range the project actually
  configures - together with the modules that reference it, found by searching
  the source rather than by a hand-kept list.
* :func:`methods_markdown` and :func:`limitations_markdown` write the methods
  and limitations sections with **every number interpolated from ``params``**.

The point of generating the prose is stated in ``caveats.figures_are_derived_not_authored``
and enforced by ``tests/test_reporting.py``: the committed ``docs/methods.md``
and ``docs/limitations.md`` must equal what these functions produce right now, so
a threshold cannot be edited in ``params.yaml`` and left stale in the write-up.

Design notes:
    * ``pandas`` is imported inside function bodies, matching the rest of the
      package, so importing this module costs nothing.
    * Nothing here touches Earth Engine.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from colombo_uhi import repo_root

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

#: Files the generated documents are written to, relative to the repo root.
METHODS_PATH = Path("docs") / "methods.md"
LIMITATIONS_PATH = Path("docs") / "limitations.md"
PROVENANCE_PATH = Path("data") / "outputs" / "data_provenance.csv"

#: Column order of :func:`provenance_frame`.
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "key",
    "collection_id",
    "type",
    "bands",
    "native_scale_m",
    "scale_factor",
    "temporal_coverage",
    "used_by",
    "note",
)

#: Keys under a ``datasets`` entry that name a band.
_BAND_KEYS: tuple[str, ...] = (
    "st_band", "band", "band_raw", "day_band", "night_band", "qc_day_band",
    "qc_night_band", "label_band", "ndvi_band", "band_occurrence",
    "band_max_extent", "band_seasonality", "band_nres", "band_obs_count",
    "band_skin_temp", "band_t2m",
)


#: Words the generated prose must not sentence-case into nonsense.
_ACRONYMS: dict[str, str] = {
    "lst": "LST", "utfvi": "UTFVI", "fdr": "FDR", "mcda": "MCDA",
    "ahp": "AHP", "suhii": "SUHII", "maup": "MAUP", "gn": "GN", "ds": "DS",
    "jrc": "JRC", "modis": "MODIS", "qc": "QC", "cmc": "CMC", "aoi": "AOI",
    "ndvi": "NDVI", "ndbi": "NDBI", "lcz": "LCZ", "rmse": "RMSE",
}


def _titleise(key: str) -> str:
    """Turn a snake_case config key into a readable heading.

    ``lst_not_air_temp`` becomes ``LST not air temp``, not ``Lst not air
    temp``: the caveat keys are dense with acronyms, and sentence-casing them
    blindly produces headings that read as typos.

    Args:
        key: A snake_case identifier.

    Returns:
        The heading text.
    """
    words = [_ACRONYMS.get(word, word) for word in str(key).split("_")]
    if words and words[0] not in _ACRONYMS.values():
        words[0] = words[0].capitalize()
    return " ".join(words)


def _plain(value: float) -> str:
    """Format a float without scientific notation, trailing zeros trimmed.

    ``0.0000275`` renders as ``2.75e-05`` under the default float repr, which
    is not how the Landsat scale factor is written in any documentation a
    reader will check this against.

    Args:
        value: The number.

    Returns:
        A decimal string.
    """
    text = f"{float(value):.12f}".rstrip("0")
    # Keep one decimal on a whole number: the Landsat offset is published as
    # "149.0", and a reader checking this against the USGS documentation should
    # find the same string, not "149".
    return f"{text}0" if text.endswith(".") else (text or "0")


def _signed_offset(value: float) -> str:
    """Render an additive offset as ``+ x`` or ``- x``, never ``+ -x``."""
    number = float(value)
    return f"- {_plain(abs(number))}" if number < 0 else f"+ {_plain(number)}"


def _band_names(entry: Mapping[str, Any]) -> list[str]:
    """Every band a ``datasets`` entry names, in a stable order."""
    names: list[str] = []
    for key in _BAND_KEYS:
        value = entry.get(key)
        if isinstance(value, str) and value not in names:
            names.append(value)
    for group in ("sr_bands", "emissivity_bands"):
        member = entry.get(group)
        if isinstance(member, Mapping):
            names.extend(str(v) for v in member.values() if str(v) not in names)
        elif isinstance(member, (list, tuple)):
            names.extend(str(v) for v in member if str(v) not in names)
    return names


def _scale_factor(key: str, entry: Mapping[str, Any], params: Mapping[str, Any]) -> str:
    """Human-readable scale factor, taken from the sensor-constant sections.

    Scale factors deliberately do NOT live in ``datasets``: they are sensor
    constants shared across four Landsat collections and two MODIS ones, and
    duplicating them per entry is how two copies drift apart. This resolves them
    from ``landsat_c2l2`` and ``modis_lst`` instead.
    """
    if key.startswith("landsat"):
        c2 = params["landsat_c2l2"]
        return (
            f"ST: DN x {_plain(c2['st_scale'])} "
            f"{_signed_offset(c2['st_offset'])} = K; "
            f"SR: DN x {_plain(c2['sr_scale'])} "
            f"{_signed_offset(c2['sr_offset'])}"
        )
    if key.startswith("modis") and "lst" in key:
        return f"LST: DN x {_plain(params['modis_lst']['lst_scale'])} = K"
    if "emissivity_scale" in entry:
        return f"DN x {entry['emissivity_scale']}"
    if key == "surface_water":
        return "percent occurrence, 0-100 (no scaling)"
    return "none (native units)"


def _coverage(entry: Mapping[str, Any]) -> str:
    """Temporal coverage as a readable string; ``null`` end means ongoing."""
    window = entry.get("availability")
    if isinstance(window, (list, tuple)) and len(window) == 2:
        start, end = window
        return f"{start} to {end or 'ongoing'}"
    for key in ("nominal_year", "nominal_period"):
        if key in entry:
            return str(entry[key])
    return "not stated in params.yaml"


def _source_text(paths: Iterable[Path]) -> dict[str, str]:
    """Read each path once; missing files are simply absent from the result."""
    text: dict[str, str] = {}
    for path in paths:
        try:
            text[path.name] = path.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - unreadable file
            continue
    return text


def _value_paths(node: Any, wanted: str, prefix: str = "") -> list[str]:
    """Dotted paths at which ``wanted`` appears as a scalar VALUE in a mapping.

    Args:
        node: Mapping, sequence or scalar to walk.
        wanted: The string to look for.
        prefix: Dotted path of ``node`` itself.

    Returns:
        Dotted paths of the entries holding that value.
    """
    hits: list[str] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            hits.extend(_value_paths(value, wanted, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, (list, tuple)):
        for value in node:
            hits.extend(_value_paths(value, wanted, prefix))
    elif isinstance(node, str) and node == wanted:
        hits.append(prefix)
    return hits


def dataset_references(
    params: Mapping[str, Any], root: Path | None = None
) -> dict[str, list[str]]:
    """Where each ``datasets`` key is reached from - modules AND config paths.

    Found by SEARCHING, not from a hand-kept table: a hand-kept table is exactly
    the thing that goes stale, and this column exists to tell a reader where a
    dataset is actually used.

    Two searches, because the project reaches a dataset two different ways.
    Some modules name the key outright (``params["datasets"]["srtm"]``). Others
    reach it INDIRECTLY through a list in the config - ``landsat.resolve_sensors``
    reads ``landsat_c2l2.sensor_keys`` and ``modis.resolve_product`` reads
    ``modis_lst.products``, so ``landsat5`` appears nowhere in the source at all.
    A source-only search would report those four Landsat collections and both
    MODIS ones as unreferenced, which is the opposite of the truth. So config
    paths at which the key appears as a VALUE are reported too, and the
    ``datasets`` block itself is excluded so a key's own definition never counts
    as a reference to it.

    Args:
        params: Parsed params mapping.
        root: Repository root; defaults to :func:`colombo_uhi.repo_root`.

    Returns:
        Mapping of dataset key to sorted references: module filenames first,
        then ``params.yaml`` paths. Empty when nothing reaches it.
    """
    base = Path(root) if root is not None else repo_root()
    # This module is excluded from its own scan. It names dataset keys in
    # _scale_factor and in docstrings, so including it would list reporting.py
    # as a consumer of srtm and surface_water - which it is not; it is the thing
    # doing the reporting.
    modules = _source_text(
        path for path in sorted((base / "src" / "colombo_uhi").glob("*.py"))
        if path.name != Path(__file__).name
    )

    elsewhere = {key: value for key, value in params.items() if key != "datasets"}
    found: dict[str, list[str]] = {}
    for key in params["datasets"]:
        hits = sorted(
            name for name, text in modules.items()
            if f'"{key}"' in text or f"'{key}'" in text
        )
        hits += [f"params.yaml:{path}" for path in sorted(set(_value_paths(elsewhere, key)))]
        found[key] = hits
    return found


def provenance_frame(params: dict[str, Any], root: Path | None = None) -> "pd.DataFrame":
    """Every data source the project configures, with its bands and constants.

    Covers ``datasets`` (Earth Engine collections), ``non_ee_sources`` (things
    CLAUDE.md names that are NOT in the catalog), and ``aoi.assets`` (the
    GN-division polygons, which no public dataset provides and which the project
    owner must upload). A source the project uses but does not configure cannot
    appear here - which is the point: the table is a statement about
    ``params.yaml``, and ``params.yaml`` is the single source of truth.

    Args:
        params: Parsed params mapping.
        root: Repository root, for the reference search.

    Returns:
        One row per source, columns :data:`PROVENANCE_COLUMNS`.
    """
    import pandas as pd

    references = dataset_references(params, root=root)
    rows: list[dict[str, Any]] = []

    for key, entry in params["datasets"].items():
        rows.append({
            "key": key,
            "collection_id": entry.get("id", ""),
            "type": entry.get("type", ""),
            "bands": ", ".join(_band_names(entry)) or "(all)",
            "native_scale_m": entry.get("scale_m", ""),
            "scale_factor": _scale_factor(key, entry, params),
            "temporal_coverage": _coverage(entry),
            "used_by": ", ".join(references.get(key, [])) or "not referenced",
            "note": " ".join(str(entry.get("note", "")).split()),
        })

    for key, entry in (params.get("non_ee_sources") or {}).items():
        rows.append({
            "key": key,
            "collection_id": "NOT IN THE EARTH ENGINE CATALOG",
            "type": "external",
            "bands": "",
            "native_scale_m": "",
            "scale_factor": "",
            "temporal_coverage": "",
            "used_by": "not used - see note",
            "note": (
                f"{entry.get('station_name', key)}"
                + (f" (WMO {entry['wmo_station_id']})" if "wmo_station_id" in entry else "")
                + f". Status: {entry.get('status', 'unresolved')}."
            ),
        })

    for name, path in (params["aoi"].get("assets") or {}).items():
        # `aoi.assets` also holds property-NAME candidates and literal filter
        # values alongside the two asset paths. Only the paths belong in a
        # provenance table; the rest are field names, not data sources.
        if not (isinstance(path, str) and "/assets/" in path):
            continue
        rows.append({
            "key": f"asset:{name}",
            "collection_id": str(path),
            "type": "user-uploaded Earth Engine asset",
            "bands": "",
            "native_scale_m": "",
            "scale_factor": "",
            "temporal_coverage": "",
            "used_by": "aoi.py",
            "note": (
                "GN-division polygons are not in GAUL, GADM or any other public "
                "Earth Engine dataset and must be uploaded by the project owner."
            ),
        })

    return pd.DataFrame(rows, columns=list(PROVENANCE_COLUMNS))


def scenario_report_from_raster(
    path: str | Path, params: dict[str, Any], kind: str = "lst_scenario"
) -> dict[str, Any]:
    """Rebuild a validation report from the metrics stamped inside a GeoTIFF.

    ``prediction.write_surface`` refuses to write an unvalidated surface and
    stamps the metrics that let it through into the file's own tags. Phase 8
    therefore does not have to trust a side-car: the report travels with the
    raster, and a raster that somehow reached disk without one cannot be plotted.

    This exists because ``data/outputs/validation_reports.json`` - written by
    notebook 06 and described there as holding *every* validation report - does
    not in fact contain the ``lst_scenario`` entry belonging to the greening
    counterfactual, the only validated raster product Phase 6 exported. The tag
    is the authoritative copy until notebook 06 is re-run.

    Args:
        path: Path to a GeoTIFF written by ``prediction.write_surface``.
        params: Parsed params mapping.
        kind: Product kind, keying ``prediction.REQUIRED_METRICS``.

    Returns:
        A validation report, as :func:`colombo_uhi.prediction.build_validation_report`
        returns.

    Raises:
        ValueError: If the file carries no ``validation`` tag, or the tag cannot
            be parsed as a metrics mapping.
    """
    import rasterio

    from colombo_uhi import prediction

    with rasterio.open(str(path)) as handle:
        tags = handle.tags()

    if "validation" not in tags:
        raise ValueError(
            f"{path} carries no 'validation' tag, so its metrics are unknown. "
            "Only surfaces written by prediction.write_surface have one; a file "
            "without it has not been through the validation guard and must not "
            "be plotted as a predictive product."
        )
    try:
        metrics = ast.literal_eval(str(tags["validation"]))
    except (ValueError, SyntaxError) as error:
        raise ValueError(
            f"{path}'s validation tag is not a readable mapping: "
            f"{tags['validation']!r}"
        ) from error
    if not isinstance(metrics, Mapping):
        raise ValueError(
            f"{path}'s validation tag parsed to {type(metrics).__name__}, "
            "expected a mapping of metric name to value"
        )

    return prediction.build_validation_report(
        kind,
        {str(name): float(value) for name, value in metrics.items()},
        params,
        held_out=True,
        notes=[
            f"Metrics read from the validation tag stamped inside {Path(path).name} "
            "by prediction.write_surface, which refuses to write an unvalidated "
            "surface."
        ],
    )


# =============================================================================
# Generated prose: docs/methods.md and docs/limitations.md
# =============================================================================
#: Header stamped at the top of both generated documents.
_GENERATED_BANNER = (
    "> **This file is generated.** Every parameter below is read from\n"
    "> `config/params.yaml` and from the committed products in `data/outputs/`\n"
    "> by `colombo_uhi.reporting`. Do not edit it by hand - edit the config, then\n"
    "> re-run `reporting.write_docs(params)`. `tests/test_reporting.py` fails if\n"
    "> the committed copy disagrees with what the code produces, which is what\n"
    "> stops a threshold being changed in one place and left stale in another.\n"
)


def _months(names: Sequence[int]) -> str:
    """Month numbers as abbreviated names, e.g. ``[1, 2, 3]`` -> ``Jan-Mar``."""
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    picked = [labels[int(month) - 1] for month in names]
    if len(picked) > 2 and list(names) == list(range(min(names), max(names) + 1)):
        return f"{picked[0]}-{picked[-1]}"
    return ", ".join(picked)


def _read_output(name: str, root: Path | None = None) -> "pd.DataFrame | None":
    """Read a committed product from ``data/outputs``; ``None`` if absent.

    The generated prose quotes MEASUREMENTS as well as parameters, and those
    live in ``data/outputs``. Returning ``None`` rather than raising lets the
    documents generate on a fresh clone before any Colab run, with an explicit
    "not yet measured" in place of a number - which is honest, and better than
    refusing to produce the document at all.
    """
    import pandas as pd

    base = Path(root) if root is not None else repo_root()
    path = base / "data" / "outputs" / name
    if not path.is_file():
        return None
    return pd.read_csv(path)


def _sensor_offset_lines(root: Path | None = None) -> list[str]:
    """The measured Landsat inter-sensor offsets, as markdown table rows.

    Pairs with too few overlapping years carry no statistic at all, and those
    rows are rendered with dashes rather than dropped. Their absence is part of
    the result: L7-L9 overlap by two dry seasons and L5 never overlaps L8 or L9,
    so three of the six possible pairs could never have been tested, and a table
    that quietly showed only the three testable ones would imply the archive had
    been checked end to end.
    """
    import pandas as pd

    frame = _read_output("sensor_offsets_cmc.csv", root=root)
    if frame is None or frame.empty:
        return ["| _not yet measured - run notebook 04 step 6_ | | | | | |"]

    def _number(value: Any, spec: str) -> str:
        return "-" if pd.isna(value) else format(float(value), spec)

    def _window(row: Mapping[str, Any]) -> str:
        if pd.isna(row["first_year"]) or pd.isna(row["last_year"]):
            return "-"
        return f"{int(row['first_year'])}-{int(row['last_year'])}"

    # Measured pairs first, in chronological order; untestable pairs after.
    ordered = frame.sort_values(
        ["first_year", "sensor_a"], na_position="last", kind="stable"
    )
    return [
        f"| {row['sensor_a']} - {row['sensor_b']} "
        f"| {_number(row['mean_offset'], '+.2f')} "
        f"| {_number(row['t_statistic'], '+.2f')} "
        f"| {int(row['n_overlap_years'])} "
        f"| {_window(row)} "
        f"| **{row['verdict']}** |"
        for _, row in ordered.iterrows()
    ]


def _validation_metrics(root: Path | None = None) -> dict[str, Any]:
    """The committed validation reports, or an empty mapping."""
    import json

    base = Path(root) if root is not None else repo_root()
    path = base / "data" / "outputs" / "validation_reports.json"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as handle:
        return dict(json.load(handle))


def methods_markdown(params: dict[str, Any], root: Path | None = None) -> str:
    """Generate the methods section, with every parameter read from ``params``.

    Prose a report author can adapt, not a config dump: each subsection states
    what was done and why, and the numbers inside it are interpolated rather
    than typed, so the document cannot drift from the code that produced the
    results.

    Args:
        params: Parsed params mapping.
        root: Repository root, for reading committed measurements.

    Returns:
        The complete markdown document, ending in a newline.
    """
    time_cfg = params["time"]
    crs = params["crs"]
    aoi = params["aoi"]
    c2 = params["landsat_c2l2"]
    modis = params["modis_lst"]
    suhii = params["uhi"]["suhii"]
    utfvi = params["uhi"]["utfvi"]
    trends_cfg = params["trends"]
    stats = params["spatial_stats"]
    pred = params["prediction"]
    green = params["greening"]

    dry = time_cfg["seasons"]["dry_window"]
    reports = _validation_metrics(root)
    fit = reports.get("lst_fit", {}).get("metrics", {})
    lulc = reports.get("lulc_projection", {}).get("metrics", {})
    ahp_derived = green["ahp"]["derived_consistency"]
    weights = green["ahp"]["derived_weights_reference"]
    unused = [key for key, hits in dataset_references(params, root=root).items() if not hits]

    lines: list[str] = [
        "# Methods",
        "",
        _GENERATED_BANNER,
        "",
        "## 1. Study area and analysis frame",
        "",
        f"The study area is Colombo, Sri Lanka, centred near "
        f"{aoi['centre']['lat']}° N, {aoi['centre']['lon']}° E, in the Köppen "
        f"**{aoi.get('koppen', 'Af')}** tropical zone. Two nested units are used "
        "throughout. The **Colombo Municipal Council (CMC)** is the urban core, "
        f"defined as the union of the {len(aoi['cmc']['gn_division_names'])} Grama "
        "Niladhari (GN) divisions the CMC's own GIS Unit lists as inside the "
        "municipality. The **Colombo District** is the wider analysis frame: "
        f"{aoi['district']['ds_divisions']} Divisional Secretariat divisions "
        f"and {aoi['district']['gn_divisions']} GN divisions.",
        "",
        "The CMC's area is **scale-dependent and must always be quoted with its "
        "reduction scale**. The gazetted figure is 37.31 km². The COD-AB polygon "
        f"measures about {aoi['expected_areas_km2']['cmc_administrative']} km² "
        "because it encloses the Colombo Port outer harbour; once water is "
        f"masked the land area is "
        f"**about {aoi['expected_areas_km2']['cmc_land_at_30m']} km² at "
        f"{crs['analysis_scale_m']} m**, and lower again at coarser reduction "
        "scales. The residual over the gazetted figure is polygon "
        "generalisation plus "
        "the water-mask thresholds, and is reported as a sensitivity rather than "
        "tuned away.",
        "",
        f"All analysis is carried out in **{crs['analysis_epsg']}** (UTM 44N) "
        "on a "
        f"**{crs['analysis_scale_m']} m** grid. GN names are **not unique within "
        "the district**, so every GN-name filter is scoped to its parent DS "
        "division or keyed on `adm4_pcode`.",
        "",
        "GN-division polygons are in no public Earth Engine dataset. They are "
        "uploaded by the project owner as an Earth Engine asset whose path is "
        "configured in `aoi.assets`; the code fails with an actionable message if "
        "the asset is absent.",
        "",
        "## 2. Data",
        "",
        f"Every source is free and public. The study period is "
        f"**{time_cfg['start_year']}-{time_cfg['end_year']}**. The complete "
        "inventory - collection ID, bands, native resolution, scale factors, "
        "temporal coverage and the module that reads each one - is in "
        "`data/outputs/data_provenance.csv` and in figure 11; it is generated "
        "from `config/params.yaml`, so it cannot disagree with the code.",
        "",
        f"The primary comparison window is the **dry season, "
        f"{_months(dry['months'])}**, the driest and least cloudy part of the "
        "year. Monsoon seasons are defined as "
        + "; ".join(
            f"{time_cfg['seasons'][key]['label']}"
            for key in time_cfg["season_partition"]
        )
        + ", which partition the twelve months exactly once each.",
        "",
    ]

    if unused:
        lines += [
            f"**{len(unused)} configured datasets are not referenced by any "
            "analysis step**: " + ", ".join(f"`{key}`" for key in unused) + ". They are "
            "retained in the configuration because `CLAUDE.md` names them, and "
            "they appear in the provenance table marked as unreferenced. The "
            "consequence for `era5_land` in particular is stated in "
            "`docs/limitations.md`: no reanalysis air-temperature comparison was "
            "carried out.",
            "",
        ]

    lines += [
        "## 3. Land surface temperature",
        "",
        "### 3.1 Landsat Collection 2 Level-2",
        "",
        "Surface temperature comes from the Collection 2 Level-2 science "
        "products of Landsat 5 TM, 7 ETM+, 8 OLI/TIRS and 9 OLI-2/TIRS-2. "
        f"Digital numbers are converted with the published constants: "
        f"`ST x {_plain(c2['st_scale'])} {_signed_offset(c2['st_offset'])}` "
        "gives kelvin, from which "
        f"{c2['kelvin_to_celsius_offset']} is subtracted for degrees Celsius; "
        f"surface reflectance is `DN x {_plain(c2['sr_scale'])} "
        f"{_signed_offset(c2['sr_offset'])}`. "
        f"Valid ST digital numbers run {c2['st_valid_dn_range'][0]}-"
        f"{c2['st_valid_dn_range'][1]}, with {c2['st_fill_dn']} as fill.",
        "",
        "Cloud masking uses `QA_PIXEL` bits "
        + ", ".join(str(bit) for bit in c2["standard_mask"]["require_zero_bits"])
        + " (fill, dilated cloud, cirrus, cloud, cloud shadow), all required to "
        f"be zero, together with `{c2['qa_radsat_band']} == "
        f"{c2['standard_mask']['require_qa_radsat_value']}`. Reflectance bands "
        "are renamed to `blue`...`swir2` so that index code is sensor-agnostic "
        "across the TM/ETM+ and OLI band-numbering change.",
        "",
        "### 3.2 MODIS",
        "",
        f"MOD11A2 (Terra) and MYD11A2 (Aqua) 8-day 1 km LST are scaled by "
        f"**{modis['lst_scale']}** to kelvin. These products are a plain average "
        "of the daily product with **no built-in quality filtering**, so "
        "`QC_Day` and `QC_Night` are decoded explicitly, and the day and "
        "night policies differ because the daytime field is far more heavily "
        "contaminated: day requires mandatory QA <= "
        f"{modis['qc_filter']['day']['mandatory_qa_max']} and LST-error class "
        f"<= {modis['qc_filter']['day']['lst_error_max']}, night requires <= "
        f"{modis['qc_filter']['night']['mandatory_qa_max']} and <= "
        f"{modis['qc_filter']['night']['lst_error_max']}. Overpass times are "
        + ", ".join(
            f"{name.replace('_', ' ')} {value}"
            for name, value in modis["overpass_local_time"].items()
        )
        + " local.",
        "",
        "### 3.3 Compositing and the observation count",
        "",
        f"Annual and dry-season composites reduce Landsat with the "
        f"**{params['composites']['annual_reducer']}** and MODIS with the "
        f"**{params['composites']['modis_reducer']}**, and every composite "
        "carries a "
        f"per-pixel `{params['composites']['obs_count_band']}` band. That band is "
        "not a diagnostic: under tropical cloud only a minority of scenes are "
        "usable, so no composite or trend product may be read without it. It is "
        "figure 3, and it is what the untested grey on the trend map means.",
        "",
        "## 4. Cross-sensor continuity - a measured failure",
        "",
        "Collection 2 is inter-calibrated across TM, ETM+ and OLI, so the "
        "documentation implies no manual harmonisation is needed. **This was "
        "tested over Colombo and it failed.** Dry-season CMC means at 100 m give:",
        "",
        "| pair | mean offset (°C) | t | overlap years | window | verdict |",
        "|---|---|---|---|---|---|",
        *_sensor_offset_lines(root),
        "",
        "The first two offsets are several times the entire 26-year trend signal, "
        "and the L7-to-L8 step alone predicts the observed decadal jump. **No "
        "multi-year trend is fitted across a Landsat changeover.** The trend "
        f"products use `{trends_cfg['pixel_sources'][0]}` - a "
        "single-sensor-family series - accepting reduced statistical power "
        "in exchange for a defensible slope. Offsets are deliberately **not** "
        "estimated and subtracted: eight to ten noisy overlap years would inject "
        "a new error rather than remove one.",
        "",
        "SUHII is unaffected, because it is a within-year urban-minus-rural "
        "difference in which a spatially common-mode step cancels.",
        "",
        "## 5. Surface urban heat island intensity (SUHII)",
        "",
        "SUHII is the mean urban LST minus the mean rural LST, computed under "
        f"**{len(suhii['rural_definitions'])} independent rural definitions** and "
        "reported as a sensitivity rather than a single number:",
        "",
        f"1. **Buffer ring** - an annulus from {suhii['buffer_ring']['inner_km']} "
        f"to {suhii['buffer_ring']['outer_km']} km beyond the "
        f"{suhii['buffer_ring']['base'].upper()} boundary, excluding "
        + " and ".join(
            name.replace("_", "-") for name in suhii["buffer_ring"]["exclude"]
        )
        + " pixels.",
        f"2. **LCZ-based** - Local Climate Zone classes "
        f"{suhii['lcz_based']['urban_classes'][0]}-"
        f"{suhii['lcz_based']['urban_classes'][-1]} as urban and "
        f"{suhii['lcz_based']['rural_classes'][0]}-"
        f"{suhii['lcz_based']['rural_classes'][-1]} as rural, both clipped to the "
        f"{suhii['lcz_based']['scope']}.",
        "",
        "Both rural masks are additionally capped at "
        f"**{suhii['rural_filters']['max_elevation_m']} m** elevation: the ring "
        "reaches inland relief, and at a tropical lapse rate that is up to "
        "several tenths of a degree of elevation-driven cooling that would "
        "otherwise be counted as urban heat island.",
        "",
        "The series is computed for "
        + ", ".join(f"`{entry['key']}`" for entry in suhii["sources"])
        + f", each reduced at its own native scale, and Landsat stays on the "
        "median while MODIS stays on the mean so that no part of a "
        "Landsat-versus-MODIS difference is a reducer artefact.",
        "",
        "## 6. Urban thermal field variance index (UTFVI)",
        "",
        f"UTFVI is `{utfvi['definition']}`, classified on the published breaks "
        + ", ".join(str(value) for value in utfvi["breaks"])
        + " into the six classes "
        + ", ".join(utfvi["labels"])
        + f". The index is computed in **{utfvi['units']}**, which is "
        "load-bearing rather than documentation: UTFVI is a ratio, so on kelvin "
        "every break would mean roughly a tenth of what it means on Celsius.",
        "",
        f"`Tmean` is the **{_titleise(utfvi['reference']).lower().replace('aoi', 'AOI')}** "
        "- each "
        "year's own "
        "spatial mean over the AOI. That is the standard formulation and keeps "
        "the published breaks meaningful, but it has a consequence that must "
        "travel with every UTFVI output: **the index measures within-year spatial "
        "structure only.** A city that warms uniformly by 2 °C shows no class "
        "change at all, so epoch-to-epoch class drift is a redistribution of "
        "heat and never evidence of warming.",
        "",
        "Epochs are "
        + ", ".join(
            f"{key} ({value[0]}-{value[1]})" for key, value in utfvi["epochs"].items()
        )
        + "; the last is short by design because the study period ends in "
        f"{time_cfg['end_year']}.",
        "",
        "## 7. Trend analysis",
        "",
        f"Trends are fitted pixel-wise on the **annual composite series** at "
        f"{trends_cfg['fit_scale_m']} m, using the non-parametric Mann-Kendall "
        f"test (`{trends_cfg['mk_reducer']}`) and Sen's slope "
        f"(`{trends_cfg['sen_reducer']}`), reported in "
        "degrees Celsius per year. A pixel is fitted only where it has at "
        f"least **{trends_cfg['min_years']} valid years**, each resting on at "
        f"least **{trends_cfg['min_valid_obs']} valid observations**; pixels "
        "below either floor are neither significant nor non-significant, and are "
        "drawn as untested grey rather than as zero trend.",
        "",
        "Significance is corrected for multiple testing **in Python on the "
        "exported p-value raster**, because the procedure needs every p-value at "
        "once and cannot be done server-side. The headline correction is "
        f"**{_titleise(trends_cfg['fdr']['method']).replace(' ', '-')}** at "
        f"alpha = {trends_cfg['fdr']['alpha']}. Benjamini-Hochberg controls the "
        "false discovery rate under independence or positive regression "
        "dependency, and a 100 m LST raster is strongly spatially autocorrelated, "
        "so **Benjamini-Yekutieli is reported beside it** as the bound valid "
        "under arbitrary dependence. Both denominators are reported: the tested "
        "set and the total, because untested pixels belong to neither.",
        "",
        f"The autocorrelation-corrected Mann-Kendall "
        f"(`{trends_cfg['mmk']['method']}`) is run alongside the plain test on "
        "the aggregate series, and the pair is reported: serial correlation "
        "inflates the variance of S, and the size of that inflation is itself a "
        "result.",
        "",
        "Decadal windows are "
        + ", ".join(
            f"{label.replace('_', '-')}" for label in trends_cfg["decades"]
        )
        + f", weighted `{trends_cfg['decade_weighting']}` so every year counts "
        "once. **The windows are unequal by construction** (11 / 10 / "
        f"{time_cfg['end_year'] - 2021 + 1} years), so any difference involving "
        "the last one rests on about half the sample and carries roughly √2 the "
        "standard error - which is why the decadal product emits a standard "
        "error and a z band rather than a difference alone.",
        "",
        "## 8. Spatial statistics",
        "",
        "Every spatial statistic is computed at **both** the GN "
        f"({aoi['district']['gn_divisions']} units) and DS "
        f"({aoi['district']['ds_divisions']} units) levels. What survives the "
        "coarsening is itself the reported result: this is the modifiable areal "
        "unit problem, and reporting one level only would hide it.",
        "",
        f"Spatial weights are **{stats['weights']['scheme']} contiguity**, "
        f"{stats['weights']['transform']}-standardised, with islands attached by "
        f"`{stats['weights']['island_policy']}`. Global Moran's I, Local Moran's "
        "I (LISA) and Getis-Ord Gi* are computed with "
        f"**{stats['permutations']} conditional permutations** at seed "
        f"{stats['random_seed']}, and local p-values carry a "
        f"{_titleise(stats['lisa']['fdr_method']).replace(' ', '-')} correction "
        "at "
        f"alpha = {stats['lisa']['alpha']}. Emerging hot spot analysis applies "
        "Mann-Kendall to the Gi* "
        "time series of each unit's space-time bins.",
        "",
        "Driver attribution follows the escalation path required by the project "
        "specification: ordinary least squares, then a test of the residual "
        "Moran's I, then a spatial lag or error model, then geographically "
        "weighted regression and multiscale GWR. Variance inflation factors are "
        "reported at every step, because the candidate drivers over Colombo are "
        "strongly collinear.",
        "",
        "## 9. Conditional scenario projection",
        "",
        f"**Nothing in this section is a forecast.** The framing is "
        f"`{pred['framing']}` throughout, and every predictive product ships "
        "with validation metrics and explicit uncertainty language.",
        "",
        f"A random forest of **{pred['rf']['n_trees']} trees** (seed "
        f"{pred['rf']['random_seed']}, minimum leaf population "
        f"{pred['rf']['min_leaf_population']}, bag fraction "
        f"{pred['rf']['bag_fraction']}) is fitted to "
        f"`{pred['rf']['response']}` on the {pred['rf']['epoch']} "
        f"`{pred['rf']['source']}` composite at {pred['rf']['scale_m']} m, from "
        f"{pred['rf']['sample_pixels']:,} sampled pixels, with predictors "
        + ", ".join(f"`{name}`" for name in pred["rf"]["predictors"])
        + f" (`{', '.join(pred['rf']['categorical'])}` categorical).",
        "",
        f"Validation uses a **spatially blocked** split at "
        f"{pred['split']['block_size_m']:.0f} m blocks, not a random one: "
        "adjacent LST pixels are near-duplicates, so a random split leaks the "
        "test set into the training set and reports an optimistic score."
        + (
            f" Held-out performance is **RMSE {fit['rmse']:.2f} °C, "
            f"R² {fit['r2']:.3f}**."
            if fit else " Held-out performance is not yet measured."
        ),
        "",
        "The **land-cover component is a measured negative result.** A CA-Markov "
        "model calibrated on Dynamic World reproduces the *quantity* of "
        "land-cover change over Colombo but cannot *allocate* it"
        + (
            f": Kappa {lulc['kappa']:.3f} against a persistence Kappa of "
            f"{lulc['persistence_kappa']:.3f} - that is "
            f"{lulc['kappa_above_null']:+.3f} against a no-change map - with a "
            f"figure of merit of {lulc['figure_of_merit']:.3f}."
            if lulc else "."
        )
        + " A validation guard therefore **refuses to export any projected "
        "land-cover product**, and no map of the 2030 or 2036 horizons is "
        "produced. The transition matrix, class areas and validation "
        "sensitivities are published as the evidence for the negative result.",
        "",
        "The **greening counterfactual** rests on the validated random forest "
        "alone, with no land-cover projection underneath it, which is why it can "
        "be mapped while the future horizons cannot. It applies a "
        f"**{pred['scenarios']['greening']['canopy_increase_fraction']:.0%} shift "
        "of each priority cell's surface character toward the observed canopy "
        "signature** and re-predicts. It is a counterfactual on observed "
        "predictors - 'if these zones were greened today' - and it assumes both "
        "that the planting happens and that the fitted relationship holds under "
        "a surface the model never observed. Extrapolation beyond the training "
        "envelope is measured and reported.",
        "",
        "## 10. Greening priority (MCDA / AHP)",
        "",
        f"All {aoi['district']['gn_divisions']} GN divisions are ranked on the "
        f"following {len(green['criteria'])} observed criteria:",
        "",
        "| criterion | direction | what it measures |",
        "|---|---|---|",
        *[
            f"| `{entry['name']}` | {entry['direction']} | {entry['label']} |"
            for entry in green["criteria"]
        ],
        "",
        "Weights are derived from a pairwise comparison matrix by "
        f"{green['ahp']['eigen_method'].replace('_', ' ')} on that matrix, never "
        "set by hand - "
        "`greening.criteria_weights` is pinned to null by a test so the weights "
        "cannot be reverse-engineered from a desired answer. The resulting "
        "weights are "
        + ", ".join(f"{name} {value:.3f}" for name, value in weights.items())
        + f", with lambda_max {ahp_derived['lambda_max']:.4f}, consistency index "
        f"{ahp_derived['consistency_index']:.4f} and **consistency ratio "
        f"{ahp_derived['consistency_ratio']:.4f}** against Saaty's "
        f"{green['ahp']['consistency_ratio_max']} threshold.",
        "",
        "The ranking is cross-checked three ways: an independent **TOPSIS** "
        "ranking under the same weights, a **leave-one-out ablation** against a "
        "heat-only ranking, and a **DS-level re-run** as the MAUP sensitivity. "
        "The ablation, not the consistency ratio, is what says how much the "
        "multi-criteria method adds - and over Colombo the answer is that it "
        "adds very little, which is a finding about the city and is reported as "
        "one.",
        "",
        f"Compliance with the **3-30-300 rule** is assessed per division: a "
        f"{green['rule_3_30_300']['canopy']['target_pct']}% tree-class share, "
        "and a green space of at least "
        f"{green['rule_3_30_300']['green_space']['min_patch_ha']} ha within "
        f"{green['rule_3_30_300']['green_space']['service_distance_m']} m by "
        f"{green['rule_3_30_300']['green_space']['distance_metric']} distance, "
        "bounded by a variant at a "
        f"{green['rule_3_30_300']['green_space']['detour_ratio']} detour ratio. "
        "The '3 trees from every window' component is marked "
        f"`{green['rule_3_30_300']['trees_in_view']['status']}` "
        "(not remotely sensable) and does not "
        "enter the score. The 30% figure is a **Dynamic World tree-class "
        "share** of a 10 m modal classification, not crown cover from a "
        "canopy-height model, and must never be quoted as canopy cover.",
        "",
        f"The exported priority set is the **top {green['top_n']}** divisions, "
        "each carrying its score gap at the cut, its tie flag, its wetland "
        "status and its land-cover coverage flag.",
        "",
        "## 11. Figures and colour",
        "",
        f"Report figures are written at **{params['report']['dpi']} dpi** into "
        f"`{params['report']['figure_dir']}/`. Every palette in the "
        "configuration is verified for colour-vision deficiency under simulated "
        + ", ".join(params["report"]["cvd"]["deficiencies"])
        + " (Viénot, Brettel & Mollon 1999), using two different tests: "
        "categorical palettes must keep a minimum pairwise CIE76 difference of "
        f"{params['report']['cvd']['min_delta_e']}, and sequential or diverging "
        "ramps must keep a monotonic L* profile spanning at least "
        f"{params['report']['cvd']['min_lightness_span']}. Judging a ramp by "
        "pairwise difference is the mistake that split avoids: adjacent stops of "
        "a continuous ramp are meant to be close. The measured result is in "
        "`data/outputs/palette_cvd_check.csv`. Two palettes failed and were "
        "changed; two are exempt for stated reasons and are given redundant "
        "encoding instead, because colour never carries a class alone.",
        "",
        "## 12. Software",
        "",
        "All processing runs in Google Colab against the Earth Engine Python "
        "API. Analysis logic lives in the importable package `colombo_uhi`; the "
        "notebooks orchestrate and display. Pure-Python logic - false-discovery "
        "correction, UTFVI classification, AHP weighting and its consistency "
        "ratio, TOPSIS, the colour-vision checks - is covered by a `pytest` "
        "suite that runs without Earth Engine credentials. Spatial statistics "
        "use `libpysal`, `esda`, `spreg` and `mgwr`; raster post-processing uses "
        "`rasterio`.",
        "",
    ]
    return "\n".join(lines).rstrip("\n") + "\n"


def limitations_markdown(params: dict[str, Any], root: Path | None = None) -> str:
    """Generate the limitations section: the standing caveats, then the findings.

    Two parts, deliberately. The first reproduces every string under ``caveats``
    **verbatim**, because those are the project's standing constraints and a
    paraphrase would weaken them. The second records what implementation
    actually turned up - limitations that were discovered by measuring, not
    anticipated in the specification.

    Args:
        params: Parsed params mapping.
        root: Repository root, for reading committed measurements.

    Returns:
        The complete markdown document, ending in a newline.
    """
    aoi = params["aoi"]
    unused = [key for key, hits in dataset_references(params, root=root).items() if not hits]
    reports = _validation_metrics(root)
    lulc = reports.get("lulc_projection", {}).get("metrics", {})

    lines: list[str] = [
        "# Limitations",
        "",
        _GENERATED_BANNER,
        "",
        "Read this document beside every figure and every number in the report. "
        "Nothing here is a disclaimer in the legal sense: each entry changes how "
        "a specific result may be read, and several of them were found only by "
        "measuring something the documentation said would be fine.",
        "",
        "## Part 1 - standing caveats",
        "",
        "These are reproduced verbatim from `config/params.yaml`, where they are "
        "stamped onto figures and exported metadata by the code itself.",
        "",
    ]

    for key, text in params["caveats"].items():
        heading = _titleise(key)
        lines += [f"### {heading}", "", f"`caveats.{key}`", "",
                  "> " + " ".join(str(text).split()), ""]

    lines += [
        "## Part 2 - limitations found during implementation",
        "",
        "### 1. Collection 2 inter-calibration fails over Colombo",
        "",
        "The Landsat Collection 2 documentation states that the archive is "
        "inter-calibrated across TM, ETM+ and OLI, which would mean no manual "
        "harmonisation is needed. Tested empirically on dry-season CMC means, it "
        "is not true here:",
        "",
        "| pair | mean offset (°C) | t | overlap years | window | verdict |",
        "|---|---|---|---|---|---|",
        *_sensor_offset_lines(root),
        "",
        "Only three of the six pairs overlap enough to be tested at all, and "
        "two of those three step by several times the entire 26-year trend "
        "signal. **Consequence:** the headline trend is fitted within a single "
        "sensor family only, over a shorter record and therefore with less "
        "statistical power. Any multi-sensor product in this project is a "
        "geography statement, never a magnitude one.",
        "",
        "### 2. The land-cover projection is a measured negative result",
        "",
        "The CA-Markov component reproduces the quantity of land-cover change "
        "but cannot allocate it"
        + (
            f", scoring Kappa {lulc['kappa']:.3f} against a persistence Kappa of "
            f"{lulc['persistence_kappa']:.3f} - worse than a no-change map - "
            f"with a figure of merit of {lulc['figure_of_merit']:.3f}"
            if lulc else ""
        )
        + ", across two class schemes and both calibration intervals Dynamic "
        "World's record supports. **Consequence:** no projected land-cover map "
        "exists, and none of the 2030 or 2036 horizons may be mapped. Deliverable "
        "2 carries a stated, quantified limitation instead of an unvalidated "
        "map. The greening counterfactual is unaffected, because it rests on the "
        "regression alone.",
        "",
        "### 3. The multi-criteria ranking reproduces a ranking by heat alone",
        "",
        "The five greening criteria are near-collinear over Colombo. A "
        "leave-one-out ablation gives a rank correlation of about 0.98 between "
        "the full five-criterion ranking and a ranking on land surface "
        "temperature alone, the first principal component carries about 91% of "
        "the variance, and the effective dimensionality is about 1.5. "
        "**Consequence:** this is a finding about Colombo, not a fault in the "
        "method, and the ranking is still the right product - but it is not "
        "adding what a five-criterion MCDA is normally assumed to add, and must "
        "not be presented as if it were.",
        "",
        "### 4. The AHP weights are judgements",
        "",
        "They were argued by the analyst from the literature and from this "
        "project's own measurements. They were **not** elicited from "
        "stakeholders, residents or the municipality. The consistency ratio "
        "tests only whether the judgements are self-consistent, never whether "
        "they are right, and a different defensible set gives a different "
        "ranking.",
        "",
        "### 5. The CMC's area is scale-dependent",
        "",
        "The administrative polygon measures about "
        f"{aoi['expected_areas_km2']['cmc_administrative']} km² because it "
        "encloses the Colombo Port outer harbour; masked to land it measures "
        f"about {aoi['expected_areas_km2']['cmc_land_at_30m']} km² at 30 m and "
        "less at coarser reduction scales, against a gazetted "
        f"{aoi['expected_areas_km2']['cmc']}.31 km². **Consequence:** any CMC "
        "area, and any per-area figure derived from one, must be quoted with "
        "its reduction scale.",
        "",
        "### 6. Two boundary sources that do not agree",
        "",
        "The GAUL district polygon and the uploaded COD-AB GN polygons differ by "
        "roughly 13 km². Exports clipped to the former while zones were burned "
        "from the latter left GN area with no exported data, and a fill-with-zero "
        "read made it indistinguishable from land the classifier failed to "
        "classify. It flagged 23 coastal and edge divisions as unobserved and, "
        "for three Colab runs, deleted Pettah - one of the hottest, densest, most "
        "treeless divisions in the district - from the greening deliverable. "
        "**Consequence:** the export now carries an explicit in-region band, and "
        "the guard message that had been printing the discrepancy all along is no "
        "longer advisory.",
        "",
        "### 7. JRC Global Surface Water does not map the ocean",
        "",
        "The cheap static water mask, whose docstring claimed it was "
        "authoritative for the ocean and the Colombo Port outer harbour, is not: "
        "the water JRC finds across the district is inland. **Consequence:** any "
        "coastal product must use the combined mask, which ORs MNDWI, the "
        "`QA_PIXEL` water-bit frequency and JRC. Anything that used the cheap "
        "mask on a coastal geometry was treating the sea as land.",
        "",
        "### 8. Strict MODIS daytime QC biases the sample toward the core",
        "",
        "The strict day policy retains only a few per cent of daytime "
        "observations and fails disproportionately over the dense coastal core - "
        "exactly where the heat island signal lives. **Consequence:** the strict "
        "Terra day series is never quoted alone; a relaxed-QC variant is run "
        "beside it and both are reported.",
        "",
        "### 9. Green-space access is straight-line and an upper bound",
        "",
        "Service areas are Euclidean, so they ignore the Kelani River, Beira "
        "Lake, the coastal railway and walled compounds, and overstate access by "
        "an unknown amount; a shorter detour-ratio variant bounds the "
        "uncertainty. Green space is counted whether public or private, so "
        "gardens, cantonments and golf courses all count toward compliance. "
        "**Consequence:** every compliance figure is an upper bound.",
        "",
        "### 10. No official wetland boundary exists",
        "",
        "There is no authoritative Colombo Wetland Complex boundary in any free "
        "dataset. The wetland layer is three earth-observation proxies plus the "
        "legally designated sites in the World Database on Protected Areas. "
        "**Consequence:** wetland status is a constructed indicator, not a legal "
        "determination.",
        "",
        "### 11. Coverage flags travel; they do not delete",
        "",
        "A division whose land-cover coverage falls below the floor is flagged in "
        "every output row, on the map as a distinct hatch, and in the metadata "
        "sidecar - but it is **not** removed from the ranking. The floor gates "
        "nothing that enters the score, and across three runs it removed the "
        "densest and hottest divisions in the district. **Consequence:** a "
        "reader may drop those divisions; the pipeline does not do it for them.",
        "",
        "### 12. Agreement between phases is not validation",
        "",
        "The greening ranking correlates strongly with the Phase 5 hot-spot "
        "result, but the two share inputs, so the agreement is not independent "
        "corroboration and is labelled as such in the output.",
        "",
        "### 13. Every zonal statistic is a property of its aggregation",
        "",
        "Coefficients and rankings computed across GN divisions describe those "
        "polygons, not pixels and not people. The DS-level re-run gives "
        "different numbers by construction, which is why both levels are always "
        "reported.",
        "",
    ]

    if unused:
        lines += [
            "### 14. Configured datasets that were never used",
            "",
            "The following are configured but referenced by no analysis step: "
            + ", ".join(f"`{key}`" for key in unused)
            + ". The consequential one is `era5_land`: the project "
            "specification names ERA5-Land `temperature_2m` for reanalysis "
            "air-temperature validation, and **that comparison was never "
            "carried out**. **Consequence:** nothing in this project "
            "independently corroborates the satellite land surface temperatures "
            "against an air-temperature record, which sharpens the standing "
            "caveat that LST is not air temperature - the gap between them is "
            "unquantified here, not merely unquoted.",
            "",
        ]

    lines += [
        "### Colour and legibility",
        "",
        "Two palettes failed the Phase 8 colour-vision check and were replaced; "
        "the measured before-and-after numbers are recorded in "
        "`config/params.yaml` beside each one, and the full result is in "
        "`data/outputs/palette_cvd_check.csv`. Two palettes are exempt with "
        "stated reasons: the Dynamic World legend colours, which are fixed by "
        "the catalog, and the 17-category emerging-hot-spot scheme, which "
        "cannot be separated by colour even for a reader with normal colour "
        "vision. Both are given redundant encoding instead.",
        "",
    ]
    return "\n".join(lines).rstrip("\n") + "\n"


def write_docs(
    params: dict[str, Any], root: Path | None = None
) -> dict[str, Path]:
    """Write ``docs/methods.md``, ``docs/limitations.md`` and the provenance CSV.

    Args:
        params: Parsed params mapping.
        root: Repository root; defaults to :func:`colombo_uhi.repo_root`.

    Returns:
        Mapping of artefact name to the path written.
    """
    base = Path(root) if root is not None else repo_root()
    written: dict[str, Path] = {}

    for name, path, text in (
        ("methods", base / METHODS_PATH, methods_markdown(params, root=root)),
        ("limitations", base / LIMITATIONS_PATH, limitations_markdown(params, root=root)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n" so the committed file is byte-identical on Windows and
        # Linux - otherwise the drift test passes on one and fails on the other.
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        written[name] = path

    provenance = base / PROVENANCE_PATH
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance_frame(params, root=root).to_csv(
        provenance, index=False, lineterminator="\n"
    )
    written["provenance"] = provenance
    return written
