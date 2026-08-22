"""Export wrappers: Export.image.toDrive / table.toDrive, and task status.

Every product that leaves Earth Engine goes through here, so that the naming
convention in ``exports.name_template`` is applied in exactly ONE place. A
product whose name is assembled at the call site is a product that will
eventually collide with another one in Drive and silently overwrite it.

Products:
    * :func:`export_name` - render ``{product}_{aoi}_{startyear}_{endyear}_{res}m``;
      the pure seam every wrapper below routes through;
    * :func:`image_to_drive` / :func:`image_to_asset` / :func:`table_to_drive` -
      thin, explicit wrappers over ``ee.batch.Export``;
    * :func:`describe_tasks` / :func:`wait_for_tasks` / :func:`find_tasks` - the
      status side, including re-attaching to exports submitted before a Colab
      disconnect.

Two things in here are easy to get wrong and are therefore pinned by unit tests:

1. **A dropped template placeholder is a silent data-loss bug.** If a custom
   template omits ``{startyear}``, two different year ranges render to the SAME
   filename and the second export overwrites the first in Drive with no warning.
   :func:`export_name` refuses a template that does not consume every field.
2. **Band ORDER in an exported GeoTIFF is not self-describing.** Earth Engine may
   or may not write band names into the file's band descriptions, so
   :func:`image_to_drive` takes an explicit ``band_order`` and selects the image
   into it before exporting. Without that, adding one band upstream shifts every
   band index in every subsequently downloaded file, and the reader maps the
   p-value band onto the slope.

Design notes:
    * ``import ee`` (and pandas) is deferred into function bodies so this module,
      and the local pytest suite, import cleanly without ``earthengine-api``.
    * Every constant comes from ``config/params.yaml`` (``exports`` section).
    * Nothing here blocks by default. :func:`wait_for_tasks` is opt-in, times out
      rather than hanging, and is safe to call repeatedly.
"""

from __future__ import annotations

import re
import string
import time
import warnings
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, never at runtime
    import ee
    import pandas as pd

#: Placeholders ``exports.name_template`` must consume. A template that omits one
#: lets two different products render to the same name.
NAME_TEMPLATE_FIELDS: tuple[str, ...] = (
    "product",
    "aoi",
    "startyear",
    "endyear",
    "res",
)

#: Earth Engine task descriptions accept only these characters. Anything else is
#: collapsed to a single underscore by :func:`sanitise_name_part`.
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_.,:;-]+")

#: Task states from which no further transition happens.
TERMINAL_STATES: frozenset[str] = frozenset({"COMPLETED", "FAILED", "CANCELLED"})

#: Column order of the table :func:`build_task_status_frame` produces.
TASK_COLUMNS: tuple[str, ...] = (
    "description",
    "id",
    "state",
    "runtime_s",
    "destination",
    "error",
)

#: Export file formats :func:`table_to_drive` knows how to request. The spelling
#: here is the CANONICAL one Earth Engine expects; callers may use any case.
TABLE_FORMATS: tuple[str, ...] = ("CSV", "GeoJSON", "KML", "KMZ", "SHP", "TFRecord")


# =============================================================================
# Pure helpers (no Earth Engine; unit-tested)
# =============================================================================
def resolve_table_format(file_format: str) -> str:
    """Normalise a table export format to the spelling Earth Engine expects.

    Matching is case-insensitive, but the CANONICAL spelling is returned. Four
    of the six names in :data:`TABLE_FORMATS` are mixed case, so comparing an
    upper-cased input against that tuple rejects every one of them - it accepts
    only ``CSV``, and produces the self-contradicting message "unsupported table
    format 'GeoJSON'; expected one of [... 'GeoJSON' ...]". That is exactly what
    happened on the first Colab run of notebook 05.

    Args:
        file_format: Requested format, in any case.

    Returns:
        The canonical member of :data:`TABLE_FORMATS`.

    Raises:
        ValueError: If the format is not one of :data:`TABLE_FORMATS`.
    """
    canonical = {name.upper(): name for name in TABLE_FORMATS}
    resolved = canonical.get(str(file_format).strip().upper())
    if resolved is None:
        raise ValueError(
            f"unsupported table format {file_format!r}; expected one of "
            f"{list(TABLE_FORMATS)} (case-insensitive)"
        )
    return resolved
def sanitise_name_part(value: str, label: str) -> str:
    """Reduce one name component to characters Earth Engine accepts.

    Lowercased, with every run of unsafe characters collapsed to a single
    underscore and leading/trailing underscores stripped. ``"CMC core"`` becomes
    ``"cmc_core"``; ``"LST/dry"`` becomes ``"lst_dry"``.

    Args:
        value: The raw component.
        label: What the component is, used in the error message.

    Returns:
        The sanitised component.

    Raises:
        ValueError: If the component is empty, or contains nothing usable. An
            empty component would collapse two placeholders into one separator
            and change the shape of the rendered name.
    """
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string, got {value!r}")
    cleaned = _UNSAFE_NAME_CHARS.sub("_", value).strip("_").lower()
    if not cleaned:
        raise ValueError(
            f"{label} is empty after removing characters Earth Engine rejects "
            f"(it accepts only A-Z a-z 0-9 . , : ; _ -); got {value!r}"
        )
    return cleaned


def resolve_name_template(template: str | None, params: dict[str, Any]) -> str:
    """Validate the export name template against the fields we can supply.

    Both directions are checked, because each failure mode is real:

    * an **unknown** placeholder makes every export raise at format time;
    * an **unused** placeholder is worse, because nothing raises — two products
      that differ only in the dropped field render to the same name and the
      second overwrites the first in Drive.

    Args:
        template: Explicit template, or ``None`` to read ``exports.name_template``.
        params: Parsed params mapping.

    Returns:
        The validated template.

    Raises:
        KeyError: If the template references a field outside
            :data:`NAME_TEMPLATE_FIELDS`.
        ValueError: If the template consumes no fields, or leaves one unused.
    """
    resolved = template or params["exports"]["name_template"]
    fields = {
        name for _, name, _, _ in string.Formatter().parse(resolved) if name
    }
    unknown = sorted(fields - set(NAME_TEMPLATE_FIELDS))
    if unknown:
        raise KeyError(
            f"export name template references unknown placeholder(s) {unknown}; "
            f"only {list(NAME_TEMPLATE_FIELDS)} are supplied. Fix "
            "exports.name_template in config/params.yaml."
        )
    unused = sorted(set(NAME_TEMPLATE_FIELDS) - fields)
    if unused:
        raise ValueError(
            f"export name template {resolved!r} does not use {unused}. Every "
            "field must appear, or two products differing only in that field "
            "render to the SAME name and the second silently overwrites the "
            "first in Drive."
        )
    return resolved


def export_name(
    product: str,
    aoi: str,
    params: dict[str, Any],
    start_year: int | None = None,
    end_year: int | None = None,
    res_m: int | None = None,
    suffix: str | None = None,
    template: str | None = None,
) -> str:
    """Render the project's export name — the one place a product gets its name.

    Args:
        product: What the file holds, e.g. ``"lst_trend"``.
        aoi: Area of interest, e.g. ``"cmc"`` or ``"district"``.
        params: Parsed params mapping.
        start_year: Defaults to ``time.start_year``.
        end_year: Defaults to ``time.end_year``.
        res_m: Resolution in metres; defaults to ``exports.default_scale_m``.
        suffix: Optional discriminator appended after the resolution, e.g. a
            rural definition or a sensor.
        template: Optional override of ``exports.name_template``.

    Returns:
        The rendered name, e.g. ``"lst_trend_district_2000_2025_100m"``.

    Raises:
        KeyError: If the template references an unsupplied placeholder.
        ValueError: If the template leaves a field unused, if ``end_year`` is
            before ``start_year``, if the resolution is not positive, or if the
            rendered name exceeds ``exports.max_name_chars``.
    """
    resolved_template = resolve_name_template(template, params)

    first = int(params["time"]["start_year"] if start_year is None else start_year)
    last = int(params["time"]["end_year"] if end_year is None else end_year)
    if last < first:
        raise ValueError(f"end_year ({last}) must be >= start_year ({first})")

    resolution = int(
        params["exports"]["default_scale_m"] if res_m is None else res_m
    )
    if resolution <= 0:
        raise ValueError(f"res_m must be a positive number of metres, got {resolution}")

    name = resolved_template.format(
        product=sanitise_name_part(product, "product"),
        aoi=sanitise_name_part(aoi, "aoi"),
        startyear=first,
        endyear=last,
        res=resolution,
    )
    if suffix:
        name = f"{name}_{sanitise_name_part(suffix, 'suffix')}"

    limit = int(params["exports"]["max_name_chars"])
    if len(name) > limit:
        raise ValueError(
            f"rendered export name is {len(name)} characters, over the "
            f"exports.max_name_chars limit of {limit}: {name!r}. Earth Engine "
            "truncates long task descriptions, which can collapse two different "
            "products onto one name. Shorten `product` or `suffix`."
        )
    return name


def resolve_export_settings(
    params: dict[str, Any],
    scale_m: int | None = None,
    crs: str | None = None,
    folder: str | None = None,
    max_pixels: int | None = None,
    cloud_optimized: bool | None = None,
) -> dict[str, Any]:
    """Resolve scale / CRS / Drive folder / pixel ceiling from params.

    Every one of these is passed to Earth Engine EXPLICITLY rather than left to a
    default. Relying on the export defaults is how a product ends up reprojected
    to EPSG:4326 at 1 degree resolution without anything looking wrong.

    Args:
        params: Parsed params mapping.
        scale_m: Export scale; defaults to ``exports.default_scale_m``.
        crs: Export CRS; defaults to ``exports.default_crs``.
        folder: Drive folder; defaults to ``exports.drive_folder``.
        max_pixels: Ceiling; defaults to ``composites.reduce_max_pixels``.
        cloud_optimized: Write a COG; defaults to ``exports.cloud_optimized``.

    Returns:
        Mapping with keys ``scale_m``, ``crs``, ``folder``, ``max_pixels`` and
        ``cloud_optimized``.

    Raises:
        ValueError: If the scale is not positive, or the CRS/folder is empty.
    """
    cfg = params["exports"]

    resolved_scale = int(cfg["default_scale_m"] if scale_m is None else scale_m)
    if resolved_scale <= 0:
        raise ValueError(f"scale_m must be a positive number of metres, got {resolved_scale}")

    resolved_crs = str(cfg["default_crs"] if crs is None else crs).strip()
    if not resolved_crs:
        raise ValueError("crs must be a non-empty EPSG string, e.g. 'EPSG:32644'")

    resolved_folder = str(cfg["drive_folder"] if folder is None else folder).strip()
    if not resolved_folder:
        raise ValueError("folder must be a non-empty Drive folder name")

    return {
        "scale_m": resolved_scale,
        "crs": resolved_crs,
        "folder": resolved_folder,
        "max_pixels": int(
            params["composites"]["reduce_max_pixels"]
            if max_pixels is None
            else max_pixels
        ),
        "cloud_optimized": bool(
            cfg["cloud_optimized"] if cloud_optimized is None else cloud_optimized
        ),
    }


def build_task_status_frame(statuses: Sequence[Mapping[str, Any]]) -> "pd.DataFrame":
    """Shape ``ee.batch.Task.status()`` dicts into the :data:`TASK_COLUMNS` table.

    Pure, so the reshaping is unit-testable without submitting an export. Earth
    Engine omits keys that do not apply (a running task has no ``error_message``,
    a fresh one has no ``destination_uris``), so every field is read with
    ``.get`` and missing values become ``None`` rather than a ``KeyError``.

    Args:
        statuses: Status mappings, one per task.

    Returns:
        ``pd.DataFrame`` with columns :data:`TASK_COLUMNS`; empty but correctly
        shaped when no tasks are given.
    """
    import pandas as pd  # Deferred: see module docstring.

    rows: list[dict[str, Any]] = []
    for status in statuses:
        start = status.get("start_timestamp_ms") or 0
        end = status.get("update_timestamp_ms") or 0
        runtime = (end - start) / 1000.0 if start and end >= start else None
        destination = status.get("destination_uris") or []
        rows.append(
            {
                "description": status.get("description"),
                "id": status.get("id"),
                "state": status.get("state"),
                "runtime_s": runtime,
                "destination": destination[0] if destination else None,
                "error": status.get("error_message"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=list(TASK_COLUMNS))
    return pd.DataFrame(rows)[list(TASK_COLUMNS)]


def all_terminal(statuses: Sequence[Mapping[str, Any]]) -> bool:
    """``True`` when every task has reached a state it cannot leave."""
    return all(status.get("state") in TERMINAL_STATES for status in statuses)


# =============================================================================
# Earth Engine exports
# =============================================================================
def missing_export_bands(
    image: "ee.Image", band_order: Sequence[str]
) -> list[str]:
    """Bands ``band_order`` names that the image does not carry.

    Costs one metadata round trip. ``bandNames`` resolves from the graph without
    computing pixels, so it is cheap next to the export it guards.

    Args:
        image: The image about to be exported.
        band_order: Bands the caller intends to select.

    Returns:
        The absent names, in the order requested; empty when every band is
        present. Also empty when the check itself cannot run, because a
        diagnostic must never replace the real failure with one of its own.
    """
    try:
        present = set(image.bandNames().getInfo() or [])
    except Exception:  # noqa: BLE001 - diagnostics never mask the real error
        return []
    return [band for band in band_order if band not in present]


def image_to_drive(
    image: "ee.Image",
    product: str,
    aoi: str,
    params: dict[str, Any],
    region: "ee.Geometry",
    band_order: Sequence[str] | None = None,
    scale_m: int | None = None,
    crs: str | None = None,
    folder: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    suffix: str | None = None,
    max_pixels: int | None = None,
    start: bool = True,
    verify_bands: bool = True,
) -> "ee.batch.Task":
    """Export an image to Drive as a GeoTIFF, under the project naming convention.

    .. warning::
        ``band_order`` is load-bearing, not cosmetic. Earth Engine does not
        reliably write band names into a GeoTIFF's band descriptions, so the
        reader falls back to position. Selecting the image into an explicit
        order here is what keeps ``trends.export_band_order`` and
        ``trends.read_trend_raster`` describing the same file. Without it,
        adding one band upstream shifts every band index in every subsequently
        downloaded file — and a p-value band read as a slope produces a
        plausible-looking, entirely wrong map.

        ``image.select`` is LAZY, so a ``band_order`` naming a band the image
        does not carry fails inside the batch task, minutes after submission and
        a Colab round trip away from the caller. Phase 8 lost two runs to
        exactly that: ``trends.decadal_band_order`` describes
        ``trends.decadal_product``, the notebook paired it with
        ``trends.decadal_means``, and Earth Engine reported
        ``Image.select: Band pattern 'diff_2011_2020_minus_2000_2010' did not
        match any bands`` six minutes in. ``verify_bands`` resolves the band
        names once and refuses before the task starts.

    .. note::
        Pass ``region`` as a rectangle (``geometry.bounds()``) and clip the image
        instead of passing a 557-vertex polygon. The vertex list travels with
        every export request.

    Args:
        image: Image to export.
        product: Product name for :func:`export_name`.
        aoi: Area name for :func:`export_name`.
        params: Parsed params mapping.
        region: Export extent.
        band_order: Bands to select, in order. ``None`` exports the image as-is.
        scale_m: Export scale; defaults to ``exports.default_scale_m``.
        crs: Export CRS; defaults to ``exports.default_crs``.
        folder: Drive folder; defaults to ``exports.drive_folder``.
        start_year: Defaults to ``time.start_year``.
        end_year: Defaults to ``time.end_year``.
        suffix: Optional name discriminator.
        max_pixels: Ceiling; defaults to ``composites.reduce_max_pixels``.
        start: Submit the task. ``False`` returns it unstarted, so a notebook can
            print what it would do without queueing anything.
        verify_bands: Check ``band_order`` against the image's real bands before
            submitting. Costs one metadata round trip; set ``False`` only when
            that round trip is genuinely unaffordable.

    Returns:
        The ``ee.batch.Task``, started or not.

    Raises:
        ValueError: If ``band_order`` names a band the image does not carry.
    """
    import ee  # Deferred: see module docstring.

    settings = resolve_export_settings(
        params, scale_m=scale_m, crs=crs, folder=folder, max_pixels=max_pixels
    )
    name = export_name(
        product,
        aoi,
        params,
        start_year=start_year,
        end_year=end_year,
        res_m=settings["scale_m"],
        suffix=suffix,
    )

    if band_order and verify_bands:
        absent = missing_export_bands(image, list(band_order))
        if absent:
            try:
                present = image.bandNames().getInfo() or []
            except Exception:  # noqa: BLE001 - report what we already know
                present = []
            raise ValueError(
                f"export '{name}' requests band(s) {absent}, which the image "
                f"does not carry. It has {sorted(present)}. Earth Engine would "
                "accept this and fail inside the batch task minutes from now. "
                "If these are decadal difference bands, build the image with "
                "trends.decadal_product, not trends.decadal_means - "
                "trends.decadal_band_order describes the former."
            )

    payload = image.select(list(band_order)) if band_order else image

    task = ee.batch.Export.image.toDrive(
        image=payload,
        description=name,
        folder=settings["folder"],
        fileNamePrefix=name,
        region=region,
        scale=settings["scale_m"],
        crs=settings["crs"],
        maxPixels=settings["max_pixels"],
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": settings["cloud_optimized"]},
    )
    if start:
        task.start()
    return task


def image_to_asset(
    image: "ee.Image",
    product: str,
    aoi: str,
    params: dict[str, Any],
    region: "ee.Geometry",
    asset_id: str | None = None,
    band_order: Sequence[str] | None = None,
    scale_m: int | None = None,
    crs: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    suffix: str | None = None,
    max_pixels: int | None = None,
    start: bool = True,
) -> "ee.batch.Task":
    """Export an image to an Earth Engine asset.

    The escape hatch behind ``trends.annual_stack_asset``: materialising the
    annual series once turns every later product from "rebuild 26 composites"
    into "read 26 bands", which is the largest single memory lever available
    when the live path keeps exceeding the user memory limit.

    Args:
        image: Image to export.
        product: Product name for :func:`export_name`.
        aoi: Area name for :func:`export_name`.
        params: Parsed params mapping.
        region: Export extent.
        asset_id: Full destination asset id. ``None`` builds
            ``projects/<gcp project>/assets/<name>``.
        band_order: Bands to select, in order.
        scale_m: Export scale.
        crs: Export CRS.
        start_year: Defaults to ``time.start_year``.
        end_year: Defaults to ``time.end_year``.
        suffix: Optional name discriminator.
        max_pixels: Ceiling; defaults to ``composites.reduce_max_pixels``.
        start: Submit the task.

    Returns:
        The ``ee.batch.Task``, started or not.

    Raises:
        ValueError: If ``asset_id`` is omitted and no Earth Engine project id is
            configured to build one from.
    """
    import ee  # Deferred: see module docstring.

    settings = resolve_export_settings(
        params, scale_m=scale_m, crs=crs, max_pixels=max_pixels
    )
    name = export_name(
        product,
        aoi,
        params,
        start_year=start_year,
        end_year=end_year,
        res_m=settings["scale_m"],
        suffix=suffix,
    )

    if asset_id is None:
        project = (params.get("gcp") or {}).get("ee_project_id")
        if not project:
            raise ValueError(
                "cannot build an asset id: gcp.ee_project_id is not set in "
                "config/params.yaml and no explicit asset_id was passed. Either "
                "set it, or pass asset_id='projects/<your-project>/assets/"
                f"{name}'."
            )
        asset_id = f"projects/{project}/assets/{name}"

    payload = image.select(list(band_order)) if band_order else image

    task = ee.batch.Export.image.toAsset(
        image=payload,
        description=name,
        assetId=asset_id,
        region=region,
        scale=settings["scale_m"],
        crs=settings["crs"],
        maxPixels=settings["max_pixels"],
    )
    if start:
        task.start()
    return task


def table_to_drive(
    collection: "ee.FeatureCollection",
    product: str,
    aoi: str,
    params: dict[str, Any],
    file_format: str = "CSV",
    selectors: Sequence[str] | None = None,
    folder: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    suffix: str | None = None,
    res_m: int | None = None,
    start: bool = True,
) -> "ee.batch.Task":
    """Export a feature collection to Drive under the project naming convention.

    Args:
        collection: Features to export.
        product: Product name for :func:`export_name`.
        aoi: Area name for :func:`export_name`.
        params: Parsed params mapping.
        file_format: One of :data:`TABLE_FORMATS`, in any case.
        selectors: Properties to keep, in order. ``None`` exports every property,
            whose column order Earth Engine does not guarantee.
        folder: Drive folder; defaults to ``exports.drive_folder``.
        start_year: Defaults to ``time.start_year``.
        end_year: Defaults to ``time.end_year``.
        suffix: Optional name discriminator.
        res_m: Resolution recorded in the name; defaults to the analysis grid.
        start: Submit the task.

    Returns:
        The ``ee.batch.Task``, started or not.

    Raises:
        ValueError: If ``file_format`` is not one of :data:`TABLE_FORMATS`.
    """
    import ee  # Deferred: see module docstring.

    fmt = resolve_table_format(file_format)

    settings = resolve_export_settings(params, folder=folder, scale_m=res_m)
    name = export_name(
        product,
        aoi,
        params,
        start_year=start_year,
        end_year=end_year,
        res_m=settings["scale_m"],
        suffix=suffix,
    )

    kwargs: dict[str, Any] = {
        "collection": collection,
        "description": name,
        "folder": settings["folder"],
        "fileNamePrefix": name,
        "fileFormat": fmt,
    }
    if selectors:
        kwargs["selectors"] = list(selectors)

    task = ee.batch.Export.table.toDrive(**kwargs)
    if start:
        task.start()
    return task


# =============================================================================
# Task status
# =============================================================================
def describe_tasks(tasks: Sequence[Any]) -> "pd.DataFrame":
    """Fetch every task's status and shape it into the :data:`TASK_COLUMNS` table.

    Args:
        tasks: Task handles from the export wrappers above.

    Returns:
        ``pd.DataFrame`` with one row per task.
    """
    return build_task_status_frame([task.status() for task in tasks])


def find_tasks(
    prefix: str,
    params: dict[str, Any],
    limit: int = 50,
) -> "pd.DataFrame":
    """Re-attach to previously submitted exports by description prefix.

    Colab disconnects. Without this, a 40-minute export submitted before a
    disconnect is invisible to the reconnected session and gets submitted a
    second time — two identical tasks writing the same filename into the same
    Drive folder.

    Args:
        prefix: Description prefix to match, e.g. ``"lst_trend_"``.
        params: Parsed params mapping (unused today; kept so callers do not
            change signature when filtering by folder becomes possible).
        limit: Maximum number of recent tasks to inspect.

    Returns:
        ``pd.DataFrame`` with columns :data:`TASK_COLUMNS`, most recent first.
    """
    import ee  # Deferred: see module docstring.

    del params  # Reserved; see Args.
    statuses = [task.status() for task in ee.batch.Task.list()[:limit]]
    matched = [
        status
        for status in statuses
        if str(status.get("description") or "").startswith(prefix)
    ]
    return build_task_status_frame(matched)


def wait_for_tasks(
    tasks: Sequence[Any],
    params: dict[str, Any],
    poll_seconds: int | None = None,
    timeout_seconds: int | None = None,
    progress: bool = True,
) -> "pd.DataFrame":
    """Poll until every task is terminal, or the timeout elapses.

    .. note::
        This RETURNS on timeout; it does not raise. A Drive export can take well
        over half an hour, and a Colab cell blocked that long risks the runtime
        disconnecting — which loses the task handles as well as the wait. The
        notebook pattern is therefore "submit, then re-run the poll cell until
        every state reads COMPLETED", and this function is safe to call again.

    Args:
        tasks: Task handles.
        params: Parsed params mapping (``exports.poll_seconds`` /
            ``exports.timeout_seconds``).
        poll_seconds: Override the poll interval.
        timeout_seconds: Override the timeout.
        progress: Print a state tally after each poll.

    Returns:
        ``pd.DataFrame`` of the final observed statuses.

    Raises:
        ValueError: If the poll interval or timeout is not positive.
    """
    cfg = params["exports"]
    interval = int(cfg["poll_seconds"] if poll_seconds is None else poll_seconds)
    timeout = int(cfg["timeout_seconds"] if timeout_seconds is None else timeout_seconds)
    if interval <= 0:
        raise ValueError(f"poll_seconds must be > 0, got {interval}")
    if timeout <= 0:
        raise ValueError(f"timeout_seconds must be > 0, got {timeout}")

    deadline = time.monotonic() + timeout
    statuses = [task.status() for task in tasks]

    while not all_terminal(statuses):
        if time.monotonic() >= deadline:
            warnings.warn(
                f"wait_for_tasks timed out after {timeout}s with "
                f"{sum(s.get('state') not in TERMINAL_STATES for s in statuses)} "
                "task(s) still running. Nothing is wrong and nothing was "
                "cancelled - the exports keep going server-side. Re-run this "
                "cell to keep polling.",
                stacklevel=2,
            )
            break
        if progress:
            tally: dict[str, int] = {}
            for status in statuses:
                state = str(status.get("state"))
                tally[state] = tally.get(state, 0) + 1
            print("  " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
        time.sleep(interval)
        statuses = [task.status() for task in tasks]

    failed = [s for s in statuses if s.get("state") == "FAILED"]
    if failed:
        warnings.warn(
            f"{len(failed)} export task(s) FAILED: "
            + "; ".join(
                f"{s.get('description')}: {s.get('error_message')}" for s in failed
            ),
            stacklevel=2,
        )
    return build_task_status_frame(statuses)
