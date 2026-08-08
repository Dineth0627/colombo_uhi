"""MODIS MOD11A2 / MYD11A2 8-day LST with real QC bit filtering.

MOD11A2 is a plain average of the daily MOD11A1 retrievals with **no built-in
quality filtering** (CLAUDE.md). Using ``LST_Day_1km`` straight out of the
collection therefore mixes good retrievals with cloud-contaminated ones. This
module applies the QC bits explicitly: mandatory QA "good quality" (bits 0-1)
AND average LST error <= 1 K (bits 6-7), both configurable for sensitivity runs.

Two catalog details that are easy to get wrong and expensive to debug:

* ``Clear_sky_days`` / ``Clear_sky_nights`` are **8-bit BITMASKS, not counts** —
  one bit per day of the 8-day window. A fully clear period reads as 255, not 8.
  :func:`clear_sky_count` does the popcount.
* ``LST_Day_1km`` / ``LST_Night_1km`` have a valid DN range starting at 7500
  (fill = 0). :func:`lst_collection` gates on the raw DN before scaling.

MODIS is also the only night-time source in this project: Landsat sees a single
~10:30 daytime overpass, so night-time UHI comes from Terra (~22:30) and Aqua
(~01:30) alone (CLAUDE.md caveat 4).

Design notes:
    * ``import ee`` is deferred into function bodies so this module (and the
      local pytest suite) imports cleanly without ``earthengine-api``.
    * Every band name, scale factor and QC bit position comes from
      ``config/params.yaml``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, never at runtime
    import ee

#: Accepted values for the ``daynight`` argument.
DAYNIGHT_OPTIONS: tuple[str, ...] = ("day", "night")

#: Band name for the decoded clear-sky day/night count.
CLEAR_SKY_COUNT_BAND = "clear_sky_count"


# =============================================================================
# Pure helpers (no Earth Engine; unit-tested)
# =============================================================================
def qc_bit_range(bits: Sequence[int]) -> tuple[int, int]:
    """Convert a contiguous QC bit range into a ``(shift, mask)`` pair.

    MODIS QC fields are multi-bit values, so they are read by shifting the
    field down to bit 0 and masking off its width:
    ``qc_bit_range([6, 7]) == (6, 0b11)``.

    Args:
        bits: Contiguous, ascending bit positions, e.g. ``[6, 7]``.

    Returns:
        ``(shift, mask)`` for ``value.rightShift(shift).bitwiseAnd(mask)``.

    Raises:
        ValueError: If ``bits`` is empty, negative, or not contiguous and
            ascending — any of which would silently read the wrong field.
    """
    positions = [int(b) for b in bits]
    if not positions:
        raise ValueError("bits must contain at least one bit position")
    if positions[0] < 0:
        raise ValueError(f"bit positions must be >= 0, got {positions}")
    expected = list(range(positions[0], positions[0] + len(positions)))
    if positions != expected:
        raise ValueError(
            f"bits must be contiguous and ascending, got {positions} "
            f"(expected {expected})"
        )
    return positions[0], (1 << len(positions)) - 1


def resolve_product(product: str, params: dict[str, Any]) -> str:
    """Map a short product name onto its key under ``datasets``.

    Args:
        product: ``"terra"`` or ``"aqua"``.
        params: Parsed params mapping.

    Returns:
        The ``datasets`` key, e.g. ``"modis_terra_lst"``.

    Raises:
        ValueError: If the product is not configured.
    """
    products = params["modis_lst"]["products"]
    if product not in products:
        raise ValueError(
            f"unknown MODIS product '{product}'; configured products are "
            f"{sorted(products)}"
        )
    return products[product]


def resolve_daynight(daynight: str) -> str:
    """Validate the day/night selector.

    Args:
        daynight: ``"day"`` or ``"night"``.

    Returns:
        The validated value.

    Raises:
        ValueError: If it is neither.
    """
    if daynight not in DAYNIGHT_OPTIONS:
        raise ValueError(
            f"daynight must be one of {list(DAYNIGHT_OPTIONS)}, got {daynight!r}"
        )
    return daynight


def clamp_start_date(
    start_date: str, product: str, params: dict[str, Any]
) -> str:
    """Clamp a requested start date to a product's first available date.

    Aqua did not launch until 2002-07-04, so a 2000-2025 request would
    otherwise return silently empty years for 2000-2002 that look like data
    gaps rather than a mission that did not exist yet.

    Args:
        start_date: Requested inclusive start, ``YYYY-MM-DD``.
        product: ``"terra"`` or ``"aqua"``.
        params: Parsed params mapping.

    Returns:
        The later of ``start_date`` and the product's availability start.
    """
    dataset_key = resolve_product(product, params)
    available_from = params["datasets"][dataset_key]["availability"][0]
    if start_date >= available_from:  # ISO dates compare correctly as strings
        return start_date
    warnings.warn(
        f"MODIS {product} ({dataset_key}) has no data before {available_from}; "
        f"clamping the requested start {start_date} to {available_from}. "
        "Years before that are absent by mission history, not by cloud cover.",
        stacklevel=3,
    )
    return available_from


# =============================================================================
# Earth Engine helpers
# =============================================================================
def qc_mask(
    qc_image: "ee.Image",
    params: dict[str, Any],
    mandatory_qa_max: int | None = None,
    lst_error_max: int | None = None,
) -> "ee.Image":
    """Quality mask from a ``QC_Day``/``QC_Night`` band.

    Both QC fields are ordinal with 0 = best, so the thresholds are applied as
    maxima:

    * mandatory QA (bits 0-1): 0 = produced, good quality; 1 = unreliable;
      2 = not produced (cloud); 3 = not produced (other);
    * LST error (bits 6-7): 0 = avg error <= 1 K; 1 = <= 2 K; 2 = <= 3 K;
      3 = > 3 K.

    With both at 0 (the params default) this is exactly CLAUDE.md's "good
    quality AND average error <= 1 K".

    Args:
        qc_image: The ``QC_Day`` or ``QC_Night`` band as an ``ee.Image``.
        params: Parsed params mapping (``modis_lst.qc_filter``).
        mandatory_qa_max: Highest acceptable mandatory-QA value; ``None`` uses
            ``qc_filter.mandatory_qa_required_value``.
        lst_error_max: Highest acceptable LST-error class; ``None`` uses
            ``qc_filter.lst_error_required_value``.

    Returns:
        Single-band 0/1 ``ee.Image`` named ``qc_good``.
    """
    cfg = params["modis_lst"]["qc_filter"]

    qa_shift, qa_mask = qc_bit_range(cfg["mandatory_qa_bits"])
    qa_ceiling = (
        cfg["mandatory_qa_required_value"]
        if mandatory_qa_max is None
        else mandatory_qa_max
    )
    good_qa = qc_image.rightShift(qa_shift).bitwiseAnd(qa_mask).lte(qa_ceiling)

    err_shift, err_mask = qc_bit_range(cfg["lst_error_bits"])
    err_ceiling = (
        cfg["lst_error_required_value"] if lst_error_max is None else lst_error_max
    )
    good_error = qc_image.rightShift(err_shift).bitwiseAnd(err_mask).lte(err_ceiling)

    return good_qa.And(good_error).rename("qc_good")


def clear_sky_count(
    image: "ee.Image", params: dict[str, Any], daynight: str
) -> "ee.Image":
    """Decode ``Clear_sky_days``/``Clear_sky_nights`` into an actual count.

    These bands are BITMASKS: bit 0 = day 1 of the 8-day window ... bit 7 =
    day 8, each set when that day was clear. Reading the raw value as a count
    would report 255 for a fully clear period instead of 8. Earth Engine has no
    popcount primitive, so the set bits are summed.

    Args:
        image: A single MOD11A2/MYD11A2 ``ee.Image``.
        params: Parsed params mapping (``modis_lst``).
        daynight: ``"day"`` or ``"night"``.

    Returns:
        Single-band ``ee.Image`` named ``clear_sky_count`` (0-8).
    """
    cfg = params["modis_lst"]
    band = cfg[f"clear_sky_{resolve_daynight(daynight)}_band"]
    n_bits = int(cfg["clear_sky_bits"])

    source = image.select(band)
    total: "ee.Image | None" = None
    for bit in range(n_bits):  # client-side loop over 8 constants, not a getInfo loop
        is_clear = source.rightShift(bit).bitwiseAnd(1)
        total = is_clear if total is None else total.add(is_clear)
    assert total is not None  # clear_sky_bits is always >= 1
    return total.rename(CLEAR_SKY_COUNT_BAND).toFloat()


def lst_collection(
    product: str,
    daynight: str,
    params: dict[str, Any],
    region: "ee.Geometry | None" = None,
    start_date: str | None = None,
    end_date: str | None = None,
    mandatory_qa_max: int | None = None,
    lst_error_max: int | None = None,
) -> "ee.ImageCollection":
    """QC-filtered 8-day MODIS LST in degrees Celsius.

    Each image carries ``LST_C`` (masked to good-quality pixels) and
    ``clear_sky_count`` (the decoded number of clear days behind that 8-day
    average), plus ``year``, ``month``, ``season``, ``sensor`` and ``product``
    properties matching the Landsat collection's schema.

    Args:
        product: ``"terra"`` (MOD11A2, ~10:30/22:30) or ``"aqua"`` (MYD11A2,
            ~13:30/01:30).
        daynight: ``"day"`` or ``"night"``.
        params: Parsed params mapping.
        region: Geometry to filter to; defaults to
            :func:`colombo_uhi.aoi.analysis_region`.
        start_date: Inclusive start; defaults to ``time.start_year``-01-01,
            clamped to the product's launch date.
        end_date: **Exclusive** end; defaults to ``time.end_year`` + 1 -01-01.
        mandatory_qa_max: See :func:`qc_mask`.
        lst_error_max: See :func:`qc_mask`.

    Returns:
        ``ee.ImageCollection`` sorted ascending by ``system:time_start``.
    """
    import ee  # Deferred: see module docstring.

    from colombo_uhi import landsat

    resolve_daynight(daynight)
    dataset_key = resolve_product(product, params)
    dataset = params["datasets"][dataset_key]
    cfg = params["modis_lst"]
    comp = params["composites"]

    if region is None:
        # Deferred import, consistent with landsat.harmonised_collection.
        from colombo_uhi import aoi

        region = aoi.analysis_region(params)

    time_cfg = params["time"]
    start = clamp_start_date(
        start_date or f"{time_cfg['start_year']}-01-01", product, params
    )
    end = end_date or f"{int(time_cfg['end_year']) + 1}-01-01"

    lst_band = dataset[f"{daynight}_band"]
    qc_band = dataset[f"qc_{daynight}_band"]
    lst_name = params["landsat_c2l2"]["lst_band_name"]
    dn_min, dn_max = cfg["valid_dn_range"]
    season_lookup = landsat.season_by_month(params)

    def prepare(image: "ee.Image") -> "ee.Image":
        raw = image.select(lst_band)
        valid = raw.gte(dn_min).And(raw.lte(dn_max))
        good = qc_mask(
            image.select(qc_band), params, mandatory_qa_max, lst_error_max
        )
        lst_c = (
            raw.multiply(cfg["lst_scale"])
            .subtract(cfg["kelvin_to_celsius_offset"])
            .rename(lst_name)
            .updateMask(valid)
            .updateMask(good)
        )
        stacked = ee.Image.cat(
            [lst_c, clear_sky_count(image, params, daynight)]
        ).toFloat()

        date = ee.Date(image.get("system:time_start"))
        month = ee.Number(date.get("month"))
        return (
            ee.Image(stacked.copyProperties(image, image.propertyNames()))
            .set("system:time_start", image.get("system:time_start"))
            .set(
                {
                    comp["year_property"]: date.get("year"),
                    "month": month,
                    comp["season_property"]: ee.List(season_lookup).get(
                        month.subtract(1)
                    ),
                    comp["sensor_property"]: dataset["id"],
                    "product": product,
                    "daynight": daynight,
                }
            )
        )

    return (
        ee.ImageCollection(dataset["id"])
        .filterBounds(region)
        .filterDate(start, end)
        .map(prepare)
        .sort("system:time_start")
    )


def annual_lst(
    product: str,
    daynight: str,
    params: dict[str, Any],
    geometry: "ee.Geometry | None" = None,
    region: "ee.Geometry | None" = None,
    reducer: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    **kwargs: Any,
) -> "ee.ImageCollection":
    """Annual MODIS LST composites, optionally clipped to a study geometry.

    Like the Landsat composites, every image carries ``obs_count`` — here the
    number of 8-day granules that survived QC filtering, which is a coarser
    quantity than the per-scene Landsat count. Read it as "how many QC-clean
    8-day windows contributed", and use ``clear_sky_count`` on the underlying
    granules for the finer picture.

    Args:
        product: ``"terra"`` or ``"aqua"``.
        daynight: ``"day"`` or ``"night"``.
        params: Parsed params mapping.
        geometry: Clip target, e.g. ``aoi.cmc_boundary(params)`` or
            ``aoi.colombo_district(params).geometry()``. ``None`` leaves the
            composites unclipped.
        region: Geometry the source granules are filtered to, forwarded to
            :func:`lst_collection`; defaults to
            :func:`colombo_uhi.aoi.analysis_region`. Distinct from ``geometry``:
            this one bounds the work, that one bounds the output.
        reducer: Central-tendency reducer; defaults to
            ``composites.modis_reducer``. Pass the SAME reducer used for the
            Landsat series when comparing the two, otherwise part of the offset
            between them is a reducer artefact rather than a sensor difference.
        start_date: Inclusive start; clamped to the product's launch date.
        end_date: Exclusive end.
        **kwargs: Forwarded to
            :func:`colombo_uhi.composites.annual_composites`.

    Returns:
        ``ee.ImageCollection`` with one composite per year.
    """
    from colombo_uhi import composites

    collection = lst_collection(
        product,
        daynight,
        params,
        region=region,
        start_date=start_date,
        end_date=end_date,
    )
    annual = composites.annual_composites(
        collection,
        params,
        band=params["landsat_c2l2"]["lst_band_name"],
        reducer=reducer or params["composites"]["modis_reducer"],
        **kwargs,
    )
    if geometry is None:
        return annual
    return annual.map(lambda image: image.clip(geometry))
