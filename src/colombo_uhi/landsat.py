"""Landsat Collection 2 Level-2 masking and scaling helpers.

Phase 1 ships only the pieces the study-area layer needs (QA_PIXEL clear mask,
surface-reflectance scaling); Phase 2 extends this module with the harmonised
L5/L7/L8/L9 LST collection, `PROCESSING_LEVEL == 'L2SP'` filtering, valid ST DN
range handling and SLC-off awareness.

Design notes:
    * ``import ee`` is deferred into function bodies so this module (and the
      local pytest suite) imports cleanly without ``earthengine-api``.
    * Every constant (band names, scale factors, QA bit positions) comes from
      ``config/params.yaml`` — no magic numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, never at runtime
    import ee


def bits_to_mask(bits: Sequence[int]) -> int:
    """Build an integer bitmask with the given bit positions set.

    Pure helper for QA-band masking, e.g. the standard Landsat C2 L2 cloud
    mask requires QA_PIXEL bits 0-4 to all be zero:
    ``bits_to_mask([0, 1, 2, 3, 4]) == 0b11111 == 31``.

    Args:
        bits: Bit positions (0-based, each >= 0). Duplicates are allowed and
            have no extra effect.

    Returns:
        Integer with exactly the given bits set (0 for an empty sequence).

    Raises:
        ValueError: If any bit position is negative.
    """
    mask = 0
    for bit in bits:
        if bit < 0:
            raise ValueError(f"bit positions must be >= 0, got {bit}")
        mask |= 1 << bit
    return mask


def qa_clear_mask(image: "ee.Image", params: dict[str, Any]) -> "ee.Image":
    """Compute the standard clear-observation mask for a C2 L2 scene.

    Standard mask per CLAUDE.md: QA_PIXEL bits 0-4 (Fill, Dilated Cloud,
    Cirrus, Cloud, Cloud Shadow) all zero AND ``QA_RADSAT == 0`` (no saturated
    bands). Bit 2 is *Unused* (always 0) on TM/ETM+, so the same mask is valid
    across L5/L7/L8/L9.

    Args:
        image: A single Landsat C2 L2 ``ee.Image``.
        params: Parsed params mapping (``landsat_c2l2`` section is used).

    Returns:
        Single-band 0/1 ``ee.Image`` (1 = clear, unsaturated observation).
    """
    import ee  # Deferred: see module docstring.

    c2l2 = params["landsat_c2l2"]
    qa_bits = bits_to_mask(c2l2["standard_mask"]["require_zero_bits"])
    qa_pixel = image.select(c2l2["qa_pixel_band"])
    qa_radsat = image.select(c2l2["qa_radsat_band"])

    clear = qa_pixel.bitwiseAnd(qa_bits).eq(0)
    unsaturated = qa_radsat.eq(c2l2["standard_mask"]["require_qa_radsat_value"])
    return clear.And(unsaturated).rename("clear")


def qa_water_flag(image: "ee.Image", params: dict[str, Any]) -> "ee.Image":
    """Extract the QA_PIXEL water bit (bit 7) as a 0/1 image.

    Args:
        image: A single Landsat C2 L2 ``ee.Image``.
        params: Parsed params mapping (``landsat_c2l2.qa_pixel_bits.water``).

    Returns:
        Single-band 0/1 ``ee.Image`` named ``water`` (1 = flagged as water).
    """
    import ee  # Deferred: see module docstring.

    c2l2 = params["landsat_c2l2"]
    water_bit = bits_to_mask([c2l2["qa_pixel_bits"]["water"]])
    return (
        image.select(c2l2["qa_pixel_band"]).bitwiseAnd(water_bit).gt(0).rename("water")
    )


def scale_sr(image: "ee.Image", params: dict[str, Any]) -> "ee.Image":
    """Convert surface-reflectance DN bands (``SR_B*``) to reflectance.

    Applies ``DN * 0.0000275 - 0.2`` (constants from params) to every SR band
    while leaving all other bands untouched.

    Args:
        image: A single Landsat C2 L2 ``ee.Image``.
        params: Parsed params mapping (``landsat_c2l2.sr_scale``/``sr_offset``).

    Returns:
        The input ``ee.Image`` with its ``SR_B.*`` bands replaced by scaled
        reflectance values (band names preserved).
    """
    import ee  # Deferred: see module docstring.

    c2l2 = params["landsat_c2l2"]
    scaled = (
        image.select("SR_B.*")
        .multiply(c2l2["sr_scale"])
        .add(c2l2["sr_offset"])
    )
    return image.addBands(scaled, overwrite=True)
