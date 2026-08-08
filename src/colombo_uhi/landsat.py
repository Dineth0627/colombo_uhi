"""Landsat Collection 2 Level-2 loading, masking, scaling (Phase 2 — NOT YET WRITTEN).

STUB — docstring only; no implementation exists yet. Do not import expecting
functionality.

Planned contents (per CLAUDE.md):
    * harmonised TM/ETM+/OLI ImageCollection across L5/L7/L8/L9 (C2 is
      inter-calibrated; no manual coefficients, verify empirically on
      overlapping years)
    * standard mask: QA_PIXEL bits 0-4 all zero AND QA_RADSAT == 0
    * PROCESSING_LEVEL == 'L2SP' filter (ST bands are fully masked in L2SR)
    * ST_B6 / ST_B10 -> Kelvin -> degC using constants from config/params.yaml
    * valid ST DN range and fill handling; per-image valid-observation flags
    * SLC-off awareness for ETM+ after 2003-05-31
"""
