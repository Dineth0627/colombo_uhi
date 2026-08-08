"""MODIS MOD11A2/MYD11A2 LST loading and QC filtering (Phase 2 — NOT YET WRITTEN).

STUB — docstring only; no implementation exists yet. Do not import expecting
functionality.

Planned contents (per CLAUDE.md):
    * Terra (MOD11A2) + Aqua (MYD11A2) 8-day LST collections, day and night
    * explicit QC_Day/QC_Night filtering — MOD11A2 has NO built-in QA: keep
      mandatory-QA "good quality" (bits 0-1 == 0) AND LST error <= 1 K
      (bits 6-7 == 0), per config/params.yaml
    * scale 0.02 -> Kelvin -> degC conversions
    * note: Aqua starts 2002-07-04; Terra-only before that
"""
