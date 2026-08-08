"""Trend analysis: Mann-Kendall, Sen's slope, FDR correction (Phase 4 — NOT YET WRITTEN).

STUB — docstring only; no implementation exists yet. Do not import expecting
functionality.

Planned contents (per CLAUDE.md):
    * pixel-wise Mann-Kendall via ``ee.Reducer.kendallsCorrelation`` and
      Sen's slope via ``ee.Reducer.sensSlope`` on the ANNUAL composite
      series; slopes reported in degC/yr
    * MK Z and p-value derivation for export
    * Benjamini-Hochberg FDR correction implemented in pure Python (numpy)
      on the exported p-value raster — unit-tested with synthetic arrays;
      only FDR-significant trends are mapped
    * per-pixel valid-observation counts carried through (caveat #2)
"""
