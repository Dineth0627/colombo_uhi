"""UHI metrics: SUHII, UTFVI, z-scores (Phase 3 — NOT YET WRITTEN).

STUB — docstring only; no implementation exists yet. Do not import expecting
functionality.

Planned contents (per CLAUDE.md):
    * SUHII = mean urban LST - mean rural LST, computed under >= 2 rural
      definitions (buffer ring AND LCZ-based) with sensitivity reporting
    * UTFVI = (Ts - Tmean) / Tmean with the six classes from params
      ``uhi.utfvi`` (Excellent < 0 ... Worst > 0.020)
    * LST z-score / normalisation helpers
    * outputs always labelled LST — never air temperature (caveat #1)
"""
