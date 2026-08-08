"""Export wrappers: Export.image.toDrive / table.toDrive (Phase 2+ — NOT YET WRITTEN).

STUB — docstring only; no implementation exists yet. Do not import expecting
functionality.

Planned contents (per CLAUDE.md):
    * thin wrappers over ``ee.batch.Export.image.toDrive`` and
      ``ee.batch.Export.table.toDrive``
    * consistent naming from params ``exports.name_template``:
      ``{product}_{aoi}_{startyear}_{endyear}_{res}m``
    * default CRS EPSG:32644 and 30 m scale from params ``exports``
"""
