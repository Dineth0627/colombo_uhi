"""Conditional scenario projection: random forest LST + CA-Markov land cover.

**Nothing in this module produces a forecast.** Every product is a *conditional
scenario projection*: "if the transition rates calibrated on the observed
land-cover pair continued, and if the fitted LST-driver relationship held, the
surface would look like this". That framing is not a disclaimer bolted on at the
end - it is enforced. :func:`require_validated` refuses to let an export or a
figure be produced from a product whose validation metrics were never computed,
and :func:`validation_caption` stamps the metrics that *were* computed onto the
figure itself.

Two assumptions carry most of the risk, and both are named in the docstrings of
the functions that make them:

* **Space-for-time substitution.** The random forest is fitted on ONE epoch's
  cross-section - where it is hot *now*, given the drivers *now* - and is then
  fed projected drivers. It has never seen time. Reading its output as "what
  2030 will be like" attributes to time a relationship that was only ever
  measured across space.
* **A random forest cannot extrapolate.** Each leaf returns the mean of the
  training rows that fell into it, so a projected pixel whose drivers lie
  outside the training envelope is pinned to the edge of what the model has
  seen. That is a systematic, directional error, not noise.
  :func:`extrapolation_flags` counts those pixels and
  :func:`require_validated` fails the product when there are too many.

Products:
    * :func:`spatial_block_ids` / :func:`blocked_split` / :func:`blocked_kfold` -
      **spatially blocked** train/test splitting, the only splitting this module
      will fit a reported model on;
    * :func:`fit_sklearn_rf` / :func:`blocked_cv_scores` /
      :func:`permutation_importance_frame` / :func:`compare_split_strategies` -
      the scikit-learn side, for inspectable importance and honest CV;
    * :func:`prediction_stack` / :func:`fit_ee_rf` / :func:`project_lst_image` -
      the Earth Engine side, which is what actually paints a district-wide map;
    * :func:`transition_matrix` / :func:`transition_probabilities` /
      :func:`markov_project` / :func:`ca_markov_project` - the CA-Markov core,
      implemented here in Python so the transition probabilities MOLUSCE reports
      can be validated rather than trusted;
    * :func:`cohen_kappa` / :func:`persistence_baseline_kappa` /
      :func:`quantity_allocation_disagreement` / :func:`figure_of_merit` - the
      land-cover validation suite;
    * :func:`interim_priority_zones` / :func:`apply_greening_scenario` /
      :func:`class_conditional_predictors` - the scenario machinery;
    * :func:`build_validation_report` / :func:`require_validated` /
      :func:`validation_caption` - the guard, and the text it puts on figures.

Seven things in here are easy to get wrong and are therefore pinned by unit
tests and stated loudly in the relevant docstrings:

1. **A random train/test split leaks spatial autocorrelation.** Two pixels 100 m
   apart are nearly the same observation; a random split puts one in train and
   one in test, and the reported R2 measures interpolation between neighbours
   rather than prediction. :func:`resolve_split` REFUSES ``"random"``.
   :func:`compare_split_strategies` is the single, explicitly-named door through
   which a random split can be fitted at all, and it exists only to *measure*
   the inflation so the report can quote it.
2. **Kappa is inflated by persistence.** Over a short land-cover interval most
   cells do not change, so a null projection that says "nothing happens" scores
   a high Kappa. :func:`persistence_baseline_kappa` computes exactly that null,
   and :func:`figure_of_merit` scores the model on CHANGED cells only - the one
   metric here a no-change projection cannot game.
3. **A transition row estimated from nothing is not an identity row by
   accident.** ``0/0`` is ``nan``, and a ``nan`` row silently poisons every
   subsequent matrix power. :func:`transition_probabilities` makes an unobserved
   row explicitly identity and says how many rows it did that to.
4. **A fractional power of a transition matrix need not be a transition
   matrix.** :func:`resolve_projection_steps` rounds to a whole step and returns
   the EFFECTIVE year beside the requested one, rather than quietly labelling a
   1.83-step projection "2035".
5. **A 0/1 land-cover raster cannot distinguish "not this class" from "never
   observed".** Masked pixels are written as 0 in a GeoTIFF. Phase 5 lost a run
   to exactly that, so :func:`lulc_class_image` emits an ``observed`` band and
   :func:`read_lulc_raster` returns it separately from the labels.
6. **Impurity importance is biased toward high-cardinality predictors**, and
   ``lcz_class`` is exactly that. :func:`permutation_importance_frame` computes
   importance on the HELD-OUT blocks; the Earth Engine ``explain()`` importance
   is reported beside it as a cross-check, never as the headline.
7. **Land cover feeds LST, so an LST projection is only as validated as the
   land-cover projection under it.** ``REQUIRED_METRICS`` therefore demands
   RMSE, R2 *and* Kappa for a projected LST product - which is exactly what
   CLAUDE.md's caveat 3 asks for - and only the two regression metrics for a
   present-day fitted surface where no CA was involved.

Design notes:
    * ``ee``, ``numpy``, ``pandas``, ``rasterio`` and ``sklearn`` are deferred
      into function bodies so this module, and the local pytest suite, import
      cleanly without ``earthengine-api`` or a raster stack.
    * Validation runs BEFORE the deferred import, so it stays unit-testable.
    * Every constant comes from ``config/params.yaml`` (``prediction``).
    * Nothing here writes to ``data/`` or ``figures/`` on its own; paths are
      always supplied by the caller.
    * The cellular automaton is deliberately simpler than MOLUSCE's
      neural-network transition potential. That is the point: it is the
      reproducible, unit-tested baseline the MOLUSCE result is *compared with*,
      not a replacement for it. See ``docs/molusce_handoff.md``.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, never at runtime
    import ee
    import numpy as np
    import pandas as pd

#: Splitting methods a *reported* model may be fitted on. ``"random"`` is
#: deliberately absent - see the module docstring, point 1.
SPLIT_METHODS: tuple[str, ...] = ("spatial_block",)

#: The method name :func:`compare_split_strategies` is allowed to use, and
#: :func:`resolve_split` is required to refuse.
RANDOM_SPLIT = "random"

#: Product kinds :func:`require_validated` knows how to gate.
PRODUCT_KINDS: tuple[str, ...] = (
    "lst_fit",
    "lst_projection",
    "lulc_projection",
)

#: Which of ``prediction.validation_metrics`` each product kind must carry.
#:
#: A projected LST surface is built on top of a projected land cover, so it
#: inherits that projection's Kappa - it is not "a regression product" that can
#: ship regression metrics alone. A present-day fitted surface involves no CA
#: and therefore has no Kappa to report.
REQUIRED_METRICS: dict[str, tuple[str, ...]] = {
    "lst_fit": ("rmse", "r2"),
    "lst_projection": ("rmse", "r2", "kappa"),
    "lulc_projection": ("kappa",),
}

#: Column order of the blocked cross-validation table.
CV_COLUMNS: tuple[str, ...] = (
    "fold",
    "n_train",
    "n_test",
    "n_train_blocks",
    "n_test_blocks",
    "rmse",
    "r2",
)

#: Column order of the permutation-importance table.
IMPORTANCE_COLUMNS: tuple[str, ...] = (
    "predictor",
    "importance_mean",
    "importance_std",
    "rank",
)

#: Column order of the projected class-area table.
AREA_COLUMNS: tuple[str, ...] = (
    "class_code",
    "class_label",
    "cells",
    "area_km2",
    "share",
)

#: Column order of the projection-horizon table.
HORIZON_COLUMNS: tuple[str, ...] = (
    "requested_year",
    "base_year",
    "interval_years",
    "steps",
    "effective_year",
    "offset_years",
)

#: Label attached to a validation report that was never actually computed.
NOT_COMPUTED = "not_computed"

#: Wrap width of :func:`validation_caption`, matching
#: :data:`colombo_uhi.viz.CAVEAT_WRAP_CHARS` so a caption and a caveat footer
#: line up when they sit on the same figure.
CAPTION_WRAP_CHARS = 110


class ValidationMissing(RuntimeError):
    """Raised when a predictive product is used without validation metrics.

    This is the guard CLAUDE.md caveat 3 asks for. It is raised by
    :func:`require_validated`, which every export and every figure helper in
    this phase calls **before** doing any work, so there is no code path from an
    unvalidated model to a written product or a plotted map.
    """


# =============================================================================
# Pure helpers (no Earth Engine; unit-tested)
# =============================================================================


def resolve_predictors(
    predictors: Sequence[str] | None, params: dict[str, Any]
) -> list[str]:
    """Validate the random-forest predictor list.

    Args:
        predictors: Override; ``None`` reads ``prediction.rf.predictors``.
        params: Parsed params mapping.

    Returns:
        Predictor names, order preserved, duplicates collapsed.

    Raises:
        ValueError: If the list is empty, or the response appears in it.
    """
    cfg = params["prediction"]["rf"]
    names: list[str] = []
    for name in list(cfg["predictors"] if predictors is None else predictors):
        text = str(name)
        if text not in names:
            names.append(text)
    if not names:
        raise ValueError(
            "at least one predictor is required; set prediction.rf.predictors"
        )
    response = str(cfg["response"])
    if response in names:
        raise ValueError(
            f"the response {response!r} appears in the predictor list; a "
            "regression of a variable on itself has R2 = 1 and means nothing"
        )
    return names


def resolve_categorical(
    params: dict[str, Any], predictors: Sequence[str] | None = None
) -> list[str]:
    """Predictors that are class codes rather than measurements.

    Args:
        params: Parsed params mapping.
        predictors: Override for the predictor list they must be a subset of.

    Returns:
        Categorical predictor names.

    Raises:
        ValueError: If a categorical name is not in the predictor list, which
            would mean the config declares an encoding for a variable the model
            never sees.
    """
    names = resolve_predictors(predictors, params)
    declared = [str(name) for name in params["prediction"]["rf"]["categorical"]]
    unknown = [name for name in declared if name not in names]
    if unknown:
        raise ValueError(
            f"prediction.rf.categorical names {unknown}, which are not in "
            f"prediction.rf.predictors {names}"
        )
    return declared


def resolve_rf_settings(
    params: dict[str, Any], **overrides: Any
) -> dict[str, Any]:
    """Resolve every random-forest hyper-parameter from params.

    Args:
        params: Parsed params mapping.
        **overrides: Any key in ``prediction.rf`` may be overridden by name.

    Returns:
        Mapping with ``n_trees``, ``variables_per_split``,
        ``min_leaf_population``, ``bag_fraction``, ``max_nodes``,
        ``random_seed``, ``response``, ``epoch``, ``source``, ``scale_m``,
        ``predictors`` and ``categorical``.

    Raises:
        ValueError: If a count is non-positive, ``bag_fraction`` is outside
            ``(0, 1]``, or an unknown override name is passed.
    """
    cfg = dict(params["prediction"]["rf"])
    unknown = sorted(set(overrides) - set(cfg))
    if unknown:
        raise ValueError(
            f"unknown random-forest setting(s) {unknown}; "
            f"prediction.rf defines {sorted(cfg)}"
        )
    cfg.update(overrides)

    settings: dict[str, Any] = {
        "n_trees": int(cfg["n_trees"]),
        "min_leaf_population": int(cfg["min_leaf_population"]),
        "bag_fraction": float(cfg["bag_fraction"]),
        "random_seed": int(cfg["random_seed"]),
        "response": str(cfg["response"]),
        "epoch": str(cfg["epoch"]),
        "source": str(cfg["source"]),
        "scale_m": int(cfg["scale_m"]),
        "sample_pixels": int(cfg["sample_pixels"]),
        "min_sample_rows": int(cfg["min_sample_rows"]),
        "lcz_encoding": str(cfg["lcz_encoding"]),
        "variables_per_split": (
            None if cfg["variables_per_split"] is None
            else int(cfg["variables_per_split"])
        ),
        "max_nodes": None if cfg["max_nodes"] is None else int(cfg["max_nodes"]),
    }
    settings["predictors"] = resolve_predictors(cfg["predictors"], params)
    settings["categorical"] = resolve_categorical(params, settings["predictors"])

    for name in ("n_trees", "min_leaf_population", "sample_pixels", "scale_m"):
        if settings[name] <= 0:
            raise ValueError(
                f"prediction.rf.{name} must be positive, got {settings[name]}"
            )
    if not 0.0 < settings["bag_fraction"] <= 1.0:
        raise ValueError(
            "prediction.rf.bag_fraction must be in (0, 1], got "
            f"{settings['bag_fraction']}"
        )
    return settings


def resolve_split(
    params: dict[str, Any], method: str | None = None
) -> dict[str, Any]:
    """Resolve and validate the train/test splitting configuration.

    .. warning::
        This function **refuses** ``"random"``. A random split of a raster
        sample puts pixels 100 m apart into both train and test, so the test set
        is effectively part of the training set and the reported R2 measures
        interpolation between neighbours. Use
        :func:`compare_split_strategies` if you want the random number - it
        exists to quantify that inflation for the report, and it says so.

    Args:
        params: Parsed params mapping.
        method: Override for ``prediction.split.method``.

    Returns:
        Mapping with ``method``, ``block_size_m``, ``test_fraction``,
        ``n_folds``, ``seed`` and ``min_blocks``.

    Raises:
        ValueError: If the method is ``"random"``, or unknown, or a numeric
            setting is out of range.
    """
    cfg = params["prediction"]["split"]
    name = str(cfg["method"] if method is None else method)
    if name == RANDOM_SPLIT:
        raise ValueError(
            "a random train/test split leaks spatial autocorrelation and "
            "inflates R2; prediction.split.method must be one of "
            f"{list(SPLIT_METHODS)}. To MEASURE the inflation for the report, "
            "call compare_split_strategies() instead."
        )
    if name not in SPLIT_METHODS:
        raise ValueError(
            f"unknown split method {name!r}; prediction.split.method accepts "
            f"{list(SPLIT_METHODS)}"
        )

    settings = {
        "method": name,
        "block_size_m": float(cfg["block_size_m"]),
        "test_fraction": float(cfg["test_fraction"]),
        "n_folds": int(cfg["n_folds"]),
        "seed": int(cfg["seed"]),
        "min_blocks": int(cfg["min_blocks"]),
    }
    if settings["block_size_m"] <= 0:
        raise ValueError(
            f"prediction.split.block_size_m must be positive, got "
            f"{settings['block_size_m']}"
        )
    if not 0.0 < settings["test_fraction"] < 1.0:
        raise ValueError(
            "prediction.split.test_fraction must be in (0, 1), got "
            f"{settings['test_fraction']}"
        )
    if settings["n_folds"] < 2:
        raise ValueError(
            f"prediction.split.n_folds must be at least 2, got "
            f"{settings['n_folds']}"
        )
    if settings["min_blocks"] < 2:
        raise ValueError(
            f"prediction.split.min_blocks must be at least 2, got "
            f"{settings['min_blocks']}"
        )
    return settings


def resolve_ca_classes(params: dict[str, Any]) -> list[int]:
    """Land-cover class codes the CA-Markov model retains.

    Args:
        params: Parsed params mapping.

    Returns:
        Sorted class codes.

    Raises:
        ValueError: If the list is empty, has duplicates, or names a code that
            is not in the scheme's legend - which would contribute an
            all-zero transition row rather than raising anywhere later.
    """
    cfg = params["prediction"]["ca_markov"]
    codes = [int(code) for code in cfg["classes"]]
    if not codes:
        raise ValueError("prediction.ca_markov.classes must not be empty")
    if len(set(codes)) != len(codes):
        raise ValueError(
            f"prediction.ca_markov.classes has duplicates: {codes}"
        )
    scheme = str(cfg["scheme"])
    legend = params["landcover"][scheme]["classes"]
    unknown = [code for code in codes if code not in legend]
    if unknown:
        raise ValueError(
            f"prediction.ca_markov.classes names {unknown}, which are not in "
            f"the {scheme} legend {sorted(legend)}"
        )
    return sorted(codes)


def resolve_scenario(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Resolve and validate one scenario definition.

    Args:
        name: Scenario key from ``prediction.scenarios``.
        params: Parsed params mapping.

    Returns:
        Mapping with ``key``, ``label``, ``canopy_increase_fraction`` and, for a
        scenario that converts anything, ``eligible_classes``, ``target_class``
        and ``protect_classes``.

    Raises:
        KeyError: If the scenario is not configured.
        ValueError: If the canopy fraction is outside ``[0, 1]``, a class code
            is outside ``prediction.ca_markov.classes``, or an eligible class is
            also protected - a contradiction that would otherwise resolve
            silently in whichever order the code happened to test.
    """
    scenarios = params["prediction"]["scenarios"]
    if name not in scenarios:
        raise KeyError(
            f"no scenario {name!r}; prediction.scenarios defines "
            f"{sorted(scenarios)}"
        )
    cfg = dict(scenarios[name])
    fraction = float(cfg.get("canopy_increase_fraction", 0.0))
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"prediction.scenarios.{name}.canopy_increase_fraction must be in "
            f"[0, 1], got {fraction}"
        )

    resolved: dict[str, Any] = {
        "key": name,
        "label": str(cfg.get("label", name)),
        "canopy_increase_fraction": fraction,
    }
    if fraction == 0.0 and "target_class" not in cfg:
        # A pure business-as-usual scenario converts nothing, so it needs no
        # class lists. Return early rather than inventing defaults for it.
        resolved.update(
            {"eligible_classes": [], "target_class": None, "protect_classes": []}
        )
        return resolved

    codes = resolve_ca_classes(params)
    eligible = [int(code) for code in cfg["eligible_classes"]]
    protect = [int(code) for code in cfg.get("protect_classes", [])]
    target = int(cfg["target_class"])

    unknown = sorted({*eligible, *protect, target} - set(codes))
    if unknown:
        raise ValueError(
            f"scenario {name!r} names class code(s) {unknown} that are not in "
            f"prediction.ca_markov.classes {codes}"
        )
    overlap = sorted(set(eligible) & set(protect))
    if overlap:
        raise ValueError(
            f"scenario {name!r} lists class code(s) {overlap} as BOTH eligible "
            "for conversion and protected from it"
        )
    if target in protect:
        raise ValueError(
            f"scenario {name!r} protects its own target class {target}"
        )
    resolved.update(
        {
            "eligible_classes": eligible,
            "target_class": target,
            "protect_classes": protect,
        }
    )
    return resolved


def resolve_projection_steps(params: dict[str, Any]) -> "pd.DataFrame":
    """Whole Markov steps to each requested horizon, and the year they land on.

    One Markov step is the calibrated interval itself - the gap between
    ``prediction.ca_markov.projection_base_years``. A horizon that is not a whole
    number of steps cannot be reached by raising the transition matrix to an
    integer power, and a *fractional* matrix power is not guaranteed to be a
    valid stochastic matrix (it can carry negative entries). So this rounds to
    the nearest whole step and returns the **effective** year beside the
    requested one. Every figure and table built from this must quote the
    effective year; nothing is silently relabelled.

    Args:
        params: Parsed params mapping.

    Returns:
        ``pandas.DataFrame`` with :data:`HORIZON_COLUMNS`.

    Raises:
        ValueError: If the base pair is not increasing, a requested year is not
            after the base year, rounding is disabled and a horizon is
            fractional, or rounding would move a horizon by more than
            ``prediction.ca_markov.max_step_offset_years``.
    """
    import pandas as pd  # Deferred: see module docstring.

    cfg = params["prediction"]["ca_markov"]
    base_start, base_end = (int(year) for year in cfg["projection_base_years"])
    interval = base_end - base_start
    if interval <= 0:
        raise ValueError(
            "prediction.ca_markov.projection_base_years must increase, got "
            f"[{base_start}, {base_end}]"
        )
    allow_round = bool(cfg.get("round_projection_steps", True))
    max_offset = int(cfg.get("max_step_offset_years", interval))

    rows: list[dict[str, Any]] = []
    for requested in [int(year) for year in cfg["projection_years"]]:
        span = requested - base_end
        if span <= 0:
            raise ValueError(
                f"projection year {requested} is not after the base year "
                f"{base_end}; prediction.ca_markov.projection_years must all "
                "lie beyond projection_base_years[1]"
            )
        exact = span / interval
        steps = int(round(exact))
        steps = max(steps, 1)
        effective = base_end + steps * interval
        offset = effective - requested
        if offset != 0 and not allow_round:
            raise ValueError(
                f"projection year {requested} is {exact:.2f} steps of "
                f"{interval} years from {base_end}, which is not a whole "
                "number. Set prediction.ca_markov.round_projection_steps true, "
                "or move projection_years onto a multiple of the interval."
            )
        if abs(offset) > max_offset:
            raise ValueError(
                f"rounding projection year {requested} to a whole step lands "
                f"on {effective}, {abs(offset)} years away, which exceeds "
                f"prediction.ca_markov.max_step_offset_years ({max_offset})"
            )
        rows.append(
            {
                "requested_year": requested,
                "base_year": base_end,
                "interval_years": interval,
                "steps": steps,
                "effective_year": effective,
                "offset_years": offset,
            }
        )
    return pd.DataFrame(rows, columns=list(HORIZON_COLUMNS))


# --- Spatially blocked splitting ---------------------------------------------


def spatial_block_ids(
    x: Sequence[float], y: Sequence[float], block_size_m: float
) -> "np.ndarray":
    """Assign each sample to a square spatial block.

    Blocks are the unit a train/test split may cut on. Cutting on rows instead
    puts neighbouring pixels on both sides of the split, which is the leak this
    whole machinery exists to avoid.

    Args:
        x: Projected easting of each sample, in metres.
        y: Projected northing of each sample, in metres.
        block_size_m: Block edge length in metres.

    Returns:
        Integer block id per sample, numbered ``0..n_blocks-1`` in a stable
        order so the same coordinates always produce the same ids.

    Raises:
        ValueError: If the coordinate arrays differ in length, are empty,
            contain non-finite values, or ``block_size_m`` is non-positive.
    """
    import numpy as np  # Deferred: see module docstring.

    size = float(block_size_m)
    if size <= 0:
        raise ValueError(f"block_size_m must be positive, got {size}")

    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.shape != ys.shape:
        raise ValueError(
            f"x and y must have the same shape, got {xs.shape} and {ys.shape}"
        )
    if xs.size == 0:
        raise ValueError("cannot block an empty coordinate set")
    if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
        raise ValueError(
            "x and y must all be finite; a NaN coordinate would silently land "
            "every affected sample in one shared block"
        )

    corners = np.column_stack(
        [np.floor(xs / size).astype(np.int64), np.floor(ys / size).astype(np.int64)]
    )
    _, ids = np.unique(corners, axis=0, return_inverse=True)
    return np.asarray(ids, dtype=np.int64).reshape(xs.shape)


def require_enough_blocks(
    block_ids: Sequence[int], params: dict[str, Any], min_blocks: int | None = None
) -> dict[str, Any]:
    """Refuse to split a sample that has too few blocks to split meaningfully.

    Args:
        block_ids: Block id per sample.
        params: Parsed params mapping.
        min_blocks: Override for ``prediction.split.min_blocks``.

    Returns:
        Mapping with ``n_rows``, ``n_blocks``, ``min_blocks`` and
        ``rows_per_block_median``.

    Raises:
        ValueError: If fewer blocks are present than the configured floor.
    """
    import numpy as np  # Deferred: see module docstring.

    settings = resolve_split(params)
    floor = int(settings["min_blocks"] if min_blocks is None else min_blocks)
    ids = np.asarray(block_ids)
    unique, counts = np.unique(ids, return_counts=True)
    report = {
        "n_rows": int(ids.size),
        "n_blocks": int(unique.size),
        "min_blocks": floor,
        "rows_per_block_median": float(np.median(counts)) if counts.size else 0.0,
    }
    if report["n_blocks"] < floor:
        raise ValueError(
            f"only {report['n_blocks']} spatial block(s) in {report['n_rows']} "
            f"rows, below prediction.split.min_blocks ({floor}). Either the "
            "sample is too small or prediction.split.block_size_m is too "
            "large for the study area. Fold-to-fold spread computed from this "
            "few blocks would not mean anything."
        )
    return report


def blocked_split(
    block_ids: Sequence[int],
    test_fraction: float,
    seed: int,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Split into train and test by whole blocks.

    Blocks are shuffled once and assigned to the test set until the requested
    *row* fraction is reached. A block is never divided, so no sample in the
    test set has a training neighbour inside the same block.

    Args:
        block_ids: Block id per sample.
        test_fraction: Target share of ROWS held out, in ``(0, 1)``.
        seed: Shuffle seed.

    Returns:
        ``(train_mask, test_mask)`` boolean arrays over the samples.

    Raises:
        ValueError: If the fraction is out of range, or the split leaves either
            side empty - which would produce an R2 of ``nan`` two steps later
            rather than an error here.
    """
    import numpy as np  # Deferred: see module docstring.

    fraction = float(test_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {fraction}")

    ids = np.asarray(block_ids)
    unique, counts = np.unique(ids, return_counts=True)
    if unique.size < 2:
        raise ValueError(
            f"cannot hold out whole blocks from {unique.size} block(s); "
            "reduce prediction.split.block_size_m or enlarge the sample"
        )

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(unique.size)
    target = fraction * float(ids.size)

    held: list[int] = []
    accumulated = 0.0
    for position in order:
        if accumulated >= target and held:
            break
        held.append(int(unique[position]))
        accumulated += float(counts[position])

    test_mask = np.isin(ids, held)
    train_mask = ~test_mask
    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            f"a {fraction:.0%} block split left {int(train_mask.sum())} "
            f"training and {int(test_mask.sum())} test rows; one side is "
            "empty. The blocks are too unevenly populated for this fraction."
        )
    return train_mask, test_mask


def blocked_kfold(
    block_ids: Sequence[int], n_folds: int, seed: int
) -> list[tuple["np.ndarray", "np.ndarray"]]:
    """Blocked k-fold cross-validation masks.

    Args:
        block_ids: Block id per sample.
        n_folds: Number of folds.
        seed: Shuffle seed.

    Returns:
        List of ``(train_mask, test_mask)`` pairs, one per fold. Every sample
        appears in exactly one test fold.

    Raises:
        ValueError: If ``n_folds`` is below 2 or exceeds the number of blocks,
            or if any fold would be empty on either side.
    """
    import numpy as np  # Deferred: see module docstring.

    folds = int(n_folds)
    if folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {folds}")

    ids = np.asarray(block_ids)
    unique = np.unique(ids)
    if folds > unique.size:
        raise ValueError(
            f"n_folds ({folds}) exceeds the number of spatial blocks "
            f"({unique.size}); a fold with no block cannot be scored"
        )

    rng = np.random.default_rng(int(seed))
    shuffled = unique[rng.permutation(unique.size)]
    groups = np.array_split(shuffled, folds)

    masks: list[tuple["np.ndarray", "np.ndarray"]] = []
    for index, group in enumerate(groups):
        test_mask = np.isin(ids, group)
        train_mask = ~test_mask
        if not train_mask.any() or not test_mask.any():
            raise ValueError(
                f"fold {index} has {int(train_mask.sum())} training and "
                f"{int(test_mask.sum())} test rows; one side is empty"
            )
        masks.append((train_mask, test_mask))
    return masks


def random_row_split(
    n_rows: int, test_fraction: float, seed: int
) -> tuple["np.ndarray", "np.ndarray"]:
    """A deliberately naive random row split, for measuring its own inflation.

    .. warning::
        **Never report a metric computed on this split as model performance.**
        It exists so :func:`compare_split_strategies` can quote the gap between
        it and the blocked split, which is the honest way to show why the
        blocked split is the one used.

    Args:
        n_rows: Number of samples.
        test_fraction: Share of rows held out.
        seed: Shuffle seed.

    Returns:
        ``(train_mask, test_mask)`` boolean arrays.

    Raises:
        ValueError: If the fraction is out of range or a side would be empty.
    """
    import numpy as np  # Deferred: see module docstring.

    fraction = float(test_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {fraction}")
    count = int(n_rows)
    n_test = int(round(fraction * count))
    if n_test < 1 or n_test >= count:
        raise ValueError(
            f"a {fraction:.0%} split of {count} rows leaves {n_test} test rows"
        )
    rng = np.random.default_rng(int(seed))
    test_mask = np.zeros(count, dtype=bool)
    test_mask[rng.permutation(count)[:n_test]] = True
    return ~test_mask, test_mask


# --- Regression metrics -------------------------------------------------------


def rmse(observed: Sequence[float], predicted: Sequence[float]) -> float:
    """Root mean squared error, in the units of the response.

    Args:
        observed: Measured values.
        predicted: Model values.

    Returns:
        RMSE. For LST products the unit is degrees Celsius.

    Raises:
        ValueError: If the arrays differ in length or are empty.
    """
    import numpy as np  # Deferred: see module docstring.

    truth = np.asarray(observed, dtype=float)
    guess = np.asarray(predicted, dtype=float)
    if truth.shape != guess.shape:
        raise ValueError(
            f"observed and predicted must match, got {truth.shape} and "
            f"{guess.shape}"
        )
    if truth.size == 0:
        raise ValueError("cannot score an empty prediction")
    return float(np.sqrt(np.mean((truth - guess) ** 2)))


def r_squared(observed: Sequence[float], predicted: Sequence[float]) -> float:
    """Coefficient of determination against the mean of the *observed* values.

    .. note::
        On a held-out set this is ``1 - SS_res / SS_tot`` where ``SS_tot`` is
        taken about the held-out mean. It is therefore **not** the training R2
        and it can go negative, which is the correct signal that the model does
        worse than predicting the held-out mean. Do not clamp it.

    Args:
        observed: Measured values.
        predicted: Model values.

    Returns:
        R2, or ``nan`` when the observed values have no variance.

    Raises:
        ValueError: If the arrays differ in length or are empty.
    """
    import numpy as np  # Deferred: see module docstring.

    truth = np.asarray(observed, dtype=float)
    guess = np.asarray(predicted, dtype=float)
    if truth.shape != guess.shape:
        raise ValueError(
            f"observed and predicted must match, got {truth.shape} and "
            f"{guess.shape}"
        )
    if truth.size == 0:
        raise ValueError("cannot score an empty prediction")
    total = float(np.sum((truth - truth.mean()) ** 2))
    if total == 0.0:
        return float("nan")
    residual = float(np.sum((truth - guess) ** 2))
    return 1.0 - residual / total


# --- Land-cover agreement metrics --------------------------------------------


def confusion_matrix(
    observed: Sequence[int], projected: Sequence[int], classes: Sequence[int]
) -> "np.ndarray":
    """Cross-tabulate observed against projected class codes.

    Args:
        observed: Reference class code per cell.
        projected: Model class code per cell.
        classes: Class codes, defining the row and column order.

    Returns:
        ``len(classes) x len(classes)`` integer array. **Rows are observed,
        columns are projected** - the orientation every other function here
        assumes.

    Raises:
        ValueError: If the arrays differ in shape, are empty, or ``classes`` is
            empty or has duplicates.
    """
    import numpy as np  # Deferred: see module docstring.

    codes = [int(code) for code in classes]
    if not codes:
        raise ValueError("classes must not be empty")
    if len(set(codes)) != len(codes):
        raise ValueError(f"classes has duplicates: {codes}")

    truth = np.asarray(observed).ravel()
    guess = np.asarray(projected).ravel()
    if truth.shape != guess.shape:
        raise ValueError(
            f"observed and projected must match, got {truth.shape} and "
            f"{guess.shape}"
        )
    if truth.size == 0:
        raise ValueError("cannot cross-tabulate an empty map")

    lookup = {code: index for index, code in enumerate(codes)}
    matrix = np.zeros((len(codes), len(codes)), dtype=np.int64)
    keep = np.isin(truth, codes) & np.isin(guess, codes)
    dropped = int((~keep).sum())
    if dropped:
        warnings.warn(
            f"{dropped} of {truth.size} cell(s) carry a class code outside "
            f"{codes} and were excluded from the confusion matrix",
            stacklevel=2,
        )
    rows = np.array([lookup[int(code)] for code in truth[keep]], dtype=np.int64)
    cols = np.array([lookup[int(code)] for code in guess[keep]], dtype=np.int64)
    if rows.size:
        np.add.at(matrix, (rows, cols), 1)
    return matrix


def cohen_kappa(matrix: "np.ndarray") -> float:
    """Cohen's kappa from a confusion matrix.

    .. warning::
        Kappa on a short land-cover interval is dominated by persistence: most
        cells do not change, so a null projection that copies the initial map
        scores highly. **Always read this beside**
        :func:`persistence_baseline_kappa` **and** :func:`figure_of_merit`.

    Args:
        matrix: Square confusion matrix, observed in rows.

    Returns:
        Kappa, or ``nan`` when expected agreement is 1 (a single-class map,
        where chance-corrected agreement is undefined). ``nan`` fails
        :func:`require_validated` rather than passing as a perfect score.

    Raises:
        ValueError: If the matrix is not square or sums to zero.
    """
    import numpy as np  # Deferred: see module docstring.

    table = np.asarray(matrix, dtype=float)
    if table.ndim != 2 or table.shape[0] != table.shape[1]:
        raise ValueError(f"matrix must be square, got shape {table.shape}")
    total = float(table.sum())
    if total <= 0:
        raise ValueError("confusion matrix sums to zero")

    observed_agreement = float(np.trace(table)) / total
    row_shares = table.sum(axis=1) / total
    col_shares = table.sum(axis=0) / total
    expected = float(np.sum(row_shares * col_shares))
    if math.isclose(expected, 1.0):
        return float("nan")
    return (observed_agreement - expected) / (1.0 - expected)


def persistence_baseline_kappa(
    initial: Sequence[int], observed: Sequence[int], classes: Sequence[int]
) -> float:
    """Kappa of the null "nothing changes" projection.

    This is the score a model has to beat to have shown anything at all. On a
    three-year Dynamic World interval it is typically very high, which is
    precisely why a bare Kappa is not evidence of skill.

    Args:
        initial: Class code per cell at the start of the projected interval.
        observed: Reference class code per cell at the end of it.
        classes: Class codes.

    Returns:
        Kappa of ``initial`` treated as the projection.
    """
    return cohen_kappa(confusion_matrix(observed, initial, classes))


def quantity_allocation_disagreement(matrix: "np.ndarray") -> dict[str, float]:
    """Decompose disagreement into quantity and allocation (Pontius & Millones).

    Kappa reports a single number for two very different failures: getting the
    *amount* of each class wrong, and getting the amount right but putting it in
    the wrong place. This separates them.

    Args:
        matrix: Square confusion matrix, observed in rows.

    Returns:
        Mapping with ``proportion_correct``, ``quantity_disagreement``,
        ``allocation_disagreement`` and ``total_disagreement``. The first three
        sum to 1 - an identity pinned by a unit test.

    Raises:
        ValueError: If the matrix is not square or sums to zero.
    """
    import numpy as np  # Deferred: see module docstring.

    table = np.asarray(matrix, dtype=float)
    if table.ndim != 2 or table.shape[0] != table.shape[1]:
        raise ValueError(f"matrix must be square, got shape {table.shape}")
    total = float(table.sum())
    if total <= 0:
        raise ValueError("confusion matrix sums to zero")

    shares = table / total
    observed_totals = shares.sum(axis=1)   # reference share of each class
    projected_totals = shares.sum(axis=0)  # projected share of each class
    diagonal = np.diag(shares)

    quantity = float(np.sum(np.abs(observed_totals - projected_totals))) / 2.0
    allocation = float(
        np.sum(
            2.0
            * np.minimum(
                observed_totals - diagonal, projected_totals - diagonal
            )
        )
    ) / 2.0
    correct = float(np.sum(diagonal))
    return {
        "proportion_correct": correct,
        "quantity_disagreement": quantity,
        "allocation_disagreement": allocation,
        "total_disagreement": quantity + allocation,
    }


def figure_of_merit(
    initial: Sequence[int],
    observed: Sequence[int],
    projected: Sequence[int],
) -> dict[str, float]:
    """Score the projection on the cells that actually changed.

    The figure of merit is ``hits / (hits + misses + false alarms + wrong
    hits)``. It ignores correctly-projected persistence entirely, which is what
    makes it the one metric here that a "nothing changes" projection cannot
    game - such a projection scores exactly 0.

    Categories, following Pontius et al. (2008):

    * **hit** - observed changed, and the projection changed it to the right
      class;
    * **wrong hit** - observed changed, the projection changed it, wrong class;
    * **miss** - observed changed, the projection kept it as it was;
    * **false alarm** - observed persisted, the projection changed it.

    Args:
        initial: Class code per cell at the start of the projected interval.
        observed: Reference class code per cell at the end of it.
        projected: Model class code per cell at the end of it.

    Returns:
        Mapping with ``hits``, ``wrong_hits``, ``misses``, ``false_alarms``,
        ``observed_change``, ``projected_change`` and ``figure_of_merit``. The
        figure of merit is ``nan`` when nothing changed and nothing was
        projected to change - there is no skill to measure, which is not the
        same as zero skill.

    Raises:
        ValueError: If the arrays differ in shape or are empty.
    """
    import numpy as np  # Deferred: see module docstring.

    start = np.asarray(initial).ravel()
    truth = np.asarray(observed).ravel()
    guess = np.asarray(projected).ravel()
    if not (start.shape == truth.shape == guess.shape):
        raise ValueError(
            "initial, observed and projected must match, got "
            f"{start.shape}, {truth.shape} and {guess.shape}"
        )
    if start.size == 0:
        raise ValueError("cannot score an empty map")

    observed_change = truth != start
    projected_change = guess != start

    hits = int(np.sum(observed_change & projected_change & (guess == truth)))
    wrong_hits = int(
        np.sum(observed_change & projected_change & (guess != truth))
    )
    misses = int(np.sum(observed_change & ~projected_change))
    false_alarms = int(np.sum(~observed_change & projected_change))

    denominator = hits + wrong_hits + misses + false_alarms
    return {
        "hits": hits,
        "wrong_hits": wrong_hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "observed_change": int(observed_change.sum()),
        "projected_change": int(projected_change.sum()),
        "figure_of_merit": (
            float("nan") if denominator == 0 else hits / denominator
        ),
    }


# --- Markov core --------------------------------------------------------------


def transition_matrix(
    from_codes: Sequence[int],
    to_codes: Sequence[int],
    classes: Sequence[int],
) -> "np.ndarray":
    """Count observed class-to-class transitions.

    Args:
        from_codes: Class code per cell at the earlier date.
        to_codes: Class code per cell at the later date.
        classes: Class codes, defining the row and column order.

    Returns:
        ``len(classes) x len(classes)`` counts, rows the earlier class.

    Raises:
        ValueError: If the arrays differ in shape, are empty, or ``classes`` is
            empty or has duplicates.
    """
    return confusion_matrix(from_codes, to_codes, classes)


def transition_probabilities(counts: "np.ndarray") -> "np.ndarray":
    """Row-normalise a transition count matrix.

    A class that was never observed at the earlier date has an all-zero row.
    ``0/0`` is ``nan``, and a single ``nan`` row silently poisons every
    subsequent matrix power, so an unobserved row is made **explicitly
    identity** - "a class we never saw leave, we do not model leaving" - and the
    number of such rows is reported through a warning.

    Args:
        counts: Square transition count matrix.

    Returns:
        Row-stochastic float matrix; every row sums to 1.

    Raises:
        ValueError: If the matrix is not square or carries negative counts.
    """
    import numpy as np  # Deferred: see module docstring.

    table = np.asarray(counts, dtype=float)
    if table.ndim != 2 or table.shape[0] != table.shape[1]:
        raise ValueError(f"counts must be square, got shape {table.shape}")
    if (table < 0).any():
        raise ValueError("transition counts must not be negative")

    totals = table.sum(axis=1)
    probabilities = np.zeros_like(table)
    observed = totals > 0
    probabilities[observed] = table[observed] / totals[observed, None]
    unobserved = np.flatnonzero(~observed)
    if unobserved.size:
        probabilities[unobserved, unobserved] = 1.0
        warnings.warn(
            f"{unobserved.size} class row(s) at index {unobserved.tolist()} "
            "were never observed at the earlier date and were set to identity; "
            "they contribute no modelled change",
            stacklevel=2,
        )
    return probabilities


def markov_project(probabilities: "np.ndarray", steps: int) -> "np.ndarray":
    """Raise a transition matrix to a whole number of steps.

    Args:
        probabilities: Row-stochastic transition matrix.
        steps: Number of whole intervals to project. Must be a positive
            integer - see :func:`resolve_projection_steps` for why a fractional
            step is refused rather than approximated.

    Returns:
        The ``steps``-step transition matrix.

    Raises:
        TypeError: If ``steps`` is not an integer.
        ValueError: If ``steps`` is below 1, the matrix is not square, or its
            rows do not sum to 1.
    """
    import numpy as np  # Deferred: see module docstring.

    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)):
        raise TypeError(
            f"steps must be a whole number of intervals, got {steps!r}. A "
            "fractional power of a transition matrix is not guaranteed to be a "
            "transition matrix; use resolve_projection_steps()."
        )
    count = int(steps)
    if count < 1:
        raise ValueError(f"steps must be at least 1, got {count}")

    table = np.asarray(probabilities, dtype=float)
    if table.ndim != 2 or table.shape[0] != table.shape[1]:
        raise ValueError(
            f"probabilities must be square, got shape {table.shape}"
        )
    if not np.allclose(table.sum(axis=1), 1.0):
        raise ValueError(
            "probabilities rows must each sum to 1; pass the output of "
            "transition_probabilities(), not raw counts"
        )
    return np.linalg.matrix_power(table, count)


def projected_class_areas(
    current_counts: Sequence[int],
    probabilities: "np.ndarray",
    steps: int,
    cell_area_m2: float,
    classes: Sequence[int],
    params: dict[str, Any],
) -> "pd.DataFrame":
    """Project how many cells of each class the Markov chain implies.

    This sets the *demand* the cellular automaton then has to allocate in space.

    Args:
        current_counts: Cell count per class at the base date, in ``classes``
            order.
        probabilities: One-step row-stochastic transition matrix.
        steps: Whole steps to project.
        cell_area_m2: Area of one cell, for the reported area column.
        classes: Class codes, in the matrix's order.
        params: Parsed params mapping, for the class labels.

    Returns:
        ``pandas.DataFrame`` with :data:`AREA_COLUMNS`. Cell counts are rounded
        to integers and the largest class absorbs the rounding residual, so the
        projected totals match the base total exactly.

    Raises:
        ValueError: If the counts do not match the matrix size.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd

    codes = [int(code) for code in classes]
    counts = np.asarray(current_counts, dtype=float)
    if counts.shape != (len(codes),):
        raise ValueError(
            f"current_counts has shape {counts.shape}, expected "
            f"({len(codes)},) to match classes {codes}"
        )

    projected = counts @ markov_project(probabilities, steps)
    total = int(round(counts.sum()))
    rounded = np.floor(projected).astype(np.int64)
    residual = total - int(rounded.sum())
    if residual != 0 and rounded.size:
        # Hand the rounding residual to the largest class rather than spreading
        # it, so the totals reconcile exactly and the adjustment is traceable.
        rounded[int(np.argmax(projected))] += residual

    scheme = str(params["prediction"]["ca_markov"]["scheme"])
    labels = params["landcover"][scheme]["classes"]
    area_km2 = rounded * float(cell_area_m2) / 1e6
    share = rounded / total if total else np.zeros_like(rounded, dtype=float)
    return pd.DataFrame(
        {
            "class_code": codes,
            "class_label": [str(labels.get(code, code)) for code in codes],
            "cells": rounded.astype(int),
            "area_km2": area_km2,
            "share": share,
        },
        columns=list(AREA_COLUMNS),
    )


# --- Cellular automaton -------------------------------------------------------


def neighbourhood_potential(
    labels: "np.ndarray", target_code: int, radius: int
) -> "np.ndarray":
    """Share of a cell's neighbourhood already occupied by a target class.

    This is the transition potential the allocator ranks candidates by. It
    encodes the one spatial rule a cellular automaton contributes over a plain
    Markov chain: change happens next to change that has already happened.

    .. note::
        This is deliberately simpler than MOLUSCE's multi-layer-perceptron
        potential, which also weighs the driver rasters. That difference is the
        whole reason both are run and compared - see ``docs/molusce_handoff.md``.

    Args:
        labels: 2-D class-code array.
        target_code: Class whose neighbourhood share is measured.
        radius: Neighbourhood radius in cells; ``2`` gives a 5x5 kernel.

    Returns:
        Float array, same shape as ``labels``, holding the share of valid
        neighbours (the centre cell excluded) carrying ``target_code``.

    Raises:
        ValueError: If ``labels`` is not 2-D or ``radius`` is below 1.
    """
    import numpy as np  # Deferred: see module docstring.

    grid = np.asarray(labels)
    if grid.ndim != 2:
        raise ValueError(f"labels must be 2-D, got shape {grid.shape}")
    span = int(radius)
    if span < 1:
        raise ValueError(f"radius must be at least 1, got {span}")

    is_target = (grid == int(target_code)).astype(np.float64)
    hits = np.zeros(grid.shape, dtype=np.float64)
    valid = np.zeros(grid.shape, dtype=np.float64)
    ones = np.ones(grid.shape, dtype=np.float64)

    for offset_y in range(-span, span + 1):
        for offset_x in range(-span, span + 1):
            if offset_y == 0 and offset_x == 0:
                continue
            hits += _shift2d(is_target, offset_y, offset_x)
            valid += _shift2d(ones, offset_y, offset_x)

    return np.divide(hits, valid, out=np.zeros_like(hits), where=valid > 0)


def _shift2d(array: "np.ndarray", offset_y: int, offset_x: int) -> "np.ndarray":
    """Shift a 2-D array, filling the vacated edge with zeros.

    Edges are filled rather than wrapped so a coastal cell is not given
    neighbours from the opposite side of the district.
    """
    import numpy as np  # Deferred: see module docstring.

    out = np.zeros_like(array)
    rows, cols = array.shape
    src_y0, dst_y0 = (0, offset_y) if offset_y >= 0 else (-offset_y, 0)
    src_x0, dst_x0 = (0, offset_x) if offset_x >= 0 else (-offset_x, 0)
    height = rows - abs(offset_y)
    width = cols - abs(offset_x)
    if height <= 0 or width <= 0:
        return out
    out[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = array[
        src_y0 : src_y0 + height, src_x0 : src_x0 + width
    ]
    return out


def ca_allocate(
    labels: "np.ndarray",
    demand: Mapping[int, int],
    params: dict[str, Any],
    changeable: "np.ndarray | None" = None,
    seed: int | None = None,
) -> tuple["np.ndarray", "pd.DataFrame"]:
    """Allocate a Markov demand in space by neighbourhood potential.

    Classes needing more cells take them from classes holding a surplus,
    choosing the candidates with the highest neighbourhood potential for the
    gaining class. Deficits are served largest first, and a cell once taken is
    not reconsidered, so the result is deterministic given ``seed``.

    Args:
        labels: 2-D class-code array at the start of the step.
        demand: Target cell count per class code.
        params: Parsed params mapping.
        changeable: Optional boolean array; ``False`` pins a cell. Cells outside
            the study area and unobserved cells belong here.
        seed: Override for ``prediction.ca_markov.allocation_seed``. Used only
            to break ties between cells of equal potential.

    Returns:
        ``(new_labels, report)`` where ``report`` is a ``pandas.DataFrame`` with
        one row per class holding ``class_code``, ``before``, ``demand``,
        ``after`` and ``shortfall``. **A non-zero shortfall is not an error** -
        it means there were not enough changeable cells to satisfy the demand -
        but it must be reported, so it is a column rather than a warning.

    Raises:
        ValueError: If ``labels`` is not 2-D, ``changeable`` does not match it,
            or ``demand`` names a class outside ``prediction.ca_markov.classes``.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd

    grid = np.asarray(labels)
    if grid.ndim != 2:
        raise ValueError(f"labels must be 2-D, got shape {grid.shape}")

    codes = resolve_ca_classes(params)
    unknown = sorted(set(int(code) for code in demand) - set(codes))
    if unknown:
        raise ValueError(
            f"demand names class code(s) {unknown} outside "
            f"prediction.ca_markov.classes {codes}"
        )

    cfg = params["prediction"]["ca_markov"]
    immutable = {int(code) for code in cfg["immutable_classes"]}
    radius = int(cfg["neighbourhood_radius_cells"])
    rng = np.random.default_rng(
        int(cfg["allocation_seed"] if seed is None else seed)
    )

    if changeable is None:
        mutable = np.ones(grid.shape, dtype=bool)
    else:
        mutable = np.asarray(changeable, dtype=bool)
        if mutable.shape != grid.shape:
            raise ValueError(
                f"changeable has shape {mutable.shape}, expected {grid.shape}"
            )
    mutable = mutable & ~np.isin(grid, sorted(immutable))

    out = grid.copy()
    before = {code: int(np.sum(grid == code)) for code in codes}
    wanted = {code: int(demand.get(code, before[code])) for code in codes}
    deficits = {
        code: wanted[code] - before[code]
        for code in codes
        if wanted[code] > before[code] and code not in immutable
    }
    surplus = {
        code: before[code] - wanted[code]
        for code in codes
        if before[code] > wanted[code] and code not in immutable
    }

    taken = np.zeros(grid.shape, dtype=bool)
    shortfall = {code: 0 for code in codes}

    # Tie-break with a tiny seeded jitter so an unbroken plateau of equal
    # potential does not resolve to raster scan order, which would put every new
    # patch in the top-left corner of the district.
    jitter = rng.random(grid.shape) * 1e-6

    for gaining in sorted(deficits, key=lambda code: -deficits[code]):
        remaining = deficits[gaining]
        while remaining > 0:
            donors = sorted(code for code, value in surplus.items() if value > 0)
            candidate = (
                mutable & ~taken & np.isin(out, donors) & (out != gaining)
            )
            if not donors or not candidate.any():
                break

            # Recomputed each pass: converting a cell changes its neighbours'
            # potential, which is the only thing the automaton adds over a plain
            # Markov chain.
            potential = neighbourhood_potential(out, gaining, radius)
            scores = np.where(candidate, potential + jitter, -np.inf)
            take = min(remaining, int(candidate.sum()))
            flat = np.argpartition(scores.ravel(), -take)[-take:]
            chosen = np.zeros(scores.size, dtype=bool)
            chosen[flat] = True
            chosen = chosen.reshape(scores.shape) & candidate

            # Never strip a donor below its own demand. When the top-scoring
            # selection over-draws one donor, the LOWEST-scoring of its cells
            # are released - and the next pass refills from another donor, so a
            # cap costs placement quality, never the demand itself.
            for donor in donors:
                donor_mask = chosen & (out == donor)
                allowed = surplus[donor]
                excess = int(donor_mask.sum()) - allowed
                if excess > 0:
                    order = np.argsort(np.where(donor_mask, scores, np.inf).ravel())
                    flat_chosen = chosen.ravel()
                    flat_chosen[order[:excess]] = False
                    chosen = flat_chosen.reshape(scores.shape)

            converted = int(chosen.sum())
            if converted == 0:
                break
            for donor in donors:
                surplus[donor] -= int(np.sum(chosen & (out == donor)))
            out[chosen] = gaining
            taken |= chosen
            remaining -= converted

        shortfall[gaining] = remaining
        surplus = {code: value for code, value in surplus.items() if value > 0}

    scheme = str(cfg["scheme"])
    class_labels = params["landcover"][scheme]["classes"]
    report = pd.DataFrame(
        {
            "class_code": codes,
            "class_label": [str(class_labels.get(code, code)) for code in codes],
            "before": [before[code] for code in codes],
            "demand": [wanted[code] for code in codes],
            "after": [int(np.sum(out == code)) for code in codes],
            "shortfall": [shortfall[code] for code in codes],
        }
    )
    return out, report


def ca_markov_project(
    early: "np.ndarray",
    late: "np.ndarray",
    params: dict[str, Any],
    steps: int,
    changeable: "np.ndarray | None" = None,
    cell_area_m2: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Calibrate a Markov chain on two dates and allocate it forward in space.

    **This is a conditional scenario projection, not a forecast.** It answers
    "if the transition rates observed between these two dates continued, and if
    change kept clustering the way it has, where would the classes be" - and
    nothing about whether those rates will continue.

    Args:
        early: 2-D class-code array at the earlier calibration date.
        late: 2-D class-code array at the later calibration date, and the state
            the projection starts from.
        params: Parsed params mapping.
        steps: Whole Markov steps to project.
        changeable: Optional boolean array; ``False`` pins a cell.
        cell_area_m2: Override for the area implied by
            ``prediction.ca_markov.raster_scale_m``.
        seed: Override for ``prediction.ca_markov.allocation_seed``.

    Returns:
        Mapping with ``labels`` (the projected array), ``counts`` (the observed
        transition counts), ``probabilities`` (one-step), ``areas`` (the Markov
        demand table) and ``allocation`` (the per-class allocation report).

    Raises:
        ValueError: If the two arrays differ in shape.
    """
    import numpy as np  # Deferred: see module docstring.

    start = np.asarray(early)
    finish = np.asarray(late)
    if start.shape != finish.shape:
        raise ValueError(
            f"early and late must match, got {start.shape} and {finish.shape}"
        )

    codes = resolve_ca_classes(params)
    cfg = params["prediction"]["ca_markov"]
    scale = float(cfg["raster_scale_m"])
    area = float(scale * scale if cell_area_m2 is None else cell_area_m2)

    if changeable is None:
        valid = np.isin(start, codes) & np.isin(finish, codes)
    else:
        valid = np.asarray(changeable, dtype=bool) & np.isin(
            start, codes
        ) & np.isin(finish, codes)

    counts = transition_matrix(start[valid], finish[valid], codes)
    probabilities = transition_probabilities(counts)

    current = np.array(
        [int(np.sum(finish[valid] == code)) for code in codes], dtype=np.int64
    )
    areas = projected_class_areas(
        current, probabilities, steps, area, codes, params
    )
    demand = dict(zip(areas["class_code"], areas["cells"]))
    labels, allocation = ca_allocate(
        finish, demand, params, changeable=valid, seed=seed
    )
    return {
        "labels": labels,
        "counts": counts,
        "probabilities": probabilities,
        "areas": areas,
        "allocation": allocation,
        "steps": int(steps),
        "framing": str(params["prediction"]["framing"]),
    }


# --- Scenario machinery -------------------------------------------------------


def interim_priority_zones(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    columns: Mapping[str, str] | None = None,
) -> "pd.DataFrame":
    """Rank zones for greening on a Phase-5 proxy, pending Phase 7's MCDA.

    .. warning::
        This is an **interim** rule, flagged as such by
        ``prediction.priority_zones.source``. It ranks on outcome (where it is
        hot, where there is little vegetation) and says nothing about
        *feasibility* - land ownership, existing use, or planting suitability -
        which is exactly what Phase 7's MCDA/AHP weighted overlay adds. Because
        :func:`apply_greening_scenario` takes a plain zone list, Phase 7 replaces
        this function's output without touching the scenario code.

    Args:
        frame: One row per zone, with a ``zone_id`` column and one column per
            criterion.
        params: Parsed params mapping.
        columns: Override mapping criterion name to frame column. Defaults to
            ``gi_star_hot`` -> ``gi_z``, ``lst_2020s`` -> ``LST_C``,
            ``ndvi_inverse`` -> ``NDVI``.

    Returns:
        ``pandas.DataFrame`` sorted by descending score, with ``zone_id``, one
        ``<criterion>_rank`` column per criterion (percentile rank in ``[0, 1]``,
        higher = higher priority), ``score``, ``rank`` and a boolean
        ``priority`` flagging the top ``prediction.priority_zones.top_n``.

    Raises:
        ValueError: If the weights do not match the criteria or do not sum to 1,
            a required column is absent, or ``zone_id`` is missing or duplicated.
    """
    import pandas as pd  # Deferred: see module docstring.

    cfg = params["prediction"]["priority_zones"]
    criteria = [str(name) for name in cfg["rank_by"]]
    weights = [float(value) for value in cfg["weights"]]
    if len(weights) != len(criteria):
        raise ValueError(
            f"prediction.priority_zones has {len(criteria)} criteria but "
            f"{len(weights)} weights"
        )
    if not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError(
            f"prediction.priority_zones.weights sum to {sum(weights)}, not 1"
        )

    default = {
        "gi_star_hot": "gi_z",
        "lst_2020s": "LST_C",
        "ndvi_inverse": "NDVI",
    }
    mapping = dict(default)
    if columns:
        mapping.update({str(k): str(v) for k, v in columns.items()})

    if "zone_id" not in frame.columns:
        raise ValueError(
            f"frame has no 'zone_id' column; it has {sorted(frame.columns)}"
        )
    if frame["zone_id"].duplicated().any():
        raise ValueError(
            "frame has duplicate zone_id values; merge the criterion tables "
            "before ranking, or the same zone competes with itself"
        )

    out = pd.DataFrame({"zone_id": frame["zone_id"].astype(str)})
    score = pd.Series(0.0, index=frame.index)
    for name, weight in zip(criteria, weights):
        column = mapping.get(name)
        if column is None:
            raise ValueError(
                f"criterion {name!r} has no column mapping; pass one via "
                f"columns=, known defaults are {sorted(default)}"
            )
        if column not in frame.columns:
            raise ValueError(
                f"criterion {name!r} needs column {column!r}, which is not in "
                f"{sorted(frame.columns)}"
            )
        values = pd.to_numeric(frame[column], errors="coerce")
        # An inverted criterion is one where LOW is high-priority.
        if name.endswith("_inverse"):
            values = -values
        ranked = values.rank(pct=True, na_option="bottom")
        out[f"{name}_rank"] = ranked.to_numpy()
        score = score + weight * ranked

    out["score"] = score.to_numpy()
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    top_n = min(int(cfg["top_n"]), len(out))
    out["priority"] = out["rank"] <= top_n
    return out


def apply_greening_scenario(
    labels: "np.ndarray",
    priority_mask: "np.ndarray",
    params: dict[str, Any],
    scenario: str,
    seed: int | None = None,
) -> tuple["np.ndarray", dict[str, Any]]:
    """Convert a share of eligible cells inside priority zones to canopy.

    Conversion is *not* random within the eligible set: cells with the highest
    neighbourhood share of the target class are converted first, because
    planting that extends existing canopy is both more plausible as policy and
    more effective thermally than the same area scattered as isolated cells.

    Args:
        labels: 2-D class-code array to modify.
        priority_mask: Boolean array, ``True`` inside a priority zone.
        params: Parsed params mapping.
        scenario: Scenario key from ``prediction.scenarios``.
        seed: Tie-break seed; defaults to
            ``prediction.ca_markov.allocation_seed``.

    Returns:
        ``(new_labels, report)``. The report carries ``scenario``, ``label``,
        ``canopy_increase_fraction``, ``n_priority_cells``, ``n_eligible``,
        ``n_converted`` and ``target_class`` - every number the figure caption
        needs to state what lever was pulled and how hard.

    Raises:
        ValueError: If the arrays differ in shape or ``labels`` is not 2-D.
    """
    import numpy as np  # Deferred: see module docstring.

    grid = np.asarray(labels)
    if grid.ndim != 2:
        raise ValueError(f"labels must be 2-D, got shape {grid.shape}")
    mask = np.asarray(priority_mask, dtype=bool)
    if mask.shape != grid.shape:
        raise ValueError(
            f"priority_mask has shape {mask.shape}, expected {grid.shape}"
        )

    cfg = resolve_scenario(scenario, params)
    ca_cfg = params["prediction"]["ca_markov"]
    rng = np.random.default_rng(
        int(ca_cfg["allocation_seed"] if seed is None else seed)
    )

    out = grid.copy()
    report: dict[str, Any] = {
        "scenario": cfg["key"],
        "label": cfg["label"],
        "canopy_increase_fraction": cfg["canopy_increase_fraction"],
        "target_class": cfg["target_class"],
        "n_priority_cells": int(mask.sum()),
        "n_eligible": 0,
        "n_converted": 0,
    }
    if cfg["canopy_increase_fraction"] <= 0 or cfg["target_class"] is None:
        return out, report

    eligible = (
        mask
        & np.isin(grid, cfg["eligible_classes"])
        & ~np.isin(grid, cfg["protect_classes"])
    )
    report["n_eligible"] = int(eligible.sum())
    take = int(math.floor(cfg["canopy_increase_fraction"] * eligible.sum()))
    if take <= 0:
        return out, report

    potential = neighbourhood_potential(
        grid, cfg["target_class"], int(ca_cfg["neighbourhood_radius_cells"])
    )
    scores = np.where(
        eligible, potential + rng.random(potential.shape) * 1e-6, -np.inf
    )
    flat = np.argpartition(scores.ravel(), -take)[-take:]
    chosen = np.zeros(scores.size, dtype=bool)
    chosen[flat] = True
    chosen = chosen.reshape(scores.shape) & eligible

    out[chosen] = cfg["target_class"]
    report["n_converted"] = int(chosen.sum())
    return out, report


def class_conditional_predictors(
    frame: "pd.DataFrame",
    class_column: str,
    predictors: Sequence[str],
    min_rows: int = 30,
) -> "pd.DataFrame":
    """Median predictor value within each observed land-cover class.

    This is how a projected land-cover map becomes a projected predictor stack:
    each projected class is painted with the value that class carries *today*.

    .. warning::
        This is a **substitution, not a simulation**. It assumes the spectral
        and structural signature of, say, "built" in 2030 is the signature of
        "built" in the training epoch. Densification within a class, or a change
        in construction materials, is invisible to it. Classes with fewer than
        ``min_rows`` observations are flagged rather than dropped, because a
        thinly-sampled class still has to be painted with something and the
        report must say which ones those were.

    Args:
        frame: Training sample, one row per pixel.
        class_column: Column holding the land-cover class code.
        predictors: Predictor columns to summarise.
        min_rows: Below this, the class is flagged ``thin``.

    Returns:
        ``pandas.DataFrame`` with ``class_code``, ``n``, ``thin`` and one column
        per predictor.

    Raises:
        ValueError: If a required column is absent or the frame is empty.
    """
    import pandas as pd  # Deferred: see module docstring.

    names = [str(name) for name in predictors]
    missing = [
        name for name in [class_column, *names] if name not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"frame is missing {missing}; it has {sorted(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("cannot summarise an empty training sample")

    rows: list[dict[str, Any]] = []
    for code, group in frame.groupby(class_column, sort=True):
        record: dict[str, Any] = {
            "class_code": int(code),
            "n": int(len(group)),
            "thin": bool(len(group) < int(min_rows)),
        }
        for name in names:
            record[name] = float(
                pd.to_numeric(group[name], errors="coerce").median()
            )
        rows.append(record)
    return pd.DataFrame(rows, columns=["class_code", "n", "thin", *names])


def paint_class_predictors(
    labels: "np.ndarray",
    table: "pd.DataFrame",
    predictors: Sequence[str],
) -> dict[str, "np.ndarray"]:
    """Paint a class map with each class's conditional predictor value.

    Args:
        labels: 2-D class-code array.
        table: Output of :func:`class_conditional_predictors`.
        predictors: Predictor names to paint.

    Returns:
        Mapping predictor name to a float array shaped like ``labels``. Cells
        whose class is absent from ``table`` are ``nan``, never 0 - a zero NDVI
        is a real value and would be indistinguishable from "unknown".

    Raises:
        ValueError: If ``labels`` is not 2-D or a predictor is missing.
    """
    import numpy as np  # Deferred: see module docstring.

    grid = np.asarray(labels)
    if grid.ndim != 2:
        raise ValueError(f"labels must be 2-D, got shape {grid.shape}")
    names = [str(name) for name in predictors]
    missing = [name for name in names if name not in table.columns]
    if missing:
        raise ValueError(
            f"the class-conditional table is missing {missing}; it has "
            f"{sorted(table.columns)}"
        )

    out: dict[str, "np.ndarray"] = {}
    for name in names:
        painted = np.full(grid.shape, np.nan, dtype=float)
        for record in table.itertuples():
            painted[grid == int(record.class_code)] = float(
                getattr(record, name)
            )
        out[name] = painted
    return out


# --- Validation reporting and the export guard --------------------------------


def extrapolation_flags(
    training: "pd.DataFrame",
    target: "pd.DataFrame",
    params: dict[str, Any],
    predictors: Sequence[str] | None = None,
) -> dict[str, Any]:
    """How far the projected predictors sit outside the training envelope.

    A random forest returns the mean of the training rows in each leaf, so it
    **cannot** produce a value outside the training response range. A projected
    pixel whose drivers lie beyond the training minimum or maximum is therefore
    pinned to the edge of what the model has seen - a systematic error in a
    known direction, not noise. Counting those pixels is the only honest way to
    say how much of a projected map is inside the model's competence.

    Args:
        training: Rows the model was fitted on.
        target: Rows the model is about to be applied to.
        params: Parsed params mapping.
        predictors: Override for ``prediction.rf.predictors``.

    Returns:
        Mapping with ``fraction`` (share of target rows outside the envelope on
        at least one predictor), ``n``, ``n_outside``, ``tolerance``,
        ``within_tolerance`` and ``by_predictor``.

    Raises:
        ValueError: If a predictor column is absent from either frame, or the
            target frame is empty.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd

    names = resolve_predictors(predictors, params)
    for label, frame in (("training", training), ("target", target)):
        missing = [name for name in names if name not in frame.columns]
        if missing:
            raise ValueError(
                f"the {label} frame is missing {missing}; it has "
                f"{sorted(frame.columns)}"
            )
    if len(target) == 0:
        raise ValueError("cannot check extrapolation against an empty target")

    tolerance = float(
        params["prediction"]["rf"]["extrapolation"]["tolerance_fraction"]
    )
    outside_any = np.zeros(len(target), dtype=bool)
    by_predictor: dict[str, float] = {}
    for name in names:
        low = float(pd.to_numeric(training[name], errors="coerce").min())
        high = float(pd.to_numeric(training[name], errors="coerce").max())
        values = pd.to_numeric(target[name], errors="coerce").to_numpy()
        outside = (values < low) | (values > high) | ~np.isfinite(values)
        by_predictor[name] = float(np.mean(outside))
        outside_any |= outside

    fraction = float(np.mean(outside_any))
    return {
        "fraction": fraction,
        "n": int(len(target)),
        "n_outside": int(outside_any.sum()),
        "tolerance": tolerance,
        "within_tolerance": bool(fraction <= tolerance),
        "by_predictor": by_predictor,
    }


def build_validation_report(
    kind: str,
    metrics: Mapping[str, float],
    params: dict[str, Any],
    held_out: bool,
    n_train: int | None = None,
    n_test: int | None = None,
    n_blocks: int | None = None,
    block_size_m: float | None = None,
    extrapolation: Mapping[str, Any] | None = None,
    notes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Assemble the validation record that must travel with a product.

    Args:
        kind: One of :data:`PRODUCT_KINDS`.
        metrics: Computed metric values, keyed by
            ``prediction.validation_metrics`` names plus any extras
            (``figure_of_merit``, ``persistence_kappa``, ...).
        params: Parsed params mapping.
        held_out: Whether the metrics were computed on data the model never saw.
            Passing ``True`` for training-set metrics is the one failure this
            guard cannot detect, so it is a required positional argument rather
            than a default.
        n_train: Training row count.
        n_test: Held-out row count.
        n_blocks: Number of spatial blocks the split cut on.
        block_size_m: Block edge length used.
        extrapolation: Output of :func:`extrapolation_flags`.
        notes: Extra lines for the figure caption.

    Returns:
        A plain mapping, safe to write to CSV or JSON.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    if kind not in PRODUCT_KINDS:
        raise ValueError(
            f"unknown product kind {kind!r}; known kinds are "
            f"{list(PRODUCT_KINDS)}"
        )
    return {
        "kind": kind,
        "framing": str(params["prediction"]["framing"]),
        "metrics": {str(k): float(v) for k, v in metrics.items()},
        "held_out": bool(held_out),
        "n_train": None if n_train is None else int(n_train),
        "n_test": None if n_test is None else int(n_test),
        "n_blocks": None if n_blocks is None else int(n_blocks),
        "block_size_m": None if block_size_m is None else float(block_size_m),
        "extrapolation": dict(extrapolation) if extrapolation else None,
        "notes": [str(note) for note in (notes or [])],
    }


def require_validated(
    report: Mapping[str, Any] | None, params: dict[str, Any]
) -> dict[str, Any]:
    """Refuse to proceed unless the product's validation metrics exist.

    This is the gate CLAUDE.md caveat 3 asks for. Every export wrapper and every
    figure helper in this phase calls it first, so there is no path from an
    unvalidated model to a written product or a plotted map.

    It checks four things:

    1. a report exists at all;
    2. it carries every metric :data:`REQUIRED_METRICS` demands for its kind;
    3. every one of those is finite - a ``nan`` Kappa from a single-class map is
       an absence of evidence, not a pass;
    4. the metrics were computed on **held-out** data, and the extrapolation
       fraction is within ``prediction.rf.extrapolation.tolerance_fraction``.

    Args:
        report: Output of :func:`build_validation_report`.
        params: Parsed params mapping.

    Returns:
        The report, unchanged, so callers can chain.

    Raises:
        ValidationMissing: On any of the four failures, naming the params key or
            the function that produces what is missing.
    """
    import math as _math

    if not report:
        raise ValidationMissing(
            "this product has no validation report. Predictive outputs are "
            "conditional scenario projections and must ship with validation "
            "metrics (CLAUDE.md caveat 3). Build one with "
            "build_validation_report() before exporting or plotting."
        )
    kind = str(report.get("kind", ""))
    if kind not in REQUIRED_METRICS:
        raise ValidationMissing(
            f"validation report has unknown kind {kind!r}; known kinds are "
            f"{list(PRODUCT_KINDS)}"
        )

    declared = [str(name) for name in params["prediction"]["validation_metrics"]]
    required = [name for name in REQUIRED_METRICS[kind] if name in declared]
    metrics = dict(report.get("metrics") or {})

    absent = [name for name in required if name not in metrics]
    if absent:
        raise ValidationMissing(
            f"a {kind!r} product requires {list(required)} and is missing "
            f"{absent}. It has {sorted(metrics)}. A projected LST surface "
            "inherits the Kappa of the land-cover projection under it - it is "
            "not validated by its regression metrics alone."
        )
    unusable = [
        name
        for name in required
        if not _math.isfinite(float(metrics[name]))
    ]
    if unusable:
        raise ValidationMissing(
            f"a {kind!r} product has non-finite {unusable}. A NaN metric is an "
            "absence of evidence, not a passing score; find out why it could "
            "not be computed before exporting anything built on it."
        )
    if not report.get("held_out"):
        raise ValidationMissing(
            f"a {kind!r} product's metrics were not computed on held-out data. "
            "Training-set metrics measure memorisation. Split with "
            "blocked_split() or blocked_kfold() and re-score."
        )

    extrapolation = report.get("extrapolation")
    if extrapolation and not extrapolation.get("within_tolerance", True):
        raise ValidationMissing(
            f"{extrapolation['fraction']:.1%} of the target pixels lie outside "
            "the training envelope on at least one predictor, above "
            "prediction.rf.extrapolation.tolerance_fraction "
            f"({extrapolation['tolerance']:.1%}). A random forest cannot "
            "extrapolate, so those pixels are pinned to the edge of the "
            "training range. Narrow the projection, widen the training sample, "
            "or raise the tolerance deliberately and say so in the report."
        )
    return dict(report)


def validation_caption(
    report: Mapping[str, Any] | None, params: dict[str, Any]
) -> str:
    """The uncertainty text that must appear on a predictive figure.

    Args:
        report: Output of :func:`build_validation_report`.
        params: Parsed params mapping.

    Returns:
        A short multi-line string: the framing sentence, the metrics with the
        split they were computed on, the extrapolation fraction, and any notes.

    Raises:
        ValidationMissing: If the report is absent or incomplete - a predictive
            figure without metrics on it is not a figure this project ships.
    """
    import textwrap

    validated = require_validated(report, params)
    metrics = validated["metrics"]

    items = [
        "CONDITIONAL SCENARIO PROJECTION, NOT A FORECAST. This is what the "
        "surface would look like IF the calibrated transition rates continued "
        "and the fitted LST-driver relationship held. It is Land Surface "
        "Temperature, not air temperature.",
    ]

    labels = {
        "rmse": "RMSE",
        "r2": "R2",
        "kappa": "Kappa",
        "persistence_kappa": "Kappa of the no-change null",
        "figure_of_merit": "Figure of merit",
    }
    parts = [
        f"{labels[name]}={metrics[name]:.3f}"
        for name in labels
        if name in metrics
    ]
    if parts:
        split = ""
        if validated.get("n_blocks") is not None:
            split = (
                f", on {validated['n_blocks']} held-out spatial blocks of "
                f"{validated['block_size_m']:.0f} m"
            )
        basis = "held-out data" if validated["held_out"] else NOT_COMPUTED
        items.append(f"Validation ({basis}): " + ", ".join(parts) + split)

    extrapolation = validated.get("extrapolation")
    if extrapolation:
        items.append(
            f"{extrapolation['fraction']:.1%} of target pixels lie outside the "
            "training envelope on at least one predictor; a random forest "
            "cannot extrapolate beyond it, so those pixels are pinned to the "
            "edge of what the model has seen."
        )
    items.extend(str(note) for note in validated.get("notes", []))

    wrapped: list[str] = []
    for item in items:
        wrapped.extend(
            textwrap.wrap(
                " ".join(item.split()),
                width=CAPTION_WRAP_CHARS,
                initial_indent="- ",
                subsequent_indent="  ",
            )
        )
    return "\n".join(wrapped)


# =============================================================================
# scikit-learn side (inspectable importance, honest cross-validation)
# =============================================================================


def fit_sklearn_rf(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    train_mask: "np.ndarray | None" = None,
    predictors: Sequence[str] | None = None,
    response: str | None = None,
) -> dict[str, Any]:
    """Fit a scikit-learn random forest with the Earth Engine hyper-parameters.

    The point of having both implementations is that ``sklearn`` exposes
    permutation importance and per-fold scores, which ``ee.Classifier`` does
    not, while Earth Engine is what can actually paint a district-wide raster.
    They are configured from the same ``prediction.rf`` block so the comparison
    is meaningful.

    Args:
        frame: Training sample, one row per pixel.
        params: Parsed params mapping.
        train_mask: Boolean row mask; ``None`` fits on every row.
        predictors: Override for ``prediction.rf.predictors``.
        response: Override for ``prediction.rf.response``.

    Returns:
        Mapping with ``model``, ``predictors``, ``response`` and ``n_train``.

    Raises:
        ValueError: If a column is absent, or fewer than
            ``prediction.rf.min_sample_rows`` usable rows remain.
    """
    import numpy as np  # Deferred: see module docstring.
    from sklearn.ensemble import RandomForestRegressor

    settings = resolve_rf_settings(params)
    names = resolve_predictors(predictors, params)
    target = str(settings["response"] if response is None else response)

    missing = [n for n in [*names, target] if n not in frame.columns]
    if missing:
        raise ValueError(
            f"the training frame is missing {missing}; it has "
            f"{sorted(frame.columns)}"
        )

    rows = frame if train_mask is None else frame.loc[np.asarray(train_mask)]
    design = rows[names].to_numpy(dtype=float)
    values = rows[target].to_numpy(dtype=float)
    keep = np.isfinite(design).all(axis=1) & np.isfinite(values)
    if int(keep.sum()) < settings["min_sample_rows"]:
        raise ValueError(
            f"only {int(keep.sum())} usable training row(s), below "
            f"prediction.rf.min_sample_rows ({settings['min_sample_rows']})"
        )

    model = RandomForestRegressor(
        n_estimators=settings["n_trees"],
        min_samples_leaf=settings["min_leaf_population"],
        max_features=settings["variables_per_split"] or "sqrt",
        max_leaf_nodes=settings["max_nodes"],
        random_state=settings["random_seed"],
        n_jobs=-1,
    )
    model.fit(design[keep], values[keep])
    return {
        "model": model,
        "predictors": names,
        "response": target,
        "n_train": int(keep.sum()),
    }


def score_rows(
    fitted: Mapping[str, Any],
    frame: "pd.DataFrame",
    mask: "np.ndarray | None" = None,
) -> dict[str, float]:
    """Score a fitted forest on a row subset.

    Args:
        fitted: Output of :func:`fit_sklearn_rf`.
        frame: Sample containing the predictor and response columns.
        mask: Boolean row mask; ``None`` scores every row.

    Returns:
        Mapping with ``n``, ``rmse`` and ``r2``.

    Raises:
        ValueError: If no usable rows remain.
    """
    import numpy as np  # Deferred: see module docstring.

    names = list(fitted["predictors"])
    target = str(fitted["response"])
    rows = frame if mask is None else frame.loc[np.asarray(mask)]
    design = rows[names].to_numpy(dtype=float)
    values = rows[target].to_numpy(dtype=float)
    keep = np.isfinite(design).all(axis=1) & np.isfinite(values)
    if not keep.any():
        raise ValueError("no usable rows to score")
    predicted = fitted["model"].predict(design[keep])
    return {
        "n": int(keep.sum()),
        "rmse": rmse(values[keep], predicted),
        "r2": r_squared(values[keep], predicted),
    }


def blocked_cv_scores(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    block_column: str = "block_id",
    predictors: Sequence[str] | None = None,
    response: str | None = None,
) -> "pd.DataFrame":
    """Blocked k-fold cross-validation of the random forest.

    Args:
        frame: Training sample carrying a block-id column.
        params: Parsed params mapping.
        block_column: Column holding the spatial block id.
        predictors: Override for ``prediction.rf.predictors``.
        response: Override for ``prediction.rf.response``.

    Returns:
        ``pandas.DataFrame`` with :data:`CV_COLUMNS`, one row per fold. Read the
        **spread** across folds, not just the mean: a large spread says the
        model's performance depends on which part of Colombo it was asked about.

    Raises:
        ValueError: If the block column is absent.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd

    if block_column not in frame.columns:
        raise ValueError(
            f"frame has no {block_column!r} column; add one with "
            "spatial_block_ids()"
        )
    settings = resolve_split(params)
    blocks = frame[block_column].to_numpy()
    require_enough_blocks(blocks, params)

    rows: list[dict[str, Any]] = []
    for index, (train_mask, test_mask) in enumerate(
        blocked_kfold(blocks, settings["n_folds"], settings["seed"])
    ):
        fitted = fit_sklearn_rf(
            frame, params, train_mask=train_mask,
            predictors=predictors, response=response,
        )
        scored = score_rows(fitted, frame, test_mask)
        rows.append(
            {
                "fold": index,
                "n_train": fitted["n_train"],
                "n_test": scored["n"],
                "n_train_blocks": int(np.unique(blocks[train_mask]).size),
                "n_test_blocks": int(np.unique(blocks[test_mask]).size),
                "rmse": scored["rmse"],
                "r2": scored["r2"],
            }
        )
    return pd.DataFrame(rows, columns=list(CV_COLUMNS))


def compare_split_strategies(
    frame: "pd.DataFrame",
    params: dict[str, Any],
    block_column: str = "block_id",
    predictors: Sequence[str] | None = None,
    response: str | None = None,
) -> "pd.DataFrame":
    """Measure how much a random split inflates R2 against a blocked one.

    This is the **only** function here that fits a random split, and it does so
    to quantify the problem rather than to report a score. The gap it returns is
    the number the write-up should quote when it explains why every reported
    metric in this phase comes from blocked held-out data.

    Args:
        frame: Training sample carrying a block-id column.
        params: Parsed params mapping.
        block_column: Column holding the spatial block id.
        predictors: Override for ``prediction.rf.predictors``.
        response: Override for ``prediction.rf.response``.

    Returns:
        ``pandas.DataFrame`` with one row per strategy: ``method``,
        ``n_train``, ``n_test``, ``rmse``, ``r2`` and ``reportable``. Only the
        blocked row has ``reportable`` true.

    Raises:
        ValueError: If the block column is absent.
    """
    import pandas as pd  # Deferred: see module docstring.

    if block_column not in frame.columns:
        raise ValueError(
            f"frame has no {block_column!r} column; add one with "
            "spatial_block_ids()"
        )
    settings = resolve_split(params)
    blocks = frame[block_column].to_numpy()

    splits = {
        "spatial_block": blocked_split(
            blocks, settings["test_fraction"], settings["seed"]
        ),
        RANDOM_SPLIT: random_row_split(
            len(frame), settings["test_fraction"], settings["seed"]
        ),
    }
    rows: list[dict[str, Any]] = []
    for method, (train_mask, test_mask) in splits.items():
        fitted = fit_sklearn_rf(
            frame, params, train_mask=train_mask,
            predictors=predictors, response=response,
        )
        scored = score_rows(fitted, frame, test_mask)
        rows.append(
            {
                "method": method,
                "n_train": fitted["n_train"],
                "n_test": scored["n"],
                "rmse": scored["rmse"],
                "r2": scored["r2"],
                "reportable": method != RANDOM_SPLIT,
            }
        )
    return pd.DataFrame(rows)


def permutation_importance_frame(
    fitted: Mapping[str, Any],
    frame: "pd.DataFrame",
    params: dict[str, Any],
    mask: "np.ndarray | None" = None,
    n_repeats: int = 10,
) -> "pd.DataFrame":
    """Permutation importance on the held-out rows.

    .. note::
        Impurity importance (``model.feature_importances_``) is biased toward
        high-cardinality predictors, and ``lcz_class`` is exactly that. Read
        this instead. It is computed on the rows the model did not see, so it
        measures what the predictor buys in *prediction*, not in fitting.

    Args:
        fitted: Output of :func:`fit_sklearn_rf`.
        frame: Sample containing the predictor and response columns.
        params: Parsed params mapping.
        mask: Boolean row mask selecting the held-out rows.
        n_repeats: Permutation repeats per predictor.

    Returns:
        ``pandas.DataFrame`` with :data:`IMPORTANCE_COLUMNS`, sorted most
        important first.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd
    from sklearn.inspection import permutation_importance

    names = list(fitted["predictors"])
    target = str(fitted["response"])
    rows = frame if mask is None else frame.loc[np.asarray(mask)]
    design = rows[names].to_numpy(dtype=float)
    values = rows[target].to_numpy(dtype=float)
    keep = np.isfinite(design).all(axis=1) & np.isfinite(values)

    result = permutation_importance(
        fitted["model"],
        design[keep],
        values[keep],
        n_repeats=int(n_repeats),
        random_state=int(params["prediction"]["rf"]["random_seed"]),
        n_jobs=-1,
    )
    out = pd.DataFrame(
        {
            "predictor": names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out[list(IMPORTANCE_COLUMNS)]


def compare_rf_implementations(
    ee_importance: Mapping[str, float],
    sklearn_importance: "pd.DataFrame",
) -> "pd.DataFrame":
    """Put the Earth Engine and scikit-learn importances side by side.

    Agreement on the *ordering* is the useful signal; the two are not on the
    same scale and should never be differenced.

    Args:
        ee_importance: The ``importance`` mapping from
            :func:`ee_rf_explain`.
        sklearn_importance: Output of :func:`permutation_importance_frame`.

    Returns:
        ``pandas.DataFrame`` with ``predictor``, ``ee_importance``,
        ``ee_rank``, ``sklearn_importance``, ``sklearn_rank`` and
        ``rank_difference``.
    """
    import pandas as pd  # Deferred: see module docstring.

    ee_frame = (
        pd.DataFrame(
            {
                "predictor": list(ee_importance),
                "ee_importance": [float(v) for v in ee_importance.values()],
            }
        )
        .sort_values("ee_importance", ascending=False)
        .reset_index(drop=True)
    )
    ee_frame["ee_rank"] = ee_frame.index + 1

    sk_frame = sklearn_importance.rename(
        columns={"importance_mean": "sklearn_importance", "rank": "sklearn_rank"}
    )[["predictor", "sklearn_importance", "sklearn_rank"]]

    merged = ee_frame.merge(sk_frame, on="predictor", how="outer")
    merged["rank_difference"] = merged["ee_rank"] - merged["sklearn_rank"]
    return merged.sort_values("sklearn_rank").reset_index(drop=True)


# =============================================================================
# Earth Engine side
# =============================================================================


def prediction_stack(
    params: dict[str, Any],
    epoch: str | None = None,
    region: "ee.Geometry | None" = None,
    source: str | None = None,
) -> "ee.Image":
    """The response band and every random-forest predictor, in one image.

    Built on :func:`colombo_uhi.spatial_stats.covariate_stack`, which already
    yields NDVI, NDBI, built fraction, population density, elevation and
    distance to coast for the Phase 5 regression ladder, then extended with the
    LCZ class.

    .. note::
        The LCZ band is added **here**, not inside ``covariate_stack``. Adding
        it there would change ``spatial_stats.zone_covariate_bands`` and so
        change the column list of every committed Phase 5 export - a silent
        invalidation of results that are already signed off.

    .. warning::
        The response comes from ``prediction.rf.source``, which must name a
        single Landsat sensor family. The Collection 2 inter-calibration was
        tested over Colombo and failed (L7-L8 = -2.48 degC, 3.4x the whole
        26-year trend signal). A model trained on a pooled series would learn
        the sensor changeover as if it were geography.

    Args:
        params: Parsed params mapping.
        epoch: Override for ``prediction.rf.epoch``.
        region: Region the source collection is filtered to; defaults to
            :func:`colombo_uhi.aoi.analysis_region`.
        source: Override for ``prediction.rf.source``.

    Returns:
        ``ee.Image`` carrying the response and every configured predictor.

    Raises:
        ValueError: If a configured predictor is neither produced by
            ``covariate_stack`` nor known to this function.
    """
    from colombo_uhi import aoi, landcover, spatial_stats

    settings = resolve_rf_settings(params)
    epoch_key = str(settings["epoch"] if epoch is None else epoch)
    source_key = str(settings["source"] if source is None else source)
    work_region = aoi.analysis_region(params) if region is None else region

    from_covariates = list(
        spatial_stats.resolve_regression_predictors(None, params)
    )
    extras = {"lcz_class"}
    unknown = [
        name
        for name in settings["predictors"]
        if name not in from_covariates and name not in extras
    ]
    if unknown:
        raise ValueError(
            f"prediction.rf.predictors names {unknown}, which "
            "spatial_stats.covariate_stack does not produce and prediction.py "
            f"does not add. It produces {from_covariates}; this module adds "
            f"{sorted(extras)}."
        )

    stack = spatial_stats.covariate_stack(
        params, epoch_key, work_region, source=source_key
    )
    if "lcz_class" in settings["predictors"]:
        stack = stack.addBands(
            landcover.lcz_class_image(params).rename(["lcz_class"]).toFloat()
        )
    keep = [settings["response"], *settings["predictors"]]
    return stack.select(keep).toFloat().set(
        {
            "epoch": epoch_key,
            "source": source_key,
            "framing": str(params["prediction"]["framing"]),
        }
    )


def training_sample_selectors(params: dict[str, Any]) -> list[str]:
    """Column order of the exported training sample.

    ``Export.table.toDrive`` does not guarantee column order unless
    ``selectors`` is passed, and a silently reordered CSV turns into a
    nonsensical coefficient three steps later.

    Args:
        params: Parsed params mapping.

    Returns:
        ``["x", "y", response, *predictors]``.
    """
    settings = resolve_rf_settings(params)
    return ["x", "y", settings["response"], *settings["predictors"]]


def training_sample_collection(
    params: dict[str, Any],
    epoch: str | None = None,
    region: "ee.Geometry | None" = None,
    source: str | None = None,
    n_pixels: int | None = None,
    seed: int | None = None,
    scale_m: int | None = None,
) -> "ee.FeatureCollection":
    """Sample pixels for the random forest, carrying projected coordinates.

    The ``x`` and ``y`` bands are not decoration: they are what
    :func:`spatial_block_ids` needs to build a spatially blocked split. Sampling
    without them forces a random split, which is the failure this phase exists
    to avoid.

    Args:
        params: Parsed params mapping.
        epoch: Override for ``prediction.rf.epoch``.
        region: Sampling region; defaults to
            :func:`colombo_uhi.aoi.analysis_region`.
        source: Override for ``prediction.rf.source``.
        n_pixels: Override for ``prediction.rf.sample_pixels``.
        seed: Override for ``prediction.rf.random_seed``.
        scale_m: Override for ``prediction.rf.scale_m``.

    Returns:
        ``ee.FeatureCollection``, one feature per sampled pixel.
    """
    import ee  # Deferred: see module docstring.

    from colombo_uhi import aoi

    settings = resolve_rf_settings(params)
    work_region = aoi.analysis_region(params) if region is None else region
    scale = int(settings["scale_m"] if scale_m is None else scale_m)
    crs = str(params["crs"]["analysis_epsg"])

    stack = prediction_stack(params, epoch=epoch, region=work_region, source=source)
    # Coordinates are taken in the ANALYSIS projection, so they are metres and
    # spatial_block_ids can cut square blocks on them. Degrees would make a
    # "2 km" block a different size at every latitude.
    coordinates = ee.Image.pixelCoordinates(ee.Projection(crs)).rename(["x", "y"])
    return stack.addBands(coordinates).sample(
        region=work_region,
        scale=scale,
        projection=ee.Projection(crs),
        numPixels=int(settings["sample_pixels"] if n_pixels is None else n_pixels),
        seed=int(settings["random_seed"] if seed is None else seed),
        dropNulls=True,
        geometries=False,
        tileScale=int(params["composites"]["tile_scale"]),
    )


def export_training_sample(
    params: dict[str, Any],
    epoch: str | None = None,
    region: "ee.Geometry | None" = None,
    source: str | None = None,
    folder: str | None = None,
    suffix: str | None = None,
    start: bool = True,
) -> "ee.batch.Task":
    """Submit the random-forest training sample as a batch export.

    Args:
        params: Parsed params mapping.
        epoch: Override for ``prediction.rf.epoch``.
        region: Sampling region.
        source: Override for ``prediction.rf.source``.
        folder: Drive folder; defaults to ``exports.drive_folder``.
        suffix: Name discriminator; defaults to ``"{epoch}_{source}"``.
        start: Submit the task.

    Returns:
        The ``ee.batch.Task``.
    """
    from colombo_uhi import exports

    settings = resolve_rf_settings(params)
    epoch_key = str(settings["epoch"] if epoch is None else epoch)
    source_key = str(settings["source"] if source is None else source)
    features = training_sample_collection(
        params, epoch=epoch_key, region=region, source=source_key
    )
    return exports.table_to_drive(
        features,
        product="rf_training_sample",
        aoi="district",
        params=params,
        file_format="CSV",
        selectors=training_sample_selectors(params),
        folder=folder,
        res_m=settings["scale_m"],
        suffix=suffix if suffix is not None else f"{epoch_key}_{source_key}",
        start=start,
    )


def read_training_sample(
    path: str | Path, params: dict[str, Any], add_blocks: bool = True
) -> "pd.DataFrame":
    """Read the exported training sample and attach spatial block ids.

    Args:
        path: Path to the downloaded ``.csv``.
        params: Parsed params mapping.
        add_blocks: Add a ``block_id`` column from ``x``/``y``.

    Returns:
        ``pandas.DataFrame`` with ``x``, ``y``, the response, every predictor,
        and (by default) ``block_id``. Rows with a non-finite response or
        predictor are dropped, and the number dropped is reported through a
        warning rather than passed on silently.

    Raises:
        ValueError: If a required column is absent, or fewer than
            ``prediction.rf.min_sample_rows`` usable rows remain.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd

    settings = resolve_rf_settings(params)
    wanted = training_sample_selectors(params)

    frame = pd.read_csv(path)
    missing = [name for name in wanted if name not in frame.columns]
    if missing:
        raise ValueError(
            f"the exported training sample is missing {missing}; it has "
            f"{sorted(frame.columns)}. Check that export_training_sample ran "
            "with the same prediction.rf.predictors list."
        )

    out = frame[wanted].apply(pd.to_numeric, errors="coerce")
    usable = np.isfinite(out.to_numpy(dtype=float)).all(axis=1)
    dropped = int((~usable).sum())
    if dropped:
        warnings.warn(
            f"dropped {dropped} of {len(out)} sampled pixel(s) with a "
            "non-finite response or predictor",
            stacklevel=2,
        )
    out = out.loc[usable].reset_index(drop=True)
    if len(out) < settings["min_sample_rows"]:
        raise ValueError(
            f"only {len(out)} usable sampled pixel(s), below "
            f"prediction.rf.min_sample_rows ({settings['min_sample_rows']}). "
            "Raise prediction.rf.sample_pixels or check the export for a "
            "predictor that came back entirely masked."
        )

    if add_blocks:
        out["block_id"] = spatial_block_ids(
            out["x"].to_numpy(),
            out["y"].to_numpy(),
            resolve_split(params)["block_size_m"],
        )
    return out


def fit_ee_rf(
    sample: "ee.FeatureCollection",
    params: dict[str, Any],
    predictors: Sequence[str] | None = None,
    response: str | None = None,
) -> "ee.Classifier":
    """Train the server-side random forest in REGRESSION mode.

    Args:
        sample: Training features, one per pixel.
        params: Parsed params mapping.
        predictors: Override for ``prediction.rf.predictors``.
        response: Override for ``prediction.rf.response``.

    Returns:
        A trained ``ee.Classifier`` whose output is a continuous LST estimate.
    """
    import ee  # Deferred: see module docstring.

    settings = resolve_rf_settings(params)
    names = resolve_predictors(predictors, params)
    target = str(settings["response"] if response is None else response)

    classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=settings["n_trees"],
        variablesPerSplit=settings["variables_per_split"],
        minLeafPopulation=settings["min_leaf_population"],
        bagFraction=settings["bag_fraction"],
        maxNodes=settings["max_nodes"],
        seed=settings["random_seed"],
    ).setOutputMode("REGRESSION")
    return classifier.train(
        features=sample, classProperty=target, inputProperties=names
    )


def ee_rf_explain(classifier: "ee.Classifier") -> dict[str, Any]:
    """Pull the server-side forest's importance and out-of-bag error.

    .. note::
        This is a single ``getInfo()``, never inside a loop. The importance it
        returns is impurity-based and therefore biased toward high-cardinality
        predictors; read :func:`permutation_importance_frame` as the headline
        and this as a cross-check.

    Args:
        classifier: A trained ``ee.Classifier``.

    Returns:
        The ``explain()`` mapping, typically carrying ``importance``,
        ``numberOfTrees`` and ``outOfBagErrorEstimate``.
    """
    return dict(classifier.explain().getInfo())


def project_lst_image(
    stack: "ee.Image",
    classifier: "ee.Classifier",
    params: dict[str, Any],
    predictors: Sequence[str] | None = None,
    band_name: str | None = None,
) -> "ee.Image":
    """Apply a trained forest to a predictor stack.

    **The result is a conditional scenario projection, not a forecast.** When
    ``stack`` carries projected predictors, the model is being asked a
    space-for-time question: it learned where it is hot *given today's drivers*,
    and it has never seen a year change.

    Args:
        stack: Image carrying every predictor the classifier was trained on.
        classifier: Trained ``ee.Classifier`` in REGRESSION mode.
        params: Parsed params mapping.
        predictors: Override for ``prediction.rf.predictors``.
        band_name: Output band name; defaults to ``"<response>_projected"``.

    Returns:
        Single-band ``ee.Image``. The band is named explicitly rather than left
        as Earth Engine's default ``"classification"``, which reads as a class
        code and is not what this is.
    """
    settings = resolve_rf_settings(params)
    names = resolve_predictors(predictors, params)
    output = str(band_name or f"{settings['response']}_projected")
    return (
        stack.select(names)
        .classify(classifier, output)
        .toFloat()
        .set({"framing": str(params["prediction"]["framing"])})
    )


def export_projection(
    image: "ee.Image",
    params: dict[str, Any],
    report: Mapping[str, Any] | None,
    region: "ee.Geometry",
    product: str = "lst_projection",
    scenario: str | None = None,
    year: int | None = None,
    scale_m: int | None = None,
    folder: str | None = None,
    suffix: str | None = None,
    start: bool = True,
) -> "ee.batch.Task":
    """Export a projected surface - **only** if it has been validated.

    :func:`require_validated` runs first, before any Earth Engine call, so a
    product without computed validation metrics never reaches ``ee.batch``.

    Args:
        image: The projected image.
        params: Parsed params mapping.
        report: Output of :func:`build_validation_report`.
        region: Export region.
        product: Product name for :func:`colombo_uhi.exports.export_name`.
        scenario: Scenario key, folded into the filename suffix.
        year: Effective projection year, folded into the filename suffix.
        scale_m: Override for ``prediction.rf.scale_m``.
        folder: Drive folder; defaults to ``exports.drive_folder``.
        suffix: Explicit suffix, overriding the scenario/year default.
        start: Submit the task.

    Returns:
        The ``ee.batch.Task``.

    Raises:
        ValidationMissing: If the report is absent or incomplete.
    """
    require_validated(report, params)

    from colombo_uhi import exports

    settings = resolve_rf_settings(params)
    scale = int(settings["scale_m"] if scale_m is None else scale_m)
    parts = [part for part in (scenario, None if year is None else str(year)) if part]
    return exports.image_to_drive(
        image,
        product=product,
        aoi="district",
        params=params,
        region=region,
        scale_m=scale,
        folder=folder,
        suffix=suffix if suffix is not None else ("_".join(parts) or None),
        start=start,
    )


def predictor_band_order(params: dict[str, Any]) -> list[str]:
    """Band order of the exported predictor raster.

    A GeoTIFF's band order is not self-describing, so it is declared here, used
    by :func:`export_predictor_raster` and read back by
    :func:`read_predictor_raster`. The two cannot drift.

    Args:
        params: Parsed params mapping.

    Returns:
        ``[response, *predictors]``.
    """
    settings = resolve_rf_settings(params)
    return [settings["response"], *settings["predictors"]]


def export_predictor_raster(
    params: dict[str, Any],
    region: "ee.Geometry",
    epoch: str | None = None,
    source: str | None = None,
    scale_m: int | None = None,
    folder: str | None = None,
    suffix: str | None = None,
    start: bool = True,
) -> "ee.batch.Task":
    """Submit the observed response-and-predictor stack as a raster.

    This is what makes Part 2 of the notebook runnable **without Earth Engine**:
    with the predictors on disk as arrays, the projected surface can be painted,
    scored and plotted in numpy, using the same scikit-learn model the blocked
    cross-validation validated.

    Args:
        params: Parsed params mapping.
        region: Export region.
        epoch: Override for ``prediction.rf.epoch``.
        source: Override for ``prediction.rf.source``.
        scale_m: Override for ``prediction.rf.scale_m``.
        folder: Drive folder; defaults to ``exports.drive_folder``.
        suffix: Name discriminator; defaults to ``"{epoch}_{source}"``.
        start: Submit the task.

    Returns:
        The ``ee.batch.Task``.
    """
    from colombo_uhi import exports

    settings = resolve_rf_settings(params)
    epoch_key = str(settings["epoch"] if epoch is None else epoch)
    source_key = str(settings["source"] if source is None else source)
    scale = int(settings["scale_m"] if scale_m is None else scale_m)
    stack = prediction_stack(
        params, epoch=epoch_key, region=region, source=source_key
    )
    return exports.image_to_drive(
        stack,
        product="predictor_stack",
        aoi="district",
        params=params,
        region=region,
        band_order=predictor_band_order(params),
        scale_m=scale,
        folder=folder,
        suffix=suffix if suffix is not None else f"{epoch_key}_{source_key}",
        start=start,
    )


def read_predictor_raster(
    path: str | Path, params: dict[str, Any]
) -> tuple[dict[str, "np.ndarray"], dict[str, Any]]:
    """Read the exported predictor raster into named arrays.

    Args:
        path: Path to the downloaded ``.tif``.
        params: Parsed params mapping.

    Returns:
        ``(arrays, profile)`` where ``arrays`` maps band name to a 2-D float
        array with the raster's nodata replaced by ``nan``.

    Raises:
        ValueError: If the band count does not match
            :func:`predictor_band_order` - which would silently mislabel every
            predictor, since band order is not carried in the file.
    """
    import numpy as np  # Deferred: see module docstring.
    import rasterio

    names = predictor_band_order(params)
    with rasterio.open(str(path)) as handle:
        if handle.count != len(names):
            raise ValueError(
                f"{path} has {handle.count} band(s) but "
                f"prediction.rf declares {len(names)}: {names}. Band order is "
                "not stored in a GeoTIFF, so a mismatch would rename every "
                "predictor. Re-export with export_predictor_raster."
            )
        arrays = {
            name: handle.read(index + 1, masked=True).filled(np.nan).astype(float)
            for index, name in enumerate(names)
        }
        profile = dict(handle.profile)
    return arrays, profile


def raster_sample_frame(
    arrays: Mapping[str, "np.ndarray"],
    labels: "np.ndarray | None" = None,
    label_column: str = "lulc_class",
) -> "pd.DataFrame":
    """Flatten named rasters into one row per pixel.

    Args:
        arrays: Mapping of band name to 2-D array, all the same shape.
        labels: Optional class-code array to attach.
        label_column: Column name for ``labels``.

    Returns:
        ``pandas.DataFrame`` with one column per band, one row per pixel, in
        row-major order - so ``frame[name].to_numpy().reshape(shape)`` recovers
        the raster.

    Raises:
        ValueError: If the arrays do not all share one shape.
    """
    import numpy as np  # Deferred: see module docstring.
    import pandas as pd

    if not arrays:
        raise ValueError("at least one array is required")
    shapes = {name: np.asarray(a).shape for name, a in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"arrays must share one shape, got {shapes}")

    frame = pd.DataFrame(
        {name: np.asarray(a, dtype=float).ravel() for name, a in arrays.items()}
    )
    if labels is not None:
        grid = np.asarray(labels)
        if grid.shape != next(iter(shapes.values())):
            raise ValueError(
                f"labels has shape {grid.shape}, expected "
                f"{next(iter(shapes.values()))}"
            )
        frame[label_column] = grid.ravel()
    return frame


def predict_surface(
    fitted: Mapping[str, Any],
    arrays: Mapping[str, "np.ndarray"],
    params: dict[str, Any],
) -> "np.ndarray":
    """Apply a fitted forest across a predictor raster stack.

    **The result is a conditional scenario projection, not a forecast**, and
    when ``arrays`` carries projected predictors it is also a space-for-time
    substitution: the model learned where it is hot *given today's drivers* and
    has never seen a year change.

    Args:
        fitted: Output of :func:`fit_sklearn_rf`.
        arrays: Mapping of predictor name to 2-D array.
        params: Parsed params mapping.

    Returns:
        2-D float array of predicted values. Pixels with a non-finite predictor
        are ``nan``, never an imputed value - a forest given a filled-in
        predictor returns a confident number built on an invention.

    Raises:
        ValueError: If a predictor is missing or the arrays differ in shape.
    """
    import numpy as np  # Deferred: see module docstring.

    names = list(fitted["predictors"])
    missing = [name for name in names if name not in arrays]
    if missing:
        raise ValueError(
            f"the predictor stack is missing {missing}; it holds "
            f"{sorted(arrays)}"
        )
    shapes = {name: np.asarray(arrays[name]).shape for name in names}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"predictor arrays must share one shape, got {shapes}")

    shape = next(iter(shapes.values()))
    design = np.column_stack(
        [np.asarray(arrays[name], dtype=float).ravel() for name in names]
    )
    usable = np.isfinite(design).all(axis=1)
    out = np.full(design.shape[0], np.nan, dtype=float)
    if usable.any():
        out[usable] = fitted["model"].predict(design[usable])
    return out.reshape(shape)


def lulc_class_image(
    params: dict[str, Any],
    year: int,
    region: "ee.Geometry | None" = None,
    scale_m: int | None = None,
) -> "ee.Image":
    """Modal land cover for one year, aggregated to the CA grid.

    .. warning::
        A single-band 0-to-8 GeoTIFF cannot distinguish "class 0, water" from
        "never classified" - masked pixels are written as 0. Phase 5 lost a run
        to exactly that (Dynamic World "green" appeared to grow 5.4x from 2016
        to 2024, almost all of it Sentinel-2 coverage). So this returns **two**
        bands, ``label`` and ``observed``, and :func:`read_lulc_raster` hands
        them back separately.

    Args:
        params: Parsed params mapping.
        year: Calendar year to composite.
        region: Region to filter to; defaults to
            :func:`colombo_uhi.aoi.analysis_region`.
        scale_m: Override for ``prediction.ca_markov.raster_scale_m``.

    Returns:
        Two-band ``ee.Image`` (``label``, ``observed``) on the analysis CRS at
        the CA grid scale, with its projection pinned. Leaving the projection to
        be inherited is what made a Phase 5 export serialise at 159 MB.

    Raises:
        ValueError: If the configured scheme is not Dynamic World, which is the
            only per-year scheme with the temporal depth this needs.
    """
    import ee  # Deferred: see module docstring.

    from colombo_uhi import aoi, landcover

    cfg = params["prediction"]["ca_markov"]
    scheme = str(cfg["scheme"])
    if scheme != "dynamic_world":
        raise ValueError(
            f"prediction.ca_markov.scheme is {scheme!r}; the CA needs a "
            "per-year classification and only 'dynamic_world' provides one. "
            "WorldCover has two epochs and LCZ has one."
        )

    work_region = aoi.analysis_region(params) if region is None else region
    crs = str(params["crs"]["analysis_epsg"])
    native = int(params["datasets"]["dynamic_world"]["scale_m"])
    target = int(cfg["raster_scale_m"] if scale_m is None else scale_m)

    labels = (
        landcover.dynamic_world_mode(params, int(year), region=work_region)
        .setDefaultProjection(crs=crs, scale=native)
    )
    # "observed" is the honest companion to the labels: 1 where Dynamic World
    # actually classified the year, 0 where it did not. Built before the
    # aggregation so it measures Sentinel-2 coverage, not aggregation artefacts.
    observed = labels.mask().reduce(ee.Reducer.min()).gt(0).rename(["observed"])

    aggregated = (
        labels.reduceResolution(reducer=ee.Reducer.mode(), maxPixels=1024)
        .reproject(crs=crs, scale=target)
        .rename(["label"])
        .toInt16()
    )
    coverage = (
        observed.setDefaultProjection(crs=crs, scale=native)
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
        .reproject(crs=crs, scale=target)
        .gte(0.5)
        .rename(["observed"])
        .toInt16()
    )
    return aggregated.addBands(coverage).set(
        {"year": int(year), "scheme": scheme, "scale_m": target}
    )


def export_lulc_raster(
    params: dict[str, Any],
    year: int,
    region: "ee.Geometry",
    scale_m: int | None = None,
    folder: str | None = None,
    suffix: str | None = None,
    start: bool = True,
) -> "ee.batch.Task":
    """Submit one year's land-cover raster as a batch export.

    This is the file both the Python cellular automaton and MOLUSCE read. Band
    order is passed explicitly because a GeoTIFF's band order is not
    self-describing.

    Args:
        params: Parsed params mapping.
        year: Calendar year.
        region: Export region.
        scale_m: Override for ``prediction.ca_markov.raster_scale_m``.
        folder: Drive folder; defaults to ``exports.drive_folder``.
        suffix: Name discriminator; defaults to the year.
        start: Submit the task.

    Returns:
        The ``ee.batch.Task``.
    """
    from colombo_uhi import exports

    cfg = params["prediction"]["ca_markov"]
    scale = int(cfg["raster_scale_m"] if scale_m is None else scale_m)
    image = lulc_class_image(params, year, region=region, scale_m=scale)
    return exports.image_to_drive(
        image,
        product="lulc",
        aoi="district",
        params=params,
        region=region,
        band_order=["label", "observed"],
        scale_m=scale,
        folder=folder,
        suffix=suffix if suffix is not None else str(int(year)),
        start=start,
    )


def read_lulc_raster(
    path: str | Path, params: dict[str, Any]
) -> tuple["np.ndarray", "np.ndarray", dict[str, Any]]:
    """Read an exported land-cover raster, keeping labels and coverage apart.

    Args:
        path: Path to the downloaded ``.tif``.
        params: Parsed params mapping.

    Returns:
        ``(labels, observed, profile)``. ``labels`` is an integer array,
        ``observed`` a boolean array that is ``False`` wherever Dynamic World
        never classified the cell, and ``profile`` the rasterio profile.

    Raises:
        ValueError: If the file does not carry both bands - which usually means
            it was written before ``lulc_class_image`` emitted ``observed``, and
            using it would silently treat unclassified cells as water.
    """
    import numpy as np  # Deferred: see module docstring.
    import rasterio

    with rasterio.open(str(path)) as handle:
        if handle.count < 2:
            raise ValueError(
                f"{path} has {handle.count} band(s); a land-cover raster for "
                "the CA must carry BOTH 'label' and 'observed'. Without the "
                "coverage band, cells Dynamic World never classified are "
                "written as 0 and read back as water."
            )
        labels = handle.read(1)
        observed = handle.read(2)
        profile = dict(handle.profile)

    return (
        np.asarray(labels).astype(np.int16),
        np.asarray(observed).astype(bool),
        profile,
    )


def write_lulc_projection(
    labels: "np.ndarray",
    profile: Mapping[str, Any],
    path: str | Path,
    params: dict[str, Any],
    report: Mapping[str, Any] | None,
) -> Path:
    """Write a projected land-cover raster - **only** if it has been validated.

    Args:
        labels: Projected class-code array.
        profile: Rasterio profile from :func:`read_lulc_raster`.
        path: Output ``.tif`` path.
        params: Parsed params mapping.
        report: Output of :func:`build_validation_report`.

    Returns:
        The written path.

    Raises:
        ValidationMissing: If the report is absent or incomplete.
    """
    require_validated(report, params)

    import numpy as np  # Deferred: see module docstring.
    import rasterio

    array = np.asarray(labels).astype(np.int16)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    out_profile = dict(profile)
    out_profile.update(
        {"count": 1, "dtype": "int16", "height": array.shape[0],
         "width": array.shape[1]}
    )
    with rasterio.open(destination, "w", **out_profile) as handle:
        handle.write(array, 1)
        handle.update_tags(
            framing=str(params["prediction"]["framing"]),
            validation=str(dict(report or {}).get("metrics", {})),
        )
    return destination


def ghsl_built_cross_check(
    params: dict[str, Any], year: int | None = None
) -> "ee.Image":
    """GHSL built surface for an independent comparison against the CA.

    ``JRC/GHSL/P2023A/GHS_BUILT_S`` runs to 2030, so the cellular automaton's
    projected built area has an outside opinion for that horizon.

    .. note::
        This is reported as **agreement or disagreement**, never used to correct
        the CA. Tuning the automaton until it matches GHSL would turn an
        independent check into a fitted parameter.

    Args:
        params: Parsed params mapping.
        year: Override for ``prediction.ghsl_cross_check_year``.

    Returns:
        Single-band ``ee.Image`` of built fraction, from
        :func:`colombo_uhi.uhi_metrics.built_up_fraction`.
    """
    from colombo_uhi import uhi_metrics

    epoch = int(
        params["prediction"]["ghsl_cross_check_year"] if year is None else year
    )
    return uhi_metrics.built_up_fraction(params, epoch)
