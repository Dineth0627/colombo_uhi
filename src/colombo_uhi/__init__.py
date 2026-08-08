"""Colombo Urban Heat Island analysis package.

Remote-sensing analysis of Land Surface Temperature (LST) trends in Colombo,
Sri Lanka (2000-2025) on free Google Earth Engine data. Notebooks under
``notebooks/`` orchestrate; all logic lives in this package.

Every constant (dataset IDs, band names, scale factors, thresholds, CRS) comes
from ``config/params.yaml`` — the single source of truth. Load it with
:func:`load_params`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

__version__ = "0.1.0"

#: Environment variable that may point at an alternative params.yaml.
PARAMS_ENV_VAR = "COLOMBO_UHI_PARAMS"


def repo_root() -> Path:
    """Return the repository root directory.

    The package lives at ``<repo>/src/colombo_uhi``, so the root is two
    levels above this file's directory.

    Returns:
        Absolute path to the repository root.
    """
    return Path(__file__).resolve().parents[2]


def load_params(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load ``config/params.yaml``, the project's single source of truth.

    Resolution order for the file location:

    1. explicit ``path`` argument,
    2. the ``COLOMBO_UHI_PARAMS`` environment variable,
    3. ``<repo_root>/config/params.yaml``.

    Args:
        path: Optional explicit path to a params YAML file.

    Returns:
        The parsed parameter mapping.

    Raises:
        FileNotFoundError: If no params file exists at the resolved location,
            with instructions on how to fix it.
    """
    if path is not None:
        resolved = Path(path)
    elif os.environ.get(PARAMS_ENV_VAR):
        resolved = Path(os.environ[PARAMS_ENV_VAR])
    else:
        resolved = repo_root() / "config" / "params.yaml"

    if not resolved.is_file():
        raise FileNotFoundError(
            f"params file not found at '{resolved}'. Expected the repository's "
            "config/params.yaml. Run notebooks from the repo root (the setup "
            f"notebook does `%cd` there), pass an explicit path, or set the "
            f"{PARAMS_ENV_VAR} environment variable."
        )

    with open(resolved, encoding="utf-8") as handle:
        params: dict[str, Any] = yaml.safe_load(handle)
    return params
