"""Earth Engine authentication and initialisation helpers for Google Colab.

Design notes:
    * ``import ee`` is deferred into function bodies so this module (and the
      local pytest suite) imports cleanly on machines without
      ``earthengine-api`` installed. Claude Code cannot run Earth Engine;
      only the user's authenticated Colab session can.
    * :func:`init_ee` is idempotent per Python session — re-running a Colab
      cell never re-prompts for authentication. Pass ``force=True`` after
      switching Google accounts or Cloud projects.
"""

from __future__ import annotations

import os
from typing import Any

from colombo_uhi import load_params

#: Environment variable checked for the Earth Engine Cloud project id.
EE_PROJECT_ENV_VAR = "EE_PROJECT"

#: Where to register a (free, noncommercial) Earth Engine Cloud project.
EE_REGISTER_URL = "https://code.earthengine.google.com/register"

# Module-level session state: the project id EE was initialised with, or None.
_initialized_project: str | None = None


def resolve_project_id(
    project_id: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """Resolve the Earth Engine Cloud project id to use.

    Precedence (first non-empty value wins):

    1. the explicit ``project_id`` argument,
    2. the ``EE_PROJECT`` environment variable,
    3. ``gcp.ee_project_id`` in ``config/params.yaml``.

    Args:
        project_id: Explicit project id, if the caller has one.
        params: Parsed params mapping (as returned by
            :func:`colombo_uhi.load_params`); optional.

    Returns:
        The resolved project id.

    Raises:
        RuntimeError: If no project id is configured anywhere, with
            instructions for all three configuration options.
    """
    from_params = None
    if params is not None:
        from_params = params.get("gcp", {}).get("ee_project_id")

    for candidate in (project_id, os.environ.get(EE_PROJECT_ENV_VAR), from_params):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    raise RuntimeError(
        "No Earth Engine Cloud project id configured. Set one of the "
        "following (checked in this order):\n"
        '  1. init_ee(project_id="your-project-id")\n'
        f'  2. the {EE_PROJECT_ENV_VAR} environment variable '
        f'(in Colab: os.environ["{EE_PROJECT_ENV_VAR}"] = "your-project-id")\n'
        "  3. gcp.ee_project_id in config/params.yaml\n"
        f"No project yet? Register free for noncommercial use: {EE_REGISTER_URL}"
    )


def init_ee(
    project_id: str | None = None,
    params_path: str | os.PathLike[str] | None = None,
    force: bool = False,
) -> str:
    """Authenticate (if needed) and initialise the Earth Engine session.

    Idempotent: once initialisation succeeds, later calls return immediately
    without touching the network, so re-running a Colab cell is safe. The
    first call in a fresh Colab VM may open a Google authentication prompt.

    Args:
        project_id: Explicit EE Cloud project id; overrides the environment
            variable and params.yaml (see :func:`resolve_project_id`).
        params_path: Optional explicit path to params.yaml.
        force: Re-initialise even if a session already exists (e.g. after
            switching accounts or projects).

    Returns:
        The project id the session is initialised with.

    Raises:
        RuntimeError: If no project id is configured, or if initialisation
            fails even after authenticating.
    """
    global _initialized_project

    if _initialized_project is not None and not force:
        return _initialized_project

    params: dict[str, Any] | None
    try:
        params = load_params(params_path)
    except FileNotFoundError:
        # Not fatal here: the id may come from the argument or the env var.
        params = None

    project = resolve_project_id(project_id, params)

    import ee  # Deferred: keeps this module importable without earthengine-api.

    try:
        ee.Initialize(project=project)
    except Exception:
        # Fresh VM / no cached credentials: authenticate, then retry once.
        ee.Authenticate()
        try:
            ee.Initialize(project=project)
        except Exception as exc:
            raise RuntimeError(
                f"ee.Initialize failed for project '{project}' even after "
                "authentication. Common causes: the project is not registered "
                f"for Earth Engine ({EE_REGISTER_URL}), the Earth Engine API "
                "is not enabled on it, or your Google account has no access "
                f"to it. Original error: {exc}"
            ) from exc

    _initialized_project = project
    return project


def ee_smoke_test(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run trivial Earth Engine calls to confirm the session works.

    Evaluates ``ee.Number(1).add(1)`` and fetches the SRTM band names (the
    dataset id comes from params.yaml, so this also sanity-checks the config).

    Args:
        params: Parsed params mapping; loaded from the default location if
            omitted.

    Returns:
        Dict with keys ``one_plus_one``, ``srtm_id``, ``srtm_bands`` and
        ``ok`` (True when both results match expectations).

    Raises:
        ee.EEException: If the session is not initialised (run
            :func:`init_ee` first).
    """
    import ee  # Deferred: see module docstring.

    if params is None:
        params = load_params()

    one_plus_one = ee.Number(1).add(1).getInfo()
    srtm_id = params["datasets"]["srtm"]["id"]
    srtm_bands = ee.Image(srtm_id).bandNames().getInfo()

    return {
        "one_plus_one": one_plus_one,
        "srtm_id": srtm_id,
        "srtm_bands": srtm_bands,
        "ok": one_plus_one == 2 and srtm_bands == ["elevation"],
    }
