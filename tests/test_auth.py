"""Tests for colombo_uhi.auth (pure-Python parts only, no Earth Engine).

``ee`` is faked via ``sys.modules`` injection, so these tests run on machines
without earthengine-api installed — including this repo's local dev machine.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from colombo_uhi import auth


@pytest.fixture(autouse=True)
def reset_auth_state(monkeypatch: pytest.MonkeyPatch):
    """Isolate module-level idempotency state and the EE_PROJECT env var."""
    monkeypatch.setattr(auth, "_initialized_project", None)
    monkeypatch.delenv(auth.EE_PROJECT_ENV_VAR, raising=False)
    yield


class FakeEE:
    """Minimal stand-in for the ``ee`` module."""

    def __init__(self, fail_first_init: bool = False) -> None:
        self.init_calls: list[dict[str, Any]] = []
        self.auth_calls = 0
        self._fail_first_init = fail_first_init

    def Initialize(self, project: str | None = None) -> None:  # noqa: N802 (ee API name)
        self.init_calls.append({"project": project})
        if self._fail_first_init and len(self.init_calls) == 1:
            raise RuntimeError("no credentials (simulated)")

    def Authenticate(self) -> None:  # noqa: N802 (ee API name)
        self.auth_calls += 1


@pytest.fixture()
def fake_ee(monkeypatch: pytest.MonkeyPatch) -> FakeEE:
    fake = FakeEE()
    monkeypatch.setitem(sys.modules, "ee", fake)  # type: ignore[arg-type]
    return fake


# --- resolve_project_id -------------------------------------------------------


def test_explicit_argument_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(auth.EE_PROJECT_ENV_VAR, "env-project")
    params = {"gcp": {"ee_project_id": "params-project"}}
    assert auth.resolve_project_id("arg-project", params) == "arg-project"


def test_env_var_beats_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(auth.EE_PROJECT_ENV_VAR, "env-project")
    params = {"gcp": {"ee_project_id": "params-project"}}
    assert auth.resolve_project_id(None, params) == "env-project"


def test_params_used_as_fallback() -> None:
    params = {"gcp": {"ee_project_id": "params-project"}}
    assert auth.resolve_project_id(None, params) == "params-project"


def test_whitespace_only_values_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(auth.EE_PROJECT_ENV_VAR, "   ")
    params = {"gcp": {"ee_project_id": "params-project"}}
    assert auth.resolve_project_id(None, params) == "params-project"


def test_missing_everywhere_raises_actionable_error() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        auth.resolve_project_id(None, {"gcp": {"ee_project_id": None}})
    message = str(excinfo.value)
    assert auth.EE_PROJECT_ENV_VAR in message
    assert "params.yaml" in message
    assert auth.EE_REGISTER_URL in message


def test_none_params_is_accepted() -> None:
    with pytest.raises(RuntimeError):
        auth.resolve_project_id(None, None)


# --- init_ee ------------------------------------------------------------------


def test_init_ee_initialises_with_resolved_project(fake_ee: FakeEE) -> None:
    project = auth.init_ee(project_id="my-project")
    assert project == "my-project"
    assert fake_ee.init_calls == [{"project": "my-project"}]
    assert fake_ee.auth_calls == 0


def test_init_ee_is_idempotent(fake_ee: FakeEE) -> None:
    auth.init_ee(project_id="my-project")
    auth.init_ee(project_id="ignored-second-time")
    assert len(fake_ee.init_calls) == 1, "second call must not re-initialise"


def test_init_ee_force_reinitialises(fake_ee: FakeEE) -> None:
    auth.init_ee(project_id="my-project")
    project = auth.init_ee(project_id="other-project", force=True)
    assert project == "other-project"
    assert len(fake_ee.init_calls) == 2


def test_init_ee_authenticates_then_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeEE(fail_first_init=True)
    monkeypatch.setitem(sys.modules, "ee", fake)  # type: ignore[arg-type]
    project = auth.init_ee(project_id="my-project")
    assert project == "my-project"
    assert fake.auth_calls == 1
    assert len(fake.init_calls) == 2


def test_init_ee_wraps_persistent_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_fail(project: str | None = None) -> None:
        raise RuntimeError("project not registered (simulated)")

    fake = SimpleNamespace(Initialize=always_fail, Authenticate=lambda: None)
    monkeypatch.setitem(sys.modules, "ee", fake)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError) as excinfo:
        auth.init_ee(project_id="bad-project")
    message = str(excinfo.value)
    assert "bad-project" in message
    assert auth.EE_REGISTER_URL in message
    # A failed init must NOT poison the idempotency flag.
    assert auth._initialized_project is None


def test_init_ee_uses_env_var_when_no_argument(
    fake_ee: FakeEE, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(auth.EE_PROJECT_ENV_VAR, "env-project")
    assert auth.init_ee() == "env-project"
    assert fake_ee.init_calls == [{"project": "env-project"}]
