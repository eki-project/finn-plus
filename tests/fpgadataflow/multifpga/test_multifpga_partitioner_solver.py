"""Tests for the creation and release of the mip solver model used by the partitioners."""

import pytest

import mip
import mip.gurobi
from types import SimpleNamespace
from typing import Any

import finn.transformation.fpgadataflow.multifpga.partitioner as partitioner
from finn.transformation.fpgadataflow.multifpga.partitioner import (
    MIP_SOLVER_MAX_ATTEMPTS_ENV,
    Partitioner,
)
from finn.util.exception import FINNMultiFPGAUserError

ENV_ERROR = "Gurobi environment could not be loaded, check your license."
MODEL_ERROR = "Could not create Gurobi model"


class FakeModelFactory:
    """Stand-in for ``mip.Model`` that fails a given number of times before succeeding."""

    def __init__(self, failures: list[str]) -> None:
        self.failures = failures
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.failures:
            raise mip.exceptions.InterfacingError(self.failures.pop(0))
        return "model"


class FakeDiagnostics:
    """Stand-in for ``Partitioner._gurobi_environment_error`` counting its invocations."""

    def __init__(self, reason: str | None) -> None:
        self.reason = reason
        self.calls = 0

    def __call__(self) -> str | None:
        self.calls += 1
        return self.reason


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []
    monkeypatch.setattr(partitioner.time, "sleep", delays.append)
    monkeypatch.setattr(partitioner.random, "uniform", lambda a, b: 1.0)  # noqa: ARG005
    monkeypatch.setattr(partitioner, "MIP_SOLVER_RETRY_DELAY", 2.0)
    monkeypatch.delenv(MIP_SOLVER_MAX_ATTEMPTS_ENV, raising=False)
    return delays


@pytest.fixture
def diagnostics(monkeypatch: pytest.MonkeyPatch) -> FakeDiagnostics:
    fake = FakeDiagnostics("Gurobi error 10009: No tokens available")
    monkeypatch.setattr(Partitioner, "_gurobi_environment_error", fake)
    return fake


@pytest.mark.multifpga
def test_transient_failure_is_retried_with_backoff(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float], diagnostics: FakeDiagnostics
) -> None:
    factory = FakeModelFactory([ENV_ERROR, ENV_ERROR])
    monkeypatch.setattr(partitioner, "Model", factory)

    assert Partitioner.create_model(mip.GUROBI) == "model"
    assert len(factory.calls) == 3
    assert factory.calls[0] == {"name": "finn_partition", "solver_name": mip.GUROBI}
    assert no_sleep == [2.0, 4.0]
    # Retries do not request an extra token just to explain the failure.
    assert diagnostics.calls == 0


@pytest.mark.multifpga
def test_persistent_failure_raises_after_all_attempts(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float], diagnostics: FakeDiagnostics
) -> None:
    factory = FakeModelFactory([ENV_ERROR] * 10)
    monkeypatch.setattr(partitioner, "Model", factory)
    monkeypatch.setenv(MIP_SOLVER_MAX_ATTEMPTS_ENV, "3")

    with pytest.raises(FINNMultiFPGAUserError) as excinfo:
        Partitioner.create_model(mip.GUROBI)
    assert len(factory.calls) == 3
    assert no_sleep == [2.0, 4.0]
    message = str(excinfo.value)
    assert "after 3 attempt(s)" in message
    assert ENV_ERROR in message
    assert "No tokens available" in message
    assert isinstance(excinfo.value.__cause__, mip.exceptions.InterfacingError)
    # Gurobi is asked for the reason exactly once, when giving up.
    assert diagnostics.calls == 1


@pytest.mark.multifpga
def test_non_transient_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float], diagnostics: FakeDiagnostics
) -> None:
    factory = FakeModelFactory([MODEL_ERROR])
    monkeypatch.setattr(partitioner, "Model", factory)

    with pytest.raises(FINNMultiFPGAUserError, match=MODEL_ERROR):
        Partitioner.create_model(mip.GUROBI)
    assert len(factory.calls) == 1
    assert no_sleep == []
    assert diagnostics.calls == 0


@pytest.mark.multifpga
def test_single_attempt_disables_retries(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float], diagnostics: FakeDiagnostics
) -> None:
    factory = FakeModelFactory([ENV_ERROR])
    monkeypatch.setattr(partitioner, "Model", factory)
    monkeypatch.setenv(MIP_SOLVER_MAX_ATTEMPTS_ENV, "1")

    with pytest.raises(FINNMultiFPGAUserError, match="after 1 attempt"):
        Partitioner.create_model(mip.GUROBI)
    assert len(factory.calls) == 1
    assert no_sleep == []


@pytest.mark.multifpga
def test_diagnostics_only_collected_for_gurobi(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float], diagnostics: FakeDiagnostics
) -> None:
    factory = FakeModelFactory([ENV_ERROR] * 10)
    monkeypatch.setattr(partitioner, "Model", factory)
    monkeypatch.setenv(MIP_SOLVER_MAX_ATTEMPTS_ENV, "2")

    with pytest.raises(FINNMultiFPGAUserError):
        Partitioner.create_model(mip.CBC)
    assert len(factory.calls) == 2
    assert diagnostics.calls == 0


@pytest.mark.multifpga
def test_gurobi_environment_error_without_gurobi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a Gurobi library the diagnostics report nothing instead of failing."""
    monkeypatch.setattr(mip.gurobi, "has_gurobi", False)
    assert Partitioner._gurobi_environment_error() is None  # noqa: SLF001


class FakeGurobiLibrary:
    """Records the free calls python-mip would forward to libgurobi."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.freed_models: list[Any] = []
        self.freed_envs: list[Any] = []
        monkeypatch.setattr(mip.gurobi, "has_gurobi", True)
        monkeypatch.setattr(mip.gurobi, "GRBfreemodel", self.freed_models.append, raising=False)
        monkeypatch.setattr(mip.gurobi, "GRBfreeenv", self.freed_envs.append, raising=False)

    @staticmethod
    def solver(owns_model: bool = True) -> mip.gurobi.SolverGurobi:
        solver = mip.gurobi.SolverGurobi.__new__(mip.gurobi.SolverGurobi)
        solver._ownsModel = owns_model  # noqa: SLF001
        solver._model = "model-handle"  # noqa: SLF001
        solver._env = "env-handle"  # noqa: SLF001
        solver._venv_loaded = True  # noqa: SLF001
        return solver


class MinimalPartitioner(Partitioner):
    """Concrete enough to be instantiated; only the resource handling of the base is tested."""


# ABCMeta computes the abstract set on class creation, so clear it afterwards.
MinimalPartitioner.__abstractmethods__ = frozenset()


def partitioner_with(model: Any) -> MinimalPartitioner:
    """Bypass __init__, which needs a full build configuration and a solver."""
    part = MinimalPartitioner.__new__(MinimalPartitioner)
    part.model = model
    return part


@pytest.mark.multifpga
def test_close_frees_gurobi_model_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    lib = FakeGurobiLibrary(monkeypatch)
    solver = lib.solver()
    part = partitioner_with(SimpleNamespace(solver=solver))

    part.close()
    assert lib.freed_models == ["model-handle"]
    assert lib.freed_envs == ["env-handle"]
    # The handles are cleared so that neither a second release nor python-mip's own
    # SolverGurobi.__del__ frees them again.
    assert not solver._model and not solver._env  # noqa: SLF001
    part.close()
    assert len(lib.freed_models) == 1 and len(lib.freed_envs) == 1


@pytest.mark.multifpga
def test_close_leaves_foreign_resources_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    lib = FakeGurobiLibrary(monkeypatch)
    # A solver wrapping a model it does not own, a non-Gurobi solver, no model at all and a
    # partitioner whose __init__ failed before creating the model.
    partitioner_with(SimpleNamespace(solver=lib.solver(owns_model=False))).close()
    partitioner_with(SimpleNamespace(solver=object())).close()
    partitioner_with(None).close()
    MinimalPartitioner.__new__(MinimalPartitioner).close()
    assert lib.freed_models == [] and lib.freed_envs == []


@pytest.mark.multifpga
def test_solver_is_released_by_context_manager_and_when_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib = FakeGurobiLibrary(monkeypatch)

    part = partitioner_with(SimpleNamespace(solver=lib.solver()))
    with part as entered:
        assert entered is part
        assert lib.freed_envs == []
    assert lib.freed_envs == ["env-handle"]

    part = partitioner_with(SimpleNamespace(solver=lib.solver()))
    del entered, part  # last references gone -> __del__ runs right away (refcount, no GC)
    assert lib.freed_envs == ["env-handle"] * 2
