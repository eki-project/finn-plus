"""Tests for the creation of the mip solver model used by the Multi-FPGA partitioners."""

import pytest

import mip
import mip.gurobi
from types import SimpleNamespace
from typing import Any

import finn.transformation.fpgadataflow.multifpga.partitioner as partitioner
from finn.transformation.fpgadataflow.multifpga.partitioner import (
    MIP_SOLVER_MAX_ATTEMPTS_ENV,
    MIP_SOLVER_RETRY_DELAY_ENV,
    Partitioner,
    create_mip_model,
    is_transient_solver_error,
    release_mip_model,
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


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []
    monkeypatch.setattr(partitioner.time, "sleep", delays.append)
    monkeypatch.setattr(partitioner.random, "uniform", lambda a, b: 1.0)  # noqa: ARG005
    monkeypatch.delenv(MIP_SOLVER_MAX_ATTEMPTS_ENV, raising=False)
    monkeypatch.delenv(MIP_SOLVER_RETRY_DELAY_ENV, raising=False)
    return delays


@pytest.mark.multifpga
def test_is_transient_solver_error() -> None:
    assert is_transient_solver_error(mip.exceptions.InterfacingError(ENV_ERROR))
    assert not is_transient_solver_error(mip.exceptions.InterfacingError(MODEL_ERROR))


@pytest.mark.multifpga
def test_transient_failure_is_retried_with_backoff(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    factory = FakeModelFactory([ENV_ERROR, ENV_ERROR])
    monkeypatch.setattr(partitioner, "Model", factory)
    monkeypatch.setattr(
        partitioner, "describe_gurobi_environment_error", lambda: "Gurobi error 10009: No tokens"
    )
    monkeypatch.setenv(MIP_SOLVER_RETRY_DELAY_ENV, "2")

    assert create_mip_model(mip.GUROBI) == "model"
    assert len(factory.calls) == 3
    assert factory.calls[0] == {"name": "finn_partition", "solver_name": mip.GUROBI}
    assert no_sleep == [2.0, 4.0]


@pytest.mark.multifpga
def test_persistent_failure_raises_after_all_attempts(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    factory = FakeModelFactory([ENV_ERROR] * 10)
    monkeypatch.setattr(partitioner, "Model", factory)
    monkeypatch.setattr(
        partitioner,
        "describe_gurobi_environment_error",
        lambda: "Gurobi error 10009: No tokens available",
    )
    monkeypatch.setenv(MIP_SOLVER_MAX_ATTEMPTS_ENV, "3")
    monkeypatch.setenv(MIP_SOLVER_RETRY_DELAY_ENV, "1")

    with pytest.raises(FINNMultiFPGAUserError) as excinfo:
        create_mip_model(mip.GUROBI)
    assert len(factory.calls) == 3
    assert no_sleep == [1.0, 2.0]
    message = str(excinfo.value)
    assert "after 3 attempt(s)" in message
    assert ENV_ERROR in message
    assert "No tokens available" in message
    assert isinstance(excinfo.value.__cause__, mip.exceptions.InterfacingError)


@pytest.mark.multifpga
def test_non_transient_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    factory = FakeModelFactory([MODEL_ERROR])
    monkeypatch.setattr(partitioner, "Model", factory)
    monkeypatch.setattr(
        partitioner,
        "describe_gurobi_environment_error",
        lambda: pytest.fail("diagnostics must not run for non-transient errors"),
    )

    with pytest.raises(FINNMultiFPGAUserError, match=MODEL_ERROR):
        create_mip_model(mip.GUROBI)
    assert len(factory.calls) == 1
    assert no_sleep == []


@pytest.mark.multifpga
def test_single_attempt_disables_retries(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    factory = FakeModelFactory([ENV_ERROR])
    monkeypatch.setattr(partitioner, "Model", factory)
    monkeypatch.setattr(partitioner, "describe_gurobi_environment_error", lambda: None)
    monkeypatch.setenv(MIP_SOLVER_MAX_ATTEMPTS_ENV, "1")

    with pytest.raises(FINNMultiFPGAUserError, match="after 1 attempt"):
        create_mip_model(mip.GUROBI)
    assert len(factory.calls) == 1
    assert no_sleep == []


@pytest.mark.multifpga
def test_diagnostics_only_collected_for_gurobi(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    factory = FakeModelFactory([ENV_ERROR])
    monkeypatch.setattr(partitioner, "Model", factory)
    monkeypatch.setattr(
        partitioner,
        "describe_gurobi_environment_error",
        lambda: pytest.fail("diagnostics must only run for Gurobi"),
    )

    assert create_mip_model(mip.CBC) == "model"
    assert len(factory.calls) == 2


@pytest.mark.multifpga
def test_describe_gurobi_environment_error_without_gurobi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a Gurobi library the diagnostics report nothing instead of failing."""
    import mip.gurobi

    monkeypatch.setattr(mip.gurobi, "has_gurobi", False)
    assert partitioner.describe_gurobi_environment_error() is None


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


@pytest.mark.multifpga
def test_release_mip_model_frees_gurobi_model_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib = FakeGurobiLibrary(monkeypatch)
    solver = lib.solver()
    model = SimpleNamespace(solver=solver)

    release_mip_model(model)
    assert lib.freed_models == ["model-handle"]
    assert lib.freed_envs == ["env-handle"]
    # The handles are cleared so that neither a second release nor python-mip's own
    # SolverGurobi.__del__ frees them again.
    assert not solver._model and not solver._env  # noqa: SLF001
    release_mip_model(model)
    assert len(lib.freed_models) == 1 and len(lib.freed_envs) == 1


@pytest.mark.multifpga
def test_release_mip_model_leaves_foreign_resources_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    lib = FakeGurobiLibrary(monkeypatch)
    # A solver wrapping a model it does not own, a non-Gurobi solver and no model at all.
    release_mip_model(SimpleNamespace(solver=lib.solver(owns_model=False)))
    release_mip_model(SimpleNamespace(solver=object()))
    release_mip_model(None)
    assert lib.freed_models == [] and lib.freed_envs == []


class MinimalPartitioner(Partitioner):
    """Concrete enough to be instantiated; only the resource handling of the base is tested."""


# ABCMeta computes the abstract set on class creation, so clear it afterwards.
MinimalPartitioner.__abstractmethods__ = frozenset()


@pytest.mark.multifpga
def test_partitioner_releases_solver_on_close_and_when_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released: list[Any] = []
    monkeypatch.setattr(partitioner, "release_mip_model", released.append)

    # Bypass __init__, which needs a full build configuration and a solver.
    part = MinimalPartitioner.__new__(MinimalPartitioner)
    part.model = "the-model"
    with part as entered:
        assert entered is part
    assert released == ["the-model"]

    part.close()
    assert released == ["the-model"] * 2

    del entered, part  # last references gone -> __del__ runs right away (refcount, no GC)
    assert released == ["the-model"] * 3

    # A partitioner whose __init__ failed before creating the model does not blow up.
    del released[:]
    broken = MinimalPartitioner.__new__(MinimalPartitioner)
    del broken
    assert released == [None]
