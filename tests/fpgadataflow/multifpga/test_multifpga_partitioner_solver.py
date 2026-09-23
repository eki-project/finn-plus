"""Tests for the creation of the mip solver model used by the Multi-FPGA partitioners."""

import pytest

import mip
from typing import Any

import finn.transformation.fpgadataflow.multifpga.partitioner as partitioner
from finn.transformation.fpgadataflow.multifpga.partitioner import (
    MIP_SOLVER_MAX_ATTEMPTS_ENV,
    MIP_SOLVER_RETRY_DELAY_ENV,
    create_mip_model,
    is_transient_solver_error,
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
