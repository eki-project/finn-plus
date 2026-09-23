"""Partitioners for Multi-FPGA usage."""

from __future__ import annotations

import locale
import mip
import os
import random
import time
import yaml
from abc import ABC, abstractmethod
from mip import Model
from pathlib import Path
from typing import TYPE_CHECKING, Any

from finn.builder.build_dataflow_config import DataflowBuildConfig, MIPSolver, PartitioningStrategy
from finn.util.basic import make_build_dir
from finn.util.exception import FINNMultiFPGAError, FINNMultiFPGAUserError
from finn.util.logging import log

if TYPE_CHECKING:
    pass

# Gurobi hands out one floating license token per environment and asks its token server for
# it every time an environment is created. In a highly parallel setting (the CI test suite
# runs dozens of pytest workers that all create partitioners) this fails now and then because
# the token server is momentarily out of tokens or does not answer in time. Gurobi's own
# recommendation for this case is to retry after a short delay, which is what
# `create_mip_model` does. Both knobs can be overridden through the environment.
MIP_SOLVER_MAX_ATTEMPTS_ENV = "FINN_MIP_SOLVER_MAX_ATTEMPTS"
MIP_SOLVER_RETRY_DELAY_ENV = "FINN_MIP_SOLVER_RETRY_DELAY"
MIP_SOLVER_MAX_ATTEMPTS_DEFAULT = 6
MIP_SOLVER_RETRY_DELAY_DEFAULT = 5.0  # seconds, doubled after every failed attempt
MIP_SOLVER_RETRY_DELAY_MAX = 60.0  # seconds

# The Gurobi diagnostics below are declared lazily and only once per process.
_gurobi_diagnostics_declared = False


def _retry_settings() -> tuple[int, float]:
    """Return the (max_attempts, initial_delay) to use when creating a mip model."""
    try:
        max_attempts = max(1, int(os.environ.get(MIP_SOLVER_MAX_ATTEMPTS_ENV, "")))
    except ValueError:
        max_attempts = MIP_SOLVER_MAX_ATTEMPTS_DEFAULT
    try:
        delay = max(0.0, float(os.environ.get(MIP_SOLVER_RETRY_DELAY_ENV, "")))
    except ValueError:
        delay = MIP_SOLVER_RETRY_DELAY_DEFAULT
    return max_attempts, delay


def is_transient_solver_error(error: mip.exceptions.InterfacingError) -> bool:
    """Tell whether an ``InterfacingError`` raised while creating a mip model is one that can
    disappear on retry. Only a failed Gurobi environment creation qualifies: everything else
    (missing solver library, failed model creation) is a persistent setup problem.
    """
    return "environment could not be loaded" in str(error)


def describe_gurobi_environment_error() -> str | None:
    """Ask Gurobi why creating an environment currently fails.

    python-mip reports every failed environment creation as "check your license", which hides
    the real reason (out of tokens, token server unreachable, expired license, ...). This
    creates an empty environment through the same cffi binding python-mip uses, starts it, and
    returns Gurobi's own error message with its error code. Returns ``None`` if the environment
    starts fine (the failure was transient and is already gone) or if the diagnostics cannot be
    collected at all.
    """
    global _gurobi_diagnostics_declared
    try:
        from mip import gurobi as mip_gurobi

        if not mip_gurobi.has_gurobi:
            return None
        ffi = mip_gurobi.ffi
        lib = mip_gurobi.grblib
        if not _gurobi_diagnostics_declared:
            ffi.cdef(
                """
                int GRBemptyenv(GRBenv **envP);
                int GRBstartenv(GRBenv *env);
                const char *GRBgeterrormsg(GRBenv *env);
                """
            )
            _gurobi_diagnostics_declared = True
        env_ptr = ffi.new("GRBenv **")
        status = lib.GRBemptyenv(env_ptr)
        if status != 0:
            return f"Gurobi error {status} while creating an empty environment"
        env = env_ptr[0]
        try:
            # A failed GRBstartenv leaves the environment as it was, so the error message can
            # be read from it and it must still be freed.
            status = lib.GRBstartenv(env)
            if status == 0:
                return None
            message = ffi.string(lib.GRBgeterrormsg(env)).decode("utf-8", errors="replace")
            return f"Gurobi error {status}: {message.strip()}"
        finally:
            lib.GRBfreeenv(env)
    except Exception as e:  # noqa: BLE001 - diagnostics must never mask the original error
        log.debug(f"Could not collect Gurobi diagnostics: {e}")
        return None


def create_mip_model(solver_name: str, name: str = "finn_partition") -> mip.Model:
    """Create a ``mip.Model`` for the given solver, retrying transient failures.

    Creating the model fails transiently for Gurobi when its floating license token server is
    out of tokens or unreachable (see the module comment). Such failures are retried with an
    exponential backoff and a bit of jitter so that many parallel processes do not all hit the
    server again at the same moment. Any other failure, and a transient failure that persists
    for all attempts, raises a ``FINNMultiFPGAUserError`` that includes Gurobi's own reason
    when it can be determined.
    """
    max_attempts, delay = _retry_settings()
    for attempt in range(1, max_attempts + 1):
        try:
            return Model(name=name, solver_name=solver_name)
        except mip.exceptions.InterfacingError as e:
            if not is_transient_solver_error(e):
                raise FINNMultiFPGAUserError(
                    f"Cannot create mip solver of type {solver_name}. Original error: {e}"
                ) from e
            reason = None
            if solver_name == mip.GUROBI:
                reason = describe_gurobi_environment_error()
            details = f" ({reason})" if reason else ""
            if attempt >= max_attempts:
                raise FINNMultiFPGAUserError(
                    f"Cannot create mip solver of type {solver_name} after {attempt} "
                    f"attempt(s). Original error: {e}{details}. If this is a floating license, "
                    f"the token server may be out of tokens or unreachable; the number of "
                    f"attempts and the initial delay can be tuned with "
                    f"{MIP_SOLVER_MAX_ATTEMPTS_ENV} and {MIP_SOLVER_RETRY_DELAY_ENV}."
                ) from e
            log.warning(
                f"Creating the {solver_name} solver environment failed (attempt "
                f"{attempt}/{max_attempts}): {e}{details}. Retrying in {delay:.1f}s."
            )
            time.sleep(delay)
            # Exponential backoff with +-25% jitter, capped.
            delay = min(delay * 2, MIP_SOLVER_RETRY_DELAY_MAX) * random.uniform(0.75, 1.25)
    raise AssertionError("unreachable")  # pragma: no cover


def release_mip_model(model: mip.Model | None) -> None:
    """Free the solver resources behind a ``mip.Model`` right away.

    python-mip frees them in the solver's ``__del__``, but ``Model`` and its solver reference
    each other, so that only runs once the cyclic garbage collector gets around to them. In a
    long-lived process with a large heap (a pytest worker of the CI suite) this happens rarely,
    and every Gurobi environment that is still alive keeps its floating license token: a running
    test suite was observed holding several hundred tokens at once. Freeing explicitly returns
    the token as soon as the partitioner is done. Safe to call more than once. The model must
    not be used any more afterwards.
    """
    solver = getattr(model, "solver", None)
    if solver is None:
        return
    try:
        from mip import gurobi as mip_gurobi

        if not (mip_gurobi.has_gurobi and isinstance(solver, mip_gurobi.SolverGurobi)):
            return
        if not solver._ownsModel:  # noqa: SLF001
            return
        # Mirrors SolverGurobi.__del__, which then finds nothing left to free.
        if solver._model:  # noqa: SLF001
            mip_gurobi.GRBfreemodel(solver._model)  # noqa: SLF001
            solver._model = mip_gurobi.ffi.NULL  # noqa: SLF001
        if solver._env and solver._venv_loaded:  # noqa: SLF001
            mip_gurobi.GRBfreeenv(solver._env)  # noqa: SLF001
            solver._env = mip_gurobi.ffi.NULL  # noqa: SLF001
    except Exception as e:  # noqa: BLE001 - never fail because cleanup failed
        log.debug(f"Could not release the mip solver: {e}")


class Partitioner(ABC):
    """Models a linear problem that can be used to solve Multi-FPGA partitioning. The idea to solve
    this in general using an LP was first devised by the AMD team for Elastic-DF and implemented as
    a prototype in finn-experimental.
    (https://github.com/Xilinx/finn-experimental/blob/main/src/finnexperimental/analysis/partitioning.py)

    We use a slightly different approach to modelling the problem and the objective function,
    however the partitioner from finn-experimental should be relativly easy to swap in
    if needed.
    """  # noqa

    def init_model(self, solver: MIPSolver | None) -> mip.Model:
        """Initialize the LP model, considering the partitioning configuration."""
        if solver is None:
            try:
                return Model()
            except OSError:
                log.warning(
                    "Creation of mip.Model failed. This might be known bug "
                    "(LD_LIBRARY_PATH only modified at runtime to point to "
                    "libgurobi instead of before). Falling back to CBC."
                )  # See finn-plus issue #67
                return Model(name="finn_partition", solver_name=mip.CBC)
            except mip.exceptions.InterfacingError as e:
                log.warning(
                    f"Could not create a default-initialized mip.Model. "
                    f"The error encountered was: {e}. Trying to fallback to a CBC based model."
                )
                return Model(name="finn_partition", solver_name=mip.CBC)
        else:
            return create_mip_model(solver.value)

    def __init__(self, cfg: DataflowBuildConfig) -> None:
        """Initialize a new partitioner. This involves creating the mip model."""
        assert cfg.partitioning_configuration is not None
        self.cfg = cfg
        self.pcfg = cfg.partitioning_configuration
        self.verbosity = cfg.partitioning_configuration.verbosity

        # Store locale. Necessary to avoid a bug, where a failed Gurobi model instantiation
        # causes the default locale/encoding to switch away from UTF8, causing file
        # IO errors later on.
        current_locale = locale.getlocale(locale.LC_CTYPE)

        # Initialize the model
        self.status: mip.OptimizationStatus | None
        self.model = self.init_model(cfg.partitioning_configuration.partition_solver)
        self.model.emphasis = cfg.partitioning_configuration.partition_solver_emphasis

        # Restore locale, as mentioned above.
        locale.setlocale(locale.LC_CTYPE, current_locale)

    def close(self) -> None:
        """Release the solver behind this partitioner (see ``release_mip_model``). Results can
        no longer be read from the mip model afterwards. Also happens automatically once the
        partitioner object is dropped.
        """
        release_mip_model(getattr(self, "model", None))

    def __enter__(self) -> Partitioner:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        # The partitioner is not part of a reference cycle, so this runs as soon as the last
        # reference is dropped and returns the Gurobi license token right away.
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 - never raise from __del__
            pass

    @abstractmethod
    def create_result(self) -> dict[str, int]:
        """Method that is used to generate a uniform solution type from the internal model.
        Any model / solver can implement its constraints and variables differently and overwrite
        this method, so that any class inheriting from the base can have a uniform result type.
        This type should map node-names to devices.
        The method should also error, if it is called before a solution was found.
        """  # noqa

    def dump_model_definition(self, dir_prefix: str = "model_definition") -> Path:
        """Write the model definition to a new temporary directory for debugging."""
        p = Path(make_build_dir(dir_prefix + "_")) / "model.lp"
        self.model.write(str(p))
        return p

    def solve(
        self,
        solver_timeout: int,
    ) -> dict[str, int] | None:
        """Try to optimize the objective function. If no feasible solution is found
        return None, otherwise return a mapping of nodes to their device. After trying
        to solve, creates a snapshot description of the model in a temp build dir, as well
        as a solution in the same dir, if one was found.
        """
        self.status = self.model.optimize(solver_timeout)  # type: ignore
        if self.status == mip.OptimizationStatus.ERROR:
            model_definition_file = self.dump_model_definition()
            raise FINNMultiFPGAError(
                f"The solver returned an ERROR optimization status! "
                f"Please check the model definition file "
                f"at: {model_definition_file.absolute()}"
            )
        if self.status in [
            mip.OptimizationStatus.INFEASIBLE,
            mip.OptimizationStatus.NO_SOLUTION_FOUND,
        ]:
            return None
        return self.create_result()

    def write_results(self, p: Path) -> None:
        """Write the partition results as a YAML to the given directory."""
        results = self.create_result()
        with p.open("w+") as f:
            yaml.dump(results, f, yaml.Dumper)

    @abstractmethod
    def _get_resource_use_relative(self) -> dict[int, dict[str, Any]]:
        """Get resources used by the device in percent. Must fail if no
        partition was calculated yet.
        """
        pass  # noqa

    def get_resource_use_relative(self) -> dict[int, dict[str, Any]] | None:
        """Return the resources used by a device. This only works if the optimization goal was
        resource usage. If no optimization was done, the dict will contain None's
        Actual implementation is left to the subclasses.
        """
        if self.pcfg.partition_strategy == PartitioningStrategy.RESOURCE_UTILIZATION:
            return self._get_resource_use_relative()
        return None
