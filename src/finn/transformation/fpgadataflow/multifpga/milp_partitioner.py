"""Base class for Multi-FPGA partitioners that are modelled as a mixed integer linear program."""

from __future__ import annotations

import locale
import mip
import os
import random
import time
from abc import abstractmethod
from mip import Model
from mip.exceptions import InterfacingError
from pathlib import Path
from typing import TYPE_CHECKING, override

from finn.transformation.fpgadataflow.multifpga.partitioner import Partitioner as BasePartitioner
from finn.util.basic import make_build_dir
from finn.util.exception import (
    FINNMultiFPGAError,
    FINNMultiFPGAUnsolvableError,
    FINNMultiFPGAUserError,
)
from finn.util.logging import log

if TYPE_CHECKING:
    from numbers import Real
    from qonnx.core.modelwrapper import ModelWrapper

    from finn.builder.build_dataflow_config import DataflowBuildConfig


class MILPPartitioner(BasePartitioner):
    """Base class for partitioners that model the partitioning as a mixed integer linear program.
    Subclasses define variables, constraints and the objective on `self.model` and translate the
    solution back into a mapping in `_create_result`. The model's definition must be done in
    `_build_model`, which is called as soon as the model is initialized.

    The idea to solve this using an LP was first
    devised by the AMD team for Elastic-DF and implemented as a prototype in finn-experimental.
    (https://github.com/Xilinx/finn-experimental/blob/main/src/finnexperimental/analysis/partitioning.py)

    The solver only exists while `solve()` runs: it is created, built and optimized there and
    released right afterwards, so that its license is held as short as possible and checks in
    `__init__` can fail before a license is requested. Afterwards `self.model` must not be used
    any more. Only `status` and `objective_value` remain. Caching of the result is done by the
    base-class solve() method.
    """

    model: mip.Model

    # Gurobi hands out one floating license token per environment and asks its token server for
    # it every time an environment is created. In a highly parallel setting (the CI test suite
    # runs dozens of pytest workers that all create partitioners) this fails now and then because
    # the token server is momentarily out of tokens or does not answer in time. Gurobi's own
    # recommendation for this case is to retry after a short delay.
    MAX_ATTEMPTS_ENV = "FINN_MIP_SOLVER_MAX_ATTEMPTS"  # overrides MAX_ATTEMPTS
    MAX_ATTEMPTS = 6
    RETRY_DELAY = 5.0  # seconds before the 2nd attempt, doubled after each further one
    RETRY_DELAY_MAX = 60.0  # seconds

    def __init__(self, model: ModelWrapper, cfg: DataflowBuildConfig) -> None:
        """Initialize the partitioner. The MILP model itself is only created by `solve()`."""
        super().__init__(model, cfg)
        self.status: mip.OptimizationStatus | None = None
        self.objective_value: Real | None = None

    @abstractmethod
    def _build_model(self) -> None:
        """Define variables, constraints and the objective on `self.model`, which was just
        created. Inheriting classes need to overwrite this with their custom definition."""

    @abstractmethod
    def _create_result(self) -> dict[str, int]:
        """Translate the solution of the MILP model into a mapping of node-names to devices.
        Must be implemented by inheriting classes since this is model specific."""

    @override
    def _solve(self) -> dict[str, int]:
        """Create the MILP model with the solver from the Multi-FPGA configuration, build and
        optimize it, and release the solver afterwards. Raises FINNMultiFPGAUnsolvableError if
        no solution was found.

        Transient failures to create a Gurobi environment are retried with an exponential backoff
        and a bit of jitter, so that many parallel processes do not all hit the token server again
        at the same moment.
        """
        solver = self.mfcfg.partition_solver
        # An empty solver name lets python-mip choose (Gurobi if available, otherwise CBC)
        solver_name = "" if solver is None else solver.value
        try:
            max_attempts = max(1, int(os.environ.get(self.MAX_ATTEMPTS_ENV, "")))
        except ValueError:
            max_attempts = self.MAX_ATTEMPTS
        delay = self.RETRY_DELAY

        # A failed Gurobi model instantiation switches the default locale/encoding away from UTF8,
        # causing file IO errors later on, so restore it afterwards.
        current_locale = locale.getlocale(locale.LC_CTYPE)
        try:
            for attempt in range(1, max_attempts + 1):
                try:
                    self.model = Model(name="finn_partition", solver_name=solver_name)
                    break
                except (OSError, InterfacingError) as e:
                    if solver is None:
                        # The OSError might be a known bug (LD_LIBRARY_PATH only modified at
                        # runtime to point to libgurobi instead of before), see finn-plus#67
                        log.warning(
                            f"Could not create a default-initialized mip.Model ({e}). "
                            f"Falling back to CBC."
                        )
                        self.model = Model(name="finn_partition", solver_name=mip.CBC)
                        break
                    # Only a failed Gurobi environment creation (no token or no answer from the
                    # token server right now) can disappear on retry. A missing solver library
                    # or a failed model creation is a persistent setup problem.
                    transient = "environment could not be loaded" in str(e)
                    if not transient or attempt == max_attempts:
                        hint = (
                            f" If this is a floating license, the token server may be out of "
                            f"tokens or unreachable; the number of attempts can be set with "
                            f"{self.MAX_ATTEMPTS_ENV}."
                            if transient
                            else ""
                        )
                        raise FINNMultiFPGAUserError(
                            f"Cannot create mip solver of type {solver_name} after {attempt} "
                            f"attempt(s). Original error: {e}{hint}"
                        ) from e
                    log.warning(
                        f"Creating the {solver_name} solver environment failed (attempt "
                        f"{attempt}/{max_attempts}): {e} Retrying in {delay:.1f}s."
                    )
                    time.sleep(delay)
                    # Exponential backoff with +-25% jitter, capped.
                    delay = min(delay * 2, self.RETRY_DELAY_MAX) * random.uniform(0.75, 1.25)
        finally:
            locale.setlocale(locale.LC_CTYPE, current_locale)

        try:
            self.model.emphasis = self.mfcfg.partition_solver_emphasis
            self._build_model()
            self.status = self.model.optimize(self.mfcfg.partition_solver_timeout)  # type: ignore
            if self.status == mip.OptimizationStatus.ERROR:
                raise FINNMultiFPGAError(
                    f"The solver returned an ERROR optimization status! Please check the model "
                    f"definition file at: {self._dump_model_definition().absolute()}"
                )
            # Any other status than OPTIMAL or FEASIBLE (e.g. INFEASIBLE, INT_INFEASIBLE,
            # NO_SOLUTION_FOUND, CUTOFF) means that no solution is available
            if self.status not in (
                mip.OptimizationStatus.OPTIMAL,
                mip.OptimizationStatus.FEASIBLE,
            ):
                raise FINNMultiFPGAUnsolvableError(
                    f"The solver found no partitioning for {self.mfcfg.num_fpgas} devices "
                    f"(optimization status: {self.status.name})."
                )
            self.objective_value = self.model.objective_value
            return self._create_result()
        finally:
            # python-mip frees the Gurobi model and environment in SolverGurobi.__del__, but
            # Model and its solver reference each other, so that only runs once the cyclic
            # garbage collector gets around to them. Until then the environment keeps its
            # floating license token: a running test suite was observed holding several hundred
            # tokens at once. So free them right away, mirroring SolverGurobi.__del__ (which
            # then finds nothing left to free).
            try:
                from mip import gurobi as mip_gurobi

                solver = self.model.solver
                if (
                    mip_gurobi.has_gurobi
                    and isinstance(solver, mip_gurobi.SolverGurobi)
                    and solver._ownsModel  # noqa: SLF001
                ):
                    if solver._model:  # noqa: SLF001
                        mip_gurobi.GRBfreemodel(solver._model)  # noqa: SLF001
                        solver._model = mip_gurobi.ffi.NULL  # noqa: SLF001
                    if solver._env and solver._venv_loaded:  # noqa: SLF001
                        mip_gurobi.GRBfreeenv(solver._env)  # noqa: SLF001
                        solver._env = mip_gurobi.ffi.NULL  # noqa: SLF001
            except Exception as e:  # never fail because cleanup failed
                log.debug(f"Could not release the mip solver: {e}")

    def _dump_model_definition(self, dir_prefix: str = "model_definition") -> Path:
        """Write the model definition to a new temporary directory for debugging."""
        p = Path(make_build_dir(dir_prefix + "_")) / "model.lp"
        self.model.write(str(p))
        return p
