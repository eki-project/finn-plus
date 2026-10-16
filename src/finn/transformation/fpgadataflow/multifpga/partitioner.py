"""Partitioners for Multi-FPGA usage."""

from __future__ import annotations

import locale
import mip
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self, cast

from finn.builder.build_dataflow_config import DataflowBuildConfig, MIPSolver, PartitioningStrategy
from finn.util.basic import make_build_dir
from finn.util.exception import FINNInternalError, FINNMultiFPGAError, FINNMultiFPGAUserError
from finn.util.logging import log
from finn.util.lp import create_mip_model, create_model_with_retries, gurobi_environment_error

# Creating the solver model (with retries for Gurobi's floating license) lives in finn.util.lp,
# shared with the floorplanner's SLR assignment.


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
        return create_mip_model(solver, "finn_partition", FINNMultiFPGAUserError)

    @staticmethod
    def create_model(solver_name: str, name: str = "finn_partition") -> mip.Model:
        """Create the ``mip.Model`` for the given solver, retrying transient failures.

        See `finn.util.lp.create_model_with_retries`; failures raise
        ``FINNMultiFPGAUserError``.
        """
        return create_model_with_retries(solver_name, name, FINNMultiFPGAUserError)

    @staticmethod
    def _gurobi_environment_error() -> str | None:
        """Ask Gurobi why creating an environment currently fails (`finn.util.lp`)."""
        return gurobi_environment_error()

    def __init__(self, cfg: DataflowBuildConfig) -> None:
        """Initialize a new partitioner. This involves creating the mip model."""
        if not (cfg.partitioning_configuration is not None):
            raise FINNInternalError("No partitioning_configuration set in the build configuration")
        self.cfg = cfg
        self.pcfg = cfg.partitioning_configuration
        self.verbosity = cfg.partitioning_configuration.verbosity

        # Store locale. Necessary to avoid a bug, where a failed Gurobi model instantiation
        # causes the default locale/encoding to switch away from UTF8, causing file
        # IO errors later on.
        current_locale = locale.getlocale(locale.LC_CTYPE)

        # Initialize the model. The solution values are cached by solve() so that they remain
        # available after the solver was released by close().
        self.status: mip.OptimizationStatus | None = None
        self.objective_value: float | None = None
        self._resource_use_relative_cache: dict[int, dict[str, Any]] | None = None
        # Explicitly annotated: without it, pyright's loop-narrowing infers a spurious
        # "Model | LinExpr" union for self.model wherever a subclass does `self.model +=
        # <constraint>` inside a for loop, since mip's LinExpr overloads comparison
        # operators (e.g. `==`) to build constraint objects instead of returning bool.
        self.model: mip.Model = self.init_model(cfg.partitioning_configuration.partition_solver)
        self.model.emphasis = cfg.partitioning_configuration.partition_solver_emphasis

        # Restore locale, as mentioned above.
        locale.setlocale(locale.LC_CTYPE, current_locale)

    def close(self) -> None:
        """Release the solver behind this partitioner right away, returning its license token.

        python-mip frees the Gurobi model and environment in the solver's ``__del__``, but
        ``Model`` and its solver reference each other, so that only runs once the cyclic garbage
        collector gets around to them. In a long-lived process with a large heap (a pytest worker
        of the CI suite) this happens rarely, and every Gurobi environment that is still alive
        keeps its floating license token: a running test suite was observed holding several
        hundred tokens at once. Safe to call more than once; also happens when the partitioner is
        used as a context manager or dropped. Afterwards the mip model must not be used any more,
        only the values cached by ``solve()`` (status, objective value, resource use) remain.
        """
        solver = getattr(getattr(self, "model", None), "solver", None)
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
        except Exception as e:  # never fail because cleanup failed
            log.debug(f"Could not release the mip solver: {e}")

    def __enter__(self) -> Self:
        """Enter a ``with`` block that releases the solver at its end."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Release the solver when the ``with`` block ends."""
        self.close()

    def __del__(self) -> None:
        """Release the solver once the partitioner is dropped.

        The partitioner is not part of a reference cycle, so this runs as soon as the last
        reference is dropped and returns the Gurobi license token right away.
        """
        # Never raise from __del__. A plain try/except instead of contextlib.suppress, since this
        # may run during interpreter shutdown when module globals are already torn down.
        try:  # noqa: SIM105
            self.close()
        except Exception:
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
        result = self.create_result()
        # Cache what is reported after solving, so that the solver can be released early.
        # mip types it as numbers.Real; the solvers return a float.
        self.objective_value = cast("float | None", self.model.objective_value)
        self.get_resource_use_relative()
        return result

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
            if self._resource_use_relative_cache is None:
                self._resource_use_relative_cache = self._get_resource_use_relative()
            return self._resource_use_relative_cache
        return None
