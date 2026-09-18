"""Partitioners for Multi-FPGA usage."""

from __future__ import annotations

import locale
import mip
import yaml
from abc import ABC, abstractmethod
from mip import Model
from pathlib import Path
from typing import Any

from finn.builder.build_dataflow_config import DataflowBuildConfig, MIPSolver, PartitioningStrategy
from finn.util.basic import make_build_dir
from finn.util.exception import FINNInternalError, FINNMultiFPGAError, FINNMultiFPGAUserError
from finn.util.logging import log


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
            try:
                return Model(name="finn_partition", solver_name=solver.value)
            except mip.exceptions.InterfacingError as e:
                raise FINNMultiFPGAUserError(
                    f"Cannot create mip solver of type {solver.value}. Original error: {e}"
                ) from e

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

        # Initialize the model
        self.status: mip.OptimizationStatus | None
        self.model = self.init_model(cfg.partitioning_configuration.partition_solver)
        self.model.emphasis = cfg.partitioning_configuration.partition_solver_emphasis

        # Restore locale, as mentioned above.
        locale.setlocale(locale.LC_CTYPE, current_locale)

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
