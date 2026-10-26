"""Abstract base class for Multi-FPGA partitioners."""

import time
import yaml
from abc import ABC, abstractmethod
from collections import Counter
from functools import cached_property
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from typing import cast

from finn.builder.build_dataflow_config import DataflowBuildConfig, MultiFPGAConfiguration
from finn.transformation.fpgadataflow.multifpga.utils import ensure_multifpga_config
from finn.util.basic import log
from finn.util.exception import FINNMissingMappingError
from finn.util.resources import (
    ResourceEstimates,
    available_resources_on_platform,
    get_estimated_model_resources,
)


class Partitioner(ABC):
    """Base class for partitioners. Partitioners only need to implement the interfacing methods
    to other parts of FINN. Partitioners should inherit and implement their necessary machinery.
    """

    @ensure_multifpga_config
    def __init__(self, model: ModelWrapper, cfg: DataflowBuildConfig) -> None:
        """Initialize a partitioner by providing it with ModelWrapper and DataflowBuildConfig."""
        assert cfg.board is not None
        self.modelwrapper = model
        self.cfg = cfg
        self.mfcfg = cast("MultiFPGAConfiguration", cfg.multifpga_configuration)
        self.board = cfg.board
        self.mapping: dict[str, int] | None = None
        self.solve_seconds: float | None = None

    @abstractmethod
    def _solve(self) -> dict[str, int]:
        """Solve the partitioning problem and return a mapping between node-names and devices.
        This method MUST raise a FINNMultiFPGAUnsolvableError in case no solution can be found."""

    def solve(self) -> dict[str, int]:
        """Solve the model/problem and return a mapping between node-names and devices. If
        the model was already solved immediately return the cached result.
        """
        if self.mapping is None:
            start = time.time()
            if self.mfcfg.num_fpgas == 1:
                # There is only one possible mapping, so skip solving
                log.warning(
                    "Skipping solving - only one device provided. Assuming "
                    "default mapping (all nodes on device 0)."
                )
                self.mapping = {node.name: 0 for node in self.modelwrapper.graph.node}
            else:
                self.mapping = self._solve()
            self.solve_seconds = time.time() - start
        return self.mapping

    @cached_property
    def resource_estimates(self) -> ResourceEstimates:
        """Estimated resources per node for all considered resource types."""
        return get_estimated_model_resources(
            self.modelwrapper,
            self.cfg._resolve_fpga_part(),  # noqa: SLF001
            self.mfcfg.considered_resources,
            True,
        )

    def get_resource_use_relative(self) -> dict[int, dict[str, float]]:
        """Given the found mapping return the relative resource utilization
        per device and resource type. Raises an error if no mapping was found yet.
        """
        if self.mapping is None:
            raise FINNMissingMappingError(
                "Cannot retrieve relative resource usage since no mapping was calculated yet."
            )
        available = available_resources_on_platform(self.board, self.mfcfg.considered_resources)
        used = {
            device: dict.fromkeys(self.mfcfg.considered_resources, 0.0)
            for device in sorted(set(self.mapping.values()))
        }
        for nodename, device in self.mapping.items():
            for restype in self.mfcfg.considered_resources:
                used[device][restype] += self.resource_estimates[nodename][restype]
        return {
            device: {restype: amount / available[restype] for restype, amount in usage.items()}
            for device, usage in used.items()
        }

    def generate_partitioning_report(self) -> str:
        """Generate a report in which the required resources, the resources available per device,
        and much more are listed. Works even if no solution was found.
        """
        s = ""

        # General information
        s += f"{' Model and Configuration ':=^80}\n"
        s += "=" * 80 + "\n"
        s += f"{'Layers: ' + str(len(self.modelwrapper.graph.node)):<10}\n"
        s += f"{'Devices: ' + str(self.mfcfg.num_fpgas):<10}\n"
        if self.solve_seconds is not None:
            s += f"{'Time elapsed: ' + str(int(self.solve_seconds)) + 's':<10}\n"

        # Resources
        resource_estimates = self.resource_estimates
        device_resources = available_resources_on_platform(
            self.board, self.mfcfg.considered_resources
        )
        maxutil = self.mfcfg.max_utilization

        s += f"\n{' Required Resources by the Model ':=^80}\n"
        s += "=" * 80 + "\n"
        for restype in self.mfcfg.considered_resources:
            total_required = sum(rv[restype] for rv in resource_estimates.values())
            factor = total_required / (maxutil * device_resources[restype])
            s += (
                f"{restype:<15}{total_required:<15_}   ({factor:.2f}x  "
                f"on  {self.board}  at  {maxutil:.2%}  max utilization)\n"
            )

        s += f"\n{' Available Resources on Devices at Utilization Percentages ':=^80}\n"
        s += "=" * 80 + "\n"
        maxutil_percent = f"{maxutil:.2%}"
        devices = self.mfcfg.num_fpgas
        s += (
            " " * 15 + f"{'1x @ ' + maxutil_percent:^15}"
            f"{'1x @ 100%':^15}"
            f"{str(devices) + 'x @ ' + maxutil_percent + '(!)':^15}"
            f"{str(devices) + 'x @ 100%':^15}\n"
        )
        for restype in self.mfcfg.considered_resources:
            res = int(device_resources[restype])
            s += (
                f"{restype:<15}"
                f"{int(res * maxutil):^15_}"
                f"{res:^15_}"
                f"{int(maxutil * res * devices):^15_}"
                f"{devices * res:^15_}\n"
            )

        s += f"\n{' Largest Nodes by Resource Type ':=^80}\n"
        s += "=" * 80 + "\n"
        for restype in self.mfcfg.considered_resources:
            largest = max(resource_estimates, key=lambda layer: resource_estimates[layer][restype])
            largest_amount = int(resource_estimates[largest][restype])
            s += f"{restype:<15}{largest_amount:<12_}{largest:<25}\n"

        # Return early if no solution was found
        if self.mapping is None:
            return s

        s += f"\n{' Nodes per Device ':=^80}\n"
        s += "=" * 80 + "\n"
        counter = Counter(self.mapping.values())
        for device in sorted(counter):
            s += f"{'Device ' + str(device) + ': ':<10}{str(counter[device]) + ' nodes':<10}\n"

        s += f"\n{' Resources per Device ':=^80}\n"
        for device, usage in self.get_resource_use_relative().items():
            s += f"{' Device ' + str(device) + ' ':-^40}\n"
            for restype, relative in usage.items():
                s += f"{restype:<15}{relative:<15.2%}\n"

        return s

    def write_mapping(self, path: Path | None = None) -> Path:
        """Write the mapping to either the configured output reports directory
        or the provided path. Errors if the mapping is not yet solved.
        """
        if self.mapping is None:
            raise FINNMissingMappingError("Cannot write non-existent mapping.")
        p = self.cfg.get_report_directory() / "device_mapping.yaml" if path is None else path
        with p.open("w") as f:
            yaml.safe_dump(self.mapping, f)
        return p

    def write_report(self, path: Path | None = None) -> Path:
        """Write the partitioning report to either the configured output reports directory
        or the provided path. Works even if no solution was found.
        """
        p = self.cfg.get_report_directory() / "partitioning_report.txt" if path is None else path
        p.write_text(self.generate_partitioning_report())
        return p
