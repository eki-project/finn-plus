"""Build a Vitis accelerator from a completed FINN design."""

# Copyright (c) 2020, Xilinx, Inc.
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from __future__ import annotations

import json
import shlex
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from qonnx.transformation.base import Transformation
from typing import TYPE_CHECKING, cast

from finn.builder.build_dataflow_config import DataflowBuildConfig, MFCommunicationKernel
from finn.transformation.fpgadataflow.floorplanning.placer_retry import (
    MAX_PLACER_RETRIES,
    OVERRIDES_FILE,
    REPORT_FILE,
    UNITS_FILE,
    VPL_LOG,
    latest_attempt,
    parse_placer_rejections,
    retry_script,
    write_overrides,
)
from finn.transformation.fpgadataflow.multifpga.aurora.link_config_transform import (
    AddAuroraToLinkConfig,
)
from finn.transformation.fpgadataflow.vitis_linking_configuration import (
    BuildBasicVitisLinkConfig,
    VitisLinkConfiguration,
)
from finn.util.exception import FINNInternalError, FINNSynthesisError, FINNUserError
from finn.util.logging import log

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


class ParallelVitisSynthesis(Transformation):
    """Execute a (parallel) synthesis on the model. Requires that the model has
    a link config. Afterwards the bitstreams are available.
    Stores the paths of all XCLBINs in as JSON, indexed by device
    id in the metadata prop "bitfile". Also stores the paths to the synthesis reports
    in vivado_synth_rpt, also as a JSON.
    """

    def __init__(self, cfg: DataflowBuildConfig) -> None:
        """Execute multiple Vitis link/synthesis runs in parallel."""
        self.cfg = cfg

    def link(self, config: VitisLinkConfiguration) -> Path:
        """Link a single config. Returns the path where the XCLBIN is
        expected to be produced.

        If implementation fails because the placer rejected a floorplan pblock as
        too small, the exact shortfall it reports is fed back to the geometry
        engine and implementation is resumed, up to `MAX_PLACER_RETRIES` times.
        Any other failure is raised at once. See `placer_retry`.
        """
        link_dir = config.run_script_path.parent
        script = config.run_script_path
        accumulated: dict[str, dict[str, int]] = {}
        for attempt in range(MAX_PLACER_RETRIES + 1):
            result = subprocess.run(
                shlex.split(f"bash {script}"),
                cwd=link_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                break

            vpl_log = link_dir / VPL_LOG
            report_path = link_dir / REPORT_FILE
            report = json.loads(report_path.read_text()) if report_path.is_file() else None
            units_path = link_dir / UNITS_FILE
            units = json.loads(units_path.read_text()) if units_path.is_file() else None
            rejections = (
                parse_placer_rejections(
                    latest_attempt(vpl_log.read_text(errors="ignore")), report, units
                )
                if vpl_log.is_file()
                else {}
            )
            if not rejections or attempt == MAX_PLACER_RETRIES:
                raise FINNSynthesisError(
                    f"Synthesis likely failed. Try checking logs at {link_dir}.",
                    link_dir / "v++_a.log",
                )
            # Shortfalls add up across attempts: a second rejection means the
            # first increase was not enough, not that it should be replaced.
            for pblock, per in rejections.items():
                acc = accumulated.setdefault(pblock, {})
                for key, short in per.items():
                    acc[key] = acc.get(key, 0) + short
            # The hook re-packs only the units named here, each with its total
            # shortfall so far, and keeps every other placement of the previous
            # attempt: a unit that was grown earlier and then placed stays where
            # it is unless the placer rejects it again.
            write_overrides(
                link_dir / OVERRIDES_FILE, {pblock: accumulated[pblock] for pblock in rejections}
            )
            script = retry_script(config.run_script_path)
            log.warning(
                f"Placer rejected {len(rejections)} floorplan pblock(s) as too small; "
                f"resuming implementation with their reported shortfalls "
                f"(attempt {attempt + 2} of {MAX_PLACER_RETRIES + 1}): {rejections}"
            )

        # Try to generate resource reports, but don't raise an exception if this fails
        gen_result = subprocess.run(
            shlex.split(f"vivado -mode batch -source {config.gen_report_xml_path}"),
            cwd=config.gen_report_xml_path.parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if gen_result.returncode != 0:
            log.error(
                f"Could not create post-synthesis resource report in "
                f"{config.gen_report_xml_path.parent}. Continuing now."
            )
        return config.run_script_path.parent / "a.xclbin"

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Run synthesis."""
        # Load all configs
        configs = VitisLinkConfiguration.load_from_model(model)

        # Figure out number of parallel synthesis runs
        if self.cfg.partitioning_configuration is not None:
            workers = self.cfg.partitioning_configuration.parallel_synthesis_workers
        else:
            workers = 1

        # Execute synthesis concurrently
        futures: dict[int, Future] = {}
        with ThreadPoolExecutor(workers) as tpe:
            for device, config in configs.items():
                log.info(
                    f"Submitting config {device} for synthesis "
                    f"(target frequency: {config.f_mhz} MHz)"
                    f"(at {config.run_script_path.parent})"
                )
                futures[device] = tpe.submit(self.link, config)
            tpe.shutdown(wait=True)

        # Check results and exceptions
        results: dict[int, str] = {}
        any_failed = False
        for i, future in futures.items():
            result = cast("Path", future.result())
            results[i] = str(result.absolute())
            if not result.exists():
                any_failed = True
                log.critical(
                    f"XCLBIN for device {i} not found. Check "
                    f"synthesis logs at {configs[i].run_script_path.parent}"
                )
        if any_failed:
            raise FINNUserError(
                "Generation of one or more bitstreams failed. Check "
                "the synthesis directory and vivado logs for more details."
            )

        # Store paths for usage in driver generation, etc.
        model.set_metadata_prop("bitfile", json.dumps(results))
        model.set_metadata_prop(
            "vivado_synth_rpt",
            json.dumps(
                {device: str(Path(p).parent / "synth_report.xml") for device, p in results.items()}
            ),
        )
        return model, False


class VitisBuild(Transformation):
    """Build an accelerator for the Vitis platform. Receives an SDP graph with all
    XO files packaged. This transform builds the configuration and starts a (parallel)
    synthesis.
    """

    def __init__(self, cfg: DataflowBuildConfig) -> None:
        """Run a Vitis build on the entire graph by creating a linking config and executing it."""
        self.cfg: DataflowBuildConfig = cfg

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Run the build."""
        log.info("Building Vitis linking configuration...")
        if self.cfg.board is None:
            raise FINNInternalError("No board set in the build configuration")
        model = model.transform(
            BuildBasicVitisLinkConfig(
                platform=self.cfg._resolve_vitis_platform(),  # noqa
                board=self.cfg.board,
                mem_type=self.cfg.fpga_memory,
                vitis_opt_strategy=self.cfg.vitis_opt_strategy,
                optimization_level=self.cfg.vitis_opt_strategy.value,
                f_mhz=round(1000.0 / self.cfg.synth_clk_period_ns),
                fpga_part=self.cfg._resolve_fpga_part(),  # noqa: SLF001
                auto_floorplanning=self.cfg.enable_auto_floorplanning,
                floorplan_resource_margin=self.cfg.floorplan_resource_margin,
                floorplan_slicem_utilization=self.cfg.floorplan_slicem_utilization,
                floorplan_logic_utilization=self.cfg.floorplan_logic_utilization,
            )
        )

        # Multi-FPGA specific config changes
        if self.cfg.partitioning_configuration is not None:
            log.info("Modifying linking configuration for Multi-FPGA...")
            match self.cfg.partitioning_configuration.communication_kernel:
                case MFCommunicationKernel.AURORA:
                    model = model.transform(
                        AddAuroraToLinkConfig(
                            board=self.cfg.board,
                            fpga_part=self.cfg._resolve_fpga_part(),  # noqa: SLF001
                        )
                    )
                case _:
                    raise FINNInternalError(
                        f"Vitis linking confíguration modifications for kernel type "
                        f"{self.cfg.partitioning_configuration.communication_kernel} "
                        f"are not yet implemented."
                    )

        # Check for errors and warnings
        configs: dict[int, VitisLinkConfiguration] = VitisLinkConfiguration.load_from_model(model)
        for device, config in configs.items():
            log.info(f"Checking configuration for device {device}...")
            errors = config.get_config_validation_errors(silent_warnings=False)
            if errors is not None:
                # TODO: Raise all exceptions as an exception group (Python 3.11)
                raise errors[0]

        # Run the synthesis
        log.info("Starting synthesis...")
        model = model.transform(ParallelVitisSynthesis(self.cfg))
        return model, False
