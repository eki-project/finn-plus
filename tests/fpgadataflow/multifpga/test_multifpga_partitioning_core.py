from __future__ import annotations

import pytest

import os
import random
import torch
from brevitas.export import export_qonnx
from contextlib import contextmanager
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from typing import TYPE_CHECKING

from finn.builder import build_dataflow_steps
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    MFCommunicationKernel,
    MFTopology,
    PartitioningConfiguration,
    PartitioningStrategy,
    default_build_dataflow_steps,
)
from finn.transformation.fpgadataflow.multifpga_partitioner import AuroraPartitioner
from finn.transformation.fpgadataflow.multifpga_utils import available_resources
from finn.util import platforms
from finn.util.test import get_test_model

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def custom_build(name: str, random_prefix: bool) -> Generator[tuple[Path, Path, Path]]:
    """Create a directory in FINN_BUILD_DIR for custom builds.
    Temporarily also set the FINN_BUILD_DIR environment variable to this new dir.
    Can be used to contain a complete build. Returns the new directory, the
    temp directory and the output directory"""
    origin_path = Path(os.environ["FINN_BUILD_DIR"])
    if not origin_path.exists():
        origin_path.mkdir(parents=True)
    dir_name = name
    if random_prefix:
        proposed_dir_name = dir_name + f"_{random.randint(0,1000000)}"
        while proposed_dir_name in os.listdir(origin_path):
            proposed_dir_name = dir_name + f"_{random.randint(0,1000000)}"
        dir_name = proposed_dir_name
    root = origin_path / dir_name
    root.mkdir()
    temps = root / "FINN_TMP"
    temps.mkdir()
    out = root / "outputs"
    out.mkdir()
    original_build_dir = os.environ["FINN_BUILD_DIR"]
    try:
        os.environ["FINN_BUILD_DIR"] = str(temps)
        yield (root, temps, out)
    finally:
        os.environ["FINN_BUILD_DIR"] = original_build_dir


# TODO: Add mobilenet
@pytest.mark.parametrize(
    "model_type",
    [
        ("CNV", 1, 1, True),
        ("CNV", 1, 2, True),
        ("CNV", 2, 2, True),
        ("LFC", 1, 1, True),
        ("LFC", 1, 2, True),
        ("SFC", 1, 1, True),
        ("SFC", 1, 2, True),
        ("SFC", 2, 2, True),
        ("TFC", 1, 1, True),
        ("TFC", 1, 2, True),
    ],
)
@pytest.mark.parametrize("devices", [2, 3, 4, 10])
@pytest.mark.parametrize("max_util", [0.95, 0.85])
@pytest.mark.parametrize("ideal_util", [0.80, 0.75])
@pytest.mark.parametrize(
    "partition_strategy",
    [PartitioningStrategy.LAYER_COUNT, PartitioningStrategy.RESOURCE_UTILIZATION],
)
@pytest.mark.parametrize("topology", [MFTopology.CHAIN])
@pytest.mark.parametrize("board", ["Pynq-Z1"])
def test_aurora_partition_solution_found(
    model_type: tuple[str, int, int, bool],
    devices: int,
    partition_strategy: PartitioningStrategy,
    topology: MFTopology,
    board: str,
    max_util: float,
    ideal_util: float,
) -> None:
    """Test some known model - fpga combinations that should
    be solveable"""

    # TODO: Fix: Certain model types fail during streamlining

    typename, wbits, abits, pretrained = model_type
    test_dir_identifier = (
        f"test_partition_solution_{typename}_{wbits}_{abits}"
        f"_p{pretrained}_dev{devices}_board{board}"
    )

    with custom_build(test_dir_identifier, True) as dirs:
        root, temps, out = dirs
        model_onnx_path = Path(root) / "fc.onnx"
        fc = get_test_model(typename, wbits, abits, pretrained)
        ishape = (1, 1, 28, 28)
        if typename == "CNV":
            ishape = (1, 3, 32, 32)
        elif typename == "mobilenet":
            ishape = (1, 3, 224, 224)
        export_qonnx(fc, torch.randn(ishape), str(model_onnx_path))
        qonnx_cleanup(str(model_onnx_path), out_file=str(model_onnx_path))
        model = ModelWrapper(str(model_onnx_path))

        cfg = DataflowBuildConfig(
            output_dir=str(out),
            synth_clk_period_ns=5.0,
            generate_outputs=[DataflowOutputType.ESTIMATE_REPORTS, DataflowOutputType.STITCHED_IP],
            board=board,
            target_fps=2000,
            save_intermediate_models=True,
            partitioning_configuration=PartitioningConfiguration(
                num_fpgas=devices,
                partition_strategy=partition_strategy,
                max_utilization=max_util,
                ideal_utilization=ideal_util,
                communication_kernel=MFCommunicationKernel.AURORA,
                topology=topology,
            ),
        )
        for transform_step in default_build_dataflow_steps:
            model = build_dataflow_steps.build_dataflow_step_lookup[transform_step](model, cfg)
            if transform_step == "step_set_fifo_depths":
                break
        model = build_dataflow_steps.step_partition_for_multifpga(model, cfg)


@pytest.mark.parametrize(
    "distribution",
    ["equal-LUT", "equal-FF"],
)
@pytest.mark.parametrize("distribution_args", [{"level": 1000}, {"level": 10000}, {"level": 10e6}])
@pytest.mark.parametrize("nodes", [*list(range(1, 10))])
@pytest.mark.parametrize("devices", [*list(range(1, 10))])
@pytest.mark.parametrize("considered_resources", [["LUT", "FF", "DSP", "BRAM_18K"]])
@pytest.mark.parametrize("board", ["U280", "Pynq-Z1"])
@pytest.mark.parametrize("max_util", [0.85])
@pytest.mark.parametrize("ideal_util", [0.75])
@pytest.mark.parametrize("topology", [MFTopology.CHAIN])
@pytest.mark.parametrize("inseperable_nodes", [[]])
@pytest.mark.parametrize("network_ports", [2])
def test_aurora_partitioning_pure_resource_optimize(
    distribution: str,
    distribution_args: dict,
    nodes: int,
    devices: int,
    considered_resources: list[str],
    board: str,
    ideal_util: float,
    max_util: float,
    topology: MFTopology,
    inseperable_nodes: list[int],
    network_ports: int,
) -> None:
    """Test partitioning with the Aurora model based on constructed data instead of real models"""
    dist_type = distribution.split("-")[0]
    dist_res = distribution.split("-")[1]
    resource_estimates = {}
    for node in range(nodes):
        resource_estimates[node] = dict(
            zip(considered_resources, [0 for _ in range(len(considered_resources))])
        )
        if dist_type == "equal":
            resource_estimates[node][dist_res] = distribution_args["level"]

    test_dir_identifier = (
        f"test_pure_aurora_resource_opt_device{devices}"
        f"_node{nodes}_topo{topology.name}_{board}_dist{distribution}"
        f"_{max_util}_{ideal_util}_ins{len(inseperable_nodes)}"
    )
    with custom_build(test_dir_identifier, True) as dirs:
        root, temps, out = dirs
        res_per_device = available_resources(platforms.platforms[board](), considered_resources)
        part = AuroraPartitioner(
            network_ports_per_device=network_ports,
            strategy=PartitioningStrategy.RESOURCE_UTILIZATION,
            devices=devices,
            nodes=nodes,
            considered_resources=considered_resources,
            resources_per_device=res_per_device,
            inseperable_nodes=[],
            topology=topology,
            max_utilization=max_util,
            ideal_utilization=ideal_util,
            resource_estimates=resource_estimates,
        )

        solution = part.solve(
            100,
            root / "snapshot.txt",
            root / "solution.txt",
            {k: f"node_{k}" for k in range(nodes)},
        )

        # Check if a solution was found
        # If the model is impossible assert that no solution was found
        # and return to skip the rest if the test
        overutilized_overall = (
            nodes * distribution_args["level"] > res_per_device[dist_res] * devices
        )
        overutilized_per_device = distribution_args["level"] > max_util * res_per_device[dist_res]
        if devices > nodes or overutilized_overall or overutilized_per_device:
            assert solution is None
            return
        assert solution is not None

        # Every device is utilized
        usage = part.get_resource_use_by_device()
        assert usage is not None
        for device in usage.keys():
            assert any(usage[device][restype] > 0 for restype in usage[device].keys())

        # max_utilization not overstepped
        for device in usage.keys():
            for restype, res in usage[device].items():
                assert res <= max_util * res_per_device[restype]

        # Atleast 2 different device IDs (to catch the qonnx nodeattr bug)
        if devices >= 2:
            assert len(set(solution.values())) >= 2

        # All nodes have an assignment
        for i in range(nodes):
            assert i in solution.keys()
            assert solution[i] is not None

        # Consecutive assignments
        for i in range(nodes - 1):
            assert abs(solution[i] - solution[i + 1]) <= 1

        # No device was visited twice
        if topology == MFTopology.CHAIN:
            visited = [solution[0]]
            for i in range(nodes - 1):
                if solution[i + 1] != solution[i]:
                    visited.append(solution[i + 1])
        assert len(set(visited)) == len(visited)


@pytest.mark.parametrize(
    "platform", [platforms.Alveo_NxU280_Platform(), platforms.Zynq7020_Platform()]
)
@pytest.mark.parametrize("topology", [MFTopology.CHAIN])
@pytest.mark.parametrize("network_ports", [2])
@pytest.mark.parametrize("ideal_max_util", [(0.8, 0.9), (0.9, 1.0), (0.2, 0.8), (0.2, 1.0)])
def test_enforce_utilization_limit(
    platform: platforms.Platform,
    topology: MFTopology,
    network_ports: int,
    ideal_max_util: tuple[float, float],
) -> None:
    """Test that the partitioner upholds the resource utilization limit"""
    test_dir_identifier = f"test_util_limit_{platform.__class__.__name__}" f"_{topology.name}"
    with custom_build(test_dir_identifier, True) as dirs:
        root, temps, out = dirs
        diff = 0.05
        max_util = ideal_max_util[1]
        ideal_util = ideal_max_util[0]
        devices = 2
        nodes = 2
        considered_resources = ["LUT", "FF", "DSP", "BRAM_18K"]
        res_per_device = available_resources(platform, considered_resources)
        # Device0 is underutilized, Device1 is overutilized
        resource_estimates = {
            0: {res: res_per_device[res] * (max_util - diff) for res in considered_resources},
            1: {res: res_per_device[res] * (max_util + diff) for res in considered_resources},
        }
        part = AuroraPartitioner(
            network_ports_per_device=network_ports,
            strategy=PartitioningStrategy.RESOURCE_UTILIZATION,
            devices=devices,
            nodes=nodes,
            considered_resources=considered_resources,
            resources_per_device=res_per_device,
            inseperable_nodes=[],
            topology=topology,
            max_utilization=max_util,
            ideal_utilization=ideal_util,
            resource_estimates=resource_estimates,
        )

        solution = part.solve(
            100,
            root / "snapshot.txt",
            root / "solution.txt",
            {k: f"node_{k}" for k in range(nodes)},
        )
        assert solution is None


@pytest.mark.parametrize("devices", [2, 3, 10, 100])
@pytest.mark.parametrize("nodes", [1, 2, 3, 10, 100, 110])
@pytest.mark.parametrize("network_ports", [2, 3, 4])
@pytest.mark.parametrize("topology", [MFTopology.CHAIN])
@pytest.mark.parametrize(
    "platform", [platforms.Alveo_NxU280_Platform(), platforms.Zynq7020_Platform()]
)
def test_impossible_inseperable_nodes(
    devices: int, nodes: int, network_ports: int, topology: MFTopology, platform: platforms.Platform
) -> None:
    """Check that impossible device node combinations are caught"""

    # TODO: Check all conditions that can fail in the partitioner with regards to
    # inseperable groups

    test_dir_identifier = f"test_all_inseperable_{devices}"
    with custom_build(test_dir_identifier, True) as dirs:
        root, temps, out = dirs
        max_util = 0.9
        ideal_util = 0.8
        considered_resources = ["LUT", "FF", "DSP", "BRAM_18K"]
        res_per_device = available_resources(platform, considered_resources)
        resource_estimates = {
            node: {res: ideal_util * res_per_device[res] for res in considered_resources}
            for node in range(nodes)
        }

        # Ignore working configs, only test wrong configurations
        if (nodes < devices) or ((devices == nodes) and (devices > 1)):
            with pytest.raises(Exception):
                part = AuroraPartitioner(
                    network_ports_per_device=network_ports,
                    strategy=PartitioningStrategy.RESOURCE_UTILIZATION,
                    devices=devices,
                    nodes=nodes,
                    considered_resources=considered_resources,
                    resources_per_device=res_per_device,
                    inseperable_nodes=[list(range(devices))],
                    topology=topology,
                    max_utilization=max_util,
                    ideal_utilization=ideal_util,
                    resource_estimates=resource_estimates,
                )
                solution = part.solve(
                    100,
                    root / "snapshot.txt",
                    root / "solution.txt",
                    {k: f"node_{k}" for k in range(nodes)},
                )
                assert solution is None
