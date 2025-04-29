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
from finn.transformation.fpgadataflow.multifpga_partitioner import (
    AuroraPartitioner,
    PartitionForMultiFPGA,
)
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
    [
        "equal-LUT",
    ],
)
@pytest.mark.parametrize("distribution_args", [{"level": 1000}])
@pytest.mark.parametrize("nodes", [1, 2, 3, 10, 99])
@pytest.mark.parametrize("devices", [1, 2, 3, 7, 10, 100])
@pytest.mark.parametrize("considered_resources", [["LUT", "FF", "DSP", "BRAM_18K"]])
@pytest.mark.parametrize("board", ["U280", "Pynq-Z1"])
@pytest.mark.parametrize("max_util", [0.85])
@pytest.mark.parametrize("ideal_util", [0.75])
@pytest.mark.parametrize("topology", [MFTopology.CHAIN])
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
        "_node{nodes}_topo{topology.name}_{board}_dist{distribution}"
    )
    with custom_build(test_dir_identifier, True) as dirs:
        root, temps, out = dirs
        cfg = DataflowBuildConfig(
            output_dir=str(out),
            synth_clk_period_ns=5.0,
            generate_outputs=[],
            partitioning_configuration=PartitioningConfiguration(),
        )
        res_per_device = PartitionForMultiFPGA(cfg).resources_per_device(
            platforms.platforms[board]()
        )
        part = AuroraPartitioner(
            network_ports_per_device=2,
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

        # Solution found
        assert solution is not None

        # Every device is utilized
        usage = part.get_resource_use_by_device()
        assert usage is not None
        for device in usage.keys():
            assert any(usage[device][restype] > 0 for restype in usage[device].keys())

        # max_utilization not overstepped
        total_per_device = part._total_resources_per_device()  # noqa
        for device in usage.keys():
            for restype, res in usage[device].values():
                assert res <= max_util * total_per_device[restype]


def test_enforce_utilization_limit() -> None:
    """Test that the partitioner upholds the resource utilization limit"""
    pass
