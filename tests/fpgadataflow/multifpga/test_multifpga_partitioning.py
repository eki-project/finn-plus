from __future__ import annotations

import pytest

import os
from pathlib import Path
from qonnx.core.datatype import DataType

from finn.builder import build_dataflow_steps
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    PartitioningConfiguration,
    ShellFlowType,
)
from finn.builder.build_dataflow_steps import step_partition_for_multifpga
from finn.transformation.fpgadataflow.multifpga_partitioner import PartitionForMultiFPGA
from finn.util.basic import make_build_dir
from tests.fpgadataflow.test_set_folding import make_multi_fclayer_model


@pytest.mark.multifpga
@pytest.mark.parametrize(
    "partition_config", [PartitioningConfiguration(10), PartitioningConfiguration(1), None]
)
def test_multifpga_metadata_info_set_after_partitioning(
    partition_config: PartitioningConfiguration | None,
) -> None:
    dt = DataType["BINARY"]
    model = make_multi_fclayer_model(3, dt, dt, dt, 10)
    output_dir = make_build_dir("test_multifpga_metadata_info_output_dir")
    cfg = DataflowBuildConfig(output_dir, 5.0, [])
    cfg.partitioning_configuration = partition_config

    # TODO: As soon as partitioning move this whole test into the proper partitioning test
    with pytest.raises(Exception):  # noqa
        model = step_partition_for_multifpga(model, cfg)
    if partition_config is not None and partition_config.num_fpgas > 1:
        assert model.get_metadata_prop("is_multifpga") == "True"
    else:
        assert model.get_metadata_prop("is_multifpga") == "False"


@pytest.mark.multifpga
@pytest.mark.slow
@pytest.mark.parametrize(
    "platform", [("U280", ShellFlowType.VITIS_ALVEO), ("Pynq-Z1", ShellFlowType.VIVADO_ZYNQ)]
)
@pytest.mark.parametrize("num_layers", [2, 10])
@pytest.mark.parametrize("dt", [DataType["UINT4"]])
def test_resource_est_for_all_layers(
    platform: tuple[str, ShellFlowType], num_layers: int, dt: DataType
) -> None:
    """Test that resource estimtates for all layers can be found"""
    board, shell = platform
    model = make_multi_fclayer_model(3, dt, dt, dt, num_layers)
    output_dir = make_build_dir("test_res_estimation_")
    steps = [
        "step_qonnx_to_finn",
        "step_tidy_up",
        "step_streamline",
        "step_convert_to_hw",
        "step_create_dataflow_partition",
        "step_specialize_layers",
        "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        "step_generate_estimate_reports",
        "step_hw_codegen",
        "step_hw_ipgen",
        "step_set_fifo_depths",
    ]
    cfg = DataflowBuildConfig(
        output_dir=str(output_dir),
        synth_clk_period_ns=5.0,
        generate_outputs=[DataflowOutputType.ESTIMATE_REPORTS],
        board=board,
        steps=steps,
        target_fps=3000,
        shell_flow_type=shell,
    )

    # Run the first half of the FINN flow
    for step in steps:
        model = build_dataflow_steps.build_dataflow_step_lookup[step](model, cfg)

    # Run the resource estimation
    part = PartitionForMultiFPGA(cfg)
    estimates = part.gather_resource_utilization(model)
    for node in model.graph.node:
        assert node.name in estimates.keys(), f"No estimate found for layer {node.name}"
        for est in estimates[node.name].values():
            assert type(est) in [int, float]  # Efficiency measures use floats

        assert any(
            est > 0 for est in estimates[node.name].values()
        ), f"Layer {node.name} does not use any resources at all: {estimates[node.name]}"


def test_partition_solution_found() -> None:
    """Test some known model - fpga combinations that should
    be solveable"""


def test_aurora_partition_valid() -> None:
    """Test known model - fpga combination solutions and check
    that they are valid for constraints that the Aurora
    Partitioner requires"""


def test_platform_resources() -> None:
    raise NotImplementedError()
