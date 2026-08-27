import pytest

from fpgadataflow.multifpga.utils import get_model
from typing import TYPE_CHECKING, cast

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    PartitioningConfiguration,
    ShellFlowType,
)
from finn.util.basic import make_build_dir
from finn.util.resources import get_estimated_model_resources

if TYPE_CHECKING:
    from numbers import Real


@pytest.mark.multifpga
@pytest.mark.slow
@pytest.mark.parametrize(
    "platform", [("U55C", ShellFlowType.VITIS_ALVEO), ("Pynq-Z1", ShellFlowType.VIVADO_ZYNQ)]
)
@pytest.mark.parametrize(
    "model_type",
    [
        ("CNV", 1, 1, True),
        ("CNV", 1, 2, True),
        ("CNV", 2, 2, True),
        ("LFC", 1, 1, True),
        ("LFC", 1, 2, True),
        ("SFC", 1, 2, True),
        ("SFC", 2, 2, True),
        ("TFC", 1, 1, True),
        ("TFC", 1, 2, True),
        ("mobilenetv1", 4, 4, True),
        ("resnet18", 4, 4, True),
    ],
)
def test_resource_est_for_all_layers(
    model_type: tuple[str, int, int, bool],
    platform: tuple[str, ShellFlowType],
    pytestconfig: pytest.Config,
) -> None:
    """Test that resource estimtates for all layers can be found."""
    board, shell = platform
    model_name, wbits, abits, pretrained = model_type

    # Create a dataflow config
    output_dir = make_build_dir("test_res_estimation_")
    cfg = DataflowBuildConfig(
        output_dir=str(output_dir),
        synth_clk_period_ns=5.0,
        generate_outputs=[DataflowOutputType.ESTIMATE_REPORTS],
        board=board,
        steps=[],
        target_fps=3000,
        shell_flow_type=shell,
        partitioning_configuration=PartitioningConfiguration(),
    )

    model, _ = get_model(
        model_name,
        wbits,
        abits,
        pretrained,
        "step_set_fifo_depths",
        skip_fifo_sizing=True,
        cfg=cfg,
        pytestconfig=pytestconfig,
    )

    # Run the resource estimation
    assert cfg.partitioning_configuration is not None
    estimates: dict[str, dict[str, int | float]] = get_estimated_model_resources(
        model,
        fpga_part=cfg._resolve_fpga_part(),  # noqa
        considered_resources=cfg.partitioning_configuration.considered_resources,
        add_missing_resources=True,
    )

    # Checks
    for node in model.graph.node:
        # Chat every layer has an estimate
        assert node.name in estimates.keys(), f"No estimate found for layer {node.name}"

        # Assert that every estimate is either an int or a float
        for est in estimates[node.name].values():
            assert type(est) in [int, float]  # Efficiency measures use floats

        # Assert that every layer uses any resource at all
        assert any(est > 0 for est in estimates[node.name].values()), (
            f"Layer {node.name} does not use " f"any resources at all: {estimates[node.name]}"
        )

    # Check that resource estimates were added if resource type was not used in a layer
    if model_name == "CNV" and wbits == 2 and abits == 2:
        no_ff = [f"ConvolutionInputGenerator_rtl_{i}" for i in range(8)]
        layers_seen: dict[str, bool] = dict.fromkeys(no_ff, False)
        for node in model.graph.node:
            if node.name in no_ff:
                layers_seen[node.name] = True
                assert estimates[node.name]["FF"] == 0
        assert all(
            list(layers_seen.values())
        ), "Not all expected layers were seen for this model type!"
