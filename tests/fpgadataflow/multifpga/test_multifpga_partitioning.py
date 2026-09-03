"""General partitioning related tests."""
import pytest

import yaml
from copy import deepcopy
from fpgadataflow.multifpga.utils import get_model
from pathlib import Path

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    PartitioningConfiguration,
    ShellFlowType,
)
from finn.transformation.fpgadataflow.multifpga.partition_model import ApplyPartitioning
from finn.util.basic import make_build_dir
from finn.util.exception import FINNError
from finn.util.fpgadataflow import get_device_id


def _dump(data: dict, name: str, path: Path) -> Path:
    p = path / name
    with p.open("w+") as f:
        yaml.dump(data, f, yaml.Dumper)
    return p


@pytest.mark.parametrize(
    "model_type",
    [
        ("CNV", 1, 1, True),
        ("CNV", 1, 2, True),
        ("CNV", 2, 2, True),
        ("LFC", 1, 1, True),
        ("LFC", 1, 2, True),
        # FIXME: SFC-1-1 has an issue with negative values during bitwidth minimization
        # ("SFC", 1, 1, True),
        ("SFC", 1, 2, True),
        ("SFC", 2, 2, True),
        ("TFC", 1, 1, True),
        ("TFC", 1, 2, True),
        ("mobilenetv1", 4, 4, True),
        ("resnet18", 4, 4, True),
    ],
)
def test_apply_partitioning(model_type: tuple[str, int, int, bool]) -> None:
    """Test that the partitioning is correctly applied to the model."""
    model_name, wbits, abits, pretrained = model_type

    # Create a dataflow config
    output_dir = make_build_dir("test_apply_partitioning_")
    dump_yaml = lambda data, name: _dump(data, name, Path(output_dir))  # noqa
    cfg = DataflowBuildConfig(
        output_dir=str(output_dir),
        synth_clk_period_ns=5.0,
        board="U55C",
        steps=[],
        target_fps=3000,
        shell_flow_type=ShellFlowType.VITIS_ALVEO,
        partitioning_configuration=PartitioningConfiguration(),
    )

    # Create a model
    model, _ = get_model(
        model_name,
        wbits,
        abits,
        pretrained,
        "step_set_fifo_depths",
        skip_fifo_sizing=True,
        cfg=cfg,
    )

    # Create configs
    partition = {node.name: i for i, node in enumerate(model.graph.node)}
    linear = dump_yaml(partition, "linear.yaml")

    partition = {node.name: 0 for node in model.graph.node}
    same = dump_yaml(partition, "same.yaml")

    partition = {node.name: 0 for node in model.graph.node}
    partition["unknown_node"] = 0
    unknown = dump_yaml(partition, "unknown.yaml")

    partition = {model.graph.node[0].name: 0}
    missing = dump_yaml(partition, "missing.yaml")

    # Standard linear partitioning
    linear_model = deepcopy(model).transform(ApplyPartitioning(linear))
    for i, node in enumerate(linear_model.graph.node):
        assert get_device_id(node) == i

    # All nodes have the same device
    same_model = deepcopy(model).transform(ApplyPartitioning(same))
    for node in same_model.graph.node:
        assert get_device_id(node) == 0

    # We don't allow an assignment to a non-existent node
    with pytest.raises(FINNError):
        _ = deepcopy(model).transform(ApplyPartitioning(unknown))

    # We require all nodes to have a device ID assigned
    with pytest.raises(FINNError):
        _ = deepcopy(model).transform(ApplyPartitioning(missing))
