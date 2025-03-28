from __future__ import annotations

import pytest

from qonnx.core.datatype import DataType

from finn.builder.build_dataflow_config import DataflowBuildConfig, PartitioningConfiguration
from finn.builder.build_dataflow_steps import step_partition_for_multifpga
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
def test_no_partition_on_splits() -> None:
    """Make sure that the partitioning transformation doesnt split up during a branch"""
    raise AssertionError()


@pytest.mark.multifpga
def test_found_a_partitioning() -> None:
    raise AssertionError()


@pytest.mark.multifpga
def test_branch_finding() -> None:
    """Test that the function listing all nodes in a branch works as expected"""
    raise AssertionError()
