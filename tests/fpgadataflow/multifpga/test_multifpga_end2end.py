import pytest

from test_multifpga_sdp_creation import create_sdp_ready_model

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    MultiFPGACommunicationScheme,
    PartitioningConfiguration,
)
from finn.builder.build_dataflow_steps import (
    step_create_multifpga_sdp,
    step_prepare_network_infrastructure,
)
from finn.util.basic import make_build_dir


@pytest.mark.multifpga
def test_multifpga_end2end_mobilenet() -> None:
    raise NotImplementedError()


@pytest.mark.multifpga
@pytest.mark.parametrize(
    "device_node_combinations", [(2, 2), (5, 10), (40, 100), (50, 50), (1, 10)]
)
@pytest.mark.parametrize("assignment_type", ["random", "equal"])
@pytest.mark.parametrize("communication_scheme", [MultiFPGACommunicationScheme.AURORA_CHAIN])
@pytest.mark.parametrize("shuffle_devices", [True, False])
def test_multifpga_end2end_artifical_network(
    device_node_combinations: tuple[int, int],
    assignment_type: str,
    communication_scheme: MultiFPGACommunicationScheme,
    shuffle_devices: bool,
) -> None:
    # TODO: Rewrite!
    devices, nodes = device_node_combinations
    assignment_order = None
    if communication_scheme == MultiFPGACommunicationScheme.AURORA_CHAIN:
        assignment_order = "linear"
    else:
        raise NotImplementedError()

    # TODO: Make this model entirely from scratch
    # TODO: Collapse all params and only pass a partitioning config
    temps = make_build_dir("test_end2end_artificial_outputs")
    cfg = DataflowBuildConfig(
        temps,
        5.0,
        [],
        partitioning_configuration=PartitioningConfiguration(
            num_fpgas=devices, communication_scheme=communication_scheme
        ),
    )
    model = create_sdp_ready_model(
        nodes, devices, assignment_type, assignment_order, shuffle_devices
    )
    model.set_metadata_prop("is_multifpga", "True")
    model = step_create_multifpga_sdp(model, cfg)
    model = step_prepare_network_infrastructure(model, cfg)
