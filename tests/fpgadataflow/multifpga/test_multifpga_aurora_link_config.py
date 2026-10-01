"""Tests for adding AuroraFlow kernels to the Vitis linking configurations."""

import pytest

import onnx.helper as oh
from onnx import TensorProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.builder.build_dataflow_config import (
    MFCommunicationKernel,
    MFVerbosity,
    PartitioningConfiguration,
)
from finn.transformation.fpgadataflow.multifpga.aurora.link_config_transform import (
    AddAuroraToLinkConfig,
)
from finn.transformation.fpgadataflow.multifpga.aurora.metadata import AuroraNetworkMetadata
from finn.transformation.fpgadataflow.multifpga.create_network_metadata import CreateNetworkMetadata
from finn.transformation.fpgadataflow.vitis_linking_configuration import VitisLinkConfiguration
from finn.util.basic import make_build_dir
from finn.util.fpgadataflow import get_device_id


@pytest.mark.multifpga
@pytest.mark.auroraflow
def test_link_config_shared_aurora_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that an Aurora kernel shared by two SDPs on the same device (one using TX, the
    other RX) is instantiated only once.
    """
    build_dir = Path(make_build_dir("test_aurora_link_config_"))

    # Returnchain n0 (device 0) -> n1 (device 1) -> n2 (device 0). On device 0, n0 uses the TX
    # and n2 the RX direction of the same Aurora kernel.
    tensors = [oh.make_tensor_value_info(f"t_{i}", TensorProto.FLOAT, [1, 1]) for i in range(4)]
    nodes = [
        oh.make_node(
            "StreamingDataflowPartition",
            [tensors[i].name],
            [tensors[i + 1].name],
            name=f"n{i}",
            domain="finn.custom_op.fpgadataflow",
            device_id=device,
        )
        for i, device in enumerate([0, 1, 0])
    ]
    model = ModelWrapper(
        qonnx_make_model(
            oh.make_graph(nodes, inputs=[tensors[0]], outputs=[tensors[-1]], name="graph")
        )
    )
    pcfg = PartitioningConfiguration(
        communication_kernel=MFCommunicationKernel.AURORA, verbosity=MFVerbosity.NONE
    )
    model = model.transform(CreateNetworkMetadata(pcfg))

    # Stand-ins for the packaged Aurora kernels (normally set by PrepareAuroraFlow)
    meta = AuroraNetworkMetadata.load_from_model(model)
    for device, kernels in meta.data.items():
        for index, kernel in enumerate(kernels):
            kernel.aurora_xo = build_dir / f"aurora_{device}_{index}.xo"
            kernel.aurora_xo.touch()
    meta.save()

    # Basic link configs that only contain the SDPs
    configs = {}
    for device in meta.data:
        config = VitisLinkConfiguration(
            build_dir / "link" / str(device) / "config.txt", 100, "", "platform"
        )
        for node in model.graph.node:
            if get_device_id(node) == device:
                config.add_cu(node.name, node.name)
        configs[device] = config
    model = VitisLinkConfiguration.store_to_model(configs, model)

    # Avoid packaging the dummy kernels with Vitis HLS and reading kernel infos from the XOs
    rx_dummy = build_dir / "rx_dummy_kernel.xo"
    tx_dummy = build_dir / "tx_dummy_kernel.xo"
    rx_dummy.touch()
    tx_dummy.touch()
    monkeypatch.setattr(
        AddAuroraToLinkConfig, "package_dummy_kernels", lambda _: (rx_dummy, tx_dummy)
    )
    monkeypatch.setattr(VitisLinkConfiguration, "_get_kerneldefs", lambda _: {})

    model = model.transform(AddAuroraToLinkConfig("U280", "xcu280-fsvh2892-2L-e"))
    configs = VitisLinkConfiguration.load_from_model(model)

    # Each device has a single Aurora kernel, instantiated and connected to its QSFP port once
    for config in configs.values():
        assert config.cu.count("aurora_flow_0") == 1
        assert config.connects.count(("aurora_flow_0/gt_port", "io_gt_qsfp0_00")) == 1
        assert not any(cu.startswith("vdk_") for cu in config.cu)

    # Both directions of the kernel on device 0 are connected to different SDPs
    assert configs[0].sc["n0.m_axis_0"] == ["aurora_flow_0.tx_axis"]
    assert configs[0].sc["aurora_flow_0.rx_axis"] == ["n2.s_axis_0"]
    assert configs[1].sc["aurora_flow_0.rx_axis"] == ["n1.s_axis_0"]
    assert configs[1].sc["n1.m_axis_0"] == ["aurora_flow_0.tx_axis"]
