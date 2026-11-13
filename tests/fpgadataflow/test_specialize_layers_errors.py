"""Tests for the typed errors of the SpecializeLayers transformation."""

import pytest

from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

import finn.transformation.fpgadataflow.specialize_layers as specialize_layers
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.exception import FINNConfigurationError, FINNUnsupportedOpError, FINNUserError

FPGA_PART = "xc7z020clg400-1"


def make_globalaccpool_model(preferred_impl_style: str = "") -> ModelWrapper:
    """Return a model with a single GlobalAccPool layer."""
    node = helper.make_node(
        "GlobalAccPool",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        name="pool",
        NumChannels=8,
        PE=1,
        inputDataType="UINT4",
        numInputVectors=[1, 4, 4],
        preferred_impl_style=preferred_impl_style,
    )
    graph = helper.make_graph(
        [node],
        "graph",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4, 4, 8])],
        [helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 1, 1, 8])],
    )
    return ModelWrapper(qonnx_make_model(graph))


@pytest.mark.fpgadataflow
@pytest.mark.parametrize("preferred_impl_style", ["", "hls", "rtl"])
def test_specialize_layers_no_hw_variant(
    monkeypatch: pytest.MonkeyPatch, preferred_impl_style: str
) -> None:
    """A layer without HLS and RTL variant raises a FINNUnsupportedOpError naming the node."""
    # pretend that no implementation of the layer is registered
    monkeypatch.setattr(specialize_layers, "hls_variants", {})
    monkeypatch.setattr(specialize_layers, "rtl_variants", {})
    model = make_globalaccpool_model(preferred_impl_style)
    with pytest.raises(FINNUnsupportedOpError) as excinfo:
        model.transform(SpecializeLayers(FPGA_PART))
    assert isinstance(excinfo.value, FINNUserError)
    assert excinfo.value.node_name == "pool"
    assert excinfo.value.op_type == "GlobalAccPool"
    assert excinfo.value.unmapped_nodes == []
    assert "Node pool with optype GlobalAccPool has no hw implementation variant" in str(
        excinfo.value
    )


@pytest.mark.fpgadataflow
def test_specialize_layers_with_hw_variant() -> None:
    """The layer specializes as usual if a variant is registered."""
    model = make_globalaccpool_model().transform(SpecializeLayers(FPGA_PART))
    assert model.graph.node[0].op_type == "GlobalAccPool_hls"


@pytest.mark.fpgadataflow
def test_specialize_layers_invalid_preferred_impl_style(monkeypatch: pytest.MonkeyPatch) -> None:
    """An invalid preferred_impl_style raises a FINNConfigurationError."""

    class InvalidImplStyle:
        """Stand-in for a custom op whose preferred_impl_style is neither hls nor rtl."""

        def get_nodeattr(self, name: str) -> str:
            """Return the invalid implementation style."""
            assert name == "preferred_impl_style"
            return "verilog"

    monkeypatch.setattr(specialize_layers, "getCustomOp", lambda _: InvalidImplStyle())
    model = make_globalaccpool_model()
    with pytest.raises(FINNConfigurationError) as excinfo:
        model.transform(SpecializeLayers(FPGA_PART))
    assert isinstance(excinfo.value, FINNUserError)
    assert "Invalid value for attribute preferred_impl_style" in str(excinfo.value)
    assert "verilog" in str(excinfo.value)
