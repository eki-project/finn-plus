import pytest

import random
import string
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.analysis.fpgadataflow.unsupported_layers import unsupported_layers
from finn.builder.build_dataflow_steps import step_create_dataflow_partition
from finn.util.exception import FINNUserError


def get_random_name():
    return "".join(random.choices(string.ascii_lowercase, k=10))


def create_tensor():
    return helper.make_tensor_value_info(get_random_name(), TensorProto.FLOAT, None)


def create_node(inputs, outputs, fpga=False, name=None):
    if fpga:
        domain = "finn.custom_op.fpgadataflow"
    else:
        domain = "somethingelse"

    return helper.make_node(
        "CustomOp", [i.name for i in inputs], [o.name for o in outputs], domain=domain, name=name
    )


@pytest.mark.analysis
def test_unsupported_layers_expected_fail():
    inp1 = create_tensor()
    inp2 = create_tensor()

    n1n3 = create_tensor()
    n2n3 = create_tensor()

    n1 = create_node([inp1], [n1n3], fpga=False)
    n2 = create_node([inp2], [n2n3], fpga=False)

    n3n5 = create_tensor()
    n3n4 = create_tensor()

    n3 = create_node([n1n3, n2n3], [n3n4, n3n5], fpga=True)

    n5n6 = create_tensor()
    n4n6 = create_tensor()

    n5 = create_node([n3n5], [n5n6], fpga=False)
    n4 = create_node([n3n4], [n4n6], fpga=True)

    n6n7 = create_tensor()

    n6 = create_node([n5n6, n4n6], [n6n7], fpga=True)

    out = create_tensor()

    n7 = create_node([n6n7], [out], fpga=True)

    mul_graph = helper.make_graph(
        nodes=[n1, n2, n3, n4, n5, n6, n7],
        name="g1",
        inputs=[inp1, inp2],
        outputs=[out],
        value_info=[n1n3, n2n3, n3n4, n3n5, n5n6, n4n6, n6n7],
    )

    model = qonnx_make_model(mul_graph)
    model = ModelWrapper(model)

    ret = unsupported_layers(model)
    assert ret[0] is False, "Model should not be supported, but was not detected as such"


@pytest.mark.analysis
def test_unsupported_layers():
    inp1 = create_tensor()
    inp2 = create_tensor()

    n1n3 = create_tensor()
    n2n3 = create_tensor()

    n1 = create_node([inp1], [n1n3], fpga=False)
    n2 = create_node([inp2], [n2n3], fpga=False)

    n3n5 = create_tensor()
    n3n4 = create_tensor()

    n3 = create_node([n1n3, n2n3], [n3n4, n3n5], fpga=True)

    n5n6 = create_tensor()
    n4n6 = create_tensor()

    n5 = create_node([n3n5], [n5n6], fpga=True)
    n4 = create_node([n3n4], [n4n6], fpga=True)

    n6n7 = create_tensor()

    n6 = create_node([n5n6, n4n6], [n6n7], fpga=True)

    out = create_tensor()

    n7 = create_node([n6n7], [out], fpga=True)

    mul_graph = helper.make_graph(
        nodes=[n1, n2, n3, n4, n5, n6, n7],
        name="g1",
        inputs=[inp1, inp2],
        outputs=[out],
        value_info=[n1n3, n2n3, n3n4, n3n5, n5n6, n4n6, n6n7],
    )

    model = qonnx_make_model(mul_graph)
    model = ModelWrapper(model)

    ret = unsupported_layers(model)
    assert ret[0] is True, "Model should be supported, but was not detected as such"


@pytest.mark.analysis
def test_unsupported_layers_loop():
    inp1 = create_tensor()
    inp2 = create_tensor()

    n1n3 = create_tensor()
    n2n3 = create_tensor()

    n1 = create_node([inp1], [n1n3], fpga=False)
    n2 = create_node([inp2], [n2n3], fpga=False)

    n3n5 = create_tensor()
    n3n4 = create_tensor()

    n3 = create_node([n1n3, n2n3], [n3n4, n3n5], fpga=True)

    n5n6 = create_tensor()
    n4n6 = create_tensor()

    n6n5 = create_tensor()

    n5 = create_node([n3n5, n6n5], [n5n6], fpga=True)
    n4 = create_node([n3n4], [n4n6], fpga=True)

    n6n7 = create_tensor()

    n6 = create_node([n5n6, n4n6], [n6n7, n6n5], fpga=True)

    out = create_tensor()

    n7 = create_node([n6n7], [out], fpga=True)

    mul_graph = helper.make_graph(
        nodes=[n1, n2, n3, n4, n5, n6, n7],
        name="g1",
        inputs=[inp1, inp2],
        outputs=[out],
        value_info=[n1n3, n2n3, n3n4, n3n5, n5n6, n4n6, n6n7],
    )

    model = qonnx_make_model(mul_graph)
    model = ModelWrapper(model)

    ret = unsupported_layers(model)
    assert ret[0] is True, "Model should be supported, but was not detected as such"


@pytest.mark.analysis
def test_large_model():
    inp1 = create_tensor()
    inp2 = create_tensor()

    n1n3 = create_tensor()
    n2n4 = create_tensor()

    n1 = create_node([inp1], [n1n3], fpga=False)
    n2 = create_node([inp2], [n2n4], fpga=False)

    n3n5 = create_tensor()
    n4n5 = create_tensor()
    n4n6 = create_tensor()

    n3 = create_node([n1n3], [n3n5], fpga=True)
    n4 = create_node([n2n4], [n4n5, n4n6], fpga=False)

    out1 = create_tensor()

    n6 = create_node([n4n6], [out1], fpga=False)

    n5n7 = create_tensor()
    n5n8 = create_tensor()

    n5 = create_node([n3n5, n4n5], [n5n7, n5n8], fpga=True)

    out2 = create_tensor()

    n7 = create_node([n5n7], [out2], fpga=True)

    n8n9 = create_tensor()

    n8 = create_node([n5n8], [n8n9], fpga=True)

    n9n10 = create_tensor()

    n9 = create_node([n8n9], [n9n10], fpga=True, name="n9")

    n10n11 = create_tensor()
    n10n12 = create_tensor()

    n10 = create_node([n9n10], [n10n11, n10n12], fpga=False, name="n10")

    out3 = create_tensor()

    n11 = create_node([n10n11], [out3], fpga=False, name="n11")

    out4 = create_tensor()

    n12 = create_node([n10n12], [out4], fpga=True, name="n12")

    mul_graph = helper.make_graph(
        nodes=[n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12],
        name="g2",
        inputs=[inp1, inp2],
        outputs=[out1, out2, out3, out4],
        value_info=[n1n3, n2n4, n3n5, n4n5, n4n6, n5n7, n5n8, n8n9, n9n10, n10n11, n10n12],
    )

    model = qonnx_make_model(mul_graph)
    model = ModelWrapper(model)

    ret = unsupported_layers(model)
    assert ret[0] is False, "Model should not be supported, but was not detected as such"


@pytest.mark.analysis
def test_large_model_step():
    inp1 = create_tensor()
    inp2 = create_tensor()

    n1n3 = create_tensor()
    n2n4 = create_tensor()

    n1 = create_node([inp1], [n1n3], fpga=False)
    n2 = create_node([inp2], [n2n4], fpga=False)

    n3n5 = create_tensor()
    n4n5 = create_tensor()
    n4n6 = create_tensor()

    n3 = create_node([n1n3], [n3n5], fpga=True)
    n4 = create_node([n2n4], [n4n5, n4n6], fpga=False)

    out1 = create_tensor()

    n6 = create_node([n4n6], [out1], fpga=False)

    n5n7 = create_tensor()
    n5n8 = create_tensor()

    n5 = create_node([n3n5, n4n5], [n5n7, n5n8], fpga=True)

    out2 = create_tensor()

    n7 = create_node([n5n7], [out2], fpga=True)

    n8n9 = create_tensor()

    n8 = create_node([n5n8], [n8n9], fpga=True)

    n9n10 = create_tensor()

    n9 = create_node([n8n9], [n9n10], fpga=True, name="n9")

    n10n11 = create_tensor()
    n10n12 = create_tensor()

    n10 = create_node([n9n10], [n10n11, n10n12], fpga=False, name="n10")

    out3 = create_tensor()

    n11 = create_node([n10n11], [out3], fpga=False, name="n11")

    out4 = create_tensor()

    n12 = create_node([n10n12], [out4], fpga=True, name="n12")

    mul_graph = helper.make_graph(
        nodes=[n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12],
        name="g2",
        inputs=[inp1, inp2],
        outputs=[out1, out2, out3, out4],
        value_info=[n1n3, n2n4, n3n5, n4n5, n4n6, n5n7, n5n8, n8n9, n9n10, n10n11, n10n12],
    )

    model = qonnx_make_model(mul_graph)
    model = ModelWrapper(model)

    with pytest.raises(FINNUserError):
        step_create_dataflow_partition(model, None)
