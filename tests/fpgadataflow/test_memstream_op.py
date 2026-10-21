"""Tests of the memstream ops: which ops carry one, and how its wrapper is configured."""

import pytest

from onnx import helper
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import gen_finn_dt_tensor

import finn.custom_op.fpgadataflow as abstract_ops
import finn.custom_op.fpgadataflow.hls as hls_ops
import finn.custom_op.fpgadataflow.rtl as rtl_ops
from finn.custom_op.fpgadataflow.memstream import MemStreamOp, memstream_cascade_height
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.general import ApplyConfig
from tests.fpgadataflow.test_fpgadataflow_mvau import make_single_fclayer_modelwrapper

REGISTRIES = [
    ("finn.custom_op.fpgadataflow", abstract_ops.custom_op),
    ("finn.custom_op.fpgadataflow.hls", hls_ops.custom_op),
    ("finn.custom_op.fpgadataflow.rtl", rtl_ops.custom_op),
]
# attributes some ops require at construction
REQUIRED_ATTRS = {"bodies": 1, "Labels": 1, "heads": 1, "num": 1, "transpose_in_shape": [1, 1]}


def instantiate_all_ops() -> dict[str, CustomOp]:
    """Return {op_type: instance} for every registered fpgadataflow op."""
    return {
        name: getCustomOp(helper.make_node(name, [], [], domain=domain, **REQUIRED_ATTRS))
        for domain, ops in REGISTRIES
        for name in ops
    }


def test_every_fpgadataflow_op_instantiates() -> None:
    """No op is left abstract: MemStreamOp.memstream_depth is implemented wherever the
    class is mixed in, and ops without a memstream need neither calc_wmem nor calc_tmem."""
    assert len(instantiate_all_ops()) > 100


def test_only_memstream_ops_carry_the_memstream() -> None:
    """The memstream code and its node attribute sit on exactly the ops that have one."""
    memstream_ops = {
        name for name, inst in instantiate_all_ops().items() if isinstance(inst, MemStreamOp)
    }
    expected = {
        name
        for _, ops in REGISTRIES
        for name in ops
        if name.startswith(("MVAU", "VVAU", "Elementwise"))
        or name in ("Thresholding_hls", "Requant_rtl", "NodeContainer")
    }
    assert memstream_ops == expected
    for name, inst in instantiate_all_ops().items():
        carries = "ram_cascade_height" in inst.get_nodeattr_types()
        assert carries == (name in expected), name
        assert hasattr(inst, "generate_hdl_memstream") == (name in expected), name


@pytest.mark.parametrize(
    "part,ram_style,depth,height",
    [
        # UltraScale+ block RAM: no cascades, except RAMB36 pairs between 32K and 64K words
        ("xcu55c-fsvh2892-2L-e", "block", 32768, 1),
        ("xcu55c-fsvh2892-2L-e", "auto", 64000, 2),
        ("xczu7ev-ffvc1156-2-e", "block", 65536, 2),
        ("xcu55c-fsvh2892-2L-e", "block", 131072, 1),
        # URAM: 4K words each, cascade what an 8:1 mux cannot take
        ("xcu55c-fsvh2892-2L-e", "ultra", 32768, 1),
        ("xcu55c-fsvh2892-2L-e", "ultra", 40000, 2),
        ("xcu55c-fsvh2892-2L-e", "ultra", 131072, 4),
        # left to synthesis: no RAM cascades, Versal, 7-series
        ("xcu55c-fsvh2892-2L-e", "distributed", 1024, 0),
        ("xcv80-lsva4737-2MHP-e-S", "block", 131072, 0),
        ("xcv80-lsva4737-2MHP-e-S", "ultra", 131072, 0),
        ("xc7z020clg400-1", "block", 131072, 0),
    ],
)
def test_memstream_cascade_height(part: str, ram_style: str, depth: int, height: int) -> None:
    """The rule picks the height per part family, RAM style and depth."""
    assert memstream_cascade_height(part, ram_style, depth) == height


@pytest.mark.parametrize("override,height", [(None, 1), (-1, 1), (0, 0), (3, 3)])
@pytest.mark.fpgadataflow
def test_fpgadataflow_mvau_memstream_cascade_height(
    tmp_path: Path, override: int | None, height: int
) -> None:
    """The memstream wrapper carries the rule's cascade height unless the folding config
    sets one (0 leaves it to synthesis)."""
    weights = gen_finn_dt_tensor(DataType["INT4"], (128, 128))
    model = make_single_fclayer_modelwrapper(
        weights, 1, 1, DataType["INT4"], DataType["INT4"], DataType["INT32"]
    )
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("mem_mode", "internal_decoupled")
    inst.set_nodeattr("ram_style", "block")
    inst.set_nodeattr("preferred_impl_style", "rtl")
    model = model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))
    node = model.graph.node[0]
    inst = getCustomOp(node)
    if override is not None:
        model = model.transform(ApplyConfig({node.name: {"ram_cascade_height": override}}))
        inst = getCustomOp(model.graph.node[0])
    assert isinstance(inst, MemStreamOp)
    inst.set_nodeattr("code_gen_dir_ipgen", str(tmp_path))
    inst.generate_hdl_memstream("xczu7ev-ffvc1156-2-e")
    wrapper = (tmp_path / f"{node.name}_memstream_wrapper.v").read_text()
    assert f"parameter  CASCADE_HEIGHT = {height}," in wrapper
