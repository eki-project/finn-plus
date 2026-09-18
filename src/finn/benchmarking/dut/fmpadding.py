"""FMPadding (RTL) microbenchmark DUT."""

from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model
from typing import Optional

from finn.benchmarking.dut.microbench_base import (
    MicrobenchDUT,
    specialize_single_node,
    stream_width_ok,
)
from finn.benchmarking.param_space import Choice, Divisor, ParamSpace


class bench_fmpadding(MicrobenchDUT):
    NAME = "fmpadding"
    OP_TYPES = ("FMPadding_rtl",)
    PARAMS = {
        "idt": "input datatype (must be able to represent 0)",
        "ch": "channels",
        "simd": "channel parallelism (must divide ch)",
        "idim": "input image size [H, W]",
        "padding": "padding [top, left, bottom, right]",
    }

    @staticmethod
    def validate(params: dict) -> Optional[str]:
        idt = DataType[params["idt"]]
        if not idt.allowed(0):
            return "idt must be able to represent zero"
        ch, simd = int(params["ch"]), int(params["simd"])
        idim_h, idim_w = (int(x) for x in params["idim"])
        padding = [int(x) for x in params["padding"]]
        if len(padding) != 4 or min(padding) < 0:
            return "padding must be four non-negative values"
        if sum(padding) == 0:
            return "padding must be > 0"
        if simd < 1 or ch % simd != 0:
            return "simd must divide ch"
        if min(idim_h, idim_w) < 1:
            return "idim must be >= 1"
        if not stream_width_ok(simd * idt.bitwidth()):
            return "stream width exceeds the instrumentation limit"
        return None

    @classmethod
    def param_space(cls) -> ParamSpace:
        return {
            "idt": Choice(["BINARY", "UINT2", "INT2", "UINT4", "INT4", "UINT8", "INT8"]),
            "ch": Choice([3, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
            "simd": Divisor("ch", pow2=False, hi=64),
            "idim": Choice(
                [
                    [4, 4],
                    [7, 7],
                    [8, 8],
                    [14, 14],
                    [16, 16],
                    [28, 28],
                    [32, 32],
                    [56, 56],
                    [112, 112],
                    [1, 128],
                ]
            ),
            "padding": Choice(
                [[1, 1, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [2, 2, 2, 2], [3, 3, 3, 3], [0, 1, 0, 1]]
            ),
        }

    @classmethod
    def make_model(cls, params: dict, fpga_part: str):
        idt = DataType[params["idt"]]
        ch, simd = int(params["ch"]), int(params["simd"])
        idim_h, idim_w = (int(x) for x in params["idim"])
        padding = [int(x) for x in params["padding"]]
        odim_h = idim_h + padding[0] + padding[2]
        odim_w = idim_w + padding[1] + padding[3]

        inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, idim_h, idim_w, ch])
        outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, odim_h, odim_w, ch])
        node = helper.make_node(
            "FMPadding",
            ["inp"],
            ["outp"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            ImgDim=[idim_h, idim_w],
            Padding=padding,
            NumChannels=ch,
            inputDataType=idt.name,
            numInputVectors=1,
            SIMD=simd,
        )
        graph = helper.make_graph([node], "fmpadding_graph", [inp], [outp])
        model = ModelWrapper(qonnx_make_model(graph, producer_name="fmpadding-model"))
        model.set_tensor_datatype("inp", idt)
        model.set_tensor_datatype("outp", idt)

        model = specialize_single_node(model, "rtl", fpga_part)
        inst = getCustomOp(model.graph.node[0])
        info = {
            "odim": [odim_h, odim_w],
            "stream_width": int(inst.get_instream_width()),
        }
        return model, info
