"""StreamingDataWidthConverter microbenchmark DUT (HLS and RTL backends)."""

from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model
from typing import Optional

from finn.benchmarking.dut.microbench_base import (
    MicrobenchDUT,
    specialize_single_node,
    stream_width_ok,
)
from finn.benchmarking.param_space import Choice, Divisor, Fixed, ParamSpace


class bench_dwc(MicrobenchDUT):
    NAME = "dwc"
    OP_TYPES = ("StreamingDataWidthConverter_hls", "StreamingDataWidthConverter_rtl")
    PARAMS = {
        "backend": "hls or rtl (rtl requires an integer width ratio)",
        "dtype": "element datatype",
        "in_elems": "elements per input beat (must divide ch)",
        "out_elems": "elements per output beat (must divide ch)",
        "ch": "elements per vector",
        "n": "vectors per input (loop bound only)",
    }

    @staticmethod
    def validate(params: dict) -> Optional[str]:
        backend = params.get("backend")
        if backend not in ("hls", "rtl"):
            return "backend must be hls or rtl"
        bits = DataType[params["dtype"]].bitwidth()
        in_elems, out_elems = int(params["in_elems"]), int(params["out_elems"])
        ch, n = int(params["ch"]), int(params["n"])
        if min(in_elems, out_elems, ch, n) < 1:
            return "dimensions must be >= 1"
        if in_elems == out_elems:
            return "in_elems and out_elems must differ"
        if ch % in_elems != 0 or ch % out_elems != 0:
            return "in_elems and out_elems must divide ch"
        in_width, out_width = in_elems * bits, out_elems * bits
        if not stream_width_ok(in_width) or not stream_width_ok(out_width):
            return "stream width exceeds the instrumentation limit"
        integer_ratio = in_width % out_width == 0 or out_width % in_width == 0
        if backend == "rtl" and not integer_ratio:
            return "rtl DWC requires an integer width ratio"
        return None

    @classmethod
    def param_space(cls) -> ParamSpace:
        return {
            "backend": Choice(["hls", "rtl"]),
            "dtype": Choice(
                ["BINARY", "UINT2", "UINT4", "INT4", "UINT8", "INT8", "INT16", "INT32"]
            ),
            "ch": Choice([24, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]),
            "in_elems": Divisor("ch", pow2=False, hi=64),
            "out_elems": Divisor("ch", pow2=False, hi=64),
            "n": Fixed(16),
        }

    @classmethod
    def make_model(cls, params: dict, fpga_part: str):
        dtype = DataType[params["dtype"]]
        bits = dtype.bitwidth()
        in_elems, out_elems = int(params["in_elems"]), int(params["out_elems"])
        ch, n = int(params["ch"]), int(params["n"])
        shape = [1, n, ch]
        in_width, out_width = in_elems * bits, out_elems * bits

        inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
        outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)
        node = helper.make_node(
            "StreamingDataWidthConverter",
            ["inp"],
            ["outp"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            inShape=shape,
            outShape=shape,
            inWidth=in_width,
            outWidth=out_width,
            dataType=dtype.name,
        )
        graph = helper.make_graph([node], "dwc_graph", [inp], [outp])
        model = ModelWrapper(qonnx_make_model(graph, producer_name="dwc-model"))
        model.set_tensor_datatype("inp", dtype)
        model.set_tensor_datatype("outp", dtype)

        model = specialize_single_node(model, params["backend"], fpga_part)
        big, small = max(in_width, out_width), min(in_width, out_width)
        info = {
            "in_width": in_width,
            "out_width": out_width,
            "ratio": big / small,
            "integer_ratio": bool(big % small == 0),
        }
        return model, info
