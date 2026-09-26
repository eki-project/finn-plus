"""Pool (HLS) microbenchmark DUT: MaxPool and QuantAvgPool."""

from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model
from typing import Optional

from finn.benchmarking.dut.microbench_base import MicrobenchDUT, check_foreign, stream_width_ok
from finn.benchmarking.param_space import Choice, Conditional, Divisor, Fixed, ParamSpace, Pow2Range
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth


def _is_int_type(name: str) -> bool:
    return name.startswith("INT") or name.startswith("UINT")


def quant_avg_pool_accum_bits(ibits: int, k: int) -> int:
    """Accumulator width of QuantAvgPool2d (as ``qonnx.custom_op.general.QuantAvgPool2d``)."""
    return int((2**ibits - 1) * k * k).bit_length()


class bench_pool(MicrobenchDUT):
    NAME = "pool"
    OP_TYPES = ("Pool_hls",)
    PARAMS = {
        "function": "MaxPool or QuantAvgPool",
        "idt": "input datatype (integer)",
        "odt": "QuantAvgPool only: output datatype (None for MaxPool, which keeps idt)",
        "ch": "channels",
        "pe": "channel parallelism (must divide ch)",
        "k": "kernel size [kh, kw] (square for QuantAvgPool)",
        "odim": "output image size [H, W] (loop bound only)",
    }

    @staticmethod
    def validate(params: dict) -> Optional[str]:
        function = params.get("function")
        if function not in ("MaxPool", "QuantAvgPool"):
            return "function must be MaxPool or QuantAvgPool"
        if not _is_int_type(params["idt"]):
            return "idt must be an INT/UINT datatype"
        idt = DataType[params["idt"]]
        ch, pe = int(params["ch"]), int(params["pe"])
        k_h, k_w = (int(x) for x in params["k"])
        if pe < 1 or ch % pe != 0:
            return "pe must divide ch"
        if min(k_h, k_w) < 1:
            return "kernel must be >= 1"
        if function == "MaxPool":
            reason = check_foreign(params, ["odt"], "MaxPool keeps the input datatype")
            if reason:
                return reason
            odt = idt
        else:
            odt_name = params.get("odt")
            if odt_name is None or not _is_int_type(odt_name):
                return "QuantAvgPool needs an INT/UINT odt"
            odt = DataType[odt_name]
            if odt.signed() != idt.signed():
                return "QuantAvgPool: odt and idt must have the same signedness"
            if odt.bitwidth() > idt.bitwidth():
                return "QuantAvgPool: odt must not be wider than idt"
            if k_h != k_w:
                return "QuantAvgPool requires a square kernel"
        if not stream_width_ok(pe * idt.bitwidth()) or not stream_width_ok(pe * odt.bitwidth()):
            return "stream width exceeds the instrumentation limit"
        return None

    @classmethod
    def param_space(cls) -> ParamSpace:
        return {
            "function": Choice(["MaxPool", "QuantAvgPool"]),
            "idt": Choice(["UINT2", "INT2", "UINT4", "INT4", "UINT8", "INT8"]),
            "odt": Conditional(
                "function", {"MaxPool": Fixed(None)}, Choice(["UINT2", "INT2", "UINT4", "INT4"])
            ),
            "ch": Pow2Range(16, 1024),
            "pe": Divisor("ch", pow2=True, hi=64),
            "k": Choice([[2, 2], [2, 2], [3, 3], [7, 7]]),
            "odim": Fixed([8, 8]),
        }

    @classmethod
    def make_model(cls, params: dict, fpga_part: str):
        function = params["function"]
        idt = DataType[params["idt"]]
        odt = idt if function == "MaxPool" else DataType[params["odt"]]
        ch, pe = int(params["ch"]), int(params["pe"])
        k_h, k_w = (int(x) for x in params["k"])
        odim_h, odim_w = (int(x) for x in params["odim"])
        if function == "QuantAvgPool":
            accum_bits = quant_avg_pool_accum_bits(idt.bitwidth(), k_h)
            size = max(accum_bits - odt.bitwidth(), 0)
        else:
            accum_bits, size = 0, 0

        inp = helper.make_tensor_value_info(
            "inp", TensorProto.FLOAT, [1, odim_h, odim_w, k_h * k_w * ch]
        )
        outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, odim_h, odim_w, ch])
        node = helper.make_node(
            "Pool_hls",
            ["inp"],
            ["outp"],
            domain="finn.custom_op.fpgadataflow.hls",
            backend="fpgadataflow",
            InputDataType=idt.name,
            OutputDataType=odt.name,
            Channels=ch,
            PE=pe,
            KernelSize=[k_h, k_w],
            Function=function,
            OutImgDims=[odim_h, odim_w],
            AccumBits=accum_bits,
            Size=size,
            BatchSize=1,
            cpp_interface="hls_vector",
        )
        graph = helper.make_graph([node], "pool_graph", [inp], [outp])
        model = ModelWrapper(qonnx_make_model(graph, producer_name="pool-model"))
        model.set_tensor_datatype("inp", idt)
        model.set_tensor_datatype("outp", odt)
        # the build's bit width minimization tightens AccumBits from the input datatype;
        # apply it here so dut_info describes the node that gets built
        model = model.transform(MinimizeAccumulatorWidth())
        inst = getCustomOp(model.graph.node[0])
        info = {
            "odt": inst.get_nodeattr("OutputDataType"),
            "in_width": int(inst.get_instream_width()),
            "accum_bits": int(inst.get_nodeattr("AccumBits")),
            "size": size,
        }
        return model, info
