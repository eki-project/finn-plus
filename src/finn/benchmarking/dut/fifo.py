"""StreamingFIFO (RTL backend, rtl or vivado implementation) microbenchmark DUT."""

from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model
from typing import Optional

from finn.benchmarking.dut.microbench_base import (
    MICROBENCH_BUILD_STEPS,
    MicrobenchDUT,
    check_foreign,
    specialize_single_node,
    stream_width_ok,
)
from finn.benchmarking.param_space import Choice, Conditional, Fixed, ParamSpace


class bench_fifo(MicrobenchDUT):
    NAME = "fifo"
    OP_TYPES = ("StreamingFIFO_rtl",)
    PARAMS = {
        "impl_style": "rtl (SRL/LUT based) or vivado (FIFO generator IP)",
        "dtype": "element datatype",
        "elems": "elements per stream beat (width = elems * bits)",
        "n": "beats per input vector (loop bound only)",
        "depth": "FIFO depth",
        "ram_style": "vivado only: auto, block, distributed or ultra (None for rtl)",
    }

    @staticmethod
    def validate(params: dict) -> Optional[str]:
        impl_style = params.get("impl_style")
        if impl_style not in ("rtl", "vivado"):
            return "impl_style must be rtl or vivado"
        bits = DataType[params["dtype"]].bitwidth()
        elems, n, depth = int(params["elems"]), int(params["n"]), int(params["depth"])
        if elems < 1 or n < 1:
            return "elems and n must be >= 1"
        if depth < 2:
            return "depth must be >= 2"
        if not stream_width_ok(elems * bits):
            return "stream width exceeds the instrumentation limit"
        if impl_style == "vivado":
            if depth < 16:
                return "vivado FIFOs need depth >= 16"
            if params.get("ram_style") not in ("auto", "block", "distributed", "ultra"):
                return "vivado ram_style must be auto, block, distributed or ultra"
            return None
        return check_foreign(params, ["ram_style"], "rtl FIFOs have no ram_style")

    @classmethod
    def param_space(cls) -> ParamSpace:
        return {
            "impl_style": Choice(["rtl", "vivado"]),
            "dtype": Choice(
                ["BINARY", "UINT2", "UINT4", "INT4", "UINT8", "INT8", "INT16", "INT32"]
            ),
            "elems": Choice([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]),
            "n": Fixed(64),
            "depth": Conditional(
                "impl_style",
                {"rtl": Choice([2, 4, 8, 10, 12, 16, 24, 32, 40, 48, 64, 72, 96, 128, 192, 256])},
                Choice([512, 1024, 2048, 4096, 8192, 16384, 32768]),
            ),
            "ram_style": Conditional(
                "impl_style",
                {"rtl": Fixed(None)},
                Choice(["auto", "block", "distributed", "ultra"]),
            ),
        }

    @classmethod
    def build_steps(cls, params: dict) -> list[str]:
        steps = list(MICROBENCH_BUILD_STEPS)
        if params.get("impl_style") == "vivado":
            # the Vivado FIFO IP cannot be rtl-simulated
            steps.remove("step_measure_rtlsim_performance")
        return steps

    @classmethod
    def make_model(cls, params: dict, fpga_part: str):
        dtype = DataType[params["dtype"]]
        elems, n, depth = int(params["elems"]), int(params["n"]), int(params["depth"])
        normal_shape = [1, n * elems]
        folded_shape = [1, n, elems]

        inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, normal_shape)
        outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, normal_shape)
        node = helper.make_node(
            "StreamingFIFO",
            ["inp"],
            ["outp"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            depth=depth,
            folded_shape=folded_shape,
            normal_shape=normal_shape,
            dataType=dtype.name,
        )
        graph = helper.make_graph([node], "fifo_graph", [inp], [outp])
        model = ModelWrapper(qonnx_make_model(graph, producer_name="fifo-model"))
        model.set_tensor_datatype("inp", dtype)
        model.set_tensor_datatype("outp", dtype)

        model = specialize_single_node(model, "rtl", fpga_part)
        inst = getCustomOp(model.graph.node[0])
        inst.set_nodeattr("impl_style", params["impl_style"])
        if params["impl_style"] == "vivado":
            inst.set_nodeattr("ram_style", params["ram_style"])
        width = int(inst.get_instream_width())
        depth_adjusted = int(inst.get_adjusted_depth())
        info = {
            "width_bits": width,
            "depth_adjusted": depth_adjusted,
            "capacity_bits": width * depth_adjusted,
        }
        return model, info
