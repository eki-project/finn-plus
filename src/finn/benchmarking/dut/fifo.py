"""StreamingFIFO (RTL) microbenchmark DUT.

All RTL FIFOs are built from ``finn-rtllib/fifo/hdl/fifo.sv``; ``ram_style`` requests the
storage (``auto`` lets the RTL's ladder decide, ``srl`` a shift register, ``block`` BRAM,
``distributed`` LUTRAM, ``ultra`` URAM) and ``StreamingFIFO.resolve_ram_style()`` predicts
what actually gets built (recorded as ``ram_style_eff``).
"""

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
from finn.benchmarking.param_space import Choice, Fixed, ParamSpace

RAM_STYLES = ("auto", "srl", "block", "distributed", "ultra")


class bench_fifo(MicrobenchDUT):
    NAME = "fifo"
    OP_TYPES = ("StreamingFIFO_rtl",)
    PARAMS = {
        "dtype": "element datatype",
        "elems": "elements per stream beat (width = elems * bits)",
        "n": "beats per input vector (loop bound only)",
        "depth": "FIFO depth",
        "ram_style": "requested storage: auto, srl, block, distributed or ultra",
    }

    @staticmethod
    def validate(params: dict) -> Optional[str]:
        bits = DataType[params["dtype"]].bitwidth()
        elems, n, depth = int(params["elems"]), int(params["n"]), int(params["depth"])
        if elems < 1 or n < 1:
            return "elems and n must be >= 1"
        if depth < 2:
            return "depth must be >= 2"
        if not stream_width_ok(elems * bits):
            return "stream width exceeds the instrumentation limit"
        if params.get("ram_style") not in RAM_STYLES:
            return f"ram_style must be one of {RAM_STYLES}"
        return None

    @classmethod
    def param_space(cls) -> ParamSpace:
        return {
            "dtype": Choice(
                ["BINARY", "UINT2", "UINT4", "INT4", "UINT8", "INT8", "INT16", "INT32"]
            ),
            "elems": Choice([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]),
            "n": Fixed(64),
            "depth": Choice(
                [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
            ),
            "ram_style": Choice(list(RAM_STYLES)),
        }

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
            ram_style=params["ram_style"],
        )
        graph = helper.make_graph([node], "fifo_graph", [inp], [outp])
        model = ModelWrapper(qonnx_make_model(graph, producer_name="fifo-model"))
        model.set_tensor_datatype("inp", dtype)
        model.set_tensor_datatype("outp", dtype)

        model = specialize_single_node(model, "rtl", fpga_part)
        inst = getCustomOp(model.graph.node[0])
        inst.set_nodeattr("impl_style", "rtl")
        width = int(inst.get_instream_width())
        info = {
            "width_bits": width,
            "ram_style_eff": inst.resolve_ram_style(),
            "capacity_bits": width * depth,
        }
        return model, info
