"""Elementwise binary operation (Add/Mul with a constant operand) microbenchmark DUT."""

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
from typing import Optional

from finn.benchmarking.dut.microbench_base import (
    MicrobenchDUT,
    resolve_part,
    specialize_single_node,
    stream_width_ok,
)
from finn.benchmarking.param_space import Choice, Conditional, Fixed, ParamSpace
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth
from finn.util.basic import is_versal

_DTYPES_HLS = ("INT", "UINT", "FLOAT32")


def _dtype_ok(name: str) -> bool:
    return name.startswith("INT") or name.startswith("UINT") or name == "FLOAT32"


def rhs_shape_for(bcast: str, shape: list[int]) -> list[int]:
    """Shape of the constant operand for a broadcast kind (scalar, channel, full)."""
    if bcast == "scalar":
        return [1]
    if bcast == "channel":
        return [int(shape[-1])]
    if bcast == "full":
        return [int(x) for x in shape]
    raise ValueError(f"unknown rhs_bcast {bcast}")


class bench_eltwise(MicrobenchDUT):
    NAME = "eltwise"
    OP_TYPES = (
        "ElementwiseAdd_hls",
        "ElementwiseAdd_rtl",
        "ElementwiseMul_hls",
        "ElementwiseMul_rtl",
    )
    PARAMS = {
        "backend": "hls or rtl (rtl: Versal, FLOAT32 and internal_decoupled only)",
        "op": "Add or Mul",
        "lhs_dtype": "datatype of the streamed operand (INT/UINT or FLOAT32)",
        "rhs_dtype": "datatype of the constant operand (INT/UINT or FLOAT32)",
        "shape": "input/output shape [1, H, W, C]",
        "rhs_bcast": "scalar, channel or full: how the constant is broadcast",
        "pe": "element parallelism (must divide C)",
        "mem_mode": "internal_embedded or internal_decoupled (constant storage)",
        "ram_style": "auto, block, distributed or ultra",
    }

    @staticmethod
    def validate(params: dict) -> Optional[str]:
        backend = params.get("backend")
        if backend not in ("hls", "rtl"):
            return "backend must be hls or rtl"
        if params.get("op") not in ("Add", "Mul"):
            return "op must be Add or Mul"
        lhs, rhs = params["lhs_dtype"], params["rhs_dtype"]
        if not _dtype_ok(lhs) or not _dtype_ok(rhs):
            return "dtypes must be INT/UINT or FLOAT32"
        shape = [int(x) for x in params["shape"]]
        if len(shape) != 4 or shape[0] != 1 or min(shape) < 1:
            return "shape must be [1, H, W, C]"
        pe = int(params["pe"])
        if pe < 1 or shape[-1] % pe != 0:
            return "pe must divide C"
        if params.get("rhs_bcast") not in ("scalar", "channel", "full"):
            return "rhs_bcast must be scalar, channel or full"
        if params.get("mem_mode") not in ("internal_embedded", "internal_decoupled"):
            return "mem_mode must be internal_embedded or internal_decoupled"
        if params.get("ram_style") not in ("auto", "block", "distributed", "ultra"):
            return "invalid ram_style"
        if backend == "rtl":
            if not is_versal(resolve_part(params)):
                return "Elementwise_rtl requires a Versal part"
            if lhs != "FLOAT32" or rhs != "FLOAT32":
                return "Elementwise_rtl requires FLOAT32 operands"
            if params["mem_mode"] != "internal_decoupled":
                return "Elementwise_rtl requires internal_decoupled"
        lhs_bits = DataType[lhs].bitwidth()
        out_bits = 32 if (lhs == "FLOAT32" or rhs == "FLOAT32") else 33
        if not stream_width_ok(pe * lhs_bits) or not stream_width_ok(pe * out_bits):
            return "stream width exceeds the instrumentation limit"
        return None

    @classmethod
    def param_space(cls) -> ParamSpace:
        return {
            "backend": Choice(["hls", "hls", "hls", "rtl"]),
            "op": Choice(["Add", "Mul"]),
            "lhs_dtype": Conditional(
                "backend",
                {"rtl": Fixed("FLOAT32")},
                Choice(["UINT4", "INT4", "UINT8", "INT8", "INT16", "FLOAT32"]),
            ),
            "rhs_dtype": Conditional(
                "backend",
                {"rtl": Fixed("FLOAT32")},
                Choice(["UINT4", "INT4", "UINT8", "INT8", "INT16", "FLOAT32"]),
            ),
            "shape": Choice(
                [
                    [1, 1, 1, 64],
                    [1, 7, 7, 512],
                    [1, 14, 14, 256],
                    [1, 28, 28, 128],
                    [1, 56, 56, 64],
                    [1, 8, 8, 1024],
                ]
            ),
            "rhs_bcast": Choice(["scalar", "channel", "channel", "full"]),
            "pe": Choice([1, 2, 4, 8, 16, 32]),
            "mem_mode": Conditional(
                "backend",
                {"rtl": Fixed("internal_decoupled")},
                Choice(["internal_embedded", "internal_decoupled"]),
            ),
            "ram_style": Choice(["auto", "block", "distributed", "ultra"]),
        }

    @classmethod
    def make_model(cls, params: dict, fpga_part: str):
        backend, op = params["backend"], params["op"]
        lhs_dt, rhs_dt = DataType[params["lhs_dtype"]], DataType[params["rhs_dtype"]]
        shape = [int(x) for x in params["shape"]]
        rhs_shape = rhs_shape_for(params["rhs_bcast"], shape)
        pe = int(params["pe"])
        if lhs_dt.is_integer() and rhs_dt.is_integer():
            out_dt = DataType["INT32"]
        else:
            out_dt = DataType["FLOAT32"]

        inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
        outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)
        node = helper.make_node(
            f"Elementwise{op}",
            ["inp", "const"],
            ["outp"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            lhs_shape=shape,
            rhs_shape=rhs_shape,
            out_shape=shape,
            lhs_dtype=lhs_dt.name,
            rhs_dtype=rhs_dt.name,
            out_dtype=out_dt.name,
            lhs_style="input",
            rhs_style="const",
            PE=pe,
            mem_mode=params["mem_mode"],
            ram_style=params["ram_style"],
        )
        graph = helper.make_graph([node], "eltwise_graph", [inp], [outp])
        model = ModelWrapper(qonnx_make_model(graph, producer_name="eltwise-model"))
        model.set_tensor_datatype("inp", lhs_dt)
        model.set_tensor_datatype("const", rhs_dt)
        model.set_tensor_datatype("outp", out_dt)
        model.set_initializer("const", gen_finn_dt_tensor(rhs_dt, rhs_shape).astype(np.float32))

        model = specialize_single_node(model, backend, fpga_part)
        model = model.transform(MinimizeWeightBitWidth())
        model = model.transform(MinimizeAccumulatorWidth())
        model = model.transform(InferDataTypes())
        inst = getCustomOp(model.graph.node[0])
        info = {
            "rhs_dtype": inst.get_nodeattr("rhs_dtype"),
            "out_dtype": inst.get_nodeattr("out_dtype"),
            "rhs_num_elems": int(np.prod(rhs_shape)),
            "wmem": int(inst.calc_wmem()),
        }
        return model, info
