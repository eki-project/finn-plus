"""Thresholding (standalone activation) microbenchmark DUT (HLS and RTL backends)."""

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model
from typing import Optional

from finn.benchmarking.dut.microbench_base import (
    MicrobenchDUT,
    check_foreign,
    specialize_single_node,
    stream_width_ok,
)
from finn.benchmarking.param_space import Choice, Conditional, Divisor, Fixed, ParamSpace, Pow2Range
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth


def _is_int_type(name: str) -> bool:
    return name.startswith("INT") or name.startswith("UINT")


class bench_thresholding(MicrobenchDUT):
    NAME = "thresholding"
    OP_TYPES = ("Thresholding_hls", "Thresholding_rtl")
    PARAMS = {
        "backend": "hls or rtl",
        "idt": "input datatype (integer)",
        "odt": "output (activation) datatype, INT/UINT 1..8 bit; numSteps = 2^bits - 1",
        "ch": "number of channels",
        "pe": "channel parallelism (must divide ch)",
        "nhw": "number of input vectors, e.g. [1, 32, 32] (loop bound only)",
        "mem_mode": "hls only: internal_embedded or internal_decoupled (None for rtl)",
        "ram_style": "hls only: distributed or block (None for rtl)",
        "depth_trigger_bram": "rtl only: memory depth from which BRAM is used (0 = off)",
        "depth_trigger_uram": "rtl only: memory depth from which URAM is used (0 = off)",
    }

    @staticmethod
    def validate(params: dict) -> Optional[str]:
        backend = params.get("backend")
        if backend not in ("hls", "rtl"):
            return "backend must be hls or rtl"
        idt_name, odt_name = params["idt"], params["odt"]
        if not _is_int_type(idt_name):
            return "idt must be an INT/UINT datatype"
        if not _is_int_type(odt_name):
            return "odt must be an INT/UINT datatype"
        idt, odt = DataType[idt_name], DataType[odt_name]
        if not 1 <= odt.bitwidth() <= 8:
            return "odt must be 1..8 bit"
        ch, pe = int(params["ch"]), int(params["pe"])
        if pe < 1 or ch % pe != 0:
            return "pe must divide ch"
        if not stream_width_ok(pe * idt.bitwidth()) or not stream_width_ok(pe * odt.bitwidth()):
            return "stream width exceeds the instrumentation limit"
        if backend == "hls":
            if params.get("mem_mode") not in ("internal_embedded", "internal_decoupled"):
                return "hls mem_mode must be internal_embedded or internal_decoupled"
            if params.get("ram_style") not in ("distributed", "block"):
                return "hls ram_style must be distributed or block"
            return check_foreign(
                params, ["depth_trigger_bram", "depth_trigger_uram"], "hls has no depth triggers"
            )
        reason = check_foreign(params, ["mem_mode", "ram_style"], "rtl has no mem_mode/ram_style")
        if reason:
            return reason
        bram, uram = int(params.get("depth_trigger_bram") or 0), int(
            params.get("depth_trigger_uram") or 0
        )
        if bram < 0 or uram < 0:
            return "depth triggers must be >= 0"
        if bram and uram and uram < bram:
            return "depth_trigger_uram must be >= depth_trigger_bram"
        return None

    @classmethod
    def param_space(cls) -> ParamSpace:
        return {
            "backend": Choice(["hls", "rtl"]),
            "idt": Choice(
                ["INT4", "UINT4", "INT8", "UINT8", "INT12", "INT16", "INT20", "INT24", "INT32"]
            ),
            "odt": Choice(
                ["UINT1", "UINT2", "INT2", "UINT3", "UINT4", "INT4", "UINT6", "UINT8", "INT8"]
            ),
            "ch": Pow2Range(16, 1024),
            "pe": Divisor("ch", pow2=True, hi=64),
            "nhw": Fixed([1, 8, 8]),
            "mem_mode": Conditional(
                "backend", {"rtl": Fixed(None)}, Choice(["internal_embedded", "internal_decoupled"])
            ),
            "ram_style": Conditional(
                "backend", {"rtl": Fixed(None)}, Choice(["distributed", "block"])
            ),
            "depth_trigger_bram": Conditional(
                "backend", {"hls": Fixed(0)}, Choice([0, 32, 64, 128, 256, 512, 1024])
            ),
            "depth_trigger_uram": Conditional(
                "backend", {"hls": Fixed(0)}, Choice([0, 0, 1024, 2048, 4096, 8192, 16384])
            ),
        }

    @classmethod
    def make_model(cls, params: dict, fpga_part: str):
        idt, odt = DataType[params["idt"]], DataType[params["odt"]]
        ch, pe = int(params["ch"]), int(params["pe"])
        nhw = [int(x) for x in params["nhw"]]
        n_steps = odt.get_num_possible_values() - 1
        actval = int(odt.min())

        # random thresholds within the input range, non-decreasing per channel
        T = np.random.randint(idt.min(), idt.max(), (ch, n_steps)).astype(np.float32)
        T = np.sort(T, axis=1)
        tdt = DataType["INT32"]

        inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, nhw + [ch])
        outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, nhw + [ch])
        node = helper.make_node(
            "Thresholding",
            ["inp", "thresh"],
            ["outp"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            PE=pe,
            NumChannels=ch,
            numSteps=n_steps,
            inputDataType=idt.name,
            weightDataType=tdt.name,
            outputDataType=odt.name,
            numInputVectors=nhw,
            ActVal=actval,
        )
        graph = helper.make_graph([node], "thresholding_graph", [inp], [outp])
        model = ModelWrapper(qonnx_make_model(graph, producer_name="thresholding-model"))
        model.set_tensor_datatype("inp", idt)
        model.set_tensor_datatype("outp", odt)
        model.set_tensor_datatype("thresh", tdt)
        model.set_initializer("thresh", T)

        model = specialize_single_node(model, params["backend"], fpga_part)
        inst = getCustomOp(model.graph.node[0])
        if params["backend"] == "hls":
            inst.set_nodeattr("mem_mode", params["mem_mode"])
            inst.set_nodeattr("ram_style", params["ram_style"])
        else:
            inst.set_nodeattr("depth_trigger_bram", int(params.get("depth_trigger_bram") or 0))
            inst.set_nodeattr("depth_trigger_uram", int(params.get("depth_trigger_uram") or 0))
        # realistic threshold datatype (the build repeats this step, which is idempotent)
        model = model.transform(MinimizeWeightBitWidth())
        inst = getCustomOp(model.graph.node[0])
        info = {
            "pe": pe,
            "num_steps": n_steps,
            "tdt": inst.get_nodeattr("weightDataType"),
            "tmem": ch // pe,
        }
        return model, info
