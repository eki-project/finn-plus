"""VVAU (vector-vector activation unit, depthwise convolution) microbenchmark DUT."""

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import (
    calculate_matvec_accumulator_range,
    gen_finn_dt_tensor,
    qonnx_make_model,
)
from typing import Optional

from finn.benchmarking.dut.microbench_base import (
    MicrobenchDUT,
    check_foreign,
    resolve_part,
    stream_width_ok,
)
from finn.benchmarking.param_space import Choice, Conditional, Divisor, Fixed, ParamSpace, Pow2Range
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth
from finn.util.basic import is_versal


class bench_vvau(MicrobenchDUT):
    NAME = "vvau"
    OP_TYPES = ("VVAU_hls", "VVAU_rtl")
    PARAMS = {
        "backend": "hls or rtl (rtl requires a Versal part)",
        "idt": "input datatype",
        "wdt": "weight datatype",
        "act": "activation (output) datatype or None for accumulator output",
        "ch": "channels",
        "dim": "spatial output size [H, W] (loop bound only)",
        "k": "kernel size [kh, kw]",
        "pe": "channel parallelism (must divide ch)",
        "simd": "kernel parallelism (must divide kh*kw)",
        "mem_mode": "internal_embedded or internal_decoupled",
        "ram_style": "auto, block, distributed or ultra",
        "resType": "hls only: lut or dsp (None for rtl)",
    }

    @staticmethod
    def validate(params: dict) -> Optional[str]:
        backend = params.get("backend")
        if backend not in ("hls", "rtl"):
            return "backend must be hls or rtl"
        idt, wdt = DataType[params["idt"]], DataType[params["wdt"]]
        act = params.get("act")
        ch, pe, simd = int(params["ch"]), int(params["pe"]), int(params["simd"])
        k_h, k_w = (int(x) for x in params["k"])
        if pe < 1 or ch % pe != 0:
            return "pe must divide ch"
        if simd < 1 or (k_h * k_w) % simd != 0:
            return "simd must divide kh*kw"
        if params.get("mem_mode") not in ("internal_embedded", "internal_decoupled"):
            return "mem_mode must be internal_embedded or internal_decoupled"
        if params.get("ram_style") not in ("auto", "block", "distributed", "ultra"):
            return "invalid ram_style"
        if backend == "hls":
            if params.get("resType") not in ("lut", "dsp"):
                return "hls resType must be lut or dsp"
        else:
            reason = check_foreign(params, ["resType"], "rtl always uses DSPs")
            if reason:
                return reason
            if not is_versal(resolve_part(params)):
                return "VVAU_rtl requires a Versal part"
            if act is not None:
                return "VVAU_rtl does not support integrated activation"
            if not wdt.signed():
                return "VVAU_rtl requires signed weights"
            if (
                wdt.bitwidth() > 8
                or idt.bitwidth() > 9
                or (idt.bitwidth() == 9 and not idt.signed())
            ):
                return "VVAU_rtl supports up to 8 bit weights and 8 bit (9 bit signed) inputs"
        odt_bits = DataType[act].bitwidth() if act is not None else 32
        if not stream_width_ok(idt.bitwidth() * simd * pe) or not stream_width_ok(odt_bits * pe):
            return "stream width exceeds the instrumentation limit"
        return None

    @classmethod
    def param_space(cls) -> ParamSpace:
        return {
            "backend": Choice(["hls", "hls", "rtl"]),
            "idt": Choice(["UINT2", "INT2", "UINT4", "INT4", "UINT8", "INT8"]),
            "wdt": Choice(["INT2", "INT3", "INT4", "INT6", "INT8"]),
            "act": Conditional(
                "backend", {"rtl": Fixed(None)}, Choice([None, "UINT2", "UINT4", "INT4", "UINT8"])
            ),
            "ch": Pow2Range(32, 1024),
            "dim": Fixed([7, 7]),
            "k": Choice([[3, 3], [3, 3], [5, 5], [1, 3], [1, 5]]),
            "pe": Divisor("ch", pow2=True, hi=64),
            "simd": Choice([1, 3, 9]),
            "mem_mode": Choice(["internal_embedded", "internal_decoupled"]),
            "ram_style": Choice(["auto", "block", "distributed", "ultra"]),
            "resType": Conditional("backend", {"rtl": Fixed(None)}, Choice(["lut", "dsp"])),
        }

    @classmethod
    def make_model(cls, params: dict, fpga_part: str):
        backend = params["backend"]
        idt, wdt = DataType[params["idt"]], DataType[params["wdt"]]
        act = params.get("act")
        ch, pe, simd = int(params["ch"]), int(params["pe"]), int(params["simd"])
        dim_h, dim_w = (int(x) for x in params["dim"])
        k_h, k_w = (int(x) for x in params["k"])

        if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
            export_wdt, export_idt, binary_xnor_mode = DataType["BINARY"], DataType["BINARY"], 1
        else:
            export_wdt, export_idt, binary_xnor_mode = wdt, idt, 0

        W = gen_finn_dt_tensor(wdt, (ch, 1, k_h, k_w))
        zero_weights = round(float((W == 0).sum()) / W.size, 2)

        if act is None:
            T, tdt, actval, no_act = None, None, 0, 1
            odt = DataType["UINT32"] if binary_xnor_mode else DataType["INT32"]
        else:
            odt = DataType[act]
            no_act = 0
            actval = 0 if odt == DataType["BIPOLAR"] else int(odt.min())
            # thresholds within the accumulator range of each channel's k_h*k_w weights
            (acc_min, acc_max) = calculate_matvec_accumulator_range(W.reshape(ch, k_h * k_w).T, idt)
            n_steps = odt.get_num_possible_values() - 1
            T = np.random.randint(acc_min, max(acc_min + 1, acc_max - 1), (ch, n_steps)).astype(
                np.float32
            )
            T = np.sort(T, axis=1)
            if binary_xnor_mode:
                tdt = DataType["UINT32"]
                T = np.ceil((T + k_h * k_w) / 2)
            else:
                tdt = DataType["INT32"]

        inp = helper.make_tensor_value_info(
            "inp", TensorProto.FLOAT, [1, dim_h, dim_w, k_h * k_w * ch]
        )
        outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, dim_h, dim_w, ch])
        node = helper.make_node(
            f"VVAU_{backend}",
            ["inp", "weights"] + (["thresh"] if T is not None else []),
            ["outp"],
            domain=f"finn.custom_op.fpgadataflow.{backend}",
            backend="fpgadataflow",
            PE=pe,
            SIMD=simd,
            Dim=[dim_h, dim_w],
            Channels=ch,
            Kernel=[k_h, k_w],
            resType=params["resType"] if backend == "hls" else "dsp",
            ActVal=actval,
            inputDataType=export_idt.name,
            weightDataType=export_wdt.name,
            outputDataType=odt.name,
            noActivation=no_act,
            mem_mode=params["mem_mode"],
            ram_style=params["ram_style"],
            binaryXnorMode=binary_xnor_mode,
        )
        graph = helper.make_graph([node], "vvau_graph", [inp], [outp])
        model = ModelWrapper(qonnx_make_model(graph, producer_name="vvau-model"))
        model.set_tensor_datatype("inp", idt)
        model.set_tensor_datatype("outp", odt)
        model.set_tensor_datatype("weights", wdt)
        model.set_initializer("weights", (W + 1) / 2 if binary_xnor_mode else W)
        if T is not None:
            model.set_tensor_datatype("thresh", tdt)
            model.set_initializer("thresh", T)

        model = model.transform(MinimizeWeightBitWidth())
        model = model.transform(MinimizeAccumulatorWidth())
        model = model.transform(InferDataTypes())
        inst = getCustomOp(model.graph.node[0])
        info = {
            "pe": pe,
            "simd": simd,
            "zero_weights": zero_weights,
            "wmem": int(inst.calc_wmem()),
            "tmem": int(inst.calc_tmem()),
        }
        return model, info
