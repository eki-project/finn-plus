"""MVAU (Matrix Vector Activation Unit) benchmarking module for FINN.

This module provides micro-benchmarking capabilities for FINN's MVAU operator.
The module supports both HLS and RTL backend implementations with configurable
sparsity patterns, data types, and folding parameters.

Key Features:
    - Synthetic MVAU model generation with configurable dimensions and data types
    - Support for various sparsity patterns (unstructured, structured row/column)
    - HLS and RTL backend compatibility with appropriate constraints
    - Automatic SIMD/PE folding parameter calculation and validation
    - Weight and threshold generation with realistic accumulator ranges
    - Integration with FINN's dataflow build pipeline for complete benchmarking

Classes:
    bench_mvau: Specialized benchmark implementation for MVAU operations
"""

import math
import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import (
    calculate_matvec_accumulator_range,
    gen_finn_dt_tensor,
    qonnx_make_model,
)
from typing import Optional

from finn.benchmarking.dut.microbench_base import MicrobenchDUT, stream_width_ok
from finn.benchmarking.param_space import Choice, Conditional, Divisor, Fixed, ParamSpace, Pow2Range
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth

SPARSITY_TYPES = (
    "none",
    "unstructured",
    "rows_random",
    "cols_random",
    "rows_regular",
    "cols_regular",
)


def resolve_folding(params: dict) -> Optional[tuple[int, int]]:
    """(simd, pe) from the sf/nf folding factors (-1 = maximum folding), None if invalid."""
    mw, mh, sf, nf = int(params["mw"]), int(params["mh"]), int(params["sf"]), int(params["nf"])
    if sf > mw or nf > mh or sf == 0 or nf == 0:
        return None
    sf = mw if sf == -1 else sf
    nf = mh if nf == -1 else nf
    simd, pe = mw // sf, mh // nf
    if mw % simd != 0 or mh % pe != 0:
        return None
    return simd, pe


class bench_mvau(MicrobenchDUT):
    """Specialized benchmark class for FINN Matrix Vector Activation Unit (MVAU) operations.

    This class extends the microbenchmark base class to provide MVAU-specific model
    generation. It supports synthetic model creation with configurable matrix dimensions,
    data types, sparsity patterns, and folding parameters for both the HLS and the RTL
    backend.

    Supported Features:
        - Matrix dimensions: configurable input/output widths (mw, mh)
        - Data types: BINARY, BIPOLAR, INT4, INT8, etc. for weights, inputs, and outputs
        - Sparsity: unstructured, structured (row/column), regular patterns
        - Folding: SIMD/PE parameters via the folding factors sf/nf (-1 = maximum)
        - Backends: HLS (LUT-based) and RTL (DSP-based) implementations
        - Memory modes: const, internal_embedded, internal_decoupled
        - Activation functions: configurable threshold-based quantization
    """

    NAME = "mvau"
    OP_TYPES = ("MVAU_hls", "MVAU_rtl")
    PARAMS = {
        "backend": "hls or rtl",
        "idt": "input datatype",
        "wdt": "weight datatype",
        "act": "activation (output) datatype or None for accumulator output",
        "nhw": "number of input vectors, e.g. [1, 32, 32] (loop bound only)",
        "mw": "matrix width (input channels)",
        "mh": "matrix height (output channels)",
        "sf": "synapse folding factor: simd = mw // sf (-1 = mw, i.e. simd = 1)",
        "nf": "neuron folding factor: pe = mh // nf (-1 = mh, i.e. pe = 1)",
        "m": "sample-level parallelism (only 1 supported unless fully unrolled)",
        "sparsity_type": "none, unstructured, rows_random, cols_random, rows_regular, cols_regular",
        "sparsity_amount": "fraction of weights forced to zero (0 for none)",
        "mem_mode": "const, internal_embedded or internal_decoupled",
        "ram_style": "weight memory: auto, block, distributed or ultra",
        "ram_style_thr": "threshold memory: auto, distributed or block",
    }

    @staticmethod
    def validate(params: dict) -> Optional[str]:
        backend = params.get("backend")
        if backend not in ("hls", "rtl"):
            return "backend must be hls or rtl"
        idt, wdt = DataType[params["idt"]], DataType[params["wdt"]]
        act = params.get("act")
        folding = resolve_folding(params)
        if folding is None:
            return "invalid sf/nf folding configuration"
        simd, pe = folding
        mw, mh, m = int(params["mw"]), int(params["mh"]), int(params.get("m", 1))
        if m > 1 and (simd != mw or pe != mh):
            return "m > 1 not possible for non-max simd/pe"
        if backend == "rtl":
            if act is not None:
                return "MVAU_rtl only supports standalone thresholds"
            if params["mem_mode"] != "internal_decoupled":
                return "MVAU_rtl only supports internal_decoupled mem_mode"
            if not wdt.signed():
                return "MVAU_rtl only supports signed weights"
            if idt.bitwidth() < 4 or idt.bitwidth() > 8:
                return "MVAU_rtl supports 4..8 bit inputs"
            if wdt.bitwidth() < 4 or wdt.bitwidth() > 8:
                return "MVAU_rtl supports 4..8 bit weights"
        sparsity_type = params.get("sparsity_type", "none")
        amount = params.get("sparsity_amount", 0) or 0
        if sparsity_type not in SPARSITY_TYPES:
            return f"unknown sparsity type {sparsity_type}"
        if sparsity_type == "none" and amount > 0:
            return "sparsity amount > 0 not applicable for none sparsity"
        if sparsity_type != "none" and amount == 0:
            return "sparsity amount = 0 not applicable for selected sparsity"
        if sparsity_type.endswith("_regular") and amount not in (0.25, 0.5, 0.75):
            return "regular sparsity only applicable for amount 0.25/0.5/0.75"
        odt_bits = DataType[act].bitwidth() if act is not None else 32
        if not stream_width_ok(idt.bitwidth() * simd) or not stream_width_ok(odt_bits * pe):
            return "stream width exceeds the instrumentation limit"
        return None

    @classmethod
    def param_space(cls) -> ParamSpace:
        return {
            "backend": Choice(["hls", "hls", "rtl"]),
            "idt": Conditional(
                "backend",
                {"rtl": Choice(["INT4", "UINT4", "INT5", "INT6", "INT8", "UINT8"])},
                Choice(
                    [
                        "BINARY",
                        "BIPOLAR",
                        "INT2",
                        "UINT2",
                        "INT3",
                        "INT4",
                        "UINT4",
                        "INT6",
                        "INT8",
                        "UINT8",
                    ]
                ),
            ),
            "wdt": Conditional(
                "backend",
                {"rtl": Choice(["INT4", "INT5", "INT6", "INT8"])},
                Choice(["BINARY", "BIPOLAR", "INT2", "INT3", "INT4", "INT6", "INT8"]),
            ),
            "act": Conditional(
                "backend",
                {"rtl": Fixed(None)},
                Choice([None, "BIPOLAR", "UINT2", "INT2", "UINT4", "INT4", "UINT8", "INT8"]),
            ),
            "nhw": Fixed([1, 8, 8]),
            "mw": Pow2Range(16, 2048),
            "mh": Pow2Range(16, 1024),
            "sf": Divisor("mw", pow2=True),
            "nf": Divisor("mh", pow2=True),
            "m": Fixed(1),
            "sparsity_type": Choice(
                ["none", "none", "none", "unstructured", "rows_random", "cols_random"]
            ),
            "sparsity_amount": Conditional(
                "sparsity_type", {"none": Fixed(0)}, Choice([0.25, 0.5, 0.75, 0.9])
            ),
            "mem_mode": Conditional(
                "backend",
                {"rtl": Fixed("internal_decoupled")},
                Choice(["internal_embedded", "internal_decoupled"]),
            ),
            "ram_style": Choice(["auto", "block", "distributed", "ultra"]),
            "ram_style_thr": Conditional(
                "act", {None: Fixed("auto")}, Choice(["auto", "distributed", "block"])
            ),
        }

    @staticmethod
    def _make_single_mvau_model(
        W,
        numInputVectors,
        pe,
        simd,
        m,
        wdt,
        idt,
        odt,
        T=None,
        tdt=None,
        mem_mode="const",
        ram_style="auto",
        ram_style_thresholds="auto",
        backend="hls",
    ):
        """Create a single MVAU ONNX model with specified parameters.

        For BIPOLAR weights and inputs, the method automatically converts to BINARY
        representation and sets binaryXnorMode=1 for efficient XNOR-based computation.
        The model undergoes bit-width minimization optimizations to reduce resource usage.
        """
        mw = W.shape[0]
        mh = W.shape[1]

        # there are two ways to implement bipolar weights and inputs for
        # MatrixVectorActivation:
        # - specify their datatypes as such
        # - specify their datatypes as BINARY as use binaryXnorMode
        if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
            # we'll internally convert weights/inputs to binary and specify the
            # datatypes as such, and also set the binaryXnorMode attribute to 1
            export_wdt = DataType["BINARY"]
            export_idt = DataType["BINARY"]
            binary_xnor_mode = 1
        else:
            export_wdt = wdt
            export_idt = idt
            binary_xnor_mode = 0

        # numInputVectors for dense = [N]
        # numInputVectors for conv  = [N, H, W]
        inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, numInputVectors + [mw])
        outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, numInputVectors + [mh])
        if T is not None:
            no_act = 0
            node_inp_list = ["inp", "weights", "thresh"]
            if odt == DataType["BIPOLAR"]:
                actval = 0
            else:
                actval = odt.min()
        else:
            # no thresholds
            node_inp_list = ["inp", "weights"]
            actval = 0
            no_act = 1

        if backend == "hls":
            customop_name = "MVAU_hls"
            domain = "finn.custom_op.fpgadataflow.hls"
            resType = "lut"
        elif backend == "rtl":
            customop_name = "MVAU_rtl"
            domain = "finn.custom_op.fpgadataflow.rtl"
            resType = "dsp"

        mvau_node = helper.make_node(
            customop_name,
            node_inp_list,
            ["outp"],
            domain=domain,
            backend="fpgadataflow",
            MW=mw,
            MH=mh,
            SIMD=simd,
            PE=pe,
            M=m,
            numInputVectors=numInputVectors,
            inputDataType=export_idt.name,
            weightDataType=export_wdt.name,
            outputDataType=odt.name,
            ActVal=actval,
            binaryXnorMode=binary_xnor_mode,
            noActivation=no_act,
            resType=resType,
            mem_mode=mem_mode,
            ram_style=ram_style,
            ram_style_thresholds=ram_style_thresholds,
            runtime_writeable_weights=0,
        )

        graph = helper.make_graph(
            nodes=[mvau_node], name="mvau_graph", inputs=[inp], outputs=[outp]
        )
        model = qonnx_make_model(graph, producer_name="mvau-model")
        model = ModelWrapper(model)

        model.set_tensor_datatype("inp", idt)
        model.set_tensor_datatype("outp", odt)
        model.set_tensor_datatype("weights", wdt)
        if binary_xnor_mode:
            # convert bipolar to binary
            model.set_initializer("weights", (W + 1) / 2)
        else:
            model.set_initializer("weights", W)
        if T is not None:
            model.set_tensor_datatype("thresh", tdt)
            model.set_initializer("thresh", T)

        # Minimize weight & accumulator width to obtain realistic resource consumption
        model = model.transform(MinimizeWeightBitWidth())
        model = model.transform(MinimizeAccumulatorWidth())
        model = model.transform(InferDataTypes())

        return model

    @staticmethod
    def _apply_sparsity(W: np.ndarray, sparsity_type: str, amount: float) -> np.ndarray:
        mw, mh = W.shape
        if sparsity_type == "none":
            return W
        if sparsity_type == "unstructured":
            idx = np.random.choice(mw * mh, size=int(amount * mw * mh), replace=False)
            W = np.reshape(W, -1)
            W[idx] = 0.0
            return np.reshape(W, (mw, mh))
        if sparsity_type == "rows_random":
            W[np.random.choice(mw, size=int(amount * mw), replace=False), :] = 0.0
            return W
        if sparsity_type == "cols_random":
            W[:, np.random.choice(mh, size=int(amount * mh), replace=False)] = 0.0
            return W
        n = mw if sparsity_type == "rows_regular" else mh
        if amount == 0.25:
            idx = np.arange(0, n, step=4)
        elif amount == 0.5:
            idx = np.arange(0, n, step=2)
        else:
            idx = np.concatenate(
                (np.arange(0, n, step=4), np.arange(1, n, step=4), np.arange(2, n, step=4))
            )
        if sparsity_type == "rows_regular":
            W[idx, :] = 0.0
        else:
            W[:, idx] = 0.0
        return W

    @classmethod
    def make_model(cls, params: dict, fpga_part: str):
        idt, wdt = DataType[params["idt"]], DataType[params["wdt"]]
        act = params.get("act")
        act = DataType[act] if act is not None else None
        numInputVectors = [int(x) for x in params["nhw"]]
        mw, mh, m = int(params["mw"]), int(params["mh"]), int(params.get("m", 1))
        simd, pe = resolve_folding(params)
        backend = params["backend"]

        # Generate weights
        W = gen_finn_dt_tensor(wdt, (mw, mh))
        W = cls._apply_sparsity(
            W, params.get("sparsity_type", "none"), params.get("sparsity_amount", 0) or 0
        )
        # TODO: implement enforce option which prevents naturally occurring sparsity
        # TODO: implement distribution option which selects between uniform/normal/??

        # log resulting sparsity statistics
        # could be higher than selected due to naturally occurring sparsity
        num_zeros = (W == 0).sum()
        num_ones = (W == 1).sum() + (W == -1).sum()
        num_p2 = 0
        for w in np.nditer(W):
            if w != 0 and w != 1 and w != -1:
                if math.log2(abs(w)).is_integer():
                    num_p2 = num_p2 + 1
        info = {
            "simd": simd,
            "pe": pe,
            "zero_weights": round(num_zeros / W.size, 2),
            "easy_weights": round((num_zeros + num_ones + num_p2) / W.size, 2),
        }

        # Generate thresholds
        if act is None:
            # no activation, produce accumulators
            T = None
            tdt = None
            if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
                odt = DataType["UINT32"]
            else:
                odt = DataType["INT32"]
        else:
            odt = act
            # set range for threshold values according to actual accumulator range
            # for the generated weights (this could result in some thresholds being
            # clipped by MinimizeAccumulatorWidth)
            (acc_min, acc_max) = calculate_matvec_accumulator_range(W, idt)
            n_steps = act.get_num_possible_values() - 1
            T = np.random.randint(acc_min, acc_max - 1, (mh, n_steps)).astype(np.float32)
            # provide non-decreasing thresholds
            T = np.sort(T, axis=1)
            # generate thresholds for activation
            if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
                tdt = DataType["UINT32"]
                # bias thresholds to be positive
                T = np.ceil((T + mw) / 2)
                assert (T >= 0).all()
            else:
                tdt = DataType["INT32"]

        model = cls._make_single_mvau_model(
            W,
            numInputVectors,
            pe,
            simd,
            m,
            wdt,
            idt,
            odt,
            T,
            tdt,
            params["mem_mode"],
            params["ram_style"],
            params["ram_style_thr"],
            backend,
        )
        return model, info
