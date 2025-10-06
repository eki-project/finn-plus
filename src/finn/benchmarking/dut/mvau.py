"""
MVAU (Matrix Vector Activation Unit) benchmarking module for FINN.

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

import json
import math
import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import (
    calculate_matvec_accumulator_range,
    gen_finn_dt_tensor,
    qonnx_make_model,
)

import finn.builder.build_dataflow_config as build_cfg
from finn.benchmarking.bench_base import bench
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth


class bench_mvau(bench):
    """
    Specialized benchmark class for FINN Matrix Vector Activation Unit (MVAU) operations.

    This class extends the base benchmark class to provide MVAU-specific model generation
    and benchmarking capabilities. It supports synthetic model creation with configurable
    matrix dimensions, data types, sparsity patterns, and folding parameters.

    The class handles both HLS and RTL backend implementations with appropriate constraints
    and optimizations. It generates realistic MVAU models for performance characterization
    and resource utilization analysis.

    Supported Features:
        - Matrix dimensions: configurable input/output widths (mw, mh)
        - Data types: BINARY, BIPOLAR, INT4, INT8, etc. for weights, inputs, and outputs
        - Sparsity: unstructured, structured (row/column), regular patterns
        - Folding: SIMD/PE parameters for parallel processing optimization
        - Backends: HLS (LUT-based) and RTL (DSP-based) implementations
        - Memory modes: const, internal_embedded, internal_decoupled
        - Activation functions: configurable threshold-based quantization
    """

    def _make_single_mvau_model(
        self,
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
        """
        Create a single MVAU ONNX model with specified parameters.

        This method constructs a complete ONNX model containing a single MVAU node
        with the given weight matrix, data types, and configuration parameters.
        It handles both HLS and RTL backend variants with appropriate optimizations.

        Args:
            W (np.ndarray): Weight matrix of shape (mw, mh) containing the weights
            numInputVectors (list): Input tensor shape prefix ([N] for dense, [N,H,W] for conv)
            pe (int): Number of output channels computed in parallel
            simd (int): Number of input channels processed in parallel
            m (int): Sample-level parallelism factor (currently unused)
            wdt (DataType): Weight data type (e.g., BINARY, INT8)
            idt (DataType): Input data type (e.g., BINARY, INT8)
            odt (DataType): Output data type (e.g., INT32, BIPOLAR)
            T (np.ndarray, optional): Threshold matrix for activation quantization.
                                    Defaults to None (no activation).
            tdt (DataType, optional): Threshold data type. Defaults to None.
            mem_mode (str, optional): Memory mode for weights storage. Options:
                                    "const", "internal_embedded", "internal_decoupled".
                                    Defaults to "const".
            ram_style (str, optional): RAM style for weight storage. Defaults to "auto".
            ram_style_thresholds (str, optional): RAM style for thresholds. Defaults to "auto".
            backend (str, optional): Implementation backend. "hls" or "rtl". Defaults to "hls".

        Returns:
            ModelWrapper: Complete ONNX model with optimized MVAU implementation

        Note:
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
        # model.set_tensor_shape("weights", (channels, 1, k_h, k_w)) from VVAU
        if binary_xnor_mode:
            # convert bipolar to binary
            model.set_initializer("weights", (W + 1) / 2)
        else:
            model.set_initializer("weights", W)
        if T is not None:
            model.set_tensor_datatype("thresh", tdt)
            model.set_initializer("thresh", T)

        # Minimize weight & accumulator width to obtain realistic resource consumption
        # model = model.transform(InferShapes())
        model = model.transform(MinimizeWeightBitWidth())
        model = model.transform(MinimizeAccumulatorWidth())
        model = model.transform(InferDataTypes())

        return model

    def _step_export_onnx(self, onnx_export_path):
        """
        Generate and export a synthetic MVAU ONNX model for benchmarking.

        This method creates a synthetic MVAU model based on the benchmark parameters,
        including matrix dimensions, data types, sparsity patterns, and folding configuration.
        It performs comprehensive parameter validation and generates realistic weights and
        thresholds for accurate performance characterization.

        Args:
            onnx_export_path (str): Path where the generated ONNX model will be saved

        Returns:
            str: "skipped" if the parameter configuration is invalid or unsupported,
                 otherwise None indicating successful model generation

        Parameter Requirements:
            - idt, wdt, act: Input, weight, and activation data types (strings)
            - nhw: Number of input vectors (list for tensor shape)
            - mw, mh: Matrix width (input features) and height (output features)
            - sf, nf: Synapse (SIMD) and Neuron (PE) folding factors (-1 for maximum folding)
            - m: Sample-level parallelism factor (currently unused)
            - mem_mode: Weight memory mode
            - ram_style, ram_style_thr: RAM styles for weights and thresholds
            - backend: "hls" or "rtl" implementation
            - sparsity_type (optional): "none", "unstructured", "rows_random", "cols_random",
                                      "rows_regular", "cols_regular"
            - sparsity_amount (optional): Fraction of weights to zero (0.0-1.0)

        The method generates some auxiliary statistics about the created model,
        including sparsity metrics and folding parameters, which are saved
        as dut_info.json for analysis.
        """
        # Read params
        idt = self._params["idt"]
        wdt = self._params["wdt"]
        act = self._params["act"]

        numInputVectors = self._params["nhw"]
        mw = self._params["mw"]
        mh = self._params["mh"]
        sf = self._params["sf"]
        nf = self._params["nf"]
        m = self._params["m"]

        mem_mode = self._params["mem_mode"]
        ram_style = self._params["ram_style"]
        ram_style_thr = self._params["ram_style_thr"]

        backend = self._params["backend"]

        output_dict = {}

        # convert string to FINN DataType
        idt = DataType[idt]
        wdt = DataType[wdt]
        if act is not None:
            act = DataType[act]

        # Determine and log folding
        if sf > mw or nf > mh:
            print("Invalid sf/nf configuration, skipping")
            return "skipped"
        if sf == -1:
            sf = mw
        simd = mw // sf
        if nf == -1:
            nf = mh
        pe = mh // nf
        if mw % simd != 0 or mh % pe != 0:
            print("Invalid simd/pe configuration, skipping")
            return "skipped"
        if m > 1 and (simd != mw or pe != mh):
            print("M > 1 not possible for non-max simd/pe, skipping")
            return "skipped"
        output_dict["simd"] = simd
        output_dict["pe"] = pe

        # Restrictions for RTL MVAU
        if backend == "rtl":
            # only standalone thresholds supported
            if act is not None:
                return "skipped"
            # only decoupled mem mode supported
            if mem_mode != "internal_decoupled":
                return "skipped"
            # only signed weights supported
            if not wdt.signed():
                return "skipped"
            # bitwidth restrictions
            if idt.bitwidth() < 4 or idt.bitwidth() > 8:
                return "skipped"
            if wdt.bitwidth() < 4 or wdt.bitwidth() > 8:
                return "skipped"
            # TODO: narrow-range restrictions for DSP48E1
            # TODO: special case of 9-bit signed input

        # Generate weights
        np.random.seed(123456)  # TODO: verify or switch to modern numpy random generation

        W = gen_finn_dt_tensor(wdt, (mw, mh))

        if "sparsity_type" in self._params:
            sparsity_type = self._params["sparsity_type"]
        else:
            sparsity_type = "none"

        if sparsity_type == "none":
            if "sparsity_amount" in self._params:
                if self._params["sparsity_amount"] > 0:
                    print("sparsity amount > 0 not applicable for none sparsity, skipping")
                    return "skipped"
        else:
            if self._params["sparsity_amount"] == 0:
                print("sparsity amount = 0 not applicable for selected sparsity, skipping")
                return "skipped"
            if sparsity_type == "unstructured":
                idx = np.random.choice(
                    mw * mh, size=int(self._params["sparsity_amount"] * mw * mh), replace=False
                )
                W = np.reshape(W, -1)
                W[idx] = 0.0
                W = np.reshape(W, (mw, mh))
            elif sparsity_type == "rows_random":
                idx_mw = np.random.choice(
                    mw, size=int(self._params["sparsity_amount"] * mw), replace=False
                )
                W[idx_mw, :] = 0.0
            elif sparsity_type == "cols_random":
                idx_mh = np.random.choice(
                    mh, size=int(self._params["sparsity_amount"] * mh), replace=False
                )
                W[:, idx_mh] = 0.0
            elif sparsity_type == "rows_regular":
                if self._params["sparsity_amount"] == 0.25:
                    idx_mw = np.arange(0, mw, step=4)
                elif self._params["sparsity_amount"] == 0.5:
                    idx_mw = np.arange(0, mw, step=2)
                elif self._params["sparsity_amount"] == 0.75:
                    idx_mw = np.concatenate(
                        (
                            np.arange(0, mw, step=4),
                            np.arange(1, mw, step=4),
                            np.arange(2, mw, step=4),
                        )
                    )
                else:
                    print("regular sparsity only applicable for amount 0.25/0.5/0.75, skipping")
                    return "skipped"
                W[idx_mw, :] = 0.0
            elif sparsity_type == "cols_regular":
                if self._params["sparsity_amount"] == 0.25:
                    idx_mh = np.arange(0, mh, step=4)
                elif self._params["sparsity_amount"] == 0.5:
                    idx_mh = np.arange(0, mh, step=2)
                elif self._params["sparsity_amount"] == 0.75:
                    idx_mh = np.concatenate(
                        (
                            np.arange(0, mh, step=4),
                            np.arange(1, mh, step=4),
                            np.arange(2, mh, step=4),
                        )
                    )
                else:
                    print("regular sparsity only applicable for amount 0.25/0.5/0.75, skipping")
                    return "skipped"
                W[:, idx_mh] = 0.0

            else:
                print("ERROR: unknown sparsity type")
                raise Exception("ERROR: unknown sparsity type")

        # TODO: implement enforce option which prevents naturally occurring sparsity
        # params["sparsity_enforce"]
        # TODO: implement distribution option which selects between uniform/normal/??
        # params["sparsity_distribution"]

        # log resulting sparsity statistics
        # could be higher than selected due to naturally occurring sparsity
        num_zeros = (W == 0).sum()
        num_ones = (W == 1).sum() + (W == -1).sum()
        num_p2 = 0
        for w in np.nditer(W):
            if w != 0 and w != 1 and w != -1:
                if w > 0:
                    if math.log2(w).is_integer():
                        num_p2 = num_p2 + 1
                else:
                    if math.log2(-w).is_integer():
                        num_p2 = num_p2 + 1
        output_dict["zero_weights"] = round(num_zeros / W.size, 2)
        output_dict["easy_weights"] = round((num_zeros + num_ones + num_p2) / W.size, 2)

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
            # set range for threshold values according to worst-case accumulator range
            # (not weight value specific)
            # this could result in some thresholds being clipped by MinimizeAccumulatorWidth
            # lower_range = calculate_matvec_accumulator_range(wdt.min() * np.ones_like(W), idt)
            # upper_range = calculate_matvec_accumulator_range(wdt.max() * np.ones_like(W), idt)
            # acc_min = min(min(lower_range), min(upper_range))
            # acc_max = max(max(lower_range), max(upper_range))
            # set range for threshold values according to actual accumulator range
            # for the generated weights
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

        # Create model
        model = self._make_single_mvau_model(
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
            mem_mode,
            ram_style,
            ram_style_thr,
            backend,
        )
        model = model.transform(GiveUniqueNodeNames())
        # node = model.get_nodes_by_op_type("MVAU_hls")[0]
        # inst = getCustomOp(node)

        # log additional info about the generated model (e.g. SIMD/PE or sparsity)
        with open(self._build_inputs["build_dir"] + "/report/dut_info.json", "w") as f:
            json.dump(output_dict, f, indent=2)

        # TODO: also generate golden I/O pair for further verification steps
        model.save(onnx_export_path)

    def _step_build_setup(self):
        """
        Configure the dataflow build pipeline for MVAU microbenchmarks.

        This method sets up a comprehensive build configuration specifically optimized
        for MVAU microbenchmark evaluation. The configuration includes all necessary
        steps for complete characterization from ONNX model to deployment package.

        Returns:
            DataflowBuildConfig: Configured build pipeline for MVAU benchmarking
        """
        # create build config for synthetic microbenchmark models
        cfg = build_cfg.DataflowBuildConfig(
            # manual folding
            target_fps=None,
            steps=[
                "step_create_dataflow_partition",
                "step_minimize_bit_width",
                "step_generate_estimate_reports",
                "step_hw_codegen",
                "step_hw_ipgen",
                "step_create_stitched_ip",
                "step_measure_rtlsim_performance",
                "step_out_of_context_synthesis",
                "step_vivado_power_estimation",
                "step_synthesize_bitfile",
                "step_make_driver",
                "step_deployment_package",
            ],
        )
        return cfg
