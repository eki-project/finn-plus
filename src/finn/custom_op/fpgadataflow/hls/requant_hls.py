# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HLS backend implementation of the uniform-affine requantization operator."""

import numpy as np
from onnx import GraphProto, NodeProto
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.quant import max_int, min_int
from typing import cast

from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.requant import NodeAttrTypes, Requant
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.exception import FINNInternalError
from finn.util.logging import log


@register_custom_op
class Requant_hls(Requant, HLSBackend):
    """HLS backend for Requant operation.

    Computes: ``clip(round(x * scale + bias), min, max)``

    Scale and bias are embedded as constants in the generated HLS code.
    Per-channel scale and bias are supported. When scale=1 and bias=0,
    the generated code skips the unnecessary multiply/add operations.

    Note: This backend is primarily for FLOAT32 inputs. For integer inputs,
    prefer Requant_rtl which is more efficient.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)
        # optimization flags derived from the constant scale/bias in generate_params
        self._scale_is_one: bool = False
        self._bias_is_zero: bool = False

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(Requant.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def get_exp_cycles(self) -> int:
        """Return expected number of cycles for execution.

        Adds a constant offset to account for HLS pipeline initialization overhead.
        """
        hls_overhead = 10
        return Requant.get_exp_cycles(self) + hls_overhead

    def generate_params(self, model: ModelWrapper, path: str | Path) -> None:
        """Generate scale and bias parameter arrays as HLS header file."""
        code_gen_dir = Path(path)
        scale = self.get_scale(model)
        bias = self.get_bias(model)

        num_channels = self.num_channels
        pe = self.pe
        cf = num_channels // pe

        # Check scale/bias for optimization and store as instance attributes
        # (not in code_gen_dict which expects lists for template substitution)
        self._scale_is_one = bool(np.allclose(scale, 1.0))
        self._bias_is_zero = bool(np.allclose(bias, 0.0))

        # Broadcast scalar values to all channels if needed
        if scale.size == 1:
            scale = np.full(num_channels, scale.item(), dtype=np.float32)
        if bias.size == 1:
            bias = np.full(num_channels, bias.item(), dtype=np.float32)

        # Reshape for PE interleaving: [CF][PE]
        scale_reshaped = scale.reshape(cf, pe)
        bias_reshaped = bias.reshape(cf, pe)

        # Write to header file
        lines = [f"static const float scales[{cf}][{pe}] = {{"]
        for c in range(cf):
            lines.append("    {" + ", ".join(f"{v:.10f}f" for v in scale_reshaped[c]) + "},")
        lines += ["};", "", f"static const float biases[{cf}][{pe}] = {{"]
        for c in range(cf):
            lines.append("    {" + ", ".join(f"{v:.10f}f" for v in bias_reshaped[c]) + "},")
        lines += ["};", ""]
        (code_gen_dir / "params.h").write_text("\n".join(lines))

    def global_includes(self) -> None:
        """Return global includes."""
        self.code_gen_dict["$GLOBALS$"] = [
            "#include <hls_math.h>",
            "#include <hls_stream.h>",
            "#include <ap_int.h>",
            '#include "flatten.hpp"',
            '#include "params.h"',
        ]

    def defines(self, var: str) -> None:  # noqa: ARG002
        """Return defines."""
        pe = self.pe
        cf = self.num_channels // pe

        idt = self.get_input_datatype(0)
        odt = self.get_output_datatype()

        # Get output range from output datatype
        signed = odt.signed()
        narrow = bool(self.narrow)
        bitwidth = odt.bitwidth()
        min_val = min_int(signed, narrow, bitwidth)
        max_val = max_int(signed, narrow, bitwidth)

        total_fold = int(np.prod(self.num_input_vectors)) * cf

        # Use explicit width constants instead of TI::width which doesn't work for float
        in_width = idt.bitwidth()
        out_width = odt.bitwidth()

        self.code_gen_dict["$DEFINES$"] = [
            f"constexpr unsigned PE = {pe};",
            f"constexpr unsigned CF = {cf};",
            f"constexpr unsigned TOTAL_FOLD = {total_fold};",
            f"constexpr int MIN_VAL = {min_val};",
            f"constexpr int MAX_VAL = {max_val};",
            # Element types
            f"using TI = {idt.get_hls_datatype_str()};",
            f"using TO = {odt.get_hls_datatype_str()};",
            # Explicit width constants (float doesn't have ::width)
            f"static constexpr auto TI_WIDTH = {in_width};",
            f"static constexpr auto TO_WIDTH = {out_width};",
            # Packed types for stream interface
            f"using InPacked = ap_uint<{self.get_instream_width(0)}>;",
            f"using OutPacked = ap_uint<{self.get_outstream_width(0)}>;",
            """
template<typename T, typename TLo, typename THi>
static inline T clip(T const x, TLo const lo, THi const hi) {
#pragma HLS inline
    if(x < lo) return lo;
    if(x > hi) return hi;
    return x;
}
            """,
        ]

    def docompute(self) -> None:
        """Return docompute."""
        # Get optimization flags (set during generate_params)
        scale_is_one = self._scale_is_one
        bias_is_zero = self._bias_is_zero

        # Build the computation expression based on what's needed
        if scale_is_one and bias_is_zero:
            # Just round and clip
            compute_expr = "float(in_val)"
        elif scale_is_one:
            # No scale, just bias
            compute_expr = "float(in_val) + biases[cf_idx][p]"
        elif bias_is_zero:
            # No bias, just scale
            compute_expr = "float(in_val) * scales[cf_idx][p]"
        else:
            # Full computation
            compute_expr = "float(in_val) * scales[cf_idx][p] + biases[cf_idx][p]"

        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"""
            // Buffer for PE parallel outputs
            TO out_buf[PE];
            #pragma HLS ARRAY_PARTITION variable=out_buf complete dim=1

            for (unsigned fold = 0; fold < TOTAL_FOLD; fold++) {{
                #pragma HLS PIPELINE II=1 style=flp

                // Read packed input and unpack using Slice
                InPacked in_packed = in0_V.read();
                auto in_slice = Slice<TI>{{}}(in_packed);

                unsigned cf_idx = fold % CF;

                for (unsigned p = 0; p < PE; p++) {{
                    #pragma HLS UNROLL
                    #pragma HLS INLINE recursive
                    // Get input value from slice
                    TI in_val = in_slice(p, 0);

                    // Compute scaled value
                    float scaled = {compute_expr};

                    // Round to nearest integer and clip
                    int rounded = clip((int)hls::lrint(scaled), MIN_VAL, MAX_VAL);

                    // Store output
                    out_buf[p] = TO(rounded);
                }}

                // Pack and write output
                out0_V.write(flatten(out_buf));
            }}
            """
        ]

    def blackboxfunction(self) -> None:
        """Return blackboxfunction."""
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"""
            void {self.onnx_node.name}(
                hls::stream<InPacked> &in0_V,
                hls::stream<OutPacked> &out0_V
            )
            """
        ]

    def read_npy_data(self) -> None:
        """Generate code for reading input data from .npy file."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        elem_hls_type = self.get_input_datatype(0).get_hls_datatype_str()
        npy_type = "half" if elem_hls_type == "half" else "float"

        self.code_gen_dict["$READNPYDATA$"] = [
            f"npy2apintstream<InPacked, TI, TI_WIDTH, {npy_type}>(",
            f'"{code_gen_dir}/input_0.npy", in0_V, false',
            ");",
        ]

    def strm_decl(self) -> None:
        """Generate stream declarations for C++ simulation."""
        self.code_gen_dict["$STREAMDECLARATIONS$"] = [
            'hls::stream<InPacked> in0_V ("in0_V");',
            'hls::stream<OutPacked> out0_V ("out0_V");',
        ]

    def dataoutstrm(self) -> None:
        """Generate code for writing output data to .npy file."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        shape = f"{{{','.join(str(i) for i in self.get_folded_output_shape(0))}}}"
        elem_hls_type = self.get_output_datatype().get_hls_datatype_str()
        npy_type = "half" if elem_hls_type == "half" else "float"

        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            f"apintstream2npy<OutPacked, TO, TO_WIDTH, {npy_type}>(",
            f'out0_V, {shape}, "{code_gen_dir}/output_0.npy", false',
            ");",
        ]

    def save_as_npy(self) -> None:
        """Empty - output saving is handled in dataoutstrm."""
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def pragmas(self) -> None:
        """Return pragmas."""
        self.code_gen_dict["$PRAGMAS$"] = [
            """
            #pragma HLS interface AXIS port=in0_V
            #pragma HLS interface AXIS port=out0_V
            #pragma HLS interface ap_ctrl_none port=return
            """
        ]

    def execute_node(
        self, context: dict[str, np.ndarray], graph: GraphProto
    ) -> None:  # noqa: ARG002
        """Execute the node using cppsim or rtlsim.

        Custom implementation that only passes input 0 (data), since scale and
        bias are embedded as parameters in the generated code.
        """
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node

        if mode == "cppsim":
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_cppsim"))
        elif mode == "rtlsim":
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
        else:
            raise FINNInternalError(
                f'{node.name}: invalid exec_mode {mode}, must be one of ("cppsim", "rtlsim")'
            )

        # Only process input 0 (data tensor), scale and bias are embedded
        inp = node.input[0]
        exp_ishape = tuple(self.get_normal_input_shape(0))
        folded_ishape = self.get_folded_input_shape(0)
        inp_val = context[inp]

        # Make sure the input has the right container datatype
        if inp_val.dtype not in [np.float32, np.float16]:
            log.warning(
                f"{node.name}: changing input container datatype from {inp_val.dtype} to float32"
            )
            inp_val = inp_val.astype(np.float32)

        if inp_val.shape != exp_ishape:
            raise FINNInternalError(f"{node.name}: input shape {inp_val.shape} != {exp_ishape}")
        export_idt = self.get_input_datatype(0)

        if export_idt == DataType["BIPOLAR"]:
            inp_val = (inp_val + 1) / 2
            export_idt = DataType["BINARY"]

        reshaped_input = inp_val.reshape(folded_ishape).copy()
        np.save(str(Path(code_gen_dir) / "input_0.npy"), reshaped_input)

        if mode == "cppsim":
            # execute the precompiled model
            self.exec_precompiled_singlenode_model()
            # load output npy file
            self.npy_to_dynamic_output(context)
            for o, outp in enumerate(node.output):
                exp_oshape = tuple(self.get_normal_output_shape(o))
                if context[outp].shape != exp_oshape:
                    raise FINNInternalError(
                        f"{node.name}: cppsim did not produce expected output shape"
                    )
                if self.get_output_datatype(o) == DataType["BIPOLAR"]:
                    context[outp] = 2 * context[outp] - 1
        elif mode == "rtlsim":
            nbits = self.get_instream_width(0)
            rtlsim_inp = npy_to_rtlsim_input(f"{code_gen_dir}/input_0.npy", export_idt, nbits)
            io_dict = {
                "inputs": {"in0": rtlsim_inp},
                "outputs": {"out0": []},
            }

            sim = self.get_rtlsim()
            self.reset_rtlsim(sim)
            self.rtlsim_multi_io(sim, io_dict)
            self.close_rtlsim(sim)

            rtlsim_output = io_dict["outputs"]["out0"]
            odt = self.get_output_datatype(0)
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width(0)
            out_npy_path = f"{code_gen_dir}/output_0.npy"
            out_shape = self.get_folded_output_shape(0)
            rtlsim_output_to_npy(
                rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
            )

            exp_oshape = tuple(self.get_normal_output_shape(0))
            output = np.load(out_npy_path)
            output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
            context[node.output[0]] = output

            if context[node.output[0]].shape != exp_oshape:
                raise FINNInternalError(
                    f"{node.name}: output shape {context[node.output[0]].shape} != {exp_oshape}"
                )
