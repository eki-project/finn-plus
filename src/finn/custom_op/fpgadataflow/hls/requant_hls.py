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
from finn.custom_op.fpgadataflow.hwcustomop import NodeAttrTypes
from finn.custom_op.fpgadataflow.requant import Requant
from finn.util.data_packing import hls_datatype_str, npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.exception import FINNInternalError
from finn.util.logging import log


@register_custom_op
class Requant_hls(Requant, HLSBackend):
    """HLS backend for Requant operation.

    Computes: ``clip(round(x * scale + bias), min, max)``

    Delegates to Requant_Batch from finn-hlslib (function-pipelined).
    Scale and bias are embedded as RequantParams in a generated header file.
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

    def generate_params(
        self,
        model: ModelWrapper,
        path: str | Path,
        fpgapart: str | None = None,  # noqa: ARG002
    ) -> None:
        """Generate the RequantParams header (scales and biases in [PE][NF] layout).

        Args:
            model: The model holding the scale/bias initializers of this node.
            path: Directory in which ``params.h`` is written.
            fpgapart: Target FPGA part (unused, the HLS parameters are device-independent).
        """
        code_gen_dir = Path(path)
        scale = self.get_scale(model)
        bias = self.get_bias(model)

        num_channels = self.num_channels
        pe = self.pe
        nf = num_channels // pe

        # Broadcast scalar values to all channels if needed
        if scale.size == 1:
            scale = np.full(num_channels, scale.item(), dtype=np.float32)
        if bias.size == 1:
            bias = np.full(num_channels, bias.item(), dtype=np.float32)

        # Reshape for PE interleaving: channel order is
        #   ch0=(nf=0,pe=0), ch1=(nf=0,pe=1), ..., ch_{PE-1}=(nf=0,pe=PE-1),
        #   ch_{PE}=(nf=1,pe=0), ...
        # Python arrays are [NF][PE], RequantParams needs [PE][NF] → transpose
        scale_nf_pe = scale.reshape(nf, pe)
        bias_nf_pe = bias.reshape(nf, pe)

        lines = [
            '#include "requant.hpp"',
            "",
            f"static RequantParams<{nf}, {pe}> const rq_params = {{",
        ]
        # scales[PE][NF]
        lines.append("    .scales = {")
        for p in range(pe):
            vals = ", ".join(f"{scale_nf_pe[n, p]:.10f}f" for n in range(nf))
            lines.append(f"        {{{vals}}},")
        lines.append("    },")
        # biases[PE][NF]
        lines.append("    .biases = {")
        for p in range(pe):
            vals = ", ".join(f"{bias_nf_pe[n, p]:.10f}f" for n in range(nf))
            lines.append(f"        {{{vals}}},")
        lines += ["    }", "};", ""]
        (code_gen_dir / "params.h").write_text("\n".join(lines))

    def global_includes(self) -> None:
        """Return global includes."""
        self.code_gen_dict["$GLOBALS$"] = [
            '#include "requant.hpp"',
            '#include "params.h"',
        ]

    def defines(self, var: str) -> None:  # noqa: ARG002
        """Return defines."""
        pe = self.pe
        num_channels = self.num_channels

        idt = self.get_input_datatype(0)
        odt = self.get_output_datatype()

        narrow = bool(self.narrow)
        signed = odt.signed()
        bitwidth = odt.bitwidth()
        min_val = min_int(signed, narrow, bitwidth)
        max_val = max_int(signed, narrow, bitwidth)

        self.code_gen_dict["$DEFINES$"] = [
            f"constexpr unsigned NumChannels1 = {num_channels};",
            f"constexpr unsigned PE1 = {pe};",
            f"constexpr int MIN_VAL = {min_val};",
            f"constexpr int MAX_VAL = {max_val};",
            f"using TI = {hls_datatype_str(idt)};",
            f"using TO = {hls_datatype_str(odt)};",
        ]

    def docompute(self) -> None:
        """Return docompute."""
        self.code_gen_dict["$DOCOMPUTE$"] = [
            "Requant_Batch<NumChannels1, PE1, MIN_VAL, MAX_VAL>(in0_V, out0_V, rq_params);"
        ]

    def blackboxfunction(self) -> None:
        """Return blackboxfunction."""
        pe = self.pe
        idt = hls_datatype_str(self.get_input_datatype(0))
        odt = hls_datatype_str(self.get_output_datatype())
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            (
                f"void {self.onnx_node.name}(hls::stream<tensor<{idt}, {pe}>> &in0_V, "
                f"hls::stream<tensor<{odt}, {pe}>> &out0_V)"
            )
        ]

    def pragmas(self) -> None:
        """Return pragmas."""
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis port=in0_V",
            "#pragma HLS INTERFACE axis port=out0_V",
            "#pragma HLS INTERFACE ap_ctrl_none port=return",
            "#pragma HLS DATAFLOW disable_start_propagation",
            "#pragma HLS ARRAY_PARTITION variable=rq_params.scales complete dim=1",
            "#pragma HLS ARRAY_PARTITION variable=rq_params.biases complete dim=1",
        ]

    def ipgen_extra_directives(self) -> list[str]:
        """Return a list of extra tcl directives for HLS synthesis."""
        return ["config_compile -pipeline_style flp"]

    def execute_node(
        self, context: dict[str, np.ndarray], graph: GraphProto  # noqa: ARG002
    ) -> None:
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
