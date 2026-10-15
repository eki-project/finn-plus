# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""RTL backend implementation of the Pad1D layer."""

import numpy as np
from onnx import NodeProto
from pathlib import Path
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING, Any, cast

from finn.custom_op.fpgadataflow.pad1d import NodeAttrTypes, Pad1D
from finn.custom_op.fpgadataflow.rtl import register_custom_op
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from onnx import GraphProto


def _rtlsrc_dir() -> Path:
    """Return the directory of the finn-rtllib pad1d HDL sources."""
    return Path(get_settings().finn_rtllib) / "pad1d" / "hdl"


@register_custom_op
class Pad1D_rtl(Pad1D, RTLBackend):
    """RTL implementation of Pad1D."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return node attribute types, combining Pad1D and RTLBackend attributes."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(Pad1D.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def _pack_value(self, value: float, dtype: BaseDataType) -> int:
        """Return the unsigned bit pattern of a single value in the given FINN datatype."""
        bitwidth = dtype.bitwidth()
        if dtype == DataType["BIPOLAR"]:
            int_value = int((value + 1) // 2)
        else:
            if dtype.is_fixed_point():
                # scale_factor() is only defined on fixed-point DataTypes, guarded above
                value = value / cast("Any", dtype).scale_factor()
            int_value = int(value)
            if int_value < 0:
                int_value += 1 << bitwidth
        return int_value & ((1 << bitwidth) - 1)

    def _get_pad_data_width(self, ind: int) -> int:
        """Return the total bit width of the pad data for pad input `ind`."""
        dtype = self.get_input_datatype()
        pad_count = self._get_pad_count(ind)
        return max(1, pad_count) * self.num_channels * dtype.bitwidth()

    def _get_pad_values(self, model: ModelWrapper, ind: int) -> np.ndarray:
        """Return the constant pad values for input `ind` expanded to the full pad length."""
        pad_count = self._get_pad_count(ind)
        if pad_count == 0:
            return np.zeros((1, 1, self.num_channels), dtype=np.float32)

        if len(self.onnx_node.input) <= ind:
            raise FINNUserError(
                f"Pad1D RTL generation requires a constant {self._get_pad_side_name(ind)} "
                "pad input."
            )

        pad_init = model.get_initializer(self.onnx_node.input[ind])
        if pad_init is None:
            raise FINNUserError(
                f"Pad1D RTL generation requires a constant {self._get_pad_side_name(ind)} "
                "pad input."
            )

        pad_values = np.asarray(pad_init, dtype=np.float32)
        self._validate_pad_shape(pad_values.shape, ind)
        if pad_values.shape[1] == 1 and pad_count > 1:
            pad_values = np.repeat(pad_values, pad_count, axis=1)
        return pad_values

    def _pack_pad_data(self, model: ModelWrapper, ind: int) -> str:
        """Return the pad values of input `ind` packed as a Verilog sized hex literal."""
        dtype = self.get_input_datatype()
        bitwidth = dtype.bitwidth()
        if self._get_pad_count(ind) == 0:
            data_width = self._get_pad_data_width(ind)
            hex_digits = (data_width + 3) // 4
            return f"{data_width}'h{0:0{hex_digits}x}"

        pad_values = self._get_pad_values(model, ind)

        if not np.vectorize(dtype.allowed)(pad_values).all():
            raise FINNUserError(
                f"Pad1D {self._get_pad_side_name(ind)} pad values cannot be represented "
                f"with {dtype.name}"
            )

        packed = 0
        for i, value in enumerate(pad_values.flatten()):
            packed |= self._pack_value(value, dtype) << (i * bitwidth)
        data_width = self._get_pad_data_width(ind)
        hex_digits = (data_width + 3) // 4
        return f"{data_width}'h{packed:0{hex_digits}x}"

    def generate_hdl(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:  # noqa: ARG002
        """Render the pad1d wrapper template with embedded pad data into the codegen dir."""
        simd = self.simd
        num_channels = self.num_channels
        if num_channels % simd != 0:
            raise FINNUserError("SIMD must divide NumChannels")

        template = (_rtlsrc_dir() / "pad1d_template.v").read_text()

        topname = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", topname)

        elem_width = self.get_input_datatype().bitwidth()
        fold_width = elem_width * simd
        code_gen_dict = {
            "TOP_MODULE_NAME": topname,
            "NUM_TOKENS": self.num_tokens,
            "NUM_CHANNELS": num_channels,
            "SIMD": simd,
            "ELEM_WIDTH": elem_width,
            "PAD_LEFT": self.pad_left,
            "PAD_RIGHT": self.pad_right,
            "FOLD_WIDTH": fold_width,
            "PAD_LEFT_DATA_WIDTH": self._get_pad_data_width(1),
            "PAD_RIGHT_DATA_WIDTH": self._get_pad_data_width(2),
            "PAD_LEFT_DATA": self._pack_pad_data(model, 1),
            "PAD_RIGHT_DATA": self._pack_pad_data(model, 2),
        }

        for key, value in code_gen_dict.items():
            template = template.replace(f"${key}$", str(value))

        code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
        (Path(code_gen_dir) / f"{topname}.v").write_text(template)
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:
        """Return the list of RTL files (pad1d core and generated wrapper) for this node."""
        gen_top_module = cast("str", self.get_nodeattr("gen_top_module"))
        if abspath:
            code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))
            return [
                str(_rtlsrc_dir() / "pad1d.sv"),
                str(code_gen_dir / f"{gen_top_module}.v"),
            ]
        return ["pad1d.sv", gen_top_module + ".v"]

    def code_generation_ipi(self) -> list[str]:
        """Construct and return the TCL for node instantiation in Vivado IPI."""
        cmd = [f"add_files -norecurse {f}" for f in self.get_rtl_file_list(abspath=True)]
        cmd.append(
            f"create_bd_cell -type module -reference {self.get_nodeattr('gen_top_module')} "
            f"{self.onnx_node.name}"
        )
        return cmd

    def execute_node(self, context: dict[str, np.ndarray], graph: "GraphProto") -> None:
        """Execute the node via Python (cppsim) or rtlsim on the streamed token input."""
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            Pad1D.execute_node(self, context, graph)
        elif mode == "rtlsim":
            node = self.onnx_node
            code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))

            # Only input 0 is streamed. Pad inputs are embedded as RTL parameters.
            inp = node.input[0]
            exp_ishape = tuple(self.get_normal_input_shape(0))
            folded_ishape = self.get_folded_input_shape(0)
            inp_val = context[inp]
            if str(inp_val.dtype) != "float32":
                raise FINNInternalError("Input datatype is not float32")
            if inp_val.shape != exp_ishape:
                raise FINNInternalError("Input shape doesn't match expected shape.")
            export_idt = self.get_input_datatype(0)

            reshaped_input = inp_val.reshape(folded_ishape)
            input_path = str(code_gen_dir / "input_0.npy")
            np.save(input_path, reshaped_input)
            rtlsim_inp = npy_to_rtlsim_input(
                input_path,
                export_idt,
                self.get_instream_width(0),
            )
            io_dict = {
                "inputs": {"in0": rtlsim_inp},
                "outputs": {"out0": []},
            }

            sim = self.get_rtlsim()
            self.reset_rtlsim(sim)
            self.rtlsim_multi_io(sim, io_dict)
            self.close_rtlsim(sim)

            odt = self.get_output_datatype(0)
            output_path = str(code_gen_dir / "output.npy")
            rtlsim_output_to_npy(
                io_dict["outputs"]["out0"],
                output_path,
                odt,
                self.get_folded_output_shape(0),
                self.get_outstream_width(0),
                odt.bitwidth(),
            )
            output = np.load(output_path)
            exp_oshape = tuple(self.get_normal_output_shape(0))
            context[node.output[0]] = np.asarray(output, dtype=np.float32).reshape(exp_oshape)
        else:
            raise FINNInternalError(
                f"Invalid value for attribute exec_mode! Is currently set to: {mode} "
                'has to be set to one of the following values ("cppsim", "rtlsim")'
            )
