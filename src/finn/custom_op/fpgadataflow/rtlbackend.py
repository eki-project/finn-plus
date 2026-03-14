# Copyright (C) 2023, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""RTL backend support for FINN custom operations.

This module provides the RTLBackend abstract base class that all RTL-based custom
operations in FINN inherit from. It includes functionality for HDL code generation,
RTL simulation, and integration with Vivado IP Integrator.
"""

import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from onnx import GraphProto
    from qonnx.core.modelwrapper import ModelWrapper

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn import xsi
from finn.util.basic import make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.exception import FINNInternalError
from finn.util.logging import log

finnxsi = xsi if xsi.is_available() else None


class RTLBackend(HWCustomOp, ABC):
    """RTLBackend class all custom ops that correspond to a module in finn-rtllib
    are using functionality of. Contains different functions every RTL
    custom node should have. Some as abstract methods, these have to be filled
    when writing a new RTL custom op node."""

    def get_nodeattr_types(
        self,
    ) -> dict[
        str,
        tuple[str, bool, int | float | str | bool | npt.NDArray | list]
        | tuple[str, bool, int | float | str | bool | npt.NDArray | list, set | None],
    ]:
        """Return 4-tuple (dtype, required, default_val, allowed_values) for attribute
        with name. allowed_values will be None if not specified.

        Returns:
            dict[ str, tuple[str, bool, int | float | str | bool | npt.NDArray | list] | tuple[
                str, bool, int | float | str | bool | npt.NDArray | list, set | None]]:
                Dictionary of node attribute types
        """
        super_attrs = super().get_nodeattr_types()
        super_attrs.update(
            {
                # attribute to save top module name - not user configurable
                "gen_top_module": ("s", False, ""),
            }
        )
        return super_attrs

    @abstractmethod
    def generate_hdl(self, model: "ModelWrapper", fpgapart: str, clk: str) -> None:
        """Generate HDL code for this node.

        Args:
            model: The FINN model containing this node
            fpgapart: Target FPGA part string
            clk: Clock period specification

        Returns:
            None
        """

    def prepare_rtlsim(self) -> None:
        """Create a xsi emulation library for the RTL code generated for this node.
        Sets the rtlsim_so attribute to the path of the generated library.

        Returns:
            None
        """
        import finn_xsi.adapter as finnxsi

        verilog_files = self.get_rtl_file_list(abspath=True)
        single_src_dir = make_build_dir("rtlsim_" + self.onnx_node.name + "_")
        trace_file = self.get_nodeattr("rtlsim_trace")
        debug = not (trace_file is None or trace_file == "")
        ret = finnxsi.compile_sim_obj(
            self.get_verilog_top_module_name(), verilog_files, single_src_dir, debug
        )
        # save generated lib filename in attribute
        self.set_nodeattr("rtlsim_so", ret[0] + "/" + ret[1])

    def get_verilog_paths(self) -> list[str]:
        """Return path to code gen directory.
        Can be overwritten to return additional paths to relevant verilog files.

        Returns:
            list[str]: List of paths to directories containing Verilog files
        """
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        return [cast("str", code_gen_dir)]

    @abstractmethod
    def get_rtl_file_list(self, abspath: bool = False) -> list[str] | list[Path]:
        """Return list of RTL files.
        Must be implemented by each subclass to provide the list of RTL files used by this node.

        Args:
            abspath: If True, return absolute paths; if False, return relative paths

        Returns:
            list[str] | list[Path]: List of paths to RTL files
        """

    @abstractmethod
    def code_generation_ipi(self) -> list[str]:
        """Generate TCL commands for IP Integrator.
        Must be implemented by each subclass to provide the TCL commands needed
        to integrate this node into Vivado IP Integrator.

        Returns:
            list[str]: List of TCL commands for IP Integrator
        """

    def code_generation_ipgen(self, model: "ModelWrapper", fpgapart: str, clk: str) -> None:
        """Generate HDL code for IP generation.
        Wrapper method that calls generate_hdl to produce the HDL code for this node.

        Args:
            model: The FINN model containing this node
            fpgapart: Target FPGA part string
            clk: Clock period specification

        Returns:
            None
        """
        self.generate_hdl(model, fpgapart, clk)

    def execute_node(
        self, context: dict[str, npt.NDArray], graph: "GraphProto"
    ) -> None:  # noqa: ARG002
        """Execute this node's RTL simulation.

        Args:
            context: Dictionary mapping tensor names to their numpy array values
            graph: The ONNX graph containing this node

        Returns:
            None

        Raises:
            Exception: If exec_mode is not set to "rtlsim"
        """
        mode = self.get_nodeattr("exec_mode")
        code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))

        if mode == "rtlsim":
            node = self.onnx_node
            inputs = {}
            for i, inp in enumerate(node.input):
                shape = self.get_normal_input_shape(i)
                if shape is None:
                    raise FINNInternalError(
                        f"Input shape for input {i} of node {node.name} is None."
                    )
                exp_ishape = tuple(shape)
                folded_ishape = self.get_folded_input_shape(i)
                if folded_ishape is None:
                    raise FINNInternalError(
                        f"Folded input shape for input {i} of node {node.name} is None."
                    )
                inp_val = context[inp]
                # Make sure the input has the right container datatype
                if inp_val.dtype != np.float32:
                    # Issue a warning to make the user aware of this type-cast
                    log.warning(
                        f"{node.name}: Changing input container datatype from "
                        f"{inp_val.dtype} to {np.float32}"
                    )
                    # Convert the input to floating point representation as the
                    # container datatype
                    inp_val = inp_val.astype(np.float32)

                assert inp_val.shape == exp_ishape, "Input shape doesn't match expected shape."
                export_idt = self.get_input_datatype(i)

                reshaped_input = inp_val.reshape(folded_ishape)
                input_path = Path(code_gen_dir) / f"input_{i}.npy"
                np.save(input_path, reshaped_input)
                nbits = self.get_instream_width(i)
                rtlsim_inp = npy_to_rtlsim_input(str(input_path), export_idt, nbits)
                inputs[f"in{i}"] = rtlsim_inp
            outputs = {}
            for o, _ in enumerate(node.output):
                outputs[f"out{o}"] = []
            # assembled execution context
            io_dict = {"inputs": inputs, "outputs": outputs}

            sim = self.get_rtlsim()
            self.reset_rtlsim(sim)
            self.rtlsim_multi_io(sim, io_dict)
            self.close_rtlsim(sim)
            for o, outp in enumerate(node.output):
                rtlsim_output = io_dict["outputs"][f"out{o}"]
                odt = self.get_output_datatype(o)
                target_bits = odt.bitwidth()
                packed_bits = self.get_outstream_width(o)
                out_npy_path = f"{code_gen_dir}/output.npy"
                out_shape = self.get_folded_output_shape(o)
                rtlsim_output_to_npy(
                    rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
                )
                # load and reshape output
                oshape = self.get_normal_output_shape(o)
                if oshape is None:
                    raise FINNInternalError(
                        f"Output shape for output {o} of node {node.name} is None."
                    )
                exp_oshape = tuple(oshape)
                output = np.load(out_npy_path)
                output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
                context[outp] = output

                assert (
                    context[outp].shape == exp_oshape
                ), "Output shape doesn't match expected shape."

        else:
            raise Exception(
                f"""Invalid value for attribute exec_mode! Is currently set to: {mode}
            has to be set to one of the following value ("cppsim", "rtlsim")"""
            )
