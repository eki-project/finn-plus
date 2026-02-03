# Copyright (C) 2024, Advanced Micro Devices, Inc.
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

import numpy as np
import os
import shutil

from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.custom_op.fpgadataflow.streamingfifo import StreamingFIFO
from finn.util.logging import log


class StreamingFIFO_rtl(StreamingFIFO, RTLBackend):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # Toggle between rtl or IPI implementation
            # rtl - use the rtl generated IP during stitching
            # vivado - use the AXI Infrastructure FIFO
            # virtual - use virtual rtl implementation for live fifo-sizing
            "impl_style": ("s", False, "rtl", {"rtl", "vivado", "virtual"}),
            # Unique FIFO ID for ring bus addressing (only for impl_style=virtual)
            "fifo_id": ("i", False, 0),
        }
        my_attrs.update(StreamingFIFO.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))

        return my_attrs

    def get_adjusted_depth(self):
        impl = self.get_nodeattr("impl_style")
        depth = self.get_nodeattr("depth")
        if impl == "vivado":
            old_depth = depth
            # round up depth to nearest power-of-2
            # Vivado FIFO impl may fail otherwise
            depth = (1 << (depth - 1).bit_length()) if impl == "vivado" else depth
            if old_depth != depth:
                log.warning(
                    f"{self.onnx_node.name}: rounding-up FIFO depth "
                    f"from {old_depth} to {depth} for impl_style=vivado"
                )

        return depth

    def get_verilog_top_module_intf_names(self):
        ret = super().get_verilog_top_module_intf_names()
        is_virtual = self.get_nodeattr("impl_style") == "virtual"
        is_rtl = self.get_nodeattr("impl_style") == "rtl"
        is_depth_monitor = self.get_nodeattr("depth_monitor") == 1
        if is_rtl and is_depth_monitor:
            ret["ap_none"] = ["maxcount"]
        if is_virtual:
            ret["ap_none"] = ["icfg", "ocfg"]
        return ret

    def generate_hdl(self, model, fpgapart, clk):
        if self.get_nodeattr("impl_style") == "virtual":
            # No HDL generation needed for virtual FIFOs
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            self.set_nodeattr("ipgen_path", code_gen_dir)
            self.set_nodeattr("ip_path", code_gen_dir)
            return

        rtlsrc = os.path.join(os.environ["FINN_RTLLIB"], "fifo/hdl")
        template_path = rtlsrc + "/fifo_template.v"

        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        topname = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", topname)

        code_gen_dict = {}
        code_gen_dict["$TOP_MODULE_NAME$"] = topname
        # make instream width a multiple of 8 for axi interface
        in_width = self.get_instream_width_padded()

        count_width = int(self.get_nodeattr("depth")).bit_length()
        depth = int(self.get_nodeattr("depth"))
        code_gen_dict["$COUNT_WIDTH$"] = f"{count_width}"
        code_gen_dict["$COUNT_RANGE$"] = "[{}:0]".format(count_width - 1)
        code_gen_dict["$IN_RANGE$"] = "[{}:0]".format(in_width - 1)
        code_gen_dict["$OUT_RANGE$"] = "[{}:0]".format(in_width - 1)
        code_gen_dict["$WIDTH$"] = str(in_width)
        code_gen_dict["$DEPTH$"] = str(depth)
        # apply code generation to templates
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(template_path, "r") as f:
            template = f.read()
        for key_name in code_gen_dict:
            key = "%s" % key_name
            template = template.replace(key, str(code_gen_dict[key_name]))
        with open(
            os.path.join(code_gen_dir, self.get_verilog_top_module_name() + ".v"),
            "w",
        ) as f:
            f.write(template)

        shutil.copy(rtlsrc + "/fifo_gauge.sv", code_gen_dir)
        shutil.copy(rtlsrc + "/Q_srl.v", code_gen_dir)
        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def code_generation_ipi(self):
        impl_style = self.get_nodeattr("impl_style")
        if impl_style == "rtl":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

            sourcefiles = [
                "fifo_gauge.sv",
                "Q_srl.v",
                self.get_nodeattr("gen_top_module") + ".v",
            ]

            sourcefiles = [os.path.join(code_gen_dir, f) for f in sourcefiles]

            cmd = []
            for f in sourcefiles:
                cmd += ["add_files -norecurse %s" % (f)]
            cmd += [
                "create_bd_cell -type module -reference %s %s"
                % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
            ]
            return cmd
        elif impl_style == "vivado":
            cmd = []
            node_name = self.onnx_node.name
            depth = self.get_adjusted_depth()
            ram_style = self.get_nodeattr("ram_style")
            # create a hierarchy for this layer, with the same port names
            clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
            rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
            dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0][0]
            din_name = self.get_verilog_top_module_intf_names()["s_axis"][0][0]
            cmd.append("create_bd_cell -type hier %s" % node_name)
            cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (node_name, clk_name))
            cmd.append("create_bd_pin -dir I -type rst /%s/%s" % (node_name, rst_name))
            cmd.append(
                "create_bd_intf_pin -mode Master "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, dout_name)
            )
            cmd.append(
                "create_bd_intf_pin -mode Slave "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, din_name)
            )
            # instantiate and configure DWC
            cmd.append(
                "create_bd_cell -type ip "
                "-vlnv xilinx.com:ip:axis_data_fifo:2.0 /%s/fifo" % node_name
            )
            cmd.append(
                "set_property -dict [list CONFIG.FIFO_DEPTH {%d}] "
                "[get_bd_cells /%s/fifo]" % (depth, node_name)
            )
            cmd.append(
                "set_property -dict [list CONFIG.FIFO_MEMORY_TYPE {%s}] "
                "[get_bd_cells /%s/fifo]" % (ram_style, node_name)
            )
            cmd.append(
                "set_property -dict [list CONFIG.TDATA_NUM_BYTES {%d}] "
                "[get_bd_cells /%s/fifo]" % (np.ceil(self.get_outstream_width() / 8), node_name)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/fifo/M_AXIS] "
                "[get_bd_intf_pins %s/%s]" % (node_name, node_name, dout_name)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/fifo/S_AXIS] "
                "[get_bd_intf_pins %s/%s]" % (node_name, node_name, din_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] "
                "[get_bd_pins %s/fifo/s_axis_aresetn]" % (node_name, rst_name, node_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] "
                "[get_bd_pins %s/fifo/s_axis_aclk]" % (node_name, clk_name, node_name)
            )
            return cmd
        elif impl_style == "virtual":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            sourcefiles = self.get_rtl_file_list(abspath=True)
            fifo_name = self.onnx_node.name
            id = self.get_nodeattr("fifo_id")
            width = int(self.get_instream_width_padded())
            fm_size = int(np.prod(self.get_folded_input_shape()[0:-1]))

            cmd = []
            for f in sourcefiles:
                cmd += [f"add_files -norecurse {f}"]
            cmd += [f"create_bd_cell -type module -reference fifo_gauge_wrapper {fifo_name}"]
            cmd += [f"set_property CONFIG.ID {id} [get_bd_cells {fifo_name}]"]
            cmd += [f"set_property CONFIG.DATA_WIDTH {width} [get_bd_cells {fifo_name}]"]
            cmd += [f"set_property CONFIG.FM_SIZE {fm_size} [get_bd_cells {fifo_name}]"]
            return cmd
        else:
            raise Exception(
                "FIFO implementation style %s not supported, please use rtl or vivado" % impl_style
            )

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/"
            if self.get_nodeattr("impl_style") == "virtual":
                rtllib_dir = os.path.join(os.environ["FINN_RTLLIB"], "fifo_virtual/hdl/")
            else:
                rtllib_dir = os.path.join(os.environ["FINN_RTLLIB"], "fifo/hdl/")
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        if self.get_nodeattr("impl_style") == "virtual":
            verilog_files = [
                rtllib_dir + "fifo_gauge_pkg.sv",
                rtllib_dir + "fifo_gauge.sv",
                rtllib_dir + "fifo_gauge_wrapper.v",
            ]
        else:
            verilog_files = [
                rtllib_dir + "Q_srl.v",
                code_gen_dir + self.get_nodeattr("gen_top_module") + ".v",
            ]
        return verilog_files

    def prepare_rtlsim(self, behav=False):
        # TODO: Support simulation of vivado-style FIFOs,
        # or ensure node-by-node rtlsim is always skipped for FIFOs in general
        if self.get_nodeattr("impl_style") != "rtl":
            log.warning(
                f"Trying to prepare rtlsim for {self.onnx_node.name}, but impl_style "
                "is set to vivado or virtual, which is not supported for simulation. Skipping. "
                "Simulation will fall back to Python simulation."
            )
            raise NotImplementedError()
        return super().prepare_rtlsim(behav)

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        impl_style = self.get_nodeattr("impl_style")
        if mode == "cppsim" or impl_style == "vivado" or impl_style == "virtual":
            # Fall back to Python simulation (no-op) for vivado or virtual style FIFOs
            StreamingFIFO.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)
