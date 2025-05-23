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

import numpy as np
import os
from abc import abstractmethod
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import roundup_to_integer_multiple

from finn.util.basic import get_liveness_threshold_cycles, is_versal
from finn.util.logging import log

try:
    import pyxsi_utils
except ModuleNotFoundError:
    pyxsi_utils = None


class HWCustomOp(CustomOp):
    """HWCustomOp class all custom ops that can be implemented with either
    HLS or RTL backend are based on. Contains different functions every fpgadataflow
    custom node should have. Some as abstract methods, these have to be filled
    when writing a new fpgadataflow custom op node."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        self.code_gen_dict = {}

    def get_nodeattr_types(self):
        return {
            "backend": ("s", True, "fpgadataflow"),
            "preferred_impl_style": ("s", False, "", {"", "hls", "rtl"}),
            "code_gen_dir_ipgen": ("s", False, ""),
            "ipgen_path": ("s", False, ""),
            "ip_path": ("s", False, ""),
            "ip_vlnv": ("s", False, ""),
            "exec_mode": ("s", False, "", {"", "rtlsim", "cppsim"}),
            "cycles_rtlsim": ("i", False, 0),
            "cycles_estimate": ("i", False, 0),
            "rtlsim_trace": ("s", False, ""),
            "res_estimate": ("s", False, ""),
            "res_synth": ("s", False, ""),
            "rtlsim_so": ("s", False, ""),
            # partitioning info
            # ID of SLR to which the Op is attached in Vitis builds
            # Set to -1 as 'don't care'
            "slr": ("i", False, -1),
            # Vitis memory port to which any AXI-MM interface
            # of this Op should be attached in Vitis builds
            # E.g.: "DDR[0]", "HBM[0]", "PLRAM[0]"
            "mem_port": ("s", False, ""),
            # Partition to which the Op belongs; all Ops with the
            # same partition_id are stitched together
            # Users should avoid setting this attribute manually
            # and instead use the floorplan transform to set
            # partition IDs from Vitis design rules and SLR IDs
            "partition_id": ("i", False, 0),
            # ID of FPGA device to which this Op is allocated, in
            # a multi-FPGA setting
            "device_id": ("i", False, 0),
            # input and output FIFO depths for multi-I/O nodes
            "inFIFODepths": ("ints", False, [2]),
            "outFIFODepths": ("ints", False, [2]),
            "output_hook": ("s", False, ""),
            # accumulated characteristic function over two periods
            "io_chrc_in": ("t", False, np.asarray([], dtype=np.int32)),
            "io_chrc_out": ("t", False, np.asarray([], dtype=np.int32)),
            # the period for which the characterization was run
            "io_chrc_period": ("i", False, 0),
            # amount of zero padding inserted during chrc.
            "io_chrc_pads_in": ("ints", False, []),
            "io_chrc_pads_out": ("ints", False, []),
        }

    def make_shape_compatible_op(self, model):
        oshape = self.get_normal_output_shape()
        # implement tensor with correct shape
        return super().make_const_shape_op(oshape)

    def get_verilog_top_module_name(self):
        "Return the Verilog top module name for this node."

        node = self.onnx_node
        prefixed_top_name = node.name

        return prefixed_top_name

    def get_verilog_top_module_intf_names(self):
        """Return a dict of names of input and output interfaces.
        The keys reflect the protocols each interface implements:
        'clk', 'rst', 'm_axis', 's_axis', 'aximm', 'axilite'.
        Values are lists of tuples (axis, aximm) or names (axilite):
        'axis' tuples correspond to the list of node inputs in order,
        each tuple is (interface_name, interface_width_bits).
        axilite always assumed to be 32 bits and is not tuple (name only).
        Each block must have at most one aximm and one axilite."""
        node = self.onnx_node
        intf_names = {}
        intf_names["clk"] = ["ap_clk"]
        intf_names["rst"] = ["ap_rst_n"]
        intf_names["s_axis"] = []
        for i in range(len(node.input)):
            # not every node input will result in an interface of the produced HW
            # filter out inputs that have no stream width associated with them
            width = self.get_instream_width_padded(i)
            if width != 0:
                intf_names["s_axis"].append(("in%d_V" % (i), self.get_instream_width_padded(i)))
        intf_names["m_axis"] = []
        for i in range(len(node.output)):
            intf_names["m_axis"].append(("out%d_V" % (i), self.get_outstream_width_padded(i)))
        intf_names["aximm"] = []
        intf_names["axilite"] = []
        intf_names["ap_none"] = []
        return intf_names

    def get_rtlsim(self):
        """Return a xsi wrapper for the emulation library
        for this node."""

        rtlsim_so = self.get_nodeattr("rtlsim_so")
        assert os.path.isfile(rtlsim_so), "Cannot find rtlsim library."

        sim_base, sim_rel = rtlsim_so.split("xsim.dir")
        sim_rel = "xsim.dir" + sim_rel
        # pass in correct tracefile from attribute
        tracefile = self.get_nodeattr("rtlsim_trace")
        if tracefile == "default":
            tracefile = self.onnx_node.name + ".wdb"
        sim = pyxsi_utils.load_sim_obj(sim_base, sim_rel, tracefile)

        return sim

    def close_rtlsim(self, sim):
        "Close and free up resources for rtlsim."
        pyxsi_utils.close_rtlsim(sim)

    def node_res_estimation(self, fpgapart):
        """Returns summarized resource estimation of BRAMs and LUTs
        of the node as a dictionary."""
        ret = dict()
        ret["BRAM_18K"] = self.bram_estimation()
        ret["BRAM_efficiency"] = self.bram_efficiency_estimation()
        ret["LUT"] = self.lut_estimation()
        ret["URAM"] = self.uram_estimation()
        ret["URAM_efficiency"] = self.uram_efficiency_estimation()
        ret["DSP"] = self.dsp_estimation(fpgapart)
        return ret

    def bram_efficiency_estimation(self):
        """Function for BRAM efficiency estimation: actual parameter storage
        needed divided by the allocated BRAM storage (from estimation)"""
        return 1

    def uram_efficiency_estimation(self):
        """Function for URAM efficiency estimation: actual parameter storage
        needed divided by the allocated URAM storage (from estimation)"""
        return 1

    def bram_estimation(self):
        """Function for BRAM resource estimation, is member function of
        HWCustomOp class but has to be filled by every node"""
        return 0

    def uram_estimation(self):
        """Function for UltraRAM resource estimation, is member function of
        HWCustomOp class but has to be filled by every node"""
        return 0

    def lut_estimation(self):
        """Function for LUT resource estimation, is member function of
        HWCustomOp class but has to be filled by every node"""
        return 0

    def dsp_estimation(self, fpgapart):
        """Function for DSP resource estimation, is member function of
        HWCustomOp class but has to be filled by every node"""
        return 0

    def get_exp_cycles(self):
        """Function for estimation of expected cycles for set folding,
        is member function of HWCustomOp class but has to be filled
        by every node"""
        return 0

    def get_op_and_param_counts(self):
        """Return a dictionary with number of ops needed per inference for
        this layer as well as parameter count (weights, thresholds, etc.).
        Entries should be in the format:
        {op_<optype> : <count>, param_<paramtype>: <count>}."""
        return {}

    def reset_rtlsim(self, sim):
        """Sets reset input in pyxsi to zero, toggles the clock and set it
        back to one"""
        pyxsi_utils.reset_rtlsim(sim)

    def toggle_clk(self, sim):
        """Toggles the clock input in pyxsi once."""
        pyxsi_utils.toggle_clk(sim)

    def rtlsim_multi_io(self, sim, io_dict, hook_postclk=None):
        "Run rtlsim for this node, supports multiple i/o streams."
        num_out_values = self.get_number_output_values()
        total_cycle_count = pyxsi_utils.rtlsim_multi_io(
            sim,
            io_dict,
            num_out_values,
            sname="_V_",
            liveness_threshold=get_liveness_threshold_cycles(),
            hook_postclk=hook_postclk,
        )

        self.set_nodeattr("cycles_rtlsim", total_cycle_count)

    def verify_node(self):
        """Can be implemented to verify that all attributes the node needs
        are there and that particular attributes are set correctly. Can also
        check if the number of inputs is equal to the expected number."""
        pass

    def generate_params(self, model, path):
        """Function to generate parameters (i.e. weights and thresholds),
        is member function of HWCustomOp class but has to be filled
        by every node that needs to generate parameters."""
        pass

    @abstractmethod
    def get_number_output_values(self):
        """Function to get the number of expected output values,
        is member function of HWCustomOp class but has to be filled
        by every node."""
        pass

    @abstractmethod
    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input stream ind."""

    @abstractmethod
    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output stream ind."""

    @abstractmethod
    def get_normal_input_shape(self, ind=0):
        """Returns normal input shape if implemented."""

    @abstractmethod
    def get_normal_output_shape(self, ind=0):
        """Returns folded output shape if implemented."""

    @abstractmethod
    def get_folded_input_shape(self, ind=0):
        """Returns folded input shape (according to synapse folding), if implemented."""

    @abstractmethod
    def get_folded_output_shape(self, ind=0):
        """Returns folded output shape (according to neuron folding), if implemented."""

    @abstractmethod
    def get_instream_width(self, ind=0):
        """Returns input stream width, if implemented."""

    @abstractmethod
    def get_outstream_width(self, ind=0):
        """Returns output stream width, if implemented."""

    def get_instream_width_padded(self, ind=0):
        """Returns input stream width padded to a multiple of 8. This is required
        by the AXI Stream spec."""
        in_width = self.get_instream_width(ind=ind)
        if in_width != 0:
            return roundup_to_integer_multiple(in_width, 8)
        else:
            return 0

    def get_outstream_width_padded(self, ind=0):
        """Returns output stream width padded to a multiple of 8. This is required
        by the AXI Stream spec."""
        out_width = self.get_outstream_width(ind=ind)
        return roundup_to_integer_multiple(out_width, 8)

    def generate_hdl_memstream(self, fpgapart, pumped_memory=0):
        """Helper function to generate verilog code for memstream component.
        Currently utilized by MVAU, VVAU and HLS Thresholding layer."""
        ops = ["MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl", "Thresholding_hls"]
        if self.onnx_node.op_type in ops:
            template_path = os.path.join(
                os.environ["FINN_RTLLIB"] + "/memstream/hdl/memstream_wrapper_template.v"
            )
            mname = self.onnx_node.name
            if self.onnx_node.op_type.startswith("Thresholding"):
                depth = self.calc_tmem()
            else:
                depth = self.calc_wmem()
            padded_width = self.get_instream_width_padded(1)
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

            ram_style = self.get_nodeattr("ram_style")
            init_file = code_gen_dir + "/memblock.dat"
            if ram_style == "ultra" and not is_versal(fpgapart):
                init_file = ""
            code_gen_dict = {
                "$MODULE_NAME$": [mname],
                "$DEPTH$": [str(depth)],
                "$WIDTH$": [str(padded_width)],
                "$INIT_FILE$": [init_file],
                "$RAM_STYLE$": [ram_style],
                "$PUMPED_MEMORY$": [str(pumped_memory)],
            }
            # apply code generation to template
            with open(template_path, "r") as f:
                template_wrapper = f.read()
            for key in code_gen_dict:
                # transform list into long string separated by '\n'
                code_gen_line = "\n".join(code_gen_dict[key])
                template_wrapper = template_wrapper.replace(key, code_gen_line)
            with open(
                os.path.join(code_gen_dir, mname + "_memstream_wrapper.v"),
                "w",
            ) as f:
                f.write(template_wrapper)
        else:
            pass

    def derive_characteristic_fxns(self, period, override_rtlsim_dict=None):
        """Return the unconstrained characteristic functions for this node."""
        # ensure rtlsim is ready
        assert self.get_nodeattr("rtlsim_so") != "", "rtlsim not ready for " + self.onnx_node.name
        if self.get_nodeattr("io_chrc_period") > 0:
            log.warning(f"Skipping node {self.onnx_node.name}: already has FIFO characteristic")
            return
        exp_cycles = self.get_exp_cycles()
        n_inps = np.prod(self.get_folded_input_shape()[:-1])
        n_outs = np.prod(self.get_folded_output_shape()[:-1])
        if exp_cycles == 0:
            # try to come up with an optimistic estimate
            exp_cycles = min(n_inps, n_outs)
        assert (
            exp_cycles <= period
        ), "Period %d too short to characterize %s : expects min %d cycles" % (
            period,
            self.onnx_node.name,
            exp_cycles,
        )
        sim = self.get_rtlsim()
        # signal name
        sname = "_V_"
        if override_rtlsim_dict is not None:
            io_dict = override_rtlsim_dict
        else:
            io_dict = {
                "inputs": {
                    "in0": [0 for i in range(n_inps)],
                },
                "outputs": {"out0": []},
            }

        # extra dicts to keep track of cycle-by-cycle transaction behavior
        # note that we restrict key names to filter out weight streams etc
        txns_in = {key: [] for (key, value) in io_dict["inputs"].items() if "in" in key}
        txns_out = {key: [] for (key, value) in io_dict["outputs"].items() if "out" in key}
        # signal name
        sname = "_V_"

        def monitor_txns(sim_obj):
            for inp in txns_in:
                in_ready = pyxsi_utils._read_signal(sim_obj, inp + sname + "TREADY") == 1
                in_valid = pyxsi_utils._read_signal(sim_obj, inp + sname + "TVALID") == 1
                if in_ready and in_valid:
                    txns_in[inp].append(1)
                else:
                    txns_in[inp].append(0)
            for outp in txns_out:
                if (
                    pyxsi_utils._read_signal(sim_obj, outp + sname + "TREADY") == 1
                    and pyxsi_utils._read_signal(sim_obj, outp + sname + "TVALID") == 1
                ):
                    txns_out[outp].append(1)
                else:
                    txns_out[outp].append(0)

        self.reset_rtlsim(sim)
        self.rtlsim_multi_io(
            sim,
            io_dict,
            hook_postclk=monitor_txns,
        )
        total_cycle_count = self.get_nodeattr("cycles_rtlsim")
        assert (
            total_cycle_count <= period
        ), """Total cycle count from rtl simulation is higher than
            specified period, please set the period higher than {}""".format(
            total_cycle_count
        )
        self.set_nodeattr("io_chrc_period", period)

        def accumulate_char_fxn(chrc):
            p = len(chrc)
            ret = []
            for t in range(2 * p):
                if t == 0:
                    ret.append(chrc[0])
                else:
                    ret.append(ret[-1] + chrc[t % p])
            return np.asarray(ret, dtype=np.int32)

        all_txns_in = np.empty((len(txns_in.keys()), 2 * period), dtype=np.int32)
        all_txns_out = np.empty((len(txns_out.keys()), 2 * period), dtype=np.int32)
        all_pad_in = []
        all_pad_out = []
        for in_idx, in_strm_nm in enumerate(txns_in.keys()):
            txn_in = txns_in[in_strm_nm]
            if len(txn_in) < period:
                pad_in = period - len(txn_in)
                txn_in += [0 for x in range(pad_in)]
            txn_in = accumulate_char_fxn(txn_in)
            all_txns_in[in_idx, :] = txn_in
            all_pad_in.append(pad_in)

        for out_idx, out_strm_nm in enumerate(txns_out.keys()):
            txn_out = txns_out[out_strm_nm]
            if len(txn_out) < period:
                pad_out = period - len(txn_out)
                txn_out += [0 for x in range(pad_out)]
            txn_out = accumulate_char_fxn(txn_out)
            all_txns_out[out_idx, :] = txn_out
            all_pad_out.append(pad_out)

        self.set_nodeattr("io_chrc_in", all_txns_in)
        self.set_nodeattr("io_chrc_out", all_txns_out)
        self.set_nodeattr("io_chrc_pads_in", all_pad_in)
        self.set_nodeattr("io_chrc_pads_out", all_pad_out)
