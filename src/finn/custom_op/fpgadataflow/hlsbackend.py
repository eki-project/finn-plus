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

"""HLS backend implementation for FINN custom operations."""

import numpy as np
import os
from abc import ABC, abstractmethod
from pathlib import Path
from qonnx.core.datatype import DataType

from finn import xsi
from finn.custom_op.fpgadataflow import templates
from finn.templates import get_templates_folder
from finn.util.basic import CppBuilder, launch_process_helper, make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.deps import get_deps_path
from finn.util.exception import FINNError, FINNUserError
from finn.util.hls import CallHLS
from finn.util.logging import log

finnxsi = xsi if xsi.is_available() else None


class HLSBackend(ABC):
    """HLSBackend class all custom ops that correspond to a finn-hlslib
    function are using functionality of. Contains different functions every HLS
    custom node should have. Some as abstract methods, these have to be filled
    when writing a new HLS custom op node."""

    def get_nodeattr_types(self):
        """Return dictionary of node attribute types and properties."""
        return {
            "code_gen_dir_cppsim": ("s", False, ""),
            "executable_path": ("s", False, ""),
            "res_hls": ("s", False, ""),
            # temporary node attribute to keep track of interface style of hls ops
            "cpp_interface": ("s", False, "packed", {"packed", "hls_vector"}),
            # temporary node attribute to keep track of execution style of hls ops
            "hls_style": ("s", False, "ifm_aware", {"ifm_aware", "freerunning"}),
        }

    def get_all_verilog_paths(self):
        """Return list of all folders containing Verilog code for this node."""

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        assert (
            code_gen_dir != ""
        ), """Node attribute "code_gen_dir_ipgen" is
        not set. Please run HLSSynthIP first."""
        verilog_path = "{}/project_{}/sol1/impl/verilog/".format(code_gen_dir, self.onnx_node.name)
        subcore_verilog_path = "{}/project_{}/sol1/impl/ip/hdl/ip/".format(
            code_gen_dir, self.onnx_node.name
        )
        # default impl only returns the HLS verilog codegen dir and subcore (impl/ip/hdl/ip) dir
        # if it exists
        ret = [verilog_path]
        if os.path.isdir(subcore_verilog_path):
            ret += [subcore_verilog_path]
        return ret

    def get_all_verilog_filenames(self, abspath=False):
        """Return list of all Verilog files used for this node."""

        verilog_files = []
        verilog_paths = self.get_all_verilog_paths()
        for verilog_path in verilog_paths:
            for f in os.listdir(verilog_path):
                if f.endswith(".v"):
                    if abspath:
                        verilog_files += [verilog_path + "/" + f]
                    else:
                        verilog_files += [f]
        return verilog_files

    def prepare_rtlsim(self, behav=False):
        """Creates a xsi emulation library for the RTL code generated
        for this node, sets the rtlsim_so attribute to its path."""

        verilog_files = self.get_all_verilog_filenames(abspath=True)
        single_src_dir = make_build_dir("rtlsim_" + self.onnx_node.name + "_")
        trace_file = self.get_nodeattr("rtlsim_trace")
        debug = not (trace_file is None or trace_file == "")
        ret = finnxsi.compile_sim_obj(
            self.get_verilog_top_module_name(), verilog_files, single_src_dir, debug, behav
        )
        # save generated lib filename in attribute
        self.set_nodeattr("rtlsim_so", ret[0] + "/" + ret[1])

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Generate C++ code and TCL script for IP generation."""
        node = self.onnx_node

        # generate top cpp file for ip generation
        path = self.get_nodeattr("code_gen_dir_ipgen")
        self.code_gen_dict["$AP_INT_MAX_W$"] = [str(self.get_ap_int_max_w())]
        self.generate_params(model, path)
        self.global_includes()
        self.defines("ipgen")
        self.blackboxfunction()
        self.pragmas()
        self.docompute()

        template = templates.ipgen_template

        for key in self.code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(self.code_gen_dict[key])
            template = template.replace(key, code_gen_line)
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        f = open(os.path.join(code_gen_dir, "top_{}.cpp".format(node.name)), "w")
        f.write(template)
        f.close()
        self.code_gen_dict.clear()

        if node.name in ["", None]:
            raise FINNUserError(
                f"[HLS Code Generation] A {node.op_type} node has no name."
                f"This will likely cause IP generation "
                f"to fail. Consider calling GiveUniqueNodeNames() beforehand."
            )

        # generate tcl script for ip generation
        self.code_gen_dict["$PROJECTNAME$"] = ["project_{}".format(node.name)]
        self.code_gen_dict["$HWSRCDIR$"] = [code_gen_dir]
        self.code_gen_dict["$FPGAPART$"] = [fpgapart]
        self.code_gen_dict["$TOPFXN$"] = [node.name]
        self.code_gen_dict["$CLKPERIOD$"] = [str(clk)]
        self.code_gen_dict["$DEFAULT_DIRECTIVES$"] = self.ipgen_default_directives()
        self.code_gen_dict["$EXTRA_DIRECTIVES$"] = self.ipgen_extra_directives()
        self.code_gen_dict["$FINNHLSLIB$"] = [str(get_deps_path() / "finn-hlslib")]
        self.code_gen_dict["$ATTENTIONHLSLIB$"] = [str(get_deps_path() / "attention-hlslib")]

        template = templates.ipgentcl_template

        for key in self.code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(self.code_gen_dict[key])
            template = template.replace(key, code_gen_line)
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        f = open(os.path.join(code_gen_dir, "hls_syn_{}.tcl".format(node.name)), "w")
        f.write(template)
        f.close()
        self.code_gen_dict.clear()

    def ipgen_default_directives(self):
        """Return list of default HLS synthesis directives."""

        default_directives = [
            "set_param hls.enable_hidden_option_error false",
            "config_compile -disable_unroll_code_size_check -pipeline_style flp",
            "config_interface -m_axi_addr64",
            "config_rtl -module_auto_prefix",
            "config_rtl -deadlock_detection none",
        ]
        return default_directives

    def ipgen_extra_directives(self):
        """Return a list of extra TCL directives for HLS synthesis."""
        return []

    def ipgen_singlenode_code(self):
        """Build the bash script for IP generation using the CallHLS utility."""
        node = self.onnx_node
        code_gen_dir = Path(self.get_nodeattr("code_gen_dir_ipgen"))
        builder = CallHLS(
            tcl_script=code_gen_dir / f"hls_syn_{node.name}.tcl",
            code_gen_dir=code_gen_dir,
            ipgen_path=code_gen_dir / f"project_{node.name}",
        )
        success = False
        while not success:
            builder.build()
            if not builder.ipgen_path.is_dir():
                raise FINNUserError(
                    f"Generated IP couldn't be found at "
                    f"{builder.ipgen_path}. Check logs at {code_gen_dir}."
                )
            ipgen_path = str(builder.ipgen_path)
            self.set_nodeattr("ipgen_path", ipgen_path)
            ip_path = ipgen_path + "/sol1/impl/ip"
            if not os.path.isdir(ip_path):
                # Workaround for possible race condition between Vitis HLS instances
                is_port_conflict = False
                xcd_log_path = os.path.join(ipgen_path, "sol1", ".autopilot", "xcd.log")
                if os.path.isfile(xcd_log_path):
                    with open(xcd_log_path, "r") as xcd_log:
                        for line in xcd_log:
                            if "Address already in use" in line:
                                is_port_conflict = True
                if is_port_conflict:
                    log.warning(
                        f"Vitis HLS IPGen ({code_gen_dir}) failed due to race condition "
                        "(XCD server port conflict). Retrying..."
                    )
                else:
                    raise FINNError(
                        f"IPGen failed: {ip_path} not found. Check log under {code_gen_dir}"
                    )
            else:
                success = True

        self.set_nodeattr("ip_path", ip_path)
        vlnv = "xilinx.com:hls:%s:1.0" % node.name
        self.set_nodeattr("ip_vlnv", vlnv)

    def code_generation_cppsim(self, model):
        """Generate C++ code for simulation (cppsim)."""
        node = self.onnx_node
        path = self.get_nodeattr("code_gen_dir_cppsim")
        self.code_gen_dict["$AP_INT_MAX_W$"] = [str(self.get_ap_int_max_w())]
        self.generate_params(model, path)
        self.global_includes()
        self.defines("cppsim")
        self.read_npy_data()
        self.strm_decl()
        self.pragmas()
        self.docompute()
        self.dataoutstrm()
        self.save_as_npy()

        if self.get_nodeattr("hls_style") == "freerunning":
            self.timeout_value()
            self.timeout_condition()
            self.timeout_read_stream()
            template = templates.docompute_template_timeout
        else:
            template = templates.docompute_template

        for key in self.code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(self.code_gen_dict[key])
            template = template.replace(key, code_gen_line)
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        f = open(os.path.join(code_gen_dir, "execute_{}.cpp".format(node.op_type)), "w")
        f.write(template)
        f.close()
        self.code_gen_dict.clear()

    def code_generation_ipi(self):
        """Construct and return the TCL for node instantiation in Vivado IPI."""
        vlnv = self.get_nodeattr("ip_vlnv")
        cmd = ["create_bd_cell -type ip -vlnv %s %s" % (vlnv, self.onnx_node.name)]
        return cmd

    def compile_singlenode_code(self):
        """Build bash script for compilation using CppBuilder and execute to produce executable."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        hls_path = os.environ.get("XILINX_HLS")
        builder = CppBuilder()
        # to enable additional debug features please uncommand the next line
        # builder.append_includes("-DDEBUG")
        builder.append_includes(f"-I{get_templates_folder()}/npy2stream")
        builder.append_includes("-I" + str(get_deps_path() / "cnpy"))
        builder.append_includes("-I" + str(get_deps_path() / "finn-hlslib"))
        # TODO: Is it ok to add this here? Add some specialization to the
        #  attention operator? Eventually integrate this into the finn-hlslib?
        builder.append_includes("-I" + str(get_deps_path() / "attention-hlslib"))
        builder.append_includes("-I$FINN_CUSTOM_HLS")
        builder.append_includes(f"-I{hls_path}/include")
        builder.append_includes("--std=c++14")
        builder.append_includes("-O3")
        builder.append_sources(code_gen_dir + "/*.cpp")
        builder.append_sources(str(get_deps_path() / "cnpy" / "cnpy.cpp"))
        builder.append_includes("-lz")
        builder.append_includes("-fno-builtin -fno-inline")
        builder.append_includes(f'-Wl,-rpath,"{hls_path}/lnx64/lib/csim"')
        builder.append_includes(f"-L{hls_path}/lnx64/lib/csim -lhlsmc++-GCC46")
        builder.append_includes(f'-Wl,-rpath,"{hls_path}/lnx64/tools/fpo_v7_1"')
        builder.append_includes(f"-L{hls_path}/lnx64/tools/fpo_v7_1 -lgmp -lmpfr")
        builder.append_includes("-lIp_floating_point_v7_1_bitacc_cmodel")
        builder.set_executable_path(code_gen_dir + "/node_model")
        builder.build(code_gen_dir)
        self.set_nodeattr("executable_path", builder.executable_path)

    def npy_to_dynamic_output(self, context):
        """Read output.npy file generated from cppsim and place into context dictionary."""
        node = self.onnx_node
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        for o, outp in enumerate(node.output):
            output = np.load("{}/output_{}.npy".format(code_gen_dir, o))
            exp_shape = self.get_normal_output_shape(o)
            context[outp] = output.reshape(exp_shape)

    def exec_precompiled_singlenode_model(self):
        """Execute precompiled executable."""
        executable_path = self.get_nodeattr("executable_path")
        if executable_path == "":
            raise Exception(
                """
Found no executable for this node, did you run the codegen and
compilation transformations?
            """
            )
        launch_process_helper(executable_path, print_stdout=False)

    # TODO: Should have been removed by refactoring (PR #1318)
    # However, it is still used by some CustomOps, namely:
    # SplitMultiHeads, MergeMultiHeads, ScaledDotProductAttention,
    # ReplicateStream, StreamingConcat
    def hls_sname(self):
        """Get the naming convention used by Vitis HLS for stream signals
        Example: the TDATA for a stream called "out" would be out_V_TDATA.
        """
        return "V"

    def execute_node(self, context, graph):
        """Execute node in specified mode (cppsim or rtlsim)."""
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node

        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )
        inputs = {}
        for i, inp in enumerate(node.input):
            nbits = self.get_instream_width(i)
            # If the stream is not exposed, it has 0 width and no npy file will be created
            # Do this check before get_normal_input_shape() because some operators (attention)
            # still return dummy shapes for some non-exposed inputs (TODO)
            if nbits == 0:
                continue
            exp_ishape = tuple(self.get_normal_input_shape(i))
            folded_ishape = self.get_folded_input_shape(i)
            inp_val = context[inp]
            # Make sure the input has the right container datatype
            if inp_val.dtype not in [np.float32, np.float16]:
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

            if export_idt == DataType["BIPOLAR"]:
                # store bipolar activations as binary
                inp_val = (inp_val + 1) / 2
                export_idt = DataType["BINARY"]

            reshaped_input = inp_val.reshape(folded_ishape)
            reshaped_input = reshaped_input.copy()
            # This npy file will be read by the cppsim executable
            np.save(os.path.join(code_gen_dir, "input_%s.npy" % i), reshaped_input)
            # The rtlsim will instead operate on a flattened int sequence from an "io_dict"
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_{}.npy".format(code_gen_dir, i), export_idt, nbits
            )
            inputs["in%s" % i] = rtlsim_inp

        if mode == "cppsim":
            # execute the precompiled model
            self.exec_precompiled_singlenode_model()
            # load output npy file
            self.npy_to_dynamic_output(context)
            for o, outp in enumerate(node.output):
                exp_oshape = tuple(self.get_normal_output_shape(o))
                assert (
                    context[outp].shape == exp_oshape
                ), "cppsim did not produce expected output shape"
                # binary -> bipolar if needed
                if self.get_output_datatype(o) == DataType["BIPOLAR"]:
                    out = context[outp]
                    out = 2 * out - 1
                    context[outp] = out
        elif mode == "rtlsim":
            outputs = {}
            for o, outp in enumerate(node.output):
                outputs["out%s" % o] = []
            # assembled execution context
            io_dict = {"inputs": inputs, "outputs": outputs}

            sim = self.get_rtlsim()
            self.reset_rtlsim(sim)
            self.rtlsim_multi_io(sim, io_dict)
            self.close_rtlsim(sim)
            for o, outp in enumerate(node.output):
                rtlsim_output = io_dict["outputs"]["out%s" % o]
                odt = self.get_output_datatype(o)
                target_bits = odt.bitwidth()
                packed_bits = self.get_outstream_width(o)
                out_npy_path = "{}/output_{}.npy".format(code_gen_dir, o)
                out_shape = self.get_folded_output_shape(o)
                rtlsim_output_to_npy(
                    rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
                )
                # load and reshape output
                exp_oshape = tuple(self.get_normal_output_shape(o))
                output = np.load(out_npy_path)
                output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
                context[outp] = output

                assert (
                    context[outp].shape == exp_oshape
                ), "Output shape doesn't match expected shape."

        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

    @abstractmethod
    def global_includes(self):
        """Function to set the global includes for c++ code that has to be generated
        for cppsim or rtlsim, is member function of HLSBackend class but has to
        be filled by every node."""
        pass

    @abstractmethod
    def defines(self, var):
        """Function to set the define commands for c++ code that has to be generated
        for cppsim or rtlsim, is member function of HLSBackend class but has to
        be filled by every node.

        var: makes it possible to reuse the function for different c++ code generation.
        I.e. if set to "ipgen" in MatrixVectorActivation additional PRAGMA defines are
        added."""
        pass

    def read_npy_data(self):
        """Generate commands for reading data from .npy file in C++.
        Might need to be overwritten depending on CustomOp."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        self.code_gen_dict["$READNPYDATA$"] = []
        cpp_interface = self.get_nodeattr("cpp_interface")

        for i, inp in enumerate(self.onnx_node.input):
            dtype = self.get_input_datatype(i)
            if dtype == DataType["BIPOLAR"]:
                # use binary for bipolar storage
                dtype = DataType["BINARY"]
            elem_hls_type = dtype.get_hls_datatype_str()
            npy_type = "half" if elem_hls_type == "half" else "float"
            npy_in = "%s/input_%s.npy" % (code_gen_dir, i)

            iwidth = self.get_instream_width(i)
            # if the stream is not exposed, it has 0 width and no npy file will be created
            if iwidth == 0:
                continue
            if cpp_interface == "packed":
                elem_bits = dtype.bitwidth()
                packed_bits = iwidth
                packed_hls_type = "ap_uint<%d>" % packed_bits
                self.code_gen_dict["$READNPYDATA$"].append(
                    'npy2apintstream<%s, %s, %d, %s>("%s", in%s_V);'
                    % (
                        packed_hls_type,
                        elem_hls_type,
                        elem_bits,
                        npy_type,
                        npy_in,
                        i,
                    )
                )
            else:
                folded_shape = self.get_folded_input_shape()
                self.code_gen_dict["$READNPYDATA$"].append(
                    'npy2vectorstream<%s, %s, %d>("%s", in%s_V, false);'
                    % (
                        elem_hls_type,
                        npy_type,
                        folded_shape[-1],
                        npy_in,
                        i,
                    )
                )

    def strm_decl(self):
        """Generate commands for stream declaration in C++.
        Might need to be overwritten depending on CustomOp."""
        node = self.onnx_node
        cpp_interface = self.get_nodeattr("cpp_interface")
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        if cpp_interface == "packed":
            for i, inp in enumerate(node.input):
                if self.get_instream_width(i):
                    self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                        'hls::stream<ap_uint<{}>> in{}_V ("in{}_V");'.format(
                            self.get_instream_width(i), i, i
                        )
                    )
            for o, outp in enumerate(node.output):
                if self.get_outstream_width(o):
                    self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                        'hls::stream<ap_uint<{}>> out{}_V ("out{}_V");'.format(
                            self.get_outstream_width(o), o, o
                        )
                    )
        else:
            for i, inp in enumerate(node.input):
                if self.get_instream_width(i):
                    dtype = self.get_input_datatype(i)
                    if dtype == DataType["BIPOLAR"]:
                        # use binary for bipolar storage
                        dtype = DataType["BINARY"]
                    elem_input_hls_type = dtype.get_hls_datatype_str()

                    self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                        'hls::stream<hls::vector<{},{}>> in{}_V ("in{}_V");'.format(
                            elem_input_hls_type, self.get_folded_input_shape(i)[-1], i, i
                        )
                    )

            for o, outp in enumerate(node.output):
                if self.get_outstream_width(o):
                    dtype = self.get_output_datatype(o)
                    if dtype == DataType["BIPOLAR"]:
                        # use binary for bipolar storage
                        dtype = DataType["BINARY"]
                    elem_output_hls_type = dtype.get_hls_datatype_str()

                    self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                        'hls::stream<hls::vector<{},{}>> out{}_V ("out{}_V");'.format(
                            elem_output_hls_type, self.get_folded_output_shape(o)[-1], o, o
                        )
                    )

            if self.get_nodeattr("hls_style") == "freerunning":
                for o, outp in enumerate(node.output):
                    if self.get_outstream_width(o):
                        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                            'hls::stream<hls::vector<{},{}>> strm{} ("strm{}");'.format(
                                elem_output_hls_type, self.get_folded_output_shape(o)[-1], o, o
                            )
                        )

    @abstractmethod
    def docompute(self):
        """Function to generate the commands for the computational part of the
        c++ code, is member function of HLSBackend class but has to be filled
        by every node."""
        pass

    def dataoutstrm(self):
        """Generate commands for reading out data from C++ and converting to npy format.
        Might need to be overwritten depending on CustomOp."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        self.code_gen_dict["$DATAOUTSTREAM$"] = []

        for o, outp in enumerate(self.onnx_node.output):
            dtype = self.get_output_datatype(o)
            if dtype == DataType["BIPOLAR"]:
                # use binary for bipolar storage
                dtype = DataType["BINARY"]
            elem_hls_type = dtype.get_hls_datatype_str()
            npy_type = "half" if elem_hls_type == "half" else "float"
            npy_out = "%s/output_%s.npy" % (code_gen_dir, o)
            oshape = self.get_folded_output_shape(o)
            oshape_cpp_str = str(oshape).replace("(", "{").replace(")", "}")

            cpp_interface = self.get_nodeattr("cpp_interface")

            if cpp_interface == "packed":
                elem_bits = dtype.bitwidth()
                packed_bits = self.get_outstream_width(o)
                packed_hls_type = "ap_uint<%d>" % packed_bits

                self.code_gen_dict["$DATAOUTSTREAM$"].append(
                    'apintstream2npy<%s, %s, %d, %s>(out%s_V, %s, "%s");'
                    % (
                        packed_hls_type,
                        elem_hls_type,
                        elem_bits,
                        npy_type,
                        o,
                        oshape_cpp_str,
                        npy_out,
                    )
                )
            else:
                folded_shape = self.get_folded_output_shape(o)
                out_vector = (
                    f"strm{o}" if self.get_nodeattr("hls_style") == "freerunning" else f"out{o}_V"
                )
                self.code_gen_dict["$DATAOUTSTREAM$"].append(
                    'vectorstream2npy<%s, %s, %d>(%s, %s, "%s");'
                    % (
                        elem_hls_type,
                        npy_type,
                        folded_shape[-1],
                        out_vector,
                        oshape_cpp_str,
                        npy_out,
                    )
                )

    def save_as_npy(self):
        """Generate commands for saving data in .npy file in C++."""
        self.code_gen_dict["$SAVEASCNPY$"] = []

    @abstractmethod
    def blackboxfunction(self):
        """Function to generate a blackbock function in c++ from which an IP block
        will be generated, is member function of HLSBackend class but has to be filled
        by every node."""
        pass

    def pragmas(self):
        """Generate pragma commands in C++.
        Might need to be overwritten depending on CustomOp."""
        # TODO: make this loop over all inputs/outputs so we don't need as much
        # specialization in the child classes (e.g., ScaledDotProductAttention)
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0_V"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

    def get_ap_int_max_w(self):
        """Return the maximum width of any ap_int used in this module. Used to set the
        AP_INT_MAX_W definition for HLS."""
        instream = self.get_instream_width()
        outstream = self.get_outstream_width()
        ret = max([instream, outstream])
        assert ret <= 8191, "AP_INT_MAX_W=%d is larger than allowed maximum of 8191" % ret
        return ret

    def timeout_value(self):
        """Set timeout value for HLS functions defined for one clock cycle."""
        self.code_gen_dict["$TIMEOUT_VALUE$"] = ["1000"]

    def timeout_condition(self):
        """Set timeout condition for HLS functions defined for one clock cycle."""
        self.code_gen_dict["$TIMEOUT_CONDITION$"] = ["out0_V.empty()"]

    def timeout_read_stream(self):
        """Set reading output stream procedure for HLS functions defined for one clock cycle."""
        self.code_gen_dict["$TIMEOUT_READ_STREAM$"] = ["strm0 << out0_V.read();"]
