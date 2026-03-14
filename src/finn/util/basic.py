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

"""
Basic utility functions and classes for FINN.

This module provides essential utility functions and classes used throughout
the FINN framework, including:

- FPGA board and part mappings (PYNQ boards, Alveo cards, etc.)
- File system utilities and path operations
- Build environment helpers (Vivado, Vitis, etc.)
- C++ compilation utilities through the CppBuilder class
- FPGA-specific functionality detection (Versal, DSP blocks, etc.)

The module serves as a foundation for other FINN components that need
basic system operations, hardware abstraction, and build tool integration.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import gen_finn_dt_tensor
from typing import Dict

from finn.util.data_packing import finnpy_to_packed_bytearray
from finn.util.logging import log
from finn.util.settings import get_settings

# test boards used for bnn pynq tests
test_board_map = ["Pynq-Z1", "KV260_SOM", "ZCU104", "U55C"]

# mapping from PYNQ board names to FPGA part names
pynq_part_map: dict[str, str] = {}
pynq_part_map["Ultra96"] = "xczu3eg-sbva484-1-e"
pynq_part_map["Ultra96-V2"] = "xczu3eg-sbva484-1-i"
pynq_part_map["Pynq-Z1"] = "xc7z020clg400-1"
pynq_part_map["Pynq-Z2"] = "xc7z020clg400-1"
pynq_part_map["ZCU102"] = "xczu9eg-ffvb1156-2-e"
pynq_part_map["ZCU104"] = "xczu7ev-ffvc1156-2-e"
pynq_part_map["ZCU111"] = "xczu28dr-ffvg1517-2-e"
pynq_part_map["RFSoC2x2"] = "xczu28dr-ffvg1517-2-e"
pynq_part_map["RFSoC4x2"] = "xczu48dr-ffvg1517-2-e"
pynq_part_map["KV260_SOM"] = "xck26-sfvc784-2LV-c"
pynq_part_map["AUP-ZU3_8GB"] = "xczu3eg-sfvc784-2-e"


# native AXI HP port width (in bits) for PYNQ boards
pynq_native_port_width: dict[str, int] = {}
pynq_native_port_width["Pynq-Z1"] = 64
pynq_native_port_width["Pynq-Z2"] = 64
pynq_native_port_width["Ultra96"] = 128
pynq_native_port_width["Ultra96-V2"] = 128
pynq_native_port_width["ZCU102"] = 128
pynq_native_port_width["ZCU104"] = 128
pynq_native_port_width["ZCU111"] = 128
pynq_native_port_width["RFSoC2x2"] = 128
pynq_native_port_width["RFSoC4x2"] = 128
pynq_native_port_width["KV260_SOM"] = 128
pynq_native_port_width["AUP-ZU3_8GB"] = 128

# Alveo device and platform mappings
alveo_part_map: dict[str, str] = {}
alveo_part_map["U50"] = "xcu50-fsvh2104-2L-e"
alveo_part_map["U200"] = "xcu200-fsgd2104-2-e"
alveo_part_map["U250"] = "xcu250-figd2104-2L-e"
alveo_part_map["U280"] = "xcu280-fsvh2892-2L-e"
alveo_part_map["U55C"] = "xcu55c-fsvh2892-2L-e"

alveo_default_platform: dict[str, str] = {}
alveo_default_platform["U50"] = "xilinx_u50_gen3x16_xdma_5_202210_1"
alveo_default_platform["U200"] = "xilinx_u200_gen3x16_xdma_2_202110_1"
alveo_default_platform["U250"] = "xilinx_u250_gen3x16_xdma_4_1_202210_1"
alveo_default_platform["U280"] = "xilinx_u280_gen3x16_xdma_1_202211_1"
alveo_default_platform["U55C"] = "xilinx_u55c_gen3x16_xdma_3_202210_1"

# Create a joint part map, encompassing other boards too
part_map: dict[str, str] = {**pynq_part_map, **alveo_part_map}
part_map["VEK280"] = "xcve2802-vsvh1760-2MP-e-S"
part_map["VCK190"] = "xcvc1902-vsva2197-2MP-e-S"
part_map["V80"] = "xcv80-lsva4737-2MHP-e-s"


def get_rtlsim_trace_depth():
    """Return the trace depth for rtlsim. Controllable
    via the RTLSIM_TRACE_DEPTH environment variable. If the env.var. is
    undefined, the default value of 1 is returned. A trace depth of 1
    will only show top-level signals and yield smaller .vcd files.

    The following depth values are of interest for whole-network stitched IP
    rtlsim:
    - level 1 shows top-level input/output streams
    - level 2 shows per-layer input/output streams
    - level 3 shows per full-layer I/O including FIFO count signals
    """

    try:
        return int(os.environ["RTLSIM_TRACE_DEPTH"])
    except KeyError:
        return 1


def get_finn_root():
    """
    Deprecated function that should not be used anymore.

    This function was previously used to get the FINN root directory,
    but has been deprecated and should not be called in new code.

    Raises:
        Exception: Always raises an exception indicating the function
                  should not be used.
    """
    raise Exception("get_finn_root() should not be used anymore.")


def get_vivado_root():
    "Return the root directory that Vivado is installed into."

    try:
        return os.environ["XILINX_VIVADO"]
    except KeyError:
        raise Exception(
            """Environment variable XILINX_VIVADO must be set
        correctly. Please ensure you have launched the Docker contaier correctly.
        """
        )


def get_liveness_threshold_cycles():
    """Return the number of no-output cycles rtlsim will wait before assuming
    the simulation is not finishing and throwing an exception."""

    return int(os.getenv("LIVENESS_THRESHOLD", 1000000))


def make_build_dir(prefix: str = "", return_as_path: bool = False) -> str:
    """Creates a folder with given prefix to be used as a build dir.
    Use this function instead of tempfile.mkdtemp to ensure any generated files
    will survive on the host after the FINN Docker container exits."""
    try:
        build_dir = get_settings().finn_build_dir
    except KeyError as keyerror:
        raise Exception("""Environment variable FINN_BUILD_DIR is missing!""") from keyerror

    if not build_dir.exists():
        raise Exception(
            f"FINN_BUILD_DIR at {build_dir} does not exist! "
            "Make sure the FINN setup ran properly!"
        )

    tmpdir = Path(tempfile.mkdtemp(prefix=prefix, dir=build_dir))
    if return_as_path:
        return tmpdir
    return str(tmpdir)


def launch_process_helper(args, proc_env=None, cwd=None, print_stdout=True, print_stderr=True):
    """Helper function to launch a process in a way that facilitates logging
    stdout/stderr with Python loggers.
    Returns (cmd_out, cmd_err) if successful, raises CalledProcessError otherwise."""
    process = subprocess.run(args, capture_output=True, env=proc_env, cwd=cwd, text=True)
    cmd_out = process.stdout.strip()
    cmd_err = process.stderr.strip()

    # Handle stdout
    if cmd_out:
        if print_stdout is True:
            log.info(cmd_out)
        else:
            # Print with DEBUG level regardless
            log.debug(cmd_out)

    # Handle stderr, depending on return code
    if process.returncode == 0:
        # Process completed successfully, log stderr only as WARNING
        if cmd_err and print_stderr:
            log.warning(cmd_err)
    else:
        # Process failed, log stderr as ERROR
        if cmd_err:
            log.error(cmd_err)

        # Log additional ERROR message
        if isinstance(args, list):
            cmd = " ".join(args)
        else:
            cmd = args
        log.error(f"Launched process returned non-zero exit code ({process.returncode}): {cmd}")

    # Raise CalledProcessError for non-zero return code
    process.check_returncode()
    return (cmd_out, cmd_err)


def which(program):
    "Python equivalent of the shell cmd 'which'."

    # source:
    # https://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
    def is_exe(fpath):
        """
        Check if a file path points to an executable file.

        Tests whether the given file path exists and has execute permissions.
        This is a helper function used by the which() function.

        Args:
            fpath (str): File path to check for executability.

        Returns:
            bool: True if the file exists and is executable, False otherwise.
        """
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


class CppBuilder:
    """Builds the g++ compiler command to produces the executable of the c++ code
    in code_gen_dir which is passed to the function build() of this class."""

    def __init__(self):
        """
        Initialize a new CppBuilder instance.

        Sets up empty lists and variables for building C++ compilation commands.
        All instance variables are initialized to empty states and should be
        populated using the various setter and append methods before calling build().

        Instance variables initialized:
            include_paths (list): List of include directory paths
            cpp_files (list): List of C++ source file paths
            executable_path (str): Path where the compiled executable will be placed
            code_gen_dir (str): Directory for code generation
            compile_components (list): List of compilation command components
            compile_script (str): Generated compilation script content
        """
        self.include_paths = []
        self.cpp_files = []
        self.executable_path = ""
        self.code_gen_dir = ""
        self.compile_components = []
        self.compile_script = ""

    def append_includes(self, library_path):
        """Adds given library path to include_paths list."""
        self.include_paths.append(library_path)

    def append_sources(self, cpp_file):
        """Adds given c++ file to cpp_files list."""
        self.cpp_files.append(cpp_file)

    def set_executable_path(self, path):
        """Sets member variable "executable_path" to given path."""
        self.executable_path = path

    def build(self, code_gen_dir):
        """Builds the g++ compiler command according to entries in include_paths
        and cpp_files lists. Saves it in bash script in given folder and
        executes it."""
        # raise error if includes are empty
        self.code_gen_dir = code_gen_dir
        self.compile_components.append("g++ -o " + str(self.executable_path))
        for cpp_file in self.cpp_files:
            self.compile_components.append(cpp_file)
        for lib in self.include_paths:
            self.compile_components.append(lib)
        bash_compile = ""
        for component in self.compile_components:
            bash_compile += str(component) + " "
        self.compile_script = str(self.code_gen_dir) + "/compile.sh"
        with open(self.compile_script, "w") as f:
            f.write("#!/bin/bash \n")
            f.write(bash_compile + "\n")
        bash_command = ["bash", self.compile_script]
        launch_process_helper(bash_command, print_stdout=False)


def is_versal(fpgapart):
    """Returns whether board is part of the Versal family"""
    return fpgapart[0:4] in ["xcvc", "xcve", "xcvp", "xcvm", "xqvc", "xqvm"] or fpgapart[0:5] in [
        "xqrvc",
        "xcv80",
    ]


def get_dsp_block(fpgapart):
    """
    Determine the DSP block type based on the FPGA part name.

    Different FPGA families and generations use different DSP block types.
    This function maps FPGA part names to their corresponding DSP block
    architecture for proper resource utilization and optimization.

    Args:
        fpgapart (str): FPGA part name/identifier (e.g., "xczu7ev-ffvc1156-2-e")

    Returns:
        str: DSP block type identifier. Returns:
             - "DSP58" for Versal family FPGAs
             - "DSP48E1" for 7-series FPGAs
             - "DSP48E2" for UltraScale/UltraScale+ FPGAs
    """
    if is_versal(fpgapart):
        return "DSP58"
    elif fpgapart[2] == "7":
        return "DSP48E1"
    else:
        return "DSP48E2"


def get_driver_shapes(model: ModelWrapper) -> Dict:
    """Get all the IO shapes for the driver."""
    idt = []
    idma_names = []
    ishape_normal = []
    ishape_folded = []
    ishape_packed = []
    for idma_ind, graph_in in enumerate(model.graph.input):
        i_tensor_name = graph_in.name
        # get inp tensor properties
        i_tensor_dt = model.get_tensor_datatype(i_tensor_name)
        i_tensor_shape_normal = tuple(model.get_tensor_shape(i_tensor_name))
        # go down into dataflow partition to get folded shape info etc
        # TODO consider setting these as attributes during dataflow partitioning
        i_consumer = model.find_consumer(i_tensor_name)
        assert (
            i_consumer.op_type == "StreamingDataflowPartition"
        ), """
            Ensure CreateDataflowPartition called before driver creation."""
        first_df_model = ModelWrapper(getCustomOp(i_consumer).get_nodeattr("model"))
        assert (
            first_df_model.graph.node[0].op_type == "IODMA_hls"
        ), "First partition must hold input IODMA"
        successors = model.find_direct_successors(i_consumer)
        successor_input_num = list(successors[0].input).index(i_consumer.output[0])
        successor_sdp = getCustomOp(successors[0])
        successor_df_model = ModelWrapper(successor_sdp.get_nodeattr("model"))
        first_node = successor_df_model.find_consumer(
            successor_df_model.graph.input[successor_input_num].name
        )
        i_tensor_shape_folded = tuple(getCustomOp(first_node).get_folded_input_shape())
        # generate dummy folded i/o tensors and their packed versions
        i_tensor_dummy_folded = gen_finn_dt_tensor(i_tensor_dt, i_tensor_shape_folded)
        i_tensor_dummy_packed = finnpy_to_packed_bytearray(i_tensor_dummy_folded, i_tensor_dt)
        i_tensor_shape_packed = i_tensor_dummy_packed.shape
        # append all input tensor info to relevant lists
        idt.append("DataType['%s']" % i_tensor_dt.name)
        ishape_normal.append(i_tensor_shape_normal)
        ishape_folded.append(i_tensor_shape_folded)
        ishape_packed.append(i_tensor_shape_packed)
        idma_names.append(getCustomOp(i_consumer).get_nodeattr("instance_name"))

    odt = []
    odma_names = []
    oshape_normal = []
    oshape_folded = []
    oshape_packed = []
    for odma_ind, graph_out in enumerate(model.graph.output):
        o_tensor_name = graph_out.name
        # get inp tensor properties
        o_tensor_dt = model.get_tensor_datatype(o_tensor_name)
        o_tensor_shape_normal = tuple(model.get_tensor_shape(o_tensor_name))
        # go down into IODMA partition to get folded shape info etc
        # TODO consider setting these as attributes during dataflow partitioning
        o_producer = model.find_producer(o_tensor_name)
        assert (
            o_producer.op_type == "StreamingDataflowPartition"
        ), """
            Ensure CreateDataflowPartition called before driver creation."""
        df_model = ModelWrapper(getCustomOp(o_producer).get_nodeattr("model"))
        assert df_model.graph.node[-1].op_type == "IODMA_hls", "Partition must hold output IODMA"
        predecessors = model.find_direct_predecessors(o_producer)
        predecessor_output_num = list(predecessors[0].output).index(o_producer.input[0])
        predecessor_sdp = getCustomOp(predecessors[0])
        predecessor_df_model = ModelWrapper(predecessor_sdp.get_nodeattr("model"))
        last_node = predecessor_df_model.find_producer(
            predecessor_df_model.graph.output[predecessor_output_num].name
        )
        o_tensor_shape_folded = tuple(getCustomOp(last_node).get_folded_output_shape())
        o_tensor_dummy_folded = gen_finn_dt_tensor(o_tensor_dt, o_tensor_shape_folded)
        o_tensor_dummy_packed = finnpy_to_packed_bytearray(o_tensor_dummy_folded, o_tensor_dt)
        o_tensor_shape_packed = o_tensor_dummy_packed.shape
        # append all output tensor info to relevant lists
        odt.append("DataType['%s']" % o_tensor_dt.name)
        oshape_normal.append(o_tensor_shape_normal)
        oshape_folded.append(o_tensor_shape_folded)
        oshape_packed.append(o_tensor_shape_packed)
        odma_names.append(getCustomOp(o_producer).get_nodeattr("instance_name"))

    return {
        "idt": idt,
        "idma_names": idma_names,
        "ishape_normal": ishape_normal,
        "ishape_folded": ishape_folded,
        "ishape_packed": ishape_packed,
        "odt": odt,
        "odma_names": odma_names,
        "oshape_normal": oshape_normal,
        "oshape_folded": oshape_folded,
        "oshape_packed": oshape_packed,
    }
