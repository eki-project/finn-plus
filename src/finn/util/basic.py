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

"""Basic utility functions and classes for FINN.

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

import contextlib
import errno
import os
import signal
import stat as statmod
import subprocess
import tempfile
import time
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import gen_finn_dt_tensor
from typing import TYPE_CHECKING, cast

from finn.util.data_packing import finnpy_to_packed_bytearray
from finn.util.exception import FINNInternalError
from finn.util.logging import log
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from onnx import NodeProto

    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

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


def wait_for_file(
    path: Path,
    timeout: float = 30.0,
    stable_for: float = 0.05,
    interval: float = 0.05,
    expect_dir: bool | None = False,
) -> bool:
    """Wait until path exists and is stable in size/mtime for stable_for seconds.

    If expect_dir is True, only a directory satisfies the check. If False,
    only a regular file satisfies it. If None, either file or directory is ok.
    """
    deadline = time.monotonic() + timeout
    last = None
    last_change = None

    while time.monotonic() < deadline:
        try:
            st = path.stat()
        except (FileNotFoundError, NotADirectoryError):
            time.sleep(interval)
            continue
        except OSError as e:
            # Under high parallelism, avoid failing immediately on transient
            # process/file table exhaustion and retry until timeout.
            if e.errno in (errno.EMFILE, errno.ENFILE):
                time.sleep(interval)
                continue
            raise

        if expect_dir is True and not statmod.S_ISDIR(st.st_mode):
            time.sleep(interval)
            continue
        if expect_dir is False and not statmod.S_ISREG(st.st_mode):
            time.sleep(interval)
            continue

        if stable_for <= 0:
            return True

        cur = (st.st_size, st.st_mtime_ns)
        now = time.monotonic()
        if cur == last:
            if last_change is not None and (now - last_change) >= stable_for:
                return True
        else:
            last = cur
            last_change = now

        time.sleep(interval)

    return False


def wait_for_dir(
    path: Path, timeout: float = 30.0, stable_for: float = 0.05, interval: float = 0.05
) -> bool:
    """Wait until directory exists and is stable in size/mtime for stable_for seconds."""
    return wait_for_file(
        path, timeout=timeout, stable_for=stable_for, interval=interval, expect_dir=True
    )


def getHWCustomOp(node: "NodeProto") -> "HWCustomOp":  # noqa: N802
    """Get the HWCustomOp from a node. Throws an error if the node is not an HWCustomOp."""
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

    n = getCustomOp(node)
    if not isinstance(n, HWCustomOp):
        raise FINNInternalError(f"Node {node.name} is not an HWCustomOp")
    return n


def get_rtlsim_trace_depth() -> int:
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


def get_vivado_root() -> str:
    """Return the root directory that Vivado is installed into."""
    try:
        return os.environ["XILINX_VIVADO"]
    except KeyError:
        raise Exception(
            """Environment variable XILINX_VIVADO must be set
        correctly.
        """
        ) from None


def get_liveness_threshold_cycles() -> int:
    """Return the number of no-output cycles rtlsim will wait before assuming
    the simulation is not finishing and throwing an exception."""
    return int(os.getenv("LIVENESS_THRESHOLD", 1000000))


def make_build_dir(prefix: str = "", return_as_path: bool = False) -> str | Path:
    """Create a folder with given prefix to be used as a build dir.
    Use this function instead of tempfile.mkdtemp to ensure any generated files
    will survive on the host after the FINN Docker container exits."""
    try:
        build_dir = get_settings().finn_build_dir
    except KeyError as keyerror:
        raise Exception("""Environment variable FINN_BUILD_DIR is missing!""") from keyerror

    if not build_dir.exists():
        raise Exception(
            f"FINN_BUILD_DIR at {build_dir} does not exist! Make sure the FINN setup ran properly!"
        )

    tmpdir = Path(tempfile.mkdtemp(prefix=prefix, dir=build_dir))
    if return_as_path:
        return tmpdir
    return str(tmpdir)


class VerboseCalledProcessError(subprocess.CalledProcessError):
    """CalledProcessError that includes captured stdout/stderr in its string representation."""

    def __str__(self) -> str:
        """Return a string representation of the error,
        including captured stdout and stderr if available."""
        base = super().__str__()
        parts = [base]
        if self.output:
            parts.append(f"stdout:\n{self.output.strip()}")
        if self.stderr:
            parts.append(f"stderr:\n{self.stderr.strip()}")
        return "\n".join(parts)


def launch_process_helper(
    args: list[str | Path] | list[str] | list[Path],
    proc_env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    print_stdout: bool = True,
    print_stderr: bool = True,
    timeout: float | None = None,
    start_new_session: bool = False,
    kill_process_group: bool = False,
) -> tuple[str, str]:
    """Launch a helper process in a way that facilitates logging
    stdout/stderr with Python loggers.
    Returns (cmd_out, cmd_err) if successful, raises CalledProcessError otherwise.
    If timeout is set, subprocess.run/communicate may raise TimeoutExpired.
    When kill_process_group is True, the whole process group is
    terminated after completion or timeout cleanup."""
    use_new_session = start_new_session or kill_process_group
    cmd = [str(arg) for arg in args]
    if not use_new_session:
        process = subprocess.run(
            cmd,
            capture_output=True,
            env=proc_env,
            cwd=cwd,
            text=True,
            timeout=timeout,
        )
        cmd_out = process.stdout.strip()
        cmd_err = process.stderr.strip()
        returncode = process.returncode
    else:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=proc_env,
            cwd=cwd,
            text=True,
            start_new_session=True,
        )
        try:
            cmd_out, cmd_err = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if kill_process_group:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    cmd_out, cmd_err = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    cmd_out, cmd_err = process.communicate()
            else:
                process.kill()
                cmd_out, cmd_err = process.communicate()
            raise
        if kill_process_group:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        cmd_out = "" if cmd_out is None else cmd_out.strip()
        cmd_err = "" if cmd_err is None else cmd_err.strip()
        returncode = process.returncode

    # Handle stdout
    if cmd_out:
        if print_stdout is True:
            log.info(cmd_out)
        else:
            # Print with DEBUG level regardless
            log.debug(cmd_out)

    # Handle stderr, depending on return code
    if returncode == 0:
        # Process completed successfully, log stderr only as WARNING
        if cmd_err and print_stderr:
            log.warning(cmd_err)
    else:
        # Process failed, log stderr as ERROR
        if cmd_err:
            log.error(cmd_err)

        # Log additional ERROR message
        cmd = " ".join(str(arg) for arg in args) if isinstance(args, list) else str(args)
        log.error(f"Launched process returned non-zero exit code ({returncode}): {cmd}")

    # Raise CalledProcessError for non-zero return code, including captured output
    if returncode != 0:
        raise VerboseCalledProcessError(
            returncode,
            args,
            output=cmd_out,
            stderr=cmd_err,
        )
    return (cmd_out, cmd_err)


def which(program: str | Path) -> str | Path | None:
    """Python equivalent of the shell cmd 'which'."""

    # source:
    # https://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
    def is_exe(fpath: str | Path) -> bool:
        """Check if a file path points to an executable file.

        Tests whether the given file path exists and has execute permissions.
        This is a helper function used by the which() function.

        Args:
            fpath (str | Path): File path to check for executability.

        Returns:
            bool: True if the file exists and is executable, False otherwise.
        """
        return Path(fpath).is_file() and os.access(fpath, os.X_OK)

    fpath, _fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = Path(path) / program
            if is_exe(exe_file):
                return exe_file

    return None


class CppBuilder:
    """Builds the g++ compiler command to produces the executable of the c++ code
    in code_gen_dir which is passed to the function build() of this class."""

    def __init__(self) -> None:
        """Initialize a new CppBuilder instance.

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
        self.include_paths: list[str] = []
        self.cpp_files: list[str] = []
        self.executable_path: str = ""
        self.code_gen_dir: str = ""
        self.compile_components: list[str] = []
        self.compile_script: Path

    def append_includes(self, library_path: str) -> None:
        """Add given library path to include_paths list."""
        self.include_paths.append(library_path)

    def append_sources(self, cpp_file: str) -> None:
        """Add given c++ file to cpp_files list."""
        self.cpp_files.append(cpp_file)

    def set_executable_path(self, path: str) -> None:
        """Set member variable "executable_path" to given path."""
        self.executable_path = path

    def build(self, code_gen_dir: str) -> None:
        """Build the g++ compiler command according to entries in include_paths
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
        self.compile_script = Path(self.code_gen_dir) / "compile.sh"
        with self.compile_script.open("w") as f:
            f.write("#!/bin/bash \n")
            f.write(bash_compile + "\n")
        bash_command = ["bash", str(self.compile_script)]
        launch_process_helper(bash_command, print_stdout=False)


def is_versal(fpgapart: str) -> bool:
    """Return whether board is part of the Versal family."""
    return fpgapart[0:4] in ["xcvc", "xcve", "xcvp", "xcvm", "xqvc", "xqvm"] or fpgapart[0:5] in [
        "xqrvc",
        "xcv80",
    ]


def get_dsp_block(fpgapart: str) -> str:
    """Determine the DSP block type based on the FPGA part name.

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
    if fpgapart[2] == "7":
        return "DSP48E1"
    return "DSP48E2"


def get_driver_shapes(model: ModelWrapper) -> dict:
    """Get all the IO shapes for the driver."""
    idt = []
    idma_names = []
    ishape_normal = []
    ishape_folded = []
    ishape_packed = []
    for _idma_ind, graph_in in enumerate(model.graph.input):
        i_tensor_name = graph_in.name
        # get inp tensor properties
        i_tensor_dt = model.get_tensor_datatype(i_tensor_name)
        tensor_shape = model.get_tensor_shape(i_tensor_name)
        if tensor_shape is None:
            raise FINNInternalError(
                f"Input tensor {i_tensor_name} has no "
                "shape information when generating driver shapes."
            )
        i_tensor_shape_normal = tuple(tensor_shape)
        # go down into dataflow partition to get folded shape info etc
        # TODO consider setting these as attributes during dataflow partitioning
        i_consumer = model.find_consumer(i_tensor_name)
        if i_consumer is None:
            raise FINNInternalError(
                f"Input tensor {i_tensor_name} has no consumer when generating driver shapes."
            )
        if i_consumer.op_type != "StreamingDataflowPartition":
            raise FINNInternalError("Ensure CreateDataflowPartition called before driver creation.")
        first_df_model = ModelWrapper(cast("str", getCustomOp(i_consumer).get_nodeattr("model")))
        assert (
            first_df_model.graph.node[0].op_type == "IODMA_hls"
        ), "First partition must hold input IODMA"
        successors = model.find_direct_successors(i_consumer)
        if successors is None or len(successors) == 0:
            raise FINNInternalError(
                f"Input tensor {i_tensor_name} has no successor when generating driver shapes."
            )
        successor_input_num = list(successors[0].input).index(i_consumer.output[0])
        successor_sdp = getCustomOp(successors[0])
        successor_df_model = ModelWrapper(cast("str", successor_sdp.get_nodeattr("model")))
        first_node = successor_df_model.find_consumer(
            successor_df_model.graph.input[successor_input_num].name
        )
        if first_node is None:
            raise FINNInternalError(
                f"Input tensor {i_tensor_name} has no consumer in the "
                "dataflow partition when generating driver shapes."
            )
        i_tensor_shape_folded = tuple(
            cast("HWCustomOp", getCustomOp(first_node)).get_folded_input_shape()
        )
        # generate dummy folded i/o tensors and their packed versions
        i_tensor_dummy_folded = gen_finn_dt_tensor(i_tensor_dt, i_tensor_shape_folded)
        i_tensor_dummy_packed = finnpy_to_packed_bytearray(i_tensor_dummy_folded, i_tensor_dt)
        i_tensor_shape_packed = i_tensor_dummy_packed.shape
        # append all input tensor info to relevant lists
        idt.append(f"DataType['{i_tensor_dt.name}']")
        ishape_normal.append(i_tensor_shape_normal)
        ishape_folded.append(i_tensor_shape_folded)
        ishape_packed.append(i_tensor_shape_packed)
        idma_names.append(getCustomOp(i_consumer).get_nodeattr("instance_name"))

    odt = []
    odma_names = []
    oshape_normal = []
    oshape_folded = []
    oshape_packed = []
    for _odma_ind, graph_out in enumerate(model.graph.output):
        o_tensor_name = graph_out.name
        # get inp tensor properties
        o_tensor_dt = model.get_tensor_datatype(o_tensor_name)
        tensor_shape = model.get_tensor_shape(o_tensor_name)
        if tensor_shape is None:
            raise FINNInternalError(
                f"Output tensor {o_tensor_name} has no "
                "shape information when generating driver shapes."
            )
        o_tensor_shape_normal = tuple(tensor_shape)
        # go down into IODMA partition to get folded shape info etc
        # TODO consider setting these as attributes during dataflow partitioning
        o_producer = model.find_producer(o_tensor_name)
        if o_producer is None:
            raise FINNInternalError(
                f"Output tensor {o_tensor_name} has no producer when generating driver shapes."
            )
        if o_producer.op_type != "StreamingDataflowPartition":
            raise FINNInternalError(
                f"Output tensor {o_tensor_name} is not part of a StreamingDataflowPartition."
            )
        df_model = ModelWrapper(cast("str", getCustomOp(o_producer).get_nodeattr("model")))
        assert df_model.graph.node[-1].op_type == "IODMA_hls", "Partition must hold output IODMA"
        predecessors = model.find_direct_predecessors(o_producer)
        if predecessors is None or len(predecessors) == 0:
            raise FINNInternalError(
                f"Output tensor {o_tensor_name} has no predecessor when generating driver shapes."
            )
        predecessor_output_num = list(predecessors[0].output).index(o_producer.input[0])
        predecessor_sdp = getCustomOp(predecessors[0])
        predecessor_df_model = ModelWrapper(cast("str", predecessor_sdp.get_nodeattr("model")))
        last_node = predecessor_df_model.find_producer(
            predecessor_df_model.graph.output[predecessor_output_num].name
        )
        if last_node is None:
            raise FINNInternalError(
                f"Output tensor {o_tensor_name} has no producer in the "
                "dataflow partition when generating driver shapes."
            )
        o_tensor_shape_folded = tuple(
            cast("HWCustomOp", getCustomOp(last_node)).get_folded_output_shape()
        )
        o_tensor_dummy_folded = gen_finn_dt_tensor(o_tensor_dt, o_tensor_shape_folded)
        o_tensor_dummy_packed = finnpy_to_packed_bytearray(o_tensor_dummy_folded, o_tensor_dt)
        o_tensor_shape_packed = o_tensor_dummy_packed.shape
        # append all output tensor info to relevant lists
        odt.append(f"DataType['{o_tensor_dt.name}']")
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
