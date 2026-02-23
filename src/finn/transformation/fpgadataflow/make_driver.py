"""Create C++ and PYNQ drivers for FINN-generated accelerators."""

# Copyright (c) 2020, Xilinx
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

import json
import multiprocessing
import numpy as np
import os
import shlex
import shutil
import subprocess
import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from string import Template
from typing import Dict, List, Optional, Tuple

import finn.util
from finn.builder.build_dataflow_config import FpgaMemoryType
from finn.templates import get_templates_folder
from finn.util.basic import get_driver_shapes, make_build_dir
from finn.util.data_packing import to_external_tensor
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log


def update_bitfile_path_after_copy(bitfile_path: str, json_path: str) -> None:
    """
    Update the xclbinPath in the JSON configuration to point to the new bitfile location.

    Args:
        json_path (str): Path to the JSON configuration file
        bitfile_path (str): New path to the bitfile (.xclbin)
    """
    if json_path is None or not os.path.exists(json_path):
        raise FINNInternalError("JSON configuration file does not exist or is not specified.")
    if bitfile_path is None or not os.path.exists(bitfile_path):
        raise FINNInternalError("Bitfile path does not exist or is not specified.")
    if not json_path.endswith(".json"):
        raise FINNInternalError("Provided path is not a JSON file.")

    # Read the current JSON configuration
    with open(json_path, "r") as f:
        data = json.load(f)

    # Update the xclbinPath for each device in the configuration
    for device_config in data:
        device_config["xclbinPath"] = os.path.abspath(bitfile_path)

    # Write the updated configuration back to the file
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


class MakeCPPDriver(Transformation):
    """Create CPP code to correctly interface the generated
    accelerator, including data packing/unpacking. Should be called
    after conversion to HLS layers, folding and the creation of
    dataflow partitions for correct operation.
    platform: has to be "alveo", otherwise an error is thrown
    Outcome if successful: sets the cpp_driver_dir attribute in the ONNX
    ModelProto's metadata_props field, with the created driver dir as the
    value.
    runtime writeable weights not yet supported.
    """

    # TODO: Enable multiple input types! Now only assumes the first one
    def resolve_dt_name(s: str) -> str:
        """Resolve datatype name for C++ driver code generation.

        Args:
            s: Datatype string to resolve

        Returns:
            Resolved C++ datatype name

        Raises:
            FINNInternalError: If datatype is unknown
        """
        s = s.replace("DataType[", "").replace("]", "")
        if s in ["BINARY", "TERNARY", "BIPOLAR"]:
            return "Datatype" + s[0] + s[1:].lower()
        elif s.startswith("U"):
            return "DatatypeUint<" + s.replace("UINT", "") + ">"
        elif s.startswith("I"):
            return "DatatypeInt<" + s.replace("INT", "") + ">"
        elif "FLOAT" in s:
            return "DatatypeFloat<" + s.replace("FLOAT", "") + ">"
        elif "FIXED" in s:
            return "DatatypeFixed" + s.replace("FIXED", "")
        else:
            raise FINNInternalError(f"Unknown datatype for C++ Driver:{s}")

    def __init__(
        self,
        platform: str,
        version: str,
        host_mem: str,
    ):
        """Initialize MakeCPPDriver transformation.

        Args:
            platform: Target platform (must be "alveo")
            version: Version of finn-cpp-driver to use ("latest" or commit hash)
            host_mem: Memory type (FpgaMemoryType.HOST_MEM or FpgaMemoryType.DEVICE_MEM)

        Raises:
            FINNUserError: If platform is not "alveo"
        """
        super().__init__()
        self.platform: str = platform

        if platform != "alveo":
            raise FINNUserError(
                "CPP driver only supported for Alveo devices, please use PYNQ driver instead."
            )
        self.version = version

        # Define variables for the repository URL and commit hash
        self.repository_url = "https://github.com/eki-project/finn-cpp-driver.git"
        if version == "latest" or version is None:
            self.commit_hash = "HEAD"
        else:
            self.commit_hash = version

        if host_mem == FpgaMemoryType.HOST_MEM:
            self.host_memory = True
        else:
            self.host_memory = False

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        """Apply the MakeCPPDriver transformation to generate C++ driver code.

        Args:
            model: ONNX model wrapper

        Returns:
            Tuple of (modified model, transformation success flag)
        """
        driver_shapes: Dict = get_driver_shapes(model)
        ext_weight_dma_cnt: int  # noqa
        weights_dir: str  # noqa
        # TODO: Enable weight file generation
        # ext_weight_dma_cnt, weights_dir = write_weights(model, cpp_driver_dir)

        # Create a temporary directory for the generated C++ driver code
        cpp_driver_dir = make_build_dir(prefix="cpp_driver_")
        # Store the driver directory path in model metadata
        model.set_metadata_prop("cpp_driver_dir", cpp_driver_dir)
        # Get the path to the FPGA bitstream from model metadata
        xclbin_path = model.get_metadata_prop("bitfile_output")
        # Define paths for configuration files
        json_path = os.path.join(cpp_driver_dir, "acceleratorconfig.json")
        header_path = os.path.join(cpp_driver_dir, "AcceleratorDatatypes.h")

        def run_command(command, cwd=None, debug=False):
            """Execute a shell command with error handling.

            Args:
                command: Shell command string to execute.
                cwd: Working directory for command execution. Defaults to None (current directory).
                debug: If True, print command output for debugging. Defaults to False.

            Raises:
                subprocess.CalledProcessError: If the command returns a non-zero exit code.
            """
            try:
                result = subprocess.run(
                    shlex.split(command), cwd=cwd, check=True, text=True, capture_output=True
                )
                if debug:
                    # Print the output for debugging purposes
                    print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {command}")
                print(f"Output:{e.stdout}; Error:{e.stderr}")
                raise e

        # Clone and set up the C++ driver repository
        log.info("Downloading C++ driver template...")
        # Initialize git repo and fetch specified version
        run_command("git init", cwd=cpp_driver_dir)
        run_command(f"git remote add origin {self.repository_url}", cwd=cpp_driver_dir)
        run_command(f"git fetch origin {self.commit_hash} --depth=1", cwd=cpp_driver_dir)
        run_command("git checkout FETCH_HEAD", cwd=cpp_driver_dir)
        # Initialize and update all git submodules
        run_command("git submodule update --init --recursive", cwd=cpp_driver_dir)

        log.info("Generating template files...")
        # Check if multiple different input/output types are used.
        if len(set(driver_shapes["idt"])) > 1 or len(set(driver_shapes["odt"])) > 1:
            raise RuntimeError(
                "Multiple different input/output types for the C++ driver\
                    are currently not supported."
            )

        # * Writing the header file
        inputDatatype: str = MakeCPPDriver.resolve_dt_name(
            driver_shapes["idt"][0].replace("'", "")
        )  # .get_canonical_name())
        outputDatatype: str = MakeCPPDriver.resolve_dt_name(
            driver_shapes["odt"][0].replace("'", "")
        )  # .get_canonical_name())
        with open(
            os.path.join(
                cpp_driver_dir, "src", "FINNCppDriver", "config", "FinnDriverUsedDatatypes.h.in"
            ),
            "r",
        ) as f_in:
            header = f_in.read()
            template_handler = Template(header)
            templated_str = template_handler.substitute(
                inputDatatype=inputDatatype, outputDatatype=outputDatatype
            )
            with open(header_path, "w+") as f:
                f.write(templated_str)

        # * Writing the json file
        # TODO: Update this for multi-fpga usage (more than one device!)
        # Path of the xclbin in the finn compiler project
        # Get kernel names using xclbinutil

        if shutil.which("xclbinutil") is None:
            raise RuntimeError(
                "xclbinutil not in PATH or not installed.\
                Required to read kernel names for driver config!"
            )

        # Extract IP layout information from the FPGA bitstream
        # Use xclbinutil to dump the IP layout section from the bitstream to a JSON file
        run_command(
            f"xclbinutil -i {xclbin_path} --dump-section IP_LAYOUT:JSON:ip_layout.json --force",
            cwd=os.path.dirname(xclbin_path),
        )
        # Load the IP layout information from the generated JSON file
        ips = None
        with open(os.path.join(os.path.dirname(xclbin_path), "ip_layout.json")) as f:
            ips = json.loads(f.read())["ip_layout"]["m_ip_data"]

        # Define a filter function to identify input/output DMA kernels
        # Filters for kernels that have valid base addresses
        # and contain "idma" or "odma" in their names
        isIO = (
            lambda x: x["m_type"] == "IP_KERNEL"
            and x["m_base_address"] != "not_used"
            and ("idma" in x["m_name"] or "odma" in x["m_name"])
        )
        # Extract lists of input and output DMA kernel names
        idmas = [x["m_name"] for x in ips if isIO(x) and "idma" in x["m_name"]]
        odmas = [x["m_name"] for x in ips if isIO(x) and "odma" in x["m_name"]]

        def formatKernelName(kname: str):
            """Format kernel name into Vitis-compatible format.

            Args:
                kname: Kernel name string in "name:instance" format.

            Returns:
                Formatted kernel name as "name:{instance}".
            """
            kparts = kname.split(":")
            return kparts[0] + ":{" + kparts[1] + "}"

        # Create JSON configuration entries for input and output DMAs
        jsonIdmas = []
        jsonOdmas = []
        # Map driver's idma names to actual kernels and include shape information
        for i in range(len(driver_shapes["idma_names"])):
            jsonIdmas.append(
                {
                    "kernelName": [
                        formatKernelName(name)
                        for name in idmas
                        if driver_shapes["idma_names"][i] in name
                    ][0],
                    "normalShape": driver_shapes["ishape_normal"][i],
                    "foldedShape": driver_shapes["ishape_folded"][i],
                    "packedShape": driver_shapes["ishape_packed"][i],
                }
            )
        # Map driver's odma names to actual kernels and include shape information
        for i in range(len(driver_shapes["odma_names"])):
            jsonOdmas.append(
                {
                    "kernelName": [
                        formatKernelName(name)
                        for name in odmas
                        if driver_shapes["odma_names"][i] in name
                    ][0],
                    "normalShape": driver_shapes["oshape_normal"][i],
                    "foldedShape": driver_shapes["oshape_folded"][i],
                    "packedShape": driver_shapes["oshape_packed"][i],
                }
            )

        # Create the final JSON configuration structure
        data = []
        data.append(
            {
                # Specify which XRT device to use (0 = first device)
                "xrtDeviceIndex": 0,
                # Store the absolute path to the bitstream
                "xclbinPath": os.path.abspath(xclbin_path),
                "name": "MainDevice",  # Assign a name to this device configuration
                "idmas": jsonIdmas,  # Include the input DMA configurations
                "odmas": jsonOdmas,  # Include the output DMA configurations
            }
        )
        # Write the complete configuration to the JSON file
        with open(json_path, "w+") as f:
            f.write(json.dumps(data, indent=4))

        log.info("Created runtime json config file")

        def configure_cmake(
            source_dir: str,  # Directory containing CMakeLists.txt
            build_dir: str,  # Directory where build files will be generated
            # Additional CMake arguments as string
            cmake_args: Optional[str] = None,
            # Command to invoke CMake
            cmake_executable: str = f"{sys.executable} -m cmake",
        ):
            """Configure CMake build system for the C++ driver.

            Args:
                source_dir: Directory containing the CMakeLists.txt file.
                build_dir: Directory where CMake build files will be generated.
                cmake_args: Additional CMake configuration arguments. Defaults to None.
                cmake_executable: Command to invoke CMake. Defaults to Python's cmake module.
            """
            # Create build directory if it doesn't exist
            os.makedirs(build_dir, exist_ok=True)
            # Split the cmake executable command into arguments
            args = shlex.split(cmake_executable)
            # Add any additional CMake arguments if provided
            if cmake_args:
                cmake_args = shlex.split(cmake_args)
                args.extend(cmake_args)
            # Set CMake policy version to ensure compatibility
            # Needed because CMake 4.0.2 is installed by FINN+ and set minimum version
            # requirements are not correctly picked up by CMake
            args.append("-DCMAKE_POLICY_VERSION_MINIMUM=3.5")
            args.append(os.path.abspath(source_dir))
            log.info(f"Configuring with: {' '.join(args)}")
            result = subprocess.run(args, cwd=build_dir, capture_output=True, text=True)
            if result.returncode != 0:
                log.critical(f"Configure failed with error:\n{result.stderr}")
                raise subprocess.CalledProcessError(
                    result.returncode, args, result.stdout, result.stderr
                )

        def build_cmake(
            build_dir: str,  # Directory containing the configured build files
            # Build tool to use (default: make)
            cmake_executable: str = "make",
            # Specific target to build (if any)
            build_target: Optional[str] = None,
            # Additional build arguments
            build_args: Optional[List[str]] = None,
        ):
            """Build the configured CMake project.

            Args:
                build_dir: Directory containing the configured build files.
                cmake_executable: Build tool to use. Defaults to "make".
                build_target: Specific target to build. Defaults to None (builds all).
                build_args: Additional build arguments. Defaults to None.

            Raises:
                subprocess.CalledProcessError: If the build fails.
            """
            # Prepare the build command with the executable
            args = [cmake_executable]
            # Add optional build target if specified
            if build_target:
                args += [build_target]
            # Add any additional build arguments
            if build_args:
                args.extend(build_args)
            log.info(f"Building with:{' '.join(args)}")
            # Execute the build command
            result = subprocess.run(args, cwd=build_dir, capture_output=True, text=True)
            # Handle build failures
            if result.returncode != 0:
                log.critical(f"Build failed with error:\n{result.stderr}")
                raise subprocess.CalledProcessError(
                    result.returncode, args, result.stdout, result.stderr
                )

        host_memory_usage = "ON" if self.host_memory else "OFF"

        # Define CMake configuration options for the driver build
        # - Release build type for optimized performance
        # - Disable sanitizers for production builds
        # - Set custom header location
        # - Disable documentation generation
        # - Enable/Disable host memory usage
        cmake_args = f"-DCMAKE_BUILD_TYPE=Release -DFINN_ENABLE_SANITIZERS=Off\
        -DFINN_HEADER_LOCATION={os.path.abspath(header_path)} -DFINN_BUILD_DOC=Off\
            -DFINN_USE_HOST_MEM={host_memory_usage}"

        # Configure the CMake project
        configure_cmake(
            source_dir=cpp_driver_dir,
            build_dir=os.path.join(cpp_driver_dir, "build"),
            cmake_args=cmake_args,
        )
        # Determine optimal number of build threads based on CPU cores
        num_cores = multiprocessing.cpu_count()
        build_cmake(
            build_dir=os.path.join(cpp_driver_dir, "build"), build_args=["-j", str(num_cores)]
        )

        def check_finn_types(bin_dir: str, expectedInputType: str, expectedOutputType: str) -> None:
            """Verify that compiled driver's datatypes match expected types.

            Args:
                bin_dir: Directory containing the finnhpc executable.
                expectedInputType: Expected input datatype string.
                expectedOutputType: Expected output datatype string.

            Raises:
                subprocess.CalledProcessError: If the datatype check command fails.
                RuntimeError: If the actual datatypes don't match expected types.
            """
            # Run the built finnhpc executable with the --check flag to output datatype information
            result = subprocess.run(
                "./finnhpc --check".split(), cwd=bin_dir, capture_output=True, text=True
            )
            if result.returncode != 0:
                log.critical(f"Running datatype check failed with error:\n{result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, result.stdout, result.stderr)
            output = result.stdout
            output_lines = output.splitlines()

            # Verify that the compiled driver's datatypes match the expected types
            # First line contains input type, second line contains output type
            if (
                expectedInputType not in output_lines[0]
                or expectedOutputType not in output_lines[1]
            ):
                log.error(
                    f"FINN types check failed. Expected Types: {expectedInputType},\
                        {expectedOutputType}"
                )
                log.error(f"                           Actual Types: {output}")
                raise FINNInternalError(
                    "Expected C++ driver types to match\
                    expected types."
                )

        # Make the compiled finnhpc executable file executable (chmod +x)
        os.chmod(os.path.join(cpp_driver_dir, "build", "bin", "finnhpc"), 0o755)

        # Verify that the driver was compiled with the correct datatypes
        check_finn_types(
            bin_dir=os.path.join(cpp_driver_dir, "build", "bin"),
            expectedInputType=inputDatatype,
            expectedOutputType=outputDatatype,
        )

        # TODO: Generating weight files
        # weights_dir = output_dir + "/runtime_weights"

        # os.makedirs(weights_dir)
        # idma_idx = 0
        # ext_weight_dma_cnt = 0

        # for node in model.graph.node:
        #     assert (
        #         node.op_type == "StreamingDataflowPartition"
        #     ), "CreateDataflowPartition needs to be applied before driver generation"

        #     if len(node.input) > 0:
        #         producer = model.find_producer(node.input[0])
        #         init_tensor = model.get_initializer(node.input[0])
        #     else:
        #         producer = None
        #         init_tensor = None

        #     if producer is None:  # input dma?
        #         sdp_inst = getCustomOp(node)
        #         idma_name = sdp_inst.get_nodeattr("instance_name")
        #         df_model = ModelWrapper(sdp_inst.get_nodeattr("model"))
        #         assert df_model.graph.node[0].op_type == "IODMA"
        #         iodma_node = getCustomOp(df_model.graph.node[0])
        #         if iodma_node.get_nodeattr("burstMode") == "wrap":  # input weights dma?
        #             init_tensor = df_model.get_initializer(iodma_node.onnx_node.input[0])
        #             ext_weight_dma_cnt += 1
        #             w_dtype = df_model.get_tensor_datatype(iodma_node.onnx_node.input[0])
        #             init_external_tensor = to_external_tensor(init_tensor, w_dtype)
        #             np.save(weights_dir + "/" + idma_name + ".npy", init_external_tensor)
        #         idma_idx += 1

        return (model, False)


class MakePYNQDriver(Transformation):
    """Create PYNQ Python code to correctly interface the generated
    accelerator, including data packing/unpacking. Should be called
    after conversion to HLS layers, folding and the creation of
    dataflow partitions for correct operation.

    platform: one of ["zynq-iodma", "alveo"]

    Outcome if successful: sets the pynq_driver_dir attribute in the ONNX
    ModelProto's metadata_props field, with the created driver dir as the
    value. If any layers use runtime-writable parameters, those will be gathered
    under the runtime_weights/ subfolder of the pynq_driver_dir.
    """

    def __init__(
        self,
        platform,
        driver_type,
        clk_period_ns=None,
        validation_datset=None,
        experiment_info=None,
        board=None,
    ):
        """Initialize PYNQ driver generation.

        Args:
            platform: Target platform, one of ["zynq-iodma", "alveo"].
            driver_type: Type/name of the driver to generate
                (e.g. "FINNDMAOverlay", "FINNDMAInstrumentationOverlay").
            clk_period_ns: Clock period in nanoseconds used for performance calculations.
            validation_datset: Validation dataset path or identifier.
            experiment_info: Path to a JSON file containing experiment metadata.
        """
        super().__init__()
        self.platform = platform
        self.driver_type = driver_type
        self.clk_period_ns = clk_period_ns
        self.validation_datset = validation_datset
        self.experiment_info = experiment_info
        self.board = board

    def _generate_driver_files(self, model):
        """Generate PYNQ driver base files."""
        # create a temporary folder for the generated driver
        pynq_driver_dir = make_build_dir(prefix="pynq_driver_")
        model.set_metadata_prop("pynq_driver_dir", pynq_driver_dir)

        # create the FINN driver
        driver_base_template = get_templates_folder() / "python_driver/driver.py"

        driver_base_py = pynq_driver_dir + "/driver.py"

        shutil.copy(driver_base_template, driver_base_py)

        # Copy validate scripts
        validate_base_template = get_templates_folder() / "validate"
        validate_target_path = pynq_driver_dir + "/validate"
        shutil.copytree(validate_base_template, validate_target_path)

        # TODO: Can we do this without packaging data_packing.py this way?
        finn_target_path = pynq_driver_dir + "/finn"
        os.makedirs(finn_target_path + "/util", exist_ok=True)
        finn_util_path = finn.util.__path__[0]
        files_to_copy = []
        files_to_copy.append(
            (
                finn_util_path + "/data_packing.py",
                finn_target_path + "/util/data_packing.py",
            )
        )
        files_to_copy.append(
            (
                finn_util_path + "/__init__.py",
                finn_target_path + "/util/__init__.py",
            )
        )
        for src_file, target_file in files_to_copy:
            shutil.copy(src_file, target_file)

    def _generate_weight_files(self, model):
        """Generate weight files for external and runtime-writable weights."""
        pynq_driver_dir = model.get_metadata_prop("pynq_driver_dir")

        external_weights = False
        runtime_weights = False

        # TODO: Check weights generation
        # generate external weights npy files
        weights_dir = pynq_driver_dir + "/runtime_weights"

        os.makedirs(weights_dir)
        idma_idx = 0
        ext_weight_dma_cnt = 0
        ext_weight_shapes_dict = {}

        for node in model.graph.node:
            assert (
                node.op_type == "StreamingDataflowPartition"
            ), "CreateDataflowPartition needs to be applied before driver generation"

            if len(node.input) > 0:
                producer = model.find_producer(node.input[0])
                init_tensor = model.get_initializer(node.input[0])
            else:
                producer = None
                init_tensor = None

            if producer is None:  # input dma?
                sdp_inst = getCustomOp(node)
                idma_name = sdp_inst.get_nodeattr("instance_name")
                df_model = ModelWrapper(sdp_inst.get_nodeattr("model"))
                assert df_model.graph.node[0].op_type == "IODMA_hls"
                iodma_node = getCustomOp(df_model.graph.node[0])
                if iodma_node.get_nodeattr("burstMode") == "wrap":  # input weights dma?
                    external_weights = True
                    dma_sdp_output = sdp_inst.onnx_node.output[0]
                    dma_target_sdp = getCustomOp(model.find_consumer(dma_sdp_output))
                    dma_target_model = ModelWrapper(dma_target_sdp.get_nodeattr("model"))
                    iodma_output_tensor = iodma_node.onnx_node.output[0]
                    dma_consumer = dma_target_model.find_consumer(iodma_output_tensor)
                    ext_weight_shapes_dict[idma_name] = dma_target_model.get_tensor_shape(
                        dma_consumer.output[0]
                    )
                    init_tensor = df_model.get_initializer(iodma_node.onnx_node.input[0])
                    ext_weight_dma_cnt += 1
                    w_dtype = df_model.get_tensor_datatype(iodma_node.onnx_node.input[0])
                    init_external_tensor = to_external_tensor(init_tensor, w_dtype)
                    np.save(weights_dir + "/" + idma_name + ".npy", init_external_tensor)
                idma_idx += 1

        external_weights_dict = {
            "external_weights": external_weights,
            "number_of_external_weights": str(ext_weight_dma_cnt),
            "external_weights_input_shapes": ext_weight_shapes_dict,
        }

        # generate weight files for runtime-writable layers
        # TODO verify
        for sdp_ind, sdp_node in enumerate(model.graph.node):
            assert sdp_node.op_type == "StreamingDataflowPartition"
            # get dataflow model
            sdp_node = getCustomOp(sdp_node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            dataflow_model = ModelWrapper(dataflow_model_filename)
            rt_layer_ind = 0
            for node in dataflow_model.graph.node:
                if node.op_type.startswith("MVAU") or node.op_type.startswith("Thresholding"):
                    node_inst = getCustomOp(node)
                    is_rt_weights = node_inst.get_nodeattr("runtime_writeable_weights")
                    if is_rt_weights == 1:
                        runtime_weights = True
                        fcl_w = dataflow_model.get_initializer(node.input[1])
                        w_filename = weights_dir + "/%d_%d_%s.dat" % (
                            sdp_ind,
                            rt_layer_ind,
                            node.name,
                        )
                        node_inst.make_weight_file(fcl_w, "decoupled_runtime", w_filename)
                        rt_layer_ind += 1
                elif node.op_type == "StreamingDataflowPartition":
                    log.warning(
                        """Nested StreamingDataflowPartition are not supported
                    """
                    )
                else:
                    continue

        if (not external_weights) and (not runtime_weights):
            os.removedirs(weights_dir)

        return external_weights_dict, runtime_weights

    def _write_fifo_widths(self, model):
        """Export FIFO widths to the settings file as well.
        At this stage, the FIFOs are already wrapped in StreamingDataflowPartitions."""
        settings = {}
        fifo_widths = {}
        for sdp_node in model.get_nodes_by_op_type("StreamingDataflowPartition"):
            sdp_node_inst = getCustomOp(sdp_node)
            dataflow_model_filename = sdp_node_inst.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)
            for node in kernel_model.graph.node:
                if node.op_type.startswith("StreamingFIFO"):
                    node_inst = getCustomOp(node)
                    # JSON doesn't support int keys
                    fifo_id = str(node_inst.get_nodeattr("fifo_id"))
                    fifo_widths[fifo_id] = node_inst.get_instream_width()
        settings["fifo_widths"] = fifo_widths
        # export original folding config to settings file,
        # so that the driver can generate a final cfg with live fifo sizes applied
        folding_path = model.get_metadata_prop("folding_config_before_lfs")
        if folding_path:
            with open(folding_path, "r") as f:
                folding_cfg = json.load(f)
            settings["folding_config_before_lfs"] = folding_cfg

        return settings

    def apply(self, model):
        """Apply the MakePYNQDriver transformation.

        Creates a PYNQ Python driver package for interfacing with the generated
        accelerator, including data packing/unpacking and runtime weight handling.

        Args:
            model: The ONNX model to generate a driver for.

        Returns:
            Tuple of (modified model, False) indicating transformation applied.
        """
        # TODO: support Alveo and Versal platforms (instr)

        self._generate_driver_files(model)

        driver_information = {}

        experiment_information = {}
        if self.experiment_info is not None:
            with open(self.experiment_info, "r") as f:
                experiment_information = json.load(f)

        driver_information["driver_type"] = self.driver_type
        if self.driver_type in ["FINNDMAOverlay", "FINNDMAInstrumentationOverlay"]:
            external_weights_dict, runtime_weights = self._generate_weight_files(model)
            driver_shapes: Dict = get_driver_shapes(model)
            driver_information["io_shape_dict"] = driver_shapes
            driver_information["io_shape_dict"]["num_inputs"] = len(driver_shapes["idma_names"])
            driver_information["io_shape_dict"]["num_outputs"] = len(driver_shapes["odma_names"])
            driver_information["io_shape_dict"]["num_outputs"] = len(driver_shapes["odma_names"])
            driver_information["io_shape_dict"][
                "number_of_external_weights"
            ] = external_weights_dict["number_of_external_weights"]
            driver_information["io_shape_dict"][
                "external_weights_input_shapes"
            ] = external_weights_dict["external_weights_input_shapes"]
            driver_information["external_weights"] = external_weights_dict["external_weights"]
            driver_information["runtime_weights"] = runtime_weights
            driver_information["platform"] = self.platform
            # TODO: also supply ext_weight_shapes_dict to driver

        if self.clk_period_ns is not None:
            driver_information["fclk_mhz"] = (1.0 / self.clk_period_ns) * 1e3

        if self.driver_type == "FINNLiveFIFOOverlay":
            driver_information.update(self._write_fifo_widths(model))

        if self.validation_datset is not None:
            driver_information["validation_dataset"] = self.validation_datset

        if "global" in experiment_information:
            if self.board is not None and "board" not in experiment_information["global"]["PAF"]:
                experiment_information["global"]["PAF"]["board"] = self.board

        settings = {
            "driver_information": driver_information,
            "experiment_information": experiment_information,
        }
        pynq_driver_dir = model.get_metadata_prop("pynq_driver_dir")
        settingsfile = pynq_driver_dir + "/settings.json"
        with open(settingsfile, "w") as f:
            json.dump(settings, f, indent=2)

        return (model, False)
