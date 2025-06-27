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
import qonnx
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
from finn.util.basic import make_build_dir
from finn.util.data_packing import get_driver_shapes, to_external_tensor
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log

from . import template_driver


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

        # Helper function to execute shell commands safely with error handling
        def run_command(command, cwd=None, debug=False):
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
        # Build required dependencies
        run_command("./buildDependencies.sh", cwd=cpp_driver_dir)

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

        # Helper function to format kernel names for the driver configuration
        # Converts "kernelName:instance" to "kernelName:{instance}" format
        def formatKernelName(kname: str):
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

        # Helper function to configure CMake build system
        def configure_cmake(
            source_dir: str,  # Directory containing CMakeLists.txt
            build_dir: str,  # Directory where build files will be generated
            # Additional CMake arguments as string
            cmake_args: Optional[str] = None,
            # Command to invoke CMake
            cmake_executable: str = f"{sys.executable} -m cmake",
        ):
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

        # Helper function to build the configured CMake project
        def build_cmake(
            build_dir: str,  # Directory containing the configured build files
            # Build tool to use (default: make)
            cmake_executable: str = "make",
            # Specific target to build (if any)
            build_target: Optional[str] = None,
            # Additional build arguments
            build_args: Optional[List[str]] = None,
        ):
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

        # Helper function to verify that the built driver uses the correct datatypes
        def check_finn_types(bin_dir: str, expectedInputType: str, expectedOutputType: str) -> None:
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


class MakePYNQDriverIODMA(Transformation):
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

    def __init__(self, platform, validation_datset=None):
        super().__init__()
        self.platform = platform
        self.validation_datset = validation_datset

    def apply(self, model):
        # create a temporary folder for the generated driver
        pynq_driver_dir = make_build_dir(prefix="pynq_driver_")
        model.set_metadata_prop("pynq_driver_dir", pynq_driver_dir)

        # create the base FINN driver -- same for all accels
        driver_base_template = os.path.join(
            os.environ["FINN_QNN_DATA"], "templates/driver/driver_base.py"
        )
        driver_base_py = pynq_driver_dir + "/driver_base.py"
        shutil.copy(driver_base_template, driver_base_py)
        # driver depends on qonnx and finn packages
        # extract individual source files and copy to driver folder
        qonnx_target_path = pynq_driver_dir + "/qonnx"
        finn_target_path = pynq_driver_dir + "/finn"
        os.makedirs(qonnx_target_path + "/core", exist_ok=True)
        os.makedirs(qonnx_target_path + "/util", exist_ok=True)
        os.makedirs(finn_target_path + "/util", exist_ok=True)
        qonnx_path = qonnx.__path__[0]
        finn_util_path = finn.util.__path__[0]
        files_to_copy = []
        files_to_copy.append(
            (qonnx_path + "/core/datatype.py", qonnx_target_path + "/core/datatype.py")
        )
        files_to_copy.append(
            (qonnx_path + "/core/__init__.py", qonnx_target_path + "/core/__init__.py")
        )
        files_to_copy.append((qonnx_path + "/util/basic.py", qonnx_target_path + "/util/basic.py"))
        files_to_copy.append(
            (qonnx_path + "/util/__init__.py", qonnx_target_path + "/util/__init__.py")
        )
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

        driver_shapes: Dict = get_driver_shapes(model)

        # generate external weights npy files
        weights_dir = pynq_driver_dir + "/runtime_weights"

        os.makedirs(weights_dir)
        idma_idx = 0
        ext_weight_dma_cnt = 0

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
                # input weights dma?
                if iodma_node.get_nodeattr("burstMode") == "wrap":
                    init_tensor = df_model.get_initializer(iodma_node.onnx_node.input[0])
                    ext_weight_dma_cnt += 1
                    w_dtype = df_model.get_tensor_datatype(iodma_node.onnx_node.input[0])
                    init_external_tensor = to_external_tensor(init_tensor, w_dtype)
                    np.save(weights_dir + "/" + idma_name + ".npy", init_external_tensor)
                idma_idx += 1

        # fill in the driver template
        driver_py = pynq_driver_dir + "/driver.py"
        driver = template_driver.pynq_driver_template

        driver = driver.replace("$PLATFORM$", self.platform)
        driver = driver.replace("$INPUT_FINN_DATATYPE$", str(driver_shapes["idt"]).replace('"', ""))
        driver = driver.replace("$INPUT_SHAPE_NORMAL$", str(driver_shapes["ishape_normal"]))
        driver = driver.replace("$INPUT_SHAPE_FOLDED$", str(driver_shapes["ishape_folded"]))
        driver = driver.replace("$INPUT_SHAPE_PACKED$", str(driver_shapes["ishape_packed"]))
        driver = driver.replace(
            "$OUTPUT_FINN_DATATYPE$", str(driver_shapes["odt"]).replace('"', "")
        )
        driver = driver.replace("$OUTPUT_SHAPE_NORMAL$", str(driver_shapes["oshape_normal"]))
        driver = driver.replace("$OUTPUT_SHAPE_FOLDED$", str(driver_shapes["oshape_folded"]))
        driver = driver.replace("$OUTPUT_SHAPE_PACKED$", str(driver_shapes["oshape_packed"]))
        driver = driver.replace("$INPUT_DMA_NAME$", "%s" % str(driver_shapes["idma_names"]))
        driver = driver.replace("$OUTPUT_DMA_NAME$", "%s" % str(driver_shapes["odma_names"]))
        driver = driver.replace("$NUM_INPUTS$", str(len(driver_shapes["idma_names"])))
        driver = driver.replace("$NUM_OUTPUTS$", str(len(driver_shapes["odma_names"])))
        driver = driver.replace("$EXT_WEIGHT_NUM$", str(ext_weight_dma_cnt))

        with open(driver_py, "w") as f:
            f.write(driver)

        # add validate.py to run full top-1 test (only for suitable networks)
        validate_py = pynq_driver_dir + "/validate.py"
        validate_template = os.path.join(
            os.environ["FINN_QNN_DATA"], "templates/driver/validate.py"
        )
        shutil.copy(validate_template, validate_py)

        # generate settings.json for generated driver
        if self.validation_datset is not None:
            settings = {
                "validation_dataset": self.validation_datset,
            }
            settingsfile = pynq_driver_dir + "/settings.json"
            with open(settingsfile, "w") as f:
                json.dump(settings, f, indent=2)

        # generate weight files for runtime-writable layers
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

        return (model, False)


class MakePYNQDriverInstrumentation(Transformation):
    def __init__(self, platform, clk_period_ns, live_fifo_sizing):
        super().__init__()
        self.platform = platform
        self.clk_period_ns = clk_period_ns
        self.live_fifo_sizing = live_fifo_sizing

    def apply(self, model: ModelWrapper):
        # TODO: support runtime-writable and external weights
        # TODO: support Alveo and Versal platforms

        # create a temporary folder for the generated driver
        pynq_driver_dir = make_build_dir(prefix="pynq_driver_")
        model.set_metadata_prop("pynq_driver_dir", pynq_driver_dir)

        # create (copy) the static instrumentation driver
        driver_template = (
            os.environ["FINN_QNN_DATA"] + "/templates/driver/driver_instrumentation.py"
        )
        if self.live_fifo_sizing:
            driver_py = pynq_driver_dir + "/driver_instrumentation.py"
        else:
            driver_py = pynq_driver_dir + "/driver.py"
        shutil.copy(driver_template, driver_py)

        # add-on driver for live fifosizing
        if self.live_fifo_sizing:
            driver_template = os.environ["FINN_QNN_DATA"] + "/templates/driver/driver_fifosizing.py"
            driver_py = pynq_driver_dir + "/driver.py"
            shutil.copy(driver_template, driver_py)

        # write default settings to driver config file
        settings = {
            "fclk_mhz": (1.0 / self.clk_period_ns) * 1e3,
        }
        if self.live_fifo_sizing:
            # export FIFO widths to the settings file as well
            # at this stage, the FIFOs are already wrapped in StreamingDataflowPartitions
            fifo_widths = {}
            for sdp_node in model.get_nodes_by_op_type("StreamingDataflowPartition"):
                sdp_node_inst = getCustomOp(sdp_node)
                # JSON doesn't support int keys
                sdp_id = str(sdp_node_inst.get_nodeattr("partition_id"))
                dataflow_model_filename = sdp_node_inst.get_nodeattr("model")
                kernel_model = ModelWrapper(dataflow_model_filename)
                for node in kernel_model.graph.node:
                    if node.op_type.startswith("StreamingFIFO"):
                        node_inst = getCustomOp(node)
                        fifo_widths[sdp_id] = node_inst.get_instream_width()
            settings["fifo_widths"] = fifo_widths
            # export original folding config to settings file,
            # so that the driver can generate a final cfg with live fifo sizes applied
            folding_path = model.get_metadata_prop("folding_config_before_lfs")
            if folding_path:
                with open(folding_path, "r") as f:
                    folding_cfg = json.load(f)
                settings["folding_config_before_lfs"] = folding_cfg

        settingsfile = pynq_driver_dir + "/settings.json"
        with open(settingsfile, "w") as f:
            json.dump(settings, f, indent=2)

        return (model, False)
