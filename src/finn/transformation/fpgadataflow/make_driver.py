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
import numpy as np
import os
import qonnx
import shlex
import shutil
import subprocess
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.util.basic import roundup_to_integer_multiple
from string import Template
from typing import Dict, Tuple

import finn.util
from finn.transformation.fpgadataflow.get_driver_shapes import get_driver_shapes
from finn.util.basic import make_build_dir
from finn.util.data_packing import hexstring2npbytearray, pack_innermost_dim_as_hex_string
from finn.util.logging import log

from . import template_driver


def to_external_tensor(init, w_dtype):
    """Return an appropriately formatted and packed numpy byte array for given
    external parameter tensor."""

    weight_width = init.shape[1] * w_dtype.bitwidth()
    weight_width_padded = roundup_to_integer_multiple(weight_width, 4)
    hex_init = pack_innermost_dim_as_hex_string(init, w_dtype, weight_width_padded, prefix="0x")
    ext_weight = np.array([], dtype=np.uint8)
    for line in hex_init:
        array_line = [x for x in reversed(hexstring2npbytearray(line, remove_prefix="0x"))]
        ext_weight = np.append(ext_weight, array_line)

    return ext_weight


class MakeCPPDriver(Transformation):
    # TODO: Enable multiple input types! Now only assumes the first one
    def resolve_dt_name(s: str) -> str:
        s = s.replace("DataType[", "").replace("]", "")
        log.info(f"Converting tensor datatype {s}")
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
            raise RuntimeError(f"Unknown datatype for C++ Driver:{s}")

    def __init__(
        self,
        platform: str,
        build_dir: str,
        version: str,
        driver_dir,
    ):
        super().__init__()
        self.platform: str = platform
        self.build_dir = build_dir
        self.version = version
        self.driver_dir = driver_dir

        # Define variables for the repository URL and commit hash
        self.repository_url = "https://github.com/eki-project/finn-cpp-driver.git"
        if version == "latest" or version is None:
            self.commit_hash = "HEAD"
        else:
            self.commit_hash = version

        # Locations of files
        self.xclbin_path = os.path.join(self.build_dir, "bitfile", "finn-accel.xclbin")
        self.json_path = os.path.join(self.driver_dir, "acceleratorconfig.json")
        self.header_path = os.path.join(self.driver_dir, "AcceleratorDatatypes.h")

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        driver_shapes: Dict = get_driver_shapes(model)
        ext_weight_dma_cnt: int  # noqa
        weights_dir: str  # noqa
        # ext_weight_dma_cnt, weights_dir = write_weights(model, cpp_driver_dir)

        # * Creating the driver dir if it doesnt exist yet
        if not os.path.isdir(self.driver_dir):
            os.mkdir(self.driver_dir)
        else:
            try:
                shutil.rmtree(self.driver_dir)
            except Exception as e:
                print(f"Failed to delete {self.driver_dir}. Reason: {e}")
                raise e
            os.mkdir(self.driver_dir)

        # Get the base C++ driver repo
        def run_command(command, cwd=None, debug=False):
            try:
                result = subprocess.run(
                    shlex.split(command), cwd=cwd, check=True, text=True, capture_output=True
                )
                if debug:
                    print(result.stdout)  # Print the output for debugging purposes
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {command}")
                print(f"Output:{e.stdout}; Error:{e.stderr}")
                raise e

        # Step-by-step equivalent of the provided bash script
        run_command("git init", cwd=self.driver_dir)
        run_command(f"git remote add origin {self.repository_url}", cwd=self.driver_dir)
        run_command(f"git fetch origin {self.commit_hash} --depth=1", cwd=self.driver_dir)
        run_command("git checkout FETCH_HEAD", cwd=self.driver_dir)
        run_command("git submodule update --init --recursive", cwd=self.driver_dir)
        run_command("./buildDependencies.sh", cwd=self.driver_dir)

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
        log.info(
            f"Writing input header file: Used datatypes\
                will be {inputDatatype} and {outputDatatype}!"
        )
        with open(
            os.path.join(
                self.driver_dir, "src", "FINNCppDriver", "config", "FinnDriverUsedDatatypes.h.in"
            ),
            "r",
        ) as f_in:
            header = f_in.read()
            template_handler = Template(header)
            templated_str = template_handler.substitute(
                inputDatatype=inputDatatype, outputDatatype=outputDatatype
            )
            with open(self.header_path, "w+") as f:
                f.write(templated_str)

        log.info("Successfully created config header file.")

        # * Writing the json file
        # TODO: Update this for multi-fpga usage (more than one device!)
        # Path of the xclbin in the finn compiler project
        # Get kernel names using xclbinutil

        if shutil.which("xclbinutil") is None:
            raise RuntimeError(
                "xclbinutil not in PATH or not installed.\
                Required to read kernel names for driver config!"
            )
        run_command(
            f"xclbinutil -i {self.xclbin_path} --dump-section IP_LAYOUT:JSON:ip_layout.json",
            cwd=os.path.join(self.build_dir, ".."),
        )
        ips = None
        with open("ip_layout.json") as f:
            ips = json.loads(f.read())["ip_layout"]["m_ip_data"]

        # Get only ips that are kernels
        isIO = (
            lambda x: x["m_type"] == "IP_KERNEL"
            and x["m_base_address"] != "not_used"
            and ("idma" in x["m_name"] or "odma" in x["m_name"])
        )
        idmas = [x["m_name"] for x in ips if isIO(x) and "idma" in x["m_name"]]
        odmas = [x["m_name"] for x in ips if isIO(x) and "odma" in x["m_name"]]

        def formatKernelName(kname: str):
            kparts = kname.split(":")
            return kparts[0] + ":{" + kparts[1] + "}"

        # Create idma and odma entries
        jsonIdmas = []
        jsonOdmas = []

        if len(driver_shapes["idma_names"]) > 1 or len(driver_shapes["odma_names"]) > 1:
            log.warning(
                "Using multiple input/output kernels in the C++ driver is supported,\
                    but not well tested. You might encounter issues using this feature."
            )

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

        data = []
        data.append(
            {
                "xrtDeviceIndex": 0,
                "xclbinPath": os.path.abspath(self.xclbin_path),
                "name": "MainDevice",
                "idmas": jsonIdmas,
                "odmas": jsonOdmas,
            }
        )
        with open(self.json_path, "w+") as f:
            f.write(json.dumps(data, indent=4))

        log.info("Created runtime json config file")

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

    def __init__(self, platform):
        super().__init__()
        self.platform = platform

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
                if iodma_node.get_nodeattr("burstMode") == "wrap":  # input weights dma?
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
