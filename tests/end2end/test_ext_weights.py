# Copyright (C) 2021-2022, Xilinx, Inc.
# Copyright (C) 2022-2024, Advanced Micro Devices, Inc.
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

import pytest

import json
import os
from pathlib import Path

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir


def verify_runtime_weights(folding_config_file, runtime_weights_dir):
    with open(folding_config_file) as file:
        dic = json.load(file)
    # count external weights layers
    num_ext_weights = 0
    for module, module_dict in dic.items():
        if module_dict.get("mem_mode") == "external":
            num_ext_weights += 1
    runtime_weights_files = os.listdir(runtime_weights_dir)
    for idx in range(num_ext_weights):
        expected_file = "idma{}.npy".format(idx)
        assert expected_file in runtime_weights_files


@pytest.mark.xdist_group(name="end2end_ext_weights")
@pytest.mark.end2end
class Test_end2end_ext_weights:
    @pytest.mark.slow
    @pytest.mark.vivado
    @pytest.mark.parametrize("topology", ["cnv", "tfc"])
    def test_end2end_ext_weights_build(self, topology):
        # Check for model file in two places:
        # 1. relative to this test file (if it is located in the cloned finn-plus repo)
        # 2. relative to current working directory (if this test file is installed elsewhere)
        f1 = (
            Path(__file__).parent.parent.parent
            / "models"
            / "bnn-pynq"
            / (topology + "-w2a2_qonnx.onnx")
        )
        f2 = (
            Path(os.environ.get("PATH_WORKDIR", "."))
            / "models"
            / "bnn-pynq"
            / (topology + "-w2a2_qonnx.onnx")
        )
        model_file = f1
        if not model_file.is_file():
            model_file = f2
            if not model_file.is_file():
                raise FileNotFoundError(
                    "Could not find model file for topology {} at {} or {}".format(topology, f1, f2)
                )
        test_data = Path(__file__).parent.parent / "example_data" / "test_ext_weights"
        folding_config_file = test_data / (topology + "-w2a2-extw.json")
        specialize_layers_config_file = test_data / (
            "specialize_layers_config_" + topology + ".json"
        )

        output_dir = make_build_dir("test_end2end_" + topology + "_ext_weights_build")
        cfg = build.DataflowBuildConfig(
            output_dir=output_dir,
            verbose=True,
            standalone_thresholds=True,
            folding_config_file=folding_config_file,
            specialize_layers_config_file=specialize_layers_config_file,
            synth_clk_period_ns=10,
            board="ZCU104",
            shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
            generate_outputs=[
                build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
                build_cfg.DataflowOutputType.BITFILE,
                build_cfg.DataflowOutputType.PYNQ_DRIVER,
                build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
            ],
        )
        build.build_dataflow_cfg(str(model_file), cfg)
        assert os.path.isfile(output_dir + "/deploy/bitfile/finn-accel.bit")
        assert os.path.isfile(output_dir + "/deploy/bitfile/finn-accel.hwh")
        assert os.path.isfile(output_dir + "/deploy/driver/driver.py")
        runtime_weights_dir = output_dir + "/deploy/driver/runtime_weights/"
        verify_runtime_weights(folding_config_file, runtime_weights_dir)
