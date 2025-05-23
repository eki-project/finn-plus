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

import os
import shutil
import subprocess
import wget

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir
from finn.util.test import load_test_checkpoint_or_skip


def get_checkpoint_name(step):
    build_dir = os.environ["FINN_BUILD_DIR"]
    onnx_dir_local = build_dir + "/onnx-models-bnn-pynq"
    if step == "build":
        # checkpoint for build step is an entire dir
        return os.environ["FINN_BUILD_DIR"] + "/end2end_ext_weights_build"
    elif step == "download":
        return onnx_dir_local + "/tfc-w2a2.onnx"
    else:
        # other checkpoints are onnx files
        return os.environ["FINN_BUILD_DIR"] + "/end2end_ext_weights_%s.onnx" % (step)


@pytest.mark.xdist_group(name="end2end_ext_weights")
@pytest.mark.end2end
class Test_end2end_ext_weights:
    def test_end2end_ext_weights_download(self):
        build_dir = os.environ["FINN_BUILD_DIR"]
        onnx_zip_local = build_dir + "/onnx-models-bnn-pynq.zip"
        onnx_dir_local = build_dir + "/onnx-models-bnn-pynq"
        onnx_zip_url = "https://github.com/Xilinx/finn-examples"
        onnx_zip_url += "/releases/download/v0.0.1a/onnx-models-bnn-pynq.zip"
        if not os.path.isfile(onnx_zip_local):
            wget.download(onnx_zip_url, out=onnx_zip_local)
        assert os.path.isfile(onnx_zip_local)
        subprocess.check_output(["unzip", "-o", onnx_zip_local, "-d", onnx_dir_local])
        assert os.path.isfile(get_checkpoint_name("download"))

    @pytest.mark.slow
    @pytest.mark.vivado
    def test_end2end_ext_weights_build(self):
        model_file = get_checkpoint_name("download")
        load_test_checkpoint_or_skip(model_file)
        test_data = os.path.join(os.environ["FINN_QNN_DATA"], "test_ext_weights")
        folding_config_file = test_data + "/tfc-w2a2-extw.json"
        specialize_layers_config_file = test_data + "/specialize_layers_config.json"
        output_dir = make_build_dir("test_end2end_ext_weights_build")
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
        build.build_dataflow_cfg(model_file, cfg)
        assert os.path.isfile(output_dir + "/deploy/bitfile/finn-accel.bit")
        assert os.path.isfile(output_dir + "/deploy/bitfile/finn-accel.hwh")
        assert os.path.isfile(output_dir + "/deploy/driver/driver.py")
        assert os.path.isfile(output_dir + "/deploy/driver/runtime_weights/idma0.npy")
        if os.path.isdir(get_checkpoint_name("build")):
            shutil.rmtree(get_checkpoint_name("build"))
        shutil.copytree(output_dir + "/deploy", get_checkpoint_name("build"))
