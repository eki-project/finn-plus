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

"""
Utility functions for FINN notebooks.

This module contains functions that are commonly used in FINN notebooks
for loading test models, generating example inputs, and other utilities.
"""

import inspect
import netron
import numpy as np
import onnx
import onnx.numpy_helper as nph
import os
import pathlib
import torch
from brevitas_examples import bnn_pynq, imagenet_classification
from IPython.display import IFrame
from pkgutil import get_data
from torch.nn import Module, Sequential


def get_notebooks_folder() -> pathlib.Path:
    return pathlib.Path(__file__).parent


# map of (wbits,abits) -> model
example_map = {
    ("CNV", 1, 1): bnn_pynq.cnv_1w1a,
    ("CNV", 1, 2): bnn_pynq.cnv_1w2a,
    ("CNV", 2, 2): bnn_pynq.cnv_2w2a,
    ("LFC", 1, 1): bnn_pynq.lfc_1w1a,
    ("LFC", 1, 2): bnn_pynq.lfc_1w2a,
    ("SFC", 1, 1): bnn_pynq.sfc_1w1a,
    ("SFC", 1, 2): bnn_pynq.sfc_1w2a,
    ("SFC", 2, 2): bnn_pynq.sfc_2w2a,
    ("TFC", 1, 1): bnn_pynq.tfc_1w1a,
    ("TFC", 1, 2): bnn_pynq.tfc_1w2a,
    ("TFC", 2, 2): bnn_pynq.tfc_2w2a,
    ("mobilenet", 4, 4): imagenet_classification.quant_mobilenet_v1_4b,
}


def get_test_model(netname, wbits, abits, pretrained):
    """Returns the model specified by input arguments from the Brevitas BNN-PYNQ
    test networks. Pretrained weights loaded if pretrained is True."""
    model_cfg = (netname, wbits, abits)
    model_def_fxn = example_map[model_cfg]
    fc = model_def_fxn(pretrained)
    return fc.eval()


def get_test_model_trained(netname, wbits, abits):
    "get_test_model with pretrained=True"
    return get_test_model(netname, wbits, abits, pretrained=True)


def get_topk(vec, k):
    "Return indices of the top-k values in given array vec (treated as 1D)."
    return np.flip(vec.flatten().argsort())[:k]


def get_example_input(topology):
    "Get example numpy input tensor for given topology."

    if "fc" in topology:
        raw_i = get_data("qonnx.data", "onnx/mnist-conv/test_data_set_0/input_0.pb")
        onnx_tensor = onnx.load_tensor_from_string(raw_i)
        return nph.to_array(onnx_tensor)
    elif topology == "cnv":
        cifar_path = (
            get_notebooks_folder() / "example_data" / "cifar10" / "cifar10-test-data-class3.npz"
        )
        x = np.load(cifar_path)["arr_0"].astype(np.float32)
        return x
    else:
        raise Exception("Unknown topology, can't return example input")


def get_trained_network_and_ishape(topology, wbits, abits):
    "Return (trained_model, shape) for given BNN-PYNQ test config."

    topology_to_ishape = {
        "tfc": (1, 1, 28, 28),
        "lfc": (1, 1, 28, 28),
        "cnv": (1, 3, 32, 32),
    }
    ishape = topology_to_ishape[topology]
    model = get_test_model_trained(topology.upper(), wbits, abits)
    return (model, ishape)


# PyTorch utility classes for notebooks


class Normalize(Module):
    """PyTorch module for normalizing input tensors with given mean and standard deviation."""

    def __init__(self, mean, std, channels):
        """Initialize the Normalize module.

        Args:
            mean: Mean values for normalization
            std: Standard deviation values for normalization
            channels: Number of channels in the input tensor
        """
        super(Normalize, self).__init__()

        self.mean = mean
        self.std = std
        self.channels = channels

    def forward(self, x):
        """Apply normalization to input tensor.

        Args:
            x: Input tensor to normalize

        Returns:
            Normalized tensor
        """
        x = x - torch.tensor(self.mean, device=x.device).reshape(1, self.channels, 1, 1)
        x = x / self.std
        return x


class ToTensor(Module):
    """PyTorch module that converts input values from [0, 255] range to [0, 1] range."""

    def __init__(self):
        """Initialize the ToTensor module."""
        super(ToTensor, self).__init__()

    def forward(self, x):
        """Convert input tensor from [0, 255] range to [0, 1] range.

        Args:
            x: Input tensor with values in [0, 255] range

        Returns:
            Tensor with values in [0, 1] range
        """
        x = x / 255
        return x


class NormalizePreProc(Module):
    """PyTorch module that combines ToTensor scaling and normalization preprocessing."""

    def __init__(self, mean, std, channels):
        """Initialize the NormalizePreProc module.

        Args:
            mean: Mean values for normalization
            std: Standard deviation values for normalization
            channels: Number of channels in the input tensor
        """
        super(NormalizePreProc, self).__init__()
        self.features = Sequential()
        scaling = ToTensor()
        self.features.add_module("scaling", scaling)
        normalize = Normalize(mean, std, channels)
        self.features.add_module("normalize", normalize)

    def forward(self, x):
        """Apply scaling and normalization preprocessing to input tensor.

        Args:
            x: Input tensor to preprocess

        Returns:
            Preprocessed tensor
        """
        return self.features(x)


# Visualization utility functions for notebooks


def showSrc(what):
    """Display the source code of a function or class."""
    print("".join(inspect.getsourcelines(what)[0]))


def showInNetron(model_filename: str, localhost_url: str = None, port: int = None):
    """Shows a ONNX model file in the Jupyter Notebook using Netron.

    :param model_filename: The path to the ONNX model file.
    :type model_filename: str

    :param localhost_url: The IP address used by the Jupyter IFrame to show the model.
     Defaults to localhost.
    :type localhost_url: str, optional

    :param port: The port number used by Netron and the Jupyter IFrame to show
     the ONNX model.  Defaults to 8081.
    :type port: int, optional

    :return: The IFrame displaying the ONNX model.
    :rtype: IPython.lib.display.IFrame
    """
    try:
        port = port or int(os.getenv("NETRON_PORT", default="8081"))
    except ValueError:
        port = 8081
    localhost_url = localhost_url or os.getenv("LOCALHOST_URL", default="localhost")
    netron.start(model_filename, address=("0.0.0.0", port), browse=False)
    return IFrame(src=f"http://{localhost_url}:{port}/", width="100%", height=400)
