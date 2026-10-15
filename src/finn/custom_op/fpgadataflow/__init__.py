# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2023-2024, Advanced Micro Devices, Inc.
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

"""Registry for the ``finn.custom_op.fpgadataflow`` ONNX domain.

Holds the abstract layer definitions that ``hls/`` and ``rtl/`` specialize, plus
the structural ops (``StreamingDataflowPartition``, ``Shuffle``, ``DNNContainer``)
that have no backend because a transformation lowers or replaces them before
synthesis.

Every operator module decorates its op class(es) with ``@register_custom_op``;
importing the submodule (below) is enough to make the op resolvable via
``qonnx.custom_op.registry``. There is no hand-maintained list of ops."""

from qonnx.custom_op.base import CustomOp

from finn.custom_op.fpgadataflow._registry import make_registry

# Dictionary of registered custom-op implementations (keyed by class name) and the
# decorator that fills it. qonnx.custom_op.registry reads ``custom_op`` to resolve
# nodes carrying this package's name as their ONNX domain.
custom_op, register_custom_op = make_registry(CustomOp)

# flake8: noqa: E402, F401
# ruff: noqa: E402, F401
# Imports below are for their registration side effects and must follow the
# make_registry call above.

import finn.custom_op.fpgadataflow.attention
import finn.custom_op.fpgadataflow.attention_heads
import finn.custom_op.fpgadataflow.concat
import finn.custom_op.fpgadataflow.convolutioninputgenerator
import finn.custom_op.fpgadataflow.crop
import finn.custom_op.fpgadataflow.dnncontainer
import finn.custom_op.fpgadataflow.duplicatestreams
import finn.custom_op.fpgadataflow.elementwise_binary
import finn.custom_op.fpgadataflow.fmpadding
import finn.custom_op.fpgadataflow.globalaccpool
import finn.custom_op.fpgadataflow.hwsoftmax
import finn.custom_op.fpgadataflow.hwwhere
import finn.custom_op.fpgadataflow.inner_shuffle
import finn.custom_op.fpgadataflow.input_dilation
import finn.custom_op.fpgadataflow.labelselect
import finn.custom_op.fpgadataflow.layernorm
import finn.custom_op.fpgadataflow.lookup
import finn.custom_op.fpgadataflow.matrixvectoractivation
import finn.custom_op.fpgadataflow.outer_shuffle
import finn.custom_op.fpgadataflow.pad1d
import finn.custom_op.fpgadataflow.pool
import finn.custom_op.fpgadataflow.pwpolyf
import finn.custom_op.fpgadataflow.requant
import finn.custom_op.fpgadataflow.reshape
import finn.custom_op.fpgadataflow.selecttoken
import finn.custom_op.fpgadataflow.shuffle
import finn.custom_op.fpgadataflow.split
import finn.custom_op.fpgadataflow.squeeze
import finn.custom_op.fpgadataflow.streamingdataflowpartition
import finn.custom_op.fpgadataflow.streamingdatawidthconverter
import finn.custom_op.fpgadataflow.streamingfifo
import finn.custom_op.fpgadataflow.thresholding
import finn.custom_op.fpgadataflow.unsqueeze
import finn.custom_op.fpgadataflow.upsampler
import finn.custom_op.fpgadataflow.vectorvectoractivation
