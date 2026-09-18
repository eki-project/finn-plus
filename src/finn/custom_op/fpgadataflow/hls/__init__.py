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

"""Registry for the ``finn.custom_op.fpgadataflow.hls`` ONNX domain.

Every ``*_hls`` module in this package decorates its op class(es) with
``@register_custom_op``; importing the submodule (below) is enough to make the
op resolvable via ``qonnx.custom_op.registry``. There is no hand-maintained
list of ops."""

from finn.custom_op.fpgadataflow._registry import make_registry
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

# Dictionary of registered custom-op implementations (keyed by class name) and the
# decorator that fills it. qonnx.custom_op.registry reads ``custom_op`` to resolve
# nodes carrying this package's name as their ONNX domain.
custom_op, register_custom_op = make_registry(HLSBackend)

# flake8: noqa: E402, F401
# ruff: noqa: E402, F401
# Imports below are for their registration side effects and must follow the
# make_registry call above.

import finn.custom_op.fpgadataflow.hls.attention_heads_hls
import finn.custom_op.fpgadataflow.hls.attention_hls
import finn.custom_op.fpgadataflow.hls.checksum_hls
import finn.custom_op.fpgadataflow.hls.concat_hls
import finn.custom_op.fpgadataflow.hls.crop_hls
import finn.custom_op.fpgadataflow.hls.duplicatestreams_hls
import finn.custom_op.fpgadataflow.hls.elementwise_binary_hls
import finn.custom_op.fpgadataflow.hls.globalaccpool_hls
import finn.custom_op.fpgadataflow.hls.hwsoftmax_hls
import finn.custom_op.fpgadataflow.hls.input_dilation_hls
import finn.custom_op.fpgadataflow.hls.iodma_hls
import finn.custom_op.fpgadataflow.hls.labelselect_hls
import finn.custom_op.fpgadataflow.hls.layernorm_hls
import finn.custom_op.fpgadataflow.hls.lookup_hls
import finn.custom_op.fpgadataflow.hls.matrixvectoractivation_hls
import finn.custom_op.fpgadataflow.hls.outer_shuffle_hls
import finn.custom_op.fpgadataflow.hls.pool_hls
import finn.custom_op.fpgadataflow.hls.requant_hls
import finn.custom_op.fpgadataflow.hls.split_hls
import finn.custom_op.fpgadataflow.hls.squeeze_hls
import finn.custom_op.fpgadataflow.hls.streamingdatawidthconverter_hls
import finn.custom_op.fpgadataflow.hls.streamingfifo_hls
import finn.custom_op.fpgadataflow.hls.thresholding_hls
import finn.custom_op.fpgadataflow.hls.tlastmarker_hls
import finn.custom_op.fpgadataflow.hls.unsqueeze_hls
import finn.custom_op.fpgadataflow.hls.upsampler_hls
import finn.custom_op.fpgadataflow.hls.vectorvectoractivation_hls
