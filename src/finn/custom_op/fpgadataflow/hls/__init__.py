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

"""Registry of HLS-backend fpgadataflow custom operators.

Every ``*_hls`` module in this package decorates its op class(es) with
``@register_custom_op``; importing the submodule (below) is enough to make the
op resolvable via ``qonnx.custom_op.registry``. There is no hand-maintained
list of ops.
"""
from qonnx.custom_op.base import CustomOp
from typing import TypeVar

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.exception import FINNInternalError

# Dictionary of registered HLSBackend implementations, keyed by class name
custom_op: dict[str, type[CustomOp]] = {}

_HLSOpT = TypeVar("_HLSOpT", bound=HLSBackend)


# Note: This must be defined before importing any custom op implementation to
# avoid "importing partially initialized module" issues.
def register_custom_op(cls: type[_HLSOpT]) -> type[_HLSOpT]:
    """Register ``cls`` (an HLSBackend implementation) into the ``custom_op`` dictionary."""
    if not issubclass(cls, HLSBackend):
        raise FINNInternalError(f"{cls} must subclass {HLSBackend}")
    custom_op[cls.__name__] = cls
    # Pass through the class unmodified so this can be used as a decorator
    return cls


# flake8: noqa: E402, F401
# ruff: noqa: E402, F401
# Imports below are for their registration side effects and must follow the
# register_custom_op definition above.

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
