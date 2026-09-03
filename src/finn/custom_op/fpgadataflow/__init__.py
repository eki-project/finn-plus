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

"""Registry of abstract fpgadataflow HW custom operators.

Every operator module in this package decorates its op class(es) with
``@register_custom_op``; importing the submodule (below) is enough to make the
op resolvable via ``qonnx.custom_op.registry``. There is no hand-maintained
list of ops.
"""
from qonnx.custom_op.base import CustomOp
from typing import TypeVar

from finn.util.exception import FINNInternalError

# Dictionary of registered custom-op implementations, keyed by class name
custom_op: dict[str, type[CustomOp]] = {}

_CustomOpT = TypeVar("_CustomOpT", bound=CustomOp)


# Note: This must be defined before importing any custom op implementation to
# avoid "importing partially initialized module" issues.
def register_custom_op(cls: type[_CustomOpT]) -> type[_CustomOpT]:
    """Register ``cls`` into the ``custom_op`` dictionary by its name."""
    if not issubclass(cls, CustomOp):
        raise FINNInternalError(f"{cls} must subclass {CustomOp}")
    custom_op[cls.__name__] = cls
    # Pass through the class unmodified so this can be used as a decorator
    return cls


# flake8: noqa: E402, F401
# ruff: noqa: E402, F401
# Imports below are for their registration side effects and must follow the
# register_custom_op definition above.

import finn.custom_op.fpgadataflow.attention
import finn.custom_op.fpgadataflow.attention_heads
import finn.custom_op.fpgadataflow.concat
import finn.custom_op.fpgadataflow.convolutioninputgenerator
import finn.custom_op.fpgadataflow.crop
import finn.custom_op.fpgadataflow.duplicatestreams
import finn.custom_op.fpgadataflow.elementwise_binary
import finn.custom_op.fpgadataflow.fmpadding
import finn.custom_op.fpgadataflow.globalaccpool
import finn.custom_op.fpgadataflow.hwsoftmax
import finn.custom_op.fpgadataflow.inner_shuffle
import finn.custom_op.fpgadataflow.input_dilation
import finn.custom_op.fpgadataflow.labelselect
import finn.custom_op.fpgadataflow.layernorm
import finn.custom_op.fpgadataflow.lookup
import finn.custom_op.fpgadataflow.matrixvectoractivation
import finn.custom_op.fpgadataflow.outer_shuffle
import finn.custom_op.fpgadataflow.pool
import finn.custom_op.fpgadataflow.requant
import finn.custom_op.fpgadataflow.reshape
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
