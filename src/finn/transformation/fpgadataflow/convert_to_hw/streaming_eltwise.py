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

"""DEPRECATED: StreamingEltwise HW layer inference, now redirects to
InferElementwiseBinaryOperation."""

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

from finn.transformation.fpgadataflow.convert_to_hw.elementwise_binary_operation import (
    InferElementwiseBinaryOperation,
)
from finn.util.logging import log


class InferStreamingEltwise(Transformation):
    """DEPRECATED: This transformation is deprecated and now redirects to
    InferElementwiseBinaryOperation.

    StreamingEltwise functionality is now covered by ElementwiseSub and
    ElementwiseAbsDiff operations (with both inputs as streaming).
    This wrapper is kept for backward compatibility.

    The ElementwiseBinary operations provide the same functionality with additional
    features like broadcasting support and more operation types.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply transformation."""
        log.warning(
            "InferStreamingEltwise is deprecated. "
            "Use InferElementwiseBinaryOperation instead. "
            "StreamingEltwise is being replaced by ElementwiseSub/ElementwiseAbsDiff.",
        )
        # Delegate to the new transformation
        return InferElementwiseBinaryOperation().apply(model)
