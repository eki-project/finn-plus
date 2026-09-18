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

"""Registry for the ``finn.custom_op.fpgadataflow.rtl`` ONNX domain.

Every ``*_rtl`` module in this package decorates its op class(es) with
``@register_custom_op``; importing the submodule (below) is enough to make the
op resolvable via ``qonnx.custom_op.registry``. There is no hand-maintained
list of ops."""

from finn.custom_op.fpgadataflow._registry import make_registry
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend

# Dictionary of registered custom-op implementations (keyed by class name) and the
# decorator that fills it. qonnx.custom_op.registry reads ``custom_op`` to resolve
# nodes carrying this package's name as their ONNX domain.
custom_op, register_custom_op = make_registry(RTLBackend)

# flake8: noqa: E402, F401
# ruff: noqa: E402, F401
# Imports below are for their registration side effects and must follow the
# make_registry call above.

import finn.custom_op.fpgadataflow.rtl.convolutioninputgenerator_rtl
import finn.custom_op.fpgadataflow.rtl.elementwise_binary_rtl
import finn.custom_op.fpgadataflow.rtl.finn_loop
import finn.custom_op.fpgadataflow.rtl.fmpadding_rtl
import finn.custom_op.fpgadataflow.rtl.inner_shuffle_rtl
import finn.custom_op.fpgadataflow.rtl.layernorm_rtl
import finn.custom_op.fpgadataflow.rtl.matrixvectoractivation_rtl
import finn.custom_op.fpgadataflow.rtl.nodecontainer
import finn.custom_op.fpgadataflow.rtl.removedatapath_rtl
import finn.custom_op.fpgadataflow.rtl.requant_rtl
import finn.custom_op.fpgadataflow.rtl.reshape_rtl
import finn.custom_op.fpgadataflow.rtl.streamingdatawidthconverter_rtl
import finn.custom_op.fpgadataflow.rtl.streamingfifo_rtl
import finn.custom_op.fpgadataflow.rtl.thresholding_rtl
import finn.custom_op.fpgadataflow.rtl.vectorvectoractivation_rtl
