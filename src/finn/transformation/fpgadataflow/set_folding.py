# Copyright (C) 2020, Xilinx, Inc.
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

# Inspect information on Python objects like modules
import inspect
import numpy as np
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames

# Import the elementwise binary operation module to extract names of all
# specializations (which require PE parallelism to be configured)
import finn.custom_op.fpgadataflow.hls.elementwise_binary_hls as elementwise_binary_hls
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from finn.util.logging import log


def divisors(num):
    for x in range(1, num + 1):
        if (num % x) == 0:
            yield x


# Find the op-type names for all HLS specializations of elementwise binary
# operations
ELEMENTWISE_BINARY_OPS = [
    op_type
    for op_type, cls in inspect.getmembers(elementwise_binary_hls, inspect.isclass)
    if issubclass(cls, elementwise_binary_hls.ElementwiseBinaryOperation_hls)
]


class SetFolding(Transformation):
    """Attempt to set parallelism attributes in all nodes to meet a specific
    target expressed as cycles per frame target_cycles_per_frame. For each
    HLSCustomOp node type, the attribute may vary but is typically one of {PE, SIMD},
    and has a certain allowed-maximum value and divisibility constraints,
    which SetFolding will take into account. Note that the algorithm implemented
    by SetFolding is very simple and it is often possible to hand-tune the returned
    parallelism configuration for better results.

    In the returned model, each node's
    cycles_estimate attribute will be set to its estimated number of cycles.

    If two_pass_relaxation is enabled,
    SetFolding will internally run a second time if the target cycles from the
    first pass could not be achieved, instead using the achievable target (which
    may be constrained by a single node) to obtain a balanced pipeline.

    Notable exceptions and special behavior:

    When folding dense convolution/FC compute engines ("MVAU"/MatrixVectorActivation),
    which have two attributes (PE and SIMD):

    * first increases SIMD while weight stream width per PE is <= mvau_wwidth_max
      (configurable in the SetFolding initializer, defaults to 36)
    * then increases PE until the target is met or max PE reached

    When folding depthwise convolutions ("VVAU"/VectorVectorActivation)
    or spatial reduction ops (Pool_Batch):

    * the producer of the node is expected to be a ConvolutionInputGenerator
      with depthwise=1, whose SIMD value will be set equal to the PE value of
      its consumer node
    * the VVAU also supports SIMD ("input window") parallelism next to
      PE ("channels"), but current ConvInpGen limitations require PE to be fully
      unfolded before SIMD is increased
    """

    def __init__(self, target_cycles_per_frame=1000, mvau_wwidth_max=36, two_pass_relaxation=True):
        super().__init__()
        self.target_cycles_per_frame = target_cycles_per_frame
        self.mvau_wwidth_max = mvau_wwidth_max
        self.two_pass_relaxation = two_pass_relaxation

    def optimize_attribute_val(self, node_inst, max_val, attr_name):
        node_inst.set_nodeattr(attr_name, 1)
        for val in divisors(max_val):
            node_inst.set_nodeattr(attr_name, val)
            cyc = node_inst.get_exp_cycles()
            if cyc <= self.target_cycles_per_frame:
                # finish if target met
                break

    def apply(self, model):
        graph = model.graph
        # these ops use PE parallelism, up to a max value of NumChannels
        pe_ops = [
            "AddStreams_hls",
            "ChannelwiseOp_hls",
            "DuplicateStreams_hls",
            "GlobalAccPool_hls",
            "Thresholding_hls",
            "Thresholding_rtl",
            "ReplicateStream_hls",
            *ELEMENTWISE_BINARY_OPS,
            "Squeeze_hls",
            "Unsqueeze_hls",
        ]
        # these ops use SIMD parallelism, up to a max value of NumChannels
        # ConvolutionInputGenerator* has a special case when depthwise=1
        # ConvolutionInputGenerator_rtl supports additional parallelism by
        # setting parallel_window=1 mode after maxing out SIMD
        simd_ops = [
            "DownSampler_hls",
            "FMPadding_hls",
            "FMPadding_rtl",
            "FMPadding_Pixel_hls",
            "ConvolutionInputGenerator_hls",
            "ConvolutionInputGenerator_rtl",
            # Streaming Split and Concat are SIMD operations
            "StreamingSplit_hls",
            "StreamingConcat_hls",
        ]
        # these ops are preceded by depthwise SWG and have special behavior,
        # as explained in the SetFolding docstring
        depthwise_op_exceptions = ["VVAU_hls", "VVAU_rtl", "Pool_hls"]
        for node in graph.node:
            if not (is_hls_node(node) or is_rtl_node(node)):
                continue
            op_type = node.op_type
            node_inst = getCustomOp(node)
            if op_type in ["MVAU_hls", "MVAU_rtl"]:
                max_simd = node_inst.get_nodeattr("MW")
                max_pe = node_inst.get_nodeattr("MH")
                node_inst.set_nodeattr("PE", 1)
                node_inst.set_nodeattr("SIMD", 1)
                # increase SIMD until either we meet
                # the target or weight stream becomes
                # too wide
                for simd_val in divisors(max_simd):
                    prev_simd_val = node_inst.get_nodeattr("SIMD")
                    node_inst.set_nodeattr("SIMD", simd_val)
                    cyc = node_inst.get_exp_cycles()
                    if cyc <= self.target_cycles_per_frame:
                        # finish if target met
                        break
                    if (
                        node_inst.get_input_datatype(1).bitwidth() * node_inst.get_nodeattr("SIMD")
                        > self.mvau_wwidth_max
                    ):
                        # revert if we've gone above width threshold
                        node_inst.set_nodeattr("SIMD", prev_simd_val)
                        break
                # increase PE until target met or reached max_pe
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type in pe_ops:
                # Note: Keep original behavior for all custom-ops defining the
                # NumChannels attribute as it is
                try:
                    max_pe = node_inst.get_nodeattr("NumChannels")
                # Note: Some of the recent additions do not define the
                # NumChannels attribute
                except AttributeError:
                    # We can extract the channels from the normal, i.e., not
                    # folded, shape of the input in these cases
                    max_pe = node_inst.get_normal_input_shape()[-1]
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type == "LabelSelect_hls":
                max_pe = node_inst.get_nodeattr("Labels")
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type in depthwise_op_exceptions:
                # init/reset SIMD of VVAU
                if op_type in ["VVAU_hls", "VVAU_rtl"]:
                    node_inst.set_nodeattr("SIMD", 1)
                max_pe = node_inst.get_nodeattr("Channels")
                self.optimize_attribute_val(node_inst, max_pe, "PE")
                # increase SIMD for VVAU once PE is exhausted
                pe = node_inst.get_nodeattr("PE")
                cyc = node_inst.get_exp_cycles()
                if (
                    op_type in ["VVAU_hls", "VVAU_rtl"]
                    and pe == max_pe
                    and cyc > self.target_cycles_per_frame
                ):
                    max_simd = np.prod(node_inst.get_nodeattr("Kernel"))
                    self.optimize_attribute_val(node_inst, max_simd, "SIMD")
                # also set the folding of the upsteam DW SWU
                # which must be identical to this node
                swu_node = model.find_producer(node.input[0])
                if swu_node.op_type.startswith("ConvolutionInputGenerator"):
                    swu_node_inst = getCustomOp(swu_node)
                    swu_node_inst.set_nodeattr("SIMD", pe)
                    # enable parallel_window mode of RTL SWG if needed
                    if swu_node.op_type == "ConvolutionInputGenerator_rtl":
                        if op_type.startswith("VVAU") and node_inst.get_nodeattr("SIMD") > 1:
                            swu_node_inst.set_nodeattr("parallel_window", 1)
                        else:
                            swu_node_inst.set_nodeattr("parallel_window", 0)
                else:
                    if op_type in ["VVAU_hls", "VVAU_rtl"]:
                        ksize = np.prod(node_inst.get_nodeattr("Kernel"))
                    elif op_type == "Pool_hls":
                        ksize = node_inst.get_nodeattr("KernelSize")
                    else:
                        raise Exception("Undefined edge case for %s" % op_type)
                    if ksize != 1:  # pointwise vvau/pool lack a SWU
                        raise Exception("Expected SWU on DW op input, found " + swu_node.op_type)
            elif op_type in simd_ops:
                if op_type.startswith("ConvolutionInputGenerator"):
                    depthwise = node_inst.get_nodeattr("depthwise")
                    if depthwise == 0:
                        max_simd = node_inst.get_nodeattr("IFMChannels")
                        # init/reset parallel_window mode of RTL SWG
                        if op_type == "ConvolutionInputGenerator_rtl":
                            node_inst.set_nodeattr("parallel_window", 0)
                        self.optimize_attribute_val(node_inst, max_simd, "SIMD")
                        # enable parallel_window mode of RTL SWG if needed
                        simd = node_inst.get_nodeattr("SIMD")
                        cyc = node_inst.get_exp_cycles()
                        if (
                            op_type == "ConvolutionInputGenerator_rtl"
                            and simd == max_simd
                            and cyc > self.target_cycles_per_frame
                        ):
                            node_inst.set_nodeattr("parallel_window", 1)
                    else:
                        # depthwise SWGs are handled separately
                        continue
                else:
                    # Note: Keep original behavior for all custom-ops defining
                    # the NumChannels attribute as it is
                    try:
                        max_simd = node_inst.get_nodeattr("NumChannels")
                    # Note: Some of the recent additions do not define the
                    # NumChannels attribute
                    except AttributeError:
                        # We can extract the channels from the normal, i.e., not
                        # folded, shape of the input in these cases
                        max_simd = node_inst.get_normal_input_shape()[-1]
                    self.optimize_attribute_val(node_inst, max_simd, "SIMD")
            else:
                log.warning(f"SetFolding doesn't know how to handle op_type {op_type}")

        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())
        if self.two_pass_relaxation:
            perf_dict = model.analysis(dataflow_performance)
            if perf_dict["max_cycles"] > self.target_cycles_per_frame:
                # run again, but with lower target (that we managed) -- this
                # may be coming from a single node's constraints, but we want
                # to balance the entire dataflow pipeline instead
                # no two_pass_relaxation this time -- no guarantee we'll
                # converge otherwise
                max_cycles_node_name = perf_dict["max_cycles_node_name"]
                max_cycles = perf_dict["max_cycles"]
                log.warning(
                    f"Node {max_cycles_node_name} is bottleneck with {max_cycles} cycles, \
                        running second pass"
                )
                model = model.transform(
                    SetFolding(
                        target_cycles_per_frame=perf_dict["max_cycles"],
                        mvau_wwidth_max=self.mvau_wwidth_max,
                        two_pass_relaxation=False,
                    )
                )

        return (model, False)
