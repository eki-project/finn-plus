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
"""Automatically sets folding, i.e., parallelism attributes for all FINN operators."""
import functools

# Inspect information on Python objects like modules
import inspect
import numpy as np
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from typing import TYPE_CHECKING, Any, Literal, cast

# Import the elementwise binary operation module to extract names of all
# specializations (which require PE parallelism to be configured)
import finn.custom_op.fpgadataflow.hls.elementwise_binary_hls as elementwise_binary_hls
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.util.basic import MAX_ALLOWED_AP_INT_W
from finn.util.exception import FINNFoldingError
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from finn.util.logging import log

if TYPE_CHECKING:
    from numpy.typing import NDArray

from collections.abc import Generator
from qonnx.core.modelwrapper import ModelWrapper

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


def divisors(num: int) -> Generator[int, Any, None]:
    """Yield divisors of num."""
    for x in range(1, num + 1):
        if (num % x) == 0:
            yield x


def common_divisors(numbers: list[int]) -> np.ndarray:
    """Return common divisors of the list of numbers."""
    separate_divisors = []
    for num in numbers:
        individual_divisors = list(divisors(num))
        separate_divisors.append(individual_divisors)

    return functools.reduce(np.intersect1d, separate_divisors)


# Find the op-type names for all HLS specializations of elementwise binary
# operations
# Note: filter on __module__ so module-level type aliases (which pass
# inspect.isclass as types.GenericAlias) and imported classes are skipped
ELEMENTWISE_BINARY_OPS = [
    op_type
    for op_type, cls in inspect.getmembers(elementwise_binary_hls, inspect.isclass)
    if getattr(cls, "__module__", None) == elementwise_binary_hls.__name__
    and issubclass(cls, elementwise_binary_hls.ElementwiseBinaryOperation_hls)
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

    Besides logging warnings, SetFolding records what it could not achieve in attributes of
    the transformation instance. They describe the most recent call of apply() (so with
    apply_to_subgraphs=True the graph that was visited last) and always refer to the requested
    target_cycles_per_frame, also if the two-pass relaxation folded the model for a lower
    throughput afterwards:

    * unhandled_op_types: op types of the HLS/RTL nodes that SetFolding does not know how to
      fold. Their folding attributes, if they have any, are left untouched.
    * achieved_cycles_per_frame: cycle estimate of the slowest HLS/RTL node after folding
      towards the requested target (None before the first call of apply()). The requested
      target was missed if and only if this is larger than target_cycles_per_frame.
    * any_throughput_target_missed: True if the search for at least one folding attribute
      ended without reaching the target, which is what the "did not reach the target
      throughput" warning reports. This is only a hint: it is also set if another folding
      attribute of the same node (e.g. PE after SIMD of an MVAU, SIMD after PE of a VVAU or
      the parallel window mode of a sliding window generator) reached the target afterwards,
      and it is not set for every op type. Use achieved_cycles_per_frame to decide whether
      the target was met.

    If strict is set, SetFolding raises a FINNFoldingError after folding towards the requested
    target if unhandled_op_types is not empty or the requested target was missed. The
    two-pass relaxation, which would only rebalance the model for the lower throughput that
    was achieved, is not run in this case. Without strict (the default) this is only logged.
    """

    def __init__(
        self,
        target_cycles_per_frame: int = 1000,
        mvau_wwidth_max: int = 36,
        two_pass_relaxation: bool = True,
        strict: bool = False,
    ) -> None:
        """Initialize the folding target and constraints."""
        super().__init__()
        self.target_cycles_per_frame = target_cycles_per_frame
        self.mvau_wwidth_max = mvau_wwidth_max
        self.two_pass_relaxation = two_pass_relaxation
        self.strict = strict
        # Results of the most recent call of apply(), see the class docstring
        self.unhandled_op_types: list[str] = []
        self.achieved_cycles_per_frame: int | None = None
        self.any_throughput_target_missed = False

        # REMINDER: The fact that this technically only applies to HLS components should
        # be remembered and considered for an eventual rework.
        if self.mvau_wwidth_max > MAX_ALLOWED_AP_INT_W:
            log.error(
                f"mvau_wwidth_max is larger than {MAX_ALLOWED_AP_INT_W} "
                f"(max. allowed AP_INT_MAX_W). "
                f"This will likely cause issues for HLS components!"
            )

    def optimize_attribute_val(self, node_inst: HWCustomOp, max_val: int, attr_name: str) -> None:
        """Optimize the folding attribute until the target cycles are met."""
        node_inst.set_nodeattr(attr_name, 1)
        for val in divisors(max_val):
            node_inst.set_nodeattr(attr_name, val)
            cyc = node_inst.get_exp_cycles()
            if cyc <= self.target_cycles_per_frame:
                # finish if target met
                return
        # We visited every divisor, so no value that matches the expected cycles was found
        self.any_throughput_target_missed = True
        log.warning(
            f"Node {node_inst.onnx_node.name} did not meet the target cycles. {attr_name} "
            f"was finalized to {node_inst.get_nodeattr(attr_name)}. Estimated: "
            f"{node_inst.get_exp_cycles()} (cyc/frame), Target: "
            f"{self.target_cycles_per_frame} (cyc/frame)."
        )

    def apply(self, model: "ModelWrapper") -> tuple[ModelWrapper, Literal[False]]:
        """Apply SetFolding to all supported nodes in the model."""
        graph = model.graph
        # these ops use PE parallelism, up to a max value of NumChannels
        pe_ops = [
            "DuplicateStreams_hls",
            "GlobalAccPool_hls",
            "PWPolyF_rtl",
            "Thresholding_hls",
            "Thresholding_rtl",
            *ELEMENTWISE_BINARY_OPS,
            "Squeeze_hls",
            "Unsqueeze_hls",
            "Reshape_rtl",
        ]
        # these ops use SIMD parallelism, up to a max value of NumChannels
        # ConvolutionInputGenerator has a special case when depthwise=1
        # ConvolutionInputGenerator_rtl supports additional parallelism by
        # setting parallel_window=1 mode after maxing out SIMD
        simd_ops = [
            "FMPadding_rtl",
            "InputDilation_hls",
            "ConvolutionInputGenerator_rtl",
            # Streaming Split and Concat are SIMD operations
            "StreamingSplit_hls",
            "StreamingConcat_hls",
            "LayerNorm_rtl",
            "Shuffle",
        ]

        # Reset the results of a previous call
        self.unhandled_op_types = []
        self.achieved_cycles_per_frame = None
        # Store whether any node does not meet the target throughput and emit a warning at the end.
        self.any_throughput_target_missed = False
        mvau_internal_decoupled_weightstream_max_width_too_large = False

        # these ops are preceded by depthwise SWG and have special behavior,
        # as explained in the SetFolding docstring
        depthwise_op_exceptions = ["VVAU_hls", "VVAU_rtl", "Pool_hls"]
        for node in graph.node:
            if not (is_hls_node(node) or is_rtl_node(node)):
                continue
            op_type = node.op_type
            node_inst = cast("HWCustomOp", getCustomOp(node))
            if op_type in ["MVAU_hls", "MVAU_rtl"]:
                max_simd = cast("int", node_inst.get_nodeattr("MW"))
                max_pe = cast("int", node_inst.get_nodeattr("MH"))
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
                        node_inst.get_input_datatype(1).bitwidth()
                        * cast("int", node_inst.get_nodeattr("SIMD"))
                        > self.mvau_wwidth_max
                    ):
                        # revert if we've gone above width threshold
                        self.any_throughput_target_missed = True
                        mvau_internal_decoupled_weightstream_max_width_too_large = True
                        node_inst.set_nodeattr("SIMD", prev_simd_val)
                        log.warning(
                            f"{node.name}: Width of weights per PE grew wider than max "
                            f"width {self.mvau_wwidth_max}. Reverting SIMD "
                            f"value {simd_val} -> {prev_simd_val}. "
                            f"Estimated: {node_inst.get_exp_cycles()} (cyc/frame), "
                            f"Target: {self.target_cycles_per_frame} (cyc/frame). "
                        )
                        break

                # Limit PE to not go wider than the weight stream allows for internal_decoupled
                if node_inst.get_nodeattr("mem_mode") == "internal_decoupled":
                    node_inst.set_nodeattr("PE", 1)
                    prev_pe_val = 1
                    for val in divisors(max_pe):
                        node_inst.set_nodeattr("PE", val)
                        cyc = node_inst.get_exp_cycles()
                        if node_inst.get_instream_width(1) > MAX_ALLOWED_AP_INT_W:
                            if is_hls_node(node):
                                # revert PE back to last value
                                node_inst.set_nodeattr("PE", prev_pe_val)
                                self.any_throughput_target_missed = True
                                mvau_internal_decoupled_weightstream_max_width_too_large = True
                                log.warning(
                                    f"{node.name}: Weight stream per PE became wider than max "
                                    f"ap_(u)int width {MAX_ALLOWED_AP_INT_W}. Reverting PE value "
                                    f"{val} -> {prev_pe_val}. "
                                    f"Estimated: {node_inst.get_exp_cycles()} (cyc/frame), "
                                    f"Target: {self.target_cycles_per_frame} (cyc/frame). "
                                )
                                break
                            log.warning(
                                f"{node.name}: Weight stream per "
                                f"PE is wider than {MAX_ALLOWED_AP_INT_W}. "
                                f"While allowed, this may cause routing issues."
                            )
                        if cyc <= self.target_cycles_per_frame:
                            # finish if target met
                            break
                        prev_pe_val = val

                else:
                    # increase PE until target met or reached max_pe
                    self.optimize_attribute_val(node_inst, max_pe, "PE")

            elif op_type in pe_ops:
                # Note: Keep original behavior for all custom-ops defining the
                # NumChannels attribute as it is
                try:
                    max_pe = cast("int", node_inst.get_nodeattr("NumChannels"))
                # Note: Some of the recent additions do not define the
                # NumChannels attribute
                except AttributeError:
                    # We can extract the channels from the normal, i.e., not
                    # folded, shape of the output in these cases
                    max_pe = node_inst.get_normal_output_shape()[-1]
                self.optimize_attribute_val(node_inst, int(max_pe), "PE")
            elif op_type == "LabelSelect_hls":
                max_pe = cast("int", node_inst.get_nodeattr("Labels"))
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type in depthwise_op_exceptions:
                # init/reset SIMD of VVAU
                if op_type in ["VVAU_hls", "VVAU_rtl"]:
                    node_inst.set_nodeattr("SIMD", 1)
                max_pe = cast("int", node_inst.get_nodeattr("Channels"))
                self.optimize_attribute_val(node_inst, max_pe, "PE")
                # increase SIMD for VVAU once PE is exhausted
                pe = cast("int", node_inst.get_nodeattr("PE"))
                cyc = node_inst.get_exp_cycles()
                if (
                    op_type in ["VVAU_hls", "VVAU_rtl"]
                    and pe == max_pe
                    and cyc > self.target_cycles_per_frame
                ):
                    max_simd = np.prod(cast("NDArray", node_inst.get_nodeattr("Kernel")))
                    self.optimize_attribute_val(node_inst, int(max_simd), "SIMD")
                # also set the folding of the upsteam DW SWU
                # which must be identical to this node
                swu_node = model.find_producer(node.input[0])
                if swu_node is None:
                    raise FINNFoldingError(
                        f"{node.name}: Expected producer node for VVAU/Pool input",
                        node_name=node.name,
                        op_type=op_type,
                    )
                if swu_node.op_type.startswith("ConvolutionInputGenerator"):
                    swu_node_inst = cast("HWCustomOp", getCustomOp(swu_node))
                    swu_node_inst.set_nodeattr("SIMD", pe)
                    # The depthwise SWU replays its buffered window once per channel
                    # fold, so it can be slower than the VVAU/Pool it feeds (e.g. a
                    # global pooling window over the whole feature map). Keep raising
                    # the shared parallelism until the SWU meets the target as well.
                    if swu_node_inst.get_exp_cycles() > self.target_cycles_per_frame:
                        for val in divisors(max_pe):
                            if val <= pe:
                                continue
                            pe = val
                            node_inst.set_nodeattr("PE", pe)
                            swu_node_inst.set_nodeattr("SIMD", pe)
                            if swu_node_inst.get_exp_cycles() <= self.target_cycles_per_frame:
                                break
                        else:
                            self.any_throughput_target_missed = True
                            log.warning(
                                f"Node {swu_node.name} did not meet the target cycles. SIMD "
                                f"was finalized to {pe} (bound to PE of {node.name}). "
                                f"Estimated: {swu_node_inst.get_exp_cycles()} (cyc/frame), "
                                f"Target: {self.target_cycles_per_frame} (cyc/frame)."
                            )
                    # enable parallel_window mode of RTL SWG if needed
                    if swu_node.op_type == "ConvolutionInputGenerator_rtl":
                        if (
                            op_type.startswith("VVAU")
                            and cast("int", node_inst.get_nodeattr("SIMD")) > 1
                        ):
                            swu_node_inst.set_nodeattr("parallel_window", 1)
                        else:
                            swu_node_inst.set_nodeattr("parallel_window", 0)
                else:
                    if op_type in ["VVAU_hls", "VVAU_rtl"]:
                        ksize = np.prod(cast("NDArray", node_inst.get_nodeattr("Kernel")))
                    elif op_type == "Pool_hls":
                        ksize = node_inst.get_nodeattr("KernelSize")
                    else:
                        raise FINNFoldingError(
                            f"{node.name}: Undefined edge case for {op_type}",
                            node_name=node.name,
                            op_type=op_type,
                        )
                    if ksize != 1:  # pointwise vvau/pool lack a SWU
                        raise FINNFoldingError(
                            f"{node.name}: Expected SWU on DW op input, found " + swu_node.op_type,
                            node_name=node.name,
                            op_type=op_type,
                        )
            elif op_type in simd_ops:
                if op_type.startswith("ConvolutionInputGenerator"):
                    depthwise = node_inst.get_nodeattr("depthwise")
                    if depthwise == 0:
                        max_simd = cast("int", node_inst.get_nodeattr("IFMChannels"))
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
                elif op_type == "StreamingConcat_hls" or op_type == "StreamingSplit_hls":
                    node_inst.set_nodeattr("SIMD", 1)
                    channels_per_stream = cast(
                        "list[int]", node_inst.get_nodeattr("ChannelsPerStream")
                    )
                    found = False
                    for simd_val in common_divisors(channels_per_stream):
                        node_inst.set_nodeattr("SIMD", int(simd_val))
                        cyc = node_inst.get_exp_cycles()
                        if cyc < self.target_cycles_per_frame:
                            found = True
                            break
                    if not found:
                        log.warning(
                            f"Node {node_inst.onnx_node.name} did not meet "
                            f"the target cycles. SIMD "
                            f"was set to {node_inst.get_nodeattr('SIMD')}. Estimated: "
                            f"{node_inst.get_exp_cycles()} (cyc/frame), Target: "
                            f"{self.target_cycles_per_frame} (cyc/frame)."
                        )
                elif op_type == "LayerNorm_rtl":
                    found = False
                    node_inst.set_nodeattr("SIMD", 1)
                    dim = int(node_inst.get_normal_input_shape()[-1])
                    for simd_val in divisors(dim):
                        if dim // simd_val > 12:
                            node_inst.set_nodeattr("SIMD", int(simd_val))
                            cyc = node_inst.get_exp_cycles()
                            if cyc < self.target_cycles_per_frame:
                                found = True
                                break
                        else:
                            found = True
                            break
                    if not found:
                        log.warning(
                            f"Node {node_inst.onnx_node.name} did not meet "
                            f"the target cycles. SIMD "
                            f"was set to {node_inst.get_nodeattr('SIMD')}. Estimated: "
                            f"{node_inst.get_exp_cycles()} (cyc/frame), Target: "
                            f"{self.target_cycles_per_frame} (cyc/frame)."
                        )
                else:
                    # Note: Keep original behavior for all custom-ops defining
                    # the NumChannels attribute as it is
                    try:
                        max_simd = cast("int", node_inst.get_nodeattr("NumChannels"))
                    # Note: Some of the recent additions do not define the
                    # NumChannels attribute
                    except AttributeError:
                        # We can extract the channels from the normal, i.e., not
                        # folded, shape of the input in these cases
                        max_simd = node_inst.get_normal_input_shape()[-1]
                    self.optimize_attribute_val(node_inst, int(max_simd), "SIMD")
            else:
                log.warning(f"SetFolding doesn't know how to handle op_type {op_type}")
                if op_type not in self.unhandled_op_types:
                    self.unhandled_op_types.append(op_type)

        # Give a warning if a throughput goal was missed
        if self.any_throughput_target_missed:
            log.warning("One or more nodes did not reach the target throughput. ")
            if mvau_internal_decoupled_weightstream_max_width_too_large:
                log.warning(
                    "SIMD parallelization for at least one node was limited by "
                    "mvau_wwidth_max >= bitwidth * SIMD. Try increasing "
                    "mvau_wwidth_max, decreasing the weight bitwidth or, "
                    "if possible, decreasing the throughput target."
                )

        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())
        # The slowest node tells whether the requested target was met
        slowest_node = None
        achieved_cycles = 0
        for node in model.graph.node:
            if is_hls_node(node) or is_rtl_node(node):
                cycles = int(cast("int", getCustomOp(node).get_nodeattr("cycles_estimate")))
                if cycles > achieved_cycles:
                    achieved_cycles = cycles
                    slowest_node = node
        self.achieved_cycles_per_frame = achieved_cycles
        if self.strict:
            problems = []
            problem_node = None
            if slowest_node is not None and achieved_cycles > self.target_cycles_per_frame:
                problems.append(
                    f"Node {slowest_node.name} missed the target of "
                    f"{self.target_cycles_per_frame} cycles per frame with an estimate of "
                    f"{achieved_cycles} cycles"
                )
                problem_node = slowest_node
            if self.unhandled_op_types:
                problems.insert(
                    0,
                    "SetFolding doesn't know how to handle op_type(s) "
                    + ", ".join(self.unhandled_op_types),
                )
                # An op type that cannot be folded is reported with its first node
                problem_node = next(
                    node for node in model.graph.node if node.op_type in self.unhandled_op_types
                )
            if problem_node is not None:
                raise FINNFoldingError(
                    "; ".join(problems),
                    node_name=problem_node.name,
                    op_type=problem_node.op_type,
                )
        if self.two_pass_relaxation:
            perf_dict = model.analysis(dataflow_performance)
            if cast("int", perf_dict["max_cycles"]) > self.target_cycles_per_frame:
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
                        target_cycles_per_frame=cast("int", perf_dict["max_cycles"]),
                        mvau_wwidth_max=self.mvau_wwidth_max,
                        two_pass_relaxation=False,
                    )
                )

        return (model, False)
