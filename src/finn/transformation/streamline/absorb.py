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

"""Streamline transformations that absorb or reorder simple ops."""

import numpy as np
import numpy.typing as npt
import qonnx.core.data_layout as qonnx_data_layout
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name
from typing import Any, cast

from finn.util.exception import FINNInternalError
from finn.util.logging import log


# Note: Old name kept for compatibility reasons but actually allows to absorb
# any bias irrespective of signedness which might result in changed signedness
# of the output type
class AbsorbSignBiasIntoMultiThreshold(Transformation):
    """Absorb scalar bias originating from signed int export back into
    MultiThreshold and re-evaluate the output datatype."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Absorb scalar bias into MultiThreshold when possible."""
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph
        for node in graph.node:
            # Only non-branching threshold operations are supported
            if (
                node.op_type == "MultiThreshold"
                and not model.is_fork_node(node)
                and not model.is_join_node(node)
            ):
                # We now we are not forking, so there is at most one consumer
                consumer = model.find_consumer(node.output[0])
                # At the end of the graph we might have no consumer. If we have
                # one, only handle Adds, turn Sub into Add first...
                if consumer is not None and consumer.op_type == "Add":
                    # Try to get the parameter tensor for the addition: Sanity
                    # check whether this is present, even though we already
                    # tested for non-joining
                    bias = model.get_initializer(consumer.input[1])

                    # Warn and skip if there is no constant bias present
                    if bias is None:
                        log.warning(
                            f"{self.__class__.__name__} ({node.name}): Bias not constant for"
                            f" {consumer.name}, skipping."
                        )
                        # Skip to next node, nothing changed so far, no need to
                        # break here
                        continue
                    bias = cast("npt.NDArray[Any]", bias)

                    # Try to get the parameter tensor for the thresholds: Sanity
                    # check whether this is present, even though we already
                    # tested for non-joining
                    thresholds = model.get_initializer(node.input[1])

                    # Warn and skip if there is no constant bias present
                    if thresholds is None:
                        log.warning(
                            f"{self.__class__.__name__} ({node.name}): Thresholds not"
                            f" constant, skipping."
                        )
                        # Skip to next node, nothing changed so far, no need to
                        # break here
                        continue
                    thresholds = cast("npt.NDArray[Any]", thresholds)

                    # Check whether the bias is as scalar as we cannot absorb
                    # full tensors into node attributes
                    if not (bias.ndim == 0 or all(x == 1 for x in bias.shape)):
                        log.warning(
                            f"{self.__class__.__name__} ({node.name}): Bias not scalar"
                            f" for {consumer.name}, skipping."
                        )
                        # Skip to next node, nothing changed so far, no need to
                        # break here
                        continue

                    # CustomOp instance of the thresholding node required for
                    # convenient attribute manipulation
                    threshold_op = getCustomOp(node)
                    # Remember the old datatype for some further checks and info
                    old_odt = cast("str", threshold_op.get_nodeattr("out_dtype"))
                    # Get the number of bits currently used to represent the
                    # output values
                    bits = DataType[old_odt].bitwidth()
                    # Check whether these thresholds have been generated from a
                    # narrow range quantizer
                    narrow = int(thresholds.shape[-1] < (2**bits - 1))

                    # Flatten effectively scalar bias tensors and extract to
                    # have "plain" scalar
                    bias = bias.flatten()[0]
                    # Shift the output bias of the thresholding operator
                    out_bias = threshold_op.get_nodeattr("out_bias") + bias
                    # Derive the new output range due to shifting the bias
                    # Note: We count thresholds steps on top of the bias
                    new_min = out_bias - narrow
                    new_max = out_bias + thresholds.shape[-1]

                    # Allows the signedness to change depending on the new
                    # output range [new_min,new_max]
                    if abs(new_min) >= abs(new_max):
                        odt = DataType.get_smallest_possible(new_min)
                    else:
                        odt = DataType.get_smallest_possible(new_max)

                    # Check whether the new range can be represented with the
                    # derived integer datatype
                    if not (odt.allowed(new_max) and odt.allowed(new_min)):
                        # Cannot be represented, warn and skip transforming
                        log.warning(
                            f"{self.__class__.__name__} ({node.name}): Cannot absorb bias"
                            f" from {consumer.name} into {node.name}: {bias}"
                        )
                        # Skip to the next candidate node
                        continue

                    # Check whether the datatype changes as this is something
                    # the "user" should be aware of
                    if odt.name != old_odt:
                        log.warning(
                            f"{self.__class__.__name__} ({node.name}): Changing output datatype "
                            f"from {old_odt} to {odt}"
                        )

                    # Up until now we did not modify the nodes/grap, just did
                    # some checks and derive the new bias and datatype. Start
                    # inserting this back into the graph now...

                    # Set new bias and datatype attributes into the threshold
                    # operator
                    threshold_op.set_nodeattr("out_bias", float(out_bias))
                    threshold_op.set_nodeattr("out_dtype", odt.name)
                    # Remove the bias operator and rewire the graph to skip the
                    # now-missing node
                    node.output[0] = consumer.output[0]
                    graph.node.remove(consumer)
                    # Update the datatype at the output of the threshold
                    # operation
                    model.set_tensor_datatype(node.output[0], odt)

                    # Graph modified so we need to apply this transformation
                    # again
                    graph_modified = True
                    # Better break now to clean up and recover annotations first
                    break
        # As we might have changes types and removed nodes better redo some
        # annotations
        model = model.transform(InferDataTypes())
        model = model.transform(InferShapes())
        # Transformed model and indication whether the transformation should be
        # applied again
        return model, graph_modified


class AbsorbAddIntoMultiThreshold(Transformation):
    """Absorb preceding Add ops into MultiThreshold by updating the threshold
    values. Only scalar/1D add vectors can be absorbed."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Absorb Add nodes into MultiThreshold thresholds."""
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            if n.op_type == "Add" and not model.is_fork_node(n) and not model.is_join_node(n):
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "MultiThreshold":
                    add_weight_name = n.input[1]
                    threshold_name = consumer.input[1]
                    a = model.get_initializer(add_weight_name)
                    t = model.get_initializer(threshold_name)
                    if a is None:
                        raise FINNInternalError("Initializer for add weights is not set.")
                    if t is None:
                        raise FINNInternalError("Initializer for thresholds is not set.")
                    a = cast("npt.NDArray[Any]", a)
                    t = cast("npt.NDArray[Any]", t)
                    start_name = n.input[0]
                    is_scalar = a.ndim == 0 or all(x == 1 for x in a.shape)

                    # Get the shape of the parameter tensor of the add
                    shape = a.shape
                    # First try to consider the tensor layout of the input for
                    # determining the number of output channels
                    layout = model.get_tensor_layout(add_weight_name)
                    # If there is no layout annotation, guess based on rank of
                    # the parameter tensor
                    # TODO: No support for Rank >= 5
                    if layout is None and len(shape) < 5:
                        # Maps tensor rank to layout annotation
                        # fmt:off
                        rank_to_layout = {
                            0: None, 1: "C", 2: "NC", 3: "NWC", 4: "NCHW"
                        }
                        # fmt:on
                        # Lookup the layout required by this input shape
                        layout = rank_to_layout[len(shape)]
                    # If there is a layout annotation, use this to determine the
                    # index of the channel dimension
                    if layout is not None and "C" in layout:
                        # Lookup the index in list
                        cdim = layout.index("C")
                    # If no layout has been annotated or there is no channel
                    # dimension, fall back to the previous default assumption
                    else:
                        # Assume the channels to be in axis 1
                        cdim = 1
                        # Issue a warning to the user, so they are aware of this
                        log.warning(
                            f"{self.__class__.__name__} ({n.name}): No "
                            f"layout annotations for {add_weight_name}:"
                            f" Assuming channel dimension at index {cdim}"
                        )

                    # We can only absorb up to per-channel (last dimension)
                    # granularity
                    if is_scalar or a.shape[cdim] == a.size:
                        # Reshape addition parameters to have the elements/PE
                        # dimension first, aligned with the thresholds.
                        t_new = t - a.reshape(-1, 1)
                        # compute new thresholds and set initializer
                        model.set_initializer(threshold_name, t_new)
                        # wire add input directly to MultiThreshold
                        consumer.input[0] = start_name
                        # remove the add node
                        graph.node.remove(n)
                        graph_modified = True
        return model, graph_modified


class AbsorbMulIntoMultiThreshold(Transformation):
    """Absorb preceding Mul ops into MultiThreshold by updating the threshold
    values. Only *positive* scalar/1D mul vectors can be absorbed."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Absorb Mul nodes into MultiThreshold thresholds when allowed."""
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            if n.op_type == "Mul" and not model.is_fork_node(n) and not model.is_join_node(n):
                mul_weight_name = n.input[1]
                a = model.get_initializer(mul_weight_name)
                if a is None:
                    raise FINNInternalError("Initializer for mul weights is not set.")
                a = cast("npt.NDArray[Any]", a)
                is_signed = (a < 0).any()
                is_scalar = a.ndim == 0 or all(x == 1 for x in a.shape)
                actual_ndims = len(tuple(filter(lambda x: x > 1, a.shape)))
                is_1d = actual_ndims == 1
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "MultiThreshold"
                    and not is_signed
                    and (is_1d or is_scalar)
                ):
                    threshold_name = consumer.input[1]
                    t = model.get_initializer(threshold_name)
                    if t is None:
                        raise FINNInternalError("Initializer for thresholds is not set.")
                    t = cast("npt.NDArray[Any]", t)
                    start_name = n.input[0]
                    # compute new thresholds and set initializer
                    t_new = t / a.reshape(-1, 1)
                    # TODO: need to handle negative A values correctly; produce
                    # mul sign mask and merge into preceding matmul?
                    model.set_initializer(threshold_name, t_new)
                    # wire add input directly to MultiThreshold
                    consumer.input[0] = start_name
                    # remove the mul node
                    graph.node.remove(n)
                    graph_modified = True
        return (model, graph_modified)


class FactorOutMulSignMagnitude(Transformation):
    """Split multiply-by-constant nodes into two multiply-by-constant nodes,
    where the first node is a bipolar vector (of signs) and the second is a
    vector of magnitudes."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Factor signed muls into sign and magnitude stages."""
        graph = model.graph
        graph_modified = False
        for node_ind, n in enumerate(graph.node, start=1):
            if n.op_type == "Mul" and not model.is_join_node(n):
                mul_weight_name = n.input[1]
                a = model.get_initializer(mul_weight_name)
                if a is None:
                    raise FINNInternalError("Initializer for mul weights is not set.")
                a = cast("npt.NDArray[Any]", a)
                is_scalar = np.prod(a.shape) == 1
                actual_ndims = len(tuple(filter(lambda x: x > 1, a.shape)))
                is_1d = actual_ndims == 1
                is_not_bipolar = model.get_tensor_datatype(mul_weight_name) != DataType["BIPOLAR"]
                is_signed = (a < 0).any()
                if is_signed and (is_scalar or is_1d) and is_not_bipolar:
                    start_name = n.input[0]
                    in_shape = model.get_tensor_shape(start_name)
                    if in_shape is None:
                        raise FINNInternalError(f"Could not determine shape of tensor {start_name}")
                    middle_name = model.make_new_valueinfo_name()
                    model.set_tensor_shape(middle_name, in_shape)
                    sign_mul_param_name = model.make_new_valueinfo_name()
                    # create new mul node with sign(A) as the operand
                    sgn = np.sign(a)
                    model.set_initializer(sign_mul_param_name, sgn)
                    model.set_tensor_datatype(sign_mul_param_name, DataType["BIPOLAR"])
                    # replace original mul weight by magnitudes
                    model.set_initializer(mul_weight_name, np.abs(a))
                    new_mul = oh.make_node("Mul", [start_name, sign_mul_param_name], [middle_name])
                    n.input[0] = middle_name
                    graph.node.insert(node_ind - 1, new_mul)
                    graph_modified = True
        return (model, graph_modified)


class Absorb1BitMulIntoMatMul(Transformation):
    """Absorb bipolar or binary multiplications into the preceding matrix
    multiply."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Absorb 1-bit muls into MatMul weights where valid."""
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            # TODO: Fork-nodes could be handled if the muls are the same in all
            #  branches, but this is not checked nor rewired at all right now.
            if n.op_type == "MatMul" and not model.is_fork_node(n):
                matmul_weight_name = n.input[1]
                w = model.get_initializer(matmul_weight_name)
                w_dt = model.get_tensor_datatype(matmul_weight_name)
                # Skip matmuls with no initializers
                if w is None:
                    continue
                w = cast("npt.NDArray[Any]", w)
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "Mul":
                    mul_weight_name = consumer.input[1]
                    a = model.get_initializer(mul_weight_name)
                    # Skip muls with no initializers
                    if a is None:
                        continue
                    a = cast("npt.NDArray[Any]", a)
                    is_1bit = model.get_tensor_datatype(mul_weight_name).bitwidth() == 1
                    if is_1bit:
                        w_new = a * w
                        if w_new.shape != w.shape:
                            raise FINNInternalError(
                                "Shape of new weights is not the same as the shape of the "
                                "weight matrix before."
                            )
                        check_fxn = np.vectorize(lambda x, w_dt=w_dt: w_dt.allowed(x))
                        # only absorb if permitted by W datatype
                        if check_fxn(w_new).all():
                            model.set_initializer(matmul_weight_name, w_new)
                            n.output[0] = consumer.output[0]
                            graph.node.remove(consumer)
                            graph_modified = True
        return (model, graph_modified)


class Absorb1BitMulIntoConv(Transformation):
    """Absorb bipolar or binary multiplications into the preceding convolution."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Absorb 1-bit muls into Conv weights where valid."""
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            if n.op_type == "Conv" and not model.is_fork_node(n):
                conv_weight_name = n.input[1]
                w = model.get_initializer(conv_weight_name)
                w_dt = model.get_tensor_datatype(conv_weight_name)
                # Skip convs with no initializers
                if w is None:
                    continue
                w = cast("npt.NDArray[Any]", w)
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "Mul":
                    mul_weight_name = consumer.input[1]
                    a = model.get_initializer(mul_weight_name)
                    # Skip muls with no initializers
                    if a is None:
                        continue
                    a = cast("npt.NDArray[Any]", a)
                    is_1bit = model.get_tensor_datatype(mul_weight_name).bitwidth() == 1
                    is_scalar = np.prod(a.shape) == 1
                    actual_ndims = len(tuple(filter(lambda x: x > 1, a.shape)))
                    is_1d = actual_ndims == 1
                    if is_1bit and (is_1d or is_scalar):
                        # move the mul to the OFM position, since the mul is
                        # applied on the outputs channelwise or as scalar
                        w_new = a.reshape(-1, 1, 1, 1) * w
                        if w_new.shape != w.shape:
                            raise FINNInternalError(
                                "Shape of new weights is not the same as the shape of the "
                                "conv weights before."
                            )
                        check_fxn = np.vectorize(lambda x, w_dt=w_dt: w_dt.allowed(x))
                        # only absorb if permitted by W datatype
                        if check_fxn(w_new).all():
                            model.set_initializer(conv_weight_name, w_new)
                            n.output[0] = consumer.output[0]
                            graph.node.remove(consumer)
                            graph_modified = True
        return (model, graph_modified)


class AbsorbTransposeIntoMultiThreshold(Transformation):
    """For (NCHWTranspose -> MultiThreshold) move Transpose past MultiThreshold
    and set its data_layout mode to NHWC."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Absorb Transpose into MultiThreshold when applicable."""
        graph = model.graph
        graph_modified = False
        nodes = list(model.graph.node)
        for node_ind, n in enumerate(nodes, start=1):
            if n.op_type == "Transpose" and not model.is_fork_node(n):
                perm_attr = get_by_name(n.attribute, "perm")
                if perm_attr is None:
                    # No permutation attribute: this is not the NCHW->NHWC
                    # transpose we are looking for, softly skip this node
                    continue
                perms = list(perm_attr.ints)
                if perms == [0, 3, 1, 2]:
                    mt_cand = model.find_consumer(n.output[0])
                    if (
                        mt_cand is not None
                        and mt_cand.op_type == "MultiThreshold"
                        # and not model.is_fork_node(mt_cand)
                    ):
                        mt_cand_orig_output = mt_cand.output[0]
                        mt = getCustomOp(mt_cand)
                        mt.set_nodeattr("data_layout", "NHWC")
                        # Rewire input of MultiThreshold node
                        mt_cand.input[0] = n.input[0]
                        # Make new intermediate tensor
                        intermediate_tensor_name = model.make_new_valueinfo_name()
                        intermediate_tensor_shape = model.get_tensor_shape(n.input[0])
                        if intermediate_tensor_shape is None:
                            raise FINNInternalError(
                                f"Could not determine shape of tensor {n.input[0]}"
                            )
                        intermediate_tensor_finn_dtype = model.get_tensor_datatype(
                            mt_cand.output[0]
                        )
                        # Create a new ValueInfoProto and set the shape
                        model.set_tensor_shape(intermediate_tensor_name, intermediate_tensor_shape)
                        # Set the tensor layout
                        model.set_tensor_layout(intermediate_tensor_name, qonnx_data_layout.NHWC)
                        # Set the tensor FINN datatype
                        model.set_tensor_datatype(
                            intermediate_tensor_name, intermediate_tensor_finn_dtype
                        )
                        # Rewire output of MT node
                        mt_cand.output[0] = intermediate_tensor_name
                        # Get rid of first transpose node
                        graph.node.remove(n)
                        # Create new Transpose node
                        new_transpose = oh.make_node(
                            "Transpose",
                            [intermediate_tensor_name],
                            [mt_cand_orig_output],
                            perm=[0, 3, 1, 2],
                        )
                        graph.node.insert(node_ind + 1, new_transpose)
                        graph_modified = True
        if graph_modified:
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class AbsorbTransposeIntoFlatten(Transformation):
    """Absorb transpose node into succeeding flatten node, if H=W=1 and the first
    dimension stays the same. Can also be applied if flatten is implemented implicitly
    by a reshape node with shape [1, -1] and the first input dimension is 1."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Absorb Transpose into Flatten or equivalent Reshape."""
        graph = model.graph
        graph_modified = False
        for node_ind, n in enumerate(graph.node, start=1):
            is_flatten_reshape = False
            if n.op_type == "Reshape":
                reshape_shape = model.get_initializer(n.input[1])
                if reshape_shape is not None:
                    is_flatten_reshape = np.array_equal(
                        cast("npt.NDArray[Any]", reshape_shape), [1, -1]
                    )
            if is_flatten_reshape or n.op_type == "Flatten":
                prod = model.find_producer(n.input[0])
                if prod is None or prod.op_type != "Transpose":
                    continue
                perm_attr = get_by_name(prod.attribute, "perm")
                # we ensure that the first dimension is not changed from the
                # transpose operation
                if perm_attr is None or perm_attr.ints[0] != 0:
                    continue
                data_layout = model.get_tensor_layout(prod.input[0])
                # check for the data layout to interpret input shape correctly
                if data_layout is None:
                    log.warning(
                        f"{self.__class__.__name__} ({n.name}): Data "
                        f"layout for input tensor of Transpose node is "
                        f"not set. To use AbsorbTransposeIntoFlatten "
                        f"transformation please set tensor data layout."
                    )
                    continue
                prod_in_shape = model.get_tensor_shape(prod.input[0])
                if prod_in_shape is None:
                    raise FINNInternalError(f"Could not determine shape of tensor {prod.input[0]}")
                if data_layout == qonnx_data_layout.NCHW:
                    (b, _c, h, w) = prod_in_shape
                    # if h=w=1 the transposition can be absorbed, otherwise
                    # the absorption would lead to an error in the behavior
                    if h != 1 or w != 1:
                        continue
                    # the flatten node from onnx keeps by default the first
                    # dim and flattens the rest, that is why this transformation
                    # can only work with b != 1 if the model contains already a
                    # flatten node and not a reshape node with shape = [1, -1].
                    # If the first  dim of the input tensor is not 1, flatten and
                    # reshape (with shape = [1, -1]) would lead to different results
                    if n.op_type == "Reshape" and b != 1:
                        continue
                elif data_layout == qonnx_data_layout.NHWC:
                    (b, h, w, _c) = prod_in_shape
                    if h != 1 or w != 1:
                        continue
                    if n.op_type == "Reshape" and b != 1:
                        continue
                # create single flatten node and remove obsolete nodes
                node = oh.make_node("Flatten", [prod.input[0]], [n.output[0]])
                graph.node.remove(n)
                graph.node.remove(prod)
                graph.node.insert(node_ind, node)
                graph_modified = True
        if graph_modified:
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class AbsorbScalarMulAddIntoTopK(Transformation):
    """Remove mul/add node prior to topk node if the op is scalar. Note that
    the TopK output probabilities will change, but the indices won't."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Remove scalar mul/add nodes before TopK when safe."""
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            if n.op_type == "TopK":
                prod = model.find_producer(n.input[0])
                if prod is not None and (prod.op_type in ["Mul", "Add"]):
                    prod_input = prod.input[0]
                    param_name = prod.input[1]
                    a = model.get_initializer(param_name)
                    if a is None:
                        log.warning(
                            f"{self.__class__.__name__} ({n.name}): Param {param_name} "
                            f"is not constant, skipping"
                        )
                        continue
                    a = cast("npt.NDArray[Any]", a)
                    is_scalar = all(x == 1 for x in a.shape)
                    is_scalar_pos_mul = is_scalar and (prod.op_type == "Mul") and a > 0
                    is_scalar_add = is_scalar and (prod.op_type == "Add")
                    if is_scalar_pos_mul or is_scalar_add:
                        # if the mul is scalar and positive, we can just delete the
                        # mul node and rewire the top k node. Because the top k node
                        # works with probabilities and their relation to each other
                        # the relation doesn't change if every value is multiplied
                        # with a scalar
                        graph.node.remove(prod)
                        n.input[0] = prod_input
                        # to avoid error the dataype is set to float32
                        model.set_tensor_datatype(n.input[0], DataType["FLOAT32"])
                        graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class AbsorbConsecutiveTransposes(Transformation):
    """Remove (Transpose -> Transpose) patterns when the input and output
    of the pattern have the same layout."""

    def are_opposite_permutations(self, perms1: list[int], perms2: list[int]) -> bool:
        """Return True if two permutations are inverses."""
        if len(perms1) != len(perms2):
            return False
        if not (0 <= max(perms2) < len(perms2)):
            raise FINNInternalError("invalid permutation")
        if not (0 <= max(perms1) < len(perms1)):
            raise FINNInternalError("invalid permutation")

        return all(perms1[p] == i for i, p in enumerate(perms2))

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Remove consecutive Transpose pairs that cancel."""
        graph = model.graph
        graph_modified = False
        for node in graph.node:
            if node.op_type == "Transpose":
                next_nodes = model.find_consumers(node.output[0])
                perm_attr = get_by_name(node.attribute, "perm")
                if perm_attr is None:
                    continue
                perms1 = list(perm_attr.ints)
                if len(next_nodes) == 0:
                    continue
                # check if all nodes after fork are opposite transposes
                all_opposite_transposes = True
                for next_node in next_nodes:
                    next_perm_attr = (
                        get_by_name(next_node.attribute, "perm")
                        if next_node is not None and next_node.op_type == "Transpose"
                        else None
                    )
                    if next_perm_attr is not None:
                        perms2 = list(next_perm_attr.ints)
                        if not self.are_opposite_permutations(perms1, perms2):
                            all_opposite_transposes = False
                            break
                    else:
                        all_opposite_transposes = False
                        break
                if not all_opposite_transposes:
                    continue
                source_tensor = node.input[0]
                for next_node in next_nodes:
                    # connect next_node's consumers' appropriate input to n's input
                    # TODO how to handle top-level outputs if any?
                    nextnode_out = next_node.output[0]
                    if nextnode_out in [x.name for x in model.graph.output]:
                        raise FINNInternalError(
                            f"Cannot absorb transpose: {nextnode_out} is a graph output"
                        )
                    consumers = model.find_consumers(nextnode_out)
                    for cons in consumers:
                        for i, iname in enumerate(cons.input):
                            if iname == nextnode_out:
                                cons.input[i] = source_tensor
                    # remove consumer transpose
                    graph.node.remove(next_node)
                # remove producer transpose
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class AbsorbTransposeIntoResize(Transformation):
    """For (NCHWTranspose -> Resize) move Transpose past Resize and
    change the Resize node's attributes accordingly."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Move Transpose past Resize and adjust scales if needed."""
        graph = model.graph
        graph_modified = False
        for node_ind, node in enumerate(graph.node, start=1):
            if node.op_type == "Transpose" and not model.is_fork_node(node):
                perm_attr = get_by_name(node.attribute, "perm")
                if perm_attr is None:
                    continue
                perms = list(perm_attr.ints)
                if perms == [0, 3, 1, 2]:
                    mt_cand = model.find_consumer(node.output[0])
                    if mt_cand is not None and mt_cand.op_type == "Resize":
                        mode_attr = get_by_name(mt_cand.attribute, "mode")
                        # ONNX Resize defaults to "nearest" if the attribute is absent
                        mode = mode_attr.s.decode("ascii") if mode_attr is not None else "nearest"
                        # skip if mode is not nearest
                        if mode != "nearest":
                            continue
                        # if sizes specified, turn into scales
                        if len(mt_cand.input) > 3:
                            sizes = model.get_initializer(mt_cand.input[3])
                        else:
                            sizes = None
                        if sizes is not None:
                            sizes = cast("npt.NDArray[Any]", sizes)
                            ishape = model.get_tensor_shape(mt_cand.input[0])
                            if ishape is None:
                                raise FINNInternalError(
                                    f"Could not determine shape of tensor {mt_cand.input[0]}"
                                )
                            ns, cs, hs, ws = sizes / np.asarray(ishape)
                            model.set_initializer(mt_cand.input[2], np.asarray([ns, cs, hs, ws]))
                            mt_cand.input.remove(mt_cand.input[3])
                        # scales already specified, transpose indices to NHWC
                        scales = model.get_initializer(mt_cand.input[2])
                        if scales is None:
                            raise FINNInternalError("Initializer for Resize scales is not set.")
                        scales = cast("npt.NDArray[Any]", scales)
                        ns, cs, hs, ws = scales
                        model.set_initializer(mt_cand.input[2], np.asarray([ns, hs, ws, cs]))
                        # get rid of first tranpose node
                        mt_cand.input[0] = node.input[0]
                        graph.node.remove(node)
                        is_last_node = mt_cand.output[0] in [x.name for x in model.graph.output]

                        new_tensor_name = model.make_new_valueinfo_name()
                        if is_last_node:
                            trans_input = new_tensor_name
                            trans_output = mt_cand.output[0]
                        else:
                            trans_input = mt_cand.output[0]
                            trans_output = new_tensor_name
                        # fix tensor shapes for Resize and Transpose
                        out_shape = model.get_tensor_shape(mt_cand.output[0])
                        if out_shape is None:
                            raise FINNInternalError(
                                f"Could not determine shape of tensor {mt_cand.output[0]}"
                            )
                        n, c, hx, wx = out_shape
                        model.set_tensor_shape(trans_input, (n, hx, wx, c))
                        model.set_tensor_shape(trans_output, (n, c, hx, wx))
                        # re-insert Transpose behind Resize
                        new_transpose = oh.make_node(
                            "Transpose",
                            [trans_input],
                            [trans_output],
                            perm=[0, 3, 1, 2],
                        )
                        # rewire nodes
                        final_t_cands = model.find_consumers(mt_cand.output[0])
                        # rewire next nodes' inputs
                        for final_t_cand in final_t_cands:
                            final_t_cand.input[0] = trans_output
                        mt_cand.output[0] = trans_input
                        graph.node.insert(node_ind + 1, new_transpose)
                        graph_modified = True
        if graph_modified:
            model = model.transform(InferDataTypes())
        return (model, graph_modified)
