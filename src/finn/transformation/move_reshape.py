"""Module to removes a flatten node if it is between two fpgadataflow nodes."""

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name
from typing import TYPE_CHECKING, cast

from finn.util.exception import FINNInternalError
from finn.util.fpgadataflow import is_fpgadataflow_node
from finn.util.logging import log

if TYPE_CHECKING:
    import numpy as np
    from onnx import NodeProto


class RemoveCNVtoFCFlatten(Transformation):
    """Removes a flatten node if it is between two fpgadataflow nodes.
    For an NHWC-Conv to FC transition, the preceding transpose is absorbed.
    The flatten operation can also be implemented by a reshape node."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply transformation."""
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            # also support implicit flatten via reshape, e.g. reshape(1,-1)
            if n.op_type == "Flatten" or n.op_type == "Reshape":
                ishape = model.get_tensor_shape(n.input[0])
                oshape = model.get_tensor_shape(n.output[0])
                if ishape is None or oshape is None:
                    raise FINNInternalError(
                        f"Could not determine tensor shape for node: {n.name}, "
                        f"input shape: {ishape}, output shape: {oshape}"
                    )
                if len(oshape) == 2 and ishape[0] == oshape[0]:
                    producer = model.find_producer(n.input[0])
                    if producer is None:
                        # Do not try to remove a Flatten/Reshape if it is the first node
                        continue
                    if is_fpgadataflow_node(producer):
                        # standalone flatten, remove
                        consumer = model.find_consumer(n.output[0])
                        if is_fpgadataflow_node(consumer):
                            graph_modified = True
                            cast("NodeProto", consumer).input[0] = n.input[0]
                            graph.node.remove(n)
                    elif producer.op_type == "Transpose":
                        # transpose + flatten, absorb into following node
                        transp_node = producer
                        # check if transpose converts NHWC to NCHW
                        ret = get_by_name(transp_node.attribute, "perm")
                        if ret is None:
                            raise FINNInternalError(
                                f"Could not find 'perm' attribute for node: {transp_node.name}"
                            )
                        perms = list(ret.ints)
                        if perms == [0, 3, 1, 2]:
                            producer = model.find_producer(transp_node.input[0])
                            if is_fpgadataflow_node(producer):
                                consumer = model.find_consumer(n.output[0])
                                if consumer is None:
                                    raise FINNInternalError(
                                        f"Could not find consumer for node: {n.name}"
                                    )
                                if consumer.op_type.startswith("MVAU"):
                                    fc_inst = getCustomOp(consumer)
                                    mw = cast("int", fc_inst.get_nodeattr("MW"))
                                    mh = cast("int", fc_inst.get_nodeattr("MH"))
                                    shape = model.get_tensor_shape(transp_node.input[0])
                                    if shape is None:
                                        raise FINNInternalError(
                                            f"Could not determine tensor shape for node: {n.name}, "
                                            f"input shape: {shape}"
                                        )
                                    (_b, h, w, c) = shape
                                    # absorb transpose into weight matrix,
                                    # allowing FC layer to operate on the NHWC input
                                    w_arr = cast(
                                        "np.ndarray", model.get_initializer(consumer.input[1])
                                    )
                                    if w_arr is None:
                                        raise FINNInternalError(
                                            "Initializer for matmul weights is not set."
                                        )
                                    w_new = w_arr.reshape(c, h, w, mh)
                                    w_new = w_new.transpose((1, 2, 0, 3))
                                    w_new = w_new.reshape(mw, mh)
                                    model.set_initializer(consumer.input[1], w_new)
                                    # remove transpose & flatten nodes
                                    consumer.input[0] = transp_node.input[0]
                                    graph.node.remove(n)
                                    graph.node.remove(transp_node)
                                    graph_modified = True
                                else:
                                    log.warning(
                                        "Could not absorb transpose->flatten \
                                        into subsequent node"
                                    )
                        else:
                            log.warning("Unsupported transpose node before flatten layer")

        return (model, graph_modified)
