"""Create SDPs for Multi-FPGA usage."""

from __future__ import annotations

from onnx import NodeProto
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from typing import TYPE_CHECKING, cast

from finn.builder.build_dataflow_config import MFVerbosity
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.externalize_params import ExternalizeParams
from finn.util.exception import FINNInternalError
from finn.util.fpgadataflow import get_device_id, get_submodel, set_device_id
from finn.util.logging import log

if TYPE_CHECKING:
    from pathlib import Path
    from qonnx.core.modelwrapper import ModelWrapper


class ClusterByNodeattribute(Transformation):
    """Cluster nodes by a "comparing attribute". If two nodes are clustered together, they
    have their "partition attribute" set to the same value. Can be run by `transform` until no more
    merges happen, then the graph is fully partitioned by the given attribute.

    Requirements:
        - An attribute that can be compared on equality (==)
        - A partition attribute of type `int`. Every node will receive a unique partition ID
            upon first call.
    """

    def __init__(
        self,
        resolve_circular_dependencies: bool,
        compare_attribute: str,
        partition_attribute: str = "partition_id",
    ) -> None:
        """Cluster by comparison attribute, by setting partition_attribute.
        If "`resolve_circular_dependencies` is given, this calls
        `ResolveCircularPartitionIDs` implicitly. If you explicitly only want
        clustered nodes, set this to False.
        """
        self.comp = compare_attribute
        self.part = partition_attribute
        self.first_call = True
        self.resolve_circular_dependencies = resolve_circular_dependencies

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Merge nodes with their neighbors. If atleast one merge happens, return modified=True."""
        # Check that all compare attributes exist
        for i, node in enumerate(model.graph.node):
            try:
                _ = getCustomOp(node).get_nodeattr(self.comp)
            except AttributeError as e:
                raise FINNInternalError(
                    f"Cannot cluster because "
                    f"comparison node attribute "
                    f"'{self.comp}' is  missing "
                    f"on at least one node ({node.name})."
                ) from e
            if self.first_call:
                try:
                    getCustomOp(node).set_nodeattr(self.part, i)
                except AttributeError as e:
                    raise FINNInternalError(
                        f"Cannot initialize partition attribute '{self.part}' "
                        f"on at least one node ({node.name}). Make sure the "
                        f"attribute exists and is of type 'int'."
                    ) from e

        # Dont re-initialize partition attributes again
        self.first_call = False

        # Merge
        modified = False
        for node in model.graph.node:
            node_op = getCustomOp(node)
            pre = model.find_direct_predecessors(node)
            suc = model.find_direct_successors(node)
            if pre is None:
                pre = []
            if suc is None:
                suc = []
            neighbors = pre + suc
            for neighbor in neighbors:
                neighbor_op = getCustomOp(neighbor)
                if node_op.get_nodeattr(self.comp) == neighbor_op.get_nodeattr(
                    self.comp
                ) and node_op.get_nodeattr(self.part) != neighbor_op.get_nodeattr(self.part):
                    neighbor_op.set_nodeattr(self.part, node_op.get_nodeattr(self.part))
                    modified = True

        # If we want to also resolve circular dependencies, do so now, when no
        # further clustering is required.
        if not modified and self.resolve_circular_dependencies:
            model = model.transform(ResolveCircularPartitionIDs(self.part))

        return model, modified


class ResolveCircularPartitionIDs(Transformation):
    """Traverse the graph. As soon as a node is seen twice, its partition_id is changed,
    and all directly connected successor nodes with the same ID have theirs changed as well.
    This might be necessary after ClusterByNodeattribute, since this might create a circular
    dependency. For example when given this graph with partition IDs A and B:
    -> A - B - A ->
       | --- /
    """  # noqa

    def __init__(self, partition_attribute: str = "partition_id") -> None:
        """Resolve circular partition IDs (or another attribute with the given name)."""
        self.part = partition_attribute

    def get_id(self, node: NodeProto) -> int:
        """Utility."""
        return cast("int", getCustomOp(node).get_nodeattr(self.part))

    def get_successors_with_id(
        self, node: NodeProto, this_id: int, model: ModelWrapper
    ) -> list[NodeProto]:
        """Get all direct successors of `node` which share the same partition ID."""
        s = model.find_direct_successors(node)
        if s is None:
            s = []
        return [node for node in s if self.get_id(node) == this_id]

    def traverse_downstream(self, node: NodeProto, model: ModelWrapper) -> list[NodeProto]:
        """Traverse downstream and list all nodes."""
        nodes = []
        queue = model.find_direct_successors(node)
        if queue is None:
            return []
        seen = []
        while len(queue) > 0:
            if queue[0].name not in seen:
                seen.append(queue[0].name)
                s = model.find_direct_successors(queue[0])
                if s is not None:
                    queue += s
                nodes.append(queue.pop(0))
            else:
                queue.pop(0)
        return nodes

    def get_circular_partition_from_node(
        self, node: NodeProto, model: ModelWrapper
    ) -> NodeProto | None:
        """From the given node traverse the graph until a node has the same partition ID
        as the origin one (unconnected). We do so by iterating all downstream nodes. If:
            - a downstream node has the same ID as the origin node
            - this downstream node has any predecessor that has a different ID but is still
                part of the downstream nodes
        then we have detected a loop.
        """
        downstream_nodes = self.traverse_downstream(node, model)
        for current in downstream_nodes:
            pre = model.find_direct_predecessors(current)
            if pre is None:
                continue
            if self.get_id(current) == self.get_id(node) and any(
                (self.get_id(p) != self.get_id(current)) and p in downstream_nodes for p in pre
            ):
                return current
        return None

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Remove a circular dependency."""
        modified = False

        # Ensure unique node names for checking cyclicity
        model = model.transform(GiveUniqueNodeNames())
        circular_node = None
        for node in model.graph.node:
            circular_node = self.get_circular_partition_from_node(node, model)
            if circular_node is not None:
                break

        # If no dependencies were detected, we can simply return
        if circular_node is None:
            return model, modified

        # Resolve
        all_ids = [self.get_id(node) for node in model.graph.node]
        new_id = max(all_ids) + 1

        # Store the ID that represents the circular path for later
        circular_id = self.get_id(circular_node)

        # Update the circular nodes ID itself
        getCustomOp(circular_node).set_nodeattr(self.part, new_id)
        modified = True

        # Filter for nodes that belong to the same partition as the node that closed the circle,
        # all other nodes are implicitly disconnected by having a
        # different partition ID and thus arent
        # part of the loop
        queue: list[NodeProto] = self.get_successors_with_id(circular_node, circular_id, model)

        # Iterate all nodes until none are left
        while len(queue) > 0:
            current = queue[0]

            # Resolve this node
            if self.get_circular_partition_from_node(current, model) is None:
                getCustomOp(current).set_nodeattr(self.part, new_id)
                queue += self.get_successors_with_id(current, circular_id, model)

            # This is done either way. If we didn't enter the if-branch, we
            # encountered a loop inside a loop. Stop resolving this branch, it will be resolved
            # in the next iteration of this transformation
            queue.pop(0)

        # Resolved every node with this circular ID, so we can now return
        return model, modified


class CreateMultiFPGAStreamingDataflowPartition(Transformation):
    """Create SDPs for Multi-FPGA models. This is done by clustering all nodes according to their
    device ID. The nodes are then packed into SDPs which have the
    same device ID as their submodel nodes.

    For different, separated sections of the model with the same device ID,
    the transformation creates separate partitions. Thus a model with
    devices A -> A -> B -> A will have 3 SDPs: 0 (A) -> 1 (B) -> 2 (A).
    """

    def __init__(  # noqa
        self, separate_iodmas: bool, dataflow_partition_directory: Path, verbosity: MFVerbosity
    ) -> None:
        """Create one SDP per all consecutive layers.

        Arguments:
        ---------
            `separate_iodmas`: If true, IODMA nodes (if detected) receive their own partition
                ID / SDP. Their device ID stays unchanged.

            `dataflow_partition_directory`: Directory in which to store logs and kernel partitions.

            `verbosity`: How verbose the transform should be.
        """
        super().__init__()
        self.verbosity = verbosity
        self.separate_iodmas = separate_iodmas
        self.cdfp_dir = dataflow_partition_directory

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Create a MultiFPGA SDP."""
        # See "create_dataflow_partition.py". Should not modify the model if
        # not necessary
        model = model.transform(ExternalizeParams())

        # Cluster the partition IDs
        model = model.transform(
            ClusterByNodeattribute(
                resolve_circular_dependencies=True,
                compare_attribute="device_id",
                partition_attribute="partition_id",
            )
        )

        # Create separate IODMAs
        all_ids = [getCustomOp(node).get_nodeattr("partition_id") for node in model.graph.node]
        if self.separate_iodmas:
            for node in model.graph.node:
                if node.op_type == "IODMA_hls":
                    current_max = max(all_ids)
                    getCustomOp(node).set_nodeattr("partition_id", current_max + 1)
                    all_ids.append(current_max + 1)
        if self.verbosity.value > MFVerbosity.MEDIUM.value:
            log.info(
                f"Clustered the graph into {len(set(all_ids))} partitions according to device ID."
            )

        # Create the SDFPs
        model = model.transform(CreateDataflowPartition(str(self.cdfp_dir)))
        model = model.transform(GiveUniqueNodeNames())

        # Set the SDP's device_id
        for node in model.graph.node:
            device_id = get_device_id(get_submodel(node)[0].graph.node[0])
            assert device_id is not None
            set_device_id(node, device_id)
        return model, False
