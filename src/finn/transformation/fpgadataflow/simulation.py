"""Manage FINN simulation variants."""
import onnx
import os
from concurrent.futures import Future, ProcessPoolExecutor
from copy import deepcopy
from onnx import NodeProto, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from typing import TYPE_CHECKING, Any, cast

from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.set_fifo_depths import xsi_fifosim
from finn.util.exception import FINNInternalError

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class Simulation:
    """Manage simulations in FINN."""

    def __init__(self, model: ModelWrapper, fpgapart: str, clk_ns: float) -> None:
        """Create a new simulation instance."""
        self.model = model
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns

    def _isolated_node_model(self, by_node: int | str | NodeProto) -> ModelWrapper:
        """Return a modelwrapper that has only the specified node.

        Args:
            by_node: If int, used as the index of the specified node. If string, assumed to be
                        the name of the node.

        Returns:
            ModelWrapper: The isolated-node modelwrapper.
        """
        # Find the node
        index = 0
        if type(by_node) is int:
            if by_node < 0 or by_node >= len(self.model.graph.node):
                raise FINNInternalError(
                    f"Cannot isolate node index {by_node}. Model has"
                    f"{len(self.model.graph.node)} nodes."
                )
            index = by_node
        elif type(by_node) is str:
            node_name = self.model.get_node_from_name(by_node)
            if node_name is None:
                raise FINNInternalError(f"Cannot isolate node {by_node}. No such node found.")
            index = [n.name for n in self.model.graph.node].index(cast("str", node_name))
        elif type(by_node) is NodeProto:
            try:
                index = self.model.graph.node.index(by_node)
            except Exception as e:
                raise FINNInternalError(f"Node {by_node.name} not found in the model.") from e
        else:
            raise FINNInternalError(
                f"Cannot find node to isolate: {by_node}. Specify either "
                f"the index (int), node name (str) or the object itself "
                f"(NodeProto)."
            )

        # Copy model to modify
        node_model = deepcopy(self.model)

        # Remove any other node
        # TODO: Refactor this following section
        for i, node in enumerate(self.model.graph.node):
            if i != index:
                node_model.graph.node.remove(node)
        target_op: HWCustomOp = getCustomOp(self.model.graph.node[0])
        inp = onnx.helper.make_tensor_value_info(
            "inp", TensorProto.FLOAT, target_op.get_folded_input_shape()
        )
        outp = onnx.helper.make_tensor_value_info(
            "outp", TensorProto.FLOAT, target_op.get_normal_output_shape()
        )

        # Remove old io
        for _ in range(len(node_model.graph.node[0].input)):
            node_model.graph.node[0].input.pop()
        for _ in range(len(node_model.graph.node[0].output)):
            node_model.graph.node[0].output.pop()

        # Set new io
        node_model.graph.node[0].input.append("inp")
        node_model.graph.node[0].output.append("outp")

        # Remove graph io
        for _ in range(len(node_model.graph.input)):
            node_model.graph.input.pop()
        for _ in range(len(node_model.graph.output)):
            node_model.graph.output.pop()

        # Set new graph io
        node_model.graph.input.append(inp)
        node_model.graph.output.append(outp)

        return node_model

    def run_sim_node_parallel_isolated(self, inputs: int) -> dict[int, Any]:
        """Simulate the given number of inputs for every layer. Layers are completely isolated
        and simulated in parallel.
        """

        def _run_simulation(node_index: int) -> Any:
            nodemodel = self._isolated_node_model(node_index)
            nodemodel = nodemodel.transform(CreateStitchedIP(self.fpgapart, self.clk_ns))
            # TODO: Remove xsi_fifosim from set_fifo_depths.py / change simulation functions
            return xsi_fifosim(nodemodel, inputs)

        workers = int(os.environ["NUM_DEFAULT_WORKERS"])
        futures: list[Future] = []
        results = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i in range(len(self.model.graph.node)):
                futures.append(pool.submit(_run_simulation, i))
            pool.shutdown(wait=True)
            for i, future in enumerate(futures):
                results[i] = future.result()
        return results

    def run_sim_node_parallel_connected(self, inputs: int) -> Any:
        """Simulate a whole model, with all layers simulated in parallel."""
        # TODO: Enable control through either Python or a seperate C++ driver
        raise NotImplementedError()

    def run_sim_complete(self) -> Any:
        raise NotImplementedError()

    def run_sim_single_node(self, node: Any) -> Any:
        raise NotImplementedError()
