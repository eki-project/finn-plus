"""Wrapper class for managing a collection of DNN submodels as a multi-DNN graph."""

from collections.abc import Callable
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.custom_op.fpgadataflow.abstract.dnncontainer import DNNContainer
from finn.util.exception import FINNInternalError, FINNUserError


class MultiDNNWrapper:
    """Wraps multiple DNN models into a single multi-DNN ONNX graph with DNNContainer nodes."""

    def __init__(self, model_dict: dict[str, str | ModelWrapper]) -> None:
        """Initialize from a dict mapping submodel names to ONNX model paths."""
        model_dict = {key: ModelWrapper(value) for key, value in model_dict.items()}
        if not all(isinstance(value, ModelWrapper) for value in model_dict.values()):
            raise FINNInternalError("All multi-DNN submodels must be ModelWrapper instances")
        self.multi_model = self._create_multi_dnn_graph(model_dict)
        self._collapsed = False

    def _create_multi_dnn_graph(self, model_dict: dict[str, ModelWrapper]) -> ModelWrapper:
        """Build and return an ONNX model wrapping each submodel in a DNNContainer."""
        nodes = []
        for key, model in model_dict.items():
            model.graph.name = key
            dnn_container = helper.make_node(
                "DNNContainer",
                inputs=[],
                outputs=[],
                domain="finn.custom_op.fpgadataflow",
                body=model.graph,
                name="dnn_container_" + key,
            )

            nodes += [dnn_container]

        graph = helper.make_graph(
            nodes=nodes,
            name="Multi_DNN_Wrapper_Graph",
            inputs=[],
            outputs=[],
        )

        model = qonnx_make_model(graph, producer_name="Multi_DNN_Wrapper_Model")
        model = ModelWrapper(model)
        return model

    def apply_step(
        self,
        step: Callable[[ModelWrapper, DataflowBuildConfig], ModelWrapper],
        targets: str | list,
        cfgs: dict[str, DataflowBuildConfig],
    ) -> None:
        """Apply a build step to the specified target submodel(s) or the wrapper model."""
        # Names Multi_DNN_Wrapper and Collapsed_Model are reserved target names.
        # Do not name submodels as such
        if ("Multi_DNN_Wrapper" in targets or "Collapsed_Model" in targets) and len(targets) > 1:
            raise FINNUserError(
                "Applying steps on Multi_DNN_Wrapper or Collapsed_Model "
                "are only allowed individually"
            )

        if "Multi_DNN_Wrapper" in targets:
            if self._collapsed is True:
                raise FINNInternalError("Model is collapsed. No Wrapper available")
            self.multi_model = step(self.multi_model, cfgs[targets[0]])

            # Always check if there are any DNNContainers in the graph.
            # If not, it has been collapsed and set the flag
            collapsed = True
            for node in self.multi_model.graph.node:
                if node.op_type == "DNNContainer":
                    collapsed = False
                    break

            if collapsed:
                self._collapsed = collapsed

        elif "Collapsed_Model" in targets:
            if self._collapsed is False:
                raise FINNInternalError("Model is not collapsed")
            self.multi_model = step(self.multi_model, cfgs[targets[0]])
            # It is currently not intendet to "uncollapse"
        else:
            if self._collapsed is True:
                raise FINNInternalError("Model is collapsed. No Submodels available")

            # Apply to submodels
            for target in targets:
                self[target] = step(self[target], cfgs[target])

    def get_container_dict(self) -> dict[str, ModelWrapper]:
        """Return a dict mapping submodel names to their ModelWrapper bodies."""
        if self._collapsed:
            raise FINNInternalError("Container dict is not defined if model is collapsed.")

        models = {}
        for node in self.multi_model.graph.node:
            if node.op_type == "DNNContainer":
                customop = getCustomOp(node)
                if not isinstance(customop, DNNContainer):
                    raise FINNInternalError(
                        f"{node.name}: expected a DNNContainer op, got {type(customop).__name__}"
                    )
                submodel = customop.get_nodeattr("body")
                models[submodel.graph.name] = submodel
        return models

    def __getitem__(self, key: str) -> ModelWrapper | None:
        """Return the submodel ModelWrapper for the given submodel name."""
        if self._collapsed:
            raise FINNInternalError("Index access is not allowed if model is collapsed.")

        # QONNX is also iterating over the entire graph
        for node in self.multi_model.graph.node:
            if node.op_type == "DNNContainer" and node.name.endswith(key):
                customop = getCustomOp(node)
                if not isinstance(customop, DNNContainer):
                    raise FINNInternalError(
                        f"{node.name}: expected a DNNContainer op, got {type(customop).__name__}"
                    )
                submodel = customop.get_nodeattr("body")
                if submodel.graph.name == key:
                    return submodel
        return None

    def __setitem__(self, key: str, value: ModelWrapper) -> None:
        """Update the submodel body for the given submodel name."""
        if self._collapsed:
            raise FINNInternalError("Index access is not allowed if model is collapsed.")

        # QONNX is also iterating over the entire graph
        for node in self.multi_model.graph.node:
            if node.op_type == "DNNContainer" and node.name.endswith(key):
                customop = getCustomOp(node)
                if not isinstance(customop, DNNContainer):
                    raise FINNInternalError(
                        f"{node.name}: expected a DNNContainer op, got {type(customop).__name__}"
                    )
                submodel = customop.get_nodeattr("body")
                if submodel.graph.name == key:
                    customop.set_nodeattr("body", value)
                    break

    def __len__(self) -> int:
        """Return the number of submodels in the wrapper."""
        if self._collapsed:
            raise FINNInternalError("Length is not defined if model is collapsed.")

        return len(self.get_container_dict())
