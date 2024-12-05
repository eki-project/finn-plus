from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation


class AssignSLR(Transformation):
    """If successful assigns the SLR node attribute of all nodes in the graph based on a solution given by an ILP solver or a handcrafted algorithm.
    This can be saved into a JSON to be read by the floorplanning transformation so that SLR assignments are possible in the vitis build flow"""

    def __init__(self):
        super().__init__()
    
    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        return model, False