import json
from math import floor
import os
from typing import Optional
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.util.platforms import Platform
from mip import Model, xsum

from typing import Optional

class SLRProblem:
    """Class containing the definition of the ILP created to describe the SLR partitioning. ILP defined upon creation, can be solved using optimize() and results, if available, can be retrieved via get_results()
    
    The results are in the format of a dictionary mapping the node names to their assigned SLRs"""

    def __init__(self, board: str, resources: dict[str, dict[str, int]]):
        self.results: dict[str, int] = None
        
        # Mapping node name -> resource type -> resource usage
        self.resources: dict[str, dict[str, int]] = resources

    def optimize(self) -> None:
        pass

    def get_results(self) -> Optional[dict[str, int]]:
        return self.results
        


class AssignSLR(Transformation):
    """If successful assigns the SLR node attribute of all nodes in the graph based on a solution given by an ILP solver or a handcrafted algorithm.
    This can be saved into a JSON to be read by the floorplanning transformation so that SLR assignments are possible in the vitis build flow"""

    def __init__(self, cfg: DataflowBuildConfig, platform: Platform):
        super().__init__()
        self.cfg = cfg
        self.platform = platform
    
    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        estimate_dir = os.path.join(self.cfg.output_dir, "report")
        resource_types = ["LUT", "FF", "BRAM_18k", "URAM", "DSP"]
        board_resources = self.platform.guide_resources
        
        # Check if we have resource estimates or must use the number of layers as a guide
        use_layer_count = False
        if not os.path.isfile(os.path.join(estimate_dir, "estimate_layer_resources.json")) or not os.path.isfile(os.path.join(estimate_dir, "estimate_layer_resources_hls.json")):
            use_layer_count = True
            print(f"Warning: Could not find both resource estimates in the reports folder {estimate_dir}. Defaulting to using layer count as a replacement metric for resource utilization")
        
        
        # Using the layer count is equivalent to every layer having the same amount of required resources
        layer_resources: dict[str, dict[str, int]] = {}
        if not use_layer_count:
            # Accumulate estimate results
            # Use the larger of the two estimates
            layer_resources = {}
            with open(os.path.join(estimate_dir, "estimate_layer_resources.json"), 'r') as f:
                layer_resources = json.load(f)
            with open(os.path.join(estimate_dir, "estimate_layer_resources_hls.json"), 'r') as f:
                hls_layer_resources = json.load(f)
                for layername in layer_resources.keys():
                    if layername in hls_layer_resources.keys():
                        for resource_type in layer_resources[layername].keys():
                            if resource_type in hls_layer_resources[layername].keys():
                                layer_resources[layername][resource_type] = max(layer_resources[layername][resource_type], hls_layer_resources[layername][resource_type])

        else:
            resource_per_layer = {}
            for i, restype in enumerate(resource_types):
                resource_per_layer[restype] = int(floor(sum([board_resources[slr][i] for slr in range(len(board_resources))]) / len(model.graph.node)))
        
            for node in model.graph.node:
                for restype in resource_types:
                    layer_resources[node.name][restype] = resource_per_layer[restype]
        


        # Creation of the problem statement
        slr_model = SLRProblem()

        # Optimizie
        slr_model.optimize()

        # Results
        assignments = slr_model.get_results()
        assert assignments is not None, f"SLR Assignment calculation failed"
        for node in model.graph.node:
            assert node.name in assignments.keys(), f"Could not assign SLR to {node.name}, no such key found in solution mapping!"
            getCustomOp(node).set_nodeattr("slr", int(assignments[node.name]))
        return model, False