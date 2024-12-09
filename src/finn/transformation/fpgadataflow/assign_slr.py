import json
from math import floor
import os
from typing import Optional
import mip
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

    def __init__(
            self, 
            index_to_name: dict[int, str],                      # Map node indices of the graph to layer/node names
            layer_resources: dict[str, dict[str, int]], 
            max_resources: dict[int, dict[str, int]], 
            default_input_slr: Optional[int] = None, 
            default_output_slr: Optional[int] = None, 
            fixed_layer_mapping: dict[str | int, int] = {}
        ):
        self.results: dict[str, int] = None
        
        # Maximum resources. Mapping SLR -> resource type -> resource usage
        self.max_resources = max_resources
        
        # Mapping node name -> resource type -> resource usage
        self.layer_resources: dict[str, dict[str, int]] = layer_resources
        layer_count = len(self.layer_resources.keys())

        ###### MODEL ######
        model = Model(sense=mip.MINIMIZE)

        # Main variable matrix [layer][slr] = 0/1
        layer_on_slr = {layername: [model.add_var(f"layer{layername}_on_slr{j}", var_type=mip.INTEGER) for j in range(self.slrs)] for layername in layer_resources.keys()}

        # Every layer can only be on one SLR
        for layername in layer_resources.keys():
            model += xsum([layer_on_slr[layername][slr] for slr in max_resources.keys()]) == 1

        # If we were given fixed mappings, apply them
        if default_input_slr is not None:
            model += layer_on_slr[index_to_name[0]][default_input_slr] == 1
        if default_output_slr is not None:
            model += layer_on_slr[index_to_name[-1]][default_input_slr] == 1
        for fixed_layer in fixed_layer_mapping.keys():
            if type(fixed_layer) == int:
                model += layer_on_slr[index_to_name[fixed_layer]][fixed_layer_mapping[fixed_layer]] == 1
            else:
                model += layer_on_slr[fixed_layer][fixed_layer_mapping[fixed_layer]] == 1
        
        # Which SLR is a given layer on?
        chosen_slr_of_layer = {layername: model.add_var(f"layer{layername}_chosen_slr", var_type=mip.INTEGER) for i in layer_resources.keys()}
        for i in layer_resources.keys():
            model += chosen_slr_of_layer[layername] == xsum([slr * layer_on_slr[layername][slr] for slr in range(len(max_resources.keys()))])
        
        # Consecutive layers need to be on consecutive SLRs (assumes SLRs are next to each other in order)
        slr_diff_of_layer = {layername: model.add_var(f"connections_of_layer{layername}", var_type=mip.INTEGER) for layername in layer_resources.keys()}
        for i in range(layer_count-1):
            layername = index_to_name[i]
            next_layername = index_to_name[i+1]
            model += slr_diff_of_layer[layername] >= chosen_slr_of_layer[layername] - chosen_slr_of_layer[next_layername]
            model += slr_diff_of_layer[layername] >= chosen_slr_of_layer[next_layername] - chosen_slr_of_layer[layername]
            model += slr_diff_of_layer[layername] <= 1
            model += slr_diff_of_layer[layername] >= 0

        # Minimize SLR connections
        slr_crossings = model.add_var("slr_crossings", var_type=mip.INTEGER)
        model += slr_crossings == xsum([slr_diff_of_layer[layername] for layername in layer_resources.keys()]) 
        slr_crossings_relative = model.add_var("slr_crossings_rel", var_type=mip.CONTINUOUS)
        model += slr_crossings_relative / layer_count # Max nr of crossings would be that a crossing happens from any layer to the next

        # Calculate resource usage 
        resource_per_slr = {}
        for slr in range(self.slrs):
            resource_per_slr[slr] = {}
            for resname in self.considered_resources:
                resource_per_slr[slr][resname] = model.add_var(f"resource_{resname}_on_slr{slr}", var_type=mip.INTEGER)
                model += resource_per_slr[slr][resname] == xsum([layer_on_slr[i][slr] * layer_resources[layername][resname] for layername in layer_resources.keys()])

        # Calculate resource usage relative / normalized per SLR and resource type
        resource_per_slr_rel = {}
        for slr in range(len(max_resources.keys())):
            resource_per_slr_rel[slr] = {}
            for resname in self.considered_resources:
                resource_per_slr_rel[slr][resname] = model.add_var(f"resource_{resname}_on_slr{slr}_relative", var_type=mip.CONTINUOUS)
                resource_per_slr_rel[slr][resname] = resource_per_slr[slr][resname] / max_resources[slr]
            
        # Get the highest relative utilization of any resource
        max_diff = model.add_var("max_diff", var_type=mip.CONTINUOUS)
        max_util_per_slr = [model.add_var(f"max_util_per_slr{slr}", var_type=mip.CONTINUOUS) for slr in range(self.slrs)]
        for slr in range(self.slrs):
            for resname in self.considered_resources:
                model += max_util_per_slr[slr] >= resource_per_slr_rel[slr][resname]

        # Calulcate largest difference to ideal
        for slr in range(self.slrs):
            model += max_diff >= max_util_per_slr[slr] - self.ideal_utilization
        
        # Try to minimize just that
        model.objective = slr_crossings_relative + max_diff


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
        max_resources = {}
        for slr in range(len(board_resources)):
            max_resources[slr] = board_resources[slr]
        
        index_to_name = {}
        for i, node in enumerate(model.graph.node):
            index_to_name[i] = node.name
        index_to_name[-1] = model.graph.node[-1].name

        # The problem object
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