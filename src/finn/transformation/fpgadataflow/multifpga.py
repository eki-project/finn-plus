import json
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

from finn.builder.build_dataflow_config import (
    PartitioningConfiguration,
    PartitioningStrategy,
)


class PartitionForMultiFPGA(Transformation):
    """
    Receive a model with only FPGADataflow nodes and partition it by assigning it's
    device node attribute. Partitioning is done with respect to the chosen strategy
    """

    def __init__(self, pcfg: PartitioningConfiguration) -> None:
        self.pcfg = pcfg

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        if self.pcfg.num_fpgas <= 1:
            # Do nothing, since no multi fpga is used
            return model, False

        if self.pcfg.num_fpgas > len(model.graph.node):
            # TODO: Exchange for custom error exception class
            raise Exception(
                f"Cannot partition: number of FPGAs set to \
               {self.pcfg.num_fpgas},but only {len(model.graph.node)} layers are \
               in the model! (num_fpgas <= #Layers must be true)"
            )

        # Collecting groups
        # grouped_together_nodes = None  # get_grouped_layers(model)  # TODO: Implement

        # Trying to open resource estimates
        resource_estimates = None
        raw_estimates = None
        if self.pcfg.optimization_goal == PartitioningStrategy.RESOURCE_UTILIZATION:
            re_path = Path(self.report_dir) / "estimate_layer_resources.json"
            re_hls_path = Path(self.report_dir) / "estimate_layer_resources_hls.json"

            # TODO: Custom exceptions
            if not Path.isfile(re_path):
                raise Exception(
                    f"Cannot find layer resource estimate in {re_path}. \
                  Check to make sure step_generate_estimate_reports and \
                  the corresponding output are set!"
                )
            if not Path.isfile(re_hls_path):
                raise Exception(
                    f"Cannot find layer resource estimate in {re_hls_path}. \
                  Check to make sure step_generate_estimate_reports and \
                  the corresponding output are set!"
                )
            with re_path.open("r") as f:
                raw_estimates = json.load(f)
            with re_hls_path.open("r") as f:
                raw_estimates.update(json.load(f))

            resource_estimates = []
            for node in model.graph.node:
                if node.name not in raw_estimates.keys():
                    raise Exception(
                        f"Could not find resource estimate for node \
                     {node.name} - cannot partition without estimate while \
                     using RESOURCE_UTILIZATION strategy"
                    )
                resource_estimates.append(raw_estimates[node.name])

            # Converting all estimates to int
            for i in range(len(resource_estimates)):
                for key in resource_estimates[i].keys():
                    resource_estimates[i][key] = int(resource_estimates[i][key])

        return model, False
