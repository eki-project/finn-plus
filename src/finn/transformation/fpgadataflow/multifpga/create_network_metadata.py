"""Creation of network metadata from the model. Implementation details are
managed by the NetworkMetadata objects."""
from __future__ import annotations

from pathlib import Path
from qonnx.transformation.base import Transformation
from typing import TYPE_CHECKING

from finn.builder.build_dataflow_config import MFVerbosity, MultiFPGAConfiguration
from finn.transformation.fpgadataflow.multifpga.backend import get_backend
from finn.util.basic import make_build_dir
from finn.util.contracts import (
    ensure_device_id_set,
    ensure_only_sdp_nodes,
    ensure_unique_node_names,
)
from finn.util.fpgadataflow import get_device_id
from finn.util.logging import log

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

    from finn.transformation.fpgadataflow.multifpga.metadata import NetworkMetadata


class CreateNetworkMetadata(Transformation):
    """Create the necessary Multi-FPGA metadata from the given graph.

    Requirements: All nodes must be StreamingDataflowPartitions with the node attribute `device_id`
        already set.

    Result: The metadata property `network_metadata` points to a file containing information
        needed by the communication kernel (which nodes are connected, on which devices, where the
        necessary IP cores per device lie, etc. The exact details differ by the type of
        communication kernel used. The metadata can be loaded automatically and inspected by
        using `NetworkMetadata.load_from_model(...)`.
    """

    def __init__(self, multifpga_configuration: MultiFPGAConfiguration) -> None:
        """Create a metadata object for the communication kernel given in the partitioning
        configuration. The metadata class reads any further settings it needs from the
        configuration (e.g. the number of ports per device).
        """
        super().__init__()
        self.verbosity = multifpga_configuration.verbosity
        self.metadata_type: type[NetworkMetadata] = get_backend(
            multifpga_configuration.network_backend
        ).metadata_type

        # Create the empty metadata object
        self.metadata: NetworkMetadata = self.metadata_type.create_from_multifpga_configuration(
            multifpga_configuration
        )

    def save_metadata(self, model: ModelWrapper, suffix: str = "yaml") -> Path:
        """Save the metadata and store the path as a metadata prop (`network_metadata`)
        in the modelwrapper instance.
        """
        metadata_dir = Path(make_build_dir("network_metadata_")).absolute()
        metadata_path = metadata_dir / ("metadata." + suffix)
        self.metadata.save(metadata_path)
        model.set_metadata_prop("network_metadata", str(metadata_path))
        return metadata_path

    @ensure_only_sdp_nodes
    @ensure_device_id_set
    @ensure_unique_node_names
    def create_metadata(self, model: ModelWrapper) -> None:
        """Walk the graph. Any time a change in devices between SDP nodes is recognized,
        this connection is added to the metadata object.
        """
        # Loop all nodes
        for node in model.graph.node:
            for out_tensor_index, output_tensor in enumerate(node.output):
                successors = model.find_consumers(output_tensor)
                for successor in successors:
                    d1 = get_device_id(node)
                    d2 = get_device_id(successor)
                    if d1 == d2:
                        continue
                    for in_tensor_index, input_tensor in enumerate(successor.input):
                        if input_tensor == output_tensor:
                            if self.verbosity.value > MFVerbosity.MEDIUM.value:
                                log.info(
                                    f"Adding connection:  {node.name}:{out_tensor_index} "
                                    f"[{d1}] ----> {successor.name}:{in_tensor_index} [{d2}]"
                                )
                            self.metadata.add_connection(
                                d1,  # type: ignore
                                node.name,
                                out_tensor_index,  # type: ignore
                                d2,  # type: ignore
                                successor.name,
                                in_tensor_index,  # type: ignore
                            )
                            break

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Create and save the metadata from the given modelwrapper."""
        self.create_metadata(model)
        self.save_metadata(model)
        return model, False
