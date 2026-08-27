"""Creation of network metadata from the model. Implementation details are
managed by the NetworkMetadata objects."""
from __future__ import annotations

from pathlib import Path
from qonnx.transformation.base import Transformation
from typing import TYPE_CHECKING, Final

from finn.builder.build_dataflow_config import MFCommunicationKernel, MFVerbosity
from finn.transformation.fpgadataflow.multifpga.aurora.metadata import AuroraNetworkMetadata
from finn.util.basic import make_build_dir
from finn.util.exception import FINNMultiFPGAError
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
        using `NetworkMetadata.from_model(...)`.
    """

    COMMUNICATION_KERNEL_METADATA_MAP: Final[dict[MFCommunicationKernel, type[NetworkMetadata]]] = {
        MFCommunicationKernel.AURORA: AuroraNetworkMetadata
    }

    def __init__(
        self,
        communication_kernel: MFCommunicationKernel,
        verbosity: MFVerbosity,
    ) -> None:
        """Create a metadata object for the given communication kernel."""
        super().__init__()
        self.verbosity = verbosity
        try:
            self.metadata_type: type[NetworkMetadata] = self.COMMUNICATION_KERNEL_METADATA_MAP[
                communication_kernel
            ]
        except KeyError as e:
            raise FINNMultiFPGAError(
                f"Communication kernel type {communication_kernel.name} "
                f"does not yet have an associated metadata class."
            ) from e

        # Create the empty metadata object
        self.metadata: NetworkMetadata = self.metadata_type()

    def save_metadata(self, model: ModelWrapper, suffix: str = "yaml") -> Path:
        """Save the metadata and store the path as a metadata prop (`network_metadata`)
        in the modelwrapper instance.
        """
        metadata_dir = Path(make_build_dir("network_metadata_")).absolute()
        metadata_path = metadata_dir / ("metadata." + suffix)
        self.metadata.save(metadata_path)
        model.set_metadata_prop("network_metadata", str(metadata_path))
        return metadata_path

    def create_metadata(self, model: ModelWrapper) -> None:
        """Walk the graph. Any time a change in devices between SDP nodes is recognized,
        this connection is added to the metadata object.
        """
        for node in model.graph.node:
            if node.op_type != "StreamingDataflowPartition":
                raise FINNMultiFPGAError(
                    f"Cannot create metadata for model: node {node.name} is "
                    f"not a StreamingDataflowPartition. Make sure to "
                    f"run CreateMultiFPGAStreamingDataflowPartition first."
                )

        for node in model.graph.node:
            suc = model.find_direct_successors(node)
            if suc is None:
                continue
            d1 = get_device_id(node)
            if d1 is None:
                raise FINNMultiFPGAError(
                    f"Node {node.name} has no device ID. "
                    f"Make sure to partition the model "
                    f"before creating the metadata."
                )
            for s in suc:
                d2 = get_device_id(s)
                if d2 is None:
                    raise FINNMultiFPGAError(
                        f"Node {s.name} has no device ID. "
                        f"Make sure to partition the model "
                        f"before creating the metadata."
                    )
                if d1 != d2:
                    if self.verbosity.value > MFVerbosity.MEDIUM.value:
                        log.info(f"Adding connection:  {node.name} [{d1}] ----> {s.name} [{d2}]")
                    self.metadata.add_connection(d1, node.name, d2, s.name)

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Create and save the metadata from the given modelwrapper."""
        self.create_metadata(model)
        self.save_metadata(model)
        return model, False
