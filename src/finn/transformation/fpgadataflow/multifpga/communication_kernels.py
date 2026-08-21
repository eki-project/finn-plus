"""Prepare communication kernels for usage (e.g. packaging IP cores)."""
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

from finn.builder.build_dataflow_config import MFCommunicationKernel, PartitioningConfiguration
from finn.transformation.fpgadataflow.multifpga.aurora.prepare_aurora import PrepareAuroraFlow
from finn.util.exception import FINNMultiFPGAUserError


class PrepareCommunicationKernels(Transformation):
    """Prepare the communication kernels for the given type. Raises an error if the communication
    kernel is unknown or not implemented. This functions as a router to the correct transform and
    prevents the user from having to differentiate it by themselves.
    """

    def __init__(self, platform: str, fpga_part: str, pcfg: PartitioningConfiguration) -> None:
        """Prepare all communication kernels."""
        super().__init__()
        self.platform = platform
        self.fpga_part = fpga_part
        self.pcfg = pcfg

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Prepare all communication kernels."""
        match self.pcfg.communication_kernel:
            case MFCommunicationKernel.AURORA:
                model = model.transform(PrepareAuroraFlow(self.platform, self.fpga_part, self.pcfg))
            case _:
                kernelname = self.pcfg.communication_kernel.name
                raise FINNMultiFPGAUserError(
                    f"Could not prepare kernels of "
                    f"type: {kernelname}, since no "
                    f"preparation transformation for this "
                    f"kind of kernel exists yet."
                )
        return model, False
