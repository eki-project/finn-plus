"""Transformation to modify an existing link config to use AuroraFlow cores."""
import subprocess
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from typing import cast

from finn.transformation.fpgadataflow.multifpga.aurora.metadata import AuroraNetworkMetadata
from finn.transformation.fpgadataflow.multifpga.metadata import DataDirection
from finn.transformation.fpgadataflow.vitis_linking_configuration import VitisLinkConfiguration
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.fpgadataflow import get_device_id
from finn.util.logging import log
from finn.util.platforms import Platform, platforms
from finn.util.settings import get_settings


class AddAuroraToLinkConfig(Transformation):
    """Iterate over an existing prepared linking configuration, adding AuroraFlow kernels and
    connecting them to the existing SDP kernels.
    """

    def __init__(self, platform_name: str) -> None:
        """Iterate over an existing prepared linking configuration, adding AuroraFlow kernels and
        connecting them to the existing SDP kernels.
        """
        super().__init__()
        self.platform: Platform = platforms[platform_name]()

    def package_dummy_kernels(self) -> tuple[Path, Path]:
        """Prepare dummy kernels that might be needed when a kernel is in duplex mode
        but only needs one connected port. Returns a tuple containing the path to
        the RX kernel .xo and the TX kernel .xo.
        """
        # TODO: Replace with unidirectional aurora
        dummy_kernel_dir = get_settings().finn_deps / "vitis_dummy_kernel"
        rx_dummy = dummy_kernel_dir / "rx_dummy_kernel.xo"
        tx_dummy = dummy_kernel_dir / "tx_dummy_kernel.xo"
        if not rx_dummy.exists() or not tx_dummy.exists():
            subprocess.run(["make"], cwd=dummy_kernel_dir, stdout=subprocess.DEVNULL)
        return rx_dummy, tx_dummy

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Modify the link config."""
        metadata = AuroraNetworkMetadata.from_model(model)
        configs = VitisLinkConfiguration.load_from_model(model)

        # Packaging dummy kernels
        rx_dummy, tx_dummy = self.package_dummy_kernels()
        dummy_per_device: dict[int, int] = {}

        # Loop through SDPs to determine which are connected to Aurora kernels
        for node in model.graph.node:
            device = cast("int", get_device_id(node))
            if device not in dummy_per_device.keys():
                dummy_per_device[device] = 0

            # Loop through all auroras
            for index, aurora_data in enumerate(metadata[device]):
                if aurora_data is None:
                    raise FINNInternalError(
                        f"Aurora metadata for device " f"{device} is completely missing."
                    )
                if aurora_data.aurora_xo is None:
                    raise FINNInternalError(
                        f"Aurora metadata for device {device} "
                        f"is incomplete: missing XO path for kernel {index}!"
                    )

                # Check that the metadata is complete with regard to the name of the
                # sending and receiving kernels
                tx_kernel_pair = aurora_data.connecting_kernels[DataDirection.TX]
                rx_kernel_pair = aurora_data.connecting_kernels[DataDirection.RX]

                # Name for the current CU
                aurora_cu = f"aurora_flow_{index}"

                # These have to be done regardless of direction
                if (tx_kernel_pair is not None and tx_kernel_pair[0] == node.name) or (
                    rx_kernel_pair is not None and rx_kernel_pair[0] == node.name
                ):
                    if self.platform.qsfp_slr is None:
                        raise FINNUserError(
                            f"Cannot place AuroraFlow kernels on device with platform"
                            f"{type(self.platform).__name__} because expected SLR placement "
                            f"of the kernel is not known. This means that the selected "
                            f"platform either does not have any QSFP ports, or that their location "
                            f"on the platforms SLRs is missing from the platform definitions "
                            f"(check platforms.py)."
                        )
                    configs[device].add_xo(aurora_data.aurora_xo)
                    configs[device].add_cu(aurora_cu, aurora_cu)
                    configs[device].add_slr(aurora_cu, self.platform.qsfp_slr)
                    configs[device].add_connect(
                        f"io_clk_qsfp{index}_refclkb_00", f"{aurora_cu}/gt_refclk_{index}"
                    )
                    configs[device].add_connect(
                        f"aurora_flow_{index}/gt_port", f"io_gt_qsfp{index}_00"
                    )
                    configs[device].add_connect(
                        f"aurora_flow_{index}/init_clk", "ii_level0_wire/ulp_m_aclk_freerun_ref_00"
                    )

                # SDP -> Aurora -> Network
                if tx_kernel_pair is not None and tx_kernel_pair[0] == node.name:
                    log.info(
                        f"Adding AuroraFlow kernel to device {device}, "
                        f"index {index} connected to {node.name} (TX)."
                    )
                    configs[device].add_sc(node.name + ".m_axis_0", f"{aurora_cu}.tx_axis")

                    # Check if we need a dummy kernel for the unused RX direction
                    if rx_kernel_pair is None:
                        configs[device].add_xo(rx_dummy)
                        dummy_cu = f"vdk_{dummy_per_device[device]}"
                        configs[device].add_cu("rx_dummy_kernel", dummy_cu)
                        configs[device].add_sc(aurora_cu + ".rx_axis", dummy_cu + ".A")
                        dummy_per_device[device] += 1

                # Network -> Aurora -> SDP
                if rx_kernel_pair is not None and rx_kernel_pair[0] == node.name:
                    log.info(
                        f"Adding AuroraFlow kernel to device {device}, "
                        f"index {index} connected to {node.name} (RX)."
                    )
                    configs[device].add_sc(f"{aurora_cu}.rx_axis", node.name + ".s_axis_0")

                    # Check if we need a dummy kernel for the unused TX direction
                    if tx_kernel_pair is None:
                        configs[device].add_xo(tx_dummy)
                        dummy_cu = f"vdk_{dummy_per_device[device]}"
                        configs[device].add_cu("tx_dummy_kernel", dummy_cu)
                        configs[device].add_sc(dummy_cu + ".A", aurora_cu + ".tx_axis")
                        dummy_per_device[device] += 1

        for config in configs.values():
            config.generate_config()
            config.generate_run_script()
        model = VitisLinkConfiguration.store_to_model(configs, model)
        return model, False
