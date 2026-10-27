"""Transformation to modify an existing link config to use AuroraFlow cores."""
import jinja2
import shlex
import shutil
import subprocess
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from typing import cast

from finn.transformation.fpgadataflow.multifpga.aurora.metadata import AuroraNetworkMetadata
from finn.transformation.fpgadataflow.vitis_linking_configuration import VitisLinkConfiguration
from finn.util.basic import make_build_dir
from finn.util.contracts import (
    ensure_device_id_set,
    ensure_metadata_prop,
    ensure_only_sdp_nodes,
    ensure_unique_node_names,
)
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.fpgadataflow import get_device_id
from finn.util.logging import log
from finn.util.platforms import Platform, platforms


class AddAuroraToLinkConfig(Transformation):
    """Iterate over an existing prepared linking configuration, adding AuroraFlow kernels and
    connecting them to the existing SDP kernels.
    """

    def __init__(self, board: str, fpga_part: str) -> None:
        """Iterate over an existing prepared linking configuration, adding AuroraFlow kernels and
        connecting them to the existing SDP kernels.
        """
        super().__init__()
        self.platform: Platform = platforms[board]()
        self.part = fpga_part
        self.rx_dummy = None
        self.tx_dummy = None

    def package_dummy_kernels(self) -> tuple[Path, Path]:
        """Prepare dummy kernels that might be needed when a kernel is in duplex mode
        but only needs one connected port. Returns a tuple containing the path to
        the RX kernel .xo and the TX kernel .xo.
        """
        width = 64
        # TODO: Replace with unidirectional aurora
        if self.rx_dummy is not None and self.tx_dummy is not None:
            return self.rx_dummy, self.tx_dummy

        # Only build once per flow
        source_dir = Path(__file__).parent / "dummy_kernel"
        dummy_dir = Path(make_build_dir("vitis_dummy_kernel_"))
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(source_dir))
        dummy_templated = dummy_dir / "main.cpp"

        # Writing the templated dummy. If the WIDTH does not match, a DWC will be inserted.
        # Also copy over the tcl script
        dummy_templated.write_text(
            env.get_template("dummy_kernel_template.cpp.jinja2").render(WIDTH=width)
        )
        shutil.copy(source_dir / "create_dummy_kernel.tcl", dummy_dir)

        # Package the kernels
        rx_dummy = dummy_dir / "rx_dummy_kernel.xo"
        tx_dummy = dummy_dir / "tx_dummy_kernel.xo"
        rx_result = subprocess.run(
            shlex.split(
                f"vitis_hls -f create_dummy_kernel.tcl {self.part} 2.5 "
                f"rx_dummy_kernel . {width} {dummy_templated}"
            ),
            capture_output=True,
            text=True,
            cwd=dummy_dir,
        )
        if rx_result.returncode != 0:
            raise FINNInternalError(
                f"There was an error building the RX vitis dummy kernel for "
                f"Aurora: \n{rx_result.stdout}\n{rx_result.stderr}"
            )
        tx_result = subprocess.run(
            shlex.split(
                f"vitis_hls -f create_dummy_kernel.tcl {self.part} 2.5 "
                f"tx_dummy_kernel . {width} {dummy_templated}"
            ),
            capture_output=True,
            text=True,
            cwd=dummy_dir,
        )
        if tx_result.returncode != 0:
            raise FINNInternalError(
                f"There was an error building the TX vitis dummy kernel for "
                f"Aurora: \n{tx_result.stdout}\n{tx_result.stderr}"
            )
        if not rx_dummy.exists():
            raise FINNInternalError(f"RX vitis dummy kernel not found at: {rx_dummy}")
        if not tx_dummy.exists():
            raise FINNInternalError(f"TX vitis dummy kernel not found at: {tx_dummy}")
        self.rx_dummy = rx_dummy
        self.tx_dummy = tx_dummy
        return self.rx_dummy, self.tx_dummy

    @ensure_only_sdp_nodes
    @ensure_device_id_set
    @ensure_unique_node_names
    @ensure_metadata_prop("network_metadata")
    @ensure_metadata_prop("vitis_link_configs")
    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Modify the link config."""
        metadata = AuroraNetworkMetadata.load_from_model(model)
        configs = VitisLinkConfiguration.load_from_model(model)

        # Packaging dummy kernels
        rx_dummy, tx_dummy = self.package_dummy_kernels()
        dummy_per_device: dict[int, int] = {}

        # An Aurora kernel can be shared by two SDPs on the same device (one using TX, the other
        # RX), so track which (device, index) kernels were already instantiated
        instantiated_kernels: set[tuple[int, int]] = set()

        # Loop through SDPs to determine which are connected to Aurora kernels
        for node in model.graph.node:
            device = cast("int", get_device_id(node))
            if device not in dummy_per_device.keys():
                dummy_per_device[device] = 0

            # Loop through all auroras
            for aurora_data in metadata[device]:
                if aurora_data is None:
                    raise FINNInternalError(
                        f"Aurora metadata for device {device} is completely missing."
                    )
                # The QSFP port of the endpoint. Its position in the metadata list may differ.
                index = aurora_data.link_index
                if index is None:
                    raise FINNInternalError(
                        f"Aurora metadata for device {device} is incomplete: missing link index "
                        f"of the endpoint connected to device {aurora_data.partner_device}!"
                    )
                if aurora_data.aurora_xo is None:
                    raise FINNInternalError(
                        f"Aurora metadata for device {device} "
                        f"is incomplete: missing XO path for kernel {index}!"
                    )

                # Whether this SDP sends or receives through this Aurora kernel
                tx = aurora_data.tx
                rx = aurora_data.rx
                is_sender = tx is not None and tx.source_kernel == node.name
                is_receiver = rx is not None and rx.target_kernel == node.name

                # Name for the current CU
                aurora_cu = f"aurora_flow_{index}"

                # Check if the SDP node is a sender or receiver (or neither)
                if (device, index) not in instantiated_kernels and (is_sender or is_receiver):
                    instantiated_kernels.add((device, index))
                    if self.platform.qsfp_slr is None:
                        raise FINNUserError(
                            f"Cannot place AuroraFlow kernels on device with platform"
                            f"{type(self.platform).__name__} because expected SLR placement "
                            f"of the kernel is not known. This means that the selected "
                            f"platform either does not have any QSFP ports, or that their location "
                            f"on the platforms SLRs is missing from the platform definitions "
                            f"(check platforms.py)."
                        )
                    if self.platform.freerunning_clock is None:
                        raise FINNUserError(
                            f"Cannot connect AuroraFlow kernels on device with platform "
                            f"{type(self.platform).__name__} because the name of its "
                            f"free-running clock is not known. It is missing from the platform "
                            f"definitions (check platforms.py)."
                        )
                    configs[device].add_xo(aurora_data.aurora_xo)
                    configs[device].add_cu(aurora_cu, aurora_cu)
                    configs[device].add_slr(aurora_cu, self.platform.qsfp_slr)
                    configs[device].add_connect(
                        self.platform.get_qsfp_refclk_interface(index),
                        f"{aurora_cu}/gt_refclk_{index}",
                    )
                    configs[device].add_connect(
                        f"{aurora_cu}/gt_port", self.platform.get_qsfp_gt_interface(index)
                    )
                    configs[device].add_connect(
                        f"{aurora_cu}/init_clk", self.platform.freerunning_clock
                    )

                # SDP -> Aurora -> Network
                if is_sender:
                    log.info(
                        f"Adding AuroraFlow kernel to device {device}, "
                        f"index {index} connected to {node.name} (TX)."
                    )
                    sdp_output = f"{node.name}.m_axis_{tx.source_kernel_port_index}"  # type: ignore
                    if configs[device].is_stream_port_connected(sdp_output):
                        raise FINNInternalError(
                            f"Cannot connect AuroraFlow kernel {index} on device {device} to "
                            f"{sdp_output}, since this port is already connected to another "
                            f"kernel."
                        )
                    configs[device].add_sc(sdp_output, f"{aurora_cu}.tx_axis")

                    # Check if we need a dummy kernel for the unused RX direction
                    if rx is None:
                        configs[device].add_xo(rx_dummy)
                        dummy_cu = f"vdk_{dummy_per_device[device]}"
                        configs[device].add_cu("rx_dummy_kernel", dummy_cu)
                        configs[device].add_sc(aurora_cu + ".rx_axis", dummy_cu + ".A")
                        dummy_per_device[device] += 1

                # Network -> Aurora -> SDP
                if is_receiver:
                    log.info(
                        f"Adding AuroraFlow kernel to device {device}, "
                        f"index {index} connected to {node.name} (RX)."
                    )
                    sdp_input = f"{node.name}.s_axis_{rx.target_kernel_port_index}"  # type: ignore
                    if configs[device].is_stream_port_connected(sdp_input):
                        raise FINNInternalError(
                            f"Cannot connect AuroraFlow kernel {index} on device {device} to "
                            f"{sdp_input}, since this port is already connected to another "
                            f"kernel."
                        )
                    configs[device].add_sc(f"{aurora_cu}.rx_axis", sdp_input)

                    # Check if we need a dummy kernel for the unused TX direction
                    if tx is None:
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
