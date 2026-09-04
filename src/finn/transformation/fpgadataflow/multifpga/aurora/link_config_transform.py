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
from finn.transformation.fpgadataflow.multifpga.metadata import DataDirection
from finn.transformation.fpgadataflow.vitis_linking_configuration import VitisLinkConfiguration
from finn.util.basic import make_build_dir
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.fpgadataflow import get_device_id
from finn.util.logging import log
from finn.util.platforms import Platform, platforms


class AddAuroraToLinkConfig(Transformation):
    """Iterate over an existing prepared linking configuration, adding AuroraFlow kernels and
    connecting them to the existing SDP kernels.
    """

    def __init__(self, platform_name: str, fpga_part: str) -> None:
        """Iterate over an existing prepared linking configuration, adding AuroraFlow kernels and
        connecting them to the existing SDP kernels.
        """
        super().__init__()
        self.platform: Platform = platforms[platform_name]()
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
                f"vitis_hls -f run.tcl {self.part} 2.5 "
                f"rx_dummy_kernel . {width} {dummy_templated}"
            ),
            capture_output=True,
            text=True,
            cwd=dummy_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if rx_result.returncode != 0:
            raise FINNInternalError(
                f"There was an error building the RX vitis dummy kernel for "
                f"Aurora: \n{rx_result.stdout}\n{rx_result.stderr}"
            )
        tx_result = subprocess.run(
            shlex.split(
                f"vitis_hls -f run.tcl {self.part} 2.5 "
                f"tx_dummy_kernel . {width} {dummy_templated}"
            ),
            capture_output=True,
            text=True,
            cwd=dummy_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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
                        f"Aurora metadata for device {device} is completely missing."
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
