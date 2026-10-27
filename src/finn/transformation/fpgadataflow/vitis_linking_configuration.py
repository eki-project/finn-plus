"""Dataclass to store a linking configuration for vitis builds. This is essential to enable multiple
transformations / steps to edit the same configuration.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from mashumaro.mixins.yaml import DataClassYAMLMixin
from pathlib import Path
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from typing import TYPE_CHECKING, Any, cast

from finn.builder.build_dataflow_config import FpgaMemoryType, VitisOptStrategy
from finn.templates import get_jinja_environment
from finn.util.basic import get_metadata_prop_path, make_build_dir
from finn.util.contracts import (
    ensure_device_id_set,
    ensure_has_xo,
    ensure_only_sdp_nodes,
    ensure_unambiguous_stream_ports,
    ensure_unique_node_names,
)
from finn.util.exception import FINNInternalError, FINNUserError, FINNVitisLinkConfigError
from finn.util.fpgadataflow import (
    get_connecting_tensor_indices,
    get_device_id,
    get_submodel,
    get_vitis_xo,
)
from finn.util.logging import log

if TYPE_CHECKING:
    from onnx import NodeProto
    from qonnx.core.modelwrapper import ModelWrapper


CU_PORT_REGEX = re.compile(r"\w+\.\w+")
"""Regex object to test whether the given CU includes a port."""

SYSTEM_PORT_REGEX = re.compile(r"^(HOST(\[\d+(:\d+)?\])?|(DDR|HBM|PLRAM)(\[\d+(:\d+)?\]))$")
"""Regex object to test whether a system port tag (sp) looks correct."""


@dataclass
class VitisLinkConfiguration(DataClassYAMLMixin):
    """Manages XO files, CU instantiations, stream connections,
    port connections, Vivado props, etc.
    It can output a linking configuration to pass to v++ and
    create a shell script to run it. Tries to be as strict and careful as possible,
    and depending on the issue raises an Exception, logs an error or warning
    or continues silently.
    """

    config_path: Path
    f_mhz: int
    optimization_level: str
    platform: str
    run_script_path: Path = field(init=False, default_factory=lambda: Path())
    gen_report_xml_path: Path = field(init=False, default_factory=lambda: Path())
    cu: list[str] = field(default_factory=list)
    nk: list[tuple[str, str]] = field(default_factory=list)
    sc: dict[str, str] = field(default_factory=dict)
    sp: dict[str, str] = field(default_factory=dict)
    xo: list[Path] = field(default_factory=list)
    slr: dict[str, str] = field(default_factory=dict)
    connects: list[tuple[str, str]] = field(default_factory=list)
    vivado_section: str = ""
    connectivity_section: str = ""

    def __post_init__(self) -> None:
        """Make sure all paths exist after initializing the config object."""
        self.config_path.parent.mkdir(exist_ok=True, parents=True)
        self.run_script_path = self.config_path.parent / "run_link.sh"
        self.run_script_path.parent.mkdir(exist_ok=True, parents=True)
        self.gen_report_xml_path = self.config_path.parent / "vitis_gen_report_xml.tcl"

    def store(self, p: Path) -> None:
        """Store the config as a YAML file, so that it can be loaded
        by other transformations again.
        """
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(self.to_yaml()))

    @staticmethod
    def load(p: Path) -> VitisLinkConfiguration:
        """Load a VitisLinkConfiguration from the given YAML file."""
        return VitisLinkConfiguration.from_yaml(p.read_text())

    @staticmethod
    def load_from_model(model: ModelWrapper) -> dict[int, VitisLinkConfiguration]:
        """Load all VitisLinkConfigurations from a modelwrapper. The path to this
        directory should be stored in the "vitis_link_configs" metadata prop.
        The function expects the configs to be stored in directories
            `link_configs/0/config.yaml`
        where 0 is the device ID and link_configs the stored path.

        Returns
        -------
            dict[int, VitisLinkConfiguration]: Maps device-IDs to their respective
                linking configurations.
        """
        storage_path = get_metadata_prop_path(
            model,
            "vitis_link_configs",
            must_exist=True,
            custom_error=(
                "Cannot load VitisLinkConfig from model, "
                "since the metadata prop "
                "'vitis_link_configs' was not found!"
            ),
        )

        configs = {}
        for device_path in storage_path.iterdir():
            configs[int(str(device_path.name))] = VitisLinkConfiguration.load(
                storage_path / device_path / "config.yaml"
            )
        return configs

    @staticmethod
    def store_to_model(  # noqa
        configs: dict[int, VitisLinkConfiguration], model: ModelWrapper
    ) -> ModelWrapper:
        """Store all VitisLinkConfigurations into a modelwrapper. The path to this
        directory will be stored in the "vitis_link_configs" metadata prop.
        The function stores the configs in directories
            `link_configs/0/config.yaml`
        where 0 is the device ID and link_configs the stored path.

        >>> from tests.fpgadataflow.vitisbuild.test_vitis_build import get_empty_modelwrapper
        >>> config = VitisLinkConfiguration(Path("conf.txt"), 100, "", "platform")
        >>> model = VitisLinkConfiguration.store_to_model({0: config}, get_empty_modelwrapper())
        >>> path = Path(model.get_metadata_prop("vitis_link_configs"))
        >>> path.exists()
        True
        >>> (path / "0" / "config.yaml").exists()
        True

        Arguments:
        ---------
            `configs`: The configuration objects to store.
            `model`: The model to update when the configs are stored.

        Returns:
        -------
            ModelWrapper: The modified modelwrapper with the updated metadata prop.

        """
        path = model.get_metadata_prop("vitis_link_configs")
        if path is None:
            path = make_build_dir("vitis_link_configs_")
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        model.set_metadata_prop("vitis_link_configs", str(path.absolute()))
        for device, config in configs.items():
            config.store(path / Path(str(device)) / "config.yaml")
        return model

    def add_cu(self, kernel_name: str, cu_name: str) -> None:
        """Add a compute unit (instance of a kernel).

        >>> lc = VitisLinkConfiguration(Path("conf.txt"), 100, "", "platform")
        >>> lc.add_cu("Kernel", "cu")
        >>> lc.cu
        ['cu']
        >>> lc.nk
        [('Kernel', 'cu')]
        """
        if cu_name in self.cu:
            kern = next(kname for kname, cname in self.nk if cname == cu_name)
            raise FINNVitisLinkConfigError(
                f"Tried creating CU {cu_name}, but a CU of this "
                f"name of kernel {kern} already exists!"
            )
        self.cu.append(cu_name)
        self.nk.append((kernel_name, cu_name))

    def add_sc(self, cu_sender: str, cu_receiver: str) -> None:
        """Add a Streaming Connection between two CUs:

        >>> lc = VitisLinkConfiguration(Path("conf.txt"), 100, "", "platform")
        >>> lc.add_cu("A", "a")
        >>> lc.add_cu("B", "b")
        >>> lc.add_sc("a.out", "b.in")
        >>> lc.sc["a.out"]
        'b.in'
        """  # noqa
        # Check formatting
        match = CU_PORT_REGEX.match(cu_sender)
        if match is None:
            raise FINNVitisLinkConfigError(
                f"Incorrect formatting "
                f"encountered while adding streaming "
                f"connection: {cu_sender} should match "
                f"the pattern <compute_unit_name>.<port>"
            )
        match = CU_PORT_REGEX.match(cu_receiver)
        if match is None:
            raise FINNVitisLinkConfigError(
                f"Incorrect formatting "
                f"encountered while adding streaming "
                f"connection: {cu_receiver} should match "
                f"the pattern <compute_unit_name>.<port>"
            )

        # Yield warning if the direction seems wrong
        sender_port = cu_sender.split(".")[1]
        receiver_port = cu_receiver.split(".")[1]
        if any(n in sender_port.lower() for n in ["s_axis", "in"]) or any(
            n in receiver_port.lower() for n in ["m_axis", "out"]
        ):
            log.error(
                f"Adding connection sc={cu_sender}:{cu_receiver}. The port "
                "names suggest that the order of sender and receiver might be "
                "swapped. Proceeding now."
            )

        # AXI-Stream connections are point-to-point: Every port may only be part of a single
        # streaming connection
        for port in (cu_sender, cu_receiver):
            if self.is_stream_port_connected(port):
                raise FINNVitisLinkConfigError(
                    f"Cannot add streaming connection {cu_sender}:{cu_receiver}, since port "
                    f"{port} is already part of another streaming connection."
                )

        # Add the connection
        self.sc[cu_sender] = cu_receiver

    def is_stream_port_connected(self, cu_port: str) -> bool:
        """Return whether the given port (<compute_unit_name>.<port>) is already used as sender
        or receiver of a streaming connection.

        >>> lc = VitisLinkConfiguration(Path("conf.txt"), 100, "", "platform")
        >>> lc.add_sc("a.out", "b.in")
        >>> lc.is_stream_port_connected("a.out"), lc.is_stream_port_connected("b.in")
        (True, True)
        >>> lc.is_stream_port_connected("b.out")
        False
        """
        return cu_port in self.sc or cu_port in self.sc.values()

    def add_slr(self, cu: str, slr: str) -> None:
        """Place the given CU on the given SLR."""
        self.slr[cu] = slr

    def add_sp(self, cu_port_name: str, mem_type: str) -> None:
        """Add an SP assignment.

        >>> lc = VitisLinkConfiguration(Path("conf.txt"), 100, "", "platform")
        >>> lc.add_sp("cu.arg", "HBM[0]")
        >>> lc.sp["cu.arg"]
        'HBM[0]'
        """
        match = SYSTEM_PORT_REGEX.match(mem_type)
        if match is None:
            log.warning(
                f"SP (system port) assignment {cu_port_name}:{mem_type} looks wrong. "
                f"This config may be incorrect. Continuing normally for now."
            )
        self.sp[cu_port_name] = mem_type

    def add_connect(self, a: str, b: str) -> None:
        """Add a connect assignment. Not to be confused with stream_connect (sc)."""
        self.connects.append((a, b))

    def add_vivado_line(self, line: str) -> None:
        """Add a custom line to the vivado section."""
        self.vivado_section += line + ("" if line[-1] != "\n" else "\n")

    def add_xo(self, xo_files: Path | list[Path] | str) -> None:
        """Add an XO file to the list of XO files that will be passed upon linking.
        Ignores duplicate calls. The files must exist when the method is called.
        """
        all_xos = []
        if type(xo_files) in [Path, str]:
            all_xos = [Path(xo_files)]  # type: ignore
        elif type(xo_files) is list:
            all_xos = xo_files
        else:
            all_xos = [Path(xo_files)]  # type: ignore

        for xo_file in all_xos:
            if xo_file in self.xo:
                log.warning(f"Ignoring duplicate addition of .xo: {xo_file.name}")
                continue
            if not xo_file.exists():
                raise FINNVitisLinkConfigError(
                    f"Tried adding .xo file which does not exist: {xo_file}"
                )
            self.xo.append(xo_file)

    def add_connectivity(self, txt: str) -> None:
        """Add further lines to the connectivity section. For example to assign clocks or ports."""
        self.connectivity_section += txt + ("" if txt[-1] != "\n" else "\n")

    def _get_kerneldefs(self) -> dict[str, dict[str, Any]]:
        """Use the `kernelinfo` utility to get information on all used kernels.

        Returns the kernel info indexed by kernel name. (Key is the kernel name,
        value is the `kernelinfo` utility output, read from JSON.)
        """
        infos = {}
        for xo in self.xo:
            result = subprocess.run(
                shlex.split(f"kernelinfo --json {xo}"), capture_output=True, text=True
            )
            if result.returncode != 0:
                log.warning(f"Could not load kernel definitions for xo file {xo.name}.")
                continue
            data = json.loads(result.stdout)["kernelDefs"][0]
            infos[data["name"]] = data
        return infos

    def _get_kernel_from_cu(self, cu: str) -> str:
        """Return the kernel of the given CU.

        >>> lc = VitisLinkConfiguration(Path("conf.txt"), 100, "", "platform")
        >>> lc.add_cu("Kernel", "cu")
        >>> lc._get_kernel_from_cu("cu")
        'Kernel'
        >>> lc._get_kernel_from_cu("cu2")
        Traceback (most recent call last):
            ...
        finn.util.exception.FINNInternalError: Cannot retrieve kernel of unknown CU: cu2
        """
        if cu not in self.cu:
            raise FINNInternalError(f"Cannot retrieve kernel of unknown CU: {cu}")
        for kernel, name in self.nk:
            if cu == name:
                return kernel
        raise FINNInternalError()

    def _get_ports_for_cu(self, cu: str, kerneldefs: dict[str, dict[str, Any]]) -> list[str]:
        """For the given CU and kernel definitions, find the kernel type of the
        CU and list all ports that this kernel has.
        """
        kernel = self._get_kernel_from_cu(cu)
        return [port["name"] for port in kerneldefs[kernel]["ports"]]

    def _get_used_ports_for_cu(self, cu: str) -> list[str]:
        """List all ports that are used in streaming-connections involving this CU.

        >>> lc = VitisLinkConfiguration(Path("conf.txt"), 100, "", "platform")
        >>> lc.add_cu("A", "a")
        >>> lc.add_cu("B", "b")
        >>> lc.add_cu("C", "c")
        >>> lc.add_sc("a.out", "b.in")
        >>> lc.add_sc("c.out", "a.in")
        >>> lc._get_used_ports_for_cu("a")
        ['out', 'in']
        >>> lc._get_used_ports_for_cu("b")
        ['in']
        >>> lc._get_used_ports_for_cu("c")
        ['out']
        """
        ports = []
        for send, recv in self.sc.items():
            send_name, send_port = send.split(".")
            recv_name, recv_port = recv.split(".")
            if send_name == cu:
                ports.append(send_port)
            if recv_name == cu:
                ports.append(recv_port)

        # Ports assigned to memory (implicitly used)
        ports += [name.split(".")[1] for name in self.sp.keys()]
        return ports

    def get_config_validation_errors(
        self, silent_warnings: bool = False
    ) -> None | list[FINNVitisLinkConfigError]:
        """Check the configuration and if errors are found, return them. Also prints warnings."""
        errors = []
        kerneldefs = self._get_kerneldefs()

        # Check kernel instantiations
        for kernel, cu in self.nk:
            if kernel not in kerneldefs.keys() and not silent_warnings:
                log.warning(
                    f"Kernel {kernel} (CU: {cu}) has no matching definition in "
                    f"the provided xo files. This may be caused by an error "
                    f"when loading the kernel definitons, or because you forgot "
                    f"to add the matching xo file to the configuration."
                )

        # Check connections
        for cu_sender, cu_receiver in self.sc.items():
            # Check if the sender of a streaming connection is a known CU
            sender_name, sender_port = cu_sender.split(".")
            if sender_name not in self.cu:
                errors.append(
                    FINNVitisLinkConfigError(
                        f"Streaming connection {cu_sender}:"
                        f"{cu_receiver} uses the unknown CU {sender_name}."
                    )
                )

            # Check if the receiver of a streaming connection is a known CU
            receiver_name, receiver_port = cu_receiver.split(".")
            if receiver_name not in self.cu:
                errors.append(
                    FINNVitisLinkConfigError(
                        f"Streaming connection {cu_sender}:"
                        f"{cu_receiver} uses the unknown "
                        f"CU {receiver_name}."
                    )
                )

            # Check that the ports on the CUs exist
            if not silent_warnings:
                sender_kernel = self._get_kernel_from_cu(sender_name)
                receiver_kernel = self._get_kernel_from_cu(receiver_name)
                if not any(
                    port["name"] == sender_port for port in kerneldefs[sender_kernel]["ports"]
                ):
                    log.warning(
                        f"Port {sender_port} in streaming connection {cu_sender} "
                        f"-> {cu_receiver} does not seem to exist on kernels of "
                        f"type {sender_kernel}."
                    )
                if not any(
                    port["name"] == receiver_port for port in kerneldefs[receiver_kernel]["ports"]
                ):
                    log.warning(
                        f"Port {receiver_port} in streaming connection {cu_sender} "
                        f"-> {cu_receiver} does not seem to exist on kernels of "
                        f"type {receiver_kernel}."
                    )

        # Check for unused ports
        if not silent_warnings:
            for cu in self.cu:
                available_ports = self._get_ports_for_cu(cu, kerneldefs)
                used_ports = self._get_used_ports_for_cu(cu)
                for port in available_ports:
                    if port not in used_ports and "s_axi_control" not in port.lower():
                        log.warning(f"CU {cu} has unused port {port}.")

        # No two same named CUs
        if len(set(self.cu)) != len(self.cu):
            errors.append(
                FINNVitisLinkConfigError(
                    "It seems that there are one or more CUs with the same name!"
                )
            )
        for kernel, cu in self.nk:
            for kernel2, cu2 in self.nk:
                if cu == cu2 and kernel != kernel2:
                    errors.append(
                        FINNVitisLinkConfigError(
                            f"There are 2 or more CUs named '{cu}' "
                            f"from different kernels ('{kernel}' "
                            f"and '{kernel2}')"
                        )
                    )
        if len(errors) > 0:
            return errors
        return None

    def generate_config(self) -> None:
        """Write the complete config. Raises an error if the
        config is invalid.
        """
        # Checking for errors
        errors = self.get_config_validation_errors(silent_warnings=True)
        if errors is not None:
            for i, err in enumerate(errors):
                log.error(f"[Linking Config Error {i+1} / {len(errors)}] {self.config_path}: {err}")
            if len(errors) == 1:
                # TODO: When we have switched to Python 3.11 use an exception group
                # TODO: to raise all exceptions at once, instead of one per run.
                raise errors[0]
            raise FINNVitisLinkConfigError(
                f"Multiple configuration errors occurred. First one is: {errors[0]}"
            )

        # Template rendering
        env = get_jinja_environment()
        template = env.get_template("vitis_link/link_config.txt.jinja")
        rendered = template.render(
            nk=self.nk,
            sc=self.sc,
            sp=self.sp,
            slr=self.slr,
            connects=self.connects,
            connectivity_extras=self.connectivity_section,
            vivado_extras=self.vivado_section,
        )
        self.config_path.write_text(rendered)

    def generate_report_tcl_script(self) -> None:
        """Generate a tcl script to output a resource report as an XML. Stored
        in the link directory as "vitis_gen_report_xml.tcl".
        """
        env = get_jinja_environment()
        template = env.get_template("vitis_link/vitis_gen_report_xml.tcl.jinja")
        rendered = template.render(VITIS_PROJ_PATH=str(self.config_path.parent.absolute()))
        self.gen_report_xml_path.write_text(rendered)

    def generate_run_script(self) -> None:
        """Generate a shell script to start v++ with the correct parameters.
        Produces the shell script next to the path of the config file
        unless a path is specified.
        """
        # Accumulate all xos
        xo_string = " ".join([str(xo) for xo in self.xo])

        # Check that a config for this link script exists
        if not self.config_path.exists():
            log.error(
                f"Writing compilation / v++ script for non-existing configuration "
                f"in {self.config_path.absolute()}. Continuing in case this is on purpose."
            )

        # Rendering the template
        env = get_jinja_environment()
        template = env.get_template("vitis_link/run_link.sh.jinja")
        rendered = template.render(
            platform=self.platform,
            xos=xo_string,
            config=self.config_path.absolute(),
            optimization=self.optimization_level,
            f_mhz=self.f_mhz,
        )
        self.run_script_path.write_text(rendered)


def is_iodma(model: ModelWrapper, node: NodeProto) -> bool:
    """Return whether the given SDP node is an I/O-DMA: its submodel contains only an IODMA
    node, and the SDP either has no predecessors (input) or no successors (output).
    """
    submodel, _ = get_submodel(node)
    if len(submodel.graph.node) != 1 or "IODMA" not in submodel.graph.node[0].op_type:
        return False
    pre = model.find_direct_predecessors(node)
    suc = model.find_direct_successors(node)
    return (pre is None) != (suc is None)


def make_cu_name(model: ModelWrapper, node: NodeProto) -> str:
    """Return the CU name of the given SDP node: idma_<name> or odma_<name> if the SDP is an
    I/O-DMA (see `is_iodma`), otherwise the node name.
    """
    if not is_iodma(model, node):
        return node.name
    if model.find_direct_predecessors(node) is None:
        return "idma_" + node.name
    return "odma_" + node.name


class BuildBasicVitisLinkConfig(Transformation):
    """Build basic configs for an SDP-only graph. If multiple devices are used, generate a config
    per device. The directory with all configurations is stored in the metadata
    prop `vitis_link_configs`. If a config already exists, emits an error and does nothing.

    Refer to `VitisLinkConfiguration.load_from_model` and
    `VitsLinkConfiguration.store_to_model` for more information.

    When done, the config is ready to link (for Single-FPGA).

    To find the path of the final config, one can load the config from the model,
    then check the `config_path` and `run_script_path` fields of the `VitisLinkConfiguration`.

    The config itself:
    - Creates a compute unit (CU) for every kernel (SDP).
    - SDPs containing only a DMA are named accordingly (see `make_cu_name`).
    - Determines IO memory banks based on board or SLR node attribute
    - Connects all SDP as in the graph
        - The n-th output of SDP A is A.m_axis_<n>
        - The n-th input of SDP B is B.s_axis_<n>
    """

    def __init__(
        self,
        platform: str,
        board: str,
        mem_type: FpgaMemoryType,
        vitis_opt_strategy: VitisOptStrategy,
        optimization_level: str,
        f_mhz: int,
    ) -> None:
        """Build a basic linking configuration and store it. Afterwards, the
        model can be linked, or the config can be modified further."""
        super().__init__()
        self.platform = platform
        self.board = board
        self.fpga_memory_type = mem_type
        self.optimization_level = optimization_level
        self.vitis_opt_strategy = vitis_opt_strategy
        self.f_mhz = f_mhz

    def apply_optimization_strategy(self, config: VitisLinkConfiguration) -> VitisLinkConfiguration:
        """Return the given config with the selected vitis opt strategy directives added."""
        match self.vitis_opt_strategy:
            case VitisOptStrategy.PERFORMANCE_BEST:
                config.add_vivado_line(
                    "prop=run.impl_1.STEPS.OPT_DESIGN.ARGS.DIRECTIVE=ExploreWithRemap\n"
                    "prop=run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=Explore\n"
                    "prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.IS_ENABLED=true\n"
                    "prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=Explore\n"
                    "prop=run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=Explore\n"
                )
            case _:
                log.info("No Vitis Optimization Strategy supplied.")
        return config

    def get_memory_port(self, node: NodeProto) -> tuple[str, int]:
        """Return memory type (HBM, DDR, HOST) and memory bank/index for the given node."""
        node_slr = getCustomOp(node).get_nodeattr("slr")
        if self.fpga_memory_type == FpgaMemoryType.HOST_MEM:
            return ("HOST", 0)
        match self.board.lower():
            case "u50" | "u280" | "u55c":
                return ("HBM", 0)
            case "u250":
                return ("DDR", 0 if node_slr == -1 else cast("int", node_slr))
            case _:
                raise FINNUserError(
                    f"Cannot do system-port placement for unknown board {self.board}"
                )

    @ensure_only_sdp_nodes
    @ensure_device_id_set
    @ensure_unique_node_names
    @ensure_unambiguous_stream_ports
    @ensure_has_xo
    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Build the configuration."""
        configs: dict[int, VitisLinkConfiguration] = {}

        # Check that we are the first to edit link configs
        vitis_link_configs = model.get_metadata_prop("vitis_link_configs")
        if vitis_link_configs is not None:
            log.error(
                f"Detected existing linking configurations in {vitis_link_configs}."
                "BuildBasicVitisLinkConfig should be the first "
                "transformation to create the initial configuration. "
                "No changes will be made."
            )
            return model, False

        # Create a config for every device
        for node in model.graph.node:
            device = cast("int", get_device_id(node))
            if device not in configs:
                configs[device] = VitisLinkConfiguration(
                    config_path=Path(make_build_dir(f"vitis_link_device_{device}_")) / "config.txt",
                    platform=self.platform,
                    optimization_level=self.optimization_level,
                    f_mhz=self.f_mhz,
                )

        # Already add optimization strategies for all devices
        for device in configs.keys():
            configs[device] = self.apply_optimization_strategy(configs[device])

        # Some temporary variables needed to construct the configs
        current_device: int
        cu_names: dict[str, str] = {}

        # KERNEL INSTANTIATION
        for node in model.graph.node:
            current_device = cast("int", get_device_id(node))
            cu_names[node.name] = make_cu_name(model, node)
            configs[current_device].add_xo(get_vitis_xo(node))
            configs[current_device].add_cu(node.name, cu_names[node.name])

        # KERNEL CONNECTION
        for node in model.graph.node:
            current_device = cast("int", get_device_id(node))
            successors = model.find_direct_successors(node)
            if successors is not None:
                for successor in successors:
                    if get_device_id(successor) == current_device:
                        # Connect ports by tensor input / output index.
                        # Derived from SDP port creation
                        source_index, target_index = get_connecting_tensor_indices(node, successor)
                        configs[current_device].add_sc(
                            f"{cu_names[node.name]}.m_axis_{source_index}",
                            f"{cu_names[successor.name]}.s_axis_{target_index}",
                        )

        # MEMORY BANK ASSIGNMENT (I/O-DMA only)
        for node in model.graph.node:
            if not is_iodma(model, node):
                continue
            current_device = cast("int", get_device_id(node))
            mem_type, mem_idx = self.get_memory_port(node)
            if mem_idx != 0:
                log.warning(
                    f"Overwriting expected mem_idx 0 of node {node.name} "
                    f"with specified Node SLR mem_idx: {mem_idx}"
                )
            configs[current_device].add_sp(
                cu_names[node.name] + ".m_axi_gmem0", f"{mem_type}[{mem_idx}]"
            )

        # Store everything, generate scripts and return
        model = VitisLinkConfiguration.store_to_model(configs, model)
        for config in configs.values():
            config.generate_config()
            config.generate_run_script()
            config.generate_report_tcl_script()
        return model, False
