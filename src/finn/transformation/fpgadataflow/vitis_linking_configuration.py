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

from finn.builder.build_dataflow_config import (
    FpgaMemoryType,
    StreamRegisterSlices,
    VitisOptStrategy,
)
from finn.templates import get_jinja_environment
from finn.transformation.fpgadataflow.floorplanning.binding import collect_floorplan_units
from finn.transformation.fpgadataflow.floorplanning.placer_retry import OVERRIDES_FILE
from finn.transformation.fpgadataflow.floorplanning.resources import (
    DEFAULT_LOGIC_UTILIZATION,
    DEFAULT_RESOURCE_MARGIN,
    DEFAULT_SLICEM_UTILIZATION,
)
from finn.transformation.fpgadataflow.floorplanning.shell_closure import load_map, vivado_version
from finn.transformation.fpgadataflow.floorplanning.tcl import render_floorplan_tcl
from finn.util.basic import get_metadata_prop_path, make_build_dir
from finn.util.exception import FINNInternalError, FINNUserError, FINNVitisLinkConfigError
from finn.util.fpgadataflow import check_all_sdp_nodes, get_device_id, get_submodel, get_vitis_xo
from finn.util.logging import log

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


CU_PORT_REGEX = re.compile(r"\w+\.\w+")
"""Regex object to test whether the given CU includes a port."""

SYSTEM_PORT_REGEX = re.compile(r"^(HOST(\[\d+(:\d+)?\])?|(DDR|HBM|PLRAM)(\[\d+(:\d+)?\]))$")
"""Regex object to test whether a system port tag (sp) looks correct."""


#: opt_design replicates the driver of any net with more loads than this inside
#: one hierarchy (``-hier_fanout_limit``). Resets and other wide control nets then
#: reach their loads from registers placed next to them.
OPT_HIER_FANOUT_LIMIT: int = 512

#: Vivado implementation run directives per `VitisOptStrategy`, as ``[vivado]`` lines of
#: the link config. DEFAULT, POWER, SIZE and BUILD_SPEED leave Vivado's defaults.
#: PERFORMANCE is the ``Explore`` set with physical optimisation switched on (what
#: PERFORMANCE_BEST used to be); PERFORMANCE_BEST is the strongest timing-driven set Vivado
#: offers: ``ExtraTimingOpt`` placement, ``AggressiveExplore`` physical optimisation and
#: routing, and post-route physical optimisation. Measured against the default on the
#: same designs (FLOORPLANNING_PLAN.md, 15 Oct 2026) rather than assumed. No opt_design
#: directive: Vivado rejects ``opt_design -directive`` together with any other option
#: ("Cannot specify additional options with '-directive'", 15 Oct 2026), and the per-SLR
#: reset handling gives opt_design ``-hier_fanout_limit`` (`OPT_HIER_FANOUT_LIMIT`), which
#: stays. (v++ ``-O3`` itself adds ``-retiming`` to place_design; that one Vivado accepts
#: next to a directive.)
IMPLEMENTATION_DIRECTIVES: dict[VitisOptStrategy, tuple[str, ...]] = {
    VitisOptStrategy.PERFORMANCE: (
        "prop=run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=Explore",
        "prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.IS_ENABLED=true",
        "prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=Explore",
        "prop=run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=Explore",
    ),
    VitisOptStrategy.PERFORMANCE_BEST: (
        "prop=run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=ExtraTimingOpt",
        "prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.IS_ENABLED=true",
        "prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=AggressiveExplore",
        "prop=run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=AggressiveExplore",
        "prop=run.impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED=true",
        "prop=run.impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE=AggressiveExplore",
    ),
}

#: ``synth_design`` options for the kernels' out-of-context synthesis per strategy. Both
#: performance strategies retime: the paths no placement can shorten on the floorplanned
#: mobilenetv1 at 200 MHz were 5-9 LUT levels of window-buffer and MVAU control logic
#: (15 Oct 2026), and moving registers across that logic is the synthesiser's job.
KERNEL_SYNTH_OPTIONS: dict[VitisOptStrategy, str] = {
    VitisOptStrategy.PERFORMANCE: "-retiming",
    VitisOptStrategy.PERFORMANCE_BEST: "-retiming",
}


def implementation_directives(strategy: VitisOptStrategy) -> tuple[str, ...]:
    """Vivado run property lines the link config gets for ``strategy`` (see the table)."""
    return IMPLEMENTATION_DIRECTIVES.get(strategy, ())


def kernel_synth_options(strategy: VitisOptStrategy) -> str:
    """Extra ``synth_design`` options for the kernels under ``strategy`` ("" for none)."""
    return KERNEL_SYNTH_OPTIONS.get(strategy, "")


#: The kernel clock pin of the Alveo dynamic platforms' clocking subsystem
#: (``<ulp>/ulp_ucs/aclk_kernel_00``); FINN's kernels run on clock 0.
KERNEL_CLOCK_PIN: str = "aclk_kernel_00"


def render_kernel_clock_tcl(f_mhz: int) -> str:
    """Tcl for the opt_design pre-hook: time the kernel clock at ``f_mhz`` in implementation.

    v++ turns ``--kernel_frequency`` into a generated clock on the kernel MMCM's
    output whose multiply/divide come from Vivado's clock-wizard solver
    (``ocl_util.tcl``, ``GetClosestSolution``). In Vivado 2023.1, 2023.2 and 2024.2
    that solver returns no solution -- 2024.2 says ``[Ip 78-110] Invalid part string
    <project>``, 2023.x fails silently -- and both values stay 1, so placement and
    routing are timed at the MMCM's input frequency (100 MHz on the U55C) whatever
    frequency was requested. 2022.2 and 2024.1 solve it (x12 / 6 for 200 MHz from 100
    MHz). Measured 16 Oct 2026 with Vitis' exact call in bare Vivado sessions; on
    2024.2 the real links show it too (every 5 ns build reported the same placer and
    router WNS as the 10 ns builds). This script re-creates the clock with the ratio
    of the requested to the input frequency, and leaves it alone when the two
    already agree (a 100 MHz build, or a Vivado whose solver works).
    """
    return "\n".join(
        [
            "# FINN+: time the kernel clock at the requested frequency in implementation",
            f"set finn_kclk_mhz {f_mhz}",
            "proc finn_kclk_gcd {a b} {",
            "    while {$b != 0} { lassign [list $b [expr {$a % $b}]] a b }",
            "    return $a",
            "}",
            "set finn_kclk_pin [get_pins -quiet -of_objects \\",
            "    [get_cells -quiet -hierarchical -filter {NAME =~ */ulp_ucs}] \\",
            f"    -filter {{REF_PIN_NAME == {KERNEL_CLOCK_PIN}}}]",
            "if {[llength $finn_kclk_pin] != 1} {",
            (
                '    puts "CRITICAL WARNING: FINN kernel clock: expected one ulp_ucs/'
                f'{KERNEL_CLOCK_PIN} pin, found [llength $finn_kclk_pin]; not changed"'
            ),
            "} else {",
            "    set finn_kclk [get_clocks -of_objects $finn_kclk_pin]",
            "    set finn_kclk_name [get_property NAME $finn_kclk]",
            "    set finn_kclk_out [lindex [get_pins [get_property SOURCE_PINS $finn_kclk]] 0]",
            "    set finn_kclk_in [get_property SOURCE $finn_kclk]",
            "    set finn_kclk_want [expr {1000.0 / $finn_kclk_mhz}]",
            "    set finn_kclk_have [get_property PERIOD $finn_kclk]",
            '    if {$finn_kclk_in eq ""} {',
            (
                '        puts "CRITICAL WARNING: FINN kernel clock: $finn_kclk_name '
                'is not a generated clock; not changed"'
            ),
            "    } elseif {abs($finn_kclk_have - $finn_kclk_want) < 0.001} {",
            '        puts "FINN kernel clock: $finn_kclk_name already at $finn_kclk_have ns"',
            "    } else {",
            (
                "        set finn_kclk_in_period [get_property PERIOD [get_clocks -of_objects "
                "[get_pins $finn_kclk_in]]]"
            ),
            "        set finn_kclk_m [expr {int(round($finn_kclk_mhz * 1000.0))}]",
            "        set finn_kclk_d [expr {int(round(1000000.0 / $finn_kclk_in_period))}]",
            "        set finn_kclk_g [finn_kclk_gcd $finn_kclk_m $finn_kclk_d]",
            "        set finn_kclk_m [expr {$finn_kclk_m / $finn_kclk_g}]",
            "        set finn_kclk_d [expr {$finn_kclk_d / $finn_kclk_g}]",
            "        create_generated_clock -name $finn_kclk_name -multiply_by $finn_kclk_m \\",
            "            -divide_by $finn_kclk_d -source $finn_kclk_in $finn_kclk_out",
            (
                '        puts "FINN kernel clock: $finn_kclk_name $finn_kclk_have ns -> '
                "[get_property PERIOD [get_clocks $finn_kclk_name]] ns "
                '(x$finn_kclk_m/$finn_kclk_d of $finn_kclk_in_period ns)"'
            ),
            "    }",
            "}",
        ]
    )


def render_slr_reset_tcl(pinned: dict[str, str]) -> str:
    """Tcl for the post-system-link hook: move each pinned kernel's reset to its SLR.

    Args:
        pinned: Compute-unit instance name -> ``"SLR<n>"``.

    The script leaves a kernel alone, and says so, when the platform has no
    ``proc_sys_reset_kernel_slr<n>`` (single-SLR platforms) or the kernel is
    already on it.
    """
    pairs = " ".join(f"{cu} {slr[3:]}" for cu, slr in sorted(pinned.items()))
    return "\n".join(
        [
            "# FINN+: connect each kernel's ap_rst_n to the kernel reset of the SLR it is",
            "# pinned to. v++ wires every kernel to proc_sys_reset_kernel_slr0, which makes",
            "# the reset of a kernel in another SLR an SLR-crossing net with a fan-out in",
            "# the thousands (vitis_linking_configuration._configure_slr_resets).",
            f"set finn_rst_pinned {{{pairs}}}",
            "foreach {inst slr} $finn_rst_pinned {",
            "    set pin [get_bd_pins -quiet $inst/ap_rst_n]",
            '    if {$pin eq ""} {',
            '        puts "FINN reset: $inst has no ap_rst_n pin; left as linked"',
            "        continue",
            "    }",
            "    set rst [get_bd_pins -quiet proc_sys_reset_kernel_slr$slr/peripheral_aresetn]",
            '    if {$rst eq ""} {',
            '        puts "FINN reset: no proc_sys_reset_kernel_slr$slr in this platform;\\',
            '            $inst keeps the reset v++ linked"',
            "        continue",
            "    }",
            "    set net [get_bd_nets -quiet -of_objects $pin]",
            '    if {$net ne ""} {',
            "        set on_net [get_property PATH [get_bd_pins -of_objects $net]]",
            "        if {[lsearch -exact $on_net [get_property PATH $rst]] >= 0} {",
            '            puts "FINN reset: $inst already takes SLR$slr\'s kernel reset"',
            "            continue",
            "        }",
            "        disconnect_bd_net $net $pin",
            "    }",
            "    connect_bd_net $rst $pin",
            '    puts "FINN reset: $inst/ap_rst_n -> proc_sys_reset_kernel_slr$slr"',
            "}",
        ]
    )


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
    sc: dict[str, list[str]] = field(default_factory=dict)
    sp: dict[str, str] = field(default_factory=dict)
    xo: list[Path] = field(default_factory=list)
    slr: dict[str, str] = field(default_factory=dict)
    connects: list[tuple[str, str]] = field(default_factory=list)
    vivado_section: str = ""
    advanced_section: str = ""
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
        ['b.in']
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

        # Add the connection
        if cu_sender not in self.sc.keys():
            self.sc[cu_sender] = []
        self.sc[cu_sender].append(cu_receiver)

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
        """Add a custom line to the vivado section (one entry per line)."""
        self.vivado_section += line if line.endswith("\n") else line + "\n"

    def add_advanced_line(self, line: str) -> None:
        """Add a custom line to the [advanced] section (``param=...`` entries)."""
        self.advanced_section += line if line.endswith("\n") else line + "\n"

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

    def _get_ports_for_cu(self, cu: str, kerneldefs: dict) -> list[str]:
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
        for send in self.sc.keys():
            for recv in self.sc[send]:
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
    ) -> list[FINNVitisLinkConfigError] | None:
        """Check the configuration and if errors are found, return them. Also prints warnings."""
        errors = []
        kerneldefs = self._get_kerneldefs()

        # Check kernel instantiations
        for kernel, cu in self.nk:
            if kernel not in kerneldefs and not silent_warnings:
                log.warning(
                    f"Kernel {kernel} (CU: {cu}) has no matching definition in "
                    f"the provided xo files. This may be caused by an error "
                    f"when loading the kernel definitons, or because you forgot "
                    f"to add the matching xo file to the configuration."
                )

        # Check connections
        for cu_sender, receivers in self.sc.items():
            for cu_receiver in receivers:
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
                        port["name"] == receiver_port
                        for port in kerneldefs[receiver_kernel]["ports"]
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
            advanced_extras=self.advanced_section,
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


class BuildBasicVitisLinkConfig(Transformation):
    """Build basic configs for an SDP-only graph. If multiple devices are used, generate a config
    per device. The directory with all configurations is stored in the metadata
    prop `vitis_link_configs`. If a config already exists, emits an error and does nothing.

    Refer to `VitisLinkConfiguration.load_from_model` and
    `VitsLinkConfiguration.store_to_model` for more information.

    When done, the config is ready to link (for Single-FPGA).

    To find the path of the final config, one can load the config from the model,
    then check the `config_path` and `run_script_path` fields of the `VitisLinkConfiguration`.
    """

    def __init__(
        self,
        platform: str,
        board: str,
        mem_type: FpgaMemoryType,
        vitis_opt_strategy: VitisOptStrategy,
        optimization_level: str,
        f_mhz: int,
        fpga_part: str = "",
        auto_floorplanning: bool = False,
        floorplan_resource_margin: float = DEFAULT_RESOURCE_MARGIN,
        floorplan_slicem_utilization: float = DEFAULT_SLICEM_UTILIZATION,
        floorplan_logic_utilization: float = DEFAULT_LOGIC_UTILIZATION,
        stream_register_slices: StreamRegisterSlices = StreamRegisterSlices.NONE,
    ) -> None:
        """Build a basic linking configuration and store it. Afterwards, the
        model can be linked, or the config can be modified further.

        Args:
            platform: Vitis platform name.
            board: Target board name.
            mem_type: Build-wide external memory choice.
            vitis_opt_strategy: Vitis optimization strategy.
            optimization_level: v++ optimization level.
            f_mhz: Target kernel clock in MHz.
            fpga_part: FPGA part identifier, needed by the floorplanner's cost models.
                Only required when auto_floorplanning is on.
            auto_floorplanning: Whether to emit soft pblocks for the placement units the
                automatic floorplanner assigned. Has no effect if the model carries no
                floorplan.
            floorplan_resource_margin: Headroom applied when sizing those pblocks.
            floorplan_slicem_utilization: Fraction of SLICEM LUTs LUTRAM/SRLs may fill.
            floorplan_logic_utilization: Fraction of a pblock's slices plain logic may fill.
            stream_register_slices: Whether the kernels carry a register slice in front of
                every node input another node drives; the floorplan units then include them
                (`DataflowBuildConfig.stream_register_slices`).
        """
        super().__init__()
        self.platform = platform
        self.board = board
        self.fpga_memory_type = mem_type
        self.optimization_level = optimization_level
        self.vitis_opt_strategy = vitis_opt_strategy
        self.f_mhz = f_mhz
        self.fpga_part = fpga_part
        self.auto_floorplanning = auto_floorplanning
        self.floorplan_resource_margin = floorplan_resource_margin
        self.floorplan_slicem_utilization = floorplan_slicem_utilization
        self.floorplan_logic_utilization = floorplan_logic_utilization
        self.stream_register_slices = stream_register_slices

    def _configure_kernel_clock(self, config: VitisLinkConfiguration) -> None:
        """Make implementation time the kernel clock at the requested frequency.

        Writes `render_kernel_clock_tcl` next to the link config and runs it as the
        opt_design pre-hook (Vitis appends a user hook to its own), so opt_design,
        placement, physical optimisation and routing all see the requested period
        and v++'s post-route frequency scaling starts from it. Applies to every
        Alveo build; at the platform's own input frequency it changes nothing.
        """
        script = config.config_path.parent / "finn_kernel_clock.tcl"
        script.write_text(render_kernel_clock_tcl(self.f_mhz) + "\n")
        config.add_vivado_line(f"prop=run.impl_1.STEPS.OPT_DESIGN.TCL.PRE={script}")

    def _configure_slr_resets(self, config: VitisLinkConfiguration) -> None:
        """Give every SLR-pinned kernel the kernel reset of its own SLR.

        v++ wires ``ap_rst_n`` of every kernel to ``proc_sys_reset_kernel_slr0``
        whatever SLR the kernel was pinned to, although the Alveo platforms carry
        one kernel reset per SLR. A kernel in another SLR then takes a reset that
        crosses the SLR boundary with a fan-out in the thousands; on the
        floorplanned mobilenetv1 (13 Oct 2026) that net was the worst FINN path,
        1.54 ns of slack at 100 MHz against 2.14 ns for the same reset inside
        SLR0, and 1 ns worse than in the unfloorplanned build.

        The rewiring is a Tcl script run as ``compiler.userPostSysLinkOverlayTcl``,
        the v++ hook that sees the finished block design before synthesis (the
        floorplanner cannot use that hook, see `_configure_floorplanning`; the
        reset is a block-design matter and can). The per-SLR resets are released a
        few cycles apart, which is harmless for a FINN dataflow design: nothing
        flows between kernels until the host starts them, long after reset.

        ``-hier_fanout_limit`` on opt_design replicates the driver of every net
        whose fan-out exceeds the limit per hierarchy, so the reset (and any other
        wide control net) is driven from registers the placer can put next to
        their loads instead of one register per SLR.
        """
        pinned = {cu: slr for cu, slr in config.slr.items() if slr.upper().startswith("SLR")}
        if not pinned:
            return
        script = config.config_path.parent / "finn_reset_per_slr.tcl"
        script.write_text(render_slr_reset_tcl(pinned) + "\n")
        config.add_advanced_line(f"param=compiler.userPostSysLinkOverlayTcl={script}")
        config.add_vivado_line(
            "prop=run.impl_1.{STEPS.OPT_DESIGN.ARGS.MORE OPTIONS}="
            f"{{-hier_fanout_limit {OPT_HIER_FANOUT_LIMIT}}}"
        )
        log.info(f"Per-SLR kernel resets for {', '.join(sorted(pinned))} via {script}")

    def _configure_floorplanning(self, model: ModelWrapper, config: VitisLinkConfiguration) -> None:
        """Point the Vivado implementation run at the floorplanner's geometry engine.

        Pblocks need an open netlist, which on Alveo only exists once v++ has reached
        the `impl_1` run. `run.impl_1.STEPS.PLACE_DESIGN.TCL.PRE` runs Tcl exactly
        there, with every kernel's internals addressable -- the same state the Zynq
        flow gets from `open_run synth_1`, so both flows drive the one geometry engine.

        Before placement, and deliberately *after* `opt_design`, for two reasons:

        * A pblock-constrained cell is one `opt_design` will not touch. Running the
          geometry engine ahead of it had the optimizer report
          `[Opt 31-1021] In phase Sweep, 1411411 netlist objects are constrained
          preventing optimization` -- essentially the whole design excluded from
          sweep, retarget and constant propagation. The netlist handed to the placer
          was therefore bigger than it needed to be, which makes every unit harder to
          fit inside its region.
        * `opt_design` rewrites the netlist: it sweeps cells away and replicates
          others. Binding cells beforehand constrains a netlist that no longer exists
          by the time anything is placed, and any cell it creates afterwards belongs
          to no pblock -- which inside a reconfigurable partition is an HDPR-18 error,
          since every cell there must be ranged.

        Placement is also the first point at which `report_utilization` reflects the
        optimized design, so the engine sizes every unit from what will really be
        placed rather than from the pre-optimization netlist.

        Not `param:compiler.userPostSysLinkOverlayTcl`: that hook does fire, but in an
        IPI block-design context where no netlist is open, so `create_pblock` fails
        outright and the kernel's internal cells are not visible.
        """
        units = collect_floorplan_units(
            model,
            self.fpga_part,
            margin=self.floorplan_resource_margin,
            stream_register_slices=self.stream_register_slices,
        )
        if not units:
            log.warning(
                "Automatic floorplanning is enabled but the model carries no placement "
                "units; linking without pblocks."
            )
            return

        script_path = config.config_path.parent / "finn_floorplan.tcl"
        report_path = config.config_path.parent / "finn_floorplan.json"
        # Sites the shell closes to the placer, measured once per platform.
        closure, closure_note = load_map(self.platform, vivado_version())
        if closure is None:
            log.warning(
                f"{closure_note}: pblocks may be sized over sites the placer closes "
                "next to the static shell, which it then rejects ([Place 30-859]). Map the "
                f"platform once with `finn floorplan map-shell --platform {self.platform}`."
            )
        else:
            log.info(
                f"Shell closure map: {len(closure.all_closed)} closed site(s) from {closure_note}"
            )
        script_path.write_text(
            "\n".join(
                render_floorplan_tcl(
                    units,
                    report_file=report_path,
                    margin=self.floorplan_resource_margin,
                    overrides_file=config.config_path.parent / OVERRIDES_FILE,
                    slicem_utilization=self.floorplan_slicem_utilization,
                    logic_utilization=self.floorplan_logic_utilization,
                    closed_sites=closure.all_closed if closure else None,
                    closed_sites_vivado=closure.vivado if closure else "",
                    closed_sites_source=closure_note,
                )
            )
            + "\n"
        )
        config.add_vivado_line(f"prop=run.impl_1.STEPS.PLACE_DESIGN.TCL.PRE={script_path}")
        # Vitis suppresses the implementation run's intermediate checkpoints
        # (project.writeIntermediateCheckpoints 0 in its _vivado_params.tcl). With
        # them written, a placer rejection resumes at place_design from the
        # optimised checkpoint instead of re-linking and re-optimising the design
        # (`placer_retry.retry_script`), and a routing failure leaves a placed
        # checkpoint to place incrementally from. One checkpoint per step, ~180 MB.
        config.add_vivado_line("param=project.writeIntermediateCheckpoints=1")
        model.set_metadata_prop("floorplan_report_json", str(report_path))
        log.info(f"Floorplanning {len(units)} placement unit(s) via {script_path}")

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

        # Differentiate Multi- and Single-FPGA cases
        number_of_device_ids = 0
        for node in model.graph.node:
            if get_device_id(node) is not None:
                number_of_device_ids += 1

        if number_of_device_ids != 0 and number_of_device_ids < len(model.graph.node):
            raise FINNVitisLinkConfigError(
                f"{number_of_device_ids} / "
                f"{len(model.graph.node)} nodes in the graph "
                f"have an associated device ID. Either all nodes "
                f"have an ID (Multi-FPGA) or none (Single-FPGA). "
                f"Stopping."
            )

        # Create a config for every device
        for node in model.graph.node:
            device = get_device_id(node)
            if not (device is not None):
                raise FINNInternalError("No device resolved for the linking configuration")
            if device not in configs:
                configs[device] = VitisLinkConfiguration(
                    config_path=Path(make_build_dir(f"vitis_link_device_{device}_")) / "config.txt",
                    platform=self.platform,
                    optimization_level=self.optimization_level,
                    f_mhz=self.f_mhz,
                )

        # The implementation directives of the strategy, for every device
        for device in configs.keys():
            for line in implementation_directives(self.vitis_opt_strategy):
                configs[device].add_vivado_line(line)

        # Some temporary variables needed to construct the configs
        current_device: int
        cu_names: dict[str, str] = {}

        # List of node names that represent an IODMA
        dma_nodes: list[str] = []

        # Current count of DMAs on the device (key)
        dmas: dict[int, int] = {}

        # What index this node is (as a DMA) on that device.
        # So A: 3 would mean that A is the 3rd DMA on that device
        dma_on_device: dict[str, int] = {}

        # Maps node names to number of input streams seen.
        # This is used to count up. If node A has 3 inputs, we count up: s_axis_0, s_axis_1, etc.
        # and increment this entry
        cu_inputs: dict[str, int] = {}
        cu_outputs: dict[str, int] = {}

        # Loop through all SDPs and add all nodes
        check_all_sdp_nodes(model)
        total_idmas = 0
        total_odmas = 0
        for node in model.graph.node:
            current_device = cast("int", get_device_id(node))
            submodel, _ = get_submodel(node)
            predecessors = model.find_direct_predecessors(node)
            successors = model.find_direct_successors(node)
            is_input = predecessors is None and successors is not None
            is_output = successors is None and predecessors is not None

            # Add the SDPs XO file
            configs[current_device].add_xo(get_vitis_xo(node))

            # Instantiate the kernel
            if len(submodel.graph.node) == 1 and "IODMA" in submodel.graph.node[0].op_type:
                dma_nodes.append(node.name)
                if current_device not in dmas:
                    dmas[current_device] = 0
                if is_input:
                    idma = f"idma_{total_idmas}"
                    total_idmas += 1
                    configs[current_device].add_cu(node.name, idma)
                    cu_names[node.name] = idma
                    dma_on_device[node.name] = dmas[current_device]
                    dmas[current_device] += 1
                if is_output:
                    odma = f"odma_{total_odmas}"
                    total_odmas += 1
                    configs[current_device].add_cu(node.name, odma)
                    cu_names[node.name] = odma
                    dma_on_device[node.name] = dmas[current_device]
                    dmas[current_device] += 1
            else:
                configs[current_device].add_cu(node.name, node.name)
                cu_names[node.name] = node.name

            # Initialize IO count for every node
            cu_inputs[node.name] = 0
            cu_outputs[node.name] = 0

        if total_idmas > 1 or total_odmas > 1:
            log.warning("Multiple IODMAs were detected. Support for this is experimental.")

        # Add connection between kernels.
        for node in model.graph.node:
            current_device = cast("int", get_device_id(node))
            submodel, _ = get_submodel(node)
            predecessors = model.find_direct_predecessors(node)
            successors = model.find_direct_successors(node)
            node_slr = getCustomOp(node).get_nodeattr("slr")

            # TODO: This is currently assuming how branching SDPs would be implemented,
            # and might need to be updated eventually
            # If successor and we are on the same device, add a connection
            if successors is not None:
                for successor in successors:
                    if get_device_id(node) == get_device_id(successor):
                        configs[current_device].add_sc(
                            f"{cu_names[node.name]}.m_axis_{cu_outputs[node.name]}",
                            f"{cu_names[successor.name]}.s_axis_{cu_inputs[successor.name]}",
                        )
                        cu_outputs[node.name] += 1
                        cu_inputs[successor.name] += 1

            # Pin the kernel to its SLR. Until now this was only ever done for the
            # Aurora transceiver CU, so ordinary kernels were placed wherever Vitis
            # chose -- including in builds that supplied a floorplan file, whose SLR
            # assignment therefore did not reach placement at all. Logged rather than
            # silent, because it changes where such a build puts its kernels.
            if node_slr != -1:
                configs[current_device].add_slr(cu_names[node.name], f"SLR{node_slr}")
                log.info(f"Pinning kernel {cu_names[node.name]} to SLR{node_slr}")

            # Add system ports. A floorplanner-populated `mem_port` wins: it is the
            # only party that knows both where the node landed and which external
            # resource it was anchored to. Without one, fall back to the original
            # board-driven derivation.
            mem_port = cast("str", getCustomOp(node).get_nodeattr("mem_port"))
            if node.name in dma_nodes and mem_port:
                configs[current_device].add_sp(cu_names[node.name] + ".m_axi_gmem0", mem_port)
                continue

            mem_type: str
            mem_idx: int
            if self.fpga_memory_type == FpgaMemoryType.HOST_MEM:
                mem_type = "HOST"
                mem_idx = 0
            else:
                match self.board.lower():
                    case "u50" | "u280" | "u55c":
                        mem_type = "HBM"
                        mem_idx = 0
                    case "u250":
                        mem_type = "DDR"
                        mem_idx = 0 if node_slr == -1 else cast("int", node_slr)
                    case _:
                        raise FINNUserError(
                            f"Cannot do system-port placement for unknown board {self.board}"
                        )
            if node.name in dma_nodes:
                if mem_idx != 0:
                    log.warning(
                        f"Overwriting expected mem_idx {dma_on_device[node.name]} "
                        f"with specified Node SLR mem_idx: {mem_idx}"
                    )
                else:
                    mem_idx = dma_on_device[node.name]
                configs[current_device].add_sp(
                    cu_names[node.name] + ".m_axi_gmem0", f"{mem_type}[{mem_idx}]"
                )

        # Store everything, generate scripts and return
        for config in configs.values():
            self._configure_kernel_clock(config)
            self._configure_slr_resets(config)
        if self.auto_floorplanning:
            for config in configs.values():
                self._configure_floorplanning(model, config)
        model = VitisLinkConfiguration.store_to_model(configs, model)
        for config in configs.values():
            config.generate_config()
            config.generate_run_script()
            config.generate_report_tcl_script()
        return model, False
