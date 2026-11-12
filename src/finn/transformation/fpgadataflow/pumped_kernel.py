"""Vitis kernels for double-pumped stitched IP.

A stitched IP with double-pumped nodes (``pumpedCompute``/``pumpedMemory``) has a second clock
input ``ap_clk2x`` that must run at exactly twice ``ap_clk``, with rising edges aligned. A Vitis
kernel gets at most two clocks from the platform, ``ap_clk`` and ``ap_clk_2``, and on the Alveo
platforms they come from two separate MMCMs with no defined phase relation. FINN therefore
clocks all of a pumped kernel's logic from ``ap_clk_2``, which the link sets to twice the FINN
clock, and derives both clocks inside the kernel:

- a ``BUFGCE_DIV`` divides ``ap_clk_2`` by two for the stitched IP's ``ap_clk``,
- a ``BUFGCE`` passes ``ap_clk_2`` on as ``ap_clk2x``,
- both share a ``CLOCK_DELAY_GROUP`` (set at link time, `render_pumped_clock_group_tcl`),
- the kernel's AXI streams cross from the platform's ``ap_clk`` (the IODMAs' clock) through
  asynchronous AXI-stream FIFOs, the reset through a reset synchronizer.

The ratio is exact by construction. v++ treats the divided clock as an unscalable system clock:
a link whose divided clock misses timing fails instead of being scaled down, and v++ allows at
most `MAX_KERNEL_CLOCK_MHZ` on any kernel clock, so the FINN clock of a pumped design is at most
half of that.
"""

from __future__ import annotations

import re
from pathlib import Path
from subprocess import CalledProcessError

from finn.templates import get_jinja_environment
from finn.util.basic import launch_process_helper
from finn.util.exception import FINNUserError

#: The highest kernel clock v++ accepts (``[v++ 60-754]`` on the U55C platform).
MAX_KERNEL_CLOCK_MHZ: int = 650

#: The highest FINN clock of a design with pumped kernels: ``ap_clk_2`` runs at twice this.
MAX_PUMPED_CLOCK_MHZ: int = MAX_KERNEL_CLOCK_MHZ // 2

#: Instance names of the clock buffers in the kernel wrapper; the link-time clock hook finds
#: them by these names.
PUMP_DIV_CELL: str = "pump_div"
PUMP_2X_CELL: str = "pump_2x"

#: Depth of the clock-crossing FIFOs on the kernel streams (the minimum of xpm_fifo_axis).
CDC_FIFO_DEPTH: int = 16

#: Second kernel clock pin of the Alveo platforms' clocking subsystem (``ulp_ucs``).
PUMPED_CLOCK_PIN: str = "aclk_kernel_01"

PUMPED_KERNEL_VENDOR: str = "xilinx_finn"
PUMPED_KERNEL_LIBRARY: str = "finn"


def is_pumped_kernel(interfaces: dict) -> bool:
    """Whether a stitched IP with these interface names (``vivado_stitch_ifnames``) is pumped."""
    return bool(interfaces.get("clk2x"))


def pumped_kernel_name(block_name: str) -> str:
    """Kernel (and top module) name of the wrapper around the stitched IP ``block_name``.

    The stitched IP's own checkpoint contains a module named ``block_name`` (its block design),
    so the wrapper needs another name. The suffix is short because the xclbin stores each CU as
    ``<kernel>:<CU>`` in at most 63 characters, and the CU keeps the name ``block_name``.
    """
    return f"{block_name}_dp"


def stub_output_ports(stub: str) -> list[str]:
    r"""Output ports declared in a Verilog synthesis stub (``write_verilog -mode synth_stub``).

    >>> stub_output_ports("module m(a, b);\n  input a;\n  output [7:0]b;\n  output c;\n")
    ['b', 'c']
    """
    return re.findall(r"^\s*output\s+(?:\[[^\]]*\]\s*)?(\w+)\s*;", stub, flags=re.MULTILINE)


def render_pumped_kernel_sv(block_name: str, interfaces: dict, stub: str) -> str:
    """SystemVerilog of the kernel wrapper around the pumped stitched IP ``block_name``.

    Args:
        block_name: Name of the stitched IP (its checkpoint's top is ``<block_name>_wrapper``).
        interfaces: The stitched IP's ``vivado_stitch_ifnames``.
        stub: The stitched IP's synthesis stub, for outputs the wrapper leaves open
            (``sim_finish``).
    """
    s_axis = [(name, int(width)) for name, width in interfaces["s_axis"]]
    m_axis = [(name, int(width)) for name, width in interfaces["m_axis"]]
    connected = {"ap_clk", "ap_clk2x", "ap_rst_n"}
    for name, _ in s_axis + m_axis:
        connected |= {f"{name}_tdata", f"{name}_tvalid", f"{name}_tready"}
    unused = [port for port in stub_output_ports(stub) if port not in connected]
    template = get_jinja_environment().get_template("vitis_kernel/pumped_kernel.sv.jinja")
    return template.render(
        module=pumped_kernel_name(block_name),
        inner=block_name,
        inner_module=f"{block_name}_wrapper",
        s_axis=s_axis,
        m_axis=m_axis,
        unused_outputs=unused,
        div_cell=PUMP_DIV_CELL,
        x2_cell=PUMP_2X_CELL,
        fifo_depth=CDC_FIFO_DEPTH,
    )


def check_pumped_kernel_interfaces(block_name: str, interfaces: dict) -> None:
    """Raise if the pumped stitched IP has interfaces the kernel wrapper cannot pass on."""
    if interfaces.get("aximm") or interfaces.get("axilite"):
        raise FINNUserError(
            f"{block_name} is double-pumped and has AXI-MM or AXI-lite interfaces "
            f"({interfaces.get('aximm')}, {interfaces.get('axilite')}); a pumped Vitis kernel "
            "can only have AXI streams (no external or runtime-writeable weights)."
        )
    if not interfaces.get("s_axis") and not interfaces.get("m_axis"):
        raise FINNUserError(f"{block_name} is double-pumped but has no AXI streams.")


def package_pumped_kernel(
    vivado_stitch_proj_dir: Path, block_name: str, interfaces: dict, fpga_part: str
) -> tuple[Path, str]:
    """Wrap the pumped stitched IP of ``vivado_stitch_proj_dir`` into a packaged Vitis kernel IP.

    Needs the out-of-context synthesis outputs of `CreateStitchedIP` with ``run_synth``:
    ``<block_name>.dcp``, the stub ``<block_name>.v`` and the constraints ``<block_name>.xdc``.

    Returns:
        The packaged IP's directory (for ``package_xo -ip_directory``) and the kernel name.
    """
    check_pumped_kernel_interfaces(block_name, interfaces)
    proj = Path(vivado_stitch_proj_dir)
    inner_dcp, inner_stub = proj / f"{block_name}.dcp", proj / f"{block_name}.v"
    for f in (inner_dcp, inner_stub):
        if not f.is_file():
            raise FINNUserError(
                f"Pumped kernel {block_name}: {f} is missing; the stitched IP must be "
                "synthesized out of context (CreateStitchedIP with run_synth=True)."
            )
    inner_xdc = proj / f"{block_name}.xdc"
    module = pumped_kernel_name(block_name)
    work_dir = proj / "pumped_kernel"
    ip_dir = work_dir / "ip"
    work_dir.mkdir(parents=True, exist_ok=True)
    wrapper_sv = work_dir / f"{module}.sv"
    wrapper_sv.write_text(render_pumped_kernel_sv(block_name, interfaces, inner_stub.read_text()))

    template = get_jinja_environment().get_template("vitis_kernel/pumped_kernel_pack.tcl.jinja")
    pack_tcl = work_dir / "pack.tcl"
    pack_tcl.write_text(
        template.render(
            module=module,
            inner=block_name,
            inner_dcp=inner_dcp,
            inner_stub=inner_stub,
            inner_xdc=inner_xdc if inner_xdc.is_file() else None,
            inner_xdc_name=inner_xdc.name,
            inner_bd_cell=f"{block_name}_i",
            wrapper_sv=wrapper_sv,
            work_dir=work_dir,
            ip_dir=ip_dir,
            part=fpga_part,
            vendor=PUMPED_KERNEL_VENDOR,
            library=PUMPED_KERNEL_LIBRARY,
            streams=[name for name, _ in interfaces["s_axis"] + interfaces["m_axis"]],
        )
        + "\n"
    )
    try:
        launch_process_helper(
            ["vivado", "-mode", "batch", "-nojournal", "-log", "pack.log", "-source", "pack.tcl"],
            cwd=work_dir,
            print_stdout=False,
        )
    except CalledProcessError as e:
        raise FINNUserError(
            f"Packaging the pumped kernel {module} failed, see {work_dir / 'pack.log'}"
        ) from e
    if not (ip_dir / "component.xml").is_file():
        raise FINNUserError(f"Pumped kernel {module} was not packaged, see {work_dir / 'pack.log'}")
    return ip_dir, module


def render_pumped_clock_group_tcl(cus: list[str]) -> str:
    """Tcl for the opt_design pre-hook: one ``CLOCK_DELAY_GROUP`` per pumped compute unit.

    The group makes the placer and router balance the divided and the undivided clock tree of
    each kernel, so the half-period paths between them see little skew.
    """
    lines = [
        "# FINN+: balance the 1x (BUFGCE_DIV) and 2x (BUFGCE) clocks of every pumped kernel",
        f"foreach finn_pump_cu {{{' '.join(sorted(cus))}}} {{",
        "    set finn_pump_nets [get_nets -quiet -of_objects [get_pins -quiet -hierarchical \\",
        (
            f'        -filter "NAME =~ */$finn_pump_cu/{PUMP_DIV_CELL}/O || '
            f'NAME =~ */$finn_pump_cu/{PUMP_2X_CELL}/O"]]'
        ),
        "    if {[llength $finn_pump_nets] == 2} {",
        "        set_property CLOCK_DELAY_GROUP finn_pump_$finn_pump_cu $finn_pump_nets",
        (
            '        puts "FINN pumped clocks: $finn_pump_cu -> '
            'CLOCK_DELAY_GROUP finn_pump_$finn_pump_cu"'
        ),
        "    } else {",
        (
            '        puts "CRITICAL WARNING: FINN pumped clocks: expected 2 clock nets in '
            '$finn_pump_cu, found [llength $finn_pump_nets]"'
        ),
        "    }",
        "}",
    ]
    return "\n".join(lines)
