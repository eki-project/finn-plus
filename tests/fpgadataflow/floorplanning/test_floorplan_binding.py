"""Tests for Phase-B unit collection and its wiring into the Zynq project."""

import pytest

from fpgadataflow.floorplanning.util import U55C_PART, attr, make_chain
from pathlib import Path
from qonnx.custom_op.registry import getCustomOp

from finn.builder.build_dataflow_config import FpgaMemoryType, VitisOptStrategy
from finn.transformation.fpgadataflow.floorplanning.binding import (
    PBLOCK_PREFIX,
    UNASSIGNED_SLR,
    collect_floorplan_units,
    pblock_name_for_unit,
)
from finn.transformation.fpgadataflow.floorplanning.grouping import ComputeFloorplanGroups
from finn.transformation.fpgadataflow.floorplanning.slr_assignment import AssignFloorplanSLRs
from finn.transformation.fpgadataflow.floorplanning.tcl import render_floorplan_tcl
from finn.transformation.fpgadataflow.templates import custom_zynq_shell_template
from finn.transformation.fpgadataflow.vitis_linking_configuration import VitisLinkConfiguration
from finn.util.basic import make_build_dir
from finn.util.platforms import platforms


def _floorplanned(length: int = 4) -> object:
    """Return a chain that has been grouped and SLR-assigned for the U55C."""
    model = make_chain(length)
    model = model.transform(ComputeFloorplanGroups())
    return model.transform(AssignFloorplanSLRs(platforms["U55C"](), U55C_PART, "U55C"))


@pytest.mark.fpgadataflow
def test_collect_returns_nothing_for_an_unfloorplanned_model() -> None:
    """A model with no floorplan yields no units, so callers can skip binding."""
    assert collect_floorplan_units(make_chain(3), U55C_PART) == []


@pytest.mark.fpgadataflow
def test_collect_builds_one_spec_per_placement_unit() -> None:
    """Every assigned unit becomes a spec carrying its nodes, SLR and demands."""
    model = _floorplanned()
    specs = collect_floorplan_units(model, U55C_PART)

    group_ids = {attr(model, n.name, "floorplan_group_id") for n in model.graph.node}
    assert len(specs) == len(group_ids)
    assert [spec.name for spec in specs] == [pblock_name_for_unit(i) for i in range(len(specs))]

    covered = {cell for spec in specs for cell in spec.cells}
    assert covered == {node.name for node in model.graph.node}
    for spec in specs:
        assert spec.slr >= 0
        assert spec.demands["luts"] > 0


@pytest.mark.fpgadataflow
def test_collect_records_the_pblock_name_on_each_node() -> None:
    """The chosen pblock name is written back so reports can find it."""
    model = _floorplanned()
    collect_floorplan_units(model, U55C_PART)
    for node in model.graph.node:
        assert str(attr(model, node.name, "floorplan_pblock")).startswith(PBLOCK_PREFIX)


@pytest.mark.fpgadataflow
def test_collect_can_run_without_recording() -> None:
    """A dry run must leave the model untouched."""
    model = _floorplanned()
    collect_floorplan_units(model, U55C_PART, record_pblock_names=False)
    assert all(attr(model, n.name, "floorplan_pblock") == "" for n in model.graph.node)


@pytest.mark.fpgadataflow
def test_unassigned_units_are_skipped_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """A grouped but unassigned unit gets no pblock rather than a bogus one."""
    model = _floorplanned()
    # Undo the SLR assignment of the first unit only.
    first_group = attr(model, model.graph.node[0].name, "floorplan_group_id")
    for node in model.graph.node:
        if getCustomOp(node).get_nodeattr("floorplan_group_id") == first_group:
            getCustomOp(node).set_nodeattr("floorplan_slr", -1)

    with caplog.at_level("WARNING"):
        specs = collect_floorplan_units(model, U55C_PART)
    assert "no SLR assignment" in caplog.text
    assert all(model.graph.node[0].name not in spec.cells for spec in specs)


@pytest.mark.fpgadataflow
def test_collected_units_render_to_valid_tcl() -> None:
    """The collection and rendering halves fit together without further massaging."""
    specs = collect_floorplan_units(_floorplanned(), U55C_PART)
    lines = render_floorplan_tcl(specs)
    call = next(line for line in lines if line.startswith("finn_fp_floorplan_units "))
    assert call.count("{") == call.count("}")
    for spec in specs:
        assert f"name {spec.name} " in call


@pytest.mark.fpgadataflow
def test_zynq_template_keeps_its_default_implementation_path() -> None:
    """The floorplan slot must not change what a build without it does.

    The template drives every Zynq bitstream, so the guard has to be such that an
    empty floorplan slot leaves the original launch_runs path in place.
    """
    assert "$FLOORPLAN_CONFIG$" in custom_zynq_shell_template
    assert "set finn_floorplan_done 0" in custom_zynq_shell_template
    assert (
        "if {$pr_flow == 0 && $finn_floorplan_done == 0} {" in custom_zynq_shell_template
    ), "the default launch_runs path must stay gated on the floorplanner not having run"


@pytest.mark.fpgadataflow
def test_alveo_hook_targets_the_implementation_run() -> None:
    """The Alveo hook must be the impl-stage one, where a netlist is actually open.

    param:compiler.userPostSysLinkOverlayTcl fires in an IPI block-design context
    with no open design -- create_pblock fails there and kernel internals are
    invisible -- so pblocks go through run.impl_1.STEPS.*.TCL.PRE instead. That is
    written into the v++ config's [vivado] section, which VitisLinkConfiguration
    already renders.

    Specifically the PLACE_DESIGN one, not OPT_DESIGN: a pblock-constrained cell is
    excluded from logic optimization, and binding before opt_design had it report
    over a million netlist objects "constrained preventing optimization".
    """
    from finn.transformation.fpgadataflow.vitis_linking_configuration import (
        BuildBasicVitisLinkConfig,
    )

    trafo = BuildBasicVitisLinkConfig(
        platform="xilinx_u55c_gen3x16_xdma_3_202210_1",
        board="U55C",
        mem_type=FpgaMemoryType.DEFAULT,
        vitis_opt_strategy=VitisOptStrategy.DEFAULT,
        optimization_level="0",
        f_mhz=200,
        fpga_part=U55C_PART,
        auto_floorplanning=True,
    )
    model = _floorplanned()
    config = VitisLinkConfiguration(
        Path(make_build_dir("test_alveo_hook_")) / "link.cfg",
        200,
        "",
        "xilinx_u55c_gen3x16_xdma_3_202210_1",
    )
    trafo._configure_floorplanning(model, config)  # noqa: SLF001

    # Before placement, and after opt_design: a pblock-constrained cell is one
    # opt_design will not optimize, and opt_design rewrites the netlist anyway.
    assert "run.impl_1.STEPS.PLACE_DESIGN.TCL.PRE=" in config.vivado_section
    # The retry resumes the implementation run from the optimised checkpoint,
    # which Vitis only writes with this parameter on.
    assert "param=project.writeIntermediateCheckpoints=1" in config.vivado_section
    assert "OPT_DESIGN.TCL.PRE" not in config.vivado_section
    assert "userPostSysLinkOverlayTcl" not in config.vivado_section
    script = config.config_path.parent / "finn_floorplan.tcl"
    assert script.is_file()
    body = script.read_text()
    assert "finn_fp_floorplan_units {" in body
    assert "source {" in body


@pytest.mark.fpgadataflow
def test_pinned_kernels_get_the_reset_of_their_slr() -> None:
    """A kernel pinned to an SLR is rewired to that SLR's kernel reset after system link.

    v++ connects every kernel to proc_sys_reset_kernel_slr0; the post-system-link
    hook moves each pinned kernel's ap_rst_n to proc_sys_reset_kernel_slr<n>, and
    opt_design replicates wide control nets per hierarchy.
    """
    from finn.transformation.fpgadataflow.vitis_linking_configuration import (
        OPT_HIER_FANOUT_LIMIT,
        BuildBasicVitisLinkConfig,
        render_slr_reset_tcl,
    )

    trafo = BuildBasicVitisLinkConfig(
        platform="xilinx_u55c_gen3x16_xdma_3_202210_1",
        board="U55C",
        mem_type=FpgaMemoryType.DEFAULT,
        vitis_opt_strategy=VitisOptStrategy.DEFAULT,
        optimization_level="0",
        f_mhz=200,
        fpga_part=U55C_PART,
        auto_floorplanning=False,
    )
    config = VitisLinkConfiguration(
        Path(make_build_dir("test_slr_reset_")) / "link.cfg",
        200,
        "",
        "xilinx_u55c_gen3x16_xdma_3_202210_1",
    )
    config.add_slr("StreamingDataflowPartition_2", "SLR1")
    config.add_slr("idma_0", "SLR0")
    trafo._configure_slr_resets(config)  # noqa: SLF001

    script = config.config_path.parent / "finn_reset_per_slr.tcl"
    assert script.is_file()
    assert f"param=compiler.userPostSysLinkOverlayTcl={script}" in config.advanced_section
    assert (
        "prop=run.impl_1.{STEPS.OPT_DESIGN.ARGS.MORE OPTIONS}="
        f"{{-hier_fanout_limit {OPT_HIER_FANOUT_LIMIT}}}" in config.vivado_section
    )
    body = script.read_text()
    assert "set finn_rst_pinned {StreamingDataflowPartition_2 1 idma_0 0}" in body
    assert "proc_sys_reset_kernel_slr$slr/peripheral_aresetn" in body
    assert (
        body
        == render_slr_reset_tcl({"StreamingDataflowPartition_2": "SLR1", "idma_0": "SLR0"}) + "\n"
    )

    # Without pinned kernels nothing is generated and the config stays as it was.
    bare = VitisLinkConfiguration(
        Path(make_build_dir("test_slr_reset_")) / "link.cfg",
        200,
        "",
        "xilinx_u55c_gen3x16_xdma_3_202210_1",
    )
    trafo._configure_slr_resets(bare)  # noqa: SLF001
    assert bare.advanced_section == "" and "hier_fanout_limit" not in bare.vivado_section


@pytest.mark.fpgadataflow
def test_config_file_carries_the_advanced_section_only_when_used() -> None:
    """The rendered v++ config gets an [advanced] section for the post-system-link hook."""
    config = VitisLinkConfiguration(
        Path(make_build_dir("test_advanced_section_")) / "link.cfg",
        200,
        "",
        "xilinx_u55c_gen3x16_xdma_3_202210_1",
    )
    config.add_cu("k", "k_1")
    config.generate_config()
    assert "[advanced]" not in config.config_path.read_text()
    config.add_advanced_line("param=compiler.userPostSysLinkOverlayTcl=/tmp/x.tcl")
    config.generate_config()
    text = config.config_path.read_text()
    assert text.rstrip().endswith("[advanced]\nparam=compiler.userPostSysLinkOverlayTcl=/tmp/x.tcl")


@pytest.mark.fpgadataflow
def test_stitched_ip_puts_a_reset_stage_in_front_of_every_node() -> None:
    """The reset pipeline is one reset_pipe cell per node, fed from the port, feeding the node."""
    from finn.transformation.fpgadataflow.create_stitched_ip import (
        RESET_PIPE_VLNV,
        CreateStitchedIP,
    )

    model = _floorplanned()
    first, second = model.graph.node[0], model.graph.node[1]
    direct = CreateStitchedIP(U55C_PART, 5.0)
    for node in (first, second):
        direct.connect_clk_rst(node)
    assert not any("reset_pipe" in c for c in direct.create_cmds + direct.connect_cmds)
    assert f"make_bd_pins_external [get_bd_pins {first.name}/ap_rst_n]" in direct.connect_cmds
    assert (
        f"connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins {second.name}/ap_rst_n]"
        in direct.connect_cmds
    )

    piped = CreateStitchedIP(U55C_PART, 5.0, reset_pipeline_stages=2)
    for node in (first, second):
        piped.connect_clk_rst(node)
    # One kernel-level tree stage, made once, fed from the port, replicated per 16 loads.
    tree = f"create_bd_cell -type ip -vlnv {RESET_PIPE_VLNV} ap_rst_n_tree"
    assert piped.create_cmds.count(tree) == 1
    assert (
        "set_property -dict [list CONFIG.STAGES {1} CONFIG.MAX_FANOUT {16}] "
        "[get_bd_cells ap_rst_n_tree]" in piped.create_cmds
    )
    assert "make_bd_pins_external [get_bd_pins ap_rst_n_tree/ap_rst_n]" in piped.connect_cmds
    assert (
        "connect_bd_net [get_bd_ports ap_clk] [get_bd_pins ap_rst_n_tree/ap_clk]"
        in piped.connect_cmds
    )
    for node in (first, second):
        cell = f"{node.name}_rst"
        assert f"create_bd_cell -type ip -vlnv {RESET_PIPE_VLNV} {cell}" in piped.create_cmds
        assert f"set_property CONFIG.STAGES {{2}} [get_bd_cells {cell}]" in piped.create_cmds
        assert (
            f"connect_bd_net [get_bd_ports ap_clk] [get_bd_pins {cell}/ap_clk]"
            in piped.connect_cmds
        )
        assert (
            f"connect_bd_net [get_bd_pins ap_rst_n_tree/rst_n_out] [get_bd_pins {cell}/ap_rst_n]"
            in piped.connect_cmds
        )
        assert (
            f"connect_bd_net [get_bd_pins {cell}/rst_n_out] [get_bd_pins {node.name}/ap_rst_n]"
            in piped.connect_cmds
        )
    # The port drives nothing but the tree stage.
    assert not any(
        c.startswith("connect_bd_net [get_bd_ports ap_rst_n]") for c in piped.connect_cmds
    )


@pytest.mark.fpgadataflow
def test_added_config_lines_stay_on_separate_lines() -> None:
    """Every added [vivado] or [advanced] entry ends up on a line of its own.

    The old newline logic was inverted (a newline was appended only when the line
    already ended with one), so a second `prop=` line was glued onto the first: v++
    then read the floorplanning hook's path with ``param=...`` appended and installed
    no hook at all (13 Oct 2026, the per_node_v3 link).
    """
    config = VitisLinkConfiguration(
        Path(make_build_dir("test_config_lines_")) / "link.cfg",
        200,
        "",
        "xilinx_u55c_gen3x16_xdma_3_202210_1",
    )
    config.add_cu("k", "k_1")
    config.add_vivado_line("prop=run.impl_1.{STEPS.OPT_DESIGN.ARGS.MORE OPTIONS}={-x 1}")
    config.add_vivado_line("prop=run.impl_1.STEPS.PLACE_DESIGN.TCL.PRE=/tmp/hook.tcl")
    config.add_vivado_line("param=project.writeIntermediateCheckpoints=1\n")
    config.add_advanced_line("param=a=1")
    config.add_advanced_line("param=b=2")
    config.generate_config()
    lines = config.config_path.read_text().splitlines()
    assert "prop=run.impl_1.{STEPS.OPT_DESIGN.ARGS.MORE OPTIONS}={-x 1}" in lines
    assert "prop=run.impl_1.STEPS.PLACE_DESIGN.TCL.PRE=/tmp/hook.tcl" in lines
    assert "param=project.writeIntermediateCheckpoints=1" in lines
    assert lines[lines.index("[advanced]") + 1 :][:2] == ["param=a=1", "param=b=2"]


@pytest.mark.fpgadataflow
def test_nodes_inserted_after_the_floorplan_get_their_own_unit() -> None:
    """A node without a group (e.g. a FIFO inserted while building the kernels) is constrained.

    It gets a unit of its own, numbered in graph order between its neighbours, in the
    SLR the model's assigned nodes share; its attrs are filled in for the reports.
    """
    from finn.transformation.fpgadataflow.floorplanning.grouping import UNASSIGNED_GROUP

    model = _floorplanned(4)
    nodes = list(model.graph.node)
    late = getCustomOp(nodes[1])
    late.set_nodeattr("floorplan_group_id", UNASSIGNED_GROUP)
    late.set_nodeattr("floorplan_slr", UNASSIGNED_SLR)
    slr = getCustomOp(nodes[0]).get_nodeattr("floorplan_slr")

    units = collect_floorplan_units(model, U55C_PART)
    assert [u.cells for u in units] == [[n.name] for n in nodes]
    assert units[1].slr == slr
    assert late.get_nodeattr("floorplan_slr") == slr
    assert late.get_nodeattr("floorplan_group_id") != UNASSIGNED_GROUP
    assert late.get_nodeattr("floorplan_pblock") == units[1].name


@pytest.mark.fpgadataflow
def test_reset_register_pattern_matches_tree_registers() -> None:
    """The reset-register pattern matches the tree and per-node pipes and nothing else."""
    import re

    from finn.transformation.fpgadataflow.create_stitched_ip import RESET_REGISTER_PATTERN

    # The registers sit below the IP instance wrapper and replicas carry a suffix.
    for name in (
        "StreamingDataflowPartition_2_MVAU_rtl_4_rst/inst/genChain.Pipe_reg[0]",
        "ap_rst_n_tree/inst/genOne.Pipe_reg[0]_replica_3",
    ):
        assert re.fullmatch(RESET_REGISTER_PATTERN, name), name
    assert not re.fullmatch(
        RESET_REGISTER_PATTERN, "StreamingDataflowPartition_2_MVAU_rtl_4/inst/core/Pipe_reg"
    )


@pytest.mark.fpgadataflow
def test_kernel_xdc_is_scoped_to_the_block_design_cell() -> None:
    """The kernel XDC is written relative to <block>_i and packaged scoped to that cell."""
    from finn.transformation.fpgadataflow.create_stitched_ip import (
        kernel_xdc_export_tcl,
        kernel_xdc_package_tcl,
    )

    assert kernel_xdc_export_tcl("StreamingDataflowPartition_3") == [
        "write_xdc -force -cell StreamingDataflowPartition_3_i StreamingDataflowPartition_3.xdc"
    ]
    package = kernel_xdc_package_tcl("StreamingDataflowPartition_3")
    assert package[0].startswith("ipx::add_file impl/StreamingDataflowPartition_3.xdc")
    assert package[1].startswith("set_property used_in [list implementation]")
    assert package[2].startswith("set_property scoped_to_cells StreamingDataflowPartition_3_i ")
    assert all(line.count("[") == line.count("]") for line in package)


@pytest.mark.fpgadataflow
def test_register_slices_sit_in_front_of_non_fifo_inputs() -> None:
    """With slices on, every node input another node drives gets an auto-pipelined slice."""
    from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
    from finn.transformation.fpgadataflow.stream_slices import (
        SLICE_REG_CONFIG,
        SLICE_VLNV,
        register_slice_inputs,
        slice_cell_name,
    )

    model = _floorplanned(3)
    first, second, third = model.graph.node
    # The first node's input comes from outside; the others are driven by a node.
    assert register_slice_inputs(first, model) == []
    assert register_slice_inputs(second, model) == [0]
    assert register_slice_inputs(third, model) == [0]

    plain = CreateStitchedIP(U55C_PART, 5.0, reset_pipeline_stages=2)
    piped = CreateStitchedIP(U55C_PART, 5.0, reset_pipeline_stages=2, stream_register_slices=True)
    for builder in (plain, piped):
        for node in model.graph.node:
            builder.connect_clk_rst(node)
    for node in (second, third):
        src = f"{first.name}/out_V" if node is second else f"{second.name}/out_V"
        piped.insert_register_slice(node, 0, src, f"{node.name}/in0_V")
    assert not any("slice" in c for c in plain.create_cmds + plain.connect_cmds)
    for node in (second, third):
        cell = slice_cell_name(node.name, 0)
        assert f"create_bd_cell -type ip -vlnv {SLICE_VLNV} {cell}" in piped.create_cmds
        assert (
            f"set_property CONFIG.REG_CONFIG {{{SLICE_REG_CONFIG}}} [get_bd_cells {cell}]"
            in piped.create_cmds
        )
        # Clocked with the kernel clock, reset together with the node it feeds.
        assert (
            f"connect_bd_net [get_bd_ports ap_clk] [get_bd_pins {cell}/aclk]" in piped.connect_cmds
        )
        assert (
            f"connect_bd_net [get_bd_pins {node.name}_rst/rst_n_out] [get_bd_pins {cell}/aresetn]"
            in piped.connect_cmds
        )
        assert any(c.endswith(f"[get_bd_intf_pins {cell}/S_AXIS]") for c in piped.connect_cmds)
        assert any(
            c.startswith(f"connect_bd_intf_net [get_bd_intf_pins {cell}/M_AXIS]")
            for c in piped.connect_cmds
        )


@pytest.mark.fpgadataflow
def test_register_slices_stay_out_of_the_units() -> None:
    """Units list only their nodes: the auto-pipelined slices must be free to spread.

    The placer inserts a slice's stages along the route between two units, which a
    pblock around either end would forbid, so the binding never adds a slice to a
    unit, whatever the kernels carry.
    """
    from finn.transformation.fpgadataflow.stream_slices import SLICE_REG_CONFIG

    model = _floorplanned(3)
    units = collect_floorplan_units(model, U55C_PART, record_pblock_names=False)
    assert [s.cells for s in units] == [[n.name] for n in model.graph.node]
    assert SLICE_REG_CONFIG == 16  # "Auto-pipelined"


@pytest.mark.fpgadataflow
def test_performance_strategies_carry_directives_and_retiming() -> None:
    """The strategy decides the link's implementation directives and the kernels' synthesis."""
    from finn.transformation.fpgadataflow.vitis_linking_configuration import (
        implementation_directives,
        kernel_synth_options,
    )

    assert implementation_directives(VitisOptStrategy.DEFAULT) == ()
    assert kernel_synth_options(VitisOptStrategy.DEFAULT) == ""
    explore = implementation_directives(VitisOptStrategy.PERFORMANCE)
    best = implementation_directives(VitisOptStrategy.PERFORMANCE_BEST)
    assert any("PLACE_DESIGN.ARGS.DIRECTIVE=Explore" in line for line in explore)
    assert any("PLACE_DESIGN.ARGS.DIRECTIVE=ExtraTimingOpt" in line for line in best)
    assert any("POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED=true" in line for line in best)
    assert not any("POST_ROUTE" in line for line in explore)
    assert all(line.startswith("prop=run.impl_1.STEPS.") for line in explore + best)
    # opt_design keeps -hier_fanout_limit, which Vivado will not combine with a directive.
    assert not any("STEPS.OPT_DESIGN." in line for line in explore + best)
    assert kernel_synth_options(VitisOptStrategy.PERFORMANCE) == "-retiming"
    assert kernel_synth_options(VitisOptStrategy.PERFORMANCE_BEST) == "-retiming"


@pytest.mark.fpgadataflow
def test_every_alveo_link_times_the_kernel_clock_at_the_requested_frequency() -> None:
    """The link config runs a kernel-clock script before opt_design, floorplanned or not."""
    from finn.transformation.fpgadataflow.vitis_linking_configuration import (
        BuildBasicVitisLinkConfig,
        render_kernel_clock_tcl,
    )

    trafo = BuildBasicVitisLinkConfig(
        platform="xilinx_u55c_gen3x16_xdma_3_202210_1",
        board="U55C",
        mem_type=FpgaMemoryType.DEFAULT,
        vitis_opt_strategy=VitisOptStrategy.DEFAULT,
        optimization_level="0",
        f_mhz=200,
        fpga_part=U55C_PART,
        auto_floorplanning=False,
    )
    config = VitisLinkConfiguration(
        Path(make_build_dir("test_kernel_clock_")) / "link.cfg",
        200,
        "",
        "xilinx_u55c_gen3x16_xdma_3_202210_1",
    )
    trafo._configure_kernel_clock(config)  # noqa: SLF001
    script = config.config_path.parent / "finn_kernel_clock.tcl"
    assert script.is_file()
    assert f"prop=run.impl_1.STEPS.OPT_DESIGN.TCL.PRE={script}" in config.vivado_section
    body = script.read_text()
    assert body == render_kernel_clock_tcl(200) + "\n"
    assert "set finn_kclk_mhz 200" in body
    assert "REF_PIN_NAME == aclk_kernel_00" in body
    assert "create_generated_clock -name $finn_kclk_name" in body
    # A clock already at the requested period is left alone; a missing pin is reported.
    assert "already at" in body and "CRITICAL WARNING: FINN kernel clock" in body
    assert body.count("{") == body.count("}")
