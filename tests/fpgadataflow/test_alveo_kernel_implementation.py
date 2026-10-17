"""Tests for the Alveo/Zynq kernel implementation features.

Kernel clock in implementation, per-SLR kernel resets, the reset pipeline, the
kernel XDC scoping, AXI-stream register slices, the performance strategies'
Vivado settings, the v++ config sections, and the platform PCIe SLR.
"""

import pytest

import numpy as np
from onnx import TensorProto
from onnx import helper as oh
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

from finn.builder.build_dataflow_config import FpgaMemoryType, VitisOptStrategy
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.vitis_linking_configuration import VitisLinkConfiguration
from finn.util.basic import make_build_dir
from finn.util.platforms import platforms

#: An Alveo U55C part.
U55C_PART: str = "xcu55c-fsvh2892-2L-e"


def make_chain(length: int) -> ModelWrapper:
    """Build a specialized linear chain of `length` small MVAUs for the U55C."""
    nodes = []
    for i in range(length):
        inp = "inp" if i == 0 else f"t{i}"
        outp = "outp" if i == length - 1 else f"t{i + 1}"
        nodes.append(
            oh.make_node(
                "MVAU",
                [inp, f"mvau{i}_w"],
                [outp],
                name=f"mvau{i}",
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                MW=4,
                MH=4,
                SIMD=1,
                PE=1,
                inputDataType="INT2",
                weightDataType="INT2",
                outputDataType="INT2",
                ActVal=0,
                binaryXnorMode=0,
                noActivation=1,
            )
        )
    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4])
    outp = oh.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 4])
    graph = oh.make_graph(nodes=nodes, name="chain", inputs=[inp], outputs=[outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="kernel-impl-test"))
    for node in nodes:
        for tensor in list(node.input) + list(node.output):
            model.set_tensor_datatype(tensor, DataType["INT2"])
        model.set_initializer(node.input[1], np.ones((4, 4), dtype=np.float32))
    model = model.transform(SpecializeLayers(U55C_PART))
    return model.transform(GiveUniqueNodeNames())


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

    model = make_chain(2)
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
    already ended with one), so a second `prop=` line was glued onto the first and v++
    read the first entry's value with the second appended (13 Oct 2026).
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

    model = make_chain(3)
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
    """Every link config runs a kernel-clock script before opt_design."""
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


@pytest.mark.fpgadataflow
def test_confirmed_pcie_slr_is_used_verbatim() -> None:
    """U55C and U280 have a confirmed PCIe SLR, which must be returned as recorded."""
    for board in ("U55C", "U280"):
        platform = platforms[board]()
        assert platform.pcie_slr == 0
        assert platform.get_pcie_slr() == 0


@pytest.mark.fpgadataflow
def test_unconfirmed_pcie_slr_falls_back_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """A multi-SLR board with no confirmed value falls back to SLR 0 and says so."""
    platform = platforms["U250"]()
    assert platform.pcie_slr is None
    with caplog.at_level("WARNING"):
        assert platform.get_pcie_slr() == 0
    assert "pcie_slr" in caplog.text


@pytest.mark.fpgadataflow
def test_single_slr_board_does_not_warn_about_pcie_slr(caplog: pytest.LogCaptureFixture) -> None:
    """On a single-SLR part there is nothing to choose, so the warning would be noise."""
    platform = platforms["Pynq-Z1"]()
    with caplog.at_level("WARNING"):
        assert platform.get_pcie_slr() == 0
    assert "pcie_slr" not in caplog.text


@pytest.mark.fpgadataflow
def test_rtl_mvau_lut_estimate_counts_weights_in_lutram() -> None:
    """The RTL MVAU's LUT estimate is no longer 0 and grows by the LUTRAM weight term."""
    from qonnx.custom_op.registry import getCustomOp

    model = make_chain(1)
    node = model.graph.node[0]
    node.op_type = "MVAU_rtl"
    node.domain = "finn.custom_op.fpgadataflow.rtl"
    inst = getCustomOp(node)
    inst.set_nodeattr("mem_mode", "internal_decoupled")
    inst.set_nodeattr("ram_style", "block")
    in_bram = inst.lut_estimation(U55C_PART)
    inst.set_nodeattr("ram_style", "distributed")
    in_lutram = inst.lut_estimation(U55C_PART)
    assert in_bram > 0
    pe, simd = inst.get_nodeattr("PE"), inst.get_nodeattr("SIMD")
    wbits = inst.get_input_datatype(1).bitwidth()
    assert in_lutram - in_bram == pe * simd * wbits * -(-inst.calc_wmem() // 64)
