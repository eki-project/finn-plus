"""Tests for Vitis kernels around double-pumped stitched IP (pumped_kernel).

The kernel wrapper, its interface checks, the link configuration of pumped kernels (second
kernel clock at twice the frequency, clock hook, clock delay groups, clock limit), the
out-of-context clock constraints, and packaging a pumped MVAU into an XO with Vivado.
"""

import pytest

import json
import numpy as np
import zipfile
from onnx import TensorProto
from onnx import helper as oh
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

from finn.builder.build_dataflow_config import FpgaMemoryType, VitisOptStrategy
from finn.transformation.fpgadataflow.create_stitched_ip import ooc_clock_tcl
from finn.transformation.fpgadataflow.pumped_kernel import (
    MAX_PUMPED_CLOCK_MHZ,
    PUMP_2X_CELL,
    PUMP_DIV_CELL,
    check_pumped_kernel_interfaces,
    is_pumped_kernel,
    pumped_kernel_name,
    render_pumped_clock_group_tcl,
    render_pumped_kernel_sv,
)
from finn.transformation.fpgadataflow.vitis_linking_configuration import (
    MAX_CU_NAME_LENGTH,
    BuildBasicVitisLinkConfig,
    VitisLinkConfiguration,
)
from finn.util.basic import make_build_dir
from finn.util.exception import FINNUserError, FINNVitisLinkConfigError

#: An Alveo U55C part and platform.
U55C_PART: str = "xcu55c-fsvh2892-2L-e"
U55C_PLATFORM: str = "xilinx_u55c_gen3x16_xdma_3_202210_1"

#: Interface names of a pumped stitched IP with one input and one output stream.
PUMPED_IFNAMES: dict = {
    "clk": ["ap_clk"],
    "clk2x": ["ap_clk2x"],
    "rst": ["ap_rst_n"],
    "s_axis": [["s_axis_0", 8]],
    "m_axis": [["m_axis_0", 16]],
    "aximm": [],
    "axilite": [],
    "ap_none": [],
}

#: The synthesis stub of such an IP, as write_verilog -mode synth_stub writes it.
PUMPED_STUB: str = """module blk_wrapper(ap_clk, ap_clk2x, ap_rst_n, m_axis_0_tdata,
  m_axis_0_tready, m_axis_0_tvalid, s_axis_0_tdata, s_axis_0_tready, s_axis_0_tvalid,
  sim_finish);
  input ap_clk;
  input ap_clk2x;
  input ap_rst_n;
  output [15:0]m_axis_0_tdata;
  input m_axis_0_tready;
  output m_axis_0_tvalid;
  input [7:0]s_axis_0_tdata;
  output s_axis_0_tready;
  input s_axis_0_tvalid;
  output sim_finish;
endmodule
"""


@pytest.mark.fpgadataflow
def test_pumped_wrapper_derives_both_clocks_from_ap_clk_2() -> None:
    """Both stitched-IP clocks come from ap_clk_2; the streams cross from ap_clk."""
    sv = render_pumped_kernel_sv("blk", PUMPED_IFNAMES, PUMPED_STUB)
    assert pumped_kernel_name("blk") == "blk_dp"
    assert "module blk_dp (" in sv and "endmodule : blk_dp" in sv
    # The 1x clock is ap_clk_2 divided by two, the 2x clock ap_clk_2 itself; both kept.
    assert (
        f'(* DONT_TOUCH = "yes" *) BUFGCE_DIV #(.BUFGCE_DIVIDE(2)) {PUMP_DIV_CELL} (.I(ap_clk_2)'
        in sv
    )
    assert f'(* DONT_TOUCH = "yes" *) BUFGCE {PUMP_2X_CELL} (.I(ap_clk_2)' in sv
    assert "blk_wrapper finn (" in sv
    assert ".ap_clk(clk), .ap_clk2x(clk2x), .ap_rst_n(rst_n)" in sv
    # One clock-crossing FIFO per stream, with the platform clock on the outer side.
    assert sv.count("xpm_fifo_axis #(") == 2
    assert "input	logic [7:0]  s_axis_0_tdata" in sv
    assert "output	logic [15:0]  m_axis_0_tdata" in sv
    assert ".s_aclk(ap_clk), .s_aresetn(ap_rst_n), .m_aclk(clk)" in sv
    assert ".s_aclk(clk), .s_aresetn(rst_n), .m_aclk(ap_clk)" in sv
    assert ".TDATA_WIDTH(8)" in sv and ".TDATA_WIDTH(16)" in sv
    # The reset enters the divided clock domain through a synchronizer.
    assert ".src_arst(ap_rst_n), .dest_clk(clk), .dest_arst(rst_n)" in sv
    # Outputs of the stitched IP that are no stream stay open.
    assert ".sim_finish()" in sv


@pytest.mark.fpgadataflow
def test_pumped_wrapper_passes_on_every_stream() -> None:
    """A stitched IP with several streams gets a port and a FIFO for each."""
    ifnames = dict(PUMPED_IFNAMES)
    ifnames["s_axis"] = [["s_axis_0", 8], ["s_axis_1", 32]]
    ifnames["m_axis"] = [["m_axis_0", 16], ["m_axis_1", 64]]
    sv = render_pumped_kernel_sv("blk", ifnames, "module blk_wrapper();\nendmodule\n")
    assert sv.count("xpm_fifo_axis #(") == 4
    for name in ("s_axis_0", "s_axis_1", "m_axis_0", "m_axis_1"):
        assert f"cdc_{name} (" in sv
        assert f".{name}_tdata({name}_i_tdata)" in sv
    assert "input	logic [31:0]  s_axis_1_tdata" in sv
    assert "output	logic [63:0]  m_axis_1_tdata" in sv


@pytest.mark.fpgadataflow
def test_pumped_kernel_accepts_only_streams() -> None:
    """External or runtime-writeable weights (AXI-MM, AXI-lite) cannot be passed through."""
    assert is_pumped_kernel(PUMPED_IFNAMES)
    assert not is_pumped_kernel({**PUMPED_IFNAMES, "clk2x": []})
    assert not is_pumped_kernel({k: v for k, v in PUMPED_IFNAMES.items() if k != "clk2x"})
    check_pumped_kernel_interfaces("blk", PUMPED_IFNAMES)
    with pytest.raises(FINNUserError, match="AXI-MM or AXI-lite"):
        check_pumped_kernel_interfaces("blk", {**PUMPED_IFNAMES, "axilite": ["s_axi_control"]})
    with pytest.raises(FINNUserError, match="AXI-MM or AXI-lite"):
        check_pumped_kernel_interfaces("blk", {**PUMPED_IFNAMES, "aximm": [["m_axi_gmem0", 64]]})
    with pytest.raises(FINNUserError, match="no AXI streams"):
        check_pumped_kernel_interfaces("blk", {**PUMPED_IFNAMES, "s_axis": [], "m_axis": []})


@pytest.mark.fpgadataflow
def test_pumped_kernel_names_fit_the_xclbin() -> None:
    """The xclbin stores each CU as ``<kernel>:<CU>`` in 63 characters; the link checks it."""
    for cu in ("StreamingDataflowPartition_1", "StreamingDataflowPartition_99"):
        assert len(f"{pumped_kernel_name(cu)}:{cu}") <= MAX_CU_NAME_LENGTH
        VitisLinkConfiguration(Path("conf.txt"), 300, "", U55C_PLATFORM).add_cu(
            pumped_kernel_name(cu), cu
        )
    config = VitisLinkConfiguration(Path("conf.txt"), 300, "", U55C_PLATFORM)
    with pytest.raises(FINNVitisLinkConfigError, match="longer than 63 characters"):
        config.add_cu("StreamingDataflowPartition_1_pumped", "StreamingDataflowPartition_1")


@pytest.mark.fpgadataflow
def test_ooc_place_and_route_clocks_ap_clk2x_at_exactly_half_the_period() -> None:
    """The OOC clocks of a pumped IP are an exact 2:1 in picoseconds."""
    assert ooc_clock_tcl(5.0, False) == ["create_clock -period 5.0 [get_ports ap_clk]"]
    for clk_ns in (2.857, 3.333, 3.0, 1.667):
        full, half = ooc_clock_tcl(clk_ns, True)
        p_full = round(float(full.split()[2]) * 1000)
        p_half = round(float(half.split()[2]) * 1000)
        assert full.endswith("[get_ports ap_clk]") and half.endswith("[get_ports ap_clk2x]")
        assert p_full == 2 * p_half
        assert abs(p_full - clk_ns * 1000) <= 1


def make_sdp_chain(build_dir: Path, pumped: bool, replicas: int = 1) -> ModelWrapper:
    """Build an IDMA -> compute -> ODMA chain of SDP nodes with XO metadata, as BuildAllXOs
    leaves it; the compute partition is pumped if ``pumped`` is set, and the DMAs serve
    ``replicas`` copies of it.
    """
    names = ["StreamingDataflowPartition_0", "StreamingDataflowPartition_1"]
    names.append("StreamingDataflowPartition_2")
    ops = ["IODMA_hls", "Identity", "IODMA_hls"]
    nodes = []
    for i, (name, op) in enumerate(zip(names, ops, strict=True)):
        domain = "finn.custom_op.fpgadataflow.hls" if op == "IODMA_hls" else ""
        sub_node = oh.make_node(op, ["x"], ["y"], name=f"{name}_n", domain=domain)
        if op == "IODMA_hls":
            sub_node.attribute.append(oh.make_attribute("numReplicas", replicas))
        sub_graph = oh.make_graph(
            [sub_node],
            name,
            [oh.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
            [oh.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        )
        sub = ModelWrapper(qonnx_make_model(sub_graph))
        xo = build_dir / f"{name}.xo"
        xo.write_text("")
        sub.set_metadata_prop("vitis_xo", str(xo))
        if op == "Identity":
            ifnames = PUMPED_IFNAMES if pumped else {**PUMPED_IFNAMES, "clk2x": []}
            sub.set_metadata_prop("vivado_stitch_ifnames", json.dumps(ifnames))
            sub.set_metadata_prop("vitis_kernel_name", pumped_kernel_name(name) if pumped else name)
        sub_path = build_dir / f"{name}.onnx"
        sub.save(str(sub_path))
        nodes.append(
            oh.make_node(
                "StreamingDataflowPartition",
                ["inp" if i == 0 else f"t{i}"],
                ["outp" if i == 2 else f"t{i + 1}"],
                name=name,
                domain="finn.custom_op.fpgadataflow",
                model=str(sub_path),
            )
        )
    graph = oh.make_graph(
        nodes,
        "sdps",
        [oh.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4])],
        [oh.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 4])],
    )
    return ModelWrapper(qonnx_make_model(graph))


@pytest.fixture
def no_kernelinfo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip reading the kernel definitions: the test XOs are empty files ``kernelinfo``
    cannot parse.
    """
    monkeypatch.setattr(VitisLinkConfiguration, "_get_kerneldefs", lambda _self: {})


def link_config_for(model: ModelWrapper, f_mhz: int) -> VitisLinkConfiguration:
    """Run BuildBasicVitisLinkConfig on ``model`` and return the device-0 configuration."""
    model = model.transform(
        BuildBasicVitisLinkConfig(
            platform=U55C_PLATFORM,
            board="U55C",
            mem_type=FpgaMemoryType.DEFAULT,
            vitis_opt_strategy=VitisOptStrategy.DEFAULT,
            optimization_level="0",
            f_mhz=f_mhz,
        )
    )
    return VitisLinkConfiguration.load_from_model(model)[0]


@pytest.mark.fpgadataflow
@pytest.mark.usefixtures("no_kernelinfo")
def test_link_clocks_pumped_kernels_from_the_second_kernel_clock() -> None:
    """A pumped compute partition is instantiated from its wrapper kernel and gets ap_clk_2
    at exactly twice the kernel clock, with the clock buffers in one delay group.
    """
    build_dir = Path(make_build_dir("test_pumped_link_"))
    config = link_config_for(make_sdp_chain(build_dir, pumped=True), 300)
    cu = "StreamingDataflowPartition_1"
    assert (pumped_kernel_name(cu), cu) in config.nk
    assert config.pumped_cus == [cu]
    assert config.kernel_frequency() == "0:300|1:600"
    assert "--kernel_frequency '0:300|1:600'" in config.run_script_path.read_text()
    script = (config.config_path.parent / "finn_kernel_clock.tcl").read_text()
    assert "set finn_kclk_mhz 300" in script and "REF_PIN_NAME == aclk_kernel_00" in script
    assert "set finn_kclk_mhz 600" in script and "REF_PIN_NAME == aclk_kernel_01" in script
    assert render_pumped_clock_group_tcl([cu]) in script
    assert f"*/$finn_pump_cu/{PUMP_DIV_CELL}/O" in script
    assert "CLOCK_DELAY_GROUP finn_pump_$finn_pump_cu" in script
    assert script.count("{") == script.count("}")
    # The streams still connect the CU by its instance name.
    assert config.sc["idma_0.m_axis_0"] == [f"{cu}.s_axis_0"]
    assert config.sc[f"{cu}.m_axis_0"] == ["odma_0.s_axis_0"]


@pytest.mark.fpgadataflow
@pytest.mark.usefixtures("no_kernelinfo")
def test_link_clocks_every_replica_of_a_pumped_kernel() -> None:
    """Each replica of a pumped compute partition is a CU of the wrapper kernel with its own
    clock delay group.
    """
    build_dir = Path(make_build_dir("test_pumped_link_replicas_"))
    config = link_config_for(make_sdp_chain(build_dir, pumped=True, replicas=3), 300)
    sdp = "StreamingDataflowPartition_1"
    cus = [f"{sdp}_r{r}" for r in range(3)]
    assert [cu for kernel, cu in config.nk if kernel == pumped_kernel_name(sdp)] == cus
    assert config.pumped_cus == cus
    assert config.kernel_frequency() == "0:300|1:600"
    script = (config.config_path.parent / "finn_kernel_clock.tcl").read_text()
    assert render_pumped_clock_group_tcl(cus) in script


@pytest.mark.fpgadataflow
@pytest.mark.usefixtures("no_kernelinfo")
def test_link_without_pumped_kernels_is_unchanged() -> None:
    """Without pumped kernels the link has one kernel clock and no second clock hook."""
    build_dir = Path(make_build_dir("test_pumped_link_"))
    config = link_config_for(make_sdp_chain(build_dir, pumped=False), 300)
    assert config.pumped_cus == []
    assert config.kernel_frequency() == "300"
    assert "--kernel_frequency 300" in config.run_script_path.read_text()
    script = (config.config_path.parent / "finn_kernel_clock.tcl").read_text()
    assert "aclk_kernel_01" not in script and "CLOCK_DELAY_GROUP" not in script


@pytest.mark.fpgadataflow
@pytest.mark.usefixtures("no_kernelinfo")
def test_link_rejects_a_pumped_clock_above_the_v_plus_plus_limit() -> None:
    """v++ allows at most 650 MHz, so pumped kernels allow at most 325 MHz."""
    build_dir = Path(make_build_dir("test_pumped_link_"))
    link_config_for(make_sdp_chain(build_dir, pumped=True), MAX_PUMPED_CLOCK_MHZ)
    with pytest.raises(FINNUserError, match="at most 325 MHz"):
        link_config_for(make_sdp_chain(build_dir, pumped=True), MAX_PUMPED_CLOCK_MHZ + 1)


def make_pumped_mvau() -> ModelWrapper:
    """Build a single specialized MVAU_rtl with pumped compute, 16x16, INT8, PE=4, SIMD=8."""
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

    node = oh.make_node(
        "MVAU",
        ["inp", "w"],
        ["outp"],
        name="mvau",
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=16,
        MH=16,
        SIMD=8,
        PE=4,
        inputDataType="INT8",
        weightDataType="INT8",
        outputDataType="INT24",
        ActVal=0,
        binaryXnorMode=0,
        noActivation=1,
        preferred_impl_style="rtl",
        resType="dsp",
        pumpedCompute=1,
    )
    graph = oh.make_graph(
        [node],
        "pumped",
        [oh.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 16])],
        [oh.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 16])],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="pumped-kernel-test"))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("w", DataType["INT8"])
    model.set_tensor_datatype("outp", DataType["INT24"])
    rng = np.random.default_rng(0)
    model.set_initializer("w", rng.integers(-128, 128, (16, 16)).astype(np.float32))
    model = model.transform(SpecializeLayers(U55C_PART))
    return model.transform(GiveUniqueNodeNames())


@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.vitis
def test_pumped_mvau_is_packaged_as_a_kernel_with_two_clocks() -> None:
    """A pumped MVAU becomes an XO whose kernel is the wrapper, with ap_clk_2 as a clock."""
    from finn.transformation.fpgadataflow.build_xo import CreateVitisXO
    from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
    from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
    from finn.transformation.fpgadataflow.prepare_ip import PrepareIP

    clk_ns = 3.333
    model = make_pumped_mvau()
    assert model.graph.node[0].op_type == "MVAU_rtl"
    assert getCustomOp(model.graph.node[0]).get_nodeattr("pumpedCompute") == 1
    model = model.transform(PrepareIP(U55C_PART, clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(U55C_PART, clk_ns, "pumped_blk", run_synth=True))
    ifnames = model.get_metadata_prop("vivado_stitch_ifnames")
    assert ifnames is not None and is_pumped_kernel(json.loads(ifnames))
    model = model.transform(CreateVitisXO("pumped_blk", U55C_PART))

    assert model.get_metadata_prop("vitis_kernel_name") == "pumped_blk_dp"
    xo_prop = model.get_metadata_prop("vitis_xo")
    assert xo_prop is not None
    xo = Path(xo_prop)
    with zipfile.ZipFile(xo) as z:
        kernel_xml = next(z.read(n).decode() for n in z.namelist() if n.endswith("kernel.xml"))
        component = next(
            z.read(n).decode()
            for n in z.namelist()
            if n.endswith("component.xml") and "pumped_blk_dp" in n
        )
    assert 'kernel name="pumped_blk_dp"' in kernel_xml
    assert 'port name="s_axis_0"' in kernel_xml and 'port name="m_axis_0"' in kernel_xml
    # Both clocks are clock interfaces of the packaged IP, so v++ connects both.
    for clk in ("ap_clk", "ap_clk_2"):
        assert f"<spirit:name>{clk}</spirit:name>" in component
    assert component.count('spirit:name="clock"') >= 2
    # The kernel is a DCP-based IP with the wrapper and the stitched IP in one checkpoint.
    assert "dcp/pumped_blk_dp.dcp" in component
