"""Transformations for generating and simulating instrumentation IP."""

import math
import numpy as np
import subprocess
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from finn.templates import load_codegen_template
from finn.util.basic import make_build_dir
from finn.util.exception import FINNInternalError
from finn.util.hls import CallHLS
from finn.util.settings import get_settings


# TODO: duplicate function from make_zynq_proj.py
def collect_ip_dirs(model: ModelWrapper, ipstitch_path: str) -> list[str]:
    """Collect list of all IP directories required by the design."""
    ip_dirs = []
    need_memstreamer = False
    for node in model.graph.node:
        node_inst = getCustomOp(node)
        ip_dir_value = node_inst.get_nodeattr("ip_path")
        if not Path(ip_dir_value).is_dir():
            raise FINNInternalError(
                f"{node.name}: the directory that should contain the generated ip blocks "
                f"doesn't exist: {ip_dir_value}"
            )
        ip_dirs += [ip_dir_value]
        if (
            node.op_type.startswith("MVAU") or node.op_type == "Thresholding_hls"
        ) and node_inst.get_nodeattr("mem_mode") == "internal_decoupled":
            need_memstreamer = True
    ip_dirs += [ipstitch_path + "/ip"]
    if need_memstreamer:
        # add RTL streamer IP
        ip_dirs.append("$::env(FINN_RTLLIB)/memstream")
    return ip_dirs


class GenerateInstrumentationIP(Transformation):
    """Generate instrumentation IP for performance monitoring."""

    def __init__(
        self,
        fpga_part: str,
        clk_period_ns: float,
        avg_n: int = 64,
        # "ip" for Vivado (Zynq) or "xo" for Vitis (Alveo/Versal)
        output_format: str = "ip",
    ) -> None:
        """Initialize instrumentation IP generation with FPGA part and clock settings."""
        super().__init__()
        self.fpga_part = fpga_part
        self.clk_period_ns = clk_period_ns
        self.avg_n = avg_n
        self.output_format = output_format

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Generate instrumentation IP core."""
        # Create directory for code-gen and HLS of instrumentation IP
        wrapper_output_dir = make_build_dir(prefix="code_gen_ipgen_Instrumentation_")
        model.set_metadata_prop("instrumentation_ipgen", wrapper_output_dir)

        # Heuristic for setting timestamp buffer size of instrumentation wrapper:
        # Currently set to minimum of 1024 or number of layers (if larger) to avoid
        # overflow issues with small designs during (initial) live FIFO sizing.
        # TODO: Implement a better heuristic.
        pending = max(len(model.graph.node), 1024)
        # query the parallelism-dependent folded input shape from the
        # node consuming the graph input
        inp_name = model.graph.input[0].name
        inp_node = getCustomOp(model.find_consumer(inp_name))
        inp_shape_folded = list(inp_node.get_folded_input_shape())
        inp_stream_width = inp_node.get_instream_width_padded()
        # number of beats per input is given by product of folded input
        # shape except the last dim (which is the stream width)
        ilen = np.prod(inp_shape_folded[:-1])
        ti = f"ap_uint<{inp_stream_width}>"
        # perform the same for the output
        out_name = model.graph.output[0].name
        out_node = getCustomOp(model.find_producer(out_name))
        out_shape_folded = list(out_node.get_folded_output_shape())
        out_stream_width = out_node.get_outstream_width_padded()
        olen = np.prod(out_shape_folded[:-1])
        to = f"ap_uint<{out_stream_width}>"
        ko = out_shape_folded[-1]
        # fill out instrumentation wrapper template
        instrwrp_cpp = (
            Path(get_settings().finn_custom_hls) / "instrumentation.template.cpp"
        ).read_text()
        instrwrp_cpp = instrwrp_cpp.replace("@PENDING@", str(pending))
        instrwrp_cpp = instrwrp_cpp.replace("@AVG_N@", str(self.avg_n))
        instrwrp_cpp = instrwrp_cpp.replace("@ILEN@", str(ilen))
        instrwrp_cpp = instrwrp_cpp.replace("@OLEN@", str(olen))
        instrwrp_cpp = instrwrp_cpp.replace("@TI@", str(ti))
        instrwrp_cpp = instrwrp_cpp.replace("@TO@", str(to))
        instrwrp_cpp = instrwrp_cpp.replace("@KO@", str(ko))
        # Derive the number of distinct tUSER values from NodeContainer nodes in the model.
        # PR and SW containers both use tUSER low-bits to select the active body/weight-set.
        # Take the max across all NodeContainers so the width covers all of them.
        num_tuser_values = 1
        for nc in model.get_nodes_by_op_type("NodeContainer"):
            nc_inst = getCustomOp(nc)
            multi_dnn_type = nc_inst.get_nodeattr("multi_dnn_type")
            if multi_dnn_type in ("partial_reconfiguration", "selectable_weights"):
                num_tuser_values = max(num_tuser_values, nc_inst.get_nodeattr("bodies"))
        tuser_width = max(math.ceil(math.log2(max(num_tuser_values, 2))), 1)
        instrwrp_cpp = instrwrp_cpp.replace("@TUSER_WIDTH@", str(tuser_width))
        instrwrp_cpp = instrwrp_cpp.replace("@NUM_TUSER_VALUES@", str(num_tuser_values))
        (Path(wrapper_output_dir) / "top_instrumentation_wrapper.cpp").write_text(instrwrp_cpp)
        # fill out HLS synthesis tcl template
        prjname = "project_instrwrap"
        ipgentcl = load_codegen_template("hls_ipgen_project.tcl")
        ipgentcl = ipgentcl.replace("$PROJECTNAME$", prjname)
        ipgentcl = ipgentcl.replace("$HWSRCDIR$", str(wrapper_output_dir))
        ipgentcl = ipgentcl.replace("$FINNHLSLIB$", str(get_settings().finn_deps / "finn-hlslib"))
        ipgentcl = ipgentcl.replace(
            "$ATTENTIONHLSLIB$", str(get_settings().finn_deps / "attention-hlslib")
        )
        ipgentcl = ipgentcl.replace("$TOPFXN$", "instrumentation_wrapper")
        ipgentcl = ipgentcl.replace("$FPGAPART$", self.fpga_part)
        ipgentcl = ipgentcl.replace("$CLKPERIOD$", str(self.clk_period_ns))
        ipgentcl = ipgentcl.replace("$DEFAULT_DIRECTIVES$", "")
        if self.output_format == "xo":
            # use Vitis RTL kernel (.xo) output instead of IP-XACT
            ipgentcl = ipgentcl.replace("$EXTRA_DIRECTIVES$", "config_export -format xo")
            ipgentcl = ipgentcl.replace(
                "export_design -format ip_catalog", "export_design -format xo"
            )
        else:
            ipgentcl = ipgentcl.replace("$EXTRA_DIRECTIVES$", "")
        (Path(wrapper_output_dir) / "hls_syn.tcl").write_text(ipgentcl)
        # build bash script to launch HLS synth and call it
        code_gen_dir = Path(wrapper_output_dir)
        builder = CallHLS(
            tcl_script=code_gen_dir / "hls_syn.tcl",
            code_gen_dir=code_gen_dir,
            ipgen_path=code_gen_dir / prjname,
        )
        builder.build()
        ipgen_path = builder.ipgen_path
        if not ipgen_path.is_dir():
            raise FINNInternalError(f"HLS IPGen failed: {ipgen_path} not found")
        ip_path = ipgen_path / "sol1" / "impl" / "ip"
        if not ip_path.is_dir():
            raise FINNInternalError(
                f"HLS IPGen failed: {ip_path} not found. Check log under {code_gen_dir}"
            )
        if self.output_format == "xo":
            raise FINNInternalError("Instrumentation IP export as .xo is not implemented")
            # TODO: export for use in VitisBuild or VersalBuild
            # xo_dir = self.output_dir + "/xo"
            # xo_dir = str(os.path.abspath(xo_dir))
            # os.makedirs(xo_dir, exist_ok=True)
            # xo_path = code_gen_dir + "/{}/sol1/impl/export.xo".format(prjname)
            # xo_instr_path = xo_dir + "/instrumentation_wrapper.xo"
            # shutil.copy(xo_path, xo_instr_path)
        # shutil.move(ip_path, self.output_dir)

        return (model, False)


class PrepareInstrumentationSim(Transformation):
    """Prepare simulation environment for instrumentation."""

    def __init__(self, fpga_part: str) -> None:
        """Initialize instrumentation simulation preparation."""
        super().__init__()
        self.fpga_part = fpga_part

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Prepare scripts for simulating instrumentation IP."""
        # Create directory for simulation of instrumentation IP + FINN IP
        sim_output_dir = make_build_dir(prefix="sim_Instrumentation_")
        model.set_metadata_prop("instrumentation_sim", sim_output_dir)

        # check if instrumentation IP was generated
        instr_ip_dir = model.get_metadata_prop("instrumentation_ipgen")
        if instr_ip_dir is None or (not Path(instr_ip_dir).is_dir()):
            raise FINNInternalError(
                "Instrumentation IP not generated, run GenerateInstrumentationIP first."
            )

        # TODO: Support simulation with AXI-lite control interfaces (e.g., for dynamic pipelines)
        # fill in testbench template
        testbench_sv = (
            Path(get_settings().finn_custom_hls) / "instrumentation_tb.template.sv"
        ).read_text()
        (Path(sim_output_dir) / "instrwrap_testbench.sv").write_text(testbench_sv)
        # fill in testbench project creator template
        testbench_tcl = (
            Path(get_settings().finn_custom_hls) / "instrumentation_sim.template.tcl"
        ).read_text()

        # collect ip repo paths for finn accelerator sub cores so Vivado can find them
        ipstitch_path = model.get_metadata_prop("vivado_stitch_proj")
        ip_dirs = ["list"]
        ip_dirs += collect_ip_dirs(model, ipstitch_path)
        ip_dirs += [instr_ip_dir]
        ip_dirs_str = f"[{' '.join(ip_dirs)}]"
        testbench_tcl = testbench_tcl.replace("@FPGA_PART@", self.fpga_part)
        testbench_tcl = testbench_tcl.replace("@IP_DIRS_STR@", ip_dirs_str)
        (Path(sim_output_dir) / "make_instrwrap_sim_proj.tcl").write_text(testbench_tcl)

        return (model, False)


class RunInstrumentationSim(Transformation):
    """Run instrumentation simulation to collect performance data."""

    def __init__(self) -> None:
        """Initialize instrumentation simulation runner."""
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Run instrumentation simulation script."""
        sim_output_dir = model.get_metadata_prop("instrumentation_sim")
        if sim_output_dir is None or (not Path(sim_output_dir).is_dir()):
            raise FINNInternalError(
                "Instrumentation sim not prepared, run PrepareInstrumentationSim first."
            )

        # Prepare bash script
        bash_script = Path.cwd() / "report_power.sh"
        with bash_script.open("w") as script:
            script.write("#!/bin/bash\n")
            script.write(f"cd {sim_output_dir}\n")
            script.write("vivado -mode batch -source make_instrwrap_sim_proj.tcl\n")

        # Run script
        print("Running Vivado simulation of instrumentation wrapper")
        sub_proc = subprocess.Popen(["bash", str(bash_script)])
        sub_proc.communicate()

        return (model, False)
