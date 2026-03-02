"""Transformations for generating and simulating instrumentation IP."""

import numpy as np
import os
import subprocess
from pathlib import Path
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from finn.custom_op.fpgadataflow.templates import ipgentcl_template
from finn.util.basic import make_build_dir
from finn.util.hls import CallHLS
from finn.util.settings import get_settings


# TODO: duplicate function from make_zynq_proj.py
def collect_ip_dirs(model, ipstitch_path):
    """Collect list of all IP directories required by the design."""
    ip_dirs = []
    need_memstreamer = False
    for node in model.graph.node:
        node_inst = getCustomOp(node)
        ip_dir_value = node_inst.get_nodeattr("ip_path")
        assert os.path.isdir(
            ip_dir_value
        ), """The directory that should
        contain the generated ip blocks doesn't exist."""
        ip_dirs += [ip_dir_value]
        if node.op_type.startswith("MVAU") or node.op_type == "Thresholding_hls":
            if node_inst.get_nodeattr("mem_mode") == "internal_decoupled":
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
        fpga_part,
        clk_period_ns,
        format="ip",  # "ip" for Vivado (Zynq) or "xo" for Vitis (Alveo/Versal)
    ):
        """Initialize instrumentation IP generation with FPGA part and clock settings."""
        super().__init__()
        self.fpga_part = fpga_part
        self.clk_period_ns = clk_period_ns
        self.format = format

    def apply(self, model):
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
        ti = "ap_uint<%d>" % inp_stream_width
        # perform the same for the output
        out_name = model.graph.output[0].name
        out_node = getCustomOp(model.find_producer(out_name))
        out_shape_folded = list(out_node.get_folded_output_shape())
        out_stream_width = out_node.get_outstream_width_padded()
        olen = np.prod(out_shape_folded[:-1])
        to = "ap_uint<%d>" % out_stream_width
        ko = out_shape_folded[-1]
        # fill out instrumentation wrapper template
        with open(
            os.path.join(get_settings().finn_custom_hls, "instrumentation.template.cpp"), "r"
        ) as f:
            instrwrp_cpp = f.read()
        instrwrp_cpp = instrwrp_cpp.replace("@PENDING@", str(pending))
        instrwrp_cpp = instrwrp_cpp.replace("@ILEN@", str(ilen))
        instrwrp_cpp = instrwrp_cpp.replace("@OLEN@", str(olen))
        instrwrp_cpp = instrwrp_cpp.replace("@TI@", str(ti))
        instrwrp_cpp = instrwrp_cpp.replace("@TO@", str(to))
        instrwrp_cpp = instrwrp_cpp.replace("@KO@", str(ko))
        with open(wrapper_output_dir + "/top_instrumentation_wrapper.cpp", "w") as f:
            f.write(instrwrp_cpp)
        # fill out HLS synthesis tcl template
        prjname = "project_instrwrap"
        ipgentcl = ipgentcl_template
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
        if self.format == "xo":
            # use Vitis RTL kernel (.xo) output instead of IP-XACT
            ipgentcl = ipgentcl.replace("$EXTRA_DIRECTIVES$", "config_export -format xo")
            ipgentcl = ipgentcl.replace(
                "export_design -format ip_catalog", "export_design -format xo"
            )
        else:
            ipgentcl = ipgentcl.replace("$EXTRA_DIRECTIVES$", "")
        with open(wrapper_output_dir + "/hls_syn.tcl", "w") as f:
            f.write(ipgentcl)
        # build bash script to launch HLS synth and call it
        code_gen_dir = Path(wrapper_output_dir)
        builder = CallHLS(
            tcl_script=code_gen_dir / "hls_syn.tcl",
            code_gen_dir=code_gen_dir,
            ipgen_path=code_gen_dir / prjname,
        )
        builder.build()
        ipgen_path = builder.ipgen_path
        assert ipgen_path.is_dir(), "HLS IPGen failed: %s not found" % (ipgen_path)
        ip_path = ipgen_path / "sol1" / "impl" / "ip"
        assert ip_path.is_dir(), "HLS IPGen failed: %s not found. Check log under %s" % (
            ip_path,
            code_gen_dir,
        )
        if self.format == "xo":
            assert False, "Not implemented"
            # TODO: export for use in VitisBuild or VersalBuild
            # xo_dir = self.output_dir + "/xo"
            # xo_dir = str(os.path.abspath(xo_dir))
            # os.makedirs(xo_dir, exist_ok=True)
            # xo_path = code_gen_dir + "/{}/sol1/impl/export.xo".format(prjname)
            # xo_instr_path = xo_dir + "/instrumentation_wrapper.xo"
            # shutil.copy(xo_path, xo_instr_path)
        else:
            # shutil.move(ip_path, self.output_dir)
            pass

        return (model, False)


class PrepareInstrumentationSim(Transformation):
    """Prepare simulation environment for instrumentation."""

    def __init__(self, fpga_part):
        """Initialize instrumentation simulation preparation."""
        super().__init__()
        self.fpga_part = fpga_part

    def apply(self, model):
        """Prepare scripts for simulating instrumentation IP."""
        # Create directory for simulation of instrumentation IP + FINN IP
        sim_output_dir = make_build_dir(prefix="sim_Instrumentation_")
        model.set_metadata_prop("instrumentation_sim", sim_output_dir)

        # check if instrumentation IP was generated
        instr_ip_dir = model.get_metadata_prop("instrumentation_ipgen")
        if instr_ip_dir is None or (not os.path.isdir(instr_ip_dir)):
            raise Exception(
                "Instrumentation IP not generated, run GenerateInstrumentationIP first."
            )

        # TODO: Support simulation with AXI-lite control interfaces (e.g., for dynamic pipelines)
        # fill in testbench template
        with open(
            os.path.join(get_settings().finn_custom_hls, "instrumentation_tb.template.sv"),
            "r",
        ) as f:
            testbench_sv = f.read()
        with open(sim_output_dir + "/instrwrap_testbench.sv", "w") as f:
            f.write(testbench_sv)
        # fill in testbench project creator template
        with open(
            os.path.join(get_settings().finn_custom_hls, "instrumentation_sim.template.tcl"),
            "r",
        ) as f:
            testbench_tcl = f.read()

        # collect ip repo paths for finn accelerator sub cores so Vivado can find them
        ipstitch_path = model.get_metadata_prop("vivado_stitch_proj")
        ip_dirs = ["list"]
        ip_dirs += collect_ip_dirs(model, ipstitch_path)
        ip_dirs += [instr_ip_dir]
        ip_dirs_str = "[%s]" % (" ".join(ip_dirs))
        testbench_tcl = testbench_tcl.replace("@FPGA_PART@", self.fpga_part)
        testbench_tcl = testbench_tcl.replace("@IP_DIRS_STR@", ip_dirs_str)
        with open(sim_output_dir + "/make_instrwrap_sim_proj.tcl", "w") as f:
            f.write(testbench_tcl)

        return (model, False)


class RunInstrumentationSim(Transformation):
    """Run instrumentation simulation to collect performance data."""

    def __init__(self):
        """Initialize instrumentation simulation runner."""
        super().__init__()

    def apply(self, model):
        """Run instrumentation simulation script."""
        sim_output_dir = model.get_metadata_prop("instrumentation_sim")
        if sim_output_dir is None or (not os.path.isdir(sim_output_dir)):
            raise Exception(
                "Instrumentation sim not prepared, run PrepareInstrumentationSim first."
            )

        # Prepare bash script
        bash_script = os.getcwd() + "/report_power.sh"
        with open(bash_script, "w") as script:
            script.write("#!/bin/bash\n")
            script.write("cd %s\n" % (sim_output_dir))
            script.write("vivado -mode batch -source make_instrwrap_sim_proj.tcl\n")

        # Run script
        print("Running Vivado simulation of instrumentation wrapper")
        sub_proc = subprocess.Popen(["bash", bash_script])
        sub_proc.communicate()

        return (model, False)
