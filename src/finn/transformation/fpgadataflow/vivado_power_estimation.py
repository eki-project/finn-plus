"""Module for vivado power estimation."""
import json
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.benchmarking.util import power_xml_to_dict
from finn.transformation.fpgadataflow.templates import (
    template_switching_simulation_sh,
    template_switching_simulation_tb,
    template_switching_simulation_xsim_tcl,
    template_vivado_open_checkpoint,
    template_vivado_power_fixed,
    template_vivado_power_from_saif,
    template_vivado_write_funcsim_netlist,
    template_vivado_write_timesim_netlist,
)
from finn.util.basic import launch_process_helper, make_build_dir


class VivadoPowerEstimation(Transformation):
    """Run Vivado power estimation on the stitched IP after OOC synthesis.
    simulate_switching_activity: False = use a fixed set of toggle rates and static probabilities.
    True = additionally simulate the switching activity of the design for power estimation."""

    def __init__(
        self,
        report_dir,
        clk_period_ns=10.0,
        simulate_switching_activity=True,
        vivado_power_simulation_type="functional",
    ):
        """Initialize instance."""
        super().__init__()
        self.report_dir = report_dir
        self.clk_period_ns = clk_period_ns
        self.simulate_switching_activity = simulate_switching_activity
        self.vivado_power_simulation_type = vivado_power_simulation_type

    def run_vivado_tcl(self, script, tmp_dir, name):
        """Write the given Tcl script into tmp_dir and run it through Vivado in batch mode."""
        tcl_path = f"{tmp_dir}/{name}.tcl"
        with open(tcl_path, "w") as tcl_file:
            tcl_file.write(script)
        self.run_bash_script(
            f"#!/bin/bash \nvivado -mode batch -source {tcl_path}\n", tmp_dir, name
        )

    def run_bash_script(self, script, tmp_dir, name):
        """Write the given shell script into tmp_dir and run it."""
        bash_path = f"{tmp_dir}/{name}.sh"
        with open(bash_path, "w") as bash_file:
            bash_file.write(script)
        launch_process_helper(["bash", bash_path], cwd=tmp_dir)

    def apply(self, model):
        """Apply transformation."""
        ooc_res_dict = eval(model.get_metadata_prop("res_total_ooc_synth"))
        # the routed out-of-context design checkpoint written by SynthOutOfContext
        routed_dcp = ooc_res_dict["routed_dcp"]
        tmp_dir = make_build_dir("vivado_power_estimation_")

        power_summary_dict = dict()
        if self.simulate_switching_activity:
            # Simulate for at least 100 cycles (in case layer is fully unrolled)
            # TODO: improve heuristic for sim duration,
            # probably too short for anything but (single layer) microbenchmarks
            perf_estimate = model.analysis(dataflow_performance)
            sim_duration_ns = max(100, perf_estimate["max_cycles"]) * self.clk_period_ns

            input_tensor = model.graph.input[0]
            output_tensor = model.graph.output[0]
            input_node_inst = getCustomOp(model.find_consumer(input_tensor.name))
            output_node_inst = getCustomOp(model.find_producer(output_tensor.name))
            in_width = input_node_inst.get_instream_width()
            out_width = output_node_inst.get_outstream_width()
            dtype_width = model.get_tensor_datatype(input_tensor.name).bitwidth()

            netlist_path = tmp_dir + "/post_impl_netlist.v"
            sdf_path = tmp_dir + "/post_impl_netlist.sdf"
            tb_path = tmp_dir + "/switching_simulation_tb.v"
            sim_tcl_path = tmp_dir + "/switching_simulation.tcl"
            saif_path = tmp_dir + "/switching.saif"

            # Prepare testbench
            # TODO: infer top module name instead of hardcoding "finn_design_wrapper"
            # top_module_name = model.get_metadata_prop("wrapper_filename")
            # top_module_name = file_to_basename(top_module_name).strip(".v")
            testbench = template_switching_simulation_tb.replace("$INSTREAM_WIDTH$", str(in_width))
            testbench = testbench.replace("$OUTSTREAM_WIDTH$", str(out_width))
            testbench = testbench.replace("$DTYPE_WIDTH$", str(dtype_width))
            testbench = testbench.replace(
                "$RANDOM_FUNCTION$", f"$urandom_range(0, {2**dtype_width - 1})"
            )
            with open(tb_path, "w") as tb_file:
                tb_file.write(testbench)

            # Export a simulation netlist of the routed design from the checkpoint
            if self.vivado_power_simulation_type == "timing":
                script = template_vivado_write_timesim_netlist.replace("$SDF_PATH$", sdf_path)
                # the timing netlist needs the SIMPRIM library and the delays of the routed
                # design annotated onto the DUT instance
                xelab_options = (
                    "-L simprims_ver -L secureip -transport_int_delays -pulse_r 0 "
                    f"-pulse_int_r 0 -sdfmax /switching_simulation_tb/dut={sdf_path}"
                )
            else:
                script = template_vivado_write_funcsim_netlist
                xelab_options = "-L unisims_ver -L unimacro_ver -L secureip"
            script = script.replace("$DCP_PATH$", routed_dcp)
            script = script.replace("$NETLIST_PATH$", netlist_path)
            self.run_vivado_tcl(script, tmp_dir, "write_sim_netlist")

            # Simulate the netlist to record switching activity into a SAIF file
            sim_tcl = template_switching_simulation_xsim_tcl.replace("$SAIF_FILE_PATH$", saif_path)
            sim_tcl = sim_tcl.replace("$SIM_DURATION_NS$", str(int(sim_duration_ns)))
            with open(sim_tcl_path, "w") as tcl_file:
                tcl_file.write(sim_tcl)
            script = template_switching_simulation_sh.replace("$NETLIST_PATH$", netlist_path)
            script = script.replace("$TB_FILE_PATH$", tb_path)
            script = script.replace("$XELAB_OPTIONS$", xelab_options)
            script = script.replace("$SIM_TCL_PATH$", sim_tcl_path)
            self.run_bash_script(script, tmp_dir, "simulate_switching_activity")

            # Read the switching activity back into the routed design and report power
            script = template_vivado_open_checkpoint.replace("$DCP_PATH$", routed_dcp)
            script = script + template_vivado_power_from_saif
            script = script.replace("$SAIF_FILE_PATH$", saif_path)
            script = script.replace("$REPORT_PATH$", self.report_dir)
            script = script.replace("$REPORT_NAME$", "power_estimate_sim")
            self.run_vivado_tcl(script, tmp_dir, "power_from_saif")

            # Parse results
            power_report_dict = power_xml_to_dict(f"{self.report_dir}/power_estimate_sim.xml")
            power_report_json = f"{self.report_dir}/power_estimate_sim.json"
            with open(power_report_json, "w") as json_file:
                json_file.write(json.dumps(power_report_dict, indent=2))

            # Separately log most important summary metrics:
            power_summary_dict["power_sim_total"] = float(
                power_report_dict["Summary"]["tables"][0]["Total On-Chip Power (W)"][0]
            )
            power_summary_dict["power_sim_static"] = float(
                power_report_dict["Summary"]["tables"][0]["Device Static (W)"][0]
            )
            power_summary_dict["power_sim_dynamic"] = float(
                power_report_dict["Summary"]["tables"][0]["Dynamic (W)"][0]
            )

        # FAST POWER ESTIMATION (no simulation of switching activity):
        # Generate power report based on the following (toggle rate, static probability) settings
        # TODO: make configurable and more fine-grained (per cell type)
        activity_settings = [(12.5, 0.5), (25, 0.5), (50, 0.5), (75, 0.5), (100, 0.5)]
        # Prepare tcl script
        script = template_vivado_open_checkpoint.replace("$DCP_PATH$", routed_dcp)
        for toggle_rate, static_prob in activity_settings:
            script = script + template_vivado_power_fixed
            script = script.replace("$TOGGLE_RATE$", str(toggle_rate))
            script = script.replace("$STATIC_PROB$", str(static_prob))
            # script = script.replace("$SWITCH_TARGET$", switch_target)
            script = script.replace("$REPORT_PATH$", self.report_dir)
            script = script.replace("$REPORT_NAME$", f"power_estimate_{toggle_rate}_{static_prob}")
        script = script + "\nclose_project\n"
        self.run_vivado_tcl(script, tmp_dir, "power_report_fixed")

        # Parse results
        for toggle_rate, static_prob in activity_settings:
            power_report_dict = power_xml_to_dict(
                f"{self.report_dir}/power_estimate_{toggle_rate}_{static_prob}.xml"
            )
            power_report_json = f"{self.report_dir}/power_estimate_{toggle_rate}_{static_prob}.json"
            with open(power_report_json, "w") as json_file:
                json_file.write(json.dumps(power_report_dict, indent=2))

            # Separately log most important summary metrics:
            power_summary_dict[f"power_{toggle_rate}_{static_prob}_total"] = float(
                power_report_dict["Summary"]["tables"][0]["Total On-Chip Power (W)"][0]
            )
            power_summary_dict[f"power_{toggle_rate}_{static_prob}_static"] = float(
                power_report_dict["Summary"]["tables"][0]["Device Static (W)"][0]
            )
            power_summary_dict[f"power_{toggle_rate}_{static_prob}_dynamic"] = float(
                power_report_dict["Summary"]["tables"][0]["Dynamic (W)"][0]
            )

        # Save summary report
        power_summary_json = f"{self.report_dir}/power_estimate_summary.json"
        with open(power_summary_json, "w") as file:
            json.dump(power_summary_dict, file, indent=2)

        return (model, False)
