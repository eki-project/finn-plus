from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.benchmarking.bench_base import bench

class bench_vgg10(bench):
    def step_build_setup(self):
        # create build config for VGG-10 (based on finn-examples)
        vgg10_build_steps = [
            "step_tidy_up",
            step_pre_streamline,
            "step_streamline",
            "step_convert_to_hw",
            step_convert_final_layers,
            "step_create_dataflow_partition",
            "step_specialize_layers",
            "step_target_fps_parallelization",
            "step_apply_folding_config",
            "step_minimize_bit_width",
            "step_generate_estimate_reports",
            "step_set_fifo_depths",
            "step_hw_codegen",
            "step_hw_ipgen",
            "step_create_stitched_ip",
            "step_measure_rtlsim_performance",
            "step_out_of_context_synthesis",
            "step_synthesize_bitfile",
            "step_make_driver",
            "step_deployment_package",
        ]

        cfg = DataflowBuildConfig(
            steps=vgg10_build_steps,
            standalone_thresholds=True,
        )

        return cfg
