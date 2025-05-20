from finn.benchmarking.bench_base import bench
from finn.builder.build_dataflow_config import DataflowBuildConfig


class bench_mobilenetv1(bench):
    def step_build_setup(self):
        # create build config for MobileNetV1 (based on finn-examples)
        mobilenet_build_steps = [
            step_mobilenet_streamline,
            step_mobilenet_lower_convs,
            step_mobilenet_convert_to_hw_layers_separate_th,
            "step_create_dataflow_partition",
            "step_specialize_layers",
            "step_apply_folding_config",
            "step_minimize_bit_width",
            "step_generate_estimate_reports",
            "step_set_fifo_depths",
            "step_hw_codegen",
            "step_hw_ipgen",
            "step_create_stitched_ip",
            "step_synthesize_bitfile",
            "step_make_driver",
            "step_deployment_package",
        ]
        # mobilenet_build_steps_alveo = [
        #     step_mobilenet_streamline,
        #     step_mobilenet_lower_convs,
        #     step_mobilenet_convert_to_hw_layers,
        #     "step_create_dataflow_partition",
        #     "step_specialize_layers",
        #     "step_apply_folding_config",
        #     "step_minimize_bit_width",
        #     "step_generate_estimate_reports",
        #     "step_hw_codegen",
        #     "step_hw_ipgen",
        #     "step_set_fifo_depths",
        #     "step_create_stitched_ip",
        #     step_mobilenet_slr_floorplan,
        #     "step_synthesize_bitfile",
        #     "step_make_pynq_driver",
        #     "step_deployment_package",
        # ]

        cfg = DataflowBuildConfig(
            steps=mobilenet_build_steps,
        )

        return cfg
