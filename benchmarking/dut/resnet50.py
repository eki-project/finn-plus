import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import alveo_default_platform

from dut.resnet50_custom_steps import (
        step_resnet50_tidy,
        step_resnet50_streamline,
        step_resnet50_convert_to_hw,
        step_resnet50_slr_floorplan,
    )

from bench_base import bench

class bench_resnet50(bench):
    def step_build_setup(self):
        # create build config for ResNet-50 (based on finn-examples)

        resnet50_build_steps = [
            step_resnet50_tidy,
            step_resnet50_streamline,
            step_resnet50_convert_to_hw,
            "step_create_dataflow_partition",
            "step_specialize_layers",
            "step_apply_folding_config",
            "step_minimize_bit_width",
            "step_generate_estimate_reports",
            "step_set_fifo_depths",
            "step_hw_codegen",
            "step_hw_ipgen",
            step_resnet50_slr_floorplan,
            "step_create_stitched_ip", # was not in finn-examples
            "step_measure_rtlsim_performance", # was not in finn-examples
            "step_out_of_context_synthesis", # was not in finn-examples
            "step_synthesize_bitfile",
            "step_make_pynq_driver",
            "step_deployment_package",
        ]

        cfg = build_cfg.DataflowBuildConfig(
            steps=resnet50_build_steps,
        )

        return cfg