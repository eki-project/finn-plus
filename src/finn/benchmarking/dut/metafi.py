import finn.builder.build_dataflow_config as build_cfg

from finn.benchmarking.bench_base import bench

# # custom steps
# from custom_steps import (
#     step_extract_absorb_bias,
#     step_pre_streamline,
#     step_residual_convert_to_hw,
#     step_residual_streamline,
#     step_residual_tidy,
#     step_residual_topo,
#     step_set_preferred_impl_style,
#     step_convert_final_layers
# )

class bench_metafi(bench):
    def step_build_setup(self):
        # create build config for MetaFi models

        steps = [
            # step_residual_tidy,
            # step_extract_absorb_bias,
            # step_residual_topo,
            # step_pre_streamline,
            # step_residual_streamline,
            # step_residual_convert_to_hw,
            "step_create_dataflow_partition",
            # step_set_preferred_impl_style,
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

        cfg = build_cfg.DataflowBuildConfig(
            steps=steps,
            target_fps=None, #23
            # folding_config_file=folding_config_file,
            # folding_config_file="/home/rz/project/finn-examples/build/vgg10-radioml/folding_config/auto_folding_config.json",
            # specialize_layers_config_file = "output_%s_%s" % (model_name, release_platform_name) + "/template_specialize_layers_config.json",
            # specialize_layers_config_file = "/home/rz/project/finn-examples/build/vgg10-radioml/specialize_layers_config/template_specialize_layers_config.json",

            #large_fifo_mem_style=build_cfg.LargeFIFOMemStyle.AUTO,
            # standalone_thresholds=True,
        )

        # where is this used and why?
        cfg.use_conv_rtl = True,  # use rtl for conv layers (MVAU cannot use rtl in our model)

        return cfg