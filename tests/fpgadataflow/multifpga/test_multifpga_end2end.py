"""End2end tests for the Multi-FPGA extension. Depending on the
execution environment with driver testing.
"""
# import pytest
#
# import mip
# from fpgadataflow.multifpga.utils import get_model
# from pathlib import Path
#
# from finn.builder.build_dataflow import build_dataflow_cfg
# from finn.builder.build_dataflow_config import (
#     AutoFIFOSizingMethod,
#     DataflowBuildConfig,
#     LargeFIFOMemStyle,
#     LogLevel,
#     MFCommunicationKernel,
#     MFTopology,
#     MFVerbosity,
#     PartitioningConfiguration,
#     PartitioningStrategy,
#     ShellFlowType,
#     VitisOptStrategy,
# )
# from finn.util.basic import make_build_dir
#
#
# @pytest.mark.parametrize(
#     "fpgas,synth_clk_period_ns,target_fps,mvau_wwidth_max,folding_two_pass,max_util,communication_kernel,topology,partition_strategy",
#     [
#         pytest.param(
#             2, 10.0, 100, 412, True, 0.85, MFCommunicationKernel.AURORA, MFTopology.CHAIN,
#             PartitioningStrategy.RESOURCE_UTILIZATION, id="slow_linear_aurora_resource"
#         )
#     ]
# )
# def test_end2end_multifpga_mobilenet(
#     fpgas: int, synth_clk_period_ns: float, target_fps: int,
#     mvau_wwidth_max: int, folding_two_pass: bool,
#     max_util: float, communication_kernel: MFCommunicationKernel, topology: MFTopology,
#     partition_strategy: PartitioningStrategy
# ) -> None:
#     """Do a complete end2end test of the Multi-FPGA variant of the mobilenet."""
#     cfg = DataflowBuildConfig(
#         partitioning_configuration=PartitioningConfiguration(
#             partitioning=None,
#             num_fpgas=fpgas,
#             ports_per_device=2,
#             partition_strategy=partition_strategy,
#             topology=topology,
#             communication_kernel=communication_kernel,
#             communication_kernel_arguments={},
#             max_utilization=max_util,
#             ideal_utilization=max_util - 0.05,
#             considered_resources=["LUT", "FF", "DSP", "BRAM_18K"],
#             partition_solver_timeout=1200,
#             partition_solver=None,
#             partition_solver_emphasis=mip.SearchEmphasis.DEFAULT,
#             parallel_synthesis_workers=2,
#             separate_iodmas=True,
#             verbosity=MFVerbosity.HIGH,
#         ),
#         steps=[
#             "step_qonnx_to_finn",
#             "step_tidy_up",
#             "finn.builder.custom_step_library.mobilenet.step_mobilenet_streamline",
#             "finn.builder.custom_step_library.mobilenet.step_mobilenet_lower_convs",
#             "finn.builder.custom_step_library.mobilenet.step_mobilenet_convert_to_hw_layers",
#             "step_create_dataflow_partition",
#             "step_specialize_layers",
#             "step_target_fps_parallelization",
#             "step_apply_folding_config",
#             "step_minimize_bit_width",
#             "step_generate_estimate_reports",
#             "step_hw_codegen",
#             "step_hw_ipgen",
#             "step_set_fifo_depths",
#             "step_prepare_synthesis",
#             "step_synthesize_bitfile",
#             "step_make_driver",
#         ],
#         output_dir=make_build_dir("mn_end2end_build_"),
#         synth_clk_period_ns=synth_clk_period_ns,
#         target_fps=target_fps,
#         mvau_wwidth_max=mvau_wwidth_max,
#         folding_two_pass_relaxation=folding_two_pass,
#         standalone_thresholds=True,
#         minimize_bit_width=True,
#         board="U55C",
#         shell_flow_type=ShellFlowType.VITIS_ALVEO,
#         auto_fifo_depths=True,
#         auto_fifo_strategy=AutoFIFOSizingMethod.DISTRIBUTED_SIMULATION,
#         large_fifo_mem_style=LargeFIFOMemStyle.AUTO,
#         vitis_opt_strategy=VitisOptStrategy.PERFORMANCE_BEST,
#         vitis_iodma_intf_max_width=512,
#         save_intermediate_models=True,
#         verbose=True,
#         console_log_level=LogLevel.DEBUG,
#     )
#     model, cfg = get_model(
#         "mobilenetv1",
#         wbits=4,
#         abits=4,
#         pretrained=True,
#         until_step=None,
#         skip_fifo_sizing=False,
#         cfg=cfg,
#     )
#     modelpath = Path(make_build_dir("mn-end2end-modelstorage_")) / "mnv1.onnx"
#     model.save(str(modelpath))
#     print("Starting...")
#     build_dataflow_cfg(str(modelpath), cfg)
#     print("Done.")
