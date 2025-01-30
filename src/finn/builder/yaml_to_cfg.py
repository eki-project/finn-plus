from __future__ import annotations

import sys
import yaml
from importlib import import_module
from pathlib import Path
from typing import Any

from finn.builder import build_dataflow, build_dataflow_steps
from finn.builder.build_dataflow_config import (
    AutoFIFOSizingMethod,
    DataflowBuildConfig,
    DataflowOutputType,
    FpgaMemoryType,
    LargeFIFOMemStyle,
    ShellFlowType,
    VitisOptStrategy,
)


def variant_from_str(enum_class, variant) -> Any:
    """If a variant exists in the given enum class, return it, else None"""
    if variant in enum_class.__members__.keys():
        return enum_class.__members__[variant]
    return None


def try_insert_setting(
    setting: str, general_config: dict, cfg: DataflowBuildConfig
) -> DataflowBuildConfig:
    if setting in general_config.keys():
        if setting in cfg.__dict__.keys():
            cfg.__setattr__(setting, general_config[setting])
        else:
            print(f"Unknown setting key {setting}. Skipping")
    return cfg


def try_insert_setting_enum(
    enum_setting: tuple, general_config: dict, cfg: DataflowBuildConfig
) -> DataflowBuildConfig:
    setting_name, enum_type = enum_setting
    if enum_setting in general_config.keys():
        if enum_setting in cfg.__dict__.keys():
            cfg.__setattr__(enum_setting, variant_from_str(enum_type, general_config[enum_setting]))
        else:
            print(f"Unknown enum setting key {enum_setting}. Skipping")
    return cfg


def process_steps(
    p: Path, data: dict, cfg: DataflowBuildConfig, step_type: str
) -> DataflowBuildConfig | None:
    if step_type in data.keys():
        steps = data[step_type]
        used_steps = []
        for step in steps:
            if step in build_dataflow_steps.build_dataflow_step_lookup.keys():
                used_steps.append(step)
            else:
                # The module name is everything except the last qualifier
                mod_name = ".".join(step.split(".")[:-1])
                step_name = step.split(".")[-1]

                # Add the custom step module path to PATH so python can import it
                # Assumes that build.py and custom_steps.py are in the same dir
                sys.path.append(str(p.parent.absolute()))

                # Import the step and add it to the list
                custom_step_module = import_module(mod_name)
                if step_name in dir(custom_step_module):
                    used_steps.append(getattr(custom_step_module, step_name))
                else:
                    print(f"Step {step} is neither an integrated step nor importable via module")
                    return None
        cfg.steps = used_steps
    return cfg


def buildcfg_from_yaml(p: Path) -> DataflowBuildConfig | None:
    """Convert a YAML build file to a dataflow build config as well as possible. If the
    given YAML file does not exist or is incorrect, this returns None"""
    if not p.exists():
        return None

    # Read yaml file
    with p.open() as f:
        data = yaml.load(f, yaml.Loader)
        if "general" not in data.keys():
            print('Missing "general" section in your build file')
            return None
        general = data["general"]
        for key in ["output_dir", "synth_clk_period_ns", "generate_outputs"]:
            if key not in general.keys():
                print(f"Missing key {key} in your build file under the general section")
                return None

        generate_outputs = [
            variant_from_str(DataflowOutputType, output) for output in general["generate_outputs"]
        ]

        # Create basic cfg with all possible default arguments
        cfg = DataflowBuildConfig(
            general["output_dir"], general["synth_clk_period_ns"], generate_outputs
        )

        # Change values with defaults now
        general_settings = [
            "standalone_thresholds",
            "mvau_wwidth_max",
            "target_fps",
            "folding_two_pass_relaxation",
            "board",
            "vitis_platform",
            "specialize_layers_config_file",
            "folding_config_file",
            "auto_fifo_depths",
            "split_large_fifos",
            "rtlsim_batch_size",
            "save_intermediate_models",
            "verify_input_npy",
            "verify_expected_output_npy",
            "verify_save_full_context",
            "verify_save_rtlsim_waveforms",
            "stitched_ip_gen_dcp",
            "signature",
            "minimize_bit_widths",
            "fpga_part",
            "force_python_rtlsim",
            "large_fifo_mem_style",
            "hls_clk_period_ns",
            "default_swg_exception",
            "vitis_floorplan_file",
            "enable_hw_debug",
            "enable_build_pdb_debug",
            "verbose",
            "start_step",
            "stop_step",
            "max_multithreshold_bit_width",
            "rtlsim_use_vivado_comps",
        ]
        for setting in general_settings:
            cfg = try_insert_setting(setting, general, cfg)

        # Setting the enums
        general_enum_settings = [
            ("shell_flow_type", ShellFlowType),
            ("auto_fifo_strategy", AutoFIFOSizingMethod),
            ("large_fifo_mem_style", LargeFIFOMemStyle),
            ("vitis_opt_strategy", VitisOptStrategy),
            ("fpga_memory", FpgaMemoryType),
        ]
        for enum_setting in general_enum_settings:
            cfg = try_insert_setting_enum(enum_setting, general, cfg)

        # Build steps list
        # Existing steps are simply passed
        # If the format is module_name.step_name, module_name is automatically imported
        cfg = process_steps(p, data, cfg, "steps")
        cfg = process_steps(p, data, cfg, "verify_steps")
        return cfg
    return None


def run_finn_from_yaml(buildfile: str, modelfile: str) -> None:
    """Entrypoint for the FINN flow when using a YAML build file"""
    build_dataflow.build_dataflow_cfg(modelfile, buildcfg_from_yaml(Path(buildfile)))
