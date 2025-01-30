from __future__ import annotations

import yaml
from enum import EnumType
from pathlib import Path
from typing import Any

from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType


def variant_from_str(enum_class: EnumType, variant: str) -> Any:
    """If a variant exists in the given enum class, return it, else None"""
    if variant in enum_class.__members__.keys():
        return enum_class.__members__[variant]
    return None


def buildcfg_from_yaml(p: Path) -> DataflowBuildConfig | None:
    """Convert a YAML build file to a dataflow build config as well as possible. If the
    given YAML file does not exist or is incorrect, this returns None"""
    if not p.exists():
        return None

    with p.open() as f:
        data = yaml.load(f)
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

        cfg = DataflowBuildConfig(
            general["output_dir"], float(general["synth_clk_period_ns"], generate_outputs)
        )

        return cfg
