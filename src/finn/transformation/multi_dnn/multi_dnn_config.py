"""Configuration loader for multi-DNN build flows."""

import json
import os
from copy import deepcopy
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from typing import ClassVar

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.util.exception import FINNUserError


class MultiDNNConfig:
    """Parses and provides access to a multi-DNN JSON configuration file."""

    virtual_keywords: ClassVar[set[str]] = {"output_dir"}

    def __init__(self, multi_dnn_config_path: str | Path) -> None:
        """Load and parse the multi-DNN config JSON from the given path."""
        with Path(multi_dnn_config_path).open() as fp_json:
            self.multi_dnn_config = json.load(fp_json)
            self.submodel_names = list(self.multi_dnn_config["Submodels"].keys())

    def get_submodel_model(self, model_name: str) -> ModelWrapper:
        """Return the ModelWrapper for the named submodel."""
        return ModelWrapper(
            self.multi_dnn_config["Submodels"][model_name].get("model_path", None), True
        )

    def get_steps(self) -> list[tuple[str, list[str]]] | None:
        """Return the list of (step_name, target_names) tuples from the config."""
        steps = self.multi_dnn_config.get("Steps", None)
        if steps is None:
            return None
        tuple_list = []
        for step_dict in steps:
            if len(step_dict) != 1:
                raise FINNUserError(
                    f"Each step dict must have exactly one entry, found {len(step_dict)} "
                    f"entries in {step_dict}"
                )
            key, value = next(iter(step_dict.items()))
            if isinstance(value, str):
                value = [value]
            tuple_list.append((key, value))
        return tuple_list

    def generate_virtual_configs(
        self, model_names: str | list[str], cfg: DataflowBuildConfig
    ) -> dict[str, DataflowBuildConfig]:
        """Create per-submodel DataflowBuildConfig copies with adjusted output paths."""
        output_configs = {}
        if "Multi_DNN_Wrapper" in model_names or "Collapsed_Model" in model_names:
            output_configs = {
                model_names[0]: cfg
            }  # We can assume that the list only has 1 element in this case
            return output_configs

        for model_name in model_names:
            copied_cfg = deepcopy(cfg)
            for k in self.virtual_keywords:
                value = getattr(cfg, k)
                if value is None:
                    continue
                normalized_path = Path(os.path.normpath(value))
                new_path = str(normalized_path.parent.parent / model_name / normalized_path.name)
                setattr(copied_cfg, k, new_path)

            for k, v in self.multi_dnn_config["Submodels"][model_name].items():
                if hasattr(cfg, k):
                    setattr(copied_cfg, k, v)
            output_configs[model_name] = copied_cfg
        return output_configs
