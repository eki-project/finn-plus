import json
import os
from copy import deepcopy
from qonnx.core.modelwrapper import ModelWrapper

from finn.builder.build_dataflow_config import DataflowBuildConfig


class MultiDNNConfig:
    virtual_keywords = {"output_dir"}

    def __init__(self, multi_dnn_config_path):
        with open(multi_dnn_config_path, "r") as fp_json:
            self.multi_dnn_config = json.load(fp_json)
            self.submodel_names = list(self.multi_dnn_config["Submodels"].keys())

    def get_submodel_model(self, model_name):
        return ModelWrapper(
            self.multi_dnn_config["Submodels"][model_name].get("model_path", None), True
        )

    def get_steps(self):
        steps = self.multi_dnn_config.get("Steps", None)
        if steps is None:
            return None
        tuple_list = []
        for step_dict in steps:
            if len(step_dict) != 1:
                raise ValueError(
                    f"Each step dict must have exactly one entry, found {len(step_dict)} ",
                    f"entries in {step_dict}",
                )
            key, value = next(iter(step_dict.items()))
            if isinstance(value, str):
                value = [value]
            tuple_list.append((key, value))
        return tuple_list

    def generate_virtual_configs(
        self, model_names: str | list[str], cfg: DataflowBuildConfig
    ) -> dict[str, DataflowBuildConfig]:
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
                normalized_path = os.path.normpath(value)
                directory, original_file = os.path.split(normalized_path)
                new_path = os.path.join(os.path.dirname(directory), model_name, original_file)
                setattr(copied_cfg, k, new_path)

            for k, v in self.multi_dnn_config["Submodels"][model_name].items():
                if hasattr(cfg, k):
                    setattr(copied_cfg, k, v)
            output_configs[model_name] = copied_cfg
        return output_configs
