import pytest

import shutil
import yaml
from pathlib import Path
from typing import Any

from finn.builder import build_dataflow_config
from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType
from finn.builder.yaml_to_cfg import buildcfg_from_yaml, variant_from_str

EXAMPLE_BUILD = Path(__file__).absolute().parent / "example_build.yaml"


def set_key(d: dict, key_path: list[str | int], value: Any) -> None:
    """Set a nested key in the given dictionary"""
    if len(key_path) > 1:
        set_key(d[key_path[0]], key_path[1:], value)
    elif len(key_path) == 1:
        d[key_path[0]] = value


def create_temp_build_file(prefix: str, template: str = "example_build.yaml") -> Path:
    """Create a temporary copy of a template build file"""
    original = Path(__file__).absolute().parent / template
    copied = original.parent / f"temp_{prefix}_{template}"
    shutil.copy(str(original), str(copied))
    return copied


def create_modified_build_file(
    prefix: str, key_path: list[str | int], value: Any, template: str = "example_build.yaml"
) -> Path:
    """Create a temporary copy of a template build file, modifying a given key and
    writing it back"""
    copied = create_temp_build_file(prefix, template)
    with copied.open() as f:
        data = yaml.load(f, yaml.Loader)
    assert data is not None
    set_key(data, key_path, value)
    with copied.open("w+") as f:
        yaml.dump(data, f, yaml.Dumper, indent=2)
    return copied


@pytest.mark.yaml_builder
def test_valid_example_yaml_build() -> None:
    cfg, model = buildcfg_from_yaml(EXAMPLE_BUILD)
    assert cfg is not None


@pytest.mark.yaml_builder
def test_default_cfg_has_empty_build_steps() -> None:
    cfg = DataflowBuildConfig(".", 4, [])
    assert cfg.steps is None


@pytest.mark.yaml_builder
def test_default_yaml_built_cfg_has_default_build_steps() -> None:
    cfg, model = buildcfg_from_yaml(EXAMPLE_BUILD)
    assert all(
        cfg.steps[i] == build_dataflow_config.default_build_dataflow_steps[i]
        for i in range(len(cfg.steps))
    )


@pytest.mark.yaml_builder
@pytest.mark.parametrize(
    "variant_and_expect_exist",
    [("STITCHED_IP", True), ("ESTIMATE_REPORTS", True), ("ABCD", False), ("", False)],
)
def test_yaml_to_enum(variant_and_expect_exist: tuple[str, bool]) -> None:
    assert (
        variant_from_str(DataflowOutputType, variant_and_expect_exist[0]) is not None
        or variant_and_expect_exist[1] is False
    )


@pytest.mark.yaml_builder
@pytest.mark.parametrize(
    "setting_value",
    [("shell_flow_type", "VITIS_ALVEO"), ("auto_fifo_strategy", "LARGEFIFO_RTLSIM")],
)
def test_yaml_enum_set_correctly(setting_value: tuple[str, str]) -> None:
    setting, value = setting_value
    copied = create_modified_build_file(
        prefix=setting + "_enum_set_correctly", key_path=["general", setting], value=value
    )
    cfg, model = buildcfg_from_yaml(copied)
    assert cfg is not None
    assert cfg.__getattribute__(setting).name == value
    # TODO: Move to teardown
    copied.unlink()


@pytest.mark.yaml_builder
def test_default_build_steps() -> None:
    cfg, model = buildcfg_from_yaml(EXAMPLE_BUILD)
    assert len(cfg.steps) > 0


@pytest.mark.yaml_builder
def test_import_custom_step() -> None:
    copied_yaml = create_modified_build_file(
        prefix="import_custom_step",
        key_path=["steps"],
        value=["example_custom_steps.step_return_100"],
    )
    cfg, model = buildcfg_from_yaml(copied_yaml)
    print(cfg.steps)
    assert cfg.steps[0]() == 100

    # TODO: Move this into setup, teardown methods
    copied_yaml.unlink()
