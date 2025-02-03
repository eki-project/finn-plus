from __future__ import annotations

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


def modify_config(p: Path, key_path: list[str | int], value: Any) -> None:
    """Modify the given file with the given keypath and store it back"""
    d = None
    with p.open() as f:
        d = yaml.load(f, yaml.Loader)
    set_key(d, key_path, value)
    with p.open("w+") as f:
        yaml.dump(d, f, yaml.Dumper)


@pytest.fixture
def temporary_buildfile():  # noqa
    """Fixture to provide a temporary yaml build file that gets cleaned up after the
    test is done running"""
    copied = create_temp_build_file("temp_")
    yield copied
    copied.unlink()


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
def test_yaml_enum_set_correctly(setting_value: tuple[str, str], temporary_buildfile: Path) -> None:
    setting, value = setting_value
    copied = temporary_buildfile
    modify_config(p=copied, key_path=["general", setting], value=value)
    cfg, model = buildcfg_from_yaml(copied)
    assert cfg is not None
    assert cfg.__getattribute__(setting).name == value


@pytest.mark.yaml_builder
def test_default_build_steps() -> None:
    cfg, model = buildcfg_from_yaml(EXAMPLE_BUILD)
    assert len(cfg.steps) > 0


@pytest.mark.yaml_builder
def test_import_custom_step(temporary_buildfile: Path) -> None:
    copied = temporary_buildfile
    modify_config(copied, ["steps"], ["example_custom_steps.step_return_100"])
    cfg, model = buildcfg_from_yaml(copied)
    print(cfg.steps)
    assert cfg.steps[0]() == 100


@pytest.mark.yaml_builder
def test_fail_on_typo_in_stepname(temporary_buildfile: Path) -> None:
    copied = temporary_buildfile
    modify_config(copied, ["steps"], ["step_hw_igen"])
    cfg, model = buildcfg_from_yaml(copied)
    assert cfg is None
