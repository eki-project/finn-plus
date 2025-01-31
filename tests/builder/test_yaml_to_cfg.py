import pytest

import shutil
import yaml
from pathlib import Path

from finn.builder import build_dataflow_config
from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType
from finn.builder.yaml_to_cfg import buildcfg_from_yaml, variant_from_str

EXAMPLE_BUILD = Path(__file__).absolute().parent / "example_build.yaml"


def create_temp_build_file(prefix: str, template: str = "example_build.yaml") -> Path:
    original = Path(__file__).absolute().parent / template
    copied = original.parent / f"temp_{prefix}_{template}"
    shutil.copy(str(original), str(copied))
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
def test_default_build_steps() -> None:
    cfg, model = buildcfg_from_yaml(EXAMPLE_BUILD)
    assert len(cfg.steps) > 0


@pytest.mark.yaml_builder
def test_import_custom_step() -> None:
    copied_yaml = create_temp_build_file("import_custom_step")
    with copied_yaml.open() as f:
        data = yaml.load(f, yaml.Loader)
    assert data is not None
    data["steps"] = ["example_custom_steps.step_return_100"]
    with copied_yaml.open("w+") as f:
        yaml.dump(data, f, yaml.Dumper)
    cfg, model = buildcfg_from_yaml(copied_yaml)
    print(cfg.steps)
    assert cfg.steps[0]() == 100

    # TODO: Move this into setup, teardown methods
    copied_yaml.unlink()
