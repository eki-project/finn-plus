"""Tests for the Multi-FPGA part of the build configuration."""
import pytest

import json
import yaml
from pathlib import Path

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.util.basic import make_build_dir
from finn.util.exception import FINNConfigurationError


@pytest.mark.multifpga
@pytest.mark.parametrize("file_format", ["yaml", "json"])
def test_renamed_multifpga_configuration_field(file_format: str) -> None:
    """Test that configs with the old field name fail with a hint to the new one."""
    output_dir = Path(make_build_dir("test_renamed_multifpga_configuration_field_"))
    old_config = {"target_fps": 1000, "partitioning_configuration": {"num_fpgas": 2}}
    new_config = {"target_fps": 1000, "multifpga_configuration": {"num_fpgas": 2}}

    def _write(data: dict, name: str) -> Path:
        p = output_dir / f"{name}.{file_format}"
        match file_format:
            case "yaml":
                p.write_text(yaml.dump(data))
            case "json":
                p.write_text(json.dumps(data))
            case _:
                raise NotImplementedError(f"Unknown config file format: {file_format}")
        return p

    with pytest.raises(FINNConfigurationError, match="renamed to 'multifpga_configuration'"):
        DataflowBuildConfig.construct_from(_write(old_config, "old"))

    cfg = DataflowBuildConfig.construct_from(_write(new_config, "new"))
    assert cfg.multifpga_configuration is not None
    assert cfg.multifpga_configuration.num_fpgas == 2
