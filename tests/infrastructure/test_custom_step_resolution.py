import pytest

from finn.builder import build_dataflow
from finn.builder.build_dataflow_config import DataflowBuildConfig


@pytest.mark.infrastructure
@pytest.mark.parametrize("custom_steps", [[("custom_steps.step_from_external_module", 1)]])
def test_custom_step_resolution(custom_steps: list[str]) -> None:
    step_names = [s[0] for s in custom_steps]
    steps = ["step_qonnx_to_finn", *step_names]
    cfg = DataflowBuildConfig(
        output_dir="", synth_clk_period_ns=1.0, generate_outputs=[], steps=steps
    )
    fxn_steps = build_dataflow.resolve_build_steps(cfg)
    assert callable(fxn_steps[0])
    for i, (_, expected_result) in enumerate(custom_steps):
        assert fxn_steps[1 + i]() == expected_result


@pytest.mark.infrastructure
def test_fail_custom_step_resolution() -> None:
    # A name but not callable
    non_callable_step = 0  # noqa

    steps = ["non_callable_step", "non_existing_step"]
    for step in steps:
        cfg = DataflowBuildConfig("", 1.0, [], steps=[step])

        # TODO: When FINNException is implemented check for specific error
        with pytest.raises(Exception):  # noqa
            _ = build_dataflow.resolve_build_steps(cfg)
