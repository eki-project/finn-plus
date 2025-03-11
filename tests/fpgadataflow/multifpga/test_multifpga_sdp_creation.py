import pytest

from qonnx.core.modelwrapper import ModelWrapper


def create_simple_model(device_list: list[int]) -> ModelWrapper:
    pass


@pytest.mark.multifpga
def test_sdp_creation(device_list: list[int]) -> None:
    pass


@pytest.mark.multifpga
def test_fail_on_split_branch_nodes() -> None:
    pass
