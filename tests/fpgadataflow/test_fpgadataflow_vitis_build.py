import pytest
from finn.transformation.fpgadataflow.vitis_build import *

@pytest.fixture
def construct_modelwrapper() -> ModelWrapper:
    return ModelWrapper("abc.onnx")


def test_model_to_config():
    pass