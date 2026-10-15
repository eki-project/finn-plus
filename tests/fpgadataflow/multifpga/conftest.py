"""Fixtures for the Multi-FPGA tests."""

import pytest

from qonnx.core.modelwrapper import ModelWrapper

from finn.util.resources import PLATFORM_RESOURCE_TYPES


@pytest.fixture
def mocked_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the resource estimation, which needs HLS synthesized nodes, by small estimates.
    Every node is estimated to use 1 of every resource type.
    """

    def estimate(model: ModelWrapper, *_: object) -> dict[str, dict[str, int]]:
        return {node.name: dict.fromkeys(PLATFORM_RESOURCE_TYPES, 1) for node in model.graph.node}

    monkeypatch.setattr(
        "finn.transformation.fpgadataflow.multifpga.partitioner.get_estimated_model_resources",
        estimate,
    )
