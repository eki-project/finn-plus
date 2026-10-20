"""Tests for the input validation and the solution checks of the Multi-FPGA partitioning."""

from __future__ import annotations

import pytest

import yaml
from fpgadataflow.multifpga.utils import chain_model, make_chain_partitioning_cfg
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from finn.builder.build_dataflow_config import PartitioningStrategy
from finn.transformation.fpgadataflow.multifpga.aurora.partitioner import (
    PointToPointBidirectionalFixedPartitioner,
)
from finn.transformation.fpgadataflow.multifpga.partition_model import (
    ApplyDeviceMapping,
    FindMultiFPGADeviceMapping,
)
from finn.util.basic import make_build_dir
from finn.util.exception import (
    FINNConfigurationError,
    FINNInternalError,
    FINNMultiFPGAConfigError,
    FINNMultiFPGAError,
    FINNMultiFPGAPartitionerError,
    FINNMultiFPGAUnsolvableError,
    FINNMultiFPGAUserError,
)
from finn.util.fpgadataflow import get_device_id
from finn.util.resources import PLATFORM_RESOURCE_TYPES, available_resources_on_platform

if TYPE_CHECKING:
    import mip
    from qonnx.core.modelwrapper import ModelWrapper


RESOURCE_STRATEGIES = [
    PartitioningStrategy.BALANCE_RESOURCE,
    PartitioningStrategy.BALANCE_RESOURCE_WEIGHTED,
    PartitioningStrategy.MINIMIZE_MAX_RESOURCE,
]


@pytest.mark.multifpga
@pytest.mark.usefixtures("mocked_resources")
def test_single_device_solution_is_rejected() -> None:
    """Test that a partitioning that puts the complete model on one device raises an error."""
    with pytest.raises(FINNMultiFPGAConfigError, match="single device"):
        chain_model(2).transform(
            FindMultiFPGADeviceMapping(make_chain_partitioning_cfg(num_fpgas=1))
        )


@pytest.mark.multifpga
@pytest.mark.usefixtures("mocked_resources")
def test_single_device_skips_solving() -> None:
    """Test that a single device directly yields the only possible mapping, without a MILP."""
    partitioner = PointToPointBidirectionalFixedPartitioner(
        make_chain_partitioning_cfg(num_fpgas=1), chain_model(3)
    )
    assert partitioner.solve() == {"n0": 0, "n1": 0, "n2": 0}
    assert partitioner._status is None  # noqa: SLF001


@pytest.mark.multifpga
@pytest.mark.usefixtures("mocked_resources")
@pytest.mark.parametrize(
    ("constraints", "match"),
    [({"unknown_node": 0}, "no node of that name"), ({"n0": -1}, "invalid device -1")],
)
def test_invalid_custom_partitioning_constraints(constraints: dict[str, int], match: str) -> None:
    """Test that custom constraints for unknown nodes or negative devices raise an error."""
    cfg = make_chain_partitioning_cfg(num_fpgas=2, custom_partitioning_constraints=constraints)
    with pytest.raises(FINNMultiFPGAConfigError, match=match):
        PointToPointBidirectionalFixedPartitioner(cfg, chain_model(2))


@pytest.mark.multifpga
@pytest.mark.usefixtures("mocked_resources")
@pytest.mark.parametrize("strategy", RESOURCE_STRATEGIES)
def test_max_utilization_above_one_is_rejected(strategy: PartitioningStrategy) -> None:
    """Test that the resource based strategies reject a max_utilization above 100% with a user
    error that automatic partitioning does not retry, while exactly 100% is accepted.
    """
    cfg = make_chain_partitioning_cfg(
        num_fpgas=2, partition_strategy=strategy, max_utilization=1.01
    )
    with pytest.raises(FINNMultiFPGAPartitionerError, match="Decrease max_utilization") as excinfo:
        PointToPointBidirectionalFixedPartitioner(cfg, chain_model(2))
    assert not isinstance(excinfo.value, FINNMultiFPGAUnsolvableError)
    cfg = make_chain_partitioning_cfg(num_fpgas=2, partition_strategy=strategy, max_utilization=1.0)
    PointToPointBidirectionalFixedPartitioner(cfg, chain_model(2))


@pytest.mark.multifpga
@pytest.mark.parametrize("strategy", RESOURCE_STRATEGIES)
def test_all_zero_resource_estimates_are_rejected(
    monkeypatch: pytest.MonkeyPatch, strategy: PartitioningStrategy
) -> None:
    """Test that the resource based strategies reject a node whose estimates are 0 for every
    resource type, since it could be placed anywhere without affecting the objective.
    """

    def estimate(model: ModelWrapper, *_: object) -> dict[str, dict[str, int]]:
        estimates = {
            node.name: dict.fromkeys(PLATFORM_RESOURCE_TYPES, 1) for node in model.graph.node
        }
        estimates["n1"] = dict.fromkeys(PLATFORM_RESOURCE_TYPES, 0)
        return estimates

    monkeypatch.setattr(
        "finn.transformation.fpgadataflow.multifpga.partitioner.get_estimated_model_resources",
        estimate,
    )
    with pytest.raises(FINNMultiFPGAError, match="faulty resource estimations"):
        PointToPointBidirectionalFixedPartitioner(
            make_chain_partitioning_cfg(num_fpgas=2, partition_strategy=strategy), chain_model(2)
        )
    # LAYER_COUNT does not use the estimates
    PointToPointBidirectionalFixedPartitioner(
        make_chain_partitioning_cfg(num_fpgas=2), chain_model(2)
    )


@pytest.mark.multifpga
def test_considered_resources_must_exist_on_board() -> None:
    """Test that considering a resource type the board does not have raises an error that lists
    the resource types the board has.
    """
    available = available_resources_on_platform("Pynq-Z1", ["LUT", "DSP"])
    assert available.keys() == {"LUT", "DSP"}
    assert all(amount > 0 for amount in available.values())
    with pytest.raises(FINNConfigurationError, match="URAM") as excinfo:
        available_resources_on_platform("Pynq-Z1", ["LUT", "URAM"])
    for resource in ["LUT", "FF", "BRAM_18K", "DSP"]:
        assert resource in str(excinfo.value)


@pytest.mark.multifpga
def test_non_integer_device_raises() -> None:
    """Test that a non-integer device in the solution raises an internal error instead of being
    truncated, while integral values are accepted.
    """
    partitioner = PointToPointBidirectionalFixedPartitioner.__new__(
        PointToPointBidirectionalFixedPartitioner
    )
    partitioner.modelwrapper = chain_model(2)
    # A solver keeps integer variables integral within its tolerance, so the solved model is
    # replaced by stand-ins with only the attributes _create_result reads. The casts tell the type
    # checker that they stand in for the mip types on purpose.
    partitioner.model = cast("mip.Model", SimpleNamespace(integer_tol=1e-6))
    partitioner.chosen_device = {
        "n0": cast("mip.Var", SimpleNamespace(x=1.0)),
        "n1": cast("mip.Var", SimpleNamespace(x=1.0)),
    }
    assert partitioner._create_result() == {"n0": 1, "n1": 1}  # noqa: SLF001
    partitioner.chosen_device["n1"] = cast("mip.Var", SimpleNamespace(x=1.9999))
    with pytest.raises(FINNInternalError, match="non-integer device"):
        partitioner._create_result()  # noqa: SLF001


@pytest.mark.multifpga
def test_apply_partitioning_from_file() -> None:
    """Test that a partitioning given as a dict or a YAML file is applied."""
    path = Path(make_build_dir("test_apply_partitioning_")) / "partitioning.yaml"
    path.write_text(yaml.safe_dump({"n0": 0, "n1": 1}))
    for mapping in [{"n0": 0, "n1": 1}, path]:
        model = chain_model(2).transform(ApplyDeviceMapping(mapping))
        assert [get_device_id(node) for node in model.graph.node] == [0, 1]


@pytest.mark.multifpga
@pytest.mark.parametrize(
    ("mapping", "match"),
    [
        ({"n0": 1, "n1": 1, "unknown_node": 1}, "unknown_node"),
        ({"n0": 1}, "does not assign a device to the node"),
    ],
)
def test_apply_invalid_partitioning(mapping: dict[str, int], match: str) -> None:
    """Test that an invalid partitioning raises a user error and leaves the model unchanged."""
    model = chain_model(2)
    with pytest.raises(FINNMultiFPGAUserError, match=match):
        ApplyDeviceMapping(mapping).apply(model)
    # device_id defaults to 0, so no node was assigned the device 1 of the mapping
    assert all(get_device_id(node) == 0 for node in model.graph.node)


@pytest.mark.multifpga
def test_apply_partitioning_rejects_invalid_files() -> None:
    """Test that a partitioning file that is no mapping, or that would construct Python objects,
    is rejected.
    """
    build_dir = Path(make_build_dir("test_apply_partitioning_"))
    not_a_mapping = build_dir / "list.yaml"
    not_a_mapping.write_text(yaml.safe_dump(["n0", "n1"]))
    with pytest.raises(FINNMultiFPGAUserError, match="Expected a mapping"):
        ApplyDeviceMapping(not_a_mapping)

    python_object = build_dir / "object.yaml"
    # The default dumper writes arbitrary objects with a !!python/object tag
    python_object.write_text(yaml.dump({"n0": SimpleNamespace(device=1), "n1": 1}))
    with pytest.raises(yaml.constructor.ConstructorError):
        ApplyDeviceMapping(python_object)
