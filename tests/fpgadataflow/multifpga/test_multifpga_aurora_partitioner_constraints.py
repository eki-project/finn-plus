"""Tests for the effect of the link and topology constraints of the Aurora Multi-FPGA partitioner
on its solution.
"""

import pytest

from fpgadataflow.multifpga.utils import chain_model, make_chain_partitioning_cfg

from finn.builder.build_dataflow_config import MFTopology
from finn.transformation.fpgadataflow.multifpga.aurora.partitioner import (
    PointToPointBidirectionalFixedPartitioner,
)
from finn.util.exception import FINNMultiFPGAConfigError, FINNMultiFPGAUnsolvableError


@pytest.mark.multifpga
@pytest.mark.usefixtures("mocked_resources")
def test_port_limit_is_enforced_simplex() -> None:
    """Test that the number of links per device constrains the partitioning. In a chain across
    three devices the middle device sends to and receives from different devices, so it needs
    two links.
    """
    model = chain_model(3)
    cfg = make_chain_partitioning_cfg(num_fpgas=3, links_per_device=1)
    with pytest.raises(FINNMultiFPGAConfigError, match="requires 2 links per device"):
        PointToPointBidirectionalFixedPartitioner(cfg, model)
    cfg = make_chain_partitioning_cfg(num_fpgas=3, links_per_device=2)
    assert PointToPointBidirectionalFixedPartitioner(cfg, model).solve() == {
        "n0": 0,
        "n1": 1,
        "n2": 2,
    }


@pytest.mark.multifpga
@pytest.mark.usefixtures("mocked_resources")
def test_port_limit_is_enforced_duplex() -> None:
    """Test that the number of links per device constrains the partitioning of a duplex chain.
    The data returns to the first device, so a chain across three devices passes the middle device
    twice, which then needs a link to each of its two neighbors.
    """
    model = chain_model(5)
    cfg = make_chain_partitioning_cfg(
        num_fpgas=3, links_per_device=1, topology=MFTopology.DUPLEX_CHAIN
    )
    with pytest.raises(FINNMultiFPGAConfigError, match="requires 2 links per device"):
        PointToPointBidirectionalFixedPartitioner(cfg, model)
    cfg = make_chain_partitioning_cfg(
        num_fpgas=3, links_per_device=2, topology=MFTopology.DUPLEX_CHAIN
    )
    assert PointToPointBidirectionalFixedPartitioner(cfg, model).solve() == {
        "n0": 0,
        "n1": 1,
        "n2": 2,
        "n3": 1,
        "n4": 0,
    }


@pytest.mark.multifpga
@pytest.mark.usefixtures("mocked_resources")
def test_duplex_chain_rejects_revisiting_devices() -> None:
    """Test that a duplex chain passes every device at most once on the way out and once on the
    way back. The partitioning 0 -> 1 -> 2 -> 1 -> 2 -> 1 -> 0 is forced with custom constraints and
    enough links for it (device 1 sends three times). A duplex chain should not allow
    this configuration.
    """
    bounce = {"n0": 0, "n1": 1, "n2": 2, "n3": 1, "n4": 2, "n5": 1, "n6": 0}
    cfg = make_chain_partitioning_cfg(
        num_fpgas=3,
        links_per_device=3,
        custom_partitioning_constraints=bounce,
        topology=MFTopology.DUPLEX_CHAIN,
    )
    with pytest.raises(FINNMultiFPGAUnsolvableError):
        PointToPointBidirectionalFixedPartitioner(cfg, chain_model(7)).solve()


@pytest.mark.multifpga
@pytest.mark.usefixtures("mocked_resources")
def test_ring() -> None:
    """Test that a ring passes the devices in the order of their IDs and closes with the edge
    from device N-1 back to device 0, which is the only edge allowed to jump back.
    """
    raise NotImplementedError


@pytest.mark.multifpga
@pytest.mark.usefixtures("mocked_resources")
def test_arbitrary_open() -> None:
    """Test that ARBITRARY_OPEN places the input nodes on device 0 and the output nodes on device
    N-1, and does not constrain the devices of the nodes in between (e.g. a fork may be split
    across devices that are not neighbours).
    """
    raise NotImplementedError


@pytest.mark.multifpga
@pytest.mark.usefixtures("mocked_resources")
def test_arbitrary_closed() -> None:
    """Test that ARBITRARY_CLOSED places the input and output nodes on device 0, and does not
    constrain the devices of the nodes in between (e.g. a fork may be split across devices that
    are not neighbours).
    """
    raise NotImplementedError
