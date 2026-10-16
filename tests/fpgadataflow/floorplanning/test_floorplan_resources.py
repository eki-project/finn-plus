"""Tests for the floorplanner's resource summing and per-SLR budgets."""

import pytest

from fpgadataflow.floorplanning.util import U55C_PART, make_chain

from finn.transformation.fpgadataflow.floorplanning.resources import (
    DEFAULT_FLOORPLAN_RESOURCES,
    PLATFORM_RESOURCE_ORDER,
    get_node_resource_estimates,
    platform_slr_capacity,
    sum_group_resources,
)
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.platforms import platforms


@pytest.mark.fpgadataflow
def test_node_estimates_cover_every_hw_node() -> None:
    """Every specialized node must get an estimate for exactly the requested types."""
    model = make_chain(3)
    estimates = get_node_resource_estimates(model, U55C_PART)
    assert set(estimates) == {node.name for node in model.graph.node}
    for res in estimates.values():
        assert set(res) == set(DEFAULT_FLOORPLAN_RESOURCES)


@pytest.mark.fpgadataflow
def test_group_sums_add_up_and_apply_margin() -> None:
    """A unit's demand is the sum of its members, rounded up after the margin."""
    node_resources = {
        "a": {"LUT": 100, "DSP": 1},
        "b": {"LUT": 55, "DSP": 0},
        "c": {"LUT": 10, "DSP": 3},
    }
    units = {0: ["a", "b"], 1: ["c"]}
    summed = sum_group_resources(node_resources, units, ("LUT", "DSP"), margin=1.0)
    assert summed == {0: {"LUT": 155, "DSP": 1}, 1: {"LUT": 10, "DSP": 3}}
    with_margin = sum_group_resources(node_resources, units, ("LUT", "DSP"), margin=1.1)
    # 155 * 1.1 = 170.5, rounded up; 1 * 1.1 = 1.1, rounded up.
    assert with_margin[0] == {"LUT": 171, "DSP": 2}


@pytest.mark.fpgadataflow
def test_margin_below_one_is_rejected() -> None:
    """A margin under 1.0 would size a unit below its own estimate."""
    with pytest.raises(FINNUserError, match="margin"):
        sum_group_resources({"a": {"LUT": 1}}, {0: ["a"]}, ("LUT",), margin=0.9)


@pytest.mark.fpgadataflow
def test_group_with_unknown_node_is_an_internal_error() -> None:
    """Grouping and estimation disagreeing about the node set is a bug, not user error."""
    with pytest.raises(FINNInternalError, match="resource estimate"):
        sum_group_resources({"a": {"LUT": 1}}, {0: ["a", "ghost"]}, ("LUT",))


@pytest.mark.fpgadataflow
def test_slr_capacity_applies_the_platform_limits() -> None:
    """The usable budget is guide_resources scaled by the platform's utilization limits."""
    platform = platforms["U55C"]()
    capacity = platform_slr_capacity(
        platform.guide_resources, platform.res_limits, DEFAULT_FLOORPLAN_RESOURCES
    )
    assert set(capacity) == set(range(platform.nslr))
    for slr, budget in capacity.items():
        for restype in DEFAULT_FLOORPLAN_RESOURCES:
            idx = PLATFORM_RESOURCE_ORDER.index(restype)
            raw = platform.guide_resources[slr][idx]
            assert budget[restype] == max(int(raw * platform.res_limits[idx]), 0)
            assert budget[restype] <= raw


@pytest.mark.fpgadataflow
def test_unknown_resource_type_is_rejected() -> None:
    """Asking for a resource the platform model does not have is a bug."""
    platform = platforms["U55C"]()
    with pytest.raises(FINNInternalError, match="not modeled"):
        platform_slr_capacity(platform.guide_resources, platform.res_limits, ("CLB",))


@pytest.mark.fpgadataflow
def test_confirmed_pcie_slr_is_used_verbatim() -> None:
    """U55C and U280 have a confirmed PCIe SLR, which must be returned as recorded."""
    for board in ("U55C", "U280"):
        platform = platforms[board]()
        assert platform.pcie_slr == 0
        assert platform.get_pcie_slr() == 0


@pytest.mark.fpgadataflow
def test_unconfirmed_pcie_slr_falls_back_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """A multi-SLR board with no confirmed value falls back to SLR 0 and says so."""
    platform = platforms["U250"]()
    assert platform.pcie_slr is None
    with caplog.at_level("WARNING"):
        assert platform.get_pcie_slr() == 0
    assert "pcie_slr" in caplog.text


@pytest.mark.fpgadataflow
def test_single_slr_board_does_not_warn_about_pcie_slr(caplog: pytest.LogCaptureFixture) -> None:
    """On a single-SLR part there is nothing to choose, so the warning would be noise."""
    platform = platforms["Pynq-Z1"]()
    with caplog.at_level("WARNING"):
        assert platform.get_pcie_slr() == 0
    assert "pcie_slr" not in caplog.text


@pytest.mark.fpgadataflow
def test_hls_reports_degrade_to_the_analytical_estimate_when_absent() -> None:
    """Asking for HLS reports before any exist must not raise, nor zero anything.

    `floorplan_use_hls_reports` defaults to True because SLR assignment runs after
    HLS synthesis in every real flow. The flip is only safe if a node without a
    report -- every RTL layer always, and an HLS layer before synthesis -- keeps its
    analytical estimate rather than failing or reporting zero.
    """
    model = make_chain(3)
    analytical = get_node_resource_estimates(model, U55C_PART, use_hls_reports=False)
    with_reports = get_node_resource_estimates(model, U55C_PART, use_hls_reports=True)
    assert set(with_reports) == set(analytical)
    for name, res in analytical.items():
        for restype, value in res.items():
            # The merge takes the larger of the two, so nothing may come out below
            # the analytical figure -- and with no report present, nothing above it.
            assert with_reports[name][restype] == value, (name, restype)
