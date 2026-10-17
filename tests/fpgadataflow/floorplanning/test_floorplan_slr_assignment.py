"""Tests for Level-1 SLR assignment and the anchor constraints feeding it."""

import pytest

import numpy as np
from fpgadataflow.floorplanning.util import (
    U55C_PART,
    ZYNQ_PART,
    attr,
    make_chain,
    make_iodma,
    make_model,
    make_mvau,
    set_device_ids,
)
from numpy.typing import NDArray
from qonnx.core.modelwrapper import ModelWrapper

from finn.builder.build_dataflow_config import FloorplanGranularity, FpgaMemoryType
from finn.transformation.fpgadataflow.floorplanning.anchors import (
    derive_slr_anchors,
    lift_anchors_to_units,
    node_has_external_memory_interface,
    resolve_mem_port,
    resolve_memory_anchor,
)
from finn.transformation.fpgadataflow.floorplanning.grouping import (
    ComputeFloorplanGroups,
    get_placement_units,
)
from finn.transformation.fpgadataflow.floorplanning.resources import (
    DEFAULT_FLOORPLAN_RESOURCES,
    get_node_resource_estimates,
    sum_group_resources,
)
from finn.transformation.fpgadataflow.floorplanning.slr_assignment import (
    AssignFloorplanSLRs,
    SLRAssignmentProblem,
    collect_unit_edges,
)
from finn.util.exception import FINNUserError
from finn.util.platforms import Platform, platforms

NO_LIMITS: NDArray[np.float64] = np.ones(5)


class TinyPlatform(Platform):
    """Multi-SLR platform with a per-SLR LUT budget the test dictates exactly.

    Real platforms are far too large for a four-node test model to ever need more than
    one SLR, so tests that want to observe a split need a device sized to the model
    rather than the other way round.
    """

    def __init__(self, nslr: int = 2, lut_per_slr: int = 10_000, sll: int = 100_000) -> None:
        """Build a platform with `nslr` identical SLRs and no memory controllers."""
        self._lut_per_slr = lut_per_slr
        sll_count = [[0 if i == j else sll for j in range(nslr)] for i in range(nslr)]
        super().__init__(
            nslr=nslr,
            sll_count=sll_count,
            ddr_slr=[],
            hbm_slr=-1,
            limits=NO_LIMITS,
            avg_constraints=[],
        )

    @property
    def compute_resources(self) -> list[list[int]]:
        """Return the per-SLR budget, with plenty of everything except LUTs."""
        return [[self._lut_per_slr, 2 * self._lut_per_slr, 10_000, 10_000, 10_000]] * self.nslr


def _total_lut_demand(model: ModelWrapper) -> int:
    """Return the margin-applied LUT demand of the model's placement units."""
    units = get_placement_units(model)
    node_resources = get_node_resource_estimates(model, U55C_PART, DEFAULT_FLOORPLAN_RESOURCES)
    summed = sum_group_resources(node_resources, units, DEFAULT_FLOORPLAN_RESOURCES)
    return sum(demand["LUT"] for demand in summed.values())


def _slrs(model: ModelWrapper) -> dict[str, int]:
    """Return the assigned SLR of every node, keyed by node name."""
    return {node.name: attr(model, node.name, "floorplan_slr") for node in model.graph.node}


# --------------------------------------------------------------------------- LP core


@pytest.mark.fpgadataflow
def test_lp_packs_a_chain_contiguously() -> None:
    """A chain that needs every SLR must be cut as few times as possible."""
    capacity = {slr: {"LUT": 100} for slr in range(3)}
    units = {unit: {"LUT": 50} for unit in range(6)}
    edges = [(unit, unit + 1, 64) for unit in range(5)]
    problem = SLRAssignmentProblem(3, units, capacity, edges, [10_000, 10_000])
    solution = problem.solve(30)
    assert solution is not None
    # Two SLRs per unit-pair, so exactly two of the five edges may cross.
    assert problem.crossing_cost() == pytest.approx(2 * 64)
    assert sorted(solution.values()) == [0, 0, 1, 1, 2, 2]


@pytest.mark.fpgadataflow
def test_lp_respects_anchors() -> None:
    """An anchored unit must land in one of its allowed SLRs."""
    capacity = {slr: {"LUT": 100} for slr in range(3)}
    units = {unit: {"LUT": 50} for unit in range(6)}
    edges = [(unit, unit + 1, 64) for unit in range(5)]
    problem = SLRAssignmentProblem(
        3, units, capacity, edges, [10_000, 10_000], unit_anchors={0: [2], 5: [0]}
    )
    solution = problem.solve(30)
    assert solution is not None
    assert solution[0] == 2
    assert solution[5] == 0


@pytest.mark.fpgadataflow
def test_lp_keeps_wide_streams_local() -> None:
    """Given a choice, the cut goes through the narrow edges, not the wide one."""
    capacity = {slr: {"LUT": 100} for slr in range(2)}
    units = {unit: {"LUT": 50} for unit in range(4)}
    edges = [(0, 1, 10), (1, 2, 5000), (2, 3, 10)]
    problem = SLRAssignmentProblem(2, units, capacity, edges, [10_000])
    solution = problem.solve(30)
    assert solution is not None
    assert solution[1] == solution[2]
    assert problem.crossing_cost() == pytest.approx(20)


@pytest.mark.fpgadataflow
def test_lp_respects_the_sll_budget() -> None:
    """A boundary budget smaller than the cheapest cut makes the problem infeasible."""
    # Two SLRs hold two units each, so the three units must be split 2/1 and exactly
    # one 500-SLL edge gets cut -- more than the 100 SLLs the boundary offers.
    capacity = {slr: {"LUT": 130} for slr in range(2)}
    units = {unit: {"LUT": 60} for unit in range(3)}
    edges = [(0, 1, 500), (1, 2, 500)]
    assert SLRAssignmentProblem(2, units, capacity, edges, [100]).solve(30) is None
    # The same problem is solvable once the budget stops being enforced.
    relaxed = SLRAssignmentProblem(2, units, capacity, edges, [100], enforce_sll_budget=False)
    assert relaxed.solve(30) is not None


@pytest.mark.fpgadataflow
def test_lp_reports_infeasibility_for_an_oversized_design() -> None:
    """A design that does not fit the device has no solution, and says how much it needs."""
    capacity = {slr: {"LUT": 100} for slr in range(3)}
    units = {unit: {"LUT": 200} for unit in range(6)}
    problem = SLRAssignmentProblem(3, units, capacity, [], [10_000, 10_000])
    assert problem.solve(30) is None
    assert "1200 required vs. 300 usable" in problem.describe_demand()


@pytest.mark.fpgadataflow
def test_lp_rejects_an_anchor_outside_the_device() -> None:
    """Anchoring to an SLR the part does not have is a configuration error."""
    capacity = {0: {"LUT": 10}, 1: {"LUT": 10}}
    with pytest.raises(FINNUserError, match="only has 2 SLR"):
        SLRAssignmentProblem(2, {0: {"LUT": 1}}, capacity, [], [10], unit_anchors={0: [5]})


# ------------------------------------------------------------------------ transformation


@pytest.mark.fpgadataflow
def test_single_slr_part_assigns_slr_zero() -> None:
    """On a single-SLR part every node goes to SLR 0 without building an LP."""
    model = make_chain(3, part=ZYNQ_PART)
    model = model.transform(ComputeFloorplanGroups())
    model = model.transform(AssignFloorplanSLRs(platforms["Pynq-Z1"](), ZYNQ_PART, "Pynq-Z1"))
    assert set(_slrs(model).values()) == {0}


@pytest.mark.fpgadataflow
def test_model_that_fits_stays_on_one_slr() -> None:
    """With no reason to split, the whole model lands in a single SLR."""
    model = make_chain(4)
    model = model.transform(ComputeFloorplanGroups())
    model = model.transform(AssignFloorplanSLRs(platforms["U55C"](), U55C_PART, "U55C"))
    assert len(set(_slrs(model).values())) == 1


@pytest.mark.fpgadataflow
def test_model_that_does_not_fit_one_slr_is_split() -> None:
    """A per-SLR budget below the total demand forces a split that still fits."""
    model = make_chain(4)
    model = model.transform(ComputeFloorplanGroups(FloorplanGranularity.PER_NODE))
    demand = _total_lut_demand(model)
    # Room for most, but not all, of the design per SLR.
    platform = TinyPlatform(nslr=2, lut_per_slr=demand - demand // 4)
    model = model.transform(AssignFloorplanSLRs(platform, U55C_PART))
    assignment = _slrs(model)
    assert len(set(assignment.values())) == 2
    node_resources = get_node_resource_estimates(model, U55C_PART, DEFAULT_FLOORPLAN_RESOURCES)
    for slr in set(assignment.values()):
        used = sum(
            node_resources[name]["LUT"] for name, value in assignment.items() if value == slr
        )
        assert used <= platform.compute_resources[slr][0]


@pytest.mark.fpgadataflow
def test_nodes_of_one_unit_share_an_slr() -> None:
    """A placement unit is atomic: all its nodes get the same SLR."""
    model = make_chain(4)
    model = model.transform(ComputeFloorplanGroups(FloorplanGranularity.NODE_WITH_ADJACENT_GLUE))
    platform = TinyPlatform(nslr=3, lut_per_slr=_total_lut_demand(model) // 2)
    model = model.transform(AssignFloorplanSLRs(platform, U55C_PART))
    per_unit: dict[int, set[int]] = {}
    for node in model.graph.node:
        unit = attr(model, node.name, "floorplan_group_id")
        per_unit.setdefault(unit, set()).add(attr(model, node.name, "floorplan_slr"))
    assert all(len(slrs) == 1 for slrs in per_unit.values())


@pytest.mark.fpgadataflow
def test_assignment_does_not_touch_device_id_or_legacy_slr() -> None:
    """SLR assignment is a different axis from device partitioning, and from `slr`."""
    model = make_chain(4)
    set_device_ids(model, {"MVAU_hls_2": 1, "MVAU_hls_3": 1})
    model = model.transform(ComputeFloorplanGroups())
    model = model.transform(AssignFloorplanSLRs(platforms["U55C"](), U55C_PART, "U55C"))
    assert attr(model, "MVAU_hls_2", "device_id") == 1
    assert attr(model, "MVAU_hls_0", "device_id") == 0
    # The legacy attr stays at its unassigned default: bridging is a separate step.
    assert all(attr(model, node.name, "slr") == -1 for node in model.graph.node)
    assert all(attr(model, node.name, "floorplan_slr") >= 0 for node in model.graph.node)


@pytest.mark.fpgadataflow
def test_each_device_is_solved_independently() -> None:
    """Two devices each get their own assignment, both starting from SLR 0."""
    model = make_chain(4)
    set_device_ids(model, {"MVAU_hls_2": 1, "MVAU_hls_3": 1})
    model = model.transform(ComputeFloorplanGroups(FloorplanGranularity.PER_NODE))
    demand = _total_lut_demand(model)
    # Sized so that the two units on each device cannot share one SLR.
    platform = TinyPlatform(nslr=2, lut_per_slr=demand // 3)
    model = model.transform(AssignFloorplanSLRs(platform, U55C_PART))
    assignment = _slrs(model)
    assert sorted(assignment[name] for name in ("MVAU_hls_0", "MVAU_hls_1")) == [0, 1]
    assert sorted(assignment[name] for name in ("MVAU_hls_2", "MVAU_hls_3")) == [0, 1]


@pytest.mark.fpgadataflow
def test_infeasible_assignment_raises_a_user_error() -> None:
    """A design that cannot be placed fails loudly, with the demand in the message."""
    model = make_chain(4)
    model = model.transform(ComputeFloorplanGroups(FloorplanGranularity.PER_NODE))
    platform = TinyPlatform(nslr=2, lut_per_slr=1)
    with pytest.raises(FINNUserError, match="infeasible"):
        model.transform(AssignFloorplanSLRs(platform, U55C_PART))


@pytest.mark.fpgadataflow
def test_explicit_anchors_override_the_derived_ones() -> None:
    """The debugging override pins a unit even when nothing would anchor it."""
    model = make_chain(4)
    model = model.transform(ComputeFloorplanGroups(FloorplanGranularity.PER_NODE))
    model = model.transform(
        AssignFloorplanSLRs(platforms["U55C"](), U55C_PART, "U55C", slr_anchors={"MVAU_hls_0": [2]})
    )
    assert attr(model, "MVAU_hls_0", "floorplan_slr") == 2


# ------------------------------------------------------------------------------ anchors


@pytest.mark.fpgadataflow
def test_iodma_is_detected_as_externally_memory_connected() -> None:
    """The generic aximm rule fires on IODMA and not on plain compute layers."""
    model = make_model([make_iodma("dma0", "inp", "t0"), make_mvau("mvau0", "t0", "outp")])
    by_name = {node.name: node for node in model.graph.node}
    assert node_has_external_memory_interface(by_name["IODMA_hls_0"])
    assert not node_has_external_memory_interface(by_name["MVAU_hls_0"])


@pytest.mark.fpgadataflow
def test_hbm_board_anchors_to_the_hbm_slr() -> None:
    """On an HBM board, memory-connected nodes are pinned to the SLR holding the HBM."""
    model = make_model([make_iodma("dma0", "inp", "t0"), make_mvau("mvau0", "t0", "outp")])
    platform = platforms["U55C"]()
    anchors = derive_slr_anchors(model, platform, "U55C")
    assert anchors == {"IODMA_hls_0": [platform.hbm_slr]}


@pytest.mark.fpgadataflow
def test_ddr_board_allows_every_bank_slr() -> None:
    """DDR banks sit on several SLRs, and the solver may pick among them."""
    platform = platforms["U250"]()
    assert resolve_memory_anchor(platform, "U250", FpgaMemoryType.DEFAULT) == platform.ddr_slr


@pytest.mark.fpgadataflow
def test_host_memory_anchors_to_the_pcie_slr() -> None:
    """With host memory selected, the anchor target is the PCIe SLR, not HBM/DDR."""
    platform = platforms["U280"]()
    assert resolve_memory_anchor(platform, "U280", FpgaMemoryType.HOST_MEM) == [
        platform.get_pcie_slr()
    ]


@pytest.mark.fpgadataflow
def test_single_slr_part_derives_no_anchors() -> None:
    """There is nothing to anchor to on a one-SLR part."""
    model = make_model(
        [make_iodma("dma0", "inp", "t0"), make_mvau("mvau0", "t0", "outp")], part=ZYNQ_PART
    )
    assert derive_slr_anchors(model, platforms["Pynq-Z1"](), "Pynq-Z1") == {}


@pytest.mark.fpgadataflow
def test_nodes_across_a_device_boundary_are_anchored_to_the_transceiver_slr() -> None:
    """The nodes that will feed and drain the link kernel are pinned next to the GTYs."""
    model = make_chain(4)
    set_device_ids(model, {"MVAU_hls_2": 1, "MVAU_hls_3": 1})
    platform = platforms["U55C"]()
    anchors = derive_slr_anchors(model, platform, "U55C")
    # qsfp_slr is stored as "SLR1" on the U55C.
    assert anchors == {"MVAU_hls_1": [1], "MVAU_hls_2": [1]}


@pytest.mark.fpgadataflow
def test_anchors_lift_onto_the_unit_a_node_belongs_to() -> None:
    """A unit inherits the anchor of whichever member carries one."""
    units = {0: ["a", "glue"], 1: ["b"]}
    assert lift_anchors_to_units({"a": [1]}, units) == {0: [1]}


@pytest.mark.fpgadataflow
def test_disjoint_anchors_in_one_unit_are_a_user_error() -> None:
    """Glue between two differently anchored nodes makes its unit unplaceable."""
    units = {0: ["a", "glue", "b"]}
    with pytest.raises(FINNUserError, match="disjoint"):
        lift_anchors_to_units({"a": [0], "b": [2]}, units)


@pytest.mark.fpgadataflow
def test_unit_edges_aggregate_and_skip_internal_streams() -> None:
    """Edges inside a unit cost nothing; parallel edges between units are summed."""
    model = make_chain(3)
    unit_of = {"MVAU_hls_0": 0, "MVAU_hls_1": 0, "MVAU_hls_2": 1}
    edges = collect_unit_edges(model, unit_of)
    assert len(edges) == 1
    unit_a, unit_b, weight = edges[0]
    assert (unit_a, unit_b) == (0, 1)
    assert weight > 0


# ------------------------------------------------------------------------- mem_port


@pytest.mark.fpgadataflow
def test_memory_bound_nodes_get_a_system_port() -> None:
    """The floorplanner records the concrete Vitis system port for anchored nodes."""
    model = make_model([make_iodma("dma0", "inp", "t0"), make_mvau("mvau0", "t0", "outp")])
    model = model.transform(ComputeFloorplanGroups(FloorplanGranularity.PER_NODE))
    model = model.transform(AssignFloorplanSLRs(platforms["U55C"](), U55C_PART, "U55C"))
    # The U55C is an HBM board, so the DMA gets an HBM channel and the MVAU, which
    # has no AXI-MM port of its own, gets nothing.
    assert attr(model, "IODMA_hls_0", "mem_port") == "HBM[0]"
    assert attr(model, "MVAU_hls_0", "mem_port") == ""


@pytest.mark.fpgadataflow
def test_ddr_board_derives_the_bank_from_the_assigned_slr() -> None:
    """On a DDR board the bank follows placement, rather than placement following it."""
    platform = platforms["U250"]()
    for slr in range(platform.nslr):
        assert resolve_mem_port(platform, "U250", FpgaMemoryType.DEFAULT, slr, 0) == f"DDR[{slr}]"


@pytest.mark.fpgadataflow
def test_host_memory_reports_a_host_port() -> None:
    """With host memory selected the port is HOST, whatever the board has on card."""
    platform = platforms["U280"]()
    assert resolve_mem_port(platform, "U280", FpgaMemoryType.HOST_MEM, 1, 3) == "HOST[0]"


@pytest.mark.fpgadataflow
def test_hbm_channels_do_not_collide() -> None:
    """Two memory-bound nodes on one device must not be given the same channel."""
    model = make_model(
        [
            make_iodma("dma0", "inp", "t0"),
            make_mvau("mvau0", "t0", "t1"),
            make_iodma("dma1", "t1", "outp", direction="out"),
        ]
    )
    model = model.transform(ComputeFloorplanGroups(FloorplanGranularity.PER_NODE))
    model = model.transform(AssignFloorplanSLRs(platforms["U55C"](), U55C_PART, "U55C"))
    ports = [attr(model, n.name, "mem_port") for n in model.graph.node]
    hbm_ports = [p for p in ports if p]
    assert len(hbm_ports) == 2
    assert len(set(hbm_ports)) == 2


@pytest.mark.fpgadataflow
def test_non_dma_memory_nodes_get_no_system_port() -> None:
    """Only IODMAs get a mem_port, even though anchoring covers every AXI-MM node.

    CreateDataflowPartition rejects a partition holding more than one non-empty
    mem_port, and nothing downstream reads the value for a non-DMA node, so writing
    one there can only break a build.
    """
    model = make_model([make_iodma("dma0", "inp", "t0"), make_mvau("mvau0", "t0", "outp")])
    model = model.transform(ComputeFloorplanGroups(FloorplanGranularity.PER_NODE))
    model = model.transform(AssignFloorplanSLRs(platforms["U55C"](), U55C_PART, "U55C"))

    ports = {n.name: attr(model, n.name, "mem_port") for n in model.graph.node}
    assert ports["IODMA_hls_0"] == "HBM[0]"
    # At most one non-empty port per partition is what the SDP check allows.
    assert sum(1 for port in ports.values() if port) == 1


@pytest.mark.fpgadataflow
def test_a_small_chain_stays_in_one_slr() -> None:
    """Crossings cost; spreading a design that fits one SLR is never a goal.

    The only thing that pushes units into another SLR is the platform's
    utilisation ceiling (``DEFAULT_RES_LIMITS``, 70-80 % of each resource). A
    30-unit chain of small MVAUs is far below it, so it must land in a single
    SLR with no crossings -- the earlier minimum-share rule that forced every SLR
    to take part was the wrong tool and is gone.
    """
    model = make_chain(30)
    model = model.transform(ComputeFloorplanGroups())
    model = model.transform(AssignFloorplanSLRs(platforms["U55C"](), U55C_PART, "U55C"))
    slrs = {attr(model, node.name, "floorplan_slr") for node in model.graph.node}
    assert len(slrs) == 1, f"a chain that fits one SLR was spread over {sorted(slrs)}"


@pytest.mark.fpgadataflow
def test_equally_cheap_cuts_are_resolved_towards_balance() -> None:
    """Among cuts of the same crossing cost, the most even split is taken.

    Both ends of the chain are anchored to SLR 0 (like DMAs next to the HBM), so the
    chain leaves SLR 0 and comes back wherever it is cut, at the same cost. SLR 0 can
    hold all but one unit; crossings alone took any of the equally cheap cuts -- on
    mobilenetv1 that put 270 of 286 units in one SLR (16 Oct 2026).
    """
    n = 10
    capacity = {slr: {"LUT": 100 * (n - 1)} for slr in range(2)}
    units = {unit: {"LUT": 100} for unit in range(n)}
    edges = [(unit, unit + 1, 10) for unit in range(n - 1)]
    anchors = {0: [0], n - 1: [0]}
    problem = SLRAssignmentProblem(2, units, capacity, edges, [10_000], unit_anchors=anchors)
    solution = problem.solve(60)
    assert solution is not None
    assert problem.crossing_cost() == pytest.approx(2 * 10)
    in_slr1 = sum(1 for slr in solution.values() if slr == 1)
    assert in_slr1 == n // 2 - 1 or in_slr1 == n // 2, solution
