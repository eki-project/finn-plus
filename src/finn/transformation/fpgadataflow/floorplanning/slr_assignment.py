"""Level-1 SLR assignment for the automatic floorplanner.

Assigns each Level-2 placement unit to one SLR of the target part, minimizing the
SLL-weighted cost of crossing SLR boundaries subject to per-SLR resource budgets, the
per-boundary SLL budget, and the external-resource anchors from `anchors`.

This is a purpose-built formulation, not a reuse of the multi-FPGA `Partitioner`. That
one models a different problem: its resource budgets are whole-device pools rather than
per-SLR, it caps inter-device connections at the number of physical QSFP ports (which
has no analogue for SLL crossings inside a die), and its chain topology forces a
monotonically increasing device index along the dataflow graph, which is exactly the
constraint SLR floorplanning must not have. Only the modeling *pattern* is shared.

Device assignment (`device_id`, which physical board) and SLR assignment (which die
region on that board) are different axes decided in sequence. This pass never re-decides
`device_id`: it treats it as a hard partition of the node set and solves one independent
problem per device.
"""

import mip
from mip import xsum
from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from typing import cast

from finn.builder.build_dataflow_config import FpgaMemoryType, MIPSolver
from finn.transformation.fpgadataflow.floorplanning.anchors import (
    AnchorSet,
    derive_slr_anchors,
    lift_anchors_to_units,
    node_has_external_memory_interface,
    resolve_mem_port,
)
from finn.transformation.fpgadataflow.floorplanning.grouping import (
    get_placement_units,
    is_placeable_node,
)
from finn.transformation.fpgadataflow.floorplanning.resources import (
    DEFAULT_FLOORPLAN_RESOURCES,
    DEFAULT_RESOURCE_MARGIN,
    PLATFORM_RESOURCE_ORDER,
    get_node_resource_estimates,
    platform_slr_capacity,
    sum_group_resources,
)
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.fpgadataflow import get_device_id
from finn.util.logging import log
from finn.util.lp import create_mip_model
from finn.util.platforms import Platform

#: Extra SLLs consumed by an AXI-Stream crossing on top of its payload width, for the
#: tvalid/tready handshake pair.
STREAM_HANDSHAKE_SLLS: int = 2

#: Default time budget handed to the solver, in seconds.
DEFAULT_SOLVER_TIMEOUT: int = 300

#: Weight of the balance term in the objective (see `SLRAssignmentProblem`). The term is
#: the mean over budgeted resource types of each one's peak per-SLR utilisation, at most 1
#: in a feasible assignment, and every crossing costs at least one SLL; a weight below 1
#: therefore only decides between assignments with the same crossing cost and never buys
#: balance with a crossing.
BALANCE_WEIGHT: float = 0.5

#: Op-type prefix of the nodes whose `mem_port` is actually consumed downstream.
IODMA_OP_PREFIX: str = "IODMA"

#: An edge between two placement units: (unit_a, unit_b, SLLs consumed if they are on
#: different SLRs).
UnitEdge = tuple[int, int, int]


def collect_unit_edges(model: ModelWrapper, unit_of: dict[str, int]) -> list[UnitEdge]:
    """Collect the inter-unit dataflow edges and their SLL cost.

    Edges inside one unit carry no cost (the unit is placed as a whole) and edges
    between two devices are not an SLR-crossing question at all, so both are skipped.
    Parallel edges between the same pair of units are aggregated into one entry, which
    keeps the LP small and makes the per-boundary SLL budget come out right.

    Args:
        model: The model to read the dataflow graph from.
        unit_of: Maps node name to placement unit id.

    Returns:
        One entry per connected pair of units, with the summed SLL cost of the streams
        running between them.
    """
    weights: dict[tuple[int, int], int] = {}
    for node in model.graph.node:
        if not is_placeable_node(node):
            continue
        producer = getCustomOp(node)
        for out_idx, tensor in enumerate(node.output):
            consumers = model.find_consumers(tensor) or []
            for consumer in consumers:
                if not is_placeable_node(consumer):
                    continue
                if get_device_id(node) != get_device_id(consumer):
                    continue
                unit_a, unit_b = unit_of[node.name], unit_of[consumer.name]
                if unit_a == unit_b:
                    continue
                key = (min(unit_a, unit_b), max(unit_a, unit_b))
                width = cast("int", producer.get_outstream_width_padded(out_idx))
                weights[key] = weights.get(key, 0) + width + STREAM_HANDSHAKE_SLLS
    return [(unit_a, unit_b, weight) for (unit_a, unit_b), weight in sorted(weights.items())]


class SLRAssignmentProblem:
    """LP model assigning placement units to SLRs on one device.

    The formulation is built around one indicator per unit and per SLR *boundary*:
    `above[u][k]` is 1 when unit `u` sits above boundary `k` (i.e. in an SLR with index
    greater than `k`). An edge then crosses boundary `k` exactly when its two endpoints
    disagree about `above[.][k]`, which makes both the objective (total SLL-weighted
    crossings) and the per-boundary SLL budget fall out of the same linear expression.
    """

    def __init__(
        self,
        nslr: int,
        unit_resources: dict[int, dict[str, int]],
        capacity: dict[int, dict[str, int]],
        edges: list[UnitEdge],
        sll_budget: list[int],
        unit_anchors: dict[int, AnchorSet] | None = None,
        avg_constraints: list[tuple[tuple[int, ...], float]] | None = None,
        enforce_sll_budget: bool = True,
        solver: MIPSolver | None = None,
    ) -> None:
        """Build the LP.

        Args:
            nslr: Number of SLRs on the device.
            unit_resources: Per-unit resource demand, already margin-applied.
            capacity: Per-SLR resource budget, as from `platform_slr_capacity`.
            edges: Inter-unit edges with their SLL cost, from `collect_unit_edges`.
            sll_budget: Available SLLs across boundary `k` (between SLR `k` and `k+1`),
                one entry per boundary.
            unit_anchors: Units restricted to a subset of SLRs. Units absent from this
                mapping may go anywhere.
            avg_constraints: The platform's averaged utilization limits, each a tuple of
                resource positions (indices into `PLATFORM_RESOURCE_ORDER`) and the mean
                relative utilization they must stay below.
            enforce_sll_budget: Whether `sll_budget` is a hard constraint. Turning this
                off keeps the crossing cost in the objective but lets the solver exceed
                the SLL budget, which is occasionally the only way to place a design
                with very wide streams.
            solver: Which MIP solver to use; `None` picks the `mip` default with a CBC
                fallback.

        Raises:
            FINNUserError: If a unit's anchor set is empty or names an SLR the device
                does not have.
            FINNInternalError: If `sll_budget` does not cover every SLR boundary.
        """
        if len(sll_budget) < max(nslr - 1, 0):
            raise FINNInternalError(
                f"A device with {nslr} SLRs has {nslr - 1} boundaries, but only "
                f"{len(sll_budget)} SLL budget(s) were given."
            )
        self.nslr = nslr
        self.unit_ids = sorted(unit_resources)
        self.unit_resources = unit_resources
        self.capacity = capacity
        self.edges = edges
        self.enforce_sll_budget = enforce_sll_budget
        self.status: mip.OptimizationStatus | None = None
        anchors = unit_anchors or {}
        resource_types = sorted({res for demand in unit_resources.values() for res in demand})

        self.model: mip.Model = create_mip_model(solver, "finn_floorplan_slr")
        self.model.verbose = 0

        # --- assignment variables: one binary per (unit, allowed SLR) ---
        self.allowed: dict[int, list[int]] = {}
        for unit in self.unit_ids:
            allowed = anchors.get(unit, list(range(nslr)))
            invalid = [slr for slr in allowed if not 0 <= slr < nslr]
            if invalid:
                raise FINNUserError(
                    f"Placement unit {unit} is anchored to SLR(s) {invalid}, but the "
                    f"target device only has {nslr} SLR(s)."
                )
            if not allowed:
                raise FINNUserError(
                    f"Placement unit {unit} has no allowed SLR left after applying its "
                    "external-resource anchors, so it cannot be placed."
                )
            self.allowed[unit] = sorted(allowed)

        self.assigned = {
            unit: {
                slr: self.model.add_var(name=f"unit{unit}_in_slr{slr}", var_type=mip.BINARY)
                for slr in self.allowed[unit]
            }
            for unit in self.unit_ids
        }
        for unit in self.unit_ids:
            self.model += xsum(self.assigned[unit].values()) == 1

        # --- per-SLR resource budgets ---
        # An SLR that no unit is allowed into (every unit anchored elsewhere) gets no
        # constraint at all: the expression would have no variables in it.
        self.usage: dict[int, dict[str, mip.LinExpr]] = {}
        self.units_per_slr: dict[int, list[int]] = {
            slr: [unit for unit in self.unit_ids if slr in self.assigned[unit]]
            for slr in range(nslr)
        }
        for slr in range(nslr):
            self.usage[slr] = {}
            if not self.units_per_slr[slr]:
                continue
            for restype in resource_types:
                used = xsum(
                    self.assigned[unit][slr] * self.unit_resources[unit][restype]
                    for unit in self.units_per_slr[slr]
                )
                self.usage[slr][restype] = used
                self.model += used <= capacity[slr][restype]

        # --- the platform's averaged utilization limits (e.g. mean of BRAM/URAM/DSP) ---
        for positions, limit in avg_constraints or []:
            considered = [
                PLATFORM_RESOURCE_ORDER[pos]
                for pos in positions
                if PLATFORM_RESOURCE_ORDER[pos] in resource_types
            ]
            for slr in range(nslr):
                if not self.units_per_slr[slr]:
                    continue
                usable = [res for res in considered if capacity[slr][res] > 0]
                if not usable:
                    continue
                self.model += xsum(
                    self.usage[slr][res] / capacity[slr][res] for res in usable
                ) <= limit * len(usable)

        # --- boundary-crossing indicators ---
        self.n_boundaries = max(nslr - 1, 0)
        self.above = {
            unit: [
                xsum(self.assigned[unit][slr] for slr in self.allowed[unit] if slr > boundary)
                for boundary in range(self.n_boundaries)
            ]
            for unit in self.unit_ids
        }
        self.crossings: dict[tuple[int, int], list[mip.Var]] = {}
        for unit_a, unit_b, _ in edges:
            crossing = []
            for boundary in range(self.n_boundaries):
                var = self.model.add_var(
                    name=f"cross_{unit_a}_{unit_b}_b{boundary}",
                    var_type=mip.CONTINUOUS,
                    lb=0.0,
                    ub=1.0,
                )
                self.model += var >= self.above[unit_a][boundary] - self.above[unit_b][boundary]
                self.model += var >= self.above[unit_b][boundary] - self.above[unit_a][boundary]
                crossing.append(var)
            self.crossings[(unit_a, unit_b)] = crossing

        # --- per-boundary SLL budget ---
        if enforce_sll_budget and edges:
            for boundary in range(self.n_boundaries):
                budget = sll_budget[boundary]
                if budget <= 0:
                    continue
                self.model += (
                    xsum(
                        self.crossings[(unit_a, unit_b)][boundary] * weight
                        for unit_a, unit_b, weight in edges
                    )
                    <= budget
                )

        # --- balance: the peak relative usage of any budgeted resource in any SLR ---
        # Crossings alone are indifferent to where a chain is cut when several cuts
        # cost the same. Anchored designs have many such cuts: mobilenetv1's DMAs sit
        # in the HBM's SLR, so its chain leaves that SLR and comes back whatever it
        # does in between. With crossings as the only objective the solver took one
        # of them at random -- 167/118 units on the old code base, 16/270 once the
        # consolidated FIFOs made the estimates fit one SLR (16 Oct 2026) -- and the
        # packer then found no room in the full SLR for three 71-DSP MVAUs whose
        # measured LUTs exceed their estimates by a third. Among equally cheap
        # cuts, the most even one leaves every SLR the most room.
        # One peak per resource type, averaged: a single peak over all of them evened
        # out only the scarcest (block RAM 540/533) and left LUTs at 36k/96k.
        self.peaks: dict[str, mip.Var] = {}
        for slr in range(nslr):
            for restype, used in self.usage[slr].items():
                if capacity[slr][restype] <= 0:
                    continue
                if restype not in self.peaks:
                    self.peaks[restype] = self.model.add_var(
                        name=f"peak_{restype}", var_type=mip.CONTINUOUS, lb=0.0
                    )
                self.model += self.peaks[restype] >= used * (1.0 / capacity[slr][restype])
        balance = xsum(self.peaks.values()) * (1.0 / max(len(self.peaks), 1))

        # --- objective: total SLL-weighted boundary crossings, then balance ---
        self.model.objective = (
            xsum(
                self.crossings[(unit_a, unit_b)][boundary] * weight
                for unit_a, unit_b, weight in edges
                for boundary in range(self.n_boundaries)
            )
            + BALANCE_WEIGHT * balance
        )
        self.model.sense = mip.MINIMIZE

    def solve(self, timeout: int = DEFAULT_SOLVER_TIMEOUT) -> dict[int, int] | None:
        """Solve the assignment problem.

        Args:
            timeout: Solver time budget in seconds.

        Returns:
            Maps placement unit id to its assigned SLR, or None if no feasible
            assignment exists.

        Raises:
            FINNInternalError: If the solver itself reports an error, as opposed to
                reporting that the problem is infeasible.
        """
        self.status = self.model.optimize(max_seconds=timeout)
        if self.status == mip.OptimizationStatus.ERROR:
            raise FINNInternalError(
                "The solver returned an ERROR status while assigning SLRs. This is a "
                "problem with the solver or its installation, not with the model."
            )
        if self.status in (
            mip.OptimizationStatus.INFEASIBLE,
            mip.OptimizationStatus.NO_SOLUTION_FOUND,
        ):
            return None
        return {
            unit: next(
                slr for slr, var in self.assigned[unit].items() if var.x is not None and var.x > 0.5
            )
            for unit in self.unit_ids
        }

    def crossing_cost(self) -> float:
        """Return the solved assignment's SLL-weighted crossing cost (no balance term)."""
        return sum(
            (self.crossings[(unit_a, unit_b)][boundary].x or 0.0) * weight
            for unit_a, unit_b, weight in self.edges
            for boundary in range(self.n_boundaries)
        )

    def describe_demand(self) -> str:
        """Return a human-readable comparison of total demand against total capacity.

        Used to explain an infeasible problem, where the usual cause is simply that the
        design does not fit the device.
        """
        lines = []
        resource_types = sorted({res for demand in self.unit_resources.values() for res in demand})
        for restype in resource_types:
            demand = sum(self.unit_resources[unit][restype] for unit in self.unit_ids)
            available = sum(self.capacity[slr][restype] for slr in range(self.nslr))
            lines.append(f"  {restype}: {demand} required vs. {available} usable")
        return "\n".join(lines)


class AssignFloorplanSLRs(Transformation):
    """Assign every placement unit to an SLR and record it in `floorplan_slr`.

    Preconditions:
        - `ComputeFloorplanGroups` has run, so every hardware node has a
          `floorplan_group_id` and no unit spans more than one device.
        - Multi-FPGA device partitioning, if used, has run, so `device_id` is final.
          This pass reads it and never changes it.
        - The graph shape is final and layers are specialized, so resource estimates and
          stream widths reflect what will actually be built.

    Postconditions:
        - Every hardware node has a `floorplan_slr` in `[0, nslr)`, identical for all
          nodes of one placement unit.
        - The legacy `slr` attr is left untouched: bridging `floorplan_slr` onto it is a
          separate, explicit step.

    On a single-SLR part this reduces to assigning SLR 0 to everything without building
    an LP at all.
    """

    def __init__(
        self,
        platform: Platform,
        fpga_part: str,
        board: str | None = None,
        fpga_memory: FpgaMemoryType = FpgaMemoryType.DEFAULT,
        resource_types: tuple[str, ...] | list[str] = DEFAULT_FLOORPLAN_RESOURCES,
        resource_margin: float = DEFAULT_RESOURCE_MARGIN,
        use_hls_reports: bool = False,
        slr_anchors: dict[str, AnchorSet] | None = None,
        anchor_link_neighbours: bool = True,
        enforce_sll_budget: bool = True,
        solver: MIPSolver | None = None,
        solver_timeout: int = DEFAULT_SOLVER_TIMEOUT,
    ) -> None:
        """Initialize the transformation.

        Args:
            platform: Platform model of the target device. One SLR-assignment problem is
                solved per `device_id` against this platform; heterogeneous multi-board
                setups (a different platform per device) are not supported yet.
            fpga_part: FPGA part identifier, needed by the DSP cost models.
            board: Board name, used to decide HBM vs. DDR when anchoring.
            fpga_memory: Build-wide memory choice, used when anchoring.
            resource_types: Resource types to budget against. FF is excluded by default,
                see `DEFAULT_FLOORPLAN_RESOURCES`.
            resource_margin: Headroom applied to each unit's estimate.
            use_hls_reports: Whether to merge Vitis HLS synthesis numbers into the
                estimates. Requires `HLSSynthIP` to have run.
            slr_anchors: Per-node anchor override, mainly for debugging. When None,
                anchors are derived automatically from the platform and the graph.
            anchor_link_neighbours: Whether nodes adjacent to a device boundary are
                pinned to the transceiver SLR.
            enforce_sll_budget: Whether the per-boundary SLL budget is a hard constraint.
            solver: MIP solver to use; None picks the `mip` default with a CBC fallback.
            solver_timeout: Solver time budget per device, in seconds.
        """
        super().__init__()
        self.platform = platform
        self.fpga_part = fpga_part
        self.board = board
        self.fpga_memory = fpga_memory
        self.resource_types = tuple(resource_types)
        self.resource_margin = resource_margin
        self.use_hls_reports = use_hls_reports
        self.slr_anchors = slr_anchors
        self.anchor_link_neighbours = anchor_link_neighbours
        self.enforce_sll_budget = enforce_sll_budget
        self.solver = solver
        self.solver_timeout = solver_timeout

    def _sll_budget(self) -> list[int]:
        """Return the SLL budget across each adjacent SLR boundary of the device."""
        sll_count = self.platform.sll_count
        budget = []
        for boundary in range(self.platform.nslr - 1):
            row = sll_count[boundary] if boundary < len(sll_count) else []
            if boundary + 1 >= len(row):
                log.warning(
                    f"Platform records no SLL count for the boundary between SLR "
                    f"{boundary} and {boundary + 1}; not budgeting that boundary."
                )
                budget.append(0)
                continue
            budget.append(int(row[boundary + 1]))
        return budget

    def _assign_single_slr(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Assign SLR 0 to every hardware node, for single-SLR parts."""
        for node in model.graph.node:
            if is_placeable_node(node):
                self._record(node, 0)
        log.info("Floorplanner: target has a single SLR, assigned every node to SLR 0.")
        return (model, False)

    def _record(self, node: NodeProto, slr: int) -> None:
        """Write the assignment and its provenance onto one node."""
        inst = getCustomOp(node)
        inst.set_nodeattr("floorplan_slr", slr)

    def _record_mem_ports(
        self,
        nodes: list[NodeProto],
        solution: dict[int, int],
        units: dict[int, list[str]],
    ) -> None:
        """Write the concrete Vitis system port onto each externally connected node.

        Closes the loop back to `BuildBasicVitisLinkConfig`: it currently derives a
        DDR bank from the `slr` attr and picks HBM channels with a plain counter that
        knows nothing about placement. The floorplanner is the party that knows both
        the SLR a node landed in and which external resource it was anchored to, so
        it records the answer in `mem_port` for the link config to read.

        Safe to write: `mem_port` has no other producer -- its one existing call site
        only copies an already-validated value up onto a new SDP node.

        Deliberately narrower than the *anchor* rule, which fires on any node with an
        AXI-MM port. Two reasons, both about not writing a value nobody reads:
        `BuildBasicVitisLinkConfig` only emits a system port for IODMA kernels, and
        `CreateDataflowPartition` rejects a partition holding more than one non-empty
        `mem_port` -- which IODMAs never trip, since `Floorplan` gives each its own
        partition, but two external-weight Lookups in one partition would. Anchoring
        stays generic because *placement* really does apply to all of them.
        """
        slr_of_node = {name: slr for unit, slr in solution.items() for name in units[unit]}
        channel = 0
        for node in nodes:
            if not node.op_type.startswith(IODMA_OP_PREFIX):
                continue
            if not node_has_external_memory_interface(node):
                continue
            port = resolve_mem_port(
                self.platform,
                self.board,
                self.fpga_memory,
                slr_of_node[node.name],
                channel,
            )
            if not port:
                continue
            getCustomOp(node).set_nodeattr("mem_port", port)
            channel += 1

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Solve one SLR-assignment problem per device and record the result."""
        if self.platform.nslr <= 1:
            return self._assign_single_slr(model)

        units = get_placement_units(model)
        if not units:
            log.warning("Floorplanner: no hardware nodes to assign, skipping SLR assignment.")
            return (model, False)
        unit_of = {name: unit for unit, names in units.items() for name in names}
        nodes_by_name = {node.name: node for node in model.graph.node if is_placeable_node(node)}

        node_resources = get_node_resource_estimates(
            model, self.fpga_part, self.resource_types, self.use_hls_reports
        )
        unit_resources = sum_group_resources(
            node_resources, units, self.resource_types, self.resource_margin
        )
        capacity = platform_slr_capacity(
            self.platform.guide_resources, self.platform.res_limits, self.resource_types
        )
        anchors = self.slr_anchors
        if anchors is None:
            anchors = derive_slr_anchors(
                model, self.platform, self.board, self.fpga_memory, self.anchor_link_neighbours
            )
        unit_anchors = lift_anchors_to_units(anchors, units)
        edges = collect_unit_edges(model, unit_of)
        sll_budget = self._sll_budget()

        # One independent problem per device: device_id is a prior, fixed decision.
        units_by_device: dict[int, list[int]] = {}
        for unit, names in units.items():
            devices = {get_device_id(nodes_by_name[name]) for name in names}
            if len(devices) != 1:
                raise FINNInternalError(
                    f"Placement unit {unit} spans device(s) {sorted(devices)}. Grouping "
                    "must never merge nodes across a device boundary."
                )
            units_by_device.setdefault(cast("int", devices.pop()), []).append(unit)

        for device in sorted(units_by_device):
            device_units = set(units_by_device[device])
            problem = SLRAssignmentProblem(
                nslr=self.platform.nslr,
                unit_resources={unit: unit_resources[unit] for unit in sorted(device_units)},
                capacity=capacity,
                edges=[e for e in edges if e[0] in device_units and e[1] in device_units],
                sll_budget=sll_budget,
                unit_anchors={u: a for u, a in unit_anchors.items() if u in device_units},
                avg_constraints=self.platform.avg_constraints,
                enforce_sll_budget=self.enforce_sll_budget,
                solver=self.solver,
            )
            solution = problem.solve(self.solver_timeout)
            if solution is None:
                raise FINNUserError(
                    f"Could not assign SLRs on device {device}: the problem is infeasible.\n"
                    f"Total demand against usable per-device capacity:\n"
                    f"{problem.describe_demand()}\n"
                    "Either the design does not fit, an external-resource anchor cannot be "
                    "satisfied, or the SLL budget between SLRs is exceeded (retry with "
                    "enforce_sll_budget disabled to rule the last one out)."
                )
            for unit, slr in solution.items():
                for name in units[unit]:
                    self._record(nodes_by_name[name], slr)
            self._record_mem_ports(
                [nodes_by_name[name] for unit in sorted(solution) for name in units[unit]],
                solution,
                units,
            )
            log.info(
                f"Floorplanner: assigned {len(solution)} placement unit(s) on device "
                f"{device} across {self.platform.nslr} SLR(s); SLL-weighted crossing cost "
                f"{problem.crossing_cost():.0f}."
            )

        return (model, False)
