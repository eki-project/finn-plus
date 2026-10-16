# Automatic Pblock Floorplanning for FINN+ — Planning Doc

Status: **First floorplanned bitfile (mobilenetv1, 4 Oct 2026, 285/285 per-layer pblocks, kernel clock 112.4 vs 110.5 MHz baseline). Milestones 1-4 implemented; Milestone 5 validation running (Oct 2026); Milestone 6
(Level-2 packer rewrite) implemented on 2 Oct 2026 — packer in Python, Vivado as a thin
shell, 131 tests green incl. the Vivado tier (3 Oct 2026). Inside real links it has packed mobilenetv1
285/285 at 1.05x the SLICEs needed; resnet18 is blocked by its memory-LUT budget, not by
packing (§6 Milestone 6, §7.1/3). First placement of a new-engine floorplan pending.** Validation measured that the shelf packer is the
root cause of the remaining waste and coverage loss, so §3.2's Level-2 design is superseded by
Milestone 6; §3.2 and §5 are kept as the original rationale with "as built" notes where
reality diverged. §7.1 lists the decisions Milestone 6 still needs.

**Contents:** [1. Goal](#1-goal) · [2. Current state of the codebase](#2-current-state-of-the-codebase-verified)
([2.1 alive](#21-what-exists-and-is-alive), [2.2 dead](#22-what-is-dead--do-not-build-on-this),
[2.3 multi-FPGA partitioner not reusable](#23-the-multifpga-partitioner-aurorapartitionerpartitioner--verified-not-reusable),
[2.4 external-resource affinity](#24-external-resource-affinity--what-already-exists-verified))
· [3. Granularity](#3-design-decision-granularity)
([3.1 Level 1: SLR assignment](#31-level-1-slr-assignment-new-purpose-built--not-aurorapartitioner),
[3.2 Level 2: pblocks](#32-level-2-pblock-granularity-and-packing))
· [4. Data model](#4-data-model-all-new-nodeattrs-no-overloading-existing-ones)
· [5. Pipeline integration](#5-pipeline-integration)
([5.0 Zynq vs. Alveo correction](#50-correction-the-zynq-and-alveo-flows-are-structurally-different-here--verified),
[5.1 Phase A](#51-phase-a--logical-assignment-early-pure-python-both-flows),
[5.2 Phase B](#52-phase-b--physical-pblock-binding-late-vivado-side-flow-specific),
[5.3 config conflicts](#53-config-conflict-rule))
· [6. Phased implementation plan](#6-phased-implementation-plan) (Milestones 1-6; Milestone 6 = packer rewrite)
· [7. Open questions](#7-open-questions-for-iteration)
([7.1 remaining](#71-remaining) — Milestone 6 decisions, [7.2 resolved](#72-resolved))
· [8. Engineering conventions and test tiers](#8-engineering-conventions-and-test-tiers)

## 1. Goal

FINN+ currently emits stitched-IP / Vivado designs with no floorplanning by default: all
kernels land wherever Vivado's placer puts them. For large designs (many layers, big SLRs
or multi-SLR/multi-die targets) this hurts timing closure and congestion, and there is
already partial infrastructure for it, but split, disconnected, and in one case dead.

This doc plans a general **automatic pblock floorplanner** that:

- works at **individual layer/kernel granularity** (MVAU, FIFO, DWC, Thresholding,
  ConvolutionInputGenerator, etc.) — not just at `StreamingDataflowPartition` (SDP)
  granularity,
- works for the normal (non-DFX) build flow, not just partial-reconfiguration designs,
- reuses existing analytical resource estimation (`node_res_estimation`) and, optionally,
  post-synthesis utilization for refinement,
- produces real Vivado pblocks/XDC — via `make_zynq_proj.py`'s existing DFX injection point
  for Zynq/raw-Vivado, and via Vitis's `userPostSysLinkOverlayTcl` hook for Alveo, since the
  two flows turn out not to share one physical-placement mechanism (§5.0),
- **respects external-resource affinity**: a node bound to a specific HBM/DDR bank must
  land in the SLR that has a direct connection to it, and multi-FPGA-facing nodes
  (IODMA/ODMA, Aurora-link-adjacent kernels) must respect both their network-interface SLR
  and the device assignment already produced by the multi-FPGA partitioner (§3.1.1).

## 2. Current state of the codebase (verified)

### 2.1 What exists and is alive

- **Per-node resource estimation.** Every `HWCustomOp` implements `node_res_estimation()`
  ([hwcustomop.py:216](src/finn/custom_op/fpgadataflow/hwcustomop.py#L216)), aggregating
  `lut_estimation()`, `bram_estimation()`, `uram_estimation()`, `dsp_estimation()`. This is
  the natural area input for a floorplanner and is already summed across a whole model by
  `analysis/fpgadataflow/partitioning.py::res_estimation_complete` (which we can lift the
  *pattern* from, see §2.2 for why not the file itself).
- **SLR/partition nodeattrs.** `slr`, `partition_id`, `device_id`, `mem_port` already exist
  as nodeattrs on `HWCustomOp`, populated by
  [`floorplan_params`](src/finn/analysis/fpgadataflow/floorplan_params.py) and consumed/set
  by [`Floorplan`](src/finn/transformation/fpgadataflow/floorplan.py). `Floorplan` requires
  a user-supplied SLR-assignment JSON (or defaults everything to one SLR) — it does not
  compute SLR assignment itself, and it is only invoked today for Alveo, single-device
  builds in `step_prepare_synthesis`
  ([build_dataflow_steps.py:1850](src/finn/builder/build_dataflow_steps.py#L1850); full
  detail on exactly how in §5.0), not in `default_build_dataflow_steps`.
- **The real, maintained multi-FPGA partitioner.**
  `transformation/fpgadataflow/multifpga/partitioner.py`'s `Partitioner` ABC +
  `aurora/partitioner.py`'s `AuroraPartitioner` is an LP-based (via `python-mip`)
  **per-node** partitioner assigning each ONNX node to a device, with
  `PartitioningStrategy.LAYER_COUNT` / `RESOURCE_UTILIZATION` objectives. This is exercised
  by `tests/fpgadataflow/multifpga/`. Its docstring explicitly says it's a from-scratch
  reimplementation of the finn-experimental idea, "slightly different approach... should be
  relatively easy to swap in if needed" (referring to the old prototype, not encouraging use
  of the dead copy in-tree — see §2.2).
- **Physical pblock generation — but DFX-only.**
  [`dfx_auto_floorplanning.tcl`](src/finn/util/vivado_scripts/dfx_auto_floorplanning.tcl)
  is a real bin-packing floorplanner: queries post-synthesis `report_utilization` per
  hierarchical cell, walks clock-region (CR) columns narrowest-first, snaps pblock
  rectangles to CR boundaries in SLICE/RAMB36/DSP48E2 coordinate spaces simultaneously, and
  packs pblocks top-to-bottom without overlap. It's called only from
  [`make_zynq_proj.py`](src/finn/transformation/fpgadataflow/make_zynq_proj.py) (~line
  1608-1628), gated on every `NodeContainer` being a `partial_reconfiguration` body with an
  empty `pblock` attribute. A normal (non-PR) build never reaches this code.
- **Device/SLR resource model.** [`util/platforms.py`](src/finn/util/platforms.py)'s
  `Platform` class has per-SLR `compute_resources`/`guide_resources` (LUT/FF/BRAM_18K/URAM/
  DSP budgets net of DDR/HBM/Eth overhead) and SLR adjacency (`sll_count`,
  `compute_connection_cost`). This is exactly the per-region budget data a floorplanner
  needs, and is already consumed by the multi-FPGA partitioner.
- **Reporting scaffolding.** `_generate_pblock_svg`
  ([build_dataflow_steps.py:244](src/finn/builder/build_dataflow_steps.py#L244)) renders a
  pblock-capacity JSON into an SVG floorplan diagram; `write_pr_resource_report` (Tcl)
  writes that JSON. Both are DFX-specific today but structurally reusable.

### 2.2 What is dead — do not build on this

**`src/finn/analysis/fpgadataflow/partitioning.py` (`ILP_partitioner`, `partition()`,
`replicate_net`, `res_estimation_complete`) is orphaned dead code.** Verified:

- Nothing in the live `finn` package imports `ILP_partitioner` or `partition` from this
  file — the only occurrences of either symbol are inside the file itself.
- Its apparent caller, `step_mobilenet_slr_floorplan` in
  [`custom_step_library/mobilenet.py:101`](src/finn/builder/custom_step_library/mobilenet.py#L101),
  imports `from finnexperimental.analysis.partitioning import partition` — a **different**
  function from the external `finn-experimental` package, which this repo's git history
  shows was deliberately removed (`Remove finn-experimental and data_loading`).
  `finnexperimental` is not installed (`ModuleNotFoundError` confirmed on this machine).
  The import is wrapped in a bare `try/except Exception: print("No SLR floorplanning
  applied")`, so this step has silently no-op'ed on every build since finn-experimental was
  removed — it never actually reaches `finn.analysis.fpgadataflow.partitioning` at all.
- `step_mobilenet_slr_floorplan` itself is not registered in any step list — doubly dead.
- No test imports `finn.analysis.fpgadataflow.partitioning`.
- Even taken at face value, it has a real bug: `task_requirements` tuples are built as
  `(luts, 0, brams, urams, dsps, 0)` — FF/register usage is hardcoded to `0` and never
  estimated (a stray unused 6th field also exists) — so it would never have produced
  FF-safe floorplans even if reachable.

**Action for this project:** ignore this file entirely for new work (candidate for
deletion in a separate cleanup, out of scope here). Use `node_res_estimation()` directly
for resource sums — do not resurrect `res_estimation_complete`. Do **not** default to
"use `AuroraPartitioner` instead" — see §2.3, it doesn't fit either.

### 2.3 The MultiFPGA partitioner (`AuroraPartitioner`/`Partitioner`) — verified NOT reusable

It was worth checking in depth whether the *live, maintained* multi-FPGA partitioner could
be repurposed for intra-die SLR assignment (Level 1, §3) instead of writing something new.
It cannot, for concrete, structural reasons — this is a different problem, not just a
differently-configured instance of the same one:

- **Resource budgets are whole-device, not per-SLR.** `AuroraPartitioner.__init__`
  ([aurora/partitioner.py:180](src/finn/transformation/fpgadataflow/multifpga/aurora/partitioner.py#L180))
  gets `device_resources` from `available_resources_on_platform()`
  ([util/resources.py:130](src/finn/util/resources.py#L130)), which explicitly **sums a
  device's resources across all of its SLRs into one flat pool**
  (`_resources_per_device_per_slr` → summed in `available_resources_on_platform`). The
  partitioner has no notion of an individual SLR's budget at all — "device" here means a
  whole physical FPGA board, one level coarser than the SLR granularity this feature needs.
- **`ports_per_device` is a physical-network constraint.**
  (`self.model += self.connections_per_device[device] <= self.pcfg.ports_per_device`,
  [line 330](src/finn/transformation/fpgadataflow/multifpga/aurora/partitioner.py#L330))
  limits inter-device connections to the number of QSFP transceivers on a board. This has
  no meaning for SLR crossings inside one die (those go over SLLs, an entirely different
  and much larger budget already modeled by `Platform.sll_count` — used by the *dead* file,
  not by `AuroraPartitioner`).
- **Chain topology constraints assume a physical board-to-board network, not a placement
  objective.** `MFTopology.CHAIN`/`RETURNCHAIN`
  ([lines 376-398](src/finn/transformation/fpgadataflow/multifpga/aurora/partitioner.py#L376))
  pin input nodes to device 0 and output nodes to device `num_fpgas - 1` (or back to 0),
  and force `chosen_device[successor] >= chosen_device[node]` for every edge — i.e. device
  index must increase monotonically along the dataflow graph. That models a **physical
  chain of boards wired in series over Aurora links**. Reusing it for SLR assignment would
  impose a wrong constraint: SLR floorplanning wants to *minimize* crossings, it has no
  reason to force a monotonic SLR ordering along the graph.
- **The config object is shaped around network partitioning.** `PartitioningConfiguration`
  ([build_dataflow_config.py:238](src/finn/builder/build_dataflow_config.py#L238)):
  `num_fpgas`, `ports_per_device`, `topology`, `single_stream_network` — none of these
  translate to "how many SLRs does this one part have and how do I balance load across
  them."
- **`Partitioner` is an extension point for link technologies, not placement strategies.**
  It has exactly one concrete subclass (`AuroraPartitioner`), and
  `PartitionForMultiFPGA.apply` selects the partitioner class by `MFCommunicationKernel`
  ([partition_model.py:115](src/finn/transformation/fpgadataflow/multifpga/partition_model.py#L115))
  — today only `AURORA` is mapped. The ABC exists so a future non-Aurora inter-device link
  can plug in, not so arbitrary partitioning problems can reuse it.

**Conclusion:** write a new, purpose-built assignment step for Level 1 (§3) rather than
reusing or subclassing `Partitioner`. It's fine (and probably worthwhile) to reuse the
*modeling pattern* — `python-mip` binary/integer vars per node×region, resource-balance
constraints, `get_inseparable_nodes`-style grouping — but the objective (minimize
SLL/crossing cost against `Platform.guide_resources` per-SLR budgets) and constraints
(no chain/topology assumption, no port limits) need to be modeled fresh. This new
assignment logic should **not** live under `multifpga/` — it isn't a multi-FPGA concept —
and should not go through `PartitioningConfiguration`/`DataflowBuildConfig.partitioning_configuration`.

### 2.4 External-resource affinity — what already exists (verified)

Checked what already ties a node to a physical memory/network interface, since Level-1
SLR assignment must respect this rather than treating every node as freely placeable:

- **`Platform` already models which SLR owns which external interface.**
  [`util/platforms.py`](src/finn/util/platforms.py) has `hbm_slr: int` (single SLR with the
  HBM controller), `ddr_slr: list[int]` (SLR(s) with a DDR bank — e.g. U250 has DDR banks
  on multiple SLRs), and `qsfp_slr: str` (SLR with the QSFP/GTY transceiver used for
  Aurora). This is exactly the anchor-target data needed.
- **The QSFP/Aurora case is already anchored today, but only for the link kernel itself.**
  `AddAuroraToLinkConfig.apply`
  ([link_config_transform.py:148](src/finn/transformation/fpgadataflow/multifpga/aurora/link_config_transform.py#L148))
  calls `configs[device].add_slr(aurora_cu, self.platform.qsfp_slr)` — the Aurora transceiver
  CU is pinned to its SLR via `VitisLinkConfiguration.add_slr` (a direct Vitis-level SLR
  pin, [vitis_linking_configuration.py:215](src/finn/transformation/fpgadataflow/vitis_linking_configuration.py#L215)).
  It does **not** anchor the FINN-side SDP node that feeds/drains the Aurora kernel — only
  the Aurora kernel itself.
- **HBM/DDR pseudo-channel assignment is live, but SLR-unaware for HBM and only
  incidentally SLR-derived for DDR.** `BuildBasicVitisLinkConfig.apply`
  ([vitis_linking_configuration.py:680-705](src/finn/transformation/fpgadataflow/vitis_linking_configuration.py#L680)):
  for HBM boards (U50/U280/U55C) every DMA node on a device gets `HBM[n]` where `n` is a
  plain **sequential per-device counter** (`dma_on_device`) — no SLR affinity considered at
  all. For U250 (DDR-only), it instead does `mem_idx = 0 if node_slr == -1 else node_slr`
  — i.e. it **derives the DDR bank index directly from the node's existing `slr` attr**,
  assuming a 1:1 SLR↔DDR-bank correspondence. This means the *current* ordering for U250 is
  "SLR decided first (arbitrarily, since nothing assigns it today), memory bank follows" —
  the reverse of what's needed once a node has a *required* external bank. When no
  floorplan is supplied, `node_slr` defaults to `-1` → `mem_idx = 0`, so today every
  DMA on U250 silently lands on DDR bank 0 regardless of where it's actually placed.
- **IODMAs can also target host memory instead of on-card HBM/DDR — and this case has no
  SLR modeling at all today.** `FpgaMemoryType`
  ([build_dataflow_config.py:155](src/finn/builder/build_dataflow_config.py#L155)) has a
  `HOST_MEM` value alongside `DEFAULT`; when `cfg.fpga_memory == FpgaMemoryType.HOST_MEM`,
  `BuildBasicVitisLinkConfig` assigns `mem_type = "HOST"`, `mem_idx = 0`
  ([vitis_linking_configuration.py:680-682](src/finn/transformation/fpgadataflow/vitis_linking_configuration.py#L680))
  for every DMA, regardless of board. Two things worth noting for anchoring:
  - This is a **global, build-config-level choice** (`cfg.fpga_memory`), not a per-node
    one — there's no mixed HBM+host-memory design today, so the anchor logic only needs to
    branch on this one flag once, not infer memory type per node.
  - **`Platform` has no field at all for which SLR hosts the PCIe/XDMA shell interface**
    (checked: no `pcie_slr`/`host_slr`/`shell_slr` or equivalent anywhere in
    `util/platforms.py`, unlike `hbm_slr`/`ddr_slr`/`qsfp_slr`/`eth_slr` which all exist).
    This is a genuine data-model gap, not just a missing anchor rule — it needs a new
    `Platform` field. **Confirmed values: U55C and U280 have PCIe in SLR0.** Other Alveo
    boards (U50, U200, U250) are unconfirmed — rather than guessing, the field defaults to
    unset and a documented fallback-with-warning policy covers the unconfirmed case (see
    §3.1.2, §6 Milestone 1). Not a concern for embedded (Zynq) platforms at all — **verified**,
    not assumed, matching the rigor of the rest of this section: all five currently-defined
    Zynq platform classes in `util/platforms.py` (`Zynq7020_Platform`, `ZU3EG_Platform`,
    `ZU7EV_Platform`, `ZU9EG_Platform`, `ZU28DR_Platform`) are instantiated with `nslr=1`
    — so there is only one SLR to anchor anything to and no PCIe-shell-location question to
    answer in the first place, for every board this repo currently defines.
- **The FINN-level `mem_port` nodeattr exists but has no independent producer.** Declared on
  `HWCustomOp` ([hwcustomop.py:109](src/finn/custom_op/fpgadataflow/hwcustomop.py#L109)) and
  validated for SDP-partition uniqueness in
  [`create_dataflow_partition.py:106-122`](src/finn/transformation/fpgadataflow/create_dataflow_partition.py#L106).
  There is exactly one call to `set_nodeattr("mem_port", ...)` in the codebase, at
  [`create_dataflow_partition.py:125`](src/finn/transformation/fpgadataflow/create_dataflow_partition.py#L125)
  — but it does not originate a value: it copies the already-validated per-node value
  (collected from the per-node attrs a few lines above, `""` if none were set) up onto the
  newly created `StreamingDataflowPartition` node. So the attr never gets a *new* value from
  live code; it only ever carries forward whatever a user's floorplan JSON set on a node via
  `ApplyConfig`, or `""` otherwise. It is a placeholder/pass-through, not an active
  mechanism, and is a different thing from the Vitis-level `add_sp` string (`"HBM[0]"` etc.)
  described above — there's no code connecting the two today.
- **Multi-FPGA `device_id` is a separate, prior, already-fixed assignment.**
  `ApplyPartitioning`/`PartitionForMultiFPGA`
  ([partition_model.py:39,93](src/finn/transformation/fpgadataflow/multifpga/partition_model.py#L39))
  set `device_id` via `set_device_id()` before any SLR-level concern exists. This is a
  different axis entirely (which physical board) from SLR assignment (which die region on
  that board) — see §3.1.1.

**Design implication:** Level-1 SLR assignment cannot be "just" an SLL-crossing
minimization (§3.1) — it needs hard/soft anchor constraints for specific node categories,
and its output needs to *feed*, not fight, the existing (if currently naive) `add_sp`/
`add_slr` logic in `vitis_linking_configuration.py`. See §3.1.2 for the anchor mechanism
and node categories, and how that reconciles with the existing DDR `slr → mem_idx`
derivation (resolved via `mem_port` reuse, §7.2).

## 3. Design decision: granularity

The DFX floorplanner's CR-column snapping exists because **partial reconfiguration**
requires pblocks aligned to clock-region boundaries (a hardware/bitstream-relocation
constraint via `HD.RECONFIGURABLE`). A **static, non-reconfigurable** pblock used purely to
guide placement for timing/congestion has no such requirement — it can be an arbitrary
rectangle in SLICE_X/Y, RAMB36_X/Y, DSP48E2_X/Y site coordinates, down to a handful of
SLICE columns. This means per-layer granularity is *more* natural for the non-PR case than
the existing CR-snapped algorithm, not less — but it needs a different packing algorithm,
not a reuse of `dfx_auto_floorplanning.tcl`'s CR-search as-is.

Consequences of going to single-node granularity (MVAU, FIFO, DWC, Thresholding, ...):

- Designs can have hundreds to thousands of nodes. One hard pblock per node risks (a)
  Vivado runtime/object-count blowup, (b) placer infeasibility from over-constraining tiny
  glue logic (a FIFO or DWC pblock a few SLICE columns wide can make routing to neighbors
  impossible), (c) an explosion of `report_utilization -cells` queries if using
  post-synthesis refinement.
- Node sizes vary wildly (a large MVAU vs. a tiny FIFO/DWC), so a naive shelf-packing
  algorithm needs to handle a long tail of very small rectangles well.
- Dataflow adjacency matters physically: placing producer/consumer nodes far apart on the
  die lengthens the interconnect and can itself hurt timing, so pure bin-packing-for-density
  is not the only objective — pack **in topological/dataflow order**, snake/serpentine
  across the available region, so neighbors in the graph tend to be neighbors on the die.

### 3.1 Level 1: SLR assignment (new, purpose-built — not `AuroraPartitioner`)

Per §2.3, this needs a fresh assignment step, not a reuse of the multi-FPGA partitioner.
Shape of it:

- One LP (or, as a cheaper fallback, a greedy heuristic) that assigns each **Level-2
  placement unit** (§3.2) — never a raw node in isolation — to one SLR of the target part.
  **Ordering is decided, not a hedge: Level-2 grouping runs before Level-1 SLR assignment.**
  Each group becomes one atomic LP variable (for `PER_NODE` granularity, a group is one
  node, so this reduces to the same thing). This guarantees by construction that a
  placement unit can never span more than one SLR — the invariant §3.2 point 4 depends on
  ("the Level-2 search space must be clipped to its parent Level-1 pblock's site ranges,"
  which presumes exactly one parent) — and, for Alveo, can never straddle two
  independently-elaborated Vitis kernel sessions (§5.2), which the
  `userPostSysLinkOverlayTcl` mechanism has no way to express anyway. Grouping first also
  correctly propagates §3.1.2's anchor constraints to the whole group a node belongs to,
  not just the anchored node itself — an IODMA glued to a neighbor under
  `NODE_WITH_ADJACENT_GLUE` now automatically drags that neighbor to the same anchored SLR,
  rather than leaving the neighbor's SLR to an independent, potentially-conflicting
  assignment. See §5.1 for the corrected step order and §3.1.1 for the equivalent
  device-boundary constraint this same reasoning implies for grouping itself.
- Objective: minimize SLR-crossing cost, modeled against `Platform.sll_count` (SLL budget
  between SLR pairs) — this is the part of the *dead* file (§2.2) that was actually solving
  a relevant problem, just needs a clean reimplementation, not resurrection of that code.
- Constraints: per-SLR budget from `Platform.guide_resources` (already correctly per-SLR,
  unlike `AuroraPartitioner`'s whole-device pool) — no chain/topology constraint, no port
  limit, no monotonic-ordering requirement.
- Scope: lives outside `multifpga/` entirely (e.g.
  `transformation/fpgadataflow/floorplan_slr_assignment.py` or similar — exact module
  location TBD), and does not touch `PartitioningConfiguration`.
- Only used when the target has `nslr > 1` and no user-supplied floorplan JSON is given;
  no-op (everything on SLR 0) when `nslr == 1`.

#### 3.1.1 Must run *per device*, respecting the multi-FPGA partitioner's output

`device_id` (which physical FPGA/board a node is on) and SLR (which die region within
that board) are different axes, decided in sequence, not jointly:

- If `PartitionForMultiFPGA`/`ApplyPartitioning` has run, every node already has a fixed,
  non-default `device_id` (§2.4). Level-1 SLR assignment must **never re-decide
  `device_id`** — it treats it as a hard partition of the node set and solves one
  independent SLR-assignment problem per distinct `device_id`, each against that device's
  own `Platform` instance (same part in practice, but keep it per-device in case of
  heterogeneous multi-FPGA setups later).
- Pipeline ordering: multi-FPGA device partitioning (existing, if used) → **Level-2
  grouping (§3.2), constrained to never merge nodes across a `device_id` boundary** →
  per-device Level-1 SLR assignment (this feature, one atomic LP variable per group,
  §3.1) → Level-2 pblock packing/binding (§3.2, §5.2). This is a strict dependency chain,
  not just a convenient ordering — SLR anchor targets in §3.1.2 (e.g. `qsfp_slr`) are
  meaningless without knowing which device's `Platform` a node belongs to first, and the
  same "a placement unit can't span more than one X" reasoning that motivates
  grouping-before-SLR (§3.1) applies one level up here too: a `NODE_WITH_ADJACENT_GLUE`
  group must never bundle a compute node with a neighboring FIFO/DWC that a prior
  multi-FPGA partitioning pass already put on a *different device* — that's not just an
  SLR mismatch, it's a physically separate chip, so the grouping step itself must check
  `device_id` equality before merging, not rely on the (per-device) Level-1 LP to catch it
  afterward.
- When multi-FPGA is *not* in use, all nodes share the implicit `device_id = 0` default
  and this collapses to a single LP instance, same as today's mental model — the
  device-boundary check in grouping is then trivially satisfied for every group.

#### 3.1.2 Anchor constraints for externally-connected nodes

Certain nodes cannot be freely assigned to any SLR — they must land in (or be restricted
to) the SLR(s) that have a direct connection to a specific external resource:

- **HBM-bound nodes** → anchored to `Platform.hbm_slr` (single value).
- **DDR-bound nodes** → anchored to the allowed set `Platform.ddr_slr` (a list — some
  boards have DDR banks on more than one SLR, e.g. U250; when more than one is valid, the
  LP should be free to pick among them based on the crossing-cost objective, not be forced
  to a single arbitrary one).
- **Multi-FPGA link-facing nodes** — both the link kernel itself (already anchored today,
  §2.4) and, new here, its **immediate FINN-side neighbor** (the SDP/node that directly
  feeds or drains the Aurora TX/RX kernel) → anchored to `Platform.qsfp_slr` **on that
  node's own device** (per §3.1.1, anchor targets must be resolved against the correct
  device's `Platform`, not a global one). This reduces cross-SLR routing pressure on a
  timing-critical streaming link, matching the existing rationale for anchoring the Aurora
  CU itself.
- **Generic rule, not a hand-enumerated node-type list: any node whose
  `get_verilog_top_module_intf_names()["aximm"]` is non-empty is externally-memory-connected
  and gets an anchor.** This is the same API `Floorplan.apply()` already uses for its
  `axilite`/`ap_none` partition-boundary checks
  ([floorplan.py:207,214](src/finn/transformation/fpgadataflow/floorplan.py#L207)), so it's
  a precedented, low-risk way to detect the category generically instead of maintaining a
  by-hand op-type list that will drift as new layers are added. Branch on the single global
  `cfg.fpga_memory` flag (§2.4 — a build-wide choice, not per-node, read once):
  - `FpgaMemoryType.DEFAULT` → HBM/DDR anchor per the two bullets above, using whatever
    `FpgaMemoryType`/board mapping `BuildBasicVitisLinkConfig` already uses so the two stay
    consistent rather than computing the mapping twice.
  - `FpgaMemoryType.HOST_MEM` → anchor to the new `Platform.pcie_slr` field (§2.4/§6
    Milestone 1). Confirmed `pcie_slr = 0` for U55C/U280; other boards fall back to SLR0
    with a warning (§6) rather than blocking on confirming every board up front. Doesn't
    apply to `nslr == 1` (embedded/Zynq) targets — skip the fallback-warning path there.
  - **Verified concrete non-IODMA categories this rule catches** (audit done, not deferred
    — grepped every `intf_names["aximm"] = ...` assignment in
    `src/finn/custom_op/fpgadataflow/`, and confirmed by reading each branch, not just the
    grep hit — see the correction below, which is exactly why that second step matters):
    - **`Lookup`/`Lookup_hls` with `mem_mode == "external"`** — same pattern,
      `intf_names["aximm"] = [("m_axi_gmem", self.ext_mem_width)]`
      ([base/lookup.py:267](src/finn/custom_op/fpgadataflow/base/lookup.py#L267)) when its
      embedding table lives in external memory rather than BRAM.
    - **`FINNLoop`** ([rtl/finn_loop.py](src/finn/custom_op/fpgadataflow/rtl/finn_loop.py)) —
      declares its own `m_axi_hbm` interface (line ~1163) *and* aggregates any `aximm`
      interfaces from nodes nested inside its loop body (lines ~1160-1175). This is a
      composite/wrapper node: the anchor logic needs to treat it as one atomic Level-1 unit
      (matching how it's already packaged as one RTL top-level module) rather than trying to
      recurse into the loop body's sub-model separately.
    - A handful of other ops (`elementwise_binary_hls`/`rtl`, `attention_hls`,
      `attention_heads_hls`, `thresholding`) declare the `aximm` key but assign it an empty
      list in every code path checked — i.e. they participate in the shared interface
      contract but don't currently have a real external-memory port. Worth a final
      confirmation pass at implementation time (this was a grep-and-read audit, not
      exhaustive execution tracing), but nothing found suggests otherwise.
    - **Correction — `MVAU`/`VVAU` with `mem_mode == "external"` is *not* a case this rule
      needs to catch, and an earlier version of this doc claimed the opposite incorrectly.**
      Checked directly against
      [base/matrixvectoractivation.py:1101](src/finn/custom_op/fpgadataflow/base/matrixvectoractivation.py#L1101)
      and the equivalent branch in
      [base/vectorvectoractivation.py:812](src/finn/custom_op/fpgadataflow/base/vectorvectoractivation.py#L812):
      the `mem_mode == "external"` branch in both files appends to **`s_axis`** (an
      AXI-Stream input, `in1_V`), not to `aximm`. External-mode MVAU/VVAU has **no direct
      AXI-MM port at all** — its weight data arrives as a stream, fed by an upstream
      producer node (an IODMA reading the weights from external memory), and *that* IODMA
      is what actually needs, and already gets, the anchor. The earlier claim ("not routed
      through a companion IODMA node — this would have been missed by an IODMA-only rule")
      had it backwards: an IODMA-based rule would have caught this correctly; it's the
      generic `aximm`-based rule that (correctly) does *not* fire on the MVAU/VVAU node
      itself here, because there is genuinely nothing to anchor there. No anchor-mechanism
      change was needed — the mechanism was never wrong, only this example of it was.
      (Incidentally, `matrixvectoractivation.py` *does* have a real, different `aximm`
      case — the `mlo_max_iter` branch two lines above, unrelated to `mem_mode` — which the
      generic rule catches automatically without needing its own bullet here, which is the
      actual point of using a mechanism-based rule instead of a hand-maintained list.)
  - This closes the audit gap cleanly: since the rule is mechanism-based
    (query the interface) rather than a maintained list, a *future* op that adds an
    `aximm` interface is caught automatically without this section needing an update.

**Mechanism**: model each anchor as an equality (single allowed SLR) or set-membership
(multiple allowed SLRs) constraint on the SLR-assignment variable in the Level-1 LP. Since
Level-1 now assigns *groups*, not raw nodes (§3.1), anchors are computed per-node from the
category rules above and then **lifted to the group they belong to**: a group's allowed-SLR
set is the intersection of its members' anchor constraints (trivially "unconstrained" if no
member has one, and — since `NODE_WITH_ADJACENT_GLUE` only glues small, non-anchored
FIFO/DWC nodes onto a compute/IODMA neighbor, §3.2 point 1 — in practice this resolves to
"the anchor of whichever member has one," not an actual multi-anchor intersection problem).
If grouping ever did produce two independently-anchored members with disjoint constraints
(e.g. a degenerate case where a DWC sits directly between two externally-memory-connected
compute nodes), the resulting empty intersection is exactly the "anchor that cannot be
satisfied" case §8 already requires a `FINNUserError`/`FINNInternalError` for — no separate
handling needed here, just noting it falls under that existing catch-all rather than being
silently unaddressed.
The Level-1 module still takes an `slr_anchors: dict[node_name, int | list[int]]` as its raw
input (derived automatically from the node-category rules, not user-authored, though a
debugging override hook is reasonable to keep), and does the per-node → per-group lift
itself rather than pushing that responsibility onto the grouping step. This isn't a new
idea to invent — the constraint-modeling pattern (equality/set-membership on an assignment
variable) is the same one independently used by
`AuroraPartitioner.custom_partitioning_constraints` (exact equality,
[aurora/partitioner.py:226](src/finn/transformation/fpgadataflow/multifpga/aurora/partitioner.py#L226))
and by the *dead* file's `abs_anchors` (equality to one-of-a-list,
[partitioning.py:757-762](src/finn/analysis/fpgadataflow/partitioning.py#L757) — the pattern
is fine to reuse even though that file's code isn't, see §2.2).

**Feeding back into the existing (naive) `add_sp`/`add_slr` logic (§2.4):** once Level-1
assignment has resolved an external-memory anchor for a node, write the concrete system-port
string (e.g. `"HBM[3]"`/`"DDR[1]"`) into `mem_port` (§4 — reusing it is safe, unlike
`slr`/`device_id`/`pblock`). `BuildBasicVitisLinkConfig` can then read `mem_port` directly
instead of (DDR) deriving `mem_idx` from `slr`, or (HBM) picking a channel via a plain
per-device counter with no SLR awareness at all — closing both gaps from §2.4 in one move,
since the floorplanner is the party that actually knows the SLR↔resource mapping and the
per-node anchor requirement. This turns the "should this feature also fix the HBM gap"
question (§7.2) into "yes, cheaply" rather than separate follow-up work, since the
mechanism (`mem_port`) is already there to carry the answer.

### 3.2 Level 2: pblock granularity and packing

> **As built (Oct 2026) — read before the numbered points.** Four things below did not
> survive contact with the Alveo DFX flow and Milestone 5 measurements:
> (a) Level-2 pblocks are **hard**, not soft — the ULP is `HD.RECONFIGURABLE` and Vivado
> discards `IS_SOFT` on every child pblock (`Constraints 18-5661`); every pblock is also
> clipped to `pblock_dynamic_SLR<n>` (`18-12550`), must range every primitive type
> (`HDPR-18`) and must hold its measured demand (`UTLZ-1` is a hard error).
> (b) Sizing is **measured only**: the hook runs as `run.impl_1.STEPS.PLACE_DESIGN.TCL.PRE`
> (after `opt_design`, not `userPostSysLinkOverlayTcl` and not before `opt_design`, which
> blocked 1.4 M objects from optimisation), reads `report_utilization -hierarchical` per
> unit, and the estimate-only / refine modes were removed (point 2 below is historical).
> (c) Chains: BRAM/DSP/URAM cascades are measured per unit from the netlist and set a
> minimum contiguous height; a block RAM cascade must additionally lie inside **one clock
> region** (`Constraints 18-5197`, hit on resnet18 2 Oct 2026).
> (d) The serpentine **shelf packer is the root cause** of the remaining waste and coverage
> loss and is replaced by Milestone 6 (§6). Points 3-5 still hold in spirit — live device
> geometry, SLR-bounded search, degrade on Level-2 failure — but the geometry *arithmetic*
> moves to Python; Vivado stays the measurement and application layer.

1. **Two-level hierarchy of pblocks:**
   - **Level 1 (hard, coarse):** one pblock per SLR, from §3.1's assignment — a hard
     boundary, required for correctness in multi-SLR/multi-FPGA designs.
   - **Level 2 (soft, fine) — soft as first try.** Within each Level-1 region, one **soft**
     pblock (`IS_SOFT` property) per placement unit, sized from `node_res_estimation()`
     and placed by a snake/serpentine packer in topological order. Soft pblocks bias
     Vivado's placer without hard-failing if the estimate is off or the region is tight —
     this avoids the infeasibility risk of hard-constraining every FIFO. Hard Level-2
     pblocks are a possible future iteration once soft placement is validated, not part of
     the first implementation.
   - **Granularity is a configurable knob, not a fixed default — needed for testing.**
     Two supported modes:
     - `PER_NODE`: strictly one placement unit per `HWCustomOp` node (MVAU, FIFO, DWC,
       Thresholding, ... all get their own unit).
     - `NODE_WITH_ADJACENT_GLUE`: one placement unit per compute node plus its directly
       attached FIFO(s) and, **by default, its DWC(s) too** (default: DWC included in the
       glue grouping, config-overridable to exclude it for testing). Rationale for
       defaulting DWCs in: the mode's whole point is capping pblock count, and DWCs sit at
       essentially every width-changing boundary, so excluding them by default would
       undercut that purpose in most real designs; grouping a DWC physically with a
       neighbor for *pblock* purposes doesn't conflict with `Floorplan`'s separate
       narrow-bus-minimizing *SLR* heuristic (§4) once the bridge (§4, on by default) runs,
       since that operates on a different attr (`slr`) at a different granularity.
       **Hard constraint on the grouping step itself, checked before merging, not after:**
       never merge a compute node with a neighboring FIFO/DWC that a prior multi-FPGA
       partitioning pass already placed on a different `device_id` (§3.1.1) — that
       neighbor is a physically separate chip, not just a different die region, so it's
       ineligible for glue-grouping regardless of adjacency in the dataflow graph.
     Both modes must be implemented and selectable via build config so they can be
     A/B-tested against each other (pblock count / Vivado runtime / achieved Fmax)
     rather than picking one by fiat up front. **Grouping runs before Level-1 SLR
     assignment** (§3.1) — see there for why, and for the resulting invariant that every
     group is guaranteed to land in exactly one SLR.
2. **Both pre- and post-synthesis sizing must be supported, and selectable independently**
   for testing — this is not an either/or:
   - *Pre-synthesis estimate mode*: sum `node_res_estimation()` per placement unit with a
     margin factor (reuse the existing 1.1x default pattern). Always available, no
     synthesis required.
   - *Post-synthesis refinement mode*: after synthesis, re-query actual per-cell
     utilization (`report_utilization`) and re-pack if materially different from the
     estimate. At fine (per-node) granularity this needs batching — one
     `report_utilization -cells { ... many cells ... }` call, not one call per node — to
     stay tractable in Tcl. **Default repack trigger**: (a) hard — any resource where
     measured usage exceeds the unit's currently allocated pblock capacity (the pblock
     literally doesn't fit, always repack regardless of margin); (b) soft, default
     **on** — measured LUT usage differs from the estimate by more than **20%** in either
     direction (loosely matching the existing 1.1×/10% margin conventions already used
     elsewhere in this codebase, e.g. `dfx_auto_floorplanning.tcl`'s margin factors). Treat
     20% as a starting point to tune during Milestone 5 validation, not a fixed constant —
     expose it as a config field rather than hard-coding it.
   - A config flag selects estimate-only / post-synthesis-only / both-with-refinement, so
     the two modes can be compared directly during development.
3. **Do the geometry search live inside Vivado, not as a static Python site-coordinate
   model.** Per-part fabric layout (SLICE/RAMB36/DSP48E2 extents, clock region layout,
   shell/static-logic exclusion zones on Alveo, PS exclusion on Zynq) is exactly what
   Vivado already exposes live via `get_sites`/`get_clock_regions`/`get_pblocks`, and it
   varies per part in ways that would be fragile to hand-maintain as static Python tables
   kept in sync with every new supported part. **Reuse and generalize the
   `generate_multi_dfx_pblocks`-style live search** (query real device extents, iterate
   candidate rectangles, check actual site availability against already-placed pblocks) as
   the packing engine, rather than building a standalone offline Python geometry model.
   Python's role is orchestration (compute placement units, resource sums, unit ordering,
   emit the Tcl call) and result parsing/reporting, not geometry computation itself.
   Note this applies to *both* the pre- and post-synthesis modes (§3 point 2) — even the
   pre-synthesis pass benefits from querying live device geometry for accurate rectangle
   placement, it just uses estimated rather than measured area demands.
4. **SLR-bounded packing region:** the Level-2 search space must be clipped to its parent
   Level-1 pblock's site ranges — `Platform` already knows per-SLR split; today's DFX
   packer implicitly assumes one contiguous region because PR designs are usually
   single-SLR-sized, this assumption needs to become explicit and enforced (pass the
   parent pblock's site ranges as a hard bound into the search, not the whole device). This
   invariant — one parent SLR per unit — is now guaranteed by construction rather than
   merely assumed: grouping-before-assignment (§3.1) means a unit can never straddle two
   SLRs in the first place, so this step never needs a fallback for the "unit spans two
   parents" case that ordering would otherwise allow.
5. **Failure handling:** since Level 2 is soft by default, "failure" mostly means "the
   soft pblock couldn't be sized/placed sensibly" rather than a hard DRC failure — degrade
   by merging with a neighboring unit, shrinking margin, or omitting the pblock for that
   unit (free placement), never hard-failing the whole build. Level-1 (SLR) infeasibility
   should still be a hard error (that's a real correctness problem, same as today).

## 4. Data model: all-new nodeattrs, no overloading existing ones

Decision: this feature must not reuse or mutate the existing `slr`/`partition_id`/
`device_id`/`mem_port`/`pblock` nodeattrs. Those are load-bearing for other, already-shipped
functionality (`Floorplan`'s DWC/FIFO SLR-crossing logic, `ClusterByNodeattribute` and the
rest of the multi-FPGA SDP-clustering pipeline, `AddAuroraToLinkConfig`'s SLR pinning for
GTY transceivers, DFX `pblock` on `NodeContainer`). Silently reusing them risks unintended
interaction with all of that — e.g. this feature's own SLR choice being picked up by
unrelated multi-FPGA clustering code, or this feature's pblock string colliding with a PR
region's pblock. New, dedicated attrs on `HWCustomOp`:

- `floorplan_group_id` (int): groups nodes into one Level-2 placement unit under the
  currently configured granularity mode (§3.2 point 1). Computed fresh every run from the
  granularity setting, not user-editable.
- `floorplan_slr` (int): this feature's own SLR assignment from §3.1. Deliberately separate
  from the existing `slr` attr — a later, explicit bridging step (copy `floorplan_slr` →
  `slr`, then run `Floorplan`) rather than implicit sharing.
  **Decision: keep the *attrs* separate for now, revisit merging later** — reasoning
  unchanged from before: merging would couple this feature's output to `Floorplan`'s
  unconditional `partition_id` re-derivation (a wider blast radius than SLR placement
  alone) and lose independent A/B-testability. Not a permanent split, just not the default
  to start from.
  **But the *bridge step itself* defaults to ON, revised from the original "optional,
  off by default"** — for two reasons, one of which only became clear once §5.0 was
  worked out in full: (1) it's needed for `Floorplan`'s DWC/FIFO SLR-crossing-safe
  insertion, and §7's success criteria are measured on WNS/congestion at SLR boundaries —
  shipping without it risks an ambiguous validation result ("floorplanner is bad" vs.
  "floorplanner is fine but nothing pipelines the crossing"); (2) **for Alveo specifically,
  the bridge is load-bearing, not optional**: `step_prepare_synthesis`'s second
  `CreateDataflowPartition` call (§5.0) creates one SDP per `Floorplan`-derived
  `partition_id` — without the bridge, `Floorplan` sees only default `slr` values and this
  feature has no effect on Alveo kernel splitting at all. A config flag still exists to
  disable the bridge for isolated placement-quality-only A/B testing (§3.2), it's just not
  what a normal run does. See §5.1 point 3, §5.2.
- `floorplan_pblock` (str): the resulting Vivado pblock name/handle for this node's
  placement unit. Deliberately separate from the DFX-only `pblock` attr on
  `rtl/nodecontainer.py`.
- `floorplan_pblock_hard` (bool): whether this unit's pblock is hard or soft. Always
  `False` in the first implementation (§3.2 point 1 — soft as first try), present so a
  later hard-pblock iteration doesn't need another attr.
- `floorplan_resource_source` (enum: `ESTIMATE` / `POST_SYNTH`): which sizing pass produced
  the numbers behind this unit's current pblock — needed for the report/debugging, and to
  know whether a post-synthesis refinement pass actually changed anything.

**Exception: `mem_port` is safe to reuse.** Per §2.4, `mem_port` is declared
([hwcustomop.py:109](src/finn/custom_op/fpgadataflow/hwcustomop.py#L109)) and has exactly
one call site that sets it (`create_dataflow_partition.py:125`), but that call doesn't
originate a value — it only copies an already-validated per-node value up onto the newly
created SDP node. So it has no independent producer: nothing currently depends on its value
being anything other than whatever a user's floorplan JSON happens to set on a node via
`ApplyConfig`, or `""` otherwise — repurposing it carries none of the collision risk that
`slr`/`partition_id`/`device_id`/`pblock` carry. Use it as the
carrier for the external-memory anchor resolved in §3.1.2 (the actual system-port string,
e.g. `"HBM[3]"`/`"DDR[1]"`), rather than inventing a separate new attr for that one purpose.
This has a real payoff beyond avoiding attr proliferation: it finally gives the existing
SDP-uniqueness check in
[`create_dataflow_partition.py:106-122`](src/finn/transformation/fpgadataflow/create_dataflow_partition.py#L106)
real values to validate (today it only ever sees `""` or a user-supplied value), and it can
directly replace `BuildBasicVitisLinkConfig`'s ad hoc `slr → mem_idx` derivation and its
SLR-unaware HBM sequential counter (§2.4) with a single, explicit, floorplanner-decided
value that both the DDR and HBM branches read the same way.

On-disk config: a new, separate floorplan-pblock config file/schema (keyed by node name or
by `floorplan_group_id`) rather than extending `floorplan.json` — keeps this feature's
config independent of the existing `Floorplan`/`ApplyConfig` schema for the same reason as
above (avoid coupling to attrs/consumers this feature doesn't own).

## 5. Pipeline integration

### 5.0 Correction: the Zynq and Alveo flows are structurally different here — verified

An earlier version of this section assumed one insertion point would serve both flows via
a single generalized "pblock" mechanism. Re-reading the actual pipeline in full shows
that's wrong: the two flows don't just run `Floorplan` at different times, they use
different physical-implementation models entirely, and the generic
`step_create_dataflow_partition` behaves differently from what the rest of this doc had
implied. Verified facts:

- **`step_create_dataflow_partition`** ([build_dataflow_steps.py:1166](src/finn/builder/build_dataflow_steps.py#L1166),
  early, shared by both flows) splits off unsupported/software nodes at the graph boundary,
  then **hard-asserts exactly one resulting `StreamingDataflowPartition`**
  (`if not (len(sdp_nodes) == 1): raise FINNInternalError(...)`,
  [line 1236](src/finn/builder/build_dataflow_steps.py#L1236)) and immediately unwraps to
  that single child model, which becomes "the model" for every step from
  `step_specialize_layers` onward. It is a software/hardware boundary split, **not** an
  SLR-partitioning mechanism — nothing about `slr`/`Floorplan` is involved here.
- **Zynq**: `step_prepare_synthesis` is a literal no-op
  (`log.warning("Currently there is nothing to be done to prepare Zynq flow synthesis
  runs.")`, [line 1888](src/finn/builder/build_dataflow_steps.py#L1888)). No further
  splitting happens after `step_create_dataflow_partition`, ever, in the default pipeline —
  the whole accelerator stays one flat model through `step_create_stitched_ip`. `Floorplan`
  is never called for Zynq today. `CreateStitchedIP` stitches every node into **one flat
  Vivado block design**, and DFX pblocks (today, PR-only) are created on `Hier_<sdp_name>`
  cells inside that one design by `make_zynq_proj.py`, reachable only via `open_run synth_1`
  post-synthesis.
- **Alveo, single-device** (`cfg.partitioning_configuration is None`):
  `step_prepare_synthesis` ([lines 1830-1885](src/finn/builder/build_dataflow_steps.py#L1830))
  runs `InsertIODMA` → `InsertDWC` → `SpecializeLayers` → `PrepareIP`/`HLSSynthIP` →
  **`Floorplan(cfg.vitis_floorplan_file)`** → **a second `CreateDataflowPartition` call**
  (this one has no single-SDP assertion, so it can legitimately produce many SDPs, one per
  `Floorplan`-derived `partition_id`) → `BuildAllXOs`. **Each resulting SDP is packaged as
  an independent Vitis kernel (`.xo`)**, not stitched into one shared Vivado block design.
  Graph shape (which nodes exist) is therefore **not final until `step_prepare_synthesis`**
  for Alveo — IODMA/DWC insertion and layer specialization happen here, after where an
  earlier draft of this plan wanted to insert its own step.
- **Alveo, multi-FPGA** (`cfg.partitioning_configuration is not None`):
  `ApplyPartitioning`/`PartitionForMultiFPGA` (`device_id`) → `CreateMultiFPGAStreamingDataflowPartition`
  → `CreateNetworkMetadata` → `PrepareCommunicationKernels` → `BuildAllXOs`. **`Floorplan`
  is not called in this branch at all** — confirms §3.1.1's premise that no SLR-level
  assignment happens for multi-FPGA today, this feature is filling a real gap there, not
  duplicating one.
- **`add_slr` (the Vitis-level per-kernel SLR pin) is only ever called for the
  special-cased Aurora CU today** ([link_config_transform.py:148](src/finn/transformation/fpgadataflow/multifpga/aurora/link_config_transform.py#L148)) —
  `BuildBasicVitisLinkConfig` never calls it for a regular SDP/kernel. Even coarse,
  SDP-level SLR pinning is unwired for ordinary kernels in the Alveo flow today, not just
  the fine-grained pblock mechanism.

**Consequence:** "generalize pblocks to the non-PR flow" is really two different pieces of
work, not one, because the two flows expose physical placement at different granularities:

- **Zynq / raw-Vivado (incl. DFX)**: one flat block design → per-node/per-cluster Vivado
  pblocks (§3.2's Level 2) generalizes directly from the existing DFX mechanism. This is
  the flow the Level-2 design in §3.2 is proven against.
- **Alveo**: each SDP is an independent Vitis kernel, placed by `add_slr` at **kernel**
  granularity. Placing individual layers *within* one packaged kernel needs a different
  hook than the Zynq flow's `open_run synth_1`, but a real, documented, currently-used one
  exists: v++'s `param:compiler.userPostSysLinkOverlayTcl` config option (set in the
  `[advanced]` section of the v++ config file) runs a user Tcl script as a **post-system-link
  hook** — after v++'s system-link step (Vivado-level `link_design`, where every kernel and
  the shell are fully elaborated) but before place/route. Individual kernel instances and
  their internal ports are live and addressable in that Vivado session (the documented use
  case — wiring extra clock/power signals per-CU — confirms sub-kernel hierarchy is visible,
  not just kernel boundaries), matching the same "elaborated design, real cell paths
  resolvable" state `open_run synth_1` reaches for Zynq. AMD's own `xbtest` Alveo test suite
  independently confirms this class of mechanism is real and production-used: it floorplans
  compute units into named per-SLR pblocks (`pblock_dynamic_SLR0/1/2`) as standard practice.
  A second, complementary hook, `--vivado.prop run.impl_1.STEPS.PLACE_DESIGN.TCL.PRE=<tcl>`
  (and `.ROUTE_DESIGN.TCL.PRE`), injects Tcl at a specific point inside the `impl_1` run —
  useful for post-synthesis refinement mode specifically, since real per-cell utilization is
  queryable there too. Not yet run against an actual FINN-built multi-layer Alveo kernel in
  this repo's Vitis 2022.2/2024.2 environment, so the Milestone-1 spike (§6) stays in the
  plan — but as a fast confirmation of a documented mechanism, not an open feasibility
  question. See §7.2 (resolved, downgraded from a "no sane default" blocker).

### 5.1 Phase A — logical assignment (early, pure Python, both flows)

A new step, e.g. `step_generate_floorplan_assignment`, between `step_generate_hardware` and
`step_create_stitched_ip` in `default_build_dataflow_steps`
([build_dataflow_config.py:338](src/finn/builder/build_dataflow_config.py#L338)). This is
unaffected by the Zynq/Alveo split above — it only decides *what* should happen, using
`node_res_estimation()`, folding, and part/board metadata, all final by this point. It:

1. **Computes Level-2 placement units first** per the configured granularity mode and the
   pre-synthesis resource sum per unit (§3.2), writing `floorplan_group_id` and the
   `ESTIMATE`-sourced `floorplan_resource_source` (§4) — checking the `device_id`-boundary
   constraint (§3.1.1, §3.2 point 1) while forming groups. **Does not create any pblocks or
   call Vivado at all** — for Zynq/raw-Vivado there is nothing to bind cells to yet
   (no elaborated hierarchy exists); for Alveo, kernels don't exist yet either (`BuildAllXOs`
   hasn't run). This is exactly why Phase A/Phase B need to be separate steps, not one.
2. **Then runs Level-1 SLR assignment (§3.1)** when `nslr > 1` and no user floorplan is
   given, over the groups from step 1 (one atomic LP variable per group, not per raw node)
   — writing `floorplan_slr` per group (§4) — not `Floorplan`/the dead ILP file/
   `AuroraPartitioner`. This order (group, then assign) is deliberate, not incidental: see
   §3.1 for why assigning raw nodes first and grouping second would let a group span more
   than one SLR, violating §3.2 point 4's invariant.
3. Bridges `floorplan_slr` → `slr` and runs `Floorplan` afterward, **on by default** (§4 —
   revised from "optional, off by default": needed so the existing DWC/FIFO
   SLR-crossing-safe insertion actually happens; see §4 for the reasoning), with a config
   flag to disable it for isolated A/B testing of placement quality alone.
4. Feature-flag via `DataflowBuildConfig` (`enable_auto_floorplanning: bool = False`), plus
   the granularity-mode and pre/post-synthesis-mode fields from §3.2. See §5.3 for what
   happens when this is combined with existing floorplan/partitioning config fields.

### 5.2 Phase B — physical pblock binding (late, Vivado-side, flow-specific)

This is where real cell paths exist and `create_pblock`/`resize_pblock` calls are actually
possible — matching the one concrete precedent in the codebase
(`make_zynq_proj.py`'s DFX pblock logic, reachable only via `open_run synth_1`
post-synthesis, [make_zynq_proj.py:1606-1628](src/finn/transformation/fpgadataflow/make_zynq_proj.py#L1606)).

- **Zynq / raw-Vivado**: generalize `make_zynq_proj.py`'s existing DFX branch so it also
  runs for non-PR designs, reading `floorplan_group_id`/the Phase-A resource sums, resolving
  them against the now-elaborated `top_i/...` hierarchy, and running the generalized
  `generate_multi_dfx_pblocks`-style search (§3.2 point 3) to emit
  `create_pblock`/`resize_pblock`/`IS_SOFT`, writing the result back into `floorplan_pblock`
  (§4) for reporting. Post-synthesis refinement mode (§3.2 point 2) is the same code path
  run again at `impl` stage if the repack trigger (§7) fires.
- **Alveo**: wire up **kernel-level `add_slr`** in `BuildBasicVitisLinkConfig` (or an
  equivalent new pass) using each SDP's `slr`, closing the gap identified in §5.0 — this
  alone gets genuine, precedented SLR placement for ordinary Alveo kernels, which doesn't
  exist at all today. For Level-2 (sub-kernel) pblocks, inject the generalized geometry
  search (§3.2 point 3, same engine as Zynq) via a `userPostSysLinkOverlayTcl` hook script
  (§5.0) rather than reusing `make_zynq_proj.py`'s `open_run synth_1` injection point
  directly — the search algorithm is shared, only where it's invoked differs per flow.

### 5.3 Config-conflict rule

If a build sets `enable_auto_floorplanning=True` together with an explicit
`vitis_floorplan_file` and/or `partitioning_configuration`: **the explicit user config wins
and auto-assignment is skipped entirely**, with a `log.warning` naming which explicit
config took precedence. No partial-merge in v1 (e.g. "auto-assign only the nodes the user's
JSON didn't cover") — that's a reasonable future enhancement, not something to build before
the simple case is validated.

## 6. Phased implementation plan

Grouped into milestones with a suggested PR/review boundary after each — this is a large
feature (a new LP solver, a generalized Tcl geometry engine, two granularity modes, two
sizing modes, new nodeattrs, a new build step, and consumption-side changes in two
structurally different downstream flows), and shipping it as one continuous, unreviewed
block is its own risk independent of the design being right.

### Milestone 1 — pure Python, no Vivado needed, `quick`-testable

0. **Spike (do first): confirm `param:compiler.userPostSysLinkOverlayTcl` works end-to-end
   against a real FINN-built multi-layer Alveo kernel** in this repo's Vitis 2022.2/2024.2
   environment (§5.0/§5.2, §7.2) — a documented, AMD-precedented mechanism (`xbtest` uses
   the same class of hook for per-SLR pblock floorplanning), not an open unknown anymore,
   but not yet exercised against FINN's own generated kernels either. One manual `v++ -c`
   + a trivial `create_pblock`/`add_cells_to_pblock` test script is enough to confirm
   before Milestones 2-3 build on top of it.
1. **Add `Platform.pcie_slr`.** Per §2.4: `pcie_slr: int | None = None`, `0` on U55C/U280
   (confirmed), unset elsewhere, `get_pcie_slr()` accessor with the `nslr > 1`-gated
   fallback-and-warning behavior described there.
2. **Resource-summing entry point.** Sums `node_res_estimation()` per placement unit with
   margin — the pre-synthesis input to packing. Explicitly *not* built on
   `res_estimation_complete` (§2.2).
3. **Level-2 unit computation (Python half only) — runs before step 4, not after.**
   Grouping by `floorplan_group_id` (granularity mode, §3.2 point 1, defaults per §7),
   including the `device_id`-boundary check on grouping itself (§3.1.1), topological/
   serpentine ordering, margin application — kept separate from the Tcl/Vivado half
   (Milestone 2) specifically so this stays `quick`-testable (§8). Must land before step 4
   since Level-1 assignment consumes these groups as its LP variables (§3.1).
4. **Level-1 SLR assignment (§3.1, §3.1.1, §3.1.2), operating on the groups from step 3.**
   New LP/heuristic module: SLL-crossing objective against
   `Platform.guide_resources`/`sll_count`, per-device execution respecting pre-existing
   `device_id` (§3.1.1), and anchor constraints (§3.1.2) — including the **generic
   `aximm`-interface-based anchor rule** (see updated §3.1.2) rather than a
   hand-enumerated node-type list. One LP variable per group, not per raw node (§3.1). Own
   attrs (§4). Fully testable without Vivado.

*Suggested PR boundary here* — everything above is independently reviewable and testable
without a Vivado license, and validates the riskiest design assumption (the spike) before
any Vivado-side engineering starts.

### Milestone 2 — Vivado-side geometry engine, `vivado`/`slow`-tier tests

5. **Tcl emission — generalize `dfx_auto_floorplanning.tcl`.** Factor out the PR-specific
   bits (`HD.RECONFIGURABLE`, CR-snapping, `CONTAIN_ROUTING`); add `IS_SOFT` support
   (absent today — the script always creates hard pblocks) and SLR-bounded search ranges
   (§3.2 point 4). Add the batched `report_utilization -cells {...}` path needed for
   post-synthesis refinement at fine granularity (§3.2 point 2).
6. **Validate the generalized Tcl engine standalone** against a synthesized checkpoint
   (not yet wired into the build pipeline) before integrating — cheaper to debug the
   geometry search in isolation than inside a full build.

*Suggested PR boundary here.*

### Milestone 3 — pipeline integration (Phase A/B from §5)

7. **Phase A build step + config flags** (§5.1) — granularity mode, sizing mode, and
   `enable_auto_floorplanning`, all real independent config fields per §3.2, plus the
   config-conflict rule (§5.3).
8. **Phase B, Zynq/raw-Vivado**: generalize `make_zynq_proj.py`'s DFX branch to the
   non-PR case (§5.2).
9. **Phase B, Alveo**: wire up kernel-level `add_slr` in `BuildBasicVitisLinkConfig`
   (§5.0/§5.2 — closes a real, currently-unwired gap even at coarse granularity); Level-2
   sub-kernel work here is contingent on the Milestone-1 spike.
10. **`mem_port`/`BuildBasicVitisLinkConfig` consumption change** (§3.1.2, §4) — read a
    floorplanner-populated `mem_port` instead of the current `slr → mem_idx`
    derivation (DDR) and SLR-unaware sequential counter (HBM).
11. **Bridge `floorplan_slr → slr` + run `Floorplan`, on by default** (§4, §5.1 point 3).

*Suggested PR boundary here* — first point at which an end-to-end build with
`enable_auto_floorplanning=True` should run successfully on a small model before moving to
real validation targets.

### Milestone 4 — reporting

12. Generalize `_generate_pblock_svg`/JSON report beyond PR regions; include
    `floorplan_resource_source` per unit.

### Milestone 5 — hardware validation

13. **Cheap dry run first**: confirm the full pipeline produces valid pblocks/XDC and
    completes without error on a small model (estimate-only or a short synthesis-only
    build) — do not go straight to full P&R runs on the real validation targets, which are
    multi-hour, Vivado-gated (per this repo's own constraints) runs.
14. **Validation**: `resnet18` and `cnv-w1a1` on the U55C (§7). Both are single-device, so
    this exercises Level-1 multi-SLR assignment and (Zynq-proven, Alveo-contingent-on-the-
    spike) Level-2 pblock packing, but not the multi-FPGA `device_id`-respecting path from
    §3.1.1 — that needs a separate validation run once a suitable multi-FPGA example is
    chosen. Compare against a no-floorplanning baseline build of the same two models, and
    across granularity/sizing-mode combinations (§3.2 points 1-2), on the success criteria
    in §7.

### Milestone 6 — Level-2 packer rewrite: 2D packing in Python, Vivado as measurement and application layer (planned, Oct 2026)

**Why — measured, not assumed.** The resnet18 `per_node` validation run of 2 Oct 2026
(`floorplan_validation/out/resnet18_per_node_attempt1/`) is the reference data set:

- 258 of 289 units constrained; the 31 left unconstrained were crammed into whatever the ULP
  had left, which is how the earlier `Place 30-365` "737 % DSP" failure arose.
- LUT capacity claimed 468 k against 129 k measured demand (3.6×). 69 % of the excess is nine
  317-DSP MVAUs: the U55C has ~141 LUTs per DSP and the units need ~18, so any rectangle
  holding 317 DSPs holds ~45 k LUTs. That part is die geometry and no packer removes it —
  only letting other units use that area does (overlap, below).
- 181 of 258 units need < 84 LUTs but each spans a 21-24-row shelf and claims ≥ 168.
- Shelf statistics: four of SLR0's nine shelves hold two units (an MVAU at 80-95 % of the
  width plus one neighbour); the other shelves use 15-35 % of their width. The SLR was
  Y-exhausted after 217 of 240 rows with 55 % of its SLICE area unused. Same pattern in SLR2.
- The shelf model is the common cause: every unit on a shelf shares its height, so a 40-LUT
  unit and a 317-DSP unit cannot be shaped independently. Every fix since the audit (shelf
  floors removed, decoupled columns, fit loop, starved-kind → next shelf) was a workaround.

Findings from the follow-up runs of the same day (all three per-node builds failed in
`place_design`; each failure was traced to its cause before anything was changed):

- **Block RAM cascades must stay inside one clock region.** resnet18 attempt 1 died with
  `[Constraints 18-5197]`: a 5-deep RAMB36 cascade whose pblock window was rows 116..120,
  four rows in one region and one in the next. Fixed in the Tcl engine by measuring the
  clock-region bands per site kind from the device (`get_clock_regions`) and requiring a
  RAMB36/RAMB18 chain's rows inside one band (`FINN_FP_CR_BOUND_KINDS`). DSP and URAM
  cascades carry no such rule. The rule carries over into the Python packer unchanged.
- **Every chain needs its own column segment, not just the longest one.** mobilenetv1
  attempt 1 (and the `diag` run, same engine) died with `[Constraints 18-11842] cascade
  length exceeding BRAM column height in constraint area` on the last MVAU. An out-of-context
  synthesis of exactly that memstream wrapper on the U55C part (the OOC idea from the
  review) measured it: 113 RAMB36 in **17 chains of up to 8**, port-A cascades only
  (`CASDINA[0]` on 96 cells, `CASDIMUX*` unused). The engine's tracer agreed on the longest
  chain (8), so measurement was right and the rule was wrong: an 11-column, 12-row window
  offers at most 11 eight-row segments for 17 chains. Fixed by measuring the chain
  *multiset* per unit and kind and packing all chains into column segments (best-fit
  decreasing; segments split at clock-region bands; RAMB18 chains folded onto RAMB36
  columns) in shelf sizing, the column cursor and the post-apply verify
  (`finn_fp_chain_lengths`, `finn_fp_kind_segments`, `finn_fp_chains_pack`,
  `finn_fp_site_runs`). Also carries over as-is.
- **The static shell carves holes the census cannot see.** mobilenetv1's output DMA was
  left unconstrained because its RAMB36 range at SLICE X9..11 Y102..159 came back empty
  after the nested clip; the same column has sites on every other shelf. The engine now
  closes the shelf and retries instead of giving up (285/285 constrained in the next run).
  In the packer this is why per-column availability must come from real site lists, not
  extents and counts.
- **The fixes made resnet18 coverage worse, not better: 258 → 241.** With the clock-region
  rule, a chain unit that opens a shelf mid-region grows it into the next band; the rows
  it skips are lost, and SLR2 ended with ~90 unusable rows in gaps (Y504..529, Y619..642,
  Y667..714) and 25 unconstrained units. The fit loop made it worse: three 317-DSP MVAUs
  "did not fit" a 7.1 k-LUT window for 6.4 k LUTs — the failing resource is not the LUTs
  — and the loop answered by widening the SLICE window 1.6×, 2.6×, 4.1×, 6.6× (to 28 k
  LUTs) without touching the resource that actually failed, until the shelf ran out and the
  unit was dropped as "SLR too short for a 2-deep chain". A loop that grows one dimension by
  a guessed factor cannot fix a shortfall in another; the packer must size each kind from
  its own measured shortfall.
- **LUTRAM/SRL legalisation needs headroom the count check cannot see.** The resnet18
  restart on the clock-region engine (attempt 2, 2 Oct 2026) passed every cascade check,
  finished global placement and died in detail placement with `[Place 30-1112] Failed to
  move LUTRAM instance ... srl_reg[30][0]_srl31 to a legal location ... over-constrained`.
  The pblock (SLICE X41..48, 33 rows; a 1953-LUT SRL FIFO) held 5 SLICEM columns = 1320
  SRL-capable LUTs, enough by count for the measured shift registers, and the placer still
  could not legalise them. The hierarchical utilisation report does carry `LUTRAMs` and
  `SRLs` columns (verified on 2024.2), so the demand was measured; what is missing is that a
  SLICEM-bound resource cannot be packed to ~100 % of the sites in a hard region. This is
  Vivado's legaliser, not the packer, and it applies to the new packer unchanged. Open
  decision in §7.1: a utilisation ceiling for SLICEM-bound LUTs, the one place where a
  ceiling is a device rule rather than a guess.
- **What did work, and stays.** Measurement after `opt_design`; the availability-filtered,
  region-scoped census; the clock-region bands; the chain multiset and segment packing; the
  `Place 30-859` overrides channel; out-of-context synthesis of a single unit as a cheap,
  offline way to inspect what synthesis made of it when a placer message is ambiguous.

**Decision (2 Oct 2026): stop patching the shelf engine; build Milestone 6 now.** Each Tcl
fix cost a 2-3 h link run and surfaced the next structural problem; coverage fell while the
fixes landed. The two runs still in flight (resnet18 restart on the clock-region engine,
mobilenetv1 restart on the multi-chain engine) run to completion for the information only
— the first placer verdict with every known cascade rule satisfied, and the LUTRAM density
figure — and are not restarted on the Tcl engine if they fail. Everything needed to build
and test the packer offline now exists: measured demands, chain multisets, census, bands,
and the three preserved attempts under `floorplan_validation/out/*_attempt1/`.

**What stays.** One pblock per layer (granularity unchanged); model/folding configs untouched;
SLR assignment pre-link by the LP (§3.1) for now; measurement after `opt_design` in the
`PLACE_DESIGN.TCL.PRE` hook; the availability-filtered, dynamic-region-scoped census; the
measured cascade chains and the clock-region rule; the DFX realities listed in §3.2's as-built
note; the `Place 30-859` override loop as last-ditch retry (`placer_retry.py`).

**Architecture — three parts, one JSON contract.**

1. **Tcl becomes a thin shell** (`finn_floorplanning.tcl`, target ≈ 300 lines, down from
   ≈ 1300): (a) census → JSON: per SLR and site kind the columns with `x`, `y0`, `y1`,
   available-site count, the clock-region bands per kind, the parent-region bounds;
   (b) measurement → JSON: per unit `luts/ffs/carries/lutrams/srls/brams/dsps/urams` and
   chain lengths per kind; (c) `exec` the packer; (d) apply the returned ranges; (e) verify
   each pblock on its real sites — counts per kind, SLICEM for LUTRAM/SRL, longest run of a
   chained kind inside one clock region — and emit the shortfalls as JSON; (f) at most three
   pack/apply/verify rounds, re-packing only failed units; (g) write the report.
2. **The packer is a Python module**, `finn.transformation.fpgadataflow.floorplanning.packer`,
   runnable as `python -m ... --census c.json --units u.json --overrides o.json --out r.json`.
   Pure function of its inputs, deterministic, no Vivado import, `quick`-tier tests on
   fixtures captured from real runs (first fixture: the U55C census and demands of the
   attempt-1 run above; add a census/demand export switch to the current Tcl *before* the
   rewrite so the fixtures exist on day one).
3. **Interpreter and environment are baked in by FINN**, not discovered by Tcl.
   `render_floorplan_tcl` writes `sys.executable` and a whitelisted environment snapshot
   (`PATH`, `PYTHONPATH` with the worktree `src`, `LD_LIBRARY_PATH` as FINN saw it *before*
   Vivado rewrote it, `CONDA_PREFIX`, `HOME`, `LANG`, every `FINN_*`) into the rendered hook,
   which launches the packer with `exec env -i <whitelist> <python> -m ...`. Scrubbing the
   environment is what keeps Vivado's own library path away from the conda Python; the
   whitelist (never a dump of `os.environ`) keeps credentials out of a Tcl file in the build
   directory. Baked absolute paths are fine: the hook is regenerated per build. First link
   run with the shell: `exec` the baked interpreter, `import numpy`, print the version into
   the Vivado log — one line that retires the only environment assumption.

   JSON contract (sketch; freeze it in a docstring with a version field):
   `census = {slr: {kind: [{x, y0, y1, n}]}, cr_bands: {kind: [[lo, hi]]}, bounds: {slr: {x0, x1, y0, y1}}}`;
   `units = [{name, slr, order, demands: {...}, chains: {kind: len}}]`;
   `overrides = {name: {kind: extra}}` (from `placer_retry`, same channel as the verify
   shortfalls); `result = {name: {ranges: {kind: [x0, y0, x1, y1]}, status, waste}}`.

**Algorithm — skyline packing with per-unit shape selection, per SLR.**

- Units are not fixed rectangles: at a chosen height `h` and start column `x`, the width a
  unit needs follows from scanning the heterogeneous columns rightwards until every kind's
  demand is met from the per-column available counts in `[y, y+h)`. Tall-and-narrow and
  short-and-wide are both legal shapes of the same unit; the shelf model only ever produced
  the second.
- Process units in dataflow order (adjacency falls out of bottom-left placement, as it did
  from the serpentine); evaluate area-descending order on the fixtures too and keep the one
  that measures better — not both.
- For each unit: candidate heights from "minimum rows that satisfy demand at full width" up
  to one clock region, aligned to a band when the unit carries a block RAM chain; for each
  skyline position and height compute the minimal width; reject if a CR-bound chain cannot
  fit; choose the placement minimising claimed-minus-ideal SLICE area, tie-break lowest `y`.
  Three hundred units × a few hundred positions × a handful of heights is negligible.
- A unit with no legal rectangle anywhere is reported with its reason (no room vs starved
  kind); the Tcl shell raises `FINNUserError`-equivalent unless the config asks to degrade.
  "Leave it unconstrained" stops being the silent default.
- Before packing, compare the SLR's ideal area to its available area: an oversubscribed SLR
  is an LP-input problem (zero estimates, §7.1) and must be reported as such, not absorbed.
- Verify failures come back as per-kind shortfalls; the packer inflates exactly those demands
  and re-packs the failed units with everything else pinned. This replaces the
  `FINN_FP_FIT_GROWTH = 1.6` / five-attempt constants with measured numbers.

**Extensions — each a separate decision, see §7.1.**

- *Overlap / nesting.* `IS_SOFT` governs containment, `EXCLUDE_PLACEMENT` governs
  exclusivity, and only `HD.RECONFIGURABLE` pblocks may not overlap; our children have
  `EXCLUDE_PLACEMENT=0` and no `HD.RECONFIGURABLE`, so hard sibling pblocks may overlap and the
  placer arbitrates shared sites. This is the only lever against the die-ratio waste: a
  DSP/BRAM-heavy host exposes its spare SLICE capacity (5.7 k of 37-45 k LUTs used) and
  LUT-only units nest inside it. Every capacity check in play (census, verify, `UTLZ-1`,
  `30-859`) is per pblock and blind to sharing, so the packer must budget the shared area
  once and the verify step must sum nested demands per host. Needs one empirical link run
  (range one small unit inside one MVAU, let the placer accept it or name the rule).
- *Choose SLRs from measurement.* The v++ `slr=` directive pins kernels before link, so a
  unit the LP oversubscribed into an SLR cannot be rescued by geometry. Dropping the directive
  and assigning SLRs in the hook from measured demand closes that gap. Check first what
  `Floorplan`'s SLR-crossing FIFO insertion and the multi-FPGA anchors (§3.1.1-3.1.2) need
  from the pre-link `slr` attribute.

**Implemented 2 Oct 2026 (same day as the decision).** `floorplanning/packer.py` (stdlib
only, run by path) with `Census`/`Unit`/`Placement`, `SlrPacker` (skyline, incremental
column scan, candidate heights 5..240 rows, area-first order, chain segments split at
clock-region bands) and a `--previous/--feedback` round protocol; `finn_floorplanning.tcl`
reduced to census export (columns with tile-grid column, SLICEM flag and available-row runs;
bands per kind), measurement, `exec env -i` of the packer, apply, verify (counts, SLICEM,
chain runs) and three rounds; `tcl.py` bakes `sys.executable`, a whitelisted environment and
the packer path into the hook. Fixtures: `tests/fpgadataflow/floorplanning/fixtures/` holds
the U55C device census and the measured units of both attempt-1 runs. On those fixtures the
packer places **285/285 mobilenetv1 and 289/289 resnet18 units** (shelf engine: 284 and 258)
at a median LUT over-claim of **1.23 / 1.27** (shelf engine: 2.83 / 5.00) in about one second;
the 317-DSP MVAUs come out 60-67 tile columns wide by 180 rows instead of full-width strips.
Quick tests: 13 packer + the shell's helper tests; Vivado-tier tests drive the whole
census → exec → apply path on a bare part.

**First link run of the new engine (resnet18, 2 Oct 2026, 16:40) and what it changed.**
The census export, the scrubbed-environment launch and the pack/apply/verify path all worked
inside the real Vitis link. The packer placed 185 of 289 units and the DRC then failed with
`HDPR-18` on BUFG cells. Four findings, all measured, each with its fix:

- **Three 122k-LUTRAM weight memories.** The post-opt measurement (now with FFs, LUTRAM and
  SRLs, which the attempt-1 fixtures lacked) shows three MVAUs of ~150k LUTs each, 122k of
  them distributed RAM: per SLR the memory LUTs come to 79-82 % of the SLICEM ceiling at 0.8.
  A rectangle around such a unit covers ~85 % of the SLR and starves everything else. This is
  what forced the **column-selective claim**: a unit's region is now the set of columns it
  needs inside its span -- SLICEM columns for a memory, DSP columns for a DSP layer, a few
  SLICE columns for logic -- and the columns in between stay with the neighbours; the pblock
  gets one range per run of claimed columns. Nothing is shared, so every capacity check stays
  exact, and the die-ratio waste of §Why is gone without overlap: mobilenetv1 claims 175k LUTs
  for 168k demanded (shelf engine: 356k; rectangle packer: 308k).
- **Clock buffers inside a unit need a `BUFGCE` range.** Synthesis promotes the memories'
  read-pointer bits to BUFGs; a pblock whose cells contain any `BUFG*` now also ranges the
  region's `BUFG*` sites (`finn_fp_range_clock_buffers`).
- **Start columns and scan direction matter more than the waste metric.** Scanning rightwards
  from the lowest column of any kind walked into a neighbour's full-height SLICEM claim and
  read as "no room"; aborting a scan on one too-high column did the same for the DSP layers
  beside the memory. Both fixed: start at the lowest columns of *each* kind the unit needs,
  skip (do not abort on) columns taken too high. Two experiments were measured and rejected:
  penalising logic on SLICEM in the waste metric (288 → 229 placed) and trimming sparse
  columns to the rows actually needed (288 → 249; kept behind `TRIM_SPARSE_COLUMNS`, off).
  Snapping only sparse users to the tile pitch is required: without it small block RAM users
  need two tiles of height and exhaust the block RAM columns (mobilenetv1 285 → 274).
- **Kind rows must be anchored at the top of the clock region.** The U55C's bottom region
  has 18 DSP rows under 60 SLICE rows because its lowest tiles carry no DSPs; a proportional
  map handed one DSP site to two stacked units. `Census.map_rows` now uses the regular pitch
  anchored at each band's top, exclusive-consistent at both ends.

- **Per-column free intervals instead of a skyline.** A 2-BRAM unit found every block RAM
  column "full" in an SLR whose block RAM demand was 46 % of capacity: a unit that has to
  start higher because of one of its columns loses every row below it on the others, and a
  skyline never gives those rows back. The packer now keeps the free row intervals of each
  column and a unit may fill a hole under a neighbour. Measured on the mobilenetv1 attempt-1
  units: SLICE claimed over SLICE needed (LUTs *and* FFs) totals 1.04x, resnet18 attempt-1
  1.05x, resnet18 measured 1.50x (the three memories claim SLICEL columns for logic before
  their SLICEM columns arrive).
- **Empty kind windows must stay empty.** Rounding an empty DSP window up to one row handed
  DSP row 0 of the bottom region (whose lowest tiles carry no DSPs) to two stacked units.

Fixtures on the *device* census: mobilenetv1 285/285, resnet18 attempt-1 289/289, resnet18
measured 287/289.

**Second link run of the new engine (resnet18, 2 Oct 2026, 19:10): 141 of 289 placed, and the
offline replay reproduces it exactly once it uses the census the hook exported.** The device
census the fixtures were built on has no static shell; the real census, clipped to the
dynamic region and filtered for available sites, has 84-91 % of the SLICEM rows (SLR1 loses
17 of 111 SLICEM columns to the shell). Against it resnet18's measured memory LUTs are **96 %,
103 % and 95 %** of the three SLRs' SLICEM ceilings at 0.8, and 98 % of the whole device's.
SLR1 cannot hold its units at that ceiling by arithmetic, and 95 % is beyond what a
per-unit packer reaches. The packer now computes this budget per SLR before packing
(`slr_budget`), writes it into its result and warns in the Vivado log; the real census is
`fixtures/census_u55c_ulp.json` and a test pins the finding. Consequences:

- This is the SLR assignment's problem (its estimates carry no memory LUTs at all) and,
  device-wide, the design's: resnet18-w3a3 as folded needs 412k memory LUTs against 421k
  available at 0.8. Per-layer hard pblocks for this network need either a higher SLICEM
  ceiling (the legaliser's limit is the open question §7.1/4 was about), a different folding
  or memory style (the configs are to stay), or a granularity decision (§7.1/3).
- The packer's "fragility" measured earlier was this tightness showing through: every
  heuristic change moved units in a resource that was already over its budget.
- Measured per-node resources before SLR assignment (the out-of-context synthesis step
  proposed on 2 Oct) are now a prerequisite, not an improvement.

**First complete new-engine floorplan in a real link (mobilenetv1, 2 Oct 2026, 21:40).** All
285 units constrained in one round, every pblock verified on Vivado's site lists, placer
initialisation (cascades, HDPR-18) passed. Global placement then rejected one pblock:
`[Place 30-859]` BRAM, 14 required, 11 available -- in a region the packer gave 24 RAMB36
sites, two rightmost block RAM columns of SLR0 next to the static shell, of which the placer
closed 13 with `place.closeUnroutableSites` ("67 BRAM sites blocked" in the region). This is
the placer-internal closure the retry channel was built for; the retry adds the reported
shortfall and resumes implementation. Whether the closure is necessary at all for pblocks
with `CONTAIN_ROUTING=0` is being measured by the `diag` variant (same floorplan,
`place.closeUnroutableSites 0`, job 3550368): if it places and routes, the hook can set the
parameter and the census needs no notion of "unroutable"; if it fails in routing, the census
must learn to keep sparse columns one tile away from the static region.

**The LUTRAM/SRL packing wall, measured (2 Oct 2026, 23:30-00:00).** The `diag` run (closure
off) passed global placement with no block RAM rejection -- so the `30-859` was the closure --
and then died in detail placement with `[Place 30-484]`: 37,357 LUTRAM/SRL primitives,
"66,489 capable slices required of 75,120" (88.5 % of the ULP's SLICEM), while the measured
memory LUTs (17,408 LUTRAM + 29,361 SRL) suggest ~6k slices at eight per slice. Out-of-context
experiments with hard pblocks on the login node pinned the rule down per unit:

- An input generator (32 RAM64M8 + 32 RAM64X1D, 320 memory LUTs) places in one SLICEM column of
  60 slices, in line with the LUT-count model.
- The FIFO the placer named (1,120 SRLC32E, 1,232 measured SRL LUTs) fails in hard regions of
  60, 120 and 180 SLICEM slices and places in 360: the real need is between 180 and 360 slices,
  i.e. roughly 3-6 shift registers per slice, not the 6.4 (eight at the 0.8 ceiling) the packer
  allows. The placer's own "3,520 required by this constraint" is an upper bound it then
  ignores once the region is large enough, so it cannot be used as the number.

Bisection (hard pblocks, real SLICEM columns, 3 Oct 2026 00:00-00:30):

- Input generator, 32 RAM64M8 + 32 RAM64X1D = 320 memory LUTs: 60 and 90 SLICEM slices fail,
  120 places, and the macros occupy **64 slices -- one per macro**, whatever its LUT count
  (a RAM64X1D is 2 LUTs and still takes a slice). By LUT count the packer gave such a unit 50.
- SRL FIFO, 1,120 SRLC32E: 280, 300 and 320 slices fail, 340 places, and the shift registers
  occupy **320 slices -- 3.50 per slice**, not eight.

Implemented the same night: the hook counts distributed-RAM macros per unit from the netlist
(`finn_fp_measure_ram_macros`, REF_NAME `RAM<digit>*`, attributed like chains) and exports them
as `ram_macros`; `Unit.slicem_slices` sizes SLICEM in *slices* -- one per macro (LUT count
under the ceiling only when no macro count exists) plus shift registers at `DEFAULT_SRL_PER_SLICE
= 3.5` -- and the shell's verify applies the identical rule. The per-SLR budget is in SLICEM
slices now. On mobilenetv1's measured units the budget stays comfortable (285/285 on the real
census, 1.09x LUTs); on resnet18 the SLICEM ratio is 0.91 per SLR even on the bare device. The
job carrying the LUT-based rule was cancelled; job 3551006 carries the slice-based one.

The retry itself needed a correction the same evening: the placer's shortfall ("14 required,
11 available") is measured against the sites *it* left open, not against the region, so
adding 3 to a demand of 17 inside a 24-site region changed nothing -- the packer kept the
same two edge columns. `placer_retry` now sizes an override past the rejected region's
reported capacity (here +10, to 27 > 24) using the exported demands, which forces a different
selection; the job that would have spent two more attempts on the unchanged region was
cancelled and resubmitted with the fix (job 3550500).

**Attempt 4 (job 3551006, slice-based SLICEM rule, 3 Oct 2026 00:00-02:00): the closure is
the census's problem, not the retry's.** The hook did what it should: 29 units carry
distributed-RAM macros, round one placed 285/285 with 195,040 LUTs claimed for 169,279 demanded
(1.15x), nothing came up short. Vivado flagged 19 pblocks as "too small (3-4 Programmable
Units)" (HDPRA-65, a warning). Placer initialisation passed -- no `30-484`, so the slice-based
SLICEM rule holds at that gate -- and global placement then failed with the same `[Place
30-859]` as attempt 3: `pblock_finn_unit_70`, 14 block RAMs required, 11 available, 67 blocked
by `place.closeUnroutableSites`. Reading it against the census: the two rightmost block RAM
columns of SLR0 (RAMB36_X12/X13, tile columns 686/709) are shell remnants with 12 free rows each
(24 sites, Y0-11); the placer closes 13 of those 24, and the 54 unused sites above them (inside
the static region, which the packer's range still spans) besides. Seventeen units were packed
onto those two columns, so the retry's +10 on unit 70 moves one unit and leaves sixteen -- and
"309 or more BRAM failed to place" says as much. The retry channel is the wrong tool here: it
learns one region per attempt at an hour each, for a fact the census could know up front. Attempt
2 of the retry was cancelled (artefacts in `out/mobilenetv1_per_node_v2_attempt4/`, including the
first attempt's Vivado log and the locked shell checkpoint `hw_bb_locked.dcp`).

Two things run instead. (1) Measurement of the closure itself on the login node: the locked shell
checkpoint carries the static routing, so for every site in the reconfigurable region the script
asks whether a locked net occupies a node on the site's own pins (rule r0) or one PIP away (r1),
and compares with the placer's 13-of-24 on the two columns. If a rule reproduces that count, the
census excludes those sites and the hook never offers them -- the root-cause fix. (2) Job 3551390
(`diag_v2b`): the same floorplan with `place.closeUnroutableSites 0` and the slice-based SLICEM
rule, so the router, not the placer's heuristic, decides whether the sites are unroutable; the
earlier diag run never got that far (it died at the LUTRAM wall).

**The closure, measured (3 Oct 2026, 02:20-03:00).** Three experiments on the login node:

- A synthetic design with the real reconfigurable-region constraints but no static shell
  placed 24 block RAMs on SLR0's two edge columns (RAMB36_X12/X13 Y0-11) without a
  rejection, both with one pblock over all 24 sites and with one pblock per site. The
  closure needs the static shell.
- Asking the locked shell checkpoint which of those sites have shell routing on their own
  pins found 2 of 24. That probe is inconclusive; its query for the region's tiles returned
  nothing.
- One block RAM per site, each in a one-site pblock, inside the empty user slot of the
  locked shell checkpoint (`hw_bb_locked.dcp`): the placer rejected **13 of the 24 sites**
  with `[Place 30-859]`, every one attributed to `place.closeUnroutableSites`. That is
  exactly the 13 the real link closed. Closed: X12 Y0/2/3/4/6 and X13 Y0-6/8. Open:
  X12 Y1/5/7-11 and X13 Y7/9-11. Each closed site blocks 3 sites (the RAMB36 and both
  RAMB18), so the RAMB18 census needs the same exclusion. The run took about 11 minutes.

The closure is therefore a property of the platform shell, not of the design. It can be
measured exactly, once per platform, by this probe, and the census can drop the closed
sites before packing. The scripts and output are in `floorplan_validation/closure_probe/`.
Open: whether SLICE, DSP and URAM sites at the shell boundary are closed as well (only block
RAM has been rejected so far), and whether the router needs the closure at all (job 3551390,
closure off).

**Closure, whole region (3 Oct 2026, 04:15-05:30).** One pblock per site column per SLR,
filled with as many probe cells as the column has sites, on the locked shell: across the user
region, block RAM closure is confined to SLR0's RAMB36_X12 (5 of 12 closed) and X13 (8 of
12), the same 13 sites; no URAM column is closed. DSP cannot be probed region-wide (the placer
refuses a region that full of DSPs, `[Place 30-4]`, and counts each DSP48E2 as eight
sub-cells against single sites), so only the shell-clipped edge columns were probed: SLR0
DSP X29 has 10 of 18 sites closed, X30 16 of 18, X31 15 of 18; SLR2 X29 46 of 96; SLR2 X30/X31
none. So the closure follows the static shell's edge, for every sparse kind. SLICE sites are
not probed yet.

**diag_v2b (job 3551390, closure off, 3 Oct 2026 03:29-04:09).** Global placement passed --
the closure was attempt 4's only block RAM problem -- and detail placement failed with
`[Place 30-484]` on four input generators (units 2, 18, 60, 120). The hook had counted their
macros correctly; the packer had given each **one SLICEM slice per macro and all LUTs at
eight per slice**, and both halves of that were wrong:

- A slice that holds a macro holds no logic, so memory slices and logic slices add up: unit
  2 (64 macros, 366 logic LUTs) needs 64 + 46 slices and got 90.
- One SLICEM per macro is the *occupancy* once placed, not a placing threshold. Out of
  context, with each generator's own synthesised netlist, SLICEM = macros placed or failed
  at random (rtl_6: exact need placed, +10 % logic failed, +25 % placed; rtl_0 never placed,
  even with 25 % more SLICEL). SLICEM = macros / 0.9 and macros / 0.8 placed all three, and
  macros / 0.8 still placed with 10 % less SLICEL than the logic count.

Fixed the same morning: macros get the SLICEM utilisation ceiling like LUT-counted RAM
(`Unit.slicem_slices`: ceil(macros / 0.8) + SRLs / 3.5), and a unit's slice need is its
memory slices plus ceil(logic LUTs / 8) (`Unit.slices_needed`, logic = all LUTs minus
LUTRAM and SRL); the shell's verify applies the same rule. On the real mobilenetv1 census
and units the packer still places 285/285 (1.43x the demanded LUTs; unit 2 now gets 140
slices). The FINN-side retry does not parse `30-484` (its "required" is the placer's
eleven-per-macro upper bound, which would be useless as a size), and the FINN CLI returned
0 although the link failed -- both noted, neither changed. Job 3551563 (`diag_v2c`): the new
rule and the clipped ranges, closure still off, to see placement and routing through.

**diag_v2c (job 3551563, new slice rule, closure off, 3 Oct 2026 06:45-07:38).** All 285
units constrained (two came up short in round one against Vivado's sites and were re-packed
in round two). Global placement passed, and detail placement passed the LUTRAM commit phase
that stopped diag_v2b -- the slice rule holds. It then failed with `[Place 30-488] Failed to
commit 9 instances` (LUT, FF, CARRY8) of two input generators, units 238 and 256 (234 and 229
logic LUTs in 30 slices). Two findings:

- **Carries were never measured.** The hierarchical utilisation report has no CARRY8 column,
  so `carries` read 0 for all 285 units. The hook now counts CARRY8 per unit from the netlist
  (`finn_fp_measure_primitives`, with the distributed-RAM macros).
- **LUT density does not explain the failure.** 103 of 285 units ran at >= 95 % of their
  slices and placed; unit 92 is the same size as unit 238 (234 LUTs, 30 slices, exactly full)
  and placed. Out-of-context sweeps of the two generators were not representative (the
  standalone sources map the window buffer to 1,680 LUTRAM cells; the link maps it to block
  RAM). So no logic margin was added -- there is no measurement to base one on -- and the
  retry learned `30-488` instead: each failing instance is attributed to its unit through the
  unit's cells (now exported in `finn_floorplan_units.json`), and the unit's demand for the
  failed type (LUT/FF/CARRY) grows by `COMMIT_RETRY_GROWTH` = 0.25 of its pblock's capacity.

**Shell closure maps (3 Oct 2026).** The closed sites are measured once per platform and
Vivado release and cached; the hook marks them PROHIBIT before the census, so the census, the
verify step and the placer see the same device (`finn_fp_prohibit_closed_sites`; a closed
RAMB36 takes its two RAMB18 halves). Pieces:

- `floorplanning/shell_closure.py` drives `util/vivado_scripts/finn_shell_closure.tcl`: it
  extracts `hw_bb_locked.dcp` from the platform's `hw/hw.xsa` (found via PLATFORM_REPO_PATHS
  or /opt/xilinx/platforms), enumerates the user region's sites (2.5 min), probes one hard
  pblock per site column filled to capacity, then one per site in the columns that lost any,
  and parses the `[Place 30-859]` "blocked with place.closeUnroutableSites" counts. Pblocks are
  created from one XDC file (the per-pblock Tcl calls cost 8 s each in a DFX design). A run
  that trips the placer's utilisation guard (`[Place 30-4]`) is split in half and retried.
- `finn floorplan map-shell --platform <name> [--jobs N]` maps a platform and writes
  `floorplanning/shell_closure_maps/<platform>__vivado-<release>.json`;
  `finn floorplan show-shell` prints a map. `FINN_SHELL_CLOSURE_PATH` adds directories.
  The link configuration loads the map for its platform and the running release
  (`load_map`); without one it warns and gives the command.
- **Resolution is one tile.** A single-site pblock covers its whole tile, so the two DSP48E2
  sites of a DSP tile report together: the per-site counts add up exactly to the per-column
  ones (SLR0 DSP X30: 16 closed sites in 9 tiles), and a tile with one closed site lists both.
  Pblocks are built from whole tiles anyway. Block RAM is exact: a second, independent probe
  closed the same 13 RAMB36 sites -- the closure is deterministic.

**The U55C map (3 Oct 2026, 12:25-13:15, `finn floorplan map-shell --jobs 3` on the login
node, 50 min).** User region: 1,776 RAMB36, 8,376 DSP48E2, 960 URAM288, 143,280 SLICE sites.

| Kind | Columns that lost sites | Sites closed | Listed (tile resolution) |
|---|---|---|---|
| RAMB36 | SLR0 X12, X13 | 13 | 13 |
| DSP48E2 | SLR0 X28-X31, SLR1 X27-X28, SLR2 X29 | 233 | 282 |
| URAM288 | none | 0 | 0 |
| SLICE | none | 0 | 0 |

For every DSP column the per-tile counts of the site pass add up exactly to the column
pass's count. SLR1 DSP X28 loses 95 of 96 sites; the edge probe had not looked there.
SLICE was mapped in a second run (`--kinds SLICE`, merged into the map). CARRY8 probe cells
did not work -- unconnected they fail a DRC (`PDCY-3`), demoted they fail to commit (a floating
carry input), packed into every slice they overwhelm the placer (`[Place 46-13]`) -- so the
SLICE probe is one LUT1, and a column pblock gets eight per site (`COLUMN_FILL`) so that one
closed slice over-subscribes it. A check on ten shell-corner columns placed 4,800 such cells
cleanly; one-site pblocks over all 1,440 slices of the shell-enclosed SLR0 corner placed
cleanly too. The full run (8 batches of ~160k LUT1, 15:10 done) placed every column cleanly:
**no SLICE site is closed** (143,280 probed). A probe run that fails without a closure line is
now bisected down to single pblocks (a failed place_design unplaces every cell, so the log
cannot say which pblock was at fault) and the isolated one is recorded as unusable.
Job 3555658 (`v2e`): mobilenetv1 with the map, the
slice rule, the carry count and the `30-488` retry, closure on.

**Attempt 5 (job 3555658, closure on, with the map, 3 Oct 2026 14:38-15:20).** The hook
prohibited the map's 295 sites plus 26 RAMB18 halves; 77 units now carry a measured CARRY8
count; 285/285 constrained (one re-packed in round two). **Global placement passed with the
closure on** -- the map removed attempt 4's `30-859` without switching anything off (four
`30-859` *warnings* on unit 82, "0 required, 0 available", are a tool-generated combined
constraint over a borrowed range on prohibited sites, harmless). Detail placement passed the
LUTRAM commit and failed with `[Place 30-1153] Failed to commit 4 macros`: carry-chain shapes
(2-3 CARRY8 plus their LUTs and flip-flops, 9-28 instances each) of one input generator, unit
256 (SDP2 rtl_5, 229 LUTs and 21 CARRY8 in one SLICE column of 30 rows). It is not a clean
density effect: unit 220 (rtl_3, 239 LUTs, 13 CARRY8, 30 slices) placed in both runs, unit 238
failed in v2c and placed here, and 60 of the 77 carry units ran at >= 90 % and placed. That is
the margin the decided policy (some overprovisioning, retry with more) covers, so the retry
learned `30-1153` too (shapes attributed through the unit's cells, LUT demand + 25 % of
capacity, at least a slice per shape). The run had no such retry yet; it was resumed by hand
from implementation with the override the new parser computes (`pblock_finn_unit_256 {luts
60}`), job 3565093, `floorplan_validation/resume_link.sh`. Artefacts:
`out/mobilenetv1_per_node_v2_attempt5/`.

**Resumed links on attempt 5 (3 Oct 2026, 15:21-17:00).** Same floorplan, same link dir
(`tmp/mn_per_node/vitis_link_device_0_xmmsllyi`); the implementation log accumulates.

| Job | Change | Result |
|---|---|---|
| 3565093 | unit 256 +60 LUTs (30 -> 45 slices, one column, X195 Y360-404) | the same 4 carry macros of rtl_5 fail (`30-1153`) |
| 3565211 | unit 256's pblock deleted (its cells left to the region) | rtl_5 places; unit 238 (rtl_4) fails, 42 instances (`30-488`) |

**The same units, out of context (3 Oct 2026, evening).** Each generator's exact link netlist
(`write_checkpoint -cell` from the partition's synthesised checkpoint), placed alone into hard
pblocks:

| Netlist | Region | Result |
|---|---|---|
| rtl_5 (21 CARRY8) | 30 slices, X195 (its link region) | fails |
| rtl_5 | 30 slices, X100 (control column) | fails |
| rtl_5 | 45 slices, X195 (the retry's region) | places |
| rtl_4 (24 CARRY8) | 30 slices (its link region) | fails |
| rtl_3 (13 CARRY8) | 30 slices (its link region) | places |

So at 30 slices both units are genuinely too small, wherever they sit: carry-heavy logic
does not pack at eight LUTs per slice (a CARRY8 ties its slice to the carry's own LUTs). But
at 45 slices rtl_5 places alone and failed in the link, so something in the full design takes
room the unit needs. The leading hypothesis is squatting: unit pblocks are non-exclusive
(`EXCLUDE_PLACEMENT FALSE`, because borrowed ranges may overlap a neighbour), so cells
outside every unit -- ULP infrastructure -- may be placed inside them. Job 3575895 tests it:
both units +60 LUTs, unit 256 exclusive, unit 238 not.

Measured per unit from the partition netlists: LUTs driving a CARRY8 input. A rule "one slice
per CARRY8 plus the remaining logic at eight per slice" ranks units 238 and 256 highest (1.37
and 1.23 of their pblocks) but also puts a dozen large MVAUs at 1.17-1.20, which placed -- so
it is not the sizing rule. The out-of-context sweep (exact netlists, `opt_design` first)
measured the thresholds instead:

| Netlist | CARRY8 | Fails at | Places at |
|---|---|---|---|
| rtl_3 | 13 | 24, 26 | 28, 30 |
| rtl_4 | 24 | 30 | 34-42 |
| rtl_5 | 21 | -- | 30-42 (30 failed once without `opt_design`) |

The link gave each of them the slice count its LUTs need at eight per slice (234-239 LUTs,
30 slices), which is exactly the threshold region for a small carry-heavy unit: rtl_5 passes
or fails at 30 depending on the run, rtl_4 needs 31-34. (LUTs / 8 is not a lower bound either:
rtl_3 places 266 LUTs in 28 slices, two small LUTs sharing a site.) The retry's step (+25 % of
the pblock, 30 -> 38) clears all three; what the sweep cannot explain is rtl_5 failing at 45
slices inside the link -- that is the question job 3575895 answers.

**Job 3575895 (3 Oct 2026, 21:13-21:42): the failure follows the location.** Both input
generators +60 LUTs, unit 256 exclusive, 238 not. The packer swapped their places: 256 got
three columns at X10/X12/X14 (rows 375-389) and passed the macro phase; 238 got the column
the retry had given 256, SLICE_X195 rows 360-404, and five of its carry macros failed
(`30-1153`). Two different units have now failed in that 45-slice column, which the same
netlist fills without trouble out of context.

**Carry chains at the shell edge, measured in the locked shell.** Ten 3-CARRY8 chains (CI
tied to ground) per 45-slice region, hard pblocks, `place_design`:

| Region | Chains that failed to commit |
|---|---|
| SLICE_X195 Y360-404 | 8 of 10 |
| SLICE_X195 Y300-344 | 7 of 10 |
| SLICE_X194 Y360-404 | 4 of 10 |
| SLICE_X100 Y360-404 (control) | 0 of 10 |

So the shell also restricts slices for carry logic next to the static region, and the
LUT1 probe that found "no SLICE site closed" could not see it: a LUT fits where a carry chain
does not. The placer gives no closure line for it -- only `[Place 30-1153]` naming the
macros -- so the probe has to find the sites by pass/fail. A per-site single-CARRY8 probe is
running to learn the pattern.

**Carry closure mapped (3-4 Oct 2026, decided: prohibit if small).** `place_cell` accepts
a carry chain anywhere and no site or tile property marks the affected slices, so only the
placer's own verdict measures it. The map gained a `CARRY8` kind: every SLICE column run is
probed as one hard pblock filled with two-slice chains; a run that loses chains is halved and
probed again, down to runs of 15 windows (30 slices), which are closed as a whole if they still
lose any (`CARRY_MIN_RUN`). Finer resolution was tried on the edge columns and abandoned: the
placer aborts its whole run at the first shape it cannot place at all (`[Place 30-642]`), so
every fully closed piece costs a run of its own (~10 min), and in those columns 45-73 % of all
windows are closed anyway. Probe runs are named by a hash of their request, so an interrupted
mapping resumes; deeper levels run in small batches so `--jobs` finds aborts in parallel.

Result (4 Oct 2026, 00:17-02:27, `--jobs 3` then 6): **3,580 of 143,280 slices (2.5 %)** take a
LUT but no carry chain -- the SLR0 corner block the shell encloses (X197-X231, rows 0-59) and
the columns along the shell's edge (X192-X196; X195 rows 60-539, X196 rows 90-539), plus a few
30-slice runs further out. That includes SLICE_X195 rows 360-404, where both input generators
failed. At 2.5 % the hook simply prohibits them (they are SLICE sites in the map, so
`finn_fp_prohibit_closed_sites` needs no change); replaying attempt 5's packing on the census
without them still places 285/285. Job 3577610 (`v2f`): mobilenetv1 with the complete map.

**Job 3577610 (`v2f`, complete map, 4 Oct 2026 04:20-).** The hook prohibited 3,875 closed
sites and 26 RAMB18 halves; 285/285 constrained in one round. Placement got furthest yet: past
the macro/LUTRAM commit where unit 256 had failed, then `[Place 30-488]` in "Place Remaining"
with 88 plain-logic instances (50 LUT, 37 FF, 1 CARRY) of four units: the three carry-heavy
input generators at 30 slices (238, 256, 272 -- the 28-34-slice threshold region the sweep
measured; no longer at the shell edge) and unit 136, an FMPadding of 46 LUTs and 33 FFs in a
10-slice pblock two columns wide and five rows tall (57 % LUT fill). The automatic retry fired
as designed and resumed implementation; attempt 2 then failed with "failed to commit all
instances" and **no instance list** anywhere in the logs. Two retry defects showed:

- A commit failure raised the demand by 25 % of the region, which can still fit the same
  region: unit 136 went from 46 to 66 LUTs and got the same 10 slices. Overrides for `30-488`
  and `30-1153` are now sized past the rejected region (`_past_capacity`: the new demand is
  1.25x what the pblock offered), like the `30-859` path.
- Vitis appends every resumed attempt to the same `runme.log`, and the retry parsed it whole:
  after attempt 2 it re-counted attempt 1's rejections (doubling those overrides) and could
  not see that attempt 2 named nothing. It now parses only the latest attempt
  (`latest_attempt`, split at the hook's banner).

Attempt 3 (doubled overrides, old code) is running.

**Attempts 2-3 of v2f, bisected (4 Oct 2026, 07:00-11:00).** Attempts 2 and 3 failed with
"failed to commit all instances" and no instance list. With an opt-in pre-place checkpoint
(`FINN_FP_PREPLACE_CHECKPOINT`, written by the hook after floorplanning) the failure reproduces
on the login node -- it is deterministic -- but a failed `place_design` unplaces most of the
design (361,870 of 632,332 primitives), so the leftovers name nothing. Bisecting the unit
pblocks (each run keeps a quarter, frees the rest) found no culprit: all four quarters failed,
each on different small units (data-width converters, FIFOs, one thresholding unit), almost
all at a fill of 0.95 or more in pblocks of 5-45 slices, two at 0.70 in 10-slice pblocks two
columns by five rows. The failures are spread over the many units that sit at their limit; a
retry can move them, not end them.

**Decided 4 Oct 2026: a logic ceiling of 0.75 and a minimum pblock size.** Plain logic (logic
LUTs, flip-flops, carries) may fill 0.75 of a pblock's slices (`floorplan_logic_utilization`,
`DEFAULT_LOGIC_UTILIZATION`; packer and the shell's verify step alike). The minimum size is
Vivado's own: a nested pblock in a reconfigurable region should span 5 Programmable Units
(DRC HDPRA-65 warns below that, "may lead to unroute"). A Programmable Unit has no fixed slice
size -- a 5-slice and a 10-slice pblock both count 3 on the U55C, and the borrowed sparse
ranges appear to count too -- so after every round the hook runs `report_drc -checks
{HDPRA-65}` and re-packs each reported pblock grown by the factor it falls short
(`finn_fp_small_pblocks`); in the last round a small pblock is kept rather than dropped.
Replayed on the full-map census, mobilenetv1 still packs 285/285, now with 278,608 LUTs
claimed for 169,279 demanded (1.65x; at most 40 % of an SLR's LUTs). Job 3578807 (`v2g`).

**Job 3578960 (`v2g`, logic ceiling 0.75 + Programmable-Unit minimum, 4 Oct 2026).** The
HDPRA-65 check re-packed 6 pblocks in round 1, 3 in round 2, and kept 2 still below 5 PUs
in round 3 (the count depends on a pblock's shape, not only its size). **Placement passed** --
the first floorplanned mobilenetv1 link to do so -- and post-placement optimisation estimated
WNS +0.003 ns, TNS 0. Routing failed in its initial phase: `[Route 35-7] Design has 33
unroutable pins`, all on the weight-memory read bus of unit 90 (MVAU_rtl_4), whose block RAM
range was exactly the 11 sites the placer leaves open in SLR0's edge columns RAMB36_X12/X13
(the other 13 are closed). The placer's closure is not the whole edge effect.

**Decided 4 Oct 2026 (option one): the column rule.** A block RAM or DSP column that loses any
site to the closure is excluded as a whole within its SLR (`close_partial_columns`, applied
by `map-shell`; the map records the added sites in `column_rule`). On the U55C: +11 RAMB36
(SLR0 X12/X13 entirely) and +150 DSP48E2 (92 of them SLR1 X27, where 4 sites close). The
alternative -- a routing probe that measures which edge sites route a wide bus -- is the
follow-up that would replace the rule with a measurement. Replayed: still 285/285. Job
3579091 (`v2h`).

**First bitfile: job 3579091 (`v2h`, 4 Oct 2026, 16:58-18:46).** With the complete map
(4,036 sites and 48 RAMB18 halves prohibited, column rule included), the logic ceiling 0.75 and
the Programmable-Unit check (8 pblocks re-packed in round 1, 2 in round 2, none short at the
end), mobilenetv1 placed, routed (initial routing clean -- the column rule removed the
`Route 35-7`), and wrote a bitstream: **285/285 units in hard per-layer pblocks**, no placer
retry needed. Against the unfloorplanned baseline (`compare.py`, Vitis' auto-scaling):

| | Baseline | Floorplanned |
|---|---|---|
| Kernel clock achievable | 110.5 MHz | 112.4 MHz (+1.7 %) |
| HBM clock achievable | 463.1 MHz | 466.2 MHz |
| Router WNS / TNS | -0.017 / -0.132 ns | -0.017 / -0.132 ns |
| Place / route time | 27:24 / 15:27 | 31:50 / 15:47 |
| LUT / SRL / FF | 267,026 / 21,366 / 289,010 | 267,767 / 21,882 / 289,079 |
| BRAM36 / DSP | 717 / 700 | 717 / 700 |

WNS/TNS are identical to the last digit, so the worst path sits in the locked static shell,
not in the user region; the kernel clock's achievable frequency is the number that measures
the floorplan. On this design the floorplan costs no timing (+1.7 % is within run-to-run
noise) and no resources; placement takes ~16 % longer. Artefacts and the xclbin:
`floorplan_validation/out/mobilenetv1_per_node_v2_attempt8_bitfile/`. Not yet run on the
board.

**Job 3579091 (`v2h`, 4 Oct 2026): first floorplanned mobilenetv1 bitfile.** Map with the
column rule (4,036 sites + 48 RAMB18 halves prohibited), logic ceiling 0.75, PU minimum (8
pblocks re-packed in round 1, 2 in round 2, none left small). Placed, routed (no unroutable
pins), bitstream written, `finn-accel-0.xclbin` produced; Vitis kept the 100 MHz kernel clock
(it estimated 112.4 MHz possible).

| | Baseline | Floorplanned |
|---|---|---|
| place / route time | 27:24 / 15:27 | 31:50 / 15:47 |
| LUT / FF / BRAM36 / DSP | 267,026 / 289,010 / 717 / 700 | 267,767 / 289,079 / 717 / 700 |
| overall worst setup slack | +0.003 ns (shell MicroBlaze, DMA clock) | identical |
| FINN worst setup slack (kernel clock) | 2.803 ns | 1.909 ns |
| FINN mean slack, 1,000 worst endpoints | 4.72 ns | 2.67 ns |
| FINN worst hold slack | 0.010 ns | 0.010 ns |

The router's WNS/TNS are identical in both because the worst path is in the static shell; the
FINN design's own timing (static timing on both routed checkpoints, paths that start and end
inside the StreamingDataflowPartitions) is **worse with the floorplan**: about 0.9 ns less on
the worst path (implied FINN Fmax ~139 -> ~124 MHz) and 2 ns less on average over the 1,000
worst endpoints. Both meet 100 MHz with margin. The worst path in both is a data-width
converter driving the clock enable of a wide SRL FIFO (two logic levels; datapath 6.87 ->
7.78 ns), i.e. wire delay between units that hard pblocks place further apart than the
unconstrained placer does. Making the floorplan improve timing rather than cost it is the next
problem: placing connected units next to each other (the packer orders by area, not by
dataflow adjacency), and the inter-unit nets that cross pblocks.

**Retry time (12 Oct 2026): resume at place_design, reuse the measurements, no
incremental placement.** A placer retry used to rerun the whole `v++ --link` (system
linking, `link_design`, `opt_design`, then the hook and the placer again). Three levers
were checked on the login node (the FPGA nodes were drained for maintenance), each on the
real path rather than assumed:

1. *Resume at `vpl.impl.place_design`.* Vitis runs the implementation as a Vivado project
   run and `v++ --from_step vpl.impl.place_design` does `reset_runs impl_1 -from_step
   place_design` and reopens the opt checkpoint -- but Vitis writes no intermediate
   checkpoints (`project.writeIntermediateCheckpoints 0`). The link configuration now
   carries `param=project.writeIntermediateCheckpoints=1`, after which the run script
   writes `level0_wrapper_opt.dcp` (and `_placed`, `_physopt`), and the retry script
   (`placer_retry.retry_script`) appends `--from_step vpl.impl.place_design`. Opening the
   checkpoint takes two minutes against five for `link_design` plus two to four for
   `opt_design`. The old path (`--from_step vpl.impl`) is gone; there is no fallback.
2. *Reuse the hook's measurements.* The hook snapshots the measured units next to the
   census (`finn_floorplan_measured.tcl`) and a resumed attempt
   (`FINN_FP_RESUMED_ATTEMPT=1`, exported by the retry script) reads both back instead of
   measuring again (three minutes). Verified in a real v++ resume ("census and measured
   units reused from the first attempt"). Missing files are an error, not a re-measure.
3. *Incremental placement from the previous attempt's placed checkpoint.* Measured and
   dropped. With 59 % of the units changed (E5) it saved nothing. In the best case -- one of
   285 units re-packed, the other 284 pinned, 98 % of cells reused (E6) -- placement took
   1487 s against 2088 s from scratch, but reading the 20 GB reference cost 455 s, so the
   net saving is ~150 s (7 %) for copying a placed checkpoint per attempt plus pinning and
   unplace logic in the hook. Not worth it.

**The placer mode trap found on the way.** Every v++-driven resume failed in the global
placer on a floorplan the first attempt had placed (`30-859`, "BRAM utilization in
constrained region pblock_finn_unit_0 ... 105 %", "DSP utilization ... 711 %"), while an
interactive `reset_runs; launch_runs` on the same checkpoint passed. The vpl child run
script carries twelve `set_param` lines the interactive launch does not; bisecting them
in parallel Vivado sessions on vpl's own opt checkpoint (twelve sessions, 12-30 minutes
each) isolated **`place.ultrathreadsUsed 0`**: Vitis turns ultrathreads placement off
(Vivado's default is 1; the passing runs log `[Place 30-1906] Running placement in
ultrathreads mode`). That placer handles the pblocks when the netlist comes from
`link_design` in the same session (every first attempt so far, including the bitfile
run) but, on a design reopened from the opt checkpoint, counts every DSP and block-RAM
sub-cell as a primitive against the pblock capacities: unit 70 holds 64 DSP48E2, the
checkpoint carries their 512 sub-cells (8 each), and the placer reports 512 / 72 sites =
711 %. Six of six resumed runs with ultrathreads on placed the floorplan; four of four
with it off did not. Two isolation runs with the mode off pin it down: the reopened
checkpoint places without FINN pblocks (1298 s), and with the pblocks but without the
shell-closure prohibits it fails as before -- the prohibits play no part. The two placers are alike on this floorplan (post-place WNS +0.003
vs -0.006 ns, 32 vs 29 min). Decision: only the resumed attempt switches the mode (the
hook sets `place.ultrathreadsUsed 1` when `FINN_FP_RESUMED_ATTEMPT` is set); the first
attempt keeps Vitis' placer, so baselines and first-attempt behaviour are unchanged.
Experiments: `floorplan_validation/tmp/mn_per_node/vitis_link_device_0_18aiv5ol`
(accumulated `vpl/runme.log` with every session) and the session directories under the
scratchpad `retry/params/{a..l,h2,h3}`.

Final check on the test link directory with the full closure map (4,036 sites): stage A,
a full `vpl.impl` under vpl with Vitis' placer, and stage B, the resumed attempt exactly as
FINN runs it (`--from_step vpl.impl.place_design`, `FINN_FP_RESUMED_ATTEMPT=1`, so measurements
reused and ultrathreads on). Both placed, routed and wrote the bitstream; the hook packed
in three rounds in both (8 and 2 pblocks re-packed), and the router's figures are the
same to the last digit:

| | stage A (first attempt) | stage B (resumed) |
|---|---|---|
| before the hook | link 4:45 + opt 1:34 + census/measure ~3 min | open_checkpoint 2:03, measurements reused |
| place_design | 28:49, post-place WNS -0.006 | 26:31, post-place WNS -0.006 |
| route_design | 13:36, WNS/TNS -0.017/-0.132 | 13:09, WNS/TNS -0.017/-0.132 |

A retry now costs about eight minutes less before the placer than the first attempt,
instead of repeating everything in front of it.

**Intra/inter-unit timing split (13 Oct 2026).** `finn floorplan timing-split`
(`floorplanning/timing_split.py` + `finn_floorplan_timing.tcl`) exports the 2,000 worst setup
paths of the kernel clock from a routed checkpoint with their endpoint tiles and the placed
bounding box of every unit, maps the endpoints to units and classifies each path as
intra-unit, inter-unit or other. For the baseline the units are mapped onto the other build's
cell names by aligning the two dataflow orders on op type (`align_cells`): names and tensor
names are renumbered per partition, so neither survives a different partitioning; the
alignment matched 283 of 285 unit cells, the two left over being the FIFOs the floorplanner
inserts at SLR crossings. Coordinates are tile columns/rows (the RPM grid is anisotropic,
~22 units per CLB column against ~2.4 per row). Run on job 3579091's routed checkpoint
(`mn_per_node/vitis_link_device_0_i5xtwwel`) against the baseline
(`mn_baseline/vitis_link_device_0_cv511dvf`); report in
`floorplan_validation/out/timing_split_mobilenetv1/timing_split.md`:

| worst 2,000 paths | floorplanned | baseline |
|---|---|---|
| intra-unit | 617 paths, worst 2.09 ns, mean 2.83, route 6.0 ns, endpoints 273 tiles apart | 85 paths, worst 3.94, mean 4.47, route 5.2 ns, 160 tiles |
| inter-unit | 553 paths, worst 1.91, mean 2.60, route 6.7 ns, 185 tiles | 242 paths, worst 2.80, mean 4.09, route 5.4 ns, 74 tiles |
| other (reset fan-out) | 830 paths, worst 1.54, mean 2.49 | 1,673 paths, worst 2.58, mean 4.24 |
| unit placed aspect ratio | mean 13.0, max 132 | mean 2.7, max 23 |
| unit placed half-perimeter | mean 97 tiles, max 534 | mean 61, max 463 |

What it says:

1. *Shape is the larger lever.* The packer's rows make units 13:1 on average (the placer's
   own are 2.7:1); 575 of the 617 intra-unit paths sit in units with aspect > 8 and their
   endpoints are ~290 tiles apart on average. The worst intra pair is a VVAU whose two ends
   are 263 tiles apart inside its own pblock. Elongation also hurts the inter-unit paths:
   the worst inter pair (DWC 192 -> FIFO 191, 1.909 ns, 218 paths) has its two boxes within
   20 tiles of each other but the connecting registers 117 tiles apart, at the far ends.
2. *Adjacency is the second.* No two floorplanned units touch (gap > 3 tiles everywhere;
   325 inter paths have 100-200 tiles between the boxes, e.g. MVAU 246 -> FIFO 247 at
   2.445 ns), where every baseline pair overlaps or touches and its endpoints are 74 tiles
   apart on average.
3. *The reset net is the worst FINN-related path, and it is not a unit path.* `ap_rst_n`
   fans out from one FDRE (plus a replica) in SLR0's `proc_sys_reset_kernel` to every unit
   and lost 1 ns (2.58 -> 1.54 ns); it will be the first casualty of a wider spread. It is
   outside the floorplanner's objects; replication per SLR or a registered reset tree would
   be the fix if it ever bites.

Sequencing from this: resource-shaped, aspect-bounded rectangles in the packer first (fixes
1 and halves the distance in 2), then dataflow-ordered snake packing so consecutive units
share an edge; both measured with this tool against these numbers. Glue stays constrained
and is not merged into neighbours (decided 13 Oct 2026).

**Compact shapes and dataflow locality in the packer (13 Oct 2026).** Two changes in
`packer.py`, both measured on the four fixtures before being kept:

1. *Half-perimeter objective.* `SlrPacker._search` ranks candidates by tile columns
   spanned plus rows (the longest Manhattan wire a path inside the unit can need), then
   SLICE waste, then bottom-left; the scan prunes any width that cannot beat the best
   half-perimeter and skips heights no shorter than it. Waste-first ranking had taken
   the first zero-waste shape at the lowest height, a strip.
2. *Locality.* In `pack`, every unit is handed the placed rectangle of its nearest
   dataflow predecessor in the same SLR (`_predecessor`); the gap to it is added to the
   half-perimeter (`LOCALITY_WEIGHT` 1: a tile of distance to the neighbour costs as much
   as a tile of width) and the columns around the anchor join the start columns
   (`ANCHOR_REACH`). Dataflow order is the default now (`pack`, `packer.py --order`,
   `render_floorplan_tcl`, the hook); area order stays available.

| fixture (order) | before: placed, aspect mean, half-perim., neighbour gap, adjacent | after: the same |
|---|---|---|
| mobilenetv1_measured (area) | 285/285, 18.2, 114, 174 tiles, 31/282 | 285/285, 3.8, 38, 30, 133/282 |
| mobilenetv1_measured (dataflow) | 285/285, 17.9, 130, 140, 83/282 | 285/285, 3.1, 37, 5.0, 231/282 |
| resnet18_attempt1 (dataflow) | 288/289, 8.7, 83, 76, 121/283 | 289/289, 3.2, 32, 2.6, 235/284 |
| resnet18_measured (dataflow) | 286/289, 14.2, 111, 121, 113/281 | 286/289, 3.4, 33, 3.4, 237/281 |

Two refinements from the first v3 link (13 Oct 2026), whose netlist (reset pipeline) left
unit 266 -- 28 block RAM chains of 4 in SLR1 -- starved in plain dataflow order: units that
carry cascade chains are packed first (most chain rows first: unit 278, eleven chains of 8,
had been left the three untouched block RAM columns of SLR0, 500 tile columns apart, when
it came after LUT-heavier carriers), the rest in dataflow order,
and a unit the start-column shortlist starves is retried from every remaining column before
it is given up. Coverage is back to 285/285 on the v3 units and unchanged on the fixtures;
the neighbour gap grows from 3-6 to 10-15 tiles (75-80 % adjacent). Median LUT over-claim
moved from 1.08-1.36 to 1.16-1.38; packing time fell (the pruning
outweighs the wider search). `summarize` now reports `half_perimeter_mean`,
`aspect_mean` and `aspect_max`. The synthetic test census (twelve SLICE columns, plain
logic limited to the SLICEL ones) cannot show compact shapes, so the two new tests use a
wider one. First link with the new packer (13 Oct 2026, test link directory, reset fix included):
285/285 units in two rounds, LUTs claimed 306k for 176k demanded (the strip packer claimed
285k for the same demand: compact shapes cost ~7 % more area), and the placer failed to
commit 84 instances of two SRL FIFOs (`30-488`, units 73 and 113, both one SLICEM column
by 20-30 rows at 77 % fill). Not a shape rule: 68 units are one tile column wide, 30 of
them SRL FIFOs of 15-30 rows, and eleven units identical to unit 113 (76 LUTs, 18 SRLs,
20 rows) placed. Legalisation noise at high fill, which the retry's growth is for; the
resumed attempt was started with +194 / +124 LUTs.

**Reset per SLR (13 Oct 2026).** v++ wires every kernel's `ap_rst_n` to
`proc_sys_reset_kernel_slr0` whatever SLR the kernel is pinned to (verified in the ULP
block design: partition 2, pinned to SLR1, sat on SLR0's reset; `proc_sys_reset_kernel_slr1`
drove nothing). `BuildBasicVitisLinkConfig._configure_slr_resets` now writes
`finn_reset_per_slr.tcl` and registers it as `compiler.userPostSysLinkOverlayTcl` (the v++
hook that sees the finished block design; the floorplanner cannot use it, the reset can),
which moves each pinned kernel to `proc_sys_reset_kernel_slr<n>`, and adds
`opt_design -hier_fanout_limit 512` so wide control nets are driven from replicas the
placer puts next to their loads. The per-SLR resets release a few cycles apart, which a
FINN dataflow design tolerates: nothing flows between kernels until the host starts them.
First real link (13 Oct 2026, test link directory): "StreamingDataflowPartition_2/ap_rst_n
-> proc_sys_reset_kernel_slr1", the other kernels already right, the option accepted by vpl.
The first attempt of that link failed to commit 156 instances of a small FIFO unit
(`30-488`, the known small-unit class); the retry grew it as FINN's loop does and resumed
at place_design. That resumed attempt, running the ultrathreads placer on the reopened
checkpoint, failed again with only "failed to commit all instances" (Phase 3.6.2 Place
Remaining) and **no instance list**: the ultrathreads placer does not print `30-488`
details, so a resumed attempt that fails leaves `placer_retry` nothing to attribute and
the loop would stop at `FINNSynthesisError`. Open: either the retry resumes at `vpl.impl`
(in-session legacy placer, diagnosable, ~8 min more per retry, no placer-mode switch) or
a resumed attempt that fails without diagnostics is followed by one from `vpl.impl` with
the same overrides. Decision pending the measurements below. Later the same day a resumed attempt failed
with `30-488` *and* its instance list (four cells of a DWC), so the ultrathreads placer
is silent only when its "Place Remaining" phase gives up, not on every commit failure.
(The resumed attempt was
also packed by the new packer -- a new v++ session packs from scratch, the "previous"
mechanism only spans rounds -- so it measured nothing cleanly.)
Second link (new packer + reset fix + fan-out replication, resumed attempt after two small
FIFOs were grown): placed in 25:34, routed in 13:23, router WNS/TNS -0.017/-0.132 as always
(shell path), bitstream written. Timing split against the baseline (worst 2,000 kernel-clock
paths):

| | attempt 8 (strip packer, SLR0 reset) | this run | baseline |
|---|---|---|---|
| intra-unit | 617 paths, worst 2.09 ns, mean 2.83 | none in the worst 2,000 | 85, worst 3.94, mean 4.47 |
| inter-unit | 553 paths, worst 1.91, mean 2.60 | 39 paths, worst 3.75, mean 3.97 | 242, worst 2.80, mean 4.09 |
| reset fan-out | 830 paths, worst 1.54, mean 2.49 | 1,961 paths, worst 1.90, mean 3.37 | 1,673, worst 2.58, mean 4.24 |
| placed aspect ratio, mean / max | 13.0 / 132 | 4.1 / 37 | 2.7 / 23 |
| placed half-perimeter, mean | 97 tiles | 40 | 61 |

The floorplanned design's own paths now beat the unconstrained placer's (its worst
inter-unit pair, DWC 192 -> FIFO 191, went from 1.91 to 3.75 ns). What is left is the reset
tree: with the SLR crossing gone, the worst reset paths run from the SLR0 driver's replica
to units 73-370 tiles away inside SLR0 (1.90 ns), so the reset is now the single limiter
of the kernel's own timing by 1.85 ns, in the baseline too. That is option 3's case (a
registered reset stage per node in the stitched IP, replicated at synthesis so the
placer keeps the replicas inside each unit's pblock). A reset-only run with the committed
packer was tried for attribution and abandoned after four attempts: the first (in-session
placer) lost the same FIFO as before, and each resumed attempt (ultrathreads placer, packed
from scratch with the grown demands) lost one or two *different* tiny units -- DWCs of 8-10
LUTs and 40-46 flip-flops in five-slice pblocks (the programmable-unit minimum) -- 52
instances on the fourth. With the strip packer and the fan-out-replicated netlist the
small units sit at the edge of what detail placement legalises, and a retry that re-packs
everything rolls the dice again. The combined run above placed on its second attempt and
v3 (below) is the measurement that matters; whether `-hier_fanout_limit` earns its netlist
churn once the reset pipeline is in is decided on v3's attempt count. Decided: every v3
run (attempt 1, attempt 2, rebuild, rebuild2, rebuild3 and the constrained re-link, all
with the limit on) placed on its first attempt, so the replicated netlist is not what
pushed the combined run into a retry, and the limit stays. It also keeps the per-SLR
reset nets and the tree stage's remaining loads below one driver per 512 sinks on the
non-pipelined parts of the kernel (DMAs, glue), which the reset pipeline does not cover.

The retry's convergence problem has a fix that is independent of any sizing rule: a
resumed attempt now keeps the previous attempt's placements and re-packs only the units
the placer rejected. The hook copies its last round's result to
`finn_floorplan_pack_final.json`; with `FINN_FP_RESUMED_ATTEMPT` it hands that file to the
packer as `--previous` and the rejected units' accumulated shortfalls as `--feedback`
(`finn_floorplan_feedback_retry.json`), the mechanism a later round already uses within one
attempt, and `vitis_build` writes only the latest rejections (each with its total so far)
into the overrides file. Every unit that placed keeps its rectangle; the dice are rolled only
for the units that failed.

**Reset pipeline per node (option 3, 13 Oct 2026).** `finn-rtllib/reset_pipe` is a packaged
IP (`AMD:user:reset_pipe:1.0`, `reset_pipe.sv`: STAGES registers on ap_rst_n, initialised
asserted, `SHREG_EXTRACT no`, `MAX_FANOUT 256` so synthesis replicates the last stage).
`CreateStitchedIP(reset_pipeline_stages=n)` creates one `<node>_rst` cell per node, makes
the port from the first cell's input and feeds every node from its own cell's output;
`collect_floorplan_units(reset_cells=True)` adds the cell to the node's unit, so the
registers and their replicas are placed inside the node's pblock and every reset path
becomes a short intra-unit one. `DataflowBuildConfig.reset_pipeline_stages` (default 2)
reaches the Vitis kernels (`BuildAllXOs`) and the Zynq kernels (`ZynqBuild`) and the two
floorplan flows; the stitched IP for rtlsim keeps the direct reset, because the XSI
harness drives inputs right after releasing reset and a FIFO in reset still asserts
ready, so a delayed release would drop the first samples there. On hardware nothing
flows until the host starts the kernels. Validation: `out/mobilenetv1_per_node_v3`, a
restart of v2 at `step_prepare_synthesis` on the login node with everything of this
day in (per-SLR resets, fan-out replication, compact/locality packer, reset pipeline):
The first v3 link exposed a bug in `VitisLinkConfiguration.add_vivado_line`
(committed code): its newline logic was inverted, so a second `[vivado]` entry was glued
onto the first. Every real build since the retry work had
`...finn_floorplan.tclparam=project.writeIntermediateCheckpoints=1` as the hook path --
no floorplanning hook installed, no checkpoint -- which the hand-edited test link
directory had hidden. Fixed with a regression test; the v3 link was killed before
placement, its config repaired by hand and the link re-run from `run_link.sh`.
The re-run (first attempt, in-session placer) placed on its first attempt and routed to
the usual -0.017 ns (shell), bitstream written. Its timing split exposed three things:

1. *The per-node stage alone moves the problem, it does not solve it.* The platform's
   reset register in SLR1 now drives the 118 per-node stages of that SLR directly: one net
   over the whole SLR, 9.8 ns of route, **0.25 ns of slack** -- below the 512 fan-out
   threshold of opt_design, so never replicated. Fix: a kernel-level `reset_pipe` stage
   (`ap_rst_n_tree`, STAGES 1, MAX_FANOUT 16) between the port and the per-node stages, so
   synthesis replicates it per 16 loads and the placer puts each replica next to its loads;
   no hop of the reset carries more. `reset_pipeline_stages` now counts the per-node stages
   (default 1; two cycles in total).
2. *Nodes inserted after the floorplan were unconstrained.* `BuildAllXOs` runs
   `InsertFIFO` on every partition when the kernels are built, after
   `step_generate_floorplan_assignment`; the two FIFOs it added had no unit, and one of
   them ended 565 tiles from the layer feeding it (1,096 paths at 1.03 ns, the worst
   non-reset paths). `_units_in_model` now gives such a node a unit of its own, in graph
   order, in the SLR the partition's assigned nodes share.
3. *One unit was starved* (266, the 28-chain MVAU), which is what the chain-first
   dataflow order above fixes; a re-link of the same kernels with it is running.

Re-link of the same kernels with the chain-first packer (`out/mobilenetv1_per_node_v3/
attempt2`): 285/285 constrained in three rounds, placed on the first attempt, routed,
bitstream written. Timing split against the baseline:

| worst 2,000 paths | attempt 8 (strip packer) | this run | baseline |
|---|---|---|---|
| intra-unit | 617, worst 2.09 ns | 834, worst 2.22, mean 4.41 | 85, worst 3.94, mean 4.47 |
| inter-unit | 553, worst 1.91 | 68, worst 3.51, mean 4.43 | 242, worst 2.80, mean 4.09 |
| reset | 830, worst 1.54 | 1,098, worst 2.72, mean 3.75 | 1,673, worst 2.58, mean 4.24 |
| unit aspect ratio, mean / max | 13.0 / 132 | 3.0 / 30 | 2.7 / 23 |
| unit half-perimeter, mean / max | 97 / 534 | 37 / 271 | 61 / 463 |

The floorplanned kernel's rectangles are now as compact as the free placer's, 51 of the
68 inter-unit paths run between adjacent units, and every class but one is at or above the
baseline. The one is a single small block RAM FIFO (unit 237: 84 LUTs, 2 RAMB36, a
15-slice pblock with its block RAM range right beside the slices): 51 paths from the
FWFT valid register to the RAM's pins, 4 logic levels, 7 ns of route for endpoints 8
tiles apart. Not a shape problem -- a routing detour at that spot, to be looked at in the
routed checkpoint; the free placer has the same FIFO at 3.94 ns. The reset's worst
paths are still the single-load nets from the platform register to each unit's stage
(2.7 ns here, 0.25 ns in attempt 1: luck of placement); the tree stage of the full rebuild
bounds them.

Full rebuild with all of this (`out/mobilenetv1_per_node_v3/rebuild`, 13 Oct 2026, 20:21-23:35
on the login node, kernels regenerated with tree and per-node stages, link through FINN's
own retry loop): 287 units (the two late FIFOs included), placed on the first attempt,
routed, `finn-accel-0.xclbin` written. Timing split:

| worst 2,000 paths | this rebuild | baseline |
|---|---|---|
| intra-unit | 1,791 paths, worst 3.64 ns, mean 4.78 | 85, worst 3.94, mean 4.47 |
| inter-unit | 107 paths, worst 3.90, mean 4.70 | 242, worst 2.80, mean 4.09 |
| reset | 102 paths, worst 0.42, mean 3.97 | 1,673, worst 2.58, mean 4.24 |
| unit aspect ratio, mean / max | 3.4 / 75 | 2.7 / 23 |
| unit half-perimeter, mean / max | 38 / 280 | 61 / 463 |

The kernel's own paths are where they should be (the worst inter-unit pair, FIFO 274 ->
FIFO 275, is the SLR crossing between partitions 2 and 3, 572 tiles by construction). The
reset tree did not do its job: a tree replica's 16 loads lie anywhere in the SLR, it settles
at their centroid, and its farthest net (to a unit's stage 377-430 tiles away) routed in
9.3 ns -- bounded fan-out does not bound distance. Fix: the per-node `reset_pipe` cells are
no longer part of the pblocks and carry two stages (`reset_pipeline_stages` default 2),
so the placer strings a node's registers out between the tree replica and the node and no
hop is longer than a third of the way; the replicas of the last stage still land next to
their loads, as any unconstrained driver of constrained loads does. Rebuilt overnight
(`out/mobilenetv1_per_node_v3/rebuild2`, 23:46-02:56, first attempt placed, routed, bitfile):

| worst 2,000 paths | rebuild2 (unconstrained 2-stage pipes + tree) | baseline |
|---|---|---|
| intra-unit | 1,741 paths, worst 2.65 ns, mean 4.76 | 85, worst 3.94, mean 4.47 |
| inter-unit | 65 paths, worst 3.78 (the SLR crossing), mean 4.53 | 242, worst 2.80, mean 4.09 |
| reset | 194 paths, worst 1.47, mean 4.49 | 1,673, worst 2.58, mean 4.24 |

The reset hop from the tree replica to a node's first register is still 302 tiles and
8.5 ns: the placer put both unconstrained registers next to the node. It is timing-driven
-- it moves cells until timing is met, and at 1.47 ns of positive slack it is met -- so an
unconstrained register chain does not get strung out on its own. Across every variant of
the day the reset class was: 1.54 ns (attempt 8, SLR0 reset for all), **1.90 ns (per-SLR
resets + opt_design replication, no pipeline)**, 0.42 ns (pipes inside the pblocks + tree),
1.47 ns (pipes unconstrained + tree). Options 1 and 2 alone did best; the pipeline as built
helps nothing unless the tools are told to keep the hops short. The one principled way to
do that is a `set_max_delay -datapath_only` on every hop from a `reset_pipe` register in the
kernel's implementation XDC (half the clock period, a requirement the placer and router then
satisfy, not a sizing guess). The first rebuild with it (`rebuild3`, 03:05-06:19) came out
*identical* to `rebuild2` in every number: the XDC was parsed but its `get_cells` pattern
assumed the registers sit directly under the IP cell, while they sit one level down
(``<cell>/inst/genChain.Pipe_reg[0]``), so it matched nothing -- Vivado is deterministic,
so the run merely repeated the previous one. Pattern fixed (``/.*Pipe_reg.*``, tested against
the real names and a replica's), and the constraint applied link-side in the same link
directory for a quick measurement (`rebuild3_constrained`, re-link from `vpl.impl`): 580
reset registers constrained, placed on the first attempt, routed, bitstream. Measuring it
needed two fixes in the timing split: Vivado reports a path's slack against its tightest
requirement, and with 580 constrained registers the reset hops filled every export, so the
exporter now takes the clock-period paths from every sequential cell except the reset
registers and the tighter group separately, and the analysis restates all slacks against
the period. Result (worst paths, slack against the 10 ns period):

| | rebuild3_constrained | baseline |
|---|---|---|
| intra-unit | 1,046 paths, worst 3.90 ns, mean 4.78 | 85, worst 3.94, mean 4.47 |
| inter-unit | 61 paths, worst 4.10 (the SLR crossing), mean 4.57 | 242, worst 2.80, mean 4.09 |
| reset hops | worst 5.49 (longest hop 4.45 ns of route, within its 5 ns) | 2.58 |
| other | worst 4.58 (HBM interconnect into the input DMA) | -- |

**The reset is no longer the limiter.** The kernel's worst path is an ordinary intra-unit
path of an MVAU at the level of the free placer's worst, and every class is at or above the
baseline. Decision: `reset_pipeline_stages` stays at 2, with the tree stage and the hop
constraint in the kernel XDC (pattern fixed); options 1 and 2 stay as well.

**The kernel XDC was never read (found 14 Oct 2026, `rebuild4`).** The confirming build
through the production path showed in its link log, for every kernel:
`CRITICAL WARNING: [Designutils 20-1275] Could not find cell 'inst' within module
'ulp_StreamingDataflowPartition_1_0' ... The XDC file ... will not be read for this
cell.` The packaged kernel is a DCP-based IP; at link time Vivado fills the kernel cell
with the checkpoint, whose top holds the block design as `<block>_i`, while the IP
integrator scopes an IP's constraint file to a cell `inst` below the IP cell by default.
Every Alveo link so far, baseline included, carries the warning: FINN's kernel XDC has
been dead weight all along (its content, XPM false paths, is applied by XPM's own
scoped constraints anyway), and rebuild3 was identical to rebuild2 for this reason as
much as for the pattern. The first measurement of the constraint
(`rebuild3_constrained`) applied it from the link-side hook and so was unaffected.

Measured on the packaged IODMA and partition-3 kernels, each instantiated in a block
design and synthesized (`out/xdc_scope/v0..v5`, ~3 min each):

| packaging | result |
|---|---|
| as packaged (v0) | `20-1275`, file dropped, hop requirement 10 ns |
| no XDC file, constraint written into the DCP (v1) | Vivado reads only the DCP's in-context XDC ("Using original IP XDC constraints instead of the XDC constraints in dcp"); not applied |
| `scoped_to_ref <block>_wrapper` (v2) | `20-1281` module not found: the block design renames the kernel's top |
| `scoped_to_cells <block>_i` (v3) and `scoped_to_ref <block>` (v4) | file read, scoped to `<cell>/<block>_i`; but `20-1307`: `if` is not allowed in an IP's XDC, so the guarded constraint still did nothing |
| v3 + `write_xdc -cell <block>_i` + a plain `set_max_delay` line (v5) | read, no warnings, reset hops at requirement 5.0 ns |

Fix: `kernel_xdc_export_tcl` writes the XDC relative to `<block>_i`,
`kernel_xdc_package_tcl` packages it with `scoped_to_cells <block>_i`, and
`reset_hop_constraints` emits one plain `set_max_delay` line (`-quiet` on the cell
query covers a kernel without reset registers). `rebuild4` was stopped in routing (it
could only reproduce rebuild2); `rebuild5` is the confirming build through the
production path.

**rebuild5 (14 Oct 2026, production path).** Every kernel XDC parsed scoped to its
`<block>_i` cell, no `20-1275`/`20-1307`, 1,672 reset-hop paths under the 5 ns
requirement, placed on the first attempt, bitstream. Slack against the period:

| | rebuild5 | rebuild3_constrained | old baseline |
|---|---|---|---|
| intra-unit | 1,076 paths, worst **2.98** | 3.90 | 3.94 |
| inter-unit | 80 paths, worst 4.19 | 4.10 | 2.80 |
| other (incl. reset) | worst 6.67 | 4.58 | 2.58 |

The reset class is gone from the top of the list for good; the worst path is now an
intra-unit net of `pblock_finn_unit_70` (partition 1's `MVAU_rtl_3`, 71 DSPs, 17 BRAMs,
10k FFs): the weight stream's output pointer `OPtr_reg[1]` driving its 1,027 output data
registers, spread over 157 rows of the unit's 180-row, 8-slice-column strip (net probe,
`out/net_probe/`). Same net, same pblock in rebuild3_constrained: 4.9 ns of route and
4.91 slack against 6.8 ns and 2.98 here -- the placer's luck within a strip decides a
±1.5 ns swing on one net. Three measurements on where the strip comes from:

- The grid is isotropic in tile units: single-level near-vertical paths cost a median
  0.040 ns per tile row, near-horizontal ones 0.039 ns per tile column (817 paths from
  three exports). The half-perimeter objective needs no row weighting.
- Replaying the packing (`replay_shapes.py`): with the predecessor anchor, unit 70 had a
  90-row box of half-perimeter 137 available (gap 140 to its predecessor) and took the
  180-row strip of half-perimeter 212 (gap 50) because `LOCALITY_WEIGHT = 1.0` counts a
  tile of gap like a tile of shape: 277 against 262. Units 150, 191 and 267 (the other
  71-DSP MVAUs) made the same trade; unit 211 got 75 x 137.
- The gap does not show in the inter-unit class: rebuild5's inter paths are at 4.36 ns
  worst for gaps of 0-20 tiles and 4.19 for the SLR crossings beyond 200 tiles, while
  intra-unit is the limiter. A locality-weight sweep on all four fixtures plus
  rebuild5's units (`locality_sweep.py`): 0.5 cuts the units taller than 90 rows from
  8-11 to 3-9 per design and the mean half-perimeter by 12-16 %, for 3-8 tiles more
  mean gap (median stays 0) and 4-10 points fewer touching pairs; every fixture places
  the same number of units. 0.25 gains little more shape and loses more locality.

Decision: `LOCALITY_WEIGHT = 0.5`, measured on `rebuild6`. Left open: the net itself is a
fan-out-1,027 pointer in `memstream`; a `MAX_FANOUT` on that register in finn-rtllib would
bound it in any shape and in the baseline too, but that is an RTL-library change outside
the floorplanner.

**rebuild6 (14 Oct 2026, `LOCALITY_WEIGHT = 0.5`, cluster job).** Placed on the second
attempt (a 5-slice FIFO short of 75 FFs on the first, the known legalisation noise),
bitstream. The shapes did what the sweep predicted: units taller than 90 rows 10 -> 3,
mean half-perimeter 41 -> 35, unit 70 a 55 x 45 box instead of 8 x 180, and its worst
intra path went from 2.98 to 3.41 ns (unit 171, 570 paths). But the design's worst path
is 0.97 ns: `o_v_reg` of FIFO unit 220 to its own address logic *through* the
ConvolutionInputGenerator it feeds (unit 221) -- the consumer's combinational ready
answering the FIFO's valid, 4.0 ns out and 3.9 ns back across 597 tiles, with 5 LUT
levels. The split had called it intra-unit because both endpoints sit in unit 220. Two
consequences:

- The exporter now records the cells a path passes through and `classify` calls a path
  that leaves a unit and comes back through another inter-unit, reported as the pair
  (`via`, `round_trips` per class). Re-exported: rebuild6 has 281 inter paths of which
  74 are round trips in 2 pairs (220->221 at 597 tiles: 0.97 ns; 238->239 at 244 tiles:
  4.33 ns); rebuild5 had one pair (220->221 at 267 tiles: 4.70 ns). Registered streams
  are unaffected by distance at this period: plain inter pairs at 597 tiles still have
  4.1 ns. The round trip is the one inter-unit path class that distance breaks, and only
  beyond roughly 300 tiles.
- Why 220 was 597 tiles from 221: chain carriers are packed first (`order_units`), so
  the window buffer 221 was placed before the run of small units 212-220, which then
  strung along from 211 anchored to their predecessors only; 220 landed next to 219 at
  the far left. Packing replay: at every weight the candidate next to 219 had gap 0 and
  nothing pulled towards 221. Fix: `_neighbours` anchors a unit to its predecessor *and*
  to the unit directly after it when that is already placed, and the cost charges the
  widest gap rather than the sum -- the sum is constant anywhere on the line between
  two anchors and never pulls a unit between them. Replay of rebuild6's units: 220 sits
  between 219 and 221 (301 and 307 tiles) and every other statistic matches rebuild6.
  Measured alternatives on the fixtures: anchoring to the nearest later placed unit
  pulled whole runs towards the next carrier (+20 % half-perimeter, tall units x1.5-2.6,
  touching pairs 52 -> 33 %) and was dropped.

`rebuild7` measures the anchoring change.

**Fair comparison (14 Oct 2026, baseline_v2 = unfloorplanned with the reset pipeline,
per-SLR resets and fan-out replication; 285 of 287 units aligned).** Slack against the
10 ns period, worst path per class:

| | baseline_v2 | rebuild5 (w=1.0) | rebuild6 (w=0.5) | rebuild7 (w=0.5, both anchors) |
|---|---|---|---|---|
| intra-unit | 3.30 (CIG unit 181, 898 paths) | 2.98 (MVAU unit 70, pointer net in a strip) | 3.41 (MVAU unit 171) | 3.65 (MVAU unit 280, the 180-row chain carrier) |
| inter-unit | 4.30 (DWC -> FIFO) | 4.19 | **0.97** (the 220->221 round trip) | 4.37 (220->221 again, now 26 paths at ~300 tiles) |
| other | 7.12 | 6.67 | 4.74 | 6.16 |
| design worst | **3.30** | **2.98** | **0.97** | **3.65** |

rebuild7 (15 Oct 2026, cluster job 3678671, placed on the first attempt) is the first
floorplanned design that beats the fair baseline, by 0.35 ns: the anchoring change put
unit 220 between 219 and 221 as the replay predicted and the round trip went from 0.97 to
4.37 ns, while the strips of rebuild5 stay gone. Its limiter is unit 280, the 125-BRAM
MVAU whose 28 chains force 180 rows -- a shape the census dictates, not the objective.

With the same reset handling on both sides the old baseline's 2.58 ns reset path is gone
and the free placer's design closes at 3.30 ns. rebuild5 and rebuild6 were at that level
or below it, each with one shape-induced worst path of its own (a strip's fan-out net, a
far round trip); rebuild7 removes both and comes out 0.35 ns ahead. At this period every
design has 3-4 ns to spare, so the margin is small against the run-to-run swing and the
decisive comparison is the 5 ns pair below.

Implementation runtimes from the Vitis logs (same nodes, cluster jobs) give the
floorplanner no edge either:

| | opt | place | phys_opt | route | bitstream |
|---|---|---|---|---|---|
| baseline_v2 | 1:40 | 25:18 | 0:50 | 16:21 | 6:57 |
| rebuild5 | 1:50 | 29:45 | 0:41 | 16:23 | 6:19 |
| rebuild6 | 0:53 (x2) | 27:53 (second attempt) | 0:45 | 16:14 | 6:50 |

Every one of these designs meets 10 ns with 3-4 ns to spare, and a timing-driven placer
stops improving a path once it meets its requirement, so the worst-slack comparison at
this period measures placer luck as much as floorplan quality (the same net in the same
pblock swung 4.9 -> 3.0 ns between rebuild3_constrained and rebuild5). A comparison that
can show a floorplanning benefit needs the design to be timing-limited: the same network
built at a period the free placer cannot close (baseline_v2's worst path closes at about
6.7 ns), or a network that fills the device. That is a full rebuild from HLS at a new
`synth_clk_period_ns`, i.e. a decision for the user, not a checkpoint restart.

**5 ns pair (15 Oct 2026, decided by Linus: "5 ns worked in the past for the floorplanned
design").** `cfg/mobilenetv1_per_node_5ns.yml` and `cfg/mobilenetv1_baseline_5ns.yml`:
the same network, folding, FIFO depths (cycle counts, unchanged) and reset handling, built
from HLS onwards at `synth_clk_period_ns: 5.0` into `out/mobilenetv1_{per_node,baseline}_5ns`
(build dirs `tmp/mn_{per_node,baseline}_5ns`; jobs 3690522 and 3690524, 12 h).

Both built in under three hours and both produced a bitfile -- but neither met 5 ns. Vitis
implements against the requested clock and, when the kernel clock fails, scales it down
(`a.xclbin.info`: "Requested Freq: 200 MHz, Achieved Freq: ..."; the warning "One or more
timing paths failed timing targeting 200 MHz ... automatically changed to" in
`vpl/vivado.log`). The routed checkpoints therefore carry the *scaled* constraint, which
is why a first look at their timing summaries showed 10 ns and positive slack; the only
2.5 ns requirements in them are FINN's own reset-hop constraints (period/2 from the kernel
XDC), and those are the worst paths of the scaled designs (0.37 / 0.54 ns).

| | baseline_5ns | per_node_5ns |
|---|---|---|
| requested | 200 MHz | 200 MHz |
| achieved after scaling | 105.7 MHz | 103.8 MHz |
| placement | first attempt | first attempt |

So the 10 ns builds' 3-4 ns of slack were not headroom to 200 MHz: both designs land near
100 MHz when asked for 200, floorplanned and free alike. The per-class picture at the 5 ns
target comes from re-constraining both routed checkpoints at 5 ns
(`net_probe/reclock_5ns.tcl`, `timing_split --clock clk_kernel_5ns`, out
`per_node_5ns/timing_split_5ns`):

| at the 5 ns target | baseline_5ns | per_node_5ns |
|---|---|---|
| worst kernel path (reset hops excluded) | -1.014 ns (MVAU unit 280 weight stream, 3 levels) | -0.758 ns (CIG unit 161, 6 levels) |
| failing endpoints, whole kernel clock | 1,737 | 295 |
| intra-unit | 1,668 paths, worst -1.01, mean -0.04 | 1,850 paths, worst -0.76, mean +0.27 |
| inter-unit | 332 paths, worst -1.01 (DWC -> FIFO), mean -0.13 | 104 paths, worst -0.52 (the 220->221 round trip at 300 tiles), mean +0.10 |
| failing paths in the 2,000-path export | 1,562 in 13 units (1,024 in MVAU unit 70) | 254 in 12 units (91 in MVAU unit 171) |

**This is the first measurement where the floorplan makes a clear difference: at 5 ns the
floorplanned design misses by 0.76 ns against 1.01 and has a sixth of the failing
endpoints.** The free placer's failures concentrate in the big MVAUs (unit 70 alone holds
1,024 of its 1,562 failing export paths, unit 280 its worst) -- the units the per-layer
pblocks keep compact. What the floorplanned design still misses:

- Logic-bound paths: its worst intra paths are 6-level window-buffer address paths in two
  ConvolutionInputGenerators (units 161, 140) and 2-level MVAU paths (unit 70); 100 of its
  254 failing export paths have 5-9 logic levels. No placement fixes those; they are the
  RTL at 200 MHz.
- The 220->221 round trip again (-0.52): 300 tiles is fine at 10 ns and not at 5. The gap a
  round trip tolerates scales with the period, which the packer does not know. Giving the
  packer the clock period and deriving the gap weight from the measured 0.04 ns/tile
  would make that one principled; left open.

**Two systematic fixes (15 Oct 2026, decided by Linus: systematic only, most important
first).**

1. *Stream register slices* (`DataflowBuildConfig.stream_register_slices`,
   `transformation/fpgadataflow/stream_slices.py`): a fully registered AXI-stream
   register slice (Vivado's `axis_register_slice`, `REG_CONFIG 8`) in front of every
   node input another node drives, FIFOs excepted (their input ready is a register:
   `Q_srl.v` drives `i_r` from `i_b_reg`). The slice's ready is a register, so the
   consumer's combinational ready -- the round trip that distance breaks -- ends inside
   the consumer's unit, where the binding puts the slice (`<node>_in<i>_slice` joins the
   unit's cells). Throughput unchanged, one cycle of latency per slice, two flip-flops
   per stream bit: 15.3 k FFs on mobilenetv1 (+9 %), measured from the model's input
   stream widths (7,653 bits at non-FIFO inputs). Hardware kernels only, like the reset
   pipeline; Vitis and Zynq.
2. *Implementation strategy harmonised* (`VitisOptStrategy`): the Vivado directives the
   link gets and the kernels' synthesis options now both follow the strategy
   (`implementation_directives`, `kernel_synth_options`). PERFORMANCE is the `Explore`
   set with physical optimisation (what PERFORMANCE_BEST used to be); PERFORMANCE_BEST is
   `ExtraTimingOpt` placement, `AggressiveExplore` physical optimisation and routing, and
   post-route physical optimisation; both synthesise the kernels with `-retiming`
   (`CreateStitchedIP(synth_options=...)`), aimed at the 5-9 level paths no placement can
   shorten. The placer retry resumes the same run, so the directives survive it. First
   attempt (jobs 3694231-3694234): the floorplanned link died in opt_design -- Vivado
   rejects `opt_design -directive ExploreWithRemap` next to the per-SLR reset handling's
   `-hier_fanout_limit 512` ("Cannot specify additional options with '-directive'"); the
   unfloorplanned link, whose kernels are not SLR-pinned and get no fan-out limit, ran
   `place_design -directive ExtraTimingOpt -retiming` (the `-retiming` is v++ `-O3`'s own)
   without complaint. The opt_design directive is dropped from both strategies; the
   fan-out limit stays.

Measured on the 5 ns pair, each side with the same settings, re-constrained at 5 ns
(`collect_5ns_pair.sh`, out `<per_node variant>/timing_split_5ns`):

| at the 5 ns target | default strategy | PERFORMANCE_BEST |
|---|---|---|
| baseline: worst kernel path / failing endpoints | -1.01 ns / 1,737 | -1.20 ns / 1,377 |
| floorplanned: worst kernel path / failing endpoints | -0.76 ns / 295 | -1.74 ns / 717 |
| floorplanned intra mean / inter mean | +0.27 / +0.10 | +0.09 / +0.08 |
| route_design | 16 min | 31 min |

**PERFORMANCE_BEST loses on both sides** (jobs 3695119/3695121, first attempt each). Its
floorplanned worst path is a one-level, 889-path fan-out net in MVAU unit 171 at -1.74 ns
-- the same class of pointer net that was 3.0 ns at 10 ns in rebuild5 -- and the baseline
drops from -1.01 to -1.20. Aggressive directives plus retiming (synthesis via the strategy,
placement via v++ `-O3`) spread the long nets of the big MVAUs further than the default
flow does; neither side gains anything on the 5-9 level logic paths either. The mapping
stays as a documented option; the default strategy stays the default, and every later
pair is measured under it. Round trips: with the same strategy on both pairs, the slice
pair (`*_5ns_perfslices`) below isolates the slices; the `*_5ns_slices` pair measures them
under the default strategy.

*PERFORMANCE_BEST + slices* (jobs 3695120/3695122, first attempt each, 141 slices in the
floorplanned design's units, measured FF demand 162.6 k -> 181.0 k): worse again on both
sides -- baseline -1.57 ns with every one of its 2,000 exported paths failing (2,402
endpoints), floorplanned -2.87 ns (1,755 endpoints). The round trips are gone as intended
(0 in both designs; the 220->221 pair no longer appears), but under this strategy the
rest moved: the floorplanned worst is a zero-level 7.3 ns net inside the 180-row chain
carrier (unit 280), and its inter class is now the SLR-crossing FIFO pair 274<->275 at 639
tiles (-1.33 ns), a registered stream that was 4.1 ns at 10 ns. The aggressive flow's
run-to-run spread swamps the slice effect; the default-strategy pair is the measurement.

That crossing pair exposed a third systematic gap: the SLRs are packed one after the
other by separate packers and `_neighbours` only looked inside the unit's own SLR, so
the two FIFOs of a crossing never saw each other -- in rebuild7's packing unit 274 sat
180 rows above the SLR boundary and 275 65 rows below it (dy 240, dx 32). Fix: the
neighbour lookup spans SLRs; `_gap` is Manhattan distance, so the crossing unit is pulled
to the boundary rows and the partner's columns. Replay of rebuild7's units: 274->275 dy 240
-> 90, dx 32 -> 2; mean half-perimeter 34.8 -> 34.4, tall units 3 -> 3; the other crossing
(155->156) keeps dy 65. Measured by `per_node_5ns_slices_xslr` (default strategy, slices,
cross-SLR anchors) against `baseline_5ns_slices`.

Both default-strategy floorplanned slice builds (`per_node_5ns_slices`, jobs 3696332 and
3696370) died in the placer with `[Place 30-484] The packing of LUTRAM/SRL instances into
capable slices could not be obeyed`: twelve RAM64M8 of the first window buffer did not fit
`pblock_finn_unit_2`, whose 64 memory macros the packer had sized at exactly the four
SLICEM columns it got (80 slices, 15 spare for everything else). The same unit, same
pblock and same demand placed under `ExtraTimingOpt`; the default placer's LUTRAM packer
did not find the arrangement. That message was not one the retry understood, so the build
stopped instead of growing the unit; `placer_retry` now parses it (attribution through the
unit's cells, demand grown like a `30-1112`: `{'pblock_finn_unit_2': {'lutrams': 288}}` on
that log). Both builds were resubmitted in full at 21:42 (jobs 3697315 and 3697316, same
configs and directories; the relink variants `*_relink.yml` submitted in parallel were
cancelled before they wrote anything, so the full rebuilds are the measurement). Both
rebuilds hit the same `30-484` on unit 2 again on their first attempt; the retry grew its
memory-LUT demand by 288, and the second attempt placed (30 min) and routed (16 min) to a
bitfile in each (links `tv_0l6fx` and `a478pkj6`). So the default placer's LUTRAM packer
fails this unit reproducibly at the packer's sizing, and the retry is what carries it --
a measured case for the memory-slice rule (`ram_macros / 0.8`) being too tight for
RAM64M8-heavy window buffers, to be checked against more such units before changing it.

*Default strategy + slices, at the 5 ns target* (`per_node_5ns_slices` -- the `_xslr`
rebuild is the identical design, both were rebuilt from the same worktree with the
cross-SLR anchoring -- against `baseline_5ns_slices`):

| at 5 ns | no slices | with slices |
|---|---|---|
| baseline: worst / failing endpoints / round trips | -1.01 / 1,737 / 0 | -1.61 (CIG unit 140, 8 levels) / 1,657 / 0 |
| floorplanned: worst / failing endpoints / round trips | -0.76 / 295 / 2 pairs | -1.75 (MVAU unit 267, 2 levels, 878 paths) / 1,151 / 0 |
| floorplanned inter-unit worst | -0.52 (the 220->221 round trip) | +0.04 (8 paths in the export) |

The slices do what they were built for: no round trip remains in either design and the
floorplanned inter-unit class is clear. They do not buy slack. Both designs got worse on
the worst path, and the floorplanned design's limiter moved to the other 180-row chain
carrier (unit 267, 126 BRAMs: 878 two-level paths at -1.75) while the baseline's is the
8-level window-buffer logic of unit 140 (960 of its 1,596 failing export paths have 8
levels). The floorplanned design still has a third of the baseline's failing endpoints.
Decision: `stream_register_slices` stays off by default -- 11 % more flip-flops for a
class that, at this period, is not what limits either design -- and remains the switch
for designs whose inter-unit paths are the limiter. The floorplanned design's recurring
limiter across every 5 ns run is now one class: one- and two-level nets inside the big
weight-streaming MVAUs (units 70, 171, 267, 280 in turn), a write-data or pointer register
driving the whole BRAM array of a 90- to 180-row unit.

**Fourth systematic fix: fan-out replication of the memory command registers (16 Oct
2026).** The nets are `axilite.sv`'s command registers (`IpEn`, `IpWe`, `IpAddr`,
`IpWData`/`IpData`: each drives every RAMB36 of the memory, 125 in units 267/280) and
`memstream.sv`'s pointers (`Ptr`: the read address of every block RAM; `OPtr`: the SRL
tap select of every output bit, 1,027 loads). Vivado's own replication only touches nets
that are already critical, so inside a pblock they were strung across the unit (6.1-7.3
ns of route). They now carry `MAX_FANOUT` attributes (32 for the command registers, 64 for
`OPtr`), which makes synthesis replicate them and lets the placer put each replica next to
its share of the array -- the same mechanism as `reset_pipe`'s tree stage. Simulation is
unaffected (attributes), the cost is a few hundred flip-flops per large MVAU, and it
applies to free and floorplanned designs alike. Measured by the `*_5ns_fanout` pair
(default strategy, no slices, both sides rebuilt from HLS).

*A fourth rejection class.* Both default-strategy floorplanned slice builds (jobs
3696332, 3696370) died in placement with `[Place 30-484] The packing of LUTRAM/SRL
instances into capable slices could not be obeyed`: the design's 38,522 LUTRAM/SRL
primitives need 68,086 of the device's 73,370 SLICEM (93 %), and twelve RAM64M8 of one
window buffer (partition 1's first ConvolutionInputGenerator, unit 2) found no SLICEM
inside its pblock. The same netlist placed under PERFORMANCE_BEST, so this is the
legalisation margin of one unit, not a capacity problem -- exactly what the convergent
retry is for, but `30-484` was not among the codes it parsed, so the builds simply
stopped. `placer_retry` now reads the primitive list of a `30-484`, attributes each
primitive to its unit and raises the unit's memory-LUT demand past its pblock's capacity
(at least by the LUTs the primitives occupy, eight per RAM64M8), like a `30-488`. Both
builds resubmitted (jobs 3697315, 3697316).

Vitis' own "achieved" frequencies (105.7 vs 103.8 MHz) do not follow the measured slack
(they would put the free placer ahead); they are what its scaler could configure on the
clock wizard, not a timing measurement, and are not used here.

**Fair baseline (`mobilenetv1_baseline_v2`, started 14 Oct 2026 15:33).** The old
baseline predates the reset pipeline and the per-SLR resets, which apply to unfloorplanned
Alveo builds as well, so its 2.58 ns reset path is not the floorplanner's gain. baseline_v2
rebuilds the unfloorplanned design from its `step_prepare_synthesis` checkpoint with the
same reset handling; the floorplan is judged against it.

**Validation focus (decided 2 Oct 2026): mobilenetv1 first.** resnet18-w3a3 is parked --
its no-floorplan baseline does not route in this setup either, and its memory footprint is
the §7.1/3 decision -- so every link run until further notice is a mobilenetv1 run.

**Sequencing.**

0. Let the two runs in flight on 2 Oct 2026 finish (resnet18 restart, clock-region engine;
   mobilenetv1 restart, multi-chain engine) and read two numbers off them: the first placer
   verdict with every known cascade rule satisfied, and the LUTRAM packing density
   (`Place 30-484`) at per-layer granularity — an earlier resnet18 run reported 703 % of
   SLICEM capability. If that stands, no packer makes per-layer hard pblocks placeable for
   resnet18 and the granularity constraint goes back to the user. Steps 1-3 do not wait for
   these runs; they are offline work on the preserved attempts.
1. Export switch in the current Tcl: census + demands + chains → JSON. Capture fixtures from
   the next link run. Cheap, useful immediately, independent of the rewrite.
2. Packer module with tests on the fixtures. Acceptance on the attempt-1 fixture: every unit
   placed, median claimed/ideal SLICE area < 1.5 (die-ratio-bound units excluded and listed),
   DSP-heavy units narrower than half the SLR, every BRAM chain inside one band.
3. Thin Tcl shell, `exec` plumbing, interpreter/environment baking in `tcl.py`, the numpy
   proof line. Delete the shelf code, `FINN_FP_FIT_GROWTH`, `FINN_FP_MAX_FIT_ATTEMPTS`, the
   discarded `IS_SOFT TRUE`, and the silent unconstrained fallback.
4. One link run per model, compared against attempt 1 on: constrained count, claimed vs
   demand per kind, placer outcome on the first try, then Milestone 5's QoR comparison.
5. Overlap and SLR-from-measurement only after 4 is green, each with its own link run.

**Test tiers (extends §8).** Packer: `quick`, pure Python on JSON fixtures. Tcl shell and
`exec` plumbing: `vivado` tier (needs a part for the census). Link runs: manual, as Milestone 5.

**Success criteria.** 100 % of units constrained on mobilenetv1 and resnet18 at `per_node`;
`place_design` passes on the first attempt (no `30-99`/`18-5197`/`30-859` retries); LUT
over-claim at or below 2× median outside die-ratio-bound units; QoR against the
no-floorplanning baseline reported as in Milestone 5.

## 7. Open questions for iteration

### 7.1 Remaining

Milestone 6 (§6) needs three decisions; everything else in the plan can proceed on its
stated default.

1. **Overlapping pblocks** — allow LUT-only units to nest inside DSP/BRAM-heavy hosts? The
   only lever against the die-ratio waste (69 % of resnet18's over-claim); legal in
   principle (§6 Milestone 6, extensions); needs one empirical link run and shared-capacity
   budgeting in the packer and the verify step.
2. **SLR assignment from measurement** — drop the v++ `slr=` directive and let the hook
   pick SLRs post-`opt_design`? Closes the zero-estimate LP gap; interacts with `Floorplan`'s
   SLR-crossing FIFO insertion and the multi-FPGA anchors.
3. **resnet18's memory LUTs exceed what per-layer hard pblocks can hold at a 0.8 SLICEM
   ceiling -- decision needed.** Measured on the real (shell-clipped) census, 2 Oct 2026:
   memory LUTs are 96 % / 103 % / 95 % of the three SLRs' SLICEM ceilings and 98 % of the
   device's. No assignment and no packer reaches that per unit. The options, none of which
   the floorplanner can take alone: raise `floorplan_slicem_utilization` toward 0.9-1.0 and
   let a run show where the legaliser stops (it failed at ~100 % in attempt 2); change the
   folding or `ram_style` of the three distributed-RAM MVAUs (the configs were to stay);
   or group LUTRAM-heavy layers with neighbours so one region is sized for several units'
   memory (the `glue` granularity, or a new memory-aware grouping). mobilenetv1's budget is
   measured by its running job and should be read before deciding.

4. ~~A ceiling for SLICEM-bound LUTs (LUTRAM + SRL).~~ **Decided 2 Oct 2026: both.**
   `DataflowBuildConfig.floorplan_slicem_utilization` (default 0.8, Vivado's pblock
   guidance) derates the SLICEM LUTs a rectangle offers to LUTRAM/SRL demand, in the packer
   and in the shell's verify step alike; and `placer_retry` now understands
   `[Place 30-1112]`, maps the quoted SLICE range back to its pblock through
   `finn_floorplan.json`, and raises that unit's memory-LUT demand by a quarter of what its
   pblock offered (`LUTRAM_RETRY_GROWTH`) for the resumed implementation. The ceiling
   prevents, the retry repairs. The 0.8 is to be checked against the measured `srls` of the
   unit that failed (its pblock offered 1320 SLICEM LUTs); the new engine exports that number
   in `finn_floorplan_units.json`.

Verified on the bare part: the conda interpreter starts under Vivado with the scrubbed, baked
environment, and the census / exec / apply / verify path runs end to end (124 tests).

### 7.2 Resolved

- [x] ~~§3.1 hedged that the Level-1 LP assigns "each node — or each Level-2 placement unit
      once granularity is chosen" — which order actually runs, and does it matter?~~ It
      matters, and the hedge was a real gap: §5.1 had (incorrectly) hard-coded Level-1
      assignment *before* Level-2 grouping, which meant nothing prevented a
      `NODE_WITH_ADJACENT_GLUE` group from bundling nodes across two different SLRs (or,
      for Alveo, two independently-elaborated Vitis kernel sessions — physically
      impossible under `userPostSysLinkOverlayTcl`), violating §3.2 point 4's stated
      invariant. **Fixed: Level-2 grouping now runs first, and Level-1 assigns each group
      as one atomic LP variable** (§3.1, §3.1.1, §3.2 point 1, §5.1, §6 Milestone 1 steps
      3-4) — this guarantees the invariant by construction instead of relying on the LP's
      objective to informally tend toward it, and also correctly lifts anchor constraints
      (§3.1.2) to the whole group a node belongs to. The same reasoning applies one level up
      to `device_id`: grouping must also never merge nodes across a multi-FPGA device
      boundary, now stated explicitly (§3.1.1, §3.2 point 1) rather than left for Level-1
      to discover per-device.
- [x] ~~Can Vitis's kernel-compile step accept a Tcl/property injection point for creating
      pblocks inside a packaged kernel's internal hierarchy?~~ **Yes — mechanism identified
      and cited** (§5.0, §5.2): `param:compiler.userPostSysLinkOverlayTcl` (v++ config,
      `[advanced]` section) runs a Tcl script as a post-system-link hook, with individual
      kernel instances and internal ports confirmed live/addressable at that point; AMD's
      own `xbtest` Alveo test suite independently confirms this class of mechanism is real
      and production-used for exactly this purpose (per-SLR `pblock_dynamic_SLR0/1/2`
      floorplanning of compute units). A secondary hook,
      `--vivado.prop run.impl_1.STEPS.PLACE_DESIGN.TCL.PRE=<tcl>`, covers post-synthesis
      refinement mode. This downgrades the Milestone 1 spike (§6, step 0) from "resolve an
      open feasibility question" to "confirm a documented mechanism against FINN's own
      generated kernels in this repo's Vitis 2022.2/2024.2 environment" — still worth
      doing, no longer a blocker to design work proceeding. Level-2 fine-grained pblock
      packing on the Alveo/U55C validation target is expected to be achievable, pending
      that confirmation.
- [x] ~~Exact new-attr names/types (§4) and the on-disk config schema~~ — good enough to
      start from as specified in §4; a second pass once implementation starts is expected,
      not a blocker.
- [x] ~~Per-board `pcie_slr` values need sourcing from real shell documentation.~~ Resolved
      for U55C/U280 (confirmed SLR0, §6 Milestone 1). Still open for U50/U200/U250 — these
      fall back to SLR0 with a warning until someone confirms their actual shell layout;
      worth tracking as a small follow-up to replace the fallback with a confirmed value.
- [x] ~~Should `floorplan_slr` ever be merged into the legacy `slr` attr?~~ Keep the attrs
      separate (§4); the *bridge step* that connects them defaults to **on** (revised — see
      §4 for why: needed for a valid WNS/congestion signal against the success criteria
      below, and load-bearing for Alveo kernel splitting specifically). Merging the attrs
      themselves is not closed permanently, just not where this starts.
- [x] ~~"Materially different" threshold for triggering a re-pack in post-synthesis
      refinement mode~~ — default: hard trigger when measured usage exceeds allocated
      capacity for any resource, soft trigger (on by default) at >20% LUT deviation from
      estimate, both configurable (§3.2 point 2). Tune the 20% during Milestone 5
      validation rather than treating it as final.
- [x] ~~`NODE_WITH_ADJACENT_GLUE`'s DWC sub-option: on or off by default?~~ On by default
      (§3.2 point 1) — keeps the mode's pblock-count-reduction purpose intact; configurable
      off for testing.
- [x] ~~What's the target list of models/boards for validation, and what metric counts as
      success?~~ **`resnet18` and `cnv-w1a1` on the U55C** (§6 Milestone 5). Success
      criteria, in priority order:
      1. **No regression in resource usage** — the primary gate. Open to a small
         regression if it comes with a significant WNS/congestion improvement (a
         discussion, not an automatic pass/fail), but resource usage is the first thing
         checked, not an afterthought to timing.
      2. **Congestion and WNS improvement** — the actual point of the feature, checked
         once criterion 1 is satisfied (or the trade-off in criterion 1 is accepted).
      The underlying goal these serve: designs should be able to run at a **faster clock**,
      or use **significantly more of the device's resources**, without hitting P&R
      failures they hit today — that's the real bar "success" is measured against, not WNS
      alone in isolation. (The U55C is an Alveo board — this validation exercises Level-1
      fully and Level-2 pending the Milestone-1 spike, per the mechanism identified in the
      first item of this list.)
- [x] ~~Full audit of which node categories carry a direct external-memory interface beyond
      IODMA~~ — done, not deferred (§3.1.2): the anchor rule is now generic
      (`get_verilog_top_module_intf_names()["aximm"]` non-empty), with concrete verified
      hits beyond IODMA (`Lookup` with `mem_mode == "external"`, `FINNLoop`'s `m_axi_hbm`,
      and `matrixvectoractivation.py`'s unrelated `mlo_max_iter` branch) — and one verified
      *non*-hit worth remembering as a cautionary example: `MVAU`/`VVAU` with
      `mem_mode == "external"` streams weights via `s_axis`, not `aximm`, so it correctly
      needs no anchor of its own (§3.1.2's correction) — so future ops are caught
      automatically rather than needing this list maintained by hand.
- [x] ~~Should this feature also fix the HBM sequential-channel-assignment gap in
      `BuildBasicVitisLinkConfig`?~~ Yes, via `mem_port` reuse (§3.1.2, §4); the actual code
      change is itemized as its own phase step in §6 Milestone 3 (step 10).
- [x] ~~Is the `mem_port` nodeattr worth wiring up?~~ Yes, reuse it directly as the
      external-memory-anchor carrier (§4) — verified safe: its one call site is a
      pass-through, not an independent producer (§2.4).
- [x] ~~Does the Zynq/`make_zynq_proj` flow need anchoring beyond SLR/pblock placement?~~
      No. `make_zynq_proj` targets are PS-attached-DDR only — Level-1 SLR/pblock placement
      is the whole story there, no §3.1.2 anchor rules apply.
- [x] ~~What happens if a build sets both `enable_auto_floorplanning=True` and an explicit
      `vitis_floorplan_file`/`partitioning_configuration`?~~ Explicit user config wins,
      auto-assignment is skipped entirely with a warning (§5.3) — no partial merge in v1.
- [x] ~~Is the Zynq "`nslr == 1`, no PCIe-anchor question" claim actually verified?~~ Yes —
      cited against all five currently-defined Zynq platform classes (§2.4).

## 8. Engineering conventions and test tiers

This repo's `CLAUDE.md` sets specific, checked conventions; calling out how this feature
meets them up front rather than leaving it implicit:

- **Docstrings on every module/class/function/method, including private ones** —
  `check_docstrings.py` walks the AST and checks this (not CI-enforced per current
  `CLAUDE.md`, but expected practice, and it's what feeds the wiki's generated API
  reference). Applies to every new class/function across all milestones in §6.
- **Document pre/postconditions on every new `Transformation` subclass explicitly** —
  `CLAUDE.md` calls "B needed A to run first" out by name as a recurring bug class, and
  this feature is unusually exposed to it: Level-1 SLR assignment must run before Level-2
  grouping, which must run before Phase B binding (§5.1-§5.2), and the
  `floorplan_slr → slr` bridge must run before `Floorplan` for its output to mean anything
  (§4). Each new transformation's docstring should state what it requires to have already
  run and what it guarantees for what comes after — not just what it does.
- **`FINNUserError`/`FINNInternalError`, never bare `assert`**, per existing convention —
  applies to, at minimum: LP infeasibility in Level-1 assignment (§3.1), an anchor that
  cannot be satisfied (§3.1.2), a Level-2 unit that can't be packed even after the
  degrade-gracefully fallback (§3.2 point 5), and a config conflict outside the resolved
  case in §5.3.
- **Full type annotations**, per existing convention across the new modules.
- **Test tiers** (mapping onto `finn test`'s existing `quick` vs. `vivado`/`slow` tiers,
  [CLAUDE.md](CLAUDE.md)'s Commands section):
  - `quick` (no Vivado): Level-2 unit computation (grouping/ordering/margin math, §6
    Milestone 1, step 3) — kept as a separate Python layer from the Tcl/Vivado half
    specifically so it stays `quick`-testable (§3.2 point 3), and run before Level-1 so its
    output can be tested as the LP's actual input, not a hypothetical one. Level-1 SLR
    assignment (§6 Milestone 1, step 4) — pure Python/`mip`, same tier
    `tests/fpgadataflow/multifpga/` already uses for `AuroraPartitioner`.
  - `vivado`/`slow`: the generalized Tcl geometry engine (§6 Milestone 2), Phase B binding
    for both flows (§6 Milestone 3, steps 8-9), and the `add_slr`/`mem_port` consumption
    changes (§6 Milestone 3, steps 9-10) — all need a real Vivado/Vitis project.
  - **Not part of routine `finn test` at all**: Milestone 5 hardware validation
    (`resnet18`/`cnv-w1a1` on real U55C hardware, multi-hour P&R runs) — a manual/CI-separate
    validation exercise, not something `finn test -v vivado` should attempt to run on every
    invocation.

**Second test example: the vision transformer (16 Oct 2026, decided by Linus).** Of the four
transformers in `models/transformer`, the vision model has a known-good U55C recipe
(`fifoNetworks/visionTransformer/transformer.yml`: 10 ns, `target_fps` 1525, the ad-hoc
transformer steps, LUT MACs; bitfile on 10 Aug 2026; 75 nodes -- 29 Thresholding_rtl,
14 MVAU_hls, 8 ScaledDotProductAttention_hls, 6 StreamingSplit_hls, 5 ElementwiseAdd_hls,
5 ReplicateStream_hls -- 33 k LUTs, 50 BRAM, no DSP). It is small for the device but it is
the first non-linear dataflow (split/replicate/attention branches) the floorplanner sees:
the grouping, the dataflow-order packing and the neighbour anchoring all assume a chain.
`cfg/transformer_vision_baseline.yml` is that recipe with the floorplanning steps added
and the two measurement steps dropped; the floorplanned `transformer_vision_per_node.yml`
restarts from the baseline's `step_generate_hardware` checkpoint (same FINN_BUILD_DIR
`tmp/tf_vision`), so FIFO depths and IP are identical. `run_cluster.sh` takes the model
through `MODEL`. Baseline job 3698764.

**Results of 16 Oct 2026.**

*Fan-out fix, 5 ns pair* (`*_5ns_fanout`, default strategy, no slices, first attempt each):
floorplanned -0.57 ns / 244 failing endpoints against the baseline's -1.31 / 1,691. The
command and pointer nets are gone from the top paths; the floorplanned design's failing
export paths are now 8-level window-buffer logic (72 of 158) and zero-level nets in the
180-row chain carrier (48). Best floorplanned result so far (previous default pair:
-0.76 / 295 against -1.01 / 1,737).

*Vision transformer, 10 ns* (`transformer_vision_{baseline,per_node}`, same checkpoint,
272 units over SLR0 and SLR1 although the design fills 69 % of one SLR's LUTs): the
floorplanned design fails, -0.21 ns / 232 endpoints, shipped at 95.4 MHz; the free
baseline passes with +0.91 ns. The failing paths are 16-28-level paths inside the
ScaledDotProductAttention units (baseline +0.9 to +1.1 ns on the same units; endpoint
distance 29 tiles in the pblocks against 14 free) and 59 paths inside the platform's HBM
subsystem (`hmss_0`, +7.1 ns in the baseline). Two hypotheses, unmeasured: deep-logic
units are spread by pblock shapes sized for area, and the packer, filling SLR0 from row 0,
takes the rows next to the HBM that the platform's own logic needs.

*The 5 ns experiments were implemented at 10 ns.* On this platform v++ constrains the
kernel clock in implementation as a 1:1 generated clock of the MMCM input
(`vpl/output/_user_impl_clk.xdc`: `-divide_by 1 -multiply_by 1` from the 100 MHz input);
`--kernel_frequency` reaches only the out-of-context synthesis XDC (`ulp_ooc_copy.xdc`:
period 5.0) and the post-route frequency scaler. Every build, 10 ns or 5 ns, reports the
identical placer and router WNS (-0.006 / -0.017, a platform path). So every "5 ns"
comparison measured placements made for 10 ns, re-timed at 5 ns: a fair comparison of
route lengths between the two flows, not of what either reaches when driven at 5 ns.
FINN users get the same: `synth_clk_period_ns` below 10 ns does not drive Vivado's
placement on the U55C. Fix to make: constrain the kernel clock at the requested period in
implementation (a `create_generated_clock` override on the kernel MMCM output from the
link hook), then rerun the 5 ns pairs.

**Plan step 1 (16 Oct 2026): two implementation bugs, both affecting every FINN Alveo build.**

1. *Kernel clock.* Vivado 2024.2's clock-wizard solver, which v++ uses to turn
   `--kernel_frequency` into the implementation clock, fails in the in-memory project of
   the impl run (`[Ip 78-110] Invalid part string Project`, reproduced in a bare
   `create_project -in_memory`), returns M = D = 1, and implementation times the kernel
   clock at the MMCM input -- 100 MHz on the U55C. Fix: `render_kernel_clock_tcl` /
   `BuildBasicVitisLinkConfig._configure_kernel_clock`, an `OPT_DESIGN.TCL.PRE` hook (Vitis
   appends it to its own) that re-creates the clock at the requested ratio; at 100 MHz it
   leaves the clock alone. Tested on a routed 5 ns checkpoint: 10.000 -> 5.000 ns, x2/1.
   Side finding: v++'s post-route scaler computes `1/(period - slack)` of the kernel
   clock's worst path, which at the re-timed 5 ns was a 9.25 ns reset-tree hop:
   1/(5 + 4.58) = 104.4 MHz, the "achieved" frequency of every 5 ns build.
2. *Reset-hop constraint never reached the main kernels.* `write_xdc -cell` ends inside
   the scope of the last XPM CDC instance it constrained (no closing `current_instance
   -quiet`), so the appended `set_max_delay` query ran inside a FIFO's reset synchroniser
   and matched nothing: the exception existed for the DMAs only (`report_exceptions`:
   2 of 3 kernels; none from the main partition's tree register). Fix: a
   `current_instance -quiet` line before it. All "reset hop" conclusions since rebuild4
   rest on the tree and per-node pipes alone, not on the constraint (rebuild7's tree hop:
   8 ns of route, 1.98 ns of slack at 10 ns, no exception).

Rebuilds: `*_5ns_clk` pair (MobileNet, both fixes, from `step_prepare_synthesis`).

**Plan step 2: the transformer.** The attention units are *more* compact in the pblocks
than free (e.g. unit 90: 56 x 90 tiles against 129 x 68), so the "pblocks spread deep
logic" hypothesis is wrong. Free placement spreads each unit at low density and keeps
each deep path's cells close (endpoint distance 14 tiles against 29); the pblocks hold
them at 72 % LUT utilisation (the packer's 0.75 ceiling) while the design fills about a
third of its two SLRs. Test: `transformer_vision_per_node_density` (logic utilisation
0.45), with `transformer_vision_{per_node,baseline}_rst` as the same-code references.
Separately the floorplanned transformer fails the *platform's* HBM clock (hbm_aclk,
-1.31 ns; hmss paths stretched over 118 tile rows) -- the only one of 24 routed builds
that does; the free build meets it (+0.007) and the floorplanned MobileNets all do
(+0.05 to +0.11). Not explained by pblock coverage of SLR0's bottom (37 % against
MobileNet's 48 %) or by the DMA pblocks' positions; watched in the three rebuilds.
