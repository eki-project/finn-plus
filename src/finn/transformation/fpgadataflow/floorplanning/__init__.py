"""Automatic pblock floorplanning for FINN+ dataflow designs.

The floorplanner works in two levels:

- **Level 1** assigns each *placement unit* to one SLR of the target part, minimizing
  SLR-crossing (SLL) cost subject to per-SLR resource budgets and external-resource
  anchors (`slr_assignment`, `anchors`).
- **Level 2** groups nodes into placement units and, later in the flow, packs one
  pblock per unit inside its parent SLR (`grouping`; the Vivado-side packer is not
  part of this module).

Level-2 *grouping* runs before Level-1 assignment, so a placement unit is guaranteed
by construction to live in exactly one SLR (and on exactly one device).

This is deliberately separate from `finn.transformation.fpgadataflow.multifpga`, which
solves the different problem of splitting a model across physical boards, and from the
legacy `Floorplan` transform, which applies a user-supplied SLR assignment rather than
computing one.
"""
