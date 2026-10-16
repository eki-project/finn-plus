# Shell closure maps

One JSON file per platform shell and Vivado release, named
`<platform>__vivado-<release>.json`. Each lists the sites inside the user region
that Vivado's placer closes as unroutable next to the static shell
(`place.closeUnroutableSites`). The floorplanner marks them `PROHIBIT` before it
takes the device census, so no pblock is ever sized over a site the placer will
not use.

## Adding a platform or a new Vivado release

With Vivado on `PATH` and the platform installed (`PLATFORM_REPO_PATHS` or
`/opt/xilinx/platforms`):

```bash
finn floorplan map-shell --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --jobs 4
finn floorplan show-shell --platform xilinx_u55c_gen3x16_xdma_3_202210_1
```

The first command extracts the locked shell checkpoint from the platform's
`hw/hw.xsa`, probes every site column of the user region, then every single site
in the columns that lost any, and writes the map here. Each probe run opens the
checkpoint (about 8 GB of memory, a few minutes); `--jobs` runs several at once.
`--out-dir` writes the map elsewhere; point `FINN_SHELL_CLOSURE_PATH` at that
directory (several may be given, separated by `:`) and the floorplanner finds it
before the maps shipped here. `--checkpoint` probes a locked checkpoint directly,
for a platform that is not installed.

Re-map when the platform or the Vivado release changes. A map measured under
another release is still used, but the floorplanning hook warns about it.

## Format

```json
{
 "format": 1,
 "platform": "xilinx_u55c_gen3x16_xdma_3_202210_1",
 "part": "xcu55c-fsvh2892-2L-e",
 "vivado": "2024.2.2",
 "kinds": ["RAMB36", "DSP48E2", "URAM288", "SLICE"],
 "probed_sites": {"RAMB36": 1776, "...": 0},
 "closed": {"RAMB36": ["RAMB36_X12Y0", "..."], "...": []},
 "generated": "2026-10-03",
 "checkpoint": "<path of the probed checkpoint>"
}
```

A closed `RAMB36` site takes both `RAMB18` halves of its tile; they are not listed
separately. A kind missing from `kinds` was not probed.

Resolution is one placement tile. A DSP tile holds two DSP48E2 sites, and a
single-site pblock covers the whole tile, so a tile with one closed site lists
both (on the U55C shell, SLR0 DSP column X30 has 16 closed sites in 9 tiles and
lists 18). Vivado builds pblocks from whole tiles anyway, so a floorplan cannot
use half a tile either.

## Carry chains (`CARRY8`)

Next to the static shell the placer also refuses carry chains in slices where a
plain LUT still fits, and it says so only by failing to commit the chains
(`[Place 30-1153]`), with no closure line. The `CARRY8` kind measures this by
placement: every column run is probed with two-slice carry chains; a run that
loses chains is halved and probed again, down to runs of 30 slices, which are
recorded as closed as a whole if they still lose any (finer pieces each abort a
whole placement run and cost about ten minutes apiece). The floorplanner
prohibits these slices like any other closed site, so they are lost to plain
logic too: on the U55C shell 3,580 of 143,280 slices (2.5 %), in the SLR0 corner
block the shell encloses and along the columns next to it.

An interrupted `map-shell` run resumes: probe runs are named after a hash of
their request, and a finished one is not repeated.

## Partly closed columns (column rule)

The placer's closure is not complete next to the shell: on the U55C it left 11
of the 24 block RAM sites in SLR0's edge columns X12/X13 open, and a weight
memory placed there had 33 pins the router could not reach (`[Route 35-7]`).
So a block RAM or DSP column that loses any site to the closure is excluded as a
whole within its SLR (`close_partial_columns`, applied by `map-shell`). The map
records what the rule added in `column_rule`; on the U55C shell that is 11 RAMB36
and 150 DSP48E2 sites on top of the 13 and 282 measured.
