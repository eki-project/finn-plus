"""Tests for the last-ditch loop that re-runs implementation on a placer rejection."""

# The log fixture below is Vivado's own output, kept verbatim so the parser is
# tested against what it will really see; its lines are as long as Vivado makes them.
# ruff: noqa: E501

import pytest

from pathlib import Path

from finn.transformation.fpgadataflow.floorplanning.placer_retry import (
    latest_attempt,
    parse_placer_rejections,
    retry_script,
    write_overrides,
)
from finn.transformation.fpgadataflow.floorplanning.tcl import (
    FloorplanUnitSpec,
    demands_from_estimate,
    render_floorplan_tcl,
)
from finn.util.basic import make_build_dir

#: Verbatim from a failed mobilenetv1 implementation: one pblock short on two
#: resources, plus a block of unrelated noise the parser must skip.
_LOG = """
INFO: [Place 30-611] Multithreading enabled for place_design using a maximum of 8 CPUs
ERROR: [Place 30-859] The number of BRAM cells that have to be placed in a constraint region exceeds the number of corresponding sites available.

Area constraint: PBlock
PBlock name is: pblock_finn_unit_50
Number of BRAM required by this constraint: 7
Number of BRAM Fixed inside this constraint: 0
Number of BRAM available in this constraint region: 5
Total number of BRAM blocked sites in this constraint region: 23
Number of BRAM sites in this constraint region blocked with place.closeUnroutableSites param: 23

Resolution: Check any pblock constraints that could prevent placement of cells in the constraint area.
ERROR: [Place 30-859] The number of DSP cells that have to be placed in a constraint region exceeds the number of corresponding sites available.

Area constraint: PBlock
PBlock name is: pblock_finn_unit_50
Number of DSP required by this constraint: 32
Number of DSP Fixed inside this constraint: 0
Number of DSP available in this constraint region: 20
Total number of DSP blocked sites in this constraint region: 16
ERROR: [Place 30-859] The number of DSP cells that have to be placed in a constraint region exceeds the number of corresponding sites available.

Area constraint: PBlock
PBlock name is: pblock_finn_unit_246
Number of DSP required by this constraint: 32
Number of DSP Fixed inside this constraint: 0
Number of DSP available in this constraint region: 24
ERROR: [Place 30-99] Placer failed with error: 'Implementation Feasibility check failed'
"""


@pytest.mark.fpgadataflow
def test_rejections_are_parsed_to_exact_shortfalls_per_pblock_and_resource() -> None:
    """The retry must ask for exactly what the placer said was missing, no more."""
    got = parse_placer_rejections(_LOG)
    assert got == {
        "pblock_finn_unit_50": {"brams": 7 - 5, "dsps": 32 - 20},
        "pblock_finn_unit_246": {"dsps": 32 - 24},
    }


@pytest.mark.fpgadataflow
def test_a_log_without_a_rejection_yields_nothing() -> None:
    """Any other failure must not trigger a retry; only a pblock rejection does."""
    assert parse_placer_rejections("ERROR: [Route 35-2] Design is not legally routed.") == {}
    assert parse_placer_rejections("") == {}


@pytest.mark.fpgadataflow
def test_overrides_are_written_as_the_tcl_dict_the_engine_sources() -> None:
    """The file must be exactly what `finn_fp_load_overrides` expects to `source`."""
    path = Path(make_build_dir("placer_retry_")) / "finn_floorplan_overrides.tcl"
    write_overrides(
        path, {"pblock_finn_unit_50": {"dsps": 12, "brams": 2}, "pblock_x": {"urams": 1}}
    )
    assert path.read_text() == (
        "set finn_fp_overrides {pblock_finn_unit_50 {brams 2 dsps 12} pblock_x {urams 1}}\n"
    )


@pytest.mark.fpgadataflow
def test_retry_script_resumes_from_implementation_and_leaves_the_original_alone() -> None:
    """Synthesis and opt_design are unaffected by a pblock resize: resume at placement."""
    d = Path(make_build_dir("placer_retry_"))
    original = d / "run_link.sh"
    original.write_text(
        "#!/bin/bash\nv++ --target hw --platform p --link a.xo --config link.cfg --save-temps\n"
    )
    retry = retry_script(original)
    assert retry != original
    lines = retry.read_text().splitlines()
    # The hook reuses the first attempt's measurements and selects the placer
    # mode for a reopened checkpoint when the attempt is marked as resumed.
    assert lines[1] == "export FINN_FP_RESUMED_ATTEMPT=1"
    assert lines[2].endswith("--save-temps --from_step vpl.impl.place_design")
    assert "--from_step" not in original.read_text()
    # Idempotent: deriving again from the original must not stack the flag.
    assert retry_script(original).read_text().count("--from_step") == 1


@pytest.mark.fpgadataflow
def test_renderer_passes_the_overrides_path_only_when_given() -> None:
    """Zynq renders without one; Alveo renders with one. The Tcl default covers the former."""
    units = [FloorplanUnitSpec("pblock_u0", 0, demands_from_estimate({"LUT": 10}))]
    plain = render_floorplan_tcl(units)
    assert _call(plain).endswith("} 1 1 1.1")
    with_file = render_floorplan_tcl(units, overrides_file="/tmp/x/finn_floorplan_overrides.tcl")
    assert _call(with_file).endswith("} 1 1 1.1 {/tmp/x/finn_floorplan_overrides.tcl}")


def _call(lines: list[str]) -> str:
    """Return the `finn_fp_floorplan_units` line of a rendered hook."""
    return next(line for line in lines if line.startswith("finn_fp_floorplan_units "))


@pytest.mark.fpgadataflow
def test_lutram_legalisation_failure_is_attributed_through_the_report() -> None:
    """`[Place 30-1112]` names a SLICE range, not a pblock; the report maps it back.

    The unit's memory-LUT demand is then raised by a quarter of the SLICEM LUTs its
    pblock offered (`LUTRAM_RETRY_GROWTH`), because the message carries no count.
    Without a report the message cannot be attributed and is skipped with a
    warning rather than mis-assigned.
    """
    log_text = (
        "ERROR: [Place 30-1112] Failed to move LUTRAM instance level0_i/ulp/x/srl_reg[30][0]_srl31"
        " to a legal location. This is most likely due to an over-constrained design. The site"
        " ranges the instance is constrained to are as follows:\n"
        "->from site SLICE_X41Y290(sliceM site) to site SLICE_X48Y322(sliceM site)\n"
    )
    report = {
        "units": {
            "pblock_finn_unit_137": {
                "grid_ranges": "SLICE_X41Y290:SLICE_X48Y322 RAMB36_X2Y58:RAMB36_X2Y63",
                "pblock_capacity": {"luts": 2112, "lutrams": 1056},
            },
            "pblock_finn_unit_1": {"grid_ranges": "SLICE_X0Y0:SLICE_X3Y4", "pblock_capacity": {}},
        }
    }
    assert parse_placer_rejections(log_text, report) == {"pblock_finn_unit_137": {"lutrams": 264}}
    assert parse_placer_rejections(log_text) == {}


@pytest.mark.fpgadataflow
def test_override_pushes_demand_past_the_rejected_region() -> None:
    """A shortfall the placer measures against its open sites must change the region.

    mobilenetv1 (2 Oct 2026): 14 BRAM required, 11 available, in a region of 24
    sites for a demand of 17. "+3" leaves 20 < 24 and the packer keeps the same
    two columns next to the shell. The override must take the demand past 24.
    """
    log_text = (
        "ERROR: [Place 30-859] The number of BRAM cells that have to be placed in a constraint"
        " region exceeds the number of corresponding sites available.\n"
        "PBlock name is: pblock_finn_unit_110\n"
        "Number of BRAM required by this constraint: 14\n"
        "Number of BRAM available in this constraint region: 11\n"
    )
    report = {"units": {"pblock_finn_unit_110": {"pblock_capacity": {"brams": 24}}}}
    units = [{"name": "pblock_finn_unit_110", "demands": {"brams": 17}}]
    assert parse_placer_rejections(log_text) == {"pblock_finn_unit_110": {"brams": 3}}
    assert parse_placer_rejections(log_text, report, units) == {
        "pblock_finn_unit_110": {"brams": 10}
    }


#: Detail placement could not commit plain logic of two input generators
#: (mobilenetv1, job 3551563, 3 Oct 2026); abridged to four of the nine instances.
_COMMIT_LOG = """Phase 3 Detail Placement | Checksum: 225d956bb
ERROR: [Place 30-488] Failed to commit 9 instances:
level0_i/ulp/StreamingDataflowPartition_2/StreamingDataflowPartition_2_i/\
StreamingDataflowPartition_2_ConvolutionInputGenerator_rtl_4/inst/impl/controller_inst/\
Window_buffer_read_addr_reg_reg[15]_i_2 with block Id: 474880 (CARRY) . No site found for this instance.
level0_i/ulp/StreamingDataflowPartition_2/StreamingDataflowPartition_2_i/\
StreamingDataflowPartition_2_ConvolutionInputGenerator_rtl_5/inst/impl/controller_inst/\
State_reg[1] with block Id: 474898 (FF) . No site found for this instance.
level0_i/ulp/StreamingDataflowPartition_2/StreamingDataflowPartition_2_i/\
StreamingDataflowPartition_2_ConvolutionInputGenerator_rtl_5/inst/impl/controller_inst/\
Window_buffer_read_addr_reg[14]_i_6 with block Id: 474924 (LUT) . No site found for this instance.
level0_i/ulp/StreamingDataflowPartition_2/StreamingDataflowPartition_2_i/\
StreamingDataflowPartition_2_ConvolutionInputGenerator_rtl_4/inst/impl/controller_inst/\
Counter_loop_kw[2]_i_2 with block Id: 474929 (LUT) . No site found for this instance.

ERROR: [Place 30-99] Placer failed with error: 'failed to commit all instances'
"""


@pytest.mark.fpgadataflow
def test_logic_commit_failure_is_attributed_through_the_unit_cells() -> None:
    """`30-488` names instances, not pblocks; their hierarchy finds the unit.

    Each failed type raises that demand by a quarter of what the pblock offered,
    at least by the number of instances that failed.
    """
    units = [
        {
            "name": "pblock_finn_unit_238",
            "cells": ["StreamingDataflowPartition_2_ConvolutionInputGenerator_rtl_4"],
        },
        {
            "name": "pblock_finn_unit_256",
            "cells": ["StreamingDataflowPartition_2_ConvolutionInputGenerator_rtl_5"],
        },
        {"name": "pblock_finn_unit_9", "cells": ["StreamingDataflowPartition_2"]},
    ]
    cap = {"luts": 240, "ffs": 480, "carries": 30, "brams": 6, "dsps": 6, "urams": 4, "lutrams": 0}
    report = {
        "units": {
            "pblock_finn_unit_238": {"pblock_capacity": cap},
            "pblock_finn_unit_256": {"pblock_capacity": cap},
        }
    }
    got = parse_placer_rejections(_COMMIT_LOG, report, units)
    assert got == {
        "pblock_finn_unit_238": {"carries": 7, "luts": 60},
        "pblock_finn_unit_256": {"ffs": 120, "luts": 60},
    }, "the enclosing partition's unit must not take the blame"
    # Without the report there is no capacity to scale: the count of failures stands in.
    assert parse_placer_rejections(_COMMIT_LOG, None, units)["pblock_finn_unit_238"] == {
        "carries": 1,
        "luts": 1,
    }


#: Detail placement could not commit carry-chain shapes of one input generator
#: (mobilenetv1, job 3555658, 3 Oct 2026); abridged to two of the four shapes.
_MACRO_LOG = """Phase 3 Detail Placement | Checksum: 1f062b5fe
ERROR: [Place 30-1153] Failed to commit 4 macros:
level0_i/ulp/StreamingDataflowPartition_2/StreamingDataflowPartition_2_i/\
StreamingDataflowPartition_2_ConvolutionInputGenerator_rtl_5/inst/impl/controller_inst/\
Window_buffer_read_addr_reg1_carry_i_1 with block Id: 309476
This block contains 9 instances.
Instance no. 0: Window_buffer_read_addr_reg1_carry_i_1
Instance no. 1: Window_buffer_read_addr_reg1_carry
This block has a USER PBLOCK associated with it. The PBLOCK has an EXCLUDE property associated with it.
level0_i/ulp/StreamingDataflowPartition_2/StreamingDataflowPartition_2_i/\
StreamingDataflowPartition_2_ConvolutionInputGenerator_rtl_5/inst/impl/controller_inst/\
Current_elem0_carry_i_1 with block Id: 309459
This block contains 13 instances.
Instance no. 0: Current_elem0_carry_i_1
This block has a USER PBLOCK associated with it. The PBLOCK has an EXCLUDE property associated with it.

ERROR: [Place 30-99] Placer failed with error: 'Could not place all shapes'
"""


@pytest.mark.fpgadataflow
def test_shape_commit_failure_grows_the_unit_it_belongs_to() -> None:
    """`30-1153` names shapes by one instance each; the unit grows by a quarter."""
    units = [
        {
            "name": "pblock_finn_unit_256",
            "cells": ["StreamingDataflowPartition_2_ConvolutionInputGenerator_rtl_5"],
        },
        {"name": "pblock_finn_unit_9", "cells": ["StreamingDataflowPartition_2"]},
    ]
    report = {"units": {"pblock_finn_unit_256": {"pblock_capacity": {"luts": 240}}}}
    assert parse_placer_rejections(_MACRO_LOG, report, units) == {
        "pblock_finn_unit_256": {"luts": 60}
    }
    # Two failed shapes and no capacity: one slice (8 LUTs) per shape.
    assert parse_placer_rejections(_MACRO_LOG, None, units) == {
        "pblock_finn_unit_256": {"luts": 16}
    }


@pytest.mark.fpgadataflow
def test_commit_failure_pushes_demand_past_the_rejected_region() -> None:
    """A unit that failed inside its region is sized beyond it, not merely raised.

    Unit 136 of mobilenetv1 (46 LUTs, 33 flip-flops in 10 slices) failed to commit;
    raising its demand by a quarter of the region (to 66 LUTs) left it in the same
    10 slices. The override now asks for 1.25x what the region offered.
    """
    log = """ERROR: [Place 30-488] Failed to commit 2 instances:
level0_i/ulp/StreamingDataflowPartition_1/StreamingDataflowPartition_1_i/\
StreamingDataflowPartition_1_FMPadding_rtl_6/inst/a_reg with block Id: 1 (FF) . No site found.
level0_i/ulp/StreamingDataflowPartition_1/StreamingDataflowPartition_1_i/\
StreamingDataflowPartition_1_FMPadding_rtl_6/inst/b_i_1 with block Id: 2 (LUT) . No site found.
"""
    units = [
        {
            "name": "pblock_finn_unit_136",
            "cells": ["StreamingDataflowPartition_1_FMPadding_rtl_6"],
            "demands": {"luts": 46, "ffs": 33, "carries": 0},
        }
    ]
    cap = {"luts": 80, "ffs": 160, "carries": 10}
    report = {"units": {"pblock_finn_unit_136": {"pblock_capacity": cap}}}
    got = parse_placer_rejections(log, report, units)["pblock_finn_unit_136"]
    assert got == {"luts": 80 + 20 - 46, "ffs": 160 + 40 - 33}
    assert 46 + got["luts"] > 80, "the new demand no longer fits the rejected region"


@pytest.mark.fpgadataflow
def test_only_the_latest_attempt_of_an_accumulated_log_is_parsed() -> None:
    """Vitis appends resumed attempts to one log; earlier rejections must not count again."""
    first = " FINN+ automatic pblock floorplanning (285 unit(s))\n" + _COMMIT_LOG
    second = (
        " FINN+ automatic pblock floorplanning (285 unit(s))\n"
        "Phase 3 Detail Placement\n"
        "ERROR: [Place 30-99] Placer failed with error: 'failed to commit all instances'\n"
    )
    units = [
        {
            "name": "pblock_finn_unit_238",
            "cells": ["StreamingDataflowPartition_2_ConvolutionInputGenerator_rtl_4"],
        },
        {
            "name": "pblock_finn_unit_256",
            "cells": ["StreamingDataflowPartition_2_ConvolutionInputGenerator_rtl_5"],
        },
    ]
    assert parse_placer_rejections(latest_attempt(first), None, units)
    assert parse_placer_rejections(latest_attempt(first + second), None, units) == {}
    assert latest_attempt("no banner at all") == "no banner at all"


@pytest.mark.fpgadataflow
def test_lutram_packing_failure_grows_the_memory_luts_of_the_unit() -> None:
    """`30-484` names the primitives that found no SLICEM; their unit asks for more memory LUTs.

    Twelve RAM64M8 of one window buffer (mobilenetv1 with register slices, 15 Oct
    2026): 96 memory LUTs that did not fit, on top of the quarter step past the
    pblock's capacity.
    """
    cig = "StreamingDataflowPartition_1_ConvolutionInputGenerator_rtl_0"
    prefix = f"level0_i/ulp/StreamingDataflowPartition_1/StreamingDataflowPartition_1_i/{cig}"
    names = "\n".join(
        f"{prefix}/inst/impl/window_buffer_inst/Ram_reg_{64 * k}_{64 * k + 63}_0_6 type RAM64M8"
        for k in range(12)
    )
    log_text = (
        "ERROR: [Place 30-484] The packing of LUTRAM/SRL instances into capable slices could not"
        " be obeyed. \n\tNumber of LUTRAMs/SRLs: 38522\n\tNumber of capable slices required is"
        " 68086 out of 73370 in the device (utilization 92.7981%)\nEven though there is a"
        " sufficient number of capable slices in the device, the packing algorithm was not able"
        " to find a solution.\nAs a result, 12 LUTRAMs/SRLs failed to place.\n"
        f"Names of these LUTRAMs/SRLs:\n{names}\nERROR: [Place 30-99] Placer failed with error:"
        " 'Could not place all lutrams'\n"
    )
    units = [
        {"name": "pblock_finn_unit_2", "cells": [cig], "demands": {"lutrams": 400, "luts": 2000}},
        {"name": "pblock_finn_unit_3", "cells": ["other_node"], "demands": {"lutrams": 10}},
    ]
    report = {"units": {"pblock_finn_unit_2": {"pblock_capacity": {"lutrams": 448}}}}
    out = parse_placer_rejections(log_text, report, units)
    assert list(out) == ["pblock_finn_unit_2"]
    # Past the 448 the region offered by a quarter step, and at least the 96 LUTs that failed.
    assert out["pblock_finn_unit_2"]["lutrams"] >= 96
    assert 400 + out["pblock_finn_unit_2"]["lutrams"] > 448
    assert parse_placer_rejections(log_text, report, None) == {}
