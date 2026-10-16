################################################################################
# FINN+ automatic pblock floorplanner - Vivado-side shell.
#
# Vivado is the measurement and application layer; the packing arithmetic lives
# in Python (finn/transformation/fpgadataflow/floorplanning/packer.py). This
# script:
#
#   1. exports the device as a census: per SLR and site kind, every column with
#      its tile-grid column and its *available* rows (IS_USED/PROHIBIT filtered,
#      clipped to the dynamic region), plus the clock-region bands per kind;
#   2. measures every unit's utilisation after opt_design and traces its
#      cascade chains (every chain, not just the longest);
#   3. runs the packer as a separate process, with the interpreter and the
#      environment FINN baked into the rendered hook (FINN_FP_PYTHON,
#      FINN_FP_PYTHON_ENV), so Vivado's own library path never reaches it;
#   4. applies the rectangles it returns as hard pblocks, verifies each one on
#      the sites Vivado actually built (counts, SLICEM, chain runs inside one
#      clock region) and feeds every shortfall back for another round, three at
#      most;
#   5. writes the JSON report the Python side enriches.
#
# Alveo realities this is written for: the user region is HD.RECONFIGURABLE, so
# IS_SOFT is discarded (Constraints 18-5661), every pblock is hard, nested
# children are clipped to pblock_dynamic_SLR<n> (18-12550), every primitive type
# must be ranged (HDPR-18) and an undersized region is a UTLZ-1 error. A block
# RAM cascade may not be forced across a clock region (18-5197) and every
# cascade needs its own column run (18-11842).
#
# Entry points:
#   finn_fp_floorplan_units        - census, measure, pack, apply, verify
#   finn_fp_export_census          - census only (fixtures for the packer tests)
#   finn_fp_query_cells_batch      - batched post-synthesis utilisation per cell
#   finn_fp_write_floorplan_report - JSON report of what was allocated
################################################################################

set FINN_FP_SLICE_LUTS   8  ;# LUTs per SLICE (UltraScale+)
set FINN_FP_SLICE_FFS    16 ;# FF/latch sites per SLICE (UltraScale+)
set FINN_FP_SLICE_CARRY  1  ;# CARRY8 per SLICE (UltraScale+)

# Pack / apply / verify rounds. Round one packs from the census; a later round
# re-packs only the units whose applied pblock came up short, with the exact
# measured shortfall added to their demand.
set FINN_FP_MAX_ROUNDS 3

# Interpreter, environment and packer path. The rendered hook sets these from
# FINN's own sys.executable and a whitelisted os.environ (see tcl.py); the
# defaults only serve a bare `vivado -source` with the script on PATH.
if {![info exists FINN_FP_PYTHON]}     { set FINN_FP_PYTHON python3 }
if {![info exists FINN_FP_PYTHON_ENV]} { set FINN_FP_PYTHON_ENV {} }
if {![info exists FINN_FP_PACKER]} {
    set FINN_FP_PACKER [file normalize [file join [file dirname [info script]] \
        .. .. transformation fpgadataflow floorplanning packer.py]]
}
if {![info exists FINN_FP_PACK_ORDER]} { set FINN_FP_PACK_ORDER dataflow }
# Fraction of a pblock's SLICEM LUTs that LUTRAM/SRLs may fill: the legaliser cannot
# pack memory LUTs to the site count in a hard region ([Place 30-1112]). Set per
# build from DataflowBuildConfig.floorplan_slicem_utilization.
if {![info exists FINN_FP_SLICEM_UTIL]} { set FINN_FP_SLICEM_UTIL 0.8 }
# Shift registers per SLICEM slice, measured on FINN's SRL FIFO in hard regions
# (packer.py, DEFAULT_SRL_PER_SLICE): 1,120 SRLC32E occupied exactly 320 slices,
# 3.5 per slice, not eight.
if {![info exists FINN_FP_SRL_PER_SLICE]} { set FINN_FP_SRL_PER_SLICE 3.5 }
# Fraction of a pblock's slices plain logic (LUTs, flip-flops, carries) may fill
# (packer.py, DEFAULT_LOGIC_UTILIZATION). Set per build from
# DataflowBuildConfig.floorplan_logic_utilization.
if {![info exists FINN_FP_LOGIC_UTIL]} { set FINN_FP_LOGIC_UTIL 0.75 }
# Programmable Units a nested pblock in a reconfigurable region should span
# (Vivado's DRC HDPRA-65: fewer "may lead to unroute"). The hook asks Vivado
# after every round and re-packs the pblocks it reports.
if {![info exists FINN_FP_MIN_PROGRAMMABLE_UNITS]} { set FINN_FP_MIN_PROGRAMMABLE_UNITS 5 }
if {![info exists FINN_FP_WORK_DIR]}   { set FINN_FP_WORK_DIR {} }

# Sites the platform shell closes to the placer (`place.closeUnroutableSites`),
# measured once per platform and Vivado release by shell_closure.py. The
# rendered hook sets the list, the release it was measured under, and where it
# came from; an empty list means no map exists for this platform.
if {![info exists FINN_FP_CLOSED_SITES]}         { set FINN_FP_CLOSED_SITES {} }
if {![info exists FINN_FP_CLOSED_SITES_VIVADO]}  { set FINN_FP_CLOSED_SITES_VIVADO {} }
if {![info exists FINN_FP_CLOSED_SITES_SOURCE]}  { set FINN_FP_CLOSED_SITES_SOURCE {} }

################################################################################
# Site kinds and their site filters / name patterns.
################################################################################

set FINN_FP_KINDS_TABLE [list \
    SLICE    {SITE_TYPE =~ "SLICE*"}   {SLICE_X(\d+)Y(\d+)} \
    RAMB36   {NAME =~ "RAMB36_*"}      {RAMB36_X(\d+)Y(\d+)} \
    RAMB18   {NAME =~ "RAMB18_*"}      {RAMB18_X(\d+)Y(\d+)} \
    DSP48E2  {SITE_TYPE == "DSP48E2"}  {DSP48E2_X(\d+)Y(\d+)} \
    URAM288  {NAME =~ "URAM288_*"}     {URAM288_X(\d+)Y(\d+)}]

set FINN_FP_SITE_AVAILABLE {IS_USED == 0 && PROHIBIT == 0 && PROHIBIT_FROM_PERSIST == 0}

# Read one demand field, treating an absent one as zero. Keeps the engine usable
# by callers whose demand dict predates a newly modelled resource.
proc finn_fp_demand {demands key} {
    if {[dict exists $demands $key]} { return [dict get $demands $key] }
    return 0
}


# A retry resumes the implementation run at place_design from the optimised
# checkpoint instead of linking again; FINN marks it with FINN_FP_RESUMED_ATTEMPT
# (placer_retry.py). The hook then reuses the first attempt's measurements and
# selects the placer mode below.
proc finn_fp_resumed_attempt {} {
    return [expr {[info exists ::env(FINN_FP_RESUMED_ATTEMPT)] &&
                  $::env(FINN_FP_RESUMED_ATTEMPT) ne "0"}]
}


proc finn_fp_apply_placer_params {} {
    if {[info exists ::env(FINN_FP_OPEN_UNROUTABLE_SITES)] &&
        $::env(FINN_FP_OPEN_UNROUTABLE_SITES) ne "0"} {
        set_param place.closeUnroutableSites 0
        puts "  DIAGNOSTIC: place.closeUnroutableSites switched off for this run"
    }
    # Vitis runs the placer with ultrathreads mode off (place.ultrathreadsUsed 0
    # in its run script; Vivado's default is 1). That placer handles the pblocks
    # in the first attempt, where the netlist comes from link_design in the same
    # session, but on a design reopened from the opt checkpoint it counts every
    # DSP and block-RAM sub-cell as a primitive of its own against the pblock
    # capacities (a 64-DSP unit shows as 711 % of its 72 DSP sites) and rejects
    # floorplans it placed before. Measured 12 Oct 2026 on the U55C with one
    # floorplan: resumed placement passed in six of six runs with the mode on
    # and failed in four of four with it off. Only the resumed attempt is
    # affected, so only it switches the mode; the placement quality of the two
    # modes on that floorplan was alike (post-place WNS +0.003 vs -0.006 ns,
    # 32 vs 29 minutes).
    if {[finn_fp_resumed_attempt]} {
        set_param place.ultrathreadsUsed 1
        puts "  resumed attempt: placer runs in ultrathreads mode (place.ultrathreadsUsed 1)"
    }
}


# Compose a site filter for one kind with the availability condition.
proc finn_fp_site_filter {kind_filter} {
    global FINN_FP_SITE_AVAILABLE
    return "($kind_filter) && ($FINN_FP_SITE_AVAILABLE)"
}


proc finn_fp_intersect_sites {a b} {
    array set in_b {}
    foreach site $b { set in_b($site) 1 }
    set out {}
    foreach site $a {
        if {[info exists in_b($site)]} { lappend out $site }
    }
    return $out
}


proc finn_fp_region_scopes {slr} {
    set scopes {}
    foreach candidate [list pblock_dynamic_SLR$slr pblock_dynamic_region] {
        set pb [get_pblocks -quiet $candidate]
        if {$pb ne ""} { lappend scopes $pb }
    }
    return $scopes
}


proc finn_fp_report_parent_containment {} {
    set seen {}
    foreach pb [get_pblocks -quiet] {
        if {![string match "pblock_dynamic*" $pb]} { continue }
        if {[lsearch -exact $seen $pb] >= 0} { continue }
        lappend seen $pb
        set props {}
        foreach prop {CONTAIN_ROUTING EXCLUDE_PLACEMENT IS_SOFT HD.RECONFIGURABLE} {
            set v [get_property -quiet $prop [get_pblocks $pb]]
            if {$v eq ""} { set v "<unset>" }
            lappend props "$prop=$v"
        }
        puts "  parent region $pb: [join $props {  }]"
    }
    if {[llength $seen] == 0} {
        puts "  no parent dynamic-region pblock found (not a DFX flow)"
    }
}


proc finn_fp_apply_pblock {pblock_name cells ranges soft} {
    if {[get_pblocks -quiet $pblock_name] ne ""} { delete_pblocks $pblock_name }
    create_pblock $pblock_name

    foreach range $ranges {
        resize_pblock [get_pblocks $pblock_name] -add $range
    }
    # `soft` is kept for the Zynq flow, where IS_SOFT survives. Inside a
    # reconfigurable partition Vivado discards it with a warning per pblock
    # (Constraints 18-5661), so it is only set where no parent region exists.
    if {$soft && [llength [get_pblocks -quiet -filter {NAME =~ "pblock_dynamic*"}]] == 0} {
        set_property IS_SOFT TRUE [get_pblocks $pblock_name]
    }

    # Routing must stay free to leave these regions. CONTAIN_ROUTING exists so a
    # PR region can be swapped at runtime; on a per-layer pblock it would force
    # every net into its own region and make the sites near each boundary
    # unroutable -- with hundreds of adjacent pblocks that is a large fraction of
    # them, and the placer then rejects the region outright ("[Place 30-859] ...
    # exceeds the number of corresponding sites available", with the shortfall
    # attributed to place.closeUnroutableSites). It defaults off; set it
    # explicitly so a changed default or an inherited value cannot turn it on
    # silently.
    set_property CONTAIN_ROUTING FALSE [get_pblocks $pblock_name]
    set_property EXCLUDE_PLACEMENT FALSE [get_pblocks $pblock_name]

    set resolved {}
    set objs {}
    foreach cell $cells {
        set obj [finn_fp_resolve_cell $cell]
        if {$obj eq ""} {
            puts "    WARNING: cell '$cell' not found, not adding it to $pblock_name"
            continue
        }
        lappend resolved $cell
        lappend objs {*}$obj
        add_cells_to_pblock [get_pblocks $pblock_name] $obj
    }
    if {[llength $resolved] == 0} {
        puts "    WARNING: $pblock_name has no cells; it will constrain nothing"
    }
    finn_fp_range_clock_buffers $pblock_name $objs
    return $ranges
}

# Every primitive type inside a reconfigurable partition must be ranged by the
# cell's pblock (HDPR-18) -- including the global clock buffers synthesis
# promotes high-fanout nets to. resnet18's distributed-RAM weight memories carry
# such BUFGs on their read pointers ("No Pblock range for cell ... Ptr1[6]_BUFG_inst
# ... A 'BUFGCE' range needs to be added", 2 Oct 2026). Site counts of the five
# packed kinds say nothing about them, so a unit that contains any BUFG* cell gets
# every BUFG* site of its SLR's region added to its ranges. Clock buffers sit in
# the clock-region spines and are shared freely between pblocks.
proc finn_fp_range_clock_buffers {pblock_name objs} {
    if {[llength $objs] == 0} { return }
    # Hierarchical names start with the unit cell's path, so one filtered
    # hierarchical query per unit cell finds its clock buffers. (`get_cells
    # -of_objects <cell>` is not a thing; with -quiet it silently returns nothing,
    # which is how the first run of this proc ranged no BUFGCE at all.)
    # `objs` holds cell *names*: Vivado objects stringify to their names when
    # they pass through a Tcl list, and `get_property NAME` on such a string is an
    # "Invalid option value ... for 'object'" error (mobilenetv1 link, 2 Oct 2026).
    set bufgs {}
    foreach path $objs {
        lappend bufgs {*}[get_cells -quiet -hierarchical \
                              -filter "REF_NAME =~ BUFG* && NAME =~ $path/*"]
    }
    if {[llength $bufgs] == 0} { return }
    set pb [get_pblocks $pblock_name]
    set scope_sites {}
    foreach scope [get_pblocks -quiet -filter {NAME =~ "pblock_dynamic*"}] {
        lappend scope_sites {*}[get_sites -quiet -of_objects $scope -filter {SITE_TYPE =~ "BUFG*"}]
    }
    if {[llength $scope_sites] == 0} {
        set scope_sites [get_sites -quiet -filter {SITE_TYPE =~ "BUFG*"}]
    }
    set scope_sites [lsort -unique $scope_sites]
    if {[llength $scope_sites] == 0} { return }
    resize_pblock $pb -add $scope_sites
    puts "    $pblock_name holds [llength $bufgs] clock buffer(s); ranged [llength $scope_sites] BUFG* site(s)"
}


proc finn_fp_resolve_cell {cell} {
    set obj [get_cells -quiet $cell]
    if {$obj ne ""} { return $obj }

    # Bare node name: find it anywhere in the hierarchy.
    set obj [get_cells -quiet -hierarchical $cell]
    if {[llength $obj] > 1} {
        puts "    WARNING: '$cell' matches [llength $obj] cells; constraining all of them"
    }
    return $obj
}


################################################################################
# JSON helpers. Vivado's Tcl ships no JSON package we can rely on, and the schema
# here is fixed, so writing it by hand is the robust choice. Reading goes the
# other way: the packer also emits its result as a Tcl dict this script sources.
################################################################################

proc finn_fp_json_str {s} {
    return "\"[string map {\\ \\\\ \" \\\"} $s]\""
}

# Collapse a list of integers into inclusive {lo hi} runs.
proc finn_fp_runs {ys} {
    set runs {}
    set start ""
    set prev ""
    foreach y [lsort -integer -unique $ys] {
        if {$start eq ""} { set start $y; set prev $y; continue }
        if {$y == $prev + 1} { set prev $y; continue }
        lappend runs [list $start $prev]
        set start $y
        set prev $y
    }
    if {$start ne ""} { lappend runs [list $start $prev] }
    return $runs
}

proc finn_fp_json_runs {runs} {
    set out {}
    foreach r $runs { lappend out "\[[lindex $r 0],[lindex $r 1]\]" }
    return "\[[join $out ,]\]"
}

################################################################################
# Census: the device as the packer needs it.
################################################################################

# Clock-region bands per kind: the rows of each site kind that share a clock
# region, measured from the device. Geometry only, so the raw kind filter.
proc finn_fp_measure_bands {} {
    global FINN_FP_KINDS_TABLE finn_fp_cr_bands
    array unset finn_fp_cr_bands
    foreach cr [get_clock_regions -quiet] {
        foreach {kind filter pattern} $FINN_FP_KINDS_TABLE {
            set ys {}
            foreach site [get_sites -quiet -of_objects $cr -filter $filter] {
                if {[regexp $pattern $site -> x y]} { lappend ys $y }
            }
            if {[llength $ys] == 0} { continue }
            set ys [lsort -integer $ys]
            set band [list [lindex $ys 0] [lindex $ys end]]
            if {![info exists finn_fp_cr_bands($kind)]} { set finn_fp_cr_bands($kind) {} }
            if {$band ni $finn_fp_cr_bands($kind)} { lappend finn_fp_cr_bands($kind) $band }
        }
    }
    foreach kind [array names finn_fp_cr_bands] {
        set finn_fp_cr_bands($kind) [lsort -integer -index 0 $finn_fp_cr_bands($kind)]
    }
}

# Write the census JSON: for every SLR and kind, each column with its own X,
# its tile-grid column (shared across kinds, which is what makes rectangles over
# interleaved kinds well defined), whether it is SLICEM, and its available rows
# as runs. Only sites inside the dynamic region that the placer may use count.
proc finn_fp_export_census {path} {
    global FINN_FP_KINDS_TABLE finn_fp_cr_bands

    finn_fp_measure_bands
    set slr_objs [get_slrs -quiet]
    if {[llength $slr_objs] == 0} {
        error "finn_fp_export_census: no SLRs reported. Open a design or link a part first."
    }
    set slrs {}
    set slr_json {}
    foreach slr_obj $slr_objs {
        if {![regexp {SLR(\d+)} [get_property NAME $slr_obj] -> slr]} {
            error "finn_fp_export_census: cannot parse an SLR index from [get_property NAME $slr_obj]"
        }
        lappend slrs $slr
        set scopes [finn_fp_region_scopes $slr]
        set kind_json {}
        set summary {}
        foreach {kind filter pattern} $FINN_FP_KINDS_TABLE {
            set sites [get_sites -quiet -of_objects $slr_obj -filter [finn_fp_site_filter $filter]]
            foreach scope $scopes {
                set sites [finn_fp_intersect_sites $sites \
                               [get_sites -quiet -of_objects $scope -filter [finn_fp_site_filter $filter]]]
            }
            array unset colys
            array unset colsite
            foreach site $sites {
                if {![regexp $pattern $site -> x y]} { continue }
                lappend colys($x) $y
                if {![info exists colsite($x)]} { set colsite($x) $site }
            }
            set cols {}
            foreach x [lsort -integer [array names colys]] {
                set site [get_sites $colsite($x)]
                set tcol [get_property COLUMN [get_tiles -quiet -of_objects $site]]
                set extra ""
                if {$kind eq "SLICE"} {
                    set m [expr {[get_property SITE_TYPE $site] eq "SLICEM" ? "true" : "false"}]
                    set extra ", \"slicem\": $m"
                }
                lappend cols "{\"x\": $x, \"tile_col\": $tcol$extra, \"runs\": [finn_fp_json_runs [finn_fp_runs $colys($x)]]}"
            }
            lappend kind_json "\"$kind\": \[[join $cols ",\n        "]\]"
            lappend summary "[llength $cols] $kind"
        }
        lappend slr_json "\"$slr\": {\n      [join $kind_json ",\n      "]\n    }"
        puts "  SLR${slr}: [join $summary {, }] columns"
    }
    set band_json {}
    foreach kind [lsort [array names finn_fp_cr_bands]] {
        set bl {}
        foreach b $finn_fp_cr_bands($kind) { lappend bl "\[[lindex $b 0],[lindex $b 1]\]" }
        lappend band_json "\"$kind\": \[[join $bl ,]\]"
    }
    foreach kind {RAMB36 RAMB18} {
        if {[info exists finn_fp_cr_bands($kind)]} {
            puts "  $kind clock-region bands: [llength $finn_fp_cr_bands($kind)] of \
[expr {[lindex $finn_fp_cr_bands($kind) 0 1] - [lindex $finn_fp_cr_bands($kind) 0 0] + 1}] rows"
        }
    }
    set part ""
    catch { set part [get_property PART [current_project]] }
    set fh [open $path w]
    puts $fh "{\n  \"version\": 1,\n  \"part\": [finn_fp_json_str $part],\n  \"slrs\": \[[join [lsort -integer $slrs] ,]\],"
    puts $fh "  \"bands\": {[join $band_json ", "]},"
    puts $fh "  \"columns\": {\n    [join $slr_json ",\n    "]\n  }\n}"
    close $fh
    puts "  census written to $path"
    return [lsort -integer $slrs]
}

################################################################################
# Units and feedback as JSON for the packer.
################################################################################

proc finn_fp_json_demands {demands} {
    set parts {}
    foreach key {luts ffs carries brams dsps urams lutrams srls ram_macros} {
        lappend parts "\"$key\": [finn_fp_demand $demands $key]"
    }
    return "{[join $parts {, }]}"
}

proc finn_fp_json_chains {chains} {
    set parts {}
    dict for {kind lens} $chains { lappend parts "\"$kind\": \[[join $lens ,]\]" }
    return "{[join $parts {, }]}"
}

proc finn_fp_export_units {units path} {
    set entries {}
    set order 0
    foreach unit $units {
        set chains {}
        if {[dict exists $unit chains]} { set chains [dict get $unit chains] }
        set cells {}
        if {[dict exists $unit cells]} {
            foreach c [dict get $unit cells] { lappend cells [finn_fp_json_str $c] }
        }
        lappend entries "{\"name\": [finn_fp_json_str [dict get $unit name]], \"slr\": [dict get $unit slr], \
\"order\": $order, \"demands\": [finn_fp_json_demands [dict get $unit demands]], \
\"chains\": [finn_fp_json_chains $chains], \"cells\": \[[join $cells {, }]\]}"
        incr order
    }
    set fh [open $path w]
    puts $fh "\[\n[join $entries ",\n"]\n\]"
    close $fh
}

proc finn_fp_export_feedback {feedback path} {
    set entries {}
    dict for {name missing} $feedback {
        set parts {}
        dict for {key value} $missing { lappend parts "[finn_fp_json_str $key]: $value" }
        lappend entries "[finn_fp_json_str $name]: {[join $parts {, }]}"
    }
    set fh [open $path w]
    puts $fh "{[join $entries ,]}"
    close $fh
}

################################################################################
# Running the packer.
################################################################################

proc finn_fp_run_packer {round census units out out_tcl {previous {}} {feedback {}}} {
    global FINN_FP_PYTHON FINN_FP_PYTHON_ENV FINN_FP_PACKER FINN_FP_PACK_ORDER
    global FINN_FP_SLICEM_UTIL FINN_FP_SRL_PER_SLICE FINN_FP_LOGIC_UTIL

    set cmd [list $FINN_FP_PYTHON $FINN_FP_PACKER --census $census --units $units \
                 --out $out --out-tcl $out_tcl --round $round --order $FINN_FP_PACK_ORDER \
                 --slicem-util $FINN_FP_SLICEM_UTIL --srl-per-slice $FINN_FP_SRL_PER_SLICE \
                 --logic-util $FINN_FP_LOGIC_UTIL]
    if {$previous ne "" && [file exists $previous]} { lappend cmd --previous $previous }
    if {$feedback ne "" && [file exists $feedback]} { lappend cmd --feedback $feedback }

    # A scrubbed environment: only what FINN whitelisted reaches the interpreter.
    # Vivado's own process carries PYTHONHOME and PYTHONPATH pointing at its
    # bundled Python 3.8 and an LD_LIBRARY_PATH of its own libraries; any of the
    # three breaks a conda interpreter at startup ("Python path configuration:
    # PYTHONHOME = .../Vivado/2024.2/tps/lnx64/python-3.8.3"). A hook rendered
    # without a whitelist (a bare `vivado -source`, or an older renderer) gets the
    # minimum instead: PATH and HOME from Vivado's environment, nothing else.
    set env_args {}
    foreach {k v} $FINN_FP_PYTHON_ENV { lappend env_args "$k=$v" }
    if {[llength $env_args] == 0} {
        foreach k {PATH HOME} {
            if {[info exists ::env($k)]} { lappend env_args "$k=$::env($k)" }
        }
        lappend env_args "LANG=C.UTF-8"
    }
    set launcher /usr/bin/env
    if {![file executable $launcher]} { set launcher env }

    puts "  packer round $round: $FINN_FP_PYTHON $FINN_FP_PACKER"
    if {[catch {set output [exec $launcher -i {*}$env_args {*}$cmd 2>@1]} err]} {
        error "FINN+ floorplanner: the packer failed:\n$err"
    }
    foreach line [split $output "\n"] { if {$line ne ""} { puts "    $line" } }
    if {![file exists $out_tcl]} {
        error "FINN+ floorplanner: the packer wrote no result at $out_tcl"
    }
    source $out_tcl
    return $::finn_fp_result
}

################################################################################
# Verify an applied pblock against the sites Vivado actually gave it. Returns
# {ok capacity missing}; `missing` holds the shortfall per demand key, or
# chains:<KIND> 1 when the chains of that kind find no runs.
################################################################################

proc finn_fp_verify_capacity {pblock_name demands {chains {}}} {
    global FINN_FP_SLICE_LUTS FINN_FP_SLICE_FFS FINN_FP_SLICE_CARRY
    global FINN_FP_SLICEM_UTIL FINN_FP_SRL_PER_SLICE FINN_FP_LOGIC_UTIL

    set pb [get_pblocks $pblock_name]
    set slices  [llength [get_sites -quiet -of_objects $pb \
                     -filter [finn_fp_site_filter {SITE_TYPE =~ "SLICE*"}]]]
    set brams   [llength [get_sites -quiet -of_objects $pb \
                     -filter [finn_fp_site_filter {NAME =~ "RAMB36_*"}]]]
    set dsps    [llength [get_sites -quiet -of_objects $pb \
                     -filter [finn_fp_site_filter {SITE_TYPE == "DSP48E2"}]]]
    set urams   [llength [get_sites -quiet -of_objects $pb \
                     -filter [finn_fp_site_filter {NAME =~ "URAM288_*"}]]]
    set slicems [llength [get_sites -quiet -of_objects $pb \
                     -filter [finn_fp_site_filter {SITE_TYPE == "SLICEM"}]]]

    set capacity [dict create \
        luts    [expr {$slices * $FINN_FP_SLICE_LUTS}] \
        ffs     [expr {$slices * $FINN_FP_SLICE_FFS}] \
        carries [expr {$slices * $FINN_FP_SLICE_CARRY}] \
        brams   $brams dsps $dsps urams $urams \
        lutrams [expr {int($slicems * $FINN_FP_SLICE_LUTS * $FINN_FP_SLICEM_UTIL)}] \
        srls    [expr {int($slicems * $FINN_FP_SLICE_LUTS * $FINN_FP_SLICEM_UTIL)}]]

    set missing [dict create]
    foreach key {ffs carries brams dsps urams} {
        set need [finn_fp_demand $demands $key]
        set have [dict get $capacity $key]
        # Flip-flops and carries are plain logic: they fill a pblock only up to the
        # logic ceiling, like logic LUTs below (same rule as the packer).
        if {$key in {ffs carries}} { set have [expr {int($have * $FINN_FP_LOGIC_UTIL)}] }
        if {$need > $have} { dict set missing $key [expr {$need - $have}] }
    }
    # SLICEM in slices: distributed-RAM macros (or, without a macro count, their
    # LUTs) under the utilisation ceiling, shift registers at their measured
    # packing density; logic LUTs fill whole slices of their own on top. Same
    # rule as the packer (Unit.slicem_slices / Unit.slices_needed).
    set ram_slices [expr {int(ceil([finn_fp_demand $demands ram_macros] / double($FINN_FP_SLICEM_UTIL)))}]
    if {$ram_slices <= 0} {
        set ram_slices [expr {int(ceil([finn_fp_demand $demands lutrams] \
                                       / ($FINN_FP_SLICE_LUTS * $FINN_FP_SLICEM_UTIL)))}]
    }
    set srl_slices [expr {int(ceil([finn_fp_demand $demands srls] / $FINN_FP_SRL_PER_SLICE))}]
    set need_slices [expr {$ram_slices + $srl_slices}]
    set logic_luts [expr {max(0, [finn_fp_demand $demands luts] - [finn_fp_demand $demands lutrams] \
                                 - [finn_fp_demand $demands srls])}]
    set all_slices [expr {$need_slices \
        + int(ceil($logic_luts / double($FINN_FP_SLICE_LUTS * $FINN_FP_LOGIC_UTIL)))}]
    if {$all_slices > $slices} {
        dict set missing luts [expr {($all_slices - $slices) * $FINN_FP_SLICE_LUTS}]
    }
    if {$need_slices > $slicems} {
        # Reported in LUTRAM LUTs so the override channel (which adds to demands)
        # moves the unit by the right amount.
        dict set missing lutrams [expr {int(($need_slices - $slicems) * $FINN_FP_SLICE_LUTS * $FINN_FP_SLICEM_UTIL)}]
    }
    # Every chain needs its own run of contiguous available sites in one column
    # (and one clock region, for block RAM). The packer sized for them from the
    # census; this checks the region Vivado actually built.
    foreach kind {RAMB36 DSP48E2 URAM288} {
        set lens [finn_fp_chain_lens_for $chains $kind]
        if {[llength $lens] == 0} { continue }
        if {![finn_fp_chains_pack [finn_fp_site_runs $pb $kind] $lens]} {
            dict set missing "chains:$kind" 1
        }
    }
    return [list [expr {[dict size $missing] == 0}] $capacity $missing]
}

################################################################################
# Apply one round's placements, verify them, collect the feedback.
################################################################################

proc finn_fp_apply_round {units result soft round} {
    global finn_fp_report FINN_FP_MAX_ROUNDS

    set feedback [dict create]
    set applied 0
    foreach unit $units {
        set name [dict get $unit name]
        if {![dict exists $result $name]} { continue }
        set p [dict get $result $name]
        if {[dict get $p status] ne "placed"} { continue }
        # A unit kept from an earlier round keeps its verified pblock.
        if {[dict get $p round] != $round && [dict exists $finn_fp_report $name]} { continue }

        set chains {}
        if {[dict exists $unit chains]} { set chains [dict get $unit chains] }
        finn_fp_apply_pblock $name [dict get $unit cells] [dict get $p ranges] $soft
        lassign [finn_fp_verify_capacity $name [dict get $unit demands] $chains] ok capacity missing
        incr applied
        if {$ok} {
            dict set finn_fp_report $name [dict create \
                slr [dict get $unit slr] ranges [dict get $p ranges] capacity $capacity \
                satisfied 1 round $round]
            continue
        }
        puts "    $name: pblock short by $missing; re-packing"
        delete_pblocks $name
        dict unset finn_fp_report $name
        dict set feedback $name $missing
    }
    # Pblocks Vivado finds too small for a reconfigurable region (HDPRA-65): grow
    # each by the factor it falls short of the minimum Programmable Unit count.
    dict for {name grow} [finn_fp_small_pblocks] {
        if {![dict exists $finn_fp_report $name]} { continue }
        if {$round >= $FINN_FP_MAX_ROUNDS} {
            # No round left to re-pack in: a small pblock beats none.
            puts "    WARNING: $name still below the minimum Programmable Units; kept"
            continue
        }
        set cap [dict get $finn_fp_report $name capacity]
        set extra [expr {int(ceil([dict get $cap luts] * $grow)) + 1}]
        puts "    $name: below the minimum Programmable Units; re-packing (+$extra LUTs)"
        delete_pblocks $name
        dict unset finn_fp_report $name
        dict set feedback $name [dict create luts $extra]
    }
    puts "  round $round: $applied pblock(s) applied, [dict size $feedback] came up short"
    return $feedback
}

# Floorplan pblocks Vivado reports as too small for a reconfigurable region
# (DRC HDPRA-65, "... nested Pblock '<name>' that is too small (<n> Programmable
# Units)"), as {name growth}: the fraction by which the pblock has to grow to
# reach FINN_FP_MIN_PROGRAMMABLE_UNITS. Vivado's own count is the measure; a
# Programmable Unit has no fixed slice size (a 5-slice and a 10-slice pblock both
# count 3 on the U55C).
proc finn_fp_small_pblocks {} {
    global FINN_FP_MIN_PROGRAMMABLE_UNITS
    set out [dict create]
    if {[catch {report_drc -checks {HDPRA-65} -return_string} rpt]} { return $out }
    foreach {- name n} [regexp -all -inline \
            {nested Pblock '(pblock_finn_unit_[^']+)' that is too small \((\d+) Programmable Units\)} $rpt] {
        if {$n < $FINN_FP_MIN_PROGRAMMABLE_UNITS && $n > 0} {
            dict set out $name [expr {double($FINN_FP_MIN_PROGRAMMABLE_UNITS) / $n - 1.0}]
        }
    }
    return $out
}

################################################################################
# Entry point.
################################################################################

# Mark the shell's closed sites PROHIBIT, so the census, the verify step and
# the placer all see the same, smaller device. A closed RAMB36 site takes the
# two RAMB18 sites of its tile with it. Returns the number of sites prohibited.
proc finn_fp_prohibit_closed_sites {} {
    global FINN_FP_CLOSED_SITES FINN_FP_CLOSED_SITES_VIVADO FINN_FP_CLOSED_SITES_SOURCE
    global finn_fp_shell_closure
    set finn_fp_shell_closure [dict create source $FINN_FP_CLOSED_SITES_SOURCE \
        vivado $FINN_FP_CLOSED_SITES_VIVADO listed [llength $FINN_FP_CLOSED_SITES] prohibited 0]
    if {[llength $FINN_FP_CLOSED_SITES] == 0} {
        if {$FINN_FP_CLOSED_SITES_SOURCE ne ""} {
            puts "  shell closure: $FINN_FP_CLOSED_SITES_SOURCE"
        }
        return 0
    }
    if {$FINN_FP_CLOSED_SITES_VIVADO ne "" && $FINN_FP_CLOSED_SITES_VIVADO ne [version -short]} {
        puts "  WARNING: shell closure map measured under Vivado $FINN_FP_CLOSED_SITES_VIVADO,\
running [version -short]; re-map the platform if placement rejects pblocks"
    }
    set sites [get_sites -quiet $FINN_FP_CLOSED_SITES]
    if {[llength $sites] != [llength $FINN_FP_CLOSED_SITES]} {
        puts "  WARNING: [expr {[llength $FINN_FP_CLOSED_SITES] - [llength $sites]}] closed\
site(s) of the shell closure map do not exist on this part"
    }
    set bram36 [get_sites -quiet $sites -filter {NAME =~ "RAMB36_*"}]
    set halves {}
    if {[llength $bram36]} {
        set halves [get_sites -quiet -of_objects [get_tiles -of_objects $bram36] \
            -filter {NAME =~ "RAMB18_*"}]
    }
    set all [concat $sites $halves]
    if {[llength $all]} { set_property PROHIBIT 1 $all }
    dict set finn_fp_shell_closure prohibited [llength $all]
    puts "  shell closure: [llength $sites] closed site(s) and [llength $halves] RAMB18\
half(s) prohibited ($FINN_FP_CLOSED_SITES_SOURCE)"
    return [llength $all]
}


proc finn_fp_floorplan_units {units {soft 1} {measure 1} {margin 1.1} {overrides_file {}}} {
    global finn_fp_report finn_fp_resource_source FINN_FP_MAX_ROUNDS FINN_FP_WORK_DIR

    puts "================================================================"
    puts " FINN+ automatic pblock floorplanning ([llength $units] unit(s))"
    puts "================================================================"

    set work $FINN_FP_WORK_DIR
    if {$work eq ""} {
        if {$overrides_file ne ""} { set work [file dirname $overrides_file] } else { set work [pwd] }
    }
    file mkdir $work

    finn_fp_report_parent_containment
    finn_fp_apply_placer_params
    finn_fp_prohibit_closed_sites
    set census [file join $work finn_floorplan_census.json]
    set measured_tcl [file join $work finn_floorplan_measured.tcl]

    # A resumed attempt runs on the same optimised netlist, so the census and
    # the measured units of the first attempt are read back instead of taken
    # again (about three minutes).
    set reuse [finn_fp_resumed_attempt]
    if {$reuse} {
        foreach f [list $census $measured_tcl] {
            if {![file exists $f]} {
                error "FINN_FP_RESUMED_ATTEMPT is set but $f from the first attempt is missing"
            }
        }
        source $measured_tcl
        set units $finn_fp_measured_units
        set finn_fp_resource_source "post_synth"
        puts "  census and measured units reused from the first attempt ($measured_tcl)"
    } else {
        finn_fp_export_census $census
    }

    # Size from what the design actually uses: the hook runs on the optimised
    # netlist, so there is nothing to estimate.
    if {$measure && !$reuse} {
        set units [finn_fp_measure_units $units $margin]
        set finn_fp_resource_source "post_synth"
        set chains [finn_fp_measure_chains $units]
        set prims [finn_fp_measure_primitives $units]
        set with_chains {}
        foreach unit $units {
            set name [dict get $unit name]
            set c {}
            if {[dict exists $chains $name]} { set c [dict get $chains $name] }
            set demands [dict get $unit demands]
            if {[dict exists $prims $name]} {
                set p [dict get $prims $name]
                dict set demands ram_macros [dict get $p ram_macros]
                dict set demands carries [expr {max([finn_fp_demand $demands carries], [dict get $p carries])}]
            }
            lappend with_chains [dict replace [dict replace $unit chains $c] demands $demands]
        }
        set units $with_chains
        set n_macro 0; set n_carry 0
        dict for {u p} $prims {
            if {[dict get $p ram_macros] > 0} { incr n_macro }
            if {[dict get $p carries] > 0} { incr n_carry }
        }
        puts "  $n_macro unit(s) hold distributed-RAM macros, $n_carry unit(s) carry chains of CARRY8"
        puts "  [dict size $chains] unit(s) carry cascade chains"
        dict for {u kinds} $chains { puts "    chains: $u $kinds" }
        # Snapshot for a resumed attempt (see finn_fp_resumed_attempt): the units
        # with their measured demands and chains, before any override.
        set fh [open $measured_tcl w]
        puts $fh "set finn_fp_measured_units [list $units]"
        close $fh
    } elseif {!$reuse} {
        set finn_fp_resource_source "estimate"
    }
    set overrides [finn_fp_load_overrides $overrides_file]
    set previous ""
    set feedback_json ""
    set final_pack [file join $work finn_floorplan_pack_final.json]
    if {$reuse} {
        # A resumed attempt keeps every placement of the attempt before it and
        # re-packs only the units the placer rejected, with their accumulated
        # shortfalls as feedback -- the mechanism a later round uses within one
        # attempt. Packing everything again from the grown demands moved every
        # unit, and with ~16 five-slice units each carrying a few percent chance
        # of failing legalisation, a run on the strip packer lost a different
        # tiny unit on each of four attempts (13 Oct 2026).
        if {![file exists $final_pack]} {
            error "FINN_FP_RESUMED_ATTEMPT is set but $final_pack from the previous attempt is missing"
        }
        if {[dict size $overrides] == 0} {
            error "FINN_FP_RESUMED_ATTEMPT is set but $overrides_file names no rejected unit"
        }
        set previous $final_pack
        set feedback_json [file join $work finn_floorplan_feedback_retry.json]
        finn_fp_export_feedback $overrides $feedback_json
        puts "  resumed attempt: re-packing [dict size $overrides] rejected unit(s) with their shortfalls ([dict keys $overrides]); every other placement is kept from $previous"
    } else {
        set units [finn_fp_apply_overrides $units $overrides]
    }

    set units_json [file join $work finn_floorplan_units.json]
    finn_fp_export_units $units $units_json

    set finn_fp_report [dict create]
    set result {}
    for {set round 1} {$round <= $FINN_FP_MAX_ROUNDS} {incr round} {
        set out     [file join $work "finn_floorplan_pack_${round}.json"]
        set out_tcl [file join $work "finn_floorplan_pack_${round}.tcl"]
        set result [finn_fp_run_packer $round $census $units_json $out $out_tcl $previous $feedback_json]
        set feedback [finn_fp_apply_round $units $result $soft $round]
        set previous $out
        if {[dict size $feedback] == 0} { break }
        set feedback_json [file join $work "finn_floorplan_feedback_${round}.json"]
        finn_fp_export_feedback $feedback $feedback_json
    }
    # The attempt's final placements, for a resumed attempt to keep (see above).
    if {$previous ne "" && $previous ne $final_pack} { file copy -force $previous $final_pack }

    set unplaced {}
    foreach unit $units {
        set name [dict get $unit name]
        if {[dict exists $finn_fp_report $name]} { continue }
        if {[get_pblocks -quiet $name] ne ""} { delete_pblocks $name }
        set why "not placed"
        if {[dict exists $result $name]} {
            set p [dict get $result $name]
            if {[dict get $p status] eq "placed"} {
                set why "pblock still short after $FINN_FP_MAX_ROUNDS rounds"
            } else {
                set why [dict get $p reason]
            }
        }
        puts "    WARNING: $name left unconstrained ($why)"
        lappend unplaced $name
    }

    set n_placed [dict size $finn_fp_report]
    puts "----------------------------------------------------------------"
    puts " Floorplanning done: $n_placed of [llength $units] unit(s) constrained"
    puts "================================================================"
    # For diagnosis: the floorplanned, not yet placed design, so a placement
    # failure can be reproduced offline (open_checkpoint; place_design) and its
    # unplaced cells inspected -- the placer does not always name them.
    if {[info exists ::env(FINN_FP_PREPLACE_CHECKPOINT)] && $::env(FINN_FP_PREPLACE_CHECKPOINT) ne ""} {
        write_checkpoint -force $::env(FINN_FP_PREPLACE_CHECKPOINT)
        puts "  pre-place checkpoint written to $::env(FINN_FP_PREPLACE_CHECKPOINT)"
    }
    return $finn_fp_report
}

# Cascade-in pins per primitive; a pin tied to a constant is not a link.
################################################################################

set FINN_FP_CASCADE_KINDS {
    RAMB36  RAMB36E2 {CASDINA[0] CASDINB[0]}
    RAMB18  RAMB18E2 {CASDINA[0] CASDINB[0]}
    DSP48E2 DSP48E2  {PCIN[0] ACIN[0] BCIN[0] CARRYCASCIN MULTSIGNIN}
    URAM288 URAM288  {CAS_IN_ADDR_A[0] CAS_IN_ADDR_B[0]}
}

# Position of every cell in its chain, 1 being the head. Cells not chained to
# anything are depth 1, so the longest chain is simply the maximum depth.

# Predecessor of every chained cell: the member whose cascade output drives
# this cell's cascade input. Cells with no chained predecessor are heads.
proc finn_fp_chain_preds {cells in_pins} {
    set member [dict create]
    foreach c $cells { dict set member $c 1 }
    set pred [dict create]
    foreach c $cells {
        foreach pn $in_pins {
            set pin [get_pins -quiet "$c/$pn"]
            if {$pin eq ""} { continue }
            set net [get_nets -quiet -of_objects $pin]
            if {$net eq "" || [string match {*<const*} $net]} { continue }
            foreach dp [get_pins -quiet -of_objects $net -filter {DIRECTION == OUT}] {
                set dc [get_cells -quiet -of_objects $dp]
                if {$dc ne $c && [dict exists $member $dc]} {
                    dict set pred $c $dc
                    break
                }
            }
            if {[dict exists $pred $c]} { break }
        }
    }
    return $pred
}


# Position of every cell in its chain, 1 being the head. Cells not chained to
# anything are depth 1, so the longest chain is simply the maximum depth.
proc finn_fp_chain_depths {cells in_pins} {
    set pred [finn_fp_chain_preds $cells $in_pins]
    set depth [dict create]
    foreach c $cells {
        set walk {}
        set x $c
        while {![dict exists $depth $x] && [dict exists $pred $x]} {
            lappend walk $x
            set x [dict get $pred $x]
        }
        if {![dict exists $depth $x]} { dict set depth $x 1 }
        set d [dict get $depth $x]
        foreach w [lreverse $walk] {
            incr d
            dict set depth $w $d
        }
    }
    return $depth
}


# Length of every chain with at least two members, keyed by its head cell.
#
# The longest chain alone is not enough: every chain needs its own column
# segment, so a unit with seventeen 8-deep chains needs seventeen segments of
# eight rows, not one. mobilenetv1's last MVAU (2 Oct 2026): 113 RAMB36 in 17
# chains of up to 8; an 11-column, 12-row window "fit" the longest chain and
# the placer rejected it ([Constraints 18-11842] cascade length exceeding BRAM
# column height in constraint area).
proc finn_fp_chain_lengths {cells in_pins} {
    set pred [finn_fp_chain_preds $cells $in_pins]
    set root [dict create]
    foreach c $cells {
        set walk {}
        set x $c
        while {![dict exists $root $x] && [dict exists $pred $x]} {
            lappend walk $x
            set x [dict get $pred $x]
        }
        set r [expr {[dict exists $root $x] ? [dict get $root $x] : $x}]
        dict set root $x $r
        foreach w $walk { dict set root $w $r }
    }
    set count [dict create]
    foreach c $cells { dict incr count [dict get $root $c] }
    set lengths [dict create]
    dict for {r n} $count { if {$n >= 2} { dict set lengths $r $n } }
    return $lengths
}


# Primitives the utilisation report does not break out per instance, counted
# from the netlist: {unit_name {ram_macros n carries n}}.
#
# ram_macros: distributed-RAM macros (REF_NAME RAM<digit>..., e.g. RAM64M8,
#   RAM64X1D); each occupies a SLICEM slice of its own, whatever its LUT count.
#   The leaf RAMD64E/RAMS64E cells inside a macro and block RAM (RAMB*) do not
#   match the pattern.
# carries: CARRY8 primitives, one per slice. The hierarchical utilisation report
#   has no carry column at all, so without this count every unit's carry demand
#   read zero (mobilenetv1, 3 Oct 2026: two input generators failed to place
#   their address counters' CARRY8s).
# A primitive belongs to the unit whose cell is its nearest ancestor.
proc finn_fp_measure_primitives {units} {
    set owner [dict create]
    foreach unit $units {
        foreach cell [dict get $unit cells] {
            foreach obj [finn_fp_resolve_cell $cell] {
                dict set owner [get_property NAME $obj] [dict get $unit name]
            }
        }
    }
    set counts [dict create]
    foreach {key filter pattern} {
        ram_macros {REF_NAME =~ "RAM*"} {^RAM\d}
        carries    {REF_NAME == "CARRY8"} {^CARRY8$}
    } {
        foreach c [get_cells -quiet -hierarchical -filter $filter] {
            if {![regexp $pattern [get_property REF_NAME $c]]} { continue }
            set parts [split $c /]
            for {set i [expr {[llength $parts] - 1}]} {$i > 0} {incr i -1} {
                set anc [join [lrange $parts 0 [expr {$i - 1}]] /]
                if {[dict exists $owner $anc]} {
                    set u [dict get $owner $anc]
                    if {![dict exists $counts $u]} { dict set counts $u [dict create ram_macros 0 carries 0] }
                    dict set counts $u $key [expr {[dict get $counts $u $key] + 1}]
                    break
                }
            }
        }
    }
    return $counts
}

# Longest chain per unit and kind: {unit_name {RAMB36 n RAMB18 n ...}}.
#
# A primitive belongs to the unit whose cell is its nearest ancestor, found by
# walking up the primitive's hierarchical name -- one dict lookup per level, so
# attribution stays cheap on hundreds of thousands of cells.
proc finn_fp_measure_chains {units} {
    global FINN_FP_CASCADE_KINDS

    set owner [dict create]
    foreach unit $units {
        foreach cell [dict get $unit cells] {
            foreach obj [finn_fp_resolve_cell $cell] {
                dict set owner [get_property NAME $obj] [dict get $unit name]
            }
        }
    }

    set chains [dict create]
    foreach {kind ref in_pins} $FINN_FP_CASCADE_KINDS {
        set cells [get_cells -quiet -hierarchical -filter "REF_NAME == $ref"]
        if {[llength $cells] == 0} { continue }
        dict for {head len} [finn_fp_chain_lengths $cells $in_pins] {
            set parts [split $head /]
            for {set i [expr {[llength $parts] - 1}]} {$i > 0} {incr i -1} {
                set anc [join [lrange $parts 0 [expr {$i - 1}]] /]
                if {[dict exists $owner $anc]} {
                    set u [dict get $owner $anc]
                    set cur {}
                    if {[dict exists $chains $u $kind]} { set cur [dict get $chains $u $kind] }
                    lappend cur $len
                    dict set chains $u $kind [lsort -integer -decreasing $cur]
                    break
                }
            }
        }
    }
    return $chains
}

# Kinds whose cascades must stay inside one clock region. Vivado rejects a block
# RAM cascade that a constraint forces across a clock-region boundary when an
# upper member uses the CASDIMUX/CASOREGIMUX pins ("[Constraints 18-5197] ...
# Bram Cascade Over-constrained", a hard placer failure). The resnet18 run of
# 2 Oct 2026 hit it with a 5-deep chain whose RAMB36 window was Y116..120, i.e.
# four rows in one region and one in the next. DSP and URAM cascades carry no
# such rule, so for them any contiguous run of rows will do.
set FINN_FP_CR_BOUND_KINDS {RAMB36 RAMB18}

# The clock-region band {lo hi} (in rows of `kind`) that row y falls in, or {}
# when the device's bands are not known (bare tclsh, or a part without clock
# regions), in which case the whole extent is treated as one band.
proc finn_fp_cr_band {kind y} {
    global finn_fp_cr_bands
    if {![info exists finn_fp_cr_bands($kind)]} { return {} }
    foreach band $finn_fp_cr_bands($kind) {
        lassign $band lo hi
        if {$y >= $lo && $y <= $hi} { return $band }
    }
    return {}
}


# Can every chain get its own run of rows? Best-fit decreasing: longest chain
# first, into the smallest segment that still holds it. A segment may host
# several short chains stacked in one column.
proc finn_fp_chains_pack {segs lens} {
    foreach len [lsort -integer -decreasing $lens] {
        set pick -1
        set pick_size 0
        for {set i 0} {$i < [llength $segs]} {incr i} {
            set sz [lindex $segs $i]
            if {$sz >= $len && ($pick < 0 || $sz < $pick_size)} {
                set pick $i
                set pick_size $sz
            }
        }
        if {$pick < 0} { return 0 }
        lset segs $pick [expr {$pick_size - $len}]
    }
    return 1
}


# Chain lengths a kind's columns must host, folding RAMB18 chains onto the
# RAMB36 columns they physically share (two RAMB18 rows per RAMB36 row).
proc finn_fp_chain_lens_for {chains kind} {
    set lens {}
    if {[dict exists $chains $kind]} { set lens [dict get $chains $kind] }
    if {$kind eq "RAMB36" && [dict exists $chains RAMB18]} {
        foreach len [dict get $chains RAMB18] { lappend lens [expr {($len + 1) / 2}] }
    }
    return $lens
}


# Every run of vertically contiguous *available* sites of one kind in an applied
# pblock, one entry per run, split at clock-region bands for the kinds whose
# cascades may not cross one -- what the chains actually get, as opposed to the
# census estimate, which cannot see holes left by prohibited sites.
proc finn_fp_site_runs {pb kind} {
    global FINN_FP_CR_BOUND_KINDS
    switch -- $kind {
        RAMB36  { set f {NAME =~ "RAMB36_*"};    set pat {RAMB36_X(\d+)Y(\d+)} }
        RAMB18  { set f {NAME =~ "RAMB18_*"};    set pat {RAMB18_X(\d+)Y(\d+)} }
        DSP48E2 { set f {SITE_TYPE == "DSP48E2"}; set pat {DSP48E2_X(\d+)Y(\d+)} }
        URAM288 { set f {NAME =~ "URAM288_*"};   set pat {URAM288_X(\d+)Y(\d+)} }
        default { return {} }
    }
    set byx [dict create]
    foreach site [get_sites -quiet -of_objects $pb -filter [finn_fp_site_filter $f]] {
        if {[regexp $pat $site -> x y]} { dict lappend byx $x $y }
    }
    set bound [expr {$kind in $FINN_FP_CR_BOUND_KINDS}]
    set runs {}
    dict for {x ys} $byx {
        set prev ""
        set run 0
        foreach y [lsort -integer -unique $ys] {
            set contiguous [expr {$prev ne "" && $y == $prev + 1}]
            if {$contiguous && $bound &&
                [finn_fp_cr_band $kind $y] ne [finn_fp_cr_band $kind $prev]} {
                set contiguous 0
            }
            if {$contiguous} {
                incr run
            } else {
                if {$run > 0} { lappend runs $run }
                set run 1
            }
            set prev $y
        }
        if {$run > 0} { lappend runs $run }
    }
    return $runs
}


proc finn_fp_longest_run {pb kind} {
    set best 0
    foreach r [finn_fp_site_runs $pb $kind] { if {$r > $best} { set best $r } }
    return $best
}


proc finn_fp_load_overrides {path} {
    if {$path eq "" || ![file exists $path]} { return {} }
    set finn_fp_overrides {}
    source $path
    puts "  Loaded placement overrides for [dict size $finn_fp_overrides] unit(s) from $path"
    return $finn_fp_overrides
}


proc finn_fp_apply_overrides {units overrides} {
    if {[dict size $overrides] == 0} { return $units }
    set out {}
    foreach unit $units {
        set name [dict get $unit name]
        if {[dict exists $overrides $name]} {
            set demands [dict get $unit demands]
            dict for {key extra} [dict get $overrides $name] {
                dict set demands $key [expr {[finn_fp_demand $demands $key] + $extra}]
            }
            set unit [dict replace $unit demands $demands]
            puts "    $name: demand raised by [dict get $overrides $name] after placer rejection"
        }
        lappend out $unit
    }
    return $out
}


proc finn_fp_measure_units {units margin} {
    set all_cells {}
    foreach unit $units {
        foreach cell [dict get $unit cells] { lappend all_cells $cell }
    }
    if {[llength $all_cells] == 0} { return $units }

    puts "  Measuring utilisation for [llength $all_cells] cell(s)"
    set measured [finn_fp_query_cells_batch $all_cells]

    set sized {}
    set fell_back 0
    foreach unit $units {
        set name    [dict get $unit name]
        set demands [dict get $unit demands]

        set actual [finn_fp_zero_demands]
        set found 0
        foreach cell [dict get $unit cells] {
            if {![dict exists $measured $cell]} { continue }
            set cell_res [dict get $measured $cell]
            dict for {key value} $cell_res {
                if {$value > 0} { set found 1 }
                dict set actual $key [expr {[dict get $actual $key] + $value}]
            }
        }

        if {!$found} {
            # Nothing of this unit appeared in the report. Keep the estimate rather
            # than sizing it to nothing.
            incr fell_back
            lappend sized $unit
            continue
        }

        set scaled [dict create]
        dict for {key value} $actual {
            dict set scaled $key [expr {int(ceil($value * $margin))}]
        }
        if {[dict get $scaled luts] < 1} { dict set scaled luts 1 }
        lappend sized [dict create \
            name $name cells [dict get $unit cells] slr [dict get $unit slr] demands $scaled]
    }

    if {$fell_back > 0} {
        puts "  WARNING: $fell_back unit(s) were not found in the utilisation report; \
their estimates were kept"
    }
    return $sized
}


proc finn_fp_query_cells_batch {cell_paths {depth 8}} {
    set result [dict create]
    foreach cell $cell_paths { dict set result $cell [finn_fp_zero_demands] }
    if {[llength $cell_paths] == 0} { return $result }

    # Map leaf name -> requested cell, so a caller may pass either form.
    array set wanted {}
    foreach cell $cell_paths { set wanted([file tail $cell]) $cell }

    set report [report_utilization -hierarchical -hierarchical_depth $depth -return_string]

    set columns {}
    set parsed 0
    foreach line [split $report "\n"] {
        set line [string trim $line]
        if {![string match "|*" $line]} { continue }
        set fields {}
        foreach field [split [string trim $line "|"] "|"] {
            lappend fields [string trim $field]
        }
        if {[llength $fields] < 2} { continue }

        if {[llength $columns] == 0} {
            if {[lsearch -exact $fields "Instance"] >= 0} { set columns $fields }
            continue
        }

        set instance [lindex $fields [lsearch -exact $columns "Instance"]]
        if {$instance eq "" || ![info exists wanted($instance)]} { continue }

        # BRAM is split across RAMB36 and RAMB18 columns; the packer counts RAMB36
        # sites, and two RAMB18s occupy one.
        set ramb36 [finn_fp_column_value $columns $fields {"RAMB36" "Block RAM Tile" "BRAM"}]
        set ramb18 [finn_fp_column_value $columns $fields {"RAMB18"}]
        set brams  [expr {$ramb36 + int(ceil(double($ramb18) / 2.0))}]

        set row [dict create \
            luts    [finn_fp_column_value $columns $fields {"Total LUTs" "CLB LUTs" "Slice LUTs"}] \
            ffs     [finn_fp_column_value $columns $fields {"FFs" "CLB Registers" "Slice Registers"}] \
            carries [finn_fp_column_value $columns $fields {"CARRY8" "Carry8"}] \
            brams   $brams \
            dsps    [finn_fp_column_value $columns $fields {"DSP Blocks" "DSPs" "DSP48 Blocks"}] \
            urams   [finn_fp_column_value $columns $fields {"URAM" "URAM288"}] \
            lutrams [finn_fp_column_value $columns $fields {"LUTRAMs" "LUT as Distributed RAM"}] \
            srls    [finn_fp_column_value $columns $fields {"SRLs" "LUT as Shift Register"}]]

        # Several rows can share one requested leaf name (a wrapper and the module
        # inside it, say). Keep the largest value per field rather than letting the
        # last row seen overwrite a bigger earlier one and under-size the pblock.
        set merged [dict get $result $wanted($instance)]
        dict for {field value} $row {
            if {$value > [dict get $merged $field]} { dict set merged $field $value }
        }
        dict set result $wanted($instance) $merged
        incr parsed
    }

    if {$parsed == 0} {
        puts "  Hierarchical utilisation table had no rows for the requested cells; \
falling back to one call per cell"
        foreach cell $cell_paths {
            set obj [finn_fp_resolve_cell $cell]
            if {$obj eq ""} {
                puts "  WARNING: cell '$cell' not found; reporting zeros for it"
                continue
            }
            dict set result $cell [finn_fp_query_cell_single $obj]
        }
    } else {
        puts "  Parsed hierarchical utilisation for $parsed of [llength $cell_paths] cell(s)"
    }
    return $result
}


proc finn_fp_zero_demands {} {
    return [dict create luts 0 ffs 0 carries 0 brams 0 dsps 0 urams 0 lutrams 0 srls 0 ram_macros 0]
}


proc finn_fp_column_value {columns fields candidates} {
    foreach candidate $candidates {
        set idx [lsearch -exact $columns $candidate]
        if {$idx < 0} { continue }
        set value [lindex $fields $idx]
        if {[string is integer -strict $value]} { return $value }
    }
    return 0
}


proc finn_fp_query_cell_single {cell_obj} {
    set report [report_utilization -cells $cell_obj -return_string]

    set luts 0; set ffs 0; set carries 0; set dsps 0
    set bram_tiles 0; set ramb36 0; set ramb18 0

    foreach line [split $report "\n"] {
        set line [string trim $line]
        if {[regexp {^\|\s+CLB LUTs\s*\*?\s*\|\s+(\d+)\s*\|} $line -> v]} { set luts $v }
        if {$luts == 0 && [regexp {^\|\s+Slice LUTs\s*\*?\s*\|\s+(\d+)\s*\|} $line -> v]} { set luts $v }
        if {$luts == 0 && [regexp {^\|\s+Total LUTs\s*\|\s+(\d+)\s*\|} $line -> v]}       { set luts $v }

        if {[regexp {^\|\s+CLB Registers\s*\|\s+(\d+)\s*\|} $line -> v]} { set ffs $v }
        if {$ffs == 0 && [regexp {^\|\s+Slice Registers\s*\|\s+(\d+)\s*\|} $line -> v]} { set ffs $v }
        if {$ffs == 0 && [regexp {^\|\s+Register as Flip Flop\s*\|\s+(\d+)\s*\|} $line -> v]} { set ffs $v }

        if {[regexp {^\|\s+CARRY8\s*\|\s+(\d+)\s*\|} $line -> v]} { set carries $v }

        if {[regexp {^\|\s+Block RAM Tile\s*\|\s+(\d+)\s*\|} $line -> v]} { set bram_tiles $v }
        if {[regexp {^\|\s+RAMB36/FIFO\*?\s*\|\s+(\d+)\s*\|} $line -> v]} { set ramb36 $v }
        if {[regexp {^\|\s+RAMB18\s*\|\s+(\d+)\s*\|} $line -> v]} { set ramb18 $v }

        if {[regexp {^\|\s+DSP48E2\s*\|\s+(\d+)\s*\|} $line -> v]} { set dsps $v }
        if {$dsps == 0 && [regexp {^\|\s+DSPs?\s*\|\s+(\d+)\s*\|} $line -> v]} { set dsps $v }
    }

    if {$bram_tiles > 0} {
        set brams $bram_tiles
    } else {
        set brams [expr {$ramb36 + int(ceil(double($ramb18) / 2.0))}]
    }
    return [dict create luts $luts ffs $ffs carries $carries brams $brams \
        dsps $dsps urams 0]
}

proc finn_fp_write_floorplan_report {report_file} {
    global finn_fp_report finn_fp_resource_source

    if {![info exists finn_fp_resource_source]} { set finn_fp_resource_source "estimate" }

    if {![info exists finn_fp_report]} {
        puts "WARNING: finn_fp_write_floorplan_report called before finn_fp_floorplan_units"
        return
    }

    set dev_max_x 0
    set dev_max_y 0
    foreach s [get_sites -quiet -filter {SITE_TYPE =~ "SLICE*" && NAME =~ "SLICE_X*Y0"}] {
        if {[regexp {SLICE_X(\d+)Y0} $s -> x] && $x > $dev_max_x} { set dev_max_x $x }
    }
    foreach s [get_sites -quiet -filter {SITE_TYPE =~ "SLICE*" && NAME =~ "SLICE_X0Y*"}] {
        if {[regexp {SLICE_X0Y(\d+)} $s -> y] && $y > $dev_max_y} { set dev_max_y $y }
    }

    set entries {}
    dict for {name info} $finn_fp_report {
        set capacity [dict get $info capacity]
        set grid ""
        if {[get_pblocks -quiet $name] ne ""} {
            set grid [join [get_property GRID_RANGES [get_pblocks $name]] " "]
        }
        set entry "    \"$name\": {"
        append entry "\n      \"slr\": [dict get $info slr],"
        append entry "\n      \"resource_source\": \"$finn_fp_resource_source\","
        append entry "\n      \"round\": [dict get $info round],"
        append entry "\n      \"satisfied\": [expr {[dict get $info satisfied] ? "true" : "false"}],"
        append entry "\n      \"grid_ranges\": \"$grid\","
        append entry "\n      \"requested_ranges\": \"[join [dict get $info ranges] " "]\","
        append entry "\n      \"pblock_capacity\": {"
        append entry "\"luts\": [dict get $capacity luts], "
        append entry "\"ffs\": [dict get $capacity ffs], "
        append entry "\"carries\": [dict get $capacity carries], "
        append entry "\"brams\": [dict get $capacity brams], "
        append entry "\"dsps\": [dict get $capacity dsps], "
        append entry "\"urams\": [dict get $capacity urams], "
        append entry "\"lutrams\": [dict get $capacity lutrams]}"
        append entry "\n    }"
        lappend entries $entry
    }

    set json "{\n"
    append json "  \"device_slice_max_x\": $dev_max_x,\n"
    append json "  \"device_slice_max_y\": $dev_max_y,\n"
    global finn_fp_shell_closure
    if {[info exists finn_fp_shell_closure]} {
        set sc $finn_fp_shell_closure
        append json "  \"shell_closure\": {\"source\": \"[string map {\\ \\\\ \" \\\"} [dict get $sc source]]\",\
\"vivado\": \"[dict get $sc vivado]\", \"listed\": [dict get $sc listed],\
\"prohibited\": [dict get $sc prohibited]},\n"
    }
    append json "  \"units\": {\n"
    append json [join $entries ",\n"]
    append json "\n  }\n}"

    set fh [open $report_file w]
    puts $fh $json
    close $fh
    puts " Floorplan report written to: $report_file"
}
