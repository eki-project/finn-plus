# Export the worst setup paths of a routed design, with the placement of
# their endpoints and the placed bounding box of every floorplan unit, for the
# floorplanner's intra/inter-unit timing split (floorplanning/timing_split.py).
#
#   vivado -mode batch -source finn_floorplan_timing.tcl \
#       -tclargs <routed.dcp> <out.json> <max_paths> <clock_pattern> <unit_roots.txt>
#
# <unit_roots.txt> lists one unit per line: "<unit name> <root cell>", the root
# cell being the node name the hook constrained (resolved anywhere in the
# hierarchy, as the hook does). The export is JSON:
#   clock:  the clocks the paths were taken from
#   units:  {name: {roots: [full cell names], cells: n, x0, y0, x1, y1}}
#           -- the tile-grid bounding box of the unit's placed leaf cells
#   paths:  [{slack, requirement, datapath, logic, route, levels,
#             start_cell, end_cell, start_site, end_site, sx, sy, ex, ey, cells}]
#           -- worst first; sx..ey are the tile column/row of the endpoint sites;
#           cells are the other cells the path passes through (a combinational
#           ready path leaves a unit and comes back through its consumer)
# Coordinates are tile COLUMN/ROW, which are uniform across the device (the RPM
# grid is not: ~22 units per CLB column against ~2.4 per row on the U55C).

set_param tcl.collectionResultDisplayLimit 0

if {[llength $argv] < 5 || [llength $argv] > 6} {
    error "usage: finn_floorplan_timing.tcl <routed.dcp> <out.json> <max_paths> <clock_pattern> <unit_roots.txt> \[<exception_sources_regexp>\]"
}
lassign $argv checkpoint out max_paths clock_pattern roots_file exception_sources

proc ts_json_str {s} {
    return "\"[string map {\\ \\\\ \" \\\"} $s]\""
}

# "Data Path Delay: 7.780ns  (logic 1.234ns (15.864%)  route 6.546ns (84.136%))"
proc ts_delay_split {path} {
    set rpt [report_timing -of_objects $path -return_string -no_header]
    if {[regexp {Data Path Delay:\s+([\d.]+)ns\s+\(logic\s+([\d.]+)ns.*?route\s+([\d.]+)ns} $rpt -> dp logic route]} {
        return [list $dp $logic $route]
    }
    return [list [get_property DATAPATH_DELAY $path] "" ""]
}

proc ts_site_xy {cell} {
    set site [get_sites -quiet -of_objects $cell]
    if {$site eq ""} { return [list "" "" ""] }
    set tile [get_tiles -of_objects $site]
    return [list [get_property NAME $site] [get_property COLUMN $tile] [get_property ROW $tile]]
}

set t0 [clock seconds]
open_checkpoint $checkpoint
puts "timing_split: checkpoint opened in [expr {[clock seconds] - $t0}] s"

set clocks [get_clocks -quiet $clock_pattern]
if {[llength $clocks] == 0} {
    error "no clock matches '$clock_pattern'; the design has: [get_clocks]"
}
puts "timing_split: clocks [get_property NAME $clocks]"

# -- units: placed bounding boxes --------------------------------------------
set fh [open $roots_file r]
set root_lines [split [string trim [read $fh]] "\n"]
close $fh
# A unit may list several root cells (one line each); their placements merge.
array set unit_xs {}
array set unit_ys {}
array set unit_n {}
array set unit_roots {}
set unit_order {}
set t0 [clock seconds]
foreach line $root_lines {
    set line [string trim $line]
    if {$line eq ""} { continue }
    lassign $line unit root
    if {![info exists unit_n($unit)]} {
        set unit_n($unit) 0; set unit_xs($unit) {}; set unit_ys($unit) {}; set unit_roots($unit) {}
        lappend unit_order $unit
    }
    set objs [get_cells -quiet $root]
    if {$objs eq ""} { set objs [get_cells -quiet -hierarchical $root] }
    # A block often holds a child of the same name (X/X); keep only the outermost.
    set outer {}
    foreach obj $objs {
        set name [get_property NAME $obj]
        set nested 0
        foreach other $objs {
            set oname [get_property NAME $other]
            if {$oname ne $name && [string match "$oname/*" $name]} { set nested 1; break }
        }
        if {!$nested} { lappend outer $obj }
    }
    set objs $outer
    foreach obj $objs {
        lappend unit_roots($unit) [ts_json_str [get_property NAME $obj]]
        set leaves [get_cells -quiet -hierarchical -filter "IS_PRIMITIVE && NAME =~ [get_property NAME $obj]/*"]
        set sites [get_sites -quiet -of_objects $leaves]
        incr unit_n($unit) [llength $leaves]
        if {[llength $sites]} {
            set tiles [get_tiles -of_objects $sites]
            lappend unit_xs($unit) {*}[get_property COLUMN $tiles]
            lappend unit_ys($unit) {*}[get_property ROW $tiles]
        }
    }
}
set unit_json {}
foreach unit $unit_order {
    set xs [lsort -integer $unit_xs($unit)]
    set ys [lsort -integer $unit_ys($unit)]
    if {[llength $xs]} {
        set box "\"x0\": [lindex $xs 0], \"y0\": [lindex $ys 0], \"x1\": [lindex $xs end], \"y1\": [lindex $ys end]"
    } else {
        set box "\"x0\": null, \"y0\": null, \"x1\": null, \"y1\": null"
    }
    lappend unit_json "[ts_json_str $unit]: {\"roots\": \[[join $unit_roots($unit) ,]\], \"cells\": $unit_n($unit), $box}"
}
puts "timing_split: [llength $unit_json] unit(s) measured in [expr {[clock seconds] - $t0}] s"

# -- paths -------------------------------------------------------------------
set t0 [clock seconds]
# Two groups: the worst paths under the clock period alone, and the worst under a
# tighter requirement (a set_max_delay, e.g. the reset tree's hops in builds of
# 14-16 Oct 2026). Ranked together
# by slack, a few hundred constrained registers would fill the whole export and hide
# the clock paths, and -filter on get_timing_paths applies only after the worst paths
# were picked. So the tighter group is taken first, and the clock-period group is
# taken from every sequential cell except the exception sources (the optional sixth
# argument, a get_cells -regexp pattern), whose paths then cannot win an endpoint.
# The analysis restates every slack against the period.
set period [get_property PERIOD [lindex $clocks 0]]
set tight [get_timing_paths -quiet -setup -max_paths $max_paths -nworst 1 -from $clocks -to $clocks \
               -filter "REQUIREMENT < $period"]
set from $clocks
if {$exception_sources ne ""} {
    set excl [get_cells -quiet -hierarchical -regexp $exception_sources]
    if {[llength $excl]} {
        set from [get_cells -hierarchical -regexp -filter "IS_SEQUENTIAL && NAME !~ {$exception_sources}"]
        puts "timing_split: [llength $excl] exception source(s) left out of the clock-period group"
    }
}
set paths [get_timing_paths -setup -max_paths $max_paths -nworst 1 -from $from -to $clocks]
puts "timing_split: [llength $paths] path(s) under the period and [llength $tight] under tighter requirements collected in [expr {[clock seconds] - $t0}] s"
set paths [concat $paths $tight]
set t0 [clock seconds]
set path_json {}
set i 0
foreach p $paths {
    set sp [get_property STARTPOINT_PIN $p]
    set ep [get_property ENDPOINT_PIN $p]
    set sc [get_cells -quiet -of_objects $sp]
    set ec [get_cells -quiet -of_objects $ep]
    lassign [ts_site_xy $sc] ssite sx sy
    lassign [ts_site_xy $ec] esite ex ey
    lassign [ts_delay_split $p] dp logic route
    set through {}
    foreach c [get_cells -quiet -of_objects $p] {
        if {$c ne $sc && $c ne $ec} { lappend through [ts_json_str [get_property NAME $c]] }
    }
    set rec [list \
        "\"slack\": [get_property SLACK $p]" \
        "\"requirement\": [get_property REQUIREMENT $p]" \
        "\"datapath\": $dp" \
        "\"logic\": [expr {$logic eq "" ? "null" : $logic}]" \
        "\"route\": [expr {$route eq "" ? "null" : $route}]" \
        "\"levels\": [get_property LOGIC_LEVELS $p]" \
        "\"start_cell\": [ts_json_str [get_property NAME $sc]]" \
        "\"end_cell\": [ts_json_str [get_property NAME $ec]]" \
        "\"start_site\": [ts_json_str $ssite]" \
        "\"end_site\": [ts_json_str $esite]" \
        "\"sx\": [expr {$sx eq "" ? "null" : $sx}]" \
        "\"sy\": [expr {$sy eq "" ? "null" : $sy}]" \
        "\"ex\": [expr {$ex eq "" ? "null" : $ex}]" \
        "\"ey\": [expr {$ey eq "" ? "null" : $ey}]" \
        "\"cells\": \[[join $through ,]\]" \
    ]
    lappend path_json "{[join $rec ,]}"
    incr i
    if {$i % 500 == 0} { puts "timing_split: $i path(s) exported" }
}
puts "timing_split: paths exported in [expr {[clock seconds] - $t0}] s"

set fh [open $out w]
puts $fh "{"
puts $fh "\"checkpoint\": [ts_json_str $checkpoint],"
puts $fh "\"clock\": [ts_json_str [get_property NAME $clocks]],"
puts $fh "\"units\": {[join $unit_json ,\n]},"
puts $fh "\"paths\": \[[join $path_json ,\n]\]"
puts $fh "}"
close $fh
puts "timing_split: wrote $out"
