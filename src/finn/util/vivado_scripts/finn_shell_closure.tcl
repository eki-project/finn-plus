################################################################################
# FINN+ shell closure probe.
#
# On a shell-based (DFX) platform such as an Alveo card, the placer closes some
# sites inside the user region as unroutable (`place.closeUnroutableSites`),
# because the locked static routing occupies the wires they would need. Which
# sites these are depends only on the platform shell and the Vivado release, so
# FINN measures them once per platform and keeps them out of the floorplanner's
# device census (see floorplanning/shell_closure.py, which drives this script).
#
# The probe is empirical: it opens the platform's locked shell checkpoint, fills
# the empty user region with unconnected probe cells, each group confined to a
# hard pblock over the sites under test, and runs place_design. The placer's
# `[Place 30-859]` rejections then name every pblock that lost sites to the
# closure, with the count ("... blocked with place.closeUnroutableSites param:
# N"). The Python driver parses those messages from the Vivado log.
#
# Usage (normally run by shell_closure.py):
#   vivado -mode batch -source finn_shell_closure.tcl \
#       -tclargs enumerate <checkpoint.dcp> <out.txt>
#   vivado -mode batch -source finn_shell_closure.tcl \
#       -tclargs probe <checkpoint.dcp> <request.txt> <out.txt>
#
# enumerate writes one line per available site of every probed kind inside the
# reconfigurable region:  SITE <kind> <site> <slr> <column-x> <row-y>
# plus INFO lines (part, Vivado version, region pblock).
#
# probe reads one line per pblock:  <kind> <pblock-name> <cells> <site> [<site> ...]
# and confines <cells> probe cells to exactly those sites. The driver asks for
# as many cells as the sites hold (one per block RAM, DSP or URAM site, eight
# LUT1 per SLICE), so a pblock with even one closed site is over-subscribed.
################################################################################

# Probe cell per site kind: library cell and the properties it needs to pass
# the design-rule checks place_design runs first.
set FINN_CS_KINDS {
    RAMB36  {ref RAMB36E2 filter {SITE_TYPE == "RAMBFIFO36"} props {READ_WIDTH_A 36 WRITE_WIDTH_A 36 READ_WIDTH_B 36 WRITE_WIDTH_B 36}}
    DSP48E2 {ref DSP48E2  filter {SITE_TYPE == "DSP48E2"}    props {}}
    URAM288 {ref URAM288  filter {SITE_TYPE == "URAM288"}    props {}}
    SLICE   {ref LUT1     filter {SITE_TYPE =~ "SLICE*"}     props {INIT 2'h1}}
    CARRY8  {ref CARRY8   filter {SITE_TYPE =~ "SLICE*"}     props {}}
}

# A CARRY8 probe is a set of carry chains of FINN_CS_CARRY_WINDOW cells each
# (the last one takes the leftover): CO[7] of one CARRY8 drives CI of the next,
# and the first CI (and every CI_TOP) is tied to ground. Next to the static shell
# a LUT fits where a carry chain does not (U55C, 3 Oct 2026), so carry logic is
# probed on its own.
if {![info exists FINN_CS_CARRY_WINDOW]} { set FINN_CS_CARRY_WINDOW 2 }

proc finn_cs_chains {rp cells} {
    global FINN_CS_CARRY_WINDOW
    set w $FINN_CS_CARRY_WINDOW
    set groups {}
    for {set i 0} {$i < [llength $cells]} {incr i $w} {
        lappend groups [lrange $cells $i [expr {$i + $w - 1}]]
    }
    if {[llength $groups] > 1 && [llength [lindex $groups end]] < $w} {
        set tail [lindex $groups end]
        set groups [lreplace $groups end end]
        lset groups end [concat [lindex $groups end] $tail]
    }
    foreach g $groups { finn_cs_chain $rp $g }
}

proc finn_cs_chain {rp cells} {
    set zero $rp/finn_cs_zero
    if {[get_nets -quiet $zero] eq ""} {
        create_cell -reference GND $rp/finn_cs_gnd
        create_net $zero
        connect_net -net $zero -objects [get_pins $rp/finn_cs_gnd/G]
    }
    set prev ""
    foreach c $cells {
        if {$prev eq ""} {
            connect_net -net $zero -objects [get_pins $c/CI]
        } else {
            create_net ${c}_ci
            connect_net -net ${c}_ci -objects [list [get_pins $prev/CO[7]] [get_pins $c/CI]]
        }
        connect_net -net $zero -objects [get_pins $c/CI_TOP]
        set prev $c
    }
}

proc finn_cs_kind {kind key} {
    global FINN_CS_KINDS
    return [dict get [dict get $FINN_CS_KINDS $kind] $key]
}

# The reconfigurable partition: the black-box cell marked HD.RECONFIGURABLE,
# and the pblock that holds it.
proc finn_cs_open {dcp} {
    open_checkpoint $dcp
    set rp [lindex [get_cells -hierarchical -quiet -filter {HD.RECONFIGURABLE == 1}] 0]
    if {$rp eq ""} { error "no HD.RECONFIGURABLE cell in $dcp: not a shell-based platform" }
    set pb [get_pblocks -quiet -of_objects $rp]
    if {$pb eq ""} { error "reconfigurable cell $rp has no pblock" }
    if {[get_property IS_BLACKBOX $rp]} { update_design -cells $rp -buffer_ports }
    return [list $rp $pb]
}

proc finn_cs_enumerate {dcp out} {
    lassign [finn_cs_open $dcp] rp pb
    set fh [open $out w]
    puts $fh "INFO part [get_property PART [current_design]]"
    puts $fh "INFO vivado [version -short]"
    puts $fh "INFO region_cell $rp"
    puts $fh "INFO region_pblock $pb"
    global FINN_CS_KINDS
    foreach kind [dict keys $FINN_CS_KINDS] {
        set sites [get_sites -quiet -of_objects $pb \
            -filter "([finn_cs_kind $kind filter]) && IS_USED == 0 && PROHIBIT == 0"]
        foreach s $sites {
            set name [get_property NAME $s]
            if {![regexp {_X(\d+)Y(\d+)$} $name -> x y]} { continue }
            set slr [get_property SLR_INDEX [get_slrs -of_objects $s]]
            puts $fh "SITE $kind $name $slr $x $y"
        }
        puts "  enumerated [llength $sites] $kind site(s)"
    }
    close $fh
}

proc finn_cs_probe {dcp request out} {
    lassign [finn_cs_open $dcp] rp pb
    # The placer reports every rejected pblock; lift the per-message limits so
    # none is suppressed.
    foreach id {{Place 30-859} {Place 30-99} {Place 30-4} {Place 30-365} {Place 30-1153} {Place 30-488} {Place 30-642}} {
        set_msg_config -id $id -limit 1000000
    }
    # Unconnected probe cells violate pin rules that are irrelevant here.
    # (PDCY: a CARRY8 with its carry input unconnected.)
    foreach chk [get_drc_checks -quiet -filter {NAME =~ "REQP-*" || NAME =~ "DSPS-*" || NAME =~ "HDPRA-*" || NAME =~ "RTSTAT-*" || NAME =~ "PDCY-*"}] {
        catch { set_property SEVERITY {Warning} $chk }
    }
    set fh [open $request r]
    set lines [split [read $fh] "\n"]
    close $fh
    set xdc [file join [file dirname $out] [file rootname [file tail $out]]_pblocks.xdc]
    set xh [open $xdc w]
    set n_pb 0; set n_cells 0
    foreach line $lines {
        set line [string trim $line]
        if {$line eq "" || [string index $line 0] eq "#"} { continue }
        set sites [lassign $line kind name count]
        set cells {}
        for {set i 0} {$i < $count} {incr i} { lappend cells "$rp/finn_cs_${name}_$i" }
        create_cell -reference [finn_cs_kind $kind ref] $cells
        set props [finn_cs_kind $kind props]
        if {[llength $props]} { set_property -dict $props [get_cells $cells] }
        if {$kind eq "CARRY8"} { finn_cs_chains $rp $cells }
        set ranges {}
        foreach s $sites { lappend ranges "$s:$s" }
        puts $xh "create_pblock $name"
        puts $xh "resize_pblock \[get_pblocks $name\] -add {$ranges}"
        puts $xh "add_cells_to_pblock \[get_pblocks $name\] \[get_cells {$cells}\]"
        puts $xh "set_property IS_SOFT FALSE \[get_pblocks $name\]"
        incr n_pb; incr n_cells [llength $cells]
    }
    close $xh
    read_xdc $xdc
    puts "FINN_CS pblocks $n_pb cells $n_cells"
    set rc [catch {place_design} msg]
    puts "FINN_CS place_rc $rc"
    set oh [open $out w]
    puts $oh "pblocks $n_pb cells $n_cells place_rc $rc"
    # Probe cells the placer left without a site, with the pblock that held them;
    # for a one-site pblock that site is one the placer would not use.
    set unplaced 0
    foreach c [get_cells -quiet "$rp/finn_cs_*"] {
        if {[get_property LOC $c] eq ""} {
            puts $oh "UNPLACED [get_property NAME $c] [get_pblocks -quiet -of_objects $c]"
            incr unplaced
        }
    }
    close $oh
    puts "FINN_CS unplaced $unplaced"
}

lassign $argv mode
switch -- $mode {
    enumerate { finn_cs_enumerate [lindex $argv 1] [lindex $argv 2] }
    probe     { finn_cs_probe [lindex $argv 1] [lindex $argv 2] [lindex $argv 3] }
    default   { error "usage: finn_shell_closure.tcl enumerate|probe ..." }
}
