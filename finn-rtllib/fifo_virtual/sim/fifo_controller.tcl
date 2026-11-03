#############################################################################
# Copyright (c) 2025, Paderborn University
#
# @author	Felix P. Jentzsch <felix.jentzsch@upb.de>
#
# Simulates the gauging ring with the fifo_controller in the loop, driven
# through its AXI-lite interface. Run from this directory:
#   vivado -mode batch -source fifo_controller.tcl
#############################################################################
set top fifo_controller
create_project -force $top $top.vivado -part xcvc1902-vsva2197-2MP-e-S

read_verilog -sv ../../axi/hdl/axilite.sv ../hdl/fifo_gauge_pkg.sv ../hdl/fifo_gauge.sv ../hdl/fifo_controller.sv

set simset [current_fileset -simset]
add_files -fileset $simset ${top}_tb.sv
set_property top ${top}_tb $simset
set_property xsim.simulate.runtime all $simset

launch_simulation
close_sim

quit
