#############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @author	Thomas B. Preußer <thomas.preusser@amd.com>
#############################################################################
set top fifo_gauge
create_project -force $top $top.vivado -part xcvc1902-vsva2197-2MP-e-S

read_verilog -sv ${top}_pkg.sv $top.sv

set simset [current_fileset -simset]
add_files -fileset $simset ${top}_tb.sv
set_property top ${top}_tb $simset
set_property xsim.simulate.runtime all $simset

launch_simulation
close_sim

quit
