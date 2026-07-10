open_project -reset static_mux_test
add_files static_mux_tb_top.cpp
add_files -tb tb.cpp
open_solution -reset sol1 -flow_target vitis
set_part  { xcu280-fsvh2892-2L-e }
create_clock -period 5.0

set_top MuxDemuxOutOfOrder
csynth_design
csim_design
cosim_design -trace_level port
exit
