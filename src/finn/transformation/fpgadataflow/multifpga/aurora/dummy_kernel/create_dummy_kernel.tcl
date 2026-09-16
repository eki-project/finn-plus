set part        [lindex $argv 2]
set clk_freq    [lindex $argv 3]
set target      [lindex $argv 4]
set export_path [lindex $argv 5]
set width       [lindex $argv 6]
set sourcefile  [lindex $argv 7]

open_project vitis_dummy_kernel
set_top $target
add_files $sourcefile
open_solution "solution1" -flow_target vitis
set_part $part
create_clock -period $clk_freq -name default
config_export -format xo -output $export_path
config_interface -m_axi_alignment_byte_size $width -m_axi_latency $width -m_axi_max_widen_bitwidth [expr $width * 8]
config_rtl -register_reset_num 3
csynth_design
export_design -rtl verilog -format xo -output $target
exit
