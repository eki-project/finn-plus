
create_project @PRJNAME@ @PRJFOLDER@ -part @FPGAPART@
set_msg_config -id {[BD 41-1753]} -suppress
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_reg_32
set_property CONFIG.TDATA_NUM_BYTES {4} [get_ips axis_reg_32]
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_8
set_property -dict [list CONFIG.TDATA_NUM_BYTES {1} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_8]
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_16
set_property -dict [list CONFIG.TDATA_NUM_BYTES {2} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_16]
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_32
set_property -dict [list CONFIG.TDATA_NUM_BYTES {4} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_32]
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_64
set_property -dict [list CONFIG.TDATA_NUM_BYTES {8} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_64]
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_128
set_property -dict [list CONFIG.TDATA_NUM_BYTES {16} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_128]
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_256
set_property -dict [list CONFIG.TDATA_NUM_BYTES {32} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_256]
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_512
set_property -dict [list CONFIG.TDATA_NUM_BYTES {64} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_512]
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_8
set_property -dict [list CONFIG.TDATA_NUM_BYTES {1} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_8]
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_16
set_property -dict [list CONFIG.TDATA_NUM_BYTES {2} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_16]
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_32
set_property -dict [list CONFIG.TDATA_NUM_BYTES {4} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_32]
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_64
set_property -dict [list CONFIG.TDATA_NUM_BYTES {8} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_64]
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_128
set_property -dict [list CONFIG.TDATA_NUM_BYTES {16} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_128]
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_256
set_property -dict [list CONFIG.TDATA_NUM_BYTES {32} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_256]
create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axisf_register_slice_512
set_property -dict [list CONFIG.TDATA_NUM_BYTES {64} CONFIG.REG_CONFIG {8} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.TUSER_WIDTH {34} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1}] [get_ips axisf_register_slice_512]
create_ip -name axi_register_slice -vendor xilinx.com -library ip -version 2.1 -module_name axi_register_slice_256
set_property -dict [list CONFIG.ADDR_WIDTH {64} CONFIG.DATA_WIDTH {256} CONFIG.HAS_QOS {0} CONFIG.HAS_REGION {0} CONFIG.REG_AW {1} CONFIG.REG_AR {1} CONFIG.REG_B {1} CONFIG.ID_WIDTH {2} CONFIG.MAX_BURST_LENGTH {14} CONFIG.NUM_READ_OUTSTANDING {32} CONFIG.NUM_WRITE_OUTSTANDING {32}] [get_ips axi_register_slice_256]
create_ip -name axi_register_slice -vendor xilinx.com -library ip -version 2.1 -module_name axi_register_slice_512
set_property -dict [list CONFIG.ADDR_WIDTH {64} CONFIG.DATA_WIDTH {512} CONFIG.HAS_QOS {0} CONFIG.HAS_REGION {0} CONFIG.REG_AW {1} CONFIG.REG_AR {1} CONFIG.REG_B {1} CONFIG.ID_WIDTH {2} CONFIG.MAX_BURST_LENGTH {14} CONFIG.NUM_READ_OUTSTANDING {32} CONFIG.NUM_WRITE_OUTSTANDING {32}] [get_ips axi_register_slice_512]
create_ip -name axi_register_slice -vendor xilinx.com -library ip -version 2.1 -module_name axil_register_slice_64
set_property -dict [list CONFIG.PROTOCOL {AXI4LITE} CONFIG.ADDR_WIDTH {64} CONFIG.HAS_PROT {0} CONFIG.DATA_WIDTH {64} CONFIG.REG_AW {1} CONFIG.REG_AR {1} CONFIG.REG_W {1} CONFIG.REG_R {1} CONFIG.REG_B {1} ]  [get_ips axil_register_slice_64]
create_ip -name axi_datamover -vendor xilinx.com -library ip -version 5.1   -module_name cdma_datamover_rd
set_property -dict [list     CONFIG.c_addr_width {64}     CONFIG.c_enable_s2mm {0}     CONFIG.c_include_mm2s_dre {true}     CONFIG.c_m_axi_mm2s_data_width {256}     CONFIG.c_m_axis_mm2s_tdata_width {256}     CONFIG.c_mm2s_burst_size {64} ] [get_ips cdma_datamover_rd]
create_ip -name axi_datamover -vendor xilinx.com -library ip -version 5.1  -module_name cdma_datamover_wr
set_property -dict [list     CONFIG.c_addr_width {64}     CONFIG.c_enable_mm2s {0}     CONFIG.c_include_s2mm_dre {true}     CONFIG.c_m_axi_s2mm_data_width {256}     CONFIG.c_s2mm_burst_size {64}     CONFIG.c_s_axis_s2mm_tdata_width {256} ] [get_ips cdma_datamover_wr]
create_ip -name axi_datamover -vendor xilinx.com -library ip -version 5.1 -module_name cdma_datamover
set_property -dict [list     CONFIG.c_include_mm2s_dre {true}     CONFIG.c_include_s2mm_dre {true}     CONFIG.c_addr_width {64}     CONFIG.c_m_axi_mm2s_data_width {256}     CONFIG.c_m_axis_mm2s_tdata_width {256}     CONFIG.c_mm2s_burst_size {64}     CONFIG.c_m_axi_s2mm_data_width {256}     CONFIG.c_s2mm_burst_size {64}     CONFIG.c_s_axis_s2mm_tdata_width {256} ] [get_ips cdma_datamover]

add_files -norecurse "$::env(FINN_RTLLIB)/cdma/cdma_top.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/cdma/krnl_counter.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/cdma/cdma_a/cdma_a.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/cdma/cdma_a/cdma_a_rd.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/cdma/cdma_a/cdma_a_wr.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/cdma/cdma_a/axi_dma_rd_a.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/cdma/cdma_a/axi_dma_wr_a.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/cdma/cdma_u/cdma_u.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/cdma/cdma_u/cdma_u_wr.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/cdma/cdma_u/cdma_u_rd.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/cdma/cdma_u/axi_dma_rd_u.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/cdma/cdma_u/axi_dma_wr_u.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/cdma/cdma_x/cdma_x.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/cdma/cdma_x/cdma_x_rd.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/cdma/cdma_x/cdma_x_wr.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/dwc/hdl/axis_adapter.v"
add_files -norecurse "$::env(FINN_RTLLIB)/dwc/hdl/axis_fifo.v"
add_files -norecurse "$::env(FINN_RTLLIB)/dwc/hdl/axis_fifo_adapter.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/skid/skid.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/ram/ram_p_c.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/mlo/infrastructure/intermediate_frames.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/mlo/infrastructure/mux.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/mlo/infrastructure/demux.sv"
add_files -norecurse "$::env(FINN_RTLLIB)/mlo/loop_control.sv"
add_files -norecurse "@TOP_VERILOG_FILE@"
add_files -norecurse "$::env(FINN_RTLLIB)/fifo/hdl/Q_srl.v"
add_files -norecurse "$::env(FINN_RTLLIB)/fifo/hdl/fifo_gauge.sv"

@IP_GEN@

# Core Cleanup Operations
set core [ipx::current_core]

# Remove all XCI references to subcores
set impl_files [ipx::get_file_groups xilinx_implementation -of $core]
foreach xci [ipx::get_files -of $impl_files {*.xci}] {
    ipx::remove_file [get_property NAME $xci] $impl_files
}

# Finalize and Save
ipx::update_checksums $core
ipx::save_core $core

# Remove stale subcore references from component.xml
file rename -force ip/component.xml ip/component.bak
set ifile [open ip/component.bak r]
set ofile [open ip/component.xml w]
set buf [list]
set kill 0
while { [eof $ifile] != 1 } {
    gets $ifile line
    if { [string match {*<spirit:fileSet>*} $line] == 1 } {
        foreach l $buf { puts $ofile $l }
        set buf [list $line]
    } elseif { [llength $buf] > 0 } {
        lappend buf $line

        if { [string match {*</spirit:fileSet>*} $line] == 1 } {
            if { $kill == 0 } { foreach l $buf { puts $ofile $l } }
            set buf [list]
            set kill 0
        } elseif { [string match {*<xilinx:subCoreRef>*} $line] == 1 } {
            set kill 1
        }
    } else {
        puts $ofile $line
    }
}
close $ifile
close $ofile

# export list of used Verilog files (for rtlsim later on)
proc find_xci_files {dir} {
    set xci_files [list]
    foreach file [glob -nocomplain -directory $dir *] {
        if {[file isdirectory $file]} {
            # Recursively search subdirectories
            set subdir_files [find_xci_files $file]
            set xci_files [concat $xci_files $subdir_files]
        } elseif {[string match *.xci $file]} {
            # Add .xci files with absolute paths to the list
            lappend xci_files [file normalize $file]
        }
    }
    return $xci_files
}
set xci_files [find_xci_files "ip/src"]
foreach xci_file $xci_files {
    read_ip $xci_file
    set ip [get_ips -of_objects [get_files $xci_file]]
    foreach ip_instance $ip {
        set ip_name [get_property NAME $ip_instance]
        if {[string match *FINNLoop* $ip_name]} {
            continue
        }
        generate_target all $ip_instance
    }
}

set all_v_files [get_files -filter {USED_IN_SYNTHESIS == 1 && (FILE_TYPE == Verilog || FILE_TYPE == SystemVerilog || FILE_TYPE =="Verilog Header" || FILE_TYPE == XCI)}]

set fp [open @PRJFOLDER@/all_verilog_srcs.txt w]
foreach vf $all_v_files {puts $fp $vf}
close $fp
