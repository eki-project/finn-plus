// (c) Copyright 1995-2025 Xilinx, Inc. All rights reserved.
// 
// This file contains confidential and proprietary information
// of Xilinx, Inc. and is protected under U.S. and
// international copyright and other intellectual property
// laws.
// 
// DISCLAIMER
// This disclaimer is not a license and does not grant any
// rights to the materials distributed herewith. Except as
// otherwise provided in a valid license issued to you by
// Xilinx, and to the maximum extent permitted by applicable
// law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
// WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
// AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
// BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
// INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
// (2) Xilinx shall not be liable (whether in contract or tort,
// including negligence, or under any other theory of
// liability) for any loss or damage of any kind or nature
// related to, arising under or in connection with these
// materials, including for any direct, or any indirect,
// special, incidental, or consequential loss or damage
// (including loss of data, profits, goodwill, or any type of
// loss or damage suffered as a result of any action brought
// by a third party) even if such damage or loss was
// reasonably foreseeable or Xilinx had been advised of the
// possibility of the same.
// 
// CRITICAL APPLICATIONS
// Xilinx products are not designed or intended to be fail-
// safe, or for use in any application requiring fail-safe
// performance, such as life-support or safety devices or
// systems, Class III medical devices, nuclear facilities,
// applications related to the deployment of airbags, or any
// other applications that could lead to death, personal
// injury, or severe property or environmental damage
// (individually and collectively, "Critical
// Applications"). Customer assumes the sole risk and
// liability of any use of Xilinx products in Critical
// Applications, subject only to applicable laws and
// regulations governing limitations on product liability.
// 
// THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
// PART OF THIS FILE AT ALL TIMES.
// 
// DO NOT MODIFY THIS FILE.


// IP VLNV: xilinx.com:module_ref:FMPadding_rtl_17:1.0
// IP Revision: 1

(* X_CORE_INFO = "FMPadding_rtl_17,Vivado 2022.2" *)
(* CHECK_LICENSE_TYPE = "finn_design_FMPadding_rtl_17_0,FMPadding_rtl_17,{}" *)
(* CORE_GENERATION_INFO = "finn_design_FMPadding_rtl_17_0,FMPadding_rtl_17,{x_ipProduct=Vivado 2022.2,x_ipVendor=xilinx.com,x_ipLibrary=module_ref,x_ipName=FMPadding_rtl_17,x_ipVersion=1.0,x_ipCoreRevision=1,x_ipLanguage=VERILOG,x_ipSimLanguage=MIXED}" *)
(* IP_DEFINITION_SOURCE = "module_ref" *)
(* DowngradeIPIdentifiedWarnings = "yes" *)
module finn_design_FMPadding_rtl_17_0 (
  ap_clk,
  ap_rst_n,
  s_axilite_AWVALID,
  s_axilite_AWREADY,
  s_axilite_AWADDR,
  s_axilite_WVALID,
  s_axilite_WREADY,
  s_axilite_WDATA,
  s_axilite_WSTRB,
  s_axilite_BVALID,
  s_axilite_BREADY,
  s_axilite_BRESP,
  s_axilite_ARVALID,
  s_axilite_ARREADY,
  s_axilite_ARADDR,
  s_axilite_RVALID,
  s_axilite_RREADY,
  s_axilite_RDATA,
  s_axilite_RRESP,
  in0_V_TREADY,
  in0_V_TVALID,
  in0_V_TDATA,
  out_V_TREADY,
  out_V_TVALID,
  out_V_TDATA
);

(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME ap_clk, ASSOCIATED_RESET ap_rst_n, ASSOCIATED_BUSIF in0_V:out_V:s_axilite, FREQ_HZ 100000000, FREQ_TOLERANCE_HZ 0, PHASE 0.0, CLK_DOMAIN finn_design_ap_clk_0, INSERT_VIP 0" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
input wire ap_clk;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME ap_rst_n, POLARITY ACTIVE_LOW, INSERT_VIP 0" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ap_rst_n RST" *)
input wire ap_rst_n;
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axilite AWVALID" *)
input wire s_axilite_AWVALID;
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axilite AWREADY" *)
output wire s_axilite_AWREADY;
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axilite AWADDR" *)
input wire [4 : 0] s_axilite_AWADDR;
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axilite WVALID" *)
input wire s_axilite_WVALID;
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axilite WREADY" *)
output wire s_axilite_WREADY;
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axilite WDATA" *)
input wire [31 : 0] s_axilite_WDATA;
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axilite WSTRB" *)
input wire [3 : 0] s_axilite_WSTRB;
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axilite BVALID" *)
output wire s_axilite_BVALID;
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axilite BREADY" *)
input wire s_axilite_BREADY;
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axilite BRESP" *)
output wire [1 : 0] s_axilite_BRESP;
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axilite ARVALID" *)
input wire s_axilite_ARVALID;
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axilite ARREADY" *)
output wire s_axilite_ARREADY;
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axilite ARADDR" *)
input wire [4 : 0] s_axilite_ARADDR;
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axilite RVALID" *)
output wire s_axilite_RVALID;
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axilite RREADY" *)
input wire s_axilite_RREADY;
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axilite RDATA" *)
output wire [31 : 0] s_axilite_RDATA;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME s_axilite, DATA_WIDTH 32, PROTOCOL AXI4LITE, FREQ_HZ 100000000, ID_WIDTH 0, ADDR_WIDTH 5, AWUSER_WIDTH 0, ARUSER_WIDTH 0, WUSER_WIDTH 0, RUSER_WIDTH 0, BUSER_WIDTH 0, READ_WRITE_MODE READ_WRITE, HAS_BURST 0, HAS_LOCK 0, HAS_PROT 0, HAS_CACHE 0, HAS_QOS 0, HAS_REGION 0, HAS_WSTRB 1, HAS_BRESP 1, HAS_RRESP 1, SUPPORTS_NARROW_BURST 0, NUM_READ_OUTSTANDING 1, NUM_WRITE_OUTSTANDING 1, MAX_BURST_LENGTH 1, PHASE 0.0, CLK_DOMAIN finn_design_ap_clk_0, NUM_READ_THREADS 1, NUM_WRITE_THREA\
DS 1, RUSER_BITS_PER_BYTE 0, WUSER_BITS_PER_BYTE 0, INSERT_VIP 0" *)
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axilite RRESP" *)
output wire [1 : 0] s_axilite_RRESP;
(* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 in0_V TREADY" *)
output wire in0_V_TREADY;
(* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 in0_V TVALID" *)
input wire in0_V_TVALID;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME in0_V, TDATA_NUM_BYTES 40, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0, HAS_TREADY 1, HAS_TSTRB 0, HAS_TKEEP 0, HAS_TLAST 0, FREQ_HZ 100000000, PHASE 0.0, CLK_DOMAIN finn_design_ap_clk_0, LAYERED_METADATA undef, INSERT_VIP 0" *)
(* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 in0_V TDATA" *)
input wire [319 : 0] in0_V_TDATA;
(* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 out_V TREADY" *)
input wire out_V_TREADY;
(* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 out_V TVALID" *)
output wire out_V_TVALID;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME out_V, TDATA_NUM_BYTES 40, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0, HAS_TREADY 1, HAS_TSTRB 0, HAS_TKEEP 0, HAS_TLAST 0, FREQ_HZ 100000000, PHASE 0.0, CLK_DOMAIN finn_design_ap_clk_0, LAYERED_METADATA undef, INSERT_VIP 0" *)
(* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 out_V TDATA" *)
output wire [319 : 0] out_V_TDATA;

  FMPadding_rtl_17 inst (
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),
    .s_axilite_AWVALID(s_axilite_AWVALID),
    .s_axilite_AWREADY(s_axilite_AWREADY),
    .s_axilite_AWADDR(s_axilite_AWADDR),
    .s_axilite_WVALID(s_axilite_WVALID),
    .s_axilite_WREADY(s_axilite_WREADY),
    .s_axilite_WDATA(s_axilite_WDATA),
    .s_axilite_WSTRB(s_axilite_WSTRB),
    .s_axilite_BVALID(s_axilite_BVALID),
    .s_axilite_BREADY(s_axilite_BREADY),
    .s_axilite_BRESP(s_axilite_BRESP),
    .s_axilite_ARVALID(s_axilite_ARVALID),
    .s_axilite_ARREADY(s_axilite_ARREADY),
    .s_axilite_ARADDR(s_axilite_ARADDR),
    .s_axilite_RVALID(s_axilite_RVALID),
    .s_axilite_RREADY(s_axilite_RREADY),
    .s_axilite_RDATA(s_axilite_RDATA),
    .s_axilite_RRESP(s_axilite_RRESP),
    .in0_V_TREADY(in0_V_TREADY),
    .in0_V_TVALID(in0_V_TVALID),
    .in0_V_TDATA(in0_V_TDATA),
    .out_V_TREADY(out_V_TREADY),
    .out_V_TVALID(out_V_TVALID),
    .out_V_TDATA(out_V_TDATA)
  );
endmodule
