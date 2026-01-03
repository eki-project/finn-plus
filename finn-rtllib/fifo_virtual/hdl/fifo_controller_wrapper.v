module fifo_controller_wrapper #(
    parameter integer  ADDR_WIDTH = 32,
    parameter integer  DATA_WIDTH = 32,
    parameter integer  IP_ADDR_WIDTH = 30,
    parameter integer  IP_DATA_WIDTH = 32
)(
	// Global Control
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi, ASSOCIATED_RESET = ap_rst_n" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	input  ap_clk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input  ap_rst_n,

	// AXI-lite Write Channels
	output  s_axi_awready,
	input  s_axi_awvalid,
	input [2:0]  s_axi_awprot,
	input [ADDR_WIDTH-1:0]  awaddr,

	output  s_axi_wready,
	input  s_axi_wvalid,
	input [DATA_WIDTH/8-1:0]  s_axi_wstrb,
	input [DATA_WIDTH  -1:0]  s_axi_wdata,

	input  s_axi_bready,
	output  s_axi_bvalid,
	output [1:0]  s_axi_bresp,

	// AXI-lite Read Channels
	output  s_axi_arready,
	input  s_axi_arvalid,
	input [2:0]  s_axi_arprot,
	input [ADDR_WIDTH-1:0]  s_axi_araddr,

	input  s_axi_rready,
	output  s_axi_rvalid,
	output [1:0]  s_axi_rresp,
	output [DATA_WIDTH-1:0]  s_axi_rdata,

	// FIFO Configuration Ring Bus
	input [7:0]  icfg,
	output [7:0]  ocfg
);

fifo_controller #(
    .ADDR_WIDTH(ADDR_WIDTH),
    .DATA_WIDTH(DATA_WIDTH),
    .IP_ADDR_WIDTH(IP_ADDR_WIDTH),
    .IP_DATA_WIDTH(IP_DATA_WIDTH)
) fifo_controller_inst (
    .aclk(ap_clk),
    .aresetn(ap_rst_n),

    .awready(s_axi_awready),
    .awvalid(s_axi_awvalid),
    .awprot(s_axi_awprot),
    .awaddr(s_axi_awaddr),

    .wready(s_axi_wready),
    .wvalid(s_axi_wvalid),
    .wstrb(s_axi_wstrb),
    .wdata(s_axi_wdata),

    .bready(s_axi_bready),
    .bvalid(s_axi_bvalid),
    .bresp(s_axi_bresp),

    .arready(s_axi_arready),
    .arvalid(s_axi_arvalid),
    .arprot(s_axi_arprot),
    .araddr(s_axi_araddr),

    .rready(s_axi_rready),
    .rvalid(s_axi_rvalid),
    .rresp(s_axi_rresp),
    .rdata(s_axi_rdata),

    .icfg(icfg),
    .ocfg(ocfg)
);

endmodule
