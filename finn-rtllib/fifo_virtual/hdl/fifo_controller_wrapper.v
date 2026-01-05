module fifo_controller_wrapper #(
    parameter integer  ADDR_WIDTH = 32,
    parameter integer  DATA_WIDTH = 32,
    parameter integer  IP_ADDR_WIDTH = 30,
    parameter integer  IP_DATA_WIDTH = 32
)(
	// Global Control
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi, ASSOCIATED_RESET ap_rst_n" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	input  ap_clk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ap_rst_n RST" *)
	input  ap_rst_n,

	// AXI-lite Write Channels
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWREADY" *)
	output  s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWVALID" *)
	input  s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWPROT" *)
	input [2:0]  s_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWADDR" *)
	input [ADDR_WIDTH-1:0]  s_axi_awaddr,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WREADY" *)
	output  s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WVALID" *)
	input  s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WSTRB" *)
	input [DATA_WIDTH/8-1:0]  s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WDATA" *)
	input [DATA_WIDTH  -1:0]  s_axi_wdata,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BREADY" *)
	input  s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BVALID" *)
	output  s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BRESP" *)
	output [1:0]  s_axi_bresp,

	// AXI-lite Read Channels
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARREADY" *)
	output  s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARVALID" *)
	input  s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARPROT" *)
	input [2:0]  s_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARADDR" *)
	input [ADDR_WIDTH-1:0]  s_axi_araddr,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RREADY" *)
	input  s_axi_rready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RVALID" *)
	output  s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RRESP" *)
	output [1:0]  s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RDATA" *)
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
