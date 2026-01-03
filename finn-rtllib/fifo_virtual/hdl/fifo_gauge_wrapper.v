module fifo_gauge_wrapper #(
	parameter [15:0]  ID = 0,
	parameter integer  DATA_WIDTH = 8,
	parameter integer  FM_SIZE = 1
)(
	// Global Control
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out0_V, ASSOCIATED_RESET = ap_rst_n" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	input  ap_clk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input  ap_rst_n,

	// Configuration Ring: Control & Status
	input [7:0]  icfg,
	output [7:0]  ocfg,

	// Input Stream
	input [DATA_WIDTH-1:0]  in0_V_TDATA,
	input  in0_V_TVALID,
	output  in0_V_TREADY,

	// Output Stream
	output [DATA_WIDTH-1:0]  out0_V_TDATA,
	output  out0_V_TVALID,
	input  out0_V_TREADY
);

fifo_gauge #(
    .ID(ID),
    .DATA_WIDTH(DATA_WIDTH),
    .FM_SIZE(FM_SIZE)
) fifo_inst (
    .clk(ap_clk),
    .rst(ap_rst_n),
    .idat(in0_V_TDATA),
    .ivld(in0_V_TVALID),
    .irdy(in0_V_TREADY),
    .odat(out0_V_TDATA),
    .ovld(out0_V_TVALID),
    .ordy(out0_V_TREADY),
    .icfg(icfg),
    .ocfg(ocfg)
);

endmodule
