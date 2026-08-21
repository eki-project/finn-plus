module $TOP_MODULE_NAME$(
//- Global Control ------------------
(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out0_V, ASSOCIATED_RESET = ap_rst_n" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
input   ap_clk,
(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
input   ap_rst_n,

//- AXI Stream - Input --------------
output   in0_V_TREADY,
input   in0_V_TVALID,
input  [$WIDTH$-1:0] in0_V_TDATA,

//- AXI Stream - Output --------------
input   out0_V_TREADY,
output   out0_V_TVALID,
output  [$WIDTH$-1:0] out0_V_TDATA
);

assign	in0_V_TREADY = out0_V_TREADY;
assign	out0_V_TVALID = in0_V_TVALID;
assign	out0_V_TDATA = 0;


endmodule
