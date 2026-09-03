module icape3_wrapper (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 CLK CLK" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME CLK, ASSOCIATED_BUSIF ICAP" *)
    input  clk,

    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME ICAP, MODE Slave" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:icap_rtl:1.0 ICAP CSIB" *)
    input  csib,
    (* X_INTERFACE_INFO = "xilinx.com:interface:icap_rtl:1.0 ICAP RDWRB" *)
    input  rdwrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:icap_rtl:1.0 ICAP I" *)
    input  [31:0] i,
    (* X_INTERFACE_INFO = "xilinx.com:interface:icap_rtl:1.0 ICAP O" *)
    output [31:0] o,
    (* X_INTERFACE_INFO = "xilinx.com:interface:icap_rtl:1.0 ICAP AVAIL" *)
    output avail,
    (* X_INTERFACE_INFO = "xilinx.com:interface:icap_rtl:1.0 ICAP PRDONE" *)
    output prdone,
    (* X_INTERFACE_INFO = "xilinx.com:interface:icap_rtl:1.0 ICAP PRERROR" *)
    output prerror
);

    ICAPE3 #(
        .DEVICE_ID(32'h03628093),
        .ICAP_AUTO_SWITCH("DISABLE"),
        .SIM_CFG_FILE_NAME("NONE")
    ) ICAPE3_inst (
        .CLK(clk),
        .CSIB(csib),
        .RDWRB(rdwrb),
        .I(i),
        .O(o),
        .AVAIL(avail),
        .PRDONE(prdone),
        .PRERROR(prerror)
    );

endmodule
