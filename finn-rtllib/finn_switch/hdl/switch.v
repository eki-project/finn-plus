module finn_switch #(
    parameter DATA_WIDTH_A = 32,
    parameter DATA_WIDTH_B = 32
)(
    input wire sel,

    // 2x1 Input 0
    input wire A_IN0_tvalid,
    input wire [DATA_WIDTH_A-1:0] A_IN0_tdata,
    input wire A_IN0_tlast,
    output wire A_IN0_tready,

    // 2x1 Input 1
    input wire A_IN1_tvalid,
    input wire [DATA_WIDTH_A-1:0] A_IN1_tdata,
    input wire A_IN1_tlast,
    output wire A_IN1_tready,

    // 2x1 Output
    output wire A_OUT_tvalid,
    output wire [DATA_WIDTH_A-1:0] A_OUT_tdata,
    output wire A_OUT_tlast,
    input wire A_OUT_tready,

    // 1x2 Input
    input wire B_IN_tvalid,
    input wire [DATA_WIDTH_B-1:0] B_IN_tdata,
    input wire B_IN_tlast,
    output wire B_IN_tready,

    // 1x2 Output 0
    output wire B_OUT0_tvalid,
    output wire [DATA_WIDTH_B-1:0] B_OUT0_tdata,
    output wire B_OUT0_tlast,
    input wire B_OUT0_tready,

    // 1x2 Output 1
    output wire B_OUT1_tvalid,
    output wire [DATA_WIDTH_B-1:0] B_OUT1_tdata,
    output wire B_OUT1_tlast,
    input wire B_OUT1_tready
);
    // 2x1
    assign A_OUT_tvalid = (sel) ? A_IN1_tvalid : A_IN0_tvalid;
    assign A_OUT_tdata = (sel) ? A_IN1_tdata : A_IN0_tdata;
    assign A_OUT_tlast = (sel) ? A_IN1_tlast : A_IN0_tlast;
    assign A_IN0_tready = A_OUT_tready;
    assign A_IN1_tready = A_OUT_tready;

    // 1x2
    assign B_OUT0_tvalid = (sel) ? 0 : B_IN_tvalid;
    assign B_OUT0_tdata = B_IN_tdata;
    assign B_OUT0_tlast = B_IN_tlast;
    assign B_OUT1_tvalid = (sel) ? B_IN_tvalid : 0;
    assign B_OUT1_tdata = B_IN_tdata;
    assign B_OUT1_tlast = B_IN_tlast;
    assign B_IN_tready = (sel) ? B_OUT1_tready : B_OUT0_tready;

endmodule
