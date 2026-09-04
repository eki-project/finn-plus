// Verilog-2001 wrapper for sw_wrapper.sv.
// Used as a Vivado block design module reference so parameters can be set via
// set_property CONFIG.* on the bd cell.
`timescale 1ns/1ps
module sw_wrapper_wrapper #(
    parameter DATA_IN_WIDTH    = 64,
    parameter DATA_OUT_WIDTH   = 64,
    parameter TUSER_WIDTH      = 2,
    parameter NUM_SETS         = 2,
    parameter NUM_OUTPUT_BEATS = 1,
    parameter SETSEL_WIDTH     = 32
) (
    input  wire                          aclk,
    input  wire                          aresetn,

    input  wire [DATA_IN_WIDTH-1:0]      s_axis_tdata,
    input  wire                          s_axis_tvalid,
    output wire                          s_axis_tready,
    input  wire                          s_axis_tlast,
    input  wire [TUSER_WIDTH-1:0]        s_axis_tuser,

    output wire [DATA_OUT_WIDTH-1:0]     m_axis_tdata,
    output wire                          m_axis_tvalid,
    input  wire                          m_axis_tready,
    output wire                          m_axis_tlast,
    output wire [TUSER_WIDTH-1:0]        m_axis_tuser,

    output wire [DATA_IN_WIDTH-1:0]      rp_m_axis_tdata,
    output wire                          rp_m_axis_tvalid,
    input  wire                          rp_m_axis_tready,
    output wire                          rp_m_axis_tlast,

    input  wire [DATA_OUT_WIDTH-1:0]     rp_s_axis_tdata,
    input  wire                          rp_s_axis_tvalid,
    output wire                          rp_s_axis_tready,
    input  wire                          rp_s_axis_tlast,

    output wire [SETSEL_WIDTH-1:0]       m_axis_setsel_tdata,
    output wire                          m_axis_setsel_tvalid,
    input  wire                          m_axis_setsel_tready
);

    sw_wrapper #(
        .DATA_IN_WIDTH    (DATA_IN_WIDTH),
        .DATA_OUT_WIDTH   (DATA_OUT_WIDTH),
        .TUSER_WIDTH      (TUSER_WIDTH),
        .NUM_SETS         (NUM_SETS),
        .NUM_OUTPUT_BEATS (NUM_OUTPUT_BEATS),
        .SETSEL_WIDTH     (SETSEL_WIDTH)
    ) inst (
        .aclk                 (aclk),
        .aresetn              (aresetn),
        .s_axis_tdata         (s_axis_tdata),
        .s_axis_tvalid        (s_axis_tvalid),
        .s_axis_tready        (s_axis_tready),
        .s_axis_tlast         (s_axis_tlast),
        .s_axis_tuser         (s_axis_tuser),
        .m_axis_tdata         (m_axis_tdata),
        .m_axis_tvalid        (m_axis_tvalid),
        .m_axis_tready        (m_axis_tready),
        .m_axis_tlast         (m_axis_tlast),
        .m_axis_tuser         (m_axis_tuser),
        .rp_m_axis_tdata      (rp_m_axis_tdata),
        .rp_m_axis_tvalid     (rp_m_axis_tvalid),
        .rp_m_axis_tready     (rp_m_axis_tready),
        .rp_m_axis_tlast      (rp_m_axis_tlast),
        .rp_s_axis_tdata      (rp_s_axis_tdata),
        .rp_s_axis_tvalid     (rp_s_axis_tvalid),
        .rp_s_axis_tready     (rp_s_axis_tready),
        .rp_s_axis_tlast      (rp_s_axis_tlast),
        .m_axis_setsel_tdata  (m_axis_setsel_tdata),
        .m_axis_setsel_tvalid (m_axis_setsel_tvalid),
        .m_axis_setsel_tready (m_axis_setsel_tready)
    );

endmodule
