// Verilog-2001 wrapper for dfx_wrapper.sv.
// Used as a Vivado block design module reference so parameters can be set via
// set_property CONFIG.* on the bd cell.
`timescale 1ns/1ps
module dfx_wrapper_wrapper #(
    parameter IN_DATA_WIDTH   = 64,
    parameter OUT_DATA_WIDTH  = 64,
    parameter TUSER_WIDTH     = 2,
    parameter NUM_RM          = 2,
    parameter RESET_CYCLES    = 16,
    parameter NUM_OUTPUT_BEATS = 1
) (
    input  wire                       aclk,
    input  wire                       aresetn,

    input  wire [IN_DATA_WIDTH-1:0]   s_axis_tdata,
    input  wire                       s_axis_tvalid,
    output wire                       s_axis_tready,
    input  wire                       s_axis_tlast,
    input  wire [TUSER_WIDTH-1:0]     s_axis_tuser,

    output wire [OUT_DATA_WIDTH-1:0]  m_axis_tdata,
    output wire                       m_axis_tvalid,
    input  wire                       m_axis_tready,
    output wire                       m_axis_tlast,
    output wire [TUSER_WIDTH-1:0]     m_axis_tuser,

    output wire [IN_DATA_WIDTH-1:0]   rp_m_axis_tdata,
    output wire                       rp_m_axis_tvalid,
    input  wire                       rp_m_axis_tready,
    output wire                       rp_m_axis_tlast,

    input  wire [OUT_DATA_WIDTH-1:0]  rp_s_axis_tdata,
    input  wire                       rp_s_axis_tvalid,
    output wire                       rp_s_axis_tready,
    input  wire                       rp_s_axis_tlast,

    output wire [NUM_RM-1:0]          controller_trigger,
    input  wire                       controller_decouple,

    output wire                       accel_reset_n
);

    dfx_wrapper #(
        .IN_DATA_WIDTH   (IN_DATA_WIDTH),
        .OUT_DATA_WIDTH  (OUT_DATA_WIDTH),
        .TUSER_WIDTH     (TUSER_WIDTH),
        .NUM_RM          (NUM_RM),
        .RESET_CYCLES    (RESET_CYCLES),
        .NUM_OUTPUT_BEATS(NUM_OUTPUT_BEATS)
    ) inst (
        .aclk                (aclk),
        .aresetn             (aresetn),
        .s_axis_tdata        (s_axis_tdata),
        .s_axis_tvalid       (s_axis_tvalid),
        .s_axis_tready       (s_axis_tready),
        .s_axis_tlast        (s_axis_tlast),
        .s_axis_tuser        (s_axis_tuser),
        .m_axis_tdata        (m_axis_tdata),
        .m_axis_tvalid       (m_axis_tvalid),
        .m_axis_tready       (m_axis_tready),
        .m_axis_tlast        (m_axis_tlast),
        .m_axis_tuser        (m_axis_tuser),
        .rp_m_axis_tdata     (rp_m_axis_tdata),
        .rp_m_axis_tvalid    (rp_m_axis_tvalid),
        .rp_m_axis_tready    (rp_m_axis_tready),
        .rp_m_axis_tlast     (rp_m_axis_tlast),
        .rp_s_axis_tdata     (rp_s_axis_tdata),
        .rp_s_axis_tvalid    (rp_s_axis_tvalid),
        .rp_s_axis_tready    (rp_s_axis_tready),
        .rp_s_axis_tlast     (rp_s_axis_tlast),
        .controller_trigger  (controller_trigger),
        .controller_decouple (controller_decouple),
        .accel_reset_n       (accel_reset_n)
    );

endmodule
