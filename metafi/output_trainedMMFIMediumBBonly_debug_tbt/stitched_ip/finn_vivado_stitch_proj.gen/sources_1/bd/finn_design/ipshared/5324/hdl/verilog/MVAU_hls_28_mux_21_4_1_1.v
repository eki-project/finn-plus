// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================

`timescale 1ns/1ps

module MVAU_hls_28_mux_21_4_1_1 #(
parameter
    ID                = 0,
    NUM_STAGE         = 1,
    din0_WIDTH       = 32,
    din1_WIDTH       = 32,
    din2_WIDTH         = 32,
    dout_WIDTH            = 32
)(
    input  [3 : 0]     din0,
    input  [3 : 0]     din1,
    input  [0 : 0]    din2,
    output [3 : 0]   dout);

// puts internal signals
wire [0 : 0]     sel;
// level 1 signals
wire [3 : 0]         mux_1_0;

assign sel = din2;

// Generate level 1 logic
assign mux_1_0 = (sel[0] == 0)? din0 : din1;

// output logic
assign dout = mux_1_0;

endmodule
