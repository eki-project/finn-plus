//Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2022.2 (lin64) Build 3671981 Fri Oct 14 04:59:54 MDT 2022
//Date        : Mon Mar 31 20:14:03 2025
//Host        : finn-container running 64-bit Ubuntu 22.04.1 LTS
//Command     : generate_target finn_design_wrapper.bd
//Design      : finn_design_wrapper
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module finn_design_wrapper
   (ap_clk,
    ap_rst_n,
    m_axis_0_tdata,
    m_axis_0_tready,
    m_axis_0_tvalid,
    s_axis_0_tdata,
    s_axis_0_tready,
    s_axis_0_tvalid);
  input ap_clk;
  input ap_rst_n;
  output [7:0]m_axis_0_tdata;
  input m_axis_0_tready;
  output m_axis_0_tvalid;
  input [23:0]s_axis_0_tdata;
  output s_axis_0_tready;
  input s_axis_0_tvalid;

  wire ap_clk;
  wire ap_rst_n;
  wire [7:0]m_axis_0_tdata;
  wire m_axis_0_tready;
  wire m_axis_0_tvalid;
  wire [23:0]s_axis_0_tdata;
  wire s_axis_0_tready;
  wire s_axis_0_tvalid;

  finn_design finn_design_i
       (.ap_clk(ap_clk),
        .ap_rst_n(ap_rst_n),
        .m_axis_0_tdata(m_axis_0_tdata),
        .m_axis_0_tready(m_axis_0_tready),
        .m_axis_0_tvalid(m_axis_0_tvalid),
        .s_axis_0_tdata(s_axis_0_tdata),
        .s_axis_0_tready(s_axis_0_tready),
        .s_axis_0_tvalid(s_axis_0_tvalid));
endmodule
