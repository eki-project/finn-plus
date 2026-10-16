// Small synthesizable design whose instance names mimic FINN node names, nested one
// level deep so the bare-name hierarchical lookup is exercised.
module fp_leaf #(parameter W = 24) (
  input  wire          clk,
  input  wire [W-1:0]  a,
  input  wire [W-1:0]  b,
  output reg  [2*W-1:0] y
);
  reg [2*W-1:0] acc;
  always @(posedge clk) begin
    acc <= a * b;
    y   <= acc ^ {(2*W){1'b1}};
  end
endmodule

module fp_mem (
  input  wire        clk,
  input  wire [10:0] addr,
  input  wire [31:0] din,
  input  wire        we,
  output reg  [31:0] dout
);
  (* ram_style = "block" *) reg [31:0] ram [0:2047];
  always @(posedge clk) begin
    if (we) ram[addr] <= din;
    dout <= ram[addr];
  end
endmodule

module fp_wrapper (
  input  wire        clk,
  input  wire [23:0] a,
  input  wire [23:0] b,
  output wire [47:0] y0,
  output wire [47:0] y1
);
  fp_leaf MVAU_hls_0 (.clk(clk), .a(a), .b(b), .y(y0));
  fp_leaf MVAU_hls_1 (.clk(clk), .a(b), .b(a), .y(y1));
endmodule

module top (
  input  wire        clk,
  input  wire [23:0] a,
  input  wire [23:0] b,
  input  wire [10:0] addr,
  input  wire [31:0] din,
  input  wire        we,
  output wire [47:0] y0,
  output wire [47:0] y1,
  output wire [31:0] dout
);
  fp_wrapper hier (.clk(clk), .a(a), .b(b), .y0(y0), .y1(y1));
  fp_mem StreamingFIFO_rtl_0 (.clk(clk), .addr(addr), .din(din), .we(we), .dout(dout));
endmodule
