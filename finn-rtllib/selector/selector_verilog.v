module selector_verilog #(
    parameter N          = 4,
    parameter ADDR_WIDTH = 8
)(
    input  wire                   aclk,
    input  wire                   aresetn,

    // AXI-Lite Write Address channel
    output wire                   s_axilite_awready,
    input  wire                   s_axilite_awvalid,
    input  wire [2:0]             s_axilite_awprot,
    input  wire [ADDR_WIDTH-1:0]  s_axilite_awaddr,

    // AXI-Lite Write Data channel
    output wire        s_axilite_wready,
    input  wire        s_axilite_wvalid,
    input  wire [3:0]  s_axilite_wstrb,
    input  wire [31:0] s_axilite_wdata,

    // AXI-Lite Write Response channel
    input  wire        s_axilite_bready,
    output wire        s_axilite_bvalid,
    output wire [1:0]  s_axilite_bresp,

    // AXI-Lite Read Address channel
    output wire                   s_axilite_arready,
    input  wire                   s_axilite_arvalid,
    input  wire [2:0]             s_axilite_arprot,
    input  wire [ADDR_WIDTH-1:0]  s_axilite_araddr,

    // AXI-Lite Read Data channel
    input  wire        s_axilite_rready,
    output wire        s_axilite_rvalid,
    output wire [1:0]  s_axilite_rresp,
    output wire [31:0] s_axilite_rdata,

    // AXI4-Stream Master Output
    output wire        m_axis_tvalid,
    input  wire        m_axis_tready,
    output wire [15:0] m_axis_tdata
);

    selector #(
        .N          (N),
        .ADDR_WIDTH (ADDR_WIDTH)
    ) u_selector (
        .aclk                (aclk),
        .aresetn             (aresetn),

        .s_axilite_awready   (s_axilite_awready),
        .s_axilite_awvalid   (s_axilite_awvalid),
        .s_axilite_awprot    (s_axilite_awprot),
        .s_axilite_awaddr    (s_axilite_awaddr),

        .s_axilite_wready    (s_axilite_wready),
        .s_axilite_wvalid    (s_axilite_wvalid),
        .s_axilite_wstrb     (s_axilite_wstrb),
        .s_axilite_wdata     (s_axilite_wdata),

        .s_axilite_bready    (s_axilite_bready),
        .s_axilite_bvalid    (s_axilite_bvalid),
        .s_axilite_bresp     (s_axilite_bresp),

        .s_axilite_arready   (s_axilite_arready),
        .s_axilite_arvalid   (s_axilite_arvalid),
        .s_axilite_arprot    (s_axilite_arprot),
        .s_axilite_araddr    (s_axilite_araddr),

        .s_axilite_rready    (s_axilite_rready),
        .s_axilite_rvalid    (s_axilite_rvalid),
        .s_axilite_rresp     (s_axilite_rresp),
        .s_axilite_rdata     (s_axilite_rdata),

        .m_axis_tvalid       (m_axis_tvalid),
        .m_axis_tready       (m_axis_tready),
        .m_axis_tdata        (m_axis_tdata)
    );

endmodule
