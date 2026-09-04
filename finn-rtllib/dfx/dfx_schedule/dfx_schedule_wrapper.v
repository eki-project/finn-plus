module dfx_schedule_wrapper #(
    parameter integer SCHEDULE_SLOTS = 4,
    parameter integer MAX_TRIGGERS   = 8,
    parameter integer ADDR_WIDTH     = 8
)(
    input  wire                    aclk,
    input  wire                    aresetn,

    output wire                    s_axilite_awready,
    input  wire                    s_axilite_awvalid,
    input  wire [2:0]              s_axilite_awprot,
    input  wire [ADDR_WIDTH-1:0]   s_axilite_awaddr,
    output wire                    s_axilite_wready,
    input  wire                    s_axilite_wvalid,
    input  wire [3:0]              s_axilite_wstrb,
    input  wire [31:0]             s_axilite_wdata,
    input  wire                    s_axilite_bready,
    output wire                    s_axilite_bvalid,
    output wire [1:0]              s_axilite_bresp,
    output wire                    s_axilite_arready,
    input  wire                    s_axilite_arvalid,
    input  wire [2:0]              s_axilite_arprot,
    input  wire [ADDR_WIDTH-1:0]   s_axilite_araddr,
    input  wire                    s_axilite_rready,
    output wire                    s_axilite_rvalid,
    output wire [1:0]              s_axilite_rresp,
    output wire [31:0]             s_axilite_rdata,

    output wire [MAX_TRIGGERS-1:0] controller_trigger,
    input  wire                    controller_decouple,
    output wire                    accel_decouple_input,
    output wire                    accel_reset,
    output wire                    accel_decouple_output
);

    dfx_schedule #(
        .SCHEDULE_SLOTS (SCHEDULE_SLOTS),
        .MAX_TRIGGERS   (MAX_TRIGGERS),
        .ADDR_WIDTH     (ADDR_WIDTH)
    ) u_dfx_schedule (
        .aclk              (aclk),
        .aresetn           (aresetn),

        .s_axilite_awready (s_axilite_awready),
        .s_axilite_awvalid (s_axilite_awvalid),
        .s_axilite_awprot  (s_axilite_awprot),
        .s_axilite_awaddr  (s_axilite_awaddr),

        .s_axilite_wready  (s_axilite_wready),
        .s_axilite_wvalid  (s_axilite_wvalid),
        .s_axilite_wstrb   (s_axilite_wstrb),
        .s_axilite_wdata   (s_axilite_wdata),

        .s_axilite_bready  (s_axilite_bready),
        .s_axilite_bvalid  (s_axilite_bvalid),
        .s_axilite_bresp   (s_axilite_bresp),

        .s_axilite_arready (s_axilite_arready),
        .s_axilite_arvalid (s_axilite_arvalid),
        .s_axilite_arprot  (s_axilite_arprot),
        .s_axilite_araddr  (s_axilite_araddr),

        .s_axilite_rready  (s_axilite_rready),
        .s_axilite_rvalid  (s_axilite_rvalid),
        .s_axilite_rresp   (s_axilite_rresp),
        .s_axilite_rdata   (s_axilite_rdata),

        .controller_trigger       (controller_trigger),
        .controller_decouple      (controller_decouple),
        .accel_decouple_input (accel_decouple_input),
        .accel_reset       (accel_reset),
        .accel_decouple_output (accel_decouple_output)
    );

endmodule
