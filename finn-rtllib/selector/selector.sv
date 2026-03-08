module selector #(
    parameter int unsigned N          = 4,
    parameter int unsigned ADDR_WIDTH = 8
)(
    input  logic  aclk,
    input  logic  aresetn,

    output logic                   s_axilite_awready,
    input  logic                   s_axilite_awvalid,
    input  logic [2:0]             s_axilite_awprot,
    input  logic [ADDR_WIDTH-1:0]  s_axilite_awaddr,
    output logic        s_axilite_wready,
    input  logic        s_axilite_wvalid,
    input  logic [3:0]  s_axilite_wstrb,
    input  logic [31:0] s_axilite_wdata,
    input  logic        s_axilite_bready,
    output logic        s_axilite_bvalid,
    output logic [1:0]  s_axilite_bresp,
    output logic                   s_axilite_arready,
    input  logic                   s_axilite_arvalid,
    input  logic [2:0]             s_axilite_arprot,
    input  logic [ADDR_WIDTH-1:0]  s_axilite_araddr,
    input  logic        s_axilite_rready,
    output logic        s_axilite_rvalid,
    output logic [1:0]  s_axilite_rresp,
    output logic [31:0] s_axilite_rdata,
    output logic        m_axis_tvalid,
    input  logic        m_axis_tready,
    output logic [15:0] m_axis_tdata
);
    localparam int unsigned BSEL_BITS      = 2;
    localparam int unsigned IP_ADDR_WIDTH0 = ADDR_WIDTH - BSEL_BITS;
    localparam int unsigned IP_ADDR_WIDTH  = IP_ADDR_WIDTH0 ? IP_ADDR_WIDTH0 : 1;

    logic                      ip_en;
    logic                      ip_wen;
    logic [IP_ADDR_WIDTH-1:0]  ip_addr;
    logic [31:0]               ip_wdata;
    logic                      ip_rack;
    logic [31:0]               ip_rdata;

    axilite #(
        .ADDR_WIDTH    (ADDR_WIDTH),
        .DATA_WIDTH    (32),
        .IP_DATA_WIDTH (32)
    ) u_axilite (
        .aclk     (aclk),
        .aresetn  (aresetn),

        .awready  (s_axilite_awready),
        .awvalid  (s_axilite_awvalid),
        .awprot   (s_axilite_awprot),
        .awaddr   (s_axilite_awaddr),

        .wready   (s_axilite_wready),
        .wvalid   (s_axilite_wvalid),
        .wstrb    (s_axilite_wstrb),
        .wdata    (s_axilite_wdata),

        .bready   (s_axilite_bready),
        .bvalid   (s_axilite_bvalid),
        .bresp    (s_axilite_bresp),

        .arready  (s_axilite_arready),
        .arvalid  (s_axilite_arvalid),
        .arprot   (s_axilite_arprot),
        .araddr   (s_axilite_araddr),

        .rready   (s_axilite_rready),
        .rvalid   (s_axilite_rvalid),
        .rresp    (s_axilite_rresp),
        .rdata    (s_axilite_rdata),

        .ip_en    (ip_en),
        .ip_wen   (ip_wen),
        .ip_addr  (ip_addr),
        .ip_wdata (ip_wdata),
        .ip_rack  (ip_rack),
        .ip_rdata (ip_rdata)
    );

    //   regs[0]   – control/status register
    //   regs[1..N] – set registers
    logic [31:0] regs [0:N];

    logic        ip_rack_d  = '0;
    logic [31:0] ip_rdata_d = '0;

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            ip_rack_d  <= '0;
            ip_rdata_d <= '0;
            for (int i = 0; i <= N; i++) regs[i] <= '0;
        end else begin
            ip_rack_d <= '0;
            if (ip_en) begin
                if (ip_wen) begin
                    if (ip_addr <= IP_ADDR_WIDTH'(N))
                        regs[ip_addr] <= ip_wdata;
                end else begin
                    ip_rack_d  <= '1;
                    ip_rdata_d <= (ip_addr <= IP_ADDR_WIDTH'(N)) ? regs[ip_addr] : '0;
                end
            end
        end
    end

    assign ip_rack  = ip_rack_d;
    assign ip_rdata = ip_rdata_d;

    logic running;
    assign running = regs[0][0];

    localparam int unsigned IDX_BITS = (N > 1) ? $clog2(N) : 1;

    logic [IDX_BITS-1:0] next_ptr = '0;
    logic loaded = '0;

    // logic [15:0] cur_reps;
    // assign cur_reps = regs[next_ptr + 1][31:16];

    logic fired;
    assign fired = m_axis_tvalid & m_axis_tready;
    assign m_axis_tvalid = running & loaded;
    assign m_axis_tdata  = regs[next_ptr + 1][15:0];

    logic [15:0] rep_cnt;

    always_ff @(posedge aclk) begin
        if (!aresetn || !running) begin
            next_ptr <= '0;
            loaded  <= '0;
            rep_cnt <= '0;
        end else begin
            if (!loaded) begin
                rep_cnt <= regs[1][31:16];
                next_ptr <= '0;
                loaded <= '1;
            end else begin
                if (fired) begin
                    rep_cnt <= rep_cnt - 16'd1;
                    if (rep_cnt == 16'd1) begin
                        if ((next_ptr == IDX_BITS'(N - 1)) || (regs[next_ptr + 2][31:16] == 16'd0)) begin
                            next_ptr <= '0;
                            rep_cnt  <= regs[1][31:16];
                        end else begin
                            next_ptr <= next_ptr + IDX_BITS'(1);
                            rep_cnt  <= regs[next_ptr + 2][31:16];
                        end
                    end
                end
            end
        end
    end
endmodule : selector
