module dfx_schedule #(
    parameter int unsigned SCHEDULE_SLOTS = 4,
    parameter int unsigned MAX_TRIGGERS   = 8,
    parameter int unsigned ADDR_WIDTH     = 8
)(
    input  logic  aclk,
    input  logic  aresetn,

    output logic                   s_axilite_awready,
    input  logic                   s_axilite_awvalid,
    input  logic [2:0]             s_axilite_awprot,
    input  logic [ADDR_WIDTH-1:0]  s_axilite_awaddr,
    output logic                   s_axilite_wready,
    input  logic                   s_axilite_wvalid,
    input  logic [3:0]             s_axilite_wstrb,
    input  logic [31:0]            s_axilite_wdata,
    input  logic                   s_axilite_bready,
    output logic                   s_axilite_bvalid,
    output logic [1:0]             s_axilite_bresp,
    output logic                   s_axilite_arready,
    input  logic                   s_axilite_arvalid,
    input  logic [2:0]             s_axilite_arprot,
    input  logic [ADDR_WIDTH-1:0]  s_axilite_araddr,
    input  logic                   s_axilite_rready,
    output logic                   s_axilite_rvalid,
    output logic [1:0]             s_axilite_rresp,
    output logic [31:0]            s_axilite_rdata,

    output logic [MAX_TRIGGERS-1:0] controller_trigger,
    input  logic                    controller_decouple,
    output logic                    accel_decouple_input,
    output logic                    accel_decouple_output,
    output logic                    accel_reset
);

    localparam int unsigned BSEL_BITS = 2;
    localparam int unsigned IP_ADDR_WIDTH0 = ADDR_WIDTH - BSEL_BITS;
    localparam int unsigned IP_ADDR_WIDTH = IP_ADDR_WIDTH0 ? IP_ADDR_WIDTH0 : 1;
    localparam int unsigned N = 7 + 3 * SCHEDULE_SLOTS;
    localparam int unsigned PREDEC_MSB_IDX = 5;
    localparam int unsigned PREDEC_LSB_IDX = 6;
    localparam int unsigned SLOT_BITS = (SCHEDULE_SLOTS > 1) ? $clog2(SCHEDULE_SLOTS) : 1;

    logic ip_en;
    logic ip_wen;
    logic [IP_ADDR_WIDTH-1:0] ip_addr;
    logic [31:0] ip_wdata;
    logic ip_rack;
    logic [31:0] ip_rdata;

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

    logic [31:0] regs [0:N-1];

    logic running;
    logic [63:0] max_cycles;
    logic [63:0] min_cycles;
    logic [63:0] pre_decouple_cycles;

    assign running = regs[0][0];
    assign max_cycles = {regs[1], regs[2]};
    assign min_cycles = {regs[3], regs[4]};
    assign pre_decouple_cycles = {regs[PREDEC_MSB_IDX], regs[PREDEC_LSB_IDX]};

    typedef enum logic [2:0] {
        S_IDLE = 3'd0,
        S_PRE_DECOUPLE = 3'd1,
        S_TRIGGER = 3'd2,
        S_WAIT_DECOUPLE = 3'd3,
        S_WAIT_CYCLES = 3'd4
    } state_t;

    state_t state = S_IDLE;
    logic [SLOT_BITS-1:0] slot_ptr = '0;
    logic [63:0] decouple_cnt = '0;
    logic [63:0] wait_cnt = '0;
    logic [63:0] pre_decouple_cnt = '0;
    logic [3:0] reset_cnt = '0;
    logic [10:0] decouple_out_dly_cnt = '0;
    logic controller_decouple_d = '0;
    logic reset_a = '1;
    logic decouple_accel_out_int = '0;

    logic [31:0] slot_rm_id;
    logic [63:0] slot_wait_cycles;

    assign slot_rm_id = regs[7 + 3 * slot_ptr];
    assign slot_wait_cycles = {regs[7 + 3 * slot_ptr + 1], regs[7 + 3 * slot_ptr + 2]};

    logic error_set;
    assign error_set = running && (
        (controller_decouple && (state == S_IDLE || state == S_WAIT_CYCLES || state == S_PRE_DECOUPLE)) ||
        ((state == S_WAIT_DECOUPLE) && (wait_cnt == 64'd0) && !controller_decouple && !controller_decouple_d)
    );

    logic decouple_done;
    assign decouple_done = (state == S_WAIT_DECOUPLE) && !controller_decouple && controller_decouple_d;
    assign accel_reset = reset_a;
    assign accel_decouple_output = decouple_accel_out_int;

    always_ff @(posedge aclk) begin
        if (!aresetn || !running) begin
            state <= S_IDLE;
            slot_ptr <= '0;
            decouple_cnt <= '0;
            wait_cnt <= '0;
            pre_decouple_cnt <= '0;
            reset_cnt <= '0;
            decouple_out_dly_cnt <= '0;
            controller_trigger <= '0;
            accel_decouple_input <= '0;
            decouple_accel_out_int <= '0;
            controller_decouple_d <= '0;
            reset_a <= '1;
        end else begin
            controller_decouple_d <= controller_decouple;

            case (state)
                S_IDLE: begin
                    slot_ptr <= '0;
                    pre_decouple_cnt <= pre_decouple_cycles;
                    accel_decouple_input <= '0;
                    decouple_accel_out_int <= '0;
                    state <= S_PRE_DECOUPLE;
                    reset_a <= '1;
                end

                S_PRE_DECOUPLE: begin
                    if (slot_rm_id == '0 || slot_wait_cycles == '0) begin
                        slot_ptr <= '0;
                        pre_decouple_cnt <= pre_decouple_cycles;
                        state <= S_PRE_DECOUPLE;
                    end else begin
                        accel_decouple_input <= 1'b1;
                        if (!controller_decouple) begin
                            if (pre_decouple_cnt == 64'd0) begin
                                state <= S_TRIGGER;
                                decouple_accel_out_int <= 1'b1;
                            end else begin
                                pre_decouple_cnt <= pre_decouple_cnt - 64'd1;
                            end
                        end
                    end
                end

                S_TRIGGER: begin
                    controller_trigger <= MAX_TRIGGERS'(slot_rm_id);
                    decouple_cnt <= '0;
                    wait_cnt <= slot_wait_cycles;
                    if (!controller_decouple)
                        state <= S_WAIT_DECOUPLE;
                end

                S_WAIT_DECOUPLE: begin
                    controller_trigger <= '0;
                    if (wait_cnt == 64'd0) begin
                        state <= S_IDLE;
                    end else if (controller_decouple) begin
                        wait_cnt <= wait_cnt - 64'd1;
                        decouple_cnt <= decouple_cnt + 64'd1;
                    end else if (controller_decouple_d) begin
                        wait_cnt <= wait_cnt - 64'd1;
                        state <= S_WAIT_CYCLES;
                        reset_a <= '0;
                        reset_cnt <= 4'd8;
                    end else begin
                        wait_cnt <= wait_cnt - 64'd1;
                        state    <= S_WAIT_DECOUPLE;
                    end
                end

                S_WAIT_CYCLES: begin
                    if (reset_cnt > 4'd0) begin
                        reset_a <= '0;
                        reset_cnt <= reset_cnt - 4'd1;
                        if (reset_cnt == 4'd1)
                            decouple_out_dly_cnt <= 10'd100;
                    end else if (decouple_out_dly_cnt > 10'd0) begin
                        reset_a <= '1;
                        decouple_out_dly_cnt <= decouple_out_dly_cnt - 10'd1;


                    end else begin
                        reset_a <= '1;
                        decouple_accel_out_int <= '0;
                        accel_decouple_input <= '0;
                    end
                    if (wait_cnt == 64'd0) begin
                        if (slot_ptr == SLOT_BITS'(SCHEDULE_SLOTS - 1))
                            slot_ptr <= '0;
                        else
                            slot_ptr <= slot_ptr + SLOT_BITS'(1);
                            pre_decouple_cnt <= pre_decouple_cycles;
                            state <= S_PRE_DECOUPLE;
                    end else begin
                        wait_cnt <= wait_cnt - 64'd1;
                    end
                end

                default: begin
                    accel_decouple_input <= '0;
                    decouple_accel_out_int <= '0;
                    state <= S_IDLE;
                end
            endcase
        end
    end

    logic ip_rack_d = '0;
    logic [31:0] ip_rdata_d = '0;

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            for (int i = 0; i < N; i++) regs[i] <= '0;
            regs[3] <= '1;
            regs[4] <= '1;
            ip_rack_d <= '0;
            ip_rdata_d <= '0;
        end else begin
            ip_rack_d <= '0;

            if (ip_en && ip_wen) begin
                if (ip_addr <= IP_ADDR_WIDTH'(N))
                    regs[ip_addr] <= ip_wdata;
            end

            if (ip_en && !ip_wen) begin
                ip_rack_d <= '1;
                ip_rdata_d <= (ip_addr <= IP_ADDR_WIDTH'(N)) ? regs[ip_addr] : '0;
            end

            if (error_set)
                regs[0][1] <= 1'b1;


            if (decouple_done) begin
                if (decouple_cnt > {regs[1], regs[2]}) begin
                    regs[1] <= decouple_cnt[63:32];
                    regs[2] <= decouple_cnt[31:0];
                end
                if (decouple_cnt < {regs[3], regs[4]}) begin
                    regs[3] <= decouple_cnt[63:32];
                    regs[4] <= decouple_cnt[31:0];
                end
            end
        end
    end

    assign ip_rack = ip_rack_d;
    assign ip_rdata = ip_rdata_d;

endmodule : dfx_schedule
