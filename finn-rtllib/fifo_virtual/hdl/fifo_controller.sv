/**
 * Copyright (c) 2025, Paderborn University
 *
 * @author	Felix P. Jentzsch <felix.jentzsch@upb.de>
 */

module fifo_controller #(
    localparam int unsigned  ADDR_WIDTH = 32,
    localparam int unsigned  DATA_WIDTH = 32,
    localparam int unsigned  IP_ADDR_WIDTH = 30,
    localparam int unsigned  IP_DATA_WIDTH = 32,
)(
	// Global Control
	input	logic  aclk,
	input	logic  aresetn,

	// AXI-lite Write Channels
	output	logic  awready,
	input	logic  awvalid,
	input	logic [2:0]  awprot,
	input	logic [ADDR_WIDTH-1:0]  awaddr,

	output	logic  wready,
	input	logic  wvalid,
	input	logic [DATA_WIDTH/8-1:0]  wstrb,
	input	logic [DATA_WIDTH  -1:0]  wdata,

	input	logic  bready,
	output	logic  bvalid,
	output	logic [1:0]  bresp,

	// AXI-lite Read Channels
	output	logic  arready,
	input	logic  arvalid,
	input	logic [2:0]  arprot,
	input	logic [ADDR_WIDTH-1:0]  araddr,

	input	logic  rready,
	output	logic  rvalid,
	output	logic [1:0]  rresp,
	output	logic [DATA_WIDTH-1:0]  rdata,

	// FIFO Configuration Ring Bus
	input	logic [7:0]  icfg,
	output	logic [7:0]  ocfg
);
	import  fifo_gauge_pkg::*;

    //-----------------------------------------------------------------------
    // AXI-lite to ap_memory Adapter
	uwire [IP_ADDR_WIDTH-1:0]  config_addr;
	uwire  config_en;
	uwire  config_wen;
	uwire  config_rack;
	uwire [IP_DATA_WIDTH-1:0]  config_wdata;
	uwire [IP_DATA_WIDTH-1:0]  config_rdata;
	axilite #(
		.ADDR_WIDTH(ADDR_WIDTH),
		.DATA_WIDTH(DATA_WIDTH),
		.IP_DATA_WIDTH(IP_DATA_WIDTH)
	) cfg (
		.aclk(aclk), .aresetn(aresetn),

		// Write Channels
		.awready, .awvalid, .awaddr, .awprot,
		.wready,  .wvalid,  .wdata,  .wstrb,
		.bready,  .bvalid,  .bresp,

		// Read Channels
		.arready, .arvalid, .araddr, .arprot,
		.rready,  .rvalid,  .rresp,  .rdata,

		// IP-side Interface
		.ip_en(config_en),
		.ip_wen(config_wen),
		.ip_addr(config_addr),
		.ip_wdata(config_wdata),
		.ip_rack(config_rack),
		.ip_rdata(config_rdata)
	);

    //-----------------------------------------------------------------------
    // Configuration bus input buffer (always the last 7 cycles = 7 bytes = 1 packet)
    // This is needed in case the ring is shallower than the maximum packet size
    // and makes reading return values easier
    logic [7:0] icfg_buffer[6:0];

	always_ff @(posedge aclk) begin
		if (!aresetn) begin
			icfg_buffer <= '{default: 8'h0};
		end else begin
			icfg_buffer[0] <= icfg;
			for (int i = 1; i < 7; i++) begin
				icfg_buffer[i] <= icfg_buffer[i-1];
			end
		end
	end

	//-----------------------------------------------------------------------
	// Main controller logic

	// State definition
	typedef enum logic [1:0] {
		IDLE  = 2'b00,
		ISSUE = 2'b01,
		WAIT  = 2'b10,
		ACK   = 2'b11
	} state_t;

	state_t state, state_next;
	logic [2:0] counter, counter_next;  // Counter for 1-7 cycles (0-6 + 1)

    // Register to buffer addr & wdata during WRITE_FILL command
    // The AXI-lite adapter expects the write to complete in a single cycle, but we need
    // multiple cycles to issue and propagate the instruction over the ring bus
    logic [IP_ADDR_WIDTH-1:0] instr_addr, instr_addr_next;
    logic [IP_DATA_WIDTH-1:0] instr_wdata, instr_wdata_next;

    // Decode instruction & FIFO ID from address (29:24 are unused)
    uwire [7:0] op = instr_addr[7:0];
    uwire [15:0] fifo_id = instr_addr[23:8]; // todo: double-check endianness throughout!
    // Packet size lookup
    uwire [2:0] packet_size;
    always_comb begin
        unique casex (op)
        M_RUN:         packet_size = 3'd1; // 1 byte packet [OP]
        BARRIER_CLEAN: packet_size = 3'd1; // 1 byte packet [OP]
        COMP_PERIOD:   packet_size = 3'd5; // 5 byte packet [OP, 32 bit value]
        READ_STALL:    packet_size = 3'd4; // 4 byte packet [OP, 16 bit ID, status byte]
        READ_FILL:     packet_size = 3'd7; // 7 byte packet [OP, 16 bit ID, 32 bit value]
        WRITE_FILL:    packet_size = 3'd7; // 7 byte packet [OP, 16 bit ID, 32 bit value]
        default:       packet_size = 3'd1; // NOP or unknown -> treat as NOP
        endcase
    end

	// State register
	always_ff @(posedge aclk) begin
		if (!aresetn) begin
			state <= IDLE;
			counter <= 3'd0;
            instr_addr <= 0;
            instr_wdata <= 0;
		end else begin
			state <= state_next;
			counter <= counter_next;
            instr_addr <= instr_addr_next;
            instr_wdata <= instr_wdata_next;
		end
	end

	// Next state logic
	always_comb begin
		state_next = state;
		counter_next = counter;
        instr_addr_next = instr_addr;
        instr_wdata_next = instr_wdata;

		unique case (state)
			IDLE: begin
				if (config_en) begin
                    // Read or write operation requested -> issue instruction
                    state_next = ISSUE;
                    counter_next = 3'd0;
                    // Register addr & wdata to hold them during the (write) command
                    instr_addr_next = config_addr;
                    instr_wdata_next = config_wdata;

                    // Special case: NOP read -> acknowledge if idle without issuing instruction
                    // This way the driver can wait for completion of the previous async write command
                    if (!config_wen && config_addr[7:0] == NOP) begin
                        state_next = ACK;
                    end
                end
			end

			ISSUE: begin
				if (counter < packet_size-1) begin
					counter_next = counter + 1'b1;
				end else begin
					state_next = WAIT;
				end
			end

			WAIT: begin
                // Wait until the entire package has been read into icfg_buffer
                if (icfg_buffer[6] != NOP) begin
                    state_next = ACK;
                    if (icfg_buffer[6] == BARRIER_DIRTY) begin
                        // Re-issue BARRIER_CLEAN instruction
                        state_next = ISSUE;
                        counter_next = 3'd0;
                    end
                    if (icfg_buffer[6] == WRITE_FILL) begin
                        // Skip ACK for write instruction
                        state_next = IDLE;
                    end
                end
			end

			ACK: begin
				state_next = IDLE;
			end
		endcase
	end

    // Read response logic: assemble return word from icfg_buffer
    // Note that icfg_buffer is already shifted by 1 cycle in the ACK state, such that the
    // packet payload is in icfg_buffer[6:1] instead of [5:0]
    assign config_rack = (state == ACK);
    always_comb begin
        casex (op)
        COMP_PERIOD: begin // 5 byte packet [OP, 32 bit value]
            config_rdata = {icfg_buffer[6], icfg_buffer[5], icfg_buffer[4], icfg_buffer[3]};
        end
        READ_STALL: begin // 4 byte packet [OP, 16 bit ID, status byte]
            config_rdata = {24'b0, icfg_buffer[4]};
        end
        READ_FILL: begin // 7 byte packet [OP, 16 bit ID, 32 bit value]
            config_rdata = {icfg_buffer[4], icfg_buffer[3], icfg_buffer[2], icfg_buffer[1]};
        end
        default: begin
            config_rdata = {24'b0, op}; // Default: return OP code in LSBs
        end
        endcase
    end

    // Configuration bus write logic
    always_comb begin
        ocfg = NOP;
        if (state == ISSUE) begin
            unique casex (op)
            M_RUN: ocfg = op; // 1 byte packet [OP]
            BARRIER_CLEAN: ocfg = op; // 1 byte packet [OP]
            COMP_PERIOD: begin // 5 byte packet [OP, 32 bit value]
                ocfg = counter == 3'd0 ? op : 8'h00;
            end
            READ_STALL: begin // 4 byte packet [OP, 16 bit ID, status byte]
                unique case (counter)
                3'd0: ocfg = op;
                3'd1: ocfg = fifo_id[7:0];
                3'd2: ocfg = fifo_id[15:8];
                3'd3: ocfg = 8'h00;
                endcase
            end
            READ_FILL: begin // 7 byte packet [OP, 16 bit ID, 32 bit value]
                unique case (counter)
                3'd0: ocfg = op;
                3'd1: ocfg = fifo_id[7:0];
                3'd2: ocfg = fifo_id[15:8];
                3'd3: ocfg = 8'h00;
                3'd4: ocfg = 8'h00;
                3'd5: ocfg = 8'h00;
                3'd6: ocfg = 8'h00;
                endcase
            end
            WRITE_FILL: begin // 7 byte packet [OP, 16 bit ID, 32 bit value]
                unique case (counter)
                3'd0: ocfg = op;
                3'd1: ocfg = fifo_id[7:0];
                3'd2: ocfg = fifo_id[15:8];
                3'd3: ocfg = wdata[7:0];
                3'd4: ocfg = wdata[15:8];
                3'd5: ocfg = wdata[23:16];
                3'd6: ocfg = wdata[31:24];
                endcase
            end
            endcase
        end
    end

endmodule : fifo_controller
