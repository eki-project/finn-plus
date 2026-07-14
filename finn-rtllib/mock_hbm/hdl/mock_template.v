module {{ TOP_MODULE_NAME }}(
//- Global Control ------------------
(* X_INTERFACE_PARAMETER = "ASSOCIATED_RESET = ap_rst_n" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
input   ap_clk,
(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
input   ap_rst_n,

//- AXI4 Slave - Write Address -----
input  [{{ ADDR_WIDTH }}-1:0] s_axi_awaddr,
input  [1:0]            s_axi_awburst,
input  [3:0]            s_axi_awcache,
input  [1:0]            s_axi_awid,
input  [7:0]            s_axi_awlen,
input                   s_axi_awlock,
input  [2:0]            s_axi_awprot,
input  [2:0]            s_axi_awsize,
input                   s_axi_awvalid,
output                  s_axi_awready,

//- AXI4 Slave - Write Data --------
input  [{{ DATA_WIDTH }}-1:0] s_axi_wdata,
input  [{{ DATA_BYTES }}-1:0] s_axi_wstrb,
input                    s_axi_wvalid,
input                    s_axi_wlast,
output                   s_axi_wready,

//- AXI4 Slave - Write Response ----
output reg [1:0]         s_axi_bid,
output reg [1:0]         s_axi_bresp,
output reg               s_axi_bvalid,
input                    s_axi_bready,

//- AXI4 Slave - Read Address ------
input  [{{ ADDR_WIDTH }}-1:0] s_axi_araddr,
input  [1:0]              s_axi_arburst,
input  [3:0]              s_axi_arcache,
input  [1:0]              s_axi_arid,
input  [7:0]              s_axi_arlen,
input                     s_axi_arlock,
input  [2:0]              s_axi_arprot,
input  [2:0]              s_axi_arsize,
input                    s_axi_arvalid,
output                   s_axi_arready,

//- AXI4 Slave - Read Data ---------
output reg [1:0]             s_axi_rid,
output reg [{{ DATA_WIDTH }}-1:0] s_axi_rdata,
output reg [1:0]             s_axi_rresp,
output reg                   s_axi_rvalid,
output reg                   s_axi_rlast,
input                        s_axi_rready
);

parameter integer LATENCY = 100;

localparam integer SHIFT_W = ({{ DATA_WIDTH }} <= 1) ? 1 : $clog2({{ DATA_WIDTH }} / 8);
localparam integer LAT_CNT_W = (LATENCY <= 1) ? 1 : $clog2(LATENCY + 1);

reg write_active;
reg write_aw_captured;
reg [LAT_CNT_W-1:0] write_latency_cnt;
reg [7:0] write_len;
reg [7:0] write_beat_idx;
reg write_all_beats_received;

reg read_active;
reg [LAT_CNT_W-1:0] read_latency_cnt;
reg [1:0] read_id;
reg [{{ ADDR_WIDTH }}-1:0] read_addr;
reg [7:0] read_len;
reg [2:0] read_size;
reg [1:0] read_burst;
reg [7:0] read_beat_idx;

function automatic [{{ ADDR_WIDTH }}-1:0] calc_next_addr;
    input [{{ ADDR_WIDTH }}-1:0] curr_addr;
    input [2:0] size;
    input [1:0] burst;
    input [7:0] len;
    reg [31:0] beat_bytes;
    reg [31:0] total_bytes;
    reg [{{ ADDR_WIDTH }}-1:0] base_addr;
    reg [{{ ADDR_WIDTH }}-1:0] next_a;
begin
    beat_bytes = (1 << size);
    if (beat_bytes == 0) beat_bytes = 1;
    case (burst)
        2'b00: begin // FIXED
            next_a = curr_addr;
        end
        2'b01: begin // INCR
            next_a = curr_addr + beat_bytes;
        end
        2'b10: begin // WRAP
            total_bytes = beat_bytes * (len + 1);
            if (total_bytes == 0) total_bytes = beat_bytes;
            base_addr = (curr_addr / total_bytes) * total_bytes;
            next_a = curr_addr + beat_bytes;
            if (next_a >= (base_addr + total_bytes)) begin
                next_a = base_addr;
            end
        end
        default: begin // reserved burst type
            next_a = curr_addr;
        end
    endcase
    calc_next_addr = next_a;
end
endfunction

assign s_axi_awready = !write_active && !write_aw_captured;
assign s_axi_wready  = write_aw_captured && write_active;
assign s_axi_arready = !read_active;

always @(posedge ap_clk) begin
	if (!ap_rst_n) begin
		write_active <= 1'b0;
		write_aw_captured <= 1'b0;
		write_latency_cnt <= {LAT_CNT_W{1'b0}};
		write_len <= 8'd0;
		write_beat_idx <= 8'd0;
		write_all_beats_received <= 1'b0;

		read_active <= 1'b0;
		read_latency_cnt <= {LAT_CNT_W{1'b0}};
		read_id <= 2'b00;
		read_addr <= {({{ ADDR_WIDTH }}){1'b0}};
		read_len <= 8'd0;
		read_size <= 3'd0;
		read_burst <= 2'b01;
		read_beat_idx <= 8'd0;

		s_axi_bid    <= 2'b00;
		s_axi_bresp  <= 2'b00; // OKAY
		s_axi_bvalid <= 1'b0;
		s_axi_rid    <= 2'b00;
		s_axi_rdata  <= {({{ DATA_WIDTH }}){1'b0}};
		s_axi_rresp  <= 2'b00; // OKAY
		s_axi_rvalid <= 1'b0;
		s_axi_rlast  <= 1'b0;
	end else begin
		// Clear response channels on handshake - ALSO clear write state
		if (s_axi_bvalid && s_axi_bready) begin
			s_axi_bvalid <= 1'b0;
			write_active <= 1'b0;
			write_aw_captured <= 1'b0;
			write_all_beats_received <= 1'b0;
		end

		if (s_axi_rvalid && s_axi_rready) begin
			s_axi_rvalid <= 1'b0;
			s_axi_rlast  <= 1'b0;
		end

		// Capture AW
		if (s_axi_awvalid && s_axi_awready) begin
			write_aw_captured <= 1'b1;
			write_active <= 1'b1;
			write_len <= s_axi_awlen;
			write_beat_idx <= 8'd0;
			write_all_beats_received <= 1'b0;
			write_latency_cnt <= LATENCY - 1;
		end

		// Write data beats - count and validate WLAST
		if (s_axi_wvalid && s_axi_wready) begin
			// Validate WLAST arrives at correct beat
			// Expected: WLAST high only on beat index == AWLEN
			if (s_axi_wlast != ((write_beat_idx + 8'd1) == (write_len + 8'd1))) begin
				// WLAST mismatch - protocol error detected
				// Continue anyway (lenient slave behavior)
			end

			// Always increment beat counter each write beat
			write_beat_idx <= write_beat_idx + 8'd1;

			// Track when all expected beats received
			if ((write_beat_idx + 8'd1) == (write_len + 8'd1)) begin
				write_all_beats_received <= 1'b1;
			end
		end

		// Write latency countdown and B response generation
		if (write_active) begin
			if (write_latency_cnt != 0) begin
				write_latency_cnt <= write_latency_cnt - 1;
			end else begin
				// Latency expired - check if all beats received and response not pending
				if (write_aw_captured && write_all_beats_received && !s_axi_bvalid) begin
					// Issue B response
					s_axi_bvalid <= 1'b1;
					// NOTE: Do NOT clear write_active/write_aw_captured here!
					// They clear only after BREADY handshake (see above)
				end
			end
		end

		// Capture AR
		if (s_axi_arvalid && s_axi_arready) begin
			read_active <= 1'b1;
			read_id <= s_axi_arid;
			read_addr <= s_axi_araddr;
			read_len <= s_axi_arlen;
			read_size <= s_axi_arsize;
			read_burst <= s_axi_arburst;
			read_beat_idx <= 8'd0;
			read_latency_cnt <= LATENCY - 1;
		end

		// Read first-word latency and data beat generation
		if (read_active) begin
			if (read_latency_cnt != 0) begin
				read_latency_cnt <= read_latency_cnt - 1;
			end else if (!s_axi_rvalid || (s_axi_rvalid && s_axi_rready)) begin
				s_axi_rid <= read_id;
				s_axi_rvalid <= 1'b1;
				s_axi_rlast <= ((read_beat_idx + 8'd1) == (read_len + 8'd1));

				if ((read_beat_idx + 8'd1) < (read_len + 8'd1)) begin
					read_addr <= calc_next_addr(read_addr, read_size, read_burst, read_len);
					read_beat_idx <= read_beat_idx + 8'd1;
				end else begin
					if (s_axi_rready || !s_axi_rvalid) begin
						read_active <= 1'b0;
					end
				end
			end
		end
	end
end

endmodule
