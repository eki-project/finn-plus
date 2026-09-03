module dfx_finn_decouple #(
	parameter int unsigned  DATA_WIDTH   = 8,
	parameter int unsigned  DECOUPLE_CNT = 1
)(
	input  logic                   aclk,
	input  logic                   aresetn,

	input  logic [DATA_WIDTH-1:0]  s_axis_tdata,
	input  logic                   s_axis_tvalid,
	output logic                   s_axis_tready,

	output logic [DATA_WIDTH-1:0]  m_axis_tdata,
	output logic                   m_axis_tvalid,
	input  logic                   m_axis_tready,

	input  logic                   decouple,
	output logic                   decouple_status
);

	localparam int unsigned  CNT_W = (DECOUPLE_CNT > 1) ? $clog2(DECOUPLE_CNT) : 1;


	logic [CNT_W-1:0]  cnt      = '0;
	logic              decoupled = 1'b0;

	logic decouple_now;
	assign decouple_now = decouple && (cnt == '0);

	assign m_axis_tdata  = s_axis_tdata;
	assign m_axis_tvalid = (decoupled || decouple_now) ? 1'b0 : s_axis_tvalid;
	assign s_axis_tready = (decoupled || decouple_now) ? 1'b0 : m_axis_tready;
	assign decouple_status = decoupled || decouple_now;

	always_ff @(posedge aclk) begin
		if (!aresetn) begin
			cnt       <= '0;
			decoupled <= 1'b0;
		end
		else if (decoupled) begin
			if (!decouple) begin
				decoupled <= 1'b0;
				cnt       <= '0;
			end
		end
		else if (decouple_now) begin
			decoupled <= 1'b1;
		end
		else begin
			if (s_axis_tvalid && m_axis_tready) begin
				if (cnt == CNT_W'(DECOUPLE_CNT - 1)) begin

					cnt <= '0;
					if (decouple)
						decoupled <= 1'b1;
				end
				else begin
					cnt <= cnt + 1'b1;
				end
			end
		end
	end

endmodule : dfx_finn_decouple
