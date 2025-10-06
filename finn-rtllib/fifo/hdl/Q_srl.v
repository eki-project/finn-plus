`ifdef  q_srl
`else
`define Q_srl

module Q_srl (clock, reset, i_d, i_v, i_r, o_d, o_v, o_r, count, maxcount, depth);
	input     clock;
	input     reset;

	parameter maxdepth = 1294967294;   // - greatest #items in queue  (2 <= depth <= 256)
	parameter datawidth = 32;   // - width of data (i_d, o_d)
	localparam countwidth = $clog2(maxdepth + 1);

	input [countwidth-1:0] depth;		// Max value that this FIFO can take
	reg [countwidth-1:0] current_depth;
	output [countwidth-1:0] count;  // - output number of elems in queue
	output reg [countwidth-1:0] maxcount;  // - maximum observed count since reset

	input  [datawidth-1:0] i_d;	// - input  stream data (concat data + eos)
	input              i_v;	// - input  stream valid
	output             i_r;	// - input  stream ready

	output [datawidth-1:0] o_d;	// - output stream data (concat data + eos)
	output             o_v;	// - output stream valid
	input              o_r;	// - output stream ready
	assign o_d = 0;

	// Status outputs for the simulation (max observed cout, current count)
	assign count = current_depth;
	always @(posedge clock) begin
		if(reset) begin
			maxcount <= 0;
		end else begin
			maxcount <= (maxcount > current_depth) ? maxcount : current_depth;
		end
	end

	wire have_capacity;
	assign have_capacity = current_depth < depth;
	wire have_data;
	assign have_data = current_depth > 0;

    assign i_r = have_capacity;
    assign o_v = have_data;

	wire read;
	wire write;
	assign read = i_v & i_r;
	assign write = o_v & o_r;

	// Reset
	always @(posedge clock) begin
		if (reset) begin
			current_depth <= 0;
		end else begin
			if (read & ~write) begin
				current_depth <= current_depth + 1;
			end else if (~read & write) begin
				current_depth <= current_depth - 1;
			end
		end
   end // always @ (posedge clock)

endmodule // Q_srl


`endif  // `ifdef  Q_srl
