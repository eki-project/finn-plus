`ifdef  q_srl
`else
`define Q_srl

module Q_srl (clock, reset, i_d, i_v, i_r, o_d, o_v, o_r, count, maxcount, depth);
	input     clock;
	input     reset;

	input [31:0] depth;		// Max value that this FIFO can take
	reg [31:0] current_depth;

   parameter maxdepth = 16;   // - greatest #items in queue  (2 <= depth <= 256)
   parameter width = 16;   // - width of data (i_d, o_d)

   localparam countwidth = $clog2(maxdepth + 1);
   localparam addrwidth = $clog2(maxdepth);

	input  [width-1:0] i_d;	// - input  stream data (concat data + eos)
	input              i_v;	// - input  stream valid
	output             i_r;	// - input  stream ready

	output [width-1:0] o_d;	// - output stream data (concat data + eos)
	output             o_v;	// - output stream valid
	input              o_r;	// - output stream ready

	assign o_d = 0;

	// Dummy values required by FINN
	output [countwidth-1:0] count;  // - output number of elems in queue
	output [countwidth-1:0] maxcount;  // - maximum observed count since reset
	assign count = 1'b1;
	assign maxcount = 1'b1;

	reg [1:0] current_state;
	reg [1:0] next_state;
	parameter idle_state = 2'b00;
	parameter consume_state = 2'b01;
	parameter produce_state = 2'b10;
	parameter produce_consume_state = 2'b11;

	wire have_capacity;
	assign have_capacity = current_depth < depth;
	wire have_data;
	assign have_data = current_depth > 0;

	wire can_read;
	wire can_write;
	assign can_read = have_capacity & i_v;
	assign can_write = have_data & o_r;




	// Comb signals
    assign i_r = (current_state == consume_state) || (current_state == produce_consume_state);
    assign o_v = (current_state == produce_state) || (current_state == produce_consume_state);


	// Reset
	always @(posedge clock) begin
		if (reset) begin
			current_state <= idle_state;
			next_state <= idle_state;
			current_depth <= 0;
		end
		else begin
			// Figure out next state
			current_state <= next_state;
			if (can_read & can_write) begin
				next_state <= produce_consume_state;
			end else if (can_read & ~can_write) begin
				next_state <= consume_state;
			end else if (~can_read & can_write) begin
				next_state <= produce_state;
			end else begin
				next_state <= idle_state;
			end

   			// Adapt occupancy. For Idle and Prod+Cons it stays the same
			if (current_state == produce_state) begin
				current_depth <= current_depth - 1;
			end else if (current_state == consume_state) begin
				current_depth <= current_depth + 1;
			end
		end
   end // always @ (posedge clock)

endmodule // Q_srl


`endif  // `ifdef  Q_srl
