/******************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Testbench for the vector pack converter.
 * @description
 *  Checks a selection of parallelism ratios, with and without padding of the
 *  last beats of a vector, for
 *   - the integrity of the data under randomly stalling source and sink,
 *     whose stall rates change from vector to vector so that the buffer
 *     runs both full and empty, and
 *   - the throughput with a free-running source and sink: unless
 *     RELAX_THROUGHPUT is set or the last beats of a vector are padded, a
 *     vector must take no more cycles than the beats on its narrower
 *     interface. The cycles per vector are reported for all configurations.
 *****************************************************************************/
module vpc_tb;

	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(8) @(posedge clk);
		rst <= 0;
	end

	// Test configurations
	typedef struct {
		int unsigned  w;	// element bit width
		int unsigned  n;	// elements per vector
		int unsigned  pi;	// input elements per beat
		int unsigned  po;	// output elements per beat
		bit  relax;
	} cfg_t;
	localparam int unsigned  CFG_CNT = 31;
	localparam cfg_t  CFGS[CFG_CNT] = '{
		// Ratios that are not integer, without padding
		'{ w: 1, n:   768, pi:  24, po:  32, relax: 0 },
		'{ w: 1, n: 49152, pi: 256, po: 192, relax: 0 },
		'{ w: 4, n:    96, pi:   6, po:   8, relax: 0 },
		'{ w: 4, n:    96, pi:   8, po:   6, relax: 0 },
		'{ w: 8, n:     6, pi:   2, po:   3, relax: 0 },
		'{ w: 8, n:     6, pi:   3, po:   2, relax: 0 },
		'{ w: 3, n:    12, pi:   4, po:   6, relax: 0 },
		'{ w: 1, n:    35, pi:   5, po:   7, relax: 0 },
		'{ w: 1, n:    35, pi:   7, po:   5, relax: 0 },
		'{ w: 2, n:    72, pi:   8, po:   9, relax: 0 },
		'{ w: 2, n:    72, pi:   9, po:   8, relax: 0 },
		'{ w: 1, n:   143, pi:  11, po:  13, relax: 0 },
		'{ w: 1, n:   143, pi:  13, po:  11, relax: 0 },
		// Ratios that are not integer, with padding
		'{ w: 8, n:    10, pi:   3, po:   4, relax: 0 },
		'{ w: 8, n:    10, pi:   4, po:   3, relax: 0 },
		'{ w: 8, n:     7, pi:   3, po:   4, relax: 0 },
		'{ w: 8, n:     7, pi:   4, po:   3, relax: 0 },
		'{ w: 5, n:    11, pi:   2, po:   5, relax: 0 },
		'{ w: 5, n:    11, pi:   5, po:   2, relax: 0 },
		'{ w: 2, n:    13, pi:   6, po:   4, relax: 0 },
		'{ w: 2, n:    50, pi:  12, po:  18, relax: 0 },
		// Integer ratios across multiple beats, with and without padding
		'{ w: 8, n:    12, pi:   1, po:   3, relax: 0 },
		'{ w: 8, n:    12, pi:   3, po:   1, relax: 0 },
		'{ w: 8, n:    10, pi:   1, po:   4, relax: 0 },
		'{ w: 8, n:    10, pi:   4, po:   1, relax: 0 },
		// Whole vector in a single wide beat, identity
		'{ w: 8, n:     3, pi:   3, po:   1, relax: 0 },
		'{ w: 8, n:     3, pi:   1, po:   3, relax: 0 },
		'{ w: 8, n:     4, pi:   2, po:   2, relax: 0 },
		// Relaxed throughput
		'{ w: 1, n:   768, pi:  24, po:  32, relax: 1 },
		'{ w: 4, n:    96, pi:   8, po:   6, relax: 1 },
		'{ w: 8, n:     7, pi:   3, po:   4, relax: 1 }
	};

	bit [CFG_CNT-1:0]  done = '0;
	bit  failed = 0;
	always_comb begin
		if(&done) begin
			if(failed)  $display("Test FAILED.");
			else        $display("Test completed successfully.");
			$finish;
		end
	end

	for(genvar  t = 0; t < CFG_CNT; t++) begin : genTests
		localparam cfg_t  CFG = CFGS[t];
		localparam int unsigned  W  = CFG.w;
		localparam int unsigned  N  = CFG.n;
		localparam int unsigned  PI = CFG.pi;
		localparam int unsigned  PO = CFG.po;
		localparam int unsigned  TRNI = 1 + (N-1)/PI;	// input beats per vector
		localparam int unsigned  TRNO = 1 + (N-1)/PO;	// output beats per vector
		localparam int unsigned  TRN_MAX = (TRNI > TRNO)? TRNI : TRNO;
		localparam bit  PADDED = (TRNI*PI != N) || (TRNO*PO != N);
		localparam bit  FULL_RATE = !CFG.relax && !PADDED;

		localparam int unsigned  VECS_RANDOM  = 64 + 16000/TRN_MAX;	// vectors passed under random stalls
		localparam int unsigned  VECS_WARMUP  = 4;
		localparam int unsigned  VECS_MEASURE = 16;

		typedef logic [W-1:0]  elem_t;
		logic [PI-1:0][W-1:0]  idat;
		logic  ivld;
		uwire  irdy;
		uwire [PO-1:0][W-1:0]  odat;
		uwire  ovld;
		logic  ordy;
		vpc #(.W(W), .N(N), .PI(PI), .PO(PO), .RELAX_THROUGHPUT(CFG.relax)) dut (
			.clk, .rst,
			.idat, .ivld, .irdy,
			.odat, .ovld, .ordy
		);

		// Random stalls are lifted for the throughput measurement
		bit  free_running = 0;

		// Stall rates in eighths, drawn anew for every vector: none, light or heavy
		function automatic int unsigned draw_stall_rate();
			unique case($urandom()%3)
			0: return  0;
			1: return  2;
			2: return  7;
			endcase
		endfunction

		// Stimulus: Feed
		elem_t  Q[$];
		initial begin
			ivld = 0;
			idat = 'x;
			@(posedge clk iff !rst);

			repeat(VECS_RANDOM + VECS_WARMUP + VECS_MEASURE + 2) begin
				automatic int unsigned  stall_rate = draw_stall_rate();
				for(int unsigned  i = 0; i < N; i += PI) begin
					automatic logic [PI-1:0][W-1:0]  dat;
					void'(std::randomize(dat));
					while(!free_running && ($urandom()%8 < stall_rate)) @(posedge clk);

					// Queue ahead of the handshake: an identity converter outputs in the same cycle.
					// Excess lanes of the last beat are not part of the vector.
					for(int unsigned  j = 0; (j < PI) && (i+j < N); j++)  Q.push_back(dat[j]);
					ivld <= 1;
					idat <= dat;
					@(posedge clk iff irdy);

					ivld <= 0;
					idat <= 'x;
				end
			end
		end

		// Output Sink and Checker
		int unsigned  OVecs = 0;	// completed output vectors
		initial begin
			ordy = 0;
			@(posedge clk iff !rst);

			forever begin
				automatic int unsigned  stall_rate = draw_stall_rate();
				for(int unsigned  i = 0; i < N; i += PO) begin
					while(!free_running && ($urandom()%8 < stall_rate)) @(posedge clk);

					ordy <= 1;
					@(posedge clk iff ovld);
					for(int unsigned  j = 0; j < PO; j++) begin
						automatic elem_t  exp = 0;	// excess lanes are padded with zeros
						if(i+j < N) begin
							assert(Q.size) else begin
								$error("[%0d] Spurious output.", t);
								$stop;
							end
							exp = Q.pop_front();
						end
						assert(odat[j] === exp) else begin
							$error("[%0d] Output mismatch on lane %0d: 0x%0x instead of 0x%0x", t, j, odat[j], exp);
							$stop;
						end
					end
					ordy <= 0;
				end
				OVecs <= OVecs + 1;
			end
		end

		// Throughput with free-running source and sink
		initial begin
			automatic time  t0;
			automatic real  cycles;

			@(posedge clk iff OVecs == VECS_RANDOM);
			free_running = 1;
			@(posedge clk iff OVecs == VECS_RANDOM + VECS_WARMUP);
			t0 = $time;
			@(posedge clk iff OVecs == VECS_RANDOM + VECS_WARMUP + VECS_MEASURE);
			cycles = real'($time - t0) / 10ns / VECS_MEASURE;

			$display(
				"[%0d] W=%0d N=%0d PI=%0d PO=%0d: %0.2f cycles per vector for %0d input and %0d output beats (%0s)",
				t, W, N, PI, PO, cycles, TRNI, TRNO, CFG.relax? "relaxed" : PADDED? "padded" : "full rate required"
			);
			assert(!FULL_RATE || (cycles <= TRN_MAX)) else begin
				$error("[%0d] Throughput below full rate.", t);
				failed = 1;
			end
			done[t] = 1;
		end

	end : genTests

endmodule : vpc_tb
