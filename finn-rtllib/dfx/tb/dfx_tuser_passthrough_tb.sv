/******************************************************************************
 * Copyright (C) 2026, Paderborn University
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Self-checking testbench of dfx_tuser_passthrough around a static segment
 * model with latency LAT, so that several frames are inside it at a time.
 * The source changes tUSER every N frames ((frame / N) % 4), as the
 * instrumentation does with mux_interval N.
 *
 * Checks: every output beat comes from the expected frame and beat and carries
 * that frame's tUSER, every frame leaves after the source has stopped, and with
 * enough tracker entries (FULL_RATE) a frame takes exactly OLEN cycles.
 * With TUSER_TRACK_DEPTH smaller than the number of frames in the segment
 * the passthrough holds frames back: slower, but still exact.
 *
 * Run: xvlog -sv ../tuser_track/tuser_track.sv ../dfx_tuser_passthrough/dfx_tuser_passthrough.sv
 *      dfx_tuser_passthrough_tb.sv && xelab dfx_tuser_passthrough_tb -s sim
 *      [-generic_top N=1 ...] && xsim sim -R
 * Prints "PASS" or stops with $fatal.
 *****************************************************************************/
`timescale 1ns/1ps
module dfx_tuser_passthrough_tb;
	parameter int N         = 1;    // frames per tUSER value
	parameter int NFRAMES   = 24;
	parameter int OLEN      = 16;   // beats per frame (input and output)
	parameter int LAT       = 100;  // segment latency: about LAT/OLEN frames inside
	parameter int RANDREADY = 0;
	parameter int RANDVALID = 0;
	parameter int TUSER_TRACK_DEPTH = 256;
	parameter int FULL_RATE = 1;    // expect one frame per OLEN cycles (no random stalls)

	localparam int  W = 32;
	localparam int  DEPTH = LAT + 4*OLEN;

	logic  clk = 0;
	always #2.5 clk = !clk;
	logic  aresetn = 0;
	longint  cyc = 0;
	always @(posedge clk)  cyc <= cyc + 1;

	//-----------------------------------------------------------------------
	// Source
	int  src_frame = 0, src_beat = 0;
	logic  src_en = 0, src_gap;
	always @(posedge clk)  src_gap <= RANDVALID? $urandom_range(0, 3) == 0 : 0;
	logic [W-1:0]  s_tdata;
	logic  s_tvalid, s_tready, s_tlast;
	logic [1:0]  s_tuser;
	assign	s_tvalid = src_en && !src_gap && (src_frame < NFRAMES);
	assign	s_tlast  = src_beat == OLEN-1;
	assign	s_tuser  = (src_frame / N) % 4;
	assign	s_tdata  = { 8'(src_frame), 8'(src_beat), 16'h0 };
	always @(posedge clk) begin
		if(s_tvalid && s_tready) begin
			if(s_tlast) begin
				src_beat  <= 0;
				src_frame <= src_frame + 1;
			end
			else  src_beat <= src_beat + 1;
		end
	end

	//-----------------------------------------------------------------------
	// DUT
	logic [W-1:0]  m_tdata, rpm_tdata, rps_tdata;
	logic  m_tvalid, m_tready, m_tlast;
	logic [1:0]  m_tuser;
	logic  rpm_tvalid, rpm_tready, rpm_tlast;
	logic  rps_tvalid, rps_tready;
	dfx_tuser_passthrough #(
		.IN_DATA_WIDTH(W), .OUT_DATA_WIDTH(W), .TUSER_WIDTH(2),
		.TUSER_TRACK_DEPTH(TUSER_TRACK_DEPTH), .NUM_OUTPUT_BEATS(OLEN)
	) dut (
		.aclk(clk), .aresetn,
		.s_axis_tdata(s_tdata), .s_axis_tvalid(s_tvalid), .s_axis_tready(s_tready),
		.s_axis_tlast(s_tlast), .s_axis_tuser(s_tuser),
		.m_axis_tdata(m_tdata), .m_axis_tvalid(m_tvalid), .m_axis_tready(m_tready),
		.m_axis_tlast(m_tlast), .m_axis_tuser(m_tuser),
		.rp_m_axis_tdata(rpm_tdata), .rp_m_axis_tvalid(rpm_tvalid),
		.rp_m_axis_tready(rpm_tready), .rp_m_axis_tlast(rpm_tlast),
		.rp_s_axis_tdata(rps_tdata), .rp_s_axis_tvalid(rps_tvalid),
		.rp_s_axis_tready(rps_tready), .rp_s_axis_tlast(1'b0)
	);

	//-----------------------------------------------------------------------
	// Segment model: fixed latency, bounded buffer
	logic [W-1:0]  buf_d[DEPTH];
	longint  buf_t[DEPTH];
	int  head = 0, tail = 0, cnt = 0;
	assign	rpm_tready = cnt < DEPTH;
	assign	rps_tvalid = (cnt > 0) && (buf_t[head] <= cyc);
	assign	rps_tdata  = buf_d[head];
	always @(posedge clk) begin
		if(!aresetn) begin
			head <= 0;
			tail <= 0;
			cnt  <= 0;
		end
		else begin
			automatic logic  pop  = rps_tvalid && rps_tready;
			automatic logic  push = rpm_tvalid && rpm_tready;
			if(pop)  head <= (head + 1) % DEPTH;
			if(push) begin
				buf_d[tail] <= rpm_tdata;
				buf_t[tail] <= cyc + LAT;
				tail <= (tail + 1) % DEPTH;
			end
			cnt <= cnt + int'(push) - int'(pop);
		end
	end

	//-----------------------------------------------------------------------
	// Sink & checker
	logic  rr;
	always @(posedge clk)  rr <= RANDREADY? $urandom_range(0, 1) : 1;
	assign	m_tready = rr;
	int  out_beats = 0, out_frames = 0, errors = 0, tuser_errors = 0, bad_intervals = 0;
	longint  last_frame_cyc = -1;
	always @(posedge clk) begin
		if(m_tvalid && m_tready) begin
			automatic int  f = m_tdata[31:24];
			automatic int  b = m_tdata[23:16];
			automatic int  exp_f = out_beats / OLEN;
			automatic int  exp_b = out_beats % OLEN;
			automatic int  exp_u = (exp_f / N) % 4;
			if((f != exp_f) || (b != exp_b) || (m_tlast != (exp_b == OLEN-1))) begin
				if(errors < 8)  $display("[%0t] output beat %0d: frame %0d beat %0d tlast %0b, expected frame %0d beat %0d",
				                         $time, out_beats, f, b, m_tlast, exp_f, exp_b);
				errors <= errors + 1;
			end
			if(m_tuser != 2'(exp_u)) begin
				if(tuser_errors < 8)  $display("[%0t] output beat %0d (frame %0d): tUSER %0d, expected %0d",
				                               $time, out_beats, exp_f, m_tuser, exp_u);
				tuser_errors <= tuser_errors + 1;
			end
			out_beats <= out_beats + 1;
			if(exp_b == OLEN-1) begin
				out_frames <= out_frames + 1;
				if((last_frame_cyc >= 0) && (cyc - last_frame_cyc != OLEN))  bad_intervals <= bad_intervals + 1;
				last_frame_cyc <= cyc;
			end
		end
	end

	initial begin
		repeat(10) @(posedge clk);
		aresetn = 1;
		repeat(20) @(posedge clk);
		src_en = 1;
		fork
			wait(out_frames == NFRAMES);
			repeat(NFRAMES * (2*OLEN + LAT + 50) + 1000) @(posedge clk);
		join_any
		repeat(LAT + 100) @(posedge clk);
		$display("N=%0d frames in %0d out %0d, beat errors %0d, tUSER errors %0d, frame intervals != %0d: %0d",
		         N, src_frame, out_frames, errors, tuser_errors, OLEN, bad_intervals);
		if(src_frame != NFRAMES)       $fatal(1, "FAIL: source stalled after %0d frames", src_frame);
		if(out_frames != NFRAMES)      $fatal(1, "FAIL: %0d of %0d frames left the segment", out_frames, NFRAMES);
		if(out_beats != NFRAMES*OLEN)  $fatal(1, "FAIL: %0d output beats, expected %0d", out_beats, NFRAMES*OLEN);
		if(errors != 0)                $fatal(1, "FAIL: %0d wrong output beats", errors);
		if(tuser_errors != 0)          $fatal(1, "FAIL: %0d output beats with a wrong tUSER", tuser_errors);
		if(FULL_RATE && !RANDREADY && !RANDVALID && (bad_intervals != 0))
			$fatal(1, "FAIL: %0d frames took longer than %0d cycles", bad_intervals, OLEN);
		$display("PASS");
		$finish;
	end
endmodule : dfx_tuser_passthrough_tb
