/******************************************************************************
 * Copyright (C) 2026, Paderborn University
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Self-checking testbench of sw_wrapper with a model of a selectable-weights
 * NodeContainer: an input FIFO feeds a compute stage with latency LAT that
 * takes one set-selection token per frame (when the frame's first beat enters
 * it) from a two-entry token buffer (as the stream_tap holds A and B), and
 * tags every output beat with the frame, beat and selected set.
 * The source selects the set (frame / N) % NUM_SETS via tUSER.
 *
 * Checks: every output beat comes from the expected frame, beat and set and
 * carries that frame's tUSER,
 * every frame leaves after the source has stopped, and without random stalls a
 * frame takes exactly OLEN cycles.
 *
 * Run: xvlog -sv ../tuser_track/tuser_track.sv ../sw_wrapper/sw_wrapper.sv sw_wrapper_tb.sv &&
 *      xelab sw_wrapper_tb -s sim [-generic_top N=1 ...] && xsim sim -R
 * Prints "PASS" or stops with $fatal.
 *****************************************************************************/
`timescale 1ns/1ps
module sw_wrapper_tb;
	parameter int N         = 2;   // frames per tUSER value
	parameter int NUM_SETS  = 2;
	parameter int NFRAMES   = 12;
	parameter int OLEN      = 16;  // beats per frame (input and output)
	parameter int LAT       = 40;
	parameter int RANDREADY = 0;
	parameter int RANDVALID = 0;
	parameter int SLOWTAP   = 0;   // token buffer accepts only every few cycles
	parameter int TUSER_TRACK_DEPTH = 256;
	parameter int FULL_RATE = 1;   // expect one frame per OLEN cycles (enough tracker entries)

	localparam int  W = 32;
	localparam int  DEPTH = LAT + 4*OLEN;
	localparam int  SETSEL_WIDTH = 32;

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
	assign	s_tuser  = (src_frame / N) % NUM_SETS;
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
	logic [SETSEL_WIDTH-1:0]  setsel_tdata;
	logic  setsel_tvalid, setsel_tready;
	sw_wrapper #(
		.DATA_IN_WIDTH(W), .DATA_OUT_WIDTH(W), .TUSER_WIDTH(2), .NUM_SETS(NUM_SETS),
		.TUSER_TRACK_DEPTH(TUSER_TRACK_DEPTH), .NUM_OUTPUT_BEATS(OLEN), .SETSEL_WIDTH(SETSEL_WIDTH)
	) dut (
		.aclk(clk), .aresetn,
		.s_axis_tdata(s_tdata), .s_axis_tvalid(s_tvalid), .s_axis_tready(s_tready),
		.s_axis_tlast(s_tlast), .s_axis_tuser(s_tuser),
		.m_axis_tdata(m_tdata), .m_axis_tvalid(m_tvalid), .m_axis_tready(m_tready),
		.m_axis_tlast(m_tlast), .m_axis_tuser(m_tuser),
		.rp_m_axis_tdata(rpm_tdata), .rp_m_axis_tvalid(rpm_tvalid),
		.rp_m_axis_tready(rpm_tready), .rp_m_axis_tlast(rpm_tlast),
		.rp_s_axis_tdata(rps_tdata), .rp_s_axis_tvalid(rps_tvalid),
		.rp_s_axis_tready(rps_tready), .rp_s_axis_tlast(1'b0),
		.m_axis_setsel_tdata(setsel_tdata), .m_axis_setsel_tvalid(setsel_tvalid),
		.m_axis_setsel_tready(setsel_tready)
	);

	//-----------------------------------------------------------------------
	// NodeContainer model
	// token buffer (2 entries)
	int  tok[2];
	int  tok_cnt = 0;
	logic  tap_slow;
	always @(posedge clk)  tap_slow <= SLOWTAP? $urandom_range(0, 2) != 0 : 0;
	assign	setsel_tready = (tok_cnt < 2) && !tap_slow;
	// input FIFO
	logic [W-1:0]  in_d[DEPTH];
	int  in_h = 0, in_t = 0, in_cnt = 0;
	assign	rpm_tready = in_cnt < DEPTH;
	// output pipeline
	logic [W-1:0]  out_d[DEPTH];
	longint  out_tm[DEPTH];
	int  out_h = 0, out_t = 0, out_cnt = 0;
	// compute stage: takes one beat per cycle, one token at each frame's first beat
	int  cur_set = -1, cbeat = 0;
	logic  tok_pop, c_take;
	assign	tok_pop = c_take && (cbeat == 0);
	assign	c_take  = (in_cnt > 0) && ((cbeat != 0) || (tok_cnt > 0)) && (out_cnt < DEPTH);
	assign	rps_tvalid = (out_cnt > 0) && (out_tm[out_h] <= cyc);
	assign	rps_tdata  = out_d[out_h];
	always @(posedge clk) begin
		if(!aresetn) begin
			tok_cnt <= 0;
			in_h <= 0; in_t <= 0; in_cnt <= 0;
			out_h <= 0; out_t <= 0; out_cnt <= 0;
			cbeat <= 0;
		end
		else begin
			automatic logic  tok_push = setsel_tvalid && setsel_tready;
			automatic logic  in_push  = rpm_tvalid && rpm_tready;
			automatic logic  out_pop  = rps_tvalid && rps_tready;
			automatic int  set = tok_pop? tok[0] : cur_set;
			// tokens
			if(tok_pop) begin
				tok[0] <= (tok_cnt == 2)? tok[1] : (tok_push? int'(setsel_tdata) : tok[0]);
				if((tok_cnt == 2) && tok_push)  tok[1] <= int'(setsel_tdata);
			end
			else if(tok_push)  tok[tok_cnt] <= int'(setsel_tdata);
			tok_cnt <= tok_cnt + int'(tok_push) - int'(tok_pop);
			// input FIFO
			if(in_push) begin
				in_d[in_t] <= rpm_tdata;
				in_t <= (in_t + 1) % DEPTH;
			end
			if(c_take)  in_h <= (in_h + 1) % DEPTH;
			in_cnt <= in_cnt + int'(in_push) - int'(c_take);
			// compute
			if(c_take) begin
				if(tok_pop)  cur_set <= tok[0];
				cbeat <= (cbeat == OLEN-1)? 0 : cbeat + 1;
				out_d[out_t] <= { in_d[in_h][31:8], 8'(set) };
				out_tm[out_t] <= cyc + LAT;
				out_t <= (out_t + 1) % DEPTH;
			end
			if(out_pop)  out_h <= (out_h + 1) % DEPTH;
			out_cnt <= out_cnt + int'(c_take) - int'(out_pop);
		end
	end

	//-----------------------------------------------------------------------
	// Sink & checker
	logic  rr;
	always @(posedge clk)  rr <= RANDREADY? $urandom_range(0, 1) : 1;
	assign	m_tready = rr;
	int  out_beats = 0, out_frames = 0, errors = 0, tuser_errors = 0, bad_intervals = 0, tokens = 0;
	longint  last_frame_cyc = -1;
	always @(posedge clk)  if(setsel_tvalid && setsel_tready)  tokens <= tokens + 1;
	always @(posedge clk) begin
		if(m_tvalid && m_tready) begin
			automatic int  f = m_tdata[31:24];
			automatic int  b = m_tdata[23:16];
			automatic int  r = m_tdata[7:0];
			automatic int  exp_f = out_beats / OLEN;
			automatic int  exp_b = out_beats % OLEN;
			automatic int  exp_r = (exp_f / N) % NUM_SETS;
			if((f != exp_f) || (b != exp_b) || (r != exp_r) || (m_tlast != (exp_b == OLEN-1))) begin
				if(errors < 8)  $display("[%0t] output beat %0d: frame %0d beat %0d set %0d tlast %0b, expected frame %0d beat %0d set %0d",
				                         $time, out_beats, f, b, r, m_tlast, exp_f, exp_b, exp_r);
				errors <= errors + 1;
			end
			if(m_tuser != 2'(exp_r)) begin
				if(tuser_errors < 8)  $display("[%0t] output beat %0d (frame %0d): tUSER %0d, expected %0d",
				                               $time, out_beats, exp_f, m_tuser, exp_r);
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
			repeat(NFRAMES * (4*OLEN + LAT + 100) + 1000) @(posedge clk);
		join_any
		repeat(LAT + 100) @(posedge clk);
		$display("N=%0d frames in %0d out %0d, tokens %0d, beat errors %0d, frame intervals != %0d: %0d",
		         N, src_frame, out_frames, tokens, errors, OLEN, bad_intervals);
		if(src_frame != NFRAMES)       $fatal(1, "FAIL: source stalled after %0d frames", src_frame);
		if(out_frames != NFRAMES)      $fatal(1, "FAIL: %0d of %0d frames left the container", out_frames, NFRAMES);
		if(out_beats != NFRAMES*OLEN)  $fatal(1, "FAIL: %0d output beats, expected %0d", out_beats, NFRAMES*OLEN);
		if(errors != 0)                $fatal(1, "FAIL: %0d wrong output beats", errors);
		if(tuser_errors != 0)          $fatal(1, "FAIL: %0d output beats with a wrong tUSER", tuser_errors);
		if(tokens != NFRAMES)          $fatal(1, "FAIL: %0d set-selection tokens for %0d frames", tokens, NFRAMES);
		if(FULL_RATE && !RANDREADY && !RANDVALID && !SLOWTAP && (bad_intervals != 0))
			$fatal(1, "FAIL: %0d frames took longer than %0d cycles", bad_intervals, OLEN);
		$display("PASS");
		$finish;
	end
endmodule : sw_wrapper_tb
