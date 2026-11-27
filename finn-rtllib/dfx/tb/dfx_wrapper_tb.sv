/******************************************************************************
 * Copyright (C) 2026, Paderborn University
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Self-checking testbench of dfx_wrapper with models of the reconfigurable
 * body (BDC), the AMD dfx_decoupler on its output and the DFX Controller.
 *
 * The body model has the properties of a stitched FINN body that matter here:
 *  - its nodes follow accel_reset_n RESET_LATENCY cycles late (reset_pipe) and
 *    a freshly configured body starts with that pipeline at its INIT value 0,
 *  - its input reports tready while the nodes are still in reset, and beats
 *    accepted then are lost (as the input FIFO does),
 *  - one output beat per input beat after LAT cycles, tagged with the frame,
 *    beat and the ID of the loaded RM.
 * The source sends NFRAMES identical-length frames and selects the RM
 * (frame / N) % 2 via tUSER, as the instrumentation does with mux_interval N.
 *
 * Checks: every output beat comes from the expected frame, beat and RM and
 * carries that frame's tUSER, every
 * frame leaves the body after the source has stopped, and without switching
 * (N >= NFRAMES) and without random stalls a frame takes exactly OLEN cycles.
 *
 * Run: xvlog -sv ../tuser_track/tuser_track.sv ../dfx_wrapper/dfx_wrapper.sv dfx_wrapper_tb.sv &&
 *      xelab dfx_wrapper_tb -s sim [-generic_top N=1 ...] && xsim sim -R
 * Prints "PASS" or stops with $fatal.
 *****************************************************************************/
`timescale 1ns/1ps
module dfx_wrapper_tb;
	parameter int N             = 2;    // frames per tUSER value
	parameter int NFRAMES       = 12;
	parameter int OLEN          = 16;   // beats per frame (input and output)
	parameter int LAT           = 40;   // body latency (> OLEN: several frames in flight)
	parameter int RESET_LATENCY = 3;    // body reset pipeline, also set on the wrapper
	parameter int RANDREADY     = 0;    // random sink tready
	parameter int RANDVALID     = 0;    // random gaps in the source
	parameter int GARBAGE       = 1;    // body drives random tvalid while reconfigured
	parameter int RECONF_CYCLES = 200;  // DFX controller: decouple time per reconfiguration
	parameter int TUSER_TRACK_DEPTH = 256;

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
	assign	s_tuser  = (src_frame / N) % 2;
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
	logic [1:0]  trigger;
	logic  decouple = 0;
	logic  accel_reset_n;
	dfx_wrapper #(
		.IN_DATA_WIDTH(W), .OUT_DATA_WIDTH(W), .TUSER_WIDTH(2), .NUM_RM(2),
		.RESET_LATENCY(RESET_LATENCY), .TUSER_TRACK_DEPTH(TUSER_TRACK_DEPTH), .NUM_OUTPUT_BEATS(OLEN)
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
		.controller_trigger(trigger), .controller_decouple(decouple),
		.accel_reset_n
	);

	//-----------------------------------------------------------------------
	// Body model
	int  rm_id = 0;           // loaded RM
	logic  reconfiguring = 0; // region being configured (outputs undefined)
	logic  node_rstn;
	if(RESET_LATENCY == 0) begin : genDirect
		assign	node_rstn = accel_reset_n && !reconfiguring;
	end
	else if(RESET_LATENCY == 1) begin : genOne
		logic  rst_pipe = 0;
		always @(posedge clk)  rst_pipe <= reconfiguring? 0 : accel_reset_n;
		assign	node_rstn = rst_pipe;
	end
	else begin : genPiped
		// INIT value 0 after a reconfiguration, like reset_pipe in a fresh body
		logic [RESET_LATENCY-1:0]  rst_pipe = '0;
		always @(posedge clk)  rst_pipe <= reconfiguring? '0 : { rst_pipe[RESET_LATENCY-2:0], accel_reset_n };
		assign	node_rstn = rst_pipe[RESET_LATENCY-1];
	end

	logic [W-1:0]  buf_d[DEPTH];
	longint  buf_t[DEPTH];
	int  head = 0, tail = 0, cnt = 0;
	logic  garbage_v;
	always @(posedge clk)  garbage_v <= $urandom_range(0, 1);
	logic [W-1:0]  bo_tdata;
	logic  bo_tvalid, bo_tready;
	// input ready even while the nodes are in reset (as the FIFO in front of them)
	assign	rpm_tready = !reconfiguring && (cnt < DEPTH);
	assign	bo_tvalid  = reconfiguring? (GARBAGE? garbage_v : 0) :
	                     node_rstn && (cnt > 0) && (buf_t[head] <= cyc);
	assign	bo_tdata   = { buf_d[head][31:8], 8'(rm_id) };
	always @(posedge clk) begin
		if(!node_rstn || reconfiguring) begin
			head <= 0;
			tail <= 0;
			cnt  <= 0;
		end
		else begin
			automatic logic  pop  = bo_tvalid && bo_tready;
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

	// AMD dfx_decoupler on the body output
	assign	rps_tdata  = bo_tdata;
	assign	rps_tvalid = bo_tvalid && !decouple;
	assign	bo_tready  = rps_tready && !decouple;

	//-----------------------------------------------------------------------
	// DFX Controller model
	int  pend = -1, ctl_cnt = 0, reconfigs = 0;
	always @(posedge clk) begin
		if(trigger != 0) begin
			if(pend >= 0)  $fatal(1, "FAIL: trigger while a reconfiguration is pending");
			if($countones(trigger) != 1)  $fatal(1, "FAIL: trigger %b not one-hot", trigger);
			pend    <= trigger[1];
			ctl_cnt <= 0;
		end
		else if(pend >= 0) begin
			ctl_cnt <= ctl_cnt + 1;
			if(ctl_cnt == 5) begin
				decouple <= 1;
				reconfiguring <= 1;
			end
			if(ctl_cnt == 5 + RECONF_CYCLES) begin
				reconfiguring <= 0;
				rm_id <= pend;
				reconfigs <= reconfigs + 1;
			end
			if(ctl_cnt == 10 + RECONF_CYCLES) begin
				decouple <= 0;
				pend <= -1;
			end
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
			automatic int  r = m_tdata[7:0];
			automatic int  exp_f = out_beats / OLEN;
			automatic int  exp_b = out_beats % OLEN;
			automatic int  exp_r = (exp_f / N) % 2;
			if((f != exp_f) || (b != exp_b) || (r != exp_r) || (m_tlast != (exp_b == OLEN-1))) begin
				if(errors < 8)  $display("[%0t] output beat %0d: frame %0d beat %0d RM %0d tlast %0b, expected frame %0d beat %0d RM %0d",
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
		repeat(50) @(posedge clk);  // the host starts long after the system reset
		src_en = 1;
		fork
			wait(out_frames == NFRAMES);
			repeat(NFRAMES * (2*OLEN + LAT + RECONF_CYCLES + 100) + 1000) @(posedge clk);
		join_any
		repeat(LAT + 100) @(posedge clk);
		$display("N=%0d frames in %0d out %0d, reconfigurations %0d, beat errors %0d, frame intervals != %0d: %0d, wrapper state %0d, frames in flight %0d",
		         N, src_frame, out_frames, reconfigs, errors, OLEN, bad_intervals, dut.state, dut.frames_in_flight);
		if(src_frame != NFRAMES)       $fatal(1, "FAIL: source stalled after %0d frames", src_frame);
		if(out_frames != NFRAMES)      $fatal(1, "FAIL: %0d of %0d frames left the body", out_frames, NFRAMES);
		if(out_beats != NFRAMES*OLEN)  $fatal(1, "FAIL: %0d output beats, expected %0d", out_beats, NFRAMES*OLEN);
		if(errors != 0)                $fatal(1, "FAIL: %0d wrong output beats", errors);
		if(tuser_errors != 0)          $fatal(1, "FAIL: %0d output beats with a wrong tUSER", tuser_errors);
		if(reconfigs != (NFRAMES-1) / N)  $fatal(1, "FAIL: %0d reconfigurations, expected %0d", reconfigs, (NFRAMES-1) / N);
		if((N >= NFRAMES) && !RANDREADY && !RANDVALID && (bad_intervals != 0))
			$fatal(1, "FAIL: %0d frames took longer than %0d cycles", bad_intervals, OLEN);
		$display("PASS");
		$finish;
	end
endmodule : dfx_wrapper_tb
