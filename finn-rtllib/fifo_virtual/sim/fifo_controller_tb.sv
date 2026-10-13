/**
 * Copyright (c) 2025, Paderborn University
 *
 * @author	Felix P. Jentzsch <felix.jentzsch@upb.de>
 *
 * @brief	Testbench for the gauging ring with the fifo_controller in the loop.
 *
 * @description
 *	Drives the same abstracted dataflow pipeline as fifo_gauge_tb, but issues all ring
 *	operations through the AXI-lite interface of the fifo_controller in the way the
 *	PYNQ driver (finn_plus_driver.overlays.live_fifo) does. It checks:
 *	 - READ_FILL and READ_STALL return their value in the low bytes of the response word
 *	   for every FIFO ID (the READ_STALL packet alignment fix),
 *	 - the peak occupancy (READ_FILL) of a bounded run with the capacities set to the
 *	   peaks of the paced run reproduces exactly these peaks and shows no producer stall,
 *	 - a deadlocked bounded run sets the run-sticky producer-blocked flag of every
 *	   gauge while the per-frame flags of the last gauge keep their previous value.
 */
module fifo_controller_tb;

	import  fifo_gauge_pkg::*;

	localparam int unsigned  LAYERS = 3;
	localparam int unsigned  FM_SIZE[0:LAYERS] = '{ 30, 42, 37, 16 };
	localparam int unsigned  SLACK[LAYERS]     = '{    7,  2, 13   };

	//=======================================================================
	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(12) @(posedge clk);
		rst <= 0;
	end

	//=======================================================================
	// Controller with AXI-lite Interface
	uwire  awready;
	logic  awvalid = 0;
	logic [31:0]  awaddr = 'x;
	uwire  wready;
	logic  wvalid = 0;
	logic [31:0]  wdata = 'x;
	logic  bready = 0;
	uwire  bvalid;
	uwire [1:0]  bresp;

	uwire  arready;
	logic  arvalid = 0;
	logic [31:0]  araddr = 'x;
	logic  rready = 0;
	uwire  rvalid;
	uwire [1:0]  rresp;
	uwire [31:0]  rdata;

	uwire [7:0]  ring_in;   // ring: controller -> first gauge
	uwire [7:0]  ring_out;  // ring: last gauge -> controller

	fifo_controller #(
		.ADDR_WIDTH(32), .DATA_WIDTH(32), .IP_ADDR_WIDTH(30), .IP_DATA_WIDTH(32)
	) ctrl (
		.aclk(clk), .aresetn(!rst),
		.awready, .awvalid, .awprot('0), .awaddr,
		.wready,  .wvalid,  .wstrb('1),  .wdata,
		.bready,  .bvalid,  .bresp,
		.arready, .arvalid, .arprot('0), .araddr,
		.rready,  .rvalid,  .rresp,      .rdata,
		.icfg(ring_out), .ocfg(ring_in)
	);

	//=======================================================================
	// Pipeline: -> FIFO -> (Layer -> FIFO)* ->
	logic [7:0]  idat;
	logic  ivld;
	uwire  irdy;
	uwire  ovld;
	logic  ordy;

	if(1) begin : blkPipeline
		uwire [7:0]  i_dat[0:LAYERS];
		uwire  i_vld[0:LAYERS];
		uwire  i_rdy[0:LAYERS];
		uwire  o_vld[0:LAYERS];
		uwire  o_rdy[0:LAYERS];
		uwire [7:0]  cfg[0:LAYERS+1];

		assign	i_dat[0] = idat;
		assign	i_vld[0] = ivld;
		assign	irdy = i_rdy[0];
		assign	cfg[0] = ring_in;
		for(genvar  l = 0; l <= LAYERS; l++) begin : genDUTs
			fifo_gauge #(.ID(l), .DATA_WIDTH(8), .FM_SIZE(FM_SIZE[l])) dut (
				.clk, .rst,
				.icfg(cfg[l]), .ocfg(cfg[l+1]),
				.idat(i_dat[l]), .ivld(i_vld[l]), .irdy(i_rdy[l]),
				.odat(),         .ovld(o_vld[l]), .ordy(o_rdy[l])
			);
		end : genDUTs
		assign	ovld = o_vld[LAYERS];
		assign	o_rdy[LAYERS] = ordy;
		assign	ring_out = cfg[LAYERS+1];

		for(genvar  l = 0; l < LAYERS; l++) begin : genLayer
			localparam int unsigned  IFM_SIZE = FM_SIZE[l];
			localparam int unsigned  OFM_SIZE = FM_SIZE[l+1];
			localparam int unsigned  MIN_PERIOD = SLACK[l] + (IFM_SIZE > OFM_SIZE? IFM_SIZE : OFM_SIZE);

			uwire  src_vld = o_vld[l];
			logic  src_rdy;
			assign	o_rdy[l] = src_rdy;
			uwire  dst_rdy = i_rdy[l+1];
			logic  dst_vld;
			logic [7:0]  dst_dat;
			assign	i_vld[l+1] = dst_vld;
			assign	i_dat[l+1] = dst_dat;
			// The layer restarts its schedule on every reset, like the accelerator does
			// when the driver resets it before a run.
			initial forever begin
				src_rdy <= 0;
				dst_vld <= 0;
				dst_dat <= 0;
				@(posedge clk iff !rst);
				fork
					forever begin
						dst_dat <= 0;
						for(int unsigned  i = 0; i < MIN_PERIOD; i++) begin
							src_rdy <= ((i+1)*IFM_SIZE)/MIN_PERIOD > (i*IFM_SIZE)/MIN_PERIOD;
							dst_vld <= ((i+1)*OFM_SIZE)/MIN_PERIOD > (i*OFM_SIZE)/MIN_PERIOD;
							forever @(posedge clk) begin
								automatic logic  stxn = src_vld && src_rdy;
								automatic logic  dtxn = dst_vld && dst_rdy;
								automatic logic  srdy = src_rdy && !src_vld;
								automatic logic  dvld = dst_vld && !dst_rdy;
								src_rdy <= srdy;
								dst_vld <= dvld;
								if(dtxn)  dst_dat <= dst_dat + 1;
								if(!srdy && !dvld)  break;
							end
						end
					end
				join_none
				@(posedge clk iff rst);
				disable fork;
			end
		end : genLayer
	end : blkPipeline

	//=======================================================================
	// AXI-lite Driver (mirrors finn_plus_driver.overlays.live_fifo)
	task automatic axi_write(input logic [31:0] addr, input logic [31:0] data);
		awaddr <= addr; awvalid <= 1;
		wdata  <= data; wvalid  <= 1;
		bready <= 1;
		fork
			begin do @(posedge clk); while(!awready); awvalid <= 0; awaddr <= 'x; end
			begin do @(posedge clk); while(!wready);  wvalid  <= 0; wdata  <= 'x; end
		join
		do @(posedge clk); while(!bvalid);
		bready <= 0;
	endtask : axi_write

	task automatic axi_read(input logic [31:0] addr, output logic [31:0] data);
		araddr <= addr; arvalid <= 1;
		rready <= 1;
		do @(posedge clk); while(!arready);
		arvalid <= 0; araddr <= 'x;
		do @(posedge clk); while(!rvalid);
		data = rdata;
		rready <= 0;
	endtask : axi_read

	function automatic logic [31:0] ctrl_addr(input ring_opcode_e op, input int unsigned fifo_id);
		return ((fifo_id << 8) | op) << 2;
	endfunction : ctrl_addr

	// ctrl_read(op, id): issue an operation and return the response word
	task automatic ctrl_read(input ring_opcode_e op, input int unsigned fifo_id, output logic [31:0] data);
		axi_read(ctrl_addr(op, fifo_id), data);
	endtask : ctrl_read

	// ctrl_set_depth: asynchronous WRITE_FILL followed by a NOP read to wait for completion.
	// The controller only accepts a request in its IDLE state, so the NOP read must not
	// arrive while the WRITE_FILL packet is still being issued. On hardware, the latency
	// of the AXI-lite access from the processing system guarantees this; here, it is
	// modelled by an explicit delay.
	task automatic ctrl_set_depth(input int unsigned fifo_id, input logic [31:0] depth);
		automatic logic [31:0]  ret;
		axi_write(ctrl_addr(WRITE_FILL, fifo_id), depth);
		repeat(32) @(posedge clk);
		ctrl_read(NOP, 0, ret);
		assert(ret == NOP) else begin
			$error("NOP read after WRITE_FILL returned 0x%08x.", ret);
			$stop;
		end
	endtask : ctrl_set_depth

	task automatic ctrl_run(input ring_opcode_e op);
		automatic logic [31:0]  ret;
		ctrl_read(op, 0, ret);
		assert(ret == op) else begin
			$error("RUN command 0x%02x acknowledged with 0x%08x.", op, ret);
			$stop;
		end
	endtask : ctrl_run

	// reset_accelerator: the driver's GPIO reset resets the whole design through
	// proc_sys_reset (gauges, layers, instrumentation and controller alike).
	task automatic reset_accelerator();
		ivld <= 0;
		ordy <= 0;
		rst <= 1;
		repeat(8) @(posedge clk);
		rst <= 0;
		repeat(4) @(posedge clk);
	endtask : reset_accelerator

	//=======================================================================
	// Traffic
	int unsigned  Period = 0;

	// Feed `frames` feature maps, each within `Period` cycles (paced), and drain the output.
	task automatic run_paced_traffic(input int unsigned frames);
		localparam int unsigned  IFM_SIZE = FM_SIZE[0];
		localparam int unsigned  OFM_SIZE = FM_SIZE[LAYERS];
		fork
			repeat(frames) begin
				automatic int unsigned  todo = IFM_SIZE;
				ivld <= 1;
				repeat(Period) begin
					@(posedge clk);
					if(ivld && irdy) begin
						if(--todo == 0)  ivld <= 0;
					end
				end
				ivld <= 0;
				assert(!todo) else begin
					$error("Couldn't feed feature map within computed period.");
					$stop;
				end
			end
			begin
				ordy <= 1;
				repeat(frames*OFM_SIZE) @(posedge clk iff ovld);
				ordy <= 0;
			end
		join
		repeat(16) @(posedge clk);
	endtask : run_paced_traffic

	//=======================================================================
	// Stimulus
	initial begin
		automatic logic [31:0]  ret;
		automatic logic [31:0]  peak_paced[0:LAYERS];
		automatic logic [31:0]  peak_bounded[0:LAYERS];
		automatic logic [31:0]  stall_before[0:LAYERS];
		automatic int unsigned  errors = 0;

		idat = 0;
		ivld = 0;
		ordy = 0;
		@(posedge clk iff !rst);
		repeat(4) @(posedge clk);

		//-------------------------------------------------------------------
		// Phase 1: detached run to determine the bottleneck period, exactly as the
		// driver does it: the instrumentation source is not started (no input to the
		// first gauge), the sink drains. The BARRIER_CLEAN read must complete once every
		// gauge has produced its four detached output frames.
		$display("[%0t] RUN_DETACHED4 (source idle)", $time);
		reset_accelerator();
		ordy <= 1;
		ctrl_run(RUN_DETACHED4);
		fork
			begin
				// the controller re-issues BARRIER_CLEAN until the ring reports clean
				ctrl_read(BARRIER_CLEAN, 0, ret);
			end
			begin
				repeat(4000) @(posedge clk);
				$error("BARRIER_CLEAN did not complete within 4000 cycles.");
				$stop;
			end
		join_any
		disable fork;
		assert(ret == BARRIER_CLEAN) else begin
			$error("BARRIER_CLEAN acknowledged with 0x%08x.", ret);
			errors++;
		end
		$display("[%0t] BARRIER_CLEAN completed", $time);
		ordy <= 0;
		ctrl_read(COMP_PERIOD, 0, ret);
		Period = ret;
		$display("[%0t] COMP_PERIOD = %0d cycles", $time, Period);
		assert(Period > 0 && Period < 1000) else begin
			$error("Implausible period %0d.", Period);
			$stop;
		end

		//-------------------------------------------------------------------
		// Phase 2: paced run, read peaks (READ_FILL) and stall flags (READ_STALL)
		$display("[%0t] RUN_PACED", $time);
		reset_accelerator();
		ctrl_run(RUN_PACED);
		run_paced_traffic(6);
		for(int unsigned  l = 0; l <= LAYERS; l++) begin
			ctrl_read(READ_FILL, l, peak_paced[l]);
			ctrl_read(READ_STALL, l, ret);
			$display("  paced: FIFO %0d peak = %0d, stall = %08b", l, peak_paced[l], ret[7:0]);
			assert(peak_paced[l] > 0 && peak_paced[l] < 1000) else begin
				$error("Implausible paced peak %0d for FIFO %0d.", peak_paced[l], l);
				errors++;
			end
			assert(ret[31:8] == 0) else begin
				$error("READ_STALL returned non-zero upper bytes 0x%08x for FIFO %0d.", ret, l);
				errors++;
			end
			// paced mode: the gauge is always ready, so the producer was never blocked ...
			assert(ret[5] == 0) else begin
				$error("Sticky producer-blocked flag set in paced mode for FIFO %0d.", l);
				errors++;
			end
			// ... but was idle at least once (the run starts with an empty pipeline)
			assert(ret[4] == 1) else begin
				$error("Sticky producer-idle flag not set in paced mode for FIFO %0d.", l);
				errors++;
			end
		end

		//-------------------------------------------------------------------
		// Phase 3: bounded run at the paced peaks must reproduce the paced execution
		$display("[%0t] RUN_BOUNDED at paced peaks", $time);
		reset_accelerator();
		for(int unsigned  l = 0; l <= LAYERS; l++)  ctrl_set_depth(l, peak_paced[l]);
		ctrl_run(RUN_BOUNDED);
		run_paced_traffic(6);
		for(int unsigned  l = 0; l <= LAYERS; l++) begin
			ctrl_read(READ_FILL, l, peak_bounded[l]);
			ctrl_read(READ_STALL, l, stall_before[l]);
			$display("  bounded: FIFO %0d peak = %0d, stall = %08b", l, peak_bounded[l], stall_before[l][7:0]);
			assert(peak_bounded[l] == peak_paced[l]) else begin
				$error("Bounded peak %0d differs from paced peak %0d for FIFO %0d.", peak_bounded[l], peak_paced[l], l);
				errors++;
			end
			assert(stall_before[l][5] == 0) else begin
				$error("Producer blocked although capacity equals the paced peak for FIFO %0d.", l);
				errors++;
			end
		end

		//-------------------------------------------------------------------
		// Phase 4: the unbounded capacity register value reproduces the paced execution
		$display("[%0t] RUN_BOUNDED with unbounded capacities", $time);
		reset_accelerator();
		for(int unsigned  l = 0; l <= LAYERS; l++)  ctrl_set_depth(l, 32'h7FFF_FFFF);
		ctrl_run(RUN_BOUNDED);
		run_paced_traffic(6);
		for(int unsigned  l = 0; l <= LAYERS; l++) begin
			ctrl_read(READ_FILL, l, ret);
			assert(ret == peak_paced[l]) else begin
				$error("Unbounded peak %0d differs from paced peak %0d for FIFO %0d.", ret, peak_paced[l], l);
				errors++;
			end
		end

		//-------------------------------------------------------------------
		// Phase 5: bounded run with a blocked output must deadlock, the sticky flags
		// must show every producer blocked while the per-frame flags of the last
		// gauge (which never completes a frame) keep their previous value.
		$display("[%0t] RUN_BOUNDED with blocked output (deadlock)", $time);
		reset_accelerator();
		for(int unsigned  l = 0; l <= LAYERS; l++)  ctrl_set_depth(l, 2);
		ctrl_run(RUN_BOUNDED);
		ivld <= 1;
		ordy <= 0;
		repeat(4000) @(posedge clk);
		assert(!irdy) else begin
			$error("Pipeline did not deadlock: input still accepted.");
			errors++;
		end
		for(int unsigned  l = 0; l <= LAYERS; l++) begin
			automatic logic [31:0]  fill;
			ctrl_read(READ_FILL, l, fill);
			ctrl_read(READ_STALL, l, ret);
			$display("  deadlock: FIFO %0d peak = %0d, stall = %08b", l, fill, ret[7:0]);
			assert(fill == 3) else begin
				$error("Deadlocked FIFO %0d holds %0d instead of capacity+1 = 3.", l, fill);
				errors++;
			end
			assert(ret[5] == 1) else begin
				$error("Sticky producer-blocked flag not set in deadlock for FIFO %0d.", l);
				errors++;
			end
		end
		assert(ret[3:0] == 4'b0000) else begin
			$error("Per-frame flags of the last gauge are %04b without a completed frame since reset.", ret[3:0]);
			errors++;
		end
		ivld <= 0;

		if(errors == 0)  $display("Test done: PASS.");
		else             $display("Test done: FAIL (%0d errors).", errors);
		$finish();
	end

	// Watchdog
	initial begin
		#2ms;
		$error("Timeout.");
		$stop;
	end

endmodule : fifo_controller_tb
