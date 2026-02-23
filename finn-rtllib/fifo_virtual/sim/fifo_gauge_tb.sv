/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Gauge FIFO testbench. See package for details.
 *****************************************************************************/
module fifo_gauge_tb;

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
	// Pipeline: -> FIFO -> (Layer -> FIFO)* ->
	logic [7:0]  icfg;
	uwire [7:0]  ocfg;
	logic [7:0]  idat;
	logic  ivld;
	uwire  irdy;
	uwire  ovld;
	logic  ordy;

	uwire ring_opcode_e  icfg_mnemonic = ring_opcode_e'(icfg);
	if(1) begin : blkPipeline

		//-------------------------------------------------------------------
		// DUTs
		uwire [7:0]  i_dat[0:LAYERS];
		uwire  i_vld[0:LAYERS];
		uwire  i_rdy[0:LAYERS];
		uwire  o_vld[0:LAYERS];
		uwire  o_rdy[0:LAYERS];
		uwire [7:0]  cfg[0:LAYERS+1];

		assign	i_dat[0] = idat;
		assign	i_vld[0] = ivld;
		assign	irdy = i_rdy[0];
		assign	cfg[0] = icfg;
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
		assign	ocfg = cfg[LAYERS+1];

		//-------------------------------------------------------------------
		// Abstracted Layers
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
			initial forever begin
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
		end : genLayer

	end : blkPipeline

	//-----------------------------------------------------------------------
	// Stimulus
	integer unsigned  Period = 'x;
	initial begin
		localparam int unsigned  IFM_SIZE = FM_SIZE[0];
		localparam int unsigned  OFM_SIZE = FM_SIZE[LAYERS];

		icfg = NOP;
		ivld = 0;
		ordy = 0;
		@(posedge clk iff !rst);

		//-------------------------------------------------------------------
		// Throughput Bounding

		// Initiate Phase
		fork
			begin
				icfg <= RUN_DETACHED4;
				@(posedge clk);
				icfg <= NOP;
			end
		join_none
		@(posedge clk iff (ocfg == RUN_DETACHED4));

		// Run four Feature Maps
		fork
			begin
				ivld <= 1;
				repeat(4*IFM_SIZE) @(posedge clk iff irdy);
				ivld <= 0;
			end
			begin
				ordy <= 1;
				repeat(4*OFM_SIZE) @(posedge clk iff ovld);
				ordy <= 0;
			end
		join

		// Check for clean Completion throughout
		forever begin
			fork
				begin
					icfg <= BARRIER_CLEAN;
					@(posedge clk);
					icfg <= NOP;
				end
			join_none
			@(posedge clk iff (ocfg ==? M_BARRIER));
			if(ocfg == BARRIER_CLEAN)  break;
		end

		// Collect Maximum Period
		fork
			begin
				icfg <= COMP_PERIOD;
				@(posedge clk);
				icfg <= 0;
				repeat(4) @(posedge clk);
				icfg <= NOP;
			end
			begin
				automatic integer unsigned  ret = 0;
				@(posedge clk iff (ocfg == COMP_PERIOD));
				repeat(4) @(posedge clk)  ret = { ret, ocfg };
				$display("Retrieved period of %0d cycles.", ret);
				Period = ret;
			end
		join

		//-------------------------------------------------------------------
		// FIFO Sizing
		fork
			begin
				icfg <= RUN_PACED;
				@(posedge clk);
				icfg <= NOP;
			end
		join_none
		@(posedge clk iff (ocfg == RUN_PACED));

		fork
			repeat(4) begin
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
				repeat(4*OFM_SIZE) @(posedge clk iff ovld);
				ordy <= 0;
			end
		join

		//-------------------------------------------------------------------
		// Validation
		fork
			begin
				icfg <= RUN_BOUNDED;
				@(posedge clk);
				icfg <= NOP;
			end
		join_none
		@(posedge clk iff (ocfg == RUN_BOUNDED));

		// Background Monitoring of Period
		ivld <= 1;
		ordy <= 1;
		fork
			// Input Feed
			begin
				automatic int unsigned  ticks = 0;
				automatic int unsigned  txns  = 0;
				idat <= 0;
				@(posedge clk iff irdy);

				forever @(posedge clk) begin
					ticks++;
					txns += irdy;
					idat <= (idat + irdy) % IFM_SIZE;
					if(txns == IFM_SIZE) begin
						$display("%s Input after %0d <= %0d.", (ticks <= Period? "." : "!"), ticks, Period);
						ticks = 0;
						txns = 0;
					end
				end
			end

			// Output Draining
			begin
				automatic int unsigned  ticks = 0;
				automatic int unsigned  txns  = 0;
				@(posedge clk iff ovld);

				forever @(posedge clk) begin
					ticks++;
					txns += ovld;
					if(txns == OFM_SIZE) begin
						$display("%s Output after %0d <= %0d.", (ticks <= Period? "." : "!"), ticks, Period);
						ticks = 0;
						txns = 0;
					end
				end
			end
		join_none
		repeat(3*OFM_SIZE) @(posedge clk iff ovld);

		// Shrink terminal FIFOs
		$display("Shrinking terminal FIFOs.");
		fork
			for(int unsigned  i = 0; i <= LAYERS; i += LAYERS) begin
				icfg <= WRITE_FILL;
				@(posedge clk);

				// ID
				icfg <= i[8+:8];
				@(posedge clk);
				icfg <= i[0+:8];
				@(posedge clk);

				// Value Slot
				icfg <= 0;
				repeat(3) @(posedge clk);
				icfg <= 1;
				@(posedge clk);
				icfg <= NOP;
			end
			repeat(2) @(posedge clk iff (ocfg == WRITE_FILL));
		join
		repeat(3*OFM_SIZE) @(posedge clk iff ovld);

		// Fill Readback
		fork
			for(int unsigned  l = 0; l <= LAYERS; l++) begin
				icfg <= READ_FILL;
				@(posedge clk);

				// ID
				icfg <= l[8+:8];
				@(posedge clk);
				icfg <= l[0+:8];
				@(posedge clk);

				// Value Slot
				icfg <= 'x;
				repeat(4) @(posedge clk);
				icfg <= NOP;
			end
			for(int unsigned  l = 0; l <= LAYERS; l++) begin
				automatic int unsigned  id   = 0;
				automatic int unsigned  fill = 0;
				@(posedge clk iff (ocfg == READ_FILL));
				repeat(2) @(posedge clk)  id   = { id,   ocfg };
				repeat(4) @(posedge clk)  fill = { fill, ocfg };
				$display("Retrieved FIFO Fill [%0d] = %0d (-> Size: %0d)", id, fill, fill+1);
			end
		join
		repeat(3*OFM_SIZE) @(posedge clk iff ovld);

		// Stall Signature Readback
		fork
			for(int unsigned  l = 0; l <= LAYERS; l++) begin
				icfg <= READ_STALL;
				@(posedge clk);

				// ID
				icfg <= l[8+:8];
				@(posedge clk);
				icfg <= l[0+:8];
				@(posedge clk);

				// Value Slot
				icfg <= 'x;
				repeat(4) @(posedge clk);
				icfg <= NOP;
			end
			for(int unsigned  l = 0; l <= LAYERS; l++) begin
				automatic int unsigned  id   = 0;
				automatic int unsigned  ret = 0;
				@(posedge clk iff (ocfg == READ_STALL));
				repeat(2) @(posedge clk)  id  = { id,  ocfg };
				repeat(4) @(posedge clk)  ret = { ret, ocfg };
				$display("Retrieved FIFO stall signature [%0d] = %4b", id, ret);
			end
		join

		$display("Test done.");
		$finish();
	end

endmodule : fifo_gauge_tb
