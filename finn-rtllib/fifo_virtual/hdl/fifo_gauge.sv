/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Gauge FIFO implementation. See package for details.
 *****************************************************************************/
module fifo_gauge #(
	bit [15:0]  ID,
	int unsigned  DATA_WIDTH,
	int unsigned  FM_SIZE
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,

	// Configuration Ring: Control & Status
	input	logic [7:0]  icfg,
	output	logic [7:0]  ocfg,

	// Input Stream
	input	logic [DATA_WIDTH-1:0]  idat,	// ignored, only for AXIS interface completeness
	input	logic  ivld,
	output	logic  irdy,

	// Output Stream
	output	logic [DATA_WIDTH-1:0]  odat,
	output	logic  ovld,
	input	logic  ordy
);
	import  fifo_gauge_pkg::*;

	//-----------------------------------------------------------------------
	// Control & Status Regs
	logic [31:0]  Period   = 0; // sticky high bit for overflow detection
	logic [ 3:0]  StallSig = 0;
	logic [31:0]  FillMax  = 0; // sticky high bit for overflow detection
	logic         fill_put;
	uwire [31:0]  fill_val;

	//-----------------------------------------------------------------------
	// IO Control & Tracking
	uwire  itxn = ivld && irdy;
	uwire  otxn = ovld && ordy;

	logic  IDirty = 0;	// partial IFM
	logic  ODirty = 0;	// partial OFM
	uwire  iclr4;	// clear detached consumption
	uwire  oclr4;	// clear detached production
	if(1) begin : blkFmTracking

		//- Feature Map & Period Tracking -----------------------------------
		localparam int unsigned  WRAP_INC = 2**$clog2(FM_SIZE)-(FM_SIZE-1);
		logic [31:0]  PCnt = 0;  // Clock Count for Period Tracking (on output side)
		logic [$clog2(FM_SIZE)+1:0]  ICnt = 0;  // Input Txn Count
		logic [$clog2(FM_SIZE)+1:0]  OCnt = 0;  // Output Txn COunt
		uwire  ilst;
		uwire  olst;
		if(FM_SIZE < 2) begin
			assign	ilst = 1;
			assign	olst = 1;
		end
		else begin
			logic  ILst = 0;
			logic  OLst = 0;
			always_ff @(posedge clk) begin
				if(rst) begin
					ILst <= 0;
					OLst <= 0;
				end
				else begin
					if(itxn)  ILst <= ((FM_SIZE % 2) || !ILst) && (((FM_SIZE-2) & ~ICnt) == 0);
					if(otxn)  OLst <= ((FM_SIZE % 2) || !OLst) && (((FM_SIZE-2) & ~OCnt) == 0);
				end
			end
			assign	ilst = ILst;
			assign	olst = OLst;
		end
		always_ff @(posedge clk) begin
			if(rst) begin
				PCnt <= 0; Period <= 0;
				ICnt <= 0; IDirty <= 0;
				OCnt <= 0; ODirty <= 0;
			end
			else begin
				automatic logic [31:0]  pcnt = PCnt + 1;
				pcnt[31] |= PCnt[31];
				if(otxn && olst) begin
					Period <= pcnt;
					pcnt = 0;
				end
				PCnt <= pcnt;

				if(itxn) begin
					ICnt <= ICnt + (ilst? WRAP_INC : 1);
					IDirty <= !ilst;
				end
				if(otxn) begin
					OCnt <= OCnt + (olst? WRAP_INC : 1);
					ODirty <= !olst;
				end
			end
		end
		assign	odat = OCnt;

		assign	iclr4 = &ICnt[$left(ICnt)-:2] && itxn && ilst;
		assign	oclr4 = &OCnt[$left(OCnt)-:2] && otxn && olst;

		//- Stall Tracking --------------------------------------------------
		typedef struct packed {
			logic  rdy;
			logic  vld;
		} stall_t;
		stall_t  IStall = '{ default: 0 };
		stall_t  OStall = '{ default: 0 };

		always_ff @(posedge clk) begin
			if(rst) begin
				IStall <= '{ default: 0 };
				OStall <= '{ default: 0 };
				StallSig <= '0;
			end
			else begin
				automatic stall_t  istall = '{
					rdy: IStall.rdy || (ivld && !irdy),
					vld: IStall.vld || (irdy && !ivld)
				};
				automatic stall_t  ostall = '{
					rdy: OStall.rdy || (ovld && !ordy),
					vld: OStall.vld || (ordy && !ovld)
				};
				if(itxn && ilst) begin
					StallSig[3:2] <= ostall;
					istall = '{ default: 0 };
				end
				if(otxn && olst) begin
					StallSig[1:0] <= istall;
					ostall = '{ default: 0 };
				end
				IStall <= istall;
				OStall <= ostall;
			end
		end

	end : blkFmTracking

	//-----------------------------------------------------------------------
	// State Machine for Ring Control
	typedef enum logic [1:0] {
		fifo = 2'b0x,
		hold = 2'b10,
		free = 2'b11
	} fifo_flow_e;
	fifo_flow_e  iflow;
	fifo_flow_e  oflow;
	if(1) begin : blkCfgCtrl

		// Protocol FSM
		typedef enum logic [3:0] {
			Start       = 4'b00xx,
			CompMax     = 4'b01xx,

			Id          = 4'b10xx,
			IdReadFill  = 4'b1000,
			IdReadStall = 4'b1001,
			IdWriteFill = 4'b1010,

			Read        = 4'b110x,
			ReadFill    = 4'b1100,
			ReadStall   = 4'b1101,

			WriteFill   = 4'b111x,

			IsRdWr      = 4'b1xxx,
			IdIsWrite   = 4'bxx1x,
			ReadIsStall = 4'bxxx1
		} state_e;
		state_e  State = Start;
		state_e  state_nxt;

		// IO Mode
		typedef enum logic [2:0] {
			Bounded       = 3'b000,
			Paced         = 3'b010,
			Detached      = 3'b1xx,
			DetachedStart = 3'b111,
			DetachedDone  = 3'b100
		} io_mode_e;
		io_mode_e  IoMode = Bounded;
		io_mode_e  io_mode_nxt;
		assign	oflow = IoMode ==? Detached? (IoMode[0]? free : hold) : fifo;
		assign	iflow = IoMode ==? Detached? (IoMode[1]? free : hold) : (IoMode ==? Paced)? free : fifo;

		// Argument Byte Counter
		typedef logic signed [2:0]  cnt_t;
		cnt_t  Cnt = -2;	// 2, 1, 0, -1 (last), -2 (idle)
		cnt_t  cnt_inc;
		uwire  cnt_lst = Cnt[$left(Cnt)];

		// Decision Selector (as in MAX computation)
		typedef struct packed { logic  mine; logic  other; }  sel_t;
		sel_t  Sel = '{ default: 'x };
		sel_t  sel_nxt;

		logic [2:0][7:0]  History = 'x; // shifted left with each icfg input

		always_ff @(posedge clk) begin
			if(rst) begin
				State  <= Start;
				IoMode <= Bounded;
				Cnt <= -2;

				Sel     <= '{ default: 'x };
				History <= 'x;
			end
			else begin
				State  <= state_nxt;
				IoMode <= io_mode_nxt;
				Cnt <= Cnt + cnt_inc;

				Sel     <= sel_nxt;
				History <= { History, icfg };
			end
		end

		//- Config Link Forwarding ----------
		logic  cfg_put; // patch control by FSM
		logic  cfg_set; // set LSB
		uwire [7:0]  cfg_val;
		if(1) begin : blkCfg

			// Forwarding Register
			logic [7:0]  OCfg = NOP;
			always_ff @(posedge clk) begin
				if(rst)  OCfg <= NOP;
				else     OCfg <= cfg_put? cfg_val : (icfg | cfg_set);
			end
			assign	ocfg = OCfg;

			// Local Patch Value Selection
			uwire [31:0]  cfg_src =
				State !=? IsRdWr?      Period  :
				State !=? ReadIsStall? FillMax :
				{ 28'b0, StallSig };
			uwire [7:0]  cfg_val_b[4] = '{
				// Cnt[1:0]
				/*  2 */ 2: cfg_src[24+:8],
				/*  1 */ 1: cfg_src[16+:8],
				/*  0 */ 0: cfg_src[ 8+:8],
				/* -1 */ 3: cfg_src[ 0+:8]
			};
			assign  cfg_val = cfg_val_b[Cnt[1:0]];

		end : blkCfg

		//- State Machine Update ------------
		assign	fill_val = State ==? IsRdWr? { History, icfg } : 0;
		always_comb begin

			io_mode_nxt = IoMode;
			if(IoMode ==? Detached) begin
				io_mode_nxt[0] &= !oclr4;
				io_mode_nxt[1] &= !iclr4;
			end
			state_nxt = State;
			cnt_inc = 0;
			sel_nxt = '{ default: 'x };

			cfg_put = 0;
			cfg_set = 0;

			fill_put =  0;

			unique casex(State)
			Start:
				unique casex(icfg)
				M_NOP: state_nxt <= Start;
				M_RUN: begin // Run Control
					unique casex(icfg)
					M_RUN_BOUNDED:  begin fill_put = 0; io_mode_nxt = Bounded; end
					M_RUN_PACED:    begin fill_put = 1; io_mode_nxt = Paced; end
					M_RUN_DETACHED: begin fill_put = 1; io_mode_nxt = DetachedStart; end
					endcase
					state_nxt = Start;
				end
				M_BARRIER: begin
					cfg_set = IDirty || ODirty || ((IoMode ==? Detached) && (IoMode !=? DetachedDone));
					state_nxt = Start;
				end
				M_COMP: begin // Compute
					state_nxt = CompMax;
					sel_nxt = '{ default: 0 };
					cnt_inc = 4;
				end
				M_READ: begin // Read Out
					state_nxt = icfg[0]? IdReadStall : IdReadFill;
					sel_nxt.mine = 1;
					cnt_inc = 2;
				end
				M_WRITE: begin // Write
					state_nxt = IdWriteFill;
					sel_nxt.mine = 1;
					cnt_inc = 2;
				end
				endcase

			CompMax: begin
				sel_nxt = Sel;
				if(!Sel && (cfg_val != icfg)) begin
					if(cfg_val < icfg)  sel_nxt.other = 1;
					else                sel_nxt.mine  = 1;
				end
				cfg_put = sel_nxt.mine;

				state_nxt = cnt_lst? Start : CompMax;
				cnt_inc = -1;
			end

			Id: begin
				localparam bit [0:1][7:0]  ID_BYTES = ID;
				sel_nxt.mine = Sel.mine && (icfg == ID_BYTES[Cnt[0]]);

				cnt_inc = -1;
				if(cnt_lst) begin
					state_nxt =
						State ==? IdIsWrite?   WriteFill :
						State ==? ReadIsStall? ReadStall :
						/* else */             ReadFill;
					cnt_inc = 3;
				end
			end

			Read: begin
				sel_nxt.mine = Sel.mine;
				cfg_put = Sel.mine;

				cnt_inc = -1;
				if(cnt_lst)  state_nxt = Start;
			end

			WriteFill: begin
				sel_nxt.mine = Sel.mine;

				cnt_inc = -1;
				if(cnt_lst) begin
					fill_put = Sel.mine;
					state_nxt = Start;
				end
			end

			endcase
		end

	end : blkCfgCtrl

	//-----------------------------------------------------------------------
	// Virtual FIFO

	// FIFO Pacing & Tracking
	logic [31:0]  FillCnt = 0;
	logic  Avl = 0;
	uwire  cap = FillCnt <= FillMax;
	always_ff @(posedge clk) begin
		if(rst) begin
			FillCnt <= 0;
			Avl <= 0;
			FillMax <= 0;
		end
		else begin
			FillCnt <= FillCnt + (itxn == otxn? 0 : itxn? 1 : -1);
			Avl <= itxn || (FillCnt[0] && !otxn) || |FillCnt[31:1];

			if(!cap && (iflow !=? fifo))  FillMax <= FillCnt;
			if(fill_put)  FillMax <= fill_val;
		end
	end
	assign	irdy = (iflow ==? free) || ((iflow ==? fifo) && cap);
	assign	ovld = (oflow ==? free) || ((oflow ==? fifo) && Avl);

endmodule : fifo_gauge
