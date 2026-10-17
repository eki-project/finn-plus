/******************************************************************************
 * Copyright (C) 2026, Paderborn University
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Registered reset stage for one dataflow node.
 *
 * The stitched IP hands every node the kernel's ap_rst_n directly, so one
 * register in the platform drives tens of thousands of synchronous-reset
 * loads across the whole SLR; that net was the worst path of the kernel
 * (mobilenetv1 on U55C, 13 Oct 2026: 2.6 ns of slack at 100 MHz, every other
 * path above 3.7 ns). This module re-registers the reset STAGES times right
 * in front of its node. MAX_FANOUT makes synthesis replicate the last stage,
 * and the placer puts the replicas next to their loads.
 *
 * The reset is synchronous already (the platform's proc_sys_reset drives
 * it), so this is a pipeline, not a synchroniser: every node leaves reset
 * STAGES cycles after the port does, all nodes together. Nothing flows
 * between nodes until the host starts the kernel, long after that.
 *****************************************************************************/

module reset_pipe #(
	int unsigned  STAGES = 2,
	int unsigned  MAX_FANOUT = 256
)(
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_RESET ap_rst_n:rst_n_out" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	input	logic  ap_clk,
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ap_rst_n RST" *)
	input	logic  ap_rst_n,
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst_n_out RST" *)
	output	logic  rst_n_out
);

	// Asserted at configuration, like the port it mirrors.
	(* SHREG_EXTRACT = "no", MAX_FANOUT = MAX_FANOUT *)
	logic [STAGES-1:0]  Pipe = '0;

	if(STAGES == 1) begin : genOne
		always_ff @(posedge ap_clk)  Pipe <= ap_rst_n;
	end
	else begin : genChain
		always_ff @(posedge ap_clk)  Pipe <= { Pipe[STAGES-2:0], ap_rst_n };
	end
	assign	rst_n_out = Pipe[STAGES-1];

endmodule : reset_pipe
