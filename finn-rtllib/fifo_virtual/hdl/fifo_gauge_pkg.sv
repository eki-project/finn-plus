/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Definitions for Communication Ring Protocol for Gauge FIFO.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *
 * @description
 *	FIFO sizing can be accomplished by the use of `fifo_gauge` modules
 *	inserted into the designated FIFO locations. These modules are controlled
 *	and monitored by a central control using a single-byte communication ring,
 *	the gauging ring. For easy placement and timing closure, its layout should
 *	be guided by the underlying dataflow topology.
 *	The gauging ring transports a continuous stream of operation packets.
 *	Gauging FIFOs on the ring cannot alter the size of any of these packets,
 *	they may only patch their contents if so implied by the requested
 *	operation. A continuous stream of NOPs (0x00) is placed on the ring by the
 *	central control while it is not requesting the execution of an operation.
 *	The FIFO sizing procedure executed by the central control would typically
 *	follow this pattern:
 *	 - Issue a `RUN_DETACHED4` command prompting each gauging FIFO to
 *	   independently produce 4 feature maps of output and consume 4 feature
 *	   maps of input without any flow dependency. Each FIFO will record the
 *	   period spent on producing individual feature map outputs.
 *	 - Issue `BARRIER_CLEAN` commands. Gauging FIFOs will replace it by a
 *	   `BARRIER_DIRTY` if the detached production and consumption of
 *	   feature maps is still in process. They forward the `BARRIER_*`
 *	   command unchanged otherwise. Once, the central control receives back a
 *	   `BARRIER_CLEAN`, the detached individual throughput measurement is
 *	   completed globally.
 *	 - Issue a `COMP_PERIOD` command with a 32-bit argument in network byte
 *	   order. The command is issued with an argument of 0. Each gauging FIFO
 *	   on the ring updates this value by the maximum between the value it
 *	   received and the feature map output period it determined locally in
 *	   the preceding detached run. This accomplishes a distributed computation
 *	   of the global maximum of the feature map initiation interval.
 *	 - Issue a `RUN_PACED` command allowing each gauging FIFO to emit output
 *	   only in correspondence to the data volume it has itself received by
 *	   any given point in time. No restriction is placed on the acceptance of
 *	   FIFO input data. However, in this process, each gauging FIFO tracks
 *	   the maximum number of transactions the input is observed to be ahead of
 *	   the output.
 *	 - The central control feeds the dataflow pipeline with input data pacing
 *	   the start of subsequent feature map inputs by the computed global
 *	   period.
 *	 - Issue `READ_FILL` commands after feeding two or more feature maps, one
 *	   for each gauging FIFO on the ring. This will inform the central control
 *	   about the maximum depth each FIFO assumed during the operation paced
 *	   to match the bottleneck throughput of the pipeline.
 *****************************************************************************/
package fifo_gauge_pkg;
	//-----------------------------------------------------------------------
	// Ring Protocol
	//  - Ring communication pure pass-through, no flow control.
	//  - Packets may be modified in contents, never in size.
	//  - Multi-byte values are transmitted in big-endian network order.
	//
	// [0] Opcode with arguments:
	//     COMP_PERIOD
	//       [1:4]	Period[31:0] - Online max with local value (big endian)
	//     READ_FILL
	//       [1:2]	ID[15:0] (big endian)
	//       [3:6]	ID matches? fill in local value : forward incoming
	//     READ_STALL
	//       [1:2]	ID[15:0] (big endian)
	//       [3]	ID matches? fill in local value : forward incoming
	//                 stall[1:0] - irdy_stall, ivld_stall
	//                 stall[3:2] - ordy_stall, ovld_stall
	//                 stall[7:4] - reserved (0)
	typedef enum logic [7:0] {
		NOP           = 8'h00,

		RUN_BOUNDED   = 8'h04,
		RUN_PACED     = 8'h05,
		RUN_DETACHED4 = 8'h07,

		BARRIER_CLEAN = 8'h08,
		BARRIER_DIRTY = 8'h09,
		COMP_PERIOD   = 8'h0A,

		READ_FILL     = 8'h0C,
		READ_STALL    = 8'h0D,
		WRITE_FILL    = 8'h0E,

		// Matching Patterns
		M_NOP          = 8'bxxxx_00xx,
		M_RUN          = 8'bxxxx_01xx,
		M_RUN_BOUNDED  = 8'bxxxx_xx00,
		M_RUN_PACED    = 8'bxxxx_xx01,
		M_RUN_DETACHED = 8'bxxxx_xx1x,
		M_BARRIER      = 8'bxxxx_100x,
		M_COMP         = 8'bxxxx_101x,
		M_IO           = 8'bxxxx_11xx,
		M_READ         = 8'bxxxx_110x,
		M_WRITE        = 8'bxxxx_111x
	} ring_opcode_e;

endpackage : fifo_gauge_pkg
