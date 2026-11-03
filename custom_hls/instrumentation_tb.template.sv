// Copyright (c) 2023 Advanced Micro Devices, Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of AMD nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


module tb #(
	// sampling period (in cycles) for reading instrumentation wrapper registers
    // TODO: make configurable or adjust automatically?
	int unsigned  INSTR_READ_PERIOD = 10000,
    // 16-bit LFSR seed for generating fixed random data
    int unsigned LFSR_SEED = 1,
    // minimum interval (cycles) between IFM starts (0 = no throttling)
    int unsigned THROTTLE_INTERVAL = 0,
    // sliding window size (frames) of the averages and the window min/max
    int unsigned AVG_WINDOW = 8,
    // number of completed output frames after which the simulation stops
    int unsigned NUM_FRAMES = 16
)();


// Clock & Reset
logic  ap_clk = 0;
always #5ns ap_clk = !ap_clk;
logic  ap_rst_n = 0;
uwire  ap_rst = !ap_rst_n;

// wires for instrumentation wrapper AXI lite interface
logic [31:0] axilite_ctrl_araddr = 'x;
uwire axilite_ctrl_arready;
logic axilite_ctrl_arvalid = 0;
logic [31:0]  axilite_ctrl_awaddr = 'x;
uwire axilite_ctrl_awready;
logic axilite_ctrl_awvalid = 0;
uwire axilite_ctrl_bready = 1;
uwire [1:0]axilite_ctrl_bresp;
uwire axilite_ctrl_bvalid;
uwire [31:0]axilite_ctrl_rdata;
logic axilite_ctrl_rready = 1;
uwire [1:0]axilite_ctrl_rresp;
uwire axilite_ctrl_rvalid;
logic [31:0]  axilite_ctrl_wdata = 'x;
uwire axilite_ctrl_wready;
uwire [3:0]axilite_ctrl_wstrb = 4'b1111;
logic  axilite_ctrl_wvalid = 0;

// Register map of the instrumentation wrapper (instrumentation.template.cpp, s_axilite
// bundle ctrl; every port occupies 16 bytes from status on: data + valid register)
localparam logic [31:0]  ADDR_CFG                = 'h010;
localparam logic [31:0]  ADDR_SEED               = 'h018;
localparam logic [31:0]  ADDR_AVG_N              = 'h020;
localparam logic [31:0]  ADDR_MUX_INTERVAL       = 'h028;
localparam logic [31:0]  ADDR_STATUS             = 'h030;
localparam logic [31:0]  ADDR_LATENCY            = 'h040;
localparam logic [31:0]  ADDR_INTERVAL           = 'h050;
localparam logic [31:0]  ADDR_CHECKSUM           = 'h060;
localparam logic [31:0]  ADDR_MIN_LATENCY        = 'h070;
localparam logic [31:0]  ADDR_LAT_SUM_LO         = 'h080;
localparam logic [31:0]  ADDR_LAT_SUM_HI         = 'h090;
localparam logic [31:0]  ADDR_INT_SUM_LO         = 'h0a0;
localparam logic [31:0]  ADDR_INT_SUM_HI         = 'h0b0;
localparam logic [31:0]  ADDR_AVG_FILL           = 'h0c0;
localparam logic [31:0]  ADDR_RUN_CYCLES_LO      = 'h0d0;
localparam logic [31:0]  ADDR_RUN_CYCLES_HI      = 'h0e0;
localparam logic [31:0]  ADDR_RUN_FRAMES         = 'h0f0;
localparam logic [31:0]  ADDR_LAT_WIN_MIN        = 'h100;
localparam logic [31:0]  ADDR_LAT_WIN_MAX        = 'h110;
localparam logic [31:0]  ADDR_INT_WIN_MIN        = 'h120;
localparam logic [31:0]  ADDR_INT_WIN_MAX        = 'h130;
localparam logic [31:0]  ADDR_IDLE_CYCLES        = 'h140;
localparam logic [31:0]  ADDR_OLDEST_PENDING_AGE = 'h150;


dut_wrapper dut_wrapper_inst (
	.ap_clk_0(ap_clk), .ap_rst_n_0(ap_rst_n),
    .s_axi_ctrl_0_araddr(axilite_ctrl_araddr),
    .s_axi_ctrl_0_arready(axilite_ctrl_arready),
    .s_axi_ctrl_0_arvalid(axilite_ctrl_arvalid),
    .s_axi_ctrl_0_awaddr(axilite_ctrl_awaddr),
    .s_axi_ctrl_0_awready(axilite_ctrl_awready),
    .s_axi_ctrl_0_awvalid(axilite_ctrl_awvalid),
    .s_axi_ctrl_0_bready(axilite_ctrl_bready),
    .s_axi_ctrl_0_bresp(axilite_ctrl_bresp),
    .s_axi_ctrl_0_bvalid(axilite_ctrl_bvalid),
    .s_axi_ctrl_0_rdata(axilite_ctrl_rdata),
    .s_axi_ctrl_0_rready(axilite_ctrl_rready),
    .s_axi_ctrl_0_rresp(axilite_ctrl_rresp),
    .s_axi_ctrl_0_rvalid(axilite_ctrl_rvalid),
    .s_axi_ctrl_0_wdata(axilite_ctrl_wdata),
    .s_axi_ctrl_0_wready(axilite_ctrl_wready),
    .s_axi_ctrl_0_wstrb(axilite_ctrl_wstrb),
    .s_axi_ctrl_0_wvalid(axilite_ctrl_wvalid)
);

//---------------------------------------------------------------------------
// AXI-lite access tasks

task automatic axi_write(input logic [31:0] addr, input logic [31:0] data);
    axilite_ctrl_awaddr  <= addr;
    axilite_ctrl_awvalid <= 1;
    axilite_ctrl_wdata   <= data;
    axilite_ctrl_wvalid  <= 1;
    repeat(8) begin
        @(posedge ap_clk);
        if(axilite_ctrl_wready && axilite_ctrl_awready)  break;
    end
    axilite_ctrl_wvalid  <= 0;
    axilite_ctrl_awvalid <= 0;
    axilite_ctrl_awaddr  <= 'x;
    axilite_ctrl_wdata   <= 'x;
    @(posedge ap_clk);
endtask : axi_write

task automatic axi_read(input logic [31:0] addr, output logic [31:0] data);
    data = 'x;
    axilite_ctrl_araddr  <= addr;
    axilite_ctrl_arvalid <= 1;
    repeat(8) begin
        @(posedge ap_clk);
        if(axilite_ctrl_rvalid) begin
            data = axilite_ctrl_rdata;
            break;
        end
    end
    axilite_ctrl_arvalid <= 0;
    axilite_ctrl_araddr  <= 'x;
    @(posedge ap_clk);
endtask : axi_read

//---------------------------------------------------------------------------

int unsigned  errors = 0;

initial begin
    logic [31:0]  status, latency, interval, checksum, min_latency, avg_fill, run_frames;
    logic [31:0]  lat_win_min, lat_win_max, int_win_min, int_win_max;
    logic [31:0]  idle_cycles, oldest_pending_age;
    logic [31:0]  prev_run_frames;

	$timeformat(-9, 2, " ns");
    prev_run_frames = 0;
	// perform reset
	repeat(100)  @(posedge ap_clk);
	ap_rst_n <= 1;
	$display("Reset complete");
    repeat(100) @(posedge ap_clk);

    // instrumentation wrapper configuration: LFSR seed, window size, no tUSER multiplexing,
    // then start the data generation (cfg[0]) with the throttle interval (cfg[31:1])
    axi_write(ADDR_SEED, LFSR_SEED << 16);
    axi_write(ADDR_AVG_N, AVG_WINDOW);
    axi_write(ADDR_MUX_INTERVAL, 0);
    axi_write(ADDR_CFG, (THROTTLE_INTERVAL << 1) | 1);

    while(1) begin
        axi_read(ADDR_STATUS,             status);
        axi_read(ADDR_LATENCY,            latency);
        axi_read(ADDR_INTERVAL,           interval);
        axi_read(ADDR_CHECKSUM,           checksum);
        axi_read(ADDR_MIN_LATENCY,        min_latency);
        axi_read(ADDR_AVG_FILL,           avg_fill);
        axi_read(ADDR_RUN_FRAMES,         run_frames);
        axi_read(ADDR_LAT_WIN_MIN,        lat_win_min);
        axi_read(ADDR_LAT_WIN_MAX,        lat_win_max);
        axi_read(ADDR_INT_WIN_MIN,        int_win_min);
        axi_read(ADDR_INT_WIN_MAX,        int_win_max);
        axi_read(ADDR_IDLE_CYCLES,        idle_cycles);
        axi_read(ADDR_OLDEST_PENDING_AGE, oldest_pending_age);
        $display("[t=%0t] STATUS = %0d  FRAMES = %0d  LATENCY = %0d  INTERVAL = %0d  CHECKSUM = %8x",
                 $time, status, run_frames, latency, interval, checksum);
        $display("[t=%0t] MIN_LATENCY = %0d  AVG_FILL = %0d  LAT_WIN = [%0d, %0d]  INT_WIN = [%0d, %0d]  IDLE = %0d  OLDEST_PENDING_AGE = %0d",
                 $time, min_latency, avg_fill, lat_win_min, lat_win_max, int_win_min, int_win_max, idle_cycles, oldest_pending_age);

        // Steady-state register checks (the window extremes lag a completion by up to
        // 2*AVG_WINDOW cycles, so only invariants are checked)
        if(status != 0) begin
            $error("Timestamp overflow/underflow reported: status = %0d", status);
            errors++;
        end
        if(lat_win_min > lat_win_max || int_win_min > int_win_max) begin
            $error("Window minimum above window maximum");
            errors++;
        end
        if(run_frames >= 2 && avg_fill >= 2) begin
            // at least one frame of the run has a valid interval in the window
            if(int_win_max == 0 || int_win_min == 0) begin
                $error("Empty interval window after %0d frames", run_frames);
                errors++;
            end
            if(lat_win_min < min_latency) begin
                $error("Window minimum latency %0d below the running minimum %0d", lat_win_min, min_latency);
                errors++;
            end
        end
        if(avg_fill > AVG_WINDOW) begin
            $error("Window fill %0d exceeds the configured window %0d", avg_fill, AVG_WINDOW);
            errors++;
        end
        if(run_frames == prev_run_frames && run_frames > 0 && idle_cycles < INSTR_READ_PERIOD) begin
            $error("No frame completed during the read period but idle_cycles = %0d", idle_cycles);
            errors++;
        end
        prev_run_frames = run_frames;

        if(run_frames >= NUM_FRAMES) begin
            $display("%0d frames completed, stopping simulation (%0d errors)", run_frames, errors);
            if(errors == 0)  $display("INSTRUMENTATION TESTBENCH PASSED");
            else             $display("INSTRUMENTATION TESTBENCH FAILED");
            $finish;
        end
        repeat(INSTR_READ_PERIOD)  @(posedge ap_clk);
    end
end


endmodule : tb
