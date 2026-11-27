// DFX tUSER Passthrough: forwards AXI-Stream tUSER across a non-reconfigurable
// segment of the FINN accelerator pipeline.
//
// FINN CustomOps do not have tUSER ports and do not generate tLast.  When such
// a segment sits between two dfx_wrapper instances the tUSER metadata (which
// carries the RM-selection request set by host software) would otherwise be
// silently discarded.  This wrapper solves both problems:
//
//   1. tUSER forwarding: remembers the tUSER of every frame inside the segment
//      (tuser_track) and re-attaches the tUSER of the frame that is leaving to
//      its output beats.  A segment may hold several frames, and the frames
//      entering it may already select another RM or weight set than the ones
//      leaving it, so the tUSER of the last input frame would mislabel them.
//
//   2. tLast regeneration: counts accepted output beats and asserts m_axis_tlast
//      every NUM_OUTPUT_BEATS transfers (parameterised at instantiation).  This
//      is necessary because the wrapped non-PR IPs do not produce tLast, yet the
//      downstream dfx_wrapper relies on it to sample tUSER and detect when to
//      trigger reconfiguration.
//
// Topology:
//   upstream (tUSER, tLast) → s_axis → [this wrapper] → rp_m_axis → non-PR IP
//   non-PR IP               → rp_s_axis → [this wrapper] → m_axis → downstream

`timescale 1ns/1ps
module dfx_tuser_passthrough #(
    parameter int IN_DATA_WIDTH    = 64,  // tdata width on input path: s_axis → rp_m_axis (bits)
    parameter int OUT_DATA_WIDTH   = 64,  // tdata width on output path: rp_s_axis → m_axis (bits)
    parameter int TUSER_WIDTH      = 2,   // tUSER width (bits)
    parameter int TUSER_TRACK_DEPTH = 256, // frames the segment may hold (see tuser_track)
    parameter int NUM_OUTPUT_BEATS = 1    // AXI-Stream beats per output frame
) (
    input  logic                       aclk,
    input  logic                       aresetn,

    // External input: carries tUSER + tLast from upstream (DMA or previous wrapper)
    input  logic [IN_DATA_WIDTH-1:0]   s_axis_tdata,
    input  logic                       s_axis_tvalid,
    output logic                       s_axis_tready,
    input  logic                       s_axis_tlast,
    input  logic [TUSER_WIDTH-1:0]     s_axis_tuser,

    // External output: tUSER restored, tLast regenerated for downstream dfx_wrapper
    output logic [OUT_DATA_WIDTH-1:0]  m_axis_tdata,
    output logic                       m_axis_tvalid,
    input  logic                       m_axis_tready,
    output logic                       m_axis_tlast,
    output logic [TUSER_WIDTH-1:0]     m_axis_tuser,

    // To wrapped non-PR IP chain (no tUSER; tLast forwarded for protocol compliance)
    output logic [IN_DATA_WIDTH-1:0]   rp_m_axis_tdata,
    output logic                       rp_m_axis_tvalid,
    input  logic                       rp_m_axis_tready,
    output logic                       rp_m_axis_tlast,

    // From wrapped non-PR IP chain (no tUSER; tLast not used — regenerated internally)
    input  logic [OUT_DATA_WIDTH-1:0]  rp_s_axis_tdata,
    input  logic                       rp_s_axis_tvalid,
    output logic                       rp_s_axis_tready,
    input  logic                       rp_s_axis_tlast  // present for interface compliance, unused
);

    // --------------------------------------------------------------------------
    // frame_start: 1 when the next accepted input beat is the first of a frame
    //   (set after reset and after every accepted tLast).
    // --------------------------------------------------------------------------
    logic frame_start;

    always_ff @(posedge aclk or negedge aresetn) begin
        if (!aresetn)
            frame_start <= 1'b1;
        else if (s_axis_tvalid & s_axis_tready)
            frame_start <= s_axis_tlast; // next beat is first of a frame iff this was the last
    end

    // --------------------------------------------------------------------------
    // Output tLast generator
    //   Counts accepted output transfers and asserts m_axis_tlast on the last
    //   beat of each NUM_OUTPUT_BEATS-deep frame.  The counter resets to zero
    //   immediately after the last beat so the next frame starts cleanly.
    // --------------------------------------------------------------------------
    localparam int CNT_W = (NUM_OUTPUT_BEATS > 1) ? $clog2(NUM_OUTPUT_BEATS) : 1;
    logic [CNT_W-1:0] beat_cnt;
    logic output_last;
    assign output_last = (beat_cnt == CNT_W'(NUM_OUTPUT_BEATS - 1));

    always_ff @(posedge aclk or negedge aresetn) begin
        if (!aresetn)
            beat_cnt <= '0;
        else if (rp_s_axis_tvalid & rp_s_axis_tready)
            beat_cnt <= output_last ? '0 : beat_cnt + 1'b1;
    end

    // --------------------------------------------------------------------------
    // Per-frame tUSER of the frames inside the segment
    //   A frame enters with its first accepted beat and leaves with its last
    //   output beat. A frame's first beat also needs room in the tracker.
    // --------------------------------------------------------------------------
    logic                   tq_in_ready;
    logic [TUSER_WIDTH-1:0] tq_out_tuser;
    logic                   input_open;
    assign input_open = !frame_start || tq_in_ready;

    tuser_track #(
        .TUSER_WIDTH(TUSER_WIDTH),
        .DEPTH      (TUSER_TRACK_DEPTH)
    ) tuser_track_i (
        .aclk     (aclk),
        .aresetn  (aresetn),
        .in_frame (s_axis_tvalid & s_axis_tready & frame_start),
        .in_tuser (s_axis_tuser),
        .in_ready (tq_in_ready),
        .out_frame(rp_s_axis_tvalid & rp_s_axis_tready & output_last),
        .out_tuser(tq_out_tuser)
    );

    // --------------------------------------------------------------------------
    // Input pass-through (strip tUSER; forward tLast for protocol compliance)
    // --------------------------------------------------------------------------
    assign rp_m_axis_tdata  = s_axis_tdata;
    assign rp_m_axis_tvalid = s_axis_tvalid & input_open;
    assign rp_m_axis_tlast  = s_axis_tlast;
    assign s_axis_tready    = rp_m_axis_tready & input_open;

    // --------------------------------------------------------------------------
    // Output pass-through (restore tUSER; inject generated tLast)
    // --------------------------------------------------------------------------
    assign m_axis_tdata     = rp_s_axis_tdata;
    assign m_axis_tvalid    = rp_s_axis_tvalid;
    assign m_axis_tlast     = output_last;
    assign m_axis_tuser     = tq_out_tuser;
    assign rp_s_axis_tready = m_axis_tready;

endmodule
