// DFX Wrapper: per-reconfigurable-region partial reconfiguration controller.
//
// Sits in the static region around one Block Design Container (BDC/RP).
// Uses AXI-Stream tLast to detect frame boundaries and tUSER to select which
// Reconfigurable Module (RM) should process each frame.
//
// Port naming convention:
//   s_axis_*   : input from upstream (host / static network)
//   m_axis_*   : output to downstream (host / static network)
//   rp_m_axis_*: output toward the BDC input (no tUSER -- FINN ops ignore it)
//   rp_s_axis_*: input from AMD dfx_decoupler/s_intf_0 (no tUSER)
//
// tUSER convention:
//   s_axis_tuser[RM_ID_W-1:0]: desired RM for THIS frame. The host sets this
//     to the same value on every beat of a frame. The wrapper peeks at it on the
//     first beat of each frame (before accepting the beat, i.e. tready=0) to
//     decide whether reconfiguration is needed before processing the frame.
//   m_axis_tuser: the full tUSER vector of the current frame, forwarded unchanged
//     so downstream wrappers receive their own RM-selection bits.
//
// State machine:
//   CHECK_TUSER  : peek at first beat of the next frame (tready=0). When tvalid
//                  goes high, latch tUSER and pending_rm_id. If pending_rm_id ==
//                  current_rm_id -> INFERENCE; else -> WAIT_FLUSH.
//   INFERENCE    : pass-through. On consumed tLast -> CHECK_TUSER.
//   WAIT_FLUSH   : block new input. Forward rp_s_axis->m_axis until the BDC
//                  pipeline is empty (frames_in_flight==0 OR last frame exits).
//   TRIGGER      : assert controller_trigger[pending_rm_id] for one cycle -> WAIT_DECOUPLE.
//   WAIT_DECOUPLE: wait for controller_decouple to go high -> RECONFIGURING.
//   RECONFIGURING: wait for controller_decouple to fall -> RESET. Update current_rm_id.
//   RESET        : hold accel_reset_n low for RESET_CYCLES cycles -> CHECK_TUSER.

`timescale 1ns/1ps
module dfx_wrapper #(
    parameter int IN_DATA_WIDTH   = 64,  // AXI-Stream tdata width on the INPUT path (s_axis / rp_m_axis)
    parameter int OUT_DATA_WIDTH  = 64,  // AXI-Stream tdata width on the OUTPUT path (rp_s_axis / m_axis)
    parameter int TUSER_WIDTH     = 2,   // tUSER width (bits); RM_ID is TUSER_WIDTH bits
    parameter int NUM_RM          = 2,   // number of Reconfigurable Modules
    parameter int RESET_CYCLES    = 16,  // clock cycles to assert accel_reset_n after reconfig
    parameter int NUM_OUTPUT_BEATS = 1   // AXI-Stream beats per output frame; used to
                                         // regenerate m_axis_tlast without relying on the
                                         // BDC generating rp_s_axis_tlast internally
) (
    input  logic                       aclk,
    input  logic                       aresetn,

    // External input (from upstream static network / DMA)
    input  logic [IN_DATA_WIDTH-1:0]   s_axis_tdata,
    input  logic                       s_axis_tvalid,
    output logic                       s_axis_tready,
    input  logic                       s_axis_tlast,
    input  logic [TUSER_WIDTH-1:0]     s_axis_tuser,

    // External output (to downstream static network / DMA)
    output logic [OUT_DATA_WIDTH-1:0]  m_axis_tdata,
    output logic                       m_axis_tvalid,
    input  logic                       m_axis_tready,
    output logic                       m_axis_tlast,
    output logic [TUSER_WIDTH-1:0]     m_axis_tuser,

    // To BDC input (no tUSER - FINN ops do not use it)
    output logic [IN_DATA_WIDTH-1:0]   rp_m_axis_tdata,
    output logic                       rp_m_axis_tvalid,
    input  logic                       rp_m_axis_tready,
    output logic                       rp_m_axis_tlast,

    // From AMD dfx_decoupler s_intf_0 (BDC output side, no tUSER)
    input  logic [OUT_DATA_WIDTH-1:0]  rp_s_axis_tdata,
    input  logic                       rp_s_axis_tvalid,
    output logic                       rp_s_axis_tready,
    input  logic                       rp_s_axis_tlast,

    // DFX controller interface
    output logic [NUM_RM-1:0]       controller_trigger,  // hw_triggers to DFX controller
    input  logic                    controller_decouple, // vsm_N_rm_decouple from DFX controller

    // Active-low reset to BDC (released after reconfig + RESET_CYCLES)
    output logic                    accel_reset_n
);

    // --------------------------------------------------------------------------
    // Derived parameters
    // --------------------------------------------------------------------------
    localparam int RM_ID_W      = (NUM_RM > 1) ? $clog2(NUM_RM) : 1;
    localparam int RESET_CNT_W  = $clog2(RESET_CYCLES + 1);
    localparam int OUT_CNT_W    = (NUM_OUTPUT_BEATS > 1) ? $clog2(NUM_OUTPUT_BEATS) : 1;
    // Width of the frames-in-flight counter. 8 bits supports up to 255 concurrent
    // frames inside the BDC pipeline, which is far more than any realistic FINN pipeline.
    localparam int INFLIGHT_W   = 8;

    // --------------------------------------------------------------------------
    // State encoding
    // --------------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_INFERENCE     = 3'd0,
        S_WAIT_FLUSH    = 3'd1,
        S_TRIGGER       = 3'd2,
        S_WAIT_DECOUPLE = 3'd3,
        S_RECONFIGURING = 3'd4,
        S_RESET         = 3'd5,
        S_CHECK_TUSER   = 3'd6  // peek at first beat of next frame before accepting it
    } state_t;

    state_t state;

    // --------------------------------------------------------------------------
    // Registers
    // --------------------------------------------------------------------------
    logic [RM_ID_W-1:0]     current_rm_id;
    logic [RM_ID_W-1:0]     pending_rm_id;
    logic [RESET_CNT_W-1:0] reset_cnt;
    logic [INFLIGHT_W-1:0]  frames_in_flight;
    logic [TUSER_WIDTH-1:0] tuser_reg;     // holds full tUSER of the current/most-recent frame
    logic                   decouple_prev; // for edge detection
    logic [OUT_CNT_W-1:0]   out_beat_cnt;  // counts accepted output beats for tLast regeneration

    // --------------------------------------------------------------------------
    // Input pass-through to BDC
    //   Only allowed in S_INFERENCE. Gate with tready from BDC.
    // --------------------------------------------------------------------------
    logic input_active;
    assign input_active = (state == S_INFERENCE);
    assign rp_m_axis_tdata  = s_axis_tdata;
    assign rp_m_axis_tvalid = s_axis_tvalid & input_active;
    assign rp_m_axis_tlast  = s_axis_tlast;
    assign s_axis_tready    = rp_m_axis_tready & input_active;

    // --------------------------------------------------------------------------
    // Output pass-through from BDC
    //   Forward rp_s_axis -> m_axis in INFERENCE and WAIT_FLUSH.
    //   In S_CHECK_TUSER: back-pressure the BDC output (tready=0) so that no
    //     output beats are consumed while peeking at s_axis_tuser. Consuming
    //     beats here without counting them (output_forward=0) would misalign
    //     out_beat_cnt, causing frame_out to fire at the wrong beat and leading
    //     to instrumentation timestamp overflow and spurious latency readings.
    //   In all other non-forwarding states (TRIGGER..RESET): absorb rp_s_axis
    //     (tready=1) since the AMD dfx_decoupler already drives tvalid=0 during
    //     this window anyway; the 1'b1 is a safe drain for any residual glitches.
    // --------------------------------------------------------------------------
    logic output_forward;
    assign output_forward = (state == S_INFERENCE) | (state == S_WAIT_FLUSH);

    // Output beat counter: counts accepted output beats and asserts out_beat_last
    // every NUM_OUTPUT_BEATS beats.  This regenerates m_axis_tlast independently
    // of rp_s_axis_tlast, so the BDC (PR bodies) do not need to generate tLast
    // internally (i.e. InsertTLastMarker is not required in PR bodies).
    logic out_beat_last;
    assign out_beat_last = (out_beat_cnt == OUT_CNT_W'(NUM_OUTPUT_BEATS - 1));

    assign m_axis_tdata     = rp_s_axis_tdata;
    assign m_axis_tvalid    = rp_s_axis_tvalid & output_forward;
    assign m_axis_tlast     = out_beat_last;
    // Forward the full upstream tUSER vector so downstream wrappers (dfx or passthrough)
    // receive the RM-selection bits intended for their own stage. tuser_reg is latched
    // in S_CHECK_TUSER from the first beat of each frame; since the FSM blocks new
    // input during reconfiguration all in-flight frames share the same tUSER value,
    // so a single hold register is sufficient.
    assign m_axis_tuser     = tuser_reg;
    assign rp_s_axis_tready = output_forward    ? m_axis_tready :
                               (state == S_CHECK_TUSER) ? 1'b0 : 1'b1;

    // --------------------------------------------------------------------------
    // Accelerator reset: active-low, de-asserted except in S_RESET
    // --------------------------------------------------------------------------
    assign accel_reset_n = (state != S_RESET) & aresetn;

    // --------------------------------------------------------------------------
    // Frames-in-flight tracking
    //   frame_in : a complete frame boundary (tLast) entered the BDC this cycle.
    //              Gated by input_active so only counted in S_INFERENCE.
    //              Uses rp_m_axis_tlast which is forwarded from s_axis_tlast (DMA).
    //   frame_out: a complete frame boundary exited the BDC this cycle.
    //              Uses out_beat_last (the beat counter) rather than rp_s_axis_tlast
    //              because PR bodies no longer generate tLast internally.
    //              Gated by output_forward (S_INFERENCE | S_WAIT_FLUSH) to avoid
    //              spurious counts while the AMD dfx_decoupler drives tvalid=0.
    // --------------------------------------------------------------------------
    logic frame_in;
    logic frame_out;
    assign frame_in  = rp_m_axis_tvalid & rp_m_axis_tready & rp_m_axis_tlast;
    assign frame_out = output_forward & rp_s_axis_tvalid & rp_s_axis_tready & out_beat_last;

    always_ff @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            frames_in_flight <= '0;
        end else begin
            unique case ({frame_in, frame_out})
                2'b10: frames_in_flight <= frames_in_flight + 1'b1; // frame entered, none exited
                2'b01: frames_in_flight <= frames_in_flight - 1'b1; // frame exited, none entered
                default: ; // 2'b00 or 2'b11 — net change is zero
            endcase
        end
    end

    // --------------------------------------------------------------------------
    // Output beat counter
    //   Counts accepted output beats (output_forward & tvalid & tready).
    //   Resets to 0 after every NUM_OUTPUT_BEATS beats (i.e. on out_beat_last).
    // --------------------------------------------------------------------------
    always_ff @(posedge aclk or negedge aresetn) begin
        if (!aresetn)
            out_beat_cnt <= '0;
        else if (output_forward & rp_s_axis_tvalid & rp_s_axis_tready)
            out_beat_cnt <= out_beat_last ? '0 : out_beat_cnt + 1'b1;
    end

    // --------------------------------------------------------------------------
    // FSM
    // --------------------------------------------------------------------------
    always_ff @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            state          <= S_CHECK_TUSER;
            current_rm_id  <= '0;
            pending_rm_id  <= '0;
            reset_cnt      <= '0;
            tuser_reg      <= '0;
            decouple_prev  <= 1'b0;
            controller_trigger <= '0;
        end else begin
            decouple_prev <= controller_decouple;
            controller_trigger <= '0; // default: no trigger

            case (state)
                // ------------------------------------------------------------------
                // Peek at the first beat of the next frame without consuming it
                // (tready=0 because input_active=0 here). When the upstream master
                // presents tvalid, sample tUSER to decide whether reconfiguration
                // is needed before this frame enters the BDC.
                // ------------------------------------------------------------------
                S_CHECK_TUSER: begin
                    if (s_axis_tvalid) begin
                        pending_rm_id <= s_axis_tuser[RM_ID_W-1:0];
                        tuser_reg     <= s_axis_tuser;
                        if (s_axis_tuser[RM_ID_W-1:0] == current_rm_id)
                            state <= S_INFERENCE;
                        else
                            state <= S_WAIT_FLUSH;
                    end
                end

                // ------------------------------------------------------------------
                S_INFERENCE: begin
                    // After every frame boundary, re-check tUSER for the next frame.
                    if (s_axis_tvalid & s_axis_tready & s_axis_tlast)
                        state <= S_CHECK_TUSER;
                end

                // ------------------------------------------------------------------
                // Drain the pipeline: wait until ALL in-flight frames have exited.
                // The pipeline may be multiple frames deep, so we track the count.
                // Also exits immediately if the pipeline was already empty when we
                // entered (frames_in_flight == 0), which happens when the BDC has
                // already drained all frames by the time the reconfiguration decision
                // is made in S_CHECK_TUSER.
                // ------------------------------------------------------------------
                S_WAIT_FLUSH: begin
                    if ((frames_in_flight == '0) ||
                        (frame_out && (frames_in_flight == {{(INFLIGHT_W-1){1'b0}}, 1'b1}))) begin
                        state <= S_TRIGGER;
                    end
                end

                // ------------------------------------------------------------------
                S_TRIGGER: begin
                    controller_trigger[pending_rm_id] <= 1'b1;
                    state <= S_WAIT_DECOUPLE;
                end

                // ------------------------------------------------------------------
                // Wait for DFX controller to assert decouple (rising edge).
                // ------------------------------------------------------------------
                S_WAIT_DECOUPLE: begin
                    if (controller_decouple & !decouple_prev) begin
                        state <= S_RECONFIGURING;
                    end
                end

                // ------------------------------------------------------------------
                // Reconfiguration in progress. Wait for decouple to fall (done).
                // ------------------------------------------------------------------
                S_RECONFIGURING: begin
                    if (!controller_decouple & decouple_prev) begin
                        current_rm_id <= pending_rm_id;
                        reset_cnt     <= RESET_CYCLES[RESET_CNT_W-1:0];
                        state         <= S_RESET;
                    end
                end

                // ------------------------------------------------------------------
                // Hold BDC in reset for RESET_CYCLES cycles.
                // ------------------------------------------------------------------
                S_RESET: begin
                    if (reset_cnt == '0) begin
                        state <= S_CHECK_TUSER;
                    end else begin
                        reset_cnt <= reset_cnt - 1'b1;
                    end
                end

                default: state <= S_CHECK_TUSER;
            endcase
        end
    end

endmodule
