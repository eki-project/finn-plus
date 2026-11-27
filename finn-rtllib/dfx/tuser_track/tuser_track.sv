// tUSER tracker: remembers the tUSER of every frame inside a wrapped body.
//
// FINN nodes carry no tUSER. A wrapper (dfx_wrapper, sw_wrapper,
// dfx_tuser_passthrough) therefore re-attaches tUSER to the body's output, and
// downstream wrappers read it on the first beat of each output frame. When the
// body holds several frames, the frame leaving it is not the one that entered
// last, so the wrapper must hand out the tUSER of the frame that is leaving.
//
// The tracker is a queue of one tUSER per frame inside the body: written when a
// frame enters (in_frame, on its first accepted beat), advanced when it leaves
// (out_frame, on its last forwarded beat). It is written at the tail and read at
// the head only, so it maps to distributed RAM. out_tuser is the tUSER of the
// oldest frame in the body; with an empty queue it is in_tuser, for a frame that
// enters and leaves in the same cycle.
//
// in_ready is low while DEPTH frames are inside the body. The wrapper then holds
// back the next frame's first beat, which limits the number of frames in flight
// but never loses or mislabels one. The default matches the 255 frames in
// flight the dfx_wrapper's frame counter allows.

`timescale 1ns/1ps
module tuser_track #(
    parameter int TUSER_WIDTH = 2,
    parameter int DEPTH       = 256   // frames that can be inside the body
) (
    input  logic                    aclk,
    input  logic                    aresetn,

    input  logic                    in_frame,   // a frame enters the body this cycle
    input  logic [TUSER_WIDTH-1:0]  in_tuser,   // its tUSER (valid with in_frame)
    output logic                    in_ready,   // another frame may enter

    input  logic                    out_frame,  // a frame leaves the body this cycle
    output logic [TUSER_WIDTH-1:0]  out_tuser   // tUSER of the frame that leaves next
);
    localparam int IDX_W = (DEPTH > 1) ? $clog2(DEPTH) : 1;
    typedef logic [IDX_W-1:0] idx_t;

    (* ram_style = "distributed" *)
    logic [TUSER_WIDTH-1:0] mem [DEPTH];
    idx_t                   head;      // oldest frame
    idx_t                   tail;      // next free entry
    logic [IDX_W:0]         frames;    // frames inside the body

    function automatic idx_t next(idx_t i);
        return (i == idx_t'(DEPTH - 1)) ? '0 : idx_t'(i + 1);
    endfunction

    // With an empty queue, a frame entering and leaving in the same cycle (bypass)
    // changes nothing.
    logic push, pop;
    assign pop  = out_frame && (frames != 0);
    assign push = in_frame && !(out_frame && (frames == 0));

    assign in_ready  = frames < (IDX_W + 1)'(DEPTH);
    assign out_tuser = (frames == 0) ? in_tuser : mem[head];

    always_ff @(posedge aclk) begin
        if (push)  mem[tail] <= in_tuser;
    end

    always_ff @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            head   <= '0;
            tail   <= '0;
            frames <= '0;
        end else begin
            if (push)  tail <= next(tail);
            if (pop)   head <= next(head);
            frames <= frames + push - pop;
        end
    end

endmodule
