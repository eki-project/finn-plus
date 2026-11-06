#include "lsp_receiver.hpp"
#include "lsp_common.hpp"
#include <ap_axi_sdata.h>

#define DEVELOP

using namespace lsp;

namespace lsp {
    namespace receiver {

        // Send a NACK frame roughly every 100 cycles
        constexpr size_t NACK_FRAME_BEAT_INTERVAL = 100;

        using BEAT_TYPE = ap_axiu<constants::BEAT_WIDTH, 1, 0, 0>;

        // Ring buffer depth (in words) of every stream, in the order of the stream arguments
        #ifdef DEVELOP
        using RING_DEPTHS = data::lsp_ring_depths_t<512, 256>;
        #else
        using RING_DEPTHS = data::lsp_ring_depths_t<$$RING_DEPTHS$$>;
        #endif


        void putNACKFrame(hls::stream<BEAT_TYPE>& txNetwork, const ap_uint<constants::FSN_WIDTH>& lastSeenFSN) {
            protocol::lsp_header_t header;
            header.version = lsp::CURRENT_VERSION;
            header._reserved = 0;
            header.frameSequenceNumber = lastSeenFSN;
            header.frameType = constants::FRAME_NACK;
            header.flags = 0;
            header.payloadChunks = 0;

            // TODO: The Ethernet kernel interface will likely still change. This is only
            // a first guess.
            BEAT_TYPE headerBits;
            headerBits.data = header.pack();
            headerBits.last = 1;
            txNetwork.write(headerBits);
        }

        enum class kernel_fsm_state : unsigned char {
            EXPECT_FRAME_FIRST_BEAT,
            EXPECT_BEAT,
            WAITING_FOR_REPLAY
        };

        /** Keeps the state of the receiver kernel. Should be static in the receiver function. */
        struct kernel_state {
            ap_uint<constants::FSN_WIDTH> lastAcknowledgedFSN = 0;
            ap_uint<constants::FSN_WIDTH> currentFSN = 0;
            protocol::lsp_header_t currentHeader;
            kernel_fsm_state state = kernel_fsm_state::EXPECT_FRAME_FIRST_BEAT;
            ap_uint<clog2(NACK_FRAME_BEAT_INTERVAL)+1> beatsSinceLastNACK = 0;
        };


        /**
        Receiver implementation, generic over the number and widths of the FINN streams.

        Stream indices: the index of a FINN stream (the `stream` field of data segments and credit records,
        and the StreamIndex of its ring buffer) is its position among the stream arguments, both here and in
        the sender kernel. The local order of the tensor connections can differ between the two devices:
        if streams X and Y run between devices A and B, and X is tensor connection 0 on A but 1 on B (Y the
        other way round), then the order of the streams is reversed between the HLS sender and receiver
        kernel. The stream arguments of both kernels must therefore be generated in the same per-link
        order, otherwise the receiver dispatches segments into the wrong streams.
        */
        template<typename RingDepths, int... StreamWidths>
        void lsp_receiver_impl(
            hls::stream<BEAT_TYPE>& rxNetwork,
            hls::stream<BEAT_TYPE>& txNetwork,
            hls::stream<ap_uint<StreamWidths>>&... streams
        ) {
            // One ring buffer per FINN stream; std::get<I>(ringBuffers) belongs to the I-th stream argument
            // and has the I-th depth of RingDepths.
            // Static, so that the buffered data survives between calls of the free-running kernel.
            static data::lsp_stream_ring_buffers_t<RingDepths, StreamWidths...> ringBuffers;

            // Segment set - the translation unit between the raw bits of the incoming data and
            // the ring buffers
            static data::lsp_segment_set_t<StreamWidths...> segmentSet;

            // Keeps reciever state, such as FSNs, replay status, etc.
            static kernel_state state;

            // First of all the NACK resend timer.
            // If we wait for a replay, and no frame has arrived yet, we send a NACK.
            if (state.state == kernel_fsm_state::WAITING_FOR_REPLAY && rxNetwork.empty()) {
                if (state.beatsSinceLastNACK == NACK_FRAME_BEAT_INTERVAL - 1) {
                    state.beatsSinceLastNACK = 0;
                    putNACKFrame(txNetwork, state.currentFSN);
                }
                state.beatsSinceLastNACK++;
                return;
            }

            // Read a beat
            BEAT_TYPE inBeat = rxNetwork.read();
            ap_uint<BEAT_TYPE::Type_data::width> data = inBeat.get_data();

            // If the frame is bad, send back NACK.
            if (inBeat.last && inBeat.user) {
                state.state = kernel_fsm_state::WAITING_FOR_REPLAY;
                putNACKFrame(txNetwork, state.currentFSN);
                state.beatsSinceLastNACK = 0;
                return;
            }

            // Check tag/ethertype for ethernet header length
            ap_uint<clog2(MAX_ETHERNET_HEADER_LENGTH_BYTES)+1> ethernetHeaderLength = MAX_ETHERNET_HEADER_LENGTH_BYTES;
            ap_uint<16> tagOrEthertype = data.range(13 * 8 - 1, 12 * 8);
            if (tagOrEthertype == 0x8100) {
                // IEEE 802.1Q
                ethernetHeaderLength = 18;
            } else if (tagOrEthertype == 0x88A8) {
                // IEEE 802.1ad
                ethernetHeaderLength = 22;
            } else if (tagOrEthertype == 0x88B5 || tagOrEthertype == 0x88B6) {
                // Local experiment ethertypes
                ethernetHeaderLength = 14;
            } else {
                // TODO: Error counter - unexpected ethertype
                ethernetHeaderLength = 14;
            }

            // Check if we are expecting the first beat of a frame and need to account for the ethernet
            // and protocol header
            ap_uint<constants::BEAT_WIDTH> beatOffset = 0;
            if (state.state == kernel_fsm_state::EXPECT_FRAME_FIRST_BEAT) {
                // We expect a protocol header. Read it, unpack it and adjust the read offset.
                beatOffset = ethernetHeaderLength;
                state.currentHeader = protocol::lsp_header_t::unpack(
                    data.range(beatOffset + protocol::lsp_header_t::WIDTH - 1, beatOffset)
                );
                beatOffset += protocol::lsp_header_t::WIDTH;
            }

            // Check if the sequence number is correct. If not, we start sending NACK frames.
            // Assignment here on purpose so that it wraps around to 0 correctly
            auto expectedFSN = state.currentFSN + 1;
            if (state.currentHeader.frameSequenceNumber == expectedFSN) {
                // Looks good, we increase
                state.currentFSN++;
            } else {
                // Start sending NACKs until we receive the correct FSN. Go back to the last one seen
                putNACKFrame(txNetwork, state.currentFSN);
                state.beatsSinceLastNACK = 0;
                state.state = kernel_fsm_state::WAITING_FOR_REPLAY;
            }

            // Decide what to do, depending on the incoming frame type and kernel state
            if (state.currentHeader.frameType == constants::FRAME_DATA) {
                while (beatOffset <= constants::BEAT_WIDTH) {
                    // TODO: Continuously read from the segments into ring buffer
                    // TODO: Assure that ethernet is padded to 64bit multiple so that we can use chunk_offset_t
                    segmentSet.set(data, beatOffset);
                    // TODO: Read the segment header to find out how long the segment was and adjust the beatOffset accordingly
                }
            } else {
                // Unexpected - we normally only receive data frames
            }

            // Check if credit updates are necessary
            // TODO

            // If this was the last beat in the frame, we expect a new frame
            if (inBeat.last) {
                state.state = kernel_fsm_state::EXPECT_FRAME_FIRST_BEAT;
            }
        }


        /**
        * Top Level Function
        */
        void lsp_receiver(
            hls::stream<BEAT_TYPE>& networkStream,
        #ifdef DEVELOP
            hls::stream<ap_uint<64>>& stream0,
            hls::stream<ap_uint<264>>& stream1
        #else
            $$STREAMS$$
        #endif
        ) {
        #pragma HLS INTERFACE axis port = net_in

        #ifdef DEVELOP
        #pragma HLS INTERFACE axis port = stream0
        #pragma HLS INTERFACE axis port = stream1
        #else
        $$STREAM_PRAGMAS$$
        #endif
        #pragma HLS INTERFACE ap_ctrl_none port = return

            // The order of the streams sets their indices, see lsp_receiver_impl
        #ifdef DEVELOP
            lsp_receiver_impl<RING_DEPTHS>(networkStream, stream0, stream1);
        #else
            lsp_receiver_impl<RING_DEPTHS>(networkStream, $$STREAM_ARGS$$);
        #endif
        }
    }
}
