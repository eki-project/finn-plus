// RX connection of the LSP receiver kernel: one per peer. Parses the frames the manager forwards,
// feeds its output streams and reports its CTRL status to the controller_tx.
#ifndef LSP_RECEIVER_CONNECTION_HPP
#define LSP_RECEIVER_CONNECTION_HPP

#include "lsp_common.hpp"
#include "lsp_receiver_utils.hpp"

namespace lsp {
    namespace receiver {
        // Same state logic as the manager
        enum class connection_state : unsigned char {
            EXPECT_FRAME_START,
            EXPECT_BEAT,
            DISCARDING_INVALID_FRAME
        };

        // TODO: Make the L2 ethertype comptime
        // TODO: connection(macStream, ringBuffer, ctrlStatus, outStream)
        inline void connection(hls::stream<beat_t>& macStream) {
            using namespace protocol;

            // Stateful variables
            static connection_state state = connection_state::EXPECT_FRAME_START;
            static ap_uint<constants::FSN_WIDTH> expectedFSN = 0;

            // Read a beat from stream
            beat_t data = macStream.read();
            if (state == connection_state::EXPECT_FRAME_START) {
                ap_uint<clog2(1 + constants::BEAT_WIDTH)> offset = 0;

                // Check for tag/ethertype. The field is read in wire byte order, so the constants
                // are byte swapped: 0x0081 = TPID 0x8100, 0xA888 = 0x88A8, 0x0091 = 0x9100
                ap_uint<16> ethertypeOrTag = data.data.range(111, 96);
                if (ethertypeOrTag == 0x0081) {
                    offset = 192; // 144 padded to 64b multiples
                } else if (ethertypeOrTag == 0xA888 || ethertypeOrTag == 0x0091) {
                    offset = 192; // 176 padded to 64b multiples
                } else {
                    offset = 128; // 112 padded to 64b multiples
                }

                // Read LSP header
                // TODO: Increase error counter if frameType != DATA.
                auto header = lsp_header_t::unpack(data.data >> offset);
                offset += lsp_header_t::WIDTH;

                // Check FSN
                if (header.frameSequenceNumber > expectedFSN) {
                    // Immediately send a NACK request and drop the entire frame.
                    // If this frame was only one beat long we can immediately expect a new one.
                    // Otherwise drop beats until tlast was seen, then expect a new frame.
                    // If the new frame is not the replay frame, we repeat the resend request
                    // - newer replay requests dont change, so we can safely re-send
                    if (data.last) {
                        state = connection_state::EXPECT_FRAME_START;
                    } else {
                        state = connection_state::DISCARDING_INVALID_FRAME;
                    }
                    // TODO: Send NACK request here
                    // TODO: Set NACK interval timeout
                } else if (header.frameSequenceNumber < expectedFSN) {
                    // Observe a replay frame that we don't need - so we simply drop it
                    // until tlast is seen as well
                    state = connection_state::DISCARDING_INVALID_FRAME;
                } else if (header.frameSequenceNumber == expectedFSN) {
                    state = connection_state::EXPECT_BEAT;
                    // TODO: Read DATA segments
                }

            } else if (state == connection_state::EXPECT_BEAT) {
                // TODO: Read DATA segments
                if (data.last) {
                    state = connection_state::EXPECT_FRAME_START;
                    // TODO: Wrap instead of increase
                    expectedFSN++;
                }
            } else if (state == connection_state::DISCARDING_INVALID_FRAME) {
                // Repeatedly discard unitl a new frame arrives
                if (data.last) {
                    state = connection_state::EXPECT_FRAME_START;
                }
            }
        }
    }
}
#endif  // LSP_RECEIVER_CONNECTION_HPP
