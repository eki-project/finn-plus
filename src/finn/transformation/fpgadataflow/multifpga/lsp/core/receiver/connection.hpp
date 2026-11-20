// RX connection of the LSP receiver, one per peer: parses its frames, fills the stream rings and reports CTRL status.
#ifndef LSP_RECEIVER_CONNECTION_HPP
#define LSP_RECEIVER_CONNECTION_HPP

#include "common.hpp"
#include "constants.hpp"
#include "utils.hpp"
#include "ring_buffer.hpp"
#include <cerrno>

namespace lsp {
    namespace receiver {
        using namespace messages;
        using namespace helpers;
        using namespace constants;
        using namespace utils;


        // Parser state, kept across beats
        enum class connection_state : unsigned char {
            EXPECT_FRAME_START,
            EXPECT_BEAT,
            DISCARDING_INVALID_FRAME,
            EXPECT_FRAME_START_AFTER_INVALID_FRAME
        };

        // Drop the whole frame like a lost one: roll back its segments and discard its remaining beats
        template <typename... Rings>
        connection_state _dropFrame(beat_t& data, std::tuple<Rings...>& buffers) {
            #pragma HLS inline
            ring_buffer::rollbackAll(buffers);
            if (data.last) {
                return connection_state::EXPECT_FRAME_START_AFTER_INVALID_FRAME;
            } else {
                return connection_state::DISCARDING_INVALID_FRAME;
            }
        }

        /** Parse the segments of one beat into the rings and return the next parser state; a malformed frame is
        dropped whole. In a PROBE frame (isProbe) every entry must be a BLOCKED report. */
        template <int... Ws, size_t... Ds>
        connection_state _readDataSegments(
            stream_widths_t<Ws...>,
            ring_depths_t<Ds...>,
            ap_uint<clog2(1 + constants::BEAT_WIDTH)> initialOffset,
            beat_t& data,
            bool isProbe,
            ctrl_status_t<sizeof...(Ws)>& creditUpdate,
            ring_buffer::stream_ring_buffers_t<ring_buffer::lsp_ring_depths_t<Ds...>, Ws...> &buffers,
            ap_uint<header_t::PAYLOAD_WIDTH>& frameChunksLeft,
            hls::stream<ctrl_status_t<sizeof...(Ws)>>& controlStream
        ) {
            #pragma HLS inline
            data_segment_t<constants::MAX_SEGMENT_PAYLOAD_WIDTH> segment;
            chunk_offset_t chunkOffset{initialOffset / 64};
            // Upper bound: every segment takes at least one chunk
            for (unsigned int i = 0; i < constants::CHUNKS_PER_BEAT; ++i) {
#pragma HLS PIPELINE II = 1
                segment = decltype(segment)::unpackMaxSizeSegment(data.data, chunkOffset);

                // Malformed: the segment crosses the beat boundary or exceeds payloadChunks
                if (segment.chunks > constants::CHUNKS_PER_BEAT - 1 - chunkOffset.index ||
                    segment.chunks + SEGMENT_HEADER_CHUNKS > frameChunksLeft) {
                    return _dropFrame(data, buffers);
                }

                // Skip the segment header
                chunkOffset.index += SEGMENT_HEADER_CHUNKS;
                frameChunksLeft -= SEGMENT_HEADER_CHUNKS;

                // 0 chunks: BLOCKED report, answer with the stream's limit at once. Otherwise: write the payload to its ring
                if (segment.chunks == 0) {
                    // An unknown stream adds no credit record, but the ACK is still due
                    creditUpdate.takeNewer(
                        ctrl_status_t<sizeof...(Ws)>::template makeACK<true>(creditUpdate.expectedFSN, segment.stream, buffers)
                    );
                } else if (isProbe) {
                    // Malformed: a PROBE entry with payload
                    return _dropFrame(data, buffers);
                } else {
                    switch (ring_buffer::dispatchSegment(buffers, segment)) {
                        case ring_buffer::lsp_stream_ring_buffer_update_result::SEGMENT_READ:
                            break;
                        // The sender's fault, retransmitting would not help: skip the segment
                        case ring_buffer::lsp_stream_ring_buffer_update_result::UNKNOWN_STREAM:
                            break;
                        // Invalid validBytesLastChunk, payload too wide, or no room: drop, so that no stream loses data
                        case ring_buffer::lsp_stream_ring_buffer_update_result::INVALID_CHUNK_BYTEMASK:
                        case ring_buffer::lsp_stream_ring_buffer_update_result::UNEXPECTED_ERROR:
                        case ring_buffer::lsp_stream_ring_buffer_update_result::BUFFER_FULL:
                        case ring_buffer::lsp_stream_ring_buffer_update_result::SENDER_BLOCKED:
                            return _dropFrame(data, buffers);
                    }
                    chunkOffset.index += segment.chunks;
                    frameChunksLeft -= segment.chunks;
                }

                // End of the entries or of the beat
                const bool frameDone = frameChunksLeft == 0;
                if (frameDone || chunkOffset.index >= constants::CHUNKS_PER_BEAT) {
                    // Malformed: the entries end before the last beat, or the last beat comes before they end
                    if (frameDone != bool(data.last)) {
                        return _dropFrame(data, buffers);
                    }

                    // Frame complete: send the BLOCKED answers collected in it, then reset them
                    if (frameDone) {
                        if (creditUpdate.ackImmediately) {
                            controlStream.write_nb(creditUpdate);
                        }
                        creditUpdate.changed = 0;
                        creditUpdate.ackImmediately = false;
                        return connection_state::EXPECT_FRAME_START;
                    } else {
                        return connection_state::EXPECT_BEAT;
                    }
                }
            }
            // Not reached: every segment takes at least one chunk, so the loop returns by the end of the beat
            return connection_state::EXPECT_BEAT;
        }

        /** Bit offset of the LSP header in beat 0, from the TPID or EtherType of the L2 header. */
        inline ap_uint<clog2(1 + BEAT_WIDTH)> _getOffsetFromEthertype(beat_t &data) {
            #pragma HLS inline
            // Read in wire byte order, so the constants are byte swapped: 0x0081 = 0x8100, 0xA888 = 0x88A8, 0x0091 = 0x9100
            ap_uint<16> ethertypeOrTag = data.data.range(111, 96);
            if (ethertypeOrTag == 0x0081) {
                return 192; // 144 padded to 64b multiples
            } else if (ethertypeOrTag == 0xA888 || ethertypeOrTag == 0x0091) {
                return 192; // 176 padded to 64b multiples
            } else {
                return 128; // 112 padded to 64b multiples
            }
        }

        // TODO: Set NACK interval timeout
        // TODO: write_nb drops the status if controlStream is full; keep one pending status, merge with takeNewer, retry every call
        // TODO: Send the gap NACK and duplicate ACK only after a good FCS at tlast, since a corrupt header may hold a false FSN
        // TODO: Make the L2 header offset a build-time constant instead of reading it from every frame

        /** One call handles at most one beat. ConnId must be unique per connection: the state lives in
        statics of the instantiation, so two connections with equal widths and depths would share it. */
        template <size_t ConnId, int... Ws, size_t... Ds>
        inline void connection(
            stream_widths_t<Ws...> widths,
            ring_depths_t<Ds...> depths,
            hls::stream<beat_t>& macStream,
            hls::stream<ctrl_status_t<sizeof...(Ws)>>& controlStream,
            hls::stream<ap_uint<Ws>>&... outStreams
        ) {
#pragma HLS pipeline II=1
            using ctrl_status_t = ctrl_status_t<sizeof...(Ws)>;

            // State kept across calls
            static connection_state state = connection_state::EXPECT_FRAME_START;
            static ap_uint<constants::FSN_WIDTH> expectedFSN = 0;
            static ring_buffer::stream_ring_buffers_t<ring_buffer::lsp_ring_depths_t<Ds...>, Ws...> buffers;
            static ap_uint<header_t::PAYLOAD_WIDTH> frameChunksLeft;
            static header_t header;

            // BLOCKED answers collected over the current frame, sent at once when it ends
            static ctrl_status_t creditUpdate;

            // Try to empty streams
            ring_buffer::tryReadIntoStream(buffers, outStreams...);

            // Credit check on every call, beat or not: report the limits of all streams whose credit update is due
            auto currentCreditStatus = ctrl_status_t::template makeACK<false>(expectedFSN, buffers);
            if (currentCreditStatus.changed != 0) {
                controlStream.write_nb(currentCreditStatus);
            }

            // Non-blocking, so the credit check above also runs while no frame arrives
            beat_t data;
            if (!macStream.read_nb(data)) {
                return;
            }
            creditUpdate.expectedFSN = expectedFSN;

            // Discard the remaining beats of a dropped frame
            if (state == connection_state::DISCARDING_INVALID_FRAME) {
                state = data.last ? connection_state::EXPECT_FRAME_START : state;
                return;
            }

            // Bad FCS, flagged in tuser on the last beat: roll back the frame without parsing this beat
            if (data.last && data.user == 1) {
                ring_buffer::rollbackAll(buffers);
                state = connection_state::EXPECT_FRAME_START;
                controlStream.write_nb(ctrl_status_t::makeNACKImmediately(expectedFSN));
                return;
            }

            // Beat 0: find and check the LSP header
            ap_uint<clog2(1 + constants::BEAT_WIDTH)> offset = 0;
            if (state == connection_state::EXPECT_FRAME_START || state == connection_state::EXPECT_FRAME_START_AFTER_INVALID_FRAME) {
                offset = _getOffsetFromEthertype(data);

                // Read header
                header = header_t::unpack(data.data >> offset);
                frameChunksLeft = header.payloadChunks;
                offset += header_t::WIDTH;

                // Neither DATA nor PROBE: discard. An unknown frame type is NACKed at once, since the frame will be
                // replayed anyway
                if (header.frameType != constants::FRAME_DATA && header.frameType != FRAME_PROBE) {
                    state = !data.last ? connection_state::DISCARDING_INVALID_FRAME : state;
                    if (header.frameType != FRAME_CREDIT_ACK && header.frameType != FRAME_NACK) {
                        controlStream.write_nb(ctrl_status_t::makeNACKImmediately(expectedFSN));
                    }
                    return;
                }

                // FSN after the expected one: gap, NACK. Before it: duplicate, ACK
                if (helpers::serialBefore(expectedFSN, header.frameSequenceNumber)) {
                    state = data.last ? connection_state::EXPECT_FRAME_START : connection_state::DISCARDING_INVALID_FRAME;
                    controlStream.write_nb(ctrl_status_t::makeNACKImmediately(expectedFSN));
                    return;
                } else if (helpers::serialBefore(header.frameSequenceNumber, expectedFSN)) {
                    // On a PROBE, a too small FSN is okay - we send the response regardless
                    if (header.frameType != FRAME_PROBE) {
                        state = !data.last ? connection_state::DISCARDING_INVALID_FRAME : state;
                        controlStream.write_nb(ctrl_status_t::makeACKImmediately(expectedFSN));
                        return;
                    }
                }
            }

            // A zero entry PROBE gets answered immedaitely
            if (header.frameType == FRAME_PROBE && header.payloadChunks == 0) {
                controlStream.write_nb(ctrl_status_t::makeACKImmediately(expectedFSN));
            }

            // Parse the segments in the rest of the beat
            if (
                state == connection_state::EXPECT_BEAT
                || state == connection_state::EXPECT_FRAME_START
                || state == connection_state::EXPECT_FRAME_START_AFTER_INVALID_FRAME
            ) {
                const bool isProbe = header.frameType == FRAME_PROBE;
                state = _readDataSegments(
                    widths, depths, offset, data, isProbe, creditUpdate, buffers, frameChunksLeft, controlStream
                );
                const bool dropped = state == connection_state::DISCARDING_INVALID_FRAME
                    || state == connection_state::EXPECT_FRAME_START_AFTER_INVALID_FRAME;
                if (dropped && !isProbe) {
                    // A malformed DATA frame will be replayed anyway: NACK it at once instead of waiting for the
                    // next frame to show the gap. A dropped PROBE gets no answer, the sender repeats it
                    controlStream.write_nb(ctrl_status_t::makeNACKImmediately(expectedFSN));
                } else if (state == connection_state::EXPECT_FRAME_START && !isProbe) {
                    expectedFSN++;
                    controlStream.write_nb(ctrl_status_t::makeFSNUpdate(expectedFSN));
                    ring_buffer::commitAll(buffers);
                }
            }

        }
    } // namespace receiver
} // namespace lsp
#endif // LSP_RECEIVER_CONNECTION_HPP
