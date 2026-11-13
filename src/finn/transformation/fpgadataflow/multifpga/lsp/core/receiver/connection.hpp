// RX connection of the LSP receiver kernel: one per peer. Parses the frames the manager forwards,
// feeds its output streams and reports its CTRL status to the controller_tx.
#ifndef LSP_RECEIVER_CONNECTION_HPP
#define LSP_RECEIVER_CONNECTION_HPP

#include "common.hpp"
#include "utils.hpp"

namespace lsp {
    namespace receiver {
        template<int... Ws> struct stream_widths_t {};
        template<size_t... Ds> struct ring_depths_t {};

        // Same state logic as the manager
        enum class connection_state : unsigned char {
            EXPECT_FRAME_START,
            EXPECT_BEAT,
            DISCARDING_INVALID_FRAME
        };

        // TODO: Make the L2 ethertype comptime
        // TODO: connection(macStream, ringBuffer, ctrlStatus, outStream)
        template<int... Ws, size_t... Ds>
        inline void connection(
            stream_widths_t<Ws...>,
            ring_depths_t<Ds...>,
            hls::stream<beat_t>& macStream,
            hls::stream<ctrl_status_t<sizeof...(Ws)>>& controlStream
        ) {
            using namespace protocol;
            using namespace data;
            using ctrl_status_t = ctrl_status_t<sizeof...(Ws)>;

            // Stateful variables
            static connection_state state = connection_state::EXPECT_FRAME_START;
            static ap_uint<constants::FSN_WIDTH> expectedFSN = 0;
            static lsp_stream_ring_buffers_t<lsp_ring_depths_t<Ds...>, Ws...> buffers;
            static ap_uint<lsp_header_t::PAYLOAD_WIDTH> frameChunksLeft;

            // TODO: controlStream.write_nb below drops the status if the stream is full. Keep one
            // pending status instead, merge every new one into it with takeNewer, and retry write_nb
            // on every call until it succeeds (spec 9.5). Worst case today: a dropped BLOCKED status,
            // since creditUpdate() has already advanced lastCreditLimitSent, so that credit is never
            // sent again until the next half-ring threshold.

            // Declare as a function here since its reused at multiple points
            auto _readDataSegments = [&](
                ap_uint<clog2(1 + constants::BEAT_WIDTH)> initialOffset,
                beat_t& data
            ) -> connection_state {
                lsp_data_segment_t<constants::MAX_SEGMENT_PAYLOAD_WIDTH> segment;
                chunk_offset_t chunkOffset{initialOffset/64};
                for (unsigned int i = 0; i < constants::CHUNKS_PER_BEAT; ++i) {
                    #pragma HLS PIPELINE II=1
                    segment = unpackMaxSizeSegment(data.data, chunkOffset);

                    // A segment never crosses the beat: more chunks than the rest
                    // of the beat make the frame malformed. Checked before advancing, which also
                    // keeps chunkOffset.index from wrapping on a corrupt chunks field
                    if (segment.chunks > constants::CHUNKS_PER_BEAT - 1 - chunkOffset.index) {
                        // Drop the whole frame, handled like a lost one (spec 5.3)
                        // TODO (#69): roll back the segments of this frame already in the rings
                        // TODO: Returning EXPECT_FRAME_START here (malformed in the last beat) looks
                        // to the caller like a finished frame, so it increments expectedFSN and
                        // ACKs. The sender then frees the frame and the rest of its data is lost.
                        // Return a distinct result (e.g. FRAME_DONE / MALFORMED / NEED_BEAT) and let
                        // the caller drop the frame without FSN++ or ACK
                        if (data.last) {
                            return connection_state::EXPECT_FRAME_START;
                        } else {
                            return connection_state::DISCARDING_INVALID_FRAME;
                        }
                    }
                    if (frameChunksLeft == 0) {
                        return connection_state::EXPECT_FRAME_START;
                    }

                    // Since we have read a header we can definitely already advance the counters
                    // by the number of chunks required by a segment header.
                    chunkOffset.index += (constants::SEGMENT_HEADER_WIDTH / constants::CHUNK_SIZE_BITS);
                    frameChunksLeft -= (constants::SEGMENT_HEADER_WIDTH / constants::CHUNK_SIZE_BITS);

                    // Normal DATA or BLOCKED?
                    if (segment.chunks == 0) {
                        // BLOCKED
                        controlStream.write_nb(
                            ctrl_status_t::makeCreditUpdateStatus(
                                expectedFSN,
                                segment.stream,
                                buffers
                            )
                        );
                    } else {
                        // DATA
                        dispatchSegment(buffers, segment);
                        chunkOffset.index += segment.chunks;
                        frameChunksLeft -= segment.chunks;
                    }
                    if (frameChunksLeft == 0) {
                        return connection_state::EXPECT_FRAME_START;
                    }
                    if (chunkOffset.index >= constants::CHUNKS_PER_BEAT) {
                        if (data.last) {
                            return connection_state::EXPECT_FRAME_START;
                        } else {
                            return connection_state::EXPECT_BEAT;
                        }
                    }
                }
                // If we hit no return until now, we expect another beat
                return connection_state::EXPECT_BEAT;
            };

            // TODO: No FCS check yet. The CMAC flags a bad FCS in tuser on the last beat
            // but both accept paths below increment expectedFSN and ACK at data.last without
            // looking at data.user. A corrupted frame is therefore accepted and its data stays in
            // the rings. At data.last with data.user set: drop the frame silently, without FSN++ or
            // ACK and roll back its segments. This also contains garbage parsed
            // from a corrupt header, e.g. a wrapped frameChunksLeft.

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
                frameChunksLeft = header.payloadChunks;
                offset += lsp_header_t::WIDTH;

                // We cannot receive anything other than DATA in this kernel.
                // Having a device act both as receiver and sender requires
                // differentiation by frame type on the top level, not here
                if (header.frameType != constants::FRAME_DATA) {
                    if (!data.last) {
                        state = connection_state::DISCARDING_INVALID_FRAME;
                    }
                    return;
                }

                // If we get a probe (Sender: "please send an ACK/CREDIT update"), we do so
                if (header.payloadChunks == 0) {
                    if (!data.last) {
                        state = connection_state::DISCARDING_INVALID_FRAME;
                    }
                    controlStream.write_nb(ctrl_status_t::makeACKNowStatus(expectedFSN));
                }

                // Check FSN
                if (helpers::serialBefore(expectedFSN, header.frameSequenceNumber)) {
                    // We must have missed a beat.
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
                    controlStream.write_nb(ctrl_status_t::makeNACKStatus(expectedFSN));
                    // TODO: Set NACK interval timeout
                } else if (helpers::serialBefore(header.frameSequenceNumber, expectedFSN)) {
                    // Observe a replay frame that we don't need - so we simply drop it
                    // until tlast is seen as well
                    if (!data.last) {
                        state = connection_state::DISCARDING_INVALID_FRAME;
                    }
                    controlStream.write_nb(ctrl_status_t::makeACKNowStatus(expectedFSN));
                } else if (header.frameSequenceNumber == expectedFSN) {
                    state = _readDataSegments(offset, data);
                    if (state == connection_state::EXPECT_FRAME_START) {
                        expectedFSN++;
                        controlStream.write_nb(ctrl_status_t::makeACKStatus(expectedFSN));
                    }
                }
            } else if (state == connection_state::EXPECT_BEAT) {
                state = _readDataSegments(0, data);
                if (state == connection_state::EXPECT_FRAME_START) {
                    expectedFSN++;
                    controlStream.write_nb(ctrl_status_t::makeACKStatus(expectedFSN));
                }
            } else if (state == connection_state::DISCARDING_INVALID_FRAME) {
                // Repeatedly discard until a new frame arrives
                if (data.last) {
                    state = connection_state::EXPECT_FRAME_START;
                }
            }

            // TODO: Check if a ring buffer update is necessary
        }
    }
}
#endif  // LSP_RECEIVER_CONNECTION_HPP
