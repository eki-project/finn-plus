// RX connection of the LSP receiver kernel: one per peer. Parses the frames the manager forwards,
// feeds its output streams and reports its CTRL status to the controller_tx.
#ifndef LSP_RECEIVER_CONNECTION_HPP
#define LSP_RECEIVER_CONNECTION_HPP

#include "common.hpp"
#include "utils.hpp"
#include "ring_buffer.hpp"

namespace lsp {
    namespace receiver {

        // Same state logic as the manager
        enum class connection_state : unsigned char {
            EXPECT_FRAME_START,
            EXPECT_BEAT,
            DISCARDING_INVALID_FRAME,
            EXPECT_FRAME_START_AFTER_INVALID_FRAME
        };

        // TODO: Make the L2 ethertype comptime
        // TODO: connection(macStream, ringBuffer, ctrlStatus, outStream)
        template<int... Ws> struct stream_widths_t {};
        template<size_t... Ds> struct ring_depths_t {};

        template<int... Ws, size_t... Ds>
        inline void connection(
            stream_widths_t<Ws...>,
            ring_depths_t<Ds...>,
            hls::stream<beat_t>& macStream,
            hls::stream<ctrl_status_t<sizeof...(Ws)>>& controlStream
        ) {
            using namespace messages;
            using namespace helpers;
            using ctrl_status_t = ctrl_status_t<sizeof...(Ws)>;

            // Stateful variables
            static connection_state state = connection_state::EXPECT_FRAME_START;
            static ap_uint<constants::FSN_WIDTH> expectedFSN = 0;
            static ring_buffer::stream_ring_buffers_t<ring_buffer::lsp_ring_depths_t<Ds...>, Ws...> buffers;
            static ap_uint<header_t::PAYLOAD_WIDTH> frameChunksLeft;
            // A NACK was sent for expectedFSN and the gap is still open. The controller_tx replaces its
            // gapOpen with the one of every newer status, so every status has to carry this value
            static bool gapOpen = false;

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
                constexpr int SEGMENT_HEADER_CHUNKS = constants::SEGMENT_HEADER_WIDTH / constants::CHUNK_SIZE_BITS;
                data_segment_t<constants::MAX_SEGMENT_PAYLOAD_WIDTH> segment;
                chunk_offset_t chunkOffset{initialOffset/64};

                // Drop the whole frame, handled like a lost one: roll back its segments, and let the
                // caller skip FSN++ and ACK
                auto _dropFrame = [&]() -> connection_state {
                    ring_buffer::rollbackAll(buffers);
                    if (data.last) {
                        return connection_state::EXPECT_FRAME_START_AFTER_INVALID_FRAME;
                    } else {
                        return connection_state::DISCARDING_INVALID_FRAME;
                    }
                };

                for (unsigned int i = 0; i < constants::CHUNKS_PER_BEAT; ++i) {
                    #pragma HLS PIPELINE II=1
                    segment = decltype(segment)::unpackMaxSizeSegment(data.data, chunkOffset);

                    // Malformed segment or chunk count. Checked before advancing, which also keeps
                    // chunkOffset.index and frameChunksLeft from wrapping on a corrupt chunks field.
                    // Cause 1: The segment goes over the beat boundary.
                    // Cause 2: The segment has more chunks than the frame's payloadChunks has left.
                    if (
                        segment.chunks > constants::CHUNKS_PER_BEAT - 1 - chunkOffset.index
                        || segment.chunks + SEGMENT_HEADER_CHUNKS > frameChunksLeft
                    ) {
                        return _dropFrame();
                    }

                    // Since we have read a header we can definitely already advance the counters
                    // by the number of chunks required by a segment header.
                    chunkOffset.index += SEGMENT_HEADER_CHUNKS;
                    frameChunksLeft -= SEGMENT_HEADER_CHUNKS;

                    // Normal DATA or BLOCKED?
                    if (segment.chunks == 0) {
                        // BLOCKED
                        controlStream.write_nb(
                            ctrl_status_t::makeCreditUpdateStatus(
                                expectedFSN,
                                gapOpen,
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

                    // The frame must end exactly with its last beat. Only then is the frame done,
                    // and since the FCS of the last beat was checked before parsing, it can be
                    // committed.
                    const bool frameDone = frameChunksLeft == 0;
                    if (frameDone || chunkOffset.index >= constants::CHUNKS_PER_BEAT) {
                        if (frameDone != bool(data.last)) {
                            // The frame ends before its last beat, or its last beat comes before
                            // the frame ends
                            return _dropFrame();
                        }
                        if (frameDone) {
                            return connection_state::EXPECT_FRAME_START;
                        } else {
                            return connection_state::EXPECT_BEAT;
                        }
                    }
                }
                // If we hit no return until now, we expect another beat
                return connection_state::EXPECT_BEAT;
            };


            // Send credit updates for streams where it is necessary. Runs on every call, also when no
            // beat arrives, since the consumer frees space independently of incoming frames. The
            // status carries the current FSN state, because the controller_tx takes it over
            ctrl_status_t creditUpdate;
            creditUpdate.expectedFSN = expectedFSN;
            creditUpdate.gapOpen = gapOpen;
            for (size_t index = 0; index < sizeof...(Ws); index++) {
                #pragma HLS unroll
                if (ring_buffer::creditUpdateDue(buffers, index)) {
                    creditUpdate.changed[index] = 1;
                    creditUpdate.limit[index] = ring_buffer::creditUpdate(buffers, index);
                }
            }
            if (creditUpdate.changed != 0) {
                controlStream.write_nb(creditUpdate);
            }

            // Read a beat from stream. Non-blocking, so that the credit check above keeps running
            // while no frames arrive
            beat_t data;
            if (!macStream.read_nb(data)) {
                return;
            }

            // The CMAC flags a bad FCS in tuser on the last beat. Drop the frame silently (no FSN++
            // or ACK) and roll back its segments. Checked before parsing, so the last beat is never
            // written. The frame ends with this beat, so the next beat starts a new frame.
            if (data.last && data.user == 1) {
                ring_buffer::rollbackAll(buffers);
                state = connection_state::EXPECT_FRAME_START;
                return;
            }

            // Handle frame reader FSM
            if (state == connection_state::EXPECT_FRAME_START || state == connection_state::EXPECT_FRAME_START_AFTER_INVALID_FRAME) {
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
                auto header = header_t::unpack(data.data >> offset);
                frameChunksLeft = header.payloadChunks;
                offset += header_t::WIDTH;

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
                    controlStream.write_nb(ctrl_status_t::makeACKNowStatus(expectedFSN, gapOpen));
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
                    gapOpen = true;
                    controlStream.write_nb(ctrl_status_t::makeNACKStatus(expectedFSN));
                    // TODO: Set NACK interval timeout
                } else if (helpers::serialBefore(header.frameSequenceNumber, expectedFSN)) {
                    // Observe a replay frame that we don't need - so we simply drop it
                    // until tlast is seen as well
                    if (!data.last) {
                        state = connection_state::DISCARDING_INVALID_FRAME;
                    }
                    controlStream.write_nb(ctrl_status_t::makeACKNowStatus(expectedFSN, gapOpen));
                } else if (header.frameSequenceNumber == expectedFSN) {
                    state = _readDataSegments(offset, data);
                    if (state == connection_state::EXPECT_FRAME_START) {
                        // The expected frame arrived, so a gap at it is closed
                        gapOpen = false;
                        expectedFSN++;
                        controlStream.write_nb(ctrl_status_t::makeACKStatus(expectedFSN));
                        ring_buffer::commitAll(buffers);
                    }
                }
            } else if (state == connection_state::EXPECT_BEAT) {
                state = _readDataSegments(0, data);
                if (state == connection_state::EXPECT_FRAME_START) {
                    // The expected frame arrived, so a gap at it is closed
                    gapOpen = false;
                    expectedFSN++;
                    controlStream.write_nb(ctrl_status_t::makeACKStatus(expectedFSN));
                    ring_buffer::commitAll(buffers);
                }
            } else if (state == connection_state::DISCARDING_INVALID_FRAME) {
                // Repeatedly discard until a new frame arrives
                if (data.last) {
                    state = connection_state::EXPECT_FRAME_START;
                }
            }

        }
    }
}
#endif  // LSP_RECEIVER_CONNECTION_HPP
