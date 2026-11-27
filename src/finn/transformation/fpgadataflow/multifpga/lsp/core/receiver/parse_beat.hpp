// RX connection of the LSP receiver, one per peer: parses its frames, fills the stream rings and reports CTRL status.
#ifndef LSP_RECEIVER_PARSE_BEAT_HPP
#define LSP_RECEIVER_PARSE_BEAT_HPP

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
        using namespace ring_buffer;


        // Parser state, kept across beats
        enum class connection_state : unsigned char {
            EXPECT_FRAME_START,
            EXPECT_FRAME_START_AFTER_INVALID_FRAME,
            EXPECT_BEAT,
            DISCARDING_INVALID_FRAME,
        };

        template<int... Ws, size_t... Ds>
        connection_state _readProbeEntries(
            stream_widths_t<Ws...>,
            ring_depths_t<Ds...>,
            ap_uint<clog2(1 + constants::BEAT_WIDTH)> initialOffset,
            beat_t& data,
            ctrl_status_t<sizeof...(Ws)>& creditUpdate,
            ring_buffers_t<ring_depths_t<Ds...>, stream_widths_t<Ws...>> &buffers,
            ap_uint<header_t::PAYLOAD_WIDTH>& frameChunksLeft,
            pending_ctrl_status_t<sizeof...(Ws)>& ctrlStatus
        ) {
#pragma HLS inline
            data_segment_t<constants::MAX_SEGMENT_PAYLOAD_WIDTH> segment;
            chunk_offset_t chunkOffset{initialOffset / 64};
            for (unsigned int i = 0; i < constants::CHUNKS_PER_BEAT; ++i) {
#pragma HLS PIPELINE II = 1
                segment = decltype(segment)::unpackMaxSizeSegment(data.data, chunkOffset);
                chunkOffset.index += SEGMENT_HEADER_CHUNKS;
                frameChunksLeft -= SEGMENT_HEADER_CHUNKS;

                // A probe has only headers, so update the ctrl status and move on
                creditUpdate.takeNewer(
                    ctrl_status_t<sizeof...(Ws)>::template makeACK<true>(creditUpdate.expectedFSN, segment.stream, buffers)
                );

                // End of frame / all entries read
                if (frameChunksLeft == 0 || chunkOffset.index >= CHUNKS_PER_BEAT) {
                    ctrlStatus.add(creditUpdate);
                    if (data.last) {
                        creditUpdate.ackImmediately = false;
                        creditUpdate.nackImmediately = false;
                        creditUpdate.changed = 0;
                    }
                    return data.last ? connection_state::EXPECT_FRAME_START : connection_state::EXPECT_BEAT;
                }

                // If payloadChunks is wrong we need to end parsing here, otherwise we drop into an endless loop,
                // since the offset of the header is now forever off.
                if (data.last) {
                    return connection_state::EXPECT_FRAME_START;
                }
            }
            return data.last ? connection_state::EXPECT_FRAME_START : connection_state::EXPECT_BEAT;
        }

        /** Parse the segments of one beat into the rings and return the next parser state. */
        template<int... Ws, size_t... Ds>
        connection_state _readDataSegments(
            stream_widths_t<Ws...>,
            ring_depths_t<Ds...>,
            ap_uint<clog2(1 + constants::BEAT_WIDTH)> initialOffset,
            beat_t& data,
            ctrl_status_t<sizeof...(Ws)>& creditUpdate,
            ring_buffers_t<ring_depths_t<Ds...>, stream_widths_t<Ws...>> &buffers,
            ap_uint<header_t::PAYLOAD_WIDTH>& frameChunksLeft,
            pending_ctrl_status_t<sizeof...(Ws)>& ctrlStatus
        ) {
#pragma HLS inline
            data_segment_t<constants::MAX_SEGMENT_PAYLOAD_WIDTH> segment;
            chunk_offset_t chunkOffset{initialOffset / 64};
            // Upper bound: every segment takes at least one chunk
            for (unsigned int i = 0; i < constants::CHUNKS_PER_BEAT; ++i) {
#pragma HLS PIPELINE II = 1
                segment = decltype(segment)::unpackMaxSizeSegment(data.data, chunkOffset);

                // Skip the segment header
                chunkOffset.index += SEGMENT_HEADER_CHUNKS;
                frameChunksLeft -= SEGMENT_HEADER_CHUNKS;

                // 0 chunks: BLOCKED report, answer with the stream's limit at once. Otherwise: write the payload to its ring
                if (segment.chunks == 0) {
                    // An unknown stream adds no credit record, but the ACK is still due
                    creditUpdate.takeNewer(
                        ctrl_status_t<sizeof...(Ws)>::template makeACK<true>(creditUpdate.expectedFSN, segment.stream, buffers)
                    );
                } else {
                    // TODO: Add error counters depending on the staging result
                    staging_result result = buffers.dispatchSegment(segment);
                    chunkOffset.index += segment.chunks;
                    frameChunksLeft -= segment.chunks;
                }

                // End of the entries or of the beat
                const bool frameDone = frameChunksLeft == 0;
                if (frameDone || chunkOffset.index >= constants::CHUNKS_PER_BEAT) {
                    // Frame complete: report the BLOCKED answers collected in it, then reset them
                    buffers.writeStagedAll();
                    if (frameDone) {
                        if (creditUpdate.ackImmediately) {
                            ctrlStatus.add(creditUpdate);
                        }
                        creditUpdate.changed = 0;
                        creditUpdate.ackImmediately = false;
                        return connection_state::EXPECT_FRAME_START;
                    } else {
                        return connection_state::EXPECT_BEAT;
                    }
                }

                // If payloadChunks is wrong we need to end parsing here, otherwise we drop into an endless loop,
                // since the offset of the header is now forever off.
                if (data.last) {
                    return connection_state::EXPECT_FRAME_START;
                }
            }
            // Not reached: every segment takes at least one chunk, so the loop returns by the end of the beat
            return data.last ? connection_state::EXPECT_FRAME_START : connection_state::EXPECT_BEAT;
        }

        /** Read and parse the header of the first beat of a frame. Returns the new state.
        Unknown frame type: Discard (and error counter). Implementation error -> not recoverable from here.
        FSN missing: NACK request
        FSN duplicate: PROBE? -> Ignore. Not PROBE? (=> DATA) -> Send ACK.
        */
        template<int... Ws>
        connection_state _readHeaders(
            ap_uint<clog2(1 + constants::BEAT_WIDTH)> &offset,
            header_t &header,
            beat_t &data,
            ap_uint<header_t::PAYLOAD_WIDTH>& frameChunksLeft,
            pending_ctrl_status_t<sizeof...(Ws)>& ctrlStatus,
            ap_uint<constants::FSN_WIDTH> &expectedFSN
        ) {
            offset = getOffsetFromEthertype(data.data);

            // Read header
            header = header_t::unpack(data.data >> offset);
            frameChunksLeft = header.payloadChunks;
            offset += header_t::WIDTH;

            // Neither DATA nor PROBE: discard.
            if (header.frameType != constants::FRAME_DATA && header.frameType != FRAME_PROBE) {
                // TODO: Error counter
                return !data.last ? connection_state::DISCARDING_INVALID_FRAME : connection_state::EXPECT_FRAME_START_AFTER_INVALID_FRAME;
            }

            // FSN after the expected one: gap, NACK. Before it: duplicate, ACK
            if (helpers::serialBefore(expectedFSN, header.frameSequenceNumber)) {
                ctrlStatus.add(ctrl_status_t<sizeof...(Ws)>::makeNACKImmediately(expectedFSN));
                return data.last ? connection_state::EXPECT_FRAME_START : connection_state::DISCARDING_INVALID_FRAME;
            } else if (helpers::serialBefore(header.frameSequenceNumber, expectedFSN)) {
                // On a PROBE, a too small FSN is okay - we send the response regardless
                if (header.frameType != FRAME_PROBE) {
                    ctrlStatus.add(ctrl_status_t<sizeof...(Ws)>::makeACKImmediately(expectedFSN));
                    return !data.last ? connection_state::DISCARDING_INVALID_FRAME : connection_state::EXPECT_FRAME_START_AFTER_INVALID_FRAME;
                }
            }

            // If nothing changes, return the state unchanged for further reading
            return connection_state::EXPECT_FRAME_START;
        }

        // TODO: Make the L2 header offset a build-time constant instead of reading it from every frame

        /** One call handles at most one beat. ConnId must be unique per connection: the state lives in
        statics of the instantiation, so two connections with equal widths and depths would share it. */
        template <size_t ConnId, int... Ws, size_t... Ds>
        inline void parseBeat(
            stream_widths_t<Ws...> widths,
            ring_depths_t<Ds...> depths,
            hls::stream<beat_t>& gatedStream,
            hls::stream<ctrl_status_t<sizeof...(Ws)>>& controlStream,
            hls::stream<ap_uint<Ws>>&... outStreams
        ) {
#pragma HLS pipeline II=1
            using ctrl_status_t = ctrl_status_t<sizeof...(Ws)>;

            // State kept across calls
            static connection_state state = connection_state::EXPECT_FRAME_START;
            static ap_uint<constants::FSN_WIDTH> expectedFSN = 0;
            static ring_buffers_t<ring_depths_t<Ds...>, stream_widths_t<Ws...>> buffers;
            static ap_uint<header_t::PAYLOAD_WIDTH> frameChunksLeft;
            static header_t header;

            // BLOCKED answers collected over the current frame, added to ctrlStatus when it ends
            static ctrl_status_t creditUpdate;

            // Everything to report to the controller_tx, sent as one status while no frame is in progress
            static pending_ctrl_status_t<sizeof...(Ws)> ctrlStatus;

            // Try to empty streams
            buffers.tryReadIntoStream(outStreams...);

            // Credit check on every call, beat or not: report the limits of all streams whose credit update is due
            auto currentCreditStatus = ctrl_status_t::template makeACK<false>(expectedFSN, buffers);
            if (currentCreditStatus.changed != 0) {
                ctrlStatus.add(currentCreditStatus);
            }

            // Handle at most one beat. A lambda, so that its early returns still reach the send below
            const auto handleBeat = [&]() {
                // Non-blocking, so the credit check above also runs while no frame arrives
                beat_t data;
                if (!gatedStream.read_nb(data)) {
                    return;
                }
                creditUpdate.expectedFSN = expectedFSN;

                // Discard the remaining beats of a dropped frame
                if (state == connection_state::DISCARDING_INVALID_FRAME) {
                    state = data.last ? connection_state::EXPECT_FRAME_START : state;
                    return;
                }

                // Bad FCS is seen on the first beat of a stream due to frameGate(...)
                if (data.user) {
                    state = data.last ? connection_state::EXPECT_FRAME_START : connection_state::DISCARDING_INVALID_FRAME;
                    ctrlStatus.add(ctrl_status_t::makeNACKImmediately(expectedFSN));
                    return;
                }

                // Beat 0: find and check the LSP header
                ap_uint<clog2(1 + constants::BEAT_WIDTH)> offset = 0;
                if (state == connection_state::EXPECT_FRAME_START || state == connection_state::EXPECT_FRAME_START_AFTER_INVALID_FRAME) {
                    state = _readHeaders(offset, header, data, frameChunksLeft, ctrlStatus, expectedFSN);
                }

                // A zero entry PROBE gets answered immedaitely
                if (header.frameType == FRAME_PROBE && header.payloadChunks == 0) {
                    ctrlStatus.add(ctrl_status_t::makeACKImmediately(expectedFSN));
                    state = connection_state::EXPECT_FRAME_START;
                    return;
                }

                // Parse the segments in the rest of the beat
                if (state == connection_state::EXPECT_BEAT || state == connection_state::EXPECT_FRAME_START) {
                    if (header.frameType == FRAME_PROBE) {
                        state = _readProbeEntries(
                            widths, depths, offset, data,
                            creditUpdate, buffers, frameChunksLeft, ctrlStatus
                        );
                    } else if (header.frameType == FRAME_DATA) {
                        state = _readDataSegments(
                            widths, depths, offset, data,
                            creditUpdate, buffers, frameChunksLeft, ctrlStatus
                        );
                    }
                    if (state == connection_state::EXPECT_FRAME_START && header.frameType != FRAME_PROBE) {
                        expectedFSN++;
                        ctrlStatus.add(ctrl_status_t::makeFSNUpdate(expectedFSN));
                    }
                }
            };
            handleBeat();

            // No frame in progress: send everything collected so far as one status. The only write to
            // controlStream, since a stream takes one write per cycle
            if (state == connection_state::EXPECT_FRAME_START || state == connection_state::EXPECT_FRAME_START_AFTER_INVALID_FRAME) {
                ctrlStatus.trySend(controlStream);
            }
        }
    } // namespace receiver
} // namespace lsp
#endif // LSP_RECEIVER_PARSE_BEAT_HPP
