#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <tuple>
#include <utility>
#include <algorithm>

#include "comptime.hpp"
#include "constants.hpp"
#include "common.hpp"

namespace lsp {
    namespace receiver {
        namespace ring_buffer {
            using namespace messages;

            /** Helper to determine the result of a ring buffer update. */
            enum class lsp_stream_ring_buffer_update_result : unsigned char {
                SEGMENT_READ,
                SENDER_BLOCKED,
                UNKNOWN_STREAM,
                BUFFER_FULL,
                INVALID_CHUNK_BYTEMASK,
                UNEXPECTED_ERROR
            };

            /**  Ring buffer that can store arriving packets. It has two jobs:

            1) FIFO Sizing (FINN internal) takes care of sizing FIFOs so that no performance degradation
            occurs. If however any other backpressure (from network connections or memory) causes stalls, we
            may need to buffer _additional_ samples, beyond what FINN's FIFOs store
            (they only consider internal stalls).

            2) Assembling a packet. When a packet is sent over frame- or beat-boundaries, it needs to be assembled.

            */
            template<size_t StreamIndex, int StreamWidth, size_t Depth>
            struct stream_ring_buffer_t {
                static_assert(StreamWidth % 8 == 0, "The stream widths must be multiples full bytes!");
                static constexpr int PointerWidth = clog2(Depth + 1);

                ap_uint<StreamWidth> buffer[Depth];
                ap_uint<PointerWidth> readPointer = 0;
                ap_uint<PointerWidth> writePointer = 0;
                ap_uint<PointerWidth> bufferSize = 0;      // Current usage, including speculative words. Not capacity.
                ap_uint<clog2(StreamWidth + 1)> inWordPointer = 0;
                ap_uint<constants::CREDIT_LIMIT_WIDTH> creditLimit = Depth;
                ap_uint<constants::CREDIT_LIMIT_WIDTH> lastCreditLimitSent = Depth;

                // The write side is speculative: payload is written as the frame arrives, but a bad
                // checksum is only reported at the frame's end. The committed write position is where
                // the write side returns to on a rollback, and the speculative words (complete words
                // written since the last commit) are hidden from the read side until the commit. Only
                // the write side is snapshotted: the read side keeps draining committed words meanwhile.
                ap_uint<PointerWidth> committedWritePointer = 0;
                ap_uint<clog2(StreamWidth + 1)> committedInWordPointer = 0;
                ap_uint<PointerWidth> speculativeWords = 0;

                // ap_uint's default constructor leaves the value uninitialized, so `= {}` would not zero it
                stream_ring_buffer_t() {
                    for (size_t i = 0; i < Depth; ++i) {
                        buffer[i] = 0;
                    }
                }

                /** Make the words written since the last commit visible to the read side. To be called
                after the frame's checksum showed no transmission errors. */
                void commit() {
                    #pragma HLS inline
                    committedWritePointer = writePointer;
                    committedInWordPointer = inWordPointer;
                    speculativeWords = 0;
                }

                /** Discard everything written since the last commit, including a partially filled word.
                To be called after the frame's checksum showed a transmission error. The bits of the
                partial word below the committed in-word position are untouched by the discarded writes. */
                void rollback() {
                    #pragma HLS inline
                    writePointer = committedWritePointer;
                    inWordPointer = committedInWordPointer;
                    bufferSize -= speculativeWords;
                    speculativeWords = 0;
                }

                /** Return whether nothing was written since the last commit. */
                bool isCommitted() const {
                    #pragma HLS inline
                    return speculativeWords == 0 && inWordPointer == committedInWordPointer;
                }

                /** Return whether a credit limit update is due (if more than half of the buffer was freed.) */
                bool creditLimitUpdateDue() {
                    return (creditLimit - lastCreditLimitSent) >= (Depth / 2);
                }

                /** Set internal memory of the last limit sent and return the limit to be sent
                to the sender via a CREDIT update. */
                ap_uint<constants::CREDIT_LIMIT_WIDTH> creditUpdate() {
                    lastCreditLimitSent = creditLimit;
                    return creditLimit;
                }

                template<int PayloadWidth>
                lsp_stream_ring_buffer_update_result updateFrom(const data_segment_t<PayloadWidth>& segment) {
                    // Figure out how wide a pointer into the segments payload needs to be
                    constexpr size_t PAYLOAD_WIDTH = PayloadWidth;
                    constexpr size_t SegmentPointerWidth = clog2(PAYLOAD_WIDTH + 1);

                    if (segment.chunks == 0) {
                        // According to spec this indicates a BLOCKED message from the sender,
                        // requiring a credit update.
                        return lsp_stream_ring_buffer_update_result::SENDER_BLOCKED;
                    }
                    if (bufferSize != Depth) {
                        // Calculate the number of bits to read:
                        // Chunks * CHUNK_SIZE, then a number of bytes determined by the validBytesLastChunk field
                        size_t bitsToRead = (
                            (segment.chunks - 1) * constants::CHUNK_SIZE_BITS
                            + segment.validBytesLastChunk * 8
                        );

                        // Some safety checks
                        if (segment.validBytesLastChunk > 8 || segment.validBytesLastChunk < 1) {
                            return lsp_stream_ring_buffer_update_result::INVALID_CHUNK_BYTEMASK;
                        }
                        // Only catches more chunks than any beat holds. The exact limit, the rest of
                        // the beat behind the header, is checked by the parser, which knows
                        // where the header sits
                        if (bitsToRead > PAYLOAD_WIDTH) {
                            return lsp_stream_ring_buffer_update_result::UNEXPECTED_ERROR;
                        }

                        // For space computation
                        const size_t expectedWordsToRead = ((inWordPointer + bitsToRead) + StreamWidth - 1) / StreamWidth;
                        // Form comptime loop unrolling
                        constexpr size_t MaxWordsToRead = (PAYLOAD_WIDTH + StreamWidth - 1) / StreamWidth + 1;

                        if (expectedWordsToRead > (Depth - bufferSize)) {
                            // No space to write all words, so nothing is written.
                            // IMPORTANT: This should never happen because the credit system should
                            // prevent the sender from emitting too many frames.
                            return lsp_stream_ring_buffer_update_result::BUFFER_FULL;
                        }

                        // We have enough space to write all words, so we can simply advance the pointer
                        // as much as required.
                        ap_uint<SegmentPointerWidth> inSegmentPointer = 0;
                        for (size_t it = 0; it < MaxWordsToRead; it++) {
    #pragma HLS PIPELINE II=1
                            // If more than the rest of the word is left write exactly that much.
                            // If less is left just write and advance the pointer.
                            size_t toWrite = std::min<size_t>(StreamWidth - inWordPointer, bitsToRead);
                            buffer[writePointer].range(
                                inWordPointer + toWrite - 1, inWordPointer
                            ) = segment.payload.range(
                                inSegmentPointer + toWrite - 1, inSegmentPointer
                            );

                            // Always move the inSegmentPointer
                            inSegmentPointer += toWrite;

                            // Only move the writePointer if we crossed a word boundary
                            if (inWordPointer + toWrite >= StreamWidth) {
                                // Avoid modulo here (HLS)
                                if (++writePointer == Depth) {
                                    writePointer = 0;
                                }

                                // We can advance without a check because we already
                                // saw above that there is enough space.
                                bufferSize++;
                                speculativeWords++;
                            }

                            // Always move the inWordPointer, regardless of word boundaries
                            // Avoid modulo here (HLS)
                            inWordPointer += toWrite;
                            if (inWordPointer > StreamWidth - 1) {
                                inWordPointer -= StreamWidth;
                            }

                            // Update bits to read
                            bitsToRead -= toWrite;

                            // Jump out if we are done
                            if (bitsToRead == 0) {
                                break;
                            }
                        }
                        return lsp_stream_ring_buffer_update_result::SEGMENT_READ;
                    }
                    return lsp_stream_ring_buffer_update_result::BUFFER_FULL;
                }

                /** Try to write one word into the user stream. The write is non-blocking, so that a full user
                stream never stalls the free-running kernel: if the stream is full, the word stays in the
                buffer and is written in a later call. */
                template<int D>
                void tryReadIntoStream(hls::stream<ap_uint<StreamWidth>, D>& stream) {
                    // Only read if the read pointer looks at committed data, and only advance if the write succeeded
                    if (bufferSize > speculativeWords && stream.write_nb(buffer[readPointer])) {
                        // Increment and wrap, avoid modulo
                        if (++readPointer == Depth) {
                            readPointer = 0;
                        }
                        bufferSize--;
                        creditLimit++;
                    }
                }
            };

            /** Ring buffer depths (in words) per stream, in stream index order. A separate type, because a
            template cannot take the depths and the widths as two non-type packs side by side. */
            template<size_t... Depths>
            using lsp_ring_depths_t = std::index_sequence<Depths...>;

            /** Implementation of lsp_stream_ring_buffers_t: pairs the stream widths with their indices
            and depths. */
            template<typename Indices, typename Depths, int... StreamWidths>
            struct _lsp_stream_ring_buffers;
            template<size_t... StreamIndices, size_t... Depths, int... StreamWidths>
            struct _lsp_stream_ring_buffers<
                std::index_sequence<StreamIndices...>, lsp_ring_depths_t<Depths...>, StreamWidths...
            > {
                static_assert(
                    sizeof...(Depths) == sizeof...(StreamWidths),
                    "Exactly one ring buffer depth per stream is required"
                );
                using type = std::tuple<stream_ring_buffer_t<StreamIndices, StreamWidths, Depths>...>;
            };

            /** One ring buffer per stream: element I is the ring buffer for stream index I with the
            I-th width and the I-th depth, i.e. the stream index is the position in `StreamWidths`.
            `Depths` is an lsp_ring_depths_t. */
            template<typename Depths, int... StreamWidths>
            using stream_ring_buffers_t = typename _lsp_stream_ring_buffers<
                std::make_index_sequence<sizeof...(StreamWidths)>, Depths, StreamWidths...
            >::type;

            /** Implementation of visitRing: one call of `f` per ring, only the matching one enabled. */
            template<typename Rings, typename F, typename R, size_t... Is>
            R _visitRing(Rings& rings, unsigned index, F& f, R result, std::index_sequence<Is...>) {
                #pragma HLS inline
                // C++14 replacement for a fold expression: checks every ring index in order. The cast
                // to void is needed because ap_int overloads the comma operator as concatenation
                int expanded[] = {0, (index == Is ? ((void) (result = f(std::get<Is>(rings))), 0) : 0)...};
                (void) expanded;
                return result;
            }

            /** Call `f` with the ring buffer at runtime position `index` of an lsp_stream_ring_buffers_t
            and return its result, or `fallback` if no ring has that index. Every ring has its own type,
            so a runtime index cannot return a reference; `f` is a generic lambda instead, and its result
            type must be the same for every ring, e.g.
                visitRing(rings, stream, [](auto& ring) { return ring.getAvailableCredits(); }, 0u)
            In hardware every ring gets its own copy of `f`, only the matching one enabled: no
            multiplexer in front of the rings, one behind them for the result. */
            template<typename... Rings, typename F, typename R>
            R visitRing(std::tuple<Rings...>& rings, unsigned index, F f, R fallback) {
                #pragma HLS inline
                return _visitRing(rings, index, f, fallback, std::index_sequence_for<Rings...>{});
            }

            /** Implementation of forEachRing: one call of `f` per ring. */
            template<typename Rings, typename F, size_t... Is>
            void _forEachRing(Rings& rings, F& f, std::index_sequence<Is...>) {
                #pragma HLS inline
                // C++14 replacement for a fold expression
                int expanded[] = {0, ((void) f(std::get<Is>(rings)), 0)...};
                (void) expanded;
            }

            /** Call `f` (a generic lambda) on every ring buffer of an lsp_stream_ring_buffers_t. In
            hardware every ring gets its own copy of `f`, all running in parallel. */
            template<typename... Rings, typename F>
            void forEachRing(std::tuple<Rings...>& rings, F f) {
                #pragma HLS inline
                _forEachRing(rings, f, std::index_sequence_for<Rings...>{});
            }

            /** Append a segment to the ring buffer of its stream: ring I (of an lsp_stream_ring_buffers_t)
            takes the segments whose stream ID is I, see #75 for the numbering. A stream ID without a ring
            returns UNKNOWN_STREAM. */
            template<typename... Rings, int PayloadWidth>
            lsp_stream_ring_buffer_update_result dispatchSegment(
                std::tuple<Rings...>& rings,
                const data_segment_t<PayloadWidth>& segment
            ) {
                #pragma HLS inline
                return visitRing(
                    rings,
                    segment.stream,
                    [&segment](auto& ring) { return ring.updateFrom(segment); },
                    lsp_stream_ring_buffer_update_result::UNKNOWN_STREAM
                );
            }

            /** Commit the writes on the ring with the given index. To be called after the final beat of
            a frame confirmed via checksum that the frame was transmitted successfully. Returns false if
            no ring has that index. */
            template<typename... Rings>
            bool commit(std::tuple<Rings...>& rings, unsigned index) {
                #pragma HLS inline
                return visitRing(rings, index, [](auto& ring) { ring.commit(); return true; }, false);
            }

            /** Discard the writes since the last commit on the ring with the given index. To be called
            after the final beat of a frame showed a checksum error. Returns false if no ring has that
            index. */
            template<typename... Rings>
            bool rollback(std::tuple<Rings...>& rings, unsigned index) {
                #pragma HLS inline
                return visitRing(rings, index, [](auto& ring) { ring.rollback(); return true; }, false);
            }

            /** Commit the writes on every ring. To be called after the final beat of a frame confirmed via
            checksum that the frame was transmitted successfully. Rings the frame did not touch are
            unchanged, so the caller does not need to track which streams a frame wrote to. */
            template<typename... Rings>
            void commitAll(std::tuple<Rings...>& rings) {
                #pragma HLS inline
                forEachRing(rings, [](auto& ring) { ring.commit(); });
            }

            /** Discard the writes since the last commit on every ring. To be called after the final beat
            of a frame showed a checksum error. Rings the frame did not touch are unchanged. */
            template<typename... Rings>
            void rollbackAll(std::tuple<Rings...>& rings) {
                #pragma HLS inline
                forEachRing(rings, [](auto& ring) { ring.rollback(); });
            }

            /** Return whether a credit update is due for this ring (if the buffer was filled with over N/2 elements, having a max capacity of N). */
            template<typename... Rings>
            bool creditUpdateDue(std::tuple<Rings...>& rings, unsigned index) {
                #pragma HLS inline
                return visitRing(rings, index, [](auto& ring) { return ring.creditLimitUpdateDue(); }, false);
            }

            /** Execute a credit update on the given ring */
            template<typename... Rings>
            ap_uint<constants::CREDIT_LIMIT_WIDTH> creditUpdate(std::tuple<Rings...>& rings, unsigned index) {
                #pragma HLS inline
                return visitRing(
                    rings, index, [](auto& ring) { return ring.creditUpdate(); },
                    ap_uint<constants::CREDIT_LIMIT_WIDTH>(0)
                );
            }

            /** Implementation of tryReadIntoStream: ring I feeds the I-th stream. */
            template<int... Ws, typename... Rings, size_t... Is>
            void _tryReadIntoStream(
                std::index_sequence<Is...>,
                std::tuple<Rings...>& rings,
                hls::stream<ap_uint<Ws>>&... streams
            ) {
                #pragma HLS inline
                int a[] = {0, ((void) std::get<Is>(rings).tryReadIntoStream(streams), 0)...};
                (void) a;
            }

            /** Try to write one committed word of every ring into its stream: ring I feeds the I-th stream. In
            hardware every ring gets its own copy, all running in parallel. */
            template<int... Ws, typename... Rings>
            void tryReadIntoStream(std::tuple<Rings...>& rings, hls::stream<ap_uint<Ws>>&... streams) {
                #pragma HLS inline
                _tryReadIntoStream(std::make_index_sequence<sizeof...(Ws)>(), rings, streams...);
            }
        }
    }
}
