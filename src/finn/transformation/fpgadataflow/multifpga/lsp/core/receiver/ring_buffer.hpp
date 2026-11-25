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

            enum class staging_result : unsigned char {
                WORD_STAGED,
                WORD_STAGED_PARTIAL,
                WORD_OVERFLOW,
                BUFFER_FULL,
                UNKNOWN_STREAM
            };

            /** Updated ring buffer. Two noticeable extra features:
            - Banking: Allows several writes to the same ring buffer in one cycle (<Banks> BRAMS with W/R slots instead of one common one)
            - Registering: Only entire words get written, so we don't need a bit-granular inWordPointer (only chunk-wise and for the accumulator)
            */
            template<int StreamWidth, size_t Depth, size_t Banks=4>
            struct stream_ring_buffer_t {
                // Banks must be a power of 2 so that wordPointer bits can be masked easily
                static_assert((Banks & (Banks - 1)) == 0, "Banks must be a power of 2");
                static_assert(
                    Depth % Banks == 0,
                    "Depth must be dividable by banks. The total capacity of the buffer is Depth, split into Banks banks."
                );

                // Rounded up amount of chunks needed to express one StreamWidth in chunks
                static constexpr int WordChunks = (StreamWidth + constants::CHUNK_SIZE_BITS - 1) / constants::CHUNK_SIZE_BITS;

                // Datawidths
                static constexpr int BankBits = clog2(1 + Banks);
                static constexpr int SlotBits = clog2(1 + Depth/Banks);
                static constexpr int WordPointerWidth = BankBits + SlotBits;

                // Dimension 1: Bank, Dimension 2: Buffer Slot
                ap_uint<StreamWidth> buffer[Banks][Depth/Banks];
#pragma HLS array_partition dim=1 variable=buffer complete
#pragma HLS bind_storage variable=buffer type=ram_s2p impl=bram

                // Lanes are registers that store the partialWords before putting
                // them into a buffer
                ap_uint<StreamWidth> lanes[Banks];
#pragma HLS array_partition variable=lanes complete

                ap_uint<clog2(1 + Banks)> laneCounter = 0;
                ap_uint<Banks> validLanes = 0;

                // Pointer across all banks and slots
                ap_uint<WordPointerWidth> writePointer = 0;
                ap_uint<WordPointerWidth> committedWritePointer = 0;

                // Store how many words are speculative
                ap_uint<WordPointerWidth> speculativeWords = 0;

                // Read pointer is also global, just as writePointer
                ap_uint<WordPointerWidth> readPointer = 0;

                // Store occupied slots in the entire buffer
                ap_uint<clog2(Depth + 1)> bufferSize = 0;
                ap_uint<clog2(Depth + 1)> committedBufferSize = 0;

                // Registers to write from. Since segments can only deliver chunks (apart from the last chunk),
                // the offset is also just in entire chunks, causing smaller shifters necessary.
                ap_uint<StreamWidth> partialWord = 0;
                ap_uint<clog2(1 + StreamWidth/constants::CHUNK_SIZE_BITS) + 1> partialWordChunkOffset = 0;
                ap_uint<StreamWidth> committedPartialWord = 0;
                ap_uint<clog2(1 + StreamWidth/constants::CHUNK_SIZE_BITS) + 1> committedPartialWordChunkOffset = 0;

                // Credit bookkeeping
                ap_uint<constants::CREDIT_LIMIT_WIDTH> creditLimit = Depth;
                ap_uint<constants::CREDIT_LIMIT_WIDTH> lastCreditLimitSent = Depth;


                ap_uint<BankBits> getBank(const ap_uint<WordPointerWidth> &ptr) {
#pragma HLS inline
                    return ptr % Banks;
                }

                ap_uint<SlotBits> getSlot(const ap_uint<WordPointerWidth> &ptr) {
#pragma HLS inline
                    return ptr / Banks;
                }

                ap_uint<BankBits> getCurrentWriteBank() {
#pragma HLS inline
                    return getBank(writePointer);
                }

                ap_uint<SlotBits> getCurrentWriteSlot() {
#pragma HLS inline
                    return getSlot(writePointer);
                }

                ap_uint<BankBits> getCurrentReadBank() {
#pragma HLS inline
                    return getBank(readPointer);
                }

                ap_uint<SlotBits> getCurrentReadSlot() {
#pragma HLS inline
                    return getSlot(readPointer);
                }

            private:
                /** Update the currently pointed to register, filling it with data from the segment. Does NOT write the buffer yet.

                Returns: Whether a word is complete in the accumulator register (partialWord).
                */
                template<size_t PayloadWidth>
                bool updateRegisters(const data_segment_t<PayloadWidth> &segment) {
                    if (partialWordChunkOffset == 0) {
                        partialWord = ap_uint<StreamWidth>(segment.payload);
                    } else {
                        partialWord |= ap_uint<StreamWidth>(segment.payload) << (partialWordChunkOffset * constants::CHUNK_SIZE_BITS);
                    }

                    // Increase offset. We need StreamWidth/CHUNK_SIZE_BITS chunks to fill one word.
                    // Since StreamWidth might not be divisible by CHUNK_SIZE_BITS, we may end up with a 1-element
                    // underestimate (30 bit width / 64 bit chunk = 0), so we need to round up (30+63 = 93. 93/64 = 1).
                    if (partialWordChunkOffset + segment.chunks == WordChunks) {
                        partialWordChunkOffset = 0;
                        return true;
                    } else {
                        partialWordChunkOffset += segment.chunks;
                        return false;
                    }
                }

            public:


                /** Stage a write to the buffer. Stores it temporarily until the write
                is finished with writeStaged.
                If more stage() calls happen than the buffer has banks, the bank index wraps
                and overwrites from the start. Make sure to call writeStaged atleast every <Banks>
                stage calls. */
                template<size_t PayloadWidth>
                staging_result stage(const data_segment_t<PayloadWidth> &segment) {
                    // If too many chunks are delivered this _can_ indicate, that the sender
                    // tried to end a word and start a new one in the same beat, which is not allowed.
                    if (partialWordChunkOffset + segment.chunks > WordChunks) {
                        return staging_result::WORD_OVERFLOW;
                    }
                    if (bufferSize + laneCounter < Depth) {
                        if (updateRegisters(segment)) {
                            ap_uint<BankBits> targetBank = (getCurrentWriteBank() + laneCounter) % Banks;
                            lanes[targetBank] = partialWord;
                            validLanes[targetBank] = 1;
                            laneCounter++;
                        } else {
                            return staging_result::WORD_STAGED_PARTIAL;
                        }
                    } else {
                        return staging_result::BUFFER_FULL;
                    }
                    return staging_result::WORD_STAGED;
                }

                /** Finish a write by moving the staged data into the buffer. Only
                this call advances the write pointer and occupies buffer space.

                We use lanes here so that the index into buffer[banks] is compile-time checkable, and HLS
                can see that every port on every BRAM is accessed at most once per call. Otherwise the tool
                could assume more than 1 write to the same port, leading to II>1.
                */
                void writeStaged() {
                    ap_uint<SlotBits> currentSlot = getCurrentWriteSlot();
                    ap_uint<SlotBits> nextSlot = (currentSlot + 1 == Depth/Banks) ? 0 : currentSlot + 1;
                    ap_uint<BankBits> currentBank = getCurrentWriteBank();
                    for (int bank = 0; bank < Banks; bank++) {
#pragma HLS unroll
                        if (validLanes[bank]) {
                            buffer[bank][(bank >= currentBank) ? currentSlot : nextSlot] = lanes[bank];
                        }

                    }
                    writePointer += laneCounter;
                    speculativeWords += laneCounter;
                    bufferSize += laneCounter;
                    if (writePointer >= Depth) {
                        writePointer -= Depth;
                    }
                    laneCounter = 0;
                    validLanes = 0;
                }

                /** Return whether nothing was written since the last commit. */
                bool isCommitted() const {
                    #pragma HLS inline
                    return speculativeWords == 0;
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

                /** Commit results into the ring buffer. */
                void commit() {
                    #pragma HLS inline
                    bufferSize -= speculativeWords;
                    committedWritePointer = writePointer;
                    speculativeWords = 0;
                    committedPartialWord = partialWord;
                    committedPartialWordChunkOffset = partialWordChunkOffset;
                }

                /** Roll back to last stable state. */
                void rollback() {
                    #pragma HLS inline
                    bufferSize = committedBufferSize;
                    writePointer = committedWritePointer;
                    speculativeWords = 0;
                    validLanes = 0;
                    laneCounter = 0;
                    partialWord = committedPartialWord;
                    partialWordChunkOffset = committedPartialWordChunkOffset;
                }

                /** Try to write one word into the user stream. The write is non-blocking, so that a full user
                stream never stalls the free-running kernel: if the stream is full, the word stays in the
                buffer and is written in a later call. */
                template<int D>
                void tryReadIntoStream(hls::stream<ap_uint<StreamWidth>, D>& stream) {
                    // Only read if the read pointer looks at committed data, and only advance if the write succeeded
                    if (bufferSize > speculativeWords && stream.write_nb(buffer[getCurrentReadBank()][getCurrentWriteSlot()])) {
                        // Increment and wrap, avoid modulo
                        if (++readPointer == Depth) {
                            readPointer = 0;
                        }
                        bufferSize--;
                        creditLimit++;
                    }
                }
            };


            template<typename Depths, typename Widths, typename Is = std::make_index_sequence<Widths::count>>
            struct ring_buffers_t;

            template<size_t... Ds, int... Ws, size_t... Is>
            struct ring_buffers_t<utils::ring_depths_t<Ds...>, utils::stream_widths_t<Ws...>, std::index_sequence<Is...>> {
                std::tuple<stream_ring_buffer_t<Ws, Ds>...> rings;

                /** Append a segment to the ring buffer of its stream: ring I (of an lsp_stream_ring_buffers_t)
                takes the segments whose stream ID is I, see #75 for the numbering. A stream ID without a ring
                returns UNKNOWN_STREAM. */
                template<int PayloadWidth>
                staging_result dispatchSegment(
                    const data_segment_t<PayloadWidth>& segment
                ) {
                    #pragma HLS inline
                    return helpers::visitTuple(
                        rings,
                        segment.stream,
                        [&segment](auto& ring) { return ring.updateFrom(segment); },
                        staging_result::UNKNOWN_STREAM
                    );
                }

                /** Commit the writes on the ring with the given index. To be called after the final beat of
                a frame confirmed via checksum that the frame was transmitted successfully. Returns false if
                no ring has that index. */
                bool commit(unsigned index) {
                    #pragma HLS inline
                    return helpers::visitTuple(rings, index, [](auto& ring) { ring.commit(); return true; }, false);
                }

                /** Discard the writes since the last commit on the ring with the given index. To be called
                after the final beat of a frame showed a checksum error. Returns false if no ring has that
                index. */
                bool rollback(unsigned index) {
                    #pragma HLS inline
                    return helpers::visitTuple(rings, index, [](auto& ring) { ring.rollback(); return true; }, false);
                }

                /** Commit the writes on every ring. To be called after the final beat of a frame confirmed via
                checksum that the frame was transmitted successfully. Rings the frame did not touch are
                unchanged, so the caller does not need to track which streams a frame wrote to. */
                void commitAll() {
                    #pragma HLS inline
                    helpers::forEachInTuple(rings,[](auto& ring) { ring.commit(); });
                }

                /** Discard the writes since the last commit on every ring. To be called after the final beat
                of a frame showed a checksum error. Rings the frame did not touch are unchanged. */
                void rollbackAll() {
                    #pragma HLS inline
                    helpers::forEachInTuple(rings,[](auto& ring) { ring.rollback(); });
                }

                /** Return whether a credit update is due for this ring (if the buffer was filled with over N/2 elements, having a max capacity of N). */
                bool creditUpdateDue(unsigned index) {
                    #pragma HLS inline
                    return helpers::visitTuple(rings, index, [](auto& ring) { return ring.creditLimitUpdateDue(); }, false);
                }

                /** Execute a credit update on the given ring */
                ap_uint<constants::CREDIT_LIMIT_WIDTH> creditUpdate(unsigned index) {
                    #pragma HLS inline
                    return helpers::visitTuple(
                        rings, index, [](auto& ring) { return ring.creditUpdate(); },
                        ap_uint<constants::CREDIT_LIMIT_WIDTH>(0)
                    );
                }

                /** Implementation of tryReadIntoStream: ring I feeds the I-th stream. */
                void _tryReadIntoStream(
                    std::index_sequence<Is...>,
                    hls::stream<ap_uint<Ws>>&... streams
                ) {
                    #pragma HLS inline
                    int a[] = {0, ((void) std::get<Is>(rings).tryReadIntoStream(streams), 0)...};
                    (void) a;
                }

                /** Try to write one committed word of every ring into its stream: ring I feeds the I-th stream. In
                hardware every ring gets its own copy, all running in parallel. */
                void tryReadIntoStream(hls::stream<ap_uint<Ws>>&... streams) {
                    #pragma HLS inline
                    _tryReadIntoStream(std::make_index_sequence<sizeof...(Ws)>(), rings, streams...);
                }
            };
        }
    }
}
