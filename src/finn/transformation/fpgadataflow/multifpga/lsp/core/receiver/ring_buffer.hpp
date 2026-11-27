#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <algorithm>

#include "comptime.hpp"
#include "constants.hpp"
#include "common.hpp"
#include "helpers.hpp"

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
            template<int StreamWidth, size_t WordDepth, int WritePipelineDelay>
            struct stream_ring_buffer_t {
                static constexpr int ChunkWidth = constants::CHUNK_SIZE_BITS;

                // Min: Need as many banks as chunks of one stream can be in a beat. A segment header always
                // takes at least one chunk, so CHUNKS_PER_BEAT - 1 would suffice, but CHUNKS_PER_BEAT keeps
                // Banks a power of two, so the bank pointer arithmetic needs no modulo logic
                // Rounded up amount of chunks needed to express one StreamWidth in chunks
                static constexpr int WordChunks = (StreamWidth + ChunkWidth - 1) / ChunkWidth;
                static constexpr int Banks = constants::CHUNKS_PER_BEAT;

                // Depth in chunks
                static constexpr int Depth = WordDepth * WordChunks;

                // Slot count, rounded up: the buffer may be physically larger than Depth. The free space
                // is still computed from Depth, and both pointers wrap at Slots * Banks, so the extra
                // chunks are simply never all occupied at the same time
                static constexpr int Slots = (Depth + Banks - 1) / Banks;

                // Datawidths
                static constexpr int BankBits = clog2(1 + Banks);
                static constexpr int SlotBits = clog2(1 + Slots);

                static_assert(
                    WritePipelineDelay > 2,
                    "Pipeline delay must be larger than BRAM access latency and pipeline delay (at least >2)"
                );

                // Dimension 1: Bank, Dimension 2: Buffer Slot
                // One BRAM per bank, partitioned in writeStaged (HLS pragmas are not allowed at class scope)
                ap_uint<ChunkWidth> buffer[Banks][Slots];

                // Staging area of one beat: lane i holds the chunk that goes to bank i. stage() spreads
                // the chunks of each segment onto the lanes, so writeStaged() writes every bank at most once
                ap_uint<ChunkWidth> lanes[Banks];
                ap_uint<Banks> validLanes = 0;

                // Chunks staged in the current beat
                ap_uint<clog2(1 + Banks)> laneCounter = 0;

                // Pointer across all banks and slots
                ap_uint<BankBits> bankWritePointer = 0;
                ap_uint<SlotBits> slotWritePointer = 0;

                // Read pointer is also global, just as writePointer
                ap_uint<BankBits> bankReadPointer = 0;
                ap_uint<SlotBits> slotReadPointer = 0;

                // Wrapping word counters. The write side (stage, writeStaged) only updates
                // written, the read side (tryReadIntoStream) only updates readCount,
                // so that neither side has a read-modify-write on a register the other side also writes.
                // The occupancy never exceeds Depth, so the modular differences below are exact:
                static constexpr int CountBits = clog2(Depth + 1);
                ap_uint<CountBits> written = 0;
                ap_uint<CountBits> readCount = 0;

                // Shift register that reports the written words delayed by the number of banks
                // This makes sure that there is never a RAW clash in one cycle. Needs to be encoded via
                // a dependence pragma as well.
                ap_uint<CountBits> writtenDelayed[WritePipelineDelay] = {};


                // If a word is wider than the bank count (e.g. 5 chunks but 4 banks)
                // then we first read 4 chunks and in the next call read 1 chunk and send the word
                static constexpr int ReadsPerCall = WordChunks > Banks ? Banks : WordChunks;
                static constexpr int StorageChunks = WordChunks <= Banks ? WordChunks : WordChunks + Banks;

                // We can, at best, send one word per cycle - the receiver cant receive more.
                // If WordChunks <= BankChunks, we are always done after the loop, so the storage must be WordChunks wide
                // If WordChunks > BankChunks, over multiple calls we store 1 word width + at most an entire slot of BankChunks
                ap_uint<StorageChunks * ChunkWidth> storage = 0;
                ap_uint<clog2(1 + StorageChunks)> chunksStored = 0;

                // Credit bookkeeping
                ap_uint<constants::CREDIT_LIMIT_WIDTH> creditLimit = Depth;
                ap_uint<constants::CREDIT_LIMIT_WIDTH> lastCreditLimitSent = Depth;

            public:
                template<int PayloadWidth>
                staging_result stage(const data_segment_t<PayloadWidth> &segment) {
                    // The free space does not depend on laneCounter and the counters do not change between
                    // stage calls, so it is shared across them and each call only compares laneCounter
                    const ap_uint<CountBits> freeSlots = Depth - ap_uint<CountBits>(written - readCount);

                    // More chunks in one beat than there are lanes: only a malformed frame does this. Reject
                    // it, so that laneCounter cannot overflow and corrupt the write pointer
                    if (laneCounter + segment.chunks > Banks) {
                        return staging_result::WORD_OVERFLOW;
                    }
                    if (laneCounter + segment.chunks > freeSlots) {
                        return staging_result::BUFFER_FULL;
                    }

                    // Chunk j of the segment goes to the bank behind the chunks staged before it
                    // No need to consider validBytesLastChunk here, since if a chunk is completely full it doesnt matter,
                    // and if its not, this chunk is the last to be picked up on the read side, so multiple segments for one
                    // word can never create a gap.
                    for (int j = 0; j < PayloadWidth / ChunkWidth; j++) {
#pragma HLS unroll
                        if (j < segment.chunks) {
                            const ap_uint<BankBits> targetBank = (bankWritePointer + laneCounter + j) % Banks;
                            lanes[targetBank] = segment.payload.range((j + 1) * ChunkWidth - 1, j * ChunkWidth);
                            validLanes[targetBank] = 1;
                        }
                    }
                    laneCounter += segment.chunks;
                    return staging_result::WORD_STAGED;
                }

                /** Finish a write by moving the staged data into the buffer. Only
                this call advances the write pointer and occupies buffer space.

                We use lanes here so that the index into buffer[banks] is compile-time checkable, and HLS
                can see that every port on every BRAM is accessed at most once per call. Otherwise the tool
                could assume more than 1 write to the same port, leading to II>1.
                */
                void writeStaged() {
#pragma HLS array_partition dim=1 variable=buffer complete
#pragma HLS bind_storage variable=buffer type=ram_s2p impl=bram
#pragma HLS array_partition variable=lanes complete
                    ap_uint<SlotBits> nextSlot = (slotWritePointer + 1 == Slots)
                        ? ap_uint<SlotBits>(0)
                        : ap_uint<SlotBits>(slotWritePointer + 1);

                    // Loop all bank indices in parallel (compile time index).
                    for (int bank = 0; bank < Banks; bank++) {
#pragma HLS unroll
                        if (validLanes[bank]) {
                            buffer[bank][(bank >= bankWritePointer) ? slotWritePointer : nextSlot] = lanes[bank];
                        }
                    }
                    slotWritePointer = slotWritePointer + ((bankWritePointer + laneCounter) / Banks);
                    bankWritePointer += laneCounter;
                    written += laneCounter;
                    if (bankWritePointer >= Banks) {
                        bankWritePointer -= Banks;
                    }
                    if (slotWritePointer >= Slots) {
                        slotWritePointer -= Slots;
                    }
                    laneCounter = 0;
                    validLanes = 0;
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

                /** Try reading an entire word into the stream. Does nothing if not enough words are available. (non-blocking) */
                template<int D>
                void tryReadIntoStream(hls::stream<ap_uint<StreamWidth>, D>& stream) {
#pragma HLS dependence variable=buffer type=inter direction=RAW dependent=false
                    // Shift words every call
                    for (int i = WritePipelineDelay - 1; i > 0; i--) {
#pragma HLS unroll
                        writtenDelayed[i] = writtenDelayed[i - 1];
                    }

                    // Insert the current written word count at the beginning of the shifter
                    writtenDelayed[0] = written;

                    // Only read if we have enough chunks to read for an entire word. Comptime loop limit.
                    // We also need enough space. If the previous iteration collected enough chunks for a word, but
                    // the receiver was busy, we didnt advance. This could cause us to write over the boundary of the storage
                    const ap_uint<CountBits> chunksInBuffer
                        = ap_uint<CountBits>(writtenDelayed[WritePipelineDelay - 1] - readCount)
                        - chunksStored;
                    const ap_uint<CountBits> chunksMissing = WordChunks - chunksStored;
                    const bool hasChunksForOneWord = chunksInBuffer >= ReadsPerCall || chunksInBuffer >= chunksMissing;
                    const bool hasSpace = (WordChunks <= Banks) || (StorageChunks - chunksStored >= ReadsPerCall);
                    const bool hasFullWordInRegister = chunksStored >= WordChunks;
                    const ap_uint<CountBits> chunksToReadThisIteration
                        = chunksInBuffer >= ReadsPerCall
                        ? ap_uint<CountBits>(ReadsPerCall)
                        : chunksMissing;
                    if (hasFullWordInRegister || (hasChunksForOneWord && hasSpace)) {
                        if (!hasFullWordInRegister && hasSpace) {
                            for (int i = 0; i < ReadsPerCall; i++) {
    #pragma HLS unroll
                                if (i < chunksToReadThisIteration) {
                                    storage.range(
                                        (chunksStored + i+1) * ChunkWidth - 1,
                                        (chunksStored + i) * ChunkWidth
                                    ) = buffer[
                                        (bankReadPointer+i) % Banks
                                    ][
                                        (slotReadPointer + ((bankReadPointer + i) / Banks)) % Slots
                                    ];
                                }
                            }
                            chunksStored += chunksToReadThisIteration;
                            slotReadPointer = slotReadPointer + ((bankReadPointer + chunksToReadThisIteration) / Banks);
                            bankReadPointer = (bankReadPointer + chunksToReadThisIteration) % Banks;
                            if (slotReadPointer >= Slots) {
                                slotReadPointer -= Slots;
                            }
                        }

                        // Do we have enough chunks to send a word?
                        if (chunksStored >= WordChunks) {
                            // Write the sliced word. Only if the write is successful do we advance the state
                            if(stream.write_nb(storage.range(StreamWidth - 1, 0))) {
                                // Read pointer updates: Automatically wraps
                                readCount += WordChunks;
                                creditLimit += WordChunks;

                                // Shift out the used data
                                storage >>= WordChunks * ChunkWidth;
                                chunksStored -= WordChunks;
                            }
                        }

                    }
                }
            };


            /** The stream rings of one connection. WritePipelineDelay is the same for all rings. */
            template<typename Depths, typename Widths, int WritePipelineDelay>
            struct ring_buffers_t;

            template<size_t... Ds, int... Ws, int WritePipelineDelay>
            struct ring_buffers_t<utils::ring_depths_t<Ds...>, utils::stream_widths_t<Ws...>, WritePipelineDelay> {
                std::tuple<stream_ring_buffer_t<Ws, Ds, WritePipelineDelay>...> rings;

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
                        [&segment](auto& ring) { return ring.stage(segment); },
                        staging_result::UNKNOWN_STREAM
                    );
                }

                /** Move the words staged by dispatchSegment into every ring. To be called once per beat, after
                its segments were dispatched, so that no ring gets more than Banks staged words. */
                void writeStagedAll() {
                    #pragma HLS inline
                    helpers::forEachInTuple(rings, [](auto& ring) { ring.writeStaged(); });
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
                template<size_t... Is>
                void _tryReadIntoStream(
                    std::index_sequence<Is...>,
                    hls::stream<ap_uint<Ws>>&... streams
                ) {
                    #pragma HLS inline
                    int a[] = {0, ((void) std::get<Is>(rings).tryReadIntoStream(streams), 0)...};
                    (void) a;
                }

                /** Try to write one word of every ring into its stream: ring I feeds the I-th stream. In
                hardware every ring gets its own copy, all running in parallel. */
                void tryReadIntoStream(hls::stream<ap_uint<Ws>>&... streams) {
                    #pragma HLS inline
                    _tryReadIntoStream(std::make_index_sequence<sizeof...(Ws)>(), streams...);
                }
            };
        }
    }
}
