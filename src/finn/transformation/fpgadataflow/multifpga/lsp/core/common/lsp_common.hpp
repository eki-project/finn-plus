// Shared definitions for the LSP sender and receiver kernels.
#ifndef LSP_COMMON_HPP
#define LSP_COMMON_HPP

#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <tuple>
#include <utility>
#include <algorithm>
#include <type_traits>

#define AP_WIDTH(x) (decltype(x)::width)

constexpr size_t MAX_ETHERNET_HEADER_LENGTH_BYTES = 22;
constexpr size_t ETHERNET_MTU = 1500;
using MAC = ap_uint<48>;

namespace lsp {
    constexpr size_t CURRENT_VERSION = 0;

    /** Ceiling of log2(n): the number of bits needed to tell n different values apart
    * (clog2(1) == 0). To store the value n itself, use clog2(n + 1). */
    constexpr int clog2(unsigned long long n) {
        int bits = 0;
        while ((1ULL << bits) < n) ++bits;
        return bits;
    }
    static_assert(clog2(1) == 0 && clog2(2) == 1 && clog2(8) == 3 && clog2(9) == 4, "clog2 is broken");


    namespace constants {
        // CONSTANTS

        // Datawidths in bits
        constexpr int BEAT_WIDTH = 512;
        constexpr int STREAM_ID_WIDTH = 10;
        constexpr int FSN_WIDTH = 32;
        constexpr int FRAME_TYPE_WIDTH = 3;
        constexpr int CREDIT_LIMIT_WIDTH = 32;

        constexpr unsigned int CHUNK_SIZE_BITS = 64;
        constexpr unsigned int CHUNK_SIZE_BYTES = CHUNK_SIZE_BITS / 8;
        constexpr int CHUNKS_PER_BEAT = BEAT_WIDTH / CHUNK_SIZE_BITS;
        constexpr int CHUNK_INDEX_WIDTH = clog2(CHUNKS_PER_BEAT);
        static_assert(
            (1 << CHUNK_INDEX_WIDTH) == CHUNKS_PER_BEAT,
            "CHUNK_INDEX_WIDTH must be log2(CHUNKS_PER_BEAT)"
        );

        // Largest frame on the CMAC AXI Stream: L2 header plus MTU, without the FCS (appended and
        // stripped by the CMAC). The L2 padding is part of the MTU, so it adds nothing.
        constexpr int MAX_FRAME_BYTES = MAX_ETHERNET_HEADER_LENGTH_BYTES + ETHERNET_MTU;
        // Beats of the largest frame, e.g. for FIFOs holding a whole frame and frame descriptors
        constexpr int MAX_BEATS_PER_FRAME = (MAX_FRAME_BYTES * 8 + BEAT_WIDTH - 1) / BEAT_WIDTH;

        // "Enum" values
        static const ap_uint<FRAME_TYPE_WIDTH> FRAME_DATA = 0;
        static const ap_uint<FRAME_TYPE_WIDTH> FRAME_CREDIT_ACK = 1;
        static const ap_uint<FRAME_TYPE_WIDTH> FRAME_NACK = 2;
    }

    namespace helpers {
        /** Return the sum of all bitwidths */
        template<int... Ws>
        constexpr int widthSum() {
            // Leading 0 keeps the array non-empty for an empty pack
            const int widths[] = {0, Ws...};
            int sum = 0;
            for (int w : widths) sum += w;
            return sum;
        }

        /** Return the largest of the given values (0 for an empty pack). */
        template<uint64_t... Values>
        constexpr uint64_t maxOf() {
            // Leading 0 keeps the array non-empty for an empty pack
            const uint64_t values[] = {0, Values...};
            uint64_t result = 0;
            for (uint64_t v : values) result = v > result ? v : result;
            return result;
        }

        /** Return whether no value occurs twice among the given values. */
        template<uint64_t... Values>
        constexpr bool allDistinct() {
            // Leading 0 keeps the array non-empty for an empty pack, skipped by starting at 1
            const uint64_t values[] = {0, Values...};
            for (size_t i = 1; i < sizeof...(Values) + 1; ++i) {
                for (size_t j = i + 1; j < sizeof...(Values) + 1; ++j) {
                    if (values[i] == values[j]) return false;
                }
            }
            return true;
        }

        /** Reverse the byte order of `value`. Converts a multi-byte field between wire order (most
        * significant byte first, so in the lowest bits of an AXI Stream beat) and its numeric value,
        * in both directions. Pure wiring in hardware. */
        template<int W>
        ap_uint<W> swapBytes(const ap_uint<W>& value) {
            static_assert(W % 8 == 0, "Only whole bytes can be swapped");
            ap_uint<W> result;
            for (int i = 0; i < W / 8; ++i) {
                #pragma HLS unroll
                result.range(8 * i + 7, 8 * i) = value.range(W - 8 * i - 1, W - 8 * i - 8);
            }
            return result;
        }

        /** `swapBytes` of a slice such as `value.range(95, 48)`. A slice's width is only known at
        * runtime, so the field width W must be given explicitly: `swapBytes<48>(beat.range(95, 48))`.
        * Checks in csim that the slice is W bits wide. */
        template<int W, int PW, bool S>
        ap_uint<W> swapBytes(const ap_range_ref<PW, S>& value) {
#ifndef __SYNTHESIS__
            assert(value.length() == W && "Slice width does not match the field width W");
#endif
            return swapBytes<W>(ap_uint<W>(value));
        }

        /** Return bits upper..lower of `value` byte swapped, as a W-bit field. W must be given
        * explicitly: `swapBytes<48>(beat, 95, 48)`. */
        template<int W, int VW>
        ap_uint<W> swapBytes(const ap_uint<VW>& value, int upper, int lower) {
            return swapBytes<W>(value.range(upper, lower));
        }

        /** Base case of `packFrom`: nothing left to pack. */
        template<int BW, int Offset>
        void packFrom(ap_uint<BW>&) {}

        /** Pack the given data into `result`, starting at bit `Offset`. The first argument goes into
        * the lowest bits, each following one directly above the previous. */
        template<int BW, int Offset, int W, int... Ws>
        void packFrom(ap_uint<BW>& result, const ap_uint<W>& data, const ap_uint<Ws>&... datas) {
            static_assert(Offset + W <= BW, "Packed data exceeds the target width");
            result.range(Offset + W - 1, Offset) = data;
            packFrom<BW, Offset + W>(result, datas...);
        }

        /** Base case of `unpackFrom`: nothing left to unpack. */
        template<int BW, int Offset>
        void unpackFrom(const ap_uint<BW>&) {}

        /** Unpack `source`, starting at bit `Offset`, into the given data. Inverse of `packFrom`: the
        * first argument is read from the lowest bits, each following one directly above the previous. */
        template<int BW, int Offset, int W, int... Ws>
        void unpackFrom(const ap_uint<BW>& source, ap_uint<W>& data, ap_uint<Ws>&... datas) {
            static_assert(Offset + W <= BW, "Unpacked data exceeds the source width");
            data = source.range(Offset + W - 1, Offset);
            unpackFrom<BW, Offset + W>(source, datas...);
        }

        /** Serial-number comparison for wrapping FSNs and credit counters: true if
        * `a` is before `b`, i.e. 0 < (b - a) mod 2^W < 2^(W-1). */
        template<int W>
        bool serialBefore(const ap_uint<W>& a, const ap_uint<W>& b) {
            const ap_uint<W> difference = b - a;
            return difference != 0 && !difference[W - 1];
        }

    }

    /** A debug struct that can be switched on at comptime. */
    template<bool Debug> struct lsp_stat_counters_t;
    template <> struct lsp_stat_counters_t<true> {
        void countZeroChunkSegment() {
            zeroChunkSegments++;
        }

        private:
        ap_uint<64> zeroChunkSegments = 0;
    };
    template <> struct lsp_stat_counters_t<false> {
        void countZeroChunkSegment() {}
    };


    /** The peers of a node: peer I has the I-th MAC address, and I is its connection index. The single
    * source of this mapping, so that every block that handles several peers (e.g. the RX manager
    * dispatching by source MAC, the CTRL controller addressing a connection's peer) agrees on it. */
    template<uint64_t... MACs>
    struct lsp_peers_t {
        static constexpr size_t N = sizeof...(MACs);
        static_assert(N > 0, "At least one peer MAC address is required");
        static_assert(
            helpers::maxOf<MACs...>() < (1ULL << MAC::width),
            "Every peer MAC address must fit into 48 bits"
        );
        static_assert(helpers::allDistinct<MACs...>(), "Peer MAC addresses must be distinct");

        // Connection index; the value N marks an unknown MAC address
        using index_t = ap_uint<clog2(N + 1)>;
        static constexpr size_t INVALID_INDEX = N;

        /** Return the connection index of the given MAC address, or INVALID_INDEX if it is unknown. */
        static index_t getIndexFromMAC(const MAC& mac) {
            #pragma HLS inline
            static constexpr uint64_t lookupTable[N] = {MACs...};
            index_t idx = INVALID_INDEX;
            for (size_t i = 0; i < N; i++) {
                #pragma HLS unroll
                if (mac == lookupTable[i]) {
                    idx = i;
                }
            }
            return idx;
        }

        /** Return whether the given MAC address belongs to a peer. */
        static bool isKnownMAC(const MAC& mac) {
            #pragma HLS inline
            return getIndexFromMAC(mac) != INVALID_INDEX;
        }

        /** Return the MAC address of the peer with the given connection index (idx < N). */
        static MAC getMACFromIndex(const index_t& idx) {
            #pragma HLS inline
            static constexpr uint64_t lookupTable[N] = {MACs...};
            MAC mac = 0;
            for (size_t i = 0; i < N; i++) {
                #pragma HLS unroll
                if (idx == i) {
                    mac = lookupTable[i];
                }
            }
            return mac;
        }
    };

    namespace protocol {
        // PROTOCOL HEADER - Found on every frame
        struct lsp_header_t {
            ap_uint<3> version;
            ap_uint<lsp::constants::FRAME_TYPE_WIDTH> frameType;
            ap_uint<8> flags;
            ap_uint<lsp::constants::FSN_WIDTH> frameSequenceNumber;
            ap_uint<10> payloadChunks;
            ap_uint<8> _reserved;

            static constexpr int WIDTH = 64;
            static constexpr int VERSION_WIDTH = decltype(version)::width;
            static constexpr int FRAME_WIDTH = decltype(frameType)::width;
            static constexpr int FLAGS_WIDTH = decltype(flags)::width;
            static constexpr int FSN_WIDTH = decltype(frameSequenceNumber)::width;
            static constexpr int PAYLOAD_WIDTH = decltype(payloadChunks)::width;

            ap_uint<WIDTH> pack() const {
                ap_uint<WIDTH> header = 0;
                helpers::packFrom<WIDTH, 0>(
                    header, version, frameType, flags,
                    frameSequenceNumber, payloadChunks, _reserved
                );
                return header;
            }

            /** Unpack a header from its packed representation. Inverse of `pack()`.
            Expects header to start at bit 0. */
            static lsp_header_t unpack(const ap_uint<WIDTH>& bits) {
                lsp_header_t header;
                constexpr size_t versionOffset = 0;
                constexpr size_t frameTypeOffset = versionOffset + AP_WIDTH(version);
                constexpr size_t flagsOffset = frameTypeOffset + AP_WIDTH(frameType);
                constexpr size_t fsnOffset = flagsOffset + AP_WIDTH(flags);
                constexpr size_t payloadChunksOffset = fsnOffset + AP_WIDTH(frameSequenceNumber);
                constexpr size_t reservedOffset = payloadChunksOffset + AP_WIDTH(payloadChunks);
                header.version = bits.range(frameTypeOffset-1, versionOffset);
                header.frameType = bits.range(flagsOffset-1, frameTypeOffset);
                header.flags = bits.range(fsnOffset-1, flagsOffset);
                header.frameSequenceNumber = bits.range(payloadChunksOffset-1, fsnOffset);
                header.payloadChunks = bits.range(reservedOffset-1, payloadChunksOffset);
                header._reserved = bits.range(WIDTH-1, reservedOffset);
                return header;
            }

            // Check that the version can be stored in the header
            static_assert(
                clog2(CURRENT_VERSION)+1 <= AP_WIDTH(version),
                "Cannot accommodate version number in protocol header!"
            );
        };

        // Enforce that the fields fill exactly WIDTH bits
        static_assert(
            helpers::widthSum<
                AP_WIDTH(lsp_header_t::version),
                AP_WIDTH(lsp_header_t::frameType),
                AP_WIDTH(lsp_header_t::flags),
                AP_WIDTH(lsp_header_t::frameSequenceNumber),
                AP_WIDTH(lsp_header_t::payloadChunks),
                AP_WIDTH(lsp_header_t::_reserved)
            >() == lsp_header_t::WIDTH,
            "Header fields must fill exactly WIDTH bits"
        );


    }

    namespace data {
        using lsp::constants::BEAT_WIDTH;

        // Width of a data segment header, one chunk
        constexpr int SEGMENT_HEADER_WIDTH = 64;
        // Largest payload a segment can carry: a segment never crosses a beat, so at most
        // the rest of the beat behind its header. Every segment fits into this width, whatever its
        // stream's word width
        constexpr int MAX_SEGMENT_PAYLOAD_WIDTH = BEAT_WIDTH - SEGMENT_HEADER_WIDTH;

        // Data segment as it is on the wire
        template<int PayloadWidth>
        struct lsp_data_segment_t {
            // HEADER
            ap_uint<lsp::constants::STREAM_ID_WIDTH> stream; // ID of the sender stream
            ap_uint<4> validBytesLastChunk;  // Number of valid bytes in last chunk. Count from LSB
            ap_uint<12> chunks;              // Number of chunks (64 bit each) of the payload that follows
            ap_uint<38> _reserved;

            // PAYLOAD
            ap_uint<PayloadWidth> payload;

            static constexpr int HEADER_WIDTH = SEGMENT_HEADER_WIDTH;
            static constexpr int PAYLOAD_WIDTH = PayloadWidth;
            static constexpr int WIDTH = HEADER_WIDTH + PayloadWidth;

            ap_uint<WIDTH> pack() const {
                ap_uint<WIDTH> segment = 0;
                helpers::packFrom<WIDTH, 0>(segment, stream, validBytesLastChunk, chunks, _reserved, payload);
                return segment;
            }

            /** Unpack a segment from its packed representation. Inverse of `pack()`. */
            static lsp_data_segment_t unpack(const ap_uint<WIDTH>& bits) {
                lsp_data_segment_t segment;
                helpers::unpackFrom<WIDTH, 0>(
                    bits, segment.stream, segment.validBytesLastChunk, segment.chunks, segment._reserved,
                    segment.payload
                );
                return segment;
            }

            static_assert(
                helpers::widthSum<
                    AP_WIDTH(stream),
                    AP_WIDTH(validBytesLastChunk),
                    AP_WIDTH(chunks),
                    AP_WIDTH(_reserved)
                >() == HEADER_WIDTH,
                "Segment header fields must fill exactly HEADER_WIDTH bits"
            );
            static_assert(
                PayloadWidth % lsp::constants::CHUNK_SIZE_BITS == 0,
                "Payload width must be a multiple of the chunk size"
            );
        };

        /** Segment of the largest possible size: exactly one beat. The receiver unpacks every segment
        into this type, because its stream (and with it the word width) is only known at runtime. */
        using max_segment_t = lsp_data_segment_t<MAX_SEGMENT_PAYLOAD_WIDTH>;
        static_assert(max_segment_t::WIDTH == BEAT_WIDTH, "A max size segment must fill exactly one beat");

        /** Pack a beat for a data frame: the protocol header (if given), followed by the given segments,
        * starting at bit `Offset` (e.g. to leave room for Ethernet / Ethernet+UDP headers).
        * The reference variants only write their own range of `beat` and keep all other bits (e.g. an
        * already written Ethernet header). The returning variants start from an all-zero beat. */
        template<int Offset = 0, int... Ws>
        void packBeat(ap_uint<BEAT_WIDTH>& beat, const lsp::protocol::lsp_header_t& header, const lsp_data_segment_t<Ws>&... segments) {
            lsp::helpers::packFrom<BEAT_WIDTH, Offset>(beat, header.pack(), segments.pack()...);
        }
        template<int Offset = 0, int... Ws>
        void packBeat(ap_uint<BEAT_WIDTH>& beat, const lsp_data_segment_t<Ws>&... segments) {
            lsp::helpers::packFrom<BEAT_WIDTH, Offset>(beat, segments.pack()...);
        }
        template<int Offset = 0, int... Ws>
        ap_uint<BEAT_WIDTH> packBeat(const lsp::protocol::lsp_header_t& header, const lsp_data_segment_t<Ws>&... segments) {
            ap_uint<BEAT_WIDTH> beat = 0;
            packBeat<Offset>(beat, header, segments...);
            return beat;
        }
        template<int Offset = 0, int... Ws>
        ap_uint<BEAT_WIDTH> packBeat(const lsp_data_segment_t<Ws>&... segments) {
            ap_uint<BEAT_WIDTH> beat = 0;
            packBeat<Offset>(beat, segments...);
            return beat;
        }

        /** Offset of a data segment within a beat, in whole chunks. Prefer it over a bit offset whenever
        * segments are chunk aligned: aligning by it costs an 8:1 mux instead of a full barrel shifter.
        * Pass it as `chunk_offset_t{n}`: a plain `{n}` selects the bit offset overloads. */
        struct chunk_offset_t {
            ap_uint<clog2(constants::CHUNKS_PER_BEAT + 1)> index;
        };

        /** Unpack the max size segment whose header starts at bit `offset`. The runtime bit shift
        * synthesizes to a full barrel shifter, so only use this for segments that are not chunk aligned.
        * Bits behind the end of the beat read as 0. */
        inline max_segment_t unpackMaxSizeSegment(const ap_uint<BEAT_WIDTH>& data, int offset) {
            return max_segment_t::unpack(data >> offset);
        }

        /** Unpack the max size segment whose header starts at chunk `offset.index`. Bits behind the end
        * of the beat read as 0. */
        inline max_segment_t unpackMaxSizeSegment(const ap_uint<BEAT_WIDTH>& data, chunk_offset_t offset) {
            // One constant shift per possible chunk index, selected by an 8:1 mux
            ap_uint<BEAT_WIDTH> aligned = 0;
            for (int i = 0; i < lsp::constants::CHUNKS_PER_BEAT; ++i) {
#pragma HLS UNROLL
                if (offset.index == i) {
                    aligned = data >> (i * lsp::constants::CHUNK_SIZE_BITS);
                }
            }
            return max_segment_t::unpack(aligned);
        }


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
        struct lsp_stream_ring_buffer_t {
            static_assert(StreamWidth % 8 == 0, "The stream widths must be multiples full bytes!");
            static constexpr int PointerWidth = clog2(Depth + 1);

            ap_uint<StreamWidth> buffer[Depth];
            ap_uint<PointerWidth> readPointer = 0;
            ap_uint<PointerWidth> writePointer = 0;
            ap_uint<PointerWidth> bufferSize = 0;      // Current usage. Not capacity.
            ap_uint<clog2(StreamWidth + 1)> inWordPointer = 0;
            ap_uint<constants::CREDIT_LIMIT_WIDTH> creditLimit = Depth;
            ap_uint<constants::CREDIT_LIMIT_WIDTH> lastCreditLimitSent = Depth;

            // ap_uint's default constructor leaves the value uninitialized, so `= {}` would not zero it
            lsp_stream_ring_buffer_t() {
                for (size_t i = 0; i < Depth; ++i) {
                    buffer[i] = 0;
                }
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
            lsp_stream_ring_buffer_update_result updateFrom(const lsp_data_segment_t<PayloadWidth>& segment) {
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
                // Only read if the read pointer looks at valid data, and only advance if the write succeeded
                if (bufferSize > 0 && stream.write_nb(buffer[readPointer])) {
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
            using type = std::tuple<lsp_stream_ring_buffer_t<StreamIndices, StreamWidths, Depths>...>;
        };

        /** One ring buffer per stream: element I is the ring buffer for stream index I with the
        I-th width and the I-th depth, i.e. the stream index is the position in `StreamWidths`.
        `Depths` is an lsp_ring_depths_t. */
        template<typename Depths, int... StreamWidths>
        using lsp_stream_ring_buffers_t = typename _lsp_stream_ring_buffers<
            std::make_index_sequence<sizeof...(StreamWidths)>, Depths, StreamWidths...
        >::type;

        /** Implementation of visitRing: one call of `f` per ring, only the matching one enabled. */
        template<typename Rings, typename F, typename R, size_t... Is>
        R _visitRing(Rings& rings, unsigned index, F& f, R result, std::index_sequence<Is...>) {
            #pragma HLS inline
            // C++14 replacement for a fold expression: checks every ring index in order
            int expanded[] = {0, (index == Is ? (result = f(std::get<Is>(rings)), 0) : 0)...};
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

        /** Append a segment to the ring buffer of its stream: ring I (of an lsp_stream_ring_buffers_t)
        takes the segments whose stream ID is I, see #75 for the numbering. A stream ID without a ring
        returns UNKNOWN_STREAM. */
        template<typename... Rings, int PayloadWidth>
        lsp_stream_ring_buffer_update_result dispatchSegment(
            std::tuple<Rings...>& rings,
            const lsp_data_segment_t<PayloadWidth>& segment
        ) {
            #pragma HLS inline
            return visitRing(
                rings,
                segment.stream,
                [&segment](auto& ring) { return ring.updateFrom(segment); },
                lsp_stream_ring_buffer_update_result::UNKNOWN_STREAM
            );
        }

    }

    namespace credit {
        // CREDIT record
        struct lsp_credit_record_t {
            ap_uint<lsp::constants::STREAM_ID_WIDTH> stream;    // Stream ID
            ap_uint<lsp::constants::CREDIT_LIMIT_WIDTH> limit;  // Credit limit. Wraps around (mod 2^32)
            ap_uint<22> _reserved;

            static constexpr int WIDTH = 64;

            ap_uint<WIDTH> pack() const {
                ap_uint<WIDTH> record = 0;
                helpers::packFrom<WIDTH, 0>(record, stream, limit, _reserved);
                return record;
            }

            /** Unpack a credit record from its packed representation. Inverse of `pack()`. */
            static lsp_credit_record_t unpack(const ap_uint<WIDTH>& bits) {
                lsp_credit_record_t record;
                // TODO
                return record;
            }
        };
        static_assert(
            helpers::widthSum<
                AP_WIDTH(lsp_credit_record_t::stream),
                AP_WIDTH(lsp_credit_record_t::limit),
                AP_WIDTH(lsp_credit_record_t::_reserved)
            >() == lsp_credit_record_t::WIDTH,
            "Credit record fields must fill exactly WIDTH bits"
        );

        /** Number of credit records that fit into a beat behind `offset` bits, and behind the protocol
        * header if `withHeader` is set. */
        constexpr int creditsPerBeat(int offset, bool withHeader) {
            return (constants::BEAT_WIDTH - offset - (withHeader ? protocol::lsp_header_t::WIDTH : 0))
                / lsp_credit_record_t::WIDTH;
        }

        // Capacities without an offset
        constexpr int CREDITS_PER_BEAT = creditsPerBeat(0, false);
        constexpr int CREDITS_PER_HEADER_BEAT = creditsPerBeat(0, true);

        /** Pack credit records into a beat: the protocol header (if given), followed by the records,
        * starting at bit `Offset` (e.g. to leave room for Ethernet / Ethernet+UDP headers).
        * At most creditsPerBeat(Offset, withHeader) records fit. `used` gives the actual count of updated
        * credit records (used = 3 means slots 0-2 are written).
        * The reference variants only write the header and the used slots and keep all other bits of
        * `beat`. The returning variants start from an all-zero beat, so unused slots are 0. */
        template<int Offset = 0>
        void packBeat(
            ap_uint<lsp::constants::BEAT_WIDTH>& beat,
            const lsp_credit_record_t (& creditRecords)[creditsPerBeat(Offset, false)],
            const unsigned int used
        ) {
            constexpr int RW = lsp_credit_record_t::WIDTH;
            for (unsigned int i = 0; i < creditsPerBeat(Offset, false); ++i) {
#pragma HLS UNROLL
                if (i < used) {
                    beat.range(Offset + (i + 1) * RW - 1, Offset + i * RW) = creditRecords[i].pack();
                }
            }
        }
        template<int Offset = 0>
        void packBeat(
            ap_uint<lsp::constants::BEAT_WIDTH>& beat,
            const protocol::lsp_header_t& header,
            const lsp_credit_record_t (& creditRecords)[creditsPerBeat(Offset, true)],
            const unsigned int used
        ) {
            constexpr int HW = protocol::lsp_header_t::WIDTH;
            constexpr int RW = lsp_credit_record_t::WIDTH;
            beat.range(Offset + HW - 1, Offset) = header.pack();
            for (unsigned int i = 0; i < creditsPerBeat(Offset, true); ++i) {
#pragma HLS UNROLL
                if (i < used) {
                    beat.range(Offset + HW + (i + 1) * RW - 1, Offset + HW + i * RW) = creditRecords[i].pack();
                }
            }
        }
        template<int Offset = 0>
        ap_uint<lsp::constants::BEAT_WIDTH> packBeat(
            const lsp_credit_record_t (& creditRecords)[creditsPerBeat(Offset, false)],
            const unsigned int used
        ) {
            ap_uint<lsp::constants::BEAT_WIDTH> beat = 0;
            packBeat<Offset>(beat, creditRecords, used);
            return beat;
        }
        template<int Offset = 0>
        ap_uint<lsp::constants::BEAT_WIDTH> packBeat(
            const protocol::lsp_header_t& header,
            const lsp_credit_record_t (& creditRecords)[creditsPerBeat(Offset, true)],
            const unsigned int used
        ) {
            ap_uint<lsp::constants::BEAT_WIDTH> beat = 0;
            packBeat<Offset>(beat, header, creditRecords, used);
            return beat;
        }

        /** Unpack all CREDITS_PER_BEAT credit record slots of a beat without a protocol header.
        * Which slots are valid is up to the caller (e.g. from the count in the frame's header). */
        inline void unpackBeat(
            const ap_uint<lsp::constants::BEAT_WIDTH>& beat,
            lsp_credit_record_t (& creditRecords)[CREDITS_PER_BEAT]
        ) {
            // TODO
        }

        /** Unpack the protocol header and all CREDITS_PER_HEADER_BEAT credit record slots of a beat.
        * Which slots are valid is up to the caller (e.g. from the count in the header). */
        inline void unpackBeat(
            const ap_uint<lsp::constants::BEAT_WIDTH>& beat,
            protocol::lsp_header_t& header,
            lsp_credit_record_t (& creditRecords)[CREDITS_PER_HEADER_BEAT]
        ) {
            // TODO
        }
    }
}

#endif  // LSP_COMMON_HPP
