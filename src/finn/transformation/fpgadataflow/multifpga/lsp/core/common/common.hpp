#ifndef LSP_COMMON_HPP
#define LSP_COMMON_HPP

#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

#include "comptime.hpp"
#include "constants.hpp"
#include "helpers.hpp"

using MAC = ap_uint<lsp::constants::MAC_WIDTH>;

namespace lsp {
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

            static constexpr int WIDTH = lsp::constants::PROTOCOL_HEADER_WIDTH;
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
                clog2(lsp::constants::CURRENT_VERSION)+1 <= AP_WIDTH(version),
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

            static constexpr int HEADER_WIDTH = lsp::constants::SEGMENT_HEADER_WIDTH;
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
        using max_segment_t = lsp_data_segment_t<lsp::constants::MAX_SEGMENT_PAYLOAD_WIDTH>;
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
    }

    namespace credit {
        using lsp::constants::creditsPerBeat;
        using lsp::constants::CREDITS_PER_BEAT;
        using lsp::constants::CREDITS_PER_HEADER_BEAT;

        // CREDIT record
        struct lsp_credit_record_t {
            ap_uint<lsp::constants::STREAM_ID_WIDTH> stream;    // Stream ID
            ap_uint<lsp::constants::CREDIT_LIMIT_WIDTH> limit;  // Credit limit. Wraps around (mod 2^32)
            ap_uint<22> _reserved;

            static constexpr int WIDTH = lsp::constants::CREDIT_RECORD_WIDTH;

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
