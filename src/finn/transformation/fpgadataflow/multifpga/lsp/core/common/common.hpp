#ifndef LSP_COMMON_HPP
#define LSP_COMMON_HPP

#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

#include "comptime.hpp"
#include "constants.hpp"
#include "helpers.hpp"


namespace lsp {
    namespace utils {
        template <int... Ws>
        struct stream_widths_t {};

        template <size_t... Ds>
        struct ring_depths_t {};

        enum ethertype_or_tag : unsigned int {
            LOCAL_ETHTYPE_1 = 0x88B5,
            LOCAL_ETHTYPE_2 = 0x88B6,
            IEEE_802_1_Q = 0x8100,
            IEEE_802_1_AD = 0x88A8,
            IEEE_802_1_AD_LEGACY = 0x9100
        };

        /** Return a beat with a prefixed L2 header prepared. */
        template<ethertype_or_tag Tag, uint64_t SourceMAC, uint64_t TargetMAC>
        ap_uint<constants::BEAT_WIDTH> ethernetHeader() {
            static_assert(
                constants::BEAT_WIDTH >= 192,
                "BEAT_WIDTH needs to be at least 192 bits wide to fit all L2 header variants"
            );
            ap_uint<constants::BEAT_WIDTH> beat = 0;

            // Set ethertype or tag
            beat.range(111, 96) = helpers::swapBytes(ap_uint<16>(Tag));

            // Set MAC addresses
            beat.range(95, 48) = helpers::swapBytes(ap_uint<48>(SourceMAC));
            beat.range(47, 0) = helpers::swapBytes(ap_uint<48>(TargetMAC));

            return beat;
        }

        /** Bit offset of the LSP header in beat 0, from the TPID or EtherType of the L2 header. */
        inline ap_uint<clog2(1 + constants::BEAT_WIDTH)> getOffsetFromEthertype(const ap_uint<constants::BEAT_WIDTH>& beat) {
            #pragma HLS inline
            // Read in wire byte order, so the constants are byte swapped: 0x0081 = 0x8100, 0xA888 = 0x88A8, 0x0091 = 0x9100
            ap_uint<16> ethertypeOrTag = beat.range(111, 96);
            if (ethertypeOrTag == 0x0081) {
                return 192; // 144 padded to 64b multiples
            } else if (ethertypeOrTag == 0xA888 || ethertypeOrTag == 0x0091) {
                return 192; // 176 padded to 64b multiples
            } else {
                return 128; // 112 padded to 64b multiples
            }
        }

        /** Bit offset of the LSP header in beat 0 for a known EtherType or TPID. Constexpr, so that the
        result can be used as a template argument. */
        constexpr int getOffsetFromEthertype(ethertype_or_tag tag) {
            switch (tag) {
                case IEEE_802_1_Q:
                    return 192; // 144 padded to 64b multiples
                case IEEE_802_1_AD_LEGACY:
                case IEEE_802_1_AD:
                    return 192; // 176 padded to 64b multiples
                case LOCAL_ETHTYPE_1:
                case LOCAL_ETHTYPE_2:
                    return 128; // 112 padded to 64b multiples
            }
            // Only reached for a value outside the enum. A negative offset fails as a template argument
            // or bit range
            return -1;
        }
    }

    namespace messages {
        using lsp::constants::BEAT_WIDTH;
        using lsp::constants::creditsPerBeat;
        using lsp::constants::CREDITS_PER_BEAT;
        using lsp::constants::CREDITS_PER_HEADER_BEAT;

        struct header_t {
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
            static header_t unpack(const ap_uint<WIDTH>& bits) {
                header_t header;
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
        };


        template<int PayloadWidth>
        struct data_segment_t {
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
            static data_segment_t unpack(const ap_uint<WIDTH>& bits) {
                data_segment_t segment;
                helpers::unpackFrom<WIDTH, 0>(
                    bits, segment.stream, segment.validBytesLastChunk, segment.chunks, segment._reserved,
                    segment.payload
                );
                return segment;
            }

            /** Unpack the max size segment whose header starts at bit `offset`. The runtime bit shift
            * synthesizes to a full barrel shifter, so only use this for segments that are not chunk aligned.
            * Bits behind the end of the beat read as 0. */
            static inline data_segment_t<constants::MAX_SEGMENT_PAYLOAD_WIDTH> unpackMaxSizeSegment(const ap_uint<BEAT_WIDTH>& data, int offset) {
                return data_segment_t<constants::MAX_SEGMENT_PAYLOAD_WIDTH>::unpack(data >> offset);
            }

            /** Unpack the max size segment whose header starts at chunk `offset.index`. Bits behind the end
            * of the beat read as 0. */
            inline static data_segment_t<constants::MAX_SEGMENT_PAYLOAD_WIDTH> unpackMaxSizeSegment(const ap_uint<BEAT_WIDTH>& data, helpers::chunk_offset_t offset) {
                // One constant shift per possible chunk index, selected by an 8:1 mux
                ap_uint<BEAT_WIDTH> aligned = 0;
                for (int i = 0; i < lsp::constants::CHUNKS_PER_BEAT; ++i) {
    #pragma HLS UNROLL
                    if (offset.index == i) {
                        aligned = data >> (i * lsp::constants::CHUNK_SIZE_BITS);
                    }
                }
                return data_segment_t<constants::MAX_SEGMENT_PAYLOAD_WIDTH>::unpack(aligned);
            }

            // Depends on the template parameter, so it is checked per instantiation
            static_assert(
                PayloadWidth % lsp::constants::CHUNK_SIZE_BITS == 0,
                "Payload width must be a multiple of the chunk size"
            );
        };


        struct credit_record_t {
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
            static credit_record_t unpack(const ap_uint<WIDTH>& bits) {
                credit_record_t record;
                // TODO
                return record;
            }
        };

    }

    namespace beat {
        using namespace constants;
        using namespace messages;

        /** Pack a beat for a data frame: the protocol header (if given), followed by the given segments,
        * starting at bit `Offset` (e.g. to leave room for Ethernet / Ethernet+UDP headers).
        * The reference variants only write their own range of `beat` and keep all other bits (e.g. an
        * already written Ethernet header). The returning variants start from an all-zero beat. */
        template<int Offset = 0, int... Ws>
        void packBeat(ap_uint<BEAT_WIDTH>& beat, const header_t& header, const data_segment_t<Ws>&... segments) {
            lsp::helpers::packFrom<BEAT_WIDTH, Offset>(beat, header.pack(), segments.pack()...);
        }
        template<int Offset = 0, int... Ws>
        void packBeat(ap_uint<BEAT_WIDTH>& beat, const data_segment_t<Ws>&... segments) {
            lsp::helpers::packFrom<BEAT_WIDTH, Offset>(beat, segments.pack()...);
        }
        template<int Offset = 0, int... Ws>
        ap_uint<BEAT_WIDTH> packBeat(const header_t& header, const data_segment_t<Ws>&... segments) {
            ap_uint<BEAT_WIDTH> beat = 0;
            packBeat<Offset>(beat, header, segments...);
            return beat;
        }
        template<int Offset = 0, int... Ws>
        ap_uint<BEAT_WIDTH> packBeat(const data_segment_t<Ws>&... segments) {
            ap_uint<BEAT_WIDTH> beat = 0;
            packBeat<Offset>(beat, segments...);
            return beat;
        }


        /** Pack credit records into a beat: the protocol header (if given), followed by the records,
        * starting at bit `Offset` (e.g. to leave room for Ethernet / Ethernet+UDP headers).
        * At most creditsPerBeat(Offset, withHeader) records fit. `used` gives the actual count of updated
        * credit records (used = 3 means slots 0-2 are written).
        * The reference variants only write the header and the used slots and keep all other bits of
        * `beat`. The returning variants start from an all-zero beat, so unused slots are 0. */
        template<int Offset = 0>
        void packBeat(
            ap_uint<lsp::constants::BEAT_WIDTH>& beat,
            const credit_record_t (& creditRecords)[creditsPerBeat(Offset, false)],
            const unsigned int used
        ) {
            constexpr int RW = credit_record_t::WIDTH;
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
            const header_t& header,
            const credit_record_t (& creditRecords)[creditsPerBeat(Offset, true)],
            const unsigned int used
        ) {
            constexpr int HW = header_t::WIDTH;
            constexpr int RW = credit_record_t::WIDTH;
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
            const credit_record_t (& creditRecords)[creditsPerBeat(Offset, false)],
            const unsigned int used
        ) {
            ap_uint<lsp::constants::BEAT_WIDTH> beat = 0;
            packBeat<Offset>(beat, creditRecords, used);
            return beat;
        }
        template<int Offset = 0>
        ap_uint<lsp::constants::BEAT_WIDTH> packBeat(
            const header_t& header,
            const credit_record_t (& creditRecords)[creditsPerBeat(Offset, true)],
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
            credit_record_t (& creditRecords)[CREDITS_PER_BEAT]
        ) {
            // TODO
        }

        /** Unpack the protocol header and all CREDITS_PER_HEADER_BEAT credit record slots of a beat.
        * Which slots are valid is up to the caller (e.g. from the count in the header). */
        inline void unpackBeat(
            const ap_uint<lsp::constants::BEAT_WIDTH>& beat,
            header_t& header,
            credit_record_t (& creditRecords)[CREDITS_PER_HEADER_BEAT]
        ) {
            // TODO
        }

    }

    // Compile time checks of the message layouts
    namespace messages {
        // Check that the version can be stored in the header
        static_assert(
            clog2(lsp::constants::CURRENT_VERSION)+1 <= AP_WIDTH(header_t::version),
            "Cannot accommodate version number in protocol header!"
        );

        // Enforce that the header fields fill exactly WIDTH bits
        static_assert(
            helpers::widthSum<
                AP_WIDTH(header_t::version),
                AP_WIDTH(header_t::frameType),
                AP_WIDTH(header_t::flags),
                AP_WIDTH(header_t::frameSequenceNumber),
                AP_WIDTH(header_t::payloadChunks),
                AP_WIDTH(header_t::_reserved)
            >() == header_t::WIDTH,
            "Header fields must fill exactly WIDTH bits"
        );

        // The segment header fields do not depend on the payload width, so checking one instantiation
        // covers all of them
        using checked_segment_t = data_segment_t<lsp::constants::MAX_SEGMENT_PAYLOAD_WIDTH>;
        static_assert(
            helpers::widthSum<
                AP_WIDTH(checked_segment_t::stream),
                AP_WIDTH(checked_segment_t::validBytesLastChunk),
                AP_WIDTH(checked_segment_t::chunks),
                AP_WIDTH(checked_segment_t::_reserved)
            >() == checked_segment_t::HEADER_WIDTH,
            "Segment header fields must fill exactly HEADER_WIDTH bits"
        );

        static_assert(
            helpers::widthSum<
                AP_WIDTH(credit_record_t::stream),
                AP_WIDTH(credit_record_t::limit),
                AP_WIDTH(credit_record_t::_reserved)
            >() == credit_record_t::WIDTH,
            "Credit record fields must fill exactly WIDTH bits"
        );
    }
}

#endif  // LSP_COMMON_HPP
