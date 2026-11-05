// Shared definitions for the LSP sender and receiver kernels.
#ifndef LSP_COMMON_HPP
#define LSP_COMMON_HPP

#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

#define AP_WIDTH(x) (decltype(x)::width)

namespace lsp {
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

        // "Enum" values
        static const ap_uint<FRAME_TYPE_WIDTH> FRAME_DATA = 0;
        static const ap_uint<FRAME_TYPE_WIDTH> FRAME_CREDIT_ACK = 1;
        static const ap_uint<FRAME_TYPE_WIDTH> FRAME_NACK = 2;
    }

    namespace helpers {
        /** Return the sum of all bitwidths */
        template<int... Ws>
        constexpr int widthSum() {
            return (0 + ... + Ws);
        }

        /** Pack the given data into `result`, starting at bit `Offset`. The first argument goes into
        * the lowest bits, each following one directly above the previous. */
        template<int BW, int Offset, int W, int... Ws>
        void packFrom(ap_uint<BW>& result, const ap_uint<W>& data, const ap_uint<Ws>&... datas) {
            static_assert(Offset + W <= BW, "Packed data exceeds the target width");
            result.range(Offset + W - 1, Offset) = data;
            if constexpr (sizeof...(Ws) > 0) {
                packFrom<BW, Offset + W>(result, datas...);
            }
        }

        /** Unpack `source`, starting at bit `Offset`, into the given data. Inverse of `packFrom`: the
        * first argument is read from the lowest bits, each following one directly above the previous. */
        template<int BW, int Offset, int W, int... Ws>
        void unpackFrom(const ap_uint<BW>& source, ap_uint<W>& data, ap_uint<Ws>&... datas) {
            static_assert(Offset + W <= BW, "Unpacked data exceeds the source width");
            // TODO
        }
    }

    namespace protocol {
        // PROTOCOL HEADER - Found on every frame
        struct lsp_header_t {
            ap_uint<3> version;                         // Protocol version
            ap_uint<lsp::constants::FRAME_TYPE_WIDTH> frameType;        // DATA, CREDIT_ACK, NACK
            ap_uint<8> flags;                           // RESERVED
            ap_uint<lsp::constants::FSN_WIDTH> frameSequenceNumber;     // Sequence number of this frame ensure arrival in order
            ap_uint<10> payloadChunks;                  // CURRENTLY UNUSED
            ap_uint<8> _reserved;

            static constexpr int WIDTH = 64;

            ap_uint<WIDTH> pack() const {
                ap_uint<WIDTH> header = 0;
                helpers::packFrom<WIDTH, 0>(
                    header, version, frameType, flags,
                    frameSequenceNumber, payloadChunks, _reserved
                );
                return header;
            }

            /** Unpack a header from its packed representation. Inverse of `pack()`. */
            static lsp_header_t unpack(const ap_uint<WIDTH>& bits) {
                lsp_header_t header;
                // TODO
                return header;
            }
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
            >() == lsp_header_t::WIDTH
        );
    }

    namespace data {
        using lsp::constants::BEAT_WIDTH;

        // Header for the DATA type frame
        template<int PayloadWidth>
        struct lsp_data_segment_t {
            // HEADER
            ap_uint<lsp::constants::STREAM_ID_WIDTH> stream; // ID of the sender stream
            ap_uint<4> validBytesLastChunk;  // Number of valid bytes in last chunk. Count from LSB
            ap_uint<12> chunks;              // Number of chunks (64 bit) payload that follow
            ap_uint<38> _reserved;

            // PAYLOAD
            ap_uint<PayloadWidth> payload;

            static constexpr int HEADER_WIDTH = 64;
            static constexpr int WIDTH = HEADER_WIDTH + PayloadWidth;

            ap_uint<WIDTH> pack() const {
                ap_uint<WIDTH> segment = 0;
                helpers::packFrom<WIDTH, 0>(segment, stream, validBytesLastChunk, chunks, _reserved, payload);
                return segment;
            }

            /** Unpack a segment from its packed representation. Inverse of `pack()`. */
            static lsp_data_segment_t unpack(const ap_uint<WIDTH>& bits) {
                lsp_data_segment_t segment;
                // TODO
                return segment;
            }

            static_assert(
                helpers::widthSum<
                    AP_WIDTH(stream),
                    AP_WIDTH(validBytesLastChunk),
                    AP_WIDTH(chunks),
                    AP_WIDTH(_reserved)
                >() == HEADER_WIDTH
            );
            static_assert(PayloadWidth % lsp::constants::CHUNK_SIZE_BITS == 0);
        };

        /** Pack a beat for a data frame: the protocol header, followed by the given segments. */
        template<int... Ws>
        ap_uint<BEAT_WIDTH> packBeat(const lsp::protocol::lsp_header_t& header, const lsp_data_segment_t<Ws>&... segments) {
            ap_uint<BEAT_WIDTH> beat = 0;
            lsp::helpers::packFrom<BEAT_WIDTH, 0>(beat, header.pack(), segments.pack()...);
            return beat;
        }

        /** Pack a beat for a data frame without a protocol header, containing only the given segments. */
        template<int... Ws>
        ap_uint<BEAT_WIDTH> packBeat(const lsp_data_segment_t<Ws>&... segments) {
            ap_uint<BEAT_WIDTH> beat = 0;
            lsp::helpers::packFrom<BEAT_WIDTH, 0>(beat, segments.pack()...);
            return beat;
        }

        /** Unpack a data frame beat into the protocol header and the given segments. Inverse of
        * `packBeat(header, segments...)`. The segment widths must be known at compile time. */
        template<int... Ws>
        void unpackBeat(
            const ap_uint<BEAT_WIDTH>& beat,
            lsp::protocol::lsp_header_t& header,
            lsp_data_segment_t<Ws>&... segments
        ) {
            // TODO
        }

        /** Unpack a data frame beat without a protocol header into the given segments. Inverse of
        * `packBeat(segments...)`. The segment widths must be known at compile time. */
        template<int... Ws>
        void unpackBeat(const ap_uint<BEAT_WIDTH>& beat, lsp_data_segment_t<Ws>&... segments) {
            // TODO
        }
    }


    namespace credit {
        // CREDIT record
        struct lsp_credit_record_t {
            ap_uint<lsp::constants::STREAM_ID_WIDTH> stream;    // Stream ID
            ap_uint<lsp::constants::CREDIT_LIMIT_WIDTH> limit;  // Credit limit. Rolls around (mod 2^32)
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
            >() == lsp_credit_record_t::WIDTH
        );

        constexpr int CREDITS_PER_BEAT = constants::BEAT_WIDTH / lsp_credit_record_t::WIDTH;
        constexpr int CREDITS_PER_HEADER_BEAT =
            (constants::BEAT_WIDTH - protocol::lsp_header_t::WIDTH) / lsp_credit_record_t::WIDTH;

        /** Pack credit records into a beat. At max CREDITS_PER_BEAT credits can be sent. `used` gives the actual count
        * of updated credit records. (used = 3 means 0,1,2 were written) The rest is 0-padded. */
        inline ap_uint<lsp::constants::BEAT_WIDTH> packBeat(
            const lsp_credit_record_t (& creditRecords)[CREDITS_PER_BEAT],
            const unsigned int used
        ) {
            constexpr int RW = lsp_credit_record_t::WIDTH;
            ap_uint<lsp::constants::BEAT_WIDTH> beat = 0;
            for (unsigned int i = 0; i < CREDITS_PER_BEAT; ++i) {
#pragma HLS UNROLL
                if (i < used) {
                    beat.range((i + 1) * RW - 1, i * RW) = creditRecords[i].pack();
                }
            }
            return beat;
        }

        /** Pack credit records into a beat with the protocol header.
        * At max CREDITS_PER_HEADER_BEAT credits can be sent. `used` gives the actual count
        * of updated credit records (used = 3 means 0,1,2 were written). The rest is 0-padded. */
        inline ap_uint<lsp::constants::BEAT_WIDTH> packBeat(
            const protocol::lsp_header_t& header,
            const lsp_credit_record_t (& creditRecords)[CREDITS_PER_HEADER_BEAT],
            const unsigned int used
        ) {
            constexpr int HW = protocol::lsp_header_t::WIDTH;
            constexpr int RW = lsp_credit_record_t::WIDTH;
            ap_uint<lsp::constants::BEAT_WIDTH> beat = 0;
            beat.range(HW - 1, 0) = header.pack();
            for (unsigned int i = 0; i < CREDITS_PER_HEADER_BEAT; ++i) {
#pragma HLS UNROLL
                if (i < used) {
                    beat.range(HW + (i + 1) * RW - 1, HW + i * RW) = creditRecords[i].pack();
                }
            }
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
