// Shared definitions for the LSP sender and receiver kernels.
#ifndef LSP_COMMON_HPP
#define LSP_COMMON_HPP

#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <tuple>
#include <utility>

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
        constexpr int CHUNKS_PER_BEAT = BEAT_WIDTH / CHUNK_SIZE_BITS;
        constexpr int CHUNK_INDEX_WIDTH = 3;  // log2(CHUNKS_PER_BEAT)
        static_assert(
            (1 << CHUNK_INDEX_WIDTH) == CHUNKS_PER_BEAT,
            "CHUNK_INDEX_WIDTH must be log2(CHUNKS_PER_BEAT)"
        );

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
            // TODO
        }


    }

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

            static constexpr int HEADER_WIDTH = 64;
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
                // TODO
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

        /** Pack a beat for a data frame: the protocol header, followed by the given segments. */
        template<int... Ws>
        ap_uint<BEAT_WIDTH> packBeat(const lsp::protocol::lsp_header_t& header, const lsp_data_segment_t<Ws>&... segments) {
            ap_uint<BEAT_WIDTH> beat = 0;
            // TODO: Replace the hardcoded 0 offset with
            // TODO: a configurable offset to consider Ethernet Header / Ethernet+UDP Headers
            lsp::helpers::packFrom<BEAT_WIDTH, 0>(beat, header.pack(), segments.pack()...);
            return beat;
        }

        /** Pack a beat for a data frame without a protocol header, containing only the given segments. */
        template<int... Ws>
        ap_uint<BEAT_WIDTH> packBeat(const lsp_data_segment_t<Ws>&... segments) {
            ap_uint<BEAT_WIDTH> beat = 0;
            // TODO: Replace the hardcoded 0 offset with
            // TODO: a configurable offset to consider Ethernet Header / Ethernet+UDP Headers
            lsp::helpers::packFrom<BEAT_WIDTH, 0>(beat, segments.pack()...);
            return beat;
        }

        /** Offset of a data segment within a beat, in whole chunks. Prefer it over a bit offset whenever
        * segments are chunk aligned: aligning by it costs an 8:1 mux instead of a full barrel shifter.
        * Pass it as `chunk_offset_t{n}`: a plain `{n}` selects the bit offset overloads. */
        struct chunk_offset_t {
            ap_uint<lsp::constants::CHUNK_INDEX_WIDTH> index;
        };

        /** Because the matching stream/width of the data is required at compile time but is only known
        at runtime, we first create the largest possible segment and then narrow it down
        by dispatching it into the correct slot in lsp_segment_set_t.

        `aligned` is expected to have the data segment header at bit 0!
        */
        inline lsp_data_segment_t<BEAT_WIDTH> _unpackMaxSizeSegment(const ap_uint<BEAT_WIDTH>& aligned) {
            using MaxSegment = lsp_data_segment_t<BEAT_WIDTH>;
            constexpr int STREAM_OFFSET = 0;
            constexpr int VALID_BYTES_OFFSET = STREAM_OFFSET + AP_WIDTH(MaxSegment::stream);
            constexpr int CHUNKS_OFFSET = VALID_BYTES_OFFSET + AP_WIDTH(MaxSegment::validBytesLastChunk);
            constexpr int RESERVED_OFFSET = CHUNKS_OFFSET + AP_WIDTH(MaxSegment::chunks);
            constexpr int PAYLOAD_OFFSET = RESERVED_OFFSET + AP_WIDTH(MaxSegment::_reserved);
            static_assert(
                PAYLOAD_OFFSET == MaxSegment::HEADER_WIDTH,
                "Field offsets must cover exactly the segment header"
            );

            MaxSegment segment;
            segment.stream = aligned.range(VALID_BYTES_OFFSET - 1, STREAM_OFFSET);
            segment.validBytesLastChunk = aligned.range(CHUNKS_OFFSET - 1, VALID_BYTES_OFFSET);
            segment.chunks = aligned.range(RESERVED_OFFSET - 1, CHUNKS_OFFSET);
            segment._reserved = aligned.range(PAYLOAD_OFFSET - 1, RESERVED_OFFSET);
            segment.payload = aligned.range(BEAT_WIDTH - 1, PAYLOAD_OFFSET);
            return segment;
        }

        /** Unpack the max size segment whose header starts at bit `offset`. The runtime bit shift
        * synthesizes to a full barrel shifter, so only use this for segments that are not chunk aligned. */
        inline lsp_data_segment_t<BEAT_WIDTH> unpackFromOffsetIntoMaxSizeSegment(
            const ap_uint<BEAT_WIDTH>& data,
            int offset
        ) {
            return _unpackMaxSizeSegment(data >> offset);
        }

        /** Unpack the max size segment whose header starts at chunk `offset.index`. */
        inline lsp_data_segment_t<BEAT_WIDTH> unpackFromOffsetIntoMaxSizeSegment(
            const ap_uint<BEAT_WIDTH>& data,
            chunk_offset_t offset
        ) {
            // One constant shift per possible chunk index, selected by an 8:1 mux
            ap_uint<BEAT_WIDTH> aligned = 0;
            for (int i = 0; i < lsp::constants::CHUNKS_PER_BEAT; ++i) {
#pragma HLS UNROLL
                if (offset.index == i) {
                    aligned = data >> (i * lsp::constants::CHUNK_SIZE_BITS);
                }
            }
            return _unpackMaxSizeSegment(aligned);
        }


        /** This struct is used as a bridge between the AXI streams from/to the user and the wire. Since
        * the number and width of streams is known at compile time, but the order and choice of data segments
        * only at runtime, we need a compatible dispatch between them.

        The `set` method receives already built data segments. This can be done because no matter
        which stream the data will be dispatched to, it can at max be as large as a BEAT. Thus we can simply
        construct the largest possible data segment and afterwards bit slice the correct range.
        */
        template<int... Ws>
        struct lsp_segment_set_t {
            std::tuple<lsp_data_segment_t<Ws>...> segments;

            // Flag to see if the data was read yet
            bool read[sizeof...(Ws)];

            /** Write the correct data segment. This is useful to narrow down the size from BEAT width
            to the actual width. Every stream on the user kernel has one matching lsp data segment. */
            void set(const lsp_data_segment_t<BEAT_WIDTH>& data) {
                setAll(data, std::make_index_sequence<sizeof...(Ws)>{});
            }

            /** The same as the set() overload with lsp_data_segment_t, but with the raw data.
            For data only beats set offset to 0, for beats with the protocol header, skip the protocol
            header by setting the offset accordingly. */
            void set(const ap_uint<BEAT_WIDTH>& data, int offset) {
                set(unpackFromOffsetIntoMaxSizeSegment(data, offset));
            }

            /** The same as set() with a bit offset, but with a chunk offset. Prefer this one for chunk
            aligned segments, see chunk_offset_t. */
            void set(const ap_uint<BEAT_WIDTH>& data, chunk_offset_t offset) {
                set(unpackFromOffsetIntoMaxSizeSegment(data, offset));
            }

            private:
            /** Set the data segment matching the stream ID from the data */
            template<std::size_t... Is>
            void setAll(const lsp_data_segment_t<BEAT_WIDTH>& data, std::index_sequence<Is...>) {
                // C++14 replacement for a fold expression: calls setIf<Is> for every index in order
                int expanded[] = {0, (setIf<Is>(data), 0)...};
                (void) expanded;
            }

            /** Set the data segment if the stream ID of the supplied segment matches. */
            template<std::size_t I>
            void setIf(const lsp_data_segment_t<BEAT_WIDTH>& data) {
                using Segment = std::tuple_element_t<I, decltype(segments)>;
                static_assert(Segment::WIDTH <= BEAT_WIDTH, "Data segment does not fit into one beat");
                auto& selected = std::get<I>(segments);
                if (data.stream == I) {
                    read[I] = false;
                    selected.stream = data.stream;
                    selected.validBytesLastChunk = data.validBytesLastChunk;
                    selected.chunks = data.chunks;
                    selected._reserved = data._reserved;
                    selected.payload = data.payload.range(Segment::PAYLOAD_WIDTH - 1, 0);
                }
            }
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
            ap_uint<StreamWidth> buffer[Depth];
            size_t readPointer = 0;
            size_t writePointer = 0;
            size_t inWordPointer = 0;

            // ap_uint's default constructor leaves the value uninitialized, so `= {}` would not zero it
            lsp_stream_ring_buffer_t() {
                for (size_t i = 0; i < Depth; ++i) {
                    buffer[i] = 0;
                }
            }

            /**
            Update the ring buffer with the data.
            The read/writes are BLOCKING - the credit system should prevent cases
            in which the receiver cannot accept data.
            */
            void updateFrom(const lsp_data_segment_t<StreamWidth>& segment) {



                // TODO



            }


        };

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

        constexpr int CREDITS_PER_BEAT = constants::BEAT_WIDTH / lsp_credit_record_t::WIDTH;
        constexpr int CREDITS_PER_HEADER_BEAT =
            (constants::BEAT_WIDTH - protocol::lsp_header_t::WIDTH) / lsp_credit_record_t::WIDTH;

        /** Pack credit records into a beat. At most CREDITS_PER_BEAT credits can be sent. `used` gives the actual count
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
        * At most CREDITS_PER_HEADER_BEAT credits can be sent. `used` gives the actual count
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
