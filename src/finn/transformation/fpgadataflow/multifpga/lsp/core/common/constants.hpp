// Constants and derived sizes of the LSP kernels, from the basic units (beat, chunk) up to the
// frame, header, segment and credit record sizes built from them. All widths are in bits unless the
// name says otherwise.
//
// Configurable constants: a constant NAME whose value is taken from the macro LSP_NAME is
// configurable. The #ifndef next to it sets the default used for development, and a build overrides
// it by defining the macro, e.g. -DLSP_ETHERNET_MTU=9000 (the CMake build: see CMakeLists.txt, an
// HLS build: the -cflags of its source files). Everything derived from it follows automatically.
#ifndef LSP_CONSTANTS_HPP
#define LSP_CONSTANTS_HPP

#include <ap_int.h>
#include <cstddef>

#include "comptime.hpp"

namespace lsp {
    namespace constants {
        // ---------------------------------------------------------------------------------------
        // Protocol version
        // ---------------------------------------------------------------------------------------

        constexpr size_t CURRENT_VERSION = 0;

        // ---------------------------------------------------------------------------------------
        // Ethernet
        // ---------------------------------------------------------------------------------------

        constexpr int MAC_WIDTH = 48;
        constexpr size_t MAX_ETHERNET_HEADER_LENGTH_BYTES = 22;
        // Largest L2 payload: 1500 B, or e.g. 9000 B with jumbo frames
#ifndef LSP_ETHERNET_MTU
#define LSP_ETHERNET_MTU 1500
#endif
        constexpr size_t ETHERNET_MTU = LSP_ETHERNET_MTU;
        static_assert(ETHERNET_MTU >= 46, "ETHERNET_MTU must be at least the minimum L2 payload of 46 B");

        // ---------------------------------------------------------------------------------------
        // Beats and chunks
        // ---------------------------------------------------------------------------------------

        // One AXI Stream word of the CMAC
        constexpr int BEAT_WIDTH = 512;

        // Unit of all LSP sizes on the wire
        constexpr unsigned int CHUNK_SIZE_BITS = 64;
        constexpr unsigned int CHUNK_SIZE_BYTES = CHUNK_SIZE_BITS / 8;
        constexpr int CHUNKS_PER_BEAT = BEAT_WIDTH / CHUNK_SIZE_BITS;
        constexpr int CHUNK_INDEX_WIDTH = clog2(CHUNKS_PER_BEAT);
        static_assert(
            (1 << CHUNK_INDEX_WIDTH) == CHUNKS_PER_BEAT,
            "CHUNK_INDEX_WIDTH must be log2(CHUNKS_PER_BEAT)"
        );

        // ---------------------------------------------------------------------------------------
        // Frames
        // ---------------------------------------------------------------------------------------

        // Largest frame on the CMAC AXI Stream: L2 header plus MTU, without the FCS (appended and
        // stripped by the CMAC). The L2 padding is part of the MTU, so it adds nothing.
        constexpr int MAX_FRAME_BYTES = MAX_ETHERNET_HEADER_LENGTH_BYTES + ETHERNET_MTU;
        // Beats of the largest frame, e.g. for FIFOs holding a whole frame and frame descriptors
        constexpr int MAX_BEATS_PER_FRAME = (MAX_FRAME_BYTES * 8 + BEAT_WIDTH - 1) / BEAT_WIDTH;

        // ---------------------------------------------------------------------------------------
        // Protocol header (found on every frame)
        // ---------------------------------------------------------------------------------------

        constexpr int PROTOCOL_HEADER_WIDTH = 64;
        constexpr int FRAME_TYPE_WIDTH = 3;
        constexpr int FSN_WIDTH = 32;

        // Frame types ("enum" values of the header's frame type field)
        static const ap_uint<FRAME_TYPE_WIDTH> FRAME_DATA = 0;
        static const ap_uint<FRAME_TYPE_WIDTH> FRAME_CREDIT_ACK = 1;
        static const ap_uint<FRAME_TYPE_WIDTH> FRAME_NACK = 2;

        // ---------------------------------------------------------------------------------------
        // Data segments
        // ---------------------------------------------------------------------------------------

        constexpr int STREAM_ID_WIDTH = 10;
        // Width of a data segment header, one chunk
        constexpr int SEGMENT_HEADER_WIDTH = 64;
        // Largest payload a segment can carry: a segment never crosses a beat, so at most
        // the rest of the beat behind its header. Every segment fits into this width, whatever its
        // stream's word width
        constexpr int MAX_SEGMENT_PAYLOAD_WIDTH = BEAT_WIDTH - SEGMENT_HEADER_WIDTH;

        // ---------------------------------------------------------------------------------------
        // Credit records
        // ---------------------------------------------------------------------------------------

        // Credit limit of a stream. Wraps around (mod 2^CREDIT_LIMIT_WIDTH)
        constexpr int CREDIT_LIMIT_WIDTH = 32;
        constexpr int CREDIT_RECORD_WIDTH = 64;

        /** Number of credit records that fit into a beat behind `offset` bits, and behind the protocol
        * header if `withHeader` is set. */
        constexpr int creditsPerBeat(int offset, bool withHeader) {
            return (BEAT_WIDTH - offset - (withHeader ? PROTOCOL_HEADER_WIDTH : 0))
                / CREDIT_RECORD_WIDTH;
        }

        // Capacities without an offset
        constexpr int CREDITS_PER_BEAT = creditsPerBeat(0, false);
        constexpr int CREDITS_PER_HEADER_BEAT = creditsPerBeat(0, true);
    }
}

#endif  // LSP_CONSTANTS_HPP
