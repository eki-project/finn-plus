// Hardware helpers of the LSP kernels: byte order conversion, bit packing of wire formats and serial
// number comparison. The compile time helpers are in comptime.hpp.
#ifndef LSP_HELPERS_HPP
#define LSP_HELPERS_HPP

#include "constants.hpp"
#include <ap_int.h>
#include <cassert>

namespace lsp {

    using MAC = ap_uint<lsp::constants::MAC_WIDTH>;

    namespace debug {
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
    }

    namespace helpers {
        /** Offset of a data segment within a beat, in whole chunks. Prefer it over a bit offset whenever
        * segments are chunk aligned: aligning by it costs an 8:1 mux instead of a full barrel shifter.
        * Pass it as `chunk_offset_t{n}`: a plain `{n}` selects the bit offset overloads. */
        struct chunk_offset_t {
            ap_uint<clog2(constants::CHUNKS_PER_BEAT + 1)> index;
        };

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

    /** The peers of a node: peer I has the I-th MAC address, and I is its connection index. The single
    * source of this mapping, so that every block that handles several peers (e.g. the RX manager
    * dispatching by source MAC, the CTRL controller addressing a connection's peer) agrees on it. */
    template<uint64_t... MACs>
    struct mac_index_lookup_t {
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
}

#endif  // LSP_HELPERS_HPP
