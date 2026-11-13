// Compile time helpers of the LSP kernels: bit width arithmetic and checks on template parameter
// packs. No hardware, only used in constant expressions and static_asserts.
#ifndef LSP_COMPTIME_HPP
#define LSP_COMPTIME_HPP

#include <cstddef>
#include <cstdint>

// Bit width of an ap_uint / ap_int member or variable
#define AP_WIDTH(x) (decltype(x)::width)

namespace lsp {
    /** Ceiling of log2(n): the number of bits needed to tell n different values apart
    * (clog2(1) == 0). To store the value n itself, use clog2(n + 1). */
    constexpr int clog2(unsigned long long n) {
        int bits = 0;
        while ((1ULL << bits) < n) ++bits;
        return bits;
    }
    static_assert(clog2(1) == 0 && clog2(2) == 1 && clog2(8) == 3 && clog2(9) == 4, "clog2 is broken");

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
    }
}

#endif  // LSP_COMPTIME_HPP
