// Helpers shared by the testbenches of the LSP receiver components. Testbench only, not synthesized.
#ifndef LSP_RECEIVER_TB_UTILS_HPP
#define LSP_RECEIVER_TB_UTILS_HPP

#include "lsp_receiver_utils.hpp"

#include <iostream>
#include <string>

// Test expression. If it doesnt hold, set flag to True (test failed) and print the supplied message.
#define assert_with_flag(expr, flag, msg) \
    do { if (!(expr)) { (flag) = true; std::cerr << __LINE__ << ": " << msg << std::endl; } } while (0)

// Compare actual with expected. If they differ, set flag to True (test failed) and print the supplied
// message with both values. Each argument is evaluated exactly once, so actual may be a stream read.
#define assert_eq_with_flag(actual, expected, flag, msg) \
    do { if (!lsp::receiver::checkEqual((actual), (expected), __LINE__, (msg))) (flag) = true; } while (0)

namespace lsp {
    namespace receiver {
        /** Implementation of assert_eq_with_flag: return whether actual == expected, print both
        values otherwise. */
        template<typename A, typename E>
        bool checkEqual(const A& actual, const E& expected, int line, const std::string& msg) {
            if (actual == expected) return true;
            std::cerr << line << ": " << msg << ": expected " << expected << " but got " << actual
                << std::endl;
            return false;
        }

        constexpr unsigned BEAT_BYTES = constants::BEAT_WIDTH / 8;

        /** Beat with the given payload and all side channels set. bytes is the number of valid bytes
        (from byte 0 upwards), only less than BEAT_BYTES on the last beat of a frame. */
        inline beat_t makeBeat(
            const ap_uint<constants::BEAT_WIDTH>& data,
            bool last,
            unsigned bytes = BEAT_BYTES
        ) {
            beat_t beat;
            beat.data = data;
            beat.keep = bytes >= BEAT_BYTES
                ? ~ap_uint<BEAT_BYTES>(0)
                : ap_uint<BEAT_BYTES>((ap_uint<BEAT_BYTES>(1) << bytes) - 1);
            beat.strb = beat.keep;
            beat.user = 0;
            beat.last = last;
            return beat;
        }
    }
}
#endif  // LSP_RECEIVER_TB_UTILS_HPP
