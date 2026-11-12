// Helpers shared by the testbenches of the LSP receiver components. Testbench only, not synthesized.
#ifndef LSP_RECEIVER_TB_UTILS_HPP
#define LSP_RECEIVER_TB_UTILS_HPP

#include "lsp_receiver_utils.hpp"

// Test expression. If it doesnt hold, set flag to True (test failed) and print the supplied message.
#define assert_with_flag(expr, flag, msg) \
    do { if (!(expr)) { (flag) = true; std::cerr << __LINE__ << ": " << msg << std::endl; } } while (0)

namespace lsp {
    namespace receiver {
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
