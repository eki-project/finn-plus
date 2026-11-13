// RX manager of the LSP receiver kernel: dispatches each frame by its source MAC address to the
// connection of that peer.
#ifndef LSP_RECEIVER_MANAGER_HPP
#define LSP_RECEIVER_MANAGER_HPP

#include "lsp_common.hpp"
#include "lsp_receiver_utils.hpp"

namespace lsp {
    namespace receiver {
        enum class manager_rx_state : unsigned char {
            EXPECT_FRAME_START,
            EXPECT_BEAT
        };

        template<typename Peers>
        void manager(
            hls::stream<beat_t, CMAC_BEAT_BUFFER>& ingress,
            hls::stream<beat_t> (&macStreams)[Peers::N]
        ) {
            // State kept across iterations
            static manager_rx_state state = manager_rx_state::EXPECT_FRAME_START;
            static MAC sourceMAC;

            // Create the dispatcher
            mac_dispatcher<Peers> dispatcher(macStreams);

            // Get the MAC and dispatch until a new frame arrives
            beat_t data = ingress.read();
            if (state == manager_rx_state::EXPECT_FRAME_START) {
                // Read the source MAC so that we can dispatch to the correct kernel (bytes 6-11)
                sourceMAC = helpers::swapBytes<48>(data.data, 95, 48);
            }
            dispatcher.forwardData(sourceMAC, data);
            if (data.last) {
                state = manager_rx_state::EXPECT_FRAME_START;
            } else {
                state = manager_rx_state::EXPECT_BEAT;
            }
        }
    }
}
#endif  // LSP_RECEIVER_MANAGER_HPP
