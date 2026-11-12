// RX manager of the LSP receiver kernel: dispatches each frame by its source MAC address to the
// connection of that peer.
#ifndef LSP_RECEIVER_MANAGER_HPP
#define LSP_RECEIVER_MANAGER_HPP

#include "lsp_receiver_utils.hpp"

namespace lsp {
    namespace receiver {
        // TODO: State machine that looks up the source MAC once per frame (beat 0) and forwards the
        // rest of the frame with it.
        template<typename Peers>
        void manager(
            hls::stream<beat_t, CMAC_BEAT_BUFFER>& ingress,
            hls::stream<beat_t> (&macStreams)[Peers::N]
        ) {
            // Create the dispatcher
            mac_dispatcher<Peers> dispatcher(macStreams);

            // EXAMPLE
            dispatcher.forwardData(0xA00001, 10);
        }
    }
}
#endif  // LSP_RECEIVER_MANAGER_HPP
