// CTRL controller (controller_tx) of the LSP receiver kernel: builds the CTRL frames (CREDIT_ACK,
// NACK) of all connections from their CTRL status and sends them back through the CMAC.
#ifndef LSP_RECEIVER_CONTROLLER_TX_HPP
#define LSP_RECEIVER_CONTROLLER_TX_HPP

#include "utils.hpp"

namespace lsp {
    namespace receiver {
        template<int Streams>
        void controllerTX(
            hls::stream<ctrl_status_t<Streams>>& controllerInStream,
            hls::stream<beat_t>& toCMAC
        ) {
            #pragma HLS pipeline II=1
            static ctrl_status_t<Streams> status;

            // Read controller stream and react to it. If no ack or nack is required, this simply overwrites the
            // current state.
            status.takeNewer(controllerInStream.read());
            if (!status.ackImmediately && !status.nackImmediately) {
                return;
            }

            if (status.ackImmediately && !status.nackImmediately) {
                // TODO
            } else if (!status.ackImmediately && status.nackImmediately) {
                // TODO
            } else if (status.ackImmediately && status.nackImmediately) {
                // TODO
            }
        }
    }
}
#endif  // LSP_RECEIVER_CONTROLLER_TX_HPP
