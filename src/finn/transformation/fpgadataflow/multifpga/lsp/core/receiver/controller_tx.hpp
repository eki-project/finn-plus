// CTRL controller (controller_tx) of the LSP receiver kernel: builds the CTRL frames (CREDIT_ACK,
// NACK) of all connections from their CTRL status and sends them back through the CMAC.
#ifndef LSP_RECEIVER_CONTROLLER_TX_HPP
#define LSP_RECEIVER_CONTROLLER_TX_HPP

#include "utils.hpp"

namespace lsp {
    namespace receiver {
        enum controller_tx_status : unsigned char {
            IDLE,
            SENDING_NACK,
            SENDING_ACK,
        };


        /** Read the requests from the connection stream. If a NACK or ACK needs to be built,
        build the first beat and send it immediately, then go into the "SENDING_(N)ACK" state,
        so that every cycle we receive a new request and send a new beat. */
        template<int Streams>
        void controllerTX(
            hls::stream<ctrl_status_t<Streams>>& controllerInStream,
            hls::stream<beat_t>& toCMAC
        ) {
            #pragma HLS pipeline II=1
            static ctrl_status_t<Streams> status;
            static ap_uint<32> nackTimer = 0;

            // Read the newest status (nonblocking)
            ctrl_status_t<Streams> newer;
            if (controllerInStream.read_nb(newer)) {
                status.takeNewer(newer);
            }

            // Always decrement timer (gives cycle count iff II=1)
            if (nackTimer != 0) {
                nackTimer--;
            }

            // NACK if we have to (either via a timeout or because the connection kernel requested it)
            if (nackTimer == 0 && status.nackImmediately) {
                // TODO: Send NACK

                // NACK takes priority, so no ACK here
                return;
            }

            // Read controller stream and react to it. If no ack or nack is required, this simply overwrites the
            // current state.
            if (!status.ackImmediately && !status.nackImmediately) {
                // Send an update if we have one
                if (status.changed != 0) {
                    // TODO
                }
            }

            // Since the NACK case is already considered, we only have to check whether ACK is set
            if (status.ackImmediately) {
                // TODO
            }
        }
    }
}
#endif  // LSP_RECEIVER_CONTROLLER_TX_HPP
