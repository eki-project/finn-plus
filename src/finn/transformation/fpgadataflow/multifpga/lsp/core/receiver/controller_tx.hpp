// CTRL controller (controller_tx) of the LSP receiver kernel: builds the CTRL frames (CREDIT_ACK,
// NACK) of all connections from their CTRL status and sends them back through the CMAC.
#ifndef LSP_RECEIVER_CONTROLLER_TX_HPP
#define LSP_RECEIVER_CONTROLLER_TX_HPP

#include "common.hpp"
#include "constants.hpp"
#include "utils.hpp"

namespace lsp {
    namespace receiver {
        enum controller_tx_status : unsigned char {
            IDLE,
            SENDING_ACK,
            NACK_WAIT_INTERVAL
        };

        /** Construct a NACK. Since a NACK only contains an ethernet header and the LSP header,
        it always fits into a one-beat-frame. */
        template<
            uint64_t SourceMAC,
            uint64_t TargetMAC,
            utils::ethertype_or_tag Tag = utils::ethertype_or_tag::LOCAL_ETHTYPE_1
        >
        beat_t constructNACK(ap_uint<constants::FSN_WIDTH> expectedFSN) {
            auto data = utils::ethernetHeader<Tag, SourceMAC, TargetMAC>();
            auto offset = utils::getOffsetFromEthertype(Tag);
            auto header = messages::header_t();
            header.frameSequenceNumber = expectedFSN;
            header.frameType = constants::FRAME_NACK;
            header.flags = 0;
            header.payloadChunks = 0;
            header.version = constants::CURRENT_VERSION;
            data.range(offset - 1 + messages::header_t::WIDTH, offset - 1) = header.pack();
            beat_t beat;
            beat.data = data;
            beat.user = 0;
            beat.last = 1;
            return beat;
        }

        /** Read the requests from the connection stream. If a NACK or ACK needs to be built,
        build the first beat and send it immediately, then go into the "SENDING_(N)ACK" state,
        so that every cycle we receive a new request and send a new beat.
        */
        template<int Streams, unsigned int NACK_REPEAT_INTERVAL, uint64_t SourceMAC, uint64_t TargetMAC>
        void controllerTX(
            hls::stream<ctrl_status_t<Streams>>& controllerInStream,
            hls::stream<beat_t>& toCMAC
        ) {
            #pragma HLS pipeline II=1
            static ctrl_status_t<Streams> status;
            static ap_uint<32> nackTimer = 0;
            static controller_tx_status fsmState = IDLE;

            // Read the newest status (nonblocking)
            ctrl_status_t<Streams> newer;
            if (controllerInStream.read_nb(newer)) {
                // Clear NACK FSM state if necessary
                if (status.nackImmediately && !newer.nackImmediately) {
                    fsmState = controller_tx_status::IDLE;
                }
                status.takeNewer(newer);
            }

            // Always decrement timer (gives cycle count iff II=1)
            if (nackTimer != 0) {
                nackTimer--;
            }

            // NACK if we have to (either via a timeout or because the connection kernel requested it)
            if (nackTimer == 0 && status.nackImmediately) {
                fsmState = controller_tx_status::NACK_WAIT_INTERVAL;
                constructNACK<SourceMAC, TargetMAC>(status.expectedFSN);

                // Start the timer (again)
                nackTimer = NACK_REPEAT_INTERVAL;

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
