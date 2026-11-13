// CTRL controller (controller_tx) of the LSP receiver kernel: builds the CTRL frames (CREDIT_ACK,
// NACK) of all connections from their CTRL status and sends them back through the CMAC.
#ifndef LSP_RECEIVER_CONTROLLER_TX_HPP
#define LSP_RECEIVER_CONTROLLER_TX_HPP

#include "lsp_receiver_utils.hpp"

namespace lsp {
    namespace receiver {
        // TODO: template<typename Peers> controller_tx(egress, egressFrames, ctrlStatus)
    }
}
#endif  // LSP_RECEIVER_CONTROLLER_TX_HPP
