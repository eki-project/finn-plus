// RX connection of the LSP receiver kernel: one per peer. Parses the frames the manager forwards,
// feeds its output streams and reports its CTRL status to the controller_tx.
#ifndef LSP_RECEIVER_CONNECTION_HPP
#define LSP_RECEIVER_CONNECTION_HPP

#include "lsp_receiver_utils.hpp"

namespace lsp {
    namespace receiver {
        // TODO: connection(macStream, ringBuffer, ctrlStatus, outStream)
    }
}
#endif  // LSP_RECEIVER_CONNECTION_HPP
