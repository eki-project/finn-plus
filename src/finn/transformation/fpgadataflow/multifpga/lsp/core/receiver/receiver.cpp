// LSP receiver kernel: receives every frame from the CMAC and serves all incoming connections.
// - The manager dispatches each frame by its source MAC address to the connection of that peer.
// - One connection per peer parses its frames, feeds its output streams and reports its CTRL status
//   (cumulative ACK, open gap, urgent ACK) to the controller_tx.
// - The controller_tx builds the CTRL frames (CREDIT_ACK, NACK) of all connections and sends them
//   back through the CMAC.
#include "lsp_receiver.hpp"
#include "lsp_common.hpp"
#include "lsp_receiver_utils.hpp"
#include "lsp_receiver_manager.hpp"
#include "lsp_receiver_connection.hpp"
#include "lsp_receiver_controller_tx.hpp"

using namespace lsp;

namespace lsp {
    namespace receiver {
        // All processes below run concurrently and free-running (e.g. hls::task, or ap_ctrl_none with
        // while(1) loops in a dataflow region). cmacIngress in particular must never wait for another
        // process, otherwise beats are lost on the CMAC RX port.
        // TODO: T he supplied streams, peers etc. are all examples. This will later be moved into a separate
        // TODO: file that will be template instantiated by cmake with the real values.
        void receiver(
            hls::stream<beat_t>& fromCMAC,
            hls::stream<beat_t>& toCMAC,
            hls::stream<ap_uint<constants::BEAT_WIDTH>>& s0,
            hls::stream<ap_uint<constants::BEAT_WIDTH>>& s1
        ) {
            using PEERS = lsp_peers_t<0xA1, 0xA2>;

            // CMAC side FIFOs, each holding at least one whole frame (see CMAC_BEAT_BUFFER)
            hls::stream<beat_t, CMAC_BEAT_BUFFER> ingress("ingress");
            hls::stream<beat_t, CMAC_BEAT_BUFFER> egress("egress");
            // One token per complete frame in egress; every frame has at least one beat
            hls::stream<bool, CMAC_BEAT_BUFFER> egressFrames("egress_frames");

            hls::stream<beat_t> macStreams[PEERS::N];

            hls::stream<ctrl_status_t<1>> ctrlStatus[PEERS::N];


            // Connect the CMAC to the manager and the manager to macStreams
            cmacIngress(fromCMAC, ingress);
            manager<PEERS>(ingress, macStreams);

            // TX to send control frames back. Also on its own sends credit updates and ACKs
            controller_tx<PEERS>(egress, egressFrames, ctrlStatus);
            cmacEgress(egress, egressFrames, toCMAC);

            // Connect macstreams to the actual parsers for the devices/connections
            connection(macStreams[0], ringBuffers[0], ctrlStatus[0], s0);
            connection(macStreams[1], ringBuffers[1], ctrlStatus[1], s1);
        }


        // Ring buffer depth (in words) of every stream, in the order of the stream arguments
        #ifdef DEVELOP
            using RING_DEPTHS = data::lsp_ring_depths_t<512, 256>;
        #else
            using RING_DEPTHS = data::lsp_ring_depths_t<$$RING_DEPTHS$$>;
        #endif
    }
}
