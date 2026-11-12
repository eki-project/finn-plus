// LSP receiver kernel: receives every frame from the CMAC and serves all incoming connections.
// - The manager dispatches each frame by its source MAC address to the connection of that peer.
// - One connection per peer parses its frames, feeds its output streams and reports its CTRL status
//   (cumulative ACK, open gap, urgent ACK) to the controller_tx.
// - The controller_tx builds the CTRL frames (CREDIT_ACK, NACK) of all connections and sends them
//   back through the CMAC.
#include "lsp_receiver.hpp"
#include "lsp_common.hpp"
#include <ap_axi_sdata.h>

using namespace lsp;

namespace lsp {
    namespace receiver {
        using beat_t = ap_axiu<constants::BEAT_WIDTH, 1, 0, 0>;

        // Number of beats of the largest frame (MTU plus L2 header). A CMAC side FIFO has to hold at
        // least one whole frame: CMAC RX cannot be paused, and CMAC TX must not see a gap in a frame.
        constexpr size_t CMAC_BEAT_BUFFER = (
            (ETHERNET_MTU + MAX_ETHERNET_HEADER_LENGTH_BYTES) * 8 + constants::BEAT_WIDTH - 1
        ) / constants::BEAT_WIDTH;

        /** CTRL status of one connection, sent by the connection to the controller_tx whenever it
        changes. The controller_tx reads it non-blocking and keeps the newest one per connection (see
        takeNewer), so the connection never has to wait for the controller_tx. */
        struct ctrl_status_t {
            // Next expected FSN: the cumulative ACK, and the FSN to NACK while gapOpen is set
            ap_uint<constants::FSN_WIDTH> expectedFSN = 0;
            // A gap was seen at expectedFSN and is still open: NACK it, and again if it stays open
            bool gapOpen = false;
            // Send a CREDIT_ACK at once instead of after the ACK delay, e.g. for an accepted probe or
            // a duplicate frame
            bool ackNow = false;

            /** Take the values of a newer status. expectedFSN and gapOpen are states, so the newer
            values replace the held ones. ackNow is an event, so a pending one stays set until the
            controller_tx has sent the ACK and cleared it. */
            void takeNewer(const ctrl_status_t& newer) {
                #pragma HLS inline
                expectedFSN = newer.expectedFSN;
                gapOpen = newer.gapOpen;
                ackNow = ackNow || newer.ackNow;
            }
        };

        // Effectively a dict that maps the MAC addresses of the peers (Peers, an lsp_peers_t) to AXI
        // Streams. These streams are connected to the connecting kernels which read the beats.
        // This struct strips the ethernet header and only passes LSP header + payload.
        template<typename Peers>
        struct mac_dispatcher {
            // Stream array used to forward data, indexed by connection index
            hls::stream<beat_t> (&streams)[Peers::N];

            // Construct with the given array
            explicit mac_dispatcher(hls::stream<beat_t> (&s)[Peers::N]) : streams(s) {}

            /** Forward the given data to the stream of the peer with the given MAC address. If the
            address is unknown, nothing is done. */
            void forwardData(const MAC& sourceMAC, const beat_t& data) {
                // An unknown MAC yields Peers::INVALID_INDEX, which matches no stream
                const typename Peers::index_t idx = Peers::getIndexFromMAC(sourceMAC);
                for (size_t i = 0; i < Peers::N; ++i) {
                    #pragma HLS unroll
                    if (i == idx) {
                        streams[i].write(data);
                    }
                }
            }
        };

        /** Copy one whole frame from the CMAC into the ingress FIFO. CMAC RX cannot be paused, so this
        process does nothing else: a stalled manager backs up into the FIFO instead of losing beats on
        the port. */
        void cmacIngress(hls::stream<beat_t>& fromCMAC, hls::stream<beat_t, CMAC_BEAT_BUFFER>& ingress) {
            beat_t beat;
            do {
                #pragma HLS PIPELINE II=1
                beat = fromCMAC.read();
                ingress.write(beat);
            } while (!beat.last);
        }

        /** Copy one whole frame from the egress FIFO to the CMAC, starting only once the frame is
        complete, so that the CMAC never sees a gap inside the frame. The controller_tx writes one
        token on egressFrames after the last beat of each frame. */
        void cmacEgress(
            hls::stream<beat_t, CMAC_BEAT_BUFFER>& egress,
            hls::stream<bool, CMAC_BEAT_BUFFER>& egressFrames,
            hls::stream<beat_t>& toCMAC
        ) {
            egressFrames.read();
            beat_t beat;
            do {
                #pragma HLS PIPELINE II=1
                beat = egress.read();
                toCMAC.write(beat);
            } while (!beat.last);
        }


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

            // CTRL status per connection. Read non-blocking by the controller_tx, which keeps the
            // newest one per connection and attaches it to the next CTRL frame
            hls::stream<ctrl_status_t> ctrlStatus[PEERS::N];

            lsp_receiver_ring_buffer& ringBuffers[2] = {...};

            // Connect the CMAC to the manager and the manager to macStreams
            cmacIngress(fromCMAC, ingress);
            manager<PEERS>(ingress, macStreams);

            // TX to send control frames back. Also on its own sends credit updates and ACKs
            controller_tx<PEERS>(egress, egressFrames, ringBuffers, ctrlStatus);
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
