// Shared types and helpers of the LSP receiver kernel: the beat type, the CMAC side FIFO depth,
// the CTRL status of a connection, the MAC dispatcher and the CMAC ingress / egress processes.
#ifndef LSP_RECEIVER_UTILS_HPP
#define LSP_RECEIVER_UTILS_HPP

#include "lsp_common.hpp"
#include <ap_axi_sdata.h>

namespace lsp {
    namespace receiver {
        using beat_t = ap_axiu<constants::BEAT_WIDTH, 1, 0, 0>;

        // Depth of a CMAC side FIFO: it has to hold at least one whole frame, because CMAC RX cannot
        // be paused and CMAC TX must not see a gap in a frame.
        constexpr size_t CMAC_BEAT_BUFFER = constants::MAX_BEATS_PER_FRAME;

        /** CTRL status of one connection with Streams streams, sent by the connection to the
        controller_tx whenever it changes. The controller_tx reads it non-blocking and keeps
        the newest one per connection (see takeNewer), so the connection never has to wait for the
        controller_tx. The connection decides when credit is due; the controller_tx
        only copies the limits of the changed streams into the credit records. */
        template<int Streams>
        struct ctrl_status_t {
            static_assert(Streams >= 1, "A connection has at least one stream");

            // Next expected FSN: the cumulative ACK, and the FSN to NACK while gapOpen is set
            ap_uint<constants::FSN_WIDTH> expectedFSN = 0;
            // A gap was seen at expectedFSN and is still open: NACK it, and again if it stays open
            bool gapOpen = false;
            // Send a CREDIT_ACK at once instead of after the ACK delay, e.g. for an accepted probe or
            // a duplicate frame
            bool ackNow = false;
            // Absolute credit limit per stream, consumed + ring capacity mod 2^32.
            // Absolute, not the free space: the sender cannot tell which of its in-flight words a
            // free count already covers, and a lost absolute limit is repaired by the next one
            ap_uint<constants::CREDIT_LIMIT_WIDTH> limit[Streams] = {};
            // Streams whose limit is due in a credit record, e.g. after half a ring was freed or at
            // once for a BLOCKED stream. Only these limits are sent, so the others may be stale
            ap_uint<Streams> changed = 0;

            /** Take the values of a newer status. expectedFSN, gapOpen and the limits are states, so
            the newer values replace the held ones. ackNow and changed are events, so pending ones stay
            set until the controller_tx has sent the CREDIT_ACK and cleared them. */
            void takeNewer(const ctrl_status_t& newer) {
                #pragma HLS inline
                expectedFSN = newer.expectedFSN;
                gapOpen = newer.gapOpen;
                ackNow = ackNow || newer.ackNow;
                for (int s = 0; s < Streams; ++s) {
                    #pragma HLS unroll
                    if (newer.changed.bit(s) == 1) {
                        limit[s] = newer.limit[s];
                    }
                }
                changed |= newer.changed;
            }

            /** Whether a CREDIT_ACK must go out at once instead of after the ACK delay: an ACK or a
            credit update is due. */
            bool creditAckNow() const {
                #pragma HLS inline
                return ackNow || changed != 0;
            }

            template<typename ...Rings>
            static ctrl_status_t makeCreditUpdateStatus(
                ap_uint<constants::FSN_WIDTH> expectedFSN,
                ap_uint<constants::STREAM_ID_WIDTH> stream,
                std::tuple<Rings...>& buffers
            ) {
                ctrl_status_t<Streams> ctrl;
                ctrl.ackNow = true;
                ctrl.expectedFSN = expectedFSN;
                ctrl.gapOpen = false;
                // If we specified an invalid stream index do nothing
                if (stream < Streams) {
                    ap_uint<Streams> m = 0;
                    m[stream] = 1;
                    ctrl.changed = m;
                }
                ctrl.limit[stream] = visitRing(
                    buffers,
                    stream,
                    [&](auto& ring){ return ring.creditUpdate(); },
                    ap_uint<constants::CREDIT_LIMIT_WIDTH>(0)
                );
                return ctrl;
            }

            static ctrl_status_t makeNACKStatus(ap_uint<constants::FSN_WIDTH> expectedFSN) {
                ctrl_status_t<Streams> ctrl;
                ctrl.ackNow = false;
                ctrl.expectedFSN = expectedFSN;
                ctrl.gapOpen = true;
                return ctrl;
            }

            static ctrl_status_t makeACKStatus(ap_uint<constants::FSN_WIDTH> expectedFSN) {
                ctrl_status_t<Streams> ctrl;
                ctrl.ackNow = false;
                ctrl.expectedFSN = expectedFSN;
                ctrl.gapOpen = false;
                return ctrl;
            }

            static ctrl_status_t makeACKNowStatus(ap_uint<constants::FSN_WIDTH> expectedFSN) {
                ctrl_status_t<Streams> ctrl;
                ctrl.ackNow = true;
                ctrl.expectedFSN = expectedFSN;
                ctrl.gapOpen = false;
                return ctrl;
            }
        };

        // Effectively a dict that maps the MAC addresses of the peers (Peers, an lsp_peers_t) to AXI
        // Streams. These streams are connected to the connecting kernels which read the beats.
        // Frames are forwarded unchanged; the connection parses the L2 header itself.
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
        inline void cmacIngress(
            hls::stream<beat_t>& fromCMAC,
            hls::stream<beat_t, CMAC_BEAT_BUFFER>& ingress
        ) {
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
        inline void cmacEgress(
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
    }
}
#endif  // LSP_RECEIVER_UTILS_HPP
