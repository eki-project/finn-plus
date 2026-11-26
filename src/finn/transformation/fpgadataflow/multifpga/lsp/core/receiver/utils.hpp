// Shared types and helpers of the LSP receiver kernel: the beat type, the CMAC side FIFO depth,
// the CTRL status of a connection, the MAC dispatcher, the CMAC ingress / egress processes and the
// round robin forwarding of the CTRL frames to the egress.
#ifndef LSP_RECEIVER_UTILS_HPP
#define LSP_RECEIVER_UTILS_HPP

#include "constants.hpp"
#include "helpers.hpp"
#include <ap_axi_sdata.h>

namespace lsp {
    namespace receiver {
        /** AXI Stream word of the CMAC ports. HLS accepts AXI Stream types only on top level ports,
        so it is converted to beat_t at the CMAC boundary (cmacIngress, cmacEgress). */
        using axis_beat_t = ap_axiu<constants::BEAT_WIDTH, 1, 0, 0>;

        /** One beat inside the kernel: the fields of axis_beat_t in a plain struct, since internal
        streams must not carry AXI Stream types. user is set by the CMAC RX on the last beat of a
        frame with a bad FCS. */
        struct beat_t {
            ap_uint<constants::BEAT_WIDTH> data;
            ap_uint<constants::BEAT_WIDTH / 8> keep;
            ap_uint<constants::BEAT_WIDTH / 8> strb;
            ap_uint<1> user;
            ap_uint<1> last;

            /** Mark every byte as valid (keep only, like ap_axiu::keep_all). */
            void keep_all() {
                #pragma HLS inline
                keep = ~ap_uint<constants::BEAT_WIDTH / 8>(0);
            }

            static beat_t fromAxis(const axis_beat_t& axis) {
                #pragma HLS inline
                beat_t beat;
                beat.data = axis.data;
                beat.keep = axis.keep;
                beat.strb = axis.strb;
                beat.user = axis.user;
                beat.last = axis.last;
                return beat;
            }

            axis_beat_t toAxis() const {
                #pragma HLS inline
                axis_beat_t axis;
                axis.data = data;
                axis.keep = keep;
                axis.strb = strb;
                axis.user = user;
                axis.last = last;
                return axis;
            }
        };

        // Depth of a CMAC side FIFO: it has to hold at least one whole frame, because CMAC RX cannot
        // be paused and CMAC TX must not see a gap in a frame.
        constexpr size_t CMAC_BEAT_BUFFER = constants::MAX_BEATS_PER_FRAME;

        /** CTRL status of one connection with Streams streams, sent by the connection to the
        controller_tx whenever a CTRL frame becomes due. The controller_tx reads it non-blocking and
        merges every status into the one it holds per connection (see takeNewer), so the connection
        never has to wait for the controller_tx.

        There are four cases:
        - NACK immediately (nackImmediately, makeNACKImmediately): a NACK for expectedFSN, at once.
        - ACK immediately (ackImmediately, makeACKImmediately or makeACK<true>): a CREDIT_ACK at
          once, with the limits of the changed streams.
        - ACK when ready (neither flag, changed != 0, makeACK<false>): a CREDIT_ACK with the limits
          of the changed streams, once the controller_tx has time.
        - FSN update (neither flag, changed == 0, makeFSNUpdate): no frame. An accepted frame is not
          ACKed on its own; its ACK rides on the next CREDIT_ACK. The update keeps the
          controller_tx's expectedFSN current, which closes an open gap.
        Both flags can be pending at once, e.g. after a malformed frame with a BLOCKED report: a NACK
        carries no credit records, so the controller_tx then sends both frames.
        */
        template<int Streams>
        struct ctrl_status_t {
            static constexpr int STREAMS = Streams;
            static_assert(Streams >= 1, "A connection has at least one stream");

            // Next expected FSN: the cumulative ACK, and the FSN a NACK replays from. A state, so
            // every status carries the current value
            ap_uint<constants::FSN_WIDTH> expectedFSN = 0;
            // Send a NACK for expectedFSN at once: a gap, or a lost, corrupt or malformed frame at
            // expectedFSN
            bool nackImmediately = false;
            // Send a CREDIT_ACK at once instead of when the controller_tx has time: a BLOCKED
            // report, a PROBE without a gap, or a duplicate frame
            bool ackImmediately = false;
            // Absolute credit limit per stream, consumed + ring capacity mod 2^32.
            // Absolute, not the free space: the sender cannot tell which of its in-flight words a
            // free count already covers, and a lost absolute limit is repaired by the next one
            ap_uint<constants::CREDIT_LIMIT_WIDTH> limit[Streams] = {};
            // Streams whose limit is due in a credit record, e.g. after half a ring was freed or for
            // a BLOCKED stream. Only these limits are sent, so the others may be stale
            ap_uint<Streams> changed = 0;

            /** Merge a newer status into this one. expectedFSN is a state, so the newer value replaces
            the held one; a limit is replaced only if the newer status marks its stream as changed, since
            the other limits of a status are not set. nackImmediately, ackImmediately and changed are
            events, so pending ones stay set until the controller_tx has sent the frame and cleared
            them. A pending NACK only stays while expectedFSN is unchanged: once it advanced, the
            NACKed frame has arrived. */
            void takeNewer(const ctrl_status_t& newer) {
                #pragma HLS inline
                nackImmediately =
                    (nackImmediately && newer.expectedFSN == expectedFSN) || newer.nackImmediately;
                expectedFSN = newer.expectedFSN;
                ackImmediately = ackImmediately || newer.ackImmediately;
                for (int s = 0; s < Streams; ++s) {
                    #pragma HLS unroll
                    if (newer.changed.bit(s) == 1) {
                        limit[s] = newer.limit[s];
                    }
                }
                changed |= newer.changed;
            }

            /** NACK immediately: a gap, or a lost, corrupt or malformed frame at expectedFSN. */
            static ctrl_status_t makeNACKImmediately(ap_uint<constants::FSN_WIDTH> expectedFSN) {
                ctrl_status_t ctrl;
                ctrl.expectedFSN = expectedFSN;
                ctrl.nackImmediately = true;
                return ctrl;
            }

            /** ACK immediately without credit: a PROBE without a gap, or a duplicate frame. */
            static ctrl_status_t makeACKImmediately(ap_uint<constants::FSN_WIDTH> expectedFSN) {
                ctrl_status_t ctrl;
                ctrl.expectedFSN = expectedFSN;
                ctrl.ackImmediately = true;
                return ctrl;
            }

            /** ACK with the current limit of one stream, e.g. a BLOCKED stream. With Immediately the
            CREDIT_ACK goes out at once, otherwise when the controller_tx has time. A stream ID the
            connection does not have adds no credit record; the status is then an ACK immediately
            without credit, or an FSN update. */
            template<bool Immediately, typename ...Rings>
            static ctrl_status_t makeACK(
                ap_uint<constants::FSN_WIDTH> expectedFSN,
                ap_uint<constants::STREAM_ID_WIDTH> stream,
                std::tuple<Rings...>& buffers
            ) {
                ctrl_status_t ctrl = makeFSNUpdate(expectedFSN);
                // Bounds check, so that no limit is written outside of the array
                if (stream < Streams) {
                    ctrl.changed[stream] = 1;
                    ctrl.limit[stream] = creditUpdate(buffers, stream);
                }
                if (Immediately) {
                    ctrl.ackImmediately = true;
                }
                return ctrl;
            }

            /** FSN update: an accepted frame. Sends no frame, but keeps the controller_tx's
            expectedFSN current, so an open gap closes. */
            static ctrl_status_t makeFSNUpdate(ap_uint<constants::FSN_WIDTH> expectedFSN) {
                ctrl_status_t ctrl;
                ctrl.expectedFSN = expectedFSN;
                return ctrl;
            }

            /** ACK with the limits of every stream whose credit update is due, after its consumer
            freed enough space. With Immediately the CREDIT_ACK goes out at once, otherwise when the
            controller_tx has time. changed stays 0 if no update is due; without Immediately the
            status is then an FSN update and need not be sent. */
            template<bool Immediately, typename ...Rings>
            static ctrl_status_t makeACK(
                ap_uint<constants::FSN_WIDTH> expectedFSN,
                std::tuple<Rings...>& buffers
            ) {
                ctrl_status_t ctrl = makeFSNUpdate(expectedFSN);
                for (int s = 0; s < Streams; ++s) {
                    #pragma HLS unroll
                    if (creditUpdateDue(buffers, s)) {
                        ctrl.changed[s] = 1;
                        ctrl.limit[s] = creditUpdate(buffers, s);
                    }
                }
                if (Immediately) {
                    ctrl.ackImmediately = true;
                }
                return ctrl;
            }
        };

        /** CTRL status collected by a connection and sent to the controller_tx as one status. A call of
        the connection can produce several statuses, but its CTRL status stream takes only one write per
        cycle, so the statuses are merged here (see takeNewer) and sent together. */
        template<int Streams>
        struct pending_ctrl_status_t {
            ctrl_status_t<Streams> status;
            bool pending = false;

            /** Merge a status into the collected one. */
            void add(const ctrl_status_t<Streams>& newer) {
                #pragma HLS inline
                status.takeNewer(newer);
                pending = true;
            }

            /** Send the collected status, if any. If the stream is full, it stays for the next try. */
            void trySend(hls::stream<ctrl_status_t<Streams>>& stream) {
                #pragma HLS inline
                if (pending && stream.write_nb(status)) {
                    // The events are sent; expectedFSN and the limits are states and stay
                    status.nackImmediately = false;
                    status.ackImmediately = false;
                    status.changed = 0;
                    pending = false;
                }
            }
        };

        // Effectively a dict that maps the MAC addresses of the peers (Peers, an mac_index_lookup_t) to AXI
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
            hls::stream<axis_beat_t>& fromCMAC,
            hls::stream<beat_t>& ingress
        ) {
            beat_t beat;
            do {
                #pragma HLS PIPELINE II=1
                beat = beat_t::fromAxis(fromCMAC.read());
                ingress.write(beat);
            } while (!beat.last);
        }

        /** Copy one whole frame from the egress FIFO to the CMAC, starting only once the frame is
        complete, so that the CMAC never sees a gap inside the frame. The controller_tx writes one
        token on egressFrames after the last beat of each frame. */
        inline void cmacEgress(
            hls::stream<beat_t>& egress,
            hls::stream<bool, CMAC_BEAT_BUFFER>& egressFrames,
            hls::stream<axis_beat_t>& toCMAC
        ) {
            egressFrames.read();
            beat_t beat;
            do {
                #pragma HLS PIPELINE II=1
                beat = egress.read();
                toCMAC.write(beat.toAxis());
            } while (!beat.last);
        }

        /** Forward from source to egress until the tlast signal is set. */
        inline void forwardCtrlToEgress(hls::stream<beat_t>& egress, hls::stream<beat_t>& source) {
            beat_t beat;
            do {
                beat = source.read();
                egress.write(beat);
            } while (!beat.last);
        }

        /** HLS-Task compatible function/kernel that selects a controller TX stream to read. Once
        selected, the stream is read until tlast is seen. Then the stateful index counter is incremented
        and its the turn of the next stream. */
        template<size_t N>
        void writeToEgress(
            hls::stream<beat_t>& egress,
            hls::stream<bool, CMAC_BEAT_BUFFER>& egressFrames,
            hls::stream<beat_t> (&ctrlStreams)[N]
        ) {
            // Selector to decide which control stream may write
            static helpers::round_robin_selector_t<N> selector;

            // Write into egress. select() returns N if no stream holds a frame
            auto selectedStream = selector.select(ctrlStreams);
            if (selectedStream == N) {
                return;
            }
            forwardCtrlToEgress(egress, ctrlStreams[selectedStream]);
            egressFrames.write(true);
            selector.advancePast(selectedStream);
        }
    }
}
#endif  // LSP_RECEIVER_UTILS_HPP
