// CTRL controller (controller_tx) of the LSP receiver kernel: builds the CTRL frames (CREDIT_ACK,
// NACK) of all connections from their CTRL status and sends them back through the CMAC.
#ifndef LSP_RECEIVER_CONTROLLER_TX_HPP
#define LSP_RECEIVER_CONTROLLER_TX_HPP

#include "common.hpp"
#include "constants.hpp"
#include "helpers.hpp"
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
            data.range(offset - 1 + messages::header_t::WIDTH, offset) = header.pack();
            beat_t beat;
            beat.data = data;
            beat.user = 0;
            beat.last = 1;
            return beat;
        }


        /** Return the number of credit records that can fit into one frame. Accounts for L2 and LSP header.
        TODO: UDP support. */
        template<size_t FrameWidthBits, utils::ethertype_or_tag Tag>
        constexpr size_t creditRecordsPerFrame() {
            static_assert(FrameWidthBits % constants::BEAT_WIDTH == 0, "Frame width needs to be a multiple of the BEAT width.");
            return (FrameWidthBits - messages::header_t::WIDTH - utils::getOffsetFromEthertype(Tag)) / messages::credit_record_t::WIDTH;
        }

        /** Construct both L2 header and LSP header for an ACK frame. The rest of the data is 0ed. */
        template<
            int Streams,
            size_t FrameWidthBits,
            uint64_t SourceMAC,
            uint64_t TargetMAC,
            utils::ethertype_or_tag Tag = utils::ethertype_or_tag::LOCAL_ETHTYPE_1
        >
        ap_uint<constants::BEAT_WIDTH> constructACKHeader(ctrl_status_t<Streams>& currentStatus) {
            auto data = utils::ethernetHeader<Tag, SourceMAC, TargetMAC>();
            auto offset = utils::getOffsetFromEthertype(Tag);
            auto header = messages::header_t();
            header.frameSequenceNumber = currentStatus.expectedFSN;
            header.frameType = constants::FRAME_CREDIT_ACK;
            header.flags = 0;
            // The greedy algorithm either stores all updates in one frame (1) or as much as the frame can fit (2).
            header.payloadChunks = (
                messages::credit_record_t::WIDTH
                * std::min(
                    ap_uint<Streams>(helpers::popcount(currentStatus.changed)),
                    ap_uint<Streams>(creditRecordsPerFrame<FrameWidthBits, Tag>())
                )
                / constants::CHUNK_SIZE_BITS
            );
            header.version = constants::CURRENT_VERSION;
            data.range(offset - 1 + messages::header_t::WIDTH, offset) = header.pack();
            return data;
        }


        /** Perform stream compaction / array packing on the changed
        credit limits of the given status. Modifies the used status fields.
        Note that this may need be to called multiple times.

        Returns whether the frame is sent completely.
        */
        template<
            int Streams,
            size_t FrameWidthBits,
            uint64_t SourceMAC,
            uint64_t TargetMAC,
            utils::ethertype_or_tag Tag = utils::ethertype_or_tag::LOCAL_ETHTYPE_1
        >
        bool sendACKBeat(hls::stream<beat_t, CMAC_BEAT_BUFFER> &toCMAC, ctrl_status_t<Streams>& status) {
            constexpr int IndexWidth = clog2(1 + Streams);
            static ap_uint<IndexWidth> indices[Streams];
#pragma HLS array_partition variable=indices complete
            static ap_uint<IndexWidth> total = 0;
            static int currentBeat = 0;
            // After a reset the next call must start a new frame. indices and total are recomputed at
            // every frame start before they are used, so they need no reset.
#pragma HLS reset variable=currentBeat

            // Perform a prefix sum over the valid/changed entries
            // TODO: This might be better at the end of the last iteration due to timing (prefix sum + compare + slot)
            if (currentBeat == 0) {
                ap_uint<IndexWidth> current = 0;
                for (int i = 0; i < Streams; i++) {
#pragma HLS unroll
                    if (status.changed[i]) {
                        indices[i] = current;
                        current++;
                    }
                }
                total = current;
            }

            // Pack streams
            beat_t beat;
            beat.keep_all();
            beat.last = 0;
            beat.data = 0;
            if (currentBeat == 0) {
                beat.data = constructACKHeader<Streams, FrameWidthBits, SourceMAC, TargetMAC, Tag>(status);
            }

            // Fill slots inside the beat. Every slot has a fixed bit range in creditRecords, so no
            // part-select depends on currentBeat. In the first beat, all records are shifted behind the
            // L2 and LSP header by a constant afterwards.
            constexpr int FirstBeatOffset = utils::getOffsetFromEthertype(Tag) + messages::header_t::WIDTH;
            constexpr int SlotsInBeat = constants::BEAT_WIDTH / messages::credit_record_t::WIDTH;
            constexpr int SlotsInFirstBeat = (constants::BEAT_WIDTH - utils::getOffsetFromEthertype(Tag) - messages::header_t::WIDTH) / messages::credit_record_t::WIDTH;
            ap_uint<constants::BEAT_WIDTH> creditRecords = 0;
            size_t overallCurrentSlot = 0;
            for (int currentSlot = 0; currentSlot < SlotsInBeat; currentSlot++) {
#pragma HLS unroll
                ap_uint<messages::credit_record_t::WIDTH> credit = 0;

                // indices[stream] is across all streams across the entire frame. To match, we need to calculate the overall slot:
                // Slots in the first beat + Full Beats so far after first beat + Slots in the current beat
                overallCurrentSlot = (currentBeat == 0) ? currentSlot : SlotsInFirstBeat + currentSlot + (currentBeat - 1) * SlotsInBeat;
                // In the first beat, slots from SlotsInFirstBeat on would be shifted out of the beat, so they take no record
                if (currentBeat != 0 || currentSlot < SlotsInFirstBeat) {
                    for (int stream = 0; stream < Streams; stream++) {
#pragma HLS unroll
                        if (status.changed[stream] && indices[stream] == overallCurrentSlot) {
                            credit |= messages::credit_record_t{stream, status.limit[stream]}.pack();
                            status.changed[stream] = 0;
                        }
                    }
                    creditRecords.range(
                        (currentSlot + 1) * messages::credit_record_t::WIDTH - 1,
                        currentSlot * messages::credit_record_t::WIDTH
                    ) = credit;
                }
            }
            const ap_uint<constants::BEAT_WIDTH> shiftedRecords = creditRecords << FirstBeatOffset;
            beat.data |= (currentBeat == 0) ? shiftedRecords : creditRecords;
            // Finished when we reached the last slot or the frame is done
            bool isLast = (SlotsInFirstBeat + currentBeat * SlotsInBeat >= total) || (currentBeat == (FrameWidthBits / constants::BEAT_WIDTH) - 1);
            beat.last = isLast;
            if (isLast) {
                currentBeat = 0;
            } else {
                currentBeat++;
            }
            toCMAC.write(beat);
            return isLast;
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
            // A reset drops pending ACKs, NACKs and credit updates and restarts the FSM and the NACK timer
#pragma HLS reset variable=status
#pragma HLS reset variable=nackTimer
#pragma HLS reset variable=fsmState

            // Read the newest status (nonblocking). Only possible if we arent constructing an ACK right now.
            ctrl_status_t<Streams> newer;
            if (fsmState != controller_tx_status::SENDING_ACK) {
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
                    beat_t nack = constructNACK<SourceMAC, TargetMAC>(status.expectedFSN);
                    toCMAC.write(nack);
                    nackTimer = NACK_REPEAT_INTERVAL;
                    return;
                }
            }

            // Only start (potentially) constructing an ACK if we arent waiting for a NACK reply right now
            if (fsmState != controller_tx_status::NACK_WAIT_INTERVAL) {
                // Read controller stream and react to it. If no ack or nack is required, this simply overwrites the
                // current state.
                if (!status.ackImmediately && !status.nackImmediately) {
                    // Send an update if we have one
                    if (status.changed != 0) {
                        bool isDone = sendACKBeat<
                            Streams,
                            constants::ETHERNET_MTU,
                            SourceMAC,
                            TargetMAC
                        >(toCMAC, status);
                        if (!isDone) {
                            fsmState = controller_tx_status::SENDING_ACK;
                        } else {
                            if (status.changed == 0) {
                                fsmState = controller_tx_status::IDLE;
                            }
                        }
                    }
                }

                // Since the NACK case is already considered, we only have to check whether ACK is set
                if (status.ackImmediately) {
                    bool isDone = sendACKBeat<
                        Streams,
                        constants::ETHERNET_MTU,
                        SourceMAC,
                        TargetMAC
                    >(toCMAC, status);
                    if (!isDone) {
                        fsmState = controller_tx_status::SENDING_ACK;
                    } else {
                        if (status.changed == 0) {
                            fsmState = controller_tx_status::IDLE;
                        }
                    }
                }
            }
        }
    }
}
#endif  // LSP_RECEIVER_CONTROLLER_TX_HPP
