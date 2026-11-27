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
        using namespace messages;
        using namespace helpers;
        using namespace constants;
        using namespace utils;

        enum controller_tx_status : unsigned char {
            IDLE,
            SENDING_ACK
        };

        /** Construct a NACK. Since a NACK only contains an ethernet header and the LSP header,
        it always fits into a one-beat-frame. */
        template<
            uint64_t SourceMAC,
            uint64_t TargetMAC,
            ethertype_or_tag Tag
        >
        beat_t constructNACK(ap_uint<FSN_WIDTH> expectedFSN) {
            auto data = ethernetHeader<Tag, SourceMAC, TargetMAC>();
            auto offset = getOffsetFromEthertype(Tag);
            auto header = header_t();
            header.frameSequenceNumber = expectedFSN;
            header.frameType = FRAME_NACK;
            header.flags = 0;
            header.payloadChunks = 0;
            header.version = CURRENT_VERSION;
            data.range(offset - 1 + header_t::WIDTH, offset) = header.pack();
            beat_t beat;
            beat.data = data;
            beat.user = 0;
            beat.last = 1;
            beat.strb = -1;
            beat.keep_all();
            return beat;
        }


        /** Construct both L2 header and LSP header for an ACK frame. The rest of the data is 0ed. */
        template<
            int Streams,
            size_t FrameWidthBits,
            uint64_t SourceMAC,
            uint64_t TargetMAC,
            ethertype_or_tag Tag
        >
        ap_uint<BEAT_WIDTH> constructACKHeader(ctrl_status_t<Streams>& currentStatus) {
            auto data = ethernetHeader<Tag, SourceMAC, TargetMAC>();
            auto offset = getOffsetFromEthertype(Tag);
            auto header = header_t();
            header.frameSequenceNumber = currentStatus.expectedFSN;
            header.frameType = FRAME_CREDIT_ACK;
            header.flags = 0;
            // The greedy algorithm either stores all updates in one frame (1) or as much as the frame can fit (2).
            // Saturate before narrowing: the count of changed streams can exceed the frame's capacity,
            // and only a value up to the capacity fits into record_count_t
            constexpr size_t Capacity = creditRecordsPerFrame<FrameWidthBits, Tag>();
            using record_count_t = ap_uint<clog2(1 + Capacity)>;
            const auto changedCount = popcount(currentStatus.changed);
            const record_count_t records = (changedCount > Capacity) ? record_count_t(Capacity) : record_count_t(changedCount);
            header.payloadChunks = credit_record_t::WIDTH * records / CHUNK_SIZE_BITS;
            header.version = CURRENT_VERSION;
            data.range(offset - 1 + header_t::WIDTH, offset) = header.pack();
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
            ethertype_or_tag Tag
        >
        bool sendACKBeat(hls::stream<beat_t> &toCMAC, ctrl_status_t<Streams>& status) {
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
            beat.strb = -1;
            beat.user = 0;
            beat.last = 0;
            beat.data = 0;
            if (currentBeat == 0) {
                beat.data = constructACKHeader<Streams, FrameWidthBits, SourceMAC, TargetMAC, Tag>(status);
            }

            // Fill slots inside the beat. Every slot has a fixed bit range in creditRecords, so no
            // part-select depends on currentBeat. In the first beat, all records are shifted behind the
            // L2 and LSP header by a constant afterwards.
            constexpr int FirstBeatOffset = getOffsetFromEthertype(Tag) + header_t::WIDTH;
            constexpr int SlotsInBeat = CREDITS_PER_BEAT;
            constexpr int SlotsInFirstBeat = creditsPerBeat(getOffsetFromEthertype(Tag), true);
            // Records per frame. Can be fewer than the frame's slots, if payloadChunks cannot count them all
            constexpr size_t Capacity = creditRecordsPerFrame<FrameWidthBits, Tag>();
            static_assert(
                Capacity <= SlotsInFirstBeat + (FrameWidthBits / BEAT_WIDTH - 1) * SlotsInBeat,
                "The frame's records must fit into its beats"
            );
            ap_uint<BEAT_WIDTH> creditRecords = 0;
            size_t overallCurrentSlot = 0;
            for (int currentSlot = 0; currentSlot < SlotsInBeat; currentSlot++) {
#pragma HLS unroll
                ap_uint<credit_record_t::WIDTH> credit = 0;

                // indices[stream] is across all streams across the entire frame. To match, we need to calculate the overall slot:
                // Slots in the first beat + Full Beats so far after first beat + Slots in the current beat
                overallCurrentSlot = (currentBeat == 0) ? currentSlot : SlotsInFirstBeat + currentSlot + (currentBeat - 1) * SlotsInBeat;
                // In the first beat, slots from SlotsInFirstBeat on would be shifted out of the beat, so they take no record.
                // Slots from Capacity on take no record either; their records go into the next frame
                if ((currentBeat != 0 || currentSlot < SlotsInFirstBeat) && overallCurrentSlot < Capacity) {
                    for (int stream = 0; stream < Streams; stream++) {
#pragma HLS unroll
                        if (status.changed[stream] && indices[stream] == overallCurrentSlot) {
                            credit |= credit_record_t{stream, status.limit[stream]}.pack();
                            status.changed[stream] = 0;
                        }
                    }
                    creditRecords.range(
                        (currentSlot + 1) * credit_record_t::WIDTH - 1,
                        currentSlot * credit_record_t::WIDTH
                    ) = credit;
                }
            }
            const ap_uint<BEAT_WIDTH> shiftedRecords = creditRecords << FirstBeatOffset;
            beat.data |= (currentBeat == 0) ? shiftedRecords : creditRecords;
            // Finished when this beat holds the last changed record or the frame's last record
            const int slotsSoFar = SlotsInFirstBeat + currentBeat * SlotsInBeat;
            bool isLast = (slotsSoFar >= total) || (slotsSoFar >= Capacity);
            beat.last = isLast;
            if (isLast) {
                currentBeat = 0;
            } else {
                currentBeat++;
            }
            toCMAC.write(beat);
            return isLast;
        }



        /** Read the requests from the connection stream. A NACK is a single beat and is sent at once.
        For an ACK, the first beat is sent at once, and the FSM stays in SENDING_ACK until the frame's
        last beat is sent, one beat per call. No new status is read while in SENDING_ACK.
        */
        template<
            int Streams,
            unsigned int NACK_REPEAT_INTERVAL,
            uint64_t SourceMAC,
            uint64_t TargetMAC,
            ethertype_or_tag Tag
        >
        void controllerTX(
            hls::stream<ctrl_status_t<Streams>>& controllerInStream,
            hls::stream<beat_t>& toCMAC
        ) {
#pragma HLS pipeline II=1
            static ctrl_status_t<Streams> status;
            static ap_uint<32> nackTimer = 0;
            static controller_tx_status fsmState = IDLE;
            static ap_uint<FSN_WIDTH> lastNACKFSN = 0;
            // A reset drops pending ACKs, NACKs and credit updates and restarts the FSM and the NACK timer
#pragma HLS reset variable=status
#pragma HLS reset variable=nackTimer
#pragma HLS reset variable=fsmState
#pragma HLS reset variable=lastNACKFSN

            // Read the newest status (nonblocking). Only possible if we arent constructing an ACK right now.
            ctrl_status_t<Streams> newer;
            if (fsmState != controller_tx_status::SENDING_ACK) {
                if (controllerInStream.read_nb(newer)) {
                    status.takeNewer(newer);
                }
            }

            // Always decrement timer (gives cycle count iff II=1)
            if (nackTimer != 0) {
                nackTimer--;
            }

            // NACK if we have to (either via a timeout or because parseBeat requested it)
            // Cannot NACK during ACK sending, because that would interrupt the frame - we have to wait until the ACK is sent
            if ((lastNACKFSN != status.expectedFSN || nackTimer == 0) && status.nackImmediately && fsmState != controller_tx_status::SENDING_ACK) {
                beat_t nack = constructNACK<SourceMAC, TargetMAC, Tag>(status.expectedFSN);
                lastNACKFSN = status.expectedFSN;
                toCMAC.write(nack);
                nackTimer = NACK_REPEAT_INTERVAL;
                return;
            }

            const bool timeForUpdate = !status.ackImmediately && !status.nackImmediately && fsmState!= controller_tx_status::SENDING_ACK && status.changed != 0;
            const bool needToSendAckBeat = status.ackImmediately || fsmState == controller_tx_status::SENDING_ACK;
            if (timeForUpdate || needToSendAckBeat) {
                bool isDone = sendACKBeat<
                    Streams,
                    maxFullBeatFrameWidth<Tag>(),
                    SourceMAC,
                    TargetMAC,
                    Tag
                >(toCMAC, status);
                if (!isDone) {
                    fsmState = controller_tx_status::SENDING_ACK;
                } else {
                    if (status.changed == 0) {
                        fsmState = controller_tx_status::IDLE;
                        status.ackImmediately = false;
                    }
                }
            }
        }
    }
}
#endif  // LSP_RECEIVER_CONTROLLER_TX_HPP
