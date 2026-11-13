// Testbench of the receiver utils. The first argument selects the component under test.
// Returns 0 on success, 77 if the test is not implemented yet, any other value on failure.
#include <iostream>
#include <string>

#include "common.hpp"
#include "utils.hpp"
#include "tb_utils.hpp"

using namespace lsp::receiver;

// takeNewer replaces expectedFSN and gapOpen, and keeps a pending ackNow
int testCtrlStatus() {
    bool failed = false;
    using status_t = ctrl_status_t<2>;
    status_t held;
    assert_with_flag(held.expectedFSN == 0 && !held.gapOpen && !held.ackNow, failed,
        "Default status should be FSN 0 without gap and pending ACK.");

    // A newer status replaces all held values
    held.takeNewer(status_t{5, true, true});
    assert_with_flag(held.expectedFSN == 5, failed, "expectedFSN = " << held.expectedFSN << ", expected 5.");
    assert_with_flag(held.gapOpen, failed, "gapOpen should be taken from the newer status.");
    assert_with_flag(held.ackNow, failed, "ackNow should be taken from the newer status.");

    // States are replaced, a pending ackNow stays set
    held.takeNewer(status_t{7, false, false});
    assert_with_flag(held.expectedFSN == 7, failed, "expectedFSN = " << held.expectedFSN << ", expected 7.");
    assert_with_flag(!held.gapOpen, failed, "A closed gap should replace the open one.");
    assert_with_flag(held.ackNow, failed, "A pending ackNow should stay set until it is cleared.");

    // Once the controller_tx cleared ackNow after sending the ACK, it stays cleared
    held.ackNow = false;
    held.takeNewer(status_t{7, false, false});
    assert_with_flag(!held.ackNow, failed, "A cleared ackNow should stay cleared.");

    // Credit: limits are states and replaced, changed is an event and merged until cleared
    auto withCredit = [](unsigned limit0, unsigned limit1, unsigned changed) {
        status_t status{7, false, false};
        status.limit[0] = limit0;
        status.limit[1] = limit1;
        status.changed = changed;
        return status;
    };
    assert_eq_with_flag(held.limit[0], 0, failed, "Default limit of stream 0");
    assert_eq_with_flag(held.limit[1], 0, failed, "Default limit of stream 1");
    assert_eq_with_flag(held.changed, 0, failed, "Default changed mask");
    assert_eq_with_flag(held.creditAckNow(), false, failed, "creditAckNow without ACK or credit due");

    held.takeNewer(withCredit(100, 200, 0b01));
    assert_eq_with_flag(held.limit[0], 100, failed, "Limit of stream 0 after the first update");
    assert_eq_with_flag(held.limit[1], 200, failed, "Limit of stream 1 after the first update");
    assert_eq_with_flag(held.changed, 0b01, failed, "changed after the first update");
    assert_eq_with_flag(held.creditAckNow(), true, failed, "creditAckNow with credit due");

    held.takeNewer(withCredit(150, 250, 0b10));
    assert_eq_with_flag(held.limit[0], 150, failed, "Limit of stream 0 after the second update");
    assert_eq_with_flag(held.limit[1], 250, failed, "Limit of stream 1 after the second update");
    assert_eq_with_flag(held.changed, 0b11, failed, "changed should merge pending streams");

    // Once the controller_tx cleared changed after sending the CREDIT_ACK, it stays cleared
    held.changed = 0;
    held.takeNewer(withCredit(160, 260, 0));
    assert_eq_with_flag(held.limit[0], 160, failed, "Limit of stream 0 without credit due");
    assert_eq_with_flag(held.changed, 0, failed, "A cleared changed mask should stay cleared");
    assert_eq_with_flag(held.creditAckNow(), false, failed, "creditAckNow after clearing");
    return failed ? 1 : 0;
}

int testMacDispatcher() {
    using PEERS = lsp::lsp_peers_t<0x1, 0x2, 0xA, 0xB>;
    hls::stream<beat_t> streams[4];
    mac_dispatcher<PEERS> dispatcher(streams);
    bool failed = false;
    beat_t data;
    auto getStream = [&](const MAC& mac) -> decltype(auto) { return streams[PEERS::getIndexFromMAC(mac)]; };

    // Try dispatching and check if the streams receive the correct data.
    dispatcher.forwardData(0x1, makeBeat(100, false));
    dispatcher.forwardData(0x1, makeBeat(200, true));
    data = getStream(0x1).read();
    assert_with_flag(data.get_data() == 100, failed, "Wrong data read from stream.");
    data = getStream(0x1).read();
    assert_with_flag(data.last == 1, failed, "Lost tlast flag in the stream.");

    // Check that all other streams are still empty
    assert_with_flag(getStream(0x2).empty(), failed, "Expected other streams to be empty.");
    assert_with_flag(getStream(0xA).empty(), failed, "Expected other streams to be empty.");
    assert_with_flag(getStream(0xB).empty(), failed, "Expected other streams to be empty.");

    // Dispatch to all streams
    dispatcher.forwardData(0x1, makeBeat(1, false));
    dispatcher.forwardData(0x2, makeBeat(2, false));
    dispatcher.forwardData(0xA, makeBeat(3, false));
    dispatcher.forwardData(0xB, makeBeat(4, false));
    dispatcher.forwardData(0x1, makeBeat(10, false));
    dispatcher.forwardData(0x2, makeBeat(20, false));
    dispatcher.forwardData(0xA, makeBeat(30, false));
    dispatcher.forwardData(0xB, makeBeat(40, false));
    assert_with_flag(getStream(0x1).read().get_data() == 1, failed, "Unexpected value");
    assert_with_flag(getStream(0x2).read().get_data() == 2, failed, "Unexpected value");
    assert_with_flag(getStream(0xA).read().get_data() == 3, failed, "Unexpected value");
    assert_with_flag(getStream(0xB).read().get_data() == 4, failed, "Unexpected value");
    assert_with_flag(!getStream(0x1).empty(), failed, "Stream should not be empty");
    assert_with_flag(!getStream(0x2).empty(), failed, "Stream should not be empty");
    assert_with_flag(!getStream(0xA).empty(), failed, "Stream should not be empty");
    assert_with_flag(!getStream(0xB).empty(), failed, "Stream should not be empty");
    getStream(0x1).read();
    getStream(0x2).read();
    getStream(0xA).read();
    getStream(0xB).read();
    return failed ? 1 : 0;
}

// One call copies exactly one frame, up to and including the beat with last set
int testCmacIngress() {
    bool failed = false;
    hls::stream<beat_t> fromCMAC;
    hls::stream<beat_t, CMAC_BEAT_BUFFER> ingress;
    fromCMAC.write(makeBeat(1, false));
    fromCMAC.write(makeBeat(2, true));
    fromCMAC.write(makeBeat(3, true));
    cmacIngress(fromCMAC, ingress);
    assert_with_flag(ingress.size() == 2 && fromCMAC.size() == 1, failed, "Expected exactly one frame to be copied.");
    assert_with_flag(ingress.read().data == 1 && ingress.read().last, failed, "Wrong beats copied.");
    return failed ? 1 : 0;
}

// One call consumes one frame token and sends exactly one frame
int testCmacEgress() {
    bool failed = false;
    hls::stream<beat_t, CMAC_BEAT_BUFFER> egress;
    hls::stream<bool, CMAC_BEAT_BUFFER> egressFrames;
    hls::stream<beat_t> toCMAC;
    egress.write(makeBeat(1, false));
    egress.write(makeBeat(2, true));
    egress.write(makeBeat(3, true));
    egressFrames.write(true);
    cmacEgress(egress, egressFrames, toCMAC);
    assert_with_flag(toCMAC.size() == 2 && egress.size() == 1 && egressFrames.empty(), failed,
        "Expected exactly one frame to be sent.");
    assert_with_flag(toCMAC.read().data == 1 && toCMAC.read().last, failed, "Wrong beats sent.");
    return failed ? 1 : 0;
}

int main(int argc, char** argv) {
    const std::string test = argc > 1 ? argv[1] : "";
    if (test == "ctrl_status") {
        return testCtrlStatus();
    } else if (test == "mac_dispatcher") {
        return testMacDispatcher();
    } else if (test == "cmac_ingress") {
        return testCmacIngress();
    } else if (test == "cmac_egress") {
        return testCmacEgress();
    }
    std::cerr << "utils_tb: unknown test '" << test << "'" << std::endl;
    return 1;
}
