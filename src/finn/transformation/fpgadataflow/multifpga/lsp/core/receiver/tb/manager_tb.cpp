// Testbench of the RX manager. Returns 0 on success, 77 if not implemented yet, any other value on
// failure.
#include "common.hpp"
#include "manager.hpp"
#include "tb_utils.hpp"
#include "utils.hpp"

using namespace lsp::receiver;

int main() {
    hls::stream<beat_t> streams[2];
    hls::stream<beat_t, CMAC_BEAT_BUFFER> fromCMAC;
    using PEERS = lsp::lsp_peers_t<0x1, 0x2>;
    bool failed = false;
    auto getStream = [&](const MAC& mac) -> decltype(auto) { return streams[PEERS::getIndexFromMAC(mac)]; };

    // These should all land in the correct stream
    ap_uint<lsp::constants::BEAT_WIDTH> header1 = 0, header2 = 0;
    header1.range(95, 48) = lsp::helpers::swapBytes<48>(0x1);
    header2.range(95, 48) = lsp::helpers::swapBytes<48>(0x2);
    fromCMAC.write(makeBeat(header1, false));
    fromCMAC.write(makeBeat(100, false));
    fromCMAC.write(makeBeat(200, false));
    fromCMAC.write(makeBeat(300, true));
    fromCMAC.write(makeBeat(header2, false));
    fromCMAC.write(makeBeat(100, false));
    fromCMAC.write(makeBeat(200, false));
    fromCMAC.write(makeBeat(300, true));

    // Test
    manager<PEERS>(fromCMAC, streams);
    manager<PEERS>(fromCMAC, streams);
    manager<PEERS>(fromCMAC, streams);
    manager<PEERS>(fromCMAC, streams);
    assert_eq_with_flag(getStream(0x1).size(), 4u, failed, "Beats in stream 0x1");
    assert_eq_with_flag(getStream(0x2).size(), 0u, failed, "Beats in the not addressed stream 0x2");
    manager<PEERS>(fromCMAC, streams);
    manager<PEERS>(fromCMAC, streams);
    manager<PEERS>(fromCMAC, streams);
    manager<PEERS>(fromCMAC, streams);
    assert_eq_with_flag(getStream(0x1).size(), 4u, failed, "Beats in stream 0x1");
    assert_eq_with_flag(getStream(0x2).size(), 4u, failed, "Beats in stream 0x2");
    assert_eq_with_flag(getStream(0x1).read().data, header1, failed, "Beat 0 from 0x1");
    assert_eq_with_flag(getStream(0x1).read().data, 100, failed, "Beat 1 from 0x1");
    assert_eq_with_flag(getStream(0x1).read().data, 200, failed, "Beat 2 from 0x1");
    assert_eq_with_flag(getStream(0x1).read().data, 300, failed, "Beat 3 from 0x1");
    assert_eq_with_flag(getStream(0x2).read().data, header2, failed, "Beat 0 from 0x2");
    assert_eq_with_flag(getStream(0x2).read().data, 100, failed, "Beat 1 from 0x2");
    assert_eq_with_flag(getStream(0x2).read().data, 200, failed, "Beat 2 from 0x2");
    assert_eq_with_flag(getStream(0x2).read().data, 300, failed, "Beat 3 from 0x2");
    return failed ? 1 : 0;
}
