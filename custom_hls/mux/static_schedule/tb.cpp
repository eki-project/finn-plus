#include <functional>
#include <array>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <bitset>

#include "static_mux_tb_top.h"


template<typename T, int F>
bool checkStream(hls::stream<T, F> &stream, int expected) {
    T value = stream.read();
    if (value != static_cast<T>(expected)) {
        std::cout << "FAIL: Expected value " << expected << " but got " << value << std::endl;
        return false;
    }
    return true;
}

bool test_mux_demux() {
    bool success = true;
    S1 a_in, a_out;
    S2 b_in, d_in, e_in, b_out, d_out, e_out;
    S3 c_in, c_out;
    SOut net_in, net_out;
    std::vector<ap_int<TOut::width>> expected;
    std::vector<TOut> received;

    // Basic test
    a_in.write(1);
    b_in.write(22);
    c_in.write(-33);
    d_in.write(-44);
    e_in.write(55);
    MuxDemuxOutOfOrder(
        a_in, b_in, c_in, d_in, e_in,
        net_in, net_out,
        a_out, b_out, c_out, d_out, e_out
    );

    // The order is defined in the testing top function.
    expected.insert(expected.end(), {1, -44, 22, -33, 55});

    // Read data to check its order and "send" it back from the network to the demux
    std::cout << "Checking order of packets..." << std::endl;
    while (!net_in.empty()) {
        received.push_back(net_in.read());
    }

    // Check that the number of elements match
    if (expected.size() != received.size()) {
        std::cout << "FAIL: Expected " << expected.size()
            << " values but received " << received.size()
            << std::endl;
        return false;
    }

    // Check that the order matches
    constexpr std::size_t streamCount = 5;
    for (std::size_t i = 0; i < expected.size(); i++) {
        auto split = splitHeader<
            bitwidth(streamCount),
            ap_int<TOut::width-bitwidth(streamCount)>,
            TOut::width
        >(
            received[i]
        );
        ap_int<TOut::width> data = std::get<1>(split);
        if (data != static_cast<ap_int<TOut::width>>(expected[i])) {
            std::cout << "FAIL: Expected ordered value "
                << expected[i] << " but got " << data
                << " (header: " << std::get<0>(split)
                << ") " << std::endl;
            success = false;
        }
    }

    // Check that the values were demuxd correctly (after sending values to demux)
    for (auto value : received) {
        net_out.write(value);
    }
    MuxDemuxOutOfOrder(
        a_in, b_in, c_in, d_in, e_in,
        net_in, net_out,
        a_out, b_out, c_out, d_out, e_out
    );

    std::cout << "Checking demux results..." << std::endl;
    success &= checkStream(a_out, 1);
    success &= checkStream(b_out, 22);
    success &= checkStream(c_out, -33);
    success &= checkStream(d_out, -44);
    success &= checkStream(e_out, 55);
    return success;
}

int main() {
    std::cout << "Checking header split/merge functions..." << std::endl;
    {
        // Data type
        constexpr int DWS = 10;
        using DTS = ap_int<DWS>;
        constexpr int DWU = 10;
        using DTU = ap_int<DWS>;

        // Header type
        constexpr int HW = 3;
        using HT = ap_uint<HW>;

        // Total packet width
        constexpr int OW = 20;

        // Index
        constexpr std::size_t idx = 2;

        bool anyErrors = false;
        DTS minSigned = -(1 << (DWS-1));
        DTS maxSigned = (1 << (DWS-1)) - 1;
        DTU maxUnsigned = (1 << (DWS-1)) - 1;

        std::cout << "Testing signed values from " << minSigned << " to " << maxSigned << std::endl;
        for (auto i = minSigned; i <= maxSigned; i++) {
            ap_uint<OW> packet = prefixHeader<HW, DTS, OW, idx>(i);
            std::tuple<HT, DTS> result = splitHeader<HW, DTS, OW>(packet);
            if (std::get<0>(result) != idx || std::get<1>(result) != i) {
                anyErrors = true;
                std::cout << "Expected DATA=" << i << ", HEADER=" << idx << std::endl;
                std::cout << "\tGot DATA=" << std::get<1>(result) << "; HEADER=" << std::get<0>(result) << std::endl;
                std::cout << "\tSigned: True. Datawidth: " << DWS << ". Headerwidth: " << HW << ". Packetwidth: " << OW << "." << std::endl;
                std::cout << "\tPacket: " << std::bitset<OW>(packet) << std::endl;
            }
            if (i == maxSigned) break;
        }

        std::cout << "Testing unsigned values from " << 0 << " to " << maxUnsigned << std::endl;
        for (auto i = 0; i <= maxUnsigned; i++) {
            ap_uint<OW> packet = prefixHeader<HW, DTU, OW, idx>(i);
            std::tuple<HT, DTU> result = splitHeader<HW, DTU, OW>(packet);
            if (std::get<0>(result) != idx || std::get<1>(result) != i) {
                anyErrors = true;
                std::cout << "Expected DATA=" << i << ", HEADER=" << idx << std::endl;
                std::cout << "\tGot DATA=" << std::get<1>(result) << "; HEADER=" << std::get<0>(result) << std::endl;
                std::cout << "\tSigned: False. Datawidth: " << DWU << ". Headerwidth: " << HW << ". Packetwidth: " << OW << "." << std::endl;
                std::cout << "\tPacket: " << std::bitset<OW>(packet) << std::endl;
            }
        }
        if (anyErrors) {
            return 1;
        }

        if (test_mux_demux()) {
            std::cout << "SUCCESS." << std::endl;
            return 0;
        }
        return 1;
    }
}
