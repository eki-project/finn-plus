#include "static_mux_tb_top.h"

void MuxDemuxOutOfOrder(
    S1 &a, S2 &b, S3 &c, S2 &d, S2 &e,
    SOut &outgoing_network, SOut &incoming_network,
    S1 &out_a, S2 &out_b, S3 &out_c, S2 &out_d, S2 &out_e
) {
#pragma HLS INTERFACE axis port=a
#pragma HLS INTERFACE axis port=b
#pragma HLS INTERFACE axis port=c
#pragma HLS INTERFACE axis port=d
#pragma HLS INTERFACE axis port=e
#pragma HLS INTERFACE axis port=out_a
#pragma HLS INTERFACE axis port=out_b
#pragma HLS INTERFACE axis port=out_c
#pragma HLS INTERFACE axis port=out_d
#pragma HLS INTERFACE axis port=out_e
#pragma HLS INTERFACE ap_ctrl_none port=return
    static_mux(std::index_sequence<0, 3, 1, 2, 4>{}, outgoing_network, a, b, c, d, e);
    while (!incoming_network.empty()) {
        static_demux(incoming_network, out_a, out_b, out_c, out_d, out_e);
    }
}
