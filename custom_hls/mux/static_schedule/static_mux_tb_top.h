#ifndef STATIC_MUX_TOP_H
#define STATIC_MUX_TOP_H

#include "hls_task.h"
#include "static_mux.hpp"

using T1 = ap_uint<2>;
using T2 = ap_int<10>;
using T3 = ap_int<14>;
using TOut = ap_uint<32>;
using S1 = hls::stream<T1, 20>;
using S2 = hls::stream<T2, 20>;
using S3 = hls::stream<T3, 20>;
using SOut = hls::stream<TOut, 20>;

void MuxDemuxOutOfOrder(
    S1 &a, S2 &b, S3 &c, S2 &d, S2 &e,
    SOut &outgoing_network, SOut &incoming_network,
    S1 &out_a, S2 &out_b, S3 &out_c, S2 &out_d, S2 &out_e
);

#endif
