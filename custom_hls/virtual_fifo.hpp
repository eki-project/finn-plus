#ifndef VIRTUAL_FIFO_HPP
#define VIRTUAL_FIFO_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

// Utility Functions, taken from instrumentation wrapper
template<typename  T>
static void move(
	hls::stream<T> &src,
	hls::stream<T> &dst
) {
#pragma HLS pipeline II=1 style=flp
	dst.write(src.read());
}

template<typename  T>
static void move(
	hls::stream<hls::axis<T, 0, 0, 0>> &src,
	hls::stream<T> &dst
) {
#pragma HLS pipeline II=1 style=flp
	dst.write(src.read().data);
}

template<typename  T>
class Payload {
public:
	using  type = T;
};
template<typename  T>
class Payload<hls::axis<T, 0, 0, 0>> {
public:
	using  type = T;
};

template<unsigned int Width>
void VirtualFIFO(hls::stream<ap_uint<Width> > &in, hls::stream<ap_uint<Width> > &out,
                ap_uint<32> mode,
                ap_uint<32> depth,
                ap_uint<32> &occupancy,
                ap_uint<32> &max_occupancy)
{
    #pragma HLS pipeline II=1 style=flp

    static ap_uint<32> c_occupancy = 0;
    static ap_uint<32> c_max_occupancy = 0;
    #pragma HLS reset variable=c_occupancy
    #pragma HLS reset variable=c_max_occupancy

    ap_uint<Width> inElem;

    bool read = mode == 0 || c_occupancy != depth;
    bool write = c_occupancy != 0;

    // INPUT
    if(read)
    {
        if(in.read_nb(inElem)) //disregard input data
        {
            c_occupancy++;
            c_max_occupancy = (c_occupancy > c_max_occupancy) ? c_occupancy : c_max_occupancy;
        }
    }

    // OUTPUT
    if(write)
    {
        if(out.write_nb(0)) //write dummy output data
        {
            c_occupancy--;
        }
    }

    // Update output status registers
    occupancy = c_occupancy;
    max_occupancy = c_max_occupancy;
}

#endif
