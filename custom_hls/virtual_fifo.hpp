#ifndef VIRTUAL_FIFO_HPP
#define VIRTUAL_FIFO_HPP

#include <ap_int.h>
#include <hls_stream.h>

// finn-hlslib: width_v and the streaming copy move()
#include "utils.hpp"

// Virtual FIFO for any stream token type T with a raw bit image constructor, i.e. ap_uint<W>
// or a finn-hlslib tensor<E, N>. Tokens are counted, not stored.
template<typename T>
void VirtualFIFO(hls::stream<T> &in, hls::stream<T> &out,
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

    T inElem;

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
        if(out.write_nb(T(ap_uint<width_v<T>>(0)))) //write dummy output data
        {
            c_occupancy--;
        }
    }

    // Update output status registers
    occupancy = c_occupancy;
    max_occupancy = c_max_occupancy;
}

#endif
