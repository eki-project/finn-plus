#ifndef STATIC_MUX_HPP
#define STATIC_MUX_HPP

#include "ap_int.h"
#include "hls_stream.h"
#include <type_traits>
#include <tuple>
#include <bitset>
#include <initializer_list>
#include <functional>
#include <utility>


/************* Helpers *************/

/** Helper to check that we receive an ap_int or ap_uint, storing the bitwidth **/
template<typename T>
struct is_arbitrary_precision_type;

template<int N>
struct is_arbitrary_precision_type<ap_int<N>> {
    static constexpr int width = N;
};

template<int N>
struct is_arbitrary_precision_type<ap_uint<N>> {
    static constexpr int width = N;
};


/** Return the number of bits needed to count to i, inclusive **/
constexpr std::size_t bitwidth(std::size_t i) {
    // Every bit halves the number of choices addressed, so the
    // next call only deals with i/2 (i>>1)
    return (i < 2 ? i : 1 + bitwidth(i >> 1));
}
static_assert(bitwidth(2) == 2, "Bitwidth for 2 should be 2");
static_assert(bitwidth(11) == 4, "Bitwidth for 11 should be 4");



/**
 * Prefix data with an unsigned header. The header is always left aligned. The data is
 * right aligned. If the data is more narrow than the maximum data width, it is
 * sign-extended.
 */
template<int HeaderBitwidth, typename DataType, int OutWidth, std::size_t idx>
ap_uint<OutWidth> prefixHeader(DataType value) {
    static_assert(OutWidth >= is_arbitrary_precision_type<DataType>::width + HeaderBitwidth);
    constexpr std::size_t DataReservedWidth = OutWidth - HeaderBitwidth;

    // Since value is more narrow than OutWidth, we implicitly cast here.
    ap_uint<OutWidth> data = value;

    // Force header bits to zero (could be 1 due to sign extension)
    data &= static_cast<ap_uint<OutWidth>>((1 << DataReservedWidth) - 1);

    // Insert the header
    data |= static_cast<ap_uint<OutWidth>>(idx) << DataReservedWidth;
    return data;
}



/**
 * Split data with a prefixed header (see prefixHeader<...>(...)) into a tuple of (header, data).
 * The data is cast to the supplied ap_(u)int type before returning.
 */
template<
    int HeaderBitwidth,
    typename DataType,
    int InWidth
>
std::tuple<ap_uint<HeaderBitwidth>, DataType> splitHeader(
    ap_uint<InWidth> value
) {
    constexpr std::size_t DataWidth = is_arbitrary_precision_type<DataType>::width;

    // Check this, otherwise, if the datawidth == outwidth, the split will fail
    static_assert(DataWidth + HeaderBitwidth <= InWidth);

    // Simply shift data to right until only the truncated header is left
    ap_uint<HeaderBitwidth> header = static_cast<ap_uint<HeaderBitwidth>>(
        value >> (InWidth - HeaderBitwidth)
    );

    // Create 1s for every data bit and mask out the header
    ap_uint<InWidth> dataMask = static_cast<ap_uint<InWidth>>((1 << DataWidth) - 1);
    DataType data = static_cast<DataType>(value & dataMask);

#if !defined(__SYNTHESIS__) && defined(MUX_DEBUG)
    std::cout << "(SplitHeader) Header: " << header << " ("
        << std::bitset<HeaderBitwidth>(header) << ") ";
    std::cout << "Data: " << data << " ("
        << std::bitset<DataWidth>(data) << ")" << std::endl;
#endif
    return std::make_tuple(header, data);
}



/************* Mux *************/


/**
 * If the selected index matches this streams index, write the streams data
 * with a prefixed header to the output stream.
 * Recursive base case.
 */
template<
    std::size_t idx = 0,
    std::size_t HeaderWidth,
    int NOut, int FOut,
    typename T, int F
>
void _static_mux_single_stream(
    std::size_t selectedIndex,
    hls::stream<ap_uint<NOut>, FOut> &out,
    hls::stream<T, F> &currentStream
) {
    static_assert(HeaderWidth + is_arbitrary_precision_type<T>::width <= NOut);
    static_assert(
        HeaderWidth + is_arbitrary_precision_type<T>::width <= AP_INT_MAX_W,
        "The incoming stream width + the required header width is larger than the max "
        "allowed AP INT width. Multi-message packets are currently not supported. "
        "Please either adjust AP_INT_MAX_W (if possible) or adjust the folding."
    );
    if (selectedIndex == idx) {
        if (!currentStream.empty()) {
            auto value = currentStream.read();
            out.write(prefixHeader<HeaderWidth, T, NOut, idx>(value));
        }
    }
}


/**
 * If the selected index matches this streams index, write the streams data
 * with a prefixed header to the output stream. Otherwise, recursively unpack
 * to check the next stream.
 */
template<
    std::size_t idx = 0,
    std::size_t HeaderWidth,
    int NOut, int FOut,
    typename T, int F,
    typename ...Ts, int ...Fs
>
void _static_mux_single_stream(
    std::size_t selectedIndex,
    hls::stream<ap_uint<NOut>, FOut> &out,
    hls::stream<T, F> &currentStream,
    hls::stream<Ts, Fs> &...others
) {
    static_assert(HeaderWidth + is_arbitrary_precision_type<T>::width <= NOut);
    static_assert(
        HeaderWidth + is_arbitrary_precision_type<T>::width <= AP_INT_MAX_W,
        "The incoming stream width + the required header width is larger than the max "
        "allowed AP INT width. Multi-message packets are currently not supported. "
        "Please either adjust AP_INT_MAX_W (if possible) or adjust the folding."
    );
    if (selectedIndex == idx) {
        if (!currentStream.empty()) {
            auto value = currentStream.read();
            out.write(prefixHeader<HeaderWidth, T, NOut, idx>(value));
        } else {
            return;
        }
    } else {
        _static_mux_single_stream<idx+1, HeaderWidth>(selectedIndex, out, others...);
    }
}



/**
 * Top level function for statically scheduled blocking multiplexing of a
 * variable number of streams. Takes as first argument a list of indices
 * and checks the streams identified by the indices in order. If a stream has
 * data available, the data is sent to the output stream. If not, the stream is
 * skipped for this call. (Non-blocking read, blocking write).
 */
template<
    int NOut,
    int FOut,
    typename ...Ts,
    int ...Fs,
    std::size_t ...I
    >
void static_mux(
    std::index_sequence<I...>,
    hls::stream<ap_uint<NOut>, FOut> &out,
    hls::stream<Ts, Fs>& ...others
) {
    for (const std::size_t idx : { I... }) {
        _static_mux_single_stream<0, bitwidth(sizeof...(others))>(idx, out, others...);
    }
}



/************* Demux *************/

/**
 * If the current stream has the correct target index, write the data onto
 * the stream. The data is cast to the type of stream its written to.
 * Recursive base case.
 */
template<
    std::size_t idx = 0,
    int DataWidth,
    int InWidth, int InDepth,
    typename T, int F
>
void _static_demux(
    std::size_t targetIndex,
    ap_uint<DataWidth> data,
    hls::stream<ap_uint<InWidth>, InDepth> &in,
    hls::stream<T, F> &current
) {
    static_assert(DataWidth > is_arbitrary_precision_type<T>::width);
    if (idx == targetIndex) {
        // We need to cast here, since we received a blank ap_uint.
        current.write(static_cast<T>(data));
    } else {
#ifndef __SYNTHESIS__
        std::cout << "Could not demultiplex to non-existing stream index " << targetIndex;
        std::cout << ". The last index found is: " << idx << "." << std::endl;
#endif
    }
}



/**
 * If the current stream has the correct target index, write the data onto
 * the stream. The data is cast to the type of stream its written to.
 * If the current stream does not match the target index, recursively unpack
 * to try the next stream.
 */
template<
    std::size_t idx = 0,
    int DataWidth,
    int InWidth, int InDepth,
    typename T, int F,
    typename ...Ts, int ...Fs
>
void _static_demux(
    std::size_t targetIndex,
    ap_uint<DataWidth> data,
    hls::stream<ap_uint<InWidth>, InDepth> &in,
    hls::stream<T, F> &current,
    hls::stream<Ts, Fs>& ...streams
) {
    static_assert(DataWidth > is_arbitrary_precision_type<T>::width);
    if (idx == targetIndex) {
        // We need to cast here, since we received a blank ap_uint.
        current.write(static_cast<T>(data));
    } else {
        _static_demux<idx+1>(targetIndex, data, in, streams...);
    }
}



/**
 * Do a BLOCKING demux operation. Reads the packet from the incoming stream (network or otherwise)
 * and splits it into header and data. It then forwards the data to the correct stream and writes it,
 * blocking. Since the write is blocking, the user needs to make sure that the buffer after the Demux is
 * large enough (see FIFO sizing, initial largest occupancy)
 */
template<
    std::size_t idx = 0,
    int InWidth, int InDepth,
    typename ...Ts, int ...Fs
>
void static_demux(
    hls::stream<ap_uint<InWidth>, InDepth> &in,
    hls::stream<Ts, Fs>& ...streams
) {
    constexpr std::size_t HeaderWidth = bitwidth(sizeof...(streams));
    constexpr std::size_t DataWidth = InWidth - HeaderWidth;
    static_assert(InWidth > HeaderWidth);

    // We can simply assume ap_uint for both, this will be casted to ap_int
    // if necessary when writing into the target stream - right now we don't
    // know the type of the target stream. (Its possible to retrieve here, but
    // not necessary).
    using D = ap_uint<DataWidth>;
    const auto split = splitHeader<HeaderWidth, D, InWidth>(in.read());
    ap_uint<HeaderWidth> header = std::get<0>(split);
    ap_uint<DataWidth> data = std::get<1>(split);
#if !defined(__SYNTHESIS__) && defined(MUX_DEBUG)
    std::cout << "(Demux) Header: " << header << " (" << std::bitset<InWidth>(header) << "). ";
    std::cout << "Data: " << data << " (" << std::bitset<InWidth>(data) << "). " << std::endl;
#endif

    // Demux the current value
    _static_demux(static_cast<std::size_t>(header), data, in, streams...);
}

#endif
