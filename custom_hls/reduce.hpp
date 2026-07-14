#ifndef REDUCE
#define REDUCE

#include "utils.hpp"

#include <algorithm>
#include <limits>
#include <type_traits>

// -----------------------------------------------------------------------------
// Detection idiom (C++14) to constrain Functor compatibility
// Required interface for F = Functor<TO>:
//   TO   init() const;
//   void operator()(TO& accu, const TI& x) const;
// -----------------------------------------------------------------------------
template <typename...> using void_t = void;

template <template <typename> class Functor, typename TO, typename TI,
          typename = void>
struct is_compatible_reducer : std::false_type {};

template <template <typename> class Functor, typename TO, typename TI>
struct is_compatible_reducer<
    Functor, TO, TI,
    void_t<decltype(std::declval<Functor<TO>>().init()),
           decltype(std::declval<Functor<TO>>()(std::declval<TO &>(),
                                                std::declval<const TI &>()))>> {
private:
  using F = Functor<TO>;

public:
  static constexpr bool value =
      std::is_convertible<decltype(std::declval<F>().init()), TO>::value &&
      std::is_same<decltype(std::declval<F>()(std::declval<TO &>(),
                                              std::declval<const TI &>())),
                   void>::value;
};

// -----------------------------------------------------------------------------
// Example functors
// -----------------------------------------------------------------------------

//-----------------------------SUM----------------------------------------------
template <typename T> struct Sum {
  T init() const {
#pragma HLS inline
    return T{};
  }

  template <typename U> void operator()(T &accu, const U &x) const {
#pragma HLS inline
    accu += static_cast<T>(x);
  }
};
// Specialization for hls::vector
template <typename T, std::size_t PE> struct Sum<hls::vector<T, PE>> {
  hls::vector<T, PE> init() const {
#pragma HLS inline
    return hls::vector<T, PE>{};
  }

  template <typename U>
  void operator()(hls::vector<T, PE> &accu, const hls::vector<U, PE> &x) const {
#pragma HLS inline
    for (std::size_t i = 0; i < PE; i++) {
#pragma HLS unroll
      accu[i] += static_cast<T>(x[i]);
    }
  }
};

//-----------------------------MIN----------------------------------------------
template <typename T> struct Min {
  T init() const {
#pragma HLS inline
    return std::numeric_limits<T>::max();
  }

  template <typename U> void operator()(T &accu, const U &x) const {
#pragma HLS inline
    T const xv = static_cast<T>(x);
    accu = (xv < accu) ? xv : accu;
  }
};
// Specialization for hls::vector
template <typename T, std::size_t PE> struct Min<hls::vector<T, PE>> {
  hls::vector<T, PE> init() const {
#pragma HLS inline
    return hls::vector<T, PE>(std::numeric_limits<T>::max());
  }

  template <typename U>
  void operator()(hls::vector<T, PE> &accu, const hls::vector<U, PE> &x) const {
#pragma HLS inline
    for (std::size_t i = 0; i < PE; i++) {
#pragma HLS unroll
      T const xv = static_cast<T>(x[i]);
      accu[i] = (xv < accu[i]) ? xv : accu[i];
    }
  }
};

//-----------------------------MAX----------------------------------------------
template <typename T> struct Max {
  T init() const {
#pragma HLS inline
    return std::numeric_limits<T>::lowest();
  }

  template <typename U> void operator()(T &accu, const U &x) const {
#pragma HLS inline
    T const xv = static_cast<T>(x);
    accu = (accu < xv) ? xv : accu;
  }
};
// Specialization for hls::vector
template <typename T, std::size_t PE> struct Max<hls::vector<T, PE>> {
  hls::vector<T, PE> init() const {
#pragma HLS inline
    return hls::vector<T, PE>(std::numeric_limits<T>::lowest());
  }

  template <typename U>
  void operator()(hls::vector<T, PE> &accu, const hls::vector<U, PE> &x) const {
#pragma HLS inline
    for (std::size_t i = 0; i < PE; i++) {
#pragma HLS unroll
      T const xv = static_cast<T>(x[i]);
      accu[i] = (accu[i] < xv) ? xv : accu[i];
    }
  }
};

// --------------------------PRODUCT-------------------------------------------
template <typename T> struct Product {
  T init() const {
#pragma HLS inline
    return T{1};
  }

  template <typename U> void operator()(T &accu, const U &x) const {
#pragma HLS inline
    accu *= static_cast<T>(x);
  }
};
// Specialization for hls::vector
template <typename T, std::size_t PE> struct Product<hls::vector<T, PE>> {
  hls::vector<T, PE> init() const {
#pragma HLS inline
    return hls::vector<T, PE>(1);
  }

  template <typename U>
  void operator()(hls::vector<T, PE> &accu, const hls::vector<U, PE> &x) const {
#pragma HLS inline
    for (std::size_t i = 0; i < PE; i++) {
#pragma HLS unroll
      accu[i] *= static_cast<T>(x[i]);
    }
  }
};

// -----------------------------------------------------------------------------
// Channel writeback helpers
//
// C * PE == the total number of channels: PE channels are processed in
// parallel per beat (TI/TO = hls::vector<T,PE>), and C is how many such
// chunks are needed to cover all of them.
//
// A spatial reduction (ReduceAccuToSingle = false) leaves every channel
// intact: each of the C accumulators -- still PE channels wide -- is
// written back on its own. Only the accumulator's numeric type may change;
// ChannelPreservingCast enforces that TDst keeps TO's width.
//
// A depthwise reduction (ReduceAccuToSingle = true) additionally reduces
// over the channels themselves, with the same Functor, until a single
// channel is left: the C chunks are combined first (handled by the loop
// below), and then, once TDst asks for a single channel, ChannelReducingCast
// folds the PE lanes of that combined result together too, using the scalar
// specialization of Functor.
//
// reduce_element<T> extracts a type's element type and width (1 for a
// plain scalar T, PE for hls::vector<T,PE>).
// -----------------------------------------------------------------------------
template <typename T> struct reduce_element {
  using type = T;
  static constexpr std::size_t width = 1;
};
template <typename T, std::size_t W> struct reduce_element<hls::vector<T, W>> {
  using type = T;
  static constexpr std::size_t width = W;
};

template <typename T> struct is_reduce_vector : std::false_type {};
template <typename T, std::size_t PE>
struct is_reduce_vector<hls::vector<T, PE>> : std::true_type {};

// Numeric cast of a single value -- never combines channels.
template <typename Sin, typename Sout> struct ElementCast {
  static_assert(std::is_convertible<Sin, Sout>::value,
                "No known conversion between the accumulator's and the "
                "output stream's element type");
  static Sout apply(const Sin &v) {
#pragma HLS inline
    return static_cast<Sout>(v);
  }
};
template <typename S> struct ElementCast<S, S> {
  static S apply(const S &v) {
#pragma HLS inline
    return v;
  }
};

// Cast + wrap a single scalar value into whatever "one channel" looks like
// as a TDst (a bare Sout, or a width-1 hls::vector<Sout,1>).
template <typename Sin, typename TDst> struct SingleValueCast {
  static TDst apply(const Sin &v) {
#pragma HLS inline
    return ElementCast<Sin, TDst>::apply(v);
  }
};
template <typename Sin, typename Sout>
struct SingleValueCast<Sin, hls::vector<Sout, 1>> {
  static hls::vector<Sout, 1> apply(const Sin &v) {
#pragma HLS inline
    hls::vector<Sout, 1> out;
    out[0] = ElementCast<Sin, Sout>::apply(v);
    return out;
  }
};

// Elementwise cast between two vectors of the same width (numeric type
// change only, channels are never combined).
template <typename TO, typename TDst> struct SameWidthVectorCast;
template <typename Sin, typename Sout, std::size_t W>
struct SameWidthVectorCast<hls::vector<Sin, W>, hls::vector<Sout, W>> {
  static hls::vector<Sout, W> apply(const hls::vector<Sin, W> &v) {
#pragma HLS inline
    hls::vector<Sout, W> out;
    for (std::size_t i = 0; i < W; i++) {
#pragma HLS unroll
      out[i] = ElementCast<Sin, Sout>::apply(v[i]);
    }
    return out;
  }
};

// -- Spatial (channel-preserving) writeback cast -----------------------------
template <typename TO, typename TDst, typename = void>
struct ChannelPreservingCast {
  static_assert(sizeof(TO) == 0,
                "ChannelPreservingCast: a spatial reduction leaves every "
                "channel intact, so TDst must keep TO's width (only its "
                "numeric type may differ). Use the depthwise "
                "(ReduceAccuToSingle = true) form if you want fewer output "
                "channels than input channels.");
};

template <typename T> struct ChannelPreservingCast<T, T, void> {
  static T apply(const T &v) {
#pragma HLS inline
    return v;
  }
};

template <typename TO, typename TDst>
struct ChannelPreservingCast<
    TO, TDst,
    typename std::enable_if<!std::is_same<TO, TDst>::value &&
                            !is_reduce_vector<TO>::value>::type> {
  static TDst apply(const TO &v) {
#pragma HLS inline
    return ElementCast<TO, TDst>::apply(v);
  }
};

template <typename TO, typename TDst>
struct ChannelPreservingCast<
    TO, TDst,
    typename std::enable_if<
        !std::is_same<TO, TDst>::value && is_reduce_vector<TO>::value &&
        (reduce_element<TO>::width == reduce_element<TDst>::width)>::type> {
  static TDst apply(const TO &v) {
#pragma HLS inline
    return SameWidthVectorCast<TO, TDst>::apply(v);
  }
};

// -- Depthwise (channel-reducing) writeback cast -----------------------------
// Folds the PE lanes of TO together with Functor, down to whatever width
// TDst asks for: 1, to leave a single channel (the defining behavior of a
// depthwise reduction), or TO's own width, to only change the numeric type
// without reducing further.
template <template <typename> class Functor, typename TVec> struct LaneReduce;

template <typename T, std::size_t PE>
struct LaneReduce<Sum, hls::vector<T, PE>> {
  // static_assert(is_compatible_reducer<Sum, T, T>::value,
  //               "The scalar specialization Sum<T> is required to fold a "
  //               "channelwise accumulator's lanes down to a single channel");
  static T apply(const hls::vector<T, PE> &v) { return v.reduce_add(); }
};

template <typename T, std::size_t PE>
struct LaneReduce<Product, hls::vector<T, PE>> {
  static T apply(const hls::vector<T, PE> &v) { return v.reduce_mul(); }
};

template <template <typename> class Functor, typename T, std::size_t PE>
struct LaneReduce<Functor, hls::vector<T, PE>> {
  static_assert(is_compatible_reducer<Functor, T, T>::value,
                "The scalar specialization Functor<T> is required to fold a "
                "channelwise accumulator's lanes down to a single channel");
  static T apply(const hls::vector<T, PE> &v) {
#pragma HLS inline
    Functor<T> fct;
    T accu = fct.init();
    for (std::size_t i = 0; i < PE; i++) {
#pragma HLS unroll
      fct(accu, v[i]);
    }
    return accu;
  }
};

template <template <typename> class Functor, typename TO, typename TDst,
          typename = void>
struct ChannelReducingCast {
  static_assert(sizeof(TO) == 0,
                "ChannelReducingCast: unsupported conversion from the "
                "internal accumulator type TO to the output stream's "
                "element type TDst.");
};

template <template <typename> class Functor, typename T>
struct ChannelReducingCast<Functor, T, T, void> {
  static T apply(const T &v) {
#pragma HLS inline
    return v;
  }
};

template <template <typename> class Functor, typename TO, typename TDst>
struct ChannelReducingCast<
    Functor, TO, TDst,
    typename std::enable_if<!std::is_same<TO, TDst>::value &&
                            !is_reduce_vector<TO>::value>::type> {
  static TDst apply(const TO &v) {
#pragma HLS inline
    return ElementCast<TO, TDst>::apply(v);
  }
};

template <template <typename> class Functor, typename TO, typename TDst>
struct ChannelReducingCast<
    Functor, TO, TDst,
    typename std::enable_if<
        !std::is_same<TO, TDst>::value && is_reduce_vector<TO>::value &&
        (reduce_element<TO>::width == reduce_element<TDst>::width)>::type> {
  static TDst apply(const TO &v) {
#pragma HLS inline
    return SameWidthVectorCast<TO, TDst>::apply(v);
  }
};

// TO = hls::vector<T,PE>, TDst carries a single channel: fold all PE lanes
// together with Functor, leaving one channel.
template <template <typename> class Functor, typename TO, typename TDst>
struct ChannelReducingCast<
    Functor, TO, TDst,
    typename std::enable_if<is_reduce_vector<TO>::value &&
                            (reduce_element<TO>::width != 1) &&
                            (reduce_element<TDst>::width == 1)>::type> {
  static TDst apply(const TO &v) {
#pragma HLS inline
    return SingleValueCast<typename reduce_element<TO>::type, TDst>::apply(
        LaneReduce<Functor, TO>::apply(v));
  }
};

// -----------------------------------------------------------------------------
// Accumulator writeback helpers
//
// TDst is the element type actually written to the output stream and
// defaults to TO (the internal accumulator type), which preserves the
// original behavior exactly.
// -----------------------------------------------------------------------------
template <bool ReduceAccuToSingle, std::size_t C,
          template <typename> class Functor, typename TO, typename TDst = TO>
struct AccuWriteback;

// Spatial: channels stay intact. Each of the C accumulators is written
// back on its own, still PE channels wide.
template <std::size_t C, template <typename> class Functor, typename TO,
          typename TDst>
struct AccuWriteback<false, C, Functor, TO, TDst> {
  static void run(hls::stream<TDst> &dst, const TO (&accu)[C],
                  const Functor<TO> &fct) {
#pragma HLS inline
    (void)fct;
    for (std::size_t c = 0; c < C; c++) {
#pragma HLS unroll
      dst.write(ChannelPreservingCast<TO, TDst>::apply(accu[c]));
    }
  }
};

// Depthwise: reduce over the channels too. The C accumulators are
// combined with Functor first; then, if TDst asks for a single channel,
// the PE lanes of that combined result are folded together with the same
// Functor.
template <std::size_t C, template <typename> class Functor, typename TO,
          typename TDst>
struct AccuWriteback<true, C, Functor, TO, TDst> {
  static void run(hls::stream<TDst> &dst, const TO (&accu)[C],
                  const Functor<TO> &fct) {
#pragma HLS inline
    TO final_accu = fct.init();
    for (std::size_t c = 0; c < C; c++) {
#pragma HLS unroll
      fct(final_accu, accu[c]);
    }
    dst.write(ChannelReducingCast<Functor, TO, TDst>::apply(final_accu));
  }
};

// -----------------------------------------------------------------------------
// InnerReduce
// -----------------------------------------------------------------------------
template <std::size_t ISIZE, std::size_t C, bool ReduceAccuToSingle,
          template <typename> class Functor, typename TI, typename TO,
          typename TDst = TO,
          typename std::enable_if<is_compatible_reducer<Functor, TO, TI>::value,
                                  int>::type = 0>
void InnerReduce(hls::stream<TI> &src, hls::stream<TDst> &dst) {
#pragma HLS inline
  Functor<TO> fct;
  TO accu[C];
#pragma HLS ARRAY_PARTITION variable = accu complete dim = 1
  for (std::size_t c = 0; c < C; c++) {
#pragma HLS unroll
    accu[c] = fct.init();
  }
  for (std::size_t i = 0; i < ISIZE; i++) {
    for (std::size_t c = 0; c < C; c++) {
#pragma HLS pipeline II = 1 style = flp
      TI x = src.read();
      fct(accu[c], x);
    }
  }

  AccuWriteback<ReduceAccuToSingle, C, Functor, TO, TDst>::run(dst, accu, fct);
}

// -----------------------------------------------------------------------------
// Helpers for C++14: product of pack, and nth element of size_t pack
// -----------------------------------------------------------------------------
template <std::size_t I, std::size_t... Ns> struct PackAt;

template <std::size_t N0, std::size_t... Ns> struct PackAt<0, N0, Ns...> {
  static constexpr std::size_t value = N0;
};

template <std::size_t I, std::size_t N0, std::size_t... Ns>
struct PackAt<I, N0, Ns...> {
  static_assert(I < sizeof...(Ns) + 1, "PackAt index out of range");
  static constexpr std::size_t value = PackAt<I - 1, Ns...>::value;
};

// Product of D[I] * D[I+1] * ... * D[end]
template <std::size_t Start, std::size_t N, std::size_t... D>
struct PackProductFromImpl {
  static_assert(Start <= N, "PackProductFrom index out of range");
  static constexpr std::size_t value =
      PackAt<Start, D...>::value *
      PackProductFromImpl<Start + 1, N, D...>::value;
};

// Stop when Start == N
template <std::size_t N, std::size_t... D>
struct PackProductFromImpl<N, N, D...> {
  static constexpr std::size_t value = 1;
};

template <std::size_t Start, std::size_t... D> struct PackProductFrom {
  static_assert(Start <= sizeof...(D), "PackProductFrom index out of range");
  static constexpr std::size_t value =
      PackProductFromImpl<Start, sizeof...(D), D...>::value;
};
// -----------------------------------------------------------------------------
// OuterReduce implementation (C++14 specialization instead of if constexpr)
// -----------------------------------------------------------------------------
template <std::size_t Level, std::size_t ReductionDimStartIndex,
          typename Enable, std::size_t... D>
struct OuterReduceImpl;

// Base case: Level == ReductionDimStartIndex
template <std::size_t Level, std::size_t ReductionDimStartIndex,
          std::size_t... D>
struct OuterReduceImpl<
    Level, ReductionDimStartIndex,
    typename std::enable_if<(Level == ReductionDimStartIndex)>::type, D...> {
  template <std::size_t C, bool ReduceAccuToSingle,
            template <typename> class Functor, typename TI, typename TO,
            typename TDst = TO>
  static void run(hls::stream<TI> &src, hls::stream<TDst> &dst) {
#pragma HLS inline
    static_assert(ReductionDimStartIndex <= sizeof...(D),
                  "ReductionDimStartIndex out of range");
    constexpr std::size_t ISIZE =
        PackProductFrom<ReductionDimStartIndex, D...>::value;
    static_assert(is_compatible_reducer<Functor, TO, TI>::value,
                  "Functor is not compatible with the given types");
    InnerReduce<ISIZE, C, ReduceAccuToSingle, Functor, TI, TO, TDst>(src, dst);
  }
};

// Recursive case: Level < ReductionDimStartIndex
template <std::size_t Level, std::size_t ReductionDimStartIndex,
          std::size_t... D>
struct OuterReduceImpl<
    Level, ReductionDimStartIndex,
    typename std::enable_if<(Level < ReductionDimStartIndex)>::type, D...> {
  template <std::size_t C, bool ReduceAccuToSingle,
            template <typename> class Functor, typename TI, typename TO,
            typename TDst = TO>
  static void run(hls::stream<TI> &src, hls::stream<TDst> &dst) {
#pragma HLS inline
    constexpr std::size_t ISIZE = PackAt<Level, D...>::value;
    for (std::size_t i = 0; i < ISIZE; i++) {
      OuterReduceImpl<Level + 1, ReductionDimStartIndex, void, D...>::
          template run<C, ReduceAccuToSingle, Functor, TI, TO, TDst>(src, dst);
    }
  }
};

// -----------------------------------------------------------------------------
// Automatic derivation of the internal accumulator type TO
//
// The accumulator type is not named by the caller: it is derived from the
// input stream's element type TI and the output stream's element type TDst.
//
//   TI = T                  (scalar)      -> TO = S
//   TI = hls::vector<T, PE>                -> TO = hls::vector<S, PE>
//
// where S is TDst's element type (TDst's own type if TDst is scalar, or its
// element type if TDst is hls::vector<S, M>). TO always keeps TI's full
// width PE, regardless of TDst's own width M, so the elementwise vector
// Functor specializations (Sum<hls::vector<S, PE>>, ...) can accumulate one
// independent partial result per channel during InnerReduce. At writeback
// time a spatial reduction keeps M = PE (ChannelPreservingCast, channels
// left intact); a depthwise reduction may narrow to M = 1
// (ChannelReducingCast, all channels folded into one with the same
// Functor) -- see the writeback helpers above.
//
// For scalar TI there is only ever one channel (PE = 1), so TO = S = TDst
// directly and there is nothing left to fold.
// -----------------------------------------------------------------------------
template <typename TI, typename TDst> struct DeduceAccuType {
  using S = typename reduce_element<TDst>::type;
  static constexpr std::size_t N = reduce_element<TI>::width;
  using type = typename std::conditional<is_reduce_vector<TI>::value,
                                         hls::vector<S, N>, S>::type;
};

// -----------------------------------------------------------------------------
// Reduction mode
// -----------------------------------------------------------------------------
enum class ReductionMode {
  // Only the ISIZE (spatial) dimension is reduced; every channel is kept
  // intact.
  Spatial,
  // The channels themselves are also reduced, with the same Functor, down
  // to a single channel (see AccuWriteback/ChannelReducingCast above).
  Depthwise,
};

// Public API
/**
 * @brief OuterReduce performs a reduction operation on a multi-dimensional
 *        stream of data, reducing along specified dimensions.
 *
 * @tparam Functor A template class that defines the reduction operation.
 * @tparam C Number of PE-wide chunks needed to cover all channels, i.e.
 *           C * PE == total number of channels (PE is the vector width of
 *           TI/TDst).
 * @tparam ReductionDimStartIndex The index of the first dimension to reduce.
 * @tparam Mode ReductionMode::Spatial (reduce ISIZE only, channels kept
 *              intact -- TDst must keep TI's vector width) or
 *              ReductionMode::Depthwise (also reduce over every channel,
 *              down to one -- TDst may be a single channel: plain T, or
 *              hls::vector<T,1>).
 * @tparam D A parameter pack representing the sizes of each dimension.
 * @tparam TI The input type of the stream elements.
 * @tparam TDst The output type of the stream elements after reduction.
 *
 * @param src The input stream containing multi-dimensional data.
 * @param dst The output stream where the reduced data will be written.
 *
 * The internal accumulator type is derived automatically from TI and TDst
 * (see DeduceAccuType above) and is never named by the caller.
 *
 * Example: PE-wide channels, tensor shape {N=1, H=4, W=2}, reducing H and
 * W (ReductionDimStartIndex = 1) down to a single channel per group of PE:
 *
 *   OuterReduce<Sum, C, 1, ReductionMode::Depthwise, 1, 4, 2>(src, dst);
 */
template <template <typename> class Functor, std::size_t C,
          std::size_t ReductionDimStartIndex, ReductionMode Mode,
          std::size_t... D, typename TI, typename TDst>
void OuterReduce(hls::stream<TI> &src, hls::stream<TDst> &dst) {
  static_assert(ReductionDimStartIndex <= sizeof...(D),
                "ReductionDimStartIndex out of range");
  using TO = typename DeduceAccuType<TI, TDst>::type;
  constexpr bool ReduceAccuToSingle = (Mode == ReductionMode::Depthwise);
  OuterReduceImpl<0, ReductionDimStartIndex, void, D...>::template run<
      C, ReduceAccuToSingle, Functor, TI, TO, TDst>(src, dst);
}

#endif /* REDUCE */
