/******************************************************************************
 *  Copyright (c) 2023, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************************************
 * @brief	Instrumentation wrapper module for FINN IP characterization.
 * @author	Thomas B. Preusser <thomas.preusser@amd.com>
 * @details
 *	Instrumentation wrapper intercepting the feature map input to and
 *	the feature map output from a FINN IP to measure processing latency and
 *	initiation interval in terms of clock cycles. The most recent readings
 *	are exposed via AXI-light.
 *	This wrapper can run the FINN IP detached from an external data source
 *	and sink by feeding LFSR-generated data and sinking the output without
 *	backpressure.
 *	This module is currently not integrated with the FINN compiler. It must
 *	be instantiated and integrated with the rest of the system in a manual
 *	process.
 *
 *	In addition to the sliding-window averages, a simple long-running
 *	throughput measurement is provided via run_cycles_lo/hi and run_frames.
 *	Both counters reset on the rising edge of cfg[0] (i.e. when LFSR
 *	generation is enabled). run_cycles counts every clock cycle elapsed since
 *	that edge; run_frames counts every completed output frame. Dividing
 *	run_cycles by run_frames in software gives the average initiation interval
 *	over an arbitrarily large window without any on-chip buffer.
 *
 * @param PENDING	maximum number of feature maps in the FINN dataflow pipeline
 * @param ILEN		number of input transactions per IFM
 * @param OLEN		number of output transactions per OFM
 * @param KO        number of subwords within output payload vector
 * @param TI		type of input payload vector
 * @param TO		type of output payload vector
 *******************************************************************************/

 #include <hls_stream.h>
 #include <ap_int.h>
 #include <ap_axi_sdata.h>
 #include <algorithm>

 // Module Configuration
 constexpr unsigned  PENDING          = @PENDING@;          // Max. feature maps in flight
 constexpr unsigned  ILEN             = @ILEN@;             // Input words per IFM
 constexpr unsigned  OLEN             = @OLEN@;             // Output words per OFM
 constexpr unsigned  KO               = @KO@;               // Subwords within OFM transaction word
 constexpr unsigned  AVG_N            = @AVG_N@;            // Max frames in averaging window
 constexpr unsigned  TUSER_WIDTH      = @TUSER_WIDTH@;      // Width of tUSER field on finnix output
 constexpr unsigned  NUM_TUSER_VALUES = @NUM_TUSER_VALUES@; // Number of round-robin tUSER values (1 = fixed at 0)
 using  TI = @TI@;  // IFM transaction word
 using  TO = @TO@;  // OFM transaction word

 //---------------------------------------------------------------------------
 // Utility Functions
 static constexpr unsigned clog2  (unsigned  x) { return  x<2? 0 : 1+clog2((x+1)/2); }
 static constexpr unsigned clog2nz(unsigned  x) { return  std::max(1u, clog2(x)); }

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

 // Unpack internal {last, user, data} word into an hls::axis beat for the interface port.
 // hls::axis types must NOT appear in internal hls::stream variables (HLS 214-208).
 template<typename T, unsigned TUW>
 static void move(
     hls::stream<ap_uint<T::width + TUW + 1>> &src,
     hls::stream<hls::axis<T, TUW, 0, 0>> &dst
 ) {
 #pragma HLS pipeline II=1 style=flp
     ap_uint<T::width + TUW + 1> const  packed = src.read();
     hls::axis<T, TUW, 0, 0>  beat;
     beat.data = packed(T::width - 1, 0);
     beat.user = packed(T::width + TUW - 1, T::width);
     beat.last = packed[T::width + TUW];
     beat.keep = -1;
     dst.write(beat);
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

 /**
  * Computes a checksum over a forwarded stream assumed to carry frames of
  * N words further subdivided into K subwords.
  *      - Subword slicing can be customized typically by using a lambda.
  *        The provided DefaultSubwordSlicer assumes an `ap_(u)int`-like word
  *        type with a member `width` and a range-based slicing operator. It
  *        further assumes a little-endian arrangement of subwords within words
  *        for the canonical subword stream order.
  *      - Subwords wider than 23 bits are folded using bitwise XOR across
  *        slices of 23 bits starting from the LSB.
  *      - The folded subword values are weighted according to their position
  *        in the stream relative to the start of frame by a periodic weight
  *        sequence 1, 2, 3, ...
  *      - The weighted folded subword values are reduced to a checksum by an
  *        accumulation module 2^24.
  *      - A checksum is emitted for each completed frame. It is the concatenation
  *        of an 8-bit (modulo 256) frame counter and the 24-bit frame checksum.
  */
 template<typename T, unsigned K>
 class DefaultSubwordSlicer {
     static_assert(T::width%K == 0, "Word size must be subword multiple.");
     static constexpr unsigned  W = T::width/K;
 public:
     ap_uint<W> operator()(T const &x, unsigned const  j) const {
 #pragma HLS inline
         return  x((j+1)*W-1, j*W);
     }
 };

 //---------------------------------------------------------------------------
 // Instrumentation Core
 template<
     unsigned  PENDING,
     unsigned  ILEN,
     unsigned  OLEN,
     unsigned  KO,
     unsigned  AVG_N,
     typename  TI,
     typename  TO
 >
 void instrument(
     hls::stream<ap_uint<TI::width + TUSER_WIDTH + 1>> &finnix,
     hls::stream<TO> &finnox,  // plain type — hls::axis stripped by move() in wrapper
     ap_uint<32>  cfg,          // [0] - 0:hold, 1:lfsr; [31:1] - minimum interval (cycles) between IFM starts
     ap_uint<32>  seed,         // [31:16] - LFSR seed (only upper 16 bits used)
     ap_uint<32>  avg_n,        // [31:0] - averaging window size (1..AVG_N frames)
     ap_uint<32>  mux_interval, // frames each tUSER value is held before advancing (0 = fixed at 0)
     ap_uint<32> &status,       // [0] - timestamp overflow; [1] - timestamp underflow
     ap_uint<32> &latency,
     ap_uint<32> &interval,
     ap_uint<32> &checksum,
     ap_uint<32> &min_latency,
     ap_uint<32> &lat_sum_lo,		// lower 32 bits of latency sliding-window sum (divide by avg_fill in SW)
     ap_uint<32> &lat_sum_hi,		// upper 32 bits of latency sliding-window sum
     ap_uint<32> &int_sum_lo,		// lower 32 bits of interval sliding-window sum
     ap_uint<32> &int_sum_hi,		// upper 32 bits of interval sliding-window sum
     ap_uint<32> &avg_fill_out,		// number of valid entries in sliding window (divisor)
     ap_uint<32> &run_cycles_lo,	// lower 32 bits of cycle count since cfg[0] rising edge
     ap_uint<32> &run_cycles_hi,	// upper 32 bits of cycle count since cfg[0] rising edge
     ap_uint<32> &run_frames		// completed output frames since cfg[0] rising edge
 ) {
 #pragma HLS pipeline II=1 style=flp

     // Timestamp Management State
     using clock_t = ap_uint<32>;
     static clock_t  cnt_clk = 0;
 #pragma HLS reset variable=cnt_clk
     hls::stream<clock_t>  timestamps;
 #pragma HLS stream variable=timestamps depth=PENDING
     static bool  timestamp_ovf = false;
     static bool  timestamp_unf = false;
 #pragma HLS reset variable=timestamp_ovf
 #pragma HLS reset variable=timestamp_unf

     // Input Feed & Generation
     constexpr unsigned  LFSR_WIDTH = (TI::width+15)/16 * 16;
     static ap_uint<clog2nz(ILEN)>  icnt = 0;
     static ap_uint<LFSR_WIDTH>  lfsr;
     static clock_t  last_ifm_start = 0;  // Timestamp of last IFM transmission start
 #pragma HLS reset variable=icnt
 #pragma HLS reset variable=lfsr off
 #pragma HLS reset variable=last_ifm_start

     // tUSER schedule state
     static ap_uint<TUSER_WIDTH>  tuser_val     = 0; // current tUSER value driven on finnix
     static ap_uint<32>           frame_mux_cnt = 0; // frames elapsed at current tuser_val
 #pragma HLS reset variable=tuser_val
 #pragma HLS reset variable=frame_mux_cnt

     if(!finnix.full()) {

         bool const  first = icnt == 0;
         bool  wr = false;

         // Rate limiting: enforce minimum interval between IFM starts
         ap_uint<31> const  min_interval = cfg(31, 1);
         bool const  interval_ok = (min_interval == 0) || ((cnt_clk - last_ifm_start) >= min_interval);

         if(first) {
             // Start of new feature map (only if minimum interval elapsed)
             if(interval_ok) {
                 wr = cfg[0];
                 if(wr) {
                     // Initialize LFSR with configurable seed
                     for(unsigned  i = 0; i < LFSR_WIDTH; i += 16) {
 #pragma HLS unroll
                         lfsr(15+i, i) = seed(31, 16) ^ (i>>4)*33331;
                     }
                     last_ifm_start = cnt_clk;  // Record start timestamp
                 }
             }
         }
         else {
             // Advance LFSR
             wr = true;
             for(unsigned  i = 0; i < LFSR_WIDTH; i += 16) {
 #pragma HLS unroll
                 lfsr(15+i, i) = (lfsr(15+i, i) >> 1) ^ ap_uint<16>(lfsr[i]? 0 : 0x8805);
             }
         }

         if(wr) {
             bool const  frame_last = (icnt == ILEN-1);
             // Pack {last[1], user[TUSER_WIDTH], data[TI::width]} into a plain ap_uint.
             // hls::axis cannot be used in internal streams (HLS 214-208); the wrapper's
             // move() dataflow process reconstructs the axis beat for the interface port.
             ap_uint<TI::width + TUSER_WIDTH + 1>  beat;
             beat(TI::width - 1, 0)                     = lfsr;
             beat(TI::width + TUSER_WIDTH - 1, TI::width) = tuser_val;
             beat[TI::width + TUSER_WIDTH]              = frame_last ? ap_uint<1>(1) : ap_uint<1>(0);
             finnix.write_nb(beat);
             if(first)  timestamp_ovf |= !timestamps.write_nb(cnt_clk);
             // After the last beat of a frame, advance the tUSER round-robin schedule
             if(frame_last && NUM_TUSER_VALUES > 1 && mux_interval > 0) {
                 if(frame_mux_cnt >= ap_uint<32>(mux_interval - 1)) {
                     frame_mux_cnt = 0;
                     tuser_val = (tuser_val == ap_uint<TUSER_WIDTH>(NUM_TUSER_VALUES - 1))
                                 ? ap_uint<TUSER_WIDTH>(0)
                                 : ap_uint<TUSER_WIDTH>(tuser_val + 1);
                 } else {
                     frame_mux_cnt++;
                 }
             }
             icnt = frame_last? decltype(icnt)(0) : decltype(icnt)(icnt + 1);
         }
     }

     // Output Tracking
     static ap_uint<clog2nz(OLEN)>  ocnt = 0;
 #pragma HLS reset variable=ocnt
     static clock_t  ts1 = 0;	// last output timestamp
     static clock_t  last_latency = 0;
     static clock_t  last_interval = 0;
     static clock_t  cur_min_latency = ~0;
 #pragma HLS reset variable=ts1
 #pragma HLS reset variable=last_latency
 #pragma HLS reset variable=last_interval
 #pragma HLS reset variable=cur_min_latency

     // Sliding-Window Averaging State
     static ap_uint<clog2nz(AVG_N)>    avg_head = 0;  // write pointer in circular buffer
     static ap_uint<clog2nz(AVG_N+1)>  avg_fill = 0;  // number of valid entries (0..AVG_N)
     static clock_t  lat_buf[AVG_N];
     static clock_t  int_buf[AVG_N];
     static ap_uint<64>  lat_sum = 0;
     static ap_uint<64>  int_sum = 0;
     static ap_uint<32>  prev_avg_n = 0;
 #pragma HLS reset variable=avg_head
 #pragma HLS reset variable=avg_fill
 #pragma HLS reset variable=lat_buf off
 #pragma HLS reset variable=int_buf off
 #pragma HLS reset variable=lat_sum
 #pragma HLS reset variable=int_sum
 #pragma HLS reset variable=prev_avg_n

     // Running Throughput Measurement State (resets on rising edge of cfg[0])
     static bool  prev_cfg0  = false;
     static bool  run_active = false;  // true after the first cfg[0] rising edge
     static ap_uint<64>  run_total_cycles = 0;
     static ap_uint<32>  run_frame_count  = 0;
 #pragma HLS reset variable=prev_cfg0
 #pragma HLS reset variable=run_active
 #pragma HLS reset variable=run_total_cycles
 #pragma HLS reset variable=run_frame_count

     static ap_uint<8>  pkts = 0;
 #pragma HLS reset variable=pkts
     static ap_uint< 2>  coeff[3];
     static ap_uint<24>  psum;
     static ap_uint<32>  last_checksum = 0;
 #pragma HLS reset variable=coeff off
 #pragma HLS reset variable=psum off
 #pragma HLS reset variable=last_checksum

     // Detect rising edge of cfg[0]: reset running throughput counters
     bool const  cur_cfg0    = cfg[0];
     if(cur_cfg0 && !prev_cfg0) {
         run_active       = true;
         run_total_cycles = 0;
         run_frame_count  = 0;
     }

     TO  oval;
     if(finnox.read_nb(oval)) {
         // Start of new output feature map
         if(ocnt == 0) {
             for(unsigned  i = 0; i < 3; i++)  coeff[i] = i+1;
             psum = 0;
         }

         // Update checksum
         for(unsigned  j = 0; j < KO; j++) {
 #pragma HLS unroll
             auto const  v0 = DefaultSubwordSlicer<TO, KO>()(oval, j);
             constexpr unsigned  W = 1 + (decltype(v0)::width-1)/23;
             ap_uint<W*23>  v = v0;	// Expand to width as multiple of 23
             ap_uint<  23>  w = 0;	// XOR across all 23-bit slices
             for(unsigned  k = 0; k < W; k++)  w ^= v(23*(k+1)-1, 23*k);
             psum += (coeff[j%3][1]? (w, ap_uint<1>(0)) : ap_uint<24>(0)) + (coeff[j%3][0]? w : ap_uint<23>(0));
         }

         // Re-align coefficients
         for(unsigned  j = 0; j < 3; j++) {
 #pragma HLS unroll
                 ap_uint<3> const  cc = coeff[j] + ap_uint<3>(KO%3);
                 coeff[j] = cc(1, 0) + cc[2];
         }

         // Track frame position
         if(ocnt != OLEN-1)  ocnt++;
         else {
             clock_t  ts0;
             if(!timestamps.read_nb(ts0))  timestamp_unf = true;
             else {
                 last_latency  = cnt_clk - ts0;	// completion - start
                 last_interval = cnt_clk - ts1;	// completion - previous completion
                 cur_min_latency = std::min(cur_min_latency, last_latency);
                 ts1 = cnt_clk;	// mark completion ^

                 // Sliding-window average update
                 // TODO: II=1 but depth is ~70 cycles, can we optimize this?
                 ap_uint<32>  win = (avg_n == 0 || avg_n > AVG_N) ? ap_uint<32>(AVG_N) : avg_n;
                 if(prev_avg_n != win) {
                     avg_head = 0;
                     avg_fill = 0;
                     lat_sum  = 0;
                     int_sum  = 0;
                     prev_avg_n = win;
                 }
                 clock_t  old_lat = lat_buf[avg_head];
                 clock_t  old_int = int_buf[avg_head];
                 lat_buf[avg_head] = last_latency;
                 int_buf[avg_head] = last_interval;
                 if(avg_fill < win) {
                     lat_sum += last_latency;
                     int_sum += last_interval;
                     avg_fill++;
                 } else {
                     lat_sum = lat_sum + last_latency - old_lat;
                     int_sum = int_sum + last_interval - old_int;
                 }
                 avg_head++;
                 if(avg_head >= ap_uint<clog2nz(AVG_N)+1>(win))  avg_head = 0;
             }
             ocnt = 0;
             if(run_active)  run_frame_count++;

             last_checksum = (pkts++, psum);
         }
     }

     // Advance Timestamp Counter
     cnt_clk++;

     // Advance Running Throughput Counters
     if(run_active)  run_total_cycles++;
     prev_cfg0 = cur_cfg0;

     // Copy Status Outputs
     status = timestamp_ovf | (timestamp_unf << 1);
     latency  = last_latency;
     interval = last_interval;
     checksum = last_checksum;
     min_latency  = cur_min_latency;
     lat_sum_lo = lat_sum(31,  0);
     lat_sum_hi = lat_sum(63, 32);
     int_sum_lo = int_sum(31,  0);
     int_sum_hi = int_sum(63, 32);
     avg_fill_out = avg_fill;
     run_cycles_lo = run_total_cycles(31,  0);
     run_cycles_hi = run_total_cycles(63, 32);
     run_frames    = run_frame_count;

 } // instrument()

 void instrumentation_wrapper(
     hls::stream<hls::axis<TI, TUSER_WIDTH, 0, 0>> &finnix,
     hls::stream<TO> &finnox,
     ap_uint<32>  cfg,
     ap_uint<32>  seed,
     ap_uint<32>  avg_n,
     ap_uint<32>  mux_interval,
     ap_uint<32> &status,
     ap_uint<32> &latency,
     ap_uint<32> &interval,
     ap_uint<32> &checksum,
     ap_uint<32> &min_latency,
     ap_uint<32> &lat_sum_lo,
     ap_uint<32> &lat_sum_hi,
     ap_uint<32> &int_sum_lo,
     ap_uint<32> &int_sum_hi,
     ap_uint<32> &avg_fill,
     ap_uint<32> &run_cycles_lo,
     ap_uint<32> &run_cycles_hi,
     ap_uint<32> &run_frames
 ) {
 #pragma HLS interface axis port=finnix
 #pragma HLS interface axis port=finnox
 #pragma HLS interface s_axilite bundle=ctrl port=cfg
 #pragma HLS interface s_axilite bundle=ctrl port=seed
 #pragma HLS interface s_axilite bundle=ctrl port=avg_n
 #pragma HLS interface s_axilite bundle=ctrl port=mux_interval
 #pragma HLS interface s_axilite bundle=ctrl port=status
 #pragma HLS interface s_axilite bundle=ctrl port=latency
 #pragma HLS interface s_axilite bundle=ctrl port=interval
 #pragma HLS interface s_axilite bundle=ctrl port=checksum
 #pragma HLS interface s_axilite bundle=ctrl port=min_latency
 #pragma HLS interface s_axilite bundle=ctrl port=lat_sum_lo
 #pragma HLS interface s_axilite bundle=ctrl port=lat_sum_hi
 #pragma HLS interface s_axilite bundle=ctrl port=int_sum_lo
 #pragma HLS interface s_axilite bundle=ctrl port=int_sum_hi
 #pragma HLS interface s_axilite bundle=ctrl port=avg_fill
 #pragma HLS interface s_axilite bundle=ctrl port=run_cycles_lo
 #pragma HLS interface s_axilite bundle=ctrl port=run_cycles_hi
 #pragma HLS interface s_axilite bundle=ctrl port=run_frames
 #pragma HLS interface ap_ctrl_none port=return

 #pragma HLS dataflow disable_start_propagation
     // Internal FIFO for finnix uses a plain ap_uint packing {last, user, data}.
     // hls::axis types are forbidden in internal hls::stream variables (HLS 214-208).
     // The move<TI, TUSER_WIDTH> overload below reconstructs the axis beat for the port.
     static hls::stream<ap_uint<TI::width + TUSER_WIDTH + 1>>  finnix0;
 #pragma HLS stream variable=finnix0 depth=2
     static hls::stream<Payload<TO>::type>  finnox0;
 #pragma HLS stream variable=finnox0 depth=2

     // AXI-Stream -> FIFO
     move(finnox, finnox0);

     // Main — writes to internal FIFO (enables II=1 in instrument)
     // TI and TO must be explicit: TI cannot be deduced from ap_uint<TI::width+...>
     instrument<PENDING, ILEN, OLEN, KO, AVG_N, TI, TO>(finnix0, finnox0, cfg, seed, avg_n, mux_interval, status, latency, interval, checksum, min_latency, lat_sum_lo, lat_sum_hi, int_sum_lo, int_sum_hi, avg_fill, run_cycles_lo, run_cycles_hi, run_frames);

     // FIFO -> AXI-Stream: reconstruct hls::axis beat from packed internal word
     move<TI, TUSER_WIDTH>(finnix0, finnix);

 } // instrumentation_wrapper
