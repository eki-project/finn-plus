#ifndef IODMA_HPP
#define IODMA_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_burst_maxi.h>
#include <etc/autopilot_ssdm_op.h>  // ap_wait

// Building blocks of the IODMA_hls kernels (see iodma_hls.py for how they are put together).
//
// Every process works on whole frames. A frame is a multiple of the memory word width IW (InsertIODMA
// picks IW that way) and of the stream width SW, hence of lcm(IW, SW), so width conversion never has
// to look at frame boundaries.
//
// Register DMAs (ap_ctrl_hs, the interface the PYNQ driver drives): each run moves numReps frames.
// Ring DMAs (ap_ctrl_none, for the C++ driver): never stop; frames are requested through a
// descriptor ring in AXI-lite space.
//
// With N replicated pipelines, frame g of the global frame sequence goes to (comes from) pipeline
// g % N. The per-port buffers between splitter/gatherer and the pipelines hold one frame each, so
// the memory side can serve the other pipelines while one of them is still consuming its frame.
//
// FOREVER = true turns a process into a free-running loop (ring DMAs); otherwise it handles the
// given amount of data and returns. HLS cannot make an endless loop a flushable pipeline and falls
// back to style=stp, where an iteration blocked on a read also stops the earlier iterations: their
// writes would wait for the next input, and the last words of a frame would only leave once the
// next frame arrives (a batch never completes). The free-running loops therefore read with read_nb
// and stall only on writes, i.e. on downstream backpressure.
//
// The ring DMAs announce a completed descriptor through cons. Nothing ties that register write to the
// memory port, and HLS left to itself writes it at the start of the descriptor, before the data has
// moved: the host then reused input buffers that were still being read and read outputs that were
// still being written. Each cons write therefore sits in a fixed-protocol region behind the last memory
// access of its descriptor.

namespace iodma {

// Frames of a run of numFrames frames that belong to pipeline P of N.
template<unsigned P, unsigned N>
unsigned port_frames(unsigned numFrames) {
#pragma HLS INLINE
	return numFrames / N + (P < numFrames % N ? 1 : 0);
}

// Register DMA: read numFrames consecutive frames (one burst-inferred loop).
template<unsigned IW>
void read_words(ap_uint<IW> const *mem, hls::stream<ap_uint<IW>> &out, unsigned wordsPerFrame,
                unsigned numFrames) {
	const unsigned numWords = numFrames * wordsPerFrame;
	for (unsigned i = 0; i < numWords; i++) {
#pragma HLS PIPELINE II=1 style=flp
		out.write(mem[i]);
	}
}

// Register DMA: write numFrames consecutive frames. The kernel reports done only after all write
// responses have arrived, so the data is visible once the run is done.
template<unsigned IW>
void write_words(hls::stream<ap_uint<IW>> &in, ap_uint<IW> *mem, unsigned wordsPerFrame,
                 unsigned numFrames) {
	const unsigned numWords = numFrames * wordsPerFrame;
	for (unsigned i = 0; i < numWords; i++) {
#pragma HLS PIPELINE II=1 style=flp
		mem[i] = in.read();
	}
}

// Width conversion of whole frames for pipeline P of N; requires IN % OUT == 0 or OUT % IN == 0
// (two calls through lcm(IN, OUT) otherwise). inWordsPerFrame counts IN-wide words.
template<unsigned IN, unsigned OUT, unsigned P, unsigned N, bool FOREVER>
void convert(hls::stream<ap_uint<IN>> &in, hls::stream<ap_uint<OUT>> &out, unsigned inWordsPerFrame,
             unsigned numFrames) {
	static_assert(IN % OUT == 0 || OUT % IN == 0, "use two conversions through lcm(IN, OUT)");
	const unsigned numIn = port_frames<P, N>(numFrames) * inWordsPerFrame;
	if (FOREVER) {
		// same conversions as below, one input word per successful read_nb
		constexpr unsigned ratio = IN > OUT ? IN / OUT : OUT / IN;
		ap_uint<IN> x;
		ap_uint<IN> e = 0;  // IN > OUT: word being split
		ap_uint<OUT> a = 0; // IN < OUT: word being assembled
		unsigned k = 0;     // IN > OUT: output words left in e; IN < OUT: input words in a
		for (;;) {
#pragma HLS PIPELINE II=1 style=flp
			if (IN == OUT) {
				if (in.read_nb(x)) out.write(x);
			} else if (IN > OUT) {
				if (k == 0 && in.read_nb(x)) {
					e = x;
					k = ratio;
				}
				if (k != 0) {
					out.write(e(OUT - 1, 0));
					e >>= OUT;
					k--;
				}
			} else if (in.read_nb(x)) {
				a = a >> IN;
				a(OUT - 1, OUT - IN) = x;
				if (k == ratio - 1) {
					out.write(a);
					k = 0;
				} else {
					k++;
				}
			}
		}
	} else if (IN == OUT) {
		for (unsigned t = 0; FOREVER || t < numIn; t++) {
#pragma HLS PIPELINE II=1 style=flp
			out.write(in.read());
		}
	} else if (IN > OUT) {
		// one input word yields IN/OUT output words, least significant first
		constexpr unsigned ratio = IN / OUT;
		ap_uint<IN> e = 0;
		unsigned o = 0;
		for (unsigned t = 0; FOREVER || t < numIn * ratio; t++) {
#pragma HLS PIPELINE II=1 style=flp
			if (o == 0) e = in.read();
			out.write(e(OUT - 1, 0));
			e >>= OUT;
			o = (o == ratio - 1) ? 0 : o + 1;
		}
	} else {
		// OUT/IN input words make one output word, the first one in the least significant bits
		constexpr unsigned ratio = OUT / IN;
		ap_uint<OUT> e = 0;
		unsigned i = 0;
		for (unsigned t = 0; FOREVER || t < numIn; t++) {
#pragma HLS PIPELINE II=1 style=flp
			e = e >> IN;
			e(OUT - 1, OUT - IN) = in.read();
			if (i == ratio - 1) {
				out.write(e);
				i = 0;
			} else {
				i++;
			}
		}
	}
}

// Distribute frames round-robin over N output streams: frame g goes to out[g % N].
template<unsigned W, unsigned N, bool FOREVER>
void split(hls::stream<ap_uint<W>> &in, hls::stream<ap_uint<W>> out[N], unsigned wordsPerFrame,
           unsigned numFrames) {
	unsigned w = 0, p = 0;
	for (unsigned t = 0; FOREVER || t < numFrames * wordsPerFrame; t++) {
#pragma HLS PIPELINE II=1 style=flp
		ap_uint<W> x;
		if (FOREVER) {
			if (!in.read_nb(x)) continue;
		} else {
			x = in.read();
		}
		out[p].write(x);
		if (w == wordsPerFrame - 1) {
			w = 0;
			p = (p == N - 1) ? 0 : p + 1;
		} else {
			w++;
		}
	}
}

// Collect frames round-robin from N input streams: frame g comes from in[g % N].
template<unsigned W, unsigned N, bool FOREVER>
void gather(hls::stream<ap_uint<W>> in[N], hls::stream<ap_uint<W>> &out, unsigned wordsPerFrame,
            unsigned numFrames) {
	unsigned w = 0, p = 0;
	for (unsigned t = 0; FOREVER || t < numFrames * wordsPerFrame; t++) {
#pragma HLS PIPELINE II=1 style=flp
		ap_uint<W> x;
		if (FOREVER) {
			if (!in[p].read_nb(x)) continue;
		} else {
			x = in[p].read();
		}
		out.write(x);
		if (w == wordsPerFrame - 1) {
			w = 0;
			p = (p == N - 1) ? 0 : p + 1;
		} else {
			w++;
		}
	}
}

// Ring DMA, memory side of the input DMA. Descriptor d (slot d % R) asks for frames[slot] frames at
// byte address (addr_hi[slot] << 32 | addr_lo[slot]). The host posts descriptors by incrementing
// prod; cons counts the descriptors whose data has been read from memory, so their buffers may be
// reused. maxDesc > 0 stops after that many descriptors (C simulation only).
template<unsigned IW, unsigned R>
void ring_read(hls::burst_maxi<ap_uint<IW>> mem, hls::stream<ap_uint<IW>> &out, const unsigned addr_lo[R],
               const unsigned addr_hi[R], const unsigned frames[R], volatile unsigned &prod,
               volatile unsigned &cons, unsigned wordsPerFrame, unsigned maxDesc) {
	unsigned done = 0;
	while (maxDesc == 0 || done != maxDesc) {
		if (prod != done) {
			const unsigned slot = done % R;
			const ap_uint<64> addr = (ap_uint<64>(addr_hi[slot]) << 32) | addr_lo[slot];
			const unsigned words = frames[slot] * wordsPerFrame;
			if (words != 0) {
				mem.read_request(addr / (IW / 8), words);
				for (unsigned i = 0; i < words; i++) {
#pragma HLS PIPELINE II=1 style=flp
					out.write(mem.read());
				}
			}
			done++;
			{
#pragma HLS PROTOCOL mode=fixed
				ap_wait();
				cons = done;
			}
		}
	}
}

// Ring DMA, memory side of the output DMA. done counts the descriptors whose data has been written
// and acknowledged by the memory (write_response), so the host may read it.
template<unsigned IW, unsigned R>
void ring_write(hls::stream<ap_uint<IW>> &in, hls::burst_maxi<ap_uint<IW>> mem, const unsigned addr_lo[R],
                const unsigned addr_hi[R], const unsigned frames[R], volatile unsigned &prod,
                volatile unsigned &cons, unsigned wordsPerFrame, unsigned maxDesc) {
	unsigned done = 0;
	while (maxDesc == 0 || done != maxDesc) {
		if (prod != done) {
			const unsigned slot = done % R;
			const ap_uint<64> addr = (ap_uint<64>(addr_hi[slot]) << 32) | addr_lo[slot];
			const unsigned words = frames[slot] * wordsPerFrame;
			if (words != 0) {
				mem.write_request(addr / (IW / 8), words);
				for (unsigned i = 0; i < words; i++) {
#pragma HLS PIPELINE II=1 style=flp
					mem.write(in.read());
				}
			}
			done++;
			{
#pragma HLS PROTOCOL mode=fixed
				if (words != 0) {
					mem.write_response();
				}
				ap_wait();
				cons = done;
			}
		}
	}
}

} // namespace iodma

#endif
