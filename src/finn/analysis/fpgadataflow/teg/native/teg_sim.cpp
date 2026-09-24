/******************************************************************************
 * Copyright (C) 2026, Paderborn University
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of FINN nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

// Native core of the timed-event-graph simulator: a line-by-line transcription of
// ``simulate.py::_Simulator.run`` (same heap order, same fixed-point handling, same
// stable-state criterion), operating on the flat CSR arrays built by ``native.py``.
// Compiled on demand with the system C++ compiler and called through ctypes.

#include <cstdint>
#include <deque>
#include <queue>
#include <vector>

namespace {

typedef int64_t i64;

const i64 NEG_INF = -(i64(1) << 60);
const int STABLE_INTERVALS = 3;
const int STABLE_OCC_FRAMES = 2;

// ---------------------------------------------------------------- stall patterns
enum PatKind { PAT_NONE = 0, PAT_BERNOULLI = 1, PAT_BURSTY = 2, PAT_SINGLE = 3, PAT_PERIODIC = 4 };

struct Pattern {
    i64 kind;
    double p;
    uint64_t seed;
    i64 on, off, phase, at, length, every;
};

inline i64 pmod(i64 a, i64 m) {
    i64 r = a % m;
    return r < 0 ? r + m : r;
}

// SplitMix64 finaliser over (seed, t), identical to patterns.py::_mix64
inline uint64_t mix64(uint64_t seed, i64 t) {
    uint64_t z = seed * 0x9E3779B97F4A7C15ULL + (uint64_t)(t + 1) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline bool pat_active(const Pattern& p, i64 t) {
    switch (p.kind) {
        case PAT_NONE:
            return true;
        case PAT_BERNOULLI: {
            double u = (double)(mix64(p.seed, t) >> 11) * (1.0 / 9007199254740992.0);
            return u >= p.p;
        }
        case PAT_BURSTY:
            if (p.off == 0) return true;
            return pmod(t - p.phase, p.on + p.off) < p.on;
        case PAT_SINGLE:
            return !(p.at <= t && t < p.at + p.length);
        default:
            return pmod(t - p.phase, p.every) == 0;
    }
}

inline i64 pat_next_true(const Pattern& p, i64 t) {
    switch (p.kind) {
        case PAT_NONE:
            return t;
        case PAT_BURSTY: {
            if (p.off == 0) return t;
            i64 period = p.on + p.off;
            i64 r = pmod(t - p.phase, period);
            return r < p.on ? t : t + (period - r);
        }
        case PAT_SINGLE:
            return (p.at <= t && t < p.at + p.length) ? p.at + p.length : t;
        case PAT_PERIODIC: {
            i64 r = pmod(t - p.phase, p.every);
            return r == 0 ? t : t + (p.every - r);
        }
        default:
            while (!pat_active(p, t)) t += 1;
            return t;
    }
}

// ---------------------------------------------------------------- heap
struct Item {
    i64 t;
    i64 prio;
    i64 c;
};
struct ItemGreater {
    bool operator()(const Item& a, const Item& b) const {
        if (a.t != b.t) return a.t > b.t;
        if (a.prio != b.prio) return a.prio > b.prio;
        return a.c > b.c;
    }
};

struct Model {
    // chains
    i64 nc;
    const i64* L;
    const i64* ev_off;   // global event index of event 0 of each chain (size nc + 1)
    const i64* gap;      // per global event
    const i64* rd_off;   // per global event (size nev + 1)
    const i64* rd_val;
    const i64* wr_off;
    const i64* wr_val;
    const i64* arc_off;  // per global event (size nev + 1), 4 values per arc
    const i64* arc_val;
    const i64* kind;
    const i64* pat;      // pattern index or -1
    const i64* period;
    const i64* keep_hist;
    const i64* hist_window;  // -1: unlimited
    const i64* group;
    const i64* freezer;
    const i64* until_empty;
    const i64* start;
    i64 ngroups;
    // edges
    i64 ne;
    const i64* lf;
    const i64* lb;
    const i64* depth;  // -1: unbounded
    const i64* init_tokens;
    const i64* sd_off;  // src_direct edges per chain (size nc + 1)
    const i64* sd_val;
    const i64* gate;    // pattern index gating writes into a direct sink edge, or -1
    // patterns
    std::vector<Pattern> patterns;
};

// frame times of the last run, fetched by teg_sim_fetch (the sources may run many frames
// ahead of the sinks, so their number is not bounded by max_frames)
std::vector<i64> g_frame_end, g_frame_start;

}  // namespace

extern "C" {

// Copy the frame end/start times of the last run into caller-provided arrays.
void teg_sim_fetch(i64* frame_end_out, i64* frame_start_out) {
    for (size_t j = 0; j < g_frame_end.size(); ++j) frame_end_out[j] = g_frame_end[j];
    for (size_t j = 0; j < g_frame_start.size(); ++j) frame_start_out[j] = g_frame_start[j];
}

// Returns 0 on success, -1 when a chain history was truncated below a referenced event.
// out_scalars: frames_done, cycles, stable, deadlock, timeout, n_frame_end, n_frame_start,
//              error chain (referencing), error chain (referenced), error event. The frame
//              times themselves are fetched with teg_sim_fetch.
i64 teg_sim_run(
    i64 nc, const i64* L, const i64* ev_off, const i64* gap, const i64* rd_off, const i64* rd_val,
    const i64* wr_off, const i64* wr_val, const i64* arc_off, const i64* arc_val, const i64* kind,
    const i64* pat, const i64* period, const i64* keep_hist, const i64* hist_window,
    const i64* group, const i64* freezer, const i64* until_empty, const i64* start, i64 ngroups,
    i64 ne, const i64* lf, const i64* lb, const i64* depth, const i64* init_tokens,
    const i64* sd_off, const i64* sd_val, const i64* gate,
    i64 npat, const i64* pat_kind, const double* pat_p, const i64* pat_i,
    i64 max_frames, i64 min_frames, i64 max_cycles, i64 stop_when_stable, i64 stable_occupancy,
    i64* maxocc, i64* first_valid, i64* out_scalars) {
    Model m;
    m.nc = nc; m.L = L; m.ev_off = ev_off; m.gap = gap; m.rd_off = rd_off; m.rd_val = rd_val;
    m.wr_off = wr_off; m.wr_val = wr_val; m.arc_off = arc_off; m.arc_val = arc_val;
    m.kind = kind; m.pat = pat; m.period = period; m.keep_hist = keep_hist;
    m.hist_window = hist_window; m.group = group; m.freezer = freezer;
    m.until_empty = until_empty; m.start = start; m.ngroups = ngroups;
    m.ne = ne; m.lf = lf; m.lb = lb; m.depth = depth; m.init_tokens = init_tokens;
    m.sd_off = sd_off; m.sd_val = sd_val; m.gate = gate;
    m.patterns.resize(npat);
    for (i64 p = 0; p < npat; ++p) {
        Pattern& q = m.patterns[p];
        q.kind = pat_kind[p];
        q.p = pat_p[p];
        q.seed = (uint64_t)pat_i[p * 7 + 0];
        q.on = pat_i[p * 7 + 1];
        q.off = pat_i[p * 7 + 2];
        q.phase = pat_i[p * 7 + 3];
        q.at = pat_i[p * 7 + 4];
        q.length = pat_i[p * 7 + 5];
        q.every = pat_i[p * 7 + 6];
    }

    std::vector<i64> freeze_count(ngroups > 0 ? ngroups : 1, 0);
    std::vector<char> freezing(nc, 0);
    std::vector<std::vector<i64>> group_waiters(ngroups > 0 ? ngroups : 1);
    std::vector<i64> idx(nc, 0), frame(nc, 0), last_t(nc, 0), cnt(nc, 0), hist_base(nc, 0);
    for (i64 c = 0; c < nc; ++c) {
        last_t[c] = L[c] > 0 ? start[c] - gap[ev_off[c]] : 0;
    }
    std::vector<std::vector<i64>> hist(nc), waiters(nc);
    std::vector<std::deque<i64>> wq(ne);
    std::vector<std::vector<i64>> rr(ne);
    std::vector<i64> wc(ne), rc(ne, 0), dw(ne, -1), sw(ne, -1);
    std::vector<char> in_dirty(ne, 0);
    std::vector<i64> dirty;
    for (i64 e = 0; e < ne; ++e) {
        for (i64 k = 0; k < init_tokens[e]; ++k) wq[e].push_back(NEG_INF);
        if (depth[e] >= 0) rr[e].assign(depth[e], NEG_INF);
        wc[e] = init_tokens[e];
        maxocc[e] = init_tokens[e];
        first_valid[e] = -1;
    }
    std::vector<i64> sinks, sources;
    for (i64 c = 0; c < nc; ++c) {
        if (kind[c] == 1) sources.push_back(c);
        if (kind[c] == 2) sinks.push_back(c);
    }
    i64 frames_done = 0;
    std::vector<i64> frame_end, frame_start;
    std::vector<std::vector<i64>> src_starts(nc);
    i64 last_occ_change = 0;
    bool stable = false, timeout = false, done = false;
    std::vector<i64> prio(nc);
    for (i64 c = 0; c < nc; ++c) prio[c] = freezer[c] ? 0 : 1;
    std::priority_queue<Item, std::vector<Item>, ItemGreater> heap;
    for (i64 c = 0; c < nc; ++c) heap.push(Item{start[c], prio[c], c});
    i64 limit = max_cycles >= 0 ? max_cycles : (i64(1) << 62);
    i64 t = 0, cur_t = 0;

    while (!heap.empty()) {
        Item it = heap.top();
        heap.pop();
        t = it.t;
        i64 c = it.c;
        if (t != cur_t) {
            for (i64 e : dirty) {
                in_dirty[e] = 0;
                i64 occ = wc[e] - rc[e];
                if (occ > maxocc[e]) {
                    maxocc[e] = occ;
                    last_occ_change = frames_done;
                }
            }
            dirty.clear();
            cur_t = t;
        }
        if (t > limit) {
            timeout = true;
            break;
        }
        i64 i = idx[c];
        i64 k = kind[c];
        i64 g = ev_off[c] + i;
        i64 tmin = last_t[c] + gap[g];
        if (k == 1) {
            if (i == 0 && period[c]) {
                i64 tp = frame[c] * period[c];
                if (tp > tmin) tmin = tp;
            }
            bool blocked = false;
            for (i64 s = sd_off[c]; s < sd_off[c + 1]; ++s) {
                i64 e = sd_val[s];
                i64 kk = wc[e];
                if (kk - rc[e] >= 1) {
                    sw[e] = c;
                    blocked = true;
                    break;
                }
                if (kk >= 1) {
                    i64 ts = rr[e][0] + lb[e];
                    if (ts > tmin) tmin = ts;
                }
            }
            if (blocked) continue;
            if (pat[c] >= 0) tmin = pat_next_true(m.patterns[pat[c]], tmin);
        }
        if (tmin > t) {
            heap.push(Item{tmin, prio[c], c});
            continue;
        }
        i64 grp = group[c];
        if (grp >= 0 && freeze_count[grp] > 0 && !freezer[c]) {
            group_waiters[grp].push_back(c);
            continue;
        }
        i64 tneed = t;
        bool blocked = false;
        for (i64 s = rd_off[g]; s < rd_off[g + 1]; ++s) {
            i64 e = rd_val[s];
            if (wc[e] <= rc[e]) {
                dw[e] = c;
                blocked = true;
                break;
            }
            i64 tr = wq[e].front() + lf[e];
            if (tr > tneed) tneed = tr;
        }
        if (blocked) continue;
        bool data_ready = tneed == t;
        for (i64 s = wr_off[g]; s < wr_off[g + 1]; ++s) {
            i64 e = wr_val[s];
            i64 d = depth[e];
            if (d >= 0) {
                i64 kk = wc[e];
                if (kk - rc[e] >= d) {
                    sw[e] = c;
                    blocked = true;
                    break;
                }
                if (kk >= d) {
                    i64 ts = rr[e][pmod(kk - d, d)] + lb[e];
                    if (ts > tneed) tneed = ts;
                }
            }
        }
        if (blocked) {
            if (data_ready && freezer[c] && !freezing[c]) {
                freezing[c] = 1;
                freeze_count[grp] += 1;
            }
            continue;
        }
        for (i64 s = arc_off[g]; s < arc_off[g + 1]; s += 4) {
            i64 src = arc_val[s], si = arc_val[s + 1], q = arc_val[s + 2], dly = arc_val[s + 3];
            i64 gi = (frame[c] - q) * L[src] + si;
            if (gi < 0) continue;
            if (gi >= cnt[src]) {
                waiters[src].push_back(c);
                blocked = true;
                break;
            }
            i64 hb = gi - hist_base[src];
            if (hb < 0) {
                out_scalars[7] = c;
                out_scalars[8] = src;
                out_scalars[9] = gi;
                return -1;
            }
            i64 ta = hist[src][hb] + dly;
            if (ta > tneed) tneed = ta;
        }
        if (blocked) continue;
        if (k == 2 && pat[c] >= 0) tneed = pat_next_true(m.patterns[pat[c]], tneed);
        for (i64 s = wr_off[g]; s < wr_off[g + 1]; ++s) {
            i64 e = wr_val[s];
            if (gate[e] >= 0) tneed = pat_next_true(m.patterns[gate[e]], tneed);
        }
        if (tneed > t) {
            if (data_ready && freezer[c] && !freezing[c]) {
                freezing[c] = 1;
                freeze_count[grp] += 1;
            }
            heap.push(Item{tneed, prio[c], c});
            continue;
        }
        if (freezing[c]) {
            bool release = true;
            if (until_empty[c]) {
                for (i64 s = rd_off[g]; s < rd_off[g + 1]; ++s) {
                    i64 e = rd_val[s];
                    if (wc[e] - rc[e] > 2) {
                        release = false;
                        break;
                    }
                }
            }
            if (release) {
                freezing[c] = 0;
                freeze_count[grp] -= 1;
                if (freeze_count[grp] == 0) {
                    std::vector<i64>& gw = group_waiters[grp];
                    for (i64 w : gw) heap.push(Item{t, prio[w], w});
                    gw.clear();
                }
            }
        }
        // ---- fire at t
        for (i64 s = rd_off[g]; s < rd_off[g + 1]; ++s) {
            i64 e = rd_val[s];
            wq[e].pop_front();
            i64 r = rc[e];
            rc[e] = r + 1;
            i64 d = depth[e];
            if (d >= 0) rr[e][pmod(r, d)] = t;
            i64 w = sw[e];
            if (w >= 0) {
                sw[e] = -1;
                heap.push(Item{t + lb[e], prio[w], w});
            }
        }
        for (i64 s = wr_off[g]; s < wr_off[g + 1]; ++s) {
            i64 e = wr_val[s];
            wq[e].push_back(t);
            wc[e] += 1;
            if (!in_dirty[e]) {
                in_dirty[e] = 1;
                dirty.push_back(e);
            }
            if (first_valid[e] < 0) first_valid[e] = t;
            i64 w = dw[e];
            if (w >= 0) {
                dw[e] = -1;
                heap.push(Item{t + lf[e], prio[w], w});
            }
        }
        last_t[c] = t;
        cnt[c] += 1;
        if (keep_hist[c]) {
            std::vector<i64>& h = hist[c];
            h.push_back(t);
            i64 win = hist_window[c];
            if (win >= 0 && (i64)h.size() > 2 * win + 64) {
                i64 drop = (i64)h.size() - win;
                h.erase(h.begin(), h.begin() + drop);
                hist_base[c] += drop;
            }
        }
        std::vector<i64>& ws = waiters[c];
        if (!ws.empty()) {
            for (i64 w : ws) heap.push(Item{t, prio[w], w});
            ws.clear();
        }
        if (k == 1 && i == 0) {
            src_starts[c].push_back(t);
            i64 fn = (i64)frame_start.size();
            bool all_started = true;
            for (i64 s : sources) {
                if ((i64)src_starts[s].size() <= fn) {
                    all_started = false;
                    break;
                }
            }
            if (all_started) {
                i64 mn = src_starts[sources[0]][fn];
                for (i64 s : sources) {
                    if (src_starts[s][fn] < mn) mn = src_starts[s][fn];
                }
                frame_start.push_back(mn);
            }
        }
        i += 1;
        if (i == L[c]) {
            i = 0;
            frame[c] += 1;
            if (k == 2) {
                i64 fd = frame[sinks[0]];
                for (i64 s : sinks) {
                    if (frame[s] < fd) fd = frame[s];
                }
                while (frames_done < fd) {
                    frames_done += 1;
                    frame_end.push_back(t);
                    i64 n = (i64)frame_end.size();
                    if (stop_when_stable && n >= min_frames && n > STABLE_INTERVALS &&
                        (!stable_occupancy || frames_done - last_occ_change >= STABLE_OCC_FRAMES)) {
                        i64 iv = frame_end[n - 1] - frame_end[n - 2];
                        bool same = true;
                        for (int j = 2; j <= STABLE_INTERVALS; ++j) {
                            if (frame_end[n - j] - frame_end[n - j - 1] != iv) {
                                same = false;
                                break;
                            }
                        }
                        if (same) {
                            stable = true;
                            done = true;
                        }
                    }
                    if (n >= max_frames) done = true;
                }
                if (done) break;
            }
        }
        idx[c] = i;
        heap.push(Item{t + gap[ev_off[c] + i], prio[c], c});
    }
    for (i64 e : dirty) {
        i64 occ = wc[e] - rc[e];
        if (occ > maxocc[e]) maxocc[e] = occ;
    }
    bool deadlock = !done && !timeout;
    out_scalars[0] = frames_done;
    out_scalars[1] = t;
    out_scalars[2] = stable ? 1 : 0;
    out_scalars[3] = deadlock ? 1 : 0;
    out_scalars[4] = timeout ? 1 : 0;
    out_scalars[5] = (i64)frame_end.size();
    out_scalars[6] = (i64)frame_start.size();
    g_frame_end.swap(frame_end);
    g_frame_start.swap(frame_start);
    return 0;
}

}  // extern "C"
