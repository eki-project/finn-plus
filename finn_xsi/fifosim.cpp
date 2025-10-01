#include <AXIS_Control.h>
#include <AXI_Control.h>
#include <Clock.h>
#include <Design.h>
#include <Kernel.h>
#include <Port.h>
#include <SharedLibrary.h>
#include <Port.h>

#include <algorithm>
#include <bitset>
#include <chrono>
#include <cstddef>
#include <deque>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// simulation_config.h contains info (Start size, MPI, etc,)
#include "simulation_config.hpp"
#ifdef MPI_FOUND
    #include <mpi.h>
    constexpr int MPI_ROOT_RANK = 0;
#endif


static void clearPorts(xsi::Design& top) noexcept;
static void reset(simDesc& desc) noexcept;

struct simDesc {
    xsi::Kernel kernel;
    xsi::Design top;
    std::vector<S_AXIS_Control> istreams;
    std::vector<M_AXIS_Control> ostreams;
    std::vector<std::reference_wrapper<xsi::Port>> fifo_ports;
    Clock clk;

    simDesc(const std::string& kernel_lib, const std::string& design_lib, const char* xsim_log_file, const char* trace_file) : kernel(kernel_lib), top(kernel, design_lib, xsim_log_file, trace_file), clk(top) {}
};

simDesc initDesign(const std::string& kernel_lib, const std::string& design_lib, const char* const xsim_log_file, const char* const trace_file) {
    simDesc desc(kernel_lib, design_lib, xsim_log_file, trace_file);

    if (trace_file) {
        // TODO make tracing more finer-grain if possible?
        desc.top.trace_all();
    }

    // Find I/O Streams and initialize their Status
    for (auto&& elem : istream_descs) {
        desc.istreams.emplace_back(desc.top, desc.clk, elem.job_size, elem.job_ticks, elem.name);
    }
    for (auto&& elem : ostream_descs) {
        desc.ostreams.emplace_back(desc.top, desc.clk, elem.job_size, elem.name);
    }

    // Find Global Control & Run Startup Sequence
    clearPorts(desc.top);
    reset(desc);

    // Make all Inputs valid & all Outputs ready
    for (auto&& s : desc.istreams) {
        s.valid();
    }
    for (auto&& s : desc.ostreams) {
        s.ready();
    }

    for (xsi::Port& p : desc.top.ports()) {
        if (std::string(p.name()).find("depth") != std::string::npos) {
            desc.fifo_ports.emplace_back(std::ref(p));
        }
    }

    return desc;
}

static void clearPorts(xsi::Design& top) noexcept {
    // Clear all input ports
    for (xsi::Port& p : top.ports()) {
        if (p.isInput()) {
            p.clear().write_back();
        }
    }
}

static void reset(simDesc& desc) noexcept {
    xsi::Port& rst_n = desc.top.getPort("ap_rst_n");
    // Reset all Inputs, Wait for Reset Period
    rst_n.set(0).write_back();
    for (unsigned i = 0; i < 16; i++) {
        desc.clk.toggle_clk();
    }
    rst_n.set(1).write_back();
}

// Helper function to process output streams
[[gnu::hot]] [[gnu::flatten]]
static bool processOutputStream(M_AXIS_Control& stream, size_t& iters, size_t& completedMaps) noexcept {
    if (!stream.is_ready() || !stream.is_valid()) [[unlikely]] {
        return false;
    }

    // Track first completion
    if (++stream.total_txns == stream.job_size) [[unlikely]] {
        stream.first_complete = iters;
    }

    // Track job completion and intervals
    if (++stream.job_txns == stream.job_size) [[unlikely]] {
        stream.interval = iters - stream.last_complete;
        stream.last_complete = iters;
        stream.job_txns = 0;
        ++completedMaps;
    }

    return true;
}

[[gnu::hot]] [[gnu::flatten]]
bool runForFeaturemaps(const size_t featuremaps, Clock& clk, std::vector<S_AXIS_Control>& istreams, std::vector<M_AXIS_Control>& ostreams, size_t& iters) {
    // Enter Simulation Loop and track Progress
    const auto begin = std::chrono::high_resolution_clock::now();
    size_t completedMaps = 0;
    size_t timeout = 0;
    constexpr size_t max_timeout_limit = max_iters * 100;

    // Pre-cache frequently accessed values
    const size_t num_ostreams = ostreams.size();
    const size_t num_istreams = istreams.size();

    //-------------------------------------------------------------------
    // Make sure all inputs are valid. We do not care about throttling
    for (size_t i = 0; i < num_istreams; ++i) {
        istreams[i].valid(true);
    }

    while (true) [[likely]] {
        //-------------------------------------------------------------------
        // Clock down - then read signal updates from design
        clk.cycle(0);

        //-------------------------------------------------------------------
        // Process output streams

        bool hasActiveOutput = false;
        for (size_t i = 0; i < num_ostreams; ++i) {
            if (processOutputStream(ostreams[i], iters, completedMaps)) [[likely]] {
                hasActiveOutput = true;
                // std::cout << "Cycle: " << iters << std::endl;
            }
        }
        timeout = hasActiveOutput ? 0 : timeout + 1;

        //-------------------------------------------------------------------
        // Clock up - then write signal updates back to design
        clk.cycle(1);

        ++iters;

        if (iters % 25000 == 0) [[unlikely]] {
            std::cout << "Iteration: " << iters << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - begin).count() << "ms" << std::endl;
        }

        if (completedMaps == featuremaps) [[unlikely]] {
            return true;
        }
        // Check for timeout
        if (timeout > max_timeout_limit) [[unlikely]] {
            return false;
        }
    }
    return true;
}

constexpr size_t computeMovingAverage(const size_t oldAvg, const size_t newValue, const size_t count) noexcept { return (oldAvg * (count - 1) + newValue) / count; }

[[gnu::hot]]
std::optional<size_t> runToStableState(Clock& clk, std::vector<S_AXIS_Control>& istreams, std::vector<M_AXIS_Control>& ostreams, size_t& iters) {
    size_t avgInterval = 0;
    size_t runs = 0;
    size_t totalMaps = 0;

    constexpr size_t minimumMaps = 3;

    // Warmup. Discard results, because empty pipeline
    bool warmup = true;

    while (true) [[likely]] {
        if (!runForFeaturemaps(1, clk, istreams, ostreams, iters)) [[unlikely]] {
            return std::nullopt;
        }
        // Do one map as a warmup without calculating the avg to fill pipeline
        if (warmup) [[unlikely]] {
            warmup = false;
            continue;
        }

        const size_t newAvgInterval = computeMovingAverage(avgInterval, ostreams[0].interval, ++runs);
        const bool notChanged = avgInterval == newAvgInterval;
        avgInterval = newAvgInterval;

        if (++totalMaps >= minimumMaps && notChanged) [[unlikely]] {
            break;
        }
    }
    // Avg Interval here is equivalent to the output stream's
    // metrics, but easier to retrieve
    return avgInterval;
}

template<typename T>
[[gnu::always_inline]] inline
std::string toBinaryString(const T data) noexcept {
    return std::bitset<sizeof(T) * 8>(data).to_string();
}


std::tuple<size_t, size_t> determineStartDepth(const std::string& kernel_lib, const std::string& design_lib, const char* const xsim_log_file, const char* const trace_file) {
    int start_depth = 1;
    int last_start_depth = start_depth;
    uint32_t last_interval = 0;

    while (true) {
        std::cout << "Starting testing start_depth: " << start_depth << std::endl;

        auto&& [kernel, top, istreams, ostreams, fifo_ports, clk] = initDesign(kernel_lib, design_lib, xsim_log_file, trace_file);

        const auto two_bin = toBinaryString(static_cast<uint32_t>(start_depth));
        for (auto&& p : fifo_ports) {
            p.get().set_binstr(two_bin).write_back();
        }
        clk.toggle_clk();
        for (auto&& s : ostreams) {
            s.job_txns = 0;
            s.total_txns = 0;
            s.first_complete = 0;
            s.last_complete = 0;
            s.interval = 0;
        }

        size_t iters = 0;
        if (auto ret = runToStableState(clk, istreams, ostreams, iters); ret) {
            auto&& interval = *ret;
            if (interval > 0 && interval == last_interval) {
                start_depth = last_start_depth;
                break;
            }
            // std::cout << "Testing start_depth: " << start_depth << ", interval: " << interval << std::endl;
            last_interval = static_cast<uint32_t>(interval);
        }

        last_start_depth = start_depth;
        start_depth <<= 1;  // Double the start depth

        if (start_depth >= (2 << 20)) {
            throw std::runtime_error("Couldn't find a working start depth, please set manually!");
        }
    }

    return std::make_tuple(start_depth, last_interval);
}

[[gnu::hot]]
std::vector<size_t> sizeIteratively(const size_t start_size, const size_t interval, const size_t startIndex, const size_t endIndex, const std::string& kernel_lib, const std::string& design_lib, const char* const xsim_log_file,
                                    const char* const trace_file) {
    const auto totalFifos = endIndex - startIndex + 1;
    std::vector<size_t> fifo_sizes(totalFifos, start_size);

    for (size_t i = 0; i < totalFifos; ++i) {
        auto&& [kernel, top, istreams, ostreams, fifo_ports, clk] = initDesign(kernel_lib, design_lib, xsim_log_file, trace_file);
        std::cout << "Sizing FIFO number " << i << std::endl;
        while (true) [[likely]] {  // do until fifo minimized
            // Try to minimize this FIFO
            const size_t oldFifoSize = fifo_sizes[i];
            fifo_sizes[i] = std::max(oldFifoSize >> 1, size_t(1));  // Use bit shift for division by 2

            const auto two_bin = toBinaryString(static_cast<uint32_t>(start_size));
            for (auto&& p : fifo_ports) {
                p.get().set_binstr(two_bin).write_back();
            }
            // Set the new fifo depth in the actual component
            fifo_ports[startIndex + i].get().set_binstr(toBinaryString(fifo_sizes[i])).write_back();
            clk.toggle_clk();
            for (auto&& s : ostreams) {
                s.job_txns = 0;
                s.total_txns = 0;
                s.first_complete = 0;
                s.last_complete = 0;
                s.interval = 0;
            }

            size_t iters = 0;
            if (auto ret = runToStableState(clk, istreams, ostreams, iters); ret) [[likely]] {
                auto&& currInterval = *ret;
                if (currInterval == 0 || currInterval > interval) [[unlikely]] {
                    // Performance drop
                    // Revert depth reduction and mark FIFO as minimized
                    fifo_sizes[i] = oldFifoSize;
                    break;
                }
            } else [[unlikely]] {
                // Timeout
                // Revert depth reduction and mark FIFO as minimized
                fifo_sizes[i] = oldFifoSize;
                break;
            }

            if (fifo_sizes[i] == 1) [[unlikely]] {
                // Minimum size reached
                break;
            }
        }
    }
    return fifo_sizes;
}

int main() {
    int world_size = 1;
    int rank = 0;
#ifdef MPI_FOUND
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "MPI Available!" << std::endl;
        std::cout << "Running on " << world_size << " ranks." << std::endl;
    }
#endif

    size_t start_depth = 1;
    size_t interval = 0;
#ifndef FIFO_START_SIZE
    if (rank == 0) {
        std::cout << "\nDetermining start depth..." << std::endl;
        auto [_start_depth, _interval] = determineStartDepth(kernel_libname, design_libname, xsim_log_filename, trace_filename);
        start_depth = _start_depth;
        std::cout << "Interval: " << _interval << "\n";
        interval = _interval;
    }
#else
    if (rank == 0) {
        std::cout << "Skipping FIFO startup sizing. Set to: " << FIFO_START_SIZE << std::endl;
    }
    start_depth = FIFO_START_SIZE;

    // Since we dont know the interval simply assume the worst and let iterations improve
    interval = std::numeric_limits<decltype(interval)>::max();
#endif
    // Create a new design just to count the FIFOs...
    simDesc design = initDesign(kernel_libname, design_libname, xsim_log_filename, trace_filename);
    size_t fifo_count = design.fifo_ports.size();

    // Check that we dont have too many ranks (results in errors later on)
    // TODO: Temporary solution. In the future this should not cause an error
    if (rank == 0) {
        if (static_cast<size_t>(world_size) > fifo_count) {
            std::cout << "Too many ranks! There are " << world_size << " ranks for " << fifo_count << " FIFOs!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Determine FIFOs to be calculated per rank
    // Which fifos this process must size
    size_t start_index = 0;
    size_t end_index = fifo_count - 1;

    // Split work across ranks
    size_t fifos_per_rank = static_cast<size_t>(static_cast<float>(fifo_count) / static_cast<float>(world_size)) + 1;
    start_index = static_cast<size_t>(rank) * fifos_per_rank;
    end_index = std::clamp((static_cast<size_t>(rank) + 1) * fifos_per_rank - 1, 0lu, fifo_count - 1);

    size_t elementsInRank = end_index - start_index + 1;
    std::cout << "Rank " << rank << " sizing " << elementsInRank << " FIFOs. (end_index: " << end_index << ", start_index: " << start_index << ")"  << std::endl;

#ifdef MPI_FOUND
    // Synchronize and send depth to all other ranks
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&start_depth, 1, MPI_UNSIGNED_LONG, MPI_ROOT_RANK, MPI_COMM_WORLD);
#endif

    if (rank == 0) {
        std::cout << "\nSizing iteratively..." << std::endl;
    }

    // Size assigned FIFOs iteratively
    const auto fifoSizes = sizeIteratively(start_depth, interval, 0, 2, kernel_libname, design_libname, xsim_log_filename, trace_filename);

    // Collect all FIFO sizes together into allSizes
    std::vector<size_t> allSizes(3);

// Gather all FIFO sizes from all ranks
#ifdef MPI_FOUND
    // Gather how many FIFOs each rank worked on (might vary, at minimum the last rank might not be filled completely)
    std::vector<int> elementsToGather(static_cast<size_t>(world_size));
    MPI_Gather(&elementsInRank, 1, MPI_UNSIGNED_LONG, &elementsToGather[0], 1, MPI_UNSIGNED_LONG, MPI_ROOT_RANK, MPI_COMM_WORLD);

    // Gather the FIFO sizes themselves
    std::vector<int> displs(static_cast<size_t>(world_size));
    std::exclusive_scan(std::begin(elementsToGather), std::end(elementsToGather), std::begin(displs), 0);
    MPI_Gatherv(&fifoSizes[0], static_cast<int>(fifoSizes.size()), MPI_UNSIGNED_LONG, &allSizes[0], &elementsToGather[0], &displs[0], MPI_UNSIGNED_LONG, MPI_ROOT_RANK, MPI_COMM_WORLD);
#else
    allSizes = fifoSizes;
#endif

    if (rank == 0) {
        for (auto elem : allSizes) {
            std::cout << "FIFO size: " << elem << std::endl;
        }
    }

    return 0;
}
