#include <AXIS_Control.h>
#include <AXI_Control.h>
#include <Clock.h>
#include <Design.h>
#include <Kernel.h>
#include <Port.h>
#include <SharedLibrary.h>

#include <algorithm>
#include <bitset>
#include <chrono>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "rtlsim_config.hpp"

// simulation_config.h contains info (Start size, MPI, etc,)
#include "simulation_config.h"
#ifdef MPI_FOUND
    #include <mpi.h>
#endif

#define MPI_ROOT_RANK 0

void clearPorts(xsi::Design& top) {
    // Clear all input ports
    for (xsi::Port& p : top.ports()) {
        if (p.isInput()) {
            p.clear().write_back();
        }
    }
}

void reset(xsi::Design& top) {
    xsi::Port& rst_n = top.getPort("ap_rst_n");
    Clock& clk = Clock::initClock(top);
    // Reset all Inputs, Wait for Reset Period
    rst_n.set(0).write_back();
    for (unsigned i = 0; i < 16; i++) {
        clk.toggle_clk();
    }
    rst_n.set(1).write_back();
}


// Helper function to process input streams with deferred writes
void processInputStream(S_AXIS_Control& stream, size_t& iters, std::vector<std::reference_wrapper<xsi::Port>>& deferredWrites, std::deque<size_t>& inflightTimestamps) {
    const bool isValid = stream.is_valid();

    // Skip if valid but not ready
    if (isValid && !stream.is_ready()) {
        return;
    }

    // Track successful transactions
    if (isValid) {
        stream.job_txns++;
    }

    const bool shouldProcessJob = (stream.job_txns < stream.job_size) || (iters >= stream.await_iter);

    // Handle throttling
    if (shouldProcessJob) {
        if (!isValid) {
            deferredWrites.emplace_back(stream.set_valid(true));
        }

        // Reset job when completed
        if (stream.job_txns == stream.job_size) {
            stream.job_txns = 0;
            stream.await_iter = iters + stream.job_ticks;
            inflightTimestamps.push_front(iters);
        }
    } else if (isValid) {
        deferredWrites.emplace_back(stream.set_valid(false));
    }
}

// Helper function to process output streams
bool processOutputStream(M_AXIS_Control& stream, size_t& iters, std::deque<size_t>& inflightTimestamps, size_t& completedMaps) {
    if (!stream.is_ready() || !stream.is_valid()) {
        return false;
    }

    const size_t totalTransactions = ++stream.total_txns;

    // Track first completion
    if (totalTransactions == stream.job_size) {
        stream.first_complete = iters;
    }

    // Track job completion and intervals
    if (++stream.job_txns == stream.job_size) {
        stream.interval = iters - stream.last_complete;
        stream.last_complete = iters;
        stream.job_txns = 0;
        stream.latency = iters - inflightTimestamps.back();
        inflightTimestamps.pop_back();  // Remove the oldest timestamp
        stream.min_latency = std::min(stream.min_latency, stream.latency);
        ++completedMaps;
    }

    return true;
}

bool runForFeaturemaps(size_t featuremaps, Clock& clk, std::vector<S_AXIS_Control>& istreams, std::vector<M_AXIS_Control>& ostreams, size_t& iters) {
    // Enter Simulation Loop and track Progress
    auto const begin = std::chrono::steady_clock::now();
    std::vector<std::reference_wrapper<xsi::Port>> to_write;
    size_t completedMaps = 0;
    size_t timeout = 0;
    std::deque<size_t> inflightTimestamps;
    inflightTimestamps.push_front(0);
    while (true) {
        //-------------------------------------------------------------------
        // Clock down - then read signal updates from design
        clk.cycle(0);

        //-------------------------------------------------------------------
        // Process input streams

        for (auto& stream : istreams) {
            processInputStream(stream, iters, to_write, inflightTimestamps);
        }

        //-------------------------------------------------------------------
        // Process output streams

        bool hasActiveOutput = false;
        for (auto& stream : ostreams) {
            if (processOutputStream(stream, iters, inflightTimestamps, completedMaps)) {
                hasActiveOutput = true;
            }
        }
        timeout = hasActiveOutput ? 0 : timeout + 1;

        //-------------------------------------------------------------------
        // Clock up - then write signal updates back to design
        clk.cycle(1);

        // Write back Ports with registered updates
        for (xsi::Port& p : to_write)
            p.write_back();
        to_write.clear();

        ++iters;

        if (iters % 25000 == 0) {
            std::cout << "Iteration: " << iters << ", Inflight Timestamps Size: " << inflightTimestamps.size() << ", in "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() << "ms" << std::endl;
        }

        if (completedMaps == featuremaps) {
            //std::cout << "Completing " << featuremaps << " maps took " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() << "ms" << std::endl;
            break;
        }
        // Check for timeout
        if (timeout > max_iters*10) {
            return false;
        }
    }
    return true;
}

size_t computeMovingAverage(const size_t oldAvg, const size_t newValue, const size_t count) { return (oldAvg * (count - 1) + newValue) / count; }

std::optional<size_t> runToStableState(Clock& clk, std::vector<S_AXIS_Control>& istreams, std::vector<M_AXIS_Control>& ostreams, size_t& iters) {
    size_t avgInterval = 0;
    size_t runs = 0;
    size_t totalMaps = 0;

    size_t minimumMaps = 5;

    //Warmup. Discard results, because empty pipeline
    bool warmup = true;

    while (true) {
        if (!runForFeaturemaps(1, clk, istreams, ostreams, iters)) {
            return std::nullopt;
        }
        //Do one map as a warmup without calculating the avg to fill pipeline
        if (warmup){
            warmup = false;
            continue;
        }

        size_t newAvgInterval = computeMovingAverage(avgInterval, ostreams[0].interval, ++runs);

        std::cout << "Finished map " << totalMaps << ". avgInterval (new): "<< newAvgInterval << ". avgInterval (old): " << avgInterval << std::endl;

        bool notChanged = avgInterval == newAvgInterval;

        avgInterval = newAvgInterval;

        if (++totalMaps >= minimumMaps && notChanged) {
            break;
        }
    }
    // Avg Interval here is equivalent to the output stream's
    // metrics, but easier to retrieve
    return avgInterval;
}

template<typename T>
std::string toBinaryString(T data) {
    return std::bitset<sizeof(T) * 8>(data).to_string();
}

struct simDesc {
    xsi::Kernel kernel;
    xsi::Design top;
    std::vector<S_AXIS_Control> istreams;
    std::vector<M_AXIS_Control> ostreams;
    std::vector<std::reference_wrapper<xsi::Port>> fifo_ports;
    Clock& clk;

    simDesc(const std::string& kernel_lib, const std::string& design_lib, const char* xsim_log_file, const char* trace_file) : kernel(kernel_lib), top(kernel, design_lib, xsim_log_file, trace_file), clk(Clock::initClock(top)) {}
};

simDesc initDesign(const std::string& kernel_lib, const std::string& design_lib, const char* xsim_log_file, const char* trace_file) {
    simDesc desc(kernel_lib, design_lib, xsim_log_file, trace_file);

    if (trace_file) {
        // TODO make tracing more finer-grain if possible?
        desc.top.trace_all();
    }

    // Find I/O Streams and initialize their Status
    for (auto&& elem : istream_descs) {
        desc.istreams.emplace_back(desc.top, elem.job_size, elem.job_ticks, elem.name);
    }
    for (auto&& elem : ostream_descs) {
        desc.ostreams.emplace_back(desc.top, elem.job_size, elem.name);
    }

    // Find Global Control & Run Startup Sequence
    clearPorts(desc.top);
    reset(desc.top);

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

std::tuple<size_t, size_t> determineStartDepth(xsi::Design& top, Clock& clk, std::vector<S_AXIS_Control>& istreams, std::vector<M_AXIS_Control>& ostreams, size_t& iters, std::vector<std::reference_wrapper<xsi::Port>>& fifo_ports) {
    int start_depth = 128;
    int last_start_depth = start_depth;
    uint32_t last_interval = 0;

    while (true) {
        std::cout << "Starting testing start_depth: " << start_depth << std::endl;

        reset(top);
        auto two_bin = toBinaryString(static_cast<uint32_t>(start_depth));
        for (auto&& p : fifo_ports) {
            p.get().set_binstr(two_bin).write_back();
        }
        clk.toggle_clk();

        if (auto ret = runToStableState(clk, istreams, ostreams, iters); ret) {
            auto&& interval = *ret;
            if (interval > 0 && interval == last_interval) {
                start_depth = last_start_depth;
                break;
            }
            std::cout << "Testing start_depth: " << start_depth << ", interval: " << interval << std::endl;
            last_interval = static_cast<uint32_t>(interval);
        }

        last_start_depth = start_depth;
        start_depth *= 2;  // Double the start depth

        if (start_depth > 1000000) {
            throw std::runtime_error("Couldn't find a working start depth, please set manually!");
        }
    }

    return std::make_tuple(start_depth, last_interval);
}

std::vector<size_t> sizeIteratively(size_t start_size, size_t interval, size_t startIndex, size_t endIndex, const std::string& kernel_lib, const std::string& design_lib, const char* xsim_log_file, const char* trace_file) {
    auto totalFifos = endIndex - startIndex + 1;
    std::vector<size_t> fifo_sizes(totalFifos, start_size);

    for (size_t i = 0; i < totalFifos; ++i) {

        auto&& [kernel, top, istreams, ostreams, fifo_ports, clk] = initDesign(kernel_lib, design_lib, xsim_log_file, trace_file);

        while (true) {  // do until fifo minimized
            // Try to minimize this FIFO
            size_t oldFifoSize = fifo_sizes[i];
            fifo_sizes[i] = std::max(oldFifoSize / 2, size_t(1));

            std::cout << "Sizing Fifo " << i << "\n";

            reset(top);
            size_t iters;

            // Set the new fifo depth in the actual component
            fifo_ports[startIndex + i].get().set_binstr(toBinaryString(fifo_sizes[i]));

            if (auto ret = runToStableState(clk, istreams, ostreams, iters); ret) {
                auto&& currInterval = *ret;
                std::cout << currInterval << "\n";
                std::cout << interval << "\n";
                if (currInterval == 0 || currInterval > interval) {
                    // Performance drop
                    // Revert depth reduction and mark FIFO as minimized
                    fifo_sizes[i] = oldFifoSize;
                    break;
                }
            } else {
                // Timeout
                // Revert depth reduction and mark FIFO as minimized
                fifo_sizes[i] = oldFifoSize;
                break;
            }

            if (fifo_sizes[i] == 1) {
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
        std::cout << "Running on " << world_size << " ranks." << std::endl;
    }
#endif
    auto&& [kernel, top, istreams, ostreams, fifo_ports, clk] = initDesign(kernel_libname, design_libname, xsim_log_filename, trace_filename);

    if (rank == 0) {
        for (auto&& port : top.ports()) {
            std::cout << "Port Name: " << port.name() << ", Direction: " << (port.dir()) << std::endl;
        }
        std::cout << "Starting data feed with idle-output timeout of " << max_iters << " cycles ...\n" << std::endl;

        if (istreams.size() > 1 || ostreams.size() > 1) {
            throw std::runtime_error(
                "This simulation is not designed to run with "
                "multiple input or output streams.");
        }
    }

    // Simulation Report Statistics
    size_t iters = 0;

    size_t start_depth = 0;
    size_t interval = 0;
#ifndef FIFO_START_SIZE
    if (rank == 0) {
        std::cout << "\nDetermining start depth..." << std::endl;
        auto [_start_depth, _interval] = determineStartDepth(top, clk, istreams, ostreams, iters, fifo_ports);
        start_depth = _start_depth;
        std::cout << "Interval: " << _interval << "\n";
        interval = _interval;
    }
#else
    start_depth = FIFO_START_SIZE;
    interval = ???;
#endif

    // Which fifos this process must size
    size_t start_index = 0;
    size_t end_index = fifo_ports.size() - 1;

    // Split work across ranks
    size_t fifos_per_rank = fifo_ports.size() / static_cast<size_t>(world_size) + 1;
    start_index = static_cast<size_t>(rank) * fifos_per_rank;
    end_index = std::clamp((static_cast<size_t>(rank) + 1) * fifos_per_rank - 1, 0lu, fifo_ports.size() - 1);

    size_t elementsInRank = end_index - start_index + 1;
    std::cout << "Rank " << rank << " sizing " << elementsInRank << " FIFOs." << std::endl;

#ifdef MPI_FOUND
    // Synchronize and send depth to all other ranks
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&start_depth, 1, MPI_UNSIGNED_LONG, MPI_ROOT_RANK, MPI_COMM_WORLD);
#endif

    if (rank == 0) {
        std::cout << "\nSizing iteratively..." << std::endl;
    }

    // Size assigned FIFOs iteratively
    auto fifoSizes = sizeIteratively(start_depth, interval, start_index, end_index, kernel_libname, design_libname, xsim_log_filename, trace_filename);

    // Collect all FIFO sizes together into allSizes
    std::vector<size_t> allSizes(fifo_ports.size());

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
