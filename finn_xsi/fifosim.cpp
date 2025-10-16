#include <AXIS_Control.h>
#include <AXI_Control.h>
#include <Clock.h>
#include <Design.h>
#include <Kernel.h>
#include <Port.h>
#include <SharedLibrary.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

// simulation_config.h contains info (Start size, MPI, etc,)
#include "simulation_config.hpp"
#ifdef MPI_FOUND
    #include <mpi.h>
constexpr int MPI_ROOT_RANK = 0;
#endif


struct simDesc {
    xsi::Kernel kernel;
    xsi::Design top;
    std::array<S_AXIS_Control, istream_descs.size()> istreams;
    std::array<M_AXIS_Control, istream_descs.size()> ostreams;
    std::vector<std::reference_wrapper<xsi::Port>> fifo_depths;
    std::vector<std::reference_wrapper<xsi::Port>> fifo_occupancies;
    std::vector<std::reference_wrapper<xsi::Port>> fifo_max_occupancies;
    Clock clk;

    simDesc(const std::string& kernel_lib, const std::string& design_lib, const char* xsim_log_file, const char* trace_file) : kernel(kernel_lib), top(kernel, design_lib, xsim_log_file, trace_file), clk(top) {}
};

static void clearPorts(xsi::Design& top) noexcept;
static void reset(simDesc& desc) noexcept;

simDesc initDesign(const std::string& kernel_lib, const std::string& design_lib, const char* const xsim_log_file, const char* const trace_file) {
    simDesc desc(kernel_lib, design_lib, xsim_log_file, trace_file);

    if (trace_file) {
        // TODO make tracing more finer-grain if possible?
        desc.top.trace_all();
    }

    // Find I/O Streams and initialize their Status
    for (size_t i = 0; i < istream_descs.size(); ++i) {
        desc.istreams[i] = S_AXIS_Control{desc.top, desc.clk, std::data(istream_descs)[i].job_size, std::data(istream_descs)[i].job_ticks, std::data(istream_descs)[i].name};
    }
    for (size_t i = 0; i < ostream_descs.size(); ++i) {
        desc.ostreams[i] = M_AXIS_Control{desc.top, desc.clk, std::data(ostream_descs)[i].job_size, std::data(ostream_descs)[i].name};
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
            desc.fifo_depths.emplace_back(std::ref(p));
        }
    }

    for (xsi::Port& p : desc.top.ports()) {
        if (std::string(p.name()).find("count") != std::string::npos && std::string(p.name()).find("maxcount") == std::string::npos) {
            desc.fifo_occupancies.emplace_back(std::ref(p));
        }
    }

    for (xsi::Port& p : desc.top.ports()) {
        if (std::string(p.name()).find("maxcount") != std::string::npos) {
            desc.fifo_max_occupancies.emplace_back(std::ref(p));
        }
    }

    if (desc.fifo_depths.empty() || desc.fifo_occupancies.empty() || desc.fifo_max_occupancies.empty()) {
        std::cout << "DEBUG: FIFO Depth Ports Found: " << desc.fifo_depths.size() << std::endl;
        std::cout << "DEBUG: FIFO Occupancy Ports Found: " << desc.fifo_occupancies.size() << std::endl;
        std::cout << "DEBUG: FIFO Max Occupancy Ports Found: " << desc.fifo_max_occupancies.size() << std::endl;
        throw std::runtime_error("FIFO detection failed! Make sure the design contains FIFOs.");
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

bool isDeadlocked(const std::vector<size_t> fifo_occupancies) {
    static std::vector<size_t> last_occupancies(fifo_occupancies.size(), 0);
    static bool init = true;
    static size_t deadlockCounter = 0;
    if (init) {
        last_occupancies = fifo_occupancies;
        init = false;
        return false;
    }

    bool unchanged = std::equal(fifo_occupancies.begin(), fifo_occupancies.end(), last_occupancies.begin());
    bool zeros = std::all_of(fifo_occupancies.begin(), fifo_occupancies.end(), [](int i) { return i==0; });

    if (unchanged && !zeros) {
        std::cout << "FIFO Occupancies: ";
        for (const auto& occ : fifo_occupancies) {
            std::cout << occ << " ";
        }
        std::cout << "\n";
        std::cout << "Last FIFO Occupancies: ";
        for (const auto& occ : last_occupancies) {
            std::cout << occ << " ";
        }
        std::cout << "\n";
    }

    last_occupancies = fifo_occupancies;
    if (unchanged && !zeros) {
        if (++deadlockCounter >= 3) {
            deadlockCounter = 0;
            return true;
        }
    } else {
        deadlockCounter = 0;
    }
    return false;
}


template<size_t istreams_size = istream_descs.size(), size_t ostreams_size = ostream_descs.size()>
[[gnu::hot]] [[gnu::flatten]]
bool runForFeaturemaps(const size_t featuremaps, Clock& clk, std::array<S_AXIS_Control, istreams_size>& istreams, std::array<M_AXIS_Control, ostreams_size>& ostreams, std::vector<std::reference_wrapper<xsi::Port>>& fifo_occupancies,
                       size_t& iters, const size_t max_inputs = std::numeric_limits<size_t>::max(), const size_t iterations_to_stable_state = std::numeric_limits<size_t>::max()) {
    size_t completedMaps = 0;
    bool inputs_valid = false;
    //-------------------------------------------------------------------
    // Make sure all inputs are valid. We do not care about throttling
    if (iters < max_inputs) {
        inputs_valid = true;
        for (auto&& stream : istreams) {
            stream.valid(true);
        }
    }

    if (iters % 2 == 1) {  // Prepare for unrolling
        //-------------------------------------------------------------------
        // Clock down - then read signal updates from design
        clk.cycle(0);

        //-------------------------------------------------------------------
        // Process output streams

        for (auto&& stream : ostreams) {
            if (stream.is_valid() && ++stream.job_txns == stream.job_size) {
                // Track job completion and intervals
                stream.interval = iters - stream.last_complete;
                stream.last_complete = iters;
                stream.job_txns = 0;
                if (++completedMaps == featuremaps) [[unlikely]] {
                    clk.cycle(1);
                    return true;
                }
            }
        }


        //-------------------------------------------------------------------
        // Clock up - then write signal updates back to design
        clk.cycle(1);

        ++iters;
    }

    // Enter Simulation Loop and track Progress
    const auto begin = std::chrono::high_resolution_clock::now();
    while (true) [[likely]] {
        //-------------------------------------------------------------------
        // Clock down - then read signal updates from design
        clk.cycle(0);

        //-------------------------------------------------------------------
        // Process output streams

        for (auto&& stream : ostreams) {
            if (stream.is_valid() && ++stream.job_txns == stream.job_size) {
                // Track job completion and intervals
                stream.interval = iters - stream.last_complete;
                stream.last_complete = iters;
                stream.job_txns = 0;
                std::cout << "Featuremap completed\n";
                if (++completedMaps == featuremaps) [[unlikely]] {
                    clk.cycle(1);
                    return true;
                }
            }
        }


        //-------------------------------------------------------------------
        // Clock up - then write signal updates back to design
        clk.cycle(1);
        //-------------------------------------------------------------------
        // Clock down - then read signal updates from design
        clk.cycle(0);

        //-------------------------------------------------------------------
        // Process output streams
        ++iters;
        for (auto&& stream : ostreams) {
            if (stream.is_valid() && ++stream.job_txns == stream.job_size) {
                // Track job completion and intervals
                stream.interval = iters - stream.last_complete;
                stream.last_complete = iters;
                stream.job_txns = 0;
                std::cout << "Featuremap completed\n";
                if (++completedMaps == featuremaps) [[unlikely]] {
                    clk.cycle(1);
                    return true;
                }
            }
        }


        //-------------------------------------------------------------------
        // Clock up - then write signal updates back to design
        clk.cycle(1);

        if ((++iters & (0x7FFF)) == 0) [[unlikely]] {  // mod 2^15 //This can only be true in even cycles therefore only check this here
            std::cout << "Iteration: " << iters << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - begin).count() << "ms" << std::endl;
            // Deadlock detection
            /* std::vector<size_t> occupancies;
            occupancies.reserve(fifo_occupancies.size());
            for (const auto& p : fifo_occupancies) {
                occupancies.emplace_back(p.get().read().as_unsigned());
            }
            if (isDeadlocked(occupancies)) {
                std::cout << "Deadlock detected after " << iters << " iterations!" << std::endl;
                return false;
            } */
        }
        // Disable new inputs after max_inputs iterations to speed up simulation
        if (inputs_valid && iters >= max_inputs) [[unlikely]] {
            std::cout << "Disabling new inputs after " << iters << " iterations." << std::endl;
            inputs_valid = false;
            for (auto&& stream : istreams) {
                stream.valid(false);
            }
        }

        if (iters >= iterations_to_stable_state) [[unlikely]] {
            std::cout << "Reached maximum number of iterations to stable state (" << iterations_to_stable_state << ")." << std::endl;
            return false;
        }
    }
}

constexpr size_t computeMovingAverage(const size_t oldAvg, const size_t newValue, const size_t count) noexcept { return (oldAvg * (count - 1) + newValue) / count; }

consteval size_t maxIStreamJobSize() {
    size_t maxSize = 0;
    for (const auto& desc : istream_descs) {
        if (desc.job_size > maxSize) {
            maxSize = desc.job_size;
        }
    }
    return maxSize;
}

[[gnu::hot]]
std::optional<std::tuple<size_t, size_t>> runToStableState(Clock& clk, std::array<S_AXIS_Control, istream_descs.size()>& istreams, std::array<M_AXIS_Control, istream_descs.size()>& ostreams, std::vector<std::reference_wrapper<xsi::Port>>& fifo_occupancies,
                                       size_t& iters, const size_t max_inputs = std::numeric_limits<size_t>::max(), const size_t iterations_to_stable_state = std::numeric_limits<size_t>::max()) {
    size_t avgInterval = 0;
    size_t runs = 0;
    size_t totalMaps = 0;

    constexpr size_t minimumMaps = 3;

    // Warmup. Discard results, because empty pipeline
    bool warmup = true;

    while (true) [[likely]] {
        if (!runForFeaturemaps(1, clk, istreams, ostreams, fifo_occupancies, iters, max_inputs, iterations_to_stable_state)) [[unlikely]] {
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
    return std::make_tuple(avgInterval, (totalMaps+10) * maxIStreamJobSize());
}

std::vector<unsigned> getMaxFIFOUtilization(const std::vector<std::reference_wrapper<xsi::Port>>& fifo_max_occupancies) {
    std::vector<unsigned> maxUtil;
    maxUtil.reserve(fifo_max_occupancies.size());
    for (const auto& p : fifo_max_occupancies) {
        maxUtil.emplace_back(p.get().read().as_unsigned());
    }
    return maxUtil;
}

std::tuple<std::vector<unsigned>, size_t, size_t, size_t> determineStartDepth(const std::string& kernel_lib, const std::string& design_lib, const char* const xsim_log_file, const char* const trace_file, int start_depth = std::numeric_limits<int>::max()) {
    auto&& [kernel, top, istreams, ostreams, fifo_ports, fifo_occupancies, fifo_max_occupancies, clk] = initDesign(kernel_lib, design_lib, xsim_log_file, trace_file);

    for (auto&& p : fifo_ports) {
        p.get().set(static_cast<unsigned>(start_depth)).write_back();
    }
    clk.toggle_clk();

    size_t iters = 0;
    if (auto ret = runToStableState(clk, istreams, ostreams, fifo_occupancies, iters); ret) {
        auto&& [interval, maxInputs] = *ret;
        if (interval > 0) {
            return std::make_tuple(getMaxFIFOUtilization(fifo_max_occupancies), static_cast<uint32_t>(interval), maxInputs, static_cast<size_t>(iters*1.1));
        }
    }
    throw std::runtime_error("Couldn't find a working start depth, please set manually!");
}

[[gnu::hot]]
std::vector<size_t> sizeIteratively(const std::vector<unsigned>& start_depths, const size_t interval, const size_t max_inputs, const size_t iterations_to_stable_state, const size_t startIndex, const size_t endIndex, const std::string& kernel_lib, const std::string& design_lib,
                                    const char* const xsim_log_file, const char* const trace_file, int rank) {
    const auto totalFifos = endIndex - startIndex + 1;
    std::vector<size_t> fifoSizes(totalFifos);

    // Initialize fifoSizes with the corresponding start depths
    for (size_t i = 0; i < totalFifos; ++i) {
        fifoSizes[i] = static_cast<size_t>(start_depths[startIndex + i]);
    }

    for (size_t i = 0; i < totalFifos; ++i) {
        if (fifoSizes[i] <= 1) {
            // Skip sizing for this FIFO, as it is already at minimum size
            continue;
        }

        std::cout << "[Rank " << rank << "] Sizing FIFO number " << i << " (from " << startIndex << " to " << endIndex << ")" << std::endl;

        // Keep track of the FIFO size of the previous iteration
        size_t previousFifoSize = fifoSizes[i];

        while (true) [[likely]] {  // do until fifo minimized
            // TODO: Bin search for more accurate sizes
            //fifoSizes[i] = std::max(previousFifoSize >> 1, size_t(1));  // Use bit shift for division by 2

            auto&& [kernel, top, istreams, ostreams, fifoPorts, fifo_occupancies, fifo_max_occupancies, clk] = initDesign(kernel_lib, design_lib, xsim_log_file, trace_file);

            // Set the start depths for all fifos in the model
            for (size_t j = 0; j < fifoPorts.size(); ++j) {
                fifoPorts[j].get().set(start_depths[j]+1).write_back();
            }

            // Try out the new FIFO depth
            fifoPorts[startIndex + i].get().set(static_cast<unsigned>(fifoSizes[i]+1)).write_back();

            std::cout << "FIFO sizes: ";
            for (auto&& p : fifoPorts) {
                std::cout << p.get().read().as_unsigned() << " ";
            }
            std::cout << std::endl;

            clk.toggle_clk();
            for (auto&& s : ostreams) {
                s.job_txns = 0;
                s.total_txns = 0;
                s.first_complete = 0;
                s.last_complete = 0;
                s.interval = 0;
            }

            size_t iters = 0;
            if (auto ret = runToStableState(clk, istreams, ostreams, fifo_occupancies, iters, max_inputs, iterations_to_stable_state); ret) [[likely]] {
                auto&& [currInterval, _maxInputs] = *ret;
                if (currInterval == 0 || currInterval > interval) [[unlikely]] {
                    // Performance drop
                    // Revert depth reduction and mark FIFO as minimized
                    fifoSizes[i] = previousFifoSize;
                    break;
                } else {
                    // currInterval <= interval: Smaller fifo depth but same performance. Thus this depth is a safe value
                    previousFifoSize = fifoSizes[i];
                }
            } else [[unlikely]] {
                // Timeout
                // Revert depth reduction and mark FIFO as minimized
                fifoSizes[i] = previousFifoSize;
                break;
            }

            if (fifoSizes[i] == 1) [[unlikely]] {
                // Minimum size reached
                break;
            }
        }
    }
    return fifoSizes;
}

int main() {
    int world_size = 1;
    int rank = 0;

#ifdef MPI_FOUND
    // Initialize MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "MPI Available!" << std::endl;
        std::cout << "Running on " << world_size << " ranks." << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // Initialize depth and interval
    std::optional<int> initial_start_depth = std::nullopt;
    std::vector<unsigned> start_depth;
    size_t interval = 0;
    size_t max_maps = std::numeric_limits<size_t>::max();
    size_t iterations_to_stable_state = std::numeric_limits<size_t>::max();

#ifdef FIFO_START_SIZE
    // If a starting size was given, use it here
    initial_start_depth = std::make_optional(FIFO_START_SIZE);
#endif

    // Run until a suitable starting depth was found
    if (rank == 0) {
        std::cout << "\n--------------------------------------------\n";
        std::cout << "\nDetermining start depth (single rank)..." << std::endl;
        if (initial_start_depth) {
            std::cout << "Setting initial start depth to: " << initial_start_depth.value() << std::endl;
            auto startDepthBegin = std::chrono::high_resolution_clock::now();
            auto [_start_depth, _interval, _max_maps, _iterations_to_stable_state] = determineStartDepth(kernel_libname, design_libname, xsim_log_filename, trace_filename, initial_start_depth.value());
            auto startDepthDuration = std::chrono::high_resolution_clock::now() - startDepthBegin;
            start_depth = _start_depth;
            std::cout << "Interval: " << _interval << "\n";
            interval = _interval;
            max_maps = _max_maps;
            iterations_to_stable_state = _iterations_to_stable_state;
            std::cout << "Finished start depth sizing in " << std::chrono::duration_cast<std::chrono::minutes>(startDepthDuration).count() << "m (" << std::chrono::duration_cast<std::chrono::hours>(startDepthDuration).count() << "h)"
                      << std::endl;
        } else {
            auto startDepthBegin = std::chrono::high_resolution_clock::now();
            auto [_start_depth, _interval, _max_iters, _iterations_to_stable_state] = determineStartDepth(kernel_libname, design_libname, xsim_log_filename, trace_filename);
            auto startDepthDuration = std::chrono::high_resolution_clock::now() - startDepthBegin;
            start_depth = _start_depth;
            std::cout << "Interval: " << _interval << "\n";
            interval = _interval;
            max_maps = _max_iters;
            iterations_to_stable_state = _iterations_to_stable_state;
            std::cout << "Finished start depth sizing in " << std::chrono::duration_cast<std::chrono::minutes>(startDepthDuration).count() << "m (" << std::chrono::duration_cast<std::chrono::hours>(startDepthDuration).count() << "h)"
                      << std::endl;
        }
    }

#ifdef MPI_FOUND
    // Synchronize and send depth to all other ranks
    MPI_Barrier(MPI_COMM_WORLD);

    // First broadcast the size of the start_depth vector
    size_t start_depth_size = start_depth.size();
    MPI_Bcast(&start_depth_size, 1, MPI_UNSIGNED_LONG, MPI_ROOT_RANK, MPI_COMM_WORLD);

    // Resize vector on non-root ranks
    if (rank != MPI_ROOT_RANK) {
        start_depth.resize(start_depth_size);
    }

    // Broadcast the vector data
    MPI_Bcast(start_depth.data(), static_cast<int>(start_depth_size), MPI_UNSIGNED, MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&interval, 1, MPI_UNSIGNED_LONG, MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&max_maps, 1, MPI_UNSIGNED_LONG, MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&iterations_to_stable_state, 1, MPI_UNSIGNED_LONG, MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
#endif  // MPI_FOUND

    // Create a new design just to count the FIFOs and make sure its destroyed directly afterwards...
    if (rank == 0) {
        std::cout << "Using start depth vector of size " << start_depth.size() << " and target interval of " << interval << std::endl;
        std::cout << "Start Depths: ";
        for (auto&& elem : start_depth) {
            std::cout << elem << " ";
        }
        std::cout << "\n";
    }
    size_t fifo_count = 0;
    {
        auto&& [kernel, top, istreams, ostreams, fifo_ports, fifo_occupancies, fifo_max_occupancies, clk] = initDesign(kernel_libname, design_libname, xsim_log_filename, trace_filename);
        fifo_count = fifo_ports.size();
    }

#ifdef MPI_FOUND
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // Determine FIFOs to be calculated per rank
    // Which fifos this process must size
    size_t start_index = 0;
    size_t end_index = 0;
    size_t elementsInRank = 0;

    // Split work across ranks
    if (static_cast<size_t>(rank) < fifo_count) {
        size_t base_fifos_per_rank = fifo_count / static_cast<size_t>(world_size);
        size_t remaining_fifos = fifo_count % static_cast<size_t>(world_size);

        // When world_size > fifo_count, base_fifos_per_rank = 0
        // Only the first fifo_count ranks get exactly 1 FIFO each
        if (base_fifos_per_rank == 0) {
            // Each of the first fifo_count ranks gets exactly 1 FIFO
            start_index = static_cast<size_t>(rank);
            end_index = static_cast<size_t>(rank);
            elementsInRank = 1;
        } else {
            // Normal case: distribute FIFOs across ranks
            // Some ranks get one extra FIFO if there's a remainder
            size_t fifos_for_this_rank = base_fifos_per_rank + (static_cast<size_t>(rank) < remaining_fifos ? 1 : 0);

            // Calculate start index
            start_index = static_cast<size_t>(rank) * base_fifos_per_rank + std::min(static_cast<size_t>(rank), remaining_fifos);

            // Calculate end index and elements count
            end_index = start_index + fifos_for_this_rank - 1;
            elementsInRank = fifos_for_this_rank;
        }
    } else {
        // This rank has no work (more ranks than FIFOs)
        start_index = 0;
        end_index = 0;
        elementsInRank = 0;
    }

#ifdef MPI_FOUND
    // Synchronize - start_depth vector already broadcast above
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (rank == 0) {
        std::cout << "\n--------------------------------------------\n";
        std::cout << "\nSizing iteratively..." << std::endl;
    }
    std::cout << "Rank " << rank << " sizing " << elementsInRank << " FIFOs. (end_index: " << end_index << ", start_index: " << start_index << ")" << std::endl;

    // Size assigned FIFOs iteratively
    std::vector<size_t> fifoSizes;
    if (elementsInRank > 0) {
        fifoSizes = sizeIteratively(start_depth, interval, max_maps, iterations_to_stable_state, start_index, end_index, kernel_libname, design_libname, xsim_log_filename, trace_filename, rank);
    } else {
        // This rank has no work, create empty vector
        fifoSizes.clear();
    }

    // Collect all FIFO sizes together into allSizes
    std::vector<size_t> allSizes(fifo_count, 0);

// Gather all FIFO sizes from all ranks
#ifdef MPI_FOUND
    // First, gather the number of elements each rank has
    std::vector<int> recvcounts(static_cast<size_t>(world_size));
    std::vector<int> displs(static_cast<size_t>(world_size));

    int my_count = static_cast<int>(elementsInRank);
    MPI_Allgather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate displacements
    displs[0] = 0;
    for (size_t i = 1; i < static_cast<size_t>(world_size); ++i) {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    // Gather all FIFO sizes using MPI_Gatherv
    MPI_Gatherv(fifoSizes.data(), my_count, MPI_UNSIGNED_LONG, allSizes.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED_LONG, MPI_ROOT_RANK, MPI_COMM_WORLD);
#else
    allSizes = fifoSizes;
#endif

    if (rank == 0) {
        for (auto elem : allSizes) {
            std::cout << "FIFO size: " << elem << std::endl;
        }
    }

#ifdef MPI_FOUND
    MPI_Finalize();
#endif  // MPI_FOUND
    return 0;
}
