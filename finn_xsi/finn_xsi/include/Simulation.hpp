#ifndef SIMULATION
#define SIMULATION
#include <AXIS_Control.h>
#include <Clock.h>
#include <Design.h>
#include <FIFO.h>
#include <Kernel.h>
#include <Port.h>
#include <SharedLibrary.h>
#include <helper.h>

//#include <InterSimulationInterface.hpp>
#include <InterprocessCommunicationChannel.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>


template<size_t IStreamsSize, size_t OStreamsSize, bool LoggingEnabled>
class Simulation {
     protected:
    std::ofstream readyLog;
    std::ofstream validLog;

     public:
    xsi::Kernel kernel;
    xsi::Design top;
    // S_AXIS_Control goes into the simulated layer
    std::array<S_AXIS_Control, IStreamsSize> istreams;
    // M_AXIS_Control comes from the simulated layer
    std::array<M_AXIS_Control, OStreamsSize> ostreams;
    Clock clk;


    Simulation(const std::string& kernel_lib, const std::string& design_lib, const char* xsim_log_file, const char* trace_file,
               std::array<StreamDescriptor, IStreamsSize> _istream_descs, std::array<StreamDescriptor, OStreamsSize> _ostream_descs)
        : kernel(kernel_lib), top(kernel, design_lib, xsim_log_file, trace_file), clk(top) {
        if (trace_file) {
            top.trace_all();
        }

        // Find I/O Streams and initialize their Status
        for (size_t i = 0; i < _istream_descs.size(); ++i) {
            istreams[i] = S_AXIS_Control{top, clk, std::data(_istream_descs)[i].job_size, std::data(_istream_descs)[i].job_ticks, std::data(_istream_descs)[i].name};
        }
        for (size_t i = 0; i < _ostream_descs.size(); ++i) {
            ostreams[i] = M_AXIS_Control{top, clk, std::data(_ostream_descs)[i].job_size, std::data(_ostream_descs)[i].name};
        }

        // Save simulation input output behaviour
        if constexpr (LoggingEnabled) {
            readyLog.open("ready_log.txt");
            validLog.open("valid_log.txt");
        }

        // Find Global Control & Run Startup Sequence
        clearPorts();
        reset();
    }

    template<std::size_t Index>
    bool hasValidOutput() {
        // static_assert(Index < ostreams.size(), "Cannot request valid status of unknown output stream index");
        return ostreams[Index].is_valid();
    }

    void clearPorts() noexcept {
        // Clear all input ports
        for (xsi::Port& p : top.ports()) {
            if (p.isInput()) {
                p.clear().write_back();
            }
        }
    }

    void reset() noexcept {
        xsi::Port& rst_n = top.getPort("ap_rst_n");
        // Reset all Inputs, Wait for Reset Period
        rst_n.set(0).write_back();
        for (unsigned i = 0; i < 16; i++) {
            clk.toggleClk();
        }
        rst_n.set(1).write_back();
    }
};

//Small struct used for exange. Will be changed later to more complex data structure.
struct CommData{
    bool data;
};

// Communication Flow:
//
//           valid      ┌──────────────────────────────────────┐     valid            valid
//   SHM   ─────────>   │         valid            valid       │    ─────────>  FIFO  ─────>   SHM
//  (pred) <───────── istream  ─────────>  xsim  ─────────> ostream <─────────        <─────  (succ)
//           ready      │      <─────────        <─────────    │     ready            ready
//                      │         ready            ready       │
//                      │                  (sim)               │
//                      └──────────────────────────────────────┘
template<size_t IStreamsSize, size_t OStreamsSize, bool LoggingEnabled, size_t NodeIndex, size_t TotalNodes>
class SingleNodeSimulation : public Simulation<IStreamsSize, OStreamsSize, LoggingEnabled> {
    using ConsumingInterface = InterprocessCommunicationChannel<CommData, CommData, false>;
    using ProducingInterface = InterprocessCommunicationChannel<CommData, CommData, true>;
    constexpr static bool FirstNode = NodeIndex == 0;
    constexpr static bool LastNode = NodeIndex == (TotalNodes - 1);
    std::array<ConsumingInterface, IStreamsSize> fromProducerInterface;
    std::array<ProducingInterface, OStreamsSize> toConsumerInterface;
    std::size_t cyclesRun = 0;
    std::size_t completedMaps = 0;
    std::array<FIFO, OStreamsSize> fifo;

    /// Communicate with predecessors and successors and update their values and our own
    [[gnu::hot, gnu::flatten, gnu::always_inline]] void communicate(std::stop_token stoken = {}) {
        if constexpr (!FirstNode) {
            for (std::size_t i = 0; i < IStreamsSize; ++i) {
                // Interface SHM <-> sim
                this->istreams[i].valid(fromProducerInterface[i].receive_request(stoken).data);
                fromProducerInterface[i].send_response(CommData{this->istreams[i].isReady()});
            }
        }
        if constexpr (!LastNode) {
            for (std::size_t i = 0; i < OStreamsSize; ++i) {
                // Interface sim -valid-> FIFO <-> SHM
                this->fifo[i].update(this->ostreams[i].isValid(), toConsumerInterface[i].send_request(CommData{this->fifo[i].isOutputValid()}, stoken).data);
                // FIFO -ready-> sim
                this->ostreams[i].ready(this->fifo[i].isInputReady());
                // Toggle FIFO clock
                this->fifo[i].toggleClock();
            }
        }
        if constexpr (LastNode) {
            for (auto&& stream : this->ostreams) {
                if (stream.isValid() && ++stream.job_txns == stream.job_size) {
                    static std::vector<std::size_t> debug_intervals;
                    // Track job completion and intervals
                    std::size_t lastComplete = stream.lastComplete;
                    stream.interval = cyclesRun - lastComplete;
                    stream.lastComplete = cyclesRun;
                    stream.job_txns = 0;
                    ++completedMaps;
                    if (lastComplete != 0) {
                        // Update stable state tracker
                        stream.stableState.update(stream.interval);
                    }
                }
            }
        }
    }

    /**
     * Initialize streams according to nodeindex
     */
    void initStreams() {
        if constexpr (FirstNode) {             // First Node; no predecessor
            for (auto&& s : this->istreams) {  // Input into sim valid
                s.valid(true);
            }
        } else if constexpr (LastNode) {       // Last Node; no successor
            for (auto&& s : this->ostreams) {  // Output from sim ready
                s.ready(true);
            }
        }
    }

    [[gnu::hot, gnu::always_inline]] void runSingleCycle(std::stop_token stoken = {}) {
        ++cyclesRun;
        communicate(stoken);
        this->clk.toggleClk();
    }

     public:
    SingleNodeSimulation(const std::string& kernel_lib, const std::string& design_lib, const char* xsim_log_file, const char* trace_file,
                         std::array<StreamDescriptor, IStreamsSize> _istream_descs, std::array<StreamDescriptor, OStreamsSize> _ostream_descs,
                         std::optional<std::string> prevNodeName = std::nullopt, std::optional<std::string> nodeName = std::nullopt, unsigned int initialFIFODepth = 2)
        : Simulation<IStreamsSize, OStreamsSize, LoggingEnabled>(kernel_lib, design_lib, xsim_log_file, trace_file, _istream_descs, _ostream_descs) {
        if (!FirstNode && !prevNodeName) {
            throw std::runtime_error("Cannot communicate with predecessor because previous node name was not given!");
        } else if (FirstNode && prevNodeName) {
            std::cout << "Simulation was passed the previous nodes name but is "
                         "NOT marked for communication with predecessor node. No "
                         "shared memory will be created."
                      << std::endl;
        }
        if (!LastNode && !nodeName) {
            throw std::runtime_error(
                "Cannot communicate with successor because "
                "current node name was not given!");
        } else if (LastNode && nodeName) {
            std::cout << "Simulation was passed the current nodes name but is NOT "
                         "marked for communication with successor node. No shared "
                         "memory will be created."
                      << std::endl;
        }
        if constexpr (!LastNode) {
            // Create FIFO buffer
            for (std::size_t i = 0; i < OStreamsSize; ++i) {
                fifo[i] = FIFO(initialFIFODepth);
            }
        }


        if constexpr (!FirstNode) {
            for (std::size_t i = 0; i < IStreamsSize; ++i) {
                fromProducerInterface[i] = std::move(ConsumingInterface(std::format("{}_{}", *prevNodeName, i)));
            }
        }

        if constexpr (!LastNode) {
            // Create consumer facing interfaces
            for (std::size_t i = 0; i < OStreamsSize; ++i) {
                toConsumerInterface[i] = std::move(ProducingInterface(std::format("{}_{}", *nodeName, i)));
            }
        }

        initStreams();
        debug("Finished initializing simulation.\nlog ------------------------------\n");
    }

    /// Reset simulation (stream and current FIFO depth, as well as cycle counter)
    void reset() {
        Simulation<IStreamsSize, OStreamsSize, LoggingEnabled>::reset();
        if constexpr (!LastNode) {
            // Reset FIFOs
            for (std::size_t i = 0; i < OStreamsSize; ++i) {
                fifo[i].reset();
            }
        }
    }

    [[gnu::hot, gnu::always_inline]] void runFeatureMaps(std::size_t featureMaps, std::stop_token stoken = {}) {
        completedMaps = 0;
        while (completedMaps < featureMaps && !stoken.stop_requested()) {
            runSingleCycle(stoken);
        }
    }

    [[gnu::hot, gnu::always_inline]] bool runToStableState(std::stop_token stoken = {}, std::size_t max_cycles = std::numeric_limits<std::size_t>::max()) {
        while (!std::all_of(this->ostreams.begin(), this->ostreams.end(), [](const M_AXIS_Control& stream) { return stream.stableState.is_stable(); }) &
               !stoken.stop_requested() & (cyclesRun <= max_cycles)) {
            runSingleCycle(stoken);
            runSingleCycle(stoken);
            runSingleCycle(stoken);
            runSingleCycle(stoken);
        }
        return cyclesRun > max_cycles;
    }

    /// Get the number of FIFOs
    std::size_t getFIFOCount() const noexcept {
        if constexpr (LastNode) {
            return 0;
        }
        return OStreamsSize;
    }

    /// Set the depth of a specific FIFO
    void setFIFODepth(std::size_t index, std::size_t depth) {
        if constexpr (LastNode) {
            throw std::runtime_error("Cannot set FIFO depth on last node (no FIFOs present)");
        }
        if (index >= OStreamsSize) {
            throw std::out_of_range(std::format("FIFO index {} out of range (max: {})", index, OStreamsSize - 1));
        }
        fifo[index].setMaxSize(depth);
    }

    /// Set the max FIFO depth of all interfaces
    void setMaxFIFODepth(std::size_t depth) {
        if constexpr (!LastNode) {
            for (FIFO& f : fifo) {
                f.setMaxSize(depth);
            }
        }
    }

    std::array<std::size_t, OStreamsSize> getFIFODepth() const noexcept {
        if constexpr (LastNode) {
            return {};
        }
        std::array<std::size_t, OStreamsSize> utilizations{};
        for (std::size_t i = 0; i < OStreamsSize; ++i) {
            utilizations[i] = fifo[i].getMaxSize();
        }
        return utilizations;
    }

    /// Get the job size of the specified output stream
    std::size_t getOutputJobSize(std::size_t outputIndex = 0) { return this->ostreams[outputIndex].job_size; }

    /// Get the job size of the specified input stream
    std::size_t getInputJobSize(std::size_t inputIndex = 0) { return this->istreams[inputIndex].job_size; }

    /// Get the number of cycles the simulation has run
    std::size_t getCyclesRun() const noexcept { return cyclesRun; }

    /// Get the number of completed feature maps
    std::size_t getCompletedMaps() const noexcept { return completedMaps; }

    /// Get the maximum FIFO utilization for each output stream
    std::array<std::size_t, OStreamsSize> getFIFOUtilization() const noexcept {
        if constexpr (LastNode) {
            return {};
        }
        std::array<std::size_t, OStreamsSize> utilizations{};
        for (std::size_t i = 0; i < OStreamsSize; ++i) {
            utilizations[i] = fifo[i].getMaxUtil();
        }
        return utilizations;
    }

    ///Get the current Ostream stable state intervals
    std::array<std::size_t, OStreamsSize> getOStreamStableStateIntervals() const noexcept {
        std::array<std::size_t, OStreamsSize> intervals{};
        if constexpr (LastNode) {
            for (std::size_t i = 0; i < OStreamsSize; ++i) {
            intervals[i] = this->ostreams[i].interval;
        }
        }
        return intervals;
    }
};


#endif /* SIMULATION */
