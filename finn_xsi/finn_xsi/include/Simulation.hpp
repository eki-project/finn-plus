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

#include <InterprocessCommunicationChannelInterface.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
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
            istreams[i] = S_AXIS_Control{top, clk, std::data(_istream_descs)[i].job_size, std::data(_istream_descs)[i].job_size, std::data(_istream_descs)[i].name};
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

// Small struct used for exange. Will be changed later to more complex data structure.
struct CommData {
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
template<size_t IStreamsSize, size_t OStreamsSize, bool LoggingEnabled, size_t NodeIndex, size_t TotalNodes, bool FirstNode, bool LastNode>
class SingleNodeSimulation : public Simulation<IStreamsSize, OStreamsSize, LoggingEnabled> {
    using ConsumingInterface = InterprocessCommunicationChannel<CommData, CommData, true>;
    using ProducingInterface = InterprocessCommunicationChannel<CommData, CommData, false>;
    std::array<ConsumingInterface, IStreamsSize> fromProducerInterface;
    std::array<ProducingInterface, OStreamsSize> toConsumerInterface;
    std::size_t cyclesRun = 0;
    std::size_t completedMaps = 0;
    std::array<FIFO, OStreamsSize> fifo;

    /**
     * Initialize streams according to nodeindex
     */
    void initStreams() {
        if constexpr (FirstNode) {             // First Node; no predecessor
            for (auto&& s : this->istreams) {  // Input into sim valid
                s.setInputValid(true);
            }
        } else if constexpr (LastNode) {       // Last Node; no successor
            for (auto&& s : this->ostreams) {  // Output from sim ready
                s.setOutputReady(true);
            }
        }
    }

    [[gnu::hot, gnu::always_inline]] bool runSingleCycle(std::stop_token stoken = {}) {
        ++cyclesRun;
        bool ret = false;
        if constexpr (!FirstNode) {
            for (std::size_t i = 0; i < IStreamsSize; ++i) {
                // Interface SHM <-> sim
                bool istreamReady = this->istreams[i].getInputReady();
                bool fifoValid = fromProducerInterface[i].send_request(CommData{istreamReady}, stoken).data;
                this->istreams[i].setValid(fifoValid);  // deferred
            }
        }
        if constexpr (!LastNode) {
            for (std::size_t i = 0; i < OStreamsSize; ++i) {
                // Interface sim -valid-> FIFO
                this->fifo[i].setInputValid(this->ostreams[i].getOutputValid(), stoken);
                // Interface FIFO <-> SHM
                this->fifo[i].setOutputReady(toConsumerInterface[i].receive_request(stoken).data, stoken);
                
                // Toggle FIFO clock
                ret |= this->fifo[i].toggleClock();
                bool fifoValid = this->fifo[i].getOutputValid();
                toConsumerInterface[i].send_response(CommData{fifoValid});
                // FIFO -ready-> sim
                this->ostreams[i].setReady(this->fifo[i].getInputReady());
            }
        }
        if constexpr (LastNode) {
            for (auto&& stream : this->ostreams) {
                if (stream.getOutputValid() && ++stream.job_txns == stream.job_size) {
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
        // ── CLOCK HIGH ─────────────────────────────────────────────────────────
        this->clk.clockHigh();  // run(1) [gap] → clk=1 → run(1)

        // ── WRITE (clock is high, commit deferred setValid / setReady) ─────────
        //
        // The deferred values were prepared at the end of the previous cycle's read
        // phase (or are defaults for the first cycle).
        for (std::size_t i = 0; i < IStreamsSize; ++i) {
            this->istreams[i].writeBack();
        }
        for (std::size_t i = 0; i < OStreamsSize; ++i) {
            this->ostreams[i].writeBack();
        }

        // ── CLOCK LOW ──────────────────────────────────────────────────────────
        this->clk.clockLow();  // run(4999) → clk=0 → run(4999)  ← sim settles
        return ret;
    }

     public:
    SingleNodeSimulation(const std::string& kernel_lib, const std::string& design_lib, const char* xsim_log_file, const char* trace_file,
                         std::array<StreamDescriptor, IStreamsSize> _istream_descs, std::array<StreamDescriptor, OStreamsSize> _ostream_descs,
                         std::array<std::string_view, IStreamsSize> inputInterfaceNames, std::array<std::string_view, OStreamsSize> outputInterfaceNames,
                         unsigned int initialFIFODepth = 2)
        : Simulation<IStreamsSize, OStreamsSize, LoggingEnabled>(kernel_lib, design_lib, xsim_log_file, trace_file, _istream_descs, _ostream_descs) {
        if (!FirstNode && inputInterfaceNames.empty()) {
            throw std::runtime_error("Cannot communicate with predecessor because previous node name was not given!");
        }
        if (!LastNode && outputInterfaceNames.empty()) {
            throw std::runtime_error(
                "Cannot communicate with successor because "
                "current node name was not given!");
        }

        if constexpr (!LastNode) {
            // Create FIFO buffer
            for (std::size_t i = 0; i < OStreamsSize; ++i) {
                fifo[i] = FIFO(initialFIFODepth);
            }
        }

        std::cout << "Initialized " << OStreamsSize << " output FIFOs with depth " << initialFIFODepth << std::endl;

        if constexpr (!LastNode) {
            // Create consumer facing interfaces
            for (std::size_t i = 0; i < OStreamsSize; ++i) {
                std::string shmName{outputInterfaceNames[i]};
                toConsumerInterface[i] = std::move(ProducingInterface(shmName));
            }
        }

        std::cout << "Initialized " << OStreamsSize << " producing interfaces for successor communication" << std::endl;

        if constexpr (!FirstNode) {
            for (std::size_t i = 0; i < IStreamsSize; ++i) {
                std::string shmName{inputInterfaceNames[i]};
                fromProducerInterface[i] = std::move(ConsumingInterface(shmName));
            }
        }

        std::cout << "Initialized " << IStreamsSize << " consuming interfaces for predecessor communication" << std::endl;

        // Verify communication works
        if constexpr (!LastNode) {
            for (std::size_t i = 0; i < OStreamsSize; ++i) {
                toConsumerInterface[i].handshake();
            }
        }
        if constexpr (!FirstNode) {
            for (std::size_t i = 0; i < IStreamsSize; ++i) {
                fromProducerInterface[i].handshake();
            }
        }

        this->clk.clockHigh();
        initStreams();
        this->clk.clockLow();
        std::cout << "Finished initializing simulation." << std::endl;
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
        bool timeout = false;
        while (!std::all_of(this->ostreams.begin(), this->ostreams.end(), [](const M_AXIS_Control& stream) { return stream.stableState.is_stable(); }) & !stoken.stop_requested() &
               (cyclesRun <= max_cycles) & !timeout) {
            timeout |= runSingleCycle(stoken);
            timeout |= runSingleCycle(stoken);
            timeout |= runSingleCycle(stoken);
            timeout |= runSingleCycle(stoken);
        }
        return timeout || cyclesRun > max_cycles;
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
            auto error = "FIFO index " + std::to_string(index) + " out of range (max: " + std::to_string(OStreamsSize - 1) + ")";
            throw std::out_of_range(error);
        }
        fifo[index].setMaxSize(depth);
    }

    void setFIFOCyclesUntilExpectedFirstValid(std::size_t index, std::size_t cycles) {
        if constexpr (LastNode) {
            throw std::runtime_error("Cannot set FIFO cycles until expected first valid on last node (no FIFOs present)");
        }
        if (index >= OStreamsSize) {
            auto error = "FIFO index " + std::to_string(index) + " out of range (max: " + std::to_string(OStreamsSize - 1) + ")";
            throw std::out_of_range(error);
        }
        fifo[index].setCyclesUntilExpectedFirstValid(cycles);
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

    std::array<std::size_t, OStreamsSize> getFIFOCyclesUntilFirstValid() const noexcept {
        if constexpr (LastNode) {
            return {};
        }
        std::array<std::size_t, OStreamsSize> cycles{};
        for (std::size_t i = 0; i < OStreamsSize; ++i) {
            cycles[i] = fifo[i].getCyclesUntilFirstValid();
        }
        return cycles;
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

    /// Get the current Ostream stable state intervals.
    /// Returns the rounded EMA of observed output intervals so that a single noisy
    /// measurement at the boundary of stability does not cause _check_performance to
    /// report a false positive or negative (raw last interval can differ from the EMA
    /// by up to the StableStateTracker stability threshold in either direction).
    /// This should not be the case, but its an additional security measure.
    std::array<std::size_t, OStreamsSize> getOStreamStableStateIntervals() const noexcept {
        std::array<std::size_t, OStreamsSize> intervals{};
        if constexpr (LastNode) {
            for (std::size_t i = 0; i < OStreamsSize; ++i) {
                const double ema = this->ostreams[i].stableState.get_ema();
                // Fall back to the raw interval when the EMA has never been updated
                // (ema == 0.0 means no second job completion has occurred yet).
                intervals[i] = (ema > 0.0) ? static_cast<std::size_t>(std::round(ema)) : this->ostreams[i].interval;
            }
        }
        return intervals;
    }
};


#endif /* SIMULATION */
