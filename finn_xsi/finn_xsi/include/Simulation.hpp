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

#include <SimulationInterface.hpp>
#include <boost/asio.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/atomic/ipc_atomic.hpp>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/detail/os_file_functions.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string_view>
using json = nlohmann::json;


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


    Simulation(const std::string& kernel_lib, const std::string& design_lib, const char* xsim_log_file, const char* trace_file, std::array<StreamDescriptor, IStreamsSize> _istream_descs,
               std::array<StreamDescriptor, OStreamsSize> _ostream_descs)
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
            clk.toggle_clk();
        }
        rst_n.set(1).write_back();
    }
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
template<size_t IStreamsSize, size_t OStreamsSize, bool LoggingEnabled, size_t NodeIndex, size_t TotalNodes, bool CommunicatesWithPredecessor, bool CommunicatesWithSuccessor>
class SingleNodeSimulation : public Simulation<IStreamsSize, OStreamsSize, LoggingEnabled> {
     private:
    using ConsumingInterface = SimulationInterface<SimulationInterfaceType::CONSUMING>;
    using ProducingInterface = SimulationInterface<SimulationInterfaceType::PRODUCING>;
    std::array<ConsumingInterface, IStreamsSize> fromProducerInterface;
    std::array<ProducingInterface, OStreamsSize> toConsumerInterface;
    std::array<FIFO, OStreamsSize> fifo;
    std::size_t cyclesRun = 0;

     public:
    SingleNodeSimulation(const std::string& kernel_lib, const std::string& design_lib, const char* xsim_log_file, const char* trace_file, std::array<StreamDescriptor, IStreamsSize> _istream_descs,
                         std::array<StreamDescriptor, OStreamsSize> _ostream_descs, std::optional<std::string> prevNodeName = std::nullopt, std::optional<std::string> nodeName = std::nullopt, unsigned int initialFIFODepth = 2)
        : Simulation<IStreamsSize, OStreamsSize, LoggingEnabled>(kernel_lib, design_lib, xsim_log_file, trace_file, _istream_descs, _ostream_descs) {
        if (CommunicatesWithPredecessor && !prevNodeName) {
            throw std::runtime_error("Cannot communicate with predecessor because previous node name was not given!");
        } else if (!CommunicatesWithPredecessor && prevNodeName) {
            std::cout << "Simulation was passed the previous nodes name but is NOT marked for communication with predecessor node. No shared memory will be created." << std::endl;
        }
        if (CommunicatesWithSuccessor && !nodeName) {
            throw std::runtime_error("Cannot communicate with successor because current node name was not given!");
        } else if (!CommunicatesWithSuccessor && nodeName) {
            std::cout << "Simulation was passed the current nodes name but is NOT marked for communication with successor node. No shared memory will be created." << std::endl;
        }
        // Create FIFO buffer
        for (std::size_t i = 0; i < OStreamsSize; ++i) {
            fifo[i] = FIFO(initialFIFODepth);
        }

        // Create consumer facing interfaces
        debug(std::format("Creating {} interfaces for communication with successors.", OStreamsSize));
        if (NodeIndex != TotalNodes - 1 && nodeName && CommunicatesWithSuccessor) {
            for (std::size_t i = 0; i < OStreamsSize; ++i) {
                toConsumerInterface[i] = std::move(ProducingInterface(std::format("{}_{}", *nodeName, i).c_str()));
            }
        }

        // Create producer facing interfaces
        debug(std::format("Creating {} interfaces for communication with predecessors.", IStreamsSize));
        if (NodeIndex != 0 && prevNodeName && CommunicatesWithPredecessor) {
            for (std::size_t i = 0; i < IStreamsSize; ++i) {
                fromProducerInterface[i] = std::move(ConsumingInterface(std::format("{}_{}", *prevNodeName, i).c_str()));
            }
        }

        // Set valid and ready
        // TODO: Currently uses NodeIndex to check if we are input or output or neither
        // This is unstable because other nodes with neither last nor first index might
        // also be inputs or outputs.
        initStreams();

        debug("Finished initializing simulation.\n------------------------------\n");
    }

     private:
    /// Communicate with predecessors and successors and update their values and our own
    [[gnu::hot]] void communicate() {
        if constexpr (NodeIndex != TotalNodes - 1 && CommunicatesWithSuccessor) {
            for (std::size_t i = 0; i < OStreamsSize; ++i) {
                // Interface sim <-> FIFO
                this->fifo[i].write(this->ostreams[i].is_valid());
                this->ostreams[i].ready(this->fifo[i].is_ready());
                // Interface FIFO <-> SHM
                this->fifo[i].ready(toConsumerInterface[i].writeToNextNode(this->fifo[i].is_valid()));
            }
        }
        if constexpr (NodeIndex != 0 && CommunicatesWithPredecessor) {
            for (std::size_t i = 0; i < IStreamsSize; ++i) {
                // Interface SHM <-> sim
                this->istreams[i].valid(fromProducerInterface[i].readFromLastNode(this->istreams[i].is_ready()));
            }
        }
    }

     public:
    /**
     * Initialize streams according to nodeindex
     */
    void initStreams() {
        if constexpr (NodeIndex == 0) {        // First Node; no predecessor
            for (auto&& s : this->istreams) {  // Input into sim valid
                s.valid();
            }
        } else if constexpr (NodeIndex == TotalNodes - 1) {  // Last Node; no successor
            for (auto&& s : this->ostreams) {                // Output from sim ready
                s.ready();
            }
            for (std::size_t i = 0; i < IStreamsSize; ++i) {  // Relay ready from sim to predecessor
                fromProducerInterface[i].readFromLastNode(this->istreams[i].is_ready());
            }
        } else {                                              // Intermediate Node; has both predecessor and successor
            for (std::size_t i = 0; i < OStreamsSize; ++i) {  // Relay ready from FIFO to sim
                this->ostreams[i].ready(this->fifo[i].is_ready());
            }
            for (std::size_t i = 0; i < IStreamsSize; ++i) {  // Relay valid from sim to predecessor
                fromProducerInterface[i].readFromLastNode(this->istreams[i].is_ready());
            }
        }
    }

    /// Reset simulation (stream and current FIFO depth)
    void reset() {
        Simulation<IStreamsSize, OStreamsSize, LoggingEnabled>::reset();
        for (std::size_t i = 0; i < OStreamsSize; ++i) {
            toConsumerInterface[i].reset();
            fifo[i].reset();
        }
        for (std::size_t i = 0; i < IStreamsSize; ++i) {
            fromProducerInterface[i].reset();
        }
    }

    [[gnu::hot]] void runSingleCycle() {
        static bool firstCycle = true;
        if (firstCycle) {
            firstCycle = false;
            communicate();  // Initial communication before first cycle
        }
        this->clk.toggle_clk();
        communicate();
        debug(std::format("Finished cycle {}\n\n", cyclesRun));
        ++cyclesRun;

        // Log the signals that this simulations set (ready to predecessor, valid to successor)
        // TODO: Collect signals in vectors and only write to file after the sim for speedup
        if constexpr (LoggingEnabled) {
            for (S_AXIS_Control& stream : this->istreams) {
                this->readyLog << stream.is_ready() << " ";
            }
            this->readyLog << "\n";
            for (M_AXIS_Control& stream : this->ostreams) {
                this->validLog << stream.is_valid() << " ";
            }
            this->validLog << "\n";
        }
    }

    /// Write the results of the simulation as a JSON file
    void writeResults(std::filesystem::path& path) {
        json j;
        for (std::size_t i = 0; i < OStreamsSize; ++i) {
            j["maxOccupation"][std::to_string(i)] = fifo[i].get_largest_occupation();
        }
        j["cyclesRun"] = cyclesRun;
        std::ofstream file(path);
        file << j;
        file.close();
    }
};


#endif /* SIMULATION */
