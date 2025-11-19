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

#include <InterSimulationInterface.hpp>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <thread>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
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

    template<std::size_t Index>
    bool hasValidOutput() {
        //static_assert(Index < ostreams.size(), "Cannot request valid status of unknown output stream index");
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
class _SingleNodeSimulation : public Simulation<IStreamsSize, OStreamsSize, LoggingEnabled> {
     private:
    using ConsumingInterface = InterSimulationInterface<true>;
    using ProducingInterface = InterSimulationInterface<false>;
    std::array<ConsumingInterface, IStreamsSize> fromProducerInterface;
    std::array<ProducingInterface, OStreamsSize> toConsumerInterface;
    std::size_t cyclesRun = 0;


     public:
    // TODO: Move to private, currently only here for debugging purposes
    std::array<FIFO, static_cast<int>(OStreamsSize)> fifo;

    _SingleNodeSimulation(const std::string& kernel_lib, const std::string& design_lib, const char* xsim_log_file, const char* trace_file, std::array<StreamDescriptor, IStreamsSize> _istream_descs,
                         std::array<StreamDescriptor, OStreamsSize> _ostream_descs, std::optional<std::string> prevNodeName = std::nullopt, std::optional<std::string> nodeName = std::nullopt, unsigned int initialFIFODepth = 2)
        : Simulation<IStreamsSize, OStreamsSize, LoggingEnabled>(kernel_lib, design_lib, xsim_log_file, trace_file, _istream_descs, _ostream_descs) {
        if (CommunicatesWithPredecessor && !prevNodeName) {
            throw std::runtime_error("Cannot communicate with predecessor because previous node name was not given!");
        } else if (!CommunicatesWithPredecessor && prevNodeName) {
            std::cout << "log Simulation was passed the previous nodes name but is "
                         "NOT marked for communication with predecessor node. No "
                         "shared memory will be created." << std::endl;
        }
        if (CommunicatesWithSuccessor && !nodeName) {
            throw std::runtime_error("Cannot communicate with successor because "
                                     "current node name was not given!");
        } else if (!CommunicatesWithSuccessor && nodeName) {
            std::cout << "log Simulation was passed the current nodes name but is NOT "
                         "marked for communication with successor node. No shared "
                         "memory will be created." << std::endl;
        }
        // Create FIFO buffer
        for (std::size_t i = 0; i < OStreamsSize; ++i) {
            fifo[i] = FIFO(initialFIFODepth);
        }

        // Create consumer facing interfaces
        for (std::size_t i = 0; i < OStreamsSize; ++i) {
            toConsumerInterface[i] = std::move(ProducingInterface(std::format("{}_{}", *nodeName, i)));
        }
        if constexpr (NodeIndex != 0) {
            for (std::size_t i = 0; i < IStreamsSize; ++i) {
                fromProducerInterface[i] = std::move(ConsumingInterface(std::format("{}_{}", *prevNodeName, i)));
            }
        }

        initStreams();
        debug("log Finished initializing simulation.\nlog ------------------------------\n");
    }


     private:
    /// Communicate with predecessors and successors and update their values and our own
    [[gnu::hot, gnu::flatten, gnu::always_inline]] void communicate() {
        if constexpr (NodeIndex != 0) {
            for (std::size_t i = 0; i < IStreamsSize; ++i) {
                // Interface SHM <-> sim
                this->istreams[i].valid(fromProducerInterface[i].exchange(this->istreams[i].isReady()));
            }
        }
        if constexpr (NodeIndex != TotalNodes - 1) {
            for (std::size_t i = 0; i < OStreamsSize; ++i) {
                this->fifo[i].update(
                    this->ostreams[i].isValid(),
                    toConsumerInterface[i].exchange(this->fifo[i].isOutputValid())
                );
                this->fifo[i].toggleClock();
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
                s.valid(true);
            }
        } else if constexpr (NodeIndex == TotalNodes - 1) {  // Last Node; no successor
            for (auto&& s : this->ostreams) {                // Output from sim ready
                s.ready(true);
            }
        }
    }

    /// Write the current ready and valid states into the filestreams
    void logReadyValidState() requires LoggingEnabled { /* TODO: Match the new extra FIFO implementation*/ }

    /// Reset simulation (stream and current FIFO depth, as well as cycle counter)
    void reset() {
        Simulation<IStreamsSize, OStreamsSize, LoggingEnabled>::reset();
        for (std::size_t i = 0; i < OStreamsSize; ++i) {
            toConsumerInterface[i].reset();
            fifo[i].reset();
        }
        if constexpr (NodeIndex != 0) {
            for (std::size_t i = 0; i < IStreamsSize; ++i) {
                fromProducerInterface[i].reset();
            }
        }
    }

    /// Return whether all connected FIFOs are empty
    bool allFIFOsEmpty() const {
        return std::all_of(fifo.begin(), fifo.end(), [](const FIFO& f) { return f.isEmpty(); });
    }

    /// Finish communication by exchanging FIFO contents with the successor, but not
    /// interacting with the predecessor anymore. Runs until the FIFO is empty
    [[gnu::hot]] void finishCommunication() {
       while (!allFIFOsEmpty()) {
            // for (std::size_t i = 0; i < toConsumerInterface.size(); ++i) {
            //     this->fifo[i].tryPop(toConsumerInterface[i].writeToNextNode(this->fifo[i].isOutputValid()));
            // }
            runSingleCycle();
       }
    }

    [[gnu::hot, gnu::always_inline]] void runSingleCycle() {
        ++cyclesRun;
        this->clk.toggleClk();
        communicate();
        if constexpr (LoggingEnabled) {
           logReadyValidState();
        }
        debug(std::format("Finished cycle {}\n\n", cyclesRun));
    }

    /// Set the max FIFO depth of all interfaces
    void setMaxFIFODepth(std::size_t depth) {
        for (FIFO& f : fifo) {
            f.setMaxSize(depth);
        }
    }

    /// Get the job size of the specified output stream
    std::size_t getOutputJobSize(std::size_t outputIndex = 0) {
        return this->ostreams[outputIndex].job_size;
    }

    /// Get the job size of the specified input stream
    std::size_t getInputJobSize(std::size_t inputIndex = 0) {
        return this->istreams[inputIndex].job_size;
    }
};



/// Single Node Simulation, thread controlled
template<size_t IStreamsSize, size_t OStreamsSize, bool LoggingEnabled, size_t NodeIndex, size_t TotalNodes, bool CommunicatesWithPredecessor, bool CommunicatesWithSuccessor>
class SingleNodeSimulation {
    private:
    std::jthread simulator;
    std::jthread communicator;

    // Only run cycles if True
    // TODO: Atomic?
    bool running;

    std::size_t samplesTarget;
    std::size_t samplesProduced;
    std::size_t validCyclesProduced;
    std::size_t cyclesTarget;
    std::size_t cyclesRun;


    // Path on which to store simulation data after stopping
    std::filesystem::path simulationDataPath;

    // The simulation itself
    _SingleNodeSimulation<IStreamsSize, OStreamsSize, LoggingEnabled, NodeIndex, TotalNodes, CommunicatesWithPredecessor, CommunicatesWithSuccessor> sim;

    public:
    SingleNodeSimulation(
            const std::string& kernel_lib,
            const std::string& design_lib,
            const char* xsim_log_file,
            const char* trace_file,
            std::array<StreamDescriptor, IStreamsSize> _istream_descs,
            std::array<StreamDescriptor, OStreamsSize> _ostream_descs,
            std::optional<std::string> prevNodeName = std::nullopt,
            std::optional<std::string> nodeName = std::nullopt,
            unsigned int initialFIFODepth = 2,
            std::string simulationDataFilename = "simulation_data.json"
    ) : running(false),
        samplesProduced(0),
        samplesTarget(0),
        validCyclesProduced(0),
        cyclesRun(0),
        cyclesTarget(0),
        simulationDataPath(simulationDataFilename),
        sim(kernel_lib, design_lib, xsim_log_file, trace_file, _istream_descs, _ostream_descs, prevNodeName, nodeName, initialFIFODepth) {}


    /// Stop and reset the simulation, reset cycle counter, target cycle counter, log queue.
    /// Leaves the simulation in a paused and reset state.
    void reset() {
        running = false;
        sim.reset();
        samplesTarget = 0;
        samplesProduced = 0;
    }

    /// Write the results of the simulation as a JSON file
    void writeResults() {
        json j;
        // TODO: Reintroduce
        // for (std::size_t i = 0; i < OStreamsSize; ++i) {
        //     j["maxOccupation"][std::to_string(i)] = sim.getLargestOccupation(i);
        // }
        //j["cyclesRun"] = outValidCycles;
        std::ofstream file(simulationDataPath);
        file << j.dump(4);
        file.close();
    }

    /// Split the given string in two, delimited by a space. Subsequent spaces are ignored. If no
    /// space is found, the second element is empty.
    std::tuple<std::string, std::string> splitSpace(std::string& s) {
        auto pos = s.find(" ");
        if (s == "") {
            return std::make_tuple("", "");
        }
        return std::make_tuple(s.substr(0, pos), s.substr(pos));
    }

    /// Read from std::cin if possible, otherwise
    /// return immediately.
    void getlineIfAvailable(std::string& buffer) {
        //std::cin.exceptions(std::istream::failbit | std::istream::badbit);
        if (std::cin.eof() || std::cin.rdbuf()->in_avail() == -1) {
            buffer = "";
            return;
        }
        std::getline(std::cin, buffer);
    }


    void sendLog(std::string message) { std::cout << "log " << message << std::endl; }
    void sendError(std::string message) { std::cout << "error " << message << std::endl; }
    void sendEnd() { std::cout << "end" << std::endl; }
    void sendCycles() { std::cout << "cycles " << cyclesRun << " " << cyclesTarget << std::endl; }
    void sendSamples() { std::cout << "samples " << samplesProduced << " " << samplesTarget << std::endl; }
    void sendStarted() { std::cout << "started" << std::endl; }
    void sendStopped() { std::cout << "stopped" << std::endl; }
    void sendReady() { std::cout << "ready" << std::endl; }


    /// Start both threads. Listen for commands on stdin.
    void start() {
        simulator = std::jthread([this](std::stop_token stop) {
            /// Prepare and wait for data from the controller
            while (samplesTarget == 0 && cyclesTarget == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            /// Run the simulation
            sendStarted();
            if constexpr(NodeIndex == TotalNodes - 1) {
                while ((samplesTarget != 0 && samplesProduced < samplesTarget) || (cyclesTarget != 0 && cyclesRun < cyclesTarget)) {
                    if (!running) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        continue;
                    }
                    sim.runSingleCycle();
                    ++cyclesRun;
                    if (std::all_of(sim.ostreams.begin(), sim.ostreams.end(), [](M_AXIS_Control& s) { return s.isValid(); })) {
                        ++validCyclesProduced;
                        // TODO: For all streams
                        samplesProduced = validCyclesProduced / sim.ostreams[0].job_size;
                        sendSamples();
                    }
                }
                sendStopped();
            } else {
                bool validSeen = false;
                while (!running) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                while (running) {
                    sim.runSingleCycle();
                    if (!validSeen && sim.ostreams[0].isValid()) {
                        validSeen = true;
                        sendLog("First valid sample seen");
                    }
                }
            }
            // Finish communicating and sent an update to the controller
            sim.finishCommunication();
            sendStopped();
            if constexpr (NodeIndex == TotalNodes - 1) {
                sendEnd();
            }
        });

        communicator = std::jthread([this]() {
            // Run initial sanity checks
            if constexpr (NodeIndex == 0) {
                if (!std::all_of(sim.istreams.begin(), sim.istreams.end(), [](S_AXIS_Control& stream) { return stream.isValid(); })) {
                    sendLog("ERROR: First node input is not set to valid!");
                    // TODO: Stop
                }
            } else if constexpr (NodeIndex == TotalNodes - 1) {
                if (!std::all_of(sim.ostreams.begin(), sim.ostreams.end(), [](M_AXIS_Control& stream) { return stream.isReady(); })) {
                    sendLog("ERROR: Last node output is not set to ready!");
                    // TODO: Stop
                }
            }
            sendReady();
            std::string input = "";
            while (true) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                // Parse incoming commands
                getlineIfAvailable(input);
                if (input == "") {
                    continue;
                }
                auto [command, argument] = splitSpace(input);

                // React to command
                if (command == "stop") {
                    simulator.request_stop();
                    writeResults();
                    // Wait for the file to be fully written
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                    // Signal python that we are done
                    sendEnd();
                    return;
                } else if (command == "fifodepth") {
                    std::size_t newDepth = static_cast<unsigned int>(std::stoul(argument));
                    running = false;
                    sim.setMaxFIFODepth(newDepth);
                    running = true;
                    sendLog(std::format("Set FIFO depth to {}", newDepth));
                } else if (command == "runCycles") {
                    cyclesTarget += static_cast<unsigned int>(std::stoul(argument));
                    running = true;
                    sendLog(std::format("Set running with {}", cyclesTarget));
                } else if (command == "runSamples") {
                    // TODO: Make generic for any number of streams (max)
                    samplesTarget += static_cast<unsigned int>(std::stoul(argument));
                    running = true;
                    sendLog(std::format("Set running with a target of {} samples!", samplesTarget));
                } else if (command == "pause") {
                    running = false;
                } else if (command == "reset") {
                    reset();
                } else if (command == "resume") {
                    running = true;
                } else if (command == "help") {
                    // TODO: Insert here or document separately
                } else {
                    sendLog("Unknown command.");
                }
            }
        });
        simulator.join();
        communicator.join();
    }
};


#endif /* SIMULATION */
