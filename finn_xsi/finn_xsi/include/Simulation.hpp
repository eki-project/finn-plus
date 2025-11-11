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
#include <queue>
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
class _SingleNodeSimulation : public Simulation<IStreamsSize, OStreamsSize, LoggingEnabled> {
     private:
    // Interface members
    using ConsumingInterface = SimulationInterface<SimulationInterfaceType::CONSUMING, NodeIndex == 0 || NodeIndex == TotalNodes - 1>;
    using ProducingInterface = SimulationInterface<SimulationInterfaceType::PRODUCING, NodeIndex == 0 || NodeIndex == TotalNodes - 1>;
    std::array<ConsumingInterface, IStreamsSize> fromProducerInterface;
    std::array<ProducingInterface, OStreamsSize> toConsumerInterface;
    std::array<FIFO, OStreamsSize> fifo;
    std::size_t cyclesRun = 0;

     public:
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
            if constexpr (NodeIndex == TotalNodes - 1) {
                toConsumerInterface[i] = std::move(ProducingInterface("output_shmem", initialFIFODepth));
            } else {
                toConsumerInterface[i] = std::move(ProducingInterface(std::format("{}_{}", *nodeName, i).c_str(), initialFIFODepth));
            }
        }
        for (std::size_t i = 0; i < IStreamsSize; ++i) {
            if constexpr (NodeIndex == 0) {
                fromProducerInterface[i] = std::move(ConsumingInterface("input_shmem", initialFIFODepth));
            } else {
                fromProducerInterface[i] = std::move(ConsumingInterface(std::format("{}_{}", *prevNodeName, i).c_str(), initialFIFODepth));
            }
        }

        debug("log Finished initializing simulation.\nlog ------------------------------\n");
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

    /// Write the current ready and valid states into the filestreams
    void logReadyValidState() requires LoggingEnabled {
        // Log the signals that this simulations set (ready to predecessor, valid to successor)
        // TODO: Collect signals in vectors and only write to file after the sim for speedup
        for (S_AXIS_Control& stream : this->istreams) {
            this->readyLog << stream.is_ready() << " ";
        }
        this->readyLog << "\n";
        for (M_AXIS_Control& stream : this->ostreams) {
            this->validLog << stream.is_valid() << " ";
        }
        this->validLog << "\n";
    }


     public:
    /// Reset simulation (stream and current FIFO depth, as well as cycle counter)
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

    /// Return the largest occupation the specified output stream / FIFO has seen
    std::size_t getLargestOccupation(std::size_t outputIndex) {
        return toConsumerInterface[outputIndex].getLargestOccupation();
    }

    [[gnu::hot]] void runSingleCycle() {
        static bool firstCycle = true;
        if (firstCycle) {
            firstCycle = false;
            communicate();  // Initial communication before first cycle
        }
        this->clk.toggle_clk();
        communicate();
        if constexpr (LoggingEnabled) {
           logReadyValidState();
        }
        debug(std::format("Finished cycle {}\n\n", cyclesRun));
        ++cyclesRun;
    }

    /// Set the max FIFO depth of all interfaces
    void setMaxFIFODepth(unsigned int depth) {
        for (ProducingInterface& prod : toConsumerInterface) {
            prod.setMaxFifoDepth(depth);
        }
        for (ConsumingInterface& cons : fromProducerInterface) {
            cons.setMaxFifoDepth(depth);
        }
    }

    /// Get the job size of the specified output stream
    std::size_t getOutputJobSize(std::size_t outputIndex = 0) {
        return this->ostreams[outputIndex].job_size;
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

    // Run until cyclesTarget are hit
    std::size_t cyclesTarget;

    // Current run cycles counter
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
        cyclesTarget(0),
        cyclesRun(0),
        simulationDataPath(simulationDataFilename),
        sim(kernel_lib, design_lib, xsim_log_file, trace_file, _istream_descs, _ostream_descs, prevNodeName, nodeName, initialFIFODepth) {}


    /// Stop and reset the simulation, reset cycle counter, target cycle counter, log queue.
    /// Leaves the simulation in a paused and reset state.
    void reset() {
        running = false;
        sim.reset();
        cyclesRun = 0;
        cyclesTarget = 0;
    }

    /// Write the results of the simulation as a JSON file
    void writeResults() {
        json j;
        for (std::size_t i = 0; i < OStreamsSize; ++i) {
            j["maxOccupation"][std::to_string(i)] = fifo[i].get_largest_occupation();
        }
        j["cyclesRun"] = cyclesRun;
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

    /// Start both threads. Listen for commands on stdin.
    void start() {
        running = true;
        simulator = std::jthread([this](std::stop_token stop) {
            while (cyclesTarget == 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    std::cout << "log Waiting.... (target: " << cyclesTarget << ")" << std::endl;
            }
            std::cout << "log Starting running with a target of " << cyclesTarget << " cycles!" << std::endl;

            // TODO: Move to communicator thread
            std::cout << "started" << std::endl;
            while (cyclesRun < cyclesTarget) {
                if (!running) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
                if (stop.stop_requested()) {
                    return;
                }
                std::cout << "cycles " << cyclesRun << " " << cyclesTarget << std::endl;
                sim.runSingleCycle();
                ++cyclesRun;
            }
            // TODO: Move to communicator thread
            std::cout << "stopped" << std::endl;
        });
        communicator = std::jthread([this]() {
            std::cout << "ready" << std::endl;
            std::string input = "";
            while (true) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (cyclesTarget != 0 && cyclesRun == cyclesTarget) {
                    std::cout << "end" << std::endl;
                    return;
                }
                if constexpr(LoggingEnabled) {
                    if (running && cyclesRun % 5000 == 1) {
                        std::cout << "cycles " << cyclesRun << " " << cyclesTarget << std::endl;
                    }
                }
                // Parse incoming commands
                getlineIfAvailable(input);
                if (input == "") {
                    continue;
                }
                auto [command, argument] = splitSpace(input);
                if (command == "stop") {
                    simulator.request_stop();
                    writeResults();
                    // Wait for the file to be fully written
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                    // Signal python that we are done
                    std::cout << "end" << std::endl;
                    return;
                } else if (command == "fifodepth") {
                    unsigned int newDepth = static_cast<unsigned int>(std::stoul(argument));
                    running = false;
                    sim.setMaxFIFODepth(newDepth);
                    running = true;
                    std::cout << "log Set FIFO depth to " << newDepth << std::endl;
                } else if (command == "runCycles") {
                    cyclesTarget += static_cast<unsigned int>(std::stoul(argument));
                    running = true;
                    std::cout << "log Set running with " << cyclesTarget << std::endl;
                } else if (command == "runSamples") {
                    cyclesTarget += static_cast<unsigned int>(std::stoul(argument)) * sim.getOutputJobSize(0);
                    running = true;
                    std::cout << "log Set running with " << cyclesTarget << std::endl;
                } else if (command == "pause") {
                    running = false;
                } else if (command == "reset") {
                    reset();
                } else if (command == "resume") {
                    running = true;
                } else if (command == "help") {
                    // TODO: Insert here or document separately
                } else {
                    std::cout << "log Unknown command " << std::endl;
                }
            }
        });
        simulator.join();
        communicator.join();
    }
};


#endif /* SIMULATION */
