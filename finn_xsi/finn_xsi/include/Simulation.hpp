#ifndef SIMULATION
#define SIMULATION
#include <AXIS_Control.h>
#include <Clock.h>
#include <Design.h>
#include <Kernel.h>
#include <Port.h>
#include <SharedLibrary.h>
#include <helper.h>
#include <boost/asio/thread_pool.hpp>
#include <boost/interprocess/detail/os_file_functions.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/atomic/ipc_atomic.hpp>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <format>
#include <iostream>
#include <memory>
#include <fstream>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/asio.hpp>
#include <string_view>
#include <optional>
#include <stdexcept>

#include <new>
#include <thread>
#ifdef __cpp_lib_hardware_interference_size
    using std::hardware_destructive_interference_size;
#else
    constexpr std::size_t hardware_destructive_interference_size = 64;
#endif

#ifdef NDEBUG
    [[maybe_unused]] inline void debug([[maybe_unused]] std::string_view s) {}
#else
    inline void debug(std::string_view s) { std::cout << "[DBG] " << s << "\n"; }
#endif

namespace ipc = boost::interprocess;


template<size_t IStreamsSize, size_t OStreamsSize, bool LoggingEnabled>
class Simulation {
    protected:
    std::ofstream readyLog;
    std::ofstream validLog;

    public:
    xsi::Kernel kernel;
    xsi::Design top;
    std::array<S_AXIS_Control, IStreamsSize> istreams;
    std::array<M_AXIS_Control, OStreamsSize> ostreams;
    Clock clk;

    /// Initialize streams to the correct valid and ready states.
    void initStreams() {
        for (auto&& s : istreams) {
            s.valid();
        }
        for (auto&& s : ostreams) {
            s.ready();
        }
    }

    Simulation(
        const std::string& kernel_lib,
        const std::string& design_lib,
        const char* xsim_log_file,
        const char* trace_file,
        std::array<StreamDescriptor, IStreamsSize> _istream_descs,
        std::array<StreamDescriptor, OStreamsSize> _ostream_descs
    ) : kernel(kernel_lib), top(kernel, design_lib, xsim_log_file, trace_file), clk(top) {
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
        if constexpr(LoggingEnabled) {
            readyLog.open("ready_log.txt");
            validLog.open("valid_log.txt");
        }

        // Find Global Control & Run Startup Sequence
        clearPorts();
        reset();
        initStreams();
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

enum class SimulationInterfaceType { PRODUCING, CONSUMING };
constexpr std::string_view to_string(SimulationInterfaceType t) {
    if (t == SimulationInterfaceType::CONSUMING) {
        return "CONSUMING";
    } else if (t == SimulationInterfaceType::PRODUCING) {
        return "PRODUCING";
    }
    return "UNKNOWN SIMULATION INTERFACE TYPE";
}

template<SimulationInterfaceType T, std::size_t ShmemSize = 1024>
class SimulationInterface {
    private:
    struct _SimulationInterface {
        alignas(hardware_destructive_interference_size) boost::ipc_atomic<bool>* iReady;
        alignas(hardware_destructive_interference_size) boost::ipc_atomic<unsigned int>* iCycle;
        alignas(hardware_destructive_interference_size) boost::ipc_atomic<bool>* oValid;
        alignas(hardware_destructive_interference_size) boost::ipc_atomic<unsigned int>* oCycle;
        alignas(hardware_destructive_interference_size) boost::ipc_atomic<unsigned int>* fifoOccupation;
        alignas(hardware_destructive_interference_size) boost::ipc_atomic<unsigned int>* maxFifoDepth;
    };
    _SimulationInterface interface;
    ipc::managed_shared_memory shmem;
    const std::string shmIdentifier;

#ifdef NDEBUG
    [[maybe_unused]] void simInterfaceDebug([[maybe_unused]] std::string_view s) {}
#else
    /// Log the given text with a header identifying the shared memory region and the interface type
    void simInterfaceDebug(std::string_view s) {
        debug(std::format("{} ({}): {}", shmIdentifier, to_string(T), s));
    }
#endif

    public:
    SimulationInterface(const char* _shmIdentifier, unsigned int initialMaxDepth = 2) : shmIdentifier(_shmIdentifier) {
        simInterfaceDebug(std::format("Creating simulation interface with {} depth.", initialMaxDepth));
        if (T == SimulationInterfaceType::PRODUCING) {
            ipc::shared_memory_object::remove(_shmIdentifier);
            simInterfaceDebug("Removed previous shared memory objects.");
            shmem = ipc::managed_shared_memory(ipc::create_only, _shmIdentifier, ShmemSize);
        } else {
            while(true) {
                try {
                    shmem = ipc::managed_shared_memory(ipc::open_only, _shmIdentifier);
                    break;
                } catch (const ipc::interprocess_exception& e) {
                    simInterfaceDebug("Producer shared memory not yet created. Waiting..");
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
        }
        simInterfaceDebug("Shared memory constructed or found.");

        // Find variables. CONSUMING interfaces only communicate with o... signals, PRODUCING ones only with i... interfaces
        interface.fifoOccupation = shmem.find_or_construct<boost::ipc_atomic<unsigned int>>("fifoOccupation")(0);
        interface.maxFifoDepth = shmem.find_or_construct<boost::ipc_atomic<unsigned int>>("maxFifoDepth")(initialMaxDepth);
        interface.iReady = shmem.find_or_construct<boost::ipc_atomic<bool>>("iReady")(initialMaxDepth > 0);
        interface.iCycle = shmem.find_or_construct<boost::ipc_atomic<unsigned int>>("iCycle")(0);
        interface.oValid = shmem.find_or_construct<boost::ipc_atomic<bool>>("oValid")(*(interface.fifoOccupation) > 9);
        interface.oCycle = shmem.find_or_construct<boost::ipc_atomic<unsigned int>>("oCycle")(0);
        simInterfaceDebug("Shared variables constructed or found.");

    }

    ~SimulationInterface() {
        // TODO: Called implicitly?
        shmem.destroy<boost::ipc_atomic<bool>>("iReady");
        shmem.destroy<boost::ipc_atomic<unsigned int>>("iCycle");
        shmem.destroy<boost::ipc_atomic<bool>>("oValid");
        shmem.destroy<boost::ipc_atomic<unsigned int>>("oCycle");
        shmem.destroy<boost::ipc_atomic<unsigned int>>("fifoOccupation");
        shmem.destroy<boost::ipc_atomic<unsigned int>>("maxFifoDepth");
    }

    /// Set the max fifo depth in this interface.
    void setMaxFifoDepth(unsigned int depth) {
        simInterfaceDebug(std::format("Setting max FIFO depth to {}", depth));
        *interface.maxFifoDepth = depth;
    }

    /// Reset all interface data fields to their defaults
    void reset(unsigned int maxFifoDepth = 2) {
        simInterfaceDebug(std::format("Resetting simulation interface (with max FIFO depth {})", maxFifoDepth));
        *interface.fifoOccupation = 0;
        *interface.maxFifoDepth = maxFifoDepth;
        *interface.iReady = true;
        *interface.iCycle = 0;
        *interface.oValid = *(interface.fifoOccupation) > 0;
        *interface.oCycle = 0;
    }

    /// Communicate with the interface from the consumer side. Pass in the consuming nodes' input_ready.
    /// If the interface has valid data, it will do the exchange.
    /// The function returns the interfaces (FIFOs) output_valid signal, which should be
    /// read by the consumer and set on their simulation port.
    bool communicate(bool consumerReady) requires (T == SimulationInterfaceType::CONSUMING) {
        // The input side must always be one cycle ahead of the output side
        // Wait until input catches up (and overtakes)
        while (*interface.iCycle <= *interface.oCycle) {}
        *interface.oValid = *interface.fifoOccupation > 0;
        if (*interface.oValid && consumerReady) {
            --(*interface.fifoOccupation);
        }
        ++(*interface.oCycle);
        simInterfaceDebug("Exchanged data with consumer.");
        return *interface.oValid;
    }

    /// Communicate with the interface from the producer side. Pass in the producing nodes' output_valid.
    /// If the interface is ready to receive data, it will do the exchange.
    /// The function returns the interfaces (FIFOs) input_ready signal, which should be
    /// read by the producer and set on their simulation port.
    bool communicate(bool producerValid) requires (T == SimulationInterfaceType::PRODUCING) {
        // The input side must always be one cycle ahead of the output side
        // Wait until output catches up
        while (*interface.oCycle != *interface.iCycle) {}
        *interface.iReady = *interface.fifoOccupation < *interface.maxFifoDepth;
        if (*interface.iReady && producerValid) {
            ++(*interface.fifoOccupation);
        }
        ++(*interface.iCycle);
        simInterfaceDebug("Exchanged data with producer.");
        return *interface.iReady;
    }
};


template<size_t IStreamsSize, size_t OStreamsSize, bool LoggingEnabled, size_t NodeIndex, size_t TotalNodes, bool CommunicatesWithPredecessor, bool CommunicatesWithSuccessor>
class SingleNodeSimulation : public Simulation<IStreamsSize, OStreamsSize, LoggingEnabled> {
    private:
    using ConsumingInterface = SimulationInterface<SimulationInterfaceType::CONSUMING>;
    using ProducingInterface = SimulationInterface<SimulationInterfaceType::PRODUCING>;
    std::array<std::unique_ptr<ConsumingInterface>, IStreamsSize> fromProducerInterface;
    std::array<std::unique_ptr<ProducingInterface>, OStreamsSize> toConsumerInterface;
    std::size_t cyclesRun = 0;

    public:
    SingleNodeSimulation(
        const std::string& kernel_lib,
        const std::string& design_lib,
        const char* xsim_log_file,
        const char* trace_file,
        std::array<StreamDescriptor, IStreamsSize> _istream_descs,
        std::array<StreamDescriptor, OStreamsSize> _ostream_descs,
        std::optional<std::string> prevNodeName = std::nullopt,
        std::optional<std::string> nodeName = std::nullopt
    ) :
    Simulation<IStreamsSize, OStreamsSize, LoggingEnabled>(kernel_lib, design_lib, xsim_log_file, trace_file, _istream_descs, _ostream_descs) {
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

        // Set valid and ready
        // TODO: Currently uses NodeIndex to check if we are input or output or neither
        // This is unstable because other nodes with neither last nor first index might
        // also be inputs or outputs.
        initStreams();

        // Create producer facing interfaces
        debug(std::format("Creating {} interfaces for communication with predecessors.", IStreamsSize));
        if (NodeIndex != 0 && prevNodeName && CommunicatesWithPredecessor) {
            for (std::size_t i = 0; i < IStreamsSize; ++i) {
                fromProducerInterface[i] = std::make_unique<ConsumingInterface>(std::format("{}_{}", *prevNodeName, i).c_str());
            }
        }

        // Create consumer facing interfaces
        debug(std::format("Creating {} interfaces for communication with successors.", OStreamsSize));
        if (NodeIndex != TotalNodes - 1 && nodeName && CommunicatesWithSuccessor) {
            for (std::size_t i = 0; i < OStreamsSize; ++i) {
                toConsumerInterface[i] = std::make_unique<ProducingInterface>(std::format("{}_{}", *nodeName, i).c_str());
            }
        }
        debug("Finished initializing simulation.\n------------------------------\n");
    }

    private:
    /// Communicate with predecessors and successors and update their values and our own
    void communicate() {
        if constexpr(NodeIndex != TotalNodes - 1 && CommunicatesWithSuccessor) {
            for (std::size_t i = 0; i < OStreamsSize; ++i) {
                this->ostreams[i].ready(toConsumerInterface[i]->communicate(this->ostreams[i].is_valid()));
            }
        }
        if constexpr(NodeIndex != 0 && CommunicatesWithPredecessor) {
            for (std::size_t i = 0; i < IStreamsSize; ++i) {
                this->istreams[i].valid(fromProducerInterface[i]->communicate(this->istreams[i].is_ready()));
            }
        }
    }

    public:
    /// Init streams according to nodeindex
    void initStreams() requires (NodeIndex == 0) { for (auto&& s : this->istreams) { s.valid(); } }
    void initStreams() requires (NodeIndex == TotalNodes - 1) { for (auto&& s : this->ostreams) { s.ready(); } }
    void initStreams() requires (NodeIndex > 0 && NodeIndex < TotalNodes - 1) { }

    /// Reset simulation (stream and current FIFO depth)
    void reset() {
        this->reset();
        for (std::size_t i = 0; i < OStreamsSize; ++i) {
            toConsumerInterface[i]->reset();
        }
        for (std::size_t i = 0; i < IStreamsSize; ++i) {
            fromProducerInterface[i]->reset();
        }
    }

    void runSingleCycle() {
        debug("Running single cycle.\n------------------");
        this->clk.toggle_clk();
        communicate();
        if constexpr(LoggingEnabled) {
            ++cyclesRun;
            debug(std::format("Finished cycle {}\n\n", cyclesRun));
        }

        // Log the signals that this simulations set (ready to predecessor, valid to successor)
        // TODO: Collect signals in vectors and only write to file after the sim for speedup
        if constexpr(LoggingEnabled) {
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

};

#endif /* SIMULATION */
