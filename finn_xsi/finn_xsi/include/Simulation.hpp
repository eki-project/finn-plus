#ifndef SIMULATION
#define SIMULATION
#include <AXIS_Control.h>
#include <Clock.h>
#include <Design.h>
#include <Kernel.h>
#include <Port.h>
#include <SharedLibrary.h>
#include <helper.h>

#include <boost/asio.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/atomic/ipc_atomic.hpp>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/detail/os_file_functions.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>
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

template<SimulationInterfaceType T, std::size_t ShmemSize = 2048>
class SimulationInterface {
     private:
    // Shared memory structure with proper cache-line alignment
    struct SharedData {
        alignas(hardware_destructive_interference_size) boost::ipc_atomic<unsigned int> fifoOccupation;
        alignas(hardware_destructive_interference_size) boost::ipc_atomic<unsigned int> maxFifoDepth;
        alignas(hardware_destructive_interference_size) boost::ipc_atomic<unsigned int> iCycle;
        alignas(hardware_destructive_interference_size) boost::ipc_atomic<unsigned int> oCycle;
        alignas(hardware_destructive_interference_size) boost::ipc_atomic<bool> iReady;
        alignas(hardware_destructive_interference_size) boost::ipc_atomic<bool> oValid;

        SharedData() : fifoOccupation(0), maxFifoDepth(0), iCycle(0), oCycle(0), iReady(false), oValid(false) {}
        SharedData(unsigned int fifoOcc, unsigned int maxDepth, unsigned int inCycle, unsigned int outCycle, bool inReady, bool outValid)
            : fifoOccupation(fifoOcc), maxFifoDepth(maxDepth), iCycle(inCycle), oCycle(outCycle), iReady(inReady), oValid(outValid) {}
        SharedData(const SharedData& other)
            : fifoOccupation(other.fifoOccupation.load()),
              maxFifoDepth(other.maxFifoDepth.load()),
              iCycle(other.iCycle.load()),
              oCycle(other.oCycle.load()),
              iReady(other.iReady.load()),
              oValid(other.oValid.load()) {}
        SharedData& operator=(const SharedData& other) {
            fifoOccupation.store(other.fifoOccupation.load());
            maxFifoDepth.store(other.maxFifoDepth.load());
            iCycle.store(other.iCycle.load());
            oCycle.store(other.oCycle.load());
            iReady.store(other.iReady.load());
            oValid.store(other.oValid.load());
            return *this;
        }
    };

    SharedData* sharedData = nullptr;
    boost::ipc_atomic<int>* refCount = nullptr;
    ipc::managed_shared_memory shmem;
    const std::string shmIdentifier;

#ifdef NDEBUG
    [[maybe_unused]] void simInterfaceDebug([[maybe_unused]] std::string_view s) {}
#else
    /// Log the given text with a header identifying the shared memory region and the interface type
    void simInterfaceDebug(std::string_view s) { debug(std::format("{} ({}): {}", shmIdentifier, to_string(T), s)); }
#endif

     public:
    // Default constructor needed for std::array
    SimulationInterface() : shmIdentifier("") {
        // Uninitialized - will be move-assigned later
    }

    SimulationInterface(const char* _shmIdentifier, unsigned int initialMaxDepth = 2) : shmIdentifier(_shmIdentifier) {
        simInterfaceDebug(std::format("Creating simulation interface with {} depth.", initialMaxDepth));
        if (T == SimulationInterfaceType::PRODUCING) {
            ipc::shared_memory_object::remove(_shmIdentifier);
            simInterfaceDebug("Removed previous shared memory objects.");
            shmem = ipc::managed_shared_memory(ipc::create_only, _shmIdentifier, ShmemSize);
        } else {
            while (true) {
                try {
                    shmem = ipc::managed_shared_memory(ipc::open_only, _shmIdentifier);
                    break;
                } catch (const ipc::interprocess_exception& e) { simInterfaceDebug("Producer shared memory not yet created. Waiting.."); }
            }
        }
        simInterfaceDebug("Shared memory constructed or found.");

        // Construct or find the reference counter (separate from SharedData)
        refCount = shmem.find_or_construct<boost::ipc_atomic<int>>("refCount")(0);

        // Increment reference count atomically
        int currentRefCount = refCount->fetch_add(1, boost::memory_order_acq_rel) + 1;
        simInterfaceDebug(std::format("Reference count incremented to {}", currentRefCount));

        // Construct or find the entire SharedData struct in shared memory
        sharedData = shmem.find_or_construct<SharedData>("data")(SharedData(0, initialMaxDepth, 0, 0, initialMaxDepth > 0, false));
        simInterfaceDebug("Shared data structure constructed or found.");
    }

    // Delete copy operations
    SimulationInterface(const SimulationInterface&) = delete;
    SimulationInterface& operator=(const SimulationInterface&) = delete;

    // Move constructor
    SimulationInterface(SimulationInterface&& other) noexcept
        : sharedData(other.sharedData), refCount(other.refCount), shmem(std::move(other.shmem)), shmIdentifier(std::move(other.shmIdentifier)) {
        // Mark other as moved-from
        other.sharedData = nullptr;
        other.refCount = nullptr;
    }

    // Move assignment operator
    SimulationInterface& operator=(SimulationInterface&& other) noexcept {
        if (this != &other) {
            sharedData = other.sharedData;
            refCount = other.refCount;
            // Note: managed_shared_memory has deleted assignment, use swap
            shmem.swap(other.shmem);
            const_cast<std::string&>(shmIdentifier) = std::move(other.shmIdentifier);

            // Mark other as moved-from
            other.sharedData = nullptr;
            other.refCount = nullptr;
        }
        return *this;
    }

    ~SimulationInterface() {
        // Skip cleanup if moved-from or default-constructed
        if (!refCount || !sharedData) {
            return;
        }

        // Decrement reference count atomically
        int remainingRefs = refCount->fetch_sub(1, boost::memory_order_acq_rel) - 1;
        simInterfaceDebug(std::format("Reference count decremented to {}", remainingRefs));

        // If we're the last process, clean up the shared memory
        if (remainingRefs == 0) {
            simInterfaceDebug("Last process exiting - cleaning up shared memory");
            shmem.destroy<SharedData>("data");
            shmem.destroy<boost::ipc_atomic<int>>("refCount");

            // Remove the shared memory segment completely
            ipc::shared_memory_object::remove(shmIdentifier.c_str());
            simInterfaceDebug("Shared memory cleaned up successfully");
        } else {
            simInterfaceDebug("Other processes still using shared memory - detaching only");
        }
    }

    /// Set the max fifo depth in this interface.
    void setMaxFifoDepth(unsigned int depth) {
        simInterfaceDebug(std::format("Setting max FIFO depth to {}", depth));
        sharedData->maxFifoDepth.store(depth, boost::memory_order_release);
    }

    /// Reset all interface data fields to their defaults
    void reset(unsigned int newMaxFifoDepth = 2) {
        simInterfaceDebug(std::format("Resetting simulation interface (with max FIFO depth {})", newMaxFifoDepth));
        sharedData->fifoOccupation.store(0, boost::memory_order_release);
        sharedData->maxFifoDepth.store(newMaxFifoDepth, boost::memory_order_release);
        sharedData->iReady.store(true, boost::memory_order_release);
        sharedData->iCycle.store(0, boost::memory_order_release);
        sharedData->oValid.store(false, boost::memory_order_release);
        sharedData->oCycle.store(0, boost::memory_order_release);
    }

    /// Communicate with the interface from the consumer side. Pass in the consuming nodes' input_ready.
    /// If the interface has valid data, it will do the exchange.
    /// The function returns the interfaces (FIFOs) output_valid signal, which should be
    /// read by the consumer and set on their simulation port.
    bool communicate(bool consumerReady)
        requires(T == SimulationInterfaceType::CONSUMING)
    {
        // The input side must always be one cycle ahead of the output side
        // Wait until input catches up (and overtakes)
        while (sharedData->iCycle <= sharedData->oCycle) {}
        sharedData->oValid = sharedData->fifoOccupation > 0;
        sharedData->fifoOccupation -= static_cast<unsigned int>(sharedData->oValid && consumerReady);
        ++(sharedData->oCycle);
        return sharedData->oValid;
    }

    /// Communicate with the interface from the producer side. Pass in the producing nodes' output_valid.
    /// If the interface is ready to receive data, it will do the exchange.
    /// The function returns the interfaces (FIFOs) input_ready signal, which should be
    /// read by the producer and set on their simulation port.
    bool communicate(bool producerValid)
        requires(T == SimulationInterfaceType::PRODUCING)
    {
        // The input side must always be one cycle ahead of the output side
        // Wait until output catches up
        while (sharedData->oCycle != sharedData->iCycle) {}
        sharedData->iReady = sharedData->fifoOccupation < sharedData->maxFifoDepth;
        sharedData->fifoOccupation += static_cast<unsigned int>(sharedData->iReady && producerValid);
        ++(sharedData->iCycle);
        return sharedData->iReady;
    }
};


template<size_t IStreamsSize, size_t OStreamsSize, bool LoggingEnabled, size_t NodeIndex, size_t TotalNodes, bool CommunicatesWithPredecessor, bool CommunicatesWithSuccessor>
class SingleNodeSimulation : public Simulation<IStreamsSize, OStreamsSize, LoggingEnabled> {
     private:
    using ConsumingInterface = SimulationInterface<SimulationInterfaceType::CONSUMING>;
    using ProducingInterface = SimulationInterface<SimulationInterfaceType::PRODUCING>;
    std::array<ConsumingInterface, IStreamsSize> fromProducerInterface;
    std::array<ProducingInterface, OStreamsSize> toConsumerInterface;
    std::size_t cyclesRun = 0;

     public:
    SingleNodeSimulation(const std::string& kernel_lib, const std::string& design_lib, const char* xsim_log_file, const char* trace_file, std::array<StreamDescriptor, IStreamsSize> _istream_descs,
                         std::array<StreamDescriptor, OStreamsSize> _ostream_descs, std::optional<std::string> prevNodeName = std::nullopt, std::optional<std::string> nodeName = std::nullopt)
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

        // Set valid and ready
        // TODO: Currently uses NodeIndex to check if we are input or output or neither
        // This is unstable because other nodes with neither last nor first index might
        // also be inputs or outputs.
        initStreams();

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

        debug("Finished initializing simulation.\n------------------------------\n");
    }

     private:
    /// Communicate with predecessors and successors and update their values and our own
    [[gnu::hot]] void communicate() {
        if constexpr (NodeIndex != TotalNodes - 1 && CommunicatesWithSuccessor) {
            for (std::size_t i = 0; i < OStreamsSize; ++i) {
                this->ostreams[i].ready(toConsumerInterface[i].communicate(this->ostreams[i].is_valid()));
            }
        }
        if constexpr (NodeIndex != 0 && CommunicatesWithPredecessor) {
            for (std::size_t i = 0; i < IStreamsSize; ++i) {
                this->istreams[i].valid(fromProducerInterface[i].communicate(this->istreams[i].is_ready()));
            }
        }
    }

     public:
    /// Init streams according to nodeindex
    void initStreams() {
        if constexpr (NodeIndex == 0) {
            for (auto&& s : this->istreams) {
                s.valid();
            }
        } else if constexpr (NodeIndex == TotalNodes - 1) {
            for (auto&& s : this->ostreams) {
                s.ready();
            }
        }
        // Middle nodes don't initialize any streams
    }

    /// Reset simulation (stream and current FIFO depth)
    void reset() {
        Simulation<IStreamsSize, OStreamsSize, LoggingEnabled>::reset();
        for (std::size_t i = 0; i < OStreamsSize; ++i) {
            toConsumerInterface[i].reset();
        }
        for (std::size_t i = 0; i < IStreamsSize; ++i) {
            fromProducerInterface[i].reset();
        }
    }

    [[gnu::hot]] void runSingleCycle() {
        this->clk.toggle_clk();
        communicate();
        if constexpr (LoggingEnabled) {
            ++cyclesRun;
            debug(std::format("Finished cycle {}\n\n", cyclesRun));
        }

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
};

#endif /* SIMULATION */
