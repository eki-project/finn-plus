#ifndef SIMULATION_INTERFACE
#define SIMULATION_INTERFACE
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
#include <cstdlib>
#include <format>
#include <new>
#include <string_view>

#ifdef __cpp_lib_hardware_interference_size
using std::hardware_destructive_interference_size;
#else
constexpr std::size_t hardware_destructive_interference_size = 64;
#endif

namespace ipc = boost::interprocess;

enum class SimulationInterfaceType { PRODUCING, CONSUMING };
constexpr std::string_view to_string(SimulationInterfaceType t) {
    if (t == SimulationInterfaceType::CONSUMING) {
        return "CONSUMING";
    } else if (t == SimulationInterfaceType::PRODUCING) {
        return "PRODUCING";
    }
    return "UNKNOWN SIMULATION INTERFACE TYPE";
}

template<SimulationInterfaceType T, bool IsIOInterface, std::size_t ShmemSize = 2048>
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
    std::atomic<std::size_t> largestOccupation;

#ifdef NDEBUG
    [[maybe_unused]] void simInterfaceDebug([[maybe_unused]] std::string_view s) {}
#else
    /// Log the given text with a header identifying the shared memory region and the interface type
    void simInterfaceDebug(std::string_view s) { debug(std::format("log {} ({}): {}", shmIdentifier, to_string(T), s)); }
#endif

     public:
    // Default constructor needed for std::array
    SimulationInterface() : shmIdentifier("") {
        // Uninitialized - will be move-assigned later
    }

    SimulationInterface(const char* _shmIdentifier, unsigned int initialMaxDepth = 2) : shmIdentifier(_shmIdentifier), largestOccupation(0) {
        simInterfaceDebug(std::format("Creating simulation interface with {} depth.", initialMaxDepth));
        if (T == SimulationInterfaceType::PRODUCING || IsIOInterface) {
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

    /// Return the largest occupation that this FIFO has had so far
    std::size_t getLargestOccupation() {
        return largestOccupation;
    }

    /// Set the max fifo depth in this interface.
    void setMaxFifoDepth(unsigned int depth) {
        sharedData->maxFifoDepth.store(depth, boost::memory_order_release);
    }

    /// Reset all interface data fields to their defaults
    void reset(unsigned int newMaxFifoDepth = 2) {
        simInterfaceDebug(std::format("Resetting simulation interface (with max FIFO depth {})", newMaxFifoDepth));
        largestOccupation = 0;
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
        if (sharedData->iReady && producerValid) {
            ++largestOccupation;
        }
        ++(sharedData->iCycle);
        return sharedData->iReady;
    }
};
#endif
