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
        alignas(hardware_destructive_interference_size) boost::ipc_atomic<unsigned int> predCycle;
        alignas(hardware_destructive_interference_size) boost::ipc_atomic<unsigned int> succCycle;
        alignas(hardware_destructive_interference_size) boost::ipc_atomic<bool> ready;
        alignas(hardware_destructive_interference_size) boost::ipc_atomic<bool> valid;

        SharedData() : predCycle(0), succCycle(0), ready(false), valid(false) {}
        SharedData(unsigned int predecessorCycle, unsigned int successorCycle, bool inReady, bool outValid) : predCycle(predecessorCycle), succCycle(successorCycle), ready(inReady), valid(outValid) {}
        SharedData(const SharedData& other) : predCycle(other.predCycle.load()), succCycle(other.succCycle.load()), ready(other.ready.load()), valid(other.valid.load()) {}
        SharedData& operator=(const SharedData& other) {
            predCycle.store(other.predCycle.load());
            succCycle.store(other.succCycle.load());
            ready.store(other.ready.load());
            valid.store(other.valid.load());
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
        sharedData = shmem.find_or_construct<SharedData>("data")(SharedData(0, 0, true, false));
        simInterfaceDebug("Shared data structure constructed or found.");
    }

    // Delete copy operations
    SimulationInterface(const SimulationInterface&) = delete;
    SimulationInterface& operator=(const SimulationInterface&) = delete;

    // Move constructor
    SimulationInterface(SimulationInterface&& other) noexcept : sharedData(other.sharedData), refCount(other.refCount), shmIdentifier(std::move(other.shmIdentifier)), shmem(std::move(other.shmem)) {
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
    void reset() {
        simInterfaceDebug("Resetting simulation interface");
        sharedData->ready.store(true, boost::memory_order_release);
        sharedData->predCycle.store(0, boost::memory_order_release);
        sharedData->valid.store(false, boost::memory_order_release);
        sharedData->succCycle.store(0, boost::memory_order_release);
    }

    /**
     * Reads valid from shm and puts ready into shm.
     * Returns the valid signal read from shm.
     * Called on consumer side.
     */
    bool readFromLastNode(bool consumerReady)
        requires(T == SimulationInterfaceType::CONSUMING)
    {
        // The predecessor must always be one cycle ahead of the successor side
        // Wait until predecessor catches up (and overtakes)
        while (sharedData->predCycle <= sharedData->succCycle) {}
        sharedData->ready = consumerReady;
        ++(sharedData->succCycle);
        return sharedData->valid;
    }

    /**
     * Reads ready from shm and puts valid into shm.
     * Returns the ready signal read from shm.
     * Called on producer side.
     */
    bool writeToNextNode(bool producerValid)
        requires(T == SimulationInterfaceType::PRODUCING)
    {
        // The predecessor side must always be at least one cycle ahead of the output side
        // Wait until output catches up
        while (sharedData->succCycle < sharedData->predCycle) {}
        sharedData->valid = producerValid;
        ++(sharedData->predCycle);
        return sharedData->ready;
    }
};
#endif
