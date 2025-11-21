#ifndef INTERSIMULATIONINTERFACE
#define INTERSIMULATIONINTERFACE

#include <atomic>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <new>
#include <thread>

#ifdef __cpp_lib_hardware_interference_size
constexpr std::size_t CACHE_LINE_SIZE = std::hardware_destructive_interference_size;
#else
constexpr std::size_t CACHE_LINE_SIZE = 64;
#endif

namespace bip = boost::interprocess;

template<bool Receiver, std::size_t SharedMemorySize = 2048>
class InterSimulationInterface {
     private:
    // ===== SHARED MEMORY STRUCTURE =====
    // This goes into shared memory and is accessible from both processes
    struct alignas(CACHE_LINE_SIZE) SharedHaloExchange {
        struct alignas(CACHE_LINE_SIZE) BufferSlot {
            std::atomic<bool> value;
            std::atomic<bool> ready;  // Ready flag for Halo Exchange NOT Simulation

            // Must explicitly initialize atomics in shared memory
            BufferSlot() : value(false), ready(false) {}
        };

        // Linearized buffers: [process_id * 2 + buffer_id]
        BufferSlot buffers[4];
        alignas(CACHE_LINE_SIZE) std::atomic<int> current_buffer;
        alignas(CACHE_LINE_SIZE) std::atomic<int> flip_barrier;

        SharedHaloExchange() : current_buffer(0), flip_barrier(0) {
            // Verify atomics are lock-free (required for shared memory)
            static_assert(std::atomic<bool>::is_always_lock_free, "std::atomic<bool> must be lock-free for inter-process use");
            static_assert(std::atomic<int>::is_always_lock_free, "std::atomic<int> must be lock-free for inter-process use");
        }

        static constexpr int idx(int proc_id, int buf_id) { return proc_id * 2 + buf_id; }
    };

    // ===== PROCESS-LOCAL STATE =====
    // This is NOT in shared memory - each process has its own copy
    template<bool IsReceiver>
    struct ProcessLocalState {
        int expected_buf;
        bool first_call;
        constexpr static int process_id = IsReceiver ? 1 : 0;  // 0 or 1
        ProcessLocalState() : expected_buf(0), first_call(true) {}
    };

    SharedHaloExchange* halo = nullptr;
    std::atomic<int>* refCount = nullptr;
    const std::string sharedMemoryName;
    bip::managed_shared_memory shmem;
    // Create process-local state
    ProcessLocalState<Receiver> local;

     public:
    // Default constructor needed for std::array
    InterSimulationInterface() : sharedMemoryName("") {
        // Uninitialized - will be move-assigned later
    }

    InterSimulationInterface(const std::string& shmName) : sharedMemoryName(shmName) {
        if constexpr (Receiver) {
            bip::shared_memory_object::remove(sharedMemoryName.c_str());
            shmem = bip::managed_shared_memory(bip::create_only, sharedMemoryName.c_str(), SharedMemorySize);
        } else {
            while (true) {
                try {
                    shmem = bip::managed_shared_memory(bip::open_only, sharedMemoryName.c_str());
                    break;
                } catch (const bip::interprocess_exception& e) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
            }
        }

        // Construct or find the reference counter (separate from SharedData)
        refCount = shmem.find_or_construct<std::atomic<int>>("refCount")(0);

        // Increment reference count atomically
        refCount->fetch_add(1, std::memory_order_acq_rel);

        // Construct the halo exchange object in shared memory
        halo = shmem.find_or_construct<SharedHaloExchange>("HaloExchange")();
    }

    // Delete copy operations
    InterSimulationInterface(const InterSimulationInterface&) = delete;
    InterSimulationInterface& operator=(const InterSimulationInterface&) = delete;

    // Move constructor
    InterSimulationInterface(InterSimulationInterface&& other) noexcept : halo(other.halo), refCount(other.refCount), sharedMemoryName(std::move(other.sharedMemoryName)), shmem(std::move(other.shmem)) {
        // Mark other as moved-from
        other.halo = nullptr;
        other.refCount = nullptr;
    }

    // Move assignment operator
    InterSimulationInterface& operator=(InterSimulationInterface&& other) noexcept {
        if (this != &other) {
            halo = other.halo;
            refCount = other.refCount;
            // Note: managed_shared_memory has deleted assignment, use swap
            shmem.swap(other.shmem);
            const_cast<std::string&>(sharedMemoryName) = std::move(other.sharedMemoryName);

            // Mark other as moved-from
            other.halo = nullptr;
            other.refCount = nullptr;
        }
        return *this;
    }

    ~InterSimulationInterface() {
        // Skip cleanup if moved-from or default-constructed
        if (!refCount || !halo) {
            return;
        }

        // Clear our local pointers before decrementing (safety)
        halo = nullptr;
        refCount = nullptr;

        // Get a raw pointer to refCount for the atomic operation
        // (we need this because we just nulled our member pointer)
        std::atomic<int>* ref_ptr = shmem.find<std::atomic<int>>("refCount").first;
        if (!ref_ptr) {
            return;  // Already destroyed somehow
        }

        // Decrement reference count atomically and get the value BEFORE decrement
        int remainingRefs = ref_ptr->fetch_sub(1, std::memory_order_acq_rel) - 1;

        // If we're the last process, clean up the shared memory
        if (remainingRefs == 0) {
            // Destroy all objects first
            shmem.destroy<SharedHaloExchange>("HaloExchange");
            shmem.destroy<std::atomic<int>>("refCount");

            // Close our handle to the shared memory
            // This doesn't delete it yet if other processes have it mapped
            shmem = bip::managed_shared_memory();

            // Now remove the shared memory segment completely
            // This is safe even if other processes still have stale mappings
            bip::shared_memory_object::remove(sharedMemoryName.c_str());
        }
    }

    // ===== EXCHANGE FUNCTION =====
    // This function runs in each process
    bool exchange(bool send_value) {
        constexpr int neighbor_id = 1 - this->local.process_id;

        // Wait for previous buffer flip (latency hiding)
        if (!local.first_call) {
            while (this->halo->current_buffer.load(std::memory_order_acquire) % 2 == local.expected_buf) {
#if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
#endif
            }
        }
        local.first_call = false;

        int buf_id = this->halo->current_buffer.load(std::memory_order_acquire) % 2;

        // Write our data
        int my_idx = SharedHaloExchange::idx(this->local.process_id, buf_id);
        this->halo->buffers[my_idx].value.store(send_value, std::memory_order_release);
        this->halo->buffers[my_idx].ready.store(true, std::memory_order_release);

        // Wait for neighbor
        int neighbor_idx = SharedHaloExchange::idx(neighbor_id, buf_id);
        bool neighbor_ready = this->halo->buffers[neighbor_idx].ready.load(std::memory_order_acquire);
        if (!neighbor_ready) {
            while (!this->halo->buffers[neighbor_idx].ready.load(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
#endif
            }
        }

        bool received = this->halo->buffers[neighbor_idx].value.load(std::memory_order_acquire);

        // Flip barrier
        if (this->halo->flip_barrier.fetch_add(1, std::memory_order_acq_rel) == 1) {
            // First process: flip the buffer
            int old_buf = buf_id;
            this->halo->buffers[SharedHaloExchange::idx(0, old_buf)].ready.store(false, std::memory_order_relaxed);
            this->halo->buffers[SharedHaloExchange::idx(1, old_buf)].ready.store(false, std::memory_order_relaxed);
            this->halo->current_buffer.fetch_add(1, std::memory_order_release);
            this->halo->flip_barrier.store(0, std::memory_order_release);
        }

        local.expected_buf = buf_id;
        return received;
    }
};
#endif /* INTERSIMULATIONINTERFACE */
