#ifndef INTERPROCESSCOMMUNICATIONCHANNEL
#define INTERPROCESSCOMMUNICATIONCHANNEL

#include <atomic>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <new>
#include <thread>
#include <iostream>

#ifndef CACHE_LINE_SIZE
    #ifdef __cpp_lib_hardware_interference_size
constexpr std::size_t CACHE_LINE_SIZE = std::hardware_destructive_interference_size;
    #else
constexpr std::size_t CACHE_LINE_SIZE = 64;
    #endif
#endif

namespace bip = boost::interprocess;

// ===== INTERPROCESS ASYMMETRIC REQUEST-RESPONSE EXCHANGE =====
// Concepts for constraining methods based on role
template<bool IsSender>
concept Sender = IsSender;

constexpr int MAX_SPIN_WAIT = 100;

template<typename Request, typename Response, bool IsSender, std::size_t SharedMemorySize = 4096>
class InterprocessCommunicationChannel {
     private:
    // ===== SHARED MEMORY STRUCTURE =====
    struct alignas(CACHE_LINE_SIZE) SharedChannelData {
        struct alignas(CACHE_LINE_SIZE) RequestSlot {
            Request data;
            std::atomic<bool> valid;

            RequestSlot() : data(), valid(false) {}
        };

        struct alignas(CACHE_LINE_SIZE) ResponseSlot {
            Response data;
            std::atomic<bool> valid;

            ResponseSlot() : data(), valid(false) {}
        };

        // Double-buffered requests and responses
        RequestSlot requests[2];
        ResponseSlot responses[2];

        alignas(CACHE_LINE_SIZE) std::atomic<int> request_write_idx;
        alignas(CACHE_LINE_SIZE) std::atomic<int> request_read_idx;
        alignas(CACHE_LINE_SIZE) std::atomic<int> response_write_idx;
        alignas(CACHE_LINE_SIZE) std::atomic<int> response_read_idx;

        SharedChannelData() : request_write_idx(0), request_read_idx(0), response_write_idx(0), response_read_idx(0) {
            // Verify atomics are lock-free (required for shared memory)
            static_assert(std::atomic<bool>::is_always_lock_free, "std::atomic<bool> must be lock-free for inter-process use");
            static_assert(std::atomic<int>::is_always_lock_free, "std::atomic<int> must be lock-free for inter-process use");
        }
    };

    // ===== PROCESS-LOCAL STATE =====
    SharedChannelData* channel = nullptr;
    std::atomic<int>* refCount = nullptr;
    const std::string sharedMemoryName;
    bip::managed_shared_memory shmem;

     public:
    // Default constructor
    InterprocessCommunicationChannel() : sharedMemoryName("") {}

    // Constructor with shared memory name
    InterprocessCommunicationChannel(const std::string& shmName) : sharedMemoryName(shmName) {
        if constexpr (IsSender) {
            // Sender creates shared memory
            bip::shared_memory_object::remove(sharedMemoryName.c_str());
            shmem = bip::managed_shared_memory(bip::create_only, sharedMemoryName.c_str(), SharedMemorySize);
            std::cout << "Created shared memory: " << sharedMemoryName << std::endl;
        } else {
            // Receiver opens existing shared memory
            std::cout << "Waiting to connect to shared memory: " << sharedMemoryName << std::endl;
            while (true) {
                try {
                    shmem = bip::managed_shared_memory(bip::open_only, sharedMemoryName.c_str());
                    break;
                } catch (const bip::interprocess_exception& e) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
            }
            std::cout << "Connected to shared memory: " << sharedMemoryName << std::endl;
        }

        // Construct or find the reference counter
        refCount = shmem.find_or_construct<std::atomic<int>>("refCount")(0);
        refCount->fetch_add(1, std::memory_order_acq_rel);

        // Construct the channel data in shared memory
        channel = shmem.find_or_construct<SharedChannelData>("ChannelData")();

    }

    void handshake() {
        // Perform handshake to verify communication works
        if constexpr (IsSender) {
            // Sender: send test request and wait for response
            std::cout << "Sending handshake test request for " << sharedMemoryName << std::endl;
            Request test_request{};
            Response test_response = send_request(test_request);
            std::cout << "Received handshake test response for " << sharedMemoryName << std::endl;
            // Communication verified if we got here without hanging
        } else {
            // Receiver: wait for test request and send response
            std::cout << "Waiting for handshake test request for " << sharedMemoryName << std::endl;
            Request test_request = receive_request();
            std::cout << "Received handshake test request for " << sharedMemoryName << std::endl;
            Response test_response{};
            send_response(test_response);
            std::cout << "Sent handshake test response for " << sharedMemoryName << std::endl;
            // Communication verified if we got here
        }
    }

    // Delete copy operations
    InterprocessCommunicationChannel(const InterprocessCommunicationChannel&) = delete;
    InterprocessCommunicationChannel& operator=(const InterprocessCommunicationChannel&) = delete;

    // Move constructor
    InterprocessCommunicationChannel(InterprocessCommunicationChannel&& other) noexcept
        : channel(other.channel), refCount(other.refCount), sharedMemoryName(std::move(other.sharedMemoryName)), shmem(std::move(other.shmem)) {
        other.channel = nullptr;
        other.refCount = nullptr;
    }

    // Move assignment operator
    InterprocessCommunicationChannel& operator=(InterprocessCommunicationChannel&& other) noexcept {
        if (this != &other) {
            channel = other.channel;
            refCount = other.refCount;
            shmem.swap(other.shmem);
            const_cast<std::string&>(sharedMemoryName) = std::move(other.sharedMemoryName);

            other.channel = nullptr;
            other.refCount = nullptr;
        }
        return *this;
    }

    ~InterprocessCommunicationChannel() {
        if (!refCount || !channel) {
            return;
        }

        channel = nullptr;
        refCount = nullptr;

        std::atomic<int>* ref_ptr = shmem.find<std::atomic<int>>("refCount").first;
        if (!ref_ptr) {
            return;
        }

        int remainingRefs = ref_ptr->fetch_sub(1, std::memory_order_acq_rel) - 1;

        if (remainingRefs == 0) {
            shmem.destroy<SharedChannelData>("ChannelData");
            shmem.destroy<std::atomic<int>>("refCount");
            shmem = bip::managed_shared_memory();
            bip::shared_memory_object::remove(sharedMemoryName.c_str());
        }
    }

    // SENDER SIDE: Send request, wait for response
    Response send_request(const Request& req, std::stop_token stoken = {})
        requires Sender<IsSender>
    {
        // Write request
        int write_slot = channel->request_write_idx.load(std::memory_order_acquire) % 2;
        channel->requests[write_slot].data = req;
        channel->requests[write_slot].valid.store(true, std::memory_order_release);
        channel->request_write_idx.fetch_add(1, std::memory_order_release);

        // Wait for response in corresponding slot
        int read_slot = channel->response_read_idx.load(std::memory_order_acquire) % 2;
        int spin_count = 0;
        while (!channel->responses[read_slot].valid.load(std::memory_order_acquire) && !stoken.stop_requested()) {
            if (spin_count++ >= MAX_SPIN_WAIT) {
                std::this_thread::yield();
                spin_count = 0;
            } else {
#if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
#elif defined(__aarch64__)
                asm volatile("yield" ::: "memory");
#endif
            }
        }

        if (stoken.stop_requested()) {
            return Response{};  // Return default-constructed response on cancellation
        }

        Response resp = channel->responses[read_slot].data;
        channel->responses[read_slot].valid.store(false, std::memory_order_release);
        channel->response_read_idx.fetch_add(1, std::memory_order_release);

        return resp;
    }

    // RECEIVER SIDE: Wait for request, send response
    Request receive_request(std::stop_token stoken = {})
        requires(!Sender<IsSender>)
    {
        int read_slot = channel->request_read_idx.load(std::memory_order_acquire) % 2;
        int spin_count = 0;

        while (!channel->requests[read_slot].valid.load(std::memory_order_acquire) && !stoken.stop_requested()) {
            if (spin_count++ >= MAX_SPIN_WAIT) {
                std::this_thread::yield();
                spin_count = 0;
            } else {
#if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
#elif defined(__aarch64__)
                asm volatile("yield" ::: "memory");
#endif
            }
        }

        if (stoken.stop_requested()) {
            return Request{};  // Return default-constructed request on cancellation
        }

        Request req = channel->requests[read_slot].data;
        channel->requests[read_slot].valid.store(false, std::memory_order_release);
        channel->request_read_idx.fetch_add(1, std::memory_order_release);

        return req;
    }

    void send_response(const Response& resp)
        requires(!Sender<IsSender>)
    {
        int write_slot = channel->response_write_idx.load(std::memory_order_acquire) % 2;
        channel->responses[write_slot].data = resp;
        channel->responses[write_slot].valid.store(true, std::memory_order_release);
        channel->response_write_idx.fetch_add(1, std::memory_order_release);
    }
};

#endif /* INTERPROCESSCOMMUNICATIONCHANNEL */
