#ifndef MPICOMMUNICATIONCHANNEL
#define MPICOMMUNICATIONCHANNEL

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>

#ifdef FINN_XSI_ENABLE_MPI
    #include <mpi.h>
#endif

template<typename Request, typename Response, bool IsSender>
class MPICommunicationChannel {
     private:
    int peerRank = -1;
    int requestTag = -1;
    int responseTag = -1;

    static int _tag_from_name(std::string_view name, std::uint32_t salt, int tag_upper_bound) {
        // 32-bit FNV-1a hash, then map into a conservative MPI tag range.
        std::uint32_t h = 2166136261u ^ salt;
        for (char c : name) {
            h ^= static_cast<std::uint8_t>(c);
            h *= 16777619u;
        }

        // Map hash into runtime MPI tag range to avoid invalid tags on
        // implementations with small MPI_TAG_UB.
        const std::uint32_t range = static_cast<std::uint32_t>(tag_upper_bound + 1);
        return static_cast<int>(h % range);
    }

    static void _check_mpi_ready() {
#ifdef FINN_XSI_ENABLE_MPI
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized == 0) {
            throw std::runtime_error("MPI channel used before MPI_Init");
        }
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (finalized != 0) {
            throw std::runtime_error("MPI channel used after MPI_Finalize");
        }
#else
        throw std::runtime_error(
            "MPI channel requested but FINN XSI was built without MPI support. "
            "Rebuild with -DFIFOSIM_ENABLE_MPI=ON.");
#endif
    }

#ifdef FINN_XSI_ENABLE_MPI
    static int _get_mpi_tag_upper_bound() {
        int tag_ub = 32767;
        int* attr_ptr = nullptr;
        int flag = 0;
        MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &attr_ptr, &flag);
        if (flag != 0 && attr_ptr != nullptr) {
            tag_ub = *attr_ptr;
        }
        if (tag_ub < 1) {
            tag_ub = 1;
        }
        return tag_ub;
    }

    static constexpr int MAX_SPIN_WAIT = 256;

    static void _pause_or_yield(int& spin_count) {
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

    static bool _wait_request_complete(MPI_Request& request, std::stop_token stoken, bool allow_cancel_on_stop) {
        int done = 0;
        int spin_count = 0;
        while (done == 0) {
            MPI_Test(&request, &done, MPI_STATUS_IGNORE);
            if (done != 0) {
                return true;
            }

            if (allow_cancel_on_stop && stoken.stop_requested()) {
                MPI_Cancel(&request);
                // Drive cancellation/completion to completion so the request is not leaked.
                while (done == 0) {
                    MPI_Test(&request, &done, MPI_STATUS_IGNORE);
                    if (done != 0) {
                        return false;
                    }
                    _pause_or_yield(spin_count);
                }
                return false;
            }

            _pause_or_yield(spin_count);
        }
        return true;
    }

    template<typename T>
    static bool _isend_and_wait(const T& payload, int dst_rank, int tag, std::stop_token stoken, bool allow_cancel_on_stop) {
        MPI_Request send_req = MPI_REQUEST_NULL;
        MPI_Isend(reinterpret_cast<const void*>(&payload), static_cast<int>(sizeof(T)), MPI_BYTE, dst_rank, tag, MPI_COMM_WORLD, &send_req);
        return _wait_request_complete(send_req, stoken, allow_cancel_on_stop);
    }

    template<typename T>
    static bool _irecv_and_wait(T& payload, int src_rank, int tag, std::stop_token stoken, bool allow_cancel_on_stop) {
        MPI_Request recv_req = MPI_REQUEST_NULL;
        MPI_Irecv(reinterpret_cast<void*>(&payload), static_cast<int>(sizeof(T)), MPI_BYTE, src_rank, tag, MPI_COMM_WORLD, &recv_req);
        return _wait_request_complete(recv_req, stoken, allow_cancel_on_stop);
    }
#endif

     public:
    MPICommunicationChannel() = default;

    MPICommunicationChannel(std::string_view channelName, int peer_rank) : peerRank(peer_rank) {
        if (peerRank < 0) {
            throw std::runtime_error("MPI channel constructed with invalid peer rank");
        }
        _check_mpi_ready();

#ifdef FINN_XSI_ENABLE_MPI
        const int tag_ub = _get_mpi_tag_upper_bound();
        requestTag = _tag_from_name(channelName, 0xA5A5A5A5u, tag_ub);
        responseTag = _tag_from_name(channelName, 0x5A5A5A5Au, tag_ub);
        if (responseTag == requestTag) {
            responseTag = (responseTag + 1) % (tag_ub + 1);
        }
#else
        (void) channelName;
#endif
    }

    void handshake() {
        // No-op for now; MPI correctness is validated by first send/recv.
        _check_mpi_ready();
    }

    Response send_request(const Request& req, std::stop_token stoken = {})
        requires(IsSender)
    {
#ifdef FINN_XSI_ENABLE_MPI
        static_assert(std::is_trivially_copyable_v<Request>, "MPI request must be trivially copyable");
        static_assert(std::is_trivially_copyable_v<Response>, "MPI response must be trivially copyable");

        // SHM-like behavior: if stop is requested, transmit a default/trash request
        // best-effort to wake the peer, then return immediately.
        if (stoken.stop_requested()) {
            Request wakeup_payload{};
            (void) _isend_and_wait(wakeup_payload, peerRank, requestTag, stoken, true);
            return Response{};
        }

        const bool request_sent = _isend_and_wait(req, peerRank, requestTag, stoken, true);
        if (!request_sent) {
            return Response{};
        }

        Response resp{};
        const bool response_received = _irecv_and_wait(resp, peerRank, responseTag, stoken, true);
        if (!response_received) {
            return Response{};
        }
        return resp;
#else
        (void) req;
        (void) stoken;
        return Response{};
#endif
    }

    Request receive_request(std::stop_token stoken = {})
        requires(!IsSender)
    {
#ifdef FINN_XSI_ENABLE_MPI
        static_assert(std::is_trivially_copyable_v<Request>, "MPI request must be trivially copyable");

        Request req{};
        const bool received = _irecv_and_wait(req, peerRank, requestTag, stoken, true);
        if (!received) {
            return Request{};
        }
        return req;
#else
        (void) stoken;
        return Request{};
#endif
    }

    void send_response(const Response& resp, std::stop_token stoken = {})
        requires(!IsSender)
    {
#ifdef FINN_XSI_ENABLE_MPI
        static_assert(std::is_trivially_copyable_v<Response>, "MPI response must be trivially copyable");
        (void) _isend_and_wait(resp, peerRank, responseTag, stoken, true);
#else
        (void) resp;
        (void) stoken;
#endif
    }
};

#endif /* MPICOMMUNICATIONCHANNEL */
