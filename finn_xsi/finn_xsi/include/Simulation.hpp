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
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/atomic/ipc_atomic.hpp>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <fstream>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/asio.hpp>
#include <optional>
#include <stdexcept>

#include <new>
#include <thread>
#include <type_traits>
#ifdef __cpp_lib_hardware_interference_size
    using std::hardware_destructive_interference_size;
#else
    constexpr std::size_t hardware_destructive_interference_size = 64;
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

template<SimulationInterfaceType T, std::size_t ShmemSize = 1024>
class SimulationInterface {
    private:
    struct _SimulationInterface {
        alignas(hardware_destructive_interference_size) boost::ipc_atomic<bool>* ready;
        alignas(hardware_destructive_interference_size) boost::ipc_atomic<bool>* valid;
        alignas(hardware_destructive_interference_size) boost::ipc_atomic<bool>* unread;
        alignas(hardware_destructive_interference_size) boost::ipc_atomic<unsigned int>* fifoOccupation;
    };
    ipc::managed_shared_memory shmem;
    _SimulationInterface interface;
    const std::string shmIdentifier;

    /// Logging
    std::ofstream out;

    public:
    SimulationInterface(const char* _shmIdentifier) : shmIdentifier(_shmIdentifier) {
        if (T == SimulationInterfaceType::PRODUCING) {
            ipc::shared_memory_object::remove(_shmIdentifier);
        }
        shmem = ipc::managed_shared_memory(ipc::open_or_create, _shmIdentifier, ShmemSize);
        interface.ready = shmem.find_or_construct<boost::ipc_atomic<bool>>("ready")(true);
        interface.valid = shmem.find_or_construct<boost::ipc_atomic<bool>>("valid")(false);
        interface.unread = shmem.find_or_construct<boost::ipc_atomic<bool>>("unread")(false);
        interface.fifoOccupation = shmem.find_or_construct<boost::ipc_atomic<unsigned int>>("fifoOccupation")(0);
    }

    ~SimulationInterface() {
        // TODO: Called implicitly?
        shmem.destroy<boost::ipc_atomic<bool>>("ready");
        shmem.destroy<boost::ipc_atomic<bool>>("valid");
        shmem.destroy<boost::ipc_atomic<bool>>("unread");
        shmem.destroy<boost::ipc_atomic<unsigned int>>("fifoOccupation");
    }

    /// Wait until predecessor has sent recent data. Then send ready.
    bool communicate(bool sendReady) requires (T == SimulationInterfaceType::CONSUMING) {
        while (!*(interface.unread)) {}
        bool validValue = *(interface.valid);
        *(interface.ready) = sendReady;
        *(interface.unread) = false;
        return validValue;
    }

    /// Wait until successor has read the previous data. Then send valid.
    bool communicate(bool sendValid) requires (T == SimulationInterfaceType::PRODUCING) {
        while (*(interface.unread)) {}
        bool readyValue = *(interface.ready);
        *(interface.valid) = sendValid;
        *(interface.unread) = true;
        return readyValue;
    }
};



template<size_t IStreamsSize, size_t OStreamsSize, bool LoggingEnabled, size_t NodeIndex, size_t TotalNodes>
class SingleNodeSimulation : public Simulation<IStreamsSize, OStreamsSize, LoggingEnabled> {
    private:
    using ConsumingInterface = SimulationInterface<SimulationInterfaceType::CONSUMING>;
    using ProducingInterface = SimulationInterface<SimulationInterfaceType::PRODUCING>;
    std::optional<std::array<std::unique_ptr<ConsumingInterface>, IStreamsSize>> fromProducerInterface;
    std::optional<std::array<std::unique_ptr<ProducingInterface>, OStreamsSize>> toConsumerInterface;

    public:
    SingleNodeSimulation(
        const std::string& kernel_lib,
        const std::string& design_lib,
        const char* xsim_log_file,
        const char* trace_file,
        std::array<StreamDescriptor, IStreamsSize> _istream_descs,
        std::array<StreamDescriptor, OStreamsSize> _ostream_descs,
        std::optional<std::string> previousNodeName = std::nullopt,
        std::optional<std::string> currentNodeName = std::nullopt
    ) :
    Simulation<IStreamsSize, OStreamsSize, LoggingEnabled>(kernel_lib, design_lib, xsim_log_file, trace_file, _istream_descs, _ostream_descs) {
        initStreams();

        if ((NodeIndex != 0 && !previousNodeName) || !currentNodeName) {
            throw std::runtime_error("Cannot construct single-node simulation without specifying the previous and this nodes names.");
        }

        // Create producer facing interfaces
        if constexpr(NodeIndex != 0) {
            fromProducerInterface = std::make_optional<std::array<std::unique_ptr<ConsumingInterface>, IStreamsSize>>();
            for (std::size_t i = 0; i < IStreamsSize; ++i) {
                (*fromProducerInterface)[i] = std::make_unique<ConsumingInterface>((*previousNodeName + std::to_string(i)).c_str());
            }
        } else {
            fromProducerInterface = std::nullopt;
        }

        // Create consumer facing interfaces
        if constexpr(NodeIndex != TotalNodes - 1) {
        toConsumerInterface = std::make_optional<std::array<std::unique_ptr<ProducingInterface>, OStreamsSize>>();
            for (std::size_t i = 0; i < OStreamsSize; ++i) {
                (*toConsumerInterface)[i] = std::make_unique<ProducingInterface>((*currentNodeName + std::to_string(i)).c_str());
            }
        } else {
            toConsumerInterface = std::nullopt;
        }

    }

    private:
    /// Communicate with predecessors and successors and update their values and our own
    void communicate() {
        if constexpr(NodeIndex != TotalNodes - 1) {
            for (std::size_t i = 0; i < OStreamsSize; ++i) {
                this->ostreams[i].ready((*toConsumerInterface)[i]->communicate(this->ostreams[i].is_valid()));
            }
        }
        if constexpr(NodeIndex != 0) {
            for (std::size_t i = 0; i < IStreamsSize; ++i) {
                this->istreams[i].valid((*fromProducerInterface)[i]->communicate(this->istreams[i].is_ready()));
            }
        }
    }

    public:
    /// Init streams according to nodeindex
    void initStreams() {
        if constexpr(NodeIndex == 0) {
            // The first node receives all valid input streams
            for (auto&& s : this->istreams) {
                s.valid();
            }
        }
        if constexpr(NodeIndex == TotalNodes - 1) {
            // The last nodes receives all ready output streams
            for (auto&& s : this->ostreams) {
                s.ready();
            }
        }
    }

    void runSingleCycle() {
        this->clk.toggle_clk();
        communicate();

        // Log the signals that this simulations set (ready to predecessor, valid to successor)
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

    /// Run for the given number of frames (frames * job_size   cycles or transactions)
    void runForFrames(std::size_t frames) {
        // TODO: Multiple IO streams: Current cycle count is hardcoded for the number of inputs of the first stream
        std::size_t cycleCount = frames * this->istreams[0].job_size;
        for (std::size_t i = 0; i < cycleCount; ++i) {
            runSingleCycle();
        }
    }
};

#endif /* SIMULATION */
