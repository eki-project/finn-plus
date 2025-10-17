#ifndef SIMULATION
#define SIMULATION
#include <AXIS_Control.h>
#include <Clock.h>
#include <Design.h>
#include <Kernel.h>
#include <Port.h>
#include <SharedLibrary.h>
#include <helper.h>
#include <atomic>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <iostream>
#include <memory>
#include <fstream>
#include "boost/interprocess/creation_tags.hpp"
#include "boost/interprocess/interprocess_fwd.hpp"

namespace ipc = boost::interprocess;

enum class SimulationInterfaceType { PRODUCING, CONSUMING };

template<SimulationInterfaceType T, std::size_t ShmemSize = 1024>
class SimulationInterface {
    private:
    ipc::managed_shared_memory shmem;
    std::atomic_bool ready;
    std::atomic_bool valid;
    std::atomic_bool unread;

    public:
    SimulationInterface(const char* shmIdentifier) {
        ipc::shared_memory_object::remove(shmIdentifier);
        shmem = ipc::managed_shared_memory(ipc::open_or_create, shmIdentifier, ShmemSize);
        ready = shmem.find_or_construct<std::atomic_bool>("ready")(true);
        valid = shmem.find_or_construct<std::atomic_bool>("valid")(false);
        unread = shmem.find_or_construct<std::atomic_bool>("unread")(false);
    }

    ~SimulationInterface() {
        // TODO: Called implicitly?
        shmem.destroy<std::atomic_bool>("ready");
        shmem.destroy<std::atomic_bool>("valid");
        shmem.destroy<std::atomic_bool>("unread");
    }

    /// Wait until predecessor has sent recent data. Then send ready.
    bool communicate(bool sendReady) requires (T == SimulationInterfaceType::CONSUMING) {
        while (!unread) {}
        ready = sendReady;
        auto validValue = valid.load();
        unread = false;
        return validValue;
    }

    /// Wait until successor has read the previous data. Then send valid.
    bool communicate(bool sendValid) requires (T == SimulationInterfaceType::PRODUCING) {
        while (unread) {}
        valid = sendValid;
        auto readyValue = ready.load();
        unread = true;
        return readyValue;
    }

};


template<size_t IStreamsSize, size_t OStreamsSize, bool LoggingEnabled, bool SingleNode>
class Simulation {
    private:
    using ConsumerInterface = SimulationInterface<SimulationInterfaceType::CONSUMING>;
    using ProducerInterface = SimulationInterface<SimulationInterfaceType::PRODUCING>;
    std::array<std::unique_ptr<ConsumerInterface>, IStreamsSize> fromProducerInterface;
    std::array<std::unique_ptr<ProducerInterface>, OStreamsSize> toConsumerInterface;
    std::ofstream readyLog;
    std::ofstream validLog;

    public:
    xsi::Kernel kernel;
    xsi::Design top;
    std::array<S_AXIS_Control, IStreamsSize> istreams;
    std::array<M_AXIS_Control, OStreamsSize> ostreams;
    Clock clk;


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

    Simulation(const std::string& previousName, const std::string& name, const std::string& kernel_lib, const std::string& design_lib, const char* xsim_log_file, const char* trace_file, std::array<StreamDescriptor, IStreamsSize> _istream_descs,
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

        // Find Global Control & Run Startup Sequence
        clearPorts();
        reset();

        // Make all Inputs valid & all Outputs ready
        for (auto&& s : istreams) {
            s.valid();
        }
        for (auto&& s : ostreams) {
            s.ready();
        }

        if constexpr(SingleNode) {
            // Create simulation interfaces
            for (std::size_t i = 0; i < IStreamsSize; ++i) {
                fromProducerInterface[i] = std::make_unique<ConsumerInterface>((previousName + std::to_string(i)).c_str());
            }
            for (std::size_t i = 0; i < OStreamsSize; ++i) {
                toConsumerInterface[i] = std::make_unique<ProducerInterface>((name + std::to_string(i)).c_str());
            }

            // Save simulation input output behaviour
            if constexpr(LoggingEnabled) {
                readyLog.open("ready_log.txt");
                validLog.open("valid_log.txt");
            }
        } else {
            // TODO
            // Entire design in one simulation
        }
    }

    /// Read valid signal from producer, write own ready signal to it
    void updateFromProducer() requires (SingleNode) {
        for (std::size_t i = 0; i < IStreamsSize; ++i) {
            istreams[i].valid(fromProducerInterface[i]->communicate(istreams[i].is_ready()));
        }
    }

    /// Read ready signal from consumer, write own valid signal to it.
    void updateToConsumer() requires (SingleNode) {
        for (std::size_t i = 0; i < OStreamsSize; ++i) {
            ostreams[i].ready(toConsumerInterface[i]->communicate(ostreams[i].is_valid()));
        }
    }

    void runSingleCycle() {
        if constexpr(SingleNode) {
            clk.toggle_clk();
            // Order: Send update forward to consumer, read update from producer second
            updateFromProducer();
            updateToConsumer();

            // Log the signals that this simulations set (ready to predecessor, valid to successor)
            if constexpr(LoggingEnabled) {
                for (S_AXIS_Control& stream : istreams) {
                    readyLog << stream.is_ready() << " ";
                }
                readyLog << "\n";
                for (M_AXIS_Control& stream : ostreams) {
                    validLog << stream.is_valid() << " ";
                }
                validLog << "\n";
            }
        } else {
            // TODO
            // Single design case
        }
    }

    /// Run for the given number of frames (frames * job_size   cycles or transactions)
    void runForFrames(std::size_t frames) {
        if constexpr(SingleNode) {
            // TODO: Multiple IO streams: Current cycle count is hardcoded for the number of inputs of the first stream
            std::size_t cycleCount = frames * istreams[0].job_size;
            for (std::size_t i = 0; i < cycleCount; ++i) {
                runSingleCycle();
            }
        } else {
            // TODO
            // Single design case
        }
    }
};

#endif /* SIMULATION */
