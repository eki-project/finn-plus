#include <Simulation.hpp>
#include "SocketServer.h"


template<size_t IStreamsSize, size_t OStreamsSize>
class IsolatedSimulation : public Simulation<IStreamsSize, OStreamsSize, false> {
    enum class LogType {READY, VALID};
    std::string readylogName;
    std::string validlogName;
    nlohmann::ordered_json readyJson;
    nlohmann::ordered_json validJson;
    std::vector<size_t> inJobSizes;
    std::vector<size_t> outJobSizes;

    /**
     * For the given streams check which has the largest job size, and return a tuple
     * (stream_index, job_size) for that stream.
     **/
    std::tuple<size_t, size_t> getLargestTxnsStream(std::vector<size_t>& jobSizes) {
        size_t l = 0;
        size_t idx = 0;
        for (size_t i = 0; i < jobSizes.size(); i++) {
            if (jobSizes[i] > l) {
                l = jobSizes[i];
                idx = i;
            }
        }
        return std::make_tuple(idx, l);
    }

    class SimState {
        public:
        bool running;
        size_t inputCyclesDone;
        size_t inputCyclesTarget;
        size_t inputLargestStreamIndex;
        size_t outputCyclesDone;
        size_t outputCyclesTarget;
        size_t outputLargestStreamIndex;
        size_t totalCycles;

        SimState(IsolatedSimulation<IStreamsSize, OStreamsSize>& sim) {
            reset(sim);
        }
        void reset(IsolatedSimulation<IStreamsSize, OStreamsSize>& sim) {
            totalCycles = 0;
            inputCyclesDone = 0;
            outputCyclesDone = 0;
            running = false;
            auto largestIn = sim.getLargestTxnsStream(sim.inJobSizes);
            auto largestOut = sim.getLargestTxnsStream(sim.outJobSizes);
            inputCyclesTarget = std::get<1>(largestIn) * 2;
            inputLargestStreamIndex = std::get<0>(largestIn);
            outputCyclesTarget = std::get<1>(largestOut) * 2;
            outputLargestStreamIndex = std::get<0>(largestOut);
        }
        inline bool inputCyclesProcessed() { return inputCyclesDone >= inputCyclesTarget; }
        inline bool outputCyclesProcessed() { return outputCyclesDone >= outputCyclesTarget; }
        inline bool allCyclesProcessed() { return inputCyclesProcessed() && outputCyclesProcessed(); }
        inline bool isRunning() { return running; }
        void setRunning(bool v) { running = v; }
        std::string getCycleStateInput() { return std::to_string(totalCycles) + "," + std::to_string(inputCyclesDone) + "," + std::to_string(inputCyclesTarget); }
        std::string getCycleStateOutput() { return std::to_string(totalCycles) + "," + std::to_string(outputCyclesDone) + "," + std::to_string(outputCyclesTarget); }
        json getStatus() {
            json j;
            if (!running && allCyclesProcessed()) {
                j["state"] = "done";
            } else {
                j["state"] = running ? "running" : "halted";
            }
            j["totalCycles"] = totalCycles;
            j["inputCyclesDone"] = inputCyclesDone;
            j["inputCyclesTarget"] = inputCyclesTarget;
            j["outputCyclesDone"] = outputCyclesDone;
            j["outputCyclesTarget"] = outputCyclesTarget;
            return j;
        }
    };


    /** Log the ready and valid signals to the JSON fields **/
    void logReady() {
        nlohmann::ordered_json j;
        j["totalCycles"] = simState.totalCycles;
        j["inputCyclesDone"] = simState.inputCyclesDone;
        j["inputCyclesTarget"] = simState.inputCyclesTarget;
        for (S_AXIS_Control& s : this->istreams) {
            j[s.name] = s.getInputReady();
        }
        readyJson.push_back(j);
    }

    void logValid() {
        nlohmann::ordered_json j;
        j["totalCycles"] = simState.totalCycles;
        j["outputCyclesDone"] = simState.outputCyclesDone;
        j["outputCyclesTarget"] = simState.outputCyclesTarget;
        for (M_AXIS_Control& s : this->ostreams) {
            j[s.name] = s.getOutputValid();
        }
        validJson.push_back(j);
    }

    SimState simState;

    public:
    IsolatedSimulation(
        const std::string& kernel_lib,
        const std::string& design_lib,
        const char* xsim_log_file,
        const char* trace_file,
        std::array<StreamDescriptor, IStreamsSize> _istream_descs,
        std::array<StreamDescriptor, OStreamsSize> _ostream_descs
    ) : Simulation<IStreamsSize, OStreamsSize, false>(
        kernel_lib, design_lib, xsim_log_file, trace_file, _istream_descs, _ostream_descs
    ), simState(*this), readyJson(json::array()), validJson(json::array()),
    readylogName("readylog.txt"), validlogName("validlog.txt") {
        // TODO: Clearly split names between connected and isolated sim (ready_log.txt and readylog.txt)
        inJobSizes.resize(_istream_descs.size());
        outJobSizes.resize(_ostream_descs.size());
        std::transform(
            _istream_descs.begin(),
            _istream_descs.end(),
            inJobSizes.begin(),
            [](StreamDescriptor& s) { return s.job_size; }
        );
        std::transform(
            _ostream_descs.begin(),
            _ostream_descs.end(),
            outJobSizes.begin(),
            [](StreamDescriptor& s) { return s.job_size; }
        );
    }

    /** Write logs to disk **/
    void commitLogsToDisk(bool clearLogs = true) {
        std::ofstream r(readylogName, std::ios::trunc);
        std::ofstream v(validlogName, std::ios::trunc);
        r << std::setw(4) << readyJson;
        std::cout << "Writing ready log: " << readyJson.size() << " elements." << std::endl;
        v << std::setw(4) << validJson;
        std::cout << "Writing valid log: " << validJson.size() << " elements." << std::endl;
        r.close();
        v.close();
        if (clearLogs) {
            readyJson = json::array();
            validJson = json::array();
        }
    }

    json getStatus() {
        return simState.getStatus();
    }

    void halt() {
        simState.setRunning(false);
    }

    void resume() {
        simState.setRunning(true);
    }

    bool isRunning() { return simState.isRunning(); }

    bool isDone() {
        return !simState.isRunning() && simState.allCyclesProcessed();
    }

    /***
     * Simulate a single cycle
     ***/
    void simulate(bool restart = false) {
        if (restart) {
            simState.reset(*this);
            simState.setRunning(true);
            std::cout << "Sim set to running: " << simState.isRunning() << std::endl;
            std::cout << "Target input/output cycles: " << simState.inputCyclesTarget << ", " << simState.outputCyclesTarget << std::endl;
            this->clearPorts();
            this->reset();
            for (S_AXIS_Control& s : this->istreams) {
                s.setInputValid(true);
            }
            for (M_AXIS_Control& s : this->ostreams) {
                s.setOutputReady(true);
            }
        }

        if (!simState.isRunning()) {
            std::cout << "Simulation not running! Send \"start\" command first." << std::endl;
            return;
        }
        if (!simState.allCyclesProcessed()) {
            logValid();
            logReady();

            if (!simState.inputCyclesProcessed() && this->istreams[simState.inputLargestStreamIndex].getInputReady()) {
                ++simState.inputCyclesDone;
            }
            if (!simState.outputCyclesProcessed() && this->ostreams[simState.outputLargestStreamIndex].getOutputValid()) {
                ++simState.outputCyclesDone;
            }
            this->clk.toggleClk();
            ++simState.totalCycles;
        } else {
           simState.setRunning(false);
        }
    }
};
