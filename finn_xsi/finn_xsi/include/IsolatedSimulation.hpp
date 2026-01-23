#include <Simulation.hpp>
#include <thread>
#include "SocketServer.h"


template<size_t IStreamsSize, size_t OStreamsSize>
class IsolatedSimulation : public Simulation<IStreamsSize, OStreamsSize, false> {
    enum class LogType {READY, VALID};
    std::ofstream readyLog; // Input side
    std::ofstream validLog; // Output side
    std::vector<size_t> inJobSizes;
    std::vector<size_t> outJobSizes;


    /**
     * Write CSV style headers to the files
     **/
    void writeLogHeaders() {
        readyLog << "totalCycles,inputCycles,doubled_targetInputCycles";
        for (S_AXIS_Control& s: this->istreams) {
            readyLog << "," << s.name;
        }
        readyLog << std::endl;
        validLog << "totalCycles,outputCycles,doubled_targetOutputCycles" << std::endl;
        for (M_AXIS_Control& s : this->ostreams) {
            validLog << "," << s.name;
        }
        validLog << std::endl;
    }


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
            std::cout << "In Job Sizes: ";
            for (auto js : sim.inJobSizes) {
                std::cout << js << " ";
            }
            std::cout << std::endl;
            std::cout << "IO cycle targets: " << inputCyclesTarget << ", " << outputCyclesTarget << std::endl;
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

    SimState simState;

    inline void writeLogEntryReady () {
        readyLog << simState.getCycleStateInput();
        for (S_AXIS_Control& s : this->istreams) { readyLog << "," << s.getInputReady(); }
    }

    inline void writeLogEntryValid() {
        validLog << simState.getCycleStateOutput();
        for (M_AXIS_Control& s : this->ostreams) { validLog << "," << s.getOutputValid(); }
    }

    public:
    IsolatedSimulation(
        const std::string& kernel_lib,
        const std::string& design_lib,
        const char* xsim_log_file,
        const char* trace_file,
        std::array<StreamDescriptor, IStreamsSize> _istream_descs,
        std::array<StreamDescriptor, OStreamsSize> _ostream_descs,
        const std::string readyLogPath,
        const std::string validLogPath
    ) : Simulation<IStreamsSize, OStreamsSize, false>(
        kernel_lib, design_lib, xsim_log_file, trace_file, _istream_descs, _ostream_descs
    ), readyLog(readyLogPath), validLog(validLogPath), simState(*this) {
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

        // Sanity check. Eventually remove
        if (!std::all_of(this->istreams.begin(), this->istreams.end(), [](S_AXIS_Control& s) {return s.isValid();})) {
            std::cout << "ERROR: An input stream is not valid!" << std::endl;
        }
        if (!std::all_of(this->ostreams.begin(), this->ostreams.end(), [](M_AXIS_Control& s) {return s.isReady();})) {
            std::cout << "ERROR: An output stream is not ready!" << std::endl;
        }

        if (!simState.isRunning()) {
            std::cout << "Simulation not running! Send \"start\" command first." << std::endl;
            return;
        }
        if (!simState.allCyclesProcessed()) {
            writeLogEntryReady();
            writeLogEntryValid();

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
