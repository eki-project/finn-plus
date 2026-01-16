#include <Simulation.hpp>


template<size_t IStreamsSize, size_t OStreamsSize>
class IsolatedSimulation : public Simulation<IStreamsSize, OStreamsSize, false> {
    enum class LogType {READY, VALID};
    std::ofstream readyLog; // Input side
    std::ofstream validLog; // Output side

    /**
     * Write CSV style headers to the files
     **/
    void writeLogHeaders() {
        readyLog << "totalCycles,inputCycles,targetInputCycles";
        for (S_AXIS_Control& s: this->istreams) {
            readyLog << "," << s.name;
        }
        readyLog << std::endl;
        validLog << "totalCycles,outputCycles,targetOutputCycles" << std::endl;
        for (M_AXIS_Control& s : this->ostreams) {
            validLog << "," << s.name;
        }
        validLog << std::endl;
    }

    inline void writeLogEntryReady (size_t cyclesTotal, size_t cycles,
            size_t targetCycles, std::span<S_AXIS_Control> axis) {
        readyLog << cyclesTotal << "," << cycles << "," << targetCycles;
        for (S_AXIS_Control& s : axis) { readyLog << "," << s.isReady(); }
    }

    inline void writeLogEntryValid(size_t cyclesTotal, size_t cycles,
            size_t targetCycles, std::span<M_AXIS_Control> axis) {
        validLog << cyclesTotal << "," << cycles << "," << targetCycles;
        for (M_AXIS_Control& s : axis) { validLog << "," << s.isValid(); }
    }

    /**
     * For the given streams check which has the largest job size, and return a tuple
     * (stream_index, job_size) for that stream.
     **/
    std::tuple<size_t, size_t> getLargestTxnsStream(std::span<AXIS_Control> axis) {
        size_t l = 0;
        size_t idx = 0;
        for (size_t i = 0; i < axis.size(); i++) {
            if (axis[i].job_size > l) {
                l = axis[i].job_size;
                idx = i;
            }
        }
        return std::make_tuple(idx, l);
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
    ), readyLog(readyLogPath), validLog(validLogPath) {}

    void simulate() {
        this->clearPorts();
        this->reset();

        // For split branches calculate the max of all incoming/outgoing stream job sizes, and which stream it is
        auto [inputCyclesTarget, inputLargestStreamIndex] = getLargestTxnsStream(this->istreams);
        auto [outputCyclesTarget, outputLargestStreamIndex] = getLargestTxnsStream(this->ostreams);
        size_t inputCycles = 0;
        size_t outputCycles = 0;

        // Set ports
        for (S_AXIS_Control& s : this->istreams) {
            s.setValid(true);
        }
        for (M_AXIS_Control& s : this->ostreams) {
            s.setReady(true);
        }

        // TODO: Check that components dont behave differently when 0 data is sent through them
        // (only relevant for behavioural simulation.)
        for (size_t cycles = 0; inputCycles < inputCyclesTarget && outputCycles < outputCyclesTarget; ++cycles) {
            // Ready log ends when the input is completely consumed
            if (inputCycles < inputCyclesTarget) {
                writeLogEntryReady(cycles, inputCycles, inputCyclesTarget, this->istreams);
            }

            // Valid log until the end
            writeLogEntryValid(cycles, outputCycles, outputCyclesTarget, this->ostreams);

            // Only if the largest stream transaction is done, is the input/output complete
            if (inputCycles < inputCyclesTarget && this->istreams[inputLargestStreamIndex].isReady()) {
                ++inputCycles;
            }
            if (outputCycles < outputCyclesTarget && this->ostreams[outputLargestStreamIndex].isValid()) {
                ++outputCycles;
            }
            this->clk.toggleClk();
        }
    }
};
