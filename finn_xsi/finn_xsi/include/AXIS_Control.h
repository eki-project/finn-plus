#ifndef AXIS_CONTROL
#define AXIS_CONTROL

#include <CommunicationChannel.hpp>
#include <StableStateTracker.hpp>
#include <functional>
#include <string>
#include <stop_token>

// Fwd declarations
namespace xsi {
    class Design;
    class Port;
}  // namespace xsi
class Clock;

class AXIS_Control : public CommunicationChannel {
     public:
    // Constructor/destructor
    AXIS_Control(xsi::Design& design, Clock& clock, size_t job_size, const std::string& prefix = "s_axis_");
    AXIS_Control() = default;
    virtual ~AXIS_Control() noexcept = default;

    AXIS_Control(AXIS_Control&& other) = default;
    AXIS_Control& operator=(AXIS_Control&& other) = default;

    void inititialized_or_throw();

    // Core functions - immediate writes
    virtual void setInputValid(bool value = true, std::stop_token stoken = {}) override;
    virtual bool getOutputValid(std::stop_token stoken = {}) noexcept override;
    virtual void setOutputReady(bool value = true, std::stop_token stoken = {}) override;
    virtual bool getInputReady(std::stop_token stoken = {}) noexcept override;

    // Deferred write functions
    std::reference_wrapper<xsi::Port> setValid(bool value = true);
    std::reference_wrapper<xsi::Port> setReady(bool value = true);

    virtual void writeBack() = 0;

    // Job Size and Transaction Statistics
    size_t job_size;
    size_t job_txns;  // [0:job_size]
    size_t total_txns;
    size_t first_complete;  // First completion timestamp

    // AXI interface prefix
    std::string name;

     protected:
    const xsi::Design* design;
    const Clock* clk;

    xsi::Port* port_vld;
    xsi::Port* port_rdy;
};

class S_AXIS_Control : public AXIS_Control {
     public:
    // Constructor/destructor
    S_AXIS_Control(xsi::Design& design, Clock& clock, size_t job_size, size_t job_ticks, const std::string& prefix = "s_axis_");
    S_AXIS_Control() = default;
    ~S_AXIS_Control() noexcept = default;

    S_AXIS_Control(S_AXIS_Control&& other) = default;
    S_AXIS_Control& operator=(S_AXIS_Control&& other) = default;

    void writeBack() override;

    size_t job_ticks;   // throttle if job_size < job_ticks
    size_t await_iter;  // iteration allowing start of next job
};

class M_AXIS_Control : public AXIS_Control {
     public:
    // Constructor/destructor
    M_AXIS_Control(xsi::Design& design, Clock& clock, size_t job_size, const std::string& prefix = "m_axis_");
    M_AXIS_Control() = default;
    ~M_AXIS_Control() noexcept = default;

    M_AXIS_Control(M_AXIS_Control&& other) = default;
    M_AXIS_Control& operator=(M_AXIS_Control&& other) = default;

    void writeBack() override;

    size_t lastComplete = 0;
    size_t interval = 0;
    StableStateTracker<> stableState;
};

#endif /* AXIS_CONTROL */
