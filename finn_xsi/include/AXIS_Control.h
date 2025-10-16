#ifndef AXIS_CONTROL
#define AXIS_CONTROL

#include <functional>
#include <limits>
#include <string>

// Fwd declarations
namespace xsi {
    class Design;
    class Port;
}  // namespace xsi
class Clock;

class AXIS_Control {
     public:
    // Constructor/destructor
    AXIS_Control(xsi::Design& design, Clock& clock, size_t job_size, const std::string& prefix = "s_axis_");
    AXIS_Control() = default;
    virtual ~AXIS_Control() noexcept = default;

    AXIS_Control(AXIS_Control&& other) = default;
    AXIS_Control& operator=(AXIS_Control&& other) = default;

    void inititialized_or_throw();

    // Core functions - immediate writes
    void valid(bool value = true);
    bool is_valid() const noexcept;
    void ready(bool value = true);
    bool is_ready() const noexcept;

    // Deferred write functions
    std::reference_wrapper<xsi::Port> set_valid(bool value = true);
    std::reference_wrapper<xsi::Port> set_ready(bool value = true);

    // Job Size and Transaction Statistics
    size_t job_size;
    size_t job_txns;  // [0:job_size]
    size_t total_txns;
    size_t first_complete;  // First completion timestamp

    // AXI interface prefix
    std::string name;

     private:
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

    size_t last_complete = 0;
    size_t interval;
    size_t latency = 0;
    size_t min_latency = std::numeric_limits<size_t>::max();  // Minimum latency observed
};

#endif /* AXIS_CONTROL */
