#ifndef AXIS_CONTROL
#define AXIS_CONTROL

#include <functional>
#include <string>
#include <vector>
#include <limits>

// Fwd declarations
namespace xsi {
class Design;
class Port;
} // namespace xsi
class Clock;

class AXIS_Control {
public:
  // Constructor/destructor
  AXIS_Control(xsi::Design &design, size_t job_size,
               const std::string &prefix = "s_axis_");
  ~AXIS_Control() = default;

  // Core functions - immediate writes
  void valid(bool value = true);
  bool is_valid() const;
  void ready(bool value = true);
  bool is_ready() const;

  // Deferred write functions
  std::reference_wrapper<xsi::Port> set_valid(bool value = true);
  std::reference_wrapper<xsi::Port> set_ready(bool value = true);

  // Job Size and Transaction Statistics
  size_t job_size;
  size_t job_txns; // [0:job_size]
  size_t total_txns;
  size_t first_complete; // First completion timestamp

  // AXI interface prefix
  std::string name;

  xsi::Port &port_vld;
  xsi::Port &port_rdy;

private:
  xsi::Design &design;
  Clock &clk;
};

class S_AXIS_Control : public AXIS_Control {
public:
  // Constructor/destructor
  S_AXIS_Control(xsi::Design &design, size_t job_size, size_t job_ticks,
                 const std::string &prefix = "s_axis_");
  ~S_AXIS_Control() = default;

  size_t job_ticks;  // throttle if job_size < job_ticks
  size_t await_iter; // iteration allowing start of next job
};

class M_AXIS_Control : public AXIS_Control {
public:
  // Constructor/destructor
  M_AXIS_Control(xsi::Design &design, size_t job_size,
                 const std::string &prefix = "m_axis_");
  ~M_AXIS_Control() = default;

  size_t last_complete;
  size_t interval;
  size_t latency = 0;
  size_t min_latency = std::numeric_limits<size_t>::max(); // Minimum latency observed
};

#endif /* AXIS_CONTROL */
