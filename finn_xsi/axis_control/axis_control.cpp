#include "axis_control.h"

#include "../clock/clock.h"
#include "../xsi_finn.hpp"
#include <iostream>

std::string sanitize_prefix(const std::string &prefix) {
  if (prefix.empty()) {
    throw std::invalid_argument("AXI prefix cannot be empty.");
  }
  std::string sanitized = prefix;
  if (sanitized.back() != '_') {
    sanitized += "_";
  }
  return sanitized;
}

xsi::Port* get_port(xsi::Design &design, const std::string &name) {
  xsi::Port *port = design.getPort(name);
  if (!port) {
    throw std::runtime_error("Port " + name + " not found in design.");
  }
  return port;
}

AXIS_Control::AXIS_Control(xsi::Design &design, size_t job_size,
                           const std::string &prefix)
    : name(sanitize_prefix(prefix)), design(design), clk(Clock::initClock(design)),
      port_vld(*get_port(design, name + "tvalid")),
      port_rdy(*get_port(design, name + "tready")), job_size(job_size),
      job_txns(0), total_txns(0), first_complete(0) {
}

void AXIS_Control::valid(bool value) {
  port_vld.set(value ? 1 : 0).write_back();
}

bool AXIS_Control::is_valid() const {
  return port_vld.read().as_bool();
}

void AXIS_Control::ready(bool value) {
  port_rdy.set(value ? 1 : 0).write_back();
}

bool AXIS_Control::is_ready() const {
  return port_rdy.read().as_bool();
}

// Deferred write functions
std::reference_wrapper<xsi::Port> AXIS_Control::set_valid(bool value) {
  return std::ref(port_vld.set(value ? 1 : 0));
}

std::reference_wrapper<xsi::Port> AXIS_Control::set_ready(bool value) {
  return std::ref(port_rdy.set(value ? 1 : 0));
}

S_AXIS_Control::S_AXIS_Control(xsi::Design &design, size_t job_size,
                               size_t job_ticks, const std::string &prefix)
    : AXIS_Control(design, job_size, prefix), job_ticks(job_ticks),
      await_iter(job_ticks) {
  if (job_size < 1 || job_ticks < 1) {
    throw std::invalid_argument("Job size and ticks must be greater than 0.");
  }
}

M_AXIS_Control::M_AXIS_Control(xsi::Design &design, size_t job_size,
                               const std::string &prefix)
    : AXIS_Control(design, job_size, prefix), last_complete(0), interval(0) {
  if (job_size < 1) {
    throw std::invalid_argument("Job size must be greater than 0.");
  }
}
