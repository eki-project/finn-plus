#include <AXIS_Control.h>
#include <Clock.h>
#include <Design.h>
#include <Port.h>

#include <iostream>

std::string sanitize_prefix(const std::string& prefix) {
    if (prefix.empty()) {
        throw std::invalid_argument("AXI prefix cannot be empty.");
    }
    std::string sanitized = prefix;
    if (sanitized.back() != '_') {
        sanitized += "_";
    }
    return sanitized;
}

AXIS_Control::AXIS_Control(xsi::Design& des, Clock& clock, size_t job_sz, const std::string& prefix)
    : job_size(job_sz), job_txns(0), total_txns(0), first_complete(0), name(sanitize_prefix(prefix)), port_vld(des.getPort(name + "tvalid")), port_rdy(des.getPort(name + "tready")), design(des), clk(clock) {}

void AXIS_Control::valid(bool value) { port_vld.set(value ? 1 : 0).write_back(); }

bool AXIS_Control::is_valid() const noexcept { return port_vld.read().as_bool(); }

void AXIS_Control::ready(bool value) { port_rdy.set(value ? 1 : 0).write_back(); }

bool AXIS_Control::is_ready() const noexcept { return port_rdy.read().as_bool(); }

// Deferred write functions
std::reference_wrapper<xsi::Port> AXIS_Control::set_valid(bool value) { return std::ref(port_vld.set(value ? 1 : 0)); }

std::reference_wrapper<xsi::Port> AXIS_Control::set_ready(bool value) { return std::ref(port_rdy.set(value ? 1 : 0)); }

S_AXIS_Control::S_AXIS_Control(xsi::Design& des, Clock& clock, size_t job_sz, size_t job_tks, const std::string& prefix) : AXIS_Control(des, clock, job_sz, prefix), job_ticks(job_tks), await_iter(job_tks) {
    if (job_sz < 1 || job_tks < 1) {
        throw std::invalid_argument("Job size and ticks must be greater than 0.");
    }
}

M_AXIS_Control::M_AXIS_Control(xsi::Design& des, Clock& clock, size_t job_sz, const std::string& prefix) : AXIS_Control(des, clock, job_sz, prefix), last_complete(0), interval(0) {
    if (job_sz < 1) {
        throw std::invalid_argument("Job size must be greater than 0.");
    }
}
