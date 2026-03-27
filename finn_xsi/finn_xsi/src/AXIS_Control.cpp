#include <AXIS_Control.h>
#include <Clock.h>
#include <Design.h>
#include <Port.h>

#include <stdexcept>

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

std::string remove_trailing_underscore(const std::string& prefix) {
    std::string clean = prefix;
    if (clean.back() == '_') {
        // If checks implicitly that pop_back() doesn't have undefined behav.
        clean.pop_back();
    }
    return clean;
}

AXIS_Control::AXIS_Control(xsi::Design& des, Clock& clock, size_t job_sz, const std::string& prefix)
    : job_size(job_sz),
      job_txns(0),
      total_txns(0),
      first_complete(0),
      name(remove_trailing_underscore(sanitize_prefix(prefix))),
      design(&des),
      clk(&clock),
      port_vld(&des.getPort(sanitize_prefix(prefix) + "tvalid")),
      port_rdy(&des.getPort(sanitize_prefix(prefix) + "tready")) {}

void AXIS_Control::inititialized_or_throw() {
    if (!design || !clk || !port_rdy || !port_vld) {
        throw std::runtime_error("AXIS Control object not correctly initialized! Aborting!");
    }
}

void AXIS_Control::setInputValid(bool value, [[maybe_unused]] std::stop_token stoken) { port_vld->set(static_cast<unsigned int>(value)).write_back(); }

bool AXIS_Control::getOutputValid([[maybe_unused]] std::stop_token stoken) noexcept { return port_vld->read().as_bool(); }

void AXIS_Control::setOutputReady(bool value, [[maybe_unused]] std::stop_token stoken) { port_rdy->set(static_cast<unsigned int>(value)).write_back(); }
bool AXIS_Control::getInputReady([[maybe_unused]] std::stop_token stoken) noexcept { return port_rdy->read().as_bool(); }

// Deferred write functions
std::reference_wrapper<xsi::Port> AXIS_Control::setValid(bool value) { return std::ref(port_vld->set(value ? 1 : 0)); }

std::reference_wrapper<xsi::Port> AXIS_Control::setReady(bool value) { return std::ref(port_rdy->set(value ? 1 : 0)); }

S_AXIS_Control::S_AXIS_Control(xsi::Design& des, Clock& clock, size_t job_sz, size_t job_tks, const std::string& prefix)
    : AXIS_Control(des, clock, job_sz, prefix), job_ticks(job_tks), await_iter(job_tks) {
    if (job_sz < 1 || job_tks < 1) {
        throw std::invalid_argument("Job size and ticks must be greater than 0.");
    }
}

void S_AXIS_Control::writeBack() {
    this->port_vld->write_back();
}

M_AXIS_Control::M_AXIS_Control(xsi::Design& des, Clock& clock, size_t job_sz, const std::string& prefix) : AXIS_Control(des, clock, job_sz, prefix), lastComplete(0), interval(0) {
    if (job_sz < 1) {
        throw std::invalid_argument("Job size must be greater than 0.");
    }
}

void M_AXIS_Control::writeBack() {
    this->port_rdy->write_back();
}
