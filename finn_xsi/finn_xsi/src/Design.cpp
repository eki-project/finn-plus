#include <Design.h>

using namespace xsi;

// Constructors
Design::Design(xsi::Kernel& kernel, const std::string& design_lib, const s_xsi_setup_info& setup_info) : _kernel(std::move(kernel)) { _kernel.open(design_lib, setup_info); }

Design::Design(xsi::Kernel& kernel, const std::string& design_lib, const char* const log_file, const char* const wdb_file)
    : Design(kernel, design_lib, s_xsi_setup_info{.logFileName = const_cast<char*>(log_file), .wdbFileName = const_cast<char*>(wdb_file)}) {}

// Destructor
Design::~Design() { _kernel.close(); }

// Move constructor
Design::Design(Design&& other) noexcept : _kernel(std::move(other._kernel)) {
    // The kernel now manages the moved design
    // No additional work needed as the kernel handles the XSI state
}

// Move assignment operator
Design& Design::operator=(Design&& other) noexcept {
    if (this != &other) {
        _kernel.close();  // Close current design
                          // Note: _kernel is a reference and cannot be reassigned
                          // The move semantics here are limited since we hold a reference
        _kernel = std::move(other._kernel);
    }
    return *this;
}

// Simulation Control & Status
void Design::trace_all() { _kernel.xsi<xsi::Kernel::Xsi::trace_all>(); }

void Design::run(const XSI_INT64 step) { _kernel.xsi<xsi::Kernel::Xsi::run>(step); }

void Design::restart() { _kernel.xsi<xsi::Kernel::Xsi::restart>(); }

int Design::get_status() const noexcept { return _kernel.xsi<xsi::Kernel::Xsi::get_status>(); }

const char* Design::get_error_info() const noexcept { return _kernel.xsi<xsi::Kernel::Xsi::get_error_info>(); }

// Port Access
int Design::num_ports() const noexcept { return static_cast<int>(_kernel.port_count()); }

xsi::Port& Design::getPort(const std::string& name) { return _kernel.getPort(name.c_str()); }

const xsi::Port& Design::getPort(const std::string& name) const { return _kernel.getPort(name.c_str()); }

std::span<xsi::Port> Design::ports() noexcept { return _kernel.ports(); }

std::span<const xsi::Port> Design::ports() const noexcept { return _kernel.ports(); }
