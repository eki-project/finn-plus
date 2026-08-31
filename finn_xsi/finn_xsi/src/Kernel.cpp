#include <Kernel.h>
#include <Port.h>
#include <SharedLibrary.h>

#include <iostream>
#include <memory>
#include <stdexcept>

using namespace xsi;

void* resolve_or_throw(xsi::SharedLibrary& lib, char const* const sym) {
    auto const res = lib.getsymbol(sym);
    if (!res) {
        throw std::runtime_error(std::string("Failed to resolve ").append(sym).append(" in ").append(lib.path()));
    }
    return *res;
}

char const* const Kernel::Xsi::FUNC_NAMES[EXTENT] = {"xsi_get_value",      "xsi_put_value",
                                                     "xsi_get_int_port",   "xsi_get_str_port",

                                                     "xsi_get_int",        "xsi_get_port_number",

                                                     "xsi_trace_all",      "xsi_run",
                                                     "xsi_restart",        "xsi_get_status",
                                                     "xsi_get_error_info",

                                                     "xsi_close"};


Kernel::Xsi::Xsi(xsi::SharedLibrary& lib) : _hdl(nullptr) {
    // Resolve XSI Functions
    for (unsigned i = 0; i < EXTENT; i++) {
        _func[i] = resolve_or_throw(lib, FUNC_NAMES[i]);
    }
}

// Xsi Move constructor
Kernel::Xsi::Xsi(Xsi&& other) noexcept : _hdl(other._hdl) {
    std::copy(std::begin(other._func), std::end(other._func), std::begin(_func));
    other._hdl = nullptr;
    std::fill(std::begin(other._func), std::end(other._func), nullptr);
}

// Xsi Move assignment operator
Kernel::Xsi& Kernel::Xsi::operator=(Xsi&& other) noexcept {
    if (this != &other) {
        _hdl = other._hdl;
        std::copy(std::begin(other._func), std::end(other._func), std::begin(_func));
        other._hdl = nullptr;
        std::fill(std::begin(other._func), std::end(other._func), nullptr);
    }
    return *this;
}

// Xsi Handle management
void Kernel::Xsi::setHandle(xsiHandle hdl) noexcept { _hdl = hdl; }

bool Kernel::Xsi::hasValidHandle() const noexcept { return _hdl != nullptr; }
//---------------------------------------------------------------------------
// Life Cycle

// Move constructor
Kernel::Kernel(Kernel&& other) noexcept : _kernel_lib(std::move(other._kernel_lib)), _xsi(std::move(other._xsi)), _design_lib(std::move(other._design_lib)), _ports() {
    // Reset source
    other._ports.clear();

    // Recreate ports if design is open
    if (_design_lib && _xsi.hasValidHandle()) {
        // Enumerate Ports
        unsigned const port_count = static_cast<unsigned>(xsi<Xsi::get_int>(xsiNumTopPorts));
        _ports.reserve(port_count);
        for (unsigned i = 0; i < port_count; ++i) {
            _ports.emplace_back(Port(*this, i));
        }
    }
}

// Move assignment operator
Kernel& Kernel::operator=(Kernel&& other) noexcept {
    if (this != &other) {
        // Clean up current state
        close();
        _ports.clear();

        // Move from other
        _kernel_lib = std::move(other._kernel_lib);
        _xsi = std::move(other._xsi);
        _design_lib = std::move(other._design_lib);

        // Reset ports in source
        other._ports.clear();

        // Recreate ports if design is open
        if (_design_lib && _xsi.hasValidHandle()) {
            // Enumerate Ports
            unsigned const port_count = static_cast<unsigned>(xsi<Xsi::get_int>(xsiNumTopPorts));
            _ports.reserve(port_count);
            for (unsigned i = 0; i < port_count; i++) {
                _ports.emplace_back(Port(*this, i));
            }
        }
    }
    return *this;
}

Kernel::Kernel(const std::string& kernel_lib) : _kernel_lib(kernel_lib), _xsi(_kernel_lib) {}

Kernel::~Kernel() {
    if (_design_lib)
        std::cerr << "Disposing XSI Kernel with open Design." << std::endl;
}

void Kernel::open(const std::string& design_lib, const s_xsi_setup_info& setup_info) {
    if (_design_lib)
        throw std::runtime_error("Kernel already has a design open: " + _design_lib.path());
    _ports.clear();
    _design_lib.open(design_lib);
    try {
        auto const f = t_fp_xsi_open(resolve_or_throw(_design_lib, "xsi_open"));
        xsiHandle const hdl = f(const_cast<p_xsi_setup_info>(&setup_info));
        if (!hdl)
            throw std::runtime_error("Loading of design failed");
        _xsi.setHandle(hdl);

        // Enumerate Ports
        unsigned const port_count = static_cast<unsigned>(xsi<Xsi::get_int>(xsiNumTopPorts));
        _ports.reserve(port_count);
        for (unsigned i = 0; i < port_count; i++) {
            _ports.emplace_back(Port(*this, i));
        }
    } catch (...) {
        std::cerr << "Exception during design open, closing design library." << std::endl;
        _design_lib.close();
        throw;
    }
}
void Kernel::close() noexcept {
    // Idempotent: only tear down an actually open design.
    if (_xsi.hasValidHandle()) {
        // Bypass the checked Kernel::xsi<> funnel, which would throw on a null handle.
        _xsi.invoke<Xsi::close>();
        _xsi.setHandle(nullptr);
    }

    // Clean up Library State: the simulator kernel caches the SystemVerilog type table of
    // the design that was opened last in a global. It has to be reset so that a design
    // opened later does not pick up the stale table in case the kernel library stays
    // loaded. Note that getsymbol() yields the *address* of the global, so the write has
    // to go through that address.
    if (_kernel_lib) {
        std::optional<void*> const vptr = _kernel_lib.getsymbol("svTypeInfo");
        if (vptr)
            *static_cast<void**>(*vptr) = nullptr;
    }

    _design_lib.close();

    // NOTE: _ports is deliberately *not* cleared here. Python-side Port wrappers are
    // non-owning views into this vector that stay alive as long as the owning Design
    // does, so destroying the elements would leave them dangling. The ports are inert
    // once the design handle is gone, because every access goes through Kernel::xsi<>.
}

bool Kernel::is_open() const noexcept { return _xsi.hasValidHandle(); }

bool Kernel::is_loaded() const noexcept { return bool(_kernel_lib); }

Port& Kernel::getPort(const char* const name) {
    int const id = xsi<Xsi::get_port_number>(name);

    if (id == -1 || id >= static_cast<int>(_ports.size())) {
        throw std::runtime_error(std::string("Port not found: ").append(name));
    }
    return _ports[static_cast<std::size_t>(id)];
}
const Port& Kernel::getPort(const char* const name) const {
    int const id = xsi<Xsi::get_port_number>(name);

    if (id == -1 || id >= static_cast<int>(_ports.size())) {
        throw std::runtime_error(std::string("Port not found: ").append(name));
    }
    return _ports[static_cast<std::size_t>(id)];
}
std::span<Port> Kernel::ports() noexcept { return std::span<Port>(_ports.data(), _ports.data() + _ports.size()); }
std::span<const Port> Kernel::ports() const noexcept { return std::span<const Port>(_ports.data(), _ports.data() + _ports.size()); }

// Port count accessor for Design class
size_t Kernel::port_count() const noexcept { return _ports.size(); }
