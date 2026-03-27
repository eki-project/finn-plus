#include <Kernel.h>
#include <Port.h>
#include <helper.h>

using namespace xsi;

Port::Port(Kernel& kernel, const unsigned id) : _kernel(kernel), _id(id), buffer((width() + 31) / 32) {}

Port::Port(Port&& other) noexcept : _kernel(other._kernel), _id(other._id), buffer(std::move(other.buffer)) {
    // Note: _kernel and _id are reference and const respectively, so they're initialized from other
    // The buffer is moved from the other object
}

Port::~Port() noexcept {}

bool Port::hasUnknown() const noexcept {
    for (auto&& elem : buffer) {
        if (elem.bVal)
            return true;
    }
    return false;
}

bool Port::isZero() const noexcept {
    for (auto&& elem : buffer) {
        if (elem.aVal)
            return false;
    }
    return true;
}

std::string Port::as_binstr() const {
    unsigned const w = width();
    std::string res(w, '?');

    auto buffer_iter = buffer.cbegin();
    auto res_iter = res.rbegin();  // Use reverse iterator to fill from right to left

    uint32_t a = 0;
    uint32_t b = 0;
    for (unsigned i = 0; i < w; i++) {
        if ((i & 31) == 0) {
            a = buffer_iter->aVal;
            b = buffer_iter->bVal;
            ++buffer_iter;
        }
        *res_iter++ = XZ10[((b & 1) << 1) | (a & 1)];
        a >>= 1;
        b >>= 1;
    }

    return res;
}

std::string Port::as_hexstr() const {
    unsigned l = (width() + 3) / 4;
    std::string res(l, '?');
    auto buffer_iter = buffer.cbegin();
    auto res_iter = res.rbegin();  // Use reverse iterator to fill from right to left

    while (l > 0) {
        uint32_t a = buffer_iter->aVal;
        uint32_t b = buffer_iter->bVal;
        ++buffer_iter;

        unsigned m = std::min(8u, l);
        l -= m;
        for (unsigned i = 0; i < m; ++i) {
            unsigned const bm = b & 0xF;
            unsigned const am = a & 0xF;

            *res_iter++ = !bm ? HEX[am] : XZ10[3 - !(am & bm)];
            a >>= 4;
            b >>= 4;
        }
    }
    return res;
}

Port& Port::clear() {
    std::fill(buffer.begin(), buffer.end(), s_xsi_vlog_logicval{.aVal = 0u, .bVal = 0u});
    return *this;
}

const char* Port::name() const noexcept { return _kernel.xsi<Kernel::Xsi::get_str_port>(static_cast<int>(_id), xsiNameTopPort); }

int Port::dir() const noexcept { return _kernel.xsi<Kernel::Xsi::get_int_port>(static_cast<int>(_id), xsiDirectionTopPort); }

unsigned Port::width() const noexcept { return static_cast<unsigned>(_kernel.xsi<Kernel::Xsi::get_int_port>(static_cast<int>(_id), xsiHDLValueSize)); }

bool Port::isInput() const noexcept { return dir() == xsiInputPort; }

bool Port::isOutput() const noexcept { return dir() == xsiOutputPort; }

bool Port::isInout() const noexcept { return dir() == xsiInoutPort; }

Port& Port::read() {
    _kernel.xsi<Kernel::Xsi::get_value>(static_cast<int>(_id), buffer.data());
    return *this;
}

void Port::write_back() { _kernel.xsi<Kernel::Xsi::put_value>(static_cast<int>(_id), buffer.data()); }

bool Port::operator[](const unsigned idx) const noexcept { return (buffer[idx / 32].aVal >> (idx % 32)) & 1; }

bool Port::as_bool() const noexcept { return buffer[0].aVal & 1; }

unsigned Port::as_unsigned() const noexcept { return buffer[0].aVal; }

Port& Port::set(const unsigned val) {
    s_xsi_vlog_logicval* const p = buffer.data();
    p->aVal = val;
    p->bVal = 0;
    return *this;
}

Port& Port::set_binstr(const std::string& val) {
    auto val_iter = val.crbegin();  // Process from right to left

    size_t chars_processed = 0;
    const size_t val_length = val.length();

    for (auto& elem : buffer) {
        uint32_t a = 0;
        uint32_t b = 0;

        // Process up to 32 characters for this buffer element
        const size_t chars_to_process = std::min(32UL, val_length - chars_processed);

        for (size_t j = 0; j < chars_to_process; ++j) {
            a <<= 1;
            b <<= 1;

            if (val_iter != val.crend()) {
                switch (*val_iter++) {
                    case '1':
                        a |= 1;
                        [[fallthrough]];
                    case '0':
                        break;
                    default:
                        a |= 1;
                        [[fallthrough]];
                    case 'Z':
                    case 'z':
                        b |= 1;
                        break;
                }
            }
        }

        elem.aVal = a;
        elem.bVal = b;

        chars_processed += chars_to_process;
        if (chars_processed >= val_length)
            break;
    }

    return *this;
}

Port& Port::set_hexstr(const std::string& val) {
    auto val_iter = val.crbegin();  // Process from right to left

    size_t chars_processed = 0;
    const size_t val_length = val.length();

    for (auto& elem : buffer) {
        uint32_t a = 0;
        uint32_t b = 0;

        // Process up to 8 hex characters (32 bits) for this buffer element
        const size_t chars_to_process = std::min(8UL, val_length - chars_processed);

        for (size_t j = 0; j < chars_to_process; ++j) {
            a <<= 4;
            b <<= 4;

            if (val_iter != val.crend()) {
                char c = *val_iter++;

                if (('0' <= c) && c <= '9') {
                    a |= c & 0xF;
                } else {
                    c |= 0x20;  // Convert to lowercase
                    if (('a' <= c) && (c <= 'f')) {
                        a |= static_cast<uint32_t>(c - ('a' - 10));
                    } else {
                        b |= 0xF;
                        if (c != 'z') {
                            a |= 0xF;
                        }
                    }
                }
            }
        }

        elem.aVal = a;
        elem.bVal = b;

        chars_processed += chars_to_process;
        if (chars_processed >= val_length)
            break;
    }

    return *this;
}
