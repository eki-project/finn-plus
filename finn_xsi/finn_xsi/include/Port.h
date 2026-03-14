#ifndef PORT_H_
#define PORT_H_

#include <string>
#include <vector>

#include "xsi.h"

namespace xsi {

    class Kernel;  // Forward declaration

    // Only exists within controlled environment within Kernel with open Design.
    class Port {
        Kernel& _kernel;
        unsigned const _id;
        std::vector<s_xsi_vlog_logicval> buffer;

         private:
        friend Kernel;
        // Con- and destruction under full control of Kernel
        Port(Port const&) = delete;
        Port& operator=(Port const&) = delete;
        Port(Kernel& kernel, const unsigned id);

         public:
        Port(Port&& other) noexcept;
        ~Port() noexcept;

         public:
        const char* name() const noexcept;
        int dir() const noexcept;
        unsigned width() const noexcept;

        bool isInput() const noexcept;
        bool isOutput() const noexcept;
        bool isInout() const noexcept;

         public:
        // Buffer Synchronization
        Port& read();
        void write_back();

        // Inspection
        bool hasUnknown() const noexcept;
        bool isZero() const noexcept;
        bool operator[](const unsigned idx) const noexcept;

        bool as_bool() const noexcept;
        unsigned as_unsigned() const noexcept;
        std::string as_binstr() const;
        std::string as_hexstr() const;

        // Manipulation
        Port& clear();
        Port& set(const unsigned val);
        Port& set_binstr(const std::string& val);
        Port& set_hexstr(const std::string& val);

    };  // class Port

}  // namespace xsi

#endif /* PORT_H_ */
