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
        Port(Kernel& kernel, unsigned const id);

         public:
        Port(Port&& other) noexcept;
        ~Port();

         public:
        char const* name() const;
        int dir() const;
        unsigned width() const;

        bool isInput() const;
        bool isOutput() const;
        bool isInout() const;

         public:
        // Buffer Synchronization
        Port& read();
        void write_back();

        // Inspection
        bool hasUnknown() const;
        bool isZero() const;
        bool operator[](unsigned const idx) const;

        bool as_bool() const;
        unsigned as_unsigned() const;
        std::string as_binstr() const;
        std::string as_hexstr() const;

        // Manipulation
        Port& clear();
        Port& set(unsigned val);
        Port& set_binstr(std::string const& val);
        Port& set_hexstr(std::string const& val);

    };  // class Port

}  // namespace xsi

#endif /* PORT_H_ */
