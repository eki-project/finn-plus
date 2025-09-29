#ifndef DESIGN
#define DESIGN

#include <Kernel.h>

namespace xsi {

    //	- non-copyable handle for exposing simulation control.
    class Design {
        xsi::Kernel _kernel;

         public:
        Design(xsi::Kernel& kernel, std::string const& design_lib, s_xsi_setup_info const& setup_info);
        Design(xsi::Kernel& kernel, std::string const& design_lib, char const* const log_file = nullptr, char const* const wdb_file = nullptr);
        ~Design();

         private:
        Design(Design const&) = delete;
        Design& operator=(Design const&) = delete;

         public:
        // Move constructor
        Design(Design&& other) noexcept;

        // Move assignment operator
        Design& operator=(Design&& other) noexcept;

        //-----------------------------------------------------------------------
        // Forwarded Access to Open Simulation

        // Simulation Control & Status
         public:
        void trace_all();
        void run(XSI_INT64 const step);
        void restart();

        int get_status() const;
        char const* get_error_info() const;

        // Port Access
         public:
        int num_ports() const;

        xsi::Port& getPort(std::string const& name);
        xsi::Port const& getPort(std::string const& name) const;

        std::span<xsi::Port> ports();
        std::span<xsi::Port const> ports() const;

    };  // class Design
}  // namespace xsi

#endif /* DESIGN */
