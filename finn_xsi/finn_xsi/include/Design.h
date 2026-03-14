#ifndef DESIGN
#define DESIGN

#include <Kernel.h>

namespace xsi {

    //	- non-copyable handle for exposing simulation control.
    class Design {
        xsi::Kernel _kernel;

         public:
        Design(xsi::Kernel& kernel, const std::string& design_lib, const s_xsi_setup_info& setup_info);
        Design(xsi::Kernel& kernel, const std::string& design_lib, const char* const log_file = nullptr, const char* const wdb_file = nullptr);
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
        void run(const XSI_INT64 step);
        void restart();

        int get_status() const noexcept;
        const char* get_error_info() const noexcept;

        // Port Access
         public:
        int num_ports() const noexcept;

        xsi::Port& getPort(const std::string& name);
        const xsi::Port& getPort(const std::string& name) const;

        std::span<xsi::Port> ports() noexcept;
        std::span<const xsi::Port> ports() const noexcept;

    };  // class Design
}  // namespace xsi

#endif /* DESIGN */
