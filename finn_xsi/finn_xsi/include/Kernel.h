#ifndef KERNEL_H_
#define KERNEL_H_

#include <SharedLibrary.h>

#include <algorithm>
#include <optional>
#include <span>
#include <vector>

#include "xsi.h"

namespace xsi {

    // Forward declarations
    class Design;
    class Port;

    class Kernel {
        //-----------------------------------------------------------------------
        // Dispatch Table for XSI Functions
        class Xsi {
            //- Statics ---------------------
             public:
            // Function Indeces
            static constexpr unsigned get_value = 0, put_value = 1, get_int_port = 2, get_str_port = 3,

                                      get_int = 4, get_port_number = 5,

                                      trace_all = 6, run = 7, restart = 8, get_status = 9, get_error_info = 10,

                                      close = 11;

             private:
            // Function Names & Types
            static constexpr unsigned EXTENT = 12;
            static char const* const FUNC_NAMES[EXTENT];
            using type_map = std::tuple<
                // Port Access
                t_fp_xsi_get_value, t_fp_xsi_put_value, t_fp_xsi_get_int_port, t_fp_xsi_get_str_port,

                // Design Inspection
                t_fp_xsi_get_int, t_fp_xsi_get_port_number,

                // Simulation Control & Status
                t_fp_xsi_trace_all, t_fp_xsi_run, t_fp_xsi_restart, t_fp_xsi_get_status, t_fp_xsi_get_error_info,

                // Closing
                t_fp_xsi_close>;

            //- Actual Contents -------------
             private:
            xsiHandle _hdl;
            void* _func[EXTENT];

            //- Lifecycle: in-place structure inside Kernel only
             public:
            Xsi(xsi::SharedLibrary& lib);
            ~Xsi() {}

             private:
            Xsi(Xsi const&) = delete;
            Xsi& operator=(Xsi const&) = delete;

             public:
            // Move constructor
            Xsi(Xsi&& other) noexcept;  // Move assignment operator
            Xsi& operator=(Xsi&& other) noexcept;

            //- Handle Update ---------------
             public:
            void setHandle(xsiHandle hdl) noexcept;
            bool hasValidHandle() const noexcept;

            //- XSI Function Invocation -----
             public:
            template<unsigned FID, typename... Args>
            auto invoke(Args&&... args) const {
                auto const f = decltype(std::get<FID>(type_map()))(_func[FID]);
                return (*f)(_hdl, std::forward<Args>(args)...);
            }

        };  // class Xsi

         private:
        // Instance State
        xsi::SharedLibrary _kernel_lib;  // Backing Kernel Library
        Xsi _xsi;                        // XSI Dispatch Table

        // Optional State once a Design in open
        xsi::SharedLibrary _design_lib;
        std::vector<Port> _ports;

         public:
        Kernel(const std::string& kernel_lib);
        Kernel(Kernel const&) = delete;
        Kernel& operator=(Kernel const&) = delete;

        // Move constructor
        Kernel(Kernel&& other) noexcept;
        // Move assignment operator
        Kernel& operator=(Kernel&& other) noexcept;

        ~Kernel();

        // Interface reserved for forwarded access through open Design
         private:
        friend Design;
        friend Port;
        template<unsigned FID, typename... Args>
        auto xsi(Args&&... args) const {
            return _xsi.invoke<FID>(std::forward<Args>(args)...);
        }

        // Port Accessors inlined below and public through Design
        Port& getPort(const char* const name);
        const Port& getPort(const char* const name) const;
        std::span<Port> ports() noexcept;
        std::span<const Port> ports() const noexcept;

        // Design con- & destruction hooks
        void open(const std::string& design_lib, const s_xsi_setup_info& setup_info);
        void close() noexcept;

         public:
        // Port count accessor for Design class
        size_t port_count() const noexcept;

    };  // class Kernel

}  // namespace xsi

#endif /* KERNEL_H_ */
