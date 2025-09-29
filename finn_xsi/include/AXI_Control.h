#ifndef AXI_CONTROL
#define AXI_CONTROL

#include <cstdint>
#include <string>

// Fwd declarations
namespace xsi {
    class Design;
    class Port;
}  // namespace xsi
class Clock;

class AXI_Control {
     public:
    // Constructor/destructor
    AXI_Control(xsi::Design& design, const std::string& axi_prefix = "AXI_Control_0_0_");
    ~AXI_Control() = default;

    // // Core register access functions
    void write_register(uint32_t addr, uint32_t data);
    uint32_t read_register(uint32_t addr);

     private:
    // AXI interface prefix
    std::string prefix;
    xsi::Design& design;
    Clock& clk;

    // Helper functions for multi-bit signal handling
    void write_addr(const std::string& signal, uint32_t addr);
    void write_data(const std::string& signal, uint32_t data);
    void write_strb(const std::string& signal, uint32_t strb);
    uint32_t read(const std::string& signal);
    void set_bool(const std::string& signal);
    void clear_bool(const std::string& signal);
    bool chk_bool(const std::string& signal);
};

#endif /* AXI_CONTROL */
