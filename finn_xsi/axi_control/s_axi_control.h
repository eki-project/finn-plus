#ifndef S_AXI_CONTROL_H
#define S_AXI_CONTROL_H

#include <cstdint>
#include <string>

//Fwd declarations
namespace xsi {
    class Design;
    class Port;
}
class Clock;

class S_AXI_Control {
public:
    // Constructor/destructor
    S_AXI_Control(xsi::Design& design, const std::string& axi_prefix = "s_axi_control_0_0_");
    ~S_AXI_Control() = default;

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

#endif // S_AXI_CONTROL_H
