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
    AXI_Control(xsi::Design& design, Clock& clock, const std::string& axi_prefix = "AXI_Control_0_0_");
    ~AXI_Control() noexcept = default;

    // // Core register access functions
    void writeRegister(uint32_t addr, uint32_t data);
    uint32_t readRegister(uint32_t addr);

     private:
    // AXI interface prefix
    std::string prefix;
    xsi::Design& design;
    Clock& clk;

    // Helper functions for multi-bit signal handling
    void writeAddr(const std::string& signal, uint32_t addr);
    void writeData(const std::string& signal, uint32_t data);
    void writeStrb(const std::string& signal, uint32_t strb);
    uint32_t read(const std::string& signal);
    void setBool(const std::string& signal);
    void clearBool(const std::string& signal);
    bool chkBool(const std::string& signal);
};

#endif /* AXI_CONTROL */
