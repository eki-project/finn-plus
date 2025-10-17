#ifndef CLOCK
#define CLOCK

#include <functional>

// Fwd declarations
namespace xsi {
    class Design;
}

class Clock {
    xsi::Design& design;

    Clock(Clock const&) = delete;
    Clock& operator=(Clock const&) = delete;
    Clock(xsi::Design& design);
    template<size_t IStreamsSize, size_t OStreamsSize, bool LoggingEnabled, bool SingleNode>
    friend class Simulation;

     public:
    Clock(Clock&&) noexcept = default;
    Clock& operator=(Clock&&) noexcept = default;
    ~Clock() noexcept = default;

    std::function<void(bool)> cycle;


    void toggle_clk() noexcept;
};

#endif /* CLOCK */
