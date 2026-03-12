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
    template<size_t IStreamsSize, size_t OStreamsSize, bool LoggingEnabled>
    friend class Simulation;

     public:
    Clock(Clock&&) noexcept = default;
    Clock& operator=(Clock&&) noexcept = default;
    ~Clock() noexcept = default;

    std::function<void()> clkHigh;
    std::function<void()> clkLow;
    std::function<void(bool)> cycle;


    void toggleClk() noexcept;

    void clockHigh() noexcept;
    void clockLow() noexcept;
};

#endif /* CLOCK */
